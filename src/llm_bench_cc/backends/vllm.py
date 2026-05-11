"""vLLM backend.

Loads the model into vLLM's `LLM` engine and generates via `llm.chat()` with
the model's chat template. Validated on gemma-4-E4B-it bf16 and FP8_BLOCK
(compressed-tensors checkpoints from `llmcompressor.model_free_ptq`) on
2026-05-10, and on bnb 4-bit NF4 + FP8 W8A8 (both in-flight quantization)
on 2026-05-11 — see scripts/vllm_spike.py for the spikes that produced the
numbers.

Three runtime quirks worth knowing:

  1. vLLM v1 runs the model in a worker subprocess. `torch.cuda.max_memory_allocated()`
     in the parent process returns 0. Peak VRAM is measured via a background
     nvidia-smi poller (same pattern as the spike script). NB: nvidia-smi sees
     vLLM's *pre-allocated KV-cache pool*, not the weight footprint — so
     peak_vram_gb collapses to roughly `gpu_memory_utilization × total_VRAM`
     regardless of quant. Quant savings in vLLM mode surface as KV-cache
     headroom (longer seqs, more concurrency), not as a smaller peak_vram_gb.
     If you need the weights-only number, grep startup logs for "model weights
     take X GiB".

  2. The Ada-generation perf cliff (vLLM issue #38887): gemma-4's heterogeneous
     attention head dims force a TRITON_ATTN fallback that disables FlashAttention.
     Manifests as a startup log line ("Using AttentionBackendEnum.TRITON_ATTN
     backend"). Empirically ~16 tok/s on L4 — slower than HF eager. The FP8
     story is still a memory + slight throughput win over vLLM bf16; just don't
     expect a speed bonanza vs HF.

  3. vLLM's bitsandbytes loader has no exposed skip-modules hook (unlike the HF
     backend's `_enumerate_non_llm_linear_paths`). bnb runs in-flight against
     every Linear it can pattern-match — vision tower included. Composite-vs-HF
     drift on `bnb-nf4-vllm` will partly reflect that scope difference.
"""

from __future__ import annotations

import logging
import subprocess
import threading
import time
from typing import Any

import torch

logger = logging.getLogger(__name__)


class _GpuMemoryPoller:
    """Background nvidia-smi poller for peak GPU memory used. See module
    docstring for why torch.cuda counters can't see vLLM's allocations."""

    def __init__(self, interval_s: float = 0.1) -> None:
        self.interval = interval_s
        self._peak_mib = 0
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._poll, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None

    def reset(self) -> None:
        with self._lock:
            self._peak_mib = 0

    def peak_gb(self) -> float:
        with self._lock:
            return self._peak_mib / 1024.0

    def _poll(self) -> None:
        while not self._stop.is_set():
            try:
                out = subprocess.check_output(
                    ["nvidia-smi", "--query-gpu=memory.used",
                     "--format=csv,noheader,nounits"],
                    text=True, timeout=1.0,
                ).strip().splitlines()[0]
                used = int(out)
                with self._lock:
                    if used > self._peak_mib:
                        self._peak_mib = used
            except (subprocess.SubprocessError, ValueError, IndexError, FileNotFoundError):
                # Transient failures are fine; the next poll will recover.
                pass
            self._stop.wait(self.interval)


class VLLMBackend:
    """vLLM runtime. Reads cfg.model.hf_id (Hub id or local checkpoint dir)
    and cfg.runtime.vllm.* knobs (gpu_memory_utilization, limit_mm_per_prompt).

    Quant routing:
      * compressed_tensors (FP8_BLOCK, W4A16) — vLLM's bundled loader picks
        these up from the checkpoint's config.json automatically; no extra
        constructor kwarg needed. fp8 kernels engage on Ada+ hardware.
      * bnb (nf4) — in-flight quantization at load time; we forward
        `quantization="bitsandbytes"` to LLM(). vLLM's bnb loader has no
        skip-modules hook, so this applies to every Linear in the model.
      * vllm_fp8 (fp8_w8a8) — in-flight dynamic FP8_E4M3 at load time; we
        forward `quantization="fp8"` to LLM(). Per-tensor static weight
        scales + dynamic per-tensor activation scales. vLLM auto-skips
        `lm_head`; on gemma-4-E4B-it the multimodal towers and embeddings
        empirically also stay in bf16 (spike load-log: "Model loading took
        10.93 GiB" vs ~16 GiB for bf16 — only adds up if non-Linear modules
        weren't touched). Requires sm_89+ (Ada Lovelace, Hopper, ...).
    """

    def __init__(self, model_cfg, vllm_cfg) -> None:
        from omegaconf import OmegaConf
        from vllm import LLM

        # vLLM accepts a dtype string ("bfloat16") directly; we keep the
        # torch.dtype version too so summary.json stays consistent across
        # backends.
        dtype_str = model_cfg.dtype
        self.dtype = getattr(torch, dtype_str)

        # OmegaConf containers don't always survive being passed straight into
        # third-party kwargs (vLLM does isinstance(dict) checks); resolve to
        # plain dicts.
        limit_mm = OmegaConf.to_container(
            vllm_cfg.get("limit_mm_per_prompt") or {"image": 1, "audio": 0},
            resolve=True,
        )
        gpu_mem_util = float(vllm_cfg.get("gpu_memory_utilization", 0.9))

        # Quant routing happens before LLM construction because bnb needs an
        # explicit `quantization=` kwarg; compressed_tensors does not (vLLM
        # auto-detects from the checkpoint's config.json).
        quant_cfg = model_cfg.get("quant")
        self.quant_backend = quant_cfg.get("backend") if quant_cfg is not None else None
        self.quant_mode = quant_cfg.get("mode") if quant_cfg is not None else None

        extra_llm_kwargs: dict[str, Any] = {}
        if self.quant_backend == "bnb":
            extra_llm_kwargs["quantization"] = "bitsandbytes"
        elif self.quant_backend == "vllm_fp8":
            extra_llm_kwargs["quantization"] = "fp8"

        logger.info("Loading via vLLM: %s (dtype=%s, gpu_memory_utilization=%.2f, quantization=%s)",
                    model_cfg.hf_id, dtype_str, gpu_mem_util,
                    extra_llm_kwargs.get("quantization", "auto"))

        self._llm = LLM(
            model=model_cfg.hf_id,
            dtype=dtype_str,
            limit_mm_per_prompt=limit_mm,
            gpu_memory_utilization=gpu_mem_util,
            **extra_llm_kwargs,
        )

        self.name = model_cfg.name
        self.hf_id = model_cfg.hf_id
        # vLLM is single-GPU here; cuda:0 is accurate for our L4 target.
        # Update if/when we add tensor-parallel.
        self.device = "cuda:0"

        self._poller = _GpuMemoryPoller()
        self._poller.start()

    def reset_peak_vram(self) -> None:
        self._poller.reset()

    def peak_vram_gb(self) -> float:
        return self._poller.peak_gb()

    def generate(
        self, image: Any, prompt: str, gen_kwargs: dict[str, Any]
    ) -> tuple[str, float]:
        from vllm import SamplingParams

        conversation = [{
            "role": "user",
            "content": [
                {"type": "image_pil", "image_pil": image},
                {"type": "text", "text": prompt},
            ],
        }]

        # Translate HF-style gen_kwargs to SamplingParams. We honor the
        # do_sample / temperature pair exactly: do_sample=False => greedy
        # (temperature=0.0) regardless of any non-zero temperature the
        # variant might have left in the config.
        max_tokens = int(gen_kwargs.get("max_new_tokens", 256))
        if gen_kwargs.get("do_sample", False):
            temperature = float(gen_kwargs.get("temperature", 1.0))
        else:
            temperature = 0.0
        sampling = SamplingParams(max_tokens=max_tokens, temperature=temperature)

        t0 = time.perf_counter()
        # use_tqdm=False suppresses vLLM's per-call progress bar — at one bar
        # per sample, an N=200 task floods Colab's output renderer and freezes
        # the cell display. The runner's outer tqdm (one bar per task) is
        # plenty signal.
        outputs = self._llm.chat(conversation, sampling_params=sampling, use_tqdm=False)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        text = outputs[0].outputs[0].text.strip()
        return text, elapsed_ms

    def __del__(self) -> None:
        # Best-effort poller cleanup; the daemon thread would exit at process
        # death anyway but stopping it cleanly avoids spurious warnings in
        # short-lived test runs.
        try:
            self._poller.stop()
        except Exception:
            pass
