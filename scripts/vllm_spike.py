"""Two-spike feasibility check for moving the harness to vLLM.

Before committing to a vLLM backend abstraction in `llm_bench_cc`, we need
two go/no-go signals on a real L4:

  Spike 1 — bf16 throughput on gemma-4-E4B-it. There is a known Ada-generation
  perf cliff for this model in vLLM (issue #38887): heterogeneous attention
  head dims (head_dim=256 / global_head_dim=512) force a TRITON_ATTN backend
  fallback that disables FlashAttention. Reported ~9 tok/s on RTX 4090 in
  v0.19.0. If we still see that on L4 with current vLLM, the FP8 memory win
  comes with a *throughput regression* vs HF eager — and the rest of the
  refactor needs to weigh that.

  Spike 2 — FP8_BLOCK checkpoint acceptance. The official Gemma 4 recipe
  covers bf16 only. Whether vLLM's compressed-tensors loader accepts a
  checkpoint produced by `llmcompressor.model_free_ptq` (vs needing vLLM's
  own AutoFP8 path) is undocumented; quickest answer is to point it at
  the checkpoint and watch the load.

This script does ONE timed multimodal generation per invocation. Run it
twice — once on the Hub bf16 model, once on the local FP8_BLOCK save dir.

Install on a fresh Colab L4 (vLLM pins its own torch/CUDA — DO NOT install
into the llm-bench-cc venv):

    pip install --pre vllm

Run:

    # Spike 1 — bf16 baseline:
    python scripts/vllm_spike.py

    # Spike 2 — FP8_BLOCK acceptance (after running quantize_fp8.py):
    python scripts/vllm_spike.py \
        --model-path /tmp/llm-bench-cc/quant/gemma-4-E4B-it-FP8_BLOCK

Eyeball the vLLM startup logs for `TRITON_ATTN` — that's the marker for
the perf-cliff bug. tok/s and peak VRAM are printed at the end.
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import threading
import time

logger = logging.getLogger("vllm_spike")


class _GpuMemoryPoller:
    """Background nvidia-smi poller for peak GPU memory used.

    `torch.cuda.max_memory_allocated()` returns 0 under vLLM v1 because the
    model runs in a worker subprocess — the parent process's CUDA context
    sees nothing. nvidia-smi reports device-wide usage and so captures the
    worker's allocations correctly. Polled at 10 Hz; the spike runs long
    enough that we won't miss a peak."""

    def __init__(self, interval_s: float = 0.1):
        self.interval = interval_s
        self.peak_mib = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _poll(self) -> None:
        while not self._stop.is_set():
            try:
                out = subprocess.check_output(
                    ["nvidia-smi", "--query-gpu=memory.used",
                     "--format=csv,noheader,nounits"],
                    text=True, timeout=1.0,
                ).strip().splitlines()[0]
                self.peak_mib = max(self.peak_mib, int(out))
            except (subprocess.SubprocessError, ValueError, IndexError, FileNotFoundError):
                pass
            self._stop.wait(self.interval)

    def __enter__(self) -> "_GpuMemoryPoller":
        self._thread = threading.Thread(target=self._poll, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)


def _make_test_image():
    """Tiny synthetic image with readable text. We don't score correctness —
    only check that vLLM accepts a multimodal input and emits non-empty text.
    Keeps the spike free of dataset dependencies."""
    from PIL import Image, ImageDraw

    img = Image.new("RGB", (384, 128), "white")
    ImageDraw.Draw(img).text((20, 50), "VLLM SMOKE TEST", fill="black")
    return img


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--model-path", default="google/gemma-4-E4B-it",
                        help="Hub id or local checkpoint dir. Use the FP8_BLOCK "
                             "save dir for spike 2.")
    parser.add_argument("--dtype", default="bfloat16",
                        help="Per the Gemma 4 recipe, bf16. Override only for debug.")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    import torch
    import vllm
    from vllm import LLM, SamplingParams

    logger.info("vLLM=%s torch=%s cuda=%s",
                vllm.__version__, torch.__version__, torch.version.cuda)
    logger.info("Loading %s (dtype=%s) ...", args.model_path, args.dtype)

    # limit_mm_per_prompt: image=1 is enough for the smoke; audio=0 avoids
    # allocating audio-tower buffers we don't exercise here.
    llm = LLM(
        model=args.model_path,
        dtype=args.dtype,
        limit_mm_per_prompt={"image": 1, "audio": 0},
    )

    image = _make_test_image()
    conversation = [{
        "role": "user",
        "content": [
            {"type": "image_pil", "image_pil": image},
            {"type": "text", "text": "Read the text in this image."},
        ],
    }]

    # Warmup — first call compiles kernels and warms caches; excluding it
    # keeps the tok/s number meaningful. 1 token is enough.
    logger.info("Warmup ...")
    llm.chat(conversation, sampling_params=SamplingParams(temperature=0.0, max_tokens=1))

    logger.info("Timed generation (max_new_tokens=%d) ...", args.max_new_tokens)
    with _GpuMemoryPoller() as poller:
        t0 = time.perf_counter()
        outputs = llm.chat(
            conversation,
            sampling_params=SamplingParams(temperature=0.0, max_tokens=args.max_new_tokens),
        )
        elapsed = time.perf_counter() - t0

    out = outputs[0].outputs[0]
    n_tokens = len(out.token_ids)
    tok_per_sec = n_tokens / elapsed if elapsed > 0 else 0.0
    peak_vram_gb = poller.peak_mib / 1024.0

    logger.info("=== RESULTS ===")
    logger.info("decoded text  : %r", out.text)
    logger.info("n_new_tokens  : %d", n_tokens)
    logger.info("elapsed_s     : %.3f", elapsed)
    logger.info("tok_per_sec   : %.2f", tok_per_sec)
    logger.info("peak_vram_gb  : %.2f", peak_vram_gb)
    logger.info("Check startup logs above for 'TRITON_ATTN' — that's the issue #38887 marker.")


if __name__ == "__main__":
    main()
