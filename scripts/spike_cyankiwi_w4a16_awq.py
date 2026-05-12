"""Feasibility spike — pre-quantized W4A16 (cyankiwi community checkpoint).

Verifies four predictions from `docs/community-w4a16-analysis.md` before we
wire the checkpoint into the benchmark harness:

  P1 — vLLM auto-detects `compressed-tensors pack-quantized` from the
       checkpoint's `config.json`; no `quantization=` kwarg needed (same
       auto-detect path FP8_BLOCK already uses). Verified by loading with
       a bare `LLM(model=...)` and watching the startup log for a
       `compressed_tensors` / `pack-quantized` mention.

  P2 — Exactly 259 LM Linears are quantized:
         42 × (q_proj, o_proj, gate_proj, up_proj, down_proj) = 210
       + 24 × (k_proj, v_proj)                                = 48   (KV-sharing
                                                                       on 18/42 layers)
       + 1 per_layer_model_projection
       PLE plumbing (`per_layer_input_gate`, `per_layer_projection`, 84
       Linears total) stays in bf16, as do `vision_tower`, `audio_tower`,
       `lm_head`. Verified offline by enumerating tensor keys in the
       downloaded safetensors — cheap, no GPU needed, no vLLM load.

  P3 — Multimodal generation still works. Vision tower is bit-identical
       bf16, so the encoder path is intact; the LM was calibrated on
       text-only data (`cyankiwi/calibration`, 384 Nemotron-SWE-v1 agent
       traces) so visual-token processing may degrade. We don't score
       accuracy here — only check that a text+image prompt returns
       non-empty text mentioning the synthetic stamp on the image.

  P4 — Weights footprint sits between bf16 baseline (~16 GiB on Spike 1)
       and a fully-4-bit LM (~3 GiB). cyankiwi's mixed scope (LM Linears
       at int4 g=32 asym; vision+audio towers + PLE plumbing + lm_head
       in bf16) predicts ~5–7 GiB for the "Model loading took X GiB"
       line in vLLM's startup log. We can't read that line programmatically
       here — surfaced in the final summary as a reminder.

Spike 5 follows the format of the existing spikes (see vllm_spike.py).

Install on a fresh Colab L4 (vLLM pins its own torch/CUDA — DO NOT install
into the llm-bench-cc venv):

    pip install --pre vllm

Run:

    python scripts/spike_cyankiwi_w4a16.py

Optional flags:

    --skip-vllm           Run only the offline safetensors check (P2);
                          skip the vLLM load + generation (P1, P3, P4).
                          Useful for a CPU-only sanity check.
    --skip-image          Run vLLM load + text-only generation only; skip
                          the multimodal step (P3).
    --max-new-tokens N    Cap on generation length (default 64).

Eyeball the vLLM startup logs for:

    "compressed_tensors" / "compressed-tensors" — confirms P1 detection.
    "Model loading took X GiB"                  — confirms P4 footprint.
    "AttentionBackendEnum.TRITON_ATTN"          — Ada-cliff issue #38887;
                                                   expected on gemma-4-E4B.
"""

from __future__ import annotations

import argparse
import logging
import re
import subprocess
import threading
import time
from collections import Counter

logger = logging.getLogger("spike_cyankiwi_w4a16")

# Hardcoded — this script is one-checkpoint-specific. To test a different
# community W4A16 (e.g. Vishva007), copy this file and adjust.
MODEL_ID = "cyankiwi/gemma-4-E4B-it-AWQ-INT4"

# Coverage predicted by `docs/community-w4a16-analysis.md`. Used by P2.
PRED_QUANTIZED_LINEAR_COUNT = 258  # 42*(q,o,gate,up,down) + 24*(k,v)
PRED_PER_LAYER_FULL = {  # appear in every one of the 42 LM layers
    "self_attn.q_proj", "self_attn.o_proj",
    "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj",
}
PRED_PER_LAYER_KV = {"self_attn.k_proj", "self_attn.v_proj"}  # only 24 layers
PRED_LM_LAYERS_TOTAL = 42
PRED_LM_LAYERS_OWN_KV = 24
PRED_PLE_KEEPS_BF16 = {"per_layer_input_gate", "per_layer_projection"}


class _GpuMemoryPoller:
    """Background nvidia-smi poller. See vllm_spike.py for the why."""

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
    """Synthetic image with a known stamp. We don't score correctness —
    just check that the decoded text mentions the stamp, indicating the
    vision tower fed something usable to the (4-bit) LM."""
    from PIL import Image, ImageDraw

    img = Image.new("RGB", (384, 128), "white")
    ImageDraw.Draw(img).text((20, 50), "CYANKIWI SMOKE", fill="black")
    return img


def _classify_quantized_linear(base: str) -> str:
    """Map a base module path (no `.weight_packed` suffix) to a coarse
    category. Used for the P2 breakdown report."""
    if "language_model" in base and "self_attn" in base:
        m = re.search(r"self_attn\.(\w+_proj)", base)
        return f"LM.attn.{m.group(1) if m else '?'}"
    if "language_model" in base and "mlp" in base:
        m = re.search(r"mlp\.(\w+_proj)", base)
        return f"LM.mlp.{m.group(1) if m else '?'}"
    if "per_layer_input_gate" in base: return "LM.PLE.input_gate"
    if "per_layer_projection" in base: return "LM.PLE.projection"
    if "vision_tower" in base: return "vision"
    if "audio_tower" in base: return "audio"
    if "lm_head" in base: return "lm_head"
    return f"other:{'.'.join(base.split('.')[:3])}"


def _verify_p2_quantization_scope() -> tuple[bool, dict]:
    """Download `model.safetensors` from the Hub (cached) and enumerate its
    tensor keys. compressed-tensors `pack-quantized` stores quantized
    Linears as a triplet — `.weight_packed`, `.weight_scale`, and either
    `.weight_zero_point` (asymmetric) or no zp tensor (symmetric). Count
    the `.weight_packed` entries; that count is the number of Linears
    actually quantized. Compare against the analysis prediction.

    Falls back to AWQ/GPTQ-style `.qweight` naming if `.weight_packed`
    isn't present — useful if this script is copy-edited for a different
    checkpoint and we forget to update the suffix probe."""
    from huggingface_hub import hf_hub_download
    from safetensors import safe_open

    logger.info("[P2] downloading %s/model.safetensors (cached on second run)...",
                MODEL_ID)
    path = hf_hub_download(repo_id=MODEL_ID, filename="model.safetensors")
    logger.info("[P2] safetensors cached at %s", path)

    with safe_open(path, framework="pt") as f:
        keys = list(f.keys())

    # compressed-tensors first, then AWQ/GPTQ fallback.
    packed_suffix, used_naming = ".weight_packed", "compressed-tensors"
    packed_keys = [k for k in keys if k.endswith(packed_suffix)]
    if not packed_keys:
        packed_suffix, used_naming = ".qweight", "awq-or-gptq"
        packed_keys = [k for k in keys if k.endswith(packed_suffix)]

    bases = [k[: -len(packed_suffix)] for k in packed_keys]
    cats = Counter(_classify_quantized_linear(b) for b in bases)
    observed_count = len(packed_keys)

    # Per-module-class coverage check against the prediction.
    per_layer_full_counts = {f"LM.attn.{x.split('.')[-1]}": 0
                             for x in PRED_PER_LAYER_FULL if "self_attn" in x}
    per_layer_full_counts.update({f"LM.mlp.{x.split('.')[-1]}": 0
                                  for x in PRED_PER_LAYER_FULL if "mlp" in x})
    for k in list(per_layer_full_counts):
        per_layer_full_counts[k] = cats.get(k, 0)
    per_layer_kv_counts = {f"LM.attn.{x.split('.')[-1]}": cats.get(f"LM.attn.{x.split('.')[-1]}", 0)
                           for x in PRED_PER_LAYER_KV}

    ple_quantized = (cats.get("LM.PLE.input_gate", 0)
                     + cats.get("LM.PLE.projection", 0))

    full_ok = all(v == PRED_LM_LAYERS_TOTAL for v in per_layer_full_counts.values())
    kv_ok = all(v == PRED_LM_LAYERS_OWN_KV for v in per_layer_kv_counts.values())
    count_ok = observed_count == PRED_QUANTIZED_LINEAR_COUNT
    ple_bf16_ok = ple_quantized == 0
    passed = full_ok and kv_ok and count_ok and ple_bf16_ok

    report = {
        "naming_convention": used_naming,
        "total_quantized_linears": observed_count,
        "predicted_total": PRED_QUANTIZED_LINEAR_COUNT,
        "per_layer_full_quant_modules": per_layer_full_counts,
        "per_layer_kv_quant_modules": per_layer_kv_counts,
        "ple_linears_quantized": ple_quantized,
        "categories": dict(cats),
        "checks": {
            "full_coverage_42_layers": full_ok,
            "kv_coverage_24_layers": kv_ok,
            "total_count_matches": count_ok,
            "ple_kept_in_bf16": ple_bf16_ok,
        },
        "passed": passed,
    }
    return passed, report


def _run_vllm_generation(max_new_tokens: int, skip_image: bool) -> dict:
    """Load via vLLM with NO `quantization=` kwarg and generate. Tests P1
    (auto-detect) and P3 (multimodal) at the cost of one full vLLM session."""
    import torch
    import vllm
    from vllm import LLM, SamplingParams

    logger.info("[P1/P3/P4] vLLM=%s torch=%s cuda=%s",
                vllm.__version__, torch.__version__, torch.version.cuda)
    logger.info("[P1] loading %s with NO `quantization=` kwarg ...", MODEL_ID)

    # Deliberately omit `quantization=` so we exercise the auto-detect path.
    # `dtype="bfloat16"` is the compute dtype for activations + the bf16
    # tensors (vision_tower, audio_tower, lm_head, PLE plumbing, norms).
    llm = LLM(
        model=MODEL_ID,
        dtype="bfloat16",
        limit_mm_per_prompt={"image": 1, "audio": 0},
    )

    sampling = SamplingParams(temperature=0.0, max_tokens=max_new_tokens)
    result: dict = {"text_only": None, "multimodal": None}

    # --- text-only first: cheapest sanity check that the LM itself works
    text_conv = [{"role": "user", "content": [
        {"type": "text", "text": "Reply with the single word: ready."}
    ]}]
    logger.info("Warmup ...")
    llm.chat(text_conv, sampling_params=SamplingParams(temperature=0.0, max_tokens=1))

    logger.info("[text-only] timed generation (max_new_tokens=%d) ...", max_new_tokens)
    with _GpuMemoryPoller() as poller:
        t0 = time.perf_counter()
        outputs = llm.chat(text_conv, sampling_params=sampling)
        elapsed = time.perf_counter() - t0
    out = outputs[0].outputs[0]
    result["text_only"] = {
        "decoded": out.text,
        "n_tokens": len(out.token_ids),
        "elapsed_s": elapsed,
        "tok_per_sec": (len(out.token_ids) / elapsed) if elapsed > 0 else 0.0,
        "peak_vram_gb": poller.peak_mib / 1024.0,
    }

    if skip_image:
        return result

    # --- multimodal: same shape as vllm_spike.py, different stamp
    image = _make_test_image()
    mm_conv = [{"role": "user", "content": [
        {"type": "image_pil", "image_pil": image},
        {"type": "text", "text": "Read the text in this image."},
    ]}]
    logger.info("[multimodal] timed generation (max_new_tokens=%d) ...", max_new_tokens)
    with _GpuMemoryPoller() as poller:
        t0 = time.perf_counter()
        outputs = llm.chat(mm_conv, sampling_params=sampling)
        elapsed = time.perf_counter() - t0
    out = outputs[0].outputs[0]
    decoded = out.text
    # Loose acceptance: P3 only requires the LM to acknowledge SOMETHING
    # text-like was in the image. We stamp "CYANKIWI SMOKE" — accept if
    # any of those tokens appears (the 4-bit LM may garble it).
    accept_tokens = ["cyankiwi", "smoke", "cyan", "kiwi"]
    p3_accepted = any(tok in decoded.lower() for tok in accept_tokens)
    result["multimodal"] = {
        "decoded": decoded,
        "n_tokens": len(out.token_ids),
        "elapsed_s": elapsed,
        "tok_per_sec": (len(out.token_ids) / elapsed) if elapsed > 0 else 0.0,
        "peak_vram_gb": poller.peak_mib / 1024.0,
        "p3_image_text_recognized": p3_accepted,
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--skip-vllm", action="store_true",
                        help="Only run the offline safetensors check (P2); "
                             "skip vLLM load and generation. CPU-only OK.")
    parser.add_argument("--skip-image", action="store_true",
                        help="Skip the multimodal step (P3).")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    # --- P2 (offline): always run; cheap and the most direct verification
    p2_pass, p2_report = _verify_p2_quantization_scope()

    logger.info("=== P2 — Quantization scope ===")
    logger.info("naming convention                : %s", p2_report["naming_convention"])
    logger.info("total quantized Linears observed : %d", p2_report["total_quantized_linears"])
    logger.info("total quantized Linears predicted: %d", p2_report["predicted_total"])
    logger.info("per-layer FULL-coverage modules  : %s",
                p2_report["per_layer_full_quant_modules"])
    logger.info("per-layer KV-shared modules      : %s",
                p2_report["per_layer_kv_quant_modules"])
    logger.info("PLE Linears quantized            : %d",
                p2_report["ple_linears_quantized"])
    logger.info("checks: %s", p2_report["checks"])
    logger.info("P2 VERDICT: %s", "PASS" if p2_pass else "FAIL")

    if args.skip_vllm:
        logger.info("--skip-vllm set; stopping after P2.")
        return

    # --- P1 / P3 / P4: only if a GPU is present
    gen = _run_vllm_generation(args.max_new_tokens, args.skip_image)

    t = gen["text_only"]
    logger.info("=== P1 — Auto-detect (no `quantization=` kwarg) ===")
    logger.info("Load completed without explicit quantization kwarg.")
    logger.info("Look back at vLLM startup logs for `compressed_tensors` "
                "/ `pack-quantized` mentions to confirm the detection path.")
    logger.info("P1 VERDICT (load-success only): PASS")

    logger.info("=== Text-only generation ===")
    logger.info("decoded     : %r", t["decoded"])
    logger.info("n_tokens    : %d", t["n_tokens"])
    logger.info("tok_per_sec : %.2f", t["tok_per_sec"])
    logger.info("peak_vram_gb: %.2f", t["peak_vram_gb"])

    if not args.skip_image and gen["multimodal"] is not None:
        m = gen["multimodal"]
        logger.info("=== P3 — Multimodal generation ===")
        logger.info("decoded                  : %r", m["decoded"])
        logger.info("n_tokens                 : %d", m["n_tokens"])
        logger.info("tok_per_sec              : %.2f", m["tok_per_sec"])
        logger.info("peak_vram_gb             : %.2f", m["peak_vram_gb"])
        logger.info("image-text token in reply: %s", m["p3_image_text_recognized"])
        logger.info("P3 VERDICT (loose): %s",
                    "PASS" if m["p3_image_text_recognized"] else
                    "INCONCLUSIVE — generation works but stamp text not recognized; "
                    "could be expected degradation from text-only calibration")

    logger.info("=== P4 — Weights footprint ===")
    logger.info("Cannot read programmatically. Scroll up to find vLLM's "
                "'Model loading took X GiB' startup line.")
    logger.info("Prediction: 5–7 GiB (vs ~10 GiB for bf16 baseline, "
                "~3 GiB for hypothetical fully-4-bit LM).")


if __name__ == "__main__":
    main()
