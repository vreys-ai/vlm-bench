"""Produces an FP8_DYNAMIC checkpoint of gemma-4-E4B-it via llmcompressor
oneshot. Data-free: no calibration set, no forward pass for activation
scales — activations are quantized dynamically per-token at inference,
weights are quantized statically per-channel from their own values.

Sibling of:
  - quantize_fp8.py:        FP8_BLOCK via model_free_ptq (static block-FP8
                            weights, no activation quantization)
  - quantize_cyankiwi.py:   W4A16 via oneshot + calibration
                            (int4 weights, bf16 activations, observer-fitted)
  - quantize_w4a16.py:      W4A16 via model_free_ptq (data-free RTN int4)

Why FP8_DYNAMIC (vs the FP8_BLOCK that quantize_fp8.py produces):
    FP8_BLOCK is a weight-only scheme: weights compress to FP8 in blocks,
    activations stay in bf16, the matmul runs as FP8×bf16. FP8_DYNAMIC is
    a W8A8 scheme: weights are static per-channel FP8, activations are
    dynamic per-token FP8 (scales computed at runtime from each batch's
    activations), and the matmul runs as FP8×FP8 on Ada/Hopper FP8 tensor
    cores. The two are different points on the speed/accuracy curve —
    FP8_DYNAMIC trades a small accuracy hit (per-token activation
    quantization) for a real arithmetic-intensity win at inference.

Why oneshot + QuantizationModifier (vs model_free_ptq):
    The vLLM FP8 docs prescribe this exact entrypoint
    (https://docs.vllm.ai/en/latest/features/quantization/fp8/). FP8_DYNAMIC
    is not a model_free_ptq scheme — model_free_ptq operates on safetensors
    files and can only emit weight-only quantization metadata. FP8_DYNAMIC
    needs an `input_activations` block in the saved config so the runtime
    knows to compute activation scales per-token, which only oneshot
    produces.

Why no calibration data:
    Weight scales come from the weights themselves (per-channel
    absmax/observer). Activation scales are computed at inference time
    from each forward pass — there is nothing to calibrate against
    offline. `oneshot(model, recipe)` is called with no `dataset=`.

Hardware:
    FP8 compute requires NVIDIA compute capability >= 8.9 (Ada Lovelace,
    Hopper). L4 is Ada Lovelace (8.9) — the project's target deploy GPU.
    The quantization step itself only needs enough VRAM to hold the bf16
    weights; no calibration forward passes run.

Install on Colab L4 (or any 16+ GB GPU). Same verified-working sequence
as quantize_cyankiwi.py (2026-05-13):

    pip install -q uv
    echo "transformers>=5.5,<6" > /tmp/transformers_override.txt
    uv pip install --system --override /tmp/transformers_override.txt \\
        "llm-bench-cc[quant] @ git+https://github.com/vreys-ai/vlm-bench@main"
    uv pip install --system --override /tmp/transformers_override.txt \\
        "llmcompressor @ git+https://github.com/vllm-project/llm-compressor.git"
    echo "compressed-tensors==0.15.1a20260503" > /tmp/compressed-tensors_override.txt
    uv pip install --system --override /tmp/compressed-tensors_override.txt vllm

Run:
    huggingface-cli login          # not required (E4B-it is public) but
                                   # the auth preflight calls whoami()
    pip install hf_transfer         # optional: faster shard downloads
    HF_HUB_ENABLE_HF_TRANSFER=1 python scripts/quantize_dynamic_fp8.py \\
        --output-dir /tmp/llm-bench-cc/quant/gemma-4-E4B-it-FP8_DYNAMIC

Output layout:
    <output-dir>/
        config.json, tokenizer.*, processor_config.json,
        preprocessor_config.json, generation_config.json,
        chat_template.jinja, *.safetensors
        quant_recipe.json   # provenance: git SHA, scheme, source
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import subprocess
from pathlib import Path

logger = logging.getLogger("quantize_dynamic_fp8")

# FP8_DYNAMIC is a preset name understood by QuantizationModifier directly
# (https://docs.vllm.ai/en/latest/features/quantization/fp8/). The preset
# expands to: weights = static per-channel FP8, activations = dynamic
# per-token FP8. No need for the config_groups form here (used in
# quantize_cyankiwi.py for non-default knobs) — the preset is well-defined
# and matches the saved-config layout vLLM expects at load time.
SCHEME = "FP8_DYNAMIC"
TARGETS = ["Linear"]

# Same gemma-4-E-variant-specific ignore set as quantize_cyankiwi.py: the
# vision/audio towers contain non-Linear modules and modality-specific
# weights we don't want to FP8, and the PLE (per-layer-embedding) bits
# must stay in bf16 because they're shared across decoder layers in a way
# that the per-channel FP8 scaling can't represent cleanly. lm_head stays
# in bf16 per the docs' example.
IGNORE = [
    "re:.*vision_tower.*",          # all 16 vision encoder layers + patch_embedder
    "re:.*audio_tower.*",           # all 12 audio layers + subsample_conv + output_proj
    "re:.*per_layer_input_gate.*",  # all 42 LM-layer PLE input gates
    "re:.*per_layer_projection.*",  # all 42 LM-layer PLE projections
    "re:.*embed_vision.*",          # vision embedding_projection
    "re:.*embed_audio.*",           # audio embedding_projection
    "lm_head",
]


def _build_recipe_payload(args: argparse.Namespace, *, git_sha: str) -> dict:
    """Build the dict that's written to `<output-dir>/quant_recipe.json`.

    Provenance-only; no consumer reads this. Captures git SHA, scheme,
    ignore list, and the entrypoint string for downstream audit."""
    return {
        "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "git_sha": git_sha,
        "model_id": args.model_id,
        "entrypoint": "llmcompressor.oneshot+QuantizationModifier",
        "scheme": SCHEME,
        "targets": TARGETS,
        "ignore_patterns": IGNORE,
        "calibration": None,  # data-free; activations quantized at runtime
        "based_on": "https://docs.vllm.ai/en/latest/features/quantization/fp8/",
    }


def _load_model(args: argparse.Namespace):
    """Load gemma-4-E4B-it onto GPU for in-place weight quantization.
    `device_map='auto'` with a capped GPU budget spills the vision/audio
    towers to CPU on L4 24GB — they're in IGNORE and never touched.

    Imported lazily so `--help` and unit tests don't pay the transformers
    import cost."""
    import torch
    from transformers import AutoModelForImageTextToText

    return AutoModelForImageTextToText.from_pretrained(
        args.model_id,
        dtype=torch.bfloat16,
        device_map="auto",
        max_memory={0: args.max_gpu_memory, "cpu": "64GiB"},
    )


def _load_processor(args: argparse.Namespace):
    """Load the multimodal processor. Cheap; not factored for reuse, just
    here to keep model and processor loads side-by-side for readability."""
    from transformers import AutoProcessor
    return AutoProcessor.from_pretrained(args.model_id)


def _run_quantization(model) -> None:
    """Run QuantizationModifier oneshot in-place. No `dataset=` arg: the
    FP8_DYNAMIC scheme is data-free (per-channel weight quantization +
    runtime activation quantization). Matches the canonical example at
    https://docs.vllm.ai/en/latest/features/quantization/fp8/."""
    from llmcompressor import oneshot
    from llmcompressor.modifiers.quantization import QuantizationModifier

    recipe = QuantizationModifier(
        targets=TARGETS,
        scheme=SCHEME,
        ignore=IGNORE,
    )
    oneshot(model=model, recipe=recipe)


def _save_artifact_and_recipe(model, processor, args: argparse.Namespace) -> None:
    """Save the quantized model + processor + provenance JSON to
    args.output_dir. `save_compressed=True` writes FP8-packed safetensors
    with the FP8_DYNAMIC quantization_config block (including the
    `input_activations` entry that tells vLLM to compute per-token
    activation scales at runtime)."""
    repo_root = Path(__file__).resolve().parent.parent

    logger.info("Saving compressed checkpoint to %s", args.output_dir)
    model.save_pretrained(str(args.output_dir), save_compressed=True)
    processor.save_pretrained(str(args.output_dir))

    payload = _build_recipe_payload(args=args, git_sha=_git_sha(repo_root))
    with (args.output_dir / "quant_recipe.json").open("w") as f:
        json.dump(payload, f, indent=2)
    logger.info("Wrote provenance to %s/quant_recipe.json", args.output_dir)


def _git_sha(repo_root: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"], text=True
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _enable_verbose_hf_logging() -> None:
    """Bump HF-stack loggers to INFO so each shard download lands in cell
    output as a normal log line. Colab notebooks routinely fail to render
    tqdm progress bars, which makes long downloads look hung.

    Pin httpx/httpcore to WARNING: huggingface_hub uses httpx under the hood,
    and at INFO httpx emits one line per HTTP request (`HTTP/1.1 200 OK`).
    With dozens of shard range-reads per download that drowns out the actual
    HF-stack progress lines we DO want to see."""
    for name in ("huggingface_hub", "transformers", "datasets", "filelock"):
        logging.getLogger(name).setLevel(logging.INFO)
    for name in ("httpx", "httpcore"):
        logging.getLogger(name).setLevel(logging.WARNING)


def _patch_transformers_torch_init_functions() -> None:
    """transformers>=5.5 dropped the public `TORCH_INIT_FUNCTIONS` mapping
    that `llmcompressor.utils.dev.skip_weights_initialize` imports at
    module load. The import chain `from llmcompressor import oneshot`
    pulls in `llmcompressor.utils.dev`, so the shim must be installed
    *before* the llmcompressor import. Reconstruct the mapping from
    `torch.nn.init`. Idempotent."""
    import torch.nn as nn
    import transformers.modeling_utils as tmu

    if hasattr(tmu, "TORCH_INIT_FUNCTIONS"):
        return
    tmu.TORCH_INIT_FUNCTIONS = {
        "uniform_": nn.init.uniform_,
        "normal_": nn.init.normal_,
        "trunc_normal_": nn.init.trunc_normal_,
        "constant_": nn.init.constant_,
        "xavier_uniform_": nn.init.xavier_uniform_,
        "xavier_normal_": nn.init.xavier_normal_,
        "kaiming_uniform_": nn.init.kaiming_uniform_,
        "kaiming_normal_": nn.init.kaiming_normal_,
        "uniform": nn.init.uniform,
        "normal": nn.init.normal,
        "xavier_uniform": nn.init.xavier_uniform,
        "xavier_normal": nn.init.xavier_normal,
        "kaiming_uniform": nn.init.kaiming_uniform,
        "kaiming_normal": nn.init.kaiming_normal,
    }


def _check_auth(model_id: str) -> None:
    """Verify HF auth before any download. gemma-4-E4B-it is currently
    *not* gated, but a misconfigured token can still manifest as a silent
    retry loop — surface it as an immediate, actionable error."""
    from huggingface_hub import HfApi

    api = HfApi()
    try:
        info = api.whoami()
        logger.info("HF auth OK (user=%s)", info.get("name") or "<unknown>")
    except Exception as e:  # noqa: BLE001 — whoami can raise many error types
        raise RuntimeError(
            f"HF auth check failed ({type(e).__name__}: {e}). Run "
            f"`huggingface-cli login` if you need authenticated access for {model_id}."
        ) from e


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--model-id", default="google/gemma-4-E4B-it")
    parser.add_argument("--output-dir", required=True, type=Path,
                        help="Where to save the compressed checkpoint (local SSD recommended).")
    parser.add_argument("--max-gpu-memory", default="22GiB",
                        help="Cap on GPU residency to leave headroom for activations on L4 24GB.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    _enable_verbose_hf_logging()

    # Must run BEFORE the llmcompressor import — llmcompressor.utils.dev
    # imports `TORCH_INIT_FUNCTIONS` at top-level, which transformers>=5.5
    # dropped.
    _patch_transformers_torch_init_functions()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # [1/4] Auth preflight.
    logger.info("[1/4] Verifying HF auth ...")
    _check_auth(args.model_id)

    # [2/4] Load model + processor. No calibration dataset — FP8_DYNAMIC
    # is data-free (per-channel weight scales from the weights themselves,
    # per-token activation scales computed at inference time).
    logger.info("[2/4] Loading model + processor ...")
    model = _load_model(args)
    processor = _load_processor(args)

    # [3/4] Run QuantizationModifier oneshot with scheme=FP8_DYNAMIC.
    # No `dataset=` arg — see _run_quantization docstring.
    logger.info("[3/4] Running QuantizationModifier oneshot (scheme=%s) ...", SCHEME)
    _run_quantization(model)

    # [4/4] Save compressed checkpoint + processor + provenance.
    logger.info("[4/4] Saving compressed checkpoint and provenance ...")
    _save_artifact_and_recipe(model, processor, args)

    logger.info("Done. Entrypoint: llmcompressor.oneshot+QuantizationModifier(scheme=%s)", SCHEME)
    logger.info("      Output:    %s", args.output_dir)


if __name__ == "__main__":
    main()
