"""Port of upstream model_free_ptq/gemma4_fp8_block.py to gemma-4-E4B-it.

Source:
    https://github.com/vllm-project/llm-compressor/blob/main/examples/model_free_ptq/gemma4_fp8_block.py

Why model_free_ptq instead of oneshot + calibration:
    Each prior attempt to use `oneshot` on this model hit a different
    incompatibility — first fx-tracing crashed on the wrapper's multimodal
    merge code, then on `*input_ids.shape` unpacking inside Gemma 4
    E-variants' per-layer-embeddings code, and finally on Gemma 4's
    KV-sharing across decoder layers (the `shared_kv_states` dict can't
    survive llmcompressor's sequential subgraph partitioning). Each fix
    surfaced the next problem one layer deeper.

    `model_free_ptq` is upstream's escape hatch for exactly this
    situation: it operates on the safetensors files directly, never
    instantiates the PyTorch model, never runs a forward pass, never
    traces. Pure per-tensor weight quantization. There's a Gemma 4
    example in the same examples/ tree that this file is a port of.

Adaptations from upstream:
  - model_id: gemma-4-31B-it -> gemma-4-E4B-it
  - --output-dir CLI arg
  - Auth preflight + verbose HF logging (Colab visibility)
  - Provenance JSON sidecar

Install on Colab L4 (or any 16+ GB GPU). llmcompressor and
compressed-tensors must come from git+main:

    pip install -q uv
    echo "transformers>=5.5,<6" > /tmp/overrides.txt
    uv pip install --system --override /tmp/overrides.txt \\
        "llm-bench-cc[quant] @ git+<your-repo-url>"
    uv pip install --system --override /tmp/overrides.txt \\
        "compressed-tensors @ git+https://github.com/neuralmagic/compressed-tensors.git" \\
        "llmcompressor @ git+https://github.com/vllm-project/llm-compressor.git"

Run:
    huggingface-cli login          # not required (E4B-it is public) but
                                   # the auth preflight calls whoami()
    pip install hf_transfer         # optional: faster shard downloads
    HF_HUB_ENABLE_HF_TRANSFER=1 python scripts/quantize_fp8.py \\
        --output-dir /tmp/llm-bench-cc/quant/gemma-4-E4B-it-FP8_BLOCK

Output layout:
    <output-dir>/
        config.json, tokenizer.*, processor_config.json, *.safetensors
        quant_recipe.json   # provenance: git SHA, scheme, source
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import subprocess
from pathlib import Path

logger = logging.getLogger("quantize_fp8")

# Scheme + ignore patterns. Scheme is verbatim from upstream
# gemma4_fp8_block.py. Ignore list adds `re:.*audio.*` because the E-variants
# have an audio tower with Conv1d weights (e.g.
# `model.audio_tower.layers.*.lconv1d.depthwise_conv1d.weight`, shape
# `(1024, 1, 5)`) — model_free_ptq's validator only accepts 2D Linear
# weights and crashes on the conv tensors. Upstream's 31B example doesn't
# hit this because that variant doesn't ship audio weights.
SCHEME = "FP8_BLOCK"
IGNORE = ["re:.*vision.*", "re:.*audio.*", "lm_head", "re:.*embed_tokens.*"]


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
    tqdm progress bars, which makes long downloads look hung."""
    for name in ("huggingface_hub", "transformers", "datasets", "filelock"):
        logging.getLogger(name).setLevel(logging.INFO)


def _patch_transformers_torch_init_functions() -> None:
    """transformers>=5.5 dropped the public `TORCH_INIT_FUNCTIONS` mapping
    that `llmcompressor.utils.dev.skip_weights_initialize` imports at
    module load. `model_free_ptq` never instantiates a PyTorch model, but
    the import chain `from llmcompressor import model_free_ptq` still
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
    parser.add_argument("--max-workers", type=int, default=8,
                        help="Parallel safetensors-shard processing workers (upstream default 8).")
    parser.add_argument("--device", default="cuda:0",
                        help="Device used to accelerate per-tensor quantization (upstream default cuda:0).")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    _enable_verbose_hf_logging()

    # Must run BEFORE the llmcompressor import — llmcompressor.utils.dev
    # imports `TORCH_INIT_FUNCTIONS` at top-level, which transformers>=5.5
    # dropped.
    _patch_transformers_torch_init_functions()

    from llmcompressor import model_free_ptq

    args.output_dir.mkdir(parents=True, exist_ok=True)
    repo_root = Path(__file__).resolve().parent.parent

    # [1/2] Auth preflight.
    logger.info("[1/2] Verifying HF auth ...")
    _check_auth(args.model_id)

    # [2/2] Run model-free PTQ. This downloads the safetensors shards (if
    # not already cached), quantizes weights per-tensor on `device`, and
    # writes new safetensors to `save_directory`. No PyTorch model is
    # ever instantiated and no forward pass runs.
    logger.info("[2/2] Running model_free_ptq (scheme=%s) on %s -> %s",
                SCHEME, args.model_id, args.output_dir)
    model_free_ptq(
        model_stub=args.model_id,
        save_directory=str(args.output_dir),
        scheme=SCHEME,
        ignore=IGNORE,
        max_workers=args.max_workers,
        device=args.device,
    )

    recipe_payload = {
        "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "git_sha": _git_sha(repo_root),
        "model_id": args.model_id,
        "scheme": SCHEME,
        "ignore_patterns": IGNORE,
        "entrypoint": "llmcompressor.model_free_ptq",
        "based_on": "examples/model_free_ptq/gemma4_fp8_block.py @ vllm-project/llm-compressor",
    }
    with (args.output_dir / "quant_recipe.json").open("w") as f:
        json.dump(recipe_payload, f, indent=2)
    logger.info("Done. Quantized checkpoint + quant_recipe.json written to %s", args.output_dir)


if __name__ == "__main__":
    main()
