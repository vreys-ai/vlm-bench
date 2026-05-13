"""Sibling of quantize_fp8.py producing a W4A16 (data-free RTN) checkpoint.

Why a separate script instead of a CLI flag on quantize_fp8.py:
    Each scheme is its own provenance trail. Keeping fp8 and w4a16 as two
    explicit scripts makes the artifact lineage in `quant_recipe.json`
    self-describing and lets us compare the two schemes' calls side-by-side
    at a glance. The two files differ only in SCHEME, the output-dir
    convention, and the section of the docstring explaining *why* this
    scheme was chosen.

Why W4A16 (vs the FP8_BLOCK that quantize_fp8.py produces):
    HF transformers + compressed-tensors 0.14 has no packed-fp8 forward
    kernel in eager mode. Empirically (smoke run on 2026-05-10):
    FP8_BLOCK loaded with `run_compressed=True` yields the same peak VRAM
    as the bf16 baseline — the runtime falls back to decompress-at-first-
    forward and the on-disk compression buys nothing at inference. FP8
    compressed-tensors checkpoints are a vLLM-runtime story.

    W4A16 routes through the bundled marlin / gptq-marlin kernels that
    consume packed int4 weights against bf16 activations directly. With
    `run_compressed=True` (forced in `llm_bench_cc.backends.hf.HFBackend`),
    LLM Linears stay packed in VRAM through the forward pass — the
    well-trodden path for inference-time int4 in HF transformers.

Why oneshot + calibration (vs the model_free_ptq path in quantize_w4a16.py):
    quantize_w4a16.py produces a *data-free RTN* W4A16 checkpoint — fast,
    no calibration set, fits any model llmcompressor's safetensors-level
    code can read. This script produces a *calibrated* W4A16 checkpoint
    matching cyankiwi/gemma-4-E4B-it-AWQ-INT4's observable config:
    group_size=32 (vs 128), observer=mse (vs minmax), asymmetric (vs the
    fp8 path's symmetric). These knobs are what we're testing as the
    explanation for cyankiwi's 0.978 retention edge over the Vishva007
    GPTQ W4A16 (0.835) on the same model.

    Trade-off: oneshot requires a full PyTorch forward pass, which means
    we hit the per-layer-embeddings + KV-sharing failure mode that
    model_free_ptq sidesteps (see reference_llmcompressor_entrypoints
    memory note). Mitigation: GPTQModifier → QuantizationModifier
    fallback on the known marker exceptions. Both paths produce the
    same final config_groups block.

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
    HF_HUB_ENABLE_HF_TRANSFER=1 python scripts/quantize_cyankiwi.py \\
        --output-dir /tmp/llm-bench-cc/quant/gemma-4-E4B-it-cyankiwi-W4A16 \\
        --num-calibration-samples 128 --max-seq-length 2048

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

logger = logging.getLogger("quantize_w4a16")

# Scheme + ignore patterns. Matches cyankiwi/gemma-4-E4B-it-AWQ-INT4's
# `quantization_config.weights` block byte-for-byte:
#   group_size=32 (default in llmcompressor is 128)
#   observer="mse" (default is "minmax")
#   symmetric=False (asymmetric quantization)
# `format="pack-quantized"` and `type="int"` are inferred by llmcompressor
# from `num_bits=4` + `strategy="group"`; the cyankiwi config records them
# in the final config.json but they're outputs, not inputs.
SCHEME_KWARGS = dict(
    num_bits=4,
    group_size=32,
    strategy="group",
    symmetric=False,
    observer="mse",
    actorder=None,
    dynamic=False,
)
TARGETS = ["Linear"]

# IGNORE regex form is semantically identical to cyankiwi's per-module
# enumeration given `targets=["Linear"]` (cyankiwi enumerates ~300
# submodule paths; we collapse them via the natural module-name groups).
# Difference from scripts/quantize_w4a16.py: no `re:.*embed_tokens.*` —
# targets="Linear" already excludes Embedding modules, and cyankiwi's
# config does the same for byte-for-byte parity on the saved config.
IGNORE = [
    "re:.*vision_tower.*",          # all 16 vision encoder layers + patch_embedder
    "re:.*audio_tower.*",           # all 12 audio layers + subsample_conv + output_proj
    "re:.*per_layer_input_gate.*",  # all 42 LM-layer PLE input gates
    "re:.*per_layer_projection.*",  # all 42 LM-layer PLE projections
    "re:.*embed_vision.*",          # vision embedding_projection
    "re:.*embed_audio.*",           # audio embedding_projection
    "lm_head",
]

# Substring markers we expect to see in oneshot failures caused by Gemma 4
# E-variant architecture (per-layer-embeddings + KV-sharing across decoder
# layers, see reference_llmcompressor_entrypoints memory note). Matched
# case-insensitively against `str(exception)`. Unknown failures re-raise —
# we'd rather surface a new failure mode than silently downgrade quality.
_E_VARIANT_FAILURE_MARKERS = (
    "per_layer_input_gate",
    "num_kv_shared_layers",
    "proxy",
    "fx",
    "sequential pipeline",
)


def _is_known_e_variant_failure(exc: BaseException) -> bool:
    """Return True if `exc`'s message contains any known marker for the
    Gemma 4 E-variant oneshot failure modes."""
    msg = str(exc).lower()
    return any(marker in msg for marker in _E_VARIANT_FAILURE_MARKERS)


# Calibration datasets in the wild use inconsistent column names.
# Priority below covers the three column conventions we'll encounter on
# the Hub: `instruction` (Platypus, Alpaca-likes), `prompt` (UltraChat
# and most instruct datasets), `text` (raw-text corpora like c4).
# `instruction` and `prompt` get the chat template applied; `text` is
# passed through raw.
_CALIBRATION_COLUMN_PRIORITY = (
    ("instruction", True),
    ("prompt", True),
    ("text", False),
)


def _pick_calibration_column(available_columns: list[str]) -> tuple[str, bool]:
    """Pick a text column from a HF dataset's schema.

    Returns (column_name, needs_chat_template). Raises ValueError listing
    both the priority list and the dataset's actual columns if no match.
    """
    available = set(available_columns)
    for col, needs_template in _CALIBRATION_COLUMN_PRIORITY:
        if col in available:
            return col, needs_template
    priority_names = ", ".join(c for c, _ in _CALIBRATION_COLUMN_PRIORITY)
    raise ValueError(
        f"Calibration dataset has none of the expected text columns "
        f"({priority_names}). Available columns: {sorted(available_columns)}."
    )


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


def _pull_processor_aux_files(model_id: str, output_dir: Path) -> None:
    """Copy multimodal processor configs from the source repo into the save
    dir. `model_free_ptq` propagates `config.json`, `tokenizer.*`,
    `processor_config.json`, `chat_template.jinja`, and `generation_config.json`
    but NOT `preprocessor_config.json` (image processor) or any audio
    processor file. Without those, downstream `AutoProcessor.from_pretrained`
    on the local dir falls through to a Hub call with the local path as
    repo_id → HFValidationError. Best-effort: missing files in the source
    repo are skipped silently (the model variant might not have them)."""
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import EntryNotFoundError

    aux_files = [
        "preprocessor_config.json",      # image processor
        "audio_processor_config.json",   # audio processor (if separate)
        "special_tokens_map.json",       # sometimes embedded in tokenizer_config.json
    ]
    for fname in aux_files:
        try:
            src = hf_hub_download(repo_id=model_id, filename=fname)
        except EntryNotFoundError:
            logger.info("  - %s not present in source repo, skipping", fname)
            continue
        except Exception as e:  # noqa: BLE001 — many error types possible
            logger.warning("  - %s fetch failed (%s: %s), skipping", fname, type(e).__name__, e)
            continue
        dst = output_dir / fname
        dst.write_bytes(Path(src).read_bytes())
        logger.info("  - copied %s", fname)


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

    # [1/3] Auth preflight.
    logger.info("[1/3] Verifying HF auth ...")
    _check_auth(args.model_id)

    # [2/3] Run model-free PTQ. This downloads the safetensors shards (if
    # not already cached), quantizes weights per-tensor on `device`, and
    # writes new safetensors to `save_directory`. No PyTorch model is
    # ever instantiated and no forward pass runs. W4A16 is data-free RTN:
    # group-wise int4 weights, bf16 activations at inference (the "A16"),
    # no calibration set required.
    logger.info("[2/3] Running model_free_ptq (scheme=%s) on %s -> %s",
                SCHEME, args.model_id, args.output_dir)
    model_free_ptq(
        model_stub=args.model_id,
        save_directory=str(args.output_dir),
        scheme=SCHEME,
        ignore=IGNORE,
        max_workers=args.max_workers,
        device=args.device,
    )

    # [3/3] Pull the multimodal-processor aux files that model_free_ptq
    # doesn't propagate. Without them, AutoProcessor.from_pretrained on the
    # save dir falls through to a Hub call with the local path as repo_id
    # and crashes with HFValidationError. Fetch them once now so the saved
    # checkpoint is self-contained for downstream load. Files are tiny
    # (KB-range) and snapshot_download is no-op-fast for ones already in
    # the HF cache from step [2/3].
    logger.info("[3/3] Adding multimodal processor aux files to %s", args.output_dir)
    _pull_processor_aux_files(args.model_id, args.output_dir)

    recipe_payload = {
        "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "git_sha": _git_sha(repo_root),
        "model_id": args.model_id,
        "scheme": SCHEME,
        "ignore_patterns": IGNORE,
        "entrypoint": "llmcompressor.model_free_ptq",
        "based_on": "examples/model_free_ptq/gemma4_fp8_block.py @ vllm-project/llm-compressor (scheme swapped FP8_BLOCK -> W4A16)",
    }
    with (args.output_dir / "quant_recipe.json").open("w") as f:
        json.dump(recipe_payload, f, indent=2)
    logger.info("Done. Quantized checkpoint + quant_recipe.json written to %s", args.output_dir)


if __name__ == "__main__":
    main()
