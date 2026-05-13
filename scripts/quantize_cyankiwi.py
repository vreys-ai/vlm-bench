"""Produces a calibrated W4A16 checkpoint matching the cyankiwi
gemma-4-E4B-it-AWQ-INT4 community artifact's observable config.

Sibling of quantize_fp8.py (FP8_BLOCK via model_free_ptq) and
quantize_w4a16.py (data-free RTN W4A16 via model_free_ptq). Each
quantization scheme is its own script for the same reason as those
two: each is a distinct provenance trail with a distinct `quant_recipe.json`
and a distinct set of failure modes. cyankiwi uses oneshot + calibration
(not model_free_ptq), so the entire pipeline differs from the other two —
see "Why oneshot + calibration" below.

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

logger = logging.getLogger("quantize_cyankiwi")

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


def _build_recipe_payload(
    args: argparse.Namespace, *, entrypoint_used: str, git_sha: str
) -> dict:
    """Build the dict that's written to `<output-dir>/quant_recipe.json`.

    The eval pipeline reads `entrypoint` to know which modifier produced
    the checkpoint (GPTQ vs observer-only). Calibration params are
    captured verbatim from the CLI args so the knob sweep is traceable.
    """
    return {
        "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "git_sha": git_sha,
        "model_id": args.model_id,
        "entrypoint": entrypoint_used,
        "scheme_kwargs": {
            "num_bits": 4,
            "group_size": args.group_size,
            "strategy": "group",
            "symmetric": args.symmetric,
            "observer": args.observer,
            "actorder": None,
            "dynamic": False,
        },
        "ignore_patterns": IGNORE,
        "calibration": {
            "dataset": args.calibration_dataset,
            "split": args.calibration_split,
            "num_samples": args.num_calibration_samples,
            "max_seq_length": args.max_seq_length,
            "seed": args.seed,
        },
        "gptq_params": {
            "block_size": args.gptq_block_size,
            "dampening_frac": args.gptq_dampening_frac,
        },
        "based_on": "https://huggingface.co/cyankiwi/gemma-4-E4B-it-AWQ-INT4/blob/main/config.json",
    }


def _load_model(args: argparse.Namespace):
    """Load gemma-4-E4B-it for calibration. `device_map='auto'` with a
    capped GPU budget spills the vision/audio towers to CPU on L4 24GB —
    they're in IGNORE and never touched during LM calibration anyway.

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


def _build_calibration_dataset(args: argparse.Namespace, processor):
    """Load + shuffle + select + chat-template-format the calibration set.

    Returns a HF Dataset of dicts with a single `text` key. `oneshot`
    takes this directly via its `dataset=` arg; it handles tokenization
    and packing internally."""
    from datasets import load_dataset

    logger.info("Loading calibration dataset %s split=%s",
                args.calibration_dataset, args.calibration_split)
    raw = load_dataset(args.calibration_dataset, split=args.calibration_split)

    text_column, needs_template = _pick_calibration_column(list(raw.column_names))
    logger.info("Picked calibration column %r (chat-template=%s)",
                text_column, needs_template)

    raw = raw.shuffle(seed=args.seed).select(range(args.num_calibration_samples))

    tokenizer = processor.tokenizer

    def _format(example):
        text = example[text_column]
        if needs_template:
            text = tokenizer.apply_chat_template(
                [{"role": "user", "content": text}],
                tokenize=False, add_generation_prompt=False,
            )
        return {"text": text}

    return raw.map(_format, remove_columns=raw.column_names)


def _config_groups_from_args(args: argparse.Namespace) -> dict:
    """Build a `config_groups` dict for GPTQModifier / QuantizationModifier.

    Why `config_groups=` instead of `scheme=`: GPTQModifier's `scheme=`
    parameter validates as either a preset name string ("W4A16") or a
    `{preset_name: targets}` dict — it rejects custom QuantizationArgs
    dicts at runtime (`Value error, scheme must either be a preset scheme
    name or a dictionary of preset scheme names`). Since we need
    non-default knobs (group_size=32, observer="mse", asymmetric), the
    `config_groups` parameter is the unambiguous path: it takes
    `{group_name: QuantizationScheme}` with explicit weights args. This
    also matches cyankiwi's saved config.json structure byte-for-byte
    (their `quantization_config.config_groups.group_0`).

    `targets` lives inside each scheme when using `config_groups`, not at
    the modifier level."""
    return {
        "group_0": {
            "targets": TARGETS,
            "weights": {
                "num_bits": 4,
                "type": "int",
                "symmetric": args.symmetric,
                "strategy": "group",
                "group_size": args.group_size,
                "observer": args.observer,
                "actorder": None,
                "dynamic": False,
            },
            "input_activations": None,
            "output_activations": None,
        }
    }


def _run_gptq(model, calib_ds, args: argparse.Namespace) -> str:
    """Run GPTQModifier oneshot. Mutates `model` in place; returns the
    entrypoint string for the recipe payload."""
    from llmcompressor import oneshot
    from llmcompressor.modifiers.quantization import GPTQModifier

    recipe = GPTQModifier(
        config_groups=_config_groups_from_args(args),
        ignore=IGNORE,
        block_size=args.gptq_block_size,
        dampening_frac=args.gptq_dampening_frac,
    )
    oneshot(
        model=model,
        dataset=calib_ds,
        recipe=recipe,
        max_seq_length=args.max_seq_length,
        num_calibration_samples=args.num_calibration_samples,
    )
    return "llmcompressor.oneshot+GPTQModifier"


def _run_observer_only(model, calib_ds, args: argparse.Namespace):
    """Fallback path. Runs QuantizationModifier oneshot on the caller's
    model in place — no Hessian, just observer-fitted scales from
    calibration activations. Returns (model, entrypoint_string).

    Why in-place (not a fresh rebuild as the spec originally called for):
    on L4 24 GB the bf16 model fills ~15 GiB. Loading a second copy
    doesn't fit, and `del model; gc.collect(); torch.cuda.empty_cache()`
    after a failed GPTQ run does not reliably release VRAM — the
    llmcompressor session retains back-refs to module submodules via
    instrumentation hooks the failed modifier registered, so the GC
    doesn't see the model as garbage.

    Safety of in-place reuse: at every marker-matched GPTQ failure point
    (fx TraceError, num_kv_shared_layers, sequential pipeline state),
    the crash happens before the modifier's `execute()` phase runs — no
    weights have been quantized or mutated. The model's bf16 weights
    are still pristine. `oneshot()` calls `session.reset()` as its
    first step, which clears any hooks GPTQ-attempt-1 left behind
    before QuantizationModifier-attempt-2 initializes. Verified on
    Gemma 4 E-variants 2026-05-13."""
    from llmcompressor import oneshot
    from llmcompressor.modifiers.quantization import QuantizationModifier

    recipe = QuantizationModifier(
        config_groups=_config_groups_from_args(args),
        ignore=IGNORE,
    )
    oneshot(
        model=model,
        dataset=calib_ds,
        recipe=recipe,
        max_seq_length=args.max_seq_length,
        num_calibration_samples=args.num_calibration_samples,
    )
    return model, "llmcompressor.oneshot+QuantizationModifier"


def _save_artifact_and_recipe(
    model, processor, args: argparse.Namespace, *, entrypoint_used: str
) -> None:
    """Save the quantized model + processor + provenance JSON to
    args.output_dir. `save_compressed=True` writes packed int4
    safetensors with the cyankiwi-shaped quantization_config block.
    `processor.save_pretrained` writes preprocessor_config.json,
    chat_template.jinja, and tokenizer files — which is why the older
    `_pull_processor_aux_files` helper is no longer called."""
    repo_root = Path(__file__).resolve().parent.parent

    logger.info("Saving compressed checkpoint to %s", args.output_dir)
    model.save_pretrained(str(args.output_dir), save_compressed=True)
    processor.save_pretrained(str(args.output_dir))

    payload = _build_recipe_payload(
        args=args,
        entrypoint_used=entrypoint_used,
        git_sha=_git_sha(repo_root),
    )
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


# NOTE: kept but uncalled. The oneshot path saves the multimodal aux
# files via `processor.save_pretrained(output_dir)` in
# _save_artifact_and_recipe, which writes preprocessor_config.json
# (image processor), chat_template.jinja, and tokenizer files. The
# model_free_ptq sibling (scripts/quantize_w4a16.py) still needs this
# helper because it never instantiates a Processor object. Left here for
# reference and for any future model_free_ptq fallback path.
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
    parser.add_argument("--calibration-dataset", default="garage-bAInd/Open-Platypus",
                        help="HF Hub repo id for the calibration corpus (text-only is fine; "
                             "vision/audio towers are in IGNORE).")
    parser.add_argument("--calibration-split", default="train",
                        help="Dataset split to load (use 'train_sft' for ultrachat_200k, etc.).")
    parser.add_argument("--num-calibration-samples", type=int, default=128,
                        help="Smoke-sized default; bump to 512 for the published llmcompressor "
                             "reference. Drives wall-clock more than any other flag.")
    parser.add_argument("--max-seq-length", type=int, default=2048,
                        help="Per-sample token cap during calibration. Matches the "
                             "llmcompressor INT4 W4A16 example default.")
    parser.add_argument("--group-size", type=int, default=32,
                        help="cyankiwi value (default in llmcompressor is 128).")
    parser.add_argument("--observer", default="mse", choices=["mse", "minmax"],
                        help="cyankiwi value (default in llmcompressor is minmax).")
    sym_group = parser.add_mutually_exclusive_group()
    sym_group.add_argument("--symmetric", dest="symmetric", action="store_true",
                           help="Use symmetric quantization (cyankiwi is asymmetric).")
    sym_group.add_argument("--no-symmetric", dest="symmetric", action="store_false",
                           help="Use asymmetric quantization (cyankiwi value, default).")
    parser.set_defaults(symmetric=False)
    parser.add_argument("--gptq-block-size", type=int, default=128,
                        help="GPTQ Hessian block size (llmcompressor default).")
    parser.add_argument("--gptq-dampening-frac", type=float, default=0.01,
                        help="GPTQ Hessian dampening fraction (llmcompressor default).")
    parser.add_argument("--max-gpu-memory", default="22GiB",
                        help="Cap on GPU residency to leave headroom for activations on L4 24GB.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Controls calibration shuffle for reproducibility.")
    parser.add_argument("--skip-gptq", action="store_true",
                        help="Force the QuantizationModifier path directly. Use when GPTQ "
                             "is known to fail on this model variant.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    _enable_verbose_hf_logging()

    # Must run BEFORE the llmcompressor import — llmcompressor.utils.dev
    # imports `TORCH_INIT_FUNCTIONS` at top-level, which transformers>=5.5
    # dropped.
    _patch_transformers_torch_init_functions()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # [1/5] Auth preflight.
    logger.info("[1/5] Verifying HF auth ...")
    _check_auth(args.model_id)

    # [2/5] Load model + processor + calibration dataset.
    logger.info("[2/5] Loading model + processor + calibration dataset ...")
    model = _load_model(args)
    processor = _load_processor(args)
    calib_ds = _build_calibration_dataset(args, processor)

    # [3/5] Try GPTQModifier first (high-quality Hessian path), fall back
    # to QuantizationModifier (observer-only) on the known Gemma 4
    # E-variant failure modes — see _is_known_e_variant_failure for the
    # marker list and reference_llmcompressor_entrypoints memory note for
    # the underlying architecture reasons.
    if args.skip_gptq:
        logger.info("[3/5] --skip-gptq set — going straight to observer-only")
        model, entrypoint_used = _run_observer_only(model, calib_ds, args)
    else:
        logger.info("[3/5] Running GPTQModifier oneshot ...")
        import torch.fx
        try:
            entrypoint_used = _run_gptq(model, calib_ds, args)
        except (torch.fx.proxy.TraceError, RuntimeError, AttributeError) as e:
            if not _is_known_e_variant_failure(e):
                raise
            logger.warning(
                "GPTQModifier failed on E-variant (%s: %s) — retrying observer-only in place",
                type(e).__name__, str(e)[:200],
            )
            # No rebuild: at every marker-matched failure point GPTQ
            # crashes before its execute() phase runs, so the model's
            # bf16 weights are still pristine. oneshot's session.reset()
            # clears any hooks the failed GPTQModifier left behind before
            # QuantizationModifier initializes. See _run_observer_only's
            # docstring for the safety argument and the L4 budget reason.
            model, entrypoint_used = _run_observer_only(model, calib_ds, args)

    # [4/5] Save the compressed checkpoint + processor + provenance.
    logger.info("[4/5] Saving compressed checkpoint and provenance ...")
    _save_artifact_and_recipe(model, processor, args, entrypoint_used=entrypoint_used)

    # [5/5] Done.
    logger.info("[5/5] Done. Entrypoint used: %s", entrypoint_used)
    logger.info("       Output: %s", args.output_dir)


if __name__ == "__main__":
    main()
