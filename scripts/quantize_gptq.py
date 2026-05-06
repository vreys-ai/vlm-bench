"""Smoke GPTQ-W4A16 quantization of gemma-4-E4B-it via llmcompressor.

Stage 1 Tier B end-to-end scaffold. Goal: get llmcompressor's oneshot
running cleanly on the VLM with a tiny calibration set and produce a
saved compressed-tensors checkpoint. Once this works, realistic
calibration (size, sources, mix) gets layered on without revisiting the
plumbing.

Quantization scope: only the LLM subtree's Linears (`language_model.*`).
Vision tower, audio tower, embed_vision, embed_audio, and lm_head stay
in bf16 — same boundary the bnb Tier-A pipeline enforces, plumbed
through llmcompressor's `ignore` instead of `llm_int8_skip_modules`.

Calibration: HuggingFaceM4/Docmatix (zero-shot-exp/train), 32 rows.
Docmatix bundles its images in the parquet shards, so it loads without
the FSX-path hazards of the_cauldron[okvqa].

Pipeline: `sequential`, fx-tracing each decoder layer in isolation.
`basic` OOMs on L4 24 GB with Gemma 4 (every Linear's activations cached
at once); `sequential` only caches one layer at a time. Target class is
auto-detected from `language_model.model.layers[0]`.

Install on Colab L4 (or any 24 GB GPU). Both llmcompressor and
compressed-tensors must come from git+main (PyPI 0.10.0.2 / 0.15.0.1
lack the transformers-5.x shim and the `compressed_tensors.distributed`
submodule, respectively). llmcompressor's setup.py also pins transformers
<= 4.57, but Gemma 4 needs >= 5.5 — we resolve the conflict with uv's
`--override`:

    pip install -q uv
    echo "transformers>=5.5,<6" > /tmp/overrides.txt
    uv pip install --system --override /tmp/overrides.txt \\
        "llm-bench-cc[quant] @ git+<your-repo-url>"
    uv pip install --system --override /tmp/overrides.txt \\
        "compressed-tensors @ git+https://github.com/neuralmagic/compressed-tensors.git" \\
        "llmcompressor @ git+https://github.com/vllm-project/llm-compressor.git"

Runtime gotcha: transformers 5.5 dropped two APIs llmcompressor still
calls — `TORCH_INIT_FUNCTIONS` and `PreTrainedModel._get_no_split_modules`.
We patch them back before importing llmcompressor — see the two
`_patch_*` helpers below.

Run:
    python scripts/quantize_gptq.py --output-dir /tmp/llm-bench-cc/quant/gptq-w4a16

Output layout:
    <output-dir>/
        config.json, tokenizer.*, processor_config.json, *.safetensors  (sharded)
        quant_recipe.json   # provenance: git SHA, calibration source, scheme
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import subprocess
from pathlib import Path
from typing import Any

logger = logging.getLogger("quantize_gptq")

# Single-source smoke calibration. Docmatix's images are bundled in the
# parquet shards (no external FSX paths), so loads are deterministic and
# don't require dataset-host infrastructure. Schema (verified via
# datasets-server): `images: Sequence(Image)`, `texts: [{user, assistant,
# source}]` — same shape as the_cauldron, so swapping in realistic mixes
# later is a column-compatible drop-in.
CALIBRATION_DATASET = "HuggingFaceM4/Docmatix"
CALIBRATION_CONFIG = "zero-shot-exp"
CALIBRATION_SPLIT = "train"
DEFAULT_NUM_CALIBRATION_SAMPLES = 32


def _git_sha(repo_root: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"], text=True
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _load_calibration_rows(
    num_samples: int, seed: int, cache_dir: str | None
) -> list[dict[str, Any]]:
    """Pull `num_samples` shuffled rows from Docmatix."""
    from datasets import load_dataset

    logger.info(
        "Loading calibration source %s [%s] split=%s -> %d rows",
        CALIBRATION_DATASET, CALIBRATION_CONFIG, CALIBRATION_SPLIT, num_samples,
    )
    ds = load_dataset(
        CALIBRATION_DATASET,
        name=CALIBRATION_CONFIG,
        split=CALIBRATION_SPLIT,
        cache_dir=cache_dir,
    )
    ds = ds.shuffle(seed=seed).select(range(num_samples))
    return list(ds)


def _row_to_messages(row: dict[str, Any]) -> tuple[Any, str]:
    """Extract (PIL image, user-text) from a Docmatix row.

    Docmatix rows have `images: [PIL, ...]` and
    `texts: [{"user": q, "assistant": a, "source": s}, ...]`. Calibration
    only needs realistic prompts, so we use the first turn's user message.
    """
    image = row["images"][0]
    if hasattr(image, "convert"):
        image = image.convert("RGB")
    user_text = row["texts"][0].get("user") or "Describe this image."
    return image, user_text


def _preprocess_for_calibration(rows, processor, max_seq_length: int):
    """Render each row through the processor's chat template. Returns a list
    of dicts of Python-native values (lists, not tensors) so it can be
    wrapped in `datasets.Dataset.from_list` — llmcompressor's oneshot
    pipeline runs its calibration data through `dataset_manager`, which
    calls `.column_names` on the input. Caller is expected to do
    `Dataset.from_list(examples).with_format("torch")`; the data_collator
    then receives torch tensors per example.

    We pass through *every* tensor field the processor returns, not just
    input_ids/attention_mask/pixel_values. Gemma 4's processor returns
    `image_position_ids` which the model needs to locate image tokens —
    dropping it makes vision_tower's forward crash with a None comparison.
    """
    examples = []
    for row in rows:
        image, user_text = _row_to_messages(row)
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": user_text},
            ],
        }]
        inputs = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        seq_len = inputs["input_ids"].shape[-1]
        truncate_to = min(seq_len, max_seq_length)
        ex: dict[str, Any] = {}
        for k, v in inputs.items():
            if not hasattr(v, "tolist") or not hasattr(v, "dim"):
                continue
            # processor returns shape [1, ...] for a single example.
            if v.dim() > 0:
                v = v[0]
            # Truncate any 1D per-token field (input_ids, attention_mask,
            # image_position_ids, token_type_ids, ...) to keep them aligned.
            # Multi-dim fields like pixel_values pass through unchanged.
            if v.dim() == 1 and v.shape[0] == seq_len:
                v = v[:truncate_to]
            ex[k] = v.tolist()
        examples.append(ex)
    return examples


def _data_collator(batch):
    # llmcompressor calls the collator with batch_size=1 in oneshot mode.
    assert len(batch) == 1, f"Expected batch_size=1 for GPTQ calibration, got {len(batch)}"
    return {k: v.unsqueeze(0) if v.dim() < 4 else v for k, v in batch[0].items()}


def _detect_decoder_layer_class(model, llm_subtree: str = "language_model") -> str:
    """Return the class name of the repeated decoder layer under the LLM
    subtree. Used as the sequential pipeline's `sequential_targets`.

    The fx-based sequential pipeline only fx-traces these target modules
    in isolation — everything outside (vision_tower, multimodal merge)
    just runs as a regular forward pass. Getting the class name right is
    the difference between tracing one ~200-line decoder layer (works)
    and tracing the whole VLM (fails with proxy iteration errors).
    """
    import torch.nn as nn

    # VLM wrappers like Gemma4ForConditionalGeneration nest sub-models
    # under `.model.` (the inner Gemma4Model holds vision_tower,
    # language_model, multi_modal_projector). Pure text models put layers
    # at `model.layers`. Try both nesting levels in order of specificity.
    candidates = [
        f"model.{llm_subtree}.layers",
        f"model.{llm_subtree}.model.layers",
        f"{llm_subtree}.layers",
        f"{llm_subtree}.model.layers",
        "model.layers",
    ]
    for path in candidates:
        try:
            mod = model.get_submodule(path)
        except AttributeError:
            continue
        if isinstance(mod, nn.ModuleList) and len(mod) > 0:
            cls_name = mod[0].__class__.__name__
            logger.info(
                "Found decoder layers at %r: class=%s, count=%d",
                path, cls_name, len(mod),
            )
            return cls_name
    raise RuntimeError(
        f"Could not locate decoder layers under '{llm_subtree}' subtree. "
        f"Tried paths: {candidates}"
    )


def _patch_transformers_for_llmcompressor() -> None:
    """Install backwards-compat shims on transformers 5.x that llmcompressor
    still calls into. Idempotent. See reference_llmcompressor_install
    memory entry for context."""
    _patch_transformers_torch_init_functions()
    _patch_pretrained_model_get_no_split_modules()


def _patch_transformers_torch_init_functions() -> None:
    """transformers>=5.5 dropped the public `TORCH_INIT_FUNCTIONS` mapping
    that llmcompressor.utils.dev.skip_weights_initialize imports.
    Reconstruct from torch.nn.init."""
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


def _patch_pretrained_model_get_no_split_modules() -> None:
    """transformers>=5.5 removed `PreTrainedModel._get_no_split_modules`,
    but the underlying `_no_split_modules` class attribute survives.
    llmcompressor still calls the getter, so we install a faithful
    backport that aggregates `_no_split_modules` across the top-level
    model and any nested PreTrainedModel children (relevant for VLMs like
    Gemma 4)."""
    from transformers import PreTrainedModel

    if hasattr(PreTrainedModel, "_get_no_split_modules"):
        return

    def _get_no_split_modules(self, device_map=None):
        no_split: set[str] = set()
        to_check = [self]
        while to_check:
            module = to_check.pop()
            if module.__class__.__name__ in no_split:
                continue
            if isinstance(module, PreTrainedModel) and module._no_split_modules is not None:
                no_split |= set(module._no_split_modules)
            to_check.extend(module.children())
        return list(no_split)

    PreTrainedModel._get_no_split_modules = _get_no_split_modules


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--model-id", default="google/gemma-4-E4B-it")
    parser.add_argument("--output-dir", required=True, type=Path,
                        help="Where to save the quantized checkpoint (local SSD recommended).")
    parser.add_argument("--num-calibration-samples", type=int,
                        default=DEFAULT_NUM_CALIBRATION_SAMPLES)
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cache-dir", default=None,
                        help="HF cache dir for both the model and calibration dataset.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")

    # Lazy imports — package's base path mustn't require these.
    import torch
    from transformers import AutoProcessor, AutoModelForImageTextToText

    _patch_transformers_for_llmcompressor()

    from llmcompressor import oneshot
    from llmcompressor.modifiers.quantization import GPTQModifier

    from llm_bench_cc.models import _enumerate_non_llm_linear_paths

    args.output_dir.mkdir(parents=True, exist_ok=True)
    repo_root = Path(__file__).resolve().parent.parent

    # 1. Build the ignore list (everything outside language_model.*) via meta-load.
    ignore = _enumerate_non_llm_linear_paths(args.model_id, cache_dir=args.cache_dir)
    logger.info("Ignore list contains %d non-LLM Linear paths", len(ignore))

    # 2. Load the bf16 model + processor.
    logger.info("Loading %s in bfloat16 ...", args.model_id)
    processor = AutoProcessor.from_pretrained(args.model_id, cache_dir=args.cache_dir)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model_id,
        dtype=torch.bfloat16,
        device_map="auto",
        low_cpu_mem_usage=True,
        cache_dir=args.cache_dir,
    )
    model.eval()

    # 3. Build calibration dataset.
    rows = _load_calibration_rows(
        num_samples=args.num_calibration_samples,
        seed=args.seed,
        cache_dir=args.cache_dir,
    )
    logger.info("Pulled %d calibration rows; preprocessing ...", len(rows))
    examples = _preprocess_for_calibration(rows, processor, max_seq_length=args.max_seq_length)

    from datasets import Dataset
    calibration_dataset = Dataset.from_list(examples).with_format("torch")
    logger.info(
        "Wrapped %d examples as Dataset (columns=%s)",
        len(calibration_dataset), calibration_dataset.column_names,
    )

    # 4. Run GPTQ via the sequential pipeline targeted at the LLM decoder layer.
    sequential_target = _detect_decoder_layer_class(model)
    recipe = GPTQModifier(
        targets=["Linear"],
        scheme="W4A16",
        ignore=ignore,
    )
    logger.info(
        "Running llmcompressor.oneshot (scheme=W4A16, pipeline=sequential, "
        "sequential_targets=[%s], n=%d, max_seq_length=%d) ...",
        sequential_target, len(examples), args.max_seq_length,
    )
    oneshot(
        model=model,
        dataset=calibration_dataset,
        recipe=recipe,
        max_seq_length=args.max_seq_length,
        num_calibration_samples=len(calibration_dataset),
        data_collator=_data_collator,
        pipeline="sequential",
        sequential_targets=[sequential_target],
    )

    # 5. Persist checkpoint and provenance.
    logger.info("Saving quantized checkpoint to %s", args.output_dir)
    model.save_pretrained(args.output_dir, save_compressed=True)
    processor.save_pretrained(args.output_dir)

    recipe_payload = {
        "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "git_sha": _git_sha(repo_root),
        "model_id": args.model_id,
        "scheme": "W4A16",
        "calibration": {
            "dataset": CALIBRATION_DATASET,
            "config": CALIBRATION_CONFIG,
            "split": CALIBRATION_SPLIT,
            "num_samples": len(examples),
            "seed": args.seed,
            "max_seq_length": args.max_seq_length,
        },
        "ignore_count": len(ignore),
        "llm_subtree": "language_model",
        "pipeline": "sequential",
        "sequential_targets": [sequential_target],
    }
    with (args.output_dir / "quant_recipe.json").open("w") as f:
        json.dump(recipe_payload, f, indent=2)
    logger.info("Done. quant_recipe.json written.")


if __name__ == "__main__":
    main()
