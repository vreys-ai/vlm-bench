"""Offline GPTQ-W4A16 quantization of gemma-4-E4B-it via llmcompressor.

Stage 1 Tier B. Quantizes only the LLM subtree (`language_model.*` Linears);
vision tower, audio tower, embed_vision, embed_audio, and lm_head are kept in
bf16 — same boundary the bnb Tier-A pipeline enforces, just plumbed through
llmcompressor's `ignore` instead of `llm_int8_skip_modules`.

Install on Colab L4 (or any 24 GB GPU). Both llmcompressor and compressed-tensors
must come from git+main (PyPI 0.10.0.2 / 0.15.0.1 lack the transformers-5.x
shim and the `compressed_tensors.distributed` submodule, respectively).
llmcompressor's setup.py also pins transformers <= 4.57, but Gemma 4 needs
>= 5.5 — we resolve the conflict with uv's `--override` (which the llmcompressor
README itself recommends), so the rest of the dep tree (loguru, pydantic,
auto-round, ...) still resolves normally:

    pip install -q uv
    echo "transformers>=5.5,<6" > /tmp/overrides.txt
    uv pip install --system --override /tmp/overrides.txt \\
        "llm-bench-cc[quant] @ git+<your-repo-url>"
    uv pip install --system --override /tmp/overrides.txt \\
        "compressed-tensors @ git+https://github.com/neuralmagic/compressed-tensors.git" \\
        "llmcompressor @ git+https://github.com/vllm-project/llm-compressor.git"

The override forces the resolver to use transformers 5.5+ regardless of any
declared upper bound. `--system` writes into the Colab Python (drop it if
you're in a venv).

Run:
    python scripts/quantize_gptq.py \
        --output-dir /tmp/llm-bench-cc/quant/gptq-w4a16 \
        --num-calibration-samples 256

Output layout:
    <output-dir>/
        config.json, tokenizer.*, processor_config.json, *.safetensors  (sharded)
        quant_recipe.json   # provenance: git SHA, calibration sources, scheme
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import random
import subprocess
from pathlib import Path
from typing import Any

logger = logging.getLogger("quantize_gptq")

# Calibration recipe: 60% the_cauldron / 40% Docmatix per stage-1 plan.
# Both are public, ungated, and image+text. Numbers are *target weights*; the
# script downscales to whatever total --num-calibration-samples requests.
DEFAULT_CALIBRATION_SOURCES = [
    # (hf_id, name/config, split, weight)
    ("HuggingFaceM4/the_cauldron", "okvqa", "train", 0.6),
    ("HuggingFaceM4/Docmatix",     "zero-shot-exp", "train", 0.4),
]


def _git_sha(repo_root: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"], text=True
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _load_calibration_rows(sources, total: int, seed: int) -> list[dict[str, Any]]:
    """Pull `total` rows from the source mix, weighted, shuffled, deterministic.

    Returns raw HF rows (image + text). The processor mapping happens later so
    we can log the exact text the model sees.
    """
    from datasets import load_dataset

    rng = random.Random(seed)
    out: list[dict[str, Any]] = []
    for hf_id, name, split, weight in sources:
        n = int(round(total * weight))
        kwargs: dict[str, Any] = {"split": split, "streaming": False}
        if name:
            kwargs["name"] = name
        logger.info("Loading calibration source %s [%s] split=%s -> %d rows", hf_id, name, split, n)
        ds = load_dataset(hf_id, **kwargs).shuffle(seed=seed).select(range(min(n, 5000)))
        # `5000` cap above keeps the in-memory shuffle cheap; we then sample
        # exactly `n` to honor the weighting precisely.
        idxs = rng.sample(range(len(ds)), k=min(n, len(ds)))
        for i in idxs:
            row = ds[i]
            out.append({"_source": hf_id, **row})
    rng.shuffle(out)
    return out[:total]


def _row_to_messages(row: dict[str, Any]) -> tuple[Any, str]:
    """Extract (PIL image, user-text) from a calibration row.

    the_cauldron: row has `images: [PIL]` and `texts: [{"user": q, "assistant": a}, ...]`.
    Docmatix:     row has `images: [PIL]` and `texts: [{"user": q, "assistant": a}, ...]`.
    Both share this shape; the only difference is content domain. Use the
    first turn's user message — calibration only needs realistic prompts, not
    full conversational context.
    """
    images = row.get("images") or [row.get("image")]
    image = images[0]
    if hasattr(image, "convert"):
        image = image.convert("RGB")
    texts = row.get("texts") or [{"user": row.get("question") or row.get("query") or ""}]
    user_text = texts[0].get("user") if isinstance(texts[0], dict) else str(texts[0])
    return image, (user_text or "Describe this image.")


def _preprocess_for_calibration(rows, processor, max_seq_length: int):
    """Render each row through the processor's chat template. Returns a list of
    dicts ready for llmcompressor's data_collator (one example per dict)."""
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
        # Truncate to max_seq_length on the token axis; calibration doesn't
        # need long contexts and over-long inputs blow up GPTQ activation memory.
        ids = inputs["input_ids"][0][:max_seq_length]
        ex: dict[str, Any] = {"input_ids": ids, "attention_mask": ids.new_ones(ids.shape)}
        if "pixel_values" in inputs:
            ex["pixel_values"] = inputs["pixel_values"][0]
        examples.append(ex)
    return examples


def _data_collator(batch):
    # llmcompressor calls the collator with batch_size=1 in oneshot mode.
    assert len(batch) == 1, f"Expected batch_size=1 for GPTQ calibration, got {len(batch)}"
    return {k: v.unsqueeze(0) if v.dim() < 4 else v for k, v in batch[0].items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-id", default="google/gemma-4-E4B-it")
    parser.add_argument("--output-dir", required=True, type=Path,
                        help="Where to save the quantized checkpoint (local SSD recommended).")
    parser.add_argument("--num-calibration-samples", type=int, default=256)
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bits", type=int, default=4)
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--scheme", default="W4A16",
                        help="llmcompressor preset; pinned to W4A16 for Tier B.")
    parser.add_argument("--cache-dir", default=None,
                        help="HF cache dir for both the model and calibration datasets.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")

    # Lazy imports — package's base path mustn't require these.
    import torch
    from transformers import AutoProcessor, AutoModelForImageTextToText
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
        DEFAULT_CALIBRATION_SOURCES, total=args.num_calibration_samples, seed=args.seed
    )
    logger.info("Pulled %d calibration rows; preprocessing ...", len(rows))
    examples = _preprocess_for_calibration(rows, processor, max_seq_length=args.max_seq_length)

    # 4. Run GPTQ.
    recipe = GPTQModifier(
        targets=["Linear"],
        scheme=args.scheme,
        ignore=ignore,
    )
    logger.info("Running llmcompressor.oneshot (scheme=%s, n=%d, max_seq_length=%d) ...",
                args.scheme, len(examples), args.max_seq_length)
    oneshot(
        model=model,
        dataset=examples,
        recipe=recipe,
        max_seq_length=args.max_seq_length,
        num_calibration_samples=len(examples),
        data_collator=_data_collator,
    )

    # 5. Persist checkpoint and provenance.
    logger.info("Saving quantized checkpoint to %s", args.output_dir)
    model.save_pretrained(args.output_dir, save_compressed=True)
    processor.save_pretrained(args.output_dir)

    recipe_payload = {
        "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "git_sha": _git_sha(repo_root),
        "model_id": args.model_id,
        "scheme": args.scheme,
        "bits": args.bits,
        "group_size": args.group_size,
        "calibration": {
            "sources": [
                {"hf_id": s[0], "name": s[1], "split": s[2], "weight": s[3]}
                for s in DEFAULT_CALIBRATION_SOURCES
            ],
            "num_samples": len(examples),
            "seed": args.seed,
            "max_seq_length": args.max_seq_length,
        },
        "ignore_count": len(ignore),
        "llm_subtree": "language_model",
    }
    with (args.output_dir / "quant_recipe.json").open("w") as f:
        json.dump(recipe_payload, f, indent=2)
    logger.info("Done. quant_recipe.json written.")


if __name__ == "__main__":
    main()
