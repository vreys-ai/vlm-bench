# Calibrated W4A16 Quantization Script Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the body of `scripts/quantize_cyankiwi.py` (currently a `model_free_ptq` sibling) with an oneshot+calibration pipeline that produces a W4A16 checkpoint matching `cyankiwi/gemma-4-E4B-it-AWQ-INT4`'s observable config (group_size=32, observer=mse, asymmetric), with a GPTQ → observer-only fallback for known Gemma 4 E-variant failures.

**Architecture:** Single-script pipeline. Pure helpers (failure-marker matcher, calibration-column detection, recipe-payload builder) are unit-tested in `tests/test_quantize_cyankiwi.py`. Integration steps (model load, calibration dataloader build, oneshot run, save) are smoke-verified by a wiring run at the end. The script reuses four existing helpers from the current stub (`_check_auth`, `_enable_verbose_hf_logging`, `_patch_transformers_torch_init_functions`, `_git_sha`) and keeps `_pull_processor_aux_files` defined but uncalled (with a comment explaining why `processor.save_pretrained` supersedes it).

**Tech Stack:** Python 3 · argparse · `llmcompressor` (git+main) · `compressed-tensors` (git+main) · `transformers` ≥ 5.5 · `datasets` · `huggingface_hub` · `pytest`

**Spec:** `docs/superpowers/specs/2026-05-13-quantize-cyankiwi-design.md` (commit `b482f3b`)

---

## File Structure

- **Modify:** `scripts/quantize_cyankiwi.py` — replace body; keep the docstring frame, `_check_auth`, `_enable_verbose_hf_logging`, `_patch_transformers_torch_init_functions`, `_git_sha`, and `_pull_processor_aux_files` (latter kept but uncalled).
- **Create:** `tests/test_quantize_cyankiwi.py` — pytest tests for the three pure helpers; runs in the local dev environment without any GPU or llmcompressor install.

---

## Task 1: New CLI flag set + recipe constants

**Files:**
- Modify: `scripts/quantize_cyankiwi.py:60-90` (the `SCHEME = "W4A16"` / `IGNORE = [...]` block and the docstring "Run:" example)

This replaces the current `SCHEME` / `IGNORE` constants and updates the docstring's "Run:" example. We don't touch `main()` yet — that comes in Task 11. After this task the script still runs the old `model_free_ptq` flow; we're just staging the new recipe constants and updating the docs.

- [ ] **Step 1: Replace the scheme/ignore block with the calibrated-W4A16 form**

In `scripts/quantize_cyankiwi.py`, replace the entire block from `# Scheme + ignore patterns.` down through the `IGNORE = [...]` line (currently lines 72–80) with:

```python
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
```

- [ ] **Step 2: Update the docstring's "Run:" example to reflect the new flags**

In the module docstring (top of file), replace the lines:

```
    HF_HUB_ENABLE_HF_TRANSFER=1 python scripts/quantize_w4a16.py \\
        --output-dir /tmp/llm-bench-cc/quant/gemma-4-E4B-it-W4A16
```

with:

```
    HF_HUB_ENABLE_HF_TRANSFER=1 python scripts/quantize_cyankiwi.py \\
        --output-dir /tmp/llm-bench-cc/quant/gemma-4-E4B-it-cyankiwi-W4A16 \\
        --num-calibration-samples 128 --max-seq-length 2048
```

Also replace the entire "Why model_free_ptq instead of oneshot + calibration:" docstring section (currently lines 25–33) with this new section:

```
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
```

- [ ] **Step 3: Commit**

```bash
git add scripts/quantize_cyankiwi.py
git commit -m "quantize_cyankiwi: stage calibrated W4A16 recipe constants

Replace SCHEME/IGNORE with SCHEME_KWARGS/IGNORE matching cyankiwi's
observable config (g=32, mse, asym). Updated docstring; main() still runs
the inherited model_free_ptq flow — that's swapped in later tasks."
```

---

## Task 2: TDD — `_is_known_e_variant_failure` failure-marker matcher

**Files:**
- Create: `tests/test_quantize_cyankiwi.py`
- Modify: `scripts/quantize_cyankiwi.py` (add helper near top of module)

Pure helper, easily unit-testable. We test it before writing it.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_quantize_cyankiwi.py`:

```python
"""Unit tests for the pure helpers in scripts/quantize_cyankiwi.py.

These tests are intentionally narrow: they cover the three pure helpers
(failure-marker matching, calibration-column detection, recipe-payload
building) without importing llmcompressor or loading any model. The
end-to-end pipeline is verified by a smoke run in the implementation
plan, not here."""

from __future__ import annotations

import sys
from pathlib import Path

# Make scripts/ importable as a top-level package for these tests.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import quantize_cyankiwi as qc  # noqa: E402


class TestIsKnownEVariantFailure:
    def test_matches_per_layer_input_gate(self):
        exc = AttributeError(
            "module 'GemmaDecoderLayer' has no attribute 'per_layer_input_gate' "
            "during fx tracing"
        )
        assert qc._is_known_e_variant_failure(exc) is True

    def test_matches_num_kv_shared_layers(self):
        exc = RuntimeError(
            "Sequential pipeline state inconsistent: num_kv_shared_layers=18 "
            "but layer 24 has no preceding KV producer"
        )
        assert qc._is_known_e_variant_failure(exc) is True

    def test_matches_fx_proxy(self):
        exc = RuntimeError("torch.fx Proxy object has no attribute __mul__")
        assert qc._is_known_e_variant_failure(exc) is True

    def test_matches_fx_substring(self):
        exc = RuntimeError("fx tracing failed at gemma4 PLE access")
        assert qc._is_known_e_variant_failure(exc) is True

    def test_matches_sequential_pipeline(self):
        exc = RuntimeError("Sequential pipeline cannot dispatch hooked layer 18")
        assert qc._is_known_e_variant_failure(exc) is True

    def test_does_not_match_oom(self):
        exc = RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")
        assert qc._is_known_e_variant_failure(exc) is False

    def test_does_not_match_auth_failure(self):
        exc = RuntimeError("401 Unauthorized: token is invalid")
        assert qc._is_known_e_variant_failure(exc) is False

    def test_case_insensitive(self):
        exc = RuntimeError("Per_Layer_Input_Gate not found")
        assert qc._is_known_e_variant_failure(exc) is True
```

- [ ] **Step 2: Run tests to verify they all fail**

Run from repo root:

```bash
pytest tests/test_quantize_cyankiwi.py -v
```

Expected: 8 errors with `AttributeError: module 'quantize_cyankiwi' has no attribute '_is_known_e_variant_failure'`.

- [ ] **Step 3: Add the helper to the script**

In `scripts/quantize_cyankiwi.py`, immediately after the `IGNORE = [...]` block (added in Task 1), insert:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_quantize_cyankiwi.py -v
```

Expected: 8 passed.

- [ ] **Step 5: Commit**

```bash
git add tests/test_quantize_cyankiwi.py scripts/quantize_cyankiwi.py
git commit -m "quantize_cyankiwi: add _is_known_e_variant_failure marker matcher

Substring matcher gating the GPTQ → observer-only fallback. Unknown
failures re-raise. Tests verify positive matches (PLE, KV-shared, fx,
sequential pipeline) and negatives (OOM, auth) plus case-insensitivity."
```

---

## Task 3: TDD — calibration-column detection helper

**Files:**
- Modify: `tests/test_quantize_cyankiwi.py` (append tests)
- Modify: `scripts/quantize_cyankiwi.py` (append helper)

The dataloader needs to figure out which column of an arbitrary HF dataset holds the text we calibrate on. Priority order: `instruction` > `prompt` > `text`. Chat template applied for the first two, raw for `text`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_quantize_cyankiwi.py`:

```python
class TestPickCalibrationColumn:
    def test_priority_instruction_first(self):
        col, needs_template = qc._pick_calibration_column(
            available_columns=["text", "instruction", "prompt"]
        )
        assert col == "instruction"
        assert needs_template is True

    def test_priority_prompt_over_text(self):
        col, needs_template = qc._pick_calibration_column(
            available_columns=["text", "prompt"]
        )
        assert col == "prompt"
        assert needs_template is True

    def test_fallback_text_no_template(self):
        col, needs_template = qc._pick_calibration_column(
            available_columns=["text", "metadata"]
        )
        assert col == "text"
        assert needs_template is False

    def test_no_match_raises_with_actual_columns(self):
        with pytest.raises(ValueError, match=r"foo.*bar"):
            qc._pick_calibration_column(available_columns=["foo", "bar"])

    def test_no_match_error_lists_priority(self):
        with pytest.raises(ValueError, match=r"instruction.*prompt.*text"):
            qc._pick_calibration_column(available_columns=["foo"])
```

Also add `import pytest` at the top of the test file if it isn't already there.

- [ ] **Step 2: Run tests to verify they all fail**

```bash
pytest tests/test_quantize_cyankiwi.py::TestPickCalibrationColumn -v
```

Expected: 5 errors with `AttributeError: module 'quantize_cyankiwi' has no attribute '_pick_calibration_column'`.

- [ ] **Step 3: Add the helper to the script**

Append to `scripts/quantize_cyankiwi.py` (right after `_is_known_e_variant_failure`):

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_quantize_cyankiwi.py::TestPickCalibrationColumn -v
```

Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add tests/test_quantize_cyankiwi.py scripts/quantize_cyankiwi.py
git commit -m "quantize_cyankiwi: add _pick_calibration_column helper

Maps HF dataset columns to (column_name, needs_chat_template). Priority
instruction > prompt > text. Hard error lists actual columns and the
priority list when no match."
```

---

## Task 4: TDD — recipe-payload builder

**Files:**
- Modify: `tests/test_quantize_cyankiwi.py` (append tests)
- Modify: `scripts/quantize_cyankiwi.py` (append helper)

The `quant_recipe.json` provenance payload is computed from `argparse.Namespace` + the entrypoint string + a git SHA. Worth a unit test because (a) it's the artifact the eval pipeline reads to know which modifier produced the checkpoint, and (b) it's pure data manipulation, trivial to test.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_quantize_cyankiwi.py`:

```python
class TestBuildRecipePayload:
    def test_includes_all_required_fields(self):
        import argparse
        args = argparse.Namespace(
            model_id="google/gemma-4-E4B-it",
            calibration_dataset="garage-bAInd/Open-Platypus",
            calibration_split="train",
            num_calibration_samples=128,
            max_seq_length=2048,
            group_size=32,
            observer="mse",
            symmetric=False,
            gptq_block_size=128,
            gptq_dampening_frac=0.01,
            seed=42,
        )
        payload = qc._build_recipe_payload(
            args=args,
            entrypoint_used="llmcompressor.oneshot+GPTQModifier",
            git_sha="abc1234",
        )
        assert payload["model_id"] == "google/gemma-4-E4B-it"
        assert payload["entrypoint"] == "llmcompressor.oneshot+GPTQModifier"
        assert payload["git_sha"] == "abc1234"
        assert payload["scheme_kwargs"]["num_bits"] == 4
        assert payload["scheme_kwargs"]["group_size"] == 32
        assert payload["scheme_kwargs"]["observer"] == "mse"
        assert payload["scheme_kwargs"]["symmetric"] is False
        assert payload["ignore_patterns"] == qc.IGNORE
        assert payload["calibration"]["dataset"] == "garage-bAInd/Open-Platypus"
        assert payload["calibration"]["num_samples"] == 128
        assert payload["calibration"]["max_seq_length"] == 2048
        assert payload["calibration"]["seed"] == 42
        assert "timestamp_utc" in payload
        # Timestamp should be ISO-8601 with seconds precision (sortable).
        assert payload["timestamp_utc"].endswith("+00:00")
        assert "based_on" in payload
        assert "cyankiwi" in payload["based_on"]

    def test_reflects_cli_overrides(self):
        import argparse
        args = argparse.Namespace(
            model_id="google/gemma-4-E4B-it",
            calibration_dataset="HuggingFaceH4/ultrachat_200k",
            calibration_split="train_sft",
            num_calibration_samples=256,
            max_seq_length=1024,
            group_size=128,
            observer="minmax",
            symmetric=True,
            gptq_block_size=64,
            gptq_dampening_frac=0.05,
            seed=99,
        )
        payload = qc._build_recipe_payload(
            args=args,
            entrypoint_used="llmcompressor.oneshot+QuantizationModifier",
            git_sha="def5678",
        )
        assert payload["scheme_kwargs"]["group_size"] == 128
        assert payload["scheme_kwargs"]["observer"] == "minmax"
        assert payload["scheme_kwargs"]["symmetric"] is True
        assert payload["calibration"]["dataset"] == "HuggingFaceH4/ultrachat_200k"
        assert payload["calibration"]["split"] == "train_sft"
        assert payload["calibration"]["num_samples"] == 256
        assert payload["calibration"]["max_seq_length"] == 1024
        assert payload["calibration"]["seed"] == 99
        assert payload["entrypoint"] == "llmcompressor.oneshot+QuantizationModifier"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_quantize_cyankiwi.py::TestBuildRecipePayload -v
```

Expected: 2 errors with `AttributeError: module 'quantize_cyankiwi' has no attribute '_build_recipe_payload'`.

- [ ] **Step 3: Add the helper to the script**

Append to `scripts/quantize_cyankiwi.py` (right after `_pick_calibration_column`):

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_quantize_cyankiwi.py -v
```

Expected: all 15 tests pass (8 from Task 2 + 5 from Task 3 + 2 from this task).

- [ ] **Step 5: Commit**

```bash
git add tests/test_quantize_cyankiwi.py scripts/quantize_cyankiwi.py
git commit -m "quantize_cyankiwi: add _build_recipe_payload provenance helper

Builds the quant_recipe.json dict from argparse + entrypoint + git SHA.
Tests verify all required fields present, scheme_kwargs reflects CLI
overrides, and timestamp is ISO-8601 with timezone."
```

---

## Task 5: Replace argparser with full CLI flag set

**Files:**
- Modify: `scripts/quantize_cyankiwi.py:186-198` (the existing `argparse.ArgumentParser` block inside `main()`)

The current `main()` has 4 flags. We replace just the parser portion (not the rest of main() yet — that's swapped wholesale in Task 11).

- [ ] **Step 1: Replace the argparser block**

In `scripts/quantize_cyankiwi.py`, find the block (inside `main()`):

```python
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
```

Replace it with:

```python
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
```

- [ ] **Step 2: Quick sanity check that the parser is syntactically valid**

```bash
python scripts/quantize_cyankiwi.py --help 2>&1 | head -40
```

Expected: argparse usage banner listing all the new flags. Won't run `model_free_ptq` because `--help` exits before that. If you see a Python traceback, the parser block has a syntax error.

- [ ] **Step 3: Commit**

```bash
git add scripts/quantize_cyankiwi.py
git commit -m "quantize_cyankiwi: full CLI flag set for calibration knob sweep

Adds --calibration-dataset, --num-calibration-samples, --max-seq-length,
--group-size, --observer, --symmetric/--no-symmetric, --gptq-block-size,
--gptq-dampening-frac, --max-gpu-memory, --seed, --skip-gptq. Removes
--max-workers and --device (model_free_ptq-specific; new pipeline uses
device_map='auto' instead)."
```

---

## Task 6: Add `_load_model` factored helper

**Files:**
- Modify: `scripts/quantize_cyankiwi.py` (append helper near other private helpers)

Used in both step 3 (initial load) and step 6 (fallback rebuild) of the pipeline. Factored to guarantee identical kwargs in both spots.

- [ ] **Step 1: Add the helper**

Append to `scripts/quantize_cyankiwi.py` (right after `_build_recipe_payload`):

```python
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
```

- [ ] **Step 2: Commit**

No test for these — they're transformers-thin wrappers and Step 11's smoke run exercises them end-to-end.

```bash
git add scripts/quantize_cyankiwi.py
git commit -m "quantize_cyankiwi: factor _load_model / _load_processor helpers

_load_model is called twice (initial load + fallback rebuild after a
GPTQ crash) so the kwargs need to be identical in both places. Lazy
imports keep --help cheap."
```

---

## Task 7: Add `_build_calibration_dataset` helper

**Files:**
- Modify: `scripts/quantize_cyankiwi.py` (append helper)

Wires the column-detection helper (Task 3) into a real HF dataset, applies the chat template if needed, shuffles, and selects N samples. Returns an object oneshot can iterate.

- [ ] **Step 1: Add the helper**

Append to `scripts/quantize_cyankiwi.py` (right after `_load_processor`):

```python
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
```

- [ ] **Step 2: Commit**

```bash
git add scripts/quantize_cyankiwi.py
git commit -m "quantize_cyankiwi: add _build_calibration_dataset

Loads HF dataset, picks text column via _pick_calibration_column, applies
the chat template when the column is instruction/prompt, shuffles +
selects N samples. Returns a Dataset with a single 'text' key for
oneshot to tokenize."
```

---

## Task 8: Add `_run_gptq` driver

**Files:**
- Modify: `scripts/quantize_cyankiwi.py` (append helper)

The primary quantization path. Mutates `model` in place and returns the entrypoint label.

- [ ] **Step 1: Add the helper**

Append to `scripts/quantize_cyankiwi.py` (right after `_build_calibration_dataset`):

```python
def _scheme_kwargs_from_args(args: argparse.Namespace) -> dict:
    """Materialize SCHEME_KWARGS with the CLI-overridable knobs filled in."""
    return dict(
        num_bits=4,
        group_size=args.group_size,
        strategy="group",
        symmetric=args.symmetric,
        observer=args.observer,
        actorder=None,
        dynamic=False,
    )


def _run_gptq(model, calib_ds, args: argparse.Namespace) -> str:
    """Run GPTQModifier oneshot. Mutates `model` in place; returns the
    entrypoint string for the recipe payload."""
    from llmcompressor import oneshot
    from llmcompressor.modifiers.quantization import GPTQModifier

    recipe = GPTQModifier(
        targets=TARGETS,
        ignore=IGNORE,
        scheme=_scheme_kwargs_from_args(args),
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
```

- [ ] **Step 2: Commit**

```bash
git add scripts/quantize_cyankiwi.py
git commit -m "quantize_cyankiwi: add _run_gptq + _scheme_kwargs_from_args

_run_gptq is the primary quantization driver (GPTQ Hessian path).
_scheme_kwargs_from_args materializes the recipe dict from CLI args
so the GPTQ and observer-only paths share an identical scheme block."
```

---

## Task 9: Add `_run_observer_only` driver

**Files:**
- Modify: `scripts/quantize_cyankiwi.py` (append helper)

Fallback path. Takes `args` (not a model) because it rebuilds the model — GPTQ may have mutated weights before crashing.

- [ ] **Step 1: Add the helper**

Append to `scripts/quantize_cyankiwi.py` (right after `_run_gptq`):

```python
def _run_observer_only(calib_ds, args: argparse.Namespace):
    """Fallback path. Rebuilds the model fresh (GPTQ may have partially
    mutated weights before crashing), then runs QuantizationModifier
    oneshot — no Hessian, just observer-fitted scales from calibration
    activations. Returns (model, entrypoint_string)."""
    from llmcompressor import oneshot
    from llmcompressor.modifiers.quantization import QuantizationModifier

    logger.info("Rebuilding model fresh before observer-only fallback")
    model = _load_model(args)

    recipe = QuantizationModifier(
        targets=TARGETS,
        ignore=IGNORE,
        scheme=_scheme_kwargs_from_args(args),
    )
    oneshot(
        model=model,
        dataset=calib_ds,
        recipe=recipe,
        max_seq_length=args.max_seq_length,
        num_calibration_samples=args.num_calibration_samples,
    )
    return model, "llmcompressor.oneshot+QuantizationModifier"
```

- [ ] **Step 2: Commit**

```bash
git add scripts/quantize_cyankiwi.py
git commit -m "quantize_cyankiwi: add _run_observer_only fallback driver

Rebuilds model fresh (GPTQ may have partially mutated weights before
crashing) then runs QuantizationModifier oneshot. No Hessian, just
observer-fitted scales from calibration activations. Same scheme block
as the GPTQ path so the saved config is identical."
```

---

## Task 10: Add `_save_artifact_and_recipe` helper

**Files:**
- Modify: `scripts/quantize_cyankiwi.py` (append helper)

Writes the packed safetensors + the provenance JSON.

- [ ] **Step 1: Add the helper**

Append to `scripts/quantize_cyankiwi.py` (right after `_run_observer_only`):

```python
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
```

- [ ] **Step 2: Commit**

```bash
git add scripts/quantize_cyankiwi.py
git commit -m "quantize_cyankiwi: add _save_artifact_and_recipe

Saves packed safetensors + processor aux files + provenance JSON.
save_compressed=True writes the cyankiwi-shaped quantization_config
block; processor.save_pretrained supersedes the now-unused
_pull_processor_aux_files helper."
```

---

## Task 11: Replace `main()` with the oneshot pipeline

**Files:**
- Modify: `scripts/quantize_cyankiwi.py` — replace the body of `main()` starting after the argparser block (Task 5 left the parser in place)

This is the wiring task that activates everything from Tasks 6-10 and removes the inherited `model_free_ptq` call.

- [ ] **Step 1: Replace the post-argparser portion of `main()`**

After the `args = parser.parse_args()` line (which Task 5 left in place), replace everything from the next line through the end of `main()` with:

```python
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
        model, entrypoint_used = _run_observer_only(calib_ds, args)
    else:
        logger.info("[3/5] Running GPTQModifier oneshot ...")
        import torch.fx
        try:
            entrypoint_used = _run_gptq(model, calib_ds, args)
        except (torch.fx.proxy.TraceError, RuntimeError, AttributeError) as e:
            if not _is_known_e_variant_failure(e):
                raise
            logger.warning(
                "GPTQModifier failed on E-variant (%s: %s) — retrying observer-only",
                type(e).__name__, str(e)[:200],
            )
            model, entrypoint_used = _run_observer_only(calib_ds, args)

    # [4/5] Save the compressed checkpoint + processor + provenance.
    logger.info("[4/5] Saving compressed checkpoint and provenance ...")
    _save_artifact_and_recipe(model, processor, args, entrypoint_used=entrypoint_used)

    # [5/5] Done.
    logger.info("[5/5] Done. Entrypoint used: %s", entrypoint_used)
    logger.info("       Output: %s", args.output_dir)
```

Also remove the now-unused `_pull_processor_aux_files` call from main() (it's gone after this replacement) — but leave the function definition in the file with a leading comment explaining why it's kept but uncalled.

- [ ] **Step 2: Add the "kept but uncalled" comment to `_pull_processor_aux_files`**

Find the `def _pull_processor_aux_files(...)` definition in the script and prepend the comment block immediately above the `def`:

```python
# NOTE: kept but uncalled. The oneshot path saves the multimodal aux
# files via `processor.save_pretrained(output_dir)` in
# _save_artifact_and_recipe, which writes preprocessor_config.json
# (image processor), chat_template.jinja, and tokenizer files. The
# model_free_ptq sibling (scripts/quantize_w4a16.py) still needs this
# helper because it never instantiates a Processor object. Left here for
# reference and for any future model_free_ptq fallback path.
```

- [ ] **Step 3: Sanity-check the script parses and `--help` still works**

```bash
python scripts/quantize_cyankiwi.py --help 2>&1 | head -40
```

Expected: argparse banner with all flags from Task 5. If you see a SyntaxError or NameError, fix it before continuing — the smoke run won't get past import otherwise.

- [ ] **Step 4: Run the unit tests one more time to make sure nothing regressed**

```bash
pytest tests/test_quantize_cyankiwi.py -v
```

Expected: 15 passed (no new tests this task, but the import path through `scripts/quantize_cyankiwi.py` is now significantly longer; we want to know if anything broke).

- [ ] **Step 5: Commit**

```bash
git add scripts/quantize_cyankiwi.py
git commit -m "quantize_cyankiwi: swap main() to oneshot + GPTQ-w/-fallback pipeline

Five-stage pipeline: auth preflight → load model+processor+calib →
GPTQModifier oneshot (with QuantizationModifier fallback on the known
E-variant marker exceptions) → save compressed + processor + recipe
json → done. --skip-gptq bypasses the primary path for cases where
GPTQ is known to fail. The model_free_ptq call and --max-workers /
--device flags are gone; _pull_processor_aux_files is kept but
uncalled (processor.save_pretrained supersedes it)."
```

---

## Task 12: Wiring smoke run + load roundtrip

**Files:** None modified — verification only.

This is the integration verification step. It is **not** unit-testable; it requires a GPU (L4 24GB or larger), the full Colab-style llmcompressor install from the docstring, and HF auth. **Skip this task on a machine without GPU/CUDA available** — the user will run it.

- [ ] **Step 1: Run the smallest end-to-end smoke run**

On a GPU machine with the install from the script's docstring complete:

```bash
HF_HUB_ENABLE_HF_TRANSFER=1 python scripts/quantize_cyankiwi.py \
    --output-dir /tmp/llm-bench-cc/quant/smoke-cyankiwi \
    --num-calibration-samples 8 \
    --max-seq-length 256
```

Expected: pipeline reaches `[5/5] Done.` in ~2-5 minutes. The `entrypoint_used` log line tells you which modifier produced the artifact (GPTQ or observer-only). If GPTQ failed and the fallback succeeded, the warning log line above [4/5] shows the original exception's first 200 chars.

If both fail: capture the full traceback, add the new marker substring to `_E_VARIANT_FAILURE_MARKERS` (Task 2), and re-run.

- [ ] **Step 2: Check provenance**

```bash
cat /tmp/llm-bench-cc/quant/smoke-cyankiwi/quant_recipe.json
```

Expected: JSON with `entrypoint`, `scheme_kwargs` (group_size=32, observer=mse, symmetric=false), `calibration` (8 samples, 256 seqlen, Open-Platypus, train, seed=42), `git_sha`, `based_on` URL.

- [ ] **Step 3: Load roundtrip check**

```bash
python -c "
from transformers import AutoModelForImageTextToText, AutoProcessor
m = AutoModelForImageTextToText.from_pretrained('/tmp/llm-bench-cc/quant/smoke-cyankiwi', dtype='auto')
p = AutoProcessor.from_pretrained('/tmp/llm-bench-cc/quant/smoke-cyankiwi')
print('OK, params:', sum(x.numel() for x in m.parameters()))
"
```

Expected: model loads without error and prints a parameter count. If `AutoProcessor` raises `HFValidationError` complaining about the local path being treated as a repo id, `processor.save_pretrained` did not write `preprocessor_config.json` for some reason — check the output dir and fall back to the old `_pull_processor_aux_files` call.

- [ ] **Step 4: Real run**

Once the smoke run is green, the actual artifact uses the configured defaults:

```bash
HF_HUB_ENABLE_HF_TRANSFER=1 python scripts/quantize_cyankiwi.py \
    --output-dir /tmp/llm-bench-cc/quant/gemma-4-E4B-it-cyankiwi-W4A16
```

Expected: ~15-30 min on L4 if GPTQ ran, ~5 min if observer-only fallback ran. Output dir has the same file shape as the smoke run, plus a larger `*.safetensors` set.

- [ ] **Step 5: Smoke eval (existing pipeline, not this script's responsibility)**

Hand the output dir off to the existing eval flow:

```bash
python -m llm_bench_cc.eval --model /tmp/llm-bench-cc/quant/gemma-4-E4B-it-cyankiwi-W4A16 --tier smoke
```

Expected: composite retention number lands somewhere in the range bracketed by Vishva007's 0.835 (lower bound) and cyankiwi's 0.978 (upper bound). The exact value answers the original question: do the recipe knobs alone (group_size=32, observer=mse, asym) explain cyankiwi's edge, or is the AWQ smoothing actually doing work?

- [ ] **Step 6: No commit needed for this task — verification only**

The artifact at `/tmp/llm-bench-cc/quant/gemma-4-E4B-it-cyankiwi-W4A16` is what eval consumes; nothing in the repo changes.

---

## Self-Review notes (completed during plan-writing)

- **Spec coverage:** every section of the spec maps to one or more tasks:
  - "Recipe" → Task 1 (constants), Task 8 (`_scheme_kwargs_from_args`)
  - "Ignore list" → Task 1
  - "Pipeline flow" steps 1–8 → Tasks 6 (load), 7 (calib), 8 (GPTQ), 9 (observer fallback), 10 (save), 11 (main wiring with the try/except)
  - "CLI flags" → Task 5
  - "Error handling" → Tasks 2 (marker matcher), 11 (try/except dispatch)
  - "Verification plan" → Task 12
  - "Out of scope" → not implemented, by design
  - "Open risks" → mitigations are Tasks 2 + 11 fallback; if both paths fail, Task 12 Step 1 instructs how to extend the marker list
- **Placeholder scan:** no TBD/TODO/"implement later" — every code block is concrete.
- **Type consistency:** `_run_gptq` returns `str` (entrypoint), mutates model in place; `_run_observer_only` returns `(model, str)` because it rebuilds the model. The call sites in Task 11 use both shapes correctly.
- **Naming:** `_pick_calibration_column` consistent across Tasks 3 and 7. `_build_recipe_payload` consistent across Tasks 4 and 10. `_load_model` consistent across Tasks 6, 9, 11. `_E_VARIANT_FAILURE_MARKERS` / `_is_known_e_variant_failure` consistent across Tasks 2 and 11.
