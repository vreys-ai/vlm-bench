# llm-bench-cc

Optimization pipeline for `google/gemma-4-E4B-it` (image-to-text). Stage 0 is the eval harness:
five-task benchmark suite (caption, ocr, docvqa, vqa, chart), W&B + CodeCarbon tracking, composite
retention score against a frozen baseline.

## Targets

- Aggregate composite ≥ 0.80 of base (unweighted mean of per-task primary-metric retention).
- Optimize for memory footprint and energy (CodeCarbon).
- L4 24 GB for both bench and deployment. (Earlier plan used T4 for training; abandoned because the
  fp16 baseline was too tight on T4 16 GB. L4 supports bf16, sdpa, and Flash-Attention 2.)

## Install

```bash
# from repo root
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
# optional, only needed for Stage 1+
pip install -e ".[quant]"
```

## One-time auth

```bash
huggingface-cli login    # gemma-4 is gated
wandb login              # or set WANDB_API_KEY; or wandb.mode=disabled
```

## Two eval tiers

| Switch | Tasks | Samples / task | Use for |
|---|---|---|---|
| `eval=smoke` | ocr only | 25 | Plumbing checks, quick A/Bs, lightning sweeps |
| `eval=full`  | caption, ocr, docvqa, vqa, chart | 200 | Real baseline + Stage 1+ retention denominators |

## Run the OCR-only smoke (lightning)

```bash
python -m llm_bench_cc.cli eval=smoke
```

## Run the full 5-task baseline

```bash
python -m llm_bench_cc.cli eval=full
```

Each run writes `<output_dir>/<run_name>/`:

- `baseline.json`     — frozen per-task primary metrics (the 100% line, only for `model.name=base`)
- `summary.json`      — composite, retention ratios, energy, latency, peak VRAM
- `preds_<task>.jsonl` — raw predictions per task
- `emissions_*.csv`   — CodeCarbon raw output
- `.hydra/`           — Hydra config snapshot

`output_dir` defaults to `/content/drive/MyDrive/vlm-bench-runs` (set in `configs/config.yaml`); change
it locally if you're not on Colab.

## Score a candidate against a saved baseline

```bash
python -m llm_bench_cc.cli \
    model=<some_quant_or_pruned_variant> \
    eval=full \
    baseline_path=<output_dir>/<baseline-run>/baseline.json
```

`composite` in `summary.json` will be the retention ratio against that baseline.

## Persistent caches (Drive on Colab)

Avoid re-downloading the model and datasets on every VM:

```bash
python -m llm_bench_cc.cli eval=full \
    model.cache_dir=/content/drive/MyDrive/hf-cache \
    eval.cache_dir=/content/drive/MyDrive/hf-cache
```

Set `*.local_files_only=true` to fail fast (no network):

```bash
python -m llm_bench_cc.cli eval=full \
    model.cache_dir=/content/drive/MyDrive/hf-cache \
    model.local_files_only=true \
    eval.cache_dir=/content/drive/MyDrive/hf-cache \
    eval.local_files_only=true
```

Per-dataset `cache_dir` / `local_files_only` (under `eval.datasets.<task>`) override the eval-level
defaults if set.

**Don't cache VQAv2 to Drive.** `lmms-lab/VQAv2`'s validation split is large and parquet-sharded;
`load_dataset(...).shuffle().select(range(N))` materializes the full split before subsetting, so
even a 200-sample run downloads several GB. Route VQA at ephemeral Colab disk while caching the
small four on Drive:

```bash
python -m llm_bench_cc.cli eval=full \
    model.cache_dir=/content/drive/MyDrive/hf-cache \
    eval.cache_dir=/content/drive/MyDrive/hf-cache \
    eval.datasets.vqa.cache_dir=/root/.cache/huggingface
```

VQA re-downloads each VM, but Drive stays clean.

## Common overrides

```bash
# disable W&B / CodeCarbon
python -m llm_bench_cc.cli wandb.mode=disabled
python -m llm_bench_cc.cli carbon.enabled=false

# a subset of tasks (full only)
python -m llm_bench_cc.cli eval=full eval.tasks='[caption,ocr]'

# different sample size
python -m llm_bench_cc.cli eval=full eval.samples_per_task=50

# move outputs (e.g. local SSD instead of Drive)
python -m llm_bench_cc.cli output_dir=/tmp/runs

# bf16 + sdpa on L4 (Ada)
python -m llm_bench_cc.cli model.dtype=bfloat16 model.attn_implementation=sdpa

# tune memory hygiene (legacy T4 settings; can usually be relaxed on L4)
python -m llm_bench_cc.cli runtime.image_max_side=null runtime.empty_cache_between_samples=false

# per-task generation overrides (full.yaml ships sensible defaults; override on the CLI if needed)
python -m llm_bench_cc.cli eval=full eval.generation_overrides.caption.max_new_tokens=128
```

`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` is set automatically by `cli.py` to reduce
fragmentation; export the env var yourself to override.

## Colab (L4)

```python
!pip install -q "llm-bench-cc[dev] @ git+<repo-or-local-path>"
!huggingface-cli login
!wandb login
!python -m llm_bench_cc.cli eval=smoke \
    model.cache_dir=/content/drive/MyDrive/hf-cache \
    eval.cache_dir=/content/drive/MyDrive/hf-cache \
    wandb.mode=online
```

If you ever fall back to T4: keep `model.dtype=float16`, `model.attn_implementation=eager`, and
expect to lower `runtime.image_max_side` (≤512) and shorten `max_new_tokens` to fit.

## Tests

```bash
pytest tests/
```

Metric tests lock the behavior the composite score depends on. If these change, every prior
`baseline.json` becomes incomparable — bump a metric version field before changing them.

## Layout

```
src/llm_bench_cc/
  configs/               Hydra: model, eval (smoke/full), top-level config
                         (shipped inside the package so pip-installed runs work)
  models.py              HF loader (AutoModelForImageTextToText)
  metrics.py             BLEU-4, ANLS, VQA-acc, relaxed-acc, char-accuracy
  composite.py           Per-task retention ratios → unweighted composite
  tasks/                 One file per task; registry exposes them by name
  tracking.py            W&B + CodeCarbon wrappers
  runner.py              Eval loop, latency/VRAM capture, baseline.json/summary.json
  cli.py                 Hydra entry point (sets PYTORCH_CUDA_ALLOC_CONF, quiets HTTP loggers)
tests/                   Metric smoke tests
```

## Dataset notes

The HF dataset paths in `src/llm_bench_cc/configs/eval/full.yaml` are verified-public mirrors —
several "natural" paths (e.g. `nlphuji/coco_captions`, `lmms-lab/OCRBench`) are gated or removed.
Each task's loader tolerates a few common field-name variants but raises clearly on missing fields.
Override per-run if a path breaks:

```bash
python -m llm_bench_cc.cli eval.datasets.caption.hf_id=<other-dataset>
```

## Pipeline stages

| Stage | Goal | Status |
|---|---|---|
| 0. Eval harness + fp16 baseline | Honest retention denominator on L4 | Plumbing validated; `eval=full` baseline regeneration is the last step |
| 1. Quantization sweep | Cut weights/energy with minimal retention loss (bnb-int8/nf4, GPTQ-4, AWQ-4, GGUF-Q4_K_M/Q5_K_M) | Pending Stage 0 final baseline |
| 2. Pruning + Distillation | Shrink architecture, recover capability | Pending Stage 1 winner |
| 3. Best-of-stack | Combine pruned → distilled → quantized | Pending Stage 2 |
