# llm-bench-cc

Optimization pipeline for `google/gemma-4-E4B-it` (image-to-text). Stage 0 is the eval harness:
five tasks (caption, ocr, docvqa, vqa, chart), W&B + CodeCarbon tracking, composite retention
score against a frozen baseline.

## Targets

- Aggregate composite ≥ 0.80 of base (unweighted mean of per-task primary-metric retention).
- Optimize for memory footprint and energy (CodeCarbon).
- Train/distill on T4 (Colab); deploy on L4 24GB.

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

## Run the smoke baseline (~25 samples per task)

```bash
python -m llm_bench_cc.cli
```

This produces `outputs/<run_name>/`:

- `baseline.json`  — frozen per-task primary metrics (the 100% line)
- `summary.json`   — run summary with composite, energy, latency, peak VRAM
- `preds_<task>.jsonl` — raw predictions per task
- `emissions_*.csv`    — CodeCarbon raw output

## Run the full baseline (200 samples per task)

```bash
python -m llm_bench_cc.cli eval=full
```

## Score a candidate against a saved baseline

```bash
python -m llm_bench_cc.cli \
    model=<some_quant_or_pruned_variant> \
    baseline_path=outputs/<baseline-run>/baseline.json
```

`composite` in `summary.json` will be the retention ratio against that baseline.

## Common overrides

```bash
# disable W&B
python -m llm_bench_cc.cli wandb.mode=disabled

# disable CodeCarbon
python -m llm_bench_cc.cli carbon.enabled=false

# a single task
python -m llm_bench_cc.cli eval.tasks='[caption]'

# different sample size
python -m llm_bench_cc.cli eval.samples_per_task=50

# bf16 on Ada+ GPUs (L4); leave as float16 on Turing (T4)
python -m llm_bench_cc.cli model.dtype=bfloat16 model.attn_implementation=sdpa
```

## Colab (T4)

```python
!pip install -q "llm-bench-cc[dev] @ git+<repo-or-local-path>"
!huggingface-cli login
!wandb login
!python -m llm_bench_cc.cli eval=smoke wandb.mode=online
```

T4 has no bf16 and no Flash-Attention — keep `model.dtype=float16` and
`model.attn_implementation=eager` there.

## Tests

```bash
pytest tests/
```

Metric tests lock the behavior the composite score depends on. If these change, every prior
`baseline.json` becomes incomparable — bump a metric version field before changing them.

## Layout

```
configs/                 Hydra: model, eval (smoke/full), top-level config
src/llm_bench_cc/
  models.py              HF loader (AutoModelForImageTextToText)
  metrics.py             BLEU-4, ANLS, VQA-acc, relaxed-acc, char-accuracy
  composite.py           Per-task retention ratios → unweighted composite
  tasks/                 One file per task; registry exposes them by name
  tracking.py            W&B + CodeCarbon wrappers
  runner.py              Eval loop, latency/VRAM capture, baseline.json/summary.json
  cli.py                 Hydra entry point
tests/                   Metric smoke tests
```

## Dataset notes

The HF dataset paths in `configs/eval/*.yaml` are defaults — if a path is unavailable or its
schema changes, swap `eval.datasets.<task>.hf_id` (and `split`/`subset`). Each task's loader
tolerates a few common field-name variants but raises clearly on missing fields.
