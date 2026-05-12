# llm-bench-cc

Optimization pipeline for `google/gemma-4-E4B-it` (image-to-text). The eval harness runs the same
five-task benchmark suite (caption, ocr, docvqa, vqa, chart) against either an **HF transformers**
or a **vLLM** runtime — selected per run via `runtime.backend`. W&B + CodeCarbon tracking, composite
retention score against a frozen per-runtime baseline. Stage 1 quantization sweep landed multiple
variants on both runtimes (see "Pipeline stages" and "Quantized variants" below).

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
# optional, only needed for HF-track bnb variants and vLLM-track bnb-nf4-vllm
pip install -e ".[quant]"
```

The default runtime is HF transformers. To run any vLLM-track variant (see "Quantized variants" below), install vLLM **separately** — it ships its own torch/CUDA build, so don't install it into the same venv on machines where you also want the HF runtime:

```bash
pip install --pre vllm
```

vLLM-track compressed-tensors / GPTQ / FP8 variants need no further installs (vLLM's bundled loaders auto-detect quantization format from each checkpoint's `config.json`).

## One-time auth

```bash
huggingface-cli login    # gemma-4 is gated
wandb login              # or set WANDB_API_KEY; or wandb.mode=disabled
```

## Three eval tiers

| Switch | Tasks | Samples / task | Use for |
|---|---|---|---|
| `eval=smoke`    | ocr | 25 | Plumbing checks, quick A/Bs, lightning sweeps |
| `eval=standard` | caption, ocr, docvqa, chart | 200 | Routine Stage-1+ candidate iteration (no VQAv2 — fast, fits on a Colab VM) |
| `eval=full`     | caption, ocr, docvqa, vqa, chart | 200 | Canonical baseline + final retention numbers |

`composite` is computed over the **intersection** of tasks present in both the
candidate's `summary.json` and the baseline's `baseline.json`, so a `standard`
candidate scores fairly against a `full` baseline (over the 4 shared tasks; a
warning is logged listing the skipped tasks).

## Run the OCR-only smoke (lightning)

```bash
python -m llm_bench_cc.cli eval=smoke
```

## Run the standard 4-task pass (no VQA)

```bash
python -m llm_bench_cc.cli eval=standard
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

## Quantized variants

Each variant ships as a Hydra model config in `configs/model/`. The runner emits `quant_backend` / `quant_mode` into `summary.json`. Pick the runtime via `runtime.backend=hf` (default) or `runtime.backend=vllm`. Each candidate must be compared against a baseline produced on **the same runtime** — vLLM bf16 outputs are not byte-identical to HF bf16 (see open question in `project_vllm_runtime_validated.md`), so cross-runtime retention conflates quant effect with runtime effect.

### HF-track (Stage 1 Tier A — bitsandbytes)

```bash
pip install -e ".[quant]"   # one-time

# 8-bit weights (LLM only; vision tower stays in bf16)
python -m llm_bench_cc.cli model=bnb-int8 eval=standard \
    baseline_path=<output_dir>/<hf-bf16-baseline>/baseline.json

# NF4 4-bit double-quant (LLM only; vision tower stays in bf16)
python -m llm_bench_cc.cli model=bnb-nf4 eval=standard \
    baseline_path=<output_dir>/<hf-bf16-baseline>/baseline.json
```

bnb's recursive Linear-swap doesn't reliably honor subtree skip names — `backends/hf.py` walks the meta-loaded model and passes the full path of every non-LLM Linear to `llm_int8_skip_modules`. Override per variant via `model.quant.skip_modules='[…]'` only if you know what you're doing.

### vLLM-track (`runtime.backend=vllm`)

Four variants ship; all four were standard-eval-verified on Colab L4 against a vLLM bf16 baseline. Headline retention numbers (composite, 4 tasks × 200, vs vLLM bf16):

| Variant | Composite | Energy Δ vs bf16 | Quant scope |
|---|---:|---:|---|
| `w4a16-vllm` (cyankiwi compressed-tensors, int4 gs=32 asym) | **0.978** | **−43.7%** | LM Linears only; vision/audio towers + PLE per-layer plumbing kept bf16 |
| `fp8-w8a8-vllm` (in-flight FP8 W8A8 E4M3) | 0.975 | −24.8% | LM Linears only; multimodal towers + embeds kept bf16 |
| `bnb-nf4-vllm` (in-flight NF4) | 0.936 | −27.9% | Every Linear vLLM's bnb loader matches (no skip hook — vision tower included) |
| `w4a16-vllm-gptq` (Vishva007 GPTQ, int4 gs=128 sym, PLE-quantized) | 0.835 | −35.3% | LM Linears incl. PLE per-layer plumbing; vision/audio towers kept bf16 |

```bash
pip install --pre vllm   # one-time, separate stack

# cyankiwi W4A16 (current vLLM-track leader)
python -m llm_bench_cc.cli model=w4a16-vllm runtime.backend=vllm eval=standard \
    baseline_path=<output_dir>/<vllm-bf16-baseline>/baseline.json

# FP8 W8A8 in-flight (no calibration)
python -m llm_bench_cc.cli model=fp8-w8a8-vllm runtime.backend=vllm eval=standard \
    baseline_path=<output_dir>/<vllm-bf16-baseline>/baseline.json

# NF4 in-flight (needs bitsandbytes; vLLM's bnb path has no skip-modules hook)
python -m llm_bench_cc.cli model=bnb-nf4-vllm runtime.backend=vllm eval=standard \
    baseline_path=<output_dir>/<vllm-bf16-baseline>/baseline.json

# Vishva007 GPTQ W4A16 (paired comparison point; loses to cyankiwi on every task)
python -m llm_bench_cc.cli model=w4a16-vllm-gptq runtime.backend=vllm eval=standard \
    baseline_path=<output_dir>/<vllm-bf16-baseline>/baseline.json
```

The two W4A16 variants both adopt community pre-quantized checkpoints — no own calibration. Provenance + paired comparison verdict lives in `docs/community-w4a16-analysis.md`. The cyankiwi conservative scope (PLE preserved, gs=32 asym) wins both retention and energy; the Vishva007 aggressive scope (PLE quantized, gs=128 sym) clears the 0.80 bar by only 3.5pp and loses on every task.

vLLM's `peak_vram_gb` is the pre-allocated KV-cache pool size (≈ `gpu_memory_utilization × total_VRAM`), not the weight footprint. Quant savings show up only in the startup log line "Model loading took X GiB". Grep there if you need the weights-only number.

### Tier B HF-track (parked)

`scripts/quantize_fp8.py` and `scripts/quantize_w4a16.py` produce compressed-tensors checkpoints via `llmcompressor.model_free_ptq`. Both are **parked on HF transformers**: FP8_BLOCK loads bf16-at-inference (no fp8 kernels), and W4A16 measured 0.79 smoke / 18.75 GB peak (Marlin overhead exceeded savings on a 5B body). The same FP8_BLOCK checkpoint runs cleanly on vLLM where fp8 kernels engage; the W4A16 path on vLLM uses the cyankiwi community checkpoint instead of our recipe. See `reference_model_free_ptq_dead_end.md` for the full closure rationale.

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

**Don't cache VQAv2 to Drive** (only relevant for `eval=full`; `eval=standard` skips VQA entirely).
`lmms-lab/VQAv2`'s validation split is large and parquet-sharded;
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

# switch to vLLM runtime (needs `pip install --pre vllm` first; see "Quantized variants")
python -m llm_bench_cc.cli runtime.backend=vllm
# vLLM-specific knobs: pre-allocated KV-cache pool fraction + per-prompt MM slot counts
python -m llm_bench_cc.cli runtime.backend=vllm runtime.vllm.gpu_memory_utilization=0.85

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
  configs/               Hydra: model (incl. all quant variants), eval (smoke/standard/full),
                         top-level config (shipped inside the package so pip-installed runs work)
  backends/              Backend protocol + HF / vLLM implementations
    base.py              Backend protocol (load + generate + peak_vram_gb)
    hf.py                AutoModelForImageTextToText loader + bnb integration with full-path
                         skip-modules enumeration (the load-bearing bnb gotcha)
    vllm.py              vLLM `LLM` engine wrapper; nvidia-smi-based VRAM poller (torch counters
                         can't see vLLM's worker subprocess)
    __init__.py          `load_backend(cfg)` dispatch on `runtime.backend`
  metrics.py             BLEU-4, ANLS, VQA-acc, relaxed-acc, char-accuracy
  composite.py           Per-task retention ratios → unweighted composite (intersection-based
                         across tasks present in both candidate and baseline)
  tasks/                 One file per task; registry exposes them by name
  tracking.py            W&B + CodeCarbon wrappers (W&B init MUST come after vLLM backend load —
                         see `reference_wandb_vllm_deadlock.md`)
  runner.py              Eval loop, latency/VRAM capture, baseline.json/summary.json
  cli.py                 Hydra entry point (sets PYTORCH_CUDA_ALLOC_CONF, quiets HTTP loggers)
scripts/                 Optional one-shot helpers (quantize_*, spike_*, vllm_spike); not on path
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
| 0. Eval harness + bf16 baseline | Honest retention denominator on L4 | Done — HF bf16 baseline locked 2026-05-01; vLLM bf16 baselines (smoke + standard) minted 2026-05-11 |
| 1. Quantization sweep | Cut weights/energy with minimal retention loss | **HF track:** bnb-int8 (0.98) and bnb-nf4 (0.92) shipped. **vLLM track:** bnb-nf4-vllm, fp8-w8a8-vllm, w4a16-vllm (cyankiwi), w4a16-vllm-gptq (Vishva007 GPTQ) all standard-eval-verified. Current leader: **w4a16-vllm (cyankiwi) at 0.978 / −43.7% energy**. Tier B HF-track parked (llm-compressor `model_free_ptq` dead on Gemma 4 E-variants). Tier C (GGUF / llama.cpp) pending upstream feasibility. |
| 2. Pruning + Distillation | Shrink architecture, recover capability | Pending Stage 1 winner |
| 3. Best-of-stack | Combine pruned → distilled → quantized | Pending Stage 2 |
