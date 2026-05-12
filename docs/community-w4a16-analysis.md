# Community W4A16 Checkpoints for gemma-4-E4B-it — Configuration Analysis

**Status:** cyankiwi standard-eval-verified 2026-05-11 (composite **0.978**, energy **−43.7%** vs bf16). Vishva007 GPTQ standard-eval-verified 2026-05-12 (composite **0.835**, energy **−35.3%** — below cyankiwi on both axes). **cyankiwi remains the vLLM-track W4A16 winner**; PLE-quantization + gs=128-sym (the Vishva007 scope) is not free on this model. Vishva007 AWQ variant analyzed from configs only; not benchmarked.
**Date:** 2026-05-11 (initial analysis); 2026-05-11 (cyankiwi spike + standard eval appended); 2026-05-12 (Vishva007 GPTQ spike + standard eval appended)
**Question being answered:** of the available community pre-quantized W4A16 checkpoints, which (if any) should we adopt as a vLLM-track 4-bit data point, given our own `llm-compressor model_free_ptq` W4A16 recipe failed on this model architecture on 2026-05-10?

## Candidates evaluated

| Repo | Format | Last modified | 30d downloads |
|---|---|---|---|
| `cyankiwi/gemma-4-E4B-it-AWQ-INT4` | compressed-tensors `pack-quantized` (despite the "AWQ" repo name) | 2026-05-03 | 45,858 |
| `Vishva007/gemma-4-E4B-it-W4A16-AutoRound-AWQ` | AWQ (GEMM) — weights under `local_model-e4b-w4g128/` subdir | 2026-05-09 | 0 |
| `Vishva007/gemma-4-E4B-it-W4A16-AutoRound-GPTQ` | GPTQ | 2026-05-09 | 37,379 |

Vishva007 AWQ's 0-download count is almost certainly the subdirectory structure breaking `vllm serve <repo-id>` auto-pull — the safetensors aren't at the repo root where `from_pretrained` looks by default. The GPTQ variant has weights at the root and shows healthy downloads.

(For provenance: I initially mis-eliminated Vishva007 by checking only `raw/main/config.json` at the root and getting a 404. The HF API `/api/models/<repo>` siblings list, which I used for cyankiwi at the end, would have revealed the subdir. Fixed.)

## Ground-truth methodology

For each checkpoint I read three sources:

1. The repo's `config.json` (HuggingFace top-level model config; contains an embedded `quantization_config` block).
2. The separate `quantization_config.json` produced by AutoRound (only the Vishva007 variants ship this).
3. The `model.safetensors.index.json` weight-map, when present — **this is the ground truth**: any Linear that was actually quantized appears as a `.qweight` + `.qzeros` + `.scales` triplet; anything left in bf16 appears as a plain `.weight`. The model card and the `modules_to_not_convert` list both can lie; the index cannot.

cyankiwi has a single safetensors file with no index, so for it I rely on the embedded `quantization_config` (which is internally consistent: `config_groups` says targets=Linear, `ignore` enumerates skips).

## Quantization parameters at a glance

| Parameter | cyankiwi | Vishva007 AWQ | Vishva007 GPTQ |
|---|---|---|---|
| `quant_method` | `compressed-tensors` | `awq` | `gptq` |
| Save format | `pack-quantized` | AWQ GEMM | GPTQ standard |
| Producer tool | llm-compressor (compressed-tensors v0.1.dev483+ga551158) | AutoRound 0.13.0 → AWQ export | AutoRound 0.13.0 → GPTQ export |
| Bits (weights) | 4 | 4 | 4 |
| Activations | bf16 | bf16 | bf16 |
| **Group size** | **32** | **128** | **128** |
| **Symmetric** | **No** (asymmetric, with zero-points) | **Yes** | **Yes** |
| Observer | MSE | (AutoRound defaults) | (AutoRound defaults) |
| Calibration samples | not stated in card | 256 | 256 |
| Calibration seq len | not stated in card | 2048 | 2048 |
| Calibration set | `cyankiwi/calibration` (text-only, 384 agent traces from `nvidia/Nemotron-SWE-v1`) | not named in card; AutoRound default = `NeelNanda/pile-10k` (text-only) | same |
| AutoRound iters | n/a | 0 (RTN mode — explicit "required for Gemma 4 compatibility") | 0 (same, explicit) |
| `desc_act` | n/a | n/a (AWQ) | `False` |
| `damp_percent` | n/a | n/a | `0.01` |
| KV-cache quant | none (`kv_cache_scheme: null`) | none | none |
| Quant calibration hardware | not stated | A100 80GB PCIe | A100 80GB PCIe |

Two parameter differences are likely to matter most for retention:

* **Group size 32 vs 128.** cyankiwi packs 4× more scales per row than Vishva007. At 4 bits, finer groups buy back accuracy at a small overhead cost. This is the single biggest accuracy-relevant knob and cyankiwi has it set 4× tighter.
* **Asymmetric vs symmetric.** cyankiwi's asymmetric quant with zero-points handles bimodal / shifted weight distributions better; symmetric quant is faster on some kernels but less flexible. For Marlin kernels (the path both will run on) both are supported.

## What was actually quantized — hard ground truth from the safetensors

The Gemma-4-E4B LM has **42 transformer layers**. KV-sharing is in effect: only **24 of those 42 layers** carry their own `k_proj` / `v_proj` parameters; the other 18 share K and V from an earlier layer and have no own KV projections. Both `q_norm` (42) and `k_norm` (24) follow that same pattern — these are RMSNorm tensors, always bf16 because norms aren't Linears.

### Vishva007 AWQ and Vishva007 GPTQ (identical scope, different format)

`.qweight` count: **342 quantized Linears each**, identical breakdown:

| Module | Count | Coverage |
|---|---|---|
| `self_attn.q_proj` | 42 | full (every layer) |
| `self_attn.o_proj` | 42 | full |
| `mlp.gate_proj` | 42 | full |
| `mlp.up_proj` | 42 | full |
| `mlp.down_proj` | 42 | full |
| `per_layer_input_gate` (PLE) | 42 | full |
| `per_layer_projection` (PLE) | 42 | full |
| `self_attn.k_proj` | 24 | every layer that has its own K |
| `self_attn.v_proj` | 24 | every layer that has its own V |

Subtotal: `42×7 + 24×2 = 342` ✓ — every quantizable Linear in the LM is quantized.

Plain (bf16) tensors inside the LM: 2 embedding tables, 42 `q_norm`, 24 `k_norm`, one global `model.language_model.norm`, one model-level `per_layer_model_projection` (a single Linear distinct from the 42 per-layer projections — appears in the skip list as the sole "LM.other" entry). Standard layer norms (RMSNorm scale params) make up the rest of the ~210 "other" plain weights inside `model.language_model.layers`.

The two Vishva007 variants are bit-for-bit equivalent in *which* Linears are quantized; they differ only in the **packing format** (AWQ-GEMM vs GPTQ standard) and in the runtime kernel they engage via vLLM. With both at gs=128, symmetric, RTN-mode (iters=0), the numerical content of the quantized weights themselves should be very close — the round-to-nearest map is the same; only the on-disk layout and the asymmetric/zero-point handling differs marginally.

### cyankiwi

No safetensors index, but `quantization_config` enumerates **334 skipped modules** with a single quant group covering all other Linears:

| Skip category | Count |
|---|---|
| `audio_tower` (incl. `embed_audio.embedding_projection`) | 135 |
| `vision_tower` (incl. `vision_tower.patch_embedder.input_proj`) | 114 |
| `per_layer_input_gate` (PLE) | 42 |
| `per_layer_projection` (PLE) | 42 |
| `lm_head` | 1 |
| **Total skipped** | **334** |

Derived quantized scope: **259 Linears** (42 q_proj + 42 o_proj + 42 gate + 42 up + 42 down + 24 k + 24 v + **1 `model.language_model.per_layer_model_projection`**) — the same per-layer q/k/v/o/gate/up/down coverage as Vishva007, **minus 84 per-layer PLE Linears**, but **plus** the single model-level PLE projection that Vishva007 keeps in bf16. The 259 count was confirmed by enumerating `.weight_packed` tensors in the live safetensors during the 2026-05-11 spike (see "Spike verification" below).

The 259 vs 342 gap between cyankiwi and Vishva007 (= 84 per-layer PLE Linears − 1 model-level PLE Linear) is the full architectural-decision delta. Note that the per-layer PLE Linears are much larger in total than the single model-level one, so cyankiwi's scope is strictly the more conservative of the two on the quantizable-Linear axis.

### The architecturally meaningful differences

There are **two** PLE-related decisions where the vendors disagree:

1. **Per-layer PLE plumbing (84 Linears: 42 `per_layer_input_gate` + 42 `per_layer_projection`).** Vishva007 quantizes them; cyankiwi does not. This is the larger of the two scope differences.
2. **Model-level `per_layer_model_projection` (1 Linear).** cyankiwi quantizes it; Vishva007 keeps it in bf16. Trivial in compression terms (a single Linear) but worth noting because the two vendors made opposite calls.

The PLE modules are Gemma-4-E-variant-specific learned Linears that inject per-layer embeddings into the residual stream. They aren't huge in aggregate but they're the part of the architecture that broke llm-compressor's `oneshot` and forced AutoRound into RTN-only mode. There's a defensible engineering argument on both sides for the per-layer decision:

* **cyankiwi conservatism:** PLE per-layer modules are small relative to attention/MLP, so the compression win from quantizing them is marginal, but the accuracy risk is real (these gates control how strongly the per-layer embedding contributes). Leaving them in bf16 trades a tiny compression budget for safety.
* **Vishva007 aggression:** RTN at gs=128 against a small Linear is usually fine if the weight distribution is well-behaved; quantizing them yields a marginally smaller checkpoint and more uniform kernel paths. AutoRound 0.13 evidently handles them without crashing (where llm-compressor's `oneshot` did not).

Without retention measurement on the full eval, neither is clearly correct.

## Calibration data — a risk that applies to all three

cyankiwi's calibration set (`cyankiwi/calibration`):

* HF dataset card tags include `modality:text` — **text-only**, confirmed.
* 384 samples drawn from `nvidia/Nemotron-SWE-v1` (agentic software-engineering conversations).
* No image column, no image URL/path field, no multimodal schema.

Vishva007 model cards do not name a calibration dataset, only "256 samples at seq_len 2048." AutoRound's default when no dataset is specified is `NeelNanda/pile-10k` — generic English text from The Pile. **Also text-only.**

Consequence for our target benchmarks: the LM weights of every candidate were calibrated on text distributions. The vision tower is preserved bit-identical in bf16, so visual *encoding* is intact, but the quantized LM has to process the resulting visual tokens with weights that were never optimized against visual-token activations. Empirically VLMs survive this with moderate degradation (typically 2–5% on visual-QA benchmarks) but it is a real risk and it is the same risk for all three candidates — it does not differentiate them.

The smoke eval on docvqa / textvqa is the go/no-go signal. Until then we should not assume retention numbers from text-only benchmarks (like MMLU) transfer to our composite.

## vLLM compatibility

* **cyankiwi (compressed-tensors / pack-quantized):** vLLM's bundled compressed-tensors loader auto-detects from `config.json` — no `--quantization` kwarg needed. Our existing `backends/vllm.py` handles this exact path today (it's how the FP8_BLOCK checkpoint loads). Routes through gptq-marlin kernels on Ada+ hardware.
* **Vishva007 AWQ:** card prescribes `--quantization autoround`; underlying format is AWQ-GEMM, so `--quantization awq` or `awq_marlin` should also work via vLLM's native AWQ loader. Also requires loading from the `local_model-e4b-w4g128/` subdirectory, which means we likely need to download the subdir and point vLLM at the local path rather than the bare repo id — extra friction.
* **Vishva007 GPTQ:** card prescribes `--quantization autoround`; underlying format is standard GPTQ, so `--quantization gptq` or `gptq_marlin` should also work. Weights are at the repo root, so `vllm serve Vishva007/gemma-4-E4B-it-W4A16-AutoRound-GPTQ` is a one-liner.

All three should pre-quantize-load with no extra calibration step. None requires our previously-broken `llm-compressor` path.

## Trust & provenance

cyankiwi: anonymous community user, 45k downloads in 30 days. Internal config consistency is high (a single group, well-formed `ignore` list, valid compressed-tensors version string). No published quant-quality measurement; the MMLU / MMMU figures on the model card are the base E4B's published scores, not measurements on the quantized variant — easy to misread as a guarantee.

Vishva007: same publisher for AWQ and GPTQ, same timestamp, same calibration run. GPTQ variant has 37k 30d downloads suggesting it loads cleanly for many users; AWQ shows 0 — most likely the subdir layout, not a quality issue.

Neither publisher has signed releases or measured retention; this is community-grade provenance for both.

## Recommendation matrix

The choice depends on what we weight more — and the safest thing is to smoke-test more than one, because the cost is small once the vLLM serve path is wired.

| If we optimize for… | Pick |
|---|---|
| Retention safety (finer quant grain, PLE preserved, asymmetric) | **cyankiwi** |
| Aggressive compression (PLE also quantized) | **Vishva007 GPTQ** (frictionless repo-root load) |
| Format diversity for the matrix (we already have compressed-tensors via FP8_BLOCK; an AWQ-format datapoint is novel) | **Vishva007 AWQ** (needs subdir handling) |
| Time-to-first-number | **Vishva007 GPTQ** — repo-root load, well-downloaded, GPTQ-Marlin path is highly optimized in vLLM |

### Concrete plan

I'd run the smoke test in two phases:

1. **First smoke: cyankiwi.** It plugs into our existing compressed-tensors-via-vLLM path with no code changes (same auto-detect mechanism we already use for FP8_BLOCK). gs=32 + asymmetric is the most accuracy-friendly of the three. If retention on smoke eval is ≥ 0.85, promote to standard eval.
2. **Second smoke: Vishva007 GPTQ.** If cyankiwi clears the bar, this gives us a comparison point on the same scheme (W4A16) with a different tradeoff (gs=128, sym, PLE-quantized). The energy delta between the two will be small (both are 4-bit on the same Linears modulo 84 small PLE projections); the retention delta will tell us whether PLE-quantization is safe.
3. **Vishva007 AWQ only if** we want the AWQ kernel data point specifically. Same checkpoint content as GPTQ for practical purposes, just different on-disk format and kernel.

If both cyankiwi and Vishva007-GPTQ retention land below 0.80 on the composite, that's our signal that the text-only-calibration penalty is too steep for VLM use and we should pivot to FP8_BLOCK (the originally-planned next step, where vision tower work isn't at risk).

## Open questions

1. **vLLM's "autoround" quantization flag** — the Vishva007 cards both prescribe `--quantization autoround` but the official vLLM stable docs list AutoAWQ and GPTQModel as supported back-ends without naming AutoRound. Either (a) the flag is a relatively recent alias that vLLM auto-detects from the `provider: auto-round` field in the quant config, or (b) the cards expect a vLLM build off main / a Docker image (`vllm/vllm-openai:gemma4`) that includes the alias. For our use, `--quantization awq_marlin` (for the AWQ variant) and `--quantization gptq_marlin` (for the GPTQ variant) should work against vLLM stable; worth confirming on first load.
2. **Why AutoRound runs only in RTN mode (iters=0)** — both cards say "required for Gemma 4 compatibility" without specifying which step of full iterative AutoRound breaks. Probably the same PLE/KV-sharing hooks issue we hit with `oneshot`. Doesn't block adoption (RTN at gs=128 sym for AWQ, or gs=32 asym for cyankiwi's compressed-tensors, both produce usable checkpoints) but means the AutoRound output is not "optimized AWQ" — it's "AWQ-formatted RTN." Equivalent statement for the GPTQ variant.
3. **cyankiwi's single PLE.projection plain-weight tensor** — the safetensors index shows 42 quantized PLE.projection entries plus 1 plain. Either the model has 43 PLE projection-like Linears (one global plus 42 per-layer) or it's a naming artifact. The Vishva007 AWQ index shows the same pattern with 42 quantized + 1 plain `per_layer_model_projection`, suggesting the latter — there is one model-level projection in addition to the 42 per-layer ones. Worth noting but not a blocker.

## Spike verification (2026-05-11, cyankiwi only)

Ran `scripts/spike_cyankiwi_w4a16.py` on a Colab L4. Four predictions tested.

| Prediction | Measured | Status |
|---|---|---|
| **P1** vLLM auto-detects compressed-tensors format, no `quantization=` kwarg | Startup log: `Using MarlinLinearKernel for CompressedTensorsWNA16` — no kwarg passed, no crash | ✅ PASS |
| **P2** 258 LM Linears quantized with the predicted structural breakdown | 259 (one more than predicted); the extra is `model.language_model.per_layer_model_projection` — analysis text was off by one. Structural checks all pass: 42-layer full coverage on q/o/gate/up/down, 24-layer KV-shared coverage on k/v, 0 PLE per-layer Linears quantized. | ✅ PASS (structure); analysis text now corrected to 259 |
| **P3** Multimodal generation still works | Decoded text from the `CYANKIWI SMOKE` synthetic stamp: `'CYANKIWI SMOKE'` (exact). 7 tokens decoded. | ✅ PASS (strong — clean OCR, not just any text) |
| **P4** Weights footprint between bf16 and full-4-bit | vLLM log: `Model loading took 9.48 GiB memory and 9.185655 seconds`. Predicted band (5–7 GiB) was anchored against the wrong bf16 baseline; corrected band based on this checkpoint's disk size is "match disk size ≈ 9.6 GB" which the measurement does. | ✅ PASS (real number; prior prediction band was anchored wrong, replaced below) |

### Corrected footprint table

The pre-spike doc anchored cyankiwi against a "~10 GiB bf16 baseline" — that was actually `fp8-w8a8-vllm`'s 10.93 GiB load number, not the bf16 baseline. The true bf16 baseline is ~16 GiB (per the existing `fp8-w8a8-vllm.yaml` comment header).

| Variant | vLLM load size | vs bf16 baseline |
|---|---|---|
| bf16 baseline | ~16 GiB | — |
| `fp8-w8a8-vllm` (2026-05-11 prior spike) | 10.93 GiB | −32% |
| **`w4a16-vllm` (cyankiwi, this spike)** | **9.48 GiB** | **−41%** |
| Hypothetical full-LM 4-bit (no bf16 anywhere) | ~3 GiB | −81% |

So **cyankiwi beats fp8-w8a8-vllm by ~14% on weights footprint** while quantizing fewer total params (only LM Linears, vs fp8-w8a8 which also touches LM activations dynamically). The retention comparison from the standard eval will tell us whether the −9% energy delta is bought cheaply.

### Throughput observation (small-sample, prefill-dominated)

Multimodal decode of 7 tokens ran at 27.49 tok/s (single sample, includes vision-encoder prefill). For comparison, the existing backend doc records fp8-w8a8 at ~16 tok/s on L4 (Ada TRITON_ATTN fallback). The Marlin W4A16 kernel may be genuinely faster on this model — or it may just be that prefill amortization differs between the two paths at this token count. Worth confirming on a long-generation run, but at minimum cyankiwi is not slower than the existing vLLM-track quants.

### TRITON_ATTN fallback observed

Startup log shows `TRITON_ATTN` as the attention backend, confirming the Ada-generation perf cliff (vLLM issue #38887, documented in `backends/vllm.py`) is in effect for this checkpoint too. Expected; not a blocker.

### The "+1" anomaly explained

A one-liner against the cached safetensors confirms the 259th quantized Linear is exactly:

```python
['model.language_model.per_layer_model_projection.weight_packed']
```

This is a model-level Linear (not per-layer), distinct from the 42 `per_layer_projection` Linears that exist inside each transformer block. Vishva007's safetensors index lists it as a plain `.weight` (kept bf16), so the two vendors made opposite calls on this single Linear. The energy/retention impact of this disagreement is negligible (one Linear out of 259/342), but it explains the off-by-one between the pre-spike subtraction estimate and the measured count.

## Standard eval results (2026-05-11)

Standard eval on Colab L4, 200 samples × 4 tasks (caption / chart / docvqa / ocr), wandb run `nbbglmkv` (`w4a16-vllm-standard-20260511-202441`). Compared against the existing vLLM-track standard runs to keep the comparison apples-to-apples.

### Composite + retention

| Run | Composite | Retention vs bf16 | Energy vs bf16 |
|---|---:|---:|---:|
| `base-vllm` (bf16) | 1.0000 | 1.0000 | — |
| `fp8-w8a8-vllm` | 0.9748 | 0.9748 | −24.8% |
| `bnb-nf4-vllm` | 0.9358 | 0.9358 | −27.9% |
| **`w4a16-vllm` (cyankiwi)** | **0.9783** | **0.9783** | **−43.7%** |

cyankiwi clears the 0.80 retention bar by 18 absolute points, edges fp8-w8a8 on retention by ~0.35pp, and roughly **doubles** the energy savings of every other quant on the matrix. It is the new vLLM-track leader on both axes.

### Per-task retention

| Task | Metric | fp8-w8a8 | bnb-nf4 | **w4a16** | Notes |
|---|---|---:|---:|---:|---|
| caption | bleu4 | 0.970 | 0.980 | **1.011** | >1.0 plausibly noise — bleu4 ~0.10 has high variance at N=200. Treat ≥0.97 as a tie. |
| chart | relaxed_acc | **0.951** | 0.916 | 0.937 | Everyone's weakest task; likely most quantization-sensitive. fp8 wins this one. |
| docvqa | anls | 1.016 | 0.897 | 0.983 | w4a16's 0.983 is the saner docvqa retention number; fp8's earlier 1.016 still looks like its own variance. |
| ocr | anls | 0.978 | 0.950 | **0.993** | w4a16 essentially matches bf16. |

### Per-task energy (kWh) and latency

Energy: cyankiwi saves 33–47% per task against bf16, with the smallest delta on docvqa and the largest on caption / chart:

| Task | bf16 | **w4a16** | Δ |
|---|---:|---:|---:|
| caption | 0.00724 | 0.00381 | **−47.3%** |
| chart   | 0.00514 | 0.00277 | **−46.2%** |
| docvqa  | 0.00288 | 0.00193 | **−33.1%** |
| ocr     | 0.00334 | 0.00195 | **−41.4%** |

Latency (mean ms/sample): −36% to −47% per task vs bf16. Specifically:

| Task | bf16 | fp8-w8a8 | bnb-nf4 | **w4a16** |
|---|---:|---:|---:|---:|
| caption | 981 | 715 | 614 | **522** |
| chart | 678 | 515 | 480 | **362** |
| docvqa | 338 | 267 | 289 | **216** |
| ocr | 436 | 327 | 361 | **253** |

So w4a16 is **faster than fp8-w8a8 on every task**, despite Marlin paying for the Ada TRITON_ATTN cliff that fp8 also pays for. The int4 weight bandwidth win evidently dominates the kernel-dispatch overhead at this batch size.

### Peak VRAM (uniform across runs — KV-cache pool, as expected)

All four runs report `avg_peak_vram_gb` ≈ 20.2 GiB because vLLM pre-allocates KV-cache to `gpu_memory_utilization × total_VRAM` (default 0.9 × 22.5 = ~20.25 on L4). Quant savings surface only in the startup log's "Model loading took X GiB" line:

| Run | Weights footprint |
|---|---:|
| bf16 baseline | ~16 GiB |
| `fp8-w8a8-vllm` | 10.93 GiB |
| **`w4a16-vllm` (cyankiwi)** | **9.48 GiB** |

### Two tasks where w4a16 didn't win

* **chart (0.937 vs fp8's 0.951)** — w4a16 gives up 1.4pp on chart. Plausible: chart_qa relaxed_acc rewards exact numeric matches in text, and fp8's 8-bit precision is less aggressive than int4 on the LM weights handling that. Not a blocker; well above the bar.
* **docvqa (0.983 vs fp8's 1.016)** — but fp8's 1.016 was already flagged as suspicious. w4a16's 0.983 is the more credible number; fp8 may be the outlier here, not the gold standard.

### Why this comes out ahead

Mechanism, in plain terms:

1. **More aggressive weight bit-budget** (int4 vs fp8) for the standard LM Linears, where most of gemma-4-E4B's parameters live → bigger memory-bandwidth win on every forward pass.
2. **Same vision-tower preservation** as fp8-w8a8 (and explicitly *more* preservation than bnb-nf4-vllm, which has no skip-modules hook), so the visual encoder path is bit-identical to bf16. Document and OCR tasks that depend on intact visual features keep retention high.
3. **PLE per-layer plumbing preserved in bf16** (the architecturally-fragile bits that broke our llm-compressor recipe), so we don't pay an accuracy tax on the part of the model that's hardest to quantize cleanly.
4. **Finer quantization grain than the AutoRound alternatives** (gs=32 vs gs=128), which buys back accuracy at small overhead.

### Not yet verified (open questions for any follow-up)

* The text-only calibration risk (vision tower bit-identical bf16, but LM weights calibrated against text-only Nemotron-SWE-v1 traces) appears to be small for our task mix — but if a future task is heavier on visual reasoning (e.g. MMMU-style chart-reasoning) the picture may shift.
* Whether the Vishva007 GPTQ variant (PLE-quantized, gs=128, sym) would beat cyankiwi by quantizing 83 more Linears, or lose retention. Not pursued — cyankiwi already wins; running Vishva007 is only justified if we want a paired comparison on PLE-quantized vs PLE-preserved.

## Vishva007 GPTQ standard eval results (2026-05-12)

Standard eval on Colab L4, 200 samples × 4 tasks (caption / chart / docvqa / ocr), wandb run `0cc1noa8` (`w4a16-vllm-gptq-standard-20260512-134032`). Compared against the same vLLM bf16 baseline (`u8fbcn7l`) cyankiwi was scored against, plus the existing vLLM-track quants.

### Composite + retention

| Run | Composite | Retention vs bf16 | Energy vs bf16 |
|---|---:|---:|---:|
| `base-vllm` (bf16) | 1.0000 | 1.0000 | — |
| `fp8-w8a8-vllm` | 0.9748 | 0.9748 | −24.8% |
| **`w4a16-vllm` (cyankiwi)** | **0.9783** | **0.9783** | **−43.7%** |
| `bnb-nf4-vllm` | 0.9358 | 0.9358 | −27.9% |
| **`w4a16-vllm-gptq` (Vishva007)** | **0.8348** | **0.8348** | **−35.3%** |

Vishva007 GPTQ clears the 0.80 retention bar by only 3.5 absolute points — cleared, but the slimmest margin of any quant on the matrix and a **14.4 absolute composite-point loss to cyankiwi**. Loses to bnb-nf4-vllm by ~10pp too. The −35.3% energy is better than fp8-w8a8 (−24.8%) and bnb-nf4-vllm (−27.9%) but trails cyankiwi (−43.7%) by ~8pp. **Loss on both axes vs cyankiwi — not a tradeoff.**

### Per-task retention

| Task | Metric | bf16 | fp8-w8a8 | bnb-nf4 | cyankiwi | **Vishva007** | gptq ret |
|---|---|---:|---:|---:|---:|---:|---:|
| caption | bleu4 | 0.1085 | 0.1053 | 0.1063 | 0.1097 | **0.0852** | **0.785** |
| chart | relaxed_acc | 0.7150 | 0.6800 | 0.6550 | 0.6700 | **0.5550** | **0.776** |
| docvqa | anls | 0.7888 | 0.8017 | 0.7075 | 0.7754 | **0.7189** | **0.911** |
| ocr | anls | 0.7431 | 0.7265 | 0.7062 | 0.7381 | **0.6438** | **0.867** |

Vishva007 GPTQ loses on **every task**. Worst hits: chart (−16.1pp absolute relaxed_acc; retention 0.776, the only sub-0.80 task-retention number on the entire vLLM matrix) and ocr (−9.9pp absolute anls). caption bleu4 drops 21.5% relative — the metric is variance-prone at N=200, but the drop is well outside what cyankiwi/fp8/bnb-nf4 showed (all three within ±3% of bf16 on caption), so it's not just noise.

### Per-task energy (kWh) and latency (ms/sample mean)

| Task | bf16 kWh | **gptq kWh** | Δ | bf16 ms | **gptq ms** | Δ | cyankiwi ms (ref) |
|---|---:|---:|---:|---:|---:|---:|---:|
| caption | 0.00724 | **0.00362** | **−50.0%** | 981 | **496** | **−49.5%** | 522 |
| chart   | 0.00514 | **0.00381** | **−25.8%** | 678 | **503** | **−25.8%** | 362 |
| docvqa  | 0.00288 | **0.00214** | **−25.6%** | 338 | **247** | **−26.9%** | 216 |
| ocr     | 0.00334 | **0.00247** | **−25.9%** | 436 | **322** | **−26.2%** | 253 |

The energy/latency pattern is uneven — caption alone gets the dramatic −50% savings; the other three tasks all settle around −26%. And against cyankiwi specifically, **Vishva007 GPTQ is slower on chart / docvqa / ocr** (e.g. chart 503 ms vs cyankiwi 362 ms, a 38.9% latency penalty) and only edges it on caption (496 vs 522). gptq-marlin at gs=128 evidently dispatches slower per-token than compressed-tensors W4A16 at gs=32 on this Ada path, despite both running the same Marlin kernel family.

### Peak VRAM

`avg_peak_vram_gb = 20.125 GiB` — same KV-cache-pool dominance as every other vLLM run; weight footprint shows up only in startup logs (vLLM startup line: weights-only number ~9.2 GiB expected based on PLE-quantized scope but not captured in summary; comparable to cyankiwi's 9.48 GiB).

### Two tasks where Vishva007 GPTQ failed the implicit retention bar

* **chart (0.776 retention)** — first sub-0.80 task-retention measurement on the vLLM matrix. PLE-quantization plausibly compounds with chart_qa's exact-numeric-match sensitivity: the per-layer PLE projections control how strongly per-layer embeddings inject into the residual stream, and quantizing them at gs=128 sym evidently corrupts the fine numeric features that chart questions require. cyankiwi (PLE preserved) gives 0.937 on the same task.
* **ocr (0.867 retention)** — best of the four for Vishva007, but still well below the other three quants (cyankiwi 0.993, fp8 0.978, bnb-nf4 0.950). The same PLE-corruption story fits: OCR demands character-level precision the per-layer plumbing helps preserve.

### What this answers (the paired comparison from the analysis section above)

The pre-eval analysis framed Vishva007-vs-cyankiwi as the test of two coupled choices: **PLE-quantization** (Vishva007: yes, +84 Linears; cyankiwi: no) and **gs=128 sym** (Vishva007) vs **gs=32 asym** (cyankiwi). The result is unambiguous:

> **cyankiwi's conservatism was earning its accuracy keep.** PLE-quantization + coarser-grain + symmetric is not safe on this architecture. The +84 quantized Linears (per-layer PLE `input_gate` + `projection`) likely account for the bulk of the accuracy loss — those are precisely the modules that broke our own `llm-compressor` recipe; Vishva007's AutoRound 0.13 in RTN mode produced loadable outputs but at a real retention cost that the gs=128 sym scheme amplified rather than masked.

There's no axis on which Vishva007 GPTQ is preferable to cyankiwi: cyankiwi wins composite by 14.4pp, every per-task retention number, energy savings by ~8pp, and latency on three of four tasks.

### Vishva007 AWQ — should we still run it?

The pre-eval doc proposed Vishva007 AWQ only if "we want the AWQ kernel data point specifically." Given Vishva007 GPTQ landed at 0.835 with the same quantization scope as Vishva007 AWQ (342 Linears, gs=128, sym, RTN mode — only the on-disk packing differs), the AWQ variant is highly likely to land within ±1pp of GPTQ. We get no new science from running it. **Park unless an AWQ-kernel-specific question surfaces later.**

### Not yet verified (open questions for any follow-up)

* Whether finer-grain GPTQ (gs=32 or gs=64) would close the gap to cyankiwi without preserving PLE. Probably yes for the grain effect, no for the PLE effect — the chart drop strongly implicates PLE-quantization specifically, not just the grain.
* Whether a Vishva007-style scope (PLE-quantized) at asymmetric + gs=32 could match cyankiwi. No such community checkpoint exists; would need to be produced. Not worth pursuing unless cyankiwi develops a problem.

## Revisions to prior memory

* `reference_model_free_ptq_dead_end.md` describes our own W4A16 attempt failing. With cyankiwi proving a clean compressed-tensors W4A16 *is* producible (just not via our recipe / our llm-compressor version), the dead-end label is correct for our path but should not generalize to "W4A16 impossible on this model." Update after first successful smoke-load of a community checkpoint.
* `project_stage1_state.md` lists "AWQ/GPTQ/INT8 W8A8/W4A16 pending on vLLM." If we adopt one of these checkpoints, the W4A16-on-vLLM row gets filled by a community artifact rather than our own calibration. Worth annotating provenance clearly so the comparison vs HF-track results is fair.
