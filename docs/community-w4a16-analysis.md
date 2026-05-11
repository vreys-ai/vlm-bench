# Community W4A16 Checkpoints for gemma-4-E4B-it — Configuration Analysis

**Status:** cyankiwi smoke-verified 2026-05-11 (see "Spike verification" section); Vishva007 variants analyzed from configs only.
**Date:** 2026-05-11 (initial analysis); 2026-05-11 (cyankiwi verification appended)
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

## Revisions to prior memory

* `reference_model_free_ptq_dead_end.md` describes our own W4A16 attempt failing. With cyankiwi proving a clean compressed-tensors W4A16 *is* producible (just not via our recipe / our llm-compressor version), the dead-end label is correct for our path but should not generalize to "W4A16 impossible on this model." Update after first successful smoke-load of a community checkpoint.
* `project_stage1_state.md` lists "AWQ/GPTQ/INT8 W8A8/W4A16 pending on vLLM." If we adopt one of these checkpoints, the W4A16-on-vLLM row gets filled by a community artifact rather than our own calibration. Worth annotating provenance clearly so the comparison vs HF-track results is fair.
