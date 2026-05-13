# Calibrated W4A16 quantization script (`scripts/quantize_cyankiwi.py`)

**Status:** Design approved 2026-05-13. Implementation shipped 2026-05-13 with one significant divergence from this design — see "Post-ship divergence" at the bottom of this file.

**Motivation:** Stage 1 vLLM-track standard eval ranks `cyankiwi/gemma-4-E4B-it-AWQ-INT4` at composite retention **0.978 / −43.7% energy** — the strongest W4A16 result we have, and notably stronger than the paired Vishva007 W4A16 GPTQ at 0.835 / −35.3%. The cyankiwi checkpoint is a community artifact we don't control. To run knob-level ablations (which parameter actually carries the retention edge — group_size, observer, asymmetry, calibration data, or some combination?) we need a script that produces a checkpoint with cyankiwi's observable config and exposes those knobs as CLI flags.

**Scope:** One new script, `scripts/quantize_cyankiwi.py`. The user has already created a stub at this path (currently a copy of `quantize_w4a16.py` / `quantize_fp8.py` — both `model_free_ptq` siblings); this design replaces its body. No changes to `llm_bench_cc/` package code. No changes to the eval pipeline.

## Background: what cyankiwi's config actually says

Fetched from `https://huggingface.co/cyankiwi/gemma-4-E4B-it-AWQ-INT4/raw/main/config.json` on 2026-05-13:

```json
"quantization_config": {
  "config_groups": {
    "group_0": {
      "format": "pack-quantized",
      "input_activations": null,
      "output_activations": null,
      "targets": ["Linear"],
      "weights": {
        "actorder": null,
        "block_structure": null,
        "dynamic": false,
        "group_size": 32,
        "num_bits": 4,
        "observer": "mse",
        "strategy": "group",
        "symmetric": false,
        "type": "int"
      }
    }
  },
  "ignore": [...],            // see "ignore list" below
  "quant_method": "compressed-tensors",
  "transform_config": {},
  "sparsity_config": {}
}
```

Despite the repo name `AWQ-INT4`, the config is observably consistent with a plain calibrated W4A16: `transform_config: {}` (no AWQ smoothing scales recorded), `input_activations: null` (weight-only). The repo README credits a "STEM and Agentic" calibration set at `cyankiwi/calibration`, which is consistent either with AWQModifier whose transforms were baked into weights pre-pack (leaving `transform_config` empty), or with a vanilla calibrated W4A16 run whose MSE observer scanned real activations.

**Decision:** target the "vanilla calibrated W4A16" interpretation. Reasons: (a) it's the simpler hypothesis that fits all observable evidence; (b) the failure mode we already have memory about — oneshot's three-deep failure on Gemma 4 E-variants from per-layer-embeddings + KV-sharing — applies to both interpretations equally, so picking AWQ doesn't dodge any risk; (c) if vanilla calibrated W4A16 reproduces cyankiwi's ~0.978 retention, we've answered the question and an AWQ script becomes a separate sibling later.

## Recipe (matches cyankiwi's `weights` block exactly)

```python
SCHEME_KWARGS = dict(
    num_bits=4,
    group_size=32,         # cyankiwi: 32; llmcompressor default: 128
    strategy="group",
    symmetric=False,       # cyankiwi: asymmetric
    observer="mse",        # cyankiwi: mse; llmcompressor default: minmax
    actorder=None,
    dynamic=False,
)
TARGETS = ["Linear"]
```

`format="pack-quantized"` and `type="int"` are inferred by llmcompressor from `num_bits=4` + `strategy="group"`; we don't pass them explicitly. The cyankiwi config records them in the final `config.json`, but they're outputs of the recipe, not inputs.

## Ignore list (semantically equivalent to cyankiwi's per-module enumeration)

cyankiwi's config enumerates every excluded submodule by full dotted name (~300 entries). The regex form below is semantically identical given `targets=["Linear"]`:

```python
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

Difference from the existing `quantize_w4a16.py`: we drop the `re:.*embed_tokens.*` defensive entry. `targets=["Linear"]` already excludes Embedding modules; cyankiwi's config does the same and we want byte-for-byte parity on the `quantization_config` block.

## Pipeline flow

1. **Auth preflight** — `_check_auth` from the existing stub. Unchanged.
2. **Patch `TORCH_INIT_FUNCTIONS`** — required before any llmcompressor import (oneshot still pulls in `utils.dev`, same import chain as `model_free_ptq`). Unchanged from the stub.
3. **Load model + processor:**
   ```python
   model = AutoModelForImageTextToText.from_pretrained(
       model_id, dtype=torch.bfloat16, device_map="auto",
       max_memory={0: args.max_gpu_memory, "cpu": "64GiB"},
   )
   processor = AutoProcessor.from_pretrained(model_id)
   ```
   `device_map="auto"` with a capped GPU budget spills audio/vision towers to CPU. They're in `IGNORE` and never touched during LM calibration anyway, so CPU residence has no measurable cost.
4. **Build calibration dataloader:**
   ```python
   ds = load_dataset(args.calibration_dataset, split=args.calibration_split, streaming=False)
   ds = ds.shuffle(seed=args.seed).select(range(args.num_calibration_samples))
   # column detection: try 'instruction', 'prompt', 'text' in priority order;
   # apply chat template if 'instruction' or 'prompt' matches, raw otherwise
   ```
   Hard error with the dataset's actual column names listed if none match.
5. **Try GPTQModifier oneshot:**
   ```python
   from llmcompressor import oneshot
   from llmcompressor.modifiers.quantization import GPTQModifier
   recipe = GPTQModifier(
       targets=TARGETS, ignore=IGNORE, scheme=SCHEME_KWARGS,
       block_size=args.gptq_block_size, dampening_frac=args.gptq_dampening_frac,
   )
   oneshot(model=model, dataset=calib_ds, recipe=recipe,
           max_seq_length=args.max_seq_length,
           num_calibration_samples=args.num_calibration_samples)
   entrypoint_used = "llmcompressor.oneshot+GPTQModifier"
   ```
6. **Fallback on known E-variant failure modes:** narrow catch list (`torch.fx.proxy.TraceError`, `RuntimeError`, `AttributeError`) with a marker-string filter:
   ```python
   except (torch.fx.proxy.TraceError, RuntimeError, AttributeError) as e:
       if not _is_known_e_variant_failure(e):
           raise
       logger.warning("GPTQModifier failed on E-variant (%s: %s) — retrying observer-only",
                      type(e).__name__, str(e)[:200])
       # rebuild model fresh — GPTQ may have mutated weights before crashing.
       # Same kwargs as step 3 (bf16, device_map="auto", capped max_memory).
       model = _load_model(args)  # factored helper, called in both step 3 and here
       from llmcompressor.modifiers.quantization import QuantizationModifier
       recipe = QuantizationModifier(targets=TARGETS, ignore=IGNORE, scheme=SCHEME_KWARGS)
       oneshot(model=model, dataset=calib_ds, recipe=recipe,
               max_seq_length=args.max_seq_length,
               num_calibration_samples=args.num_calibration_samples)
       entrypoint_used = "llmcompressor.oneshot+QuantizationModifier"
   ```
   `_is_known_e_variant_failure` matches error message substrings: `per_layer_input_gate`, `num_kv_shared_layers`, `proxy`, `fx`, `sequential pipeline`. Unknown failures re-raise — better to surface a new mode than silently downgrade quality.
7. **Save:** `model.save_pretrained(output_dir, save_compressed=True)` + `processor.save_pretrained(output_dir)`. The standard llmcompressor save path; writes packed `*.safetensors`, the `quantization_config` block in `config.json`, and (via `processor.save_pretrained`) all multimodal aux files including `preprocessor_config.json`. The existing stub's `_pull_processor_aux_files` becomes unnecessary — we keep the function definition in the file but don't call it, with a comment pointing to the `processor.save_pretrained` line that supersedes it.
8. **Write `quant_recipe.json`:** provenance payload extended from the stub:
   ```json
   {
     "timestamp_utc": "...",
     "git_sha": "...",
     "model_id": "google/gemma-4-E4B-it",
     "entrypoint": "<entrypoint_used from steps 5 or 6>",
     "scheme_kwargs": { ...SCHEME_KWARGS... },
     "ignore_patterns": [...IGNORE...],
     "calibration": {
       "dataset": "garage-bAInd/Open-Platypus",
       "split": "train",
       "num_samples": 128,
       "max_seq_length": 2048,
       "seed": 42
     },
     "based_on": "https://huggingface.co/cyankiwi/gemma-4-E4B-it-AWQ-INT4/blob/main/config.json"
   }
   ```

## CLI flags

| Flag | Default | Notes |
|---|---|---|
| `--model-id` | `google/gemma-4-E4B-it` | unchanged from stub |
| `--output-dir` | required | unchanged from stub |
| `--calibration-dataset` | `garage-bAInd/Open-Platypus` | Hub repo id |
| `--calibration-split` | `train` | overridable for validation-only datasets |
| `--num-calibration-samples` | `128` | smoke-sized to find failure modes fast |
| `--max-seq-length` | `2048` | matches llmcompressor INT4 examples |
| `--group-size` | `32` | cyankiwi value; flag for the knob sweep |
| `--observer` | `mse` | cyankiwi value; `minmax` also valid |
| `--symmetric` / `--no-symmetric` | `--no-symmetric` | cyankiwi is asymmetric |
| `--gptq-block-size` | `128` | llmcompressor GPTQ default |
| `--gptq-dampening-frac` | `0.01` | llmcompressor GPTQ default |
| `--max-gpu-memory` | `22GiB` | leaves ~2GiB headroom on L4 24GB |
| `--seed` | `42` | controls calibration shuffle |
| `--skip-gptq` | `False` | force the QuantizationModifier path (debug aid) |

No `--device` flag — `device_map="auto"` is doing the placement work; explicit `--device` would fight it.

## Error handling

| Failure | Action |
|---|---|
| HF auth fails | hard error, existing `_check_auth` message |
| Calibration dataset gated/missing | hard error with dataset name + `huggingface-cli login` hint |
| Tokenizer can't find known text column | hard error listing actual column names + priority list tried |
| Model load OOM | hard error suggesting `--max-gpu-memory` reduction |
| `GPTQModifier` raises a known-marker exception | logged warning, rebuild model, retry observer-only |
| `GPTQModifier` raises anything else | re-raise — new failure modes must be visible |
| `QuantizationModifier` fallback raises | re-raise; no safety nets left |
| Save fails | re-raise |

## Verification plan

1. **Wiring check:** `--num-calibration-samples 8 --max-seq-length 256` → confirm pipeline reaches save in ~2 min. Don't keep the output.
2. **Real run:** `--num-calibration-samples 128 --max-seq-length 2048` to the actual output dir.
3. **Provenance check:** `cat <out>/quant_recipe.json` — confirm `entrypoint` field reflects which modifier produced the artifact (GPTQ vs observer-only); this gets surfaced to wandb on the eval run.
4. **Load roundtrip:** `AutoModelForImageTextToText.from_pretrained(out_dir, dtype="auto")` — minimum bar before eval GPU time.
5. **Smoke eval:** `python -m llm_bench_cc.eval --model <out> --tier smoke` (existing pipeline). Out of scope for this script — quantize and eval are separate concerns.

## Out of scope (YAGNI)

- Multi-GPU support — L4 single-device is the only target hardware.
- Checkpoint/resume — the expensive part is GPTQ's Hessian accumulation, which has no natural checkpoint boundary. Re-run from scratch on failure.
- The `cyankiwi/calibration` dataset's specific schema — if a user passes `--calibration-dataset cyankiwi/calibration` and the column auto-detect fails, the "known columns" error gives them what they need to preprocess separately.
- AWQ. If the calibrated W4A16 path here reproduces ~0.978 retention, the AWQ question is answered (recipe knobs explain the edge). If it doesn't, a sibling `quantize_awq.py` is the right place to investigate — separate provenance, separate failure modes.

## Open risks

- **Oneshot may still fail on Gemma 4 E-variants.** The memory note `[[reference_llmcompressor_entrypoints]]` flags a three-deep failure chain (fx-trace on PLE access, sequential pipeline state on KV-shared layers). Newer llmcompressor (≥0.1.dev483, which cyankiwi's checkpoint is stamped with) may or may not have fixed this. The GPTQ→observer-only fallback is our primary mitigation. If **both** paths fail, the next move is a PLE-aware sequential-pipeline shim, which is a separate piece of work and gets its own design doc.
- **L4 24GB headroom.** Model bf16 ≈ 16 GB; GPTQ adds per-Linear Hessian buffers (256×256 bf16 ≈ 128 KB each, ~50 MB total across the LM); calibration forward passes with KV cache on 2048-token sequences add a few GB. Tight but should fit with `--max-gpu-memory 22GiB`. If OOM strikes, lower `--num-calibration-samples` or `--max-seq-length` per the error message.

## Provenance

- Based on cyankiwi's published `config.json` (URL above), fetched 2026-05-13.
- Sibling pattern follows existing `scripts/quantize_fp8.py` and `scripts/quantize_w4a16.py`.
- llmcompressor entrypoint choice and oneshot vs model_free_ptq tradeoffs follow `[[reference_llmcompressor_entrypoints]]`.

## Post-ship divergence (2026-05-13)

The original design specified an automatic `GPTQModifier → QuantizationModifier` fallback on known E-variant marker exceptions. The implementation initially shipped that design, but Colab L4 smoke runs surfaced two blockers that the design had not anticipated:

1. **`scheme=` parameter shape.** `GPTQModifier`'s `scheme=` parameter validates as either a string preset name or `{preset_name: targets}` — it rejects custom `QuantizationArgs` dicts. Fixed by switching to `config_groups=` with explicit `{group_0: {targets, weights, input_activations, output_activations}}`. This is also closer to cyankiwi's saved `config.json` structure.

2. **The GPTQ → in-place fallback path crashes at save.** GPTQ at fx-trace failure registers `weight_scale` Parameter slots on each Linear before crashing. QuantizationModifier-on-the-same-model writes those slots in a way inconsistent with `accelerate`'s offload tracking (we use `device_map="auto"`). At save, llmcompressor's `from_accelerate` cleanup hits a `TypeError`: it tries to `setattr(module, 'weight_scale', tensor)` but the Parameter slot rejects a raw Tensor. The original design also called for a "fresh rebuild" between attempts — that's infeasible on L4 24 GB because the bf16 model fills ~15 GiB and `del + gc + empty_cache` doesn't reliably release VRAM (the llmcompressor session retains submodule back-refs).

**What shipped instead:**

- Default path: `QuantizationModifier` (observer-only). The proven-working path on Gemma 4 E-variants, and also the closer match to cyankiwi's published config (`actorder: null`, no Hessian fingerprint in `config_groups`).
- `--try-gptq` opt-in flag for `GPTQModifier`. Failure under this flag is **terminal** — no automatic fallback, error message points to re-running without the flag.
- `_run_gptq`, `_is_known_e_variant_failure`, and the marker substrings are left in place behind the opt-in flag for future llmcompressor / transformers fixes where GPTQ on Gemma 4 E becomes viable.

**Why this is acceptable:** cyankiwi's published config has no GPTQ fingerprint either. Observer-only is the more faithful replica of their recipe. The script still hits its primary goal: produce a calibrated W4A16 checkpoint with the cyankiwi knob set (group_size=32, observer=mse, asymmetric) so the retention edge can be ablated.

**Open follow-ups:**

- If we ever want GPTQ to work on Gemma 4 E (e.g., to compare GPTQ vs observer-only as a knob), the structural fix is to drop `device_map="auto"` and load single-device. The model fits on L4 24 GB without offloading (LM 16 + vision 0.8 + audio 0.3 ≈ 17 GB). No `accelerate` hooks → no `from_accelerate` cleanup → no Parameter/Tensor type clash. Not shipped here because (a) the proven path works, (b) the eval ablation doesn't require GPTQ.
