# HQ Sampler Debug Handoff — 2026-03-10

## Goal
Get the HQ sampler (res2s second-order Runge-Kutta) working for LTX-2 video generation.
The HQ pipeline uses the **dev model** (full CFG-guided) instead of the distilled model,
with the distilled LoRA applied at partial strength to stabilize outputs.

## What Works
- **Distilled BF16 + res2s + CFG** → GOOD output (std=1.21, range [-4.8, 4.8])
- **Distilled FP8 + res2s + CFG** → GOOD output (std=1.17, range [-5.5, 5.6])
  - Test: `/tmp/test_distilled_fp8_res2s.py`
  - Output: `/home/alex/serenity/output/ltx2/ltx2_hq_20260310_110717_s4242.mp4` (verified good frame at `/tmp/distilled_fp8_hq_frame.png`)
- This proves: **FP8 + Stagehand block-swap + res2s = fine**. The sampler, scheduler, and pipeline plumbing all work.

## What Doesn't Work
- **Dev-FP8 model** (with or without distilled LoRA merged) through HQ pipeline → **garbage noise output**
- Every test with dev model weights produces output with ~2x std and ~3-4x range vs distilled

## Root Cause Chain (3 bugs found, 1 remaining)

### Bug 1: Merge Script Key Prefix Mismatch (FIXED)
- **File**: `/home/alex/ltx2-app/merge_hq_checkpoint.py`
- **Problem**: LoRA keys use `diffusion_model.X.lora_A.weight`, model keys use `model.diffusion_model.X.weight`. Original merge matched on `base_key` without stripping `model.` prefix → **0 layers merged**, checkpoint was just a copy of dev-fp8.
- **Fix**: Build `model_key_map` that strips `model.` prefix for matching. Now merges 1660 layers.

### Bug 2: FP8 Quantization Format Mismatch (ROOT CAUSE OF GARBAGE — PARTIALLY FIXED)
- **The critical discovery**: Dev-FP8 and Distilled-FP8 use DIFFERENT FP8 formats:
  - **Distilled-FP8**: 2408 FP8 tensors, **0 scale keys** → "cast-only" FP8
  - **Dev-FP8**: 1496 FP8 tensors, **2992 scale keys** (1496 weight_scale + 1496 input_scale) → "scaled MM" FP8
- **Our pipeline** (`pipeline.py:378`) detects "fp8" in filename → applies `QuantizationPolicy.fp8_cast()`
- **fp8_cast inference** (`ltx_core/quantization/fp8_cast.py:77-86`): replaces `Linear.forward` with `_upcast_and_round(layer.weight, x.dtype)` which does `weight.to(dtype)` — converts raw FP8 values to BF16 **WITHOUT multiplying by weight_scale**
- **Result**: For scaled-FP8 weights, the true weight = fp8_value * weight_scale. But fp8_cast just uses fp8_value directly. The weights are used at WRONG magnitudes → garbage output.
- **This is why distilled-FP8 works**: it has NO scales, so cast-only is correct.

### Bug 3: ltx-core LoRA Fusion Transpose Bug (BLOCKED runtime LoRA approach)
- **File**: `/home/alex/LTX-2/packages/ltx-core/src/ltx_core/loader/fuse_loras.py:122`
- **Problem**: `_fuse_delta_with_scaled_fp8` does `weight.t()` assuming all FP8 weights are stored transposed. For non-square weights like `gate_logits` (32, 4096), transpose gives (4096, 32) but LoRA delta is (32, 4096) → size mismatch crash.
- **Error**: `RuntimeError: The size of tensor a (32) must match the size of tensor b (4096) at non-singleton dimension 1`
- **Impact**: Cannot apply distilled LoRA to dev-fp8 at runtime via ModelLedger. Rules out the "match official pipeline exactly" approach.

## The Solution (IN PROGRESS — merge script v3)

The merge script at `/home/alex/ltx2-app/merge_hq_checkpoint.py` was updated to:

1. **Convert scaled-FP8 → cast-only FP8** BEFORE merging:
   - For each FP8 weight with `weight_scale`: `true_weight = fp8_value * weight_scale`
   - Save as simple FP8: `true_weight.to(float8_e4m3fn)` (no scale)
   - **Drop all `weight_scale` and `input_scale` keys** from output
   - This makes the checkpoint compatible with `fp8_cast` policy

2. **Then merge LoRA** into the now-cast-only FP8 weights:
   - Simple: `(weight.to(float32) + delta).to(weight.dtype)`

3. **Save with original metadata** (config, model_version, license, etc.)

**STATUS**: The v3 merge script is written and saved but **NOT YET RUN**. The session crashed before execution.

## Files Modified

### `/home/alex/ltx2-app/merge_hq_checkpoint.py` (REWRITTEN — v3, not yet run)
- Added `convert_to_cast_fp8()` function
- Fixed key prefix mapping (`model.` → stripped)
- Simplified LoRA merge (no scale handling needed after conversion)
- Preserves safetensors metadata from original dev-fp8

### `/home/alex/ltx2-app/pipeline.py` (MODIFIED — has debug logging)
Lines changed:
- **~803-818**: HQ ledger building — currently uses pre-merged checkpoints approach:
  ```python
  ledger_hq_s1 = self._build_ledger(checkpoint_override=cfg.hq_checkpoint_s1, skip_user_loras=True)
  ledger_s2 = self._build_ledger(checkpoint_override=cfg.hq_checkpoint_s2, skip_user_loras=True)
  ```
- **~349-352**: `_build_ledger()` has `skip_user_loras` and `distilled_lora_strength_override` params (added earlier)
- **~1100-1110**: `wrapped_fn` in `_generate_two_stage()` has DIAG logging (calls 1-5) that should be REMOVED once debugging is done:
  ```python
  if _call[0] <= 5:
      dv, da = result
      import torch as _t
      sig_val = sigmas[step_index].item() if step_index < len(sigmas) else -1
      logger.info(f"DIAG call {_call[0]} sigma={sig_val:.4f}: ...")
  ```
- **`wrapped_fn` parameter**: Changed from `sigmas_arg` to `sigmas` to match res2s keyword calling convention (line ~1095)
- **Sigma schedule**: Uses latent-aware shifting via `LTX2Scheduler().execute(latent=empty_latent, steps=hq_steps)` (line ~905)

## Existing Pre-Merged Checkpoints (STALE — need rebuild)
- `/home/alex/EriDiffusion/Models/ltx2/ltx2-hq-s1-fp8.safetensors` (28GB) — WRONG: has scaled-FP8 format with LoRA merged into raw FP8 values without dequantizing first
- `/home/alex/EriDiffusion/Models/ltx2/ltx2-hq-s2-fp8.safetensors` (28GB) — same issue

## Key Reference Files
- **Official HQ pipeline**: `/home/alex/LTX-2/packages/ltx-pipelines/src/ltx_pipelines/ti2vid_two_stages_hq.py`
  - Uses single dev checkpoint + distilled LoRA at runtime (0.25 s1, 0.5 s2)
  - Stage 1: `multi_modal_guider_denoising_func` (CFG with negative prompt)
  - Stage 2: `simple_denoising_func` (no CFG, just positive prompt)
  - No NAG
- **LoRA fusion code**: `/home/alex/LTX-2/packages/ltx-core/src/ltx_core/loader/fuse_loras.py` — has transpose bug at line 122
- **fp8_cast code**: `/home/alex/LTX-2/packages/ltx-core/src/ltx_core/quantization/fp8_cast.py` — inference hook at line 77-86
- **Quantization policy**: `/home/alex/LTX-2/packages/ltx-core/src/ltx_core/quantization/policy.py`
- **Scheduler**: `/home/alex/LTX-2/packages/ltx-core/src/ltx_core/components/schedulers.py` — `LTX2Scheduler` with token-count-dependent sigma shifting
- **res2s sampler**: `/home/alex/LTX-2/packages/ltx-pipelines/src/ltx_pipelines/utils/samplers.py` — line 261 positional, line 315 keyword `sigmas=`
- **Distilled LoRA**: `/home/alex/EriDiffusion/Models/ltx2/ltx-2.3-22b-distilled-lora-384.safetensors` — 3320 keys, `diffusion_model.` prefix, rank 32-384
- **Dev FP8 checkpoint**: `/home/alex/EriDiffusion/Models/ltx2/ltx-2.3-22b-dev-fp8.safetensors` — 8939 keys, scaled FP8
- **Distilled FP8 checkpoint**: `/home/alex/EriDiffusion/Models/ltx2/ltx-2.3-22b-distilled-fp8.safetensors` — 5947 keys, cast-only FP8

## Test Scripts
| Script | Purpose | Result |
|--------|---------|--------|
| `/home/alex/ltx2-app/test_hq_5s.py` | Main HQ test (pre-merged checkpoints) | Garbage (wrong FP8 format) |
| `/tmp/test_distilled_fp8_res2s.py` | Distilled-FP8 + res2s | **GOOD** — proves pipeline works |
| `/tmp/test_hq_dev_raw.py` | Raw dev-fp8, no LoRA | Garbage |
| `/tmp/test_dev_euler.py` | Dev + Euler (not res2s) | Garbage |
| `/tmp/test_distilled_res2s.py` | Distilled BF16 + res2s | **GOOD** |
| `/tmp/test_dev_no_fp8_policy.py` | Dev without fp8_cast policy | Garbage |
| `/tmp/test_dev_nan_check.py` | NaN diagnostic | 0 NaN, 0 Inf |
| `/tmp/test_hq_no_nag.py` | HQ with NAG disabled | Garbage (std=2.66) |

## Next Steps (in order)

1. **Run the v3 merge script** (`/home/alex/ltx2-app/merge_hq_checkpoint.py`):
   ```bash
   cd /home/alex/ltx2-app && /home/alex/serenity/venv/bin/python merge_hq_checkpoint.py
   ```
   This converts scaled-FP8 → cast-only FP8, then merges distilled LoRA. Should take ~10 min.

2. **Test with existing `test_hq_5s.py`** or create a new test. The pre-merged checkpoints should now be in cast-only FP8 format compatible with fp8_cast policy.

3. **If still garbage**: Check whether the naive FP8 cast (`true_weight.to(float8_e4m3fn)`) introduces too much quantization error. The dev model's weight range might be different from distilled. May need to compare a few weights before/after conversion.

4. **If works**: Remove DIAG logging from `pipeline.py` wrapped_fn, do A/B quality comparison vs distilled mode.

5. **Consider NAG for HQ mode**: Official pipeline doesn't use NAG. Our pipeline applies NAG unconditionally. May need to disable NAG when `use_hq=True` in `_generate_two_stage()` (lines ~1114-1118 and ~1176-1179).

## FP8 Format Summary

| Checkpoint | FP8 Tensors | Scale Keys | Format | Correct Policy |
|-----------|------------|------------|--------|---------------|
| distilled (BF16) | 0 | 0 | All BF16 | None |
| distilled-fp8 | 2408 | 0 | Cast-only FP8 | fp8_cast |
| dev-fp8 | 1496 | 2992 | Scaled MM FP8 | fp8_scaled_mm (needs tensorrt_llm!) |
| HQ pre-merged (v2, current) | 1496 | 2992 | Scaled MM FP8 (WRONG) | fp8_cast applied incorrectly |
| HQ pre-merged (v3, planned) | ~1496+2408? | 0 | Cast-only FP8 | fp8_cast (correct) |

## Key Diagnostic Stats

| Test | Call 1 std | Call 1 range | Result |
|------|-----------|-------------|--------|
| Distilled BF16 + res2s | 1.21 | [-4.8, 4.8] | GOOD |
| Distilled FP8 + res2s | 1.17 | [-5.5, 5.6] | GOOD |
| Dev FP8 raw (no LoRA) | 1.98 | [-15.9, 17.0] | Garbage |
| HQ pre-merged v2 + NAG | 2.35 | [-18.6, 14.8] | Garbage |
| HQ pre-merged v2 no NAG | 2.66 | [-16.0, 16.8] | Garbage |

The working runs have std ~1.2, range ~[-5, 5]. Garbage runs have std >1.9, range >[-15, 15].
The expected behavior after v3 merge: stats should be between distilled and dev (LoRA at 0.25 pulls toward distilled).
