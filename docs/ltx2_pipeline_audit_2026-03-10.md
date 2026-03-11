# LTX-2 Pipeline Audit — 2026-03-10

Comprehensive audit of LTX-2 pipeline stack (`/home/alex/LTX-2/packages/`) vs our implementation (`/home/alex/ltx2-app/`).
Five parallel agents analyzed samplers, guidance, pipelines, VAE/audio, and text encoder/loader systems.

## Summary of Fixes Applied

| # | Issue | Severity | Status |
|---|-------|----------|--------|
| 1 | `hq_distilled_lora_strength_s2` default 0.25 → 0.5 | Critical | Fixed |
| 2 | res2s `noise_seed` not tied to generation seed | Medium | Fixed |
| 3 | HQ mode wasn't being used (distilled pipeline running instead) | Critical | Fixed (earlier session) |
| 4 | Negative prompt not encoded in HQ mode | Critical | Fixed (earlier session) |

## Findings by Category

### 1. Samplers & Denoising Loops

**What exists in LTX-2:**
- `euler_denoising_loop` — Standard first-order Euler (our distilled mode)
- `res2s_audio_video_denoising_loop` — Second-order Runge-Kutta with SDE noise injection + bong iteration refinement (our HQ mode)
- `gradient_estimating_euler` — Velocity-tracking Euler variant that maintains gradient history for smoother trajectories

**What we use:** euler (distilled), res2s (HQ)

**Gap:** `gradient_estimating_euler` is unused. It could serve as a quality middle ground — better than Euler, faster than res2s (single model eval per step vs two). Worth testing for a "medium quality" preset.

**Schedulers available but unused:**
- `LinearQuadraticScheduler` — front-loads noise removal, could improve early structure formation
- `BetaScheduler` — beta-distribution-weighted schedule, could improve specific content types
- Both in `ltx_core.components.schedulers`

**res2s parameters we match official HQ:**
- `bongmath=True`, `bongmath_max_iter=100` (defaults) ✓
- `legacy_mode=True` (default) ✓
- `noise_seed` now tied to generation seed ✓

### 2. Guidance & Guiders

**What exists:**
- `MultiModalGuider` — Constant CFG guidance (used in HQ mode)
- `MultiModalGuiderFactory` — Sigma-dependent guidance that varies scale with noise level (used in dev mode)
- `STG` (Spatiotemporal Guidance) — Selective attention perturbation via `stg_blocks`

**HQ mode correctly:**
- Uses `MultiModalGuider` (not factory) ✓
- Disables STG (`stg_scale=0.0`, `stg_blocks=[]`) ✓
- Uses CFG `video_cfg_scale=3.0`, `video_rescale=0.45` ✓

**Issue found:** `audio_rescale` should be `1.0` in HQ mode (per `LTX_2_3_HQ_PARAMS`), our default is `0.7`. Not impactful for video-only generation but will matter for A2V. No fix needed yet.

**Unused perturbation types:**
- `SKIP_A2V_CROSS_ATTN` — Skip audio-to-video cross attention
- `SKIP_V2A_CROSS_ATTN` — Skip video-to-audio cross attention
- These could improve audio-video coherence as selective guidance for A2V generation

### 3. Pipelines & Conditioning

**7 pipeline variants exist in LTX-2:**

| # | Pipeline | Description | We Implement? |
|---|----------|-------------|---------------|
| 1 | `TI2VidTwoStagesDistilled` | Fast distilled (8 Euler + 3 refine) | Yes |
| 2 | `TI2VidTwoStagesHQ` | HQ res2s + CFG (15 steps) | Yes |
| 3 | `TI2VidTwoStagesDev` | Full dev mode (30 steps, CFG+STG) | Partial (via dev mode) |
| 4 | `KeyframeInterpolation` | FML2V keyframe injection | Yes (UI wired) |
| 5 | `ICLoRA` | In-Context LoRA (reference image style transfer) | No |
| 6 | `RetakePipeline` | Temporal region regeneration | No |
| 7 | `AudioVideoDistilled` | Audio-conditioned video | Partial (A2V supported) |

**IC-LoRA (In-Context LoRA):**
- Uses reference image downscaling + attention masking for zero-shot style transfer
- No fine-tuning needed — injects style via attention mechanism
- Could be valuable for consistent character/style across generations
- Implementation complexity: moderate (needs reference processing + attention mask injection)

**RetakePipeline:**
- Re-renders specific temporal regions of a generated video
- Useful for fixing a bad section without regenerating the whole clip
- Implementation complexity: moderate (temporal masking + partial denoising)

**LoRA strength mismatch (FIXED):**
- Official HQ defaults: s1=0.25, s2=0.5
- Our s2 was 0.25 — now corrected to 0.5

### 4. VAE, Upsampler & Audio

**Spatial upsampler:**
- Configured and working (2x upscale between stages) ✓
- Path: `spatial_upsampler_path` in config

**Temporal upsampler:**
- Config field exists (`temporal_upscaler_path`) but empty string (disabled)
- Would enable 2x frame interpolation (e.g., 25fps → 50fps)
- Model file not present — need to source the temporal upscaler checkpoint

**VAE tiling:**
- Our settings are reasonable for 24GB VRAM ✓
- `vae_temporal_tiles=2`, `vae_spatial_tile_pixels=512`, overlap=64px

**Video CRF control:**
- No per-generation CRF quality setting for output compression
- Currently hardcoded in the video writer
- Minor quality-of-life improvement for output file size control

**Decoder noise injection:**
- Implemented and configurable via `decoder_noise_*` settings ✓
- Adds subtle variation to decoded frames

### 5. Text Encoder & Model Loading

**Registry caching opportunity:**
- `ModelLedger` has `with_loras()`/`with_additional_loras()` methods for swapping LoRA configs without full pipeline rebuild
- Currently we rebuild the full ledger between stages
- Could save 5-10 seconds per generation by reusing the base ledger and only swapping LoRA strengths

**LoRA format detection:**
- Only diffusers format (`lora_A`/`lora_B` with `transformer.*` prefix) is detected
- Kohya format (different key naming) silently ignored
- ComfyUI format also not detected
- Users with Kohya/ComfyUI LoRAs would get silent failures

**Gemma3 text encoder:**
- Loading and caching working correctly ✓
- All 49 hidden states packed properly ✓

## Priority Recommendations

### Do Now (quick wins)
1. ~~Fix `hq_distilled_lora_strength_s2` default~~ ✓ Done
2. ~~Pass generation seed to res2s noise generators~~ ✓ Done

### Next Sprint
3. Add `gradient_estimating_euler` as "Medium Quality" option — single model eval, velocity tracking
4. Use `ModelLedger.with_loras()` for stage 2 to skip full rebuild
5. Add video CRF output quality control to UI

### Future Features
6. IC-LoRA pipeline for zero-shot style transfer
7. RetakePipeline for temporal region editing
8. Temporal upsampler integration (need model file)
9. Kohya/ComfyUI LoRA format auto-detection
10. Alternative schedulers (LinearQuadratic, Beta) as UI options
