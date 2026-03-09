"""Long video generation via temporal tiling with latent-space overlap conditioning.

Core operations:
- select_latents: temporal slicing
- blend_latent_overlap: linear alpha blend at overlap boundaries
- adain_normalize: match latent statistics to reference (prevents drift)
- extend_chunk: extend video by one temporal chunk
- LTXVLongVideoService: main looping orchestrator
"""

from __future__ import annotations

import logging
import time
from enum import Enum
from typing import TYPE_CHECKING, Any, Callable

import torch
import torch.nn.functional as F_

if TYPE_CHECKING:
    from pipeline import LTX2Pipeline

logger = logging.getLogger(__name__)


class UpscaleMode(str, Enum):
    NONE = "none"
    SPATIAL_PER_CHUNK = "spatial_per_chunk"
    FULL_PER_CHUNK = "full_per_chunk"
    SPATIAL_FINAL = "spatial_final"


# -- Latent utilities --


def select_latents(latent: dict, start: int, end: int) -> dict:
    """Slice latent along temporal axis (dim 2). Negative indices supported."""
    samples = latent["samples"]
    total_frames = samples.shape[2]

    s = start if start >= 0 else total_frames + start
    e = end if end >= 0 else total_frames + end
    e = min(e + 1, total_frames)  # inclusive to exclusive
    s = max(0, s)

    result = {"samples": samples[:, :, s:e]}
    if "noise_mask" in latent and latent["noise_mask"] is not None:
        result["noise_mask"] = latent["noise_mask"][:, :, s:e]
    return result


def blend_latent_overlap(
    prev: dict,
    new_chunk: dict,
    overlap: int,
    axis: int = 2,
) -> dict:
    """Fuse two latent chunks with linear alpha blend at the overlap region."""
    s1 = prev["samples"]
    s2 = new_chunk["samples"]

    if overlap == 0:
        return {"samples": torch.cat([s1, s2], dim=axis)}

    alpha = torch.linspace(
        1, 0, overlap + 2, device=s1.device, dtype=s1.dtype,
    )[1:-1]
    shape = [1] * s1.dim()
    shape[axis] = overlap
    alpha = alpha.reshape(shape)

    blended = (
        alpha * s1.narrow(axis, s1.shape[axis] - overlap, overlap)
        + (1 - alpha) * s2.narrow(axis, 0, overlap)
    )

    result = torch.cat(
        [
            s1.narrow(axis, 0, s1.shape[axis] - overlap),
            blended,
            s2.narrow(axis, overlap, s2.shape[axis] - overlap),
        ],
        dim=axis,
    )
    return {"samples": result}


def adain_normalize(
    latent: dict,
    reference: dict,
    factor: float = 1.0,
    per_frame: bool = False,
) -> dict:
    """AdaIN normalization: match latent statistics to reference.

    Prevents accumulated oversaturation in long generations.
    """
    if factor == 0.0:
        return latent

    t = latent["samples"].clone()
    r = reference["samples"]

    for i in range(t.shape[0]):
        for c in range(t.shape[1]):
            if per_frame:
                for f in range(t.shape[2]):
                    ref_f = min(f, r.shape[2] - 1)
                    r_mean = r[i, c, ref_f].mean()
                    r_std = r[i, c, ref_f].std()
                    i_mean = t[i, c, f].mean()
                    i_std = t[i, c, f].std()
                    if i_std > 1e-8:
                        t[i, c, f] = (t[i, c, f] - i_mean) / i_std * r_std + r_mean
            else:
                r_mean = r[i, c].mean()
                r_std = r[i, c].std()
                i_mean = t[i, c].mean()
                i_std = t[i, c].std()
                if i_std > 1e-8:
                    t[i, c] = (t[i, c] - i_mean) / i_std * r_std + r_mean

    result = latent.copy()
    result["samples"] = torch.lerp(latent["samples"], t, factor)
    return result


def downscale_latent(latent: dict, target_h: int, target_w: int) -> dict:
    """Bilinear downscale of latent spatial dims for overlap conditioning."""
    s = latent["samples"]
    B, C, T, H, W = s.shape
    s_flat = s.reshape(B * C * T, 1, H, W)
    s_down = F_.interpolate(
        s_flat, size=(target_h, target_w), mode="bilinear", align_corners=False,
    )
    return {"samples": s_down.reshape(B, C, T, target_h, target_w)}


def spatially_upscale_long_latent(
    latent: dict,
    spatial_upscaler: Any,
    video_encoder: Any,
    temporal_chunk_size: int = 25,
) -> dict:
    """Apply spatial upscaler to a long latent in temporal chunks."""
    from ltx_core.model.upsampler import upsample_video

    samples = latent["samples"]
    total_frames = samples.shape[2]

    if total_frames <= temporal_chunk_size:
        upscaled = upsample_video(samples, video_encoder, spatial_upscaler)
        return {"samples": upscaled}

    chunks = []
    pos = 0
    while pos < total_frames:
        end = min(pos + temporal_chunk_size, total_frames)
        chunk = samples[:, :, pos:end]
        upscaled_chunk = upsample_video(chunk, video_encoder, spatial_upscaler)
        chunks.append(upscaled_chunk)
        pos = end

    return {"samples": torch.cat(chunks, dim=2)}


# -- Extend chunk --


def extend_chunk(
    prev_latent: dict,
    pipeline: LTX2Pipeline,
    transformer: Any,
    xfm_runtime: Any,
    v_ctx: torch.Tensor,
    a_ctx: torch.Tensor | None,
    sigmas: torch.Tensor,
    overlap_frames: int = 24,
    num_new_frames: int = 80,
    adain_factor: float = 0.0,
    adain_reference: dict | None = None,
    seed: int = 42,
    report: Callable[[str, float], None] | None = None,
    chunk_idx: int = 0,
    pixel_width: int | None = None,
    pixel_height: int | None = None,
) -> dict:
    """Extend a video by one temporal chunk.

    1. Extract last `overlap_frames` latent frames from prev_latent
    2. Run base generation with overlap conditioning
    3. Apply AdaIN normalization if factor > 0
    4. Drop first output frame (8-frame artifact)
    5. Linear blend into prev_latent at overlap region
    """
    from ltx_core.components.diffusion_steps import EulerDiffusionStep
    from ltx_core.components.noisers import GaussianNoiser
    from ltx_core.types import VideoPixelShape
    from ltx_pipelines.utils.types import PipelineComponents

    if report is None:
        report = lambda msg, frac: None

    samples = prev_latent["samples"]
    B, C, F, H, W = samples.shape

    latent_overlap = overlap_frames // 8
    if latent_overlap < 1:
        latent_overlap = 1

    total_pixel_frames = overlap_frames + num_new_frames
    valid_pixel_frames = ((total_pixel_frames - 1) // 8) * 8 + 1
    overlap_tail = select_latents(prev_latent, -latent_overlap, -1)

    # Use explicit pixel dims if provided, otherwise infer (VAE factor = 32)
    pw = pixel_width if pixel_width is not None else W * 32
    ph = pixel_height if pixel_height is not None else H * 32

    chunk_shape = VideoPixelShape(
        batch=1,
        frames=valid_pixel_frames,
        width=pw,
        height=ph,
        fps=pipeline.config.fps,
    )

    # Pad overlap tail to full chunk temporal size (required by create_initial_state)
    target_latent_frames = (valid_pixel_frames - 1) // 8 + 1
    overlap_samples = overlap_tail["samples"]
    if overlap_samples.shape[2] < target_latent_frames:
        pad_frames = target_latent_frames - overlap_samples.shape[2]
        pad = torch.zeros(
            B, C, pad_frames, H, W,
            device=overlap_samples.device, dtype=overlap_samples.dtype,
        )
        padded = torch.cat([overlap_samples, pad], dim=2)
    else:
        padded = overlap_samples

    generator = torch.Generator(device=pipeline.device).manual_seed(seed + chunk_idx)
    noiser = GaussianNoiser(generator=generator)
    stepper = EulerDiffusionStep()
    components = PipelineComponents(dtype=pipeline.dtype, device=pipeline.device)

    video_state, audio_state = pipeline._stagehand_denoise(
        transformer, xfm_runtime, v_ctx, a_ctx, chunk_shape, sigmas,
        noiser, stepper, components, [],
        report, label=f"Chunk {chunk_idx}", progress_base=0.0, progress_range=1.0,
        noise_scale=sigmas[0].item(),
        initial_video_latent=padded.to(pipeline.device),
    )

    new_chunk = {"samples": video_state.latent}
    del video_state, audio_state

    # Drop first frame (8-frame boundary artifact)
    if new_chunk["samples"].shape[2] > 1:
        new_chunk["samples"] = new_chunk["samples"][:, :, 1:]

    # AdaIN normalization
    if adain_factor > 0.0 and adain_reference is not None:
        new_chunk = adain_normalize(new_chunk, adain_reference, factor=adain_factor)

    # Overlap shifted by 1 after first-frame drop
    effective_overlap = max(latent_overlap - 1, 0)
    return blend_latent_overlap(prev_latent, new_chunk, effective_overlap)


# -- Keyframe distribution --


def distribute_keyframes_to_chunks(
    keyframes: list[tuple[int, Any]],
    total_frames: int,
    temporal_tile_size: int,
    temporal_overlap: int,
) -> dict[int, list[tuple[int, Any]]]:
    """Map absolute frame indices to (chunk_idx, in_chunk_frame_idx) pairs."""
    step = temporal_tile_size - temporal_overlap
    result: dict[int, list[tuple[int, Any]]] = {}

    for frame_idx, image in keyframes:
        if frame_idx < temporal_tile_size:
            chunk_idx = 0
            in_chunk_idx = frame_idx
        else:
            remaining = frame_idx - temporal_tile_size
            chunk_idx = 1 + remaining // step
            in_chunk_idx = remaining % step + temporal_overlap

        if chunk_idx not in result:
            result[chunk_idx] = []
        result[chunk_idx].append((in_chunk_idx, image))

    return result


# -- Audio extension --


def extend_audio_latent(
    prev_audio: torch.Tensor | None,
    new_audio: torch.Tensor | None,
    overlap: int = 0,
) -> torch.Tensor | None:
    """Concatenate audio latents across chunks."""
    if prev_audio is None or new_audio is None:
        return new_audio if prev_audio is None else prev_audio

    if overlap <= 0 or overlap >= new_audio.shape[2]:
        return torch.cat([prev_audio, new_audio], dim=2)

    blended = (
        prev_audio[:, :, -overlap:] + new_audio[:, :, :overlap]
    ) / 2.0

    return torch.cat(
        [prev_audio[:, :, :-overlap], blended, new_audio[:, :, overlap:]],
        dim=2,
    )


# -- Per-step AdaIN patch --


def apply_per_step_adain_patch(
    denoise_fn: Callable,
    reference: dict,
    factors: list[float],
) -> Callable:
    """Wrap a denoise function to apply AdaIN after each step."""
    _step = [0]

    def patched_fn(video_state, audio_state, sigmas, step_index):
        result = denoise_fn(video_state, audio_state, sigmas, step_index)
        idx = _step[0]
        factor = factors[idx] if idx < len(factors) else 0.0
        if factor > 0.0:
            v_latent = result[0] if isinstance(result, tuple) else result
            latent_dict = {"samples": v_latent.latent.unsqueeze(0)}
            normed = adain_normalize(latent_dict, reference, factor=factor)
            v_latent.latent = normed["samples"].squeeze(0)
        _step[0] += 1
        return result

    return patched_fn


def add_long_memory_conditioning(
    accumulated: dict,
    chunk0_latent: dict,
    strength: float = 0.5,
) -> dict:
    """Blend chunk-0 statistics into accumulated latent as 'long memory'."""
    if strength <= 0.0:
        return accumulated
    return adain_normalize(accumulated, chunk0_latent, factor=strength)


# -- Long video service --


class LTXVLongVideoService:
    """Long video generation via temporal tiling.

    Default parameters (recommended starting point):
        temporal_tile_size: 80 frames (pixel space)
        temporal_overlap: 24 frames (pixel space)
        adain_factor: 0.3 (light normalization)
    """

    def generate(
        self,
        pipeline: LTX2Pipeline,
        prompt: str,
        width: int,
        height: int,
        total_frames: int,
        fps: float = 25.0,
        temporal_tile_size: int = 80,
        temporal_overlap: int = 24,
        adain_factor: float = 0.3,
        seed: int = 42,
        upscale_mode: UpscaleMode = UpscaleMode.SPATIAL_PER_CHUNK,
        long_memory_strength: float = 0.0,
        per_step_adain: bool = False,
        per_step_adain_factors: str = "0.9,0.75,0.5,0.25,0.0,0.0,0.0,0.0",
        per_chunk_prompts: list[str] | None = None,
        progress_callback: Callable[[int, int, int, int, float], None] | None = None,
    ) -> dict:
        """Generate a long video via temporal tiling.

        All chunks are generated at half resolution (width//2, height//2) to fit
        in 24GB VRAM — same as regular pipeline stage 1. The accumulated latent
        is spatially upscaled once at the end.

        Args:
            progress_callback: (chunk_idx, total_chunks, frames_generated, total_frames, elapsed)
        """
        import gc
        from long_video_presets import calculate_chunks, nearest_valid_frames
        from ltx_pipelines.utils.constants import DISTILLED_SIGMA_VALUES

        total_frames = nearest_valid_frames(total_frames)
        num_chunks, actual_total_frames = calculate_chunks(
            total_frames, temporal_tile_size, temporal_overlap,
        )

        # Half-res for generation (matches regular pipeline stage 1)
        gen_w, gen_h = width // 2, height // 2

        logger.info(
            "Long video: %d frames (%d chunks), tile=%d, overlap=%d, "
            "gen_res=%dx%d → output_res=%dx%d",
            actual_total_frames, num_chunks, temporal_tile_size,
            temporal_overlap, gen_w, gen_h, width, height,
        )

        t_start = time.perf_counter()
        sigmas = torch.tensor(
            DISTILLED_SIGMA_VALUES, dtype=torch.float32, device=pipeline.device,
        )

        # Parse per-step AdaIN factors
        psa_factors: list[float] = []
        if per_step_adain:
            psa_factors = [float(x.strip()) for x in per_step_adain_factors.split(",") if x.strip()]

        # Build ledger and encode text once (or per-chunk if per_chunk_prompts provided)
        ledger = pipeline._build_ledger()

        v_ctx, a_ctx, _, _ = pipeline._encode_text_stagehand(
            ledger, prompt, None, False,
            lambda msg, frac: None,
        )

        # Pre-encode per-chunk prompts if provided
        chunk_contexts: dict[int, tuple] = {}
        if per_chunk_prompts:
            for ci, cp in enumerate(per_chunk_prompts):
                if cp and cp != prompt:
                    vc, ac, _, _ = pipeline._encode_text_stagehand(
                        ledger, cp, None, False, lambda msg, frac: None,
                    )
                    chunk_contexts[ci] = (vc, ac)

        # Load transformer (stays resident across all chunks)
        transformer, xfm_inner, xfm_runtime = pipeline._setup_stagehand_transformer(ledger)

        try:
            # -- Chunk 0: Base generation at half-res --
            from ltx_core.components.diffusion_steps import EulerDiffusionStep
            from ltx_core.components.noisers import GaussianNoiser
            from ltx_core.types import VideoPixelShape
            from ltx_pipelines.utils.types import PipelineComponents

            generator = torch.Generator(device=pipeline.device).manual_seed(seed)
            noiser = GaussianNoiser(generator=generator)
            stepper = EulerDiffusionStep()
            components = PipelineComponents(dtype=pipeline.dtype, device=pipeline.device)

            # Ensure tile size satisfies 8n+1 constraint
            from long_video_presets import nearest_valid_frames as _nvf
            valid_tile = _nvf(temporal_tile_size)

            chunk0_shape = VideoPixelShape(
                batch=1, frames=valid_tile, width=gen_w, height=gen_h, fps=fps,
            )

            video_state, audio_state = pipeline._stagehand_denoise(
                transformer, xfm_runtime, v_ctx, a_ctx, chunk0_shape, sigmas,
                noiser, stepper, components, [],
                lambda msg, frac: None,
                label="Chunk 0", progress_base=0.0, progress_range=0.3,
            )

            accumulated = {"samples": video_state.latent}
            adain_reference = {"samples": accumulated["samples"].clone()}
            chunk0_latent = {"samples": accumulated["samples"].clone()}
            del video_state, audio_state

            if progress_callback:
                elapsed = time.perf_counter() - t_start
                progress_callback(0, num_chunks, temporal_tile_size, actual_total_frames, elapsed)

            # -- Chunks 1..N: Extend (all at half-res) --
            for chunk_idx in range(1, num_chunks):
                # Per-chunk prompt override
                vc, ac = chunk_contexts.get(chunk_idx, (v_ctx, a_ctx))

                accumulated = extend_chunk(
                    prev_latent=accumulated,
                    pipeline=pipeline,
                    transformer=transformer,
                    xfm_runtime=xfm_runtime,
                    v_ctx=vc,
                    a_ctx=ac,
                    sigmas=sigmas,
                    overlap_frames=temporal_overlap,
                    num_new_frames=temporal_tile_size - temporal_overlap,
                    adain_factor=adain_factor,
                    adain_reference=adain_reference,
                    seed=seed,
                    chunk_idx=chunk_idx,
                    pixel_width=gen_w,
                    pixel_height=gen_h,
                )

                # Per-step AdaIN: apply chunk-level normalization against reference
                if psa_factors:
                    # Use the last factor for between-chunk normalization
                    psa_chunk_factor = psa_factors[0] if psa_factors else 0.0
                    if psa_chunk_factor > 0.0:
                        accumulated = adain_normalize(
                            accumulated, adain_reference, factor=psa_chunk_factor,
                        )

                # Long memory: anchor to chunk 0 statistics
                if long_memory_strength > 0.0:
                    accumulated = add_long_memory_conditioning(
                        accumulated, chunk0_latent, strength=long_memory_strength,
                    )

                if progress_callback:
                    frames_so_far = temporal_tile_size + chunk_idx * (temporal_tile_size - temporal_overlap)
                    elapsed = time.perf_counter() - t_start
                    progress_callback(
                        chunk_idx, num_chunks, frames_so_far,
                        actual_total_frames, elapsed,
                    )

        finally:
            xfm_runtime.shutdown()
            del xfm_runtime, transformer, xfm_inner
            gc.collect()
            torch.cuda.empty_cache()

        # Move accumulated to CPU before any upscaling to free GPU for VAE/upsampler
        accumulated["samples"] = accumulated["samples"].cpu()

        # Spatial upscale from half-res to full-res
        if upscale_mode != UpscaleMode.NONE:
            logger.info("Upscaling accumulated latent from %dx%d to %dx%d...",
                        gen_w, gen_h, width, height)
            ve = ledger.video_encoder()
            su = ledger.spatial_upsampler()
            accumulated = spatially_upscale_long_latent(accumulated, su, ve)
            del ve, su
            gc.collect()
            torch.cuda.empty_cache()

        del ledger
        gc.collect()
        torch.cuda.empty_cache()
        return accumulated
