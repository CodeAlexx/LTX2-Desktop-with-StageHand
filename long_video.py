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
    """Apply spatial upscaler to a long latent in temporal chunks.

    Runs on CUDA if available, falling back to CPU for Conv3d compatibility
    (PyTorch slow_conv3d_forward has no CUDA kernel in some builds).
    """
    from ltx_core.model.upsampler import upsample_video

    samples = latent["samples"]
    total_frames = samples.shape[2]

    # Try CUDA first, fall back to CPU if slow_conv3d not available
    device = next(spatial_upscaler.parameters()).device
    try:
        test_chunk = samples[:, :, :1].to(device)
        _ = upsample_video(test_chunk, video_encoder, spatial_upscaler)
        del test_chunk, _
        run_device = device
        logger.info("Spatial upscaler running on %s", run_device)
    except NotImplementedError:
        logger.warning("Conv3d not available on CUDA, falling back to CPU for spatial upscale")
        spatial_upscaler = spatial_upscaler.cpu().float()
        video_encoder = video_encoder.cpu().float()
        run_device = torch.device("cpu")

    chunks = []
    pos = 0
    while pos < total_frames:
        end = min(pos + temporal_chunk_size, total_frames)
        chunk = samples[:, :, pos:end].to(run_device)
        if run_device.type == "cpu":
            chunk = chunk.float()
        upscaled_chunk = upsample_video(chunk, video_encoder, spatial_upscaler)
        chunks.append(upscaled_chunk.cpu())
        logger.info("Upscaled frames %d-%d / %d", pos, end, total_frames)
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
    prev_audio: torch.Tensor | None = None,
) -> tuple[dict, torch.Tensor | None]:
    """Extend a video by one temporal chunk using official conditioning API.

    Uses VideoConditionByLatentIndex to inject overlap frames at position 0,
    which sets denoise_mask=0 so the noiser preserves them and the model sees
    them as temporal context during attention. After generation, overlap frames
    are discarded (exact copies) and only new frames are concatenated.
    """
    from ltx_core.components.diffusion_steps import EulerDiffusionStep
    from ltx_core.components.noisers import GaussianNoiser
    from ltx_core.conditioning import VideoConditionByLatentIndex
    from ltx_core.types import AudioLatentShape, VideoPixelShape
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

    pw = pixel_width if pixel_width is not None else W * 32
    ph = pixel_height if pixel_height is not None else H * 32

    chunk_shape = VideoPixelShape(
        batch=1,
        frames=valid_pixel_frames,
        width=pw,
        height=ph,
        fps=pipeline.config.fps,
    )

    # Extract overlap tail from accumulated video latent
    overlap_tail = samples[:, :, -latent_overlap:].to(pipeline.device)

    # Official conditioning: inject overlap at frame 0 with strength=1.0
    # This sets denoise_mask=0 for overlap tokens BEFORE noising, so the
    # noiser never touches them and post_process_latent preserves them.
    conditioning = VideoConditionByLatentIndex(
        latent=overlap_tail,
        strength=1.0,
        latent_idx=0,
    )

    # Prepare audio: pass overlap tail as initial_audio_latent (bias, not preserved)
    audio_initial = None
    audio_overlap_frames = 0
    if prev_audio is not None:
        audio_overlap_frames = overlap_frames
        audio_target_shape = AudioLatentShape.from_video_pixel_shape(chunk_shape)
        audio_target_frames = audio_target_shape.frames

        audio_tail = prev_audio[:, :, -audio_overlap_frames:]
        if audio_tail.shape[2] < audio_target_frames:
            audio_pad = torch.zeros(
                audio_tail.shape[0], audio_tail.shape[1],
                audio_target_frames - audio_tail.shape[2],
                audio_tail.shape[3],
                device=audio_tail.device, dtype=audio_tail.dtype,
            )
            audio_initial = torch.cat([audio_tail, audio_pad], dim=2)
        else:
            audio_initial = audio_tail[:, :, :audio_target_frames]

    generator = torch.Generator(device=pipeline.device).manual_seed(seed + chunk_idx)
    noiser = GaussianNoiser(generator=generator)
    stepper = EulerDiffusionStep()
    components = PipelineComponents(dtype=pipeline.dtype, device=pipeline.device)

    # Denoise via the same pipeline path as chunk 0, with overlap conditioning
    video_state, audio_state = pipeline._stagehand_denoise(
        transformer, xfm_runtime, v_ctx, a_ctx, chunk_shape, sigmas,
        noiser, stepper, components,
        [conditioning],
        report,
        label=f"Chunk {chunk_idx}",
        initial_audio_latent=audio_initial.to(pipeline.device) if audio_initial is not None else None,
    )

    new_chunk = {"samples": video_state.latent}
    new_audio = audio_state.latent if audio_state.latent is not None else None
    del video_state, audio_state

    # AdaIN normalization on new frames only (skip overlap)
    if adain_factor > 0.0 and adain_reference is not None:
        new_only = {"samples": new_chunk["samples"][:, :, latent_overlap:]}
        new_only = adain_normalize(new_only, adain_reference, factor=adain_factor)
        new_chunk["samples"] = torch.cat(
            [new_chunk["samples"][:, :, :latent_overlap], new_only["samples"]], dim=2,
        )

    # Discard overlap frames (preserved copies of accumulated tail)
    new_frames = {"samples": new_chunk["samples"][:, :, latent_overlap:]}
    result = {"samples": torch.cat([prev_latent["samples"], new_frames["samples"]], dim=2)}

    # Discard audio overlap frames
    if new_audio is not None and audio_overlap_frames > 0:
        if audio_overlap_frames < new_audio.shape[2]:
            new_audio = new_audio[:, :, audio_overlap_frames:]

    return result, new_audio


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

    def _stage2_refine(
        self,
        accumulated: dict,
        pipeline: LTX2Pipeline,
        ledger: Any,
        v_ctx: torch.Tensor,
        a_ctx: torch.Tensor | None,
        nag_enabled: bool,
        nag_v_ctx: torch.Tensor | None,
        nag_a_ctx: torch.Tensor | None,
        pixel_width: int,
        pixel_height: int,
        fps: float,
        temporal_chunk_latent: int = 6,
        temporal_overlap_latent: int = 2,
    ) -> dict:
        """Stage 2 refinement: 3 distilled steps on upscaled latent.

        Matches pipeline._generate_two_stage Stage 2. Processes long latents
        in overlapping temporal chunks with linear blending at boundaries
        to prevent discontinuities.

        Args:
            temporal_chunk_latent: latent frames per chunk (6 ≈ 41 pixel frames)
            temporal_overlap_latent: overlap in latent frames for blending (2 ≈ 17 pixel frames)
        """
        import gc
        from ltx_core.components.diffusion_steps import EulerDiffusionStep
        from ltx_core.components.noisers import GaussianNoiser
        from ltx_core.types import VideoPixelShape
        from ltx_pipelines.utils.constants import STAGE_2_DISTILLED_SIGMA_VALUES
        from ltx_pipelines.utils.types import PipelineComponents

        samples = accumulated["samples"]  # (B, C, T, H, W)
        total_latent_frames = samples.shape[2]

        s2_sigmas = torch.tensor(
            STAGE_2_DISTILLED_SIGMA_VALUES, dtype=torch.float32, device=pipeline.device,
        )

        stride = temporal_chunk_latent - temporal_overlap_latent
        logger.info(
            "Stage 2 refinement: %d latent frames, %d steps, chunk=%d, overlap=%d, stride=%d",
            total_latent_frames, len(s2_sigmas) - 1,
            temporal_chunk_latent, temporal_overlap_latent, stride,
        )

        # Load transformer for Stage 2
        transformer2, xfm_inner2, xfm_runtime2 = pipeline._setup_stagehand_transformer(ledger)

        nag_patch2 = None
        if nag_enabled and nag_v_ctx is not None:
            from nag import NAGPatch
            nag_patch2 = NAGPatch(
                nag_v_ctx, nag_a_ctx,
                pipeline.config.nag_scale, pipeline.config.nag_alpha, pipeline.config.nag_tau,
            )
            nag_patch2.apply(transformer2)
            logger.info("NAG applied for Stage 2 refinement")

        try:
            # Accumulator: sum of weighted refined latents + weight map
            B, C, T, H, W = samples.shape
            result_sum = torch.zeros_like(samples)
            weight_sum = torch.zeros(1, 1, T, 1, 1, dtype=samples.dtype, device=samples.device)

            pos = 0
            chunk_idx = 0
            while pos < total_latent_frames:
                end = min(pos + temporal_chunk_latent, total_latent_frames)
                chunk_latent = samples[:, :, pos:end].to(pipeline.device)

                n_latent = end - pos
                n_pixel = (n_latent - 1) * 8 + 1

                chunk_shape = VideoPixelShape(
                    batch=1, frames=n_pixel,
                    width=pixel_width * 2,
                    height=pixel_height * 2,
                    fps=fps,
                )

                generator = torch.Generator(device=pipeline.device).manual_seed(42)
                noiser = GaussianNoiser(generator=generator)
                stepper = EulerDiffusionStep()
                components = PipelineComponents(dtype=pipeline.dtype, device=pipeline.device)

                video_state, _ = pipeline._stagehand_denoise(
                    transformer2, xfm_runtime2, v_ctx, a_ctx, chunk_shape, s2_sigmas,
                    noiser, stepper, components, [],
                    lambda msg, frac: None,
                    label=f"Stage2 chunk {chunk_idx}",
                    noise_scale=s2_sigmas[0].item(),
                    initial_video_latent=chunk_latent,
                )

                refined = video_state.latent.cpu()
                del video_state, chunk_latent

                # Build per-frame weight: 1.0 in the middle, linear ramp at edges
                w = torch.ones(n_latent, dtype=samples.dtype)
                if pos > 0 and temporal_overlap_latent > 0:
                    # Ramp up at start (overlap with previous chunk)
                    ramp_len = min(temporal_overlap_latent, n_latent)
                    w[:ramp_len] = torch.linspace(0, 1, ramp_len + 2, dtype=samples.dtype)[1:-1]
                if end < total_latent_frames and temporal_overlap_latent > 0:
                    # Ramp down at end (overlap with next chunk)
                    ramp_len = min(temporal_overlap_latent, n_latent)
                    w[-ramp_len:] = torch.linspace(1, 0, ramp_len + 2, dtype=samples.dtype)[1:-1]
                w = w.reshape(1, 1, n_latent, 1, 1)

                result_sum[:, :, pos:end] += refined * w
                weight_sum[:, :, pos:end] += w

                logger.info("Stage 2 refined frames %d-%d / %d", pos, end, total_latent_frames)
                pos += stride
                chunk_idx += 1

        finally:
            if nag_patch2 is not None:
                nag_patch2.remove(transformer2)
            xfm_runtime2.shutdown()
            del xfm_runtime2, transformer2, xfm_inner2
            gc.collect()
            torch.cuda.empty_cache()

        # Normalize by weight (avoid div-by-zero)
        weight_sum = weight_sum.clamp(min=1e-8)
        return {"samples": result_sum / weight_sum}

    @torch.inference_mode()
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
    ) -> tuple[dict, torch.Tensor | None]:
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

        # Generate at full resolution — each chunk fits in 24GB with Stagehand
        gen_w, gen_h = width, height

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
        # NOTE: pipeline.generate() has @torch.inference_mode() but long video
        # is called separately — must enable it here to avoid autograd OOM.
        ledger = pipeline._build_ledger()
        cfg = pipeline.config
        nag_enabled = getattr(cfg, "nag_enabled", False)

        v_ctx, a_ctx, _, _, nag_v_ctx, nag_a_ctx = pipeline._encode_text_stagehand(
            ledger, prompt, None, False,
            lambda msg, frac: None,
            encode_nag=nag_enabled,
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

        # Apply NAG if enabled in pipeline config
        nag_patch = None
        if nag_enabled and nag_v_ctx is not None:
            from nag import NAGPatch
            nag_patch = NAGPatch(nag_v_ctx, nag_a_ctx, cfg.nag_scale, cfg.nag_alpha, cfg.nag_tau)
            nag_patch.apply(transformer)
            logger.info("NAG applied for long video (scale=%.1f)", cfg.nag_scale)

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
            accumulated_audio = audio_state.latent if audio_state.latent is not None else None
            del video_state, audio_state

            if progress_callback:
                elapsed = time.perf_counter() - t_start
                progress_callback(0, num_chunks, temporal_tile_size, actual_total_frames, elapsed)

            # -- Chunks 1..N: Extend (all at half-res) --
            for chunk_idx in range(1, num_chunks):
                # Per-chunk prompt override
                vc, ac = chunk_contexts.get(chunk_idx, (v_ctx, a_ctx))

                # Last chunk gets maximum overlap for more context
                is_last_chunk = (chunk_idx == num_chunks - 1)
                if is_last_chunk and num_chunks > 2:
                    # Double the overlap, capped so we still generate >=8 new frames
                    chunk_overlap = min(temporal_overlap * 2, temporal_tile_size - 8)
                    # Ensure it satisfies 8-frame alignment
                    chunk_overlap = (chunk_overlap // 8) * 8
                    chunk_new = temporal_tile_size - chunk_overlap
                    logger.info(
                        "Last chunk %d: boosted overlap %d→%d (new_frames=%d)",
                        chunk_idx, temporal_overlap, chunk_overlap, chunk_new,
                    )
                else:
                    chunk_overlap = temporal_overlap
                    chunk_new = temporal_tile_size - temporal_overlap

                accumulated, chunk_audio = extend_chunk(
                    prev_latent=accumulated,
                    pipeline=pipeline,
                    transformer=transformer,
                    xfm_runtime=xfm_runtime,
                    v_ctx=vc,
                    a_ctx=ac,
                    sigmas=sigmas,
                    overlap_frames=chunk_overlap,
                    num_new_frames=chunk_new,
                    adain_factor=adain_factor,
                    adain_reference=adain_reference,
                    seed=seed,
                    chunk_idx=chunk_idx,
                    pixel_width=gen_w,
                    pixel_height=gen_h,
                    prev_audio=accumulated_audio,
                )
                # Accumulate audio — overlap frames already discarded by extend_chunk
                accumulated_audio = extend_audio_latent(
                    accumulated_audio, chunk_audio, overlap=0,
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
            if nag_patch is not None:
                nag_patch.remove(transformer)
            xfm_runtime.shutdown()
            del xfm_runtime, transformer, xfm_inner
            gc.collect()
            torch.cuda.empty_cache()

        # Move accumulated to CPU before any upscaling to free GPU for VAE/upsampler
        accumulated["samples"] = accumulated["samples"].cpu()

        # Spatial upscale from half-res to full-res
        if upscale_mode != UpscaleMode.NONE:
            logger.info("Upscaling accumulated latent %s ...", tuple(accumulated["samples"].shape))
            ve = ledger.video_encoder()
            su = ledger.spatial_upsampler()
            accumulated = spatially_upscale_long_latent(accumulated, su, ve)
            logger.info("Upscaled result: %s", tuple(accumulated["samples"].shape))
            del ve, su
            gc.collect()
            torch.cuda.empty_cache()

            # =========================================================
            # Stage 2: Refine upscaled latent (3 steps, distilled)
            # Same as pipeline._generate_two_stage Stage 2 — adds detail
            # that the initial 8-step generation cannot produce.
            # =========================================================
            accumulated = self._stage2_refine(
                accumulated, pipeline, ledger, v_ctx, a_ctx,
                nag_enabled, nag_v_ctx, nag_a_ctx,
                width, height, fps,
            )

        # Move audio to CPU
        if accumulated_audio is not None:
            accumulated_audio = accumulated_audio.cpu()

        del ledger
        gc.collect()
        torch.cuda.empty_cache()
        return accumulated, accumulated_audio
