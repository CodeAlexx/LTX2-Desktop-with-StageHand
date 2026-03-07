"""LTX-2 pipeline — distilled & dev modes with Stagehand block-swap.

Uses ltx_core/ltx_pipelines directly. No diffusers.
ModelLedger for component loading, Stagehand for streaming transformer/TE blocks
through 24GB VRAM.

Features:
  - Two-stage distilled denoise (8+3 steps)
  - Dev mode with CFG/STG guidance + negative prompt
  - Prompt enhancement via Gemma 3 generate()
  - I2V image conditioning (first frame or keyframes)
  - LoRA support via ModelLedger
  - Audio generation + vocoder decode
  - Spatial upscaler between stages
"""

from __future__ import annotations

import gc
import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn

from ltx_core.components.diffusion_steps import EulerDiffusionStep
from ltx_core.components.guiders import (
    MultiModalGuiderParams,
    create_multimodal_guider_factory,
)
from ltx_core.components.noisers import GaussianNoiser
from ltx_core.components.schedulers import LTX2Scheduler
from ltx_core.model.audio_vae import decode_audio as vae_decode_audio
from ltx_core.model.upsampler import upsample_video
from ltx_core.model.video_vae import TilingConfig, get_video_chunks_number
from ltx_core.model.video_vae import decode_video as vae_decode_video
from ltx_core.types import VideoPixelShape
from ltx_pipelines.utils import (
    ModelLedger,
    denoise_audio_video,
    euler_denoising_loop,
    multi_modal_guider_factory_denoising_func,
    simple_denoising_func,
)
from ltx_pipelines.utils.args import ImageConditioningInput
from ltx_pipelines.utils.constants import (
    DISTILLED_SIGMA_VALUES,
    STAGE_2_DISTILLED_SIGMA_VALUES,
)
from ltx_pipelines.utils.helpers import combined_image_conditionings
from ltx_pipelines.utils.media_io import encode_video
from ltx_pipelines.utils.types import PipelineComponents

from stagehand import StagehandConfig, StagehandRuntime

from config import AppConfig

if TYPE_CHECKING:
    from collections.abc import Callable

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _flush() -> None:
    gc.collect()
    torch.cuda.empty_cache()


def _log_vram(msg: str) -> None:
    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        used = (total - free) / 1e9
        logger.info("[VRAM %.1f/%.1fGB] %s", used, total / 1e9, msg)
    else:
        logger.info(msg)


def _get_gemma_block_module(text_encoder: Any) -> nn.Module:
    """Navigate Gemma3ForConditionalGeneration to find the blockable layers."""
    model = getattr(text_encoder, "model", text_encoder)

    # Path 1: model.language_model.model (has .layers)
    lm = getattr(model, "language_model", None)
    if lm is not None:
        inner = getattr(lm, "model", None)
        if inner is not None and hasattr(inner, "layers"):
            return inner

    # Path 2: model.model.language_model (newer transformers)
    model_attr = getattr(model, "model", None)
    if model_attr is not None:
        lm2 = getattr(model_attr, "language_model", None)
        if lm2 is not None and hasattr(lm2, "layers"):
            return lm2

    # Path 3: model.model (direct causal LM)
    if model_attr is not None and hasattr(model_attr, "layers"):
        return model_attr

    raise AttributeError(
        f"Cannot find blockable layers on {type(text_encoder).__name__}"
    )


def _unwrap_to_blocks(model: Any) -> nn.Module:
    """Navigate X0Model -> velocity_model (LTXModel with transformer_blocks)."""
    for attr in ("velocity_model", "model", "module"):
        inner = getattr(model, attr, None)
        if inner is not None:
            if hasattr(inner, "transformer_blocks"):
                return inner
            for attr2 in ("velocity_model", "model", "module"):
                inner2 = getattr(inner, attr2, None)
                if inner2 is not None and hasattr(inner2, "transformer_blocks"):
                    return inner2
    if hasattr(model, "transformer_blocks"):
        return model
    raise AttributeError(f"Cannot find transformer_blocks on {type(model).__name__}")


def _move_non_blocks_to_device(root_module: nn.Module, block_container: nn.Module, device: torch.device) -> None:
    """Move ALL params/buffers in root_module to device, EXCEPT those inside block_container.layers.

    This handles the full model tree — embeddings, norms, lm_head, projectors, etc.
    Only the actual transformer/decoder blocks stay on CPU for Stagehand.
    """
    layers = getattr(block_container, "layers", None) or getattr(block_container, "transformer_blocks", None)
    if layers is None:
        raise AttributeError("block_container has no .layers or .transformer_blocks")

    block_param_ids = set(id(p) for p in layers.parameters())
    block_buf_ids = set(id(b) for b in layers.buffers())

    with torch.no_grad():
        for p in root_module.parameters():
            if id(p) not in block_param_ids and p.device != device:
                p.data = p.data.to(device, non_blocking=True)
        for name, buf in root_module.named_buffers():
            if id(buf) not in block_buf_ids and buf.device != device:
                # Walk the module tree to set buffer properly
                parts = name.rsplit(".", 1)
                if len(parts) == 2:
                    parent = root_module.get_submodule(parts[0])
                    parent._buffers[parts[1]] = buf.to(device, non_blocking=True)
                else:
                    root_module._buffers[name] = buf.to(device, non_blocking=True)


# ---------------------------------------------------------------------------
# Stagehand runtime builders
# ---------------------------------------------------------------------------

def _stagehand_config_te() -> StagehandConfig:
    """Stagehand config for Gemma 3 12B text encoder."""
    return StagehandConfig(
        pinned_pool_mb=6144,
        pinned_slab_mb=512,
        vram_high_watermark_mb=18000,
        vram_low_watermark_mb=14000,
        prefetch_window_blocks=2,
        max_inflight_transfers=2,
        telemetry_enabled=False,
    )


def _stagehand_config_xfm() -> StagehandConfig:
    """Stagehand config for 22B transformer (48 blocks, ~800MB each in bf16)."""
    return StagehandConfig(
        pinned_pool_mb=6400,
        pinned_slab_mb=800,
        vram_high_watermark_mb=18000,
        vram_low_watermark_mb=14000,
        prefetch_window_blocks=1,
        max_inflight_transfers=1,
        telemetry_enabled=False,
    )


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class LTX2Pipeline:
    """Wraps LTX-2.3 two-stage pipeline with Stagehand block-swap.

    Supports both distilled mode (no CFG, 8+3 steps) and dev mode
    (CFG/STG guidance, N+3 steps, distilled LoRA for stage 2).
    """

    def __init__(self, config: AppConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.bfloat16

    def _build_ledger(self, *, with_distilled_lora: bool = False) -> ModelLedger:
        """Build ModelLedger with LoRA support."""
        loras = []
        if self.config.lora_paths:
            from ltx_core.loader import LoraPathStrengthAndSDOps
            for path, strength in zip(self.config.lora_paths, self.config.lora_strengths):
                if path:
                    loras.append(LoraPathStrengthAndSDOps(path=path, strength=strength, sd_ops=None))

        # For dev mode stage 2, add the distilled LoRA
        if with_distilled_lora and self.config.distilled_lora_path:
            from ltx_core.loader import LoraPathStrengthAndSDOps
            loras.append(LoraPathStrengthAndSDOps(
                path=self.config.distilled_lora_path, strength=1.0, sd_ops=None,
            ))

        return ModelLedger(
            dtype=self.dtype,
            device=self.device,
            checkpoint_path=self.config.checkpoint_path,
            gemma_root_path=self.config.gemma_root,
            spatial_upsampler_path=self.config.spatial_upsampler_path,
            loras=tuple(loras) if loras else (),
        )

    def _encode_text_stagehand(
        self,
        ledger: ModelLedger,
        prompt: str,
        negative_prompt: str | None,
        enhance_prompt: bool,
        report: Callable[[str, float], None],
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
        """Encode prompt (and optional negative) through Gemma 3 12B with Stagehand.

        Returns (v_ctx_pos, a_ctx_pos, v_ctx_neg, a_ctx_neg).
        Negative contexts are None if negative_prompt is None/empty.
        """
        report("Loading text encoder...", 0.02)
        t0 = time.perf_counter()

        # Load text encoder to CPU (too large for 24GB GPU)
        ledger.device = torch.device("cpu")
        text_encoder = ledger.text_encoder()
        ledger.device = self.device

        # Set up Stagehand on Gemma decoder layers
        block_module = _get_gemma_block_module(text_encoder)
        _move_non_blocks_to_device(text_encoder, block_module, self.device)

        te_runtime = StagehandRuntime(
            model=block_module,
            config=_stagehand_config_te(),
            block_pattern=r"^layers\.\d+$",
            group="text_encoder",
            dtype=self.dtype,
            inference_mode=True,
        )
        _log_vram(f"Stagehand TE ready ({len(te_runtime._registry)} blocks)")

        # --- Prompt enhancement ---
        if enhance_prompt:
            report("Enhancing prompt...", 0.03)
            te_runtime.begin_step(0)
            with te_runtime.managed_forward():
                enhanced = text_encoder.enhance_t2v(prompt)
            te_runtime.end_step()
            logger.info("Enhanced prompt: %s", enhanced[:200])
            prompt = enhanced
            _log_vram(f"Prompt enhanced ({time.perf_counter() - t0:.1f}s)")

        # --- Encode prompts ---
        prompts_to_encode = [prompt]
        if negative_prompt:
            prompts_to_encode.append(negative_prompt)

        all_raw_hs = []
        all_raw_masks = []

        inner_model = text_encoder.model.model
        step_idx = 1 if enhance_prompt else 0

        for i, p in enumerate(prompts_to_encode):
            label = "positive" if i == 0 else "negative"
            report(f"Encoding {label} prompt...", 0.04 + i * 0.01)

            token_pairs = text_encoder.tokenizer.tokenize_with_weights(p)["gemma"]
            input_ids = torch.tensor([[t[0] for t in token_pairs]], device=self.device)
            attention_mask = torch.tensor([[w[1] for w in token_pairs]], device=self.device)

            te_runtime.begin_step(step_idx + i)
            with te_runtime.managed_forward():
                outputs = inner_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                )
            all_raw_hs.append(tuple(h.cpu() for h in outputs.hidden_states))
            all_raw_masks.append(attention_mask.cpu())
            del outputs
            te_runtime.end_step()

        # Cleanup text encoder
        te_runtime.shutdown()
        del te_runtime, text_encoder, block_module, inner_model
        _flush()
        _log_vram(f"Text encoding done ({time.perf_counter() - t0:.1f}s)")

        # Process embeddings through feature extractor + connectors
        report("Processing embeddings...", 0.08)
        emb_proc = ledger.gemma_embeddings_processor()
        emb_proc.to(self.device)

        results = []
        for raw_hs, raw_mask in zip(all_raw_hs, all_raw_masks):
            gpu_hs = tuple(h.to(self.device) for h in raw_hs)
            gpu_mask = raw_mask.to(self.device)
            ctx = emb_proc.process_hidden_states(gpu_hs, gpu_mask)
            v = ctx.video_encoding.detach().clone()
            a = ctx.audio_encoding.detach().clone() if ctx.audio_encoding is not None else None
            results.append((v, a))
            del ctx, gpu_hs, gpu_mask

        del emb_proc, all_raw_hs, all_raw_masks
        _flush()

        v_ctx_pos, a_ctx_pos = results[0]
        v_ctx_neg, a_ctx_neg = results[1] if len(results) > 1 else (None, None)

        _log_vram(f"Embeddings: video={tuple(v_ctx_pos.shape)}")
        return v_ctx_pos, a_ctx_pos, v_ctx_neg, a_ctx_neg

    def _setup_stagehand_transformer(
        self, ledger: ModelLedger,
    ) -> tuple[Any, nn.Module, StagehandRuntime]:
        """Load transformer to CPU, set up Stagehand. Returns (transformer, xfm_inner, runtime)."""
        ledger.device = torch.device("cpu")
        transformer = ledger.transformer()
        ledger.device = self.device

        xfm_inner = _unwrap_to_blocks(transformer)
        _move_non_blocks_to_device(transformer, xfm_inner, self.device)

        transformer.requires_grad_(False)

        xfm_runtime = StagehandRuntime(
            model=xfm_inner,
            config=_stagehand_config_xfm(),
            block_pattern=r"^transformer_blocks\.\d+$",
            group="transformer",
            dtype=self.dtype,
            inference_mode=True,
        )
        _log_vram(f"Stagehand transformer ready ({len(xfm_runtime._registry)} blocks)")
        return transformer, xfm_inner, xfm_runtime

    def _stagehand_denoise(
        self,
        transformer: Any,
        xfm_runtime: StagehandRuntime,
        v_ctx: torch.Tensor,
        a_ctx: torch.Tensor | None,
        shape: VideoPixelShape,
        sigmas: torch.Tensor,
        noiser: GaussianNoiser,
        stepper: EulerDiffusionStep,
        pipeline_components: PipelineComponents,
        conditionings: list,
        report: Callable[[str, float], None],
        label: str = "",
        progress_base: float = 0.0,
        progress_range: float = 0.3,
        noise_scale: float | None = None,
        initial_video_latent: torch.Tensor | None = None,
        initial_audio_latent: torch.Tensor | None = None,
        # Dev mode guidance (None = distilled/simple mode)
        v_ctx_neg: torch.Tensor | None = None,
        a_ctx_neg: torch.Tensor | None = None,
        video_guider_params: MultiModalGuiderParams | None = None,
        audio_guider_params: MultiModalGuiderParams | None = None,
    ) -> tuple[Any, Any]:
        """Run denoising with Stagehand-managed transformer.

        If video_guider_params is provided, uses CFG/STG guidance (dev mode).
        Otherwise uses simple_denoising_func (distilled mode).
        """
        use_guidance = video_guider_params is not None

        if use_guidance:
            video_guider_factory = create_multimodal_guider_factory(
                params=video_guider_params,
                negative_context=v_ctx_neg,
            )
            audio_guider_factory = create_multimodal_guider_factory(
                params=audio_guider_params,
                negative_context=a_ctx_neg,
            ) if audio_guider_params is not None else None

            base_fn = multi_modal_guider_factory_denoising_func(
                video_guider_factory=video_guider_factory,
                audio_guider_factory=audio_guider_factory,
                v_context=v_ctx,
                a_context=a_ctx,
                transformer=transformer,
            )
        else:
            base_fn = simple_denoising_func(
                video_context=v_ctx,
                audio_context=a_ctx,
                transformer=transformer,
            )

        _step = [0]
        n_steps = len(sigmas) - 1

        def wrapped_fn(video_state, audio_state, sigmas_arg, step_index):
            xfm_runtime.begin_step(_step[0])
            with xfm_runtime.managed_forward():
                result = base_fn(video_state, audio_state, sigmas_arg, step_index)
            xfm_runtime.end_step()
            _step[0] += 1
            frac = progress_base + progress_range * (_step[0] / n_steps)
            report(f"{label} step {_step[0]}/{n_steps}", frac)
            return result

        def loop_fn(sigmas_arg, video_state, audio_state, stepper_arg):
            return euler_denoising_loop(
                sigmas=sigmas_arg,
                video_state=video_state,
                audio_state=audio_state,
                stepper=stepper_arg,
                denoise_fn=wrapped_fn,
            )

        kwargs = {}
        if noise_scale is not None:
            kwargs["noise_scale"] = noise_scale
        if initial_video_latent is not None:
            kwargs["initial_video_latent"] = initial_video_latent
        if initial_audio_latent is not None:
            kwargs["initial_audio_latent"] = initial_audio_latent

        return denoise_audio_video(
            output_shape=shape,
            conditionings=conditionings,
            noiser=noiser,
            sigmas=sigmas,
            stepper=stepper,
            denoising_loop_fn=loop_fn,
            components=pipeline_components,
            dtype=self.dtype,
            device=self.device,
            **kwargs,
        )

    @torch.inference_mode()
    def generate(
        self,
        prompt: str,
        seed: int | None = None,
        width: int | None = None,
        height: int | None = None,
        num_frames: int | None = None,
        fps: float | None = None,
        enhance_prompt: bool | None = None,
        image_path: str | None = None,
        image_strength: float | None = None,
        image_crf: int = 35,
        negative_prompt: str | None = None,
        num_inference_steps: int | None = None,
        video_cfg_scale: float | None = None,
        video_stg_scale: float | None = None,
        video_rescale: float | None = None,
        audio_cfg_scale: float | None = None,
        audio_stg_scale: float | None = None,
        audio_rescale: float | None = None,
        a2v_scale: float | None = None,
        v2a_scale: float | None = None,
        stg_blocks: list[int] | None = None,
        progress_cb: Callable[[str, float], None] | None = None,
        **_kwargs: Any,
    ) -> Path:
        """Run full two-stage generation. Returns path to output video.

        Supports both distilled mode (simple denoising, fixed sigmas) and
        dev mode (CFG/STG guidance, LTX2Scheduler, distilled LoRA for stage 2).
        """
        cfg = self.config
        is_distilled = cfg.is_distilled
        s = seed if seed is not None else cfg.seed
        w = width or cfg.width
        h = height or cfg.height
        nf = num_frames or cfg.num_frames
        fr = fps or cfg.fps
        do_enhance = enhance_prompt if enhance_prompt is not None else cfg.enhance_prompt
        img_path = image_path or cfg.image_path
        img_strength = image_strength if image_strength is not None else cfg.image_strength
        neg_prompt = negative_prompt if negative_prompt is not None else (cfg.negative_prompt if not is_distilled else None)
        n_steps = num_inference_steps or cfg.num_inference_steps

        # Guidance params (dev mode only)
        v_cfg = video_cfg_scale if video_cfg_scale is not None else cfg.video_cfg_scale
        v_stg = video_stg_scale if video_stg_scale is not None else cfg.video_stg_scale
        v_rsc = video_rescale if video_rescale is not None else cfg.video_rescale
        a_cfg = audio_cfg_scale if audio_cfg_scale is not None else cfg.audio_cfg_scale
        a_stg = audio_stg_scale if audio_stg_scale is not None else cfg.audio_stg_scale
        a_rsc = audio_rescale if audio_rescale is not None else cfg.audio_rescale
        a2v = a2v_scale if a2v_scale is not None else cfg.a2v_scale
        v2a = v2a_scale if v2a_scale is not None else cfg.v2a_scale
        stg_blks = stg_blocks if stg_blocks is not None else cfg.stg_blocks

        t_total = time.perf_counter()

        def report(phase: str, frac: float) -> None:
            if progress_cb:
                progress_cb(phase, frac)

        report("Initializing...", 0.0)
        mode_str = "distilled" if is_distilled else f"dev({n_steps}steps, cfg={v_cfg})"
        _log_vram(f"Starting generation: {w}x{h}, {nf}f, seed={s}, mode={mode_str}, enhance={do_enhance}, i2v={bool(img_path)}")

        # Build ModelLedger (loads nothing until .transformer() etc. called)
        ledger = self._build_ledger()

        # For dev mode stage 2, we need a separate ledger with distilled LoRA
        ledger_s2 = self._build_ledger(with_distilled_lora=True) if not is_distilled else ledger

        # Build image conditionings for I2V
        images: list[ImageConditioningInput] = []
        if img_path:
            images.append(ImageConditioningInput(
                path=img_path,
                frame_idx=0,
                strength=img_strength,
                crf=image_crf,
            ))

        # =====================================================================
        # Text encoding (Gemma 3 12B via Stagehand)
        # =====================================================================
        v_ctx, a_ctx, v_ctx_neg, a_ctx_neg = self._encode_text_stagehand(
            ledger, prompt, neg_prompt if not is_distilled else None, do_enhance, report,
        )

        # Common components
        generator = torch.Generator(device=self.device).manual_seed(s)
        noiser = GaussianNoiser(generator=generator)
        stepper = EulerDiffusionStep()
        pipeline_components = PipelineComponents(dtype=self.dtype, device=self.device)

        # Build guidance params for dev mode
        video_guider_params = None
        audio_guider_params = None
        if not is_distilled:
            video_guider_params = MultiModalGuiderParams(
                cfg_scale=v_cfg, stg_scale=v_stg, rescale_scale=v_rsc,
                modality_scale=a2v, skip_step=0, stg_blocks=stg_blks,
            )
            audio_guider_params = MultiModalGuiderParams(
                cfg_scale=a_cfg, stg_scale=a_stg, rescale_scale=a_rsc,
                modality_scale=v2a, skip_step=0, stg_blocks=stg_blks,
            )

        # =====================================================================
        # Stage 1: Half-res denoise
        # =====================================================================
        report("Stage 1: Loading transformer...", 0.10)
        t0 = time.perf_counter()

        s1_w, s1_h = w // 2, h // 2
        s1_shape = VideoPixelShape(batch=1, frames=nf, width=s1_w, height=s1_h, fps=fr)

        if is_distilled:
            s1_sigmas = torch.tensor(DISTILLED_SIGMA_VALUES, dtype=torch.float32, device=self.device)
        else:
            s1_sigmas = LTX2Scheduler().execute(steps=n_steps).to(dtype=torch.float32, device=self.device)

        # Load video encoder for image conditioning (needed before transformer)
        s1_conditionings = []
        video_encoder_for_cond = None
        if images:
            report("Stage 1: Encoding conditioning image...", 0.11)
            video_encoder_for_cond = ledger.video_encoder()
            s1_conditionings = combined_image_conditionings(
                images=images,
                height=s1_h,
                width=s1_w,
                video_encoder=video_encoder_for_cond,
                dtype=self.dtype,
                device=self.device,
            )
            del video_encoder_for_cond
            _flush()

        transformer, xfm_inner, xfm_runtime = self._setup_stagehand_transformer(ledger)

        video_state, audio_state = self._stagehand_denoise(
            transformer, xfm_runtime, v_ctx, a_ctx, s1_shape, s1_sigmas,
            noiser, stepper, pipeline_components, s1_conditionings, report,
            label="Stage 1", progress_base=0.12, progress_range=0.30,
            v_ctx_neg=v_ctx_neg, a_ctx_neg=a_ctx_neg,
            video_guider_params=video_guider_params,
            audio_guider_params=audio_guider_params,
        )
        _log_vram(f"Stage 1 done ({time.perf_counter() - t0:.1f}s)")

        # Save latents, free transformer for upsampler
        s1_video_latent = video_state.latent.cpu()
        s1_audio_latent = audio_state.latent.cpu()
        xfm_runtime.shutdown()
        del xfm_runtime, transformer, xfm_inner
        _flush()
        _log_vram("Transformer freed")

        # =====================================================================
        # Stage 2: Spatial upsample + refine (3 steps, always distilled/simple)
        # =====================================================================
        report("Stage 2: Upsampling...", 0.45)
        t0 = time.perf_counter()

        video_encoder = ledger.video_encoder()
        upsampler = ledger.spatial_upsampler()
        upscaled = upsample_video(
            latent=s1_video_latent.to(self.device)[:1],
            video_encoder=video_encoder,
            upsampler=upsampler,
        )

        # Build stage 2 conditionings (at full resolution)
        s2_conditionings = []
        if images:
            s2_conditionings = combined_image_conditionings(
                images=images,
                height=h,
                width=w,
                video_encoder=video_encoder,
                dtype=self.dtype,
                device=self.device,
            )

        del video_encoder, upsampler, s1_video_latent
        _flush()
        _log_vram(f"Upsampled latent: {tuple(upscaled.shape)}")

        # Reload transformer for stage 2 (with distilled LoRA in dev mode)
        report("Stage 2: Reloading transformer...", 0.50)
        transformer2, xfm_inner2, xfm_runtime2 = self._setup_stagehand_transformer(ledger_s2)

        s2_shape = VideoPixelShape(batch=1, frames=nf, width=w, height=h, fps=fr)
        s2_sigmas = torch.tensor(STAGE_2_DISTILLED_SIGMA_VALUES, dtype=torch.float32, device=self.device)

        # Stage 2 always uses simple denoising (distilled LoRA handles quality)
        video_state, audio_state = self._stagehand_denoise(
            transformer2, xfm_runtime2, v_ctx, a_ctx, s2_shape, s2_sigmas,
            noiser, stepper, pipeline_components, s2_conditionings, report,
            label="Stage 2", progress_base=0.55, progress_range=0.15,
            noise_scale=s2_sigmas[0].item(),
            initial_video_latent=upscaled,
            initial_audio_latent=s1_audio_latent.to(self.device),
        )
        _log_vram(f"Stage 2 done ({time.perf_counter() - t0:.1f}s)")

        # Free transformer before VAE decode
        xfm_runtime2.shutdown()
        del xfm_runtime2, transformer2, xfm_inner2, upscaled, s1_audio_latent
        del v_ctx, a_ctx, v_ctx_neg, a_ctx_neg
        _flush()
        _log_vram("Transformer freed, starting decode")

        # =====================================================================
        # VAE decode + save
        # =====================================================================
        report("Decoding video...", 0.75)
        t0 = time.perf_counter()

        tiling = TilingConfig.default()
        decoded_video = vae_decode_video(
            video_state.latent,
            ledger.video_decoder(),
            tiling,
            generator,
        )
        decoded_audio = vae_decode_audio(
            audio_state.latent,
            ledger.audio_decoder(),
            ledger.vocoder(),
        )
        _log_vram(f"VAE decode done ({time.perf_counter() - t0:.1f}s)")

        # Save output
        report("Saving...", 0.95)
        out_dir = cfg.ensure_output_dir()
        import datetime
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        mode_tag = "distilled" if is_distilled else "dev"
        out_path = out_dir / f"ltx2_{mode_tag}_{ts}_s{s}.mp4"

        video_chunks_number = get_video_chunks_number(nf, tiling)
        encode_video(
            video=decoded_video,
            fps=fr,
            audio=decoded_audio,
            output_path=str(out_path),
            video_chunks_number=video_chunks_number,
        )

        _log_vram(f"TOTAL: {time.perf_counter() - t_total:.1f}s")
        report("Done!", 1.0)
        return out_path
