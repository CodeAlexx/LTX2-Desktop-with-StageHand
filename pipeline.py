"""LTX-2 pipeline — distilled & dev modes with Stagehand block-swap.

Uses ltx_core/ltx_pipelines directly. No diffusers.
ModelLedger for component loading, Stagehand for streaming transformer/TE blocks
through 24GB VRAM.

Features:
  - Two-stage distilled denoise (8+3 steps)
  - Four-pass pipeline (base → spatial upscale → temporal upscale → refinement)
  - Dev mode with CFG/STG guidance + negative prompt
  - Prompt enhancement via Gemma 3 generate()
  - I2V image conditioning (first frame or keyframes)
  - LoRA support via ModelLedger
  - Audio generation + vocoder decode
  - Spatial + temporal upscaler between stages
"""

from __future__ import annotations

import gc
import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn

from ltx_core.components.diffusion_steps import EulerDiffusionStep, Res2sDiffusionStep
from ltx_core.components.guiders import (
    MultiModalGuider,
    MultiModalGuiderParams,
    create_multimodal_guider_factory,
)
from ltx_core.components.noisers import GaussianNoiser
from ltx_core.components.schedulers import LTX2Scheduler
from ltx_core.model.video_vae import SpatialTilingConfig, TemporalTilingConfig
from ltx_core.model.audio_vae import decode_audio as vae_decode_audio
from ltx_core.model.upsampler import upsample_video
from ltx_core.model.video_vae import TilingConfig, get_video_chunks_number
from ltx_core.model.video_vae import decode_video as vae_decode_video
from ltx_core.types import VideoPixelShape
from ltx_pipelines.utils import (
    ModelLedger,
    denoise_audio_video,
    euler_denoising_loop,
    multi_modal_guider_denoising_func,
    multi_modal_guider_factory_denoising_func,
    res2s_audio_video_denoising_loop,
    simple_denoising_func,
)
from ltx_pipelines.utils.args import ImageConditioningInput
from ltx_pipelines.utils.constants import (
    DISTILLED_SIGMA_VALUES,
    LTX_2_3_HQ_PARAMS,
    STAGE_2_DISTILLED_SIGMA_VALUES,
)

# Four-pass manual sigmas (from community CC prompt)
FOUR_PASS_STAGE4_SIGMAS = torch.tensor(
    [1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0],
    dtype=torch.float32,
)
from ltx_core.model.audio_vae import encode_audio as vae_encode_audio
from ltx_core.types import Audio, AudioLatentShape
from ltx_pipelines.utils.helpers import combined_image_conditionings
from ltx_pipelines.utils.media_io import decode_audio_from_file, load_video_conditioning
from ltx_pipelines.utils.media_io import encode_video
from ltx_pipelines.utils.types import PipelineComponents

from ltx_core.quantization.policy import QuantizationPolicy
from stagehand import StagehandConfig, StagehandRuntime

from config import AppConfig
from audio_utils import normalize_output_audio
from hq_runtime_loader import attach_runtime_lora_merge, attach_runtime_scaled_fp8_dequantization
from nag import NAGPatch

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


def rescale_sigmas(sigmas: torch.Tensor, factor: float) -> torch.Tensor:
    """Scale non-zero sigmas by factor."""
    return torch.where(sigmas != 0, sigmas * factor, sigmas)


def _read_ic_lora_downscale(lora_paths: list[str]) -> int:
    """Read reference_downscale_factor from IC-LoRA safetensors metadata."""
    from safetensors import safe_open
    factor = 1
    for path in lora_paths:
        try:
            with safe_open(path, framework="pt") as f:
                meta = f.metadata() or {}
                scale = int(meta.get("reference_downscale_factor", 1))
                if scale != 1:
                    factor = scale
        except Exception:
            pass
    return factor


def _needs_spatial_tiling(h_pixels: int, w_pixels: int, tile_pixels: int) -> bool:
    """Return True if resolution exceeds a single tile."""
    return h_pixels > tile_pixels or w_pixels > tile_pixels


def validate_inputs(width: int, height: int, num_frames: int) -> None:
    """Validate generation inputs."""
    if width % 32 != 0:
        raise ValueError(f"width must be divisible by 32, got {width}")
    if height % 32 != 0:
        raise ValueError(f"height must be divisible by 32, got {height}")
    if (num_frames - 1) % 8 != 0:
        raise ValueError(
            f"num_frames must satisfy 8n+1, got {num_frames}. "
            f"Nearest valid: {nearest_valid_frames(num_frames)}"
        )


def nearest_valid_frames(n: int) -> int:
    """Round n to nearest valid LTX-2.3 frame count (8n+1)."""
    remainder = (n - 1) % 8
    if remainder <= 4:
        return n - remainder
    return n + (8 - remainder)


def bong_tangent_sigmas(
    steps: int,
    start: float = 1.0,
    middle: float = 0.5,
    end: float = 0.0,
    pivot_1: float = 0.6,
    pivot_2: float = 0.6,
    slope_1: float = 0.2,
    slope_2: float = 0.2,
) -> torch.Tensor:
    """Two-stage arctan sigma schedule from RES4LYF (ClownsharkBatwing).

    Splits the schedule at a midpoint into two arctan-sigmoid curves:
    Stage 1: start → middle, Stage 2: middle → end.
    Slopes are normalized by step count for consistent shape.
    """
    import math
    n = steps + 2  # pad for boundary handling

    midpoint = int((n * pivot_1 + n * pivot_2) / 2)
    piv1 = int(n * pivot_1)
    piv2 = int(n * pivot_2)
    s1 = slope_1 / (n / 40)
    s2 = slope_2 / (n / 40)

    stage_2_len = n - midpoint
    stage_1_len = n - stage_2_len

    def _atan_sigmas(count, slope, pivot, s_start, s_end):
        smax = ((2 / math.pi) * math.atan(-slope * (0 - pivot)) + 1) / 2
        smin = ((2 / math.pi) * math.atan(-slope * ((count - 1) - pivot)) + 1) / 2
        srange = smax - smin
        sscale = s_start - s_end
        return [((((2 / math.pi) * math.atan(-slope * (x - pivot)) + 1) / 2) - smin)
                * (1 / srange) * sscale + s_end for x in range(count)]

    part1 = _atan_sigmas(stage_1_len, s1, piv1, start, middle)[:-1]
    part2 = _atan_sigmas(stage_2_len, s2, piv2 - stage_1_len, middle, end)
    combined = part1 + part2 + [0.0]
    return torch.tensor(combined, dtype=torch.float32)


def inject_noise(latent: torch.Tensor, strength: float, generator: torch.Generator) -> torch.Tensor:
    """Inject Gaussian noise into a latent tensor at given strength."""
    noise = torch.randn(latent.shape, device=latent.device, dtype=latent.dtype, generator=generator)
    return latent + noise * strength


def apply_decoder_noise(
    latent: torch.Tensor, scale: float, shift: float, seed: int,
) -> torch.Tensor:
    """Add noise to latent before VAE decode (ComfyUI Set VAE Decoder Noise pattern)."""
    gen = torch.Generator(device=latent.device).manual_seed(seed)
    noise = torch.randn(latent.shape, device=latent.device, dtype=latent.dtype, generator=gen)
    return latent + noise * scale + shift


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

    target_dtype = torch.bfloat16
    with torch.no_grad():
        for p in root_module.parameters():
            if id(p) not in block_param_ids and (p.device != device or p.dtype != target_dtype):
                p.data = p.data.to(device, dtype=target_dtype, non_blocking=True)
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

def _stagehand_config_te(gemma_root: str = "") -> StagehandConfig:
    """Stagehand config for Gemma 3 12B text encoder.

    Auto-adjusts slab/pool sizes for FP4 quantized checkpoints (~64MB/layer
    vs ~128MB for BF16). Currently no FP4 checkpoints exist in ltx_core;
    this is forward-compatible.
    """
    is_fp4 = "fp4" in gemma_root.lower()
    return StagehandConfig(
        pinned_pool_mb=4096 if is_fp4 else 6144,
        pinned_slab_mb=256 if is_fp4 else 512,
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

    @staticmethod
    def _detect_lora_sd_ops(path: str):
        """Auto-detect LoRA key format and return appropriate SDOps.

        Lightricks/ComfyUI LoRAs use ``diffusion_model.`` prefix.
        Serenity LoRAs use ``transformer.`` prefix.
        Returns SDOps that normalises either format so ltx_core can match keys.
        """
        from ltx_core.loader import SDOps
        try:
            from safetensors import safe_open
            with safe_open(path, framework="pt", device="cpu") as f:
                first_key = next(iter(f.keys()), "")
            if first_key.startswith("transformer."):
                return SDOps("serenity_lora").with_matching().with_replacement("transformer.", "")
            if first_key.startswith("diffusion_model."):
                return SDOps("comfyui_lora").with_matching().with_replacement("diffusion_model.", "")
        except Exception:
            pass
        return None

    def _build_ledger(self, *, with_distilled_lora: bool = False,
                      distilled_lora_strength_override: float | None = None,
                      checkpoint_override: str | tuple[str, ...] | None = None,
                      skip_user_loras: bool = False,
                      dequantize_scaled_fp8_to_bf16: bool = False) -> ModelLedger:
        """Build ModelLedger with LoRA support."""
        ckpt_path = checkpoint_override or self.config.checkpoint_path
        ckpt_desc = " | ".join(ckpt_path) if isinstance(ckpt_path, tuple) else ckpt_path

        loras = []
        if self.config.lora_paths and not skip_user_loras:
            from ltx_core.loader import LoraPathStrengthAndSDOps
            for path, strength in zip(self.config.lora_paths, self.config.lora_strengths):
                if path:
                    sd_ops = self._detect_lora_sd_ops(path)
                    loras.append(LoraPathStrengthAndSDOps(path=path, strength=strength, sd_ops=sd_ops))

        # For dev mode stage 2, add the distilled LoRA directly via ltx-core apply_loras.
        # HQ mode uses a transformer load-time merge instead to avoid the expensive full-model pass.
        if with_distilled_lora and self.config.distilled_lora_path:
            from ltx_core.loader import LoraPathStrengthAndSDOps
            sd_ops = self._detect_lora_sd_ops(self.config.distilled_lora_path)
            strength = distilled_lora_strength_override if distilled_lora_strength_override is not None else self.config.distilled_lora_strength
            loras.append(LoraPathStrengthAndSDOps(
                path=self.config.distilled_lora_path,
                strength=strength,
                sd_ops=sd_ops,
            ))

        # Auto-detect FP8 checkpoint → pass QuantizationPolicy so ModelLedger
        # wraps Linear layers with upcast-on-forward (fp8_cast policy).
        quantization = None
        if dequantize_scaled_fp8_to_bf16:
            logger.info("FP8 checkpoint detected — dequantizing scaled-FP8 transformer weights to BF16 at load time")
        elif "fp8" in ckpt_desc.lower():
            quantization = QuantizationPolicy.fp8_cast()
            logger.info("FP8 checkpoint detected — using fp8_cast quantization policy")

        ledger = ModelLedger(
            dtype=self.dtype,
            device=self.device,
            checkpoint_path=ckpt_path,
            gemma_root_path=self.config.gemma_root,
            spatial_upsampler_path=self.config.spatial_upsampler_path,
            loras=tuple(loras) if loras else (),
            quantization=quantization,
        )
        if dequantize_scaled_fp8_to_bf16:
            attach_runtime_scaled_fp8_dequantization(ledger)
        return ledger

    def _encode_text_stagehand(
        self,
        ledger: ModelLedger,
        prompt: str,
        negative_prompt: str | None,
        enhance_prompt: bool,
        report: Callable[[str, float], None],
        encode_nag: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None,
               torch.Tensor | None, torch.Tensor | None]:
        """Encode prompt (and optional negative/NAG) through Gemma 3 12B with Stagehand.

        Returns (v_ctx_pos, a_ctx_pos, v_ctx_neg, a_ctx_neg, nag_v_ctx, nag_a_ctx).
        Negative contexts are None if negative_prompt is None/empty.
        NAG contexts are None if encode_nag is False.

        Sequencing audit (Phase 5, Task 3 — verified correct):
          1. Load Gemma to CPU via ledger.text_encoder()
          2. Move non-block params to GPU (_move_non_blocks_to_device)
          3. Create single StagehandRuntime on decoder blocks
          4. Optional: enhancement pass (enhance_t2v) in same runtime
          5. Encoding pass(es) for pos/neg/NAG prompts in same runtime
          6. te_runtime.shutdown() — Gemma evicted once, no redundant reload
          7. Process hidden states via emb_proc (small model, stays on GPU)
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
            config=_stagehand_config_te(self.config.gemma_root),
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
        if encode_nag:
            prompts_to_encode.append("")  # empty string for NAG null context

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
            del outputs, input_ids, attention_mask
            te_runtime.end_step()
            torch.cuda.empty_cache()  # free activation memory before next encoding

        _log_vram("encode_done")

        # Cleanup text encoder
        te_runtime.shutdown()
        del te_runtime, text_encoder, block_module, inner_model
        _flush()
        _log_vram("te_evicted")

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
        v_ctx_neg, a_ctx_neg = (None, None)
        nag_v_ctx, nag_a_ctx = (None, None)

        idx = 1
        if negative_prompt:
            v_ctx_neg, a_ctx_neg = results[idx]
            idx += 1
        if encode_nag and idx < len(results):
            nag_v_ctx, nag_a_ctx = results[idx]

        _log_vram(f"embeddings_done: video={tuple(v_ctx_pos.shape)}")
        return v_ctx_pos, a_ctx_pos, v_ctx_neg, a_ctx_neg, nag_v_ctx, nag_a_ctx

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

        # Apply chunked FFN if configured
        if getattr(self.config, "ffn_chunks", 1) > 1:
            from chunk_ffn import apply_ffn_chunking
            n_patched = apply_ffn_chunking(
                xfm_inner,
                num_chunks=self.config.ffn_chunks,
                threshold=self.config.ffn_chunk_threshold,
            )
            _log_vram(f"ChunkFFN applied ({n_patched} blocks, chunks={self.config.ffn_chunks})")

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
        stepper: EulerDiffusionStep | Res2sDiffusionStep,
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
        # HQ mode: use res2s sampler instead of Euler
        use_res2s: bool = False,
        noise_seed: int = -1,
    ) -> tuple[Any, Any]:
        """Run denoising with Stagehand-managed transformer.

        If video_guider_params is provided, uses CFG/STG guidance.
        Otherwise uses simple_denoising_func (distilled mode).
        If use_res2s is True, uses res2s second-order sampler instead of Euler.
        """
        use_guidance = video_guider_params is not None

        if use_guidance:
            if use_res2s:
                # HQ mode: use MultiModalGuider directly (not factory)
                video_guider = MultiModalGuider(
                    params=video_guider_params,
                    negative_context=v_ctx_neg,
                )
                audio_guider = MultiModalGuider(
                    params=audio_guider_params,
                    negative_context=a_ctx_neg,
                ) if audio_guider_params is not None else MultiModalGuider(
                    params=video_guider_params,
                    negative_context=v_ctx_neg,
                )
                base_fn = multi_modal_guider_denoising_func(
                    video_guider=video_guider,
                    audio_guider=audio_guider,
                    v_context=v_ctx,
                    a_context=a_ctx,
                    transformer=transformer,
                )
            else:
                # Dev mode: use factory pattern (per-step sigma-based guiders)
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

        # res2s does 2 model evals per step (current + midpoint), so count
        # model calls for progress, not denoising-loop iterations.
        _call = [0]
        n_steps = len(sigmas) - 1
        # res2s calls denoise_fn ~2× per step + final step
        n_total_calls = (n_steps * 2 + 1) if use_res2s else n_steps

        def wrapped_fn(video_state, audio_state, sigmas, step_index):
            xfm_runtime.begin_step(_call[0])
            with xfm_runtime.managed_forward():
                result = base_fn(video_state, audio_state, sigmas, step_index)
            xfm_runtime.end_step()
            _call[0] += 1
            # Diagnostic: check first 5 calls for NaN/Inf and value ranges
            if _call[0] <= 5:
                dv, da = result
                import torch as _t
                sig_val = sigmas[step_index].item() if step_index < len(sigmas) else -1
                logger.info(f"DIAG call {_call[0]} sigma={sig_val:.4f}: denoised nan={_t.isnan(dv).sum().item()}, "
                            f"inf={_t.isinf(dv).sum().item()}, "
                            f"mean={dv.float().mean():.4f}, std={dv.float().std():.4f}, "
                            f"range=[{dv.float().min():.4f}, {dv.float().max():.4f}]")
                logger.info(f"DIAG call {_call[0]}: input nan={_t.isnan(video_state.latent).sum().item()}, "
                            f"mean={video_state.latent.float().mean():.4f}, std={video_state.latent.float().std():.4f}, "
                            f"range=[{video_state.latent.float().min():.4f}, {video_state.latent.float().max():.4f}]")
            frac = progress_base + progress_range * (_call[0] / n_total_calls)
            report(f"{label} eval {_call[0]}/{n_total_calls}", frac)
            return result

        if use_res2s:
            _res2s_seed = noise_seed
            def loop_fn(sigmas_arg, video_state, audio_state, stepper_arg):
                return res2s_audio_video_denoising_loop(
                    sigmas=sigmas_arg,
                    video_state=video_state,
                    audio_state=audio_state,
                    stepper=stepper_arg,
                    denoise_fn=wrapped_fn,
                    noise_seed=_res2s_seed,
                )
        else:
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

    def _make_pass1_sigmas(self, latent: torch.Tensor) -> torch.Tensor:
        """Build Pass 1 sigmas via LTX2Scheduler with terminal=0.1."""
        return LTX2Scheduler().execute(
            steps=8,
            latent=latent,
            max_shift=2.05,
            base_shift=0.95,
            stretch=True,
            terminal=0.1,
        ).to(dtype=torch.float32, device=self.device)

    def _make_stage4_sigmas(self, factor: float) -> torch.Tensor:
        """Build Pass 4 refinement sigmas (manual schedule × rescale factor)."""
        return rescale_sigmas(FOUR_PASS_STAGE4_SIGMAS, factor).to(device=self.device)

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
        image_crf: int | None = None,
        audio_path: str | None = None,
        audio_start_time: float = 0.0,
        audio_max_duration: float | None = None,
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
        video_skip_step: int | None = None,
        audio_skip_step: int | None = None,
        stg_blocks: list[int] | None = None,
        keyframes: list[ImageConditioningInput] | None = None,
        ic_lora_video_path: str | None = None,
        ic_lora_strength: float | None = None,
        ic_lora_attention_strength: float | None = None,
        progress_cb: Callable[[str, float], None] | None = None,
        **_kwargs: Any,
    ) -> Path:
        """Run full two-stage generation. Returns path to output video.

        Supports both distilled mode (simple denoising, fixed sigmas) and
        dev mode (CFG/STG guidance, LTX2Scheduler, distilled LoRA for stage 2).
        """
        from dataclasses import replace as _dc_replace
        cfg = _dc_replace(self.config)  # snapshot — immune to UI writes
        is_distilled = cfg.is_distilled
        s = seed if seed is not None else cfg.seed
        w = width or cfg.width
        h = height or cfg.height
        nf = num_frames or cfg.num_frames
        fr = fps or cfg.fps
        do_enhance = enhance_prompt if enhance_prompt is not None else cfg.enhance_prompt
        img_path = image_path or cfg.image_path
        img_strength = image_strength if image_strength is not None else cfg.image_strength
        img_crf = image_crf if image_crf is not None else cfg.image_crf
        audio_limit = audio_max_duration if audio_max_duration is not None else cfg.audio_max_duration
        neg_prompt = negative_prompt if negative_prompt is not None else (cfg.negative_prompt if (not is_distilled or cfg.hq_sampler_enabled) else None)
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
        video_skip = video_skip_step if video_skip_step is not None else cfg.video_skip_step
        audio_skip = audio_skip_step if audio_skip_step is not None else cfg.audio_skip_step
        stg_blks = stg_blocks if stg_blocks is not None else cfg.stg_blocks

        t_total = time.perf_counter()

        def report(phase: str, frac: float) -> None:
            if progress_cb:
                progress_cb(phase, frac)

        report("Initializing...", 0.0)
        mode_str = (
            f"hq({cfg.hq_num_inference_steps}steps, cfg={v_cfg})"
            if cfg.hq_sampler_enabled
            else ("distilled" if is_distilled else f"dev({n_steps}steps, cfg={v_cfg})")
        )
        _log_vram(f"Starting generation: {w}x{h}, {nf}f, seed={s}, mode={mode_str}, enhance={do_enhance}, i2v={bool(img_path)}")

        # Build ModelLedger (loads nothing until .transformer() etc. called)
        use_hq = cfg.hq_sampler_enabled

        if use_hq:
            # HQ mode follows the official two-stage path: the full dev checkpoint is used
            # for all components, with the distilled LoRA applied at runtime at different
            # strengths for stage 1 and stage 2.
            hq_dev_checkpoint = cfg.resolve_hq_dev_checkpoint_path()
            hq_s1_checkpoint, hq_s2_checkpoint = cfg.resolve_hq_transformer_checkpoint_paths()
            logger.info("HQ mode using dev checkpoint: %s", hq_dev_checkpoint)
            ledger = self._build_ledger(checkpoint_override=hq_dev_checkpoint, skip_user_loras=True)
            if hq_s1_checkpoint and hq_s2_checkpoint:
                logger.info(
                    "HQ mode using premerged BF16 transformer checkpoints: %s | %s",
                    hq_s1_checkpoint,
                    hq_s2_checkpoint,
                )
                ledger_hq_s1 = self._build_ledger(
                    checkpoint_override=hq_s1_checkpoint,
                    skip_user_loras=True,
                )
                ledger_s2 = self._build_ledger(
                    checkpoint_override=hq_s2_checkpoint,
                    skip_user_loras=True,
                )
            else:
                ledger_hq_s1 = self._build_ledger(
                    checkpoint_override=hq_dev_checkpoint,
                    skip_user_loras=True,
                )
                attach_runtime_lora_merge(
                    ledger_hq_s1,
                    lora_path=cfg.distilled_lora_path,
                    strength=cfg.hq_distilled_lora_strength_s1,
                )
                ledger_s2 = self._build_ledger(
                    checkpoint_override=hq_dev_checkpoint,
                    skip_user_loras=True,
                )
                attach_runtime_lora_merge(
                    ledger_s2,
                    lora_path=cfg.distilled_lora_path,
                    strength=cfg.hq_distilled_lora_strength_s2,
                )
        else:
            ledger = self._build_ledger()
            ledger_hq_s1 = None
            # For dev mode stage 2, we need a separate ledger with distilled LoRA
            ledger_s2 = self._build_ledger(with_distilled_lora=True) if not is_distilled else ledger

        # Build image conditionings for I2V
        # Feature #6: preprocess_input_image controls CRF compression (0 = bypass)
        effective_crf = img_crf if cfg.preprocess_input_image else 0
        images: list[ImageConditioningInput] = []
        if img_path:
            images.append(ImageConditioningInput(
                path=img_path,
                frame_idx=0,
                strength=img_strength,
                crf=effective_crf,
            ))
        # Append additional keyframe conditionings (FML2V)
        if keyframes:
            images.extend(keyframes)

        # IC-LoRA video conditioning
        ic_video = ic_lora_video_path or cfg.ic_lora_video_path or None
        ic_strength = ic_lora_strength if ic_lora_strength is not None else cfg.ic_lora_strength
        ic_attn = ic_lora_attention_strength if ic_lora_attention_strength is not None else cfg.ic_lora_attention_strength

        # =====================================================================
        # Text encoding (Gemma 3 12B via Stagehand)
        # =====================================================================
        # Feature #5: Zero negative conditioning for I2V skips negative encoding
        skip_neg_encode = cfg.zero_negative_conditioning and bool(img_path)
        needs_neg = not is_distilled or use_hq  # HQ mode needs negative prompt for CFG
        encode_neg = None if (not needs_neg or skip_neg_encode) else neg_prompt

        v_ctx, a_ctx, v_ctx_neg, a_ctx_neg, nag_v_ctx, nag_a_ctx = self._encode_text_stagehand(
            ledger, prompt, encode_neg, do_enhance, report,
            encode_nag=cfg.nag_enabled,
        )

        # Feature #5: Replace negative with zeros if I2V + zero_negative_conditioning
        if skip_neg_encode and needs_neg:
            v_ctx_neg = torch.zeros_like(v_ctx)
            a_ctx_neg = torch.zeros_like(a_ctx) if a_ctx is not None else None

        # =====================================================================
        # A2V: Encode audio conditioning (if provided)
        # =====================================================================
        a2v_audio_path = audio_path or cfg.audio_path
        initial_audio_latent = None
        if a2v_audio_path:
            report("Encoding audio...", 0.09)
            decoded_audio = decode_audio_from_file(
                a2v_audio_path, self.device,
                start_time=audio_start_time,
                max_duration=audio_limit if audio_limit is not None else (nf / fr if nf and fr else None),
            )
            if decoded_audio is not None:
                audio_encoder = ledger.audio_encoder()
                initial_audio_latent = vae_encode_audio(decoded_audio, audio_encoder)
                # Trim/pad to match expected audio latent frames
                s1_audio_shape = AudioLatentShape.from_video_pixel_shape(
                    VideoPixelShape(batch=1, frames=nf, width=w // 2, height=h // 2, fps=fr),
                )
                expected_frames = s1_audio_shape.frames
                actual_frames = initial_audio_latent.shape[2]
                if actual_frames > expected_frames:
                    initial_audio_latent = initial_audio_latent[:, :, :expected_frames]
                elif actual_frames < expected_frames:
                    pad = torch.zeros(
                        initial_audio_latent.shape[0], initial_audio_latent.shape[1],
                        expected_frames - actual_frames, initial_audio_latent.shape[3],
                        device=self.device, dtype=self.dtype,
                    )
                    initial_audio_latent = torch.cat([initial_audio_latent, pad], dim=2)
                del audio_encoder
                _flush()
                _log_vram(f"Audio encoded: {tuple(initial_audio_latent.shape)}")
            else:
                logger.warning("No audio stream found in %s", a2v_audio_path)

        # Common components
        use_hq = cfg.hq_sampler_enabled
        generator = torch.Generator(device=self.device).manual_seed(s)
        noiser = GaussianNoiser(generator=generator)
        stepper = Res2sDiffusionStep() if use_hq else EulerDiffusionStep()
        pipeline_components = PipelineComponents(dtype=self.dtype, device=self.device)

        # Build guidance params for dev mode or HQ mode
        video_guider_params = None
        audio_guider_params = None
        if use_hq:
            hq_video = LTX_2_3_HQ_PARAMS.video_guider_params
            hq_audio = LTX_2_3_HQ_PARAMS.audio_guider_params
            video_guider_params = MultiModalGuiderParams(
                cfg_scale=hq_video.cfg_scale,
                stg_scale=hq_video.stg_scale,
                rescale_scale=hq_video.rescale_scale,
                modality_scale=hq_video.modality_scale,
                skip_step=hq_video.skip_step,
                stg_blocks=list(hq_video.stg_blocks),
            )
            audio_guider_params = MultiModalGuiderParams(
                cfg_scale=hq_audio.cfg_scale,
                stg_scale=hq_audio.stg_scale,
                rescale_scale=hq_audio.rescale_scale,
                modality_scale=hq_audio.modality_scale,
                skip_step=hq_audio.skip_step,
                stg_blocks=list(hq_audio.stg_blocks),
            )
        elif not is_distilled:
            video_guider_params = MultiModalGuiderParams(
                cfg_scale=v_cfg, stg_scale=v_stg, rescale_scale=v_rsc,
                modality_scale=a2v, skip_step=max(0, int(video_skip)), stg_blocks=stg_blks,
            )
            audio_guider_params = MultiModalGuiderParams(
                cfg_scale=a_cfg, stg_scale=a_stg, rescale_scale=a_rsc,
                modality_scale=v2a, skip_step=max(0, int(audio_skip)), stg_blocks=stg_blks,
            )

        if cfg.use_four_pass:
            video_latent, audio_latent, fr = self._generate_four_pass(
                cfg=cfg, w=w, h=h, nf=nf, fr=fr, s=s,
                ledger=ledger, ledger_s2=ledger_s2,
                v_ctx=v_ctx, a_ctx=a_ctx, v_ctx_neg=v_ctx_neg, a_ctx_neg=a_ctx_neg,
                nag_v_ctx=nag_v_ctx, nag_a_ctx=nag_a_ctx,
                noiser=noiser, stepper=stepper, pipeline_components=pipeline_components,
                generator=generator, images=images,
                initial_audio_latent=initial_audio_latent,
                video_guider_params=video_guider_params,
                audio_guider_params=audio_guider_params,
                report=report,
            )
        else:
            video_latent, audio_latent, fr = self._generate_two_stage(
                cfg=cfg, is_distilled=is_distilled, w=w, h=h, nf=nf, fr=fr, n_steps=n_steps, seed=s,
                ledger=ledger_hq_s1 or ledger, ledger_s2=ledger_s2,
                vae_ledger=ledger if ledger_hq_s1 else None,
                v_ctx=v_ctx, a_ctx=a_ctx, v_ctx_neg=v_ctx_neg, a_ctx_neg=a_ctx_neg,
                nag_v_ctx=nag_v_ctx, nag_a_ctx=nag_a_ctx,
                noiser=noiser, stepper=stepper, pipeline_components=pipeline_components,
                generator=generator, images=images,
                initial_audio_latent=initial_audio_latent,
                video_guider_params=video_guider_params,
                audio_guider_params=audio_guider_params,
                report=report,
                ic_lora_video_path=ic_video,
                ic_lora_strength=ic_strength,
                ic_lora_attention_strength=ic_attn,
            )

        del v_ctx, a_ctx, v_ctx_neg, a_ctx_neg, nag_v_ctx, nag_a_ctx
        _flush()
        _log_vram("Transformer freed, starting decode")

        # =====================================================================
        # VAE decode + save
        # =====================================================================
        report("Decoding video...", 0.90)
        t0 = time.perf_counter()

        # Feature #3: VAE decoder noise injection
        if cfg.decoder_noise_enabled:
            video_latent = apply_decoder_noise(
                video_latent, cfg.decoder_noise_scale, cfg.decoder_noise_shift, cfg.decoder_noise_seed,
            )
            _log_vram(f"Decoder noise applied (scale={cfg.decoder_noise_scale}, shift={cfg.decoder_noise_shift})")

        # Feature #4: Configurable tiled spatio-temporal VAE decode (always on)
        tiling = TilingConfig(
            spatial_config=SpatialTilingConfig(
                tile_size_in_pixels=cfg.vae_spatial_tile_pixels,
                tile_overlap_in_pixels=cfg.vae_tile_overlap_pixels,
            ),
            temporal_config=TemporalTilingConfig(
                tile_size_in_frames=cfg.vae_temporal_tile_frames,
                tile_overlap_in_frames=cfg.vae_temporal_overlap_frames,
            ),
        )
        decoded_video = vae_decode_video(
            video_latent,
            ledger.video_decoder(),
            tiling,
            generator,
        )
        decoded_audio = None
        if audio_latent is not None:
            decoded_audio = vae_decode_audio(
                audio_latent.to(self.device),
                ledger.audio_decoder(),
                ledger.vocoder(),
            )
            decoded_audio = normalize_output_audio(decoded_audio)
        _log_vram(f"vae_decode_done ({time.perf_counter() - t0:.1f}s)")

        # Save output
        report("Saving...", 0.95)
        out_dir = cfg.ensure_output_dir()
        import datetime
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        mode_tag = "hq" if cfg.hq_sampler_enabled else ("distilled" if is_distilled else "dev")
        if ic_video:
            mode_tag += "_iclora"
        four_pass_tag = "_4pass" if cfg.use_four_pass else ""
        out_path = out_dir / f"ltx2_{mode_tag}{four_pass_tag}_{ts}_s{s}.mp4"

        actual_nf = video_latent.shape[2] * 8 + 1 if cfg.use_four_pass else nf
        video_chunks_number = get_video_chunks_number(actual_nf, tiling)
        encode_video(
            video=decoded_video,
            fps=fr,
            audio=decoded_audio,
            output_path=str(out_path),
            video_chunks_number=video_chunks_number,
        )

        # Final GPU cleanup
        del decoded_video, decoded_audio, video_latent, audio_latent
        del generator, noiser, stepper, pipeline_components
        del ledger_s2, ledger
        _flush()

        _log_vram(f"generation_complete — TOTAL: {time.perf_counter() - t_total:.1f}s")
        report("Done!", 1.0)
        return out_path

    def _generate_two_stage(
        self, *, cfg, is_distilled, w, h, nf, fr, n_steps, seed,
        ledger, ledger_s2, v_ctx, a_ctx, v_ctx_neg, a_ctx_neg,
        nag_v_ctx, nag_a_ctx, noiser, stepper, pipeline_components,
        generator, images, initial_audio_latent,
        video_guider_params, audio_guider_params, report,
        vae_ledger=None,
        ic_lora_video_path: str | None = None,
        ic_lora_strength: float = 1.0,
        ic_lora_attention_strength: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor | None, float]:
        """Standard two-stage pipeline: half-res → spatial upscale + refine.

        Returns (video_latent, audio_latent, fps).
        """
        # For HQ mode, vae_ledger points to the distilled ledger (matching VAE/connector dims)
        # while ledger points to the HQ-merged checkpoint (for transformer only)
        _vae_ledger = vae_ledger or ledger

        # =================================================================
        # Stage 1: Half-res denoise
        # =================================================================
        report("Stage 1: Loading transformer...", 0.10)
        t0 = time.perf_counter()

        s1_w, s1_h = w // 2, h // 2
        s1_shape = VideoPixelShape(batch=1, frames=nf, width=s1_w, height=s1_h, fps=fr)

        use_hq = cfg.hq_sampler_enabled

        if use_hq:
            # HQ mode: use LTX2Scheduler with latent-aware sigma shifting (matches official HQ pipeline)
            from ltx_core.types import VideoLatentShape
            hq_steps = cfg.hq_num_inference_steps
            empty_latent = torch.empty(VideoLatentShape.from_pixel_shape(s1_shape).to_torch_shape())
            s1_sigmas = LTX2Scheduler().execute(latent=empty_latent, steps=hq_steps).to(dtype=torch.float32, device=self.device)
        elif is_distilled:
            s1_sigmas = torch.tensor(DISTILLED_SIGMA_VALUES, dtype=torch.float32, device=self.device)
        else:
            if cfg.scheduler_type == "bong_tangent":
                s1_sigmas = bong_tangent_sigmas(n_steps).to(device=self.device)
            else:
                s1_sigmas = LTX2Scheduler().execute(steps=n_steps).to(dtype=torch.float32, device=self.device)

        # Load video encoder for image/video conditioning (needed before transformer)
        s1_conditionings: list = []
        need_encoder = bool(images) or bool(ic_lora_video_path)
        if need_encoder:
            video_encoder_for_cond = _vae_ledger.video_encoder()

        if images:
            report("Stage 1: Encoding conditioning image...", 0.11)
            s1_conditionings = combined_image_conditionings(
                images=images, height=s1_h, width=s1_w,
                video_encoder=video_encoder_for_cond,
                dtype=self.dtype, device=self.device,
            )

        # IC-LoRA: encode reference video and create conditioning
        ic_lora_ref_downscale = 1
        if ic_lora_video_path:
            report("Stage 1: Encoding IC-LoRA reference video...", 0.11)
            from ltx_core.conditioning import (
                ConditioningItemAttentionStrengthWrapper,
                VideoConditionByReferenceLatent,
            )
            from ltx_core.types import VideoLatentShape
            # Read downscale factor from LoRA metadata (if any)
            ic_lora_ref_downscale = _read_ic_lora_downscale(cfg.lora_paths)
            scale = ic_lora_ref_downscale
            ref_h = s1_h // scale
            ref_w = s1_w // scale
            ref_video = load_video_conditioning(
                video_path=ic_lora_video_path,
                height=ref_h, width=ref_w,
                frame_cap=nf,
                dtype=self.dtype, device=self.device,
            )
            encoded_ref = video_encoder_for_cond(ref_video)
            cond = VideoConditionByReferenceLatent(
                latent=encoded_ref,
                downscale_factor=scale,
                strength=ic_lora_strength,
            )
            if ic_lora_attention_strength < 1.0:
                cond = ConditioningItemAttentionStrengthWrapper(
                    cond, attention_mask=ic_lora_attention_strength,
                )
            s1_conditionings.append(cond)
            del ref_video, encoded_ref
            _log_vram(f"IC-LoRA reference encoded (scale={scale}, strength={ic_lora_strength})")

        if need_encoder:
            del video_encoder_for_cond
            _flush()

        transformer, xfm_inner, xfm_runtime = self._setup_stagehand_transformer(ledger)

        # Apply NAG to cross-attention modules before denoising
        nag_patch = None
        if cfg.nag_enabled and nag_v_ctx is not None:
            nag_patch = NAGPatch(nag_v_ctx, nag_a_ctx, cfg.nag_scale, cfg.nag_alpha, cfg.nag_tau)
            nag_patch.apply(transformer)

        video_state, audio_state = self._stagehand_denoise(
            transformer, xfm_runtime, v_ctx, a_ctx, s1_shape, s1_sigmas,
            noiser, stepper, pipeline_components, s1_conditionings, report,
            label="Stage 1", progress_base=0.12, progress_range=0.30,
            initial_audio_latent=initial_audio_latent,
            v_ctx_neg=v_ctx_neg, a_ctx_neg=a_ctx_neg,
            video_guider_params=video_guider_params,
            audio_guider_params=audio_guider_params,
            use_res2s=use_hq, noise_seed=seed,
        )
        _log_vram(f"stage1_done ({time.perf_counter() - t0:.1f}s)")

        # Feature #2: Two-stage sampling — inject noise between stages
        s1_video_latent = video_state.latent
        if cfg.two_stage_sampling:
            report("Injecting inter-stage noise...", 0.43)
            s1_video_latent = inject_noise(s1_video_latent, cfg.two_stage_noise_strength, generator)

        if nag_patch is not None:
            nag_patch.remove(transformer)

        s1_video_latent = s1_video_latent.cpu()
        s1_audio_latent = audio_state.latent.cpu() if audio_state.latent is not None else None
        xfm_runtime.shutdown()
        del xfm_runtime, transformer, xfm_inner
        _flush()
        _log_vram("xfm_evicted_stage1")

        # =================================================================
        # Stage 2: Spatial upsample + refine (3 steps, always distilled/simple)
        # =================================================================
        report("Stage 2: Upsampling...", 0.45)
        t0 = time.perf_counter()

        video_encoder = _vae_ledger.video_encoder()
        upsampler = _vae_ledger.spatial_upsampler()
        upscaled = upsample_video(
            latent=s1_video_latent.to(self.device)[:1],
            video_encoder=video_encoder, upsampler=upsampler,
        )

        s2_conditionings: list = []
        if images:
            s2_conditionings = combined_image_conditionings(
                images=images, height=h, width=w,
                video_encoder=video_encoder,
                dtype=self.dtype, device=self.device,
            )

        del video_encoder, upsampler, s1_video_latent
        _flush()
        _log_vram(f"spatial_upsample_done: {tuple(upscaled.shape)}")

        report("Stage 2: Reloading transformer...", 0.50)
        transformer2, xfm_inner2, xfm_runtime2 = self._setup_stagehand_transformer(ledger_s2)

        nag_patch2 = None
        if cfg.nag_enabled and nag_v_ctx is not None:
            nag_patch2 = NAGPatch(nag_v_ctx, nag_a_ctx, cfg.nag_scale, cfg.nag_alpha, cfg.nag_tau)
            nag_patch2.apply(transformer2)

        s2_sigmas = torch.tensor(STAGE_2_DISTILLED_SIGMA_VALUES, dtype=torch.float32, device=self.device)

        # Spatial tiling: split stage 2 into overlapping tiles for high-res
        use_tiling = (
            cfg.spatial_tile_enabled
            and not s2_conditionings  # tiling + I2V not yet supported
            and _needs_spatial_tiling(h, w, cfg.spatial_tile_pixels)
        )

        if use_tiling:
            from spatial_tiling import compute_tiles, blend_tiles, SPATIAL_SCALE
            tiles = compute_tiles(h, w, cfg.spatial_tile_pixels, cfg.spatial_tile_overlap)
            _log_vram(f"Spatial tiling: {len(tiles)} tiles at {cfg.spatial_tile_pixels}px")
            S = SPATIAL_SCALE
            tile_latents: list[torch.Tensor] = []
            for ti, tile in enumerate(tiles):
                tile_initial = upscaled[:, :, :, tile.y0 // S : tile.y1 // S, tile.x0 // S : tile.x1 // S]
                tile_shape = VideoPixelShape(batch=1, frames=nf, width=tile.w, height=tile.h, fps=fr)
                vs, _ = self._stagehand_denoise(
                    transformer2, xfm_runtime2, v_ctx, a_ctx, tile_shape, s2_sigmas,
                    noiser, stepper, pipeline_components, [], report,
                    label=f"Tile {ti + 1}/{len(tiles)}", progress_base=0.55, progress_range=0.15,
                    noise_scale=s2_sigmas[0].item(),
                    initial_video_latent=tile_initial.contiguous(),
                )
                tile_latents.append(vs.latent)
            blended = blend_tiles(tiles, tile_latents, h, w)
            # Wrap as LatentState for downstream compatibility
            from ltx_core.types import LatentState
            video_state = LatentState(
                latent=blended,
                denoise_mask=torch.ones(1, device=self.device),
                positions=torch.zeros(1, device=self.device),
                clean_latent=blended,
            )
            audio_state = LatentState(
                latent=s1_audio_latent.to(self.device) if s1_audio_latent is not None else torch.zeros(1, device=self.device),
                denoise_mask=torch.zeros(1, device=self.device),
                positions=torch.zeros(1, device=self.device),
                clean_latent=torch.zeros(1, device=self.device),
            )
        else:
            s2_shape = VideoPixelShape(batch=1, frames=nf, width=w, height=h, fps=fr)
            video_state, audio_state = self._stagehand_denoise(
                transformer2, xfm_runtime2, v_ctx, a_ctx, s2_shape, s2_sigmas,
                noiser, stepper, pipeline_components, s2_conditionings, report,
                label="Stage 2", progress_base=0.55, progress_range=0.15,
                noise_scale=s2_sigmas[0].item(),
                initial_video_latent=upscaled,
                initial_audio_latent=s1_audio_latent.to(self.device) if s1_audio_latent is not None else None,
                use_res2s=use_hq, noise_seed=seed + 1,
            )
        _log_vram(f"stage2_done ({time.perf_counter() - t0:.1f}s)")

        if nag_patch2 is not None:
            nag_patch2.remove(transformer2)
        xfm_runtime2.shutdown()
        del xfm_runtime2, transformer2, xfm_inner2, upscaled, s1_audio_latent
        _flush()
        _log_vram("xfm_evicted_final")

        video_latent = video_state.latent
        audio_latent = audio_state.latent.cpu() if audio_state.latent is not None else None
        return video_latent, audio_latent, fr

    def _generate_four_pass(
        self, *, cfg, w, h, nf, fr, s,
        ledger, ledger_s2, v_ctx, a_ctx, v_ctx_neg, a_ctx_neg,
        nag_v_ctx, nag_a_ctx, noiser, stepper, pipeline_components,
        generator, images, initial_audio_latent,
        video_guider_params, audio_guider_params, report,
    ) -> tuple[torch.Tensor, torch.Tensor | None, float]:
        """Four-pass pipeline: full-res Pass 1 → spatial upscale → temporal upscale → refinement.

        Pass 1: Full-res denoise with LTX2Scheduler(terminal=0.1), 8 steps
        Pass 2: Spatial upscale 2x (latent-space, no denoise)
        Pass 3: Temporal upscale 2x (latent-space, no denoise)
        Pass 4: Refinement with manual sigmas × rescale_factor

        Transformer is loaded ONCE (via ledger_s2 for distilled LoRA) and kept
        resident across Pass 1 and Pass 4.

        Returns (video_latent, audio_latent, fps).
        """
        # =================================================================
        # Pass 1: Full-res denoise (8 steps, terminal=0.1)
        # =================================================================
        report("Pass 1: Loading transformer...", 0.10)
        t0 = time.perf_counter()

        p1_shape = VideoPixelShape(batch=1, frames=nf, width=w, height=h, fps=fr)

        # Build image conditionings at full resolution
        p1_conditionings: list = []
        video_encoder_for_cond = None
        if images:
            report("Pass 1: Encoding conditioning image...", 0.11)
            video_encoder_for_cond = ledger.video_encoder()
            p1_conditionings = combined_image_conditionings(
                images=images, height=h, width=w,
                video_encoder=video_encoder_for_cond,
                dtype=self.dtype, device=self.device,
            )
            del video_encoder_for_cond
            _flush()

        # Load transformer once — use ledger_s2 (has distilled LoRA)
        transformer, xfm_inner, xfm_runtime = self._setup_stagehand_transformer(ledger_s2)

        # Apply NAG
        nag_patch = None
        if cfg.nag_enabled and nag_v_ctx is not None:
            nag_patch = NAGPatch(nag_v_ctx, nag_a_ctx, cfg.nag_scale, cfg.nag_alpha, cfg.nag_tau)
            nag_patch.apply(transformer)

        # Build Pass 1 sigmas — need a dummy latent for scheduler's latent-aware shifting
        # Shape: [1, 128, nf_latent, h_latent, w_latent]
        nf_lat = (nf - 1) // 8
        h_lat, w_lat = h // 16, w // 16
        dummy_latent = torch.zeros(1, 128, nf_lat, h_lat, w_lat, device=self.device, dtype=self.dtype)
        p1_sigmas = self._make_pass1_sigmas(dummy_latent)
        del dummy_latent

        video_state, audio_state = self._stagehand_denoise(
            transformer, xfm_runtime, v_ctx, a_ctx, p1_shape, p1_sigmas,
            noiser, stepper, pipeline_components, p1_conditionings, report,
            label="Pass 1", progress_base=0.12, progress_range=0.25,
            initial_audio_latent=initial_audio_latent,
            v_ctx_neg=v_ctx_neg, a_ctx_neg=a_ctx_neg,
            video_guider_params=video_guider_params,
            audio_guider_params=audio_guider_params,
        )
        _log_vram(f"stage1_done ({time.perf_counter() - t0:.1f}s)")

        video_latent = video_state.latent
        audio_latent = audio_state.latent.cpu() if audio_state.latent is not None else None
        del video_state, audio_state

        # =================================================================
        # Pass 2: Spatial upscale 2x (latent-space, no denoise)
        # =================================================================
        report("Pass 2: Spatial upscale...", 0.40)
        t0 = time.perf_counter()

        video_encoder = ledger.video_encoder()
        upsampler = ledger.spatial_upsampler()
        video_latent = upsample_video(
            latent=video_latent[:1],
            video_encoder=video_encoder,
            upsampler=upsampler,
        )
        del video_encoder, upsampler
        _flush()
        _log_vram(f"spatial_upsample_done ({time.perf_counter() - t0:.1f}s): {tuple(video_latent.shape)}")

        # =================================================================
        # Pass 3: Temporal upscale 2x (latent-space, no denoise)
        # =================================================================
        if cfg.temporal_upscaler_path:
            report("Pass 3: Temporal upscale...", 0.50)
            t0 = time.perf_counter()

            video_encoder = ledger.video_encoder()
            temporal_upsampler = self._load_temporal_upsampler()
            video_latent = upsample_video(
                latent=video_latent,
                video_encoder=video_encoder,
                upsampler=temporal_upsampler,
            )
            del temporal_upsampler, video_encoder
            _flush()
            fr = cfg.stage4_fps  # FPS doubles after temporal upscale
            _log_vram(f"temporal_upsample_done ({time.perf_counter() - t0:.1f}s): {tuple(video_latent.shape)}")

        # =================================================================
        # Pass 4: Refinement sampling at upscaled resolution
        # =================================================================
        report("Pass 4: Refinement...", 0.60)
        t0 = time.perf_counter()

        s4_sigmas = self._make_stage4_sigmas(cfg.stage4_rescale_factor)

        s4_generator = torch.Generator(device=self.device).manual_seed(cfg.stage4_seed)
        s4_noiser = GaussianNoiser(generator=s4_generator)

        upscaled_nf = video_latent.shape[2]
        s4_shape = VideoPixelShape(
            batch=1,
            frames=upscaled_nf * 8 + 1,
            width=video_latent.shape[4] * 16,
            height=video_latent.shape[3] * 16,
            fps=fr,
        )

        # Build conditionings at upscaled resolution
        s4_conditionings: list = []
        if images:
            ve = ledger.video_encoder()
            s4_conditionings = combined_image_conditionings(
                images=images,
                height=s4_shape.height, width=s4_shape.width,
                video_encoder=ve, dtype=self.dtype, device=self.device,
            )
            del ve
            _flush()

        video_state_s4, audio_state_s4 = self._stagehand_denoise(
            transformer, xfm_runtime, v_ctx, a_ctx, s4_shape, s4_sigmas,
            s4_noiser, stepper, pipeline_components, s4_conditionings, report,
            label="Pass 4", progress_base=0.65, progress_range=0.20,
            noise_scale=s4_sigmas[0].item(),
            initial_video_latent=video_latent,
            initial_audio_latent=audio_latent.to(self.device) if audio_latent is not None else None,
        )
        _log_vram(f"stage2_done ({time.perf_counter() - t0:.1f}s)")

        video_latent = video_state_s4.latent
        audio_latent = audio_state_s4.latent.cpu() if audio_state_s4.latent is not None else None
        del video_state_s4, audio_state_s4

        # Free transformer
        if nag_patch is not None:
            nag_patch.remove(transformer)
        xfm_runtime.shutdown()
        del xfm_runtime, transformer, xfm_inner
        _flush()
        _log_vram("xfm_evicted_final")

        return video_latent, audio_latent, fr

    def _load_temporal_upsampler(self) -> Any:
        """Load temporal upsampler directly (ModelLedger doesn't support it)."""
        if not self.config.temporal_upscaler_path:
            raise ValueError("temporal_upscaler_path not configured")

        from safetensors.torch import load_file
        from ltx_core.model.upsampler import LatentUpsamplerConfigurator

        model = LatentUpsamplerConfigurator.from_config({
            "temporal_upsample": True,
            "spatial_upsample": False,
        })
        state_dict = load_file(self.config.temporal_upscaler_path)
        model.load_state_dict(state_dict)
        return model.to(device=self.device, dtype=self.dtype).eval()
