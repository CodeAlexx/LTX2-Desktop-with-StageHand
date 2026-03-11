"""Application configuration."""

from __future__ import annotations

import json as _json
import logging
from glob import glob
from dataclasses import dataclass, field
from pathlib import Path

_log = logging.getLogger(__name__)
_CONFIG_FILE = Path.home() / ".config" / "ltx2_desktop" / "config.json"
_LTX23_HF_SNAPSHOT = Path(
    "/home/alex/.cache/huggingface/hub/models--Lightricks--LTX-2.3/"
    "snapshots/5a9c1c680bc66c159f708143bf274739961ecd08"
)
_LTX23_HF_DEV_BF16 = _LTX23_HF_SNAPSHOT / "ltx-2.3-22b-dev.safetensors"
_LTX23_HQ_S1_BF16_TRANSFORMER_GLOB = "/home/alex/EriDiffusion/Models/ltx2/ltx2-hq-s1-bf16-transformer-*.safetensors"
_LTX23_HQ_S2_BF16_TRANSFORMER_GLOB = "/home/alex/EriDiffusion/Models/ltx2/ltx2-hq-s2-bf16-transformer-*.safetensors"


@dataclass
class AppConfig:
    """Central configuration for the LTX-2 desktop app."""

    # --- Model paths (update these to your local model locations) ---
    distilled_checkpoint_path: str = "/home/alex/EriDiffusion/Models/ltx2/ltx-2.3-22b-distilled.safetensors"
    dev_checkpoint_path: str = "/home/alex/EriDiffusion/Models/ltx2/ltx-2.3-22b-dev-fp8.safetensors"
    hq_checkpoint_s1: str = "/home/alex/EriDiffusion/Models/ltx2/ltx2-hq-s1-fp8.safetensors"
    hq_checkpoint_s2: str = "/home/alex/EriDiffusion/Models/ltx2/ltx2-hq-s2-fp8.safetensors"
    gemma_root: str = "/home/alex/models/gemma-3-12b-it"
    distilled_lora_path: str = "/home/alex/EriDiffusion/Models/ltx2/ltx-2.3-22b-distilled-lora-384.safetensors"
    distilled_lora_strength: float = 0.25  # 0.25-0.3 recommended; 1.0 over-sharpens
    spatial_upsampler_path: str = "/home/alex/EriDiffusion/Models/ltx2/ltx-2.3-spatial-upscaler-x2-1.0.safetensors"
    temporal_upscaler_path: str = ""

    # Output
    output_dir: str = "/home/alex/serenity/output/ltx2"

    # UI
    theme_name: str = "Serenity"
    ui_scale: float = 0.0  # 0 = auto from display resolution

    # --- Pipeline mode ---
    pipeline_mode: str = "distilled"  # "distilled" or "dev"

    # HQ sampler (res_2s second-order RK instead of Euler)
    hq_sampler_enabled: bool = False
    hq_num_inference_steps: int = 15  # LTX2Scheduler steps for stage 1
    hq_distilled_lora_strength_s1: float = 0.25  # distilled LoRA strength for stage 1
    hq_distilled_lora_strength_s2: float = 0.5  # distilled LoRA strength for stage 2 (official default)

    # Generation defaults
    width: int = 768
    height: int = 512
    num_frames: int = 81  # 8*10+1
    fps: float = 25.0
    seed: int = 42

    # Prompt enhancement (uses Gemma 3 to rewrite prompts for better quality)
    enhance_prompt: bool = False

    # I2V image conditioning
    image_path: str = ""  # empty = T2V mode, path = I2V mode
    image_strength: float = 0.9
    image_crf: int = 35

    # A2V audio conditioning
    audio_path: str = ""  # empty = no audio input, path = A2V mode
    audio_start_time: float = 0.0
    audio_max_duration: float | None = None  # None = use full audio

    # Dev-mode guidance (ignored in distilled mode)
    num_inference_steps: int = 30
    video_cfg_scale: float = 3.0
    video_stg_scale: float = 1.0
    video_rescale: float = 0.7
    audio_cfg_scale: float = 7.0
    audio_stg_scale: float = 1.0
    audio_rescale: float = 0.7
    a2v_scale: float = 3.0
    v2a_scale: float = 3.0
    video_skip_step: int = 0
    audio_skip_step: int = 0
    stg_blocks: list[int] = field(default_factory=lambda: [28])
    negative_prompt: str = (
        "blurry, out of focus, overexposed, underexposed, low contrast, washed out colors, "
        "excessive noise, grainy texture, poor lighting, flickering, motion blur, distorted "
        "proportions, deformed facial features, extra limbs, disfigured hands, artifacts, "
        "inconsistent perspective, camera shake, cartoonish rendering, 3D CGI look, "
        "unrealistic materials, uncanny valley effect"
    )

    # Scheduler (dev mode only)
    scheduler_type: str = "default"  # "default" or "bong_tangent"

    # Two-stage sampling (inject noise between stages)
    two_stage_sampling: bool = False
    two_stage_noise_strength: float = 7.0

    # VAE decoder noise injection
    decoder_noise_enabled: bool = False
    decoder_noise_scale: float = 0.05
    decoder_noise_shift: float = 0.025
    decoder_noise_seed: int = 666

    # VAE tiling (always on, configurable params)
    vae_temporal_tiles: int = 2
    vae_spatial_tile_pixels: int = 512
    vae_tile_overlap_pixels: int = 64
    vae_temporal_tile_frames: int = 64
    vae_temporal_overlap_frames: int = 24

    # I2V-only options
    zero_negative_conditioning: bool = False
    preprocess_input_image: bool = False

    # Chunked FFN (activation VRAM reduction)
    ffn_chunks: int = 1             # 1 = disabled; 2 = ~50% peak FFN VRAM reduction
    ffn_chunk_threshold: int = 4096  # only chunk sequences longer than this

    # Spatial tiling (high-res denoising)
    spatial_tile_enabled: bool = False
    spatial_tile_pixels: int = 512   # tile size in pixels (must be div by 32)
    spatial_tile_overlap: int = 128  # overlap in pixels (must be div by 32)

    # Four-pass pipeline
    use_four_pass: bool = False
    stage4_rescale_factor: float = 0.895
    stage4_seed: int = 42
    stage4_fps: float = 50.0  # FPS after temporal upscale (2x base)

    # Long video
    long_video_enabled: bool = False
    long_video_preset: str = "balanced"
    long_video_total_seconds: float = 10.0
    long_video_temporal_tile_size: int = 80
    long_video_temporal_overlap: int = 24
    long_video_adain_factor: float = 0.0
    long_video_per_step_adain: bool = False
    long_video_per_step_adain_factors: str = "0.9,0.75,0.5,0.25,0.0,0.0"
    long_video_long_memory_strength: float = 0.0
    long_video_anchor_frames: int = 1

    # Keyframe conditioning (FML2V — First/Middle/Last frame injection)
    keyframe_first_path: str = ""
    keyframe_first_strength: float = 0.9
    keyframe_middle_path: str = ""
    keyframe_middle_strength: float = 0.9
    keyframe_last_path: str = ""
    keyframe_last_strength: float = 0.9

    # IC-LoRA — In-Context LoRA video conditioning
    ic_lora_video_path: str = ""          # path to reference/control video
    ic_lora_strength: float = 1.0         # conditioning strength (0=denoised, 1=clean reference)
    ic_lora_attention_strength: float = 1.0  # attention weight (0=ignore, 1=full influence)

    # NAG — Normalized Attention Guidance
    # Community default: scale=11, alpha=0.25, tau=2.5
    nag_enabled: bool = False
    nag_scale: float = 11.0
    nag_alpha: float = 0.25
    nag_tau: float = 2.5

    # LoRA
    lora_paths: list[str] = field(default_factory=list)
    lora_strengths: list[float] = field(default_factory=list)

    @property
    def is_distilled(self) -> bool:
        return self.pipeline_mode == "distilled"

    @property
    def checkpoint_path(self) -> str:
        return self.distilled_checkpoint_path if self.is_distilled else self.dev_checkpoint_path

    def resolve_hq_dev_checkpoint_path(self) -> str:
        """Prefer the official BF16 dev checkpoint for HQ mode when it exists locally."""
        if _LTX23_HF_DEV_BF16.exists():
            return str(_LTX23_HF_DEV_BF16)
        return self.dev_checkpoint_path

    def resolve_hq_transformer_checkpoint_paths(self) -> tuple[tuple[str, ...] | None, tuple[str, ...] | None]:
        """Return optional premerged BF16 HQ transformer checkpoint shard lists."""
        s1 = tuple(sorted(glob(_LTX23_HQ_S1_BF16_TRANSFORMER_GLOB)))
        s2 = tuple(sorted(glob(_LTX23_HQ_S2_BF16_TRANSFORMER_GLOB)))
        if s1 and s2:
            return s1, s2
        return None, None

    def ensure_output_dir(self) -> Path:
        p = Path(self.output_dir)
        p.mkdir(parents=True, exist_ok=True)
        return p

    def restore_v1_baseline(self) -> None:
        """Restore the closest practical approximation of the original v1 app defaults."""
        self.pipeline_mode = "distilled"
        self.hq_sampler_enabled = False
        self.hq_num_inference_steps = 15
        self.hq_distilled_lora_strength_s1 = 0.25
        self.hq_distilled_lora_strength_s2 = 0.5
        self.width = 768
        self.height = 512
        self.num_frames = 81
        self.fps = 25.0
        self.seed = 42
        self.enhance_prompt = False
        self.image_path = ""
        self.image_strength = 0.9
        self.image_crf = 35
        self.audio_path = ""
        self.audio_start_time = 0.0
        self.audio_max_duration = None
        self.num_inference_steps = 30
        self.video_cfg_scale = 3.0
        self.video_stg_scale = 1.0
        self.video_rescale = 0.7
        self.audio_cfg_scale = 7.0
        self.audio_stg_scale = 1.0
        self.audio_rescale = 0.7
        self.a2v_scale = 3.0
        self.v2a_scale = 3.0
        self.video_skip_step = 0
        self.audio_skip_step = 0
        self.stg_blocks = [28]
        self.negative_prompt = (
            "blurry, out of focus, overexposed, underexposed, low contrast, washed out colors, "
            "excessive noise, grainy texture, poor lighting, flickering, motion blur, distorted "
            "proportions, deformed facial features, extra limbs, disfigured hands, artifacts, "
            "inconsistent perspective, camera shake, cartoonish rendering, 3D CGI look, "
            "unrealistic materials, uncanny valley effect"
        )
        self.scheduler_type = "default"
        self.two_stage_sampling = False
        self.two_stage_noise_strength = 7.0
        self.decoder_noise_enabled = False
        self.decoder_noise_scale = 0.05
        self.decoder_noise_shift = 0.025
        self.decoder_noise_seed = 666
        self.zero_negative_conditioning = False
        self.preprocess_input_image = False
        self.use_four_pass = False
        self.long_video_enabled = False
        self.long_video_preset = "balanced"
        self.long_video_total_seconds = 10.0
        self.long_video_temporal_tile_size = 80
        self.long_video_temporal_overlap = 24
        self.long_video_adain_factor = 0.0
        self.long_video_per_step_adain = False
        self.long_video_per_step_adain_factors = "0.9,0.75,0.5,0.25,0.0,0.0"
        self.long_video_long_memory_strength = 0.0
        self.long_video_anchor_frames = 1
        self.keyframe_first_path = ""
        self.keyframe_first_strength = 0.9
        self.keyframe_middle_path = ""
        self.keyframe_middle_strength = 0.9
        self.keyframe_last_path = ""
        self.keyframe_last_strength = 0.9
        self.ic_lora_video_path = ""
        self.ic_lora_strength = 1.0
        self.ic_lora_attention_strength = 1.0
        self.nag_enabled = False
        self.nag_scale = 11.0
        self.nag_alpha = 0.25
        self.nag_tau = 2.5
        self.ffn_chunks = 1
        self.ffn_chunk_threshold = 4096
        self.spatial_tile_enabled = False
        self.spatial_tile_pixels = 512
        self.spatial_tile_overlap = 128
        self.stage4_rescale_factor = 0.895
        self.stage4_seed = 42
        self.stage4_fps = 50.0
        self.lora_paths = []
        self.lora_strengths = []

    def save(self, path: Path | None = None) -> None:
        """Persist config to disk as JSON."""
        p = Path(path) if path else _CONFIG_FILE
        p.parent.mkdir(parents=True, exist_ok=True)
        data = {}
        for name in self.__dataclass_fields__:
            val = getattr(self, name)
            data[name] = val
        p.write_text(_json.dumps(data, indent=2))
        _log.info("Config saved to %s", p)

    @classmethod
    def load(cls, path: Path | None = None) -> AppConfig:
        """Load config from disk, returning defaults if file missing/corrupt."""
        p = Path(path) if path else _CONFIG_FILE
        if not p.exists():
            return cls()
        try:
            raw = _json.loads(p.read_text())
            known = {k: v for k, v in raw.items() if k in cls.__dataclass_fields__}
            return cls(**known)
        except Exception as e:
            _log.warning("Failed to load config from %s: %s — using defaults", p, e)
            return cls()
