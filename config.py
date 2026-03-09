"""Application configuration."""

from __future__ import annotations

import json as _json
import logging
from dataclasses import dataclass, field
from pathlib import Path

_log = logging.getLogger(__name__)
_CONFIG_FILE = Path.home() / ".config" / "ltx2_desktop" / "config.json"


@dataclass
class AppConfig:
    """Central configuration for the LTX-2 desktop app."""

    # --- Model paths (update these to your local model locations) ---
    distilled_checkpoint_path: str = "/home/alex/EriDiffusion/Models/ltx2/ltx-2.3-22b-distilled.safetensors"
    dev_checkpoint_path: str = "/home/alex/EriDiffusion/Models/ltx2/ltx-2.3-22b-dev-fp8.safetensors"
    gemma_root: str = "/home/alex/models/gemma-3-12b-it"
    distilled_lora_path: str = "/home/alex/EriDiffusion/Models/ltx2/ltx-2.3-22b-distilled-lora-384.safetensors"
    distilled_lora_strength: float = 0.25  # 0.25-0.3 recommended; 1.0 over-sharpens
    spatial_upsampler_path: str = "/home/alex/EriDiffusion/Models/ltx2/ltx-2.3-spatial-upscaler-x2-1.0.safetensors"
    temporal_upscaler_path: str = ""

    # Output
    output_dir: str = "/home/alex/serenity/output/ltx2"

    # --- Pipeline mode ---
    pipeline_mode: str = "distilled"

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
    long_video_adain_factor: float = 0.3
    long_video_per_step_adain: bool = False
    long_video_per_step_adain_factors: str = "0.9,0.75,0.5,0.25,0.0,0.0"
    long_video_long_memory_strength: float = 0.0

    # NAG — Normalized Attention Guidance
    # Community default: scale=11, alpha=0.25, tau=2.5
    nag_enabled: bool = True
    nag_scale: float = 11.0
    nag_alpha: float = 0.25
    nag_tau: float = 2.5

    # LoRA
    lora_paths: list[str] = field(default_factory=lambda: ["/home/alex/serenity/output/ltx2_eri2_lora/lora_last.safetensors"])
    lora_strengths: list[float] = field(default_factory=lambda: [1.0])

    @property
    def is_distilled(self) -> bool:
        return self.pipeline_mode == "distilled"

    @property
    def checkpoint_path(self) -> str:
        return self.distilled_checkpoint_path if self.is_distilled else self.dev_checkpoint_path

    def ensure_output_dir(self) -> Path:
        p = Path(self.output_dir)
        p.mkdir(parents=True, exist_ok=True)
        return p

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
