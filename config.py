"""Application configuration."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class AppConfig:
    """Central configuration for the LTX-2 desktop app."""

    # --- Model paths (update these to your local model locations) ---
    distilled_checkpoint_path: str = "./models/ltx-2.3-22b-distilled.safetensors"
    dev_checkpoint_path: str = "./models/ltx-2.3-22b-dev-fp8.safetensors"
    gemma_root: str = "./models/gemma-3-12b-it"
    distilled_lora_path: str = "./models/ltx-2.3-22b-distilled-lora-384.safetensors"
    spatial_upsampler_path: str = "./models/ltx-2.3-spatial-upscaler-x2-1.0.safetensors"

    # Output
    output_dir: str = "./output"

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

    # LoRA
    lora_paths: list[str] = field(default_factory=list)
    lora_strengths: list[float] = field(default_factory=list)

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
