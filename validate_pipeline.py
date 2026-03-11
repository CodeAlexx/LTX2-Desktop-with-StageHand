#!/usr/bin/env python3
"""Hardware validation for LTX2-Desktop + Stagehand.

Run: python validate_pipeline.py

Generates short test videos and reports VRAM peaks at every checkpoint.
Validates both distilled BF16 and FP8 paths if checkpoints exist.
"""

from __future__ import annotations

import logging
import os
import sys
import time
from pathlib import Path

# Add ltx packages to path
_ltx_packages = Path(os.environ.get("LTX_PACKAGES", "/home/alex/LTX-2/packages"))
for pkg in ("ltx-core/src", "ltx-pipelines/src"):
    p = str(_ltx_packages / pkg)
    if p not in sys.path:
        sys.path.insert(0, p)

_app_dir = str(Path(__file__).parent)
if _app_dir not in sys.path:
    sys.path.insert(0, _app_dir)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def _vram_mb() -> float:
    """Current VRAM usage in MB."""
    import torch
    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        return (total - free) / 1e6
    return 0.0


def run_test(label: str, cfg, prompt: str, **gen_kwargs) -> None:
    """Run a single generation test and report VRAM peak."""
    from pipeline import LTX2Pipeline

    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")

    import torch
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    p = LTX2Pipeline(cfg)
    t0 = time.perf_counter()

    out = p.generate(
        prompt=prompt,
        progress_cb=lambda phase, frac: print(f"  [{frac:5.1%}] {phase}"),
        **gen_kwargs,
    )

    elapsed = time.perf_counter() - t0

    if torch.cuda.is_available():
        peak = torch.cuda.max_memory_allocated() / 1e9
        current = _vram_mb() / 1e3
        print(f"\n  Peak VRAM: {peak:.2f} GB")
        print(f"  Current VRAM: {current:.2f} GB")
        if peak > 23.5:
            print(f"  *** WARNING: Peak VRAM {peak:.2f} GB exceeds 23.5 GB safety limit ***")
    else:
        print("\n  (No CUDA device — VRAM stats unavailable)")

    print(f"  Elapsed: {elapsed:.1f}s")
    print(f"  Output: {out}")
    del p


def main() -> None:
    from config import AppConfig

    cfg = AppConfig.load()

    # Show current model paths
    print("Model paths:")
    print(f"  Distilled BF16: {cfg.distilled_checkpoint_path}")
    print(f"  Dev FP8:        {cfg.dev_checkpoint_path}")
    print(f"  Gemma root:     {cfg.gemma_root}")
    print(f"  Spatial up:     {cfg.spatial_upsampler_path}")
    print(f"  Temporal up:    {cfg.temporal_upscaler_path or '(not set)'}")
    print()

    # Validate paths exist
    missing = []
    for name, path in [
        ("distilled_checkpoint", cfg.distilled_checkpoint_path),
        ("gemma_root", cfg.gemma_root),
        ("spatial_upsampler", cfg.spatial_upsampler_path),
    ]:
        if not Path(path).exists():
            missing.append(f"  {name}: {path}")
    if missing:
        print("Missing required files:")
        for m in missing:
            print(m)
        print("\nUpdate config paths before running validation.")
        sys.exit(1)

    # ==========================================
    # Test 1: T2V distilled (small, fast)
    # ==========================================
    cfg.pipeline_mode = "distilled"
    cfg.nag_enabled = True
    cfg.enhance_prompt = False
    cfg.use_four_pass = False

    run_test(
        "Test 1: T2V distilled 512x512 25f (two-stage)",
        cfg,
        prompt="A serene lake at sunrise, gentle ripples on the water surface",
        width=512, height=512, num_frames=25, fps=25, seed=42,
    )

    # ==========================================
    # Test 2: FP8 distilled (if checkpoint exists)
    # ==========================================
    fp8_distilled = Path(cfg.distilled_checkpoint_path.replace(".safetensors", "-fp8.safetensors"))
    # Also check without the double suffix
    if not fp8_distilled.exists():
        fp8_distilled = Path(str(cfg.distilled_checkpoint_path).replace(
            "ltx-2.3-22b-distilled.safetensors",
            "ltx-2.3-22b-distilled-fp8.safetensors",
        ))

    if fp8_distilled.exists():
        cfg_fp8 = AppConfig.load()
        cfg_fp8.distilled_checkpoint_path = str(fp8_distilled)
        cfg_fp8.pipeline_mode = "distilled"
        cfg_fp8.nag_enabled = True
        cfg_fp8.enhance_prompt = False
        cfg_fp8.use_four_pass = False

        run_test(
            "Test 2: T2V distilled FP8 512x512 25f (two-stage)",
            cfg_fp8,
            prompt="A serene lake at sunrise, gentle ripples on the water surface",
            width=512, height=512, num_frames=25, fps=25, seed=42,
        )
    else:
        print(f"\n  Skipping FP8 test — {fp8_distilled} not found")

    # ==========================================
    # Test 3: I2V distilled (if PIL available)
    # ==========================================
    try:
        from PIL import Image
        img = Image.new("RGB", (512, 512), color=(100, 150, 200))
        test_img_path = "/tmp/ltx2_validate_i2v.png"
        img.save(test_img_path)

        cfg_i2v = AppConfig.load()
        cfg_i2v.pipeline_mode = "distilled"
        cfg_i2v.nag_enabled = True
        cfg_i2v.enhance_prompt = False
        cfg_i2v.use_four_pass = False

        run_test(
            "Test 3: I2V distilled 512x512 25f (two-stage)",
            cfg_i2v,
            prompt="The scene comes to life, gentle camera pan right",
            width=512, height=512, num_frames=25, fps=25, seed=42,
            image_path=test_img_path,
        )
    except ImportError:
        print("\n  Skipping I2V test — PIL not available")

    print("\n" + "="*60)
    print("  Validation complete. Check VRAM logs above for peaks.")
    print("="*60)


if __name__ == "__main__":
    main()
