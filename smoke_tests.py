#!/usr/bin/env python3
"""Phase 7 — Full validation smoke tests on real hardware.

Run: python smoke_tests.py

Executes 4 smoke tests and reports VRAM peaks + timing.
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

import torch


def _reset_vram():
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def _peak_gb() -> float:
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / 1e9
    return 0.0


def run_smoke(label: str, cfg, prompt: str, **gen_kwargs) -> dict:
    from pipeline import LTX2Pipeline

    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")

    _reset_vram()
    p = LTX2Pipeline(cfg)
    t0 = time.perf_counter()

    try:
        out = p.generate(
            prompt=prompt,
            progress_cb=lambda phase, frac: print(f"  [{frac:5.1%}] {phase}"),
            **gen_kwargs,
        )
        elapsed = time.perf_counter() - t0
        peak = _peak_gb()
        print(f"\n  Result: PASS")
        print(f"  Peak VRAM: {peak:.2f} GB")
        print(f"  Wall clock: {elapsed:.1f}s ({elapsed/60:.1f}min)")
        print(f"  Output: {out}")
        if peak > 23.5:
            print(f"  *** WARNING: Peak {peak:.2f} GB exceeds 23.5 GB ***")
        del p
        return {"result": "PASS", "peak_gb": peak, "elapsed_s": elapsed, "output": str(out)}
    except Exception as e:
        elapsed = time.perf_counter() - t0
        peak = _peak_gb()
        print(f"\n  Result: FAIL")
        print(f"  Error: {e}")
        print(f"  Peak VRAM at failure: {peak:.2f} GB")
        del p
        return {"result": "FAIL", "peak_gb": peak, "elapsed_s": elapsed, "error": str(e)}


def main():
    from config import AppConfig

    results = {}

    # ==========================================
    # Smoke Test 1: T2V distilled 768x512 97f
    # ==========================================
    cfg1 = AppConfig.load()
    cfg1.pipeline_mode = "distilled"
    cfg1.nag_enabled = True
    cfg1.enhance_prompt = False
    cfg1.use_four_pass = False

    results["smoke1_t2v_distilled"] = run_smoke(
        "Smoke Test 1: T2V distilled 768x512 97f",
        cfg1,
        prompt="A serene mountain lake at golden hour, gentle ripples on the water",
        width=768, height=512, num_frames=97, fps=25, seed=42,
    )

    # ==========================================
    # Smoke Test 2: Dev mode (CFG/STG) 512x512 25f
    # ==========================================
    cfg2 = AppConfig.load()
    cfg2.pipeline_mode = "dev"
    cfg2.nag_enabled = True
    cfg2.enhance_prompt = False
    cfg2.use_four_pass = False

    results["smoke2_dev_mode"] = run_smoke(
        "Smoke Test 2: Dev mode (CFG/STG) 512x512 25f",
        cfg2,
        prompt="A cat sitting by a window, sunlight streaming in",
        width=512, height=512, num_frames=25, fps=25, seed=42,
        negative_prompt="blurry, low quality",
        num_inference_steps=20,
        video_cfg_scale=3.0,
        video_stg_scale=1.0,
    )

    # ==========================================
    # Smoke Test 3: I2V distilled 768x512 97f
    # ==========================================
    try:
        from PIL import Image
        test_img = Path("/tmp/ltx2_smoke_i2v.png")
        img = Image.new("RGB", (768, 512), color=(120, 180, 220))
        img.save(str(test_img))

        cfg3 = AppConfig.load()
        cfg3.pipeline_mode = "distilled"
        cfg3.nag_enabled = True
        cfg3.enhance_prompt = False
        cfg3.use_four_pass = False

        results["smoke3_i2v"] = run_smoke(
            "Smoke Test 3: I2V distilled 768x512 97f",
            cfg3,
            prompt="The scene comes to life, gentle camera pan right across the landscape",
            width=768, height=512, num_frames=97, fps=25, seed=42,
            image_path=str(test_img),
            image_strength=0.9,
        )
    except ImportError:
        print("\n  Skipping I2V test — PIL not available")
        results["smoke3_i2v"] = {"result": "SKIP", "error": "PIL not available"}

    # ==========================================
    # Smoke Test 4: NAG comparison (same seed, NAG on vs off)
    # ==========================================
    cfg4a = AppConfig.load()
    cfg4a.pipeline_mode = "distilled"
    cfg4a.nag_enabled = True
    cfg4a.enhance_prompt = False
    cfg4a.use_four_pass = False

    results["smoke4_nag_on"] = run_smoke(
        "Smoke Test 4a: NAG enabled (scale=11, alpha=0.25, tau=2.5)",
        cfg4a,
        prompt="A serene mountain lake at golden hour, gentle ripples on the water",
        width=512, height=512, num_frames=25, fps=25, seed=42,
    )

    cfg4b = AppConfig.load()
    cfg4b.pipeline_mode = "distilled"
    cfg4b.nag_enabled = False
    cfg4b.enhance_prompt = False
    cfg4b.use_four_pass = False

    results["smoke4_nag_off"] = run_smoke(
        "Smoke Test 4b: NAG disabled (baseline)",
        cfg4b,
        prompt="A serene mountain lake at golden hour, gentle ripples on the water",
        width=512, height=512, num_frames=25, fps=25, seed=42,
    )

    # ==========================================
    # Final Report
    # ==========================================
    print("\n" + "="*60)
    print("  SMOKE TEST REPORT")
    print("="*60)

    for name, r in results.items():
        res = r.get("result", "?")
        peak = r.get("peak_gb", 0)
        elapsed = r.get("elapsed_s", 0)
        err = r.get("error", "")
        print(f"\n  {name}:")
        print(f"    Result: {res}")
        if peak:
            print(f"    VRAM peak: {peak:.2f} GB")
        if elapsed:
            print(f"    Wall clock: {elapsed:.1f}s ({elapsed/60:.1f}min)")
        if err:
            print(f"    Error: {err}")

    # NAG comparison note
    nag_on = results.get("smoke4_nag_on", {})
    nag_off = results.get("smoke4_nag_off", {})
    if nag_on.get("result") == "PASS" and nag_off.get("result") == "PASS":
        print(f"\n  NAG comparison:")
        print(f"    NAG ON:  {nag_on.get('output', '?')}")
        print(f"    NAG OFF: {nag_off.get('output', '?')}")
        print(f"    (Compare these files visually to confirm NAG effect)")

    all_pass = all(r.get("result") in ("PASS", "SKIP") for r in results.values())
    print(f"\n  Overall: {'ALL PASS' if all_pass else 'FAILURES DETECTED'}")
    print("="*60)


if __name__ == "__main__":
    main()
