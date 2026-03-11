#!/usr/bin/env python3
"""Run only Smoke Tests 4a/4b — NAG comparison."""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

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

from smoke_tests import run_smoke


def main():
    from config import AppConfig

    # 4a: NAG enabled
    cfg4a = AppConfig.load()
    cfg4a.pipeline_mode = "distilled"
    cfg4a.nag_enabled = True
    cfg4a.enhance_prompt = False
    cfg4a.use_four_pass = False

    r4a = run_smoke(
        "Smoke Test 4a: NAG enabled (scale=11, alpha=0.25, tau=2.5)",
        cfg4a,
        prompt="A serene mountain lake at golden hour, gentle ripples on the water",
        width=512, height=512, num_frames=25, fps=25, seed=42,
    )

    # 4b: NAG disabled
    cfg4b = AppConfig.load()
    cfg4b.pipeline_mode = "distilled"
    cfg4b.nag_enabled = False
    cfg4b.enhance_prompt = False
    cfg4b.use_four_pass = False

    r4b = run_smoke(
        "Smoke Test 4b: NAG disabled (baseline)",
        cfg4b,
        prompt="A serene mountain lake at golden hour, gentle ripples on the water",
        width=512, height=512, num_frames=25, fps=25, seed=42,
    )

    print(f"\n{'='*60}")
    print("  NAG COMPARISON REPORT")
    print(f"{'='*60}")
    print(f"  NAG ON:  {r4a.get('result')} | Peak: {r4a.get('peak_gb', 0):.2f} GB | Time: {r4a.get('elapsed_s', 0):.1f}s")
    print(f"  NAG OFF: {r4b.get('result')} | Peak: {r4b.get('peak_gb', 0):.2f} GB | Time: {r4b.get('elapsed_s', 0):.1f}s")
    if r4a.get("result") == "PASS" and r4b.get("result") == "PASS":
        print(f"\n  NAG ON output:  {r4a.get('output')}")
        print(f"  NAG OFF output: {r4b.get('output')}")
        print(f"  (Compare these files visually to confirm NAG effect)")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
