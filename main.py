#!/usr/bin/env python3
"""LTX-2 Video Generator — standalone DearPyGui desktop app.

Usage:
    python main.py

Requires ltx_core and ltx_pipelines on sys.path.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

# Add ltx packages to path
import os
_ltx_packages = Path(os.environ.get("LTX_PACKAGES", "/home/alex/LTX-2/packages"))
for pkg in ("ltx-core/src", "ltx-pipelines/src"):
    p = str(_ltx_packages / pkg)
    if p not in sys.path:
        sys.path.insert(0, p)

# Add app directory to path
_app_dir = str(Path(__file__).parent)
if _app_dir not in sys.path:
    sys.path.insert(0, _app_dir)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


def main() -> None:
    from app import LTX2App
    from config import AppConfig

    config = AppConfig.load()
    app = LTX2App(config)
    app.build()
    app.run()


if __name__ == "__main__":
    main()
