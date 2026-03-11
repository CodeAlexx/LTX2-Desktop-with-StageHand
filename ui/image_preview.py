"""Static image preview widget for conditioning images."""

from __future__ import annotations

import logging
from pathlib import Path

import dearpygui.dearpygui as dpg
import numpy as np

try:
    from PIL import Image, ImageOps
except ImportError:  # pragma: no cover - optional at runtime
    Image = None
    ImageOps = None

logger = logging.getLogger(__name__)

MAX_DISPLAY_W = 768
MAX_DISPLAY_H = 480


class ImagePreview:
    """Display a single input image inside DearPyGui."""

    def __init__(self, parent_tag: int | str):
        self._parent = parent_tag
        self._texture_tag: int | None = None
        self._image_tag: int | None = None
        self._status_tag: int | None = None
        self._path_tag: int | None = None
        self._blank_frame = np.zeros((MAX_DISPLAY_H, MAX_DISPLAY_W, 4), dtype=np.float32).ravel().tolist()

        self._build_ui()

    def _build_ui(self) -> None:
        with dpg.group(parent=self._parent):
            dpg.add_text("Conditioning Image Preview")
            with dpg.texture_registry():
                self._texture_tag = dpg.add_dynamic_texture(
                    width=MAX_DISPLAY_W,
                    height=MAX_DISPLAY_H,
                    default_value=self._blank_frame,
                )
            self._image_tag = dpg.add_image(
                self._texture_tag,
                width=MAX_DISPLAY_W,
                height=MAX_DISPLAY_H,
            )
            self._status_tag = dpg.add_text("No image selected")
            self._path_tag = dpg.add_text("")

    def clear(self, message: str = "No image selected") -> None:
        if self._texture_tag is not None:
            dpg.set_value(self._texture_tag, self._blank_frame)
        if self._image_tag is not None:
            dpg.configure_item(self._image_tag, width=MAX_DISPLAY_W, height=MAX_DISPLAY_H)
        if self._status_tag is not None:
            dpg.set_value(self._status_tag, message)
        if self._path_tag is not None:
            dpg.set_value(self._path_tag, "")

    def load(self, image_path: str | None) -> None:
        if not image_path:
            self.clear()
            return

        if Image is None or ImageOps is None:
            self.clear("Pillow is not installed")
            return

        path = Path(image_path).expanduser()
        if not path.exists():
            self.clear(f"Missing file: {path}")
            return

        try:
            with Image.open(path) as img:
                img = ImageOps.exif_transpose(img).convert("RGBA")
                native_w, native_h = img.size
                scale = min(MAX_DISPLAY_W / native_w, MAX_DISPLAY_H / native_h, 1.0)
                display_w = max(1, int(native_w * scale))
                display_h = max(1, int(native_h * scale))
                display_w -= display_w % 2
                display_h -= display_h % 2
                display_w = max(2, display_w)
                display_h = max(2, display_h)
                if (display_w, display_h) != img.size:
                    resampling = getattr(getattr(Image, "Resampling", Image), "LANCZOS")
                    img = img.resize((display_w, display_h), resampling)
                rgba = np.asarray(img, dtype=np.float32) * (1.0 / 255.0)
        except Exception as exc:  # pragma: no cover - depends on user files
            logger.error("Failed to load preview image %s: %s", path, exc)
            self.clear(f"Failed to load image: {exc}")
            return

        if display_h < MAX_DISPLAY_H or display_w < MAX_DISPLAY_W:
            padded = np.zeros((MAX_DISPLAY_H, MAX_DISPLAY_W, 4), dtype=np.float32)
            padded[:display_h, :display_w, :] = rgba
            rgba = padded

        dpg.set_value(self._texture_tag, rgba.ravel().tolist())
        dpg.configure_item(self._image_tag, width=display_w, height=display_h)
        dpg.set_value(self._status_tag, f"{native_w}x{native_h} -> {display_w}x{display_h}")
        dpg.set_value(self._path_tag, str(path))
