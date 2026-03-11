"""Local Serenity-style DearPyGui theme helpers for the LTX desktop app."""

from __future__ import annotations

import logging
import subprocess
from dataclasses import dataclass
from pathlib import Path

import dearpygui.dearpygui as dpg

log = logging.getLogger(__name__)

__all__ = [
    "THEME_NAMES",
    "create_theme",
    "create_start_button_theme",
    "create_stop_button_theme",
    "detect_screen_size",
    "get_ui_scale",
    "scaled",
    "setup_fonts",
]


def detect_screen_size() -> tuple[int, int]:
    """Detect primary display size, with sane fallbacks."""
    try:
        import tkinter as tk

        root = tk.Tk()
        root.withdraw()
        width = int(root.winfo_screenwidth())
        height = int(root.winfo_screenheight())
        root.destroy()
        if width > 0 and height > 0:
            return width, height
    except Exception:
        pass

    try:
        result = subprocess.run(
            ["xrandr", "--query"],
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
        for line in result.stdout.splitlines():
            if "*" not in line:
                continue
            res = line.split()[0]
            width_s, height_s = res.split("x", 1)
            return int(width_s), int(height_s)
    except Exception:
        pass

    return 1920, 1080


def _compute_auto_ui_scale(screen_height: int) -> float:
    if screen_height <= 1200:
        return 1.0
    if screen_height <= 1600:
        return 1.15
    return 1.4


def get_ui_scale(override: float = 0.0) -> float:
    if override > 0:
        return max(0.8, min(override, 3.0))
    _, screen_h = detect_screen_size()
    return _compute_auto_ui_scale(screen_h)


def scaled(value: int | float, ui_scale: float | None = None) -> int:
    scale = ui_scale if ui_scale is not None else get_ui_scale()
    return round(float(value) * scale)


_FONT_SEARCH_PATHS = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf",
    "/usr/share/fonts/truetype/ubuntu/Ubuntu-R.ttf",
    "/usr/share/fonts/TTF/DejaVuSans.ttf",
    "/usr/share/fonts/dejavu-sans-fonts/DejaVuSans.ttf",
]


def _find_font(paths: list[str]) -> str | None:
    for path in paths:
        if Path(path).exists():
            return path
    return None


def setup_fonts(ui_scale: float | None = None, size: int = 0) -> None:
    scale = ui_scale if ui_scale is not None else get_ui_scale()
    font_size = size or scaled(16, scale)
    font_path = _find_font(_FONT_SEARCH_PATHS)
    if font_path is None:
        log.warning("No TTF font found, using global font scale fallback")
        dpg.set_global_font_scale(max(1.0, scale))
        return

    with dpg.font_registry():
        font = dpg.add_font(font_path, font_size)
    dpg.bind_font(font)
    dpg.set_global_font_scale(1.0)


@dataclass(frozen=True)
class ThemePalette:
    name: str
    window_bg: tuple
    child_bg: tuple
    popup_bg: tuple
    border: tuple
    border_shadow: tuple
    text: tuple
    text_disabled: tuple
    text_selected_bg: tuple
    frame_bg: tuple
    frame_bg_hover: tuple
    frame_bg_active: tuple
    button: tuple
    button_hover: tuple
    button_active: tuple
    header: tuple
    header_hover: tuple
    header_active: tuple
    tab: tuple
    tab_hover: tuple
    tab_active: tuple
    tab_unfocused: tuple
    tab_unfocused_active: tuple
    scrollbar_bg: tuple
    scrollbar_grab: tuple
    scrollbar_hover: tuple
    scrollbar_active: tuple
    slider_grab: tuple
    slider_grab_active: tuple
    checkmark: tuple
    separator: tuple
    separator_hover: tuple
    separator_active: tuple
    resize_grip: tuple
    resize_grip_hover: tuple
    resize_grip_active: tuple
    title_bg: tuple
    title_bg_active: tuple
    title_bg_collapsed: tuple
    menubar_bg: tuple
    plot_bg: tuple
    plot_histogram: tuple
    modal_dim_bg: tuple
    window_rounding: float = 4.0
    child_rounding: float = 4.0
    frame_rounding: float = 4.0
    tab_rounding: float = 4.0
    grab_rounding: float = 3.0
    scrollbar_rounding: float = 6.0
    window_border_size: float = 1.0
    frame_border_size: float = 0.0


def _f(*rgba: float) -> tuple[int, int, int, int]:
    return tuple(int(round(c * 255)) for c in rgba)


_PALETTES: dict[str, ThemePalette] = {
    "Serenity": ThemePalette(
        name="Serenity",
        window_bg=(26, 26, 46, 255),
        child_bg=(22, 33, 62, 255),
        popup_bg=(30, 30, 52, 255),
        border=(42, 42, 74, 255),
        border_shadow=(15, 15, 30, 255),
        text=(232, 232, 232, 255),
        text_disabled=(102, 102, 102, 255),
        text_selected_bg=(67, 97, 238, 100),
        frame_bg=(22, 33, 62, 255),
        frame_bg_hover=(30, 42, 78, 255),
        frame_bg_active=(38, 52, 94, 255),
        button=(67, 97, 238, 255),
        button_hover=(90, 120, 240, 255),
        button_active=(52, 81, 222, 255),
        header=(15, 52, 96, 255),
        header_hover=(20, 65, 120, 255),
        header_active=(25, 78, 140, 255),
        tab=(30, 30, 52, 255),
        tab_hover=(50, 50, 80, 255),
        tab_active=(67, 97, 238, 255),
        tab_unfocused=(25, 25, 44, 255),
        tab_unfocused_active=(50, 70, 160, 255),
        scrollbar_bg=(18, 18, 36, 255),
        scrollbar_grab=(55, 55, 90, 255),
        scrollbar_hover=(70, 70, 110, 255),
        scrollbar_active=(85, 85, 130, 255),
        slider_grab=(67, 97, 238, 255),
        slider_grab_active=(90, 120, 240, 255),
        checkmark=(67, 97, 238, 255),
        separator=(42, 42, 74, 255),
        separator_hover=(90, 120, 240, 255),
        separator_active=(52, 81, 222, 255),
        resize_grip=(67, 97, 238, 65),
        resize_grip_hover=(67, 97, 238, 170),
        resize_grip_active=(67, 97, 238, 240),
        title_bg=(15, 15, 30, 255),
        title_bg_active=(26, 26, 46, 255),
        title_bg_collapsed=(20, 20, 40, 255),
        menubar_bg=(20, 20, 38, 255),
        plot_bg=(18, 18, 36, 255),
        plot_histogram=(67, 97, 238, 255),
        modal_dim_bg=(0, 0, 0, 140),
    ),
    "Moonlight": ThemePalette(
        name="Moonlight",
        window_bg=(20, 22, 26, 255),
        child_bg=(24, 26, 30, 255),
        popup_bg=(20, 22, 26, 255),
        border=(40, 43, 49, 255),
        border_shadow=(20, 22, 26, 255),
        text=(255, 255, 255, 255),
        text_disabled=(70, 81, 115, 255),
        text_selected_bg=(180, 180, 200, 100),
        frame_bg=(29, 32, 39, 255),
        frame_bg_hover=(40, 43, 49, 255),
        frame_bg_active=(40, 43, 49, 255),
        button=(30, 34, 38, 255),
        button_hover=(46, 48, 50, 255),
        button_active=(39, 39, 39, 255),
        header=(36, 42, 53, 255),
        header_hover=(27, 27, 27, 255),
        header_active=(20, 22, 26, 255),
        tab=(20, 22, 26, 255),
        tab_hover=(30, 34, 38, 255),
        tab_active=(30, 34, 38, 255),
        tab_unfocused=(20, 22, 26, 255),
        tab_unfocused_active=(32, 70, 146, 255),
        scrollbar_bg=(12, 14, 18, 255),
        scrollbar_grab=(30, 34, 38, 255),
        scrollbar_hover=(40, 43, 49, 255),
        scrollbar_active=(30, 34, 38, 255),
        slider_grab=(248, 255, 127, 255),
        slider_grab_active=(255, 203, 127, 255),
        checkmark=(248, 255, 127, 255),
        separator=(33, 38, 49, 255),
        separator_hover=(40, 47, 64, 255),
        separator_active=(40, 47, 64, 255),
        resize_grip=(37, 37, 37, 255),
        resize_grip_hover=(248, 255, 127, 255),
        resize_grip_active=(255, 255, 255, 255),
        title_bg=(12, 14, 18, 255),
        title_bg_active=(12, 14, 18, 255),
        title_bg_collapsed=(20, 22, 26, 255),
        menubar_bg=(25, 27, 31, 255),
        plot_bg=(12, 14, 18, 255),
        plot_histogram=(248, 255, 127, 255),
        modal_dim_bg=(50, 45, 139, 128),
        window_rounding=6.0,
        frame_rounding=6.0,
        tab_rounding=4.0,
        grab_rounding=6.0,
        window_border_size=0.0,
        frame_border_size=0.0,
    ),
    "Nord": ThemePalette(
        name="Nord",
        window_bg=_f(0.18, 0.20, 0.25, 1.00),
        child_bg=_f(0.16, 0.17, 0.20, 1.00),
        popup_bg=_f(0.23, 0.26, 0.32, 1.00),
        border=_f(0.14, 0.16, 0.19, 1.00),
        border_shadow=_f(0.09, 0.09, 0.09, 0.00),
        text=_f(0.85, 0.87, 0.91, 0.88),
        text_disabled=_f(0.49, 0.50, 0.53, 1.00),
        text_selected_bg=_f(0.37, 0.51, 0.67, 1.00),
        frame_bg=_f(0.23, 0.26, 0.32, 1.00),
        frame_bg_hover=_f(0.56, 0.74, 0.73, 1.00),
        frame_bg_active=_f(0.53, 0.75, 0.82, 1.00),
        button=_f(0.18, 0.20, 0.25, 1.00),
        button_hover=_f(0.51, 0.63, 0.76, 1.00),
        button_active=_f(0.37, 0.51, 0.67, 1.00),
        header=_f(0.51, 0.63, 0.76, 1.00),
        header_hover=_f(0.53, 0.75, 0.82, 1.00),
        header_active=_f(0.37, 0.51, 0.67, 1.00),
        tab=_f(0.18, 0.20, 0.25, 1.00),
        tab_hover=_f(0.22, 0.24, 0.31, 1.00),
        tab_active=_f(0.23, 0.26, 0.32, 1.00),
        tab_unfocused=_f(0.13, 0.15, 0.18, 1.00),
        tab_unfocused_active=_f(0.17, 0.19, 0.23, 1.00),
        scrollbar_bg=_f(0.18, 0.20, 0.25, 1.00),
        scrollbar_grab=_f(0.23, 0.26, 0.32, 0.60),
        scrollbar_hover=_f(0.23, 0.26, 0.32, 1.00),
        scrollbar_active=_f(0.23, 0.26, 0.32, 1.00),
        slider_grab=_f(0.51, 0.63, 0.76, 1.00),
        slider_grab_active=_f(0.37, 0.51, 0.67, 1.00),
        checkmark=_f(0.37, 0.51, 0.67, 1.00),
        separator=_f(0.14, 0.16, 0.19, 1.00),
        separator_hover=_f(0.56, 0.74, 0.73, 1.00),
        separator_active=_f(0.53, 0.75, 0.82, 1.00),
        resize_grip=_f(0.53, 0.75, 0.82, 0.86),
        resize_grip_hover=_f(0.61, 0.74, 0.87, 1.00),
        resize_grip_active=_f(0.37, 0.51, 0.67, 1.00),
        title_bg=_f(0.16, 0.16, 0.20, 1.00),
        title_bg_active=_f(0.16, 0.16, 0.20, 1.00),
        title_bg_collapsed=_f(0.16, 0.16, 0.20, 1.00),
        menubar_bg=_f(0.16, 0.16, 0.20, 1.00),
        plot_bg=_f(0.18, 0.20, 0.25, 1.00),
        plot_histogram=_f(0.56, 0.74, 0.73, 1.00),
        modal_dim_bg=_f(0.10, 0.10, 0.15, 0.60),
    ),
    "Blender": ThemePalette(
        name="Blender",
        window_bg=_f(0.22, 0.22, 0.22, 1.00),
        child_bg=_f(0.19, 0.19, 0.19, 1.00),
        popup_bg=_f(0.09, 0.09, 0.09, 1.00),
        border=_f(0.17, 0.17, 0.17, 1.00),
        border_shadow=_f(0.10, 0.10, 0.10, 0.00),
        text=_f(0.84, 0.84, 0.84, 1.00),
        text_disabled=_f(0.50, 0.50, 0.50, 1.00),
        text_selected_bg=_f(0.28, 0.45, 0.70, 1.00),
        frame_bg=_f(0.33, 0.33, 0.33, 1.00),
        frame_bg_hover=_f(0.47, 0.47, 0.47, 1.00),
        frame_bg_active=_f(0.16, 0.16, 0.16, 1.00),
        button=_f(0.33, 0.33, 0.33, 1.00),
        button_hover=_f(0.40, 0.40, 0.40, 1.00),
        button_active=_f(0.28, 0.45, 0.70, 1.00),
        header=_f(0.27, 0.27, 0.27, 1.00),
        header_hover=_f(0.28, 0.45, 0.70, 1.00),
        header_active=_f(0.27, 0.27, 0.27, 1.00),
        tab=_f(0.11, 0.11, 0.11, 1.00),
        tab_hover=_f(0.14, 0.14, 0.14, 1.00),
        tab_active=_f(0.19, 0.19, 0.19, 1.00),
        tab_unfocused=_f(0.11, 0.11, 0.11, 1.00),
        tab_unfocused_active=_f(0.14, 0.14, 0.14, 1.00),
        scrollbar_bg=_f(0.19, 0.19, 0.19, 1.00),
        scrollbar_grab=_f(0.33, 0.33, 0.33, 1.00),
        scrollbar_hover=_f(0.33, 0.33, 0.33, 1.00),
        scrollbar_active=_f(0.35, 0.35, 0.35, 1.00),
        slider_grab=_f(0.28, 0.45, 0.70, 1.00),
        slider_grab_active=_f(0.28, 0.45, 0.70, 1.00),
        checkmark=_f(0.28, 0.45, 0.70, 1.00),
        separator=_f(0.18, 0.18, 0.18, 1.00),
        separator_hover=_f(0.28, 0.45, 0.70, 1.00),
        separator_active=_f(0.28, 0.45, 0.70, 1.00),
        resize_grip=_f(0.54, 0.54, 0.54, 1.00),
        resize_grip_hover=_f(0.28, 0.45, 0.70, 1.00),
        resize_grip_active=_f(0.19, 0.39, 0.69, 1.00),
        title_bg=_f(0.11, 0.11, 0.11, 1.00),
        title_bg_active=_f(0.28, 0.45, 0.70, 1.00),
        title_bg_collapsed=_f(0.11, 0.11, 0.11, 1.00),
        menubar_bg=_f(0.11, 0.11, 0.11, 1.00),
        plot_bg=_f(0.19, 0.19, 0.19, 1.00),
        plot_histogram=_f(0.28, 0.45, 0.70, 1.00),
        modal_dim_bg=_f(0.10, 0.10, 0.10, 0.60),
        frame_rounding=3.0,
        window_border_size=0.0,
    ),
    "Cyberpunk": ThemePalette(
        name="Cyberpunk",
        window_bg=_f(0.00, 0.04, 0.12, 1.00),
        child_bg=_f(0.03, 0.04, 0.22, 1.00),
        popup_bg=_f(0.12, 0.06, 0.27, 1.00),
        border=_f(0.61, 0.00, 1.00, 1.00),
        border_shadow=_f(0.00, 0.00, 0.00, 0.00),
        text=_f(0.00, 0.82, 1.00, 1.00),
        text_disabled=_f(0.00, 0.36, 0.63, 1.00),
        text_selected_bg=_f(0.00, 0.82, 1.00, 0.40),
        frame_bg=_f(0.00, 0.75, 1.00, 0.20),
        frame_bg_hover=_f(0.34, 0.00, 1.00, 1.00),
        frame_bg_active=_f(0.08, 0.00, 1.00, 1.00),
        button=_f(0.00, 0.98, 1.00, 0.52),
        button_hover=_f(0.94, 0.00, 1.00, 0.80),
        button_active=_f(0.01, 0.00, 1.00, 1.00),
        header=_f(0.00, 0.95, 1.00, 0.40),
        header_hover=_f(0.94, 0.00, 1.00, 0.80),
        header_active=_f(0.01, 0.00, 1.00, 1.00),
        tab=_f(0.36, 0.00, 1.00, 1.00),
        tab_hover=_f(0.00, 0.92, 1.00, 0.80),
        tab_active=_f(0.62, 0.00, 0.80, 1.00),
        tab_unfocused=_f(0.10, 0.00, 0.30, 1.00),
        tab_unfocused_active=_f(0.30, 0.00, 0.60, 1.00),
        scrollbar_bg=_f(0.00, 0.88, 1.00, 1.00),
        scrollbar_grab=_f(0.61, 0.00, 1.00, 1.00),
        scrollbar_hover=_f(0.01, 0.00, 1.00, 1.00),
        scrollbar_active=_f(0.95, 0.19, 0.67, 1.00),
        slider_grab=_f(0.00, 1.00, 0.95, 1.00),
        slider_grab_active=_f(0.81, 0.00, 1.00, 1.00),
        checkmark=_f(0.95, 0.19, 0.92, 1.00),
        separator=_f(0.74, 0.00, 1.00, 0.50),
        separator_hover=_f(0.34, 0.00, 1.00, 0.78),
        separator_active=_f(0.00, 1.00, 0.85, 1.00),
        resize_grip=_f(0.61, 0.00, 1.00, 1.00),
        resize_grip_hover=_f(0.89, 0.26, 0.98, 0.67),
        resize_grip_active=_f(0.00, 0.88, 1.00, 0.95),
        title_bg=_f(0.00, 0.81, 0.95, 1.00),
        title_bg_active=_f(0.61, 0.00, 1.00, 1.00),
        title_bg_collapsed=_f(0.25, 0.00, 0.54, 0.81),
        menubar_bg=_f(0.61, 0.00, 1.00, 1.00),
        plot_bg=_f(0.00, 0.04, 0.12, 1.00),
        plot_histogram=_f(0.00, 1.00, 0.88, 1.00),
        modal_dim_bg=_f(0.05, 0.00, 0.20, 0.60),
        frame_rounding=2.0,
        window_rounding=3.0,
        frame_border_size=1.0,
        window_border_size=1.0,
    ),
}

THEME_NAMES = list(_PALETTES.keys())
_theme_cache: dict[tuple[str, int], int] = {}


def _build_theme(palette: ThemePalette, ui_scale: float) -> int:
    with dpg.theme() as theme:
        with dpg.theme_component(dpg.mvAll):
            dpg.add_theme_color(dpg.mvThemeCol_WindowBg, palette.window_bg)
            dpg.add_theme_color(dpg.mvThemeCol_ChildBg, palette.child_bg)
            dpg.add_theme_color(dpg.mvThemeCol_PopupBg, palette.popup_bg)
            dpg.add_theme_color(dpg.mvThemeCol_Border, palette.border)
            dpg.add_theme_color(dpg.mvThemeCol_BorderShadow, palette.border_shadow)
            dpg.add_theme_color(dpg.mvThemeCol_Text, palette.text)
            dpg.add_theme_color(dpg.mvThemeCol_TextDisabled, palette.text_disabled)
            dpg.add_theme_color(dpg.mvThemeCol_TextSelectedBg, palette.text_selected_bg)
            dpg.add_theme_color(dpg.mvThemeCol_FrameBg, palette.frame_bg)
            dpg.add_theme_color(dpg.mvThemeCol_FrameBgHovered, palette.frame_bg_hover)
            dpg.add_theme_color(dpg.mvThemeCol_FrameBgActive, palette.frame_bg_active)
            dpg.add_theme_color(dpg.mvThemeCol_Button, palette.button)
            dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, palette.button_hover)
            dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, palette.button_active)
            dpg.add_theme_color(dpg.mvThemeCol_Header, palette.header)
            dpg.add_theme_color(dpg.mvThemeCol_HeaderHovered, palette.header_hover)
            dpg.add_theme_color(dpg.mvThemeCol_HeaderActive, palette.header_active)
            dpg.add_theme_color(dpg.mvThemeCol_Tab, palette.tab)
            dpg.add_theme_color(dpg.mvThemeCol_TabHovered, palette.tab_hover)
            dpg.add_theme_color(dpg.mvThemeCol_TabActive, palette.tab_active)
            dpg.add_theme_color(dpg.mvThemeCol_TabUnfocused, palette.tab_unfocused)
            dpg.add_theme_color(dpg.mvThemeCol_TabUnfocusedActive, palette.tab_unfocused_active)
            dpg.add_theme_color(dpg.mvThemeCol_ScrollbarBg, palette.scrollbar_bg)
            dpg.add_theme_color(dpg.mvThemeCol_ScrollbarGrab, palette.scrollbar_grab)
            dpg.add_theme_color(dpg.mvThemeCol_ScrollbarGrabHovered, palette.scrollbar_hover)
            dpg.add_theme_color(dpg.mvThemeCol_ScrollbarGrabActive, palette.scrollbar_active)
            dpg.add_theme_color(dpg.mvThemeCol_SliderGrab, palette.slider_grab)
            dpg.add_theme_color(dpg.mvThemeCol_SliderGrabActive, palette.slider_grab_active)
            dpg.add_theme_color(dpg.mvThemeCol_CheckMark, palette.checkmark)
            dpg.add_theme_color(dpg.mvThemeCol_Separator, palette.separator)
            dpg.add_theme_color(dpg.mvThemeCol_SeparatorHovered, palette.separator_hover)
            dpg.add_theme_color(dpg.mvThemeCol_SeparatorActive, palette.separator_active)
            dpg.add_theme_color(dpg.mvThemeCol_ResizeGrip, palette.resize_grip)
            dpg.add_theme_color(dpg.mvThemeCol_ResizeGripHovered, palette.resize_grip_hover)
            dpg.add_theme_color(dpg.mvThemeCol_ResizeGripActive, palette.resize_grip_active)
            dpg.add_theme_color(dpg.mvThemeCol_TitleBg, palette.title_bg)
            dpg.add_theme_color(dpg.mvThemeCol_TitleBgActive, palette.title_bg_active)
            dpg.add_theme_color(dpg.mvThemeCol_TitleBgCollapsed, palette.title_bg_collapsed)
            dpg.add_theme_color(dpg.mvThemeCol_MenuBarBg, palette.menubar_bg)
            dpg.add_theme_color(dpg.mvThemeCol_PlotHistogram, palette.plot_histogram)
            if hasattr(dpg, "mvThemeCol_ModalWindowDimBg"):
                dpg.add_theme_color(dpg.mvThemeCol_ModalWindowDimBg, palette.modal_dim_bg)
            if hasattr(dpg, "mvThemeCol_PlotBg"):
                dpg.add_theme_color(dpg.mvThemeCol_PlotBg, palette.plot_bg)

            s = lambda value: scaled(value, ui_scale)
            dpg.add_theme_style(dpg.mvStyleVar_WindowPadding, s(10), s(10))
            dpg.add_theme_style(dpg.mvStyleVar_FramePadding, s(6), s(4))
            dpg.add_theme_style(dpg.mvStyleVar_ItemSpacing, s(8), s(6))
            dpg.add_theme_style(dpg.mvStyleVar_ItemInnerSpacing, s(6), s(4))
            dpg.add_theme_style(dpg.mvStyleVar_ScrollbarSize, s(14))
            dpg.add_theme_style(dpg.mvStyleVar_ScrollbarRounding, s(palette.scrollbar_rounding))
            dpg.add_theme_style(dpg.mvStyleVar_GrabMinSize, s(12))
            dpg.add_theme_style(dpg.mvStyleVar_GrabRounding, s(palette.grab_rounding))
            dpg.add_theme_style(dpg.mvStyleVar_WindowRounding, s(palette.window_rounding))
            dpg.add_theme_style(dpg.mvStyleVar_ChildRounding, s(palette.child_rounding))
            dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, s(palette.frame_rounding))
            dpg.add_theme_style(dpg.mvStyleVar_TabRounding, s(palette.tab_rounding))
            dpg.add_theme_style(dpg.mvStyleVar_WindowBorderSize, palette.window_border_size)
            dpg.add_theme_style(dpg.mvStyleVar_FrameBorderSize, palette.frame_border_size)
            dpg.add_theme_style(dpg.mvStyleVar_IndentSpacing, s(20))
            dpg.add_theme_style(dpg.mvStyleVar_PopupRounding, s(palette.frame_rounding))
    return theme


def create_theme(name: str = "Serenity", ui_scale: float | None = None) -> int:
    scale = ui_scale if ui_scale is not None else get_ui_scale()
    theme_name = name if name in _PALETTES else "Serenity"
    cache_key = (theme_name, round(scale * 100))
    if cache_key not in _theme_cache:
        _theme_cache[cache_key] = _build_theme(_PALETTES[theme_name], scale)
    dpg.bind_theme(_theme_cache[cache_key])
    return _theme_cache[cache_key]


_GREEN = (30, 160, 95)
_GREEN_HI = (25, 135, 80)
_GREEN_ACT = (20, 110, 65)
_RED = (210, 55, 65)
_RED_HI = (180, 45, 55)
_RED_ACT = (155, 38, 48)


def create_start_button_theme() -> int:
    with dpg.theme() as theme:
        with dpg.theme_component(dpg.mvButton):
            dpg.add_theme_color(dpg.mvThemeCol_Button, _GREEN)
            dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, _GREEN_HI)
            dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, _GREEN_ACT)
    return theme


def create_stop_button_theme() -> int:
    with dpg.theme() as theme:
        with dpg.theme_component(dpg.mvButton):
            dpg.add_theme_color(dpg.mvThemeCol_Button, _RED)
            dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, _RED_HI)
            dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, _RED_ACT)
    return theme
