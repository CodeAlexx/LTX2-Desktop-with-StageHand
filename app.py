"""Main application window — DearPyGui lifecycle."""

from __future__ import annotations

import math

import dearpygui.dearpygui as dpg

from config import AppConfig
from inference_worker import InferenceWorker
from pipeline import LTX2Pipeline
from ui.generate_tab import GenerateTab
from ui.lora_tab import LoRATab
from ui.settings_tab import SettingsTab
from ui.theme import create_theme, detect_screen_size, get_ui_scale, setup_fonts


def _compute_viewport_size(screen_w: int, screen_h: int, scale: float) -> tuple[int, int]:
    """Scale the default viewport up on high-resolution displays."""
    width = max(1360, int(math.floor(1360 * scale)))
    height = max(1100, int(math.floor(1100 * min(scale, 1.35))))
    width = min(width, max(1024, screen_w - 120))
    height = min(height, max(820, screen_h - 120))
    return width, height


class LTX2App:
    """Top-level application managing the DearPyGui window."""

    def __init__(self, config: AppConfig | None = None):
        self.config = config or AppConfig()
        self.worker = InferenceWorker()
        self.generate_tab = GenerateTab(
            config=self.config,
            worker=self.worker,
            pipeline_factory=lambda: LTX2Pipeline(self.config),
        )
        self.lora_tab = LoRATab(
            config=self.config,
            on_change=self.generate_tab.invalidate_pipeline,
        )
        self.settings_tab = SettingsTab(config=self.config, on_change=self._on_settings_changed)

    def _apply_appearance(self) -> None:
        ui_scale = get_ui_scale(self.config.ui_scale)
        create_theme(self.config.theme_name, ui_scale=ui_scale)
        setup_fonts(ui_scale=ui_scale)

    def _on_settings_changed(self) -> None:
        self.generate_tab.invalidate_pipeline()
        try:
            self._apply_appearance()
        except Exception:
            pass

    def build(self) -> None:
        dpg.create_context()
        screen_w, screen_h = detect_screen_size()
        ui_scale = get_ui_scale(self.config.ui_scale)
        viewport_w, viewport_h = _compute_viewport_size(screen_w, screen_h, ui_scale)

        dpg.create_viewport(title="LTX-2.3 Video Generator", width=viewport_w, height=viewport_h)

        with dpg.window(tag="primary_window"):
            dpg.add_text("LTX-2.3 Video Generator", color=(180, 220, 255))
            dpg.add_separator()

            with dpg.tab_bar() as tab_bar:
                self.generate_tab.build(tab_bar)
                self.lora_tab.build(tab_bar)
                self.settings_tab.build(tab_bar)

        dpg.set_primary_window("primary_window", True)
        dpg.setup_dearpygui()
        self._apply_appearance()
        dpg.show_viewport()

    def run(self) -> None:
        """Main render loop with polling."""
        while dpg.is_dearpygui_running():
            self.generate_tab.poll()
            dpg.render_dearpygui_frame()

        dpg.destroy_context()


if __name__ == "__main__":
    app = LTX2App(AppConfig.load())
    app.build()
    app.run()
