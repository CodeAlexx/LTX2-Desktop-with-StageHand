"""Main application window — DearPyGui lifecycle."""

from __future__ import annotations

import dearpygui.dearpygui as dpg

from config import AppConfig
from inference_worker import InferenceWorker
from pipeline import LTX2Pipeline
from ui.generate_tab import GenerateTab
from ui.lora_tab import LoRATab
from ui.settings_tab import SettingsTab


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
        self.settings_tab = SettingsTab(config=self.config)

    def build(self) -> None:
        dpg.create_context()
        dpg.create_viewport(title="LTX-2.3 Video Generator", width=900, height=1100)

        with dpg.window(tag="primary_window"):
            dpg.add_text("LTX-2.3 Video Generator", color=(180, 220, 255))
            dpg.add_separator()

            with dpg.tab_bar() as tab_bar:
                self.generate_tab.build(tab_bar)
                self.lora_tab.build(tab_bar)
                self.settings_tab.build(tab_bar)

        dpg.set_primary_window("primary_window", True)
        dpg.setup_dearpygui()
        dpg.show_viewport()

    def run(self) -> None:
        """Main render loop with polling."""
        while dpg.is_dearpygui_running():
            self.generate_tab.poll()
            dpg.render_dearpygui_frame()

        dpg.destroy_context()
