"""Settings tab — model paths and output directory."""

from __future__ import annotations

import dearpygui.dearpygui as dpg

from config import AppConfig


class SettingsTab:
    """Panel for configuring model paths and output directory."""

    def __init__(self, config: AppConfig):
        self.config = config
        self._distilled_ckpt_tag = None
        self._dev_ckpt_tag = None
        self._gemma_tag = None
        self._upsampler_tag = None
        self._distilled_lora_tag = None
        self._output_dir_tag = None
        self._mode_tag = None

    def build(self, parent: int | str) -> None:
        with dpg.tab(label="Settings", parent=parent):
            dpg.add_text("Pipeline Mode")
            self._mode_tag = dpg.add_combo(
                items=["distilled", "dev"],
                default_value=self.config.pipeline_mode,
                width=140,
            )

            dpg.add_separator()
            dpg.add_text("Model Paths")
            dpg.add_separator()

            dpg.add_text("Distilled Checkpoint (.safetensors)")
            with dpg.group(horizontal=True):
                self._distilled_ckpt_tag = dpg.add_input_text(
                    default_value=self.config.distilled_checkpoint_path, width=600,
                )
                dpg.add_button(label="Browse", callback=lambda: self._browse_file(self._distilled_ckpt_tag))

            dpg.add_text("Dev Checkpoint (.safetensors)")
            with dpg.group(horizontal=True):
                self._dev_ckpt_tag = dpg.add_input_text(
                    default_value=self.config.dev_checkpoint_path, width=600,
                )
                dpg.add_button(label="Browse", callback=lambda: self._browse_file(self._dev_ckpt_tag))

            dpg.add_text("Distilled LoRA (for dev mode stage 2)")
            with dpg.group(horizontal=True):
                self._distilled_lora_tag = dpg.add_input_text(
                    default_value=self.config.distilled_lora_path, width=600,
                )
                dpg.add_button(label="Browse", callback=lambda: self._browse_file(self._distilled_lora_tag))

            dpg.add_text("Gemma Root")
            with dpg.group(horizontal=True):
                self._gemma_tag = dpg.add_input_text(
                    default_value=self.config.gemma_root, width=600,
                )
                dpg.add_button(label="Browse", callback=lambda: self._browse_dir(self._gemma_tag))

            dpg.add_text("Spatial Upsampler")
            with dpg.group(horizontal=True):
                self._upsampler_tag = dpg.add_input_text(
                    default_value=self.config.spatial_upsampler_path, width=600,
                )
                dpg.add_button(label="Browse", callback=lambda: self._browse_file(self._upsampler_tag))

            dpg.add_separator()
            dpg.add_text("Output Directory")
            with dpg.group(horizontal=True):
                self._output_dir_tag = dpg.add_input_text(
                    default_value=self.config.output_dir, width=600,
                )
                dpg.add_button(label="Browse", callback=lambda: self._browse_dir(self._output_dir_tag))

            dpg.add_separator()
            dpg.add_button(label="Save Settings", callback=self._save)

    def _browse_file(self, target_tag: int) -> None:
        def _selected(sender, app_data):
            selections = app_data.get("selections", {})
            if selections:
                dpg.set_value(target_tag, list(selections.values())[0])

        with dpg.file_dialog(callback=_selected, width=700, height=400, show=True):
            dpg.add_file_extension(".safetensors", color=(0, 255, 0, 255))

    def _browse_dir(self, target_tag: int) -> None:
        def _selected(sender, app_data):
            path = app_data.get("file_path_name", "")
            if path:
                dpg.set_value(target_tag, path)

        dpg.add_file_dialog(
            callback=_selected,
            width=700, height=400,
            directory_selector=True,
            show=True,
        )

    def _save(self) -> None:
        self.config.pipeline_mode = dpg.get_value(self._mode_tag)
        self.config.distilled_checkpoint_path = dpg.get_value(self._distilled_ckpt_tag)
        self.config.dev_checkpoint_path = dpg.get_value(self._dev_ckpt_tag)
        self.config.distilled_lora_path = dpg.get_value(self._distilled_lora_tag)
        self.config.gemma_root = dpg.get_value(self._gemma_tag)
        self.config.spatial_upsampler_path = dpg.get_value(self._upsampler_tag)
        self.config.output_dir = dpg.get_value(self._output_dir_tag)
