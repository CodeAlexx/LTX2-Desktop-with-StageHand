"""Settings tab — model paths and output directory."""

from __future__ import annotations

import dearpygui.dearpygui as dpg

from config import AppConfig


class SettingsTab:
    """Panel for configuring model paths and output directory."""

    def __init__(self, config: AppConfig, on_change=None):
        self.config = config
        self._on_change = on_change
        self._distilled_ckpt_tag = None
        self._dev_ckpt_tag = None
        self._gemma_tag = None
        self._upsampler_tag = None
        self._temporal_upsampler_tag = None
        self._distilled_lora_tag = None
        self._output_dir_tag = None
        self._mode_tag = None
        self._distilled_lora_strength_tag = None
        self._nag_enabled_tag = None
        self._nag_scale_tag = None
        self._nag_alpha_tag = None
        self._nag_tau_tag = None
        self._ffn_chunks_tag = None
        self._spatial_tile_enabled_tag = None
        self._spatial_tile_pixels_tag = None
        self._spatial_tile_overlap_tag = None

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

            with dpg.group(horizontal=True):
                dpg.add_text("Distilled LoRA Strength")
                self._distilled_lora_strength_tag = dpg.add_slider_float(
                    default_value=self.config.distilled_lora_strength,
                    min_value=0.0, max_value=1.0, width=200,
                    format="%.2f",
                )
                dpg.add_text("  (0.25-0.3 recommended)", color=(180, 180, 180))

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

            dpg.add_text("Temporal Upsampler (optional, for 4-pass mode)")
            with dpg.group(horizontal=True):
                self._temporal_upsampler_tag = dpg.add_input_text(
                    default_value=self.config.temporal_upscaler_path, width=600,
                )
                dpg.add_button(label="Browse", callback=lambda: self._browse_file(self._temporal_upsampler_tag))

            dpg.add_separator()
            dpg.add_text("Output Directory")
            with dpg.group(horizontal=True):
                self._output_dir_tag = dpg.add_input_text(
                    default_value=self.config.output_dir, width=600,
                )
                dpg.add_button(label="Browse", callback=lambda: self._browse_dir(self._output_dir_tag))

            dpg.add_separator()
            dpg.add_text("NAG — Normalized Attention Guidance")
            dpg.add_text("  Used in all community workflows. Improves coherence.", color=(180, 180, 180))
            self._nag_enabled_tag = dpg.add_checkbox(
                label="NAG enabled", default_value=self.config.nag_enabled,
            )
            with dpg.group(horizontal=True):
                dpg.add_text("Scale")
                self._nag_scale_tag = dpg.add_slider_float(
                    default_value=self.config.nag_scale, min_value=0.0, max_value=20.0,
                    width=150, format="%.1f",
                )
                dpg.add_text("  Alpha")
                self._nag_alpha_tag = dpg.add_slider_float(
                    default_value=self.config.nag_alpha, min_value=0.0, max_value=1.0,
                    width=100, format="%.2f",
                )
                dpg.add_text("  Tau")
                self._nag_tau_tag = dpg.add_slider_float(
                    default_value=self.config.nag_tau, min_value=0.5, max_value=5.0,
                    width=100, format="%.1f",
                )

            dpg.add_separator()
            dpg.add_text("Advanced — VRAM Optimization")
            with dpg.group(horizontal=True):
                dpg.add_text("FFN Chunks")
                self._ffn_chunks_tag = dpg.add_input_int(
                    default_value=self.config.ffn_chunks, width=60, step=0,
                )
                dpg.add_text("  (1=off, 2=50% peak reduction, 4=75%)", color=(180, 180, 180))

            dpg.add_separator()
            dpg.add_text("Spatial Tiling (high-res denoising)")
            dpg.add_text("  Tiles the latent during denoising for higher resolutions.", color=(180, 180, 180))
            self._spatial_tile_enabled_tag = dpg.add_checkbox(
                label="Enable spatial tiling", default_value=self.config.spatial_tile_enabled,
            )
            with dpg.group(horizontal=True):
                dpg.add_text("Tile Size (px)")
                self._spatial_tile_pixels_tag = dpg.add_input_int(
                    default_value=self.config.spatial_tile_pixels, width=80, step=0,
                )
                dpg.add_text("  Overlap (px)")
                self._spatial_tile_overlap_tag = dpg.add_input_int(
                    default_value=self.config.spatial_tile_overlap, width=80, step=0,
                )

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
        self.config.distilled_lora_strength = dpg.get_value(self._distilled_lora_strength_tag)
        self.config.gemma_root = dpg.get_value(self._gemma_tag)
        self.config.spatial_upsampler_path = dpg.get_value(self._upsampler_tag)
        self.config.temporal_upscaler_path = dpg.get_value(self._temporal_upsampler_tag)
        self.config.output_dir = dpg.get_value(self._output_dir_tag)
        self.config.nag_enabled = dpg.get_value(self._nag_enabled_tag)
        self.config.nag_scale = dpg.get_value(self._nag_scale_tag)
        self.config.nag_alpha = dpg.get_value(self._nag_alpha_tag)
        self.config.nag_tau = dpg.get_value(self._nag_tau_tag)
        self.config.ffn_chunks = max(1, dpg.get_value(self._ffn_chunks_tag))
        self.config.spatial_tile_enabled = dpg.get_value(self._spatial_tile_enabled_tag)
        self.config.spatial_tile_pixels = max(32, dpg.get_value(self._spatial_tile_pixels_tag))
        self.config.spatial_tile_overlap = max(0, dpg.get_value(self._spatial_tile_overlap_tag))
        if self._on_change:
            self._on_change()
        self.config.save()
