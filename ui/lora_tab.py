"""LoRA management tab."""

from __future__ import annotations

import dearpygui.dearpygui as dpg

from config import AppConfig


class LoRATab:
    """Panel for managing LoRA paths and strengths."""

    def __init__(self, config: AppConfig, on_change: callable = None):
        self.config = config
        self._on_change = on_change
        self._list_tag = None
        self._rows: list[tuple[int, int]] = []  # (path_tag, strength_tag)

    def build(self, parent: int | str) -> None:
        with dpg.tab(label="LoRA", parent=parent):
            dpg.add_text("LoRA Adapters")
            dpg.add_separator()
            dpg.add_button(label="Add LoRA", callback=self._add_row)
            dpg.add_separator()
            self._list_tag = dpg.add_group()
            dpg.add_separator()
            dpg.add_button(label="Apply to Config", callback=self._apply)
        # Pre-populate from config defaults
        for i, path in enumerate(self.config.lora_paths):
            strength = self.config.lora_strengths[i] if i < len(self.config.lora_strengths) else 1.0
            self._add_row_with_values(path, strength)

    def _add_row_with_values(self, path: str = "", strength: float = 1.0) -> None:
        with dpg.group(horizontal=True, parent=self._list_tag) as row:
            path_tag = dpg.add_input_text(default_value=path, hint="Path to .safetensors", width=400)
            strength_tag = dpg.add_input_float(default_value=strength, width=80, step=0)
            dpg.add_button(label="Browse", callback=lambda: self._browse(path_tag))
            dpg.add_button(
                label="Remove",
                callback=lambda s, a, u: self._remove_row(u),
                user_data=(row, path_tag, strength_tag),
            )
            self._rows.append((path_tag, strength_tag))

    def _add_row(self) -> None:
        with dpg.group(horizontal=True, parent=self._list_tag) as row:
            path_tag = dpg.add_input_text(hint="Path to .safetensors", width=400)
            strength_tag = dpg.add_input_float(default_value=1.0, width=80, step=0)
            dpg.add_button(label="Browse", callback=lambda: self._browse(path_tag))
            dpg.add_button(
                label="Remove",
                callback=lambda s, a, u: self._remove_row(u),
                user_data=(row, path_tag, strength_tag),
            )
            self._rows.append((path_tag, strength_tag))

    def _browse(self, path_tag: int) -> None:
        def _selected(sender, app_data):
            selections = app_data.get("selections", {})
            if selections:
                dpg.set_value(path_tag, list(selections.values())[0])

        with dpg.file_dialog(callback=_selected, width=700, height=400, show=True):
            dpg.add_file_extension(".safetensors", color=(0, 255, 0, 255))

    def _remove_row(self, user_data: tuple) -> None:
        row, path_tag, strength_tag = user_data
        self._rows = [(p, s) for p, s in self._rows if p != path_tag]
        dpg.delete_item(row)

    def _apply(self) -> None:
        paths = []
        strengths = []
        for path_tag, strength_tag in self._rows:
            p = dpg.get_value(path_tag)
            s = dpg.get_value(strength_tag)
            if p:
                paths.append(p)
                strengths.append(s)
        self.config.lora_paths = paths
        self.config.lora_strengths = strengths
        # Invalidate cached pipeline so it reloads with new LoRAs
        if self._on_change:
            self._on_change()
        self.config.save()
