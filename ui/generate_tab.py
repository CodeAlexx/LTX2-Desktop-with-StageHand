"""Generate tab — workflow-oriented generation workspace with preview tools."""

from __future__ import annotations

import random
import subprocess

import dearpygui.dearpygui as dpg

from config import AppConfig
from inference_worker import InferenceWorker, WorkerState
from audio_utils import normalize_output_audio
from ui.image_preview import ImagePreview
from ui.theme import create_start_button_theme, create_stop_button_theme
from ui.video_player import VideoPlayer


# Duration presets in seconds -> frame count (n%8==1, at 25fps)
DURATION_PRESETS = {
    "1s": 25,
    "2s": 49,
    "3s": 73,
    "5s": 121,
    "8s": 193,
    "10s": 249,
    "15s": 377,
    "20s": 497,
    "30s": 753,
    "45s": 1121,
    "60s": 1497,
}

# Resolution presets (w x h, both divisible by 32)
RESOLUTION_PRESETS = {
    "512x320": (512, 320),
    "512x512": (512, 512),
    "768x512": (768, 512),
    "768x768": (768, 768),
    "1024x576": (1024, 576),
    "1280x704": (1280, 704),
}

RESOLUTION_ITEMS = [*RESOLUTION_PRESETS.keys(), "Custom"]
DURATION_ITEMS = [*DURATION_PRESETS.keys(), "Custom"]


class GenerateTab:
    """Builds and manages the generation panel with embedded preview tools."""

    def __init__(self, config: AppConfig, worker: InferenceWorker, pipeline_factory):
        self.config = config
        self.worker = worker
        self.pipeline_factory = pipeline_factory
        self._pipeline = None

        # Widget tags
        self._prompt_tag = None
        self._resolution_tag = None
        self._width_tag = None
        self._height_tag = None
        self._duration_tag = None
        self._frames_tag = None
        self._fps_tag = None
        self._seed_tag = None
        self._enhance_tag = None
        self._image_path_tag = None
        self._image_strength_tag = None
        self._image_crf_tag = None
        self._audio_path_tag = None
        self._audio_start_tag = None
        self._audio_max_duration_tag = None
        self._progress_bar_tag = None
        self._phase_text_tag = None
        self._output_text_tag = None
        self._mode_text_tag = None
        self._render_path_tag = None
        self._generate_btn_tag = None
        self._cancel_btn_tag = None

        # Keyframe conditioning (FML2V)
        self._keyframe_first_path_tag = None
        self._keyframe_first_strength_tag = None
        self._keyframe_middle_path_tag = None
        self._keyframe_middle_strength_tag = None
        self._keyframe_last_path_tag = None
        self._keyframe_last_strength_tag = None

        # IC-LoRA widget tags
        self._ic_lora_video_path_tag = None
        self._ic_lora_strength_tag = None
        self._ic_lora_attention_tag = None

        # HQ sampler widget tags
        self._hq_sampler_tag = None
        self._hq_steps_tag = None

        # Prompt / sampler widget tags
        self._scheduler_tag = None
        self._two_stage_tag = None
        self._two_stage_noise_tag = None
        self._decoder_noise_tag = None
        self._decoder_noise_scale_tag = None
        self._decoder_noise_shift_tag = None
        self._decoder_noise_seed_tag = None
        self._zero_neg_cond_tag = None
        self._preprocess_image_tag = None

        # Four-pass / Long video widget tags
        self._four_pass_tag = None
        self._long_video_tag = None
        self._long_video_group_tag = None
        self._long_video_preset_tag = None
        self._long_video_seconds_tag = None
        self._long_video_tile_tag = None
        self._long_video_overlap_tag = None
        self._long_video_adain_tag = None
        self._long_video_memory_tag = None
        self._long_video_anchor_frames_tag = None

        # Dev mode widget tags
        self._negative_prompt_tag = None
        self._steps_tag = None
        self._video_cfg_tag = None
        self._video_stg_tag = None
        self._video_rescale_tag = None
        self._video_skip_step_tag = None
        self._audio_cfg_tag = None
        self._audio_stg_tag = None
        self._audio_rescale_tag = None
        self._audio_skip_step_tag = None
        self._a2v_tag = None
        self._v2a_tag = None
        self._stg_blocks_tag = None

        # Video player
        self._player: VideoPlayer | None = None
        self._player_parent_tag = None
        self._image_preview: ImagePreview | None = None
        self._image_preview_parent_tag = None
        self._preview_source_tag = None
        self._last_mode_text = None
        self._last_render_path_text = None
        self._last_preview_path = None
        self._last_preview_source = None
        self._start_button_theme = None
        self._stop_button_theme = None

    def build(self, parent: int | str) -> None:
        with dpg.tab(label="Generate", parent=parent):
            dpg.add_text("Generation Workspace", color=(180, 220, 255))
            dpg.add_text(
                "HQ single-shot is the recommended path. Long video chunking is isolated under Experimental.",
                color=(180, 180, 180),
            )
            dpg.add_separator()

            self._build_status_section()

            with dpg.tab_bar():
                self._build_workflow_tab()
                self._build_inputs_tab()
                self._build_guides_tab()
                self._build_sampling_tab()
                self._build_experimental_tab()
                self._build_preview_tab()

    def _build_status_section(self) -> None:
        with dpg.child_window(border=True, height=150, autosize_x=True):
            self._mode_text_tag = dpg.add_text("Active mode: T2V (text-only)", color=(180, 220, 255))
            self._render_path_tag = dpg.add_text("", color=(220, 210, 170))
            with dpg.group(horizontal=True):
                self._generate_btn_tag = dpg.add_button(
                    label="Generate", callback=self._on_generate, width=120, height=30,
                )
                self._cancel_btn_tag = dpg.add_button(
                    label="Cancel", callback=self._on_cancel, width=80, height=30, enabled=False,
                )
                self._start_button_theme = create_start_button_theme()
                self._stop_button_theme = create_stop_button_theme()
                dpg.bind_item_theme(self._generate_btn_tag, self._start_button_theme)
                dpg.bind_item_theme(self._cancel_btn_tag, self._stop_button_theme)
                dpg.add_button(label="Open Output Folder", callback=self._open_output_folder, width=150)
            self._progress_bar_tag = dpg.add_progress_bar(default_value=0.0, width=-1)
            self._phase_text_tag = dpg.add_text("Idle")
            self._output_text_tag = dpg.add_text("(no output yet)")

    def _build_workflow_tab(self) -> None:
        with dpg.tab(label="Workflow"):
            with dpg.collapsing_header(label="Prompt", default_open=True):
                self._prompt_tag = dpg.add_input_text(
                    multiline=True,
                    height=120,
                    width=-1,
                    default_value=(
                        "A cinematic shot of a golden retriever running through "
                        "a sunlit meadow, slow motion, warm afternoon light"
                    ),
                )
                self._enhance_tag = dpg.add_checkbox(
                    label="Enhance prompt with Gemma rewrite",
                    default_value=self.config.enhance_prompt,
                )
                dpg.add_text("Prompt enhancement is slower and only applies on the single-shot path.", color=(180, 180, 180))

            with dpg.collapsing_header(label="Frame And Size", default_open=True):
                dpg.add_text("Frames must be 8n+1. Width and height must be divisible by 32.", color=(180, 180, 180))
                with dpg.group(horizontal=True):
                    dpg.add_text("Resolution preset")
                    self._resolution_tag = dpg.add_combo(
                        items=RESOLUTION_ITEMS,
                        default_value=self._resolution_preset_label(),
                        width=140,
                        callback=self._apply_resolution_preset,
                    )
                    dpg.add_text("Width")
                    self._width_tag = dpg.add_input_int(
                        default_value=self.config.width,
                        width=90,
                        step=0,
                        min_value=32,
                        min_clamped=True,
                        callback=self._mark_resolution_custom,
                    )
                    dpg.add_text("Height")
                    self._height_tag = dpg.add_input_int(
                        default_value=self.config.height,
                        width=90,
                        step=0,
                        min_value=32,
                        min_clamped=True,
                        callback=self._mark_resolution_custom,
                    )

                with dpg.group(horizontal=True):
                    dpg.add_text("Duration preset")
                    self._duration_tag = dpg.add_combo(
                        items=DURATION_ITEMS,
                        default_value=self._duration_preset_label(),
                        width=140,
                        callback=self._apply_duration_preset,
                    )
                    dpg.add_text("Frames")
                    self._frames_tag = dpg.add_input_int(
                        default_value=self.config.num_frames,
                        width=90,
                        step=0,
                        min_value=9,
                        min_clamped=True,
                        callback=self._mark_duration_custom,
                    )
                    dpg.add_text("FPS")
                    self._fps_tag = dpg.add_input_float(
                        default_value=self.config.fps,
                        width=80,
                        step=0,
                        min_value=1.0,
                        min_clamped=True,
                    )

                with dpg.group(horizontal=True):
                    dpg.add_text("Seed")
                    self._seed_tag = dpg.add_input_int(
                        default_value=self.config.seed,
                        width=120,
                        step=0,
                    )
                    dpg.add_button(label="Random", callback=self._randomize_seed, width=80)
                    dpg.add_text(f"Pipeline mode from Settings: {self.config.pipeline_mode}", color=(180, 180, 180))

    def _build_inputs_tab(self) -> None:
        with dpg.tab(label="Inputs"):
            with dpg.collapsing_header(label="Source Image", default_open=True):
                dpg.add_text("Used for I2V, or combined with audio for A2V + image.", color=(180, 180, 180))
                with dpg.group(horizontal=True):
                    self._image_path_tag = dpg.add_input_text(
                        hint="Path to conditioning image",
                        width=540,
                        default_value=self.config.image_path,
                    )
                    dpg.add_button(label="Browse", callback=self._browse_image)
                    dpg.add_button(
                        label="View",
                        callback=lambda: self._show_image_preview("Primary I2V / A2V image"),
                    )
                    dpg.add_button(label="Clear", callback=lambda: dpg.set_value(self._image_path_tag, ""))

                with dpg.group(horizontal=True):
                    dpg.add_text("Strength")
                    self._image_strength_tag = dpg.add_input_float(
                        default_value=self.config.image_strength,
                        width=90,
                        step=0,
                        min_value=0.0,
                        max_value=1.0,
                        min_clamped=True,
                        max_clamped=True,
                    )
                    dpg.add_text("Image CRF")
                    self._image_crf_tag = dpg.add_input_int(
                        default_value=self.config.image_crf,
                        width=90,
                        step=0,
                        min_value=0,
                        max_value=51,
                        min_clamped=True,
                        max_clamped=True,
                    )
                    dpg.add_text("Only used when image preprocessing is enabled.", color=(180, 180, 180))

                with dpg.group(horizontal=True):
                    self._zero_neg_cond_tag = dpg.add_checkbox(
                        label="Zero negative conditioning",
                        default_value=self.config.zero_negative_conditioning,
                    )
                    self._preprocess_image_tag = dpg.add_checkbox(
                        label="Preprocess input image",
                        default_value=self.config.preprocess_input_image,
                    )

            with dpg.collapsing_header(label="Audio Conditioning", default_open=True):
                dpg.add_text("Single-shot A2V is supported. Long-video A2V is still blocked.", color=(180, 180, 180))
                with dpg.group(horizontal=True):
                    self._audio_path_tag = dpg.add_input_text(
                        hint="Path to audio file",
                        width=540,
                        default_value=self.config.audio_path,
                    )
                    dpg.add_button(label="Browse", callback=self._browse_audio)
                    dpg.add_button(label="Clear", callback=lambda: dpg.set_value(self._audio_path_tag, ""))

                with dpg.group(horizontal=True):
                    dpg.add_text("Start (s)")
                    self._audio_start_tag = dpg.add_input_float(
                        default_value=self.config.audio_start_time,
                        width=90,
                        step=0,
                        min_value=0.0,
                        min_clamped=True,
                    )
                    dpg.add_text("Max duration (s)")
                    self._audio_max_duration_tag = dpg.add_input_float(
                        default_value=self.config.audio_max_duration or 0.0,
                        width=100,
                        step=0,
                        min_value=0.0,
                        min_clamped=True,
                    )
                    dpg.add_text("0 means auto-match the clip length.", color=(180, 180, 180))

    def _build_guides_tab(self) -> None:
        with dpg.tab(label="Guides"):
            with dpg.collapsing_header(label="Keyframe Conditioning", default_open=True):
                dpg.add_text("First / middle / last frame injection for tighter shot planning.", color=(180, 180, 180))
                self._build_keyframe_row(
                    label="First frame",
                    path_attr="_keyframe_first_path_tag",
                    strength_attr="_keyframe_first_strength_tag",
                    default_path=self.config.keyframe_first_path,
                    default_strength=self.config.keyframe_first_strength,
                    browse_id="first",
                    preview_label="First keyframe",
                    clear_suffix="kf_first",
                )
                self._build_keyframe_row(
                    label="Middle frame",
                    path_attr="_keyframe_middle_path_tag",
                    strength_attr="_keyframe_middle_strength_tag",
                    default_path=self.config.keyframe_middle_path,
                    default_strength=self.config.keyframe_middle_strength,
                    browse_id="middle",
                    preview_label="Middle keyframe",
                    clear_suffix="kf_mid",
                )
                self._build_keyframe_row(
                    label="Last frame",
                    path_attr="_keyframe_last_path_tag",
                    strength_attr="_keyframe_last_strength_tag",
                    default_path=self.config.keyframe_last_path,
                    default_strength=self.config.keyframe_last_strength,
                    browse_id="last",
                    preview_label="Last keyframe",
                    clear_suffix="kf_last",
                )

            with dpg.collapsing_header(label="IC-LoRA Reference Video", default_open=True):
                dpg.add_text("Use a reference video for pose, depth, edge, or motion control.", color=(180, 180, 180))
                with dpg.group(horizontal=True):
                    self._ic_lora_video_path_tag = dpg.add_input_text(
                        hint="Path to reference or control video",
                        width=540,
                        default_value=self.config.ic_lora_video_path,
                    )
                    dpg.add_button(label="Browse", callback=self._browse_ic_lora_video)
                    dpg.add_button(
                        label="Clear",
                        callback=lambda: dpg.set_value(self._ic_lora_video_path_tag, ""),
                    )
                with dpg.group(horizontal=True):
                    dpg.add_text("Strength")
                    self._ic_lora_strength_tag = dpg.add_input_float(
                        default_value=self.config.ic_lora_strength,
                        width=90,
                        step=0,
                        min_value=0.0,
                        max_value=1.0,
                        min_clamped=True,
                        max_clamped=True,
                    )
                    dpg.add_text("Attention")
                    self._ic_lora_attention_tag = dpg.add_input_float(
                        default_value=self.config.ic_lora_attention_strength,
                        width=90,
                        step=0,
                        min_value=0.0,
                        max_value=1.0,
                        min_clamped=True,
                        max_clamped=True,
                    )

    def _build_sampling_tab(self) -> None:
        with dpg.tab(label="Sampling"):
            with dpg.collapsing_header(label="Quality Path", default_open=True):
                with dpg.group(horizontal=True):
                    self._hq_sampler_tag = dpg.add_checkbox(
                        label="HQ sampler (Res2s)",
                        default_value=self.config.hq_sampler_enabled,
                    )
                    dpg.add_text("Stage 1 steps")
                    self._hq_steps_tag = dpg.add_input_int(
                        default_value=self.config.hq_num_inference_steps,
                        width=90,
                        step=0,
                        min_value=5,
                        max_value=50,
                        min_clamped=True,
                        max_clamped=True,
                    )
                    self._four_pass_tag = dpg.add_checkbox(
                        label="Four-pass pipeline",
                        default_value=self.config.use_four_pass,
                    )

                with dpg.group(horizontal=True):
                    dpg.add_text("Scheduler")
                    self._scheduler_tag = dpg.add_combo(
                        items=["default", "bong_tangent"],
                        default_value=self.config.scheduler_type,
                        width=160,
                    )
                    dpg.add_text("Pipeline mode from Settings controls distilled vs dev.", color=(180, 180, 180))

            with dpg.collapsing_header(label="Dev Guidance", default_open=True):
                dpg.add_text("These settings are ignored in distilled mode. HQ uses its own official guider params.", color=(180, 180, 180))
                dpg.add_text("Negative Prompt")
                self._negative_prompt_tag = dpg.add_input_text(
                    multiline=True,
                    height=80,
                    width=-1,
                    default_value=self.config.negative_prompt,
                )

                with dpg.group(horizontal=True):
                    dpg.add_text("Steps")
                    self._steps_tag = dpg.add_input_int(
                        default_value=self.config.num_inference_steps,
                        width=90,
                        step=0,
                        min_value=1,
                        min_clamped=True,
                    )
                    dpg.add_text("STG blocks")
                    self._stg_blocks_tag = dpg.add_input_text(
                        default_value=",".join(str(b) for b in self.config.stg_blocks),
                        width=120,
                    )

                with dpg.group(horizontal=True):
                    dpg.add_text("Video CFG")
                    self._video_cfg_tag = dpg.add_input_float(
                        default_value=self.config.video_cfg_scale,
                        width=90,
                        step=0,
                    )
                    dpg.add_text("STG")
                    self._video_stg_tag = dpg.add_input_float(
                        default_value=self.config.video_stg_scale,
                        width=90,
                        step=0,
                    )
                    dpg.add_text("Rescale")
                    self._video_rescale_tag = dpg.add_input_float(
                        default_value=self.config.video_rescale,
                        width=90,
                        step=0,
                    )
                    dpg.add_text("Skip")
                    self._video_skip_step_tag = dpg.add_input_int(
                        default_value=self.config.video_skip_step,
                        width=90,
                        step=0,
                        min_value=0,
                        min_clamped=True,
                    )

                with dpg.group(horizontal=True):
                    dpg.add_text("Audio CFG")
                    self._audio_cfg_tag = dpg.add_input_float(
                        default_value=self.config.audio_cfg_scale,
                        width=90,
                        step=0,
                    )
                    dpg.add_text("STG")
                    self._audio_stg_tag = dpg.add_input_float(
                        default_value=self.config.audio_stg_scale,
                        width=90,
                        step=0,
                    )
                    dpg.add_text("Rescale")
                    self._audio_rescale_tag = dpg.add_input_float(
                        default_value=self.config.audio_rescale,
                        width=90,
                        step=0,
                    )
                    dpg.add_text("Skip")
                    self._audio_skip_step_tag = dpg.add_input_int(
                        default_value=self.config.audio_skip_step,
                        width=90,
                        step=0,
                        min_value=0,
                        min_clamped=True,
                    )

                with dpg.group(horizontal=True):
                    dpg.add_text("A2V scale")
                    self._a2v_tag = dpg.add_input_float(
                        default_value=self.config.a2v_scale,
                        width=90,
                        step=0,
                    )
                    dpg.add_text("V2A scale")
                    self._v2a_tag = dpg.add_input_float(
                        default_value=self.config.v2a_scale,
                        width=90,
                        step=0,
                    )

            with dpg.collapsing_header(label="Noise And Decode", default_open=False):
                with dpg.group(horizontal=True):
                    self._two_stage_tag = dpg.add_checkbox(
                        label="Two-stage sampling",
                        default_value=self.config.two_stage_sampling,
                    )
                    dpg.add_text("Noise strength")
                    self._two_stage_noise_tag = dpg.add_input_float(
                        default_value=self.config.two_stage_noise_strength,
                        width=90,
                        step=0,
                        min_value=0.0,
                        min_clamped=True,
                    )

                with dpg.group(horizontal=True):
                    self._decoder_noise_tag = dpg.add_checkbox(
                        label="VAE decoder noise",
                        default_value=self.config.decoder_noise_enabled,
                    )
                    dpg.add_text("Scale")
                    self._decoder_noise_scale_tag = dpg.add_input_float(
                        default_value=self.config.decoder_noise_scale,
                        width=90,
                        step=0,
                    )
                    dpg.add_text("Shift")
                    self._decoder_noise_shift_tag = dpg.add_input_float(
                        default_value=self.config.decoder_noise_shift,
                        width=90,
                        step=0,
                    )
                    dpg.add_text("Seed")
                    self._decoder_noise_seed_tag = dpg.add_input_int(
                        default_value=self.config.decoder_noise_seed,
                        width=110,
                        step=0,
                    )

    def _build_experimental_tab(self) -> None:
        with dpg.tab(label="Experimental"):
            dpg.add_text("Chunked long video remains available here, but it is not the recommended path.", color=(220, 180, 120))
            with dpg.group(horizontal=True):
                self._long_video_tag = dpg.add_checkbox(
                    label="Enable long video chunking",
                    default_value=self.config.long_video_enabled,
                    callback=self._toggle_long_video,
                )
                dpg.add_text("Audio-conditioned long video is still blocked.", color=(180, 180, 180))

            self._long_video_group_tag = dpg.add_group(show=self.config.long_video_enabled)
            with dpg.group(parent=self._long_video_group_tag):
                with dpg.group(horizontal=True):
                    dpg.add_text("Preset")
                    self._long_video_preset_tag = dpg.add_combo(
                        items=["quality", "balanced", "fast", "two_minute", "max_length"],
                        default_value=self.config.long_video_preset,
                        width=140,
                        callback=self._apply_preset,
                    )
                    dpg.add_text("Total seconds")
                    self._long_video_seconds_tag = dpg.add_input_float(
                        default_value=self.config.long_video_total_seconds,
                        width=100,
                        step=0,
                        min_value=1.0,
                        min_clamped=True,
                    )
                    dpg.add_text("Anchor frames")
                    self._long_video_anchor_frames_tag = dpg.add_input_int(
                        default_value=self.config.long_video_anchor_frames,
                        width=90,
                        step=0,
                        min_value=1,
                        min_clamped=True,
                    )

                with dpg.group(horizontal=True):
                    dpg.add_text("Tile size")
                    self._long_video_tile_tag = dpg.add_input_int(
                        default_value=self.config.long_video_temporal_tile_size,
                        width=90,
                        step=0,
                        min_value=9,
                        min_clamped=True,
                    )
                    dpg.add_text("Overlap")
                    self._long_video_overlap_tag = dpg.add_input_int(
                        default_value=self.config.long_video_temporal_overlap,
                        width=90,
                        step=0,
                        min_value=1,
                        min_clamped=True,
                    )
                    dpg.add_text("AdaIN")
                    self._long_video_adain_tag = dpg.add_input_float(
                        default_value=self.config.long_video_adain_factor,
                        width=90,
                        step=0,
                        min_value=0.0,
                        max_value=1.0,
                        min_clamped=True,
                        max_clamped=True,
                    )
                    dpg.add_text("Memory")
                    self._long_video_memory_tag = dpg.add_input_float(
                        default_value=self.config.long_video_long_memory_strength,
                        width=90,
                        step=0,
                        min_value=0.0,
                        max_value=1.0,
                        min_clamped=True,
                        max_clamped=True,
                    )

    def _build_preview_tab(self) -> None:
        with dpg.tab(label="Preview"):
            with dpg.tab_bar():
                with dpg.tab(label="Output Video"):
                    self._player_parent_tag = dpg.add_group()
                    self._player = VideoPlayer(self._player_parent_tag)

                with dpg.tab(label="Image Preview"):
                    with dpg.group(horizontal=True):
                        dpg.add_text("Preview source")
                        self._preview_source_tag = dpg.add_combo(
                            items=[
                                "Primary I2V / A2V image",
                                "First keyframe",
                                "Middle keyframe",
                                "Last keyframe",
                            ],
                            default_value="Primary I2V / A2V image",
                            width=220,
                            callback=lambda: self._refresh_image_preview(force=True),
                        )
                        dpg.add_button(
                            label="Refresh",
                            callback=lambda: self._refresh_image_preview(force=True),
                        )
                    self._image_preview_parent_tag = dpg.add_group()
                    self._image_preview = ImagePreview(self._image_preview_parent_tag)

    def _build_keyframe_row(
        self,
        *,
        label: str,
        path_attr: str,
        strength_attr: str,
        default_path: str,
        default_strength: float,
        browse_id: str,
        preview_label: str,
        clear_suffix: str,
    ) -> None:
        with dpg.group(horizontal=True):
            dpg.add_text(label)
            path_tag = dpg.add_input_text(
                hint=f"Path to {label.lower()} image",
                width=460,
                default_value=default_path,
            )
            setattr(self, path_attr, path_tag)
            dpg.add_button(label=f"Browse##{clear_suffix}", callback=lambda: self._browse_keyframe(browse_id))
            dpg.add_button(
                label=f"View##{clear_suffix}",
                callback=lambda: self._show_image_preview(preview_label),
            )
            dpg.add_button(label=f"Clear##{clear_suffix}", callback=lambda: dpg.set_value(path_tag, ""))
            dpg.add_text("Strength")
            strength_tag = dpg.add_input_float(
                default_value=default_strength,
                width=80,
                step=0,
                min_value=0.0,
                max_value=1.0,
                min_clamped=True,
                max_clamped=True,
            )
            setattr(self, strength_attr, strength_tag)

    def _resolution_preset_label(self) -> str:
        current = (self.config.width, self.config.height)
        for label, dims in RESOLUTION_PRESETS.items():
            if dims == current:
                return label
        return "Custom"

    def _duration_preset_label(self) -> str:
        for label, frames in DURATION_PRESETS.items():
            if frames == self.config.num_frames:
                return label
        return "Custom"

    def _apply_resolution_preset(self, sender=None, app_data=None) -> None:
        preset = dpg.get_value(self._resolution_tag)
        dims = RESOLUTION_PRESETS.get(preset)
        if not dims:
            return
        width, height = dims
        dpg.set_value(self._width_tag, width)
        dpg.set_value(self._height_tag, height)

    def _apply_duration_preset(self, sender=None, app_data=None) -> None:
        preset = dpg.get_value(self._duration_tag)
        frames = DURATION_PRESETS.get(preset)
        if frames is None:
            return
        dpg.set_value(self._frames_tag, frames)

    def _mark_resolution_custom(self, sender=None, app_data=None) -> None:
        selected = dpg.get_value(self._resolution_tag)
        dims = RESOLUTION_PRESETS.get(selected)
        current = (dpg.get_value(self._width_tag), dpg.get_value(self._height_tag))
        if dims != current:
            dpg.set_value(self._resolution_tag, "Custom")

    def _mark_duration_custom(self, sender=None, app_data=None) -> None:
        selected = dpg.get_value(self._duration_tag)
        frames = DURATION_PRESETS.get(selected)
        if frames != dpg.get_value(self._frames_tag):
            dpg.set_value(self._duration_tag, "Custom")

    def _browse_image(self) -> None:
        def _selected(sender, app_data):
            selections = app_data.get("selections", {})
            if selections:
                dpg.set_value(self._image_path_tag, list(selections.values())[0])
                self._show_image_preview("Primary I2V / A2V image")

        with dpg.file_dialog(callback=_selected, width=700, height=400, show=True):
            dpg.add_file_extension(".png", color=(0, 255, 0, 255))
            dpg.add_file_extension(".jpg", color=(0, 255, 0, 255))
            dpg.add_file_extension(".jpeg", color=(0, 255, 0, 255))
            dpg.add_file_extension(".webp", color=(0, 255, 0, 255))

    def _browse_keyframe(self, which: str) -> None:
        tag_map = {
            "first": self._keyframe_first_path_tag,
            "middle": self._keyframe_middle_path_tag,
            "last": self._keyframe_last_path_tag,
        }
        target_tag = tag_map[which]

        def _selected(sender, app_data):
            selections = app_data.get("selections", {})
            if selections:
                dpg.set_value(target_tag, list(selections.values())[0])
                preview_map = {
                    "first": "First keyframe",
                    "middle": "Middle keyframe",
                    "last": "Last keyframe",
                }
                self._show_image_preview(preview_map[which])

        with dpg.file_dialog(callback=_selected, width=700, height=400, show=True):
            dpg.add_file_extension(".png", color=(0, 255, 0, 255))
            dpg.add_file_extension(".jpg", color=(0, 255, 0, 255))
            dpg.add_file_extension(".jpeg", color=(0, 255, 0, 255))
            dpg.add_file_extension(".webp", color=(0, 255, 0, 255))

    def _browse_audio(self) -> None:
        def _selected(sender, app_data):
            selections = app_data.get("selections", {})
            if selections:
                dpg.set_value(self._audio_path_tag, list(selections.values())[0])

        with dpg.file_dialog(callback=_selected, width=700, height=400, show=True):
            dpg.add_file_extension(".mp3", color=(0, 200, 255, 255))
            dpg.add_file_extension(".wav", color=(0, 200, 255, 255))
            dpg.add_file_extension(".flac", color=(0, 200, 255, 255))
            dpg.add_file_extension(".mp4", color=(0, 255, 0, 255))

    def _browse_ic_lora_video(self) -> None:
        def _selected(sender, app_data):
            selections = app_data.get("selections", {})
            if selections:
                dpg.set_value(self._ic_lora_video_path_tag, list(selections.values())[0])

        with dpg.file_dialog(callback=_selected, width=700, height=400, show=True):
            dpg.add_file_extension(".mp4", color=(0, 255, 0, 255))
            dpg.add_file_extension(".webm", color=(0, 255, 0, 255))
            dpg.add_file_extension(".avi", color=(0, 255, 0, 255))
            dpg.add_file_extension(".mov", color=(0, 255, 0, 255))

    def _toggle_long_video(self, sender=None, app_data=None) -> None:
        show = dpg.get_value(self._long_video_tag)
        dpg.configure_item(self._long_video_group_tag, show=show)

    def _apply_preset(self, sender=None, app_data=None) -> None:
        from long_video_presets import PRESETS

        preset_name = dpg.get_value(self._long_video_preset_tag)
        preset = PRESETS.get(preset_name)
        if preset:
            dpg.set_value(self._long_video_tile_tag, preset["temporal_tile_size"])
            dpg.set_value(self._long_video_overlap_tag, preset["temporal_overlap"])
            dpg.set_value(self._long_video_adain_tag, preset["adain_factor"])
            dpg.set_value(self._long_video_memory_tag, preset.get("long_memory_strength", 0.0))

    def _randomize_seed(self) -> None:
        dpg.set_value(self._seed_tag, random.randint(0, 2**31 - 1))

    def _get_preview_path(self) -> str:
        if not self._preview_source_tag:
            return ""
        source = dpg.get_value(self._preview_source_tag)
        source_map = {
            "Primary I2V / A2V image": self._image_path_tag,
            "First keyframe": self._keyframe_first_path_tag,
            "Middle keyframe": self._keyframe_middle_path_tag,
            "Last keyframe": self._keyframe_last_path_tag,
        }
        target = source_map.get(source)
        if not target:
            return ""
        return dpg.get_value(target).strip()

    def _show_image_preview(self, source: str) -> None:
        if self._preview_source_tag:
            dpg.set_value(self._preview_source_tag, source)
        self._refresh_image_preview(force=True)

    def _refresh_image_preview(self, force: bool = False) -> None:
        if not self._image_preview or not self._preview_source_tag:
            return
        source = dpg.get_value(self._preview_source_tag)
        path = self._get_preview_path()
        if not force and source == self._last_preview_source and path == self._last_preview_path:
            return
        self._last_preview_source = source
        self._last_preview_path = path
        self._image_preview.load(path or None)

    def _update_mode_text(self) -> None:
        if not self._mode_text_tag:
            return

        has_image = bool(dpg.get_value(self._image_path_tag).strip()) if self._image_path_tag else False
        has_audio = bool(dpg.get_value(self._audio_path_tag).strip()) if self._audio_path_tag else False
        has_keyframes = all(
            tag is not None for tag in (
                self._keyframe_first_path_tag,
                self._keyframe_middle_path_tag,
                self._keyframe_last_path_tag,
            )
        ) and any(
            dpg.get_value(tag).strip()
            for tag in (
                self._keyframe_first_path_tag,
                self._keyframe_middle_path_tag,
                self._keyframe_last_path_tag,
            )
        )
        has_ic_lora = bool(dpg.get_value(self._ic_lora_video_path_tag).strip()) if self._ic_lora_video_path_tag else False

        if has_image and has_audio:
            mode_text = "Active mode: I2V + A2V (source image + audio)"
        elif has_image:
            mode_text = "Active mode: I2V (source image)"
        elif has_audio:
            mode_text = "Active mode: A2V (audio-driven)"
        elif has_keyframes:
            mode_text = "Active mode: T2V + keyframes"
        else:
            mode_text = "Active mode: T2V (text-only)"

        if has_ic_lora:
            mode_text += " + IC-LoRA"

        if mode_text != self._last_mode_text:
            dpg.set_value(self._mode_text_tag, mode_text)
            self._last_mode_text = mode_text

    def _update_render_path_text(self) -> None:
        if not self._render_path_tag:
            return

        mode = self.config.pipeline_mode
        is_hq = dpg.get_value(self._hq_sampler_tag) if self._hq_sampler_tag else self.config.hq_sampler_enabled
        is_four_pass = dpg.get_value(self._four_pass_tag) if self._four_pass_tag else self.config.use_four_pass
        is_long_video = dpg.get_value(self._long_video_tag) if self._long_video_tag else self.config.long_video_enabled

        if is_long_video:
            path_text = f"Render path: Experimental long-video chunking on {mode} base"
        elif is_hq:
            path_text = "Render path: HQ two-stage single shot"
        elif is_four_pass:
            path_text = f"Render path: Four-pass {mode} pipeline"
        elif mode == "distilled":
            path_text = "Render path: Distilled two-stage single shot"
        else:
            path_text = "Render path: Dev two-stage single shot"

        if path_text != self._last_render_path_text:
            dpg.set_value(self._render_path_tag, path_text)
            self._last_render_path_text = path_text

    def _parse_stg_blocks(self) -> list[int]:
        raw = dpg.get_value(self._stg_blocks_tag).strip()
        if not raw:
            return []
        try:
            return [int(x.strip()) for x in raw.split(",") if x.strip()]
        except ValueError:
            return self.config.stg_blocks

    def _validate_inputs(
        self,
        *,
        width: int,
        height: int,
        num_frames: int,
        fps: float,
        image_crf: int,
        audio_max_duration: float | None,
        long_video_enabled: bool,
        long_video_tile: int,
        long_video_overlap: int,
        long_video_anchor_frames: int,
        four_pass_enabled: bool,
    ) -> str | None:
        if width <= 0 or height <= 0:
            return "Width and height must be positive."
        if width % 32 or height % 32:
            return "Width and height must be divisible by 32."
        if num_frames <= 0 or (num_frames - 1) % 8 != 0:
            return "Frame count must satisfy 8n+1."
        if fps <= 0:
            return "FPS must be greater than 0."
        if image_crf < 0 or image_crf > 51:
            return "Image CRF must be between 0 and 51."
        if audio_max_duration is not None and audio_max_duration <= 0:
            return "Audio max duration must be greater than 0, or 0 for auto."
        if long_video_enabled:
            if four_pass_enabled:
                return "Disable Four-pass when Long video is enabled."
            if long_video_tile <= 0:
                return "Long video tile size must be greater than 0."
            if long_video_overlap <= 0:
                return "Long video overlap must be greater than 0."
            if long_video_overlap >= long_video_tile:
                return "Long video overlap must be smaller than tile size."
            if long_video_anchor_frames <= 0:
                return "Long video anchor frames must be at least 1."
        return None

    def _on_generate(self) -> None:
        if self.worker.is_busy:
            return

        if self._player:
            self._player.stop()

        prompt = dpg.get_value(self._prompt_tag)
        width = dpg.get_value(self._width_tag)
        height = dpg.get_value(self._height_tag)
        num_frames = dpg.get_value(self._frames_tag)
        fps = dpg.get_value(self._fps_tag)
        seed = dpg.get_value(self._seed_tag)
        enhance = dpg.get_value(self._enhance_tag)
        image_path = dpg.get_value(self._image_path_tag).strip()
        image_strength = dpg.get_value(self._image_strength_tag)
        image_crf = dpg.get_value(self._image_crf_tag)
        audio_path = dpg.get_value(self._audio_path_tag).strip()
        audio_start = dpg.get_value(self._audio_start_tag)
        audio_max_duration_raw = dpg.get_value(self._audio_max_duration_tag)
        audio_max_duration = audio_max_duration_raw if audio_max_duration_raw > 0 else None

        negative_prompt = dpg.get_value(self._negative_prompt_tag).strip()
        num_inference_steps = dpg.get_value(self._steps_tag)
        video_cfg_scale = dpg.get_value(self._video_cfg_tag)
        video_stg_scale = dpg.get_value(self._video_stg_tag)
        video_rescale = dpg.get_value(self._video_rescale_tag)
        video_skip_step = dpg.get_value(self._video_skip_step_tag)
        audio_cfg_scale = dpg.get_value(self._audio_cfg_tag)
        audio_stg_scale = dpg.get_value(self._audio_stg_tag)
        audio_rescale = dpg.get_value(self._audio_rescale_tag)
        audio_skip_step = dpg.get_value(self._audio_skip_step_tag)
        a2v_scale = dpg.get_value(self._a2v_tag)
        v2a_scale = dpg.get_value(self._v2a_tag)
        stg_blocks = self._parse_stg_blocks()

        scheduler_type = dpg.get_value(self._scheduler_tag)
        two_stage = dpg.get_value(self._two_stage_tag)
        two_stage_noise = dpg.get_value(self._two_stage_noise_tag)
        decoder_noise = dpg.get_value(self._decoder_noise_tag)
        decoder_noise_scale = dpg.get_value(self._decoder_noise_scale_tag)
        decoder_noise_shift = dpg.get_value(self._decoder_noise_shift_tag)
        decoder_noise_seed = dpg.get_value(self._decoder_noise_seed_tag)
        zero_neg_cond = dpg.get_value(self._zero_neg_cond_tag)
        preprocess_image = dpg.get_value(self._preprocess_image_tag)

        hq_sampler = dpg.get_value(self._hq_sampler_tag)
        hq_steps = dpg.get_value(self._hq_steps_tag)
        four_pass = dpg.get_value(self._four_pass_tag)
        long_video = dpg.get_value(self._long_video_tag)
        long_video_preset = dpg.get_value(self._long_video_preset_tag)
        long_video_seconds = dpg.get_value(self._long_video_seconds_tag)
        long_video_tile = dpg.get_value(self._long_video_tile_tag)
        long_video_overlap = dpg.get_value(self._long_video_overlap_tag)
        long_video_adain = dpg.get_value(self._long_video_adain_tag)
        long_video_memory = dpg.get_value(self._long_video_memory_tag)
        long_video_anchor_frames = dpg.get_value(self._long_video_anchor_frames_tag)

        validation_error = self._validate_inputs(
            width=width,
            height=height,
            num_frames=num_frames,
            fps=fps,
            image_crf=image_crf,
            audio_max_duration=audio_max_duration,
            long_video_enabled=long_video,
            long_video_tile=long_video_tile,
            long_video_overlap=long_video_overlap,
            long_video_anchor_frames=long_video_anchor_frames,
            four_pass_enabled=four_pass,
        )
        if validation_error:
            dpg.set_value(self._output_text_tag, f"ERROR: {validation_error}")
            return

        self.config.hq_sampler_enabled = hq_sampler
        self.config.hq_num_inference_steps = hq_steps
        self.config.use_four_pass = four_pass
        self.config.long_video_enabled = long_video
        self.config.long_video_preset = long_video_preset
        self.config.long_video_total_seconds = long_video_seconds
        self.config.long_video_temporal_tile_size = long_video_tile
        self.config.long_video_temporal_overlap = long_video_overlap
        self.config.long_video_adain_factor = long_video_adain
        self.config.long_video_long_memory_strength = long_video_memory
        self.config.long_video_anchor_frames = long_video_anchor_frames
        self.config.scheduler_type = scheduler_type
        self.config.two_stage_sampling = two_stage
        self.config.two_stage_noise_strength = two_stage_noise
        self.config.decoder_noise_enabled = decoder_noise
        self.config.decoder_noise_scale = decoder_noise_scale
        self.config.decoder_noise_shift = decoder_noise_shift
        self.config.decoder_noise_seed = decoder_noise_seed
        self.config.zero_negative_conditioning = zero_neg_cond
        self.config.preprocess_input_image = preprocess_image
        self.config.width = width
        self.config.height = height
        self.config.num_frames = num_frames
        self.config.fps = fps
        self.config.seed = seed
        self.config.enhance_prompt = enhance
        self.config.image_path = image_path
        self.config.image_strength = image_strength
        self.config.image_crf = image_crf
        self.config.audio_path = audio_path
        self.config.audio_start_time = audio_start
        self.config.audio_max_duration = audio_max_duration
        self.config.video_skip_step = video_skip_step
        self.config.audio_skip_step = audio_skip_step

        ic_lora_video = dpg.get_value(self._ic_lora_video_path_tag).strip()
        ic_lora_strength = dpg.get_value(self._ic_lora_strength_tag)
        ic_lora_attention = dpg.get_value(self._ic_lora_attention_tag)
        self.config.ic_lora_video_path = ic_lora_video
        self.config.ic_lora_strength = ic_lora_strength
        self.config.ic_lora_attention_strength = ic_lora_attention

        from ltx_pipelines.utils.args import ImageConditioningInput

        keyframes: list[ImageConditioningInput] = []
        kf_first = dpg.get_value(self._keyframe_first_path_tag).strip()
        kf_middle = dpg.get_value(self._keyframe_middle_path_tag).strip()
        kf_last = dpg.get_value(self._keyframe_last_path_tag).strip()
        kf_first_strength = dpg.get_value(self._keyframe_first_strength_tag)
        kf_middle_strength = dpg.get_value(self._keyframe_middle_strength_tag)
        kf_last_strength = dpg.get_value(self._keyframe_last_strength_tag)
        self.config.keyframe_first_path = kf_first
        self.config.keyframe_middle_path = kf_middle
        self.config.keyframe_last_path = kf_last
        self.config.keyframe_first_strength = kf_first_strength
        self.config.keyframe_middle_strength = kf_middle_strength
        self.config.keyframe_last_strength = kf_last_strength
        if kf_first:
            keyframes.append(ImageConditioningInput(path=kf_first, frame_idx=0, strength=kf_first_strength, crf=0))
        if kf_middle:
            keyframes.append(
                ImageConditioningInput(path=kf_middle, frame_idx=num_frames // 2, strength=kf_middle_strength, crf=0)
            )
        if kf_last:
            keyframes.append(ImageConditioningInput(path=kf_last, frame_idx=num_frames - 1, strength=kf_last_strength, crf=0))

        if self._pipeline is None:
            self._pipeline = self.pipeline_factory()

        if long_video and audio_path:
            dpg.set_value(
                self._output_text_tag,
                "ERROR: Long video A2V input is not wired yet. Disable Long video for audio-conditioned runs.",
            )
            return

        dpg.configure_item(self._generate_btn_tag, enabled=False)
        dpg.configure_item(self._cancel_btn_tag, enabled=True)

        if long_video:
            from long_video_presets import seconds_to_frames

            total_frames = seconds_to_frames(long_video_seconds, fps)
            self.worker.submit(
                self._run_long_video,
                prompt=prompt,
                seed=seed,
                width=width,
                height=height,
                total_frames=total_frames,
                fps=fps,
                temporal_tile_size=long_video_tile,
                temporal_overlap=long_video_overlap,
                adain_factor=long_video_adain,
                long_memory_strength=long_video_memory,
                per_step_adain=self.config.long_video_per_step_adain,
                per_step_adain_factors=self.config.long_video_per_step_adain_factors,
                anchor_frames=long_video_anchor_frames,
                image_path=image_path or None,
                image_strength=image_strength,
            )
            return

        self.worker.submit(
            self._pipeline.generate,
            prompt=prompt,
            seed=seed,
            width=width,
            height=height,
            num_frames=num_frames,
            fps=fps,
            enhance_prompt=enhance,
            image_path=image_path or None,
            image_strength=image_strength,
            image_crf=image_crf,
            audio_path=audio_path or None,
            audio_start_time=audio_start,
            audio_max_duration=audio_max_duration,
            negative_prompt=negative_prompt or None,
            num_inference_steps=num_inference_steps,
            video_cfg_scale=video_cfg_scale,
            video_stg_scale=video_stg_scale,
            video_rescale=video_rescale,
            audio_cfg_scale=audio_cfg_scale,
            audio_stg_scale=audio_stg_scale,
            audio_rescale=audio_rescale,
            a2v_scale=a2v_scale,
            v2a_scale=v2a_scale,
            video_skip_step=video_skip_step,
            audio_skip_step=audio_skip_step,
            stg_blocks=stg_blocks,
            keyframes=keyframes or None,
            ic_lora_video_path=ic_lora_video or None,
            ic_lora_strength=ic_lora_strength,
            ic_lora_attention_strength=ic_lora_attention,
        )

    def _run_long_video(self, progress_cb=None, **kwargs) -> Path:
        """Run long video generation and decode/save the result."""
        from long_video import LTXVLongVideoService
        from ltx_core.model.video_vae import TilingConfig, get_video_chunks_number
        from ltx_core.model.video_vae import decode_video as vae_decode_video
        from ltx_pipelines.utils.media_io import encode_video
        import datetime

        def _progress_adapter(chunk_idx, total_chunks, frames_gen, total_frames, elapsed):
            if progress_cb:
                frac = (chunk_idx + 1) / total_chunks
                progress_cb(f"Chunk {chunk_idx + 1}/{total_chunks} ({frames_gen}/{total_frames}f)", frac * 0.85)

        service = LTXVLongVideoService()
        accumulated, accumulated_audio = service.generate(
            pipeline=self._pipeline,
            progress_callback=_progress_adapter,
            **kwargs,
        )

        ledger = self._pipeline._build_ledger()
        import torch

        generator = torch.Generator(device=self._pipeline.device).manual_seed(kwargs.get("seed", 42))
        tiling = TilingConfig.default()

        decoded_video = vae_decode_video(
            accumulated["samples"],
            ledger.video_decoder(),
            tiling,
            generator,
        )

        decoded_audio = None
        if accumulated_audio is not None:
            from ltx_core.model.audio_vae import decode_audio as vae_decode_audio

            decoded_audio = vae_decode_audio(
                accumulated_audio.to(self._pipeline.device),
                ledger.audio_decoder(),
                ledger.vocoder(),
            )
            decoded_audio = normalize_output_audio(decoded_audio)

        out_dir = self.config.ensure_output_dir()
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        fps = kwargs.get("fps", 25.0)
        out_path = out_dir / f"ltx2_longvideo_{ts}.mp4"

        nf = accumulated["samples"].shape[2] * 8 + 1
        video_chunks_number = get_video_chunks_number(nf, tiling)
        encode_video(
            video=decoded_video,
            fps=fps,
            audio=decoded_audio,
            output_path=str(out_path),
            video_chunks_number=video_chunks_number,
        )
        return out_path

    def _on_cancel(self) -> None:
        self.worker.cancel()

    def _open_output_folder(self) -> None:
        out = self.config.ensure_output_dir()
        subprocess.Popen(["xdg-open", str(out)])

    def invalidate_pipeline(self) -> None:
        self._pipeline = None

    def poll(self) -> None:
        if self._player:
            self._player.update()
        self._update_mode_text()
        self._update_render_path_text()
        self._refresh_image_preview()

        snap = self.worker.status.snapshot()
        state = snap["state"]
        dpg.set_value(self._progress_bar_tag, snap["progress"])
        dpg.set_value(self._phase_text_tag, snap["phase"] or "Idle")

        if state in (WorkerState.DONE, WorkerState.ERROR, WorkerState.IDLE):
            dpg.configure_item(self._generate_btn_tag, enabled=True)
            dpg.configure_item(self._cancel_btn_tag, enabled=False)

        if state == WorkerState.DONE and snap["output_path"]:
            dpg.set_value(self._output_text_tag, snap["output_path"])
            if self._player:
                self._player.load(snap["output_path"])
            self.worker.reset()

        if state == WorkerState.ERROR:
            dpg.set_value(self._output_text_tag, f"ERROR: {snap['error']}")
            self.worker.reset()
