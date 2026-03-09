"""Generate tab — main T2V/I2V generation panel with inline video player."""

from __future__ import annotations

import random
import subprocess
from pathlib import Path

import dearpygui.dearpygui as dpg

from config import AppConfig
from inference_worker import InferenceWorker, WorkerState
from ui.video_player import VideoPlayer


# Duration presets in seconds -> frame count (n%8==1, at 25fps)
DURATION_PRESETS = {
    "1s": 25,     # 1.0s
    "2s": 49,     # 2.0s
    "3s": 73,     # 3.0s
    "5s": 121,    # 5.0s
    "8s": 193,    # 8.0s
    "10s": 249,   # 10.0s
    "15s": 377,   # 15.1s
    "20s": 497,   # 19.9s
    "30s": 753,   # 30.1s
    "45s": 1121,  # 44.8s
    "60s": 1497,  # 59.9s
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


class GenerateTab:
    """Builds and manages the generation panel with embedded video player."""

    def __init__(self, config: AppConfig, worker: InferenceWorker, pipeline_factory):
        self.config = config
        self.worker = worker
        self.pipeline_factory = pipeline_factory
        self._pipeline = None

        # Widget tags
        self._prompt_tag = None
        self._resolution_tag = None
        self._duration_tag = None
        self._fps_tag = None
        self._seed_tag = None
        self._enhance_tag = None
        self._image_path_tag = None
        self._image_strength_tag = None
        self._audio_path_tag = None
        self._audio_start_tag = None
        self._progress_bar_tag = None
        self._phase_text_tag = None
        self._output_text_tag = None
        self._generate_btn_tag = None
        self._cancel_btn_tag = None

        # CC prompt widget tags
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

        # Dev mode widget tags
        self._negative_prompt_tag = None
        self._steps_tag = None
        self._video_cfg_tag = None
        self._video_stg_tag = None
        self._video_rescale_tag = None
        self._audio_cfg_tag = None
        self._audio_stg_tag = None
        self._audio_rescale_tag = None
        self._a2v_tag = None
        self._v2a_tag = None
        self._stg_blocks_tag = None
        self._dev_group_tag = None

        # Video player
        self._player: VideoPlayer | None = None
        self._player_parent_tag = None

    def build(self, parent: int | str) -> None:
        with dpg.tab(label="Generate", parent=parent):
            # Prompt section
            dpg.add_text("Prompt")
            self._prompt_tag = dpg.add_input_text(
                multiline=True, height=100, width=-1,
                default_value=(
                    "A cinematic shot of a golden retriever running through "
                    "a sunlit meadow, slow motion, warm afternoon light"
                ),
            )
            self._enhance_tag = dpg.add_checkbox(
                label="Enhance prompt (Gemma rewrites — SLOW with Stagehand)",
                default_value=self.config.enhance_prompt,
            )

            dpg.add_separator()

            # I2V section
            dpg.add_text("Image Conditioning (optional — leave empty for T2V)")
            with dpg.group(horizontal=True):
                self._image_path_tag = dpg.add_input_text(
                    hint="Path to conditioning image", width=500,
                    default_value=self.config.image_path,
                )
                dpg.add_button(label="Browse", callback=self._browse_image)
                dpg.add_button(label="Clear", callback=lambda: dpg.set_value(self._image_path_tag, ""))

            with dpg.group(horizontal=True):
                dpg.add_text("Strength")
                self._image_strength_tag = dpg.add_input_float(
                    default_value=self.config.image_strength,
                    width=80, step=0, min_value=0.0, max_value=1.0,
                    min_clamped=True, max_clamped=True,
                )

            # A2V audio conditioning
            dpg.add_text("Audio Conditioning (optional — mp3/wav/video with audio)")
            with dpg.group(horizontal=True):
                self._audio_path_tag = dpg.add_input_text(
                    hint="Path to audio file", width=500,
                    default_value=self.config.audio_path,
                )
                dpg.add_button(label="Browse", callback=self._browse_audio)
                dpg.add_button(label="Clear", callback=lambda: dpg.set_value(self._audio_path_tag, ""))
            with dpg.group(horizontal=True):
                dpg.add_text("Start (s)")
                self._audio_start_tag = dpg.add_input_float(
                    default_value=self.config.audio_start_time,
                    width=60, step=0, min_value=0.0, min_clamped=True,
                )

            dpg.add_separator()

            # I2V-only options
            with dpg.group(horizontal=True):
                self._zero_neg_cond_tag = dpg.add_checkbox(
                    label="Zero negative conditioning (I2V only)",
                    default_value=self.config.zero_negative_conditioning,
                )
                self._preprocess_image_tag = dpg.add_checkbox(
                    label="Preprocess input image (I2V only)",
                    default_value=self.config.preprocess_input_image,
                )

            dpg.add_separator()

            # Four-pass & Long video
            self._four_pass_tag = dpg.add_checkbox(
                label="Four-pass pipeline (spatial + temporal upscale + refinement)",
                default_value=self.config.use_four_pass,
            )

            self._long_video_tag = dpg.add_checkbox(
                label="Long video (temporal tiling)",
                default_value=self.config.long_video_enabled,
                callback=self._toggle_long_video,
            )

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
                    dpg.add_text("  Total seconds")
                    self._long_video_seconds_tag = dpg.add_input_float(
                        default_value=self.config.long_video_total_seconds,
                        width=80, step=0, min_value=1.0, min_clamped=True,
                    )

                with dpg.group(horizontal=True):
                    dpg.add_text("Tile size")
                    self._long_video_tile_tag = dpg.add_input_int(
                        default_value=self.config.long_video_temporal_tile_size,
                        width=60, step=0,
                    )
                    dpg.add_text("  Overlap")
                    self._long_video_overlap_tag = dpg.add_input_int(
                        default_value=self.config.long_video_temporal_overlap,
                        width=60, step=0,
                    )
                    dpg.add_text("  AdaIN")
                    self._long_video_adain_tag = dpg.add_input_float(
                        default_value=self.config.long_video_adain_factor,
                        width=60, step=0, min_value=0.0, max_value=1.0,
                        min_clamped=True, max_clamped=True,
                    )
                    dpg.add_text("  Memory")
                    self._long_video_memory_tag = dpg.add_input_float(
                        default_value=self.config.long_video_long_memory_strength,
                        width=60, step=0, min_value=0.0, max_value=1.0,
                        min_clamped=True, max_clamped=True,
                    )

            dpg.add_separator()

            # Settings row
            with dpg.group(horizontal=True):
                dpg.add_text("Resolution")
                self._resolution_tag = dpg.add_combo(
                    items=list(RESOLUTION_PRESETS.keys()),
                    default_value="768x512",
                    width=140,
                )
                dpg.add_text("  Duration")
                self._duration_tag = dpg.add_combo(
                    items=list(DURATION_PRESETS.keys()),
                    default_value="3s",
                    width=80,
                )

            with dpg.group(horizontal=True):
                dpg.add_text("FPS")
                self._fps_tag = dpg.add_input_float(
                    default_value=25.0, width=60, step=0,
                )
                dpg.add_text("  Seed")
                self._seed_tag = dpg.add_input_int(
                    default_value=self.config.seed, width=100, step=0,
                )
                dpg.add_button(label="Random", callback=self._randomize_seed, width=60)

            dpg.add_separator()

            # === Dev mode guidance controls ===
            self._dev_group_tag = dpg.add_group()
            with dpg.group(parent=self._dev_group_tag):
                dpg.add_text("Dev Mode Guidance (ignored in distilled mode)")

                dpg.add_text("Negative Prompt")
                self._negative_prompt_tag = dpg.add_input_text(
                    multiline=True, height=60, width=-1,
                    default_value=self.config.negative_prompt,
                )

                with dpg.group(horizontal=True):
                    dpg.add_text("Steps")
                    self._steps_tag = dpg.add_input_int(
                        default_value=self.config.num_inference_steps, width=60, step=0,
                    )
                    dpg.add_text("  STG Blocks")
                    self._stg_blocks_tag = dpg.add_input_text(
                        default_value=",".join(str(b) for b in self.config.stg_blocks),
                        width=80,
                    )

                with dpg.group(horizontal=True):
                    dpg.add_text("Video CFG")
                    self._video_cfg_tag = dpg.add_input_float(
                        default_value=self.config.video_cfg_scale, width=60, step=0,
                    )
                    dpg.add_text("  STG")
                    self._video_stg_tag = dpg.add_input_float(
                        default_value=self.config.video_stg_scale, width=60, step=0,
                    )
                    dpg.add_text("  Rescale")
                    self._video_rescale_tag = dpg.add_input_float(
                        default_value=self.config.video_rescale, width=60, step=0,
                    )

                with dpg.group(horizontal=True):
                    dpg.add_text("Audio CFG")
                    self._audio_cfg_tag = dpg.add_input_float(
                        default_value=self.config.audio_cfg_scale, width=60, step=0,
                    )
                    dpg.add_text("  STG")
                    self._audio_stg_tag = dpg.add_input_float(
                        default_value=self.config.audio_stg_scale, width=60, step=0,
                    )
                    dpg.add_text("  Rescale")
                    self._audio_rescale_tag = dpg.add_input_float(
                        default_value=self.config.audio_rescale, width=60, step=0,
                    )

                with dpg.group(horizontal=True):
                    dpg.add_text("A2V Scale")
                    self._a2v_tag = dpg.add_input_float(
                        default_value=self.config.a2v_scale, width=60, step=0,
                    )
                    dpg.add_text("  V2A Scale")
                    self._v2a_tag = dpg.add_input_float(
                        default_value=self.config.v2a_scale, width=60, step=0,
                    )

            dpg.add_separator()

            # === CC Prompt Controls ===
            with dpg.group(horizontal=True):
                dpg.add_text("Scheduler")
                self._scheduler_tag = dpg.add_combo(
                    items=["default", "bong_tangent"],
                    default_value=self.config.scheduler_type,
                    width=140,
                )

            with dpg.group(horizontal=True):
                self._two_stage_tag = dpg.add_checkbox(
                    label="Two-stage sampling",
                    default_value=self.config.two_stage_sampling,
                )
                dpg.add_text("  Noise strength")
                self._two_stage_noise_tag = dpg.add_input_float(
                    default_value=self.config.two_stage_noise_strength,
                    width=60, step=0, min_value=0.0, min_clamped=True,
                )

            with dpg.group(horizontal=True):
                self._decoder_noise_tag = dpg.add_checkbox(
                    label="VAE decoder noise",
                    default_value=self.config.decoder_noise_enabled,
                )
                dpg.add_text("  Scale")
                self._decoder_noise_scale_tag = dpg.add_input_float(
                    default_value=self.config.decoder_noise_scale,
                    width=60, step=0,
                )
                dpg.add_text("  Shift")
                self._decoder_noise_shift_tag = dpg.add_input_float(
                    default_value=self.config.decoder_noise_shift,
                    width=60, step=0,
                )
                dpg.add_text("  Seed")
                self._decoder_noise_seed_tag = dpg.add_input_int(
                    default_value=self.config.decoder_noise_seed,
                    width=80, step=0,
                )

            dpg.add_separator()

            # Generate / Cancel
            with dpg.group(horizontal=True):
                self._generate_btn_tag = dpg.add_button(
                    label="Generate", callback=self._on_generate, width=120, height=30,
                )
                self._cancel_btn_tag = dpg.add_button(
                    label="Cancel", callback=self._on_cancel, width=80, height=30, enabled=False,
                )

            # Progress
            self._progress_bar_tag = dpg.add_progress_bar(default_value=0.0, width=-1)
            self._phase_text_tag = dpg.add_text("Idle")

            dpg.add_separator()

            # Output path + open folder
            self._output_text_tag = dpg.add_text("(no output yet)")
            dpg.add_button(label="Open Output Folder", callback=self._open_output_folder)

            dpg.add_separator()

            # Video player (embedded inline)
            self._player_parent_tag = dpg.add_group()
            self._player = VideoPlayer(self._player_parent_tag)

    def _browse_image(self) -> None:
        def _selected(sender, app_data):
            selections = app_data.get("selections", {})
            if selections:
                dpg.set_value(self._image_path_tag, list(selections.values())[0])

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

    def _parse_stg_blocks(self) -> list[int]:
        """Parse comma-separated STG block indices."""
        raw = dpg.get_value(self._stg_blocks_tag).strip()
        if not raw:
            return []
        try:
            return [int(x.strip()) for x in raw.split(",") if x.strip()]
        except ValueError:
            return self.config.stg_blocks

    def _on_generate(self) -> None:
        if self.worker.is_busy:
            return

        # Stop any current playback
        if self._player:
            self._player.stop()

        # Read UI values
        prompt = dpg.get_value(self._prompt_tag)
        res_key = dpg.get_value(self._resolution_tag)
        w, h = RESOLUTION_PRESETS.get(res_key, (768, 512))
        dur_key = dpg.get_value(self._duration_tag)
        nf = DURATION_PRESETS.get(dur_key, 73)
        fps = dpg.get_value(self._fps_tag)
        seed = dpg.get_value(self._seed_tag)
        enhance = dpg.get_value(self._enhance_tag)
        image_path = dpg.get_value(self._image_path_tag).strip()
        image_strength = dpg.get_value(self._image_strength_tag)
        audio_path = dpg.get_value(self._audio_path_tag).strip()
        audio_start = dpg.get_value(self._audio_start_tag)

        # Dev mode params
        negative_prompt = dpg.get_value(self._negative_prompt_tag).strip()
        num_inference_steps = dpg.get_value(self._steps_tag)
        video_cfg_scale = dpg.get_value(self._video_cfg_tag)
        video_stg_scale = dpg.get_value(self._video_stg_tag)
        video_rescale = dpg.get_value(self._video_rescale_tag)
        audio_cfg_scale = dpg.get_value(self._audio_cfg_tag)
        audio_stg_scale = dpg.get_value(self._audio_stg_tag)
        audio_rescale = dpg.get_value(self._audio_rescale_tag)
        a2v_scale = dpg.get_value(self._a2v_tag)
        v2a_scale = dpg.get_value(self._v2a_tag)
        stg_blocks = self._parse_stg_blocks()

        # CC prompt settings
        scheduler_type = dpg.get_value(self._scheduler_tag)
        two_stage = dpg.get_value(self._two_stage_tag)
        two_stage_noise = dpg.get_value(self._two_stage_noise_tag)
        decoder_noise = dpg.get_value(self._decoder_noise_tag)
        decoder_noise_scale = dpg.get_value(self._decoder_noise_scale_tag)
        decoder_noise_shift = dpg.get_value(self._decoder_noise_shift_tag)
        decoder_noise_seed = dpg.get_value(self._decoder_noise_seed_tag)
        zero_neg_cond = dpg.get_value(self._zero_neg_cond_tag)
        preprocess_image = dpg.get_value(self._preprocess_image_tag)

        # Update config with generation settings
        self.config.use_four_pass = dpg.get_value(self._four_pass_tag)
        self.config.scheduler_type = scheduler_type
        self.config.two_stage_sampling = two_stage
        self.config.two_stage_noise_strength = two_stage_noise
        self.config.decoder_noise_enabled = decoder_noise
        self.config.decoder_noise_scale = decoder_noise_scale
        self.config.decoder_noise_shift = decoder_noise_shift
        self.config.decoder_noise_seed = decoder_noise_seed
        self.config.zero_negative_conditioning = zero_neg_cond
        self.config.preprocess_input_image = preprocess_image

        # Lazy-create pipeline (invalidate if config changed)
        if self._pipeline is None:
            self._pipeline = self.pipeline_factory()

        dpg.configure_item(self._generate_btn_tag, enabled=False)
        dpg.configure_item(self._cancel_btn_tag, enabled=True)

        # Check if long video mode
        if dpg.get_value(self._long_video_tag):
            from long_video import LTXVLongVideoService
            from long_video_presets import seconds_to_frames

            total_seconds = dpg.get_value(self._long_video_seconds_tag)
            total_frames = seconds_to_frames(total_seconds, fps)

            self.worker.submit(
                self._run_long_video,
                prompt=prompt,
                seed=seed,
                width=w,
                height=h,
                total_frames=total_frames,
                fps=fps,
                temporal_tile_size=dpg.get_value(self._long_video_tile_tag),
                temporal_overlap=dpg.get_value(self._long_video_overlap_tag),
                adain_factor=dpg.get_value(self._long_video_adain_tag),
                long_memory_strength=dpg.get_value(self._long_video_memory_tag),
                per_step_adain=self.config.long_video_per_step_adain,
                per_step_adain_factors=self.config.long_video_per_step_adain_factors,
            )
        else:
            self.worker.submit(
                self._pipeline.generate,
                prompt=prompt,
                seed=seed,
                width=w,
                height=h,
                num_frames=nf,
                fps=fps,
                enhance_prompt=enhance,
                image_path=image_path or None,
                image_strength=image_strength,
                audio_path=audio_path or None,
                audio_start_time=audio_start,
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
                stg_blocks=stg_blocks,
            )

    def _run_long_video(self, progress_cb=None, **kwargs) -> Path:
        """Run long video generation and decode/save the result."""
        from long_video import LTXVLongVideoService
        from ltx_core.model.video_vae import TilingConfig, get_video_chunks_number
        from ltx_core.model.video_vae import decode_video as vae_decode_video
        from ltx_pipelines.utils.media_io import encode_video
        import datetime

        # Adapt progress_cb (phase, frac) to long video callback (chunk, total, frames, total_frames, elapsed)
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

        # Decode accumulated latent to video
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

        # Decode audio if available
        decoded_audio = None
        if accumulated_audio is not None:
            from ltx_core.model.audio_vae import decode_audio as vae_decode_audio
            decoded_audio = vae_decode_audio(
                accumulated_audio.to(self._pipeline.device),
                ledger.audio_decoder(),
                ledger.vocoder(),
            )

        # Save
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
        """Force pipeline recreation on next generate (e.g., after LoRA changes)."""
        self._pipeline = None

    def poll(self) -> None:
        """Called from the render loop to update progress + player."""
        if self._player:
            self._player.update()

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
