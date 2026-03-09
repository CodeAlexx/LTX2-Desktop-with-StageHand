"""Inline DearPyGui video player using PyAV for decoding.

Renders video frames into a dpg.dynamic_texture updated each render frame.
Playback runs in a background thread that decodes and queues frames.
Audio via sounddevice synced to video PTS, graceful fallback if unavailable.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from pathlib import Path

import av
import numpy as np
import dearpygui.dearpygui as dpg

logger = logging.getLogger(__name__)

# Try sounddevice for audio (needs libportaudio)
try:
    import sounddevice as sd
    HAS_AUDIO = True
except (ImportError, OSError):
    sd = None
    HAS_AUDIO = False
    logger.info("sounddevice not available — video-only playback")

# Max preview dimensions (resize larger videos to fit)
MAX_DISPLAY_W = 768
MAX_DISPLAY_H = 480


class VideoPlayer:
    """Inline video player with play/pause/seek/scrub and audio."""

    def __init__(self, parent_tag: int | str):
        self._parent = parent_tag
        self._video_path: str | None = None
        self._container: av.container.InputContainer | None = None

        # Video info
        self._duration: float = 0.0
        self._fps: float = 24.0
        self._frame_count: int = 0
        self._native_w: int = 0
        self._native_h: int = 0
        self._display_w: int = MAX_DISPLAY_W
        self._display_h: int = MAX_DISPLAY_H

        # Playback state
        self._playing = False
        self._paused = False
        self._current_pts: float = 0.0
        self._playback_start_time: float = 0.0
        self._playback_start_pts: float = 0.0

        # Threading
        self._decode_thread: threading.Thread | None = None
        self._audio_thread: threading.Thread | None = None
        self._frame_queue: queue.Queue[tuple[float, np.ndarray] | None] = queue.Queue(maxsize=30)
        self._stop_event = threading.Event()
        self._seek_event = threading.Event()
        self._seek_target: float = 0.0
        self._lock = threading.Lock()

        # Audio
        self._audio_stream: sd.OutputStream | None = None
        self._audio_queue: queue.Queue[np.ndarray | None] = queue.Queue(maxsize=60)
        self._has_audio_track = False

        # DPG widget tags
        self._texture_tag: int | None = None
        self._image_tag: int | None = None
        self._seek_slider_tag: int | None = None
        self._time_label_tag: int | None = None
        self._play_btn_tag: int | None = None
        self._status_tag: int | None = None

        # Pre-allocated frame buffer (RGBA float32, flattened)
        self._frame_buffer: list[float] = [0.0] * (MAX_DISPLAY_W * MAX_DISPLAY_H * 4)
        self._pending_frame: np.ndarray | None = None
        self._frame_dirty = False

        self._build_ui()

    def _build_ui(self) -> None:
        """Build player widgets inside parent."""
        with dpg.group(parent=self._parent):
            dpg.add_text("Video Player")

            # Texture — create at max size, reuse for all videos
            with dpg.texture_registry():
                self._texture_tag = dpg.add_dynamic_texture(
                    width=MAX_DISPLAY_W, height=MAX_DISPLAY_H,
                    default_value=self._frame_buffer,
                )
            self._image_tag = dpg.add_image(self._texture_tag, width=MAX_DISPLAY_W, height=MAX_DISPLAY_H)

            # Transport controls
            with dpg.group(horizontal=True):
                dpg.add_button(label="<< -10s", callback=lambda: self.seek(self._current_pts - 10), width=60)
                dpg.add_button(label="|< Start", callback=lambda: self.seek(0), width=60)
                self._play_btn_tag = dpg.add_button(
                    label="Play", callback=self._toggle_play, width=70,
                )
                dpg.add_button(label="Stop", callback=self.stop, width=50)
                dpg.add_button(label="+10s >>", callback=lambda: self.seek(self._current_pts + 10), width=60)

            # Seek bar
            self._seek_slider_tag = dpg.add_slider_float(
                label="", min_value=0.0, max_value=1.0, width=-1,
                callback=self._on_seek_drag,
            )

            # Time + status
            with dpg.group(horizontal=True):
                self._time_label_tag = dpg.add_text("00:00 / 00:00")
                dpg.add_spacer(width=20)
                self._status_tag = dpg.add_text("No video loaded")

    def load(self, video_path: str) -> None:
        """Load a video file and display the first frame."""
        self.stop()

        self._video_path = video_path
        try:
            container = av.open(video_path)
        except Exception as e:
            logger.error("Failed to open %s: %s", video_path, e)
            dpg.set_value(self._status_tag, f"Error: {e}")
            return

        video_stream = container.streams.video[0]
        self._native_w = video_stream.codec_context.width
        self._native_h = video_stream.codec_context.height
        self._fps = float(video_stream.average_rate or 24)

        if video_stream.duration and video_stream.time_base:
            self._duration = float(video_stream.duration * video_stream.time_base)
        elif container.duration:
            self._duration = container.duration / av.time_base
        else:
            self._duration = 0.0

        # Check for audio
        self._has_audio_track = len(container.streams.audio) > 0

        # Compute display size (fit within MAX bounds, maintain aspect ratio)
        scale = min(MAX_DISPLAY_W / self._native_w, MAX_DISPLAY_H / self._native_h, 1.0)
        self._display_w = int(self._native_w * scale)
        self._display_h = int(self._native_h * scale)
        # Ensure even dimensions
        self._display_w = self._display_w - (self._display_w % 2)
        self._display_h = self._display_h - (self._display_h % 2)

        # Resize image widget
        dpg.configure_item(self._image_tag, width=self._display_w, height=self._display_h)

        # Decode first frame
        container.seek(0)
        for frame in container.decode(video=0):
            self._show_frame(frame)
            break

        container.close()
        self._current_pts = 0.0

        # Update UI
        dpg.configure_item(self._seek_slider_tag, max_value=max(self._duration, 0.01))
        dpg.set_value(self._seek_slider_tag, 0.0)
        self._update_time_label()

        audio_str = " + audio" if self._has_audio_track else " (no audio)"
        dpg.set_value(
            self._status_tag,
            f"{self._native_w}x{self._native_h} @ {self._fps:.1f}fps, {self._duration:.1f}s{audio_str}",
        )
        dpg.set_value(self._play_btn_tag, "Play")
        dpg.configure_item(self._play_btn_tag, label="Play")

    def _show_frame(self, frame: av.VideoFrame) -> None:
        """Convert an av.VideoFrame to RGBA float32 and stage for display."""
        rgb = frame.reformat(
            width=self._display_w,
            height=self._display_h,
            format="rgb24",
        ).to_ndarray()  # (H, W, 3) uint8

        h, w, _ = rgb.shape
        # Build RGBA float32 in a single numpy operation
        rgba = np.empty((h, w, 4), dtype=np.float32)
        rgba[:, :, :3] = rgb.astype(np.float32) * (1.0 / 255.0)
        rgba[:, :, 3] = 1.0

        # Pad to MAX dimensions (texture is fixed-size)
        if h < MAX_DISPLAY_H or w < MAX_DISPLAY_W:
            padded = np.zeros((MAX_DISPLAY_H, MAX_DISPLAY_W, 4), dtype=np.float32)
            padded[:h, :w, :] = rgba
            rgba = padded

        with self._lock:
            self._pending_frame = rgba.ravel()
            self._frame_dirty = True

    def _toggle_play(self) -> None:
        if not self._video_path:
            return
        if self._playing and not self._paused:
            self.pause()
        elif self._playing and self._paused:
            self._resume()
        else:
            self.play()

    def play(self) -> None:
        """Start playback from current position."""
        if not self._video_path:
            return

        self.stop()
        self._playing = True
        self._paused = False
        self._stop_event.clear()
        self._seek_event.clear()
        self._playback_start_time = time.monotonic()
        self._playback_start_pts = self._current_pts

        dpg.configure_item(self._play_btn_tag, label="Pause")

        self._decode_thread = threading.Thread(target=self._decode_loop, daemon=True)
        self._decode_thread.start()

        if self._has_audio_track and HAS_AUDIO:
            self._audio_thread = threading.Thread(target=self._audio_loop, daemon=True)
            self._audio_thread.start()

    def pause(self) -> None:
        """Pause playback."""
        if self._playing:
            self._paused = True
            dpg.configure_item(self._play_btn_tag, label="Resume")

    def _resume(self) -> None:
        """Resume from pause."""
        self._paused = False
        self._playback_start_time = time.monotonic()
        self._playback_start_pts = self._current_pts
        dpg.configure_item(self._play_btn_tag, label="Pause")

    def stop(self) -> None:
        """Stop playback and reset."""
        self._stop_event.set()
        self._playing = False
        self._paused = False

        # Drain queues
        while not self._frame_queue.empty():
            try:
                self._frame_queue.get_nowait()
            except queue.Empty:
                break
        while not self._audio_queue.empty():
            try:
                self._audio_queue.get_nowait()
            except queue.Empty:
                break

        if self._decode_thread and self._decode_thread.is_alive():
            self._decode_thread.join(timeout=2.0)
        if self._audio_thread and self._audio_thread.is_alive():
            self._audio_thread.join(timeout=2.0)

        if self._audio_stream is not None:
            try:
                self._audio_stream.stop()
                self._audio_stream.close()
            except Exception:
                pass
            self._audio_stream = None

        dpg.configure_item(self._play_btn_tag, label="Play")

    def seek(self, seconds: float) -> None:
        """Seek to a specific timestamp."""
        seconds = max(0.0, min(seconds, self._duration))
        self._current_pts = seconds

        if self._playing:
            # Signal decode thread to seek
            self._seek_target = seconds
            self._seek_event.set()
            self._playback_start_time = time.monotonic()
            self._playback_start_pts = seconds
        else:
            # Not playing — decode and show the frame at this position
            self._seek_and_show(seconds)

        dpg.set_value(self._seek_slider_tag, seconds)
        self._update_time_label()

    def _seek_and_show(self, seconds: float) -> None:
        """Seek to timestamp and display that frame (when stopped)."""
        if not self._video_path:
            return
        try:
            container = av.open(self._video_path)
            # Seek to nearest keyframe before target
            target_ts = int(seconds / container.streams.video[0].time_base)
            container.seek(target_ts, stream=container.streams.video[0])
            for frame in container.decode(video=0):
                pts = float(frame.pts * frame.time_base) if frame.pts is not None else seconds
                if pts >= seconds - 0.05:
                    self._show_frame(frame)
                    break
            container.close()
        except Exception as e:
            logger.debug("Seek error: %s", e)

    def _on_seek_drag(self, sender, app_data) -> None:
        """Handle seek slider drag."""
        self.seek(app_data)

    def _decode_loop(self) -> None:
        """Background thread: decode video frames and queue them."""
        try:
            container = av.open(self._video_path)
            video_stream = container.streams.video[0]

            # Seek to start position
            if self._playback_start_pts > 0.1:
                target_ts = int(self._playback_start_pts / video_stream.time_base)
                container.seek(target_ts, stream=video_stream)

            for frame in container.decode(video=0):
                if self._stop_event.is_set():
                    break

                # Handle seek request
                if self._seek_event.is_set():
                    self._seek_event.clear()
                    target_ts = int(self._seek_target / video_stream.time_base)
                    container.seek(target_ts, stream=video_stream)
                    # Drain queue
                    while not self._frame_queue.empty():
                        try:
                            self._frame_queue.get_nowait()
                        except queue.Empty:
                            break
                    continue

                # Wait while paused
                while self._paused and not self._stop_event.is_set():
                    time.sleep(0.05)

                if self._stop_event.is_set():
                    break

                pts = float(frame.pts * frame.time_base) if frame.pts is not None else 0.0

                # Timing: wait until it's time to show this frame
                elapsed = time.monotonic() - self._playback_start_time
                target_elapsed = pts - self._playback_start_pts
                wait = target_elapsed - elapsed
                if wait > 0.001:
                    time.sleep(wait)

                # Convert and queue
                rgb = frame.reformat(
                    width=self._display_w,
                    height=self._display_h,
                    format="rgb24",
                ).to_ndarray()

                h, w, _ = rgb.shape
                rgba = np.empty((h, w, 4), dtype=np.float32)
                rgba[:, :, :3] = rgb.astype(np.float32) * (1.0 / 255.0)
                rgba[:, :, 3] = 1.0

                if h < MAX_DISPLAY_H or w < MAX_DISPLAY_W:
                    padded = np.zeros((MAX_DISPLAY_H, MAX_DISPLAY_W, 4), dtype=np.float32)
                    padded[:h, :w, :] = rgba
                    rgba = padded

                try:
                    self._frame_queue.put((pts, rgba.ravel()), timeout=0.5)
                except queue.Full:
                    pass

            container.close()

            # Signal end of stream
            if not self._stop_event.is_set():
                self._frame_queue.put(None)

        except Exception as e:
            logger.error("Decode error: %s", e)

    def _audio_loop(self) -> None:
        """Background thread: decode and play audio."""
        if not HAS_AUDIO or not self._has_audio_track:
            return

        try:
            container = av.open(self._video_path)
            audio_stream = container.streams.audio[0]
            sample_rate = audio_stream.codec_context.sample_rate or 44100
            channels = audio_stream.codec_context.channels or 2

            if self._playback_start_pts > 0.1:
                target_ts = int(self._playback_start_pts / audio_stream.time_base)
                container.seek(target_ts, stream=audio_stream)

            self._audio_stream = sd.OutputStream(
                samplerate=sample_rate,
                channels=channels,
                dtype="float32",
            )
            self._audio_stream.start()

            resampler = av.AudioResampler(format="flt", layout="stereo" if channels >= 2 else "mono", rate=sample_rate)

            for frame in container.decode(audio=0):
                if self._stop_event.is_set():
                    break

                while self._paused and not self._stop_event.is_set():
                    time.sleep(0.05)

                if self._stop_event.is_set():
                    break

                resampled = resampler.resample(frame)
                for r_frame in resampled:
                    audio_array = r_frame.to_ndarray().T  # (samples, channels)
                    if audio_array.ndim == 1:
                        audio_array = audio_array.reshape(-1, 1)
                    try:
                        self._audio_stream.write(audio_array)
                    except Exception:
                        break

            container.close()
        except Exception as e:
            logger.debug("Audio playback error: %s", e)
        finally:
            if self._audio_stream is not None:
                try:
                    self._audio_stream.stop()
                    self._audio_stream.close()
                except Exception:
                    pass
                self._audio_stream = None

    def _update_time_label(self) -> None:
        cur = self._format_time(self._current_pts)
        total = self._format_time(self._duration)
        dpg.set_value(self._time_label_tag, f"{cur} / {total}")

    @staticmethod
    def _format_time(seconds: float) -> str:
        m = int(seconds) // 60
        s = int(seconds) % 60
        return f"{m:02d}:{s:02d}"

    def update(self) -> None:
        """Called every render frame from the main loop.

        Drains the frame queue and pushes the latest frame to the texture.
        """
        if not self._playing:
            # Still check for dirty frame from seek/load
            with self._lock:
                if self._frame_dirty and self._pending_frame is not None:
                    dpg.set_value(self._texture_tag, self._pending_frame)
                    self._frame_dirty = False
            return

        # Drain queue, keep only the latest frame
        latest_pts = None
        latest_data = None

        while True:
            try:
                item = self._frame_queue.get_nowait()
            except queue.Empty:
                break

            if item is None:
                # End of stream
                self._playing = False
                self._current_pts = self._duration
                dpg.configure_item(self._play_btn_tag, label="Play")
                break

            pts, data = item
            latest_pts = pts
            latest_data = data

        if latest_data is not None:
            dpg.set_value(self._texture_tag, latest_data)
            self._current_pts = latest_pts
            dpg.set_value(self._seek_slider_tag, self._current_pts)
            self._update_time_label()
