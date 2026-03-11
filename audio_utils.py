"""Helpers for post-decoding audio cleanup."""

from __future__ import annotations

import math

import torch

from ltx_core.types import Audio


def normalize_output_audio(
    audio: Audio | None,
    *,
    target_peak: float = 0.92,
    max_gain_db: float = 18.0,
    silence_floor: float = 1e-4,
) -> Audio | None:
    """Boost quiet generated audio without attenuating already-healthy output."""
    if audio is None:
        return None

    waveform = audio.waveform
    if waveform.numel() == 0:
        return audio

    peak = float(waveform.detach().abs().amax().item())
    if not math.isfinite(peak) or peak < silence_floor or peak >= target_peak:
        return audio

    max_gain = 10.0 ** (max_gain_db / 20.0)
    gain = min(target_peak / peak, max_gain)
    boosted = (waveform * gain).clamp(-0.99, 0.99)
    return Audio(waveform=boosted, sampling_rate=audio.sampling_rate)
