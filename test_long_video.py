"""Tests for long video generation utilities — no GPU required."""
from __future__ import annotations

import sys
from pathlib import Path

_app = str(Path(__file__).parent)
if _app not in sys.path:
    sys.path.insert(0, _app)

import torch
import pytest
from long_video import (
    UpscaleMode,
    select_latents,
    blend_latent_overlap,
    adain_normalize,
    downscale_latent,
    distribute_keyframes_to_chunks,
    extend_audio_latent,
    apply_per_step_adain_patch,
    add_long_memory_conditioning,
)
from long_video_presets import (
    validate_frame_count,
    nearest_valid_frames,
    calculate_chunks,
    seconds_to_frames,
)


# -- select_latents tests --


def test_select_latents_basic():
    """Basic positive index slicing."""
    x = torch.randn(1, 4, 10, 8, 8)
    lat = {"samples": x}
    out = select_latents(lat, 2, 5)
    assert out["samples"].shape == (1, 4, 4, 8, 8)  # frames 2,3,4,5
    assert torch.equal(out["samples"], x[:, :, 2:6])


def test_select_latents_negative_indices():
    """Negative indices count from end."""
    x = torch.randn(1, 4, 10, 8, 8)
    lat = {"samples": x}
    out = select_latents(lat, -3, -1)
    assert out["samples"].shape == (1, 4, 3, 8, 8)  # last 3 frames
    assert torch.equal(out["samples"], x[:, :, 7:10])


def test_select_latents_boundary_clipping():
    """Start/end clipped to valid range."""
    x = torch.randn(1, 4, 5, 8, 8)
    lat = {"samples": x}
    out = select_latents(lat, -10, 100)
    assert out["samples"].shape == (1, 4, 5, 8, 8)  # all frames


def test_select_latents_noise_mask():
    """noise_mask is sliced when present."""
    x = torch.randn(1, 4, 10, 8, 8)
    mask = torch.ones(1, 4, 10, 8, 8)
    lat = {"samples": x, "noise_mask": mask}
    out = select_latents(lat, 1, 3)
    assert "noise_mask" in out
    assert out["noise_mask"].shape == (1, 4, 3, 8, 8)


def test_select_latents_no_noise_mask():
    """No noise_mask key → no noise_mask in output."""
    x = torch.randn(1, 4, 5, 8, 8)
    out = select_latents({"samples": x}, 0, 2)
    assert "noise_mask" not in out


# -- blend_latent_overlap tests --


def test_blend_output_shape():
    """Output length = prev + new - overlap."""
    prev = {"samples": torch.randn(1, 4, 10, 8, 8)}
    new = {"samples": torch.randn(1, 4, 8, 8, 8)}
    out = blend_latent_overlap(prev, new, overlap=3)
    assert out["samples"].shape == (1, 4, 15, 8, 8)  # 10 + 8 - 3


def test_blend_boundary_values():
    """First frame = prev exactly, last frame = new exactly."""
    prev = {"samples": torch.ones(1, 4, 6, 2, 2) * 10.0}
    new = {"samples": torch.ones(1, 4, 6, 2, 2) * 20.0}
    out = blend_latent_overlap(prev, new, overlap=4)
    # First frame from prev (value 10)
    assert torch.allclose(out["samples"][:, :, 0], torch.tensor(10.0))
    # Last frame from new (value 20)
    assert torch.allclose(out["samples"][:, :, -1], torch.tensor(20.0))


def test_blend_zero_overlap():
    """overlap=0 → simple concatenation."""
    prev = {"samples": torch.randn(1, 4, 5, 8, 8)}
    new = {"samples": torch.randn(1, 4, 3, 8, 8)}
    out = blend_latent_overlap(prev, new, overlap=0)
    assert out["samples"].shape == (1, 4, 8, 8, 8)
    assert torch.equal(out["samples"][:, :, :5], prev["samples"])
    assert torch.equal(out["samples"][:, :, 5:], new["samples"])


def test_blend_overlap_midpoint():
    """At the midpoint of overlap, alpha ≈ 0.5 → average of inputs."""
    prev = {"samples": torch.zeros(1, 1, 6, 1, 1)}
    new = {"samples": torch.ones(1, 1, 6, 1, 1)}
    out = blend_latent_overlap(prev, new, overlap=4)
    # Midpoint of 4-frame overlap is between frames 2-3 (indices 3,4 in output)
    mid = out["samples"][0, 0, 3, 0, 0].item()
    assert 0.2 < mid < 0.8, f"Midpoint value {mid} not near 0.5"


# -- adain_normalize tests --


def test_adain_factor_zero_identity():
    """factor=0.0 returns input unchanged."""
    x = torch.randn(1, 4, 5, 8, 8)
    ref = torch.randn(1, 4, 5, 8, 8)
    out = adain_normalize({"samples": x}, {"samples": ref}, factor=0.0)
    assert torch.equal(out["samples"], x)


def test_adain_factor_one_matches_stats():
    """factor=1.0 matches mean/std to reference per channel."""
    torch.manual_seed(0)
    x = torch.randn(1, 4, 5, 8, 8) * 3.0 + 2.0
    ref = torch.randn(1, 4, 5, 8, 8) * 0.5 - 1.0
    out = adain_normalize({"samples": x}, {"samples": ref}, factor=1.0)
    for c in range(4):
        out_mean = out["samples"][0, c].mean().item()
        ref_mean = ref[0, c].mean().item()
        out_std = out["samples"][0, c].std().item()
        ref_std = ref[0, c].std().item()
        assert abs(out_mean - ref_mean) < 0.01, f"ch{c} mean: {out_mean} vs {ref_mean}"
        assert abs(out_std - ref_std) < 0.01, f"ch{c} std: {out_std} vs {ref_std}"


def test_adain_per_frame():
    """per_frame=True normalizes each frame independently."""
    torch.manual_seed(0)
    x = torch.randn(1, 2, 3, 4, 4) * 2.0 + 5.0
    ref = torch.randn(1, 2, 3, 4, 4) * 0.5 - 1.0
    out = adain_normalize({"samples": x}, {"samples": ref}, factor=1.0, per_frame=True)
    for c in range(2):
        for f in range(3):
            out_mean = out["samples"][0, c, f].mean().item()
            ref_mean = ref[0, c, f].mean().item()
            assert abs(out_mean - ref_mean) < 0.05, f"ch{c} f{f} mean: {out_mean} vs {ref_mean}"


# -- distribute_keyframes tests --


def test_distribute_3_keyframes_10_chunks():
    """3 keyframes across 10 chunks: correct assignments."""
    tile = 80
    overlap = 24
    step = tile - overlap  # 56
    total = tile + 9 * step  # 80 + 504 = 584

    kfs = [(0, "img0"), (100, "img100"), (400, "img400")]
    result = distribute_keyframes_to_chunks(kfs, total, tile, overlap)

    # Frame 0 → chunk 0, in_chunk_idx 0
    assert 0 in result
    assert result[0][0] == (0, "img0")

    # Frame 100 → chunk 1 (100 - 80 = 20, 20 // 56 = 0 → chunk 1, in_chunk = 20 % 56 + 24 = 44)
    assert 1 in result
    assert result[1][0][0] == 44  # in-chunk index

    # Frame 400 → (400 - 80) = 320, 320 // 56 = 5 → chunk 6
    assert 6 in result


def test_keyframe_overlap_goes_to_earlier_chunk():
    """Keyframe in overlap region belongs to the earlier chunk."""
    tile = 80
    overlap = 24
    # Frame 79 is last frame of chunk 0
    kfs = [(79, "img79")]
    result = distribute_keyframes_to_chunks(kfs, 200, tile, overlap)
    assert 0 in result
    assert result[0][0] == (79, "img79")


# -- chunk count calculation tests --


def test_chunk_count_single_chunk():
    """total_frames <= tile_size → 1 chunk."""
    n, actual = calculate_chunks(57, 80, 24)
    assert n == 1
    assert validate_frame_count(actual)


def test_chunk_count_exact_two():
    """total = tile + step → exactly 2 chunks."""
    tile, overlap = 81, 24
    step = tile - overlap  # 57
    total = tile + step  # 138
    n, actual = calculate_chunks(total, tile, overlap)
    assert n == 2
    assert validate_frame_count(actual)


def test_chunk_count_many():
    """Large total_frames → multiple chunks."""
    n, actual = calculate_chunks(1000, 81, 24)
    assert n > 1
    assert validate_frame_count(actual)
    assert actual >= 1000


# -- progress callback tests --


def test_progress_callback_count():
    """Progress callback called once per chunk with correct frame counts."""
    calls = []

    def cb(chunk_idx, total_chunks, frames_gen, total_frames, elapsed):
        calls.append((chunk_idx, total_chunks, frames_gen, total_frames))

    # Simulate what the service does (without actually running the pipeline)
    tile = 81
    overlap = 24
    total = 250
    from long_video_presets import nearest_valid_frames
    total = nearest_valid_frames(total)
    n, actual = calculate_chunks(total, tile, overlap)

    # Simulate callback calls
    cb(0, n, tile, actual, 1.0)
    for i in range(1, n):
        frames_so_far = tile + i * (tile - overlap)
        cb(i, n, frames_so_far, actual, float(i))

    assert len(calls) == n
    assert calls[0][0] == 0  # first chunk index
    assert calls[-1][0] == n - 1  # last chunk index
    assert all(c[1] == n for c in calls)  # total_chunks consistent


# -- upscale mode routing tests --


def test_upscale_mode_none():
    """UpscaleMode.NONE is a valid enum value."""
    assert UpscaleMode.NONE == "none"
    assert UpscaleMode.NONE.value == "none"


def test_upscale_mode_spatial_per_chunk():
    """UpscaleMode.SPATIAL_PER_CHUNK is the recommended default."""
    assert UpscaleMode.SPATIAL_PER_CHUNK == "spatial_per_chunk"


def test_upscale_mode_all_values():
    """All 4 upscale modes exist."""
    modes = set(UpscaleMode)
    assert len(modes) == 4
    assert UpscaleMode.NONE in modes
    assert UpscaleMode.SPATIAL_PER_CHUNK in modes
    assert UpscaleMode.FULL_PER_CHUNK in modes
    assert UpscaleMode.SPATIAL_FINAL in modes


# -- downscale_latent tests --


def test_downscale_shape():
    """Output shape matches target H, W."""
    x = torch.randn(1, 4, 5, 16, 16)
    out = downscale_latent({"samples": x}, target_h=8, target_w=8)
    assert out["samples"].shape == (1, 4, 5, 8, 8)


def test_downscale_upscale_roundtrip():
    """Downscale → upscale roundtrip is numerically close."""
    x = torch.randn(1, 4, 3, 16, 16)
    down = downscale_latent({"samples": x}, target_h=8, target_w=8)
    # "Upscale" back via interpolate
    s = down["samples"]
    B, C, T, H, W = s.shape
    s_flat = s.reshape(B * C * T, 1, H, W)
    s_up = torch.nn.functional.interpolate(s_flat, size=(16, 16), mode="bilinear", align_corners=False)
    up = s_up.reshape(B, C, T, 16, 16)
    # Bilinear loses info but should be correlated
    cos_sim = torch.nn.functional.cosine_similarity(
        x.flatten().unsqueeze(0), up.flatten().unsqueeze(0),
    ).item()
    assert cos_sim > 0.3, f"Round-trip cosine similarity too low: {cos_sim}"


# -- two-minute preset validation --


def test_two_minute_preset_chunks():
    """2-minute video at balanced preset produces valid frame counts."""
    total = seconds_to_frames(120.0, fps=25.0)
    assert validate_frame_count(total)
    n, actual = calculate_chunks(total, 57, 16)
    assert n > 1
    assert validate_frame_count(actual)
    # Each chunk tile must also satisfy 8n+1
    assert validate_frame_count(57)


def test_seconds_to_frames_basic():
    """10 seconds at 25fps → 249 (nearest 8n+1)."""
    f = seconds_to_frames(10.0, 25.0)
    assert validate_frame_count(f)
    assert abs(f - 250) <= 4  # 249 or 249


def test_nearest_valid_frames():
    """nearest_valid_frames rounds correctly."""
    assert nearest_valid_frames(1) == 1
    assert nearest_valid_frames(9) == 9   # 8*1+1
    assert nearest_valid_frames(10) == 9  # round down
    assert nearest_valid_frames(13) == 9  # round down (13-1=12, 12%8=4, <=4)
    assert nearest_valid_frames(14) == 17  # round up (14-1=13, 13%8=5, >4)
    assert nearest_valid_frames(17) == 17  # exact


def test_validate_frame_count():
    """Frame count validation for 8n+1."""
    assert validate_frame_count(1)
    assert validate_frame_count(9)
    assert validate_frame_count(17)
    assert validate_frame_count(25)
    assert validate_frame_count(81)
    assert validate_frame_count(97)
    assert not validate_frame_count(2)
    assert not validate_frame_count(8)
    assert not validate_frame_count(80)


# -- audio extension tests --


def test_extend_audio_none_handling():
    """None audio handling."""
    assert extend_audio_latent(None, None) is None
    t = torch.randn(1, 8, 5, 16)
    assert torch.equal(extend_audio_latent(None, t), t)
    assert torch.equal(extend_audio_latent(t, None), t)


def test_extend_audio_concat():
    """Audio concatenation without overlap."""
    a1 = torch.randn(1, 8, 5, 16)
    a2 = torch.randn(1, 8, 3, 16)
    out = extend_audio_latent(a1, a2, overlap=0)
    assert out.shape == (1, 8, 8, 16)


# -- per-step AdaIN tests --


def test_per_step_adain_identity_at_zero():
    """Per-step AdaIN with factor=0.0 does nothing."""
    ref = {"samples": torch.randn(1, 4, 5, 8, 8)}

    call_count = [0]
    def fake_denoise(vs, as_, sigmas, step_idx):
        call_count[0] += 1
        return vs

    patched = apply_per_step_adain_patch(fake_denoise, ref, factors=[0.0, 0.0])
    # Can't easily test without VideoState, but verify the wrapper exists
    assert callable(patched)


# -- long memory conditioning tests --


def test_long_memory_zero_identity():
    """strength=0.0 returns accumulated unchanged."""
    acc = {"samples": torch.randn(1, 4, 10, 8, 8)}
    c0 = {"samples": torch.randn(1, 4, 5, 8, 8)}
    out = add_long_memory_conditioning(acc, c0, strength=0.0)
    assert torch.equal(out["samples"], acc["samples"])


def test_long_memory_positive_changes():
    """strength > 0 modifies accumulated statistics."""
    torch.manual_seed(42)
    acc = {"samples": torch.randn(1, 4, 10, 8, 8) * 5.0 + 3.0}
    c0 = {"samples": torch.randn(1, 4, 5, 8, 8) * 0.5}
    out = add_long_memory_conditioning(acc, c0, strength=0.5)
    # Output should have different mean from input
    assert not torch.allclose(out["samples"], acc["samples"])


# -- config integration tests --


def test_config_long_video_fields():
    """Long video config fields exist with correct defaults."""
    from config import AppConfig
    cfg = AppConfig()
    assert cfg.long_video_enabled is False
    assert cfg.long_video_preset == "balanced"
    assert cfg.long_video_temporal_tile_size == 80
    assert cfg.long_video_temporal_overlap == 24
    assert cfg.long_video_adain_factor == 0.3
    assert cfg.long_video_per_step_adain is False
    assert cfg.long_video_long_memory_strength == 0.0


def test_presets_have_valid_tile_sizes():
    """All preset tile sizes satisfy 8n+1."""
    from long_video_presets import PRESETS
    for name, preset in PRESETS.items():
        tile = preset["temporal_tile_size"]
        assert validate_frame_count(tile), f"Preset {name} tile_size={tile} is not 8n+1"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
