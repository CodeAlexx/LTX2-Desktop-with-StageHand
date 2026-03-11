"""Long video generation presets and frame arithmetic."""

from __future__ import annotations

from long_video import UpscaleMode

PRESETS = {
    "quality": {
        "temporal_tile_size": 97,
        "temporal_overlap": 32,
        "overlap_cond_strength": 0.5,
        "adain_factor": 0.0,
        "steps": 8,
        "upscale_mode": UpscaleMode.SPATIAL_PER_CHUNK,
        "notes": (
            "Best coherence, slowest. 64 new base frames per chunk. "
            "At 25fps: ~2.56s new content per chunk. "
            "Output: 1536x1024 25fps. "
            "2 minutes ~ 47 chunks, ~3-4 hours on 3090."
        ),
    },
    "balanced": {
        "temporal_tile_size": 81,
        "temporal_overlap": 24,
        "overlap_cond_strength": 0.5,
        "adain_factor": 0.0,
        "steps": 8,
        "upscale_mode": UpscaleMode.SPATIAL_PER_CHUNK,
        "notes": (
            "56 new base frames per chunk = ~2.24s new content. "
            "Output: 1536x1024 25fps. "
            "2 minutes ~ 54 chunks, ~2.5-3.5 hours on 3090. "
            "Recommended default."
        ),
    },
    "fast": {
        "temporal_tile_size": 57,
        "temporal_overlap": 16,
        "overlap_cond_strength": 0.4,
        "adain_factor": 0.0,
        "steps": 6,
        "upscale_mode": UpscaleMode.SPATIAL_PER_CHUNK,
        "notes": (
            "41 new base frames per chunk = ~1.64s new content. "
            "Output: 1536x1024 25fps. "
            "2 minutes ~ 74 chunks, ~2-3 hours on 3090. "
            "More temporal drift risk with smaller overlap."
        ),
    },
    "two_minute": {
        "temporal_tile_size": 57,
        "temporal_overlap": 16,
        "overlap_cond_strength": 0.4,
        "adain_factor": 0.4,
        "per_step_adain": True,
        "per_step_adain_factors": "0.9,0.75,0.5,0.25,0.0,0.0",
        "long_memory_strength": 0.5,
        "steps": 6,
        "upscale_mode": UpscaleMode.SPATIAL_PER_CHUNK,
        "notes": (
            "Tuned for 90-120 second targets. "
            "All drift-prevention mechanisms enabled. "
            "Output: 1536x1024 25fps. "
            "Expected wall time: 2-4 hours on 3090."
        ),
    },
    "max_length": {
        "temporal_tile_size": 57,
        "temporal_overlap": 16,
        "overlap_cond_strength": 0.4,
        "adain_factor": 0.5,
        "per_step_adain": True,
        "per_step_adain_factors": "0.9,0.75,0.5,0.25,0.0,0.0",
        "long_memory_strength": 0.7,
        "steps": 6,
        "upscale_mode": UpscaleMode.NONE,
        "notes": (
            "For 3+ minute targets where coherence matters more than resolution. "
            "Base resolution (768x512) throughout. "
            "Strongest drift-prevention settings. "
            "Apply SPATIAL_FINAL upscale separately if higher res needed."
        ),
    },
}


def validate_frame_count(frames: int) -> bool:
    """LTX-2.3 requires frames = 8n+1."""
    return (frames - 1) % 8 == 0


def nearest_valid_frames(n: int) -> int:
    """Round n to nearest valid LTX-2.3 frame count (8n+1)."""
    remainder = (n - 1) % 8
    if remainder <= 4:
        return n - remainder
    return n + (8 - remainder)


def calculate_chunks(
    total_frames: int,
    tile_size: int,
    overlap: int,
) -> tuple[int, int]:
    """Calculate chunk count and actual total frames.

    Chunk 0 generates tile_size frames.
    Each subsequent chunk adds (tile_size - overlap) new frames.

    Returns (num_chunks, actual_total_frames).
    """
    if total_frames <= tile_size:
        return 1, nearest_valid_frames(total_frames)

    remaining = total_frames - tile_size
    step = tile_size - overlap
    num_extension_chunks = (remaining + step - 1) // step
    actual = tile_size + num_extension_chunks * step
    return 1 + num_extension_chunks, nearest_valid_frames(actual)


def seconds_to_frames(seconds: float, fps: float = 25.0) -> int:
    """Convert seconds to nearest valid LTX-2.3 frame count."""
    raw = int(seconds * fps)
    return nearest_valid_frames(raw)


PERFORMANCE_ESTIMATES = {
    "seconds_per_chunk_768x512_8steps": (30, 60),
    "seconds_per_chunk_768x512_6steps": (22, 45),
    "spatial_upscale_overhead_seconds": (5, 15),
}
