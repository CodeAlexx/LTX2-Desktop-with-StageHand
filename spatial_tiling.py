"""Spatial tiling — overlapping tiles for high-res denoising, linear blend."""

from __future__ import annotations

import torch
from dataclasses import dataclass

__all__ = ["TileSpec", "compute_tiles", "blend_tiles", "needs_tiling"]

SPATIAL_SCALE = 32  # LTX-2 pixel-to-latent ratio


@dataclass
class TileSpec:
    """A spatial tile region in pixel space (all values divisible by 32)."""
    y0: int
    y1: int
    x0: int
    x1: int

    @property
    def h(self) -> int:
        return self.y1 - self.y0

    @property
    def w(self) -> int:
        return self.x1 - self.x0


def needs_tiling(h_pixels: int, w_pixels: int, tile_pixels: int) -> bool:
    """Return True if the resolution exceeds a single tile."""
    return h_pixels > tile_pixels or w_pixels > tile_pixels


def compute_tiles(
    h_pixels: int, w_pixels: int,
    tile_pixels: int = 512, overlap_pixels: int = 128,
) -> list[TileSpec]:
    """Compute tile positions with overlap, aligned to 32px boundaries."""
    tile_pixels = _align(tile_pixels)
    overlap_pixels = _align(min(overlap_pixels, tile_pixels // 2))
    ys = _tiles_1d(h_pixels, tile_pixels, overlap_pixels)
    xs = _tiles_1d(w_pixels, tile_pixels, overlap_pixels)
    return [TileSpec(y0, y1, x0, x1) for y0, y1 in ys for x0, x1 in xs]


def blend_tiles(
    tiles: list[TileSpec],
    tile_latents: list[torch.Tensor],
    h_pixels: int,
    w_pixels: int,
) -> torch.Tensor:
    """Blend tile latents [B,C,F,th,tw] into full latent using linear weight ramps."""
    ref = tile_latents[0]
    B, C, F = ref.shape[:3]
    h_lat = h_pixels // SPATIAL_SCALE
    w_lat = w_pixels // SPATIAL_SCALE
    S = SPATIAL_SCALE

    out = torch.zeros(B, C, F, h_lat, w_lat, device=ref.device, dtype=ref.dtype)
    wsum = torch.zeros(1, 1, 1, h_lat, w_lat, device=ref.device, dtype=ref.dtype)

    # Collect all tile boundaries for overlap detection
    all_y = [(t.y0, t.y1) for t in tiles]
    all_x = [(t.x0, t.x1) for t in tiles]

    for tile, lat in zip(tiles, tile_latents):
        y0l, y1l = tile.y0 // S, tile.y1 // S
        x0l, x1l = tile.x0 // S, tile.x1 // S
        wy = _edge_ramp(y1l - y0l, tile.y0, tile.y1, all_y, ref.device, ref.dtype)
        wx = _edge_ramp(x1l - x0l, tile.x0, tile.x1, all_x, ref.device, ref.dtype)
        mask = wy[:, None] * wx[None, :]  # [th, tw]
        mask = mask[None, None, None, :, :]  # broadcast [1,1,1,th,tw]
        out[:, :, :, y0l:y1l, x0l:x1l] += lat * mask
        wsum[:, :, :, y0l:y1l, x0l:x1l] += mask

    return out / wsum.clamp(min=1e-8)


# ---- internal helpers ----

def _align(v: int) -> int:
    """Round down to nearest SPATIAL_SCALE multiple (min SPATIAL_SCALE)."""
    return max(SPATIAL_SCALE, (v // SPATIAL_SCALE) * SPATIAL_SCALE)


def _tiles_1d(size: int, tile: int, overlap: int) -> list[tuple[int, int]]:
    """Tile start/end positions along one axis."""
    if size <= tile:
        return [(0, size)]
    stride = tile - overlap
    result: list[tuple[int, int]] = []
    pos = 0
    while pos < size:
        end = min(pos + tile, size)
        result.append((pos, end))
        if end >= size:
            break
        pos += stride
    return result


def _edge_ramp(
    size_lat: int,
    start_px: int,
    end_px: int,
    all_ranges: list[tuple[int, int]],
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Build 1D weight with linear ramps where this tile overlaps neighbours."""
    w = torch.ones(size_lat, device=device, dtype=dtype)
    # Overlap at start: another tile ends inside our region
    for s, e in all_ranges:
        if s < start_px < e < end_px:
            n = (e - start_px) // SPATIAL_SCALE
            if n > 0 and n <= size_lat:
                w[:n] = torch.linspace(0, 1, n + 2, device=device, dtype=dtype)[1:n + 1]
            break
    # Overlap at end: another tile starts inside our region
    for s, e in all_ranges:
        if start_px < s < end_px < e:
            n = (end_px - s) // SPATIAL_SCALE
            if n > 0 and n <= size_lat:
                w[-n:] = torch.linspace(1, 0, n + 2, device=device, dtype=dtype)[1:n + 1]
            break
    return w
