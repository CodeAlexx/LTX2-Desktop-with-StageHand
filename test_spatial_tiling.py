"""Tests for spatial tiling — no GPU required."""
from __future__ import annotations

import sys
from pathlib import Path

_app = str(Path(__file__).parent)
if _app not in sys.path:
    sys.path.insert(0, _app)

import torch
from spatial_tiling import TileSpec, compute_tiles, blend_tiles, needs_tiling, SPATIAL_SCALE


def test_no_tiling_when_small():
    """Single tile returned when resolution fits."""
    tiles = compute_tiles(512, 768, tile_pixels=768, overlap_pixels=128)
    assert len(tiles) == 1
    assert tiles[0] == TileSpec(0, 512, 0, 768)


def test_needs_tiling_flag():
    assert not needs_tiling(512, 768, tile_pixels=768)
    assert needs_tiling(1024, 1024, tile_pixels=512)


def test_tiles_cover_full_area():
    """Tiles must cover every pixel of the output."""
    h, w = 1024, 1536
    tiles = compute_tiles(h, w, tile_pixels=512, overlap_pixels=128)
    assert len(tiles) > 1
    # Check coverage: every latent pixel must be in at least one tile
    S = SPATIAL_SCALE
    covered_y = set()
    covered_x = set()
    for t in tiles:
        assert t.y0 % S == 0 and t.y1 % S == 0, "Y not aligned to 32"
        assert t.x0 % S == 0 and t.x1 % S == 0, "X not aligned to 32"
        for y in range(t.y0, t.y1):
            covered_y.add(y)
        for x in range(t.x0, t.x1):
            covered_x.add(x)
    assert max(covered_y) >= h - 1, "Tiles don't cover full height"
    assert max(covered_x) >= w - 1, "Tiles don't cover full width"


def test_tile_alignment():
    """All tile boundaries must be divisible by SPATIAL_SCALE (32)."""
    tiles = compute_tiles(1024, 1536, tile_pixels=512, overlap_pixels=128)
    S = SPATIAL_SCALE
    for t in tiles:
        assert t.y0 % S == 0
        assert t.y1 % S == 0
        assert t.x0 % S == 0
        assert t.x1 % S == 0


def test_blend_single_tile():
    """Single tile blend should be identity."""
    h, w = 512, 768
    tiles = compute_tiles(h, w, tile_pixels=768, overlap_pixels=128)
    latent = torch.randn(1, 128, 3, h // SPATIAL_SCALE, w // SPATIAL_SCALE)
    result = blend_tiles(tiles, [latent], h, w)
    assert torch.allclose(result, latent, atol=1e-5)


def test_blend_preserves_shape():
    """Blended output must match full resolution."""
    h, w = 1024, 1024
    tiles = compute_tiles(h, w, tile_pixels=512, overlap_pixels=128)
    S = SPATIAL_SCALE
    tile_latents = [
        torch.randn(1, 128, 3, t.h // S, t.w // S) for t in tiles
    ]
    result = blend_tiles(tiles, tile_latents, h, w)
    assert result.shape == (1, 128, 3, h // S, w // S)


def test_blend_uniform_value():
    """Blending tiles with constant value should produce that constant."""
    h, w = 1024, 1024
    tiles = compute_tiles(h, w, tile_pixels=512, overlap_pixels=128)
    S = SPATIAL_SCALE
    val = 3.14
    tile_latents = [
        torch.full((1, 4, 2, t.h // S, t.w // S), val) for t in tiles
    ]
    result = blend_tiles(tiles, tile_latents, h, w)
    assert torch.allclose(result, torch.full_like(result, val), atol=1e-4), \
        f"Blend of constant tiles should be constant, got range [{result.min():.4f}, {result.max():.4f}]"


if __name__ == "__main__":
    test_no_tiling_when_small()
    test_needs_tiling_flag()
    test_tiles_cover_full_area()
    test_tile_alignment()
    test_blend_single_tile()
    test_blend_preserves_shape()
    test_blend_uniform_value()
    print("All spatial_tiling tests passed!")
