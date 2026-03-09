"""Tests for chunked FFN — no GPU required."""
from __future__ import annotations

import sys
from pathlib import Path

_app = str(Path(__file__).parent)
if _app not in sys.path:
    sys.path.insert(0, _app)

import torch
from chunk_ffn import apply_ffn_chunking, remove_ffn_chunking


class FakeFeedForward(torch.nn.Module):
    def __init__(self, dim=64):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(dim, dim * 4),
            torch.nn.GELU(),
            torch.nn.Linear(dim * 4, dim),
        )

    def forward(self, x):
        return self.net(x)


class FakeBlock(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.ff = FakeFeedForward()


class FakeTransformer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.transformer_blocks = torch.nn.ModuleList([FakeBlock() for _ in range(4)])


def test_chunked_output_matches_unchunked():
    """Chunked FFN must produce same output as unchunked."""
    torch.manual_seed(0)
    model = FakeTransformer()
    x = torch.randn(1, 100, 64)

    original_ff = model.transformer_blocks[0].ff
    ref = original_ff(x.clone())

    n = apply_ffn_chunking(model, num_chunks=4, threshold=10)
    assert n == 4, f"Expected 4 blocks patched, got {n}"

    chunked = model.transformer_blocks[0].ff(x.clone())
    assert torch.allclose(ref, chunked, atol=1e-5), "Chunked output differs from unchunked"


def test_no_chunking_below_threshold():
    """When seq_len < threshold, chunking is not applied."""
    model = FakeTransformer()
    apply_ffn_chunking(model, num_chunks=2, threshold=10000)
    x = torch.randn(1, 50, 64)
    out = model.transformer_blocks[0].ff(x)
    assert out.shape == (1, 50, 64)


def test_chunks_1_is_noop():
    """num_chunks=1 should not patch any blocks."""
    model = FakeTransformer()
    n = apply_ffn_chunking(model, num_chunks=1)
    assert n == 0
    assert not hasattr(model.transformer_blocks[0].ff, "_ffn_num_chunks")


def test_remove_restores_forward():
    """remove_ffn_chunking restores original forward method."""
    model = FakeTransformer()
    ff = model.transformer_blocks[0].ff
    apply_ffn_chunking(model, num_chunks=2, threshold=10)
    assert hasattr(ff, "_ffn_num_chunks")
    remove_ffn_chunking(model)
    assert not hasattr(ff, "_ffn_num_chunks")
    assert "forward" not in ff.__dict__


if __name__ == "__main__":
    test_chunked_output_matches_unchunked()
    test_no_chunking_below_threshold()
    test_chunks_1_is_noop()
    test_remove_restores_forward()
    print("All chunk_ffn tests passed!")
