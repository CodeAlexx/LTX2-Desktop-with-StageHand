"""Chunked FFN forward — reduces peak activation VRAM on long sequences.
Source: ComfyUI-KJNodes LTXVChunkFeedForward. Works with any transformer with .ff blocks.
"""
from __future__ import annotations

import logging
import types

import torch

logger = logging.getLogger(__name__)


def _ffn_chunked_forward(self, x: torch.Tensor) -> torch.Tensor:
    """Replacement forward() that chunks FFN when sequence > threshold."""
    if x.shape[1] <= self._ffn_chunk_threshold:
        return self.net(x)

    n = self._ffn_num_chunks
    seq_len = x.shape[1]
    chunk_size = seq_len // n

    for i in range(n):
        start = i * chunk_size
        end = (i + 1) * chunk_size if i < n - 1 else seq_len
        x[:, start:end] = self.net(x[:, start:end])
    return x


def apply_ffn_chunking(model, num_chunks: int = 2, threshold: int = 4096) -> int:
    """Patch all FeedForward blocks to use chunked forward. Returns count patched."""
    if num_chunks <= 1:
        return 0

    inner = _find_transformer(model)
    if inner is None:
        logger.warning("apply_ffn_chunking: no transformer found in %s", type(model).__name__)
        return 0

    blocks = (getattr(inner, "transformer_blocks", None)
              or getattr(inner, "blocks", None)
              or getattr(inner, "layers", None))
    if blocks is None:
        logger.warning("apply_ffn_chunking: no block list on %s", type(inner).__name__)
        return 0

    patched = 0
    for block in blocks:
        ff = getattr(block, "ff", None)
        if ff is None:
            continue
        ff._ffn_num_chunks = num_chunks
        ff._ffn_chunk_threshold = threshold
        ff.forward = types.MethodType(_ffn_chunked_forward, ff)
        patched += 1

    if patched:
        logger.info("apply_ffn_chunking: patched %d blocks (chunks=%d, threshold=%d)",
                    patched, num_chunks, threshold)
    return patched


def remove_ffn_chunking(model) -> None:
    """Remove chunking patch — restore default FFN forward."""
    inner = _find_transformer(model)
    if inner is None:
        return
    blocks = (getattr(inner, "transformer_blocks", None)
              or getattr(inner, "blocks", None)
              or getattr(inner, "layers", None))
    if blocks is None:
        return
    for block in blocks:
        ff = getattr(block, "ff", None)
        if ff is not None and hasattr(ff, "_ffn_num_chunks"):
            del ff._ffn_num_chunks, ff._ffn_chunk_threshold
            if "forward" in ff.__dict__:
                del ff.__dict__["forward"]


def _find_transformer(model):
    """Navigate to the module that holds transformer_blocks."""
    m = getattr(model, "diffusion_model", model)
    for attr in ("velocity_model", "model"):
        inner = getattr(m, attr, None)
        if inner is not None and hasattr(inner, "transformer_blocks"):
            return inner
    if hasattr(m, "transformer_blocks"):
        return m
    for attr in ("velocity_model", "model", "module"):
        inner = getattr(m, attr, None)
        if inner is None:
            continue
        for attr2 in ("velocity_model", "model"):
            inner2 = getattr(inner, attr2, None)
            if inner2 is not None and hasattr(inner2, "transformer_blocks"):
                return inner2
    return None
