"""Normalized Attention Guidance for LTX-2.3.

Patches cross-attention (attn2) modules on each transformer block to apply
CFG-style guidance in attention space using a null text encoding as baseline.

Source: ComfyUI-KJNodes/nodes/ltxv_nodes.py (adapted for ltx_core Attention)
Paper: https://github.com/ChenDarYen/Normalized-Attention-Guidance
Default params (all community workflows): scale=11, alpha=0.25, tau=2.5
"""

from __future__ import annotations

import logging
from typing import Any

import torch

logger = logging.getLogger(__name__)


def _nag_combine(
    x_pos: torch.Tensor,
    x_neg: torch.Tensor,
    scale: float,
    alpha: float,
    tau: float,
) -> torch.Tensor:
    """CFG in attention output space with L1-norm clipping.

    guidance = pos * scale - neg * (scale - 1)
    if ||guidance|| / ||pos|| > tau: rescale guidance down
    output = guidance * alpha + pos * (1 - alpha)
    """
    guidance = x_pos * scale - x_neg * (scale - 1)

    norm_pos = torch.norm(x_pos, p=1, dim=-1, keepdim=True)
    norm_guid = torch.norm(guidance, p=1, dim=-1, keepdim=True)

    ratio = norm_guid / (norm_pos + 1e-7)
    mask = ratio > tau

    adjustment = (norm_pos * tau) / (norm_guid + 1e-7)
    guidance = torch.where(mask, guidance * adjustment, guidance)

    # Blend: output = guidance * alpha + pos * (1 - alpha)
    return guidance * alpha + x_pos * (1 - alpha)


class NAGPatch:
    """Manages NAG patches on a transformer's cross-attention modules."""

    def __init__(
        self,
        nag_v_ctx: torch.Tensor,
        nag_a_ctx: torch.Tensor | None,
        scale: float,
        alpha: float,
        tau: float,
    ):
        self.nag_v_ctx = nag_v_ctx
        self.nag_a_ctx = nag_a_ctx
        self.scale = scale
        self.alpha = alpha
        self.tau = tau
        self._originals: dict[int, Any] = {}  # id(module) -> original forward

    def _make_nag_forward(self, module: Any, nag_ctx: torch.Tensor) -> Any:
        """Create a NAG-wrapped forward for an Attention module."""
        original = module.forward
        scale, alpha, tau = self.scale, self.alpha, self.tau

        def nag_forward(
            x: torch.Tensor,
            context: torch.Tensor | None = None,
            mask: torch.Tensor | None = None,
            pe: torch.Tensor | None = None,
            k_pe: torch.Tensor | None = None,
            perturbation_mask: torch.Tensor | None = None,
            all_perturbed: bool = False,
        ) -> torch.Tensor:
            # Standard attention with positive context
            out_pos = original(x, context, mask, pe, k_pe, perturbation_mask, all_perturbed)

            # Attention with null context (no mask needed for null)
            ctx = nag_ctx.to(device=x.device, dtype=x.dtype)
            if ctx.shape[0] != x.shape[0]:
                ctx = ctx.expand(x.shape[0], -1, -1)
            out_neg = original(x, ctx, None, pe, k_pe, perturbation_mask, all_perturbed)

            return _nag_combine(out_pos, out_neg, scale, alpha, tau)

        return nag_forward

    def apply(self, transformer: Any) -> None:
        """Patch attn2 (and audio_attn2) on all transformer blocks."""
        inner = _find_block_container(transformer)
        blocks = getattr(inner, "transformer_blocks", None)
        if blocks is None:
            logger.warning("No transformer_blocks found — NAG not applied")
            return

        count = 0
        for block in blocks:
            # Video text cross-attention
            attn2 = getattr(block, "attn2", None)
            if attn2 is not None:
                self._originals[id(attn2)] = attn2.forward
                attn2.forward = self._make_nag_forward(attn2, self.nag_v_ctx)
                count += 1

            # Audio text cross-attention
            if self.nag_a_ctx is not None:
                audio_attn2 = getattr(block, "audio_attn2", None)
                if audio_attn2 is not None:
                    self._originals[id(audio_attn2)] = audio_attn2.forward
                    audio_attn2.forward = self._make_nag_forward(audio_attn2, self.nag_a_ctx)

        logger.info("NAG applied to %d cross-attention modules (scale=%.1f, alpha=%.2f, tau=%.1f)",
                     count, self.scale, self.alpha, self.tau)

    def remove(self, transformer: Any) -> None:
        """Restore all patched modules to original forward."""
        inner = _find_block_container(transformer)
        blocks = getattr(inner, "transformer_blocks", None)
        if blocks is None:
            return

        for block in blocks:
            for attr_name in ("attn2", "audio_attn2"):
                module = getattr(block, attr_name, None)
                if module is not None and id(module) in self._originals:
                    self._originals.pop(id(module))
                    # Remove instance override so class method is used again
                    module.__dict__.pop("forward", None)

        logger.info("NAG removed")


def _find_block_container(transformer: Any) -> Any:
    """Navigate X0Model -> velocity_model (LTXModel with transformer_blocks)."""
    for attr in ("velocity_model", "model", "module"):
        inner = getattr(transformer, attr, None)
        if inner is not None:
            if hasattr(inner, "transformer_blocks"):
                return inner
            for attr2 in ("velocity_model", "model", "module"):
                inner2 = getattr(inner, attr2, None)
                if inner2 is not None and hasattr(inner2, "transformer_blocks"):
                    return inner2
    if hasattr(transformer, "transformer_blocks"):
        return transformer
    return transformer  # return as-is, apply() will warn
