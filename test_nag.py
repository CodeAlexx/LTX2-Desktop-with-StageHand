"""Tests for NAG — no GPU required."""

from __future__ import annotations

import torch
import torch.nn as nn
from nag import _nag_combine, NAGPatch, _find_block_container


def test_nag_combine_output_shape():
    """Output shape matches input shape."""
    B, T, D = 1, 16, 64
    x_pos = torch.randn(B, T, D)
    x_neg = torch.randn(B, T, D)
    out = _nag_combine(x_pos, x_neg, scale=11.0, alpha=0.25, tau=2.5)
    assert out.shape == (B, T, D)


def test_nag_combine_clips_magnitude():
    """With tau=1.0, guidance norm should not exceed positive norm."""
    B, T, D = 2, 8, 32
    x_pos = torch.randn(B, T, D)
    x_neg = torch.randn(B, T, D) * 5  # large negative
    out = _nag_combine(x_pos, x_neg, scale=11.0, alpha=1.0, tau=1.0)
    norm_out = torch.norm(out, p=1, dim=-1)
    norm_pos = torch.norm(x_pos, p=1, dim=-1)
    # After clipping at tau=1.0 with alpha=1.0 (pure guidance), should be bounded
    assert torch.all(norm_out <= norm_pos * 1.05)  # 5% tolerance for float ops


def test_nag_combine_identity_at_scale_1():
    """At scale=1.0, guidance = pos (no negative contribution)."""
    B, T, D = 1, 4, 16
    x_pos = torch.randn(B, T, D)
    x_neg = torch.randn(B, T, D)
    # scale=1 means guidance = pos*1 - neg*0 = pos
    out = _nag_combine(x_pos, x_neg, scale=1.0, alpha=1.0, tau=10.0)
    assert torch.allclose(out, x_pos, atol=1e-6)


def test_nag_combine_alpha_zero():
    """At alpha=0.0, output = positive (no guidance blend)."""
    B, T, D = 1, 4, 16
    x_pos = torch.randn(B, T, D)
    x_neg = torch.randn(B, T, D) * 3
    out = _nag_combine(x_pos, x_neg, scale=11.0, alpha=0.0, tau=2.5)
    assert torch.allclose(out, x_pos, atol=1e-6)


class FakeAttention(nn.Module):
    """Minimal Attention stand-in for testing NAG patch."""
    def __init__(self, dim: int = 64):
        super().__init__()
        self.linear = nn.Linear(dim, dim)

    def forward(self, x, context=None, mask=None, pe=None, k_pe=None,
                perturbation_mask=None, all_perturbed=False):
        ctx = context if context is not None else x
        return self.linear(ctx.mean(dim=1, keepdim=True).expand_as(x))


class FakeBlock(nn.Module):
    def __init__(self, dim=64):
        super().__init__()
        self.attn2 = FakeAttention(dim)
        self.audio_attn2 = FakeAttention(dim)


class FakeTransformer(nn.Module):
    def __init__(self, n_blocks=4, dim=64):
        super().__init__()
        self.transformer_blocks = nn.ModuleList([FakeBlock(dim) for _ in range(n_blocks)])


def test_nag_patch_apply_and_remove():
    """Patch changes forward; remove restores it."""
    xfm = FakeTransformer(n_blocks=2, dim=32)
    attn2 = xfm.transformer_blocks[0].attn2
    # Check that forward is initially the class method (not instance-overridden)
    assert "forward" not in attn2.__dict__

    nag_v = torch.zeros(1, 4, 32)
    nag_a = torch.zeros(1, 4, 32)
    patch = NAGPatch(nag_v, nag_a, scale=11.0, alpha=0.25, tau=2.5)
    patch.apply(xfm)

    # Forward should be patched (instance attribute overrides class method)
    assert "forward" in attn2.__dict__

    # Should still produce valid output
    x = torch.randn(1, 8, 32)
    ctx = torch.randn(1, 4, 32)
    out = attn2.forward(x, context=ctx)
    assert out.shape == x.shape

    # Remove should restore (remove instance override, class method used again)
    patch.remove(xfm)
    assert "forward" not in attn2.__dict__


def test_nag_patch_with_wrapped_transformer():
    """NAG should navigate X0Model -> inner to find transformer_blocks."""
    inner = FakeTransformer(n_blocks=2)

    class Wrapper(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.velocity_model = model

    wrapper = Wrapper(inner)
    found = _find_block_container(wrapper)
    assert hasattr(found, "transformer_blocks")


def test_config_nag_fields():
    """Config has NAG fields with correct defaults."""
    from config import AppConfig
    cfg = AppConfig()
    assert cfg.nag_enabled is True
    assert cfg.nag_scale == 11.0
    assert cfg.nag_alpha == 0.25
    assert cfg.nag_tau == 2.5


def test_config_save_load(tmp_path):
    """Config round-trips through save/load."""
    from config import AppConfig
    cfg = AppConfig()
    cfg.nag_scale = 15.0
    cfg.distilled_lora_strength = 0.3
    cfg.pipeline_mode = "dev"

    p = tmp_path / "test_config.json"
    cfg.save(p)

    loaded = AppConfig.load(p)
    assert loaded.nag_scale == 15.0
    assert loaded.distilled_lora_strength == 0.3
    assert loaded.pipeline_mode == "dev"


if __name__ == "__main__":
    test_nag_combine_output_shape()
    test_nag_combine_clips_magnitude()
    test_nag_combine_identity_at_scale_1()
    test_nag_combine_alpha_zero()
    test_nag_patch_apply_and_remove()
    test_nag_patch_with_wrapped_transformer()
    test_config_nag_fields()
    print("All tests passed!")
