"""Runtime HQ helpers for loading the dev scaled-FP8 checkpoint.

The official LTX-2 HQ pipeline keeps the dev checkpoint intact and applies the
distilled LoRA at runtime. On this machine we do not have the dev BF16
checkpoint, only the scaled-FP8 variant. To avoid baking a lossy cast-only FP8
checkpoint, we dequantize the transformer's scaled-FP8 weights to BF16 during
load and let ltx-core fuse the LoRA into BF16 weights at runtime.
"""

from __future__ import annotations

from dataclasses import replace

import torch
from safetensors.torch import load_file

from ltx_core.loader import SDOps
from ltx_core.loader.sd_ops import KeyValueOperationResult

_LORA_PAIR_CACHE: dict[str, dict[str, tuple[torch.Tensor, torch.Tensor]]] = {}


def _make_scaled_fp8_to_bf16_op():
    pending_weights: dict[str, torch.Tensor] = {}
    pending_scales: dict[str, torch.Tensor] = {}

    def _emit(weight_key: str, weight: torch.Tensor, scale: torch.Tensor) -> list[KeyValueOperationResult]:
        dequantized = weight.to(torch.float32)
        dequantized.mul_(scale.to(torch.float32))
        return [KeyValueOperationResult(weight_key, dequantized.to(torch.bfloat16))]

    def op(key: str, value: torch.Tensor) -> list[KeyValueOperationResult]:
        if key.endswith(".input_scale"):
            return []

        if key.endswith(".weight_scale"):
            weight_key = key.replace(".weight_scale", ".weight")
            weight = pending_weights.pop(weight_key, None)
            if weight is None:
                pending_scales[weight_key] = value
                return []
            return _emit(weight_key, weight, value)

        if key.endswith(".weight") and value.dtype == torch.float8_e4m3fn:
            scale = pending_scales.pop(key, None)
            if scale is None:
                pending_weights[key] = value
                return []
            return _emit(key, value, scale)

        return [KeyValueOperationResult(key, value)]

    return op


def make_scaled_fp8_to_bf16_sd_ops() -> SDOps:
    """Create a state-dict transform that dequantizes scaled-FP8 weights to BF16."""
    return SDOps("scaled_fp8_to_bf16").with_kv_operation(_make_scaled_fp8_to_bf16_op())


def attach_runtime_scaled_fp8_dequantization(ledger) -> None:
    """Patch a ModelLedger transformer builder to dequantize scaled-FP8 on load."""
    builder = getattr(ledger, "transformer_builder", None)
    if builder is None:
        return

    dequant_ops = make_scaled_fp8_to_bf16_sd_ops()
    model_sd_ops = builder.model_sd_ops
    if model_sd_ops is None:
        chained = dequant_ops
    else:
        chained = SDOps(
            name=f"{model_sd_ops.name}+{dequant_ops.name}",
            mapping=(*model_sd_ops.mapping, *dequant_ops.mapping),
        )

    ledger.transformer_builder = replace(builder, model_sd_ops=chained)


def _normalize_lora_base_key(key: str) -> str:
    if key.startswith("diffusion_model."):
        key = key[len("diffusion_model."):]
    elif key.startswith("transformer."):
        key = key[len("transformer."):]

    key = key.replace(".lora_A.weight", ".weight")
    key = key.replace(".lora_B.weight", ".weight")
    key = key.replace(".lora_down.weight", ".weight")
    key = key.replace(".lora_up.weight", ".weight")
    return key


def _load_lora_pairs(path: str) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    cached = _LORA_PAIR_CACHE.get(path)
    if cached is not None:
        return cached

    sd = load_file(path, device="cpu")
    pairs: dict[str, list[torch.Tensor | None]] = {}
    for key, tensor in sd.items():
        if key.endswith(".lora_A.weight") or key.endswith(".lora_down.weight"):
            base = _normalize_lora_base_key(key)
            entry = pairs.setdefault(base, [None, None])
            entry[0] = tensor
        elif key.endswith(".lora_B.weight") or key.endswith(".lora_up.weight"):
            base = _normalize_lora_base_key(key)
            entry = pairs.setdefault(base, [None, None])
            entry[1] = tensor

    finalized = {
        base: (entry[0], entry[1])
        for base, entry in pairs.items()
        if entry[0] is not None and entry[1] is not None
    }
    _LORA_PAIR_CACHE[path] = finalized
    return finalized


def _make_runtime_lora_merge_op(
    lora_path: str,
    strength: float,
):
    lora_pairs = _load_lora_pairs(lora_path)

    def op(key: str, value: torch.Tensor) -> list[KeyValueOperationResult]:
        pair = lora_pairs.get(key)
        if pair is None:
            return [KeyValueOperationResult(key, value)]

        a, b = pair
        delta = torch.matmul(
            b.to(dtype=torch.float32, device=value.device) * strength,
            a.to(dtype=torch.float32, device=value.device),
        )
        merged = value.to(torch.float32)
        merged.add_(delta)
        return [KeyValueOperationResult(key, merged.to(dtype=value.dtype))]

    return op


def attach_runtime_lora_merge(
    ledger,
    *,
    lora_path: str,
    strength: float,
) -> None:
    """Fuse a LoRA into transformer weights during checkpoint load."""
    builder = getattr(ledger, "transformer_builder", None)
    if builder is None:
        return

    merge_ops = SDOps(f"runtime_lora_merge:{strength}").with_kv_operation(
        _make_runtime_lora_merge_op(lora_path, strength),
        key_suffix=".weight",
    )
    model_sd_ops = builder.model_sd_ops
    if model_sd_ops is None:
        chained = merge_ops
    else:
        chained = SDOps(
            name=f"{model_sd_ops.name}+{merge_ops.name}",
            mapping=(*model_sd_ops.mapping, *merge_ops.mapping),
        )

    ledger.transformer_builder = replace(builder, model_sd_ops=chained)
