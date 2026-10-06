"""Shared per-token gates on every decoder layer's attention Value projection."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any

from offline_token_model_utils import decoder_layers


def value_gate_points(model: Any) -> list[Any]:
    """Return one module whose output is the final Value stream per layer.

    Models such as Gemma4 Unified RMS-normalize Value states after projection;
    some global-attention layers also derive Value from the Key projection and
    have no ``v_proj``. Hooking ``v_norm`` handles both cases without changing
    the Key stream. Other decoder families retain the existing ``v_proj`` hook.
    """
    gate_points = []
    for layer_index, layer in enumerate(decoder_layers(model)):
        attention = getattr(layer, "self_attn", None)
        if attention is None:
            attention = getattr(layer, "attention", None)
        gate_point = getattr(attention, "v_norm", None) if attention is not None else None
        if gate_point is None:
            gate_point = getattr(attention, "v_proj", None) if attention is not None else None
        if gate_point is None:
            raise AttributeError(
                "cannot find an independent post-Key Value path "
                f"(v_norm or v_proj) in decoder layer {layer_index}"
            )
        gate_points.append(gate_point)
    if not gate_points:
        raise AttributeError("model has no decoder attention Value gate points")
    return gate_points


@contextmanager
def gated_value_flow(model: Any, gate: Any, response_mask: Any):
    """Multiply each response token's V vector by one shared scalar gate."""
    effective_gate = gate * response_mask + (1.0 - response_mask)
    scale = effective_gate.unsqueeze(-1)

    def apply_gate(_module: Any, _inputs: Any, output: Any) -> Any:
        # Most v_proj modules return [batch, sequence, hidden], while Gemma4's
        # post-split v_norm returns [batch, sequence, heads, head_dim].  Append
        # singleton dimensions so the same per-token gate broadcasts only over
        # feature axes for either representation.
        output_scale = scale
        while output_scale.ndim < output.ndim:
            output_scale = output_scale.unsqueeze(-1)
        return output * output_scale.to(device=output.device, dtype=output.dtype)

    handles = [
        gate_point.register_forward_hook(apply_gate)
        for gate_point in value_gate_points(model)
    ]
    try:
        yield
    finally:
        for handle in handles:
            handle.remove()
