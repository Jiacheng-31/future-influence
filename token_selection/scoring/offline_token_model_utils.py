"""Model-backbone helpers shared by Value-gate token scorers."""

from __future__ import annotations

from typing import Any


def unwrap_model(model: Any) -> Any:
    """Return the decoder backbone instead of the causal-LM wrapper."""
    while hasattr(model, "module"):
        model = model.module

    # Prefer the text decoder nested inside multimodal wrappers. Gemma-4
    # Unified stores it at model.language_model; causal LMs still match model.
    for path in ("model.language_model", "language_model", "model", "transformer"):
        candidate = model
        for part in path.split("."):
            candidate = getattr(candidate, part, None)
            if candidate is None:
                break
        if candidate is not None and candidate is not model and hasattr(candidate, "get_input_embeddings"):
            return candidate

    if hasattr(model, "get_base_model"):
        candidate = model.get_base_model()
        if candidate is not None and candidate is not model:
            nested = getattr(candidate, "model", None)
            if nested is not None and nested is not candidate and hasattr(
                nested, "get_input_embeddings"
            ):
                return nested
            return candidate
    return model


def decoder_layers(model: Any) -> Any:
    base_model = unwrap_model(model)
    for path in (
        "layers",
        "language_model.layers",
        "model.layers",
        "model.language_model.layers",
        "model.model.layers",
        "transformer.h",
    ):
        value = base_model
        for part in path.split("."):
            value = getattr(value, part, None)
            if value is None:
                break
        if value is not None:
            return value
    raise AttributeError("cannot find decoder layers for Value-gate token scoring")


def response_token_losses(
    model: Any,
    hidden_states: Any,
    input_ids: Any,
    target_positions: Any,
    chunk_size: int = 1024,
) -> Any:
    """Apply the frozen LM head only at requested response-target positions."""
    import torch
    import torch.nn.functional as F
    from torch.utils.checkpoint import checkpoint

    lm_head = model.get_output_embeddings()
    config = getattr(model, "config", None)
    text_config = getattr(config, "text_config", None) or config
    final_logit_softcapping = getattr(text_config, "final_logit_softcapping", None)

    def chunk_losses(hidden_chunk: Any, target_chunk: Any) -> Any:
        logits = lm_head(hidden_chunk).float()
        # Gemma-family causal LMs soft-cap logits before cross entropy.  The
        # scorer bypasses the LM wrapper to avoid materializing full-sequence
        # vocabulary logits, so it must reproduce that transformation here.
        # Qwen, MiMo and Llama configs leave this unset and retain the existing
        # unmodified CE path.
        if final_logit_softcapping is not None:
            logits = (
                torch.tanh(logits / final_logit_softcapping)
                * final_logit_softcapping
            )
        return F.cross_entropy(logits, target_chunk, reduction="none")

    losses = []
    use_checkpoint = torch.is_grad_enabled() and target_positions.numel() > chunk_size
    for start in range(0, target_positions.numel(), chunk_size):
        positions = target_positions[start : start + chunk_size]
        hidden_chunk = hidden_states[0, positions - 1, :]
        target_chunk = input_ids[0, positions]
        if use_checkpoint:
            losses.append(
                checkpoint(
                    chunk_losses,
                    hidden_chunk,
                    target_chunk,
                    use_reentrant=False,
                )
            )
        else:
            losses.append(chunk_losses(hidden_chunk, target_chunk))
    return torch.cat(losses)
