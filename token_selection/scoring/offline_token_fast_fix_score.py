#!/usr/bin/env python3
"""One-pass token scores from gates on every attention Value projection."""

from __future__ import annotations

import argparse
import json
from typing import Any

from offline_token_model_utils import (
    decoder_layers,
    response_token_losses,
    unwrap_model,
)
from offline_token_score_common import (
    add_common_arguments,
    make_input,
    sample_header,
    write_results,
)
from offline_token_value_gate import gated_value_flow


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run one forward with an all-one per-token gate on every decoder "
            "attention Value projection, then obtain all token gradients in "
            "one backward pass."
        )
    )
    add_common_arguments(parser)
    return parser.parse_args()


def score_one(
    *,
    model: Any,
    tokenizer: Any,
    device: Any,
    sample: Any,
    record: dict[str, Any],
    args: argparse.Namespace,
    score_definition: str,
) -> dict[str, Any]:
    import math
    import torch

    del tokenizer
    input_ids = make_input(sample, device)
    gate_mask = torch.zeros_like(input_ids, dtype=torch.float32)
    gate_mask[:, sample.response_positions] = 1.0
    gate = torch.ones_like(gate_mask, dtype=torch.float32, requires_grad=True)
    target_positions = torch.tensor(
        sample.response_target_positions,
        device=device,
        dtype=torch.long,
    )

    if not getattr(model, "_token_score_parameters_frozen", False):
        model.requires_grad_(False)
        model.eval()
        model._token_score_parameters_frozen = True

    checkpoint_layers = decoder_layers(model)
    checkpoint_min_length = getattr(args, "activation_checkpoint_min_length", 8192)
    use_activation_checkpointing = (
        input_ids.shape[1] >= checkpoint_min_length
        and hasattr(model, "gradient_checkpointing_enable")
    )
    if use_activation_checkpointing and not getattr(
        model, "is_gradient_checkpointing", False
    ):
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    for layer in checkpoint_layers:
        layer.training = use_activation_checkpointing

    model.zero_grad(set_to_none=True)
    with torch.enable_grad(), gated_value_flow(model, gate, gate_mask):
        outputs = unwrap_model(model)(
            input_ids=input_ids,
            attention_mask=input_ids.new_ones(input_ids.shape),
            use_cache=False,
        )
        token_losses = response_token_losses(
            model,
            outputs.last_hidden_state,
            input_ids,
            target_positions,
            chunk_size=getattr(args, "lm_head_chunk_size", 1024),
        )
        # Use the same summed response-target NLL scale as finite Value ablation.
        # Causality makes each token gate's derivative depend only on targets
        # after that token, without a per-sample target-count normalization.
        response_loss = token_losses.sum()
        gradient = torch.autograd.grad(
            response_loss,
            gate,
            retain_graph=False,
            create_graph=False,
        )[0][0].detach().cpu()
    baseline_response_targets = token_losses.detach().float().cpu()

    tokens: list[float] = []
    for position in sample.response_positions:
        signed_gradient = float(gradient[position].item())
        if not math.isfinite(signed_gradient):
            raise FloatingPointError(
                f"non-finite Value-gate gradient at sequence position {position}: "
                f"{signed_gradient}"
            )
        # The lossless form is the signed first-order loss change. A negative
        # value means decreasing this token's Value gate is predicted to reduce
        # NLL. ``positive`` exists only to reproduce the legacy score files.
        raw_score = -signed_gradient
        tokens.append(
            raw_score
            if getattr(args, "score_transform", "signed") == "signed"
            else max(0.0, raw_score)
        )

    result = sample_header(sample, record, args, score_definition)
    result.update(
        baseline_response_target_nll_sum=math.fsum(
            float(value) for value in baseline_response_targets.tolist()
        ),
        token_scores=tokens,
    )
    return result


def main() -> None:
    args = parse_args()
    transform_tag = "signed_raw" if args.score_transform == "signed" else "positive_clipped"
    # v3 records that the chunked hidden-state loss path reproduces a model's
    # final_logit_softcapping before CE. This only changes soft-capped models;
    # keep v2 for Qwen/MiMo/Llama so their completed caches remain reusable.
    config = json.loads(
        (args.model_dir.expanduser().resolve() / "config.json").read_text(
            encoding="utf-8"
        )
    )
    text_config = config.get("text_config") or config
    method_version = "v3" if text_config.get("final_logit_softcapping") is not None else "v2"
    write_results(
        args,
        "value_gradient",
        f"response_value_gate_gradient_{transform_tag}_{method_version}",
    )


if __name__ == "__main__":
    main()
