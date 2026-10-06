#!/usr/bin/env python3
"""Finite per-token Value-gate ablation in the requested model dtype."""

from __future__ import annotations

import argparse
from typing import Any

from offline_token_model_utils import response_token_losses, unwrap_model
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
            "Set one response token's shared attention-Value gate below one per "
            "forward and measure its future response-target NLL increase. FP32 "
            "is the default; BF16 is accepted for controlled precision studies."
        )
    )
    add_common_arguments(parser)
    parser.set_defaults(dtype="float32")
    parser.add_argument(
        "--ablated-gate-value",
        type=float,
        required=True,
        help="Replacement Value-gate value in [0, 1).",
    )
    return parser.parse_args()


def gated_forward_losses(
    model: Any,
    input_ids: Any,
    gate: Any,
    gate_mask: Any,
    target_positions: Any,
) -> Any:
    with gated_value_flow(model, gate, gate_mask):
        outputs = unwrap_model(model)(
            input_ids=input_ids,
            attention_mask=input_ids.new_ones(input_ids.shape),
            use_cache=False,
        )
    return response_token_losses(
        model,
        outputs.last_hidden_state,
        input_ids,
        target_positions,
    )


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
    if not getattr(model, "_token_score_parameters_frozen", False):
        model.requires_grad_(False)
        model.eval()
        model._token_score_parameters_frozen = True

    input_ids = make_input(sample, device)
    gate_mask = torch.zeros_like(input_ids, dtype=torch.float32)
    gate_mask[:, sample.response_positions] = 1.0
    all_one_gate = torch.ones_like(gate_mask, dtype=torch.float32)
    all_target_positions = torch.tensor(
        sample.response_target_positions,
        device=device,
        dtype=torch.long,
    )

    with torch.inference_mode():
        baseline_losses = gated_forward_losses(
            model,
            input_ids,
            all_one_gate,
            gate_mask,
            all_target_positions,
        )
        baseline_by_position = dict(
            zip(sample.response_target_positions, baseline_losses.tolist())
        )

        token_scores: list[float] = []
        response_token_count = len(sample.response_positions)
        progress_interval = max(10, response_token_count // 100)
        for response_index, position in enumerate(sample.response_positions):
            future_positions = [
                target
                for target in sample.response_target_positions
                if target > position
            ]
            if not future_positions:
                token_scores.append(0.0)
            else:
                ablated_gate = all_one_gate.clone()
                ablated_gate[0, position] = args.ablated_gate_value
                future_tensor = torch.tensor(
                    future_positions,
                    device=device,
                    dtype=torch.long,
                )
                ablated_losses = gated_forward_losses(
                    model,
                    input_ids,
                    ablated_gate,
                    gate_mask,
                    future_tensor,
                )
                baseline_sum = math.fsum(
                    baseline_by_position[target] for target in future_positions
                )
                ablated_sum = math.fsum(ablated_losses.tolist())
                raw_score = ablated_sum - baseline_sum
                token_scores.append(
                    raw_score
                    if getattr(args, "score_transform", "signed") == "signed"
                    else max(0.0, raw_score)
                )

            completed = response_index + 1
            if completed % progress_interval == 0 or completed == response_token_count:
                print(
                    f"[source {sample.source_index}] value_ablation_progress="
                    f"{completed}/{response_token_count} "
                    f"({100.0 * completed / response_token_count:.1f}%)",
                    flush=True,
                )

    result = sample_header(sample, record, args, score_definition)
    result.update(
        baseline_response_target_nll_sum=math.fsum(baseline_losses.tolist()),
        token_scores=token_scores,
    )
    return result


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.ablated_gate_value < 1.0:
        raise ValueError("--ablated-gate-value must be in [0, 1)")
    gate_tag = format(args.ablated_gate_value, ".8g").replace(".", "p")
    transform_tag = "signed_raw" if args.score_transform == "signed" else "positive_clipped"
    dtype_tag = {
        "auto": "auto",
        "bfloat16": "bf16",
        "float16": "fp16",
        "float32": "fp32",
    }[args.dtype]
    write_results(
        args,
        "value_ablation",
        f"value_gate_{gate_tag}_future_response_target_nll_delta_{dtype_tag}_{transform_tag}_v2",
    )


if __name__ == "__main__":
    main()
