#!/usr/bin/env python3
"""Per-response-token NLL scores for entropy-based token selection."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys
from typing import Any


SCORING_DIR = Path(__file__).resolve().parent.parent / "scoring"
if str(SCORING_DIR) not in sys.path:
    sys.path.insert(0, str(SCORING_DIR))

from offline_token_model_utils import response_token_losses, unwrap_model  # noqa: E402
from offline_token_score_common import (  # noqa: E402
    add_common_arguments,
    make_input,
    sample_header,
    write_results,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute the conditional negative log-likelihood of every response "
            "content token and store it in the standard token-score cache schema."
        )
    )
    add_common_arguments(parser, include_score_transform=False)
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
    """Return one exact teacher-forced NLL value per response content token."""
    import torch

    del tokenizer
    input_ids = make_input(sample, device)
    target_positions = torch.tensor(
        sample.response_target_positions,
        device=device,
        dtype=torch.long,
    )

    if not getattr(model, "_token_score_parameters_frozen", False):
        model.requires_grad_(False)
        model.eval()
        model._token_score_parameters_frozen = True

    with torch.inference_mode():
        outputs = unwrap_model(model)(
            input_ids=input_ids,
            attention_mask=input_ids.new_ones(input_ids.shape),
            use_cache=False,
        )
        all_target_losses = response_token_losses(
            model,
            outputs.last_hidden_state,
            input_ids,
            target_positions,
            chunk_size=getattr(args, "lm_head_chunk_size", 1024),
        ).detach().float().cpu()

    loss_by_position = dict(
        zip(sample.response_target_positions, all_target_losses.tolist(), strict=True)
    )
    token_scores = [float(loss_by_position[position]) for position in sample.response_positions]
    for response_index, value in enumerate(token_scores):
        if not math.isfinite(value) or value < 0.0:
            raise FloatingPointError(
                f"invalid token NLL at response index {response_index}: {value}"
            )

    result = sample_header(sample, record, args, score_definition)
    result.update(
        baseline_response_target_nll_sum=math.fsum(
            float(value) for value in all_target_losses.tolist()
        ),
        token_scores=token_scores,
    )
    return result


def main() -> None:
    write_results(
        parse_args(),
        "token_nll",
        "response_content_token_negative_log_likelihood_v1",
    )


if __name__ == "__main__":
    main()
