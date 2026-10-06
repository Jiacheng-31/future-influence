"""Think-only score selection and matched-count random-mask rules."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Sequence

from algorithm import SelectiveTrainingConfig, select_ignored_score_indices


@dataclass(frozen=True)
class ThinkSelectiveTrainingConfig(SelectiveTrainingConfig):
    """Selective settings for experiments restricted to ``<think>`` content."""

    selection_scope: str = "think"
    random_mask_zero_count: bool = False
    random_mask_seed: int = 42

    def validate(self) -> None:
        super().validate()
        if self.selection_scope not in {"think", "response"}:
            raise ValueError("selective_training.selection_scope must be think or response.")
        if not self.ignore_zero_scores_only:
            raise ValueError(
                "think-selective training currently requires ignore_zero_scores_only=true."
            )
        if self.random_mask_zero_count and self.selection_scope != "response":
            raise ValueError("random_mask_zero_count=true requires selection_scope=response.")
        if not self.random_mask_zero_count and self.selection_scope != "think":
            raise ValueError("score-based think selection requires selection_scope=think.")
        if isinstance(self.random_mask_seed, bool) or not isinstance(self.random_mask_seed, int):
            raise TypeError("selective_training.random_mask_seed must be an integer.")


def _random_rank(sample_id: str, score_index: int, seed: int) -> bytes:
    payload = f"{seed}\0{sample_id}\0{score_index}".encode("utf-8")
    return hashlib.sha256(payload).digest()


def select_scoped_ignored_score_indices(
    scores: Sequence[float],
    eligible_indices: Sequence[int],
    *,
    sample_id: str,
    random_mask_zero_count: bool,
    random_mask_seed: int,
) -> set[int]:
    """Mask eligible non-positive scores, or randomly with the same count."""
    eligible = [int(index) for index in eligible_indices]
    if not eligible:
        return set()
    if not random_mask_zero_count:
        return select_ignored_score_indices(
            scores,
            eligible,
            exponent=0.0,
            ignore_rate=None,
            sample_id=sample_id,
            zero_scores_only=True,
        )

    zero_count = sum(float(scores[index]) <= 0.0 for index in eligible)
    ranked = sorted(
        eligible,
        key=lambda index: (_random_rank(sample_id, index, random_mask_seed), index),
    )
    return set(ranked[:zero_count])
