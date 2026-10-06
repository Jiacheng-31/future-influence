"""Pure score normalization and hard token-selection rules."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Sequence


@dataclass(frozen=True)
class SelectiveTrainingConfig:
    """Configuration consumed outside LLaMAFactory's own argument parser."""

    enabled: bool = True
    normalization_exponent: float = 0.5
    ignore_rate: float | None = None
    ignore_zero_scores_only: bool = False
    train_zero_scores_only: bool = False
    train_nonpositive_scores_only: bool = False
    score_file: str | None = None
    score_files: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, values: dict[str, Any] | None) -> "SelectiveTrainingConfig":
        config = cls(**(values or {}))
        config.validate()
        return config

    def validate(self) -> None:
        if not self.enabled:
            raise ValueError(
                "Set selective_training.enabled=true or use the standard LLaMAFactory launcher."
            )
        if not 0.0 <= self.normalization_exponent <= 1.0:
            raise ValueError("selective_training.normalization_exponent must be in [0, 1].")
        exclusive_modes = sum(
            (
                self.ignore_zero_scores_only,
                self.train_zero_scores_only,
                self.train_nonpositive_scores_only,
            )
        )
        if exclusive_modes > 1:
            raise ValueError(
                "ignore_zero_scores_only, train_zero_scores_only, and "
                "train_nonpositive_scores_only are mutually exclusive."
            )
        if exclusive_modes:
            if self.ignore_rate is not None:
                raise ValueError(
                    "zero-score selection modes and ignore_rate are mutually exclusive."
                )
        elif self.ignore_rate is None:
            raise ValueError(
                "selective_training.ignore_rate is required unless "
                "ignore_zero_scores_only=true, train_zero_scores_only=true, or "
                "train_nonpositive_scores_only=true."
            )
        elif not 0.0 <= self.ignore_rate <= 1.0:
            raise ValueError("selective_training.ignore_rate must be in [0, 1].")
        if self.score_file and self.score_files:
            raise ValueError("Use either selective_training.score_file or score_files, not both.")
        if not self.score_file and not self.score_files:
            raise ValueError("A selective_training score_file or score_files mapping is required.")
        if any(not isinstance(key, str) or not isinstance(value, str) for key, value in self.score_files.items()):
            raise TypeError("selective_training.score_files must map dataset names to paths.")

    def score_paths(self, dataset_names: Sequence[str]) -> dict[str, str]:
        names = list(dataset_names)
        if self.score_file is not None:
            if len(names) != 1:
                raise ValueError("score_file is only valid when exactly one dataset is configured.")
            return {names[0]: self.score_file}

        missing = [name for name in names if name not in self.score_files]
        if missing:
            raise ValueError(f"Missing score_files entries for datasets: {missing}")
        return {name: self.score_files[name] for name in names}


def normalize_future_scores(scores: Sequence[float], exponent: float) -> list[float]:
    """Return s_i / (T + 1 - i)^beta for zero-based input positions."""
    if not 0.0 <= exponent <= 1.0:
        raise ValueError("normalization exponent must be in [0, 1]")
    total = len(scores)
    if total == 0:
        raise ValueError("token score sequence must not be empty")

    normalized: list[float] = []
    for index, raw_value in enumerate(scores):
        value = float(raw_value)
        if not math.isfinite(value):
            raise ValueError(f"token score at index {index} must be finite")
        future_length = total - index
        normalized.append(value / math.pow(future_length, exponent))
    return normalized


def select_ignored_score_indices(
    scores: Sequence[float],
    retained_indices: Sequence[int],
    *,
    exponent: float,
    ignore_rate: float | None,
    sample_id: str,
    zero_scores_only: bool = False,
    train_zero_scores_only: bool = False,
    train_nonpositive_scores_only: bool = False,
) -> set[int]:
    """Select an exact low-score fraction among retained response content tokens.

    Returned indices address the original full score sequence. Ties are broken by
    response-token position, which makes preprocessing deterministic.
    """
    normalized = normalize_future_scores(scores, exponent)
    retained = [int(index) for index in retained_indices]
    if not retained:
        raise ValueError(f"sample_id={sample_id} has no retained response content token")
    if len(set(retained)) != len(retained):
        raise ValueError(f"sample_id={sample_id} has duplicate retained score indices")
    if retained != sorted(retained) or retained[0] < 0 or retained[-1] >= len(scores):
        raise ValueError(f"sample_id={sample_id} retained score indices are invalid")

    # Scores <= 0 are non-training tokens.  The scorer can emit signed values,
    # and the standard selective mode must treat negative and exact-zero scores
    # identically instead of accidentally training negative-score tokens.
    zero_count = sum(float(scores[index]) <= 0.0 for index in retained)
    if sum((zero_scores_only, train_zero_scores_only, train_nonpositive_scores_only)) > 1:
        raise ValueError(
            "zero_scores_only, train_zero_scores_only, and "
            "train_nonpositive_scores_only are mutually exclusive"
        )
    if zero_scores_only:
        if ignore_rate is not None:
            raise ValueError("zero_scores_only and ignore_rate are mutually exclusive")
        return {index for index in retained if float(scores[index]) <= 0.0}
    if train_zero_scores_only:
        if ignore_rate is not None:
            raise ValueError("train_zero_scores_only and ignore_rate are mutually exclusive")
        # Reverse selective training: positive-score response content is masked,
        # leaving only the zero-score (normally unselected) content trainable.
        return {index for index in retained if float(scores[index]) != 0.0}
    if train_nonpositive_scores_only:
        if ignore_rate is not None:
            raise ValueError("train_nonpositive_scores_only and ignore_rate are mutually exclusive")
        # Exact reverse of signed-score selective training over response
        # content: mask positive tokens and train only scores <= 0.
        return {index for index in retained if float(scores[index]) > 0.0}
    if ignore_rate is None or not 0.0 <= ignore_rate <= 1.0:
        raise ValueError("ignore_rate must be in [0, 1]")
    zero_rate = zero_count / len(retained)
    if ignore_rate + 1e-12 < zero_rate:
        raise ValueError(
            f"sample_id={sample_id} ignore_rate={ignore_rate:.6f} is smaller than "
            f"the retained zero-score rate={zero_rate:.6f} ({zero_count}/{len(retained)})"
        )

    ignore_count = math.ceil(ignore_rate * len(retained))
    ranked = sorted(retained, key=lambda index: (normalized[index], index))
    return set(ranked[:ignore_count])
