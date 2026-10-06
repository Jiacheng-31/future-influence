#!/usr/bin/env python3

from __future__ import annotations

import math
from pathlib import Path
import sys
import unittest
from types import SimpleNamespace


SELECTIVE_DIR = Path(__file__).resolve().parents[1] / "token_selection" / "training"
LLAMAFACTORY_SRC = Path(__file__).resolve().parents[1].parent / "Qwen" / "LLaMA-Factory" / "src"
sys.path.insert(0, str(SELECTIVE_DIR))
sys.path.insert(0, str(LLAMAFACTORY_SRC))

from algorithm import (  # noqa: E402
    SelectiveTrainingConfig,
    normalize_future_scores,
    select_ignored_score_indices,
)
from dataset import SelectiveSupervisedProcessor  # noqa: E402


class _FakeMultimodalPlugin:
    def process_messages(self, messages, images, videos, audios, processor):
        return messages

    def process_token_ids(self, input_ids, labels, images, videos, audios, tokenizer, processor):
        return input_ids, labels


class _FakeFormatter:
    def apply(self, content):
        return [99]


class _FakeTemplate:
    mm_plugin = _FakeMultimodalPlugin()
    format_assistant = _FakeFormatter()
    efficient_eos = False

    def _convert_elements_to_ids(self, tokenizer, elements):
        return elements

    def encode_multiturn(self, tokenizer, messages, system, tools, cutoff_len):
        return [([10, 11], [20, 21, 99])]


class _FakeDataArgs:
    cutoff_len = 16
    train_on_prompt = False


class _FakeTokenizer:
    eos_token_id = 2


class SelectiveTrainingAlgorithmTest(unittest.TestCase):
    def test_future_length_normalization(self) -> None:
        normalized = normalize_future_scores([4.0, 1.0], 0.5)
        self.assertAlmostEqual(normalized[0], 4.0 / math.sqrt(2.0))
        self.assertAlmostEqual(normalized[1], 1.0)

    def test_signed_scores_are_preserved_for_ranking(self) -> None:
        normalized = normalize_future_scores([-2.0, 1.0], 0.0)
        self.assertEqual(normalized, [-2.0, 1.0])

    def test_lowest_ceil_fraction_is_ignored_with_stable_ties(self) -> None:
        ignored = select_ignored_score_indices(
            [0.0, 0.0, 2.0, 1.0, 3.0],
            [0, 1, 2, 3, 4],
            exponent=0.0,
            ignore_rate=0.41,
            sample_id="7",
        )
        self.assertEqual(ignored, {0, 1, 3})

    def test_ignore_rate_must_cover_zero_score_fraction(self) -> None:
        with self.assertRaisesRegex(ValueError, "zero-score rate"):
            select_ignored_score_indices(
                [0.0, 0.0, 1.0, 2.0],
                [0, 1, 2, 3],
                exponent=0.5,
                ignore_rate=0.49,
                sample_id="8",
            )

    def test_zero_only_mode_ignores_no_positive_scores(self) -> None:
        ignored = select_ignored_score_indices(
            [0.0, -0.2, 0.0, 0.1],
            [0, 1, 2, 3],
            exponent=0.5,
            ignore_rate=None,
            sample_id="zero-only",
            zero_scores_only=True,
        )
        self.assertEqual(ignored, {0, 1, 2})

    def test_reverse_mode_trains_only_zero_scores(self) -> None:
        ignored = select_ignored_score_indices(
            [0.0, -0.2, 0.0, 0.1],
            [0, 1, 2, 3],
            exponent=0.5,
            ignore_rate=None,
            sample_id="reverse-zero-only",
            train_zero_scores_only=True,
        )
        self.assertEqual(ignored, {1, 3})

    def test_reverse_signed_mode_trains_nonpositive_scores(self) -> None:
        ignored = select_ignored_score_indices(
            [0.0, -0.2, 0.1, 2.0],
            [0, 1, 2, 3],
            exponent=0.5,
            ignore_rate=None,
            sample_id="reverse-nonpositive-only",
            train_nonpositive_scores_only=True,
        )
        self.assertEqual(ignored, {2, 3})

    def test_truncated_sequence_uses_full_future_lengths_but_retained_selection(self) -> None:
        ignored = select_ignored_score_indices(
            [4.0, 3.0, 2.0, 0.0],
            [0, 1],
            exponent=1.0,
            ignore_rate=0.5,
            sample_id="9",
        )
        # Full-sequence normalization is [1, 1, 1, 0], and the stable tie
        # therefore ignores the earlier retained token.
        self.assertEqual(ignored, {0})

    def test_config_bounds_and_score_mapping(self) -> None:
        config = SelectiveTrainingConfig.from_dict(
            {
                "normalization_exponent": 0.5,
                "ignore_rate": 0.5,
                "score_files": {"train": "scores.jsonl"},
            }
        )
        self.assertEqual(config.score_paths(["train"]), {"train": "scores.jsonl"})
        with self.assertRaisesRegex(ValueError, r"\[0, 1\]"):
            SelectiveTrainingConfig.from_dict(
                {
                    "normalization_exponent": 1.1,
                    "ignore_rate": 0.5,
                    "score_file": "scores.jsonl",
                }
            )
        zero_only = SelectiveTrainingConfig.from_dict(
            {
                "ignore_zero_scores_only": True,
                "score_file": "scores.jsonl",
            }
        )
        self.assertIsNone(zero_only.ignore_rate)
        reverse = SelectiveTrainingConfig.from_dict(
            {
                "train_zero_scores_only": True,
                "score_file": "scores.jsonl",
            }
        )
        self.assertIsNone(reverse.ignore_rate)
        reverse_signed = SelectiveTrainingConfig.from_dict(
            {
                "train_nonpositive_scores_only": True,
                "score_file": "scores.jsonl",
            }
        )
        self.assertIsNone(reverse_signed.ignore_rate)
        with self.assertRaisesRegex(ValueError, "mutually exclusive"):
            SelectiveTrainingConfig.from_dict(
                {
                    "ignore_zero_scores_only": True,
                    "train_zero_scores_only": True,
                    "score_file": "scores.jsonl",
                }
            )
        with self.assertRaisesRegex(ValueError, "mutually exclusive"):
            SelectiveTrainingConfig.from_dict(
                {
                    "ignore_zero_scores_only": True,
                    "train_nonpositive_scores_only": True,
                    "score_file": "scores.jsonl",
                }
            )

    def test_zero_scores_are_masked_without_removing_input_tokens_or_response_end(self) -> None:
        processor = SelectiveSupervisedProcessor(
            template=_FakeTemplate(),
            tokenizer=_FakeTokenizer(),
            processor=None,
            data_args=_FakeDataArgs(),
            selection_config=SelectiveTrainingConfig.from_dict(
                {
                    "ignore_zero_scores_only": True,
                    "score_file": "scores.jsonl",
                }
            ),
        )
        input_ids, labels, retained_count, ignored_count = processor._encode_example(
            prompt=[{"role": "user", "content": "prompt"}],
            response=[{"role": "assistant", "content": "response"}],
            system=None,
            tools=None,
            images=[],
            videos=[],
            audios=[],
            sample_id="label-mask",
            expected_sequence_count=5,
            expected_response_count=2,
            scores=[0.0, -1.0],
        )
        self.assertEqual(input_ids, [10, 11, 20, 21, 99])
        self.assertEqual(labels, [-100, -100, -100, -100, 99])
        self.assertEqual((retained_count, ignored_count), (2, 2))

    def test_reverse_processor_keeps_only_zero_score_content_and_response_end(self) -> None:
        processor = SelectiveSupervisedProcessor(
            template=_FakeTemplate(),
            tokenizer=_FakeTokenizer(),
            processor=None,
            data_args=_FakeDataArgs(),
            selection_config=SelectiveTrainingConfig.from_dict(
                {
                    "train_zero_scores_only": True,
                    "score_file": "scores.jsonl",
                }
            ),
        )
        input_ids, labels, retained_count, ignored_count = processor._encode_example(
            prompt=[{"role": "user", "content": "prompt"}],
            response=[{"role": "assistant", "content": "response"}],
            system=None,
            tools=None,
            images=[],
            videos=[],
            audios=[],
            sample_id="reverse-label-mask",
            expected_sequence_count=5,
            expected_response_count=2,
            scores=[0.0, 1.0],
        )
        self.assertEqual(input_ids, [10, 11, 20, 21, 99])
        self.assertEqual(labels, [-100, -100, 20, -100, 99])
        self.assertEqual((retained_count, ignored_count), (2, 1))

    def test_truncated_score_metadata_matches_training_cutoff(self) -> None:
        processor = SelectiveSupervisedProcessor(
            template=_FakeTemplate(),
            tokenizer=_FakeTokenizer(),
            processor=None,
            data_args=SimpleNamespace(cutoff_len=4, train_on_prompt=False),
            selection_config=SelectiveTrainingConfig.from_dict(
                {
                    "ignore_zero_scores_only": True,
                    "score_file": "scores.jsonl",
                }
            ),
        )
        input_ids, labels, retained_count, ignored_count = processor._encode_example(
            prompt=[{"role": "user", "content": "prompt"}],
            response=[{"role": "assistant", "content": "response"}],
            system=None,
            tools=None,
            images=[],
            videos=[],
            audios=[],
            sample_id="truncated-score-contract",
            expected_sequence_count=4,
            expected_response_count=2,
            scores=[0.0, 1.0],
        )
        self.assertEqual(input_ids, [10, 11, 20, 21])
        self.assertEqual(labels, [-100, -100, -100, 21])
        self.assertEqual((retained_count, ignored_count), (2, 1))


if __name__ == "__main__":
    unittest.main()
