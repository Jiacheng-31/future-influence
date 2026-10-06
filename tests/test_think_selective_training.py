#!/usr/bin/env python3

from __future__ import annotations

from pathlib import Path
import sys
import unittest


SELECTIVE_DIR = Path(__file__).resolve().parents[1] / "token_selection" / "training"
LLAMAFACTORY_SRC = Path(__file__).resolve().parents[1].parent / "Qwen" / "LLaMA-Factory" / "src"
sys.path.insert(0, str(SELECTIVE_DIR))
sys.path.insert(0, str(LLAMAFACTORY_SRC))

from think_algorithm import (  # noqa: E402
    ThinkSelectiveTrainingConfig,
    select_scoped_ignored_score_indices,
)
from think_dataset import (  # noqa: E402
    ThinkSelectiveSupervisedProcessor,
    find_think_content_indices,
)


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
        # Content: <think>, reason-a, reason-b, </think>, answer-a, answer-b
        return [([10, 11], [30, 20, 21, 31, 22, 23, 99])]


class _FakeDataArgs:
    cutoff_len = 32
    train_on_prompt = False


class _FakeTokenizer:
    eos_token_id = 2

    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        return {"<think>": [30], "</think>": [31]}[text]


def _config(*, random_mask: bool) -> ThinkSelectiveTrainingConfig:
    return ThinkSelectiveTrainingConfig.from_dict(
        {
            "selection_scope": "response" if random_mask else "think",
            "ignore_zero_scores_only": True,
            "random_mask_zero_count": random_mask,
            "random_mask_seed": 42,
            "score_file": "scores.jsonl",
        }
    )


class ThinkSelectiveTrainingTest(unittest.TestCase):
    def test_finds_only_tokens_strictly_inside_markers(self) -> None:
        indices = find_think_content_indices(
            [30, 7, 8, 31, 9], [30], [31], sample_id="1", turn_index=0
        )
        self.assertEqual(indices, {1, 2})

    def test_missing_think_marker_means_no_think_selection(self) -> None:
        indices = find_think_content_indices(
            [7, 8, 31], [30], [31], sample_id="2", turn_index=0
        )
        self.assertEqual(indices, set())

    def test_unclosed_think_block_extends_to_response_end(self) -> None:
        indices = find_think_content_indices(
            [30, 7, 8, 9], [30], [31], sample_id="3", turn_index=0
        )
        self.assertEqual(indices, {1, 2, 3})

    def test_random_mask_is_deterministic_and_matches_zero_count(self) -> None:
        scores = [5.0, 0.0, 2.0, -0.1, 3.0]
        eligible = [1, 2, 3, 4]
        first = select_scoped_ignored_score_indices(
            scores,
            eligible,
            sample_id="random-count",
            random_mask_zero_count=True,
            random_mask_seed=42,
        )
        second = select_scoped_ignored_score_indices(
            scores,
            eligible,
            sample_id="random-count",
            random_mask_zero_count=True,
            random_mask_seed=42,
        )
        self.assertEqual(first, second)
        self.assertEqual(len(first), 2)
        self.assertTrue(first.issubset(set(eligible)))

    def test_score_selection_masks_think_zero_but_keeps_answer_zeros(self) -> None:
        processor = ThinkSelectiveSupervisedProcessor(
            template=_FakeTemplate(),
            tokenizer=_FakeTokenizer(),
            processor=None,
            data_args=_FakeDataArgs(),
            selection_config=_config(random_mask=False),
        )
        input_ids, labels, retained_count, ignored_count = processor._encode_example(
            prompt=[{"role": "user", "content": "prompt"}],
            response=[{"role": "assistant", "content": "unused by fake template"}],
            system=None,
            tools=None,
            images=[],
            videos=[],
            audios=[],
            sample_id="think-zero",
            expected_sequence_count=9,
            expected_response_count=6,
            # Only index 1 is non-positive inside think. Answer indices 4 and 5
            # are also non-positive and must remain trainable.
            scores=[5.0, 0.0, 2.0, 5.0, -0.1, 0.0],
        )
        self.assertEqual(input_ids, [10, 11, 30, 20, 21, 31, 22, 23, 99])
        self.assertEqual(labels, [-100, -100, 30, -100, 21, 31, 22, 23, 99])
        self.assertEqual((retained_count, ignored_count), (6, 1))

    def test_random_processor_masks_full_response_with_same_count(self) -> None:
        processor = ThinkSelectiveSupervisedProcessor(
            template=_FakeTemplate(),
            tokenizer=_FakeTokenizer(),
            processor=None,
            data_args=_FakeDataArgs(),
            selection_config=_config(random_mask=True),
        )
        _input_ids, labels, _retained_count, ignored_count = processor._encode_example(
            prompt=[{"role": "user", "content": "prompt"}],
            response=[{"role": "assistant", "content": "unused by fake template"}],
            system=None,
            tools=None,
            images=[],
            videos=[],
            audios=[],
            sample_id="think-random",
            expected_sequence_count=9,
            expected_response_count=6,
            scores=[5.0, 0.0, 2.0, 5.0, 0.0, 0.0],
        )
        # The full response has three zero scores. Random masking may choose
        # reasoning, marker, or answer tokens, but never the template end.
        self.assertEqual(ignored_count, 3)
        self.assertEqual(sum(label == -100 for label in labels[2:8]), 3)
        self.assertEqual(labels[8], 99)


if __name__ == "__main__":
    unittest.main()
