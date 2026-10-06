"""Selective processor that limits masking to ``<think>...</think>`` content."""

from __future__ import annotations

from functools import cached_property
from typing import Any, Optional, Sequence

import dataset as base_dataset
from dataset import IGNORE_INDEX, SelectiveSupervisedProcessor
from think_algorithm import ThinkSelectiveTrainingConfig, select_scoped_ignored_score_indices


def _find_subsequence(sequence: Sequence[int], subsequence: Sequence[int], start: int = 0) -> int:
    width = len(subsequence)
    if width == 0:
        raise ValueError("think marker token sequence must not be empty")
    for index in range(start, len(sequence) - width + 1):
        if list(sequence[index : index + width]) == list(subsequence):
            return index
    return -1


def find_think_content_indices(
    content_ids: Sequence[int],
    open_ids: Sequence[int],
    close_ids: Sequence[int],
    *,
    sample_id: str,
    turn_index: int,
) -> set[int]:
    """Return content-token indices strictly inside the first think block."""
    open_start = _find_subsequence(content_ids, open_ids)
    if open_start < 0:
        return set()
    body_start = open_start + len(open_ids)
    close_start = _find_subsequence(content_ids, close_ids, body_start)
    if close_start < 0:
        # Some prefiltered CoT generations end at the length limit while still
        # reasoning. Everything after the opening marker is think content.
        close_start = len(content_ids)
    return set(range(body_start, close_start))


class ThinkSelectiveSupervisedProcessor(SelectiveSupervisedProcessor):
    selection_config: ThinkSelectiveTrainingConfig

    @cached_property
    def _think_marker_ids(self) -> tuple[list[int], list[int]]:
        open_ids = self.tokenizer.encode("<think>", add_special_tokens=False)
        close_ids = self.tokenizer.encode("</think>", add_special_tokens=False)
        if not open_ids or not close_ids:
            raise ValueError("tokenizer cannot encode <think> markers")
        return list(open_ids), list(close_ids)

    def _encode_example(
        self,
        prompt: list[dict[str, str]],
        response: list[dict[str, str]],
        system: Optional[str],
        tools: Optional[str],
        images: list[Any],
        videos: list[Any],
        audios: list[Any],
        sample_id: str,
        expected_sequence_count: int,
        expected_response_count: int,
        scores: Sequence[float],
    ) -> tuple[list[int], list[int], int, int]:
        from llamafactory.data.processor.processor_utils import infer_seqlen

        messages = self.template.mm_plugin.process_messages(
            prompt + response, images, videos, audios, self.processor
        )
        initial_ids, initial_labels = self.template.mm_plugin.process_token_ids(
            [], [], images, videos, audios, self.tokenizer, self.processor
        )
        encoded_pairs = self.template.encode_multiturn(
            self.tokenizer, messages, system, tools, False
        )
        response_end_ids = self._response_end_ids()
        if not response_end_ids:
            raise ValueError("cannot identify template-added response end tokens")

        content_lengths: list[int] = []
        selection_local_indices: list[set[int]] = []
        marker_ids = self._think_marker_ids if self.selection_config.selection_scope == "think" else None
        for turn_index, (_source_ids, target_ids) in enumerate(encoded_pairs):
            if (
                len(target_ids) < len(response_end_ids)
                or target_ids[-len(response_end_ids) :] != response_end_ids
            ):
                raise ValueError(
                    f"sample_id={sample_id} turn={turn_index} does not end with the expected template tokens"
                )
            content_length = len(target_ids) - len(response_end_ids)
            content_lengths.append(content_length)
            if marker_ids is None:
                selection_local_indices.append(set(range(content_length)))
            else:
                open_ids, close_ids = marker_ids
                selection_local_indices.append(
                    find_think_content_indices(
                        target_ids[:content_length],
                        open_ids,
                        close_ids,
                        sample_id=sample_id,
                        turn_index=turn_index,
                    )
                )

        full_sequence_count = len(initial_ids) + sum(
            len(source_ids) + len(target_ids) for source_ids, target_ids in encoded_pairs
        )
        if self.template.efficient_eos:
            full_sequence_count += 1
        input_ids = list(initial_ids)
        labels = list(initial_labels)
        total_length = len(input_ids) + (1 if self.template.efficient_eos else 0)
        retained_turns: list[tuple[int, int, list[int]]] = []
        for turn_index, (source_ids, target_ids) in enumerate(encoded_pairs):
            if total_length >= self.data_args.cutoff_len:
                break
            source_len, target_len = infer_seqlen(
                len(source_ids),
                len(target_ids),
                self.data_args.cutoff_len - total_length,
            )
            source_ids = source_ids[:source_len]
            target_ids = target_ids[:target_len]
            total_length += source_len + target_len

            source_label = (
                source_ids if self.data_args.train_on_prompt else [IGNORE_INDEX] * source_len
            )
            target_label = list(target_ids)
            target_start = len(labels) + source_len
            retained_content = min(content_lengths[turn_index], target_len)
            retained_turns.append(
                (turn_index, target_start, list(range(retained_content)))
            )

            input_ids += source_ids + target_ids
            labels += source_label + target_label

        if self.template.efficient_eos:
            input_ids.append(self.tokenizer.eos_token_id)
            labels.append(self.tokenizer.eos_token_id)

        full_response_count = sum(content_lengths)
        truncated_sequence_count = len(input_ids)
        truncated_response_count = sum(len(local_indices) for _, _, local_indices in retained_turns)
        full_contract = (
            expected_sequence_count == full_sequence_count
            and expected_response_count == full_response_count
        )
        truncated_contract = (
            expected_sequence_count == truncated_sequence_count
            and expected_response_count == truncated_response_count
        )
        if not full_contract and not truncated_contract:
            raise ValueError(
                f"sample_id={sample_id} score/tokenization mismatch: "
                f"score=({expected_sequence_count} sequence, {expected_response_count} response), "
                f"LLaMAFactory full=({full_sequence_count}, {full_response_count}), "
                f"truncated=({truncated_sequence_count}, {truncated_response_count}); "
                "check model tokenizer, template, and scoring truncation"
            )
        if len(scores) != expected_response_count:
            raise ValueError(
                f"sample_id={sample_id} response token mismatch: score metadata={expected_response_count}, "
                f"score array={len(scores)}"
            )

        score_index_to_label_position: dict[int, int] = {}
        retained_selection_indices: list[int] = []
        score_offset = 0
        for turn_index, target_start, local_indices in retained_turns:
            for local_index in local_indices:
                # A long source response may be cut at cutoff_len before
                # score generation. In that case score positions address the
                # concatenated retained labels, not the original full target.
                score_index = score_offset + local_index
                score_index_to_label_position[score_index] = target_start + local_index
                if local_index in selection_local_indices[turn_index]:
                    retained_selection_indices.append(score_index)
            if full_contract:
                score_offset += content_lengths[turn_index]
            else:
                score_offset += len(local_indices)

        ignored_indices = select_scoped_ignored_score_indices(
            scores,
            retained_selection_indices,
            sample_id=sample_id,
            random_mask_zero_count=self.selection_config.random_mask_zero_count,
            random_mask_seed=self.selection_config.random_mask_seed,
        )
        for score_index in ignored_indices:
            labels[score_index_to_label_position[score_index]] = IGNORE_INDEX
        return input_ids, labels, len(score_index_to_label_position), len(ignored_indices)


def install_think_selective_dataset(selection_config: ThinkSelectiveTrainingConfig) -> None:
    """Install the existing dataset pipeline with the think-only processor."""
    base_dataset.SelectiveSupervisedProcessor = ThinkSelectiveSupervisedProcessor
    base_dataset.install_selective_dataset(selection_config)
