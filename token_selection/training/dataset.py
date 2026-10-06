"""LLaMAFactory-compatible tokenized datasets with hard selective labels."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence

from algorithm import SelectiveTrainingConfig, select_ignored_score_indices


IGNORE_INDEX = -100
SELECTION_COLUMNS = (
    "_selection_sample_id",
    "_selection_sequence_token_count",
    "_selection_response_token_count",
    "_selection_token_scores",
)
SELECTION_STAT_COLUMNS = (
    "_selection_retained_content_token_count",
    "_selection_ignored_content_token_count",
)


def _score_source_indices(score_dataset: Any, raw_size: int, score_path: Path) -> list[int]:
    indices: list[int] = []
    previous = -1
    for row_index, value in enumerate(score_dataset["sample_id"]):
        try:
            source_index = int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"non-integer sample_id in {score_path} at row {row_index + 1}: {value!r}") from exc
        if source_index <= previous:
            raise ValueError(f"sample_id must be strictly increasing in {score_path}: {source_index}")
        if source_index < 0 or source_index >= raw_size:
            raise ValueError(
                f"sample_id={source_index} from {score_path} is outside source dataset size {raw_size}"
            )
        indices.append(source_index)
        previous = source_index
    if not indices:
        raise ValueError(f"score file contains no records: {score_path}")
    return indices


def _load_local_json_dataset(path: Path, num_proc: int, cache_dir: str | None) -> Any:
    from datasets import load_dataset

    if path.is_dir():
        files = sorted(str(item) for item in path.iterdir() if item.suffix.lower() in {".json", ".jsonl"})
    elif path.is_file():
        files = [str(path)]
    else:
        raise FileNotFoundError(path)
    if not files:
        raise ValueError(f"no JSON/JSONL files found under {path}")
    return load_dataset(
        "json",
        data_files=files,
        split="train",
        cache_dir=cache_dir,
        num_proc=num_proc,
    )


def _attach_scores(raw_dataset: Any, score_path: Path, num_proc: int, cache_dir: str | None) -> Any:
    from datasets import concatenate_datasets

    score_dataset = _load_local_json_dataset(score_path, num_proc, cache_dir)
    required = {
        "sample_id",
        "sequence_token_count",
        "response_token_count",
        "token_scores",
    }
    missing = required.difference(score_dataset.column_names)
    if missing:
        raise ValueError(f"score file {score_path} is missing columns: {sorted(missing)}")
    source_indices = _score_source_indices(score_dataset, len(raw_dataset), score_path)
    selected = raw_dataset.select(source_indices)
    score_metadata = score_dataset.select_columns(
        ["sample_id", "sequence_token_count", "response_token_count", "token_scores"]
    ).rename_columns(
        {
            "sample_id": "_selection_sample_id",
            "sequence_token_count": "_selection_sequence_token_count",
            "response_token_count": "_selection_response_token_count",
            "token_scores": "_selection_token_scores",
        }
    )
    # Keep the large nested token-score Arrow column in its native buffers.
    # Dataset.add_column() rebuilds that column from Python values and can copy
    # many gigabytes before preprocessing even starts.
    return concatenate_datasets([selected, score_metadata], axis=1)


def _convert_scored_example(example: dict[str, Any], converter: Any) -> dict[str, Any]:
    metadata = {name: example[name] for name in SELECTION_COLUMNS}
    converted = converter(example)
    converted.update(metadata)
    return converted


@dataclass
class SelectiveSupervisedProcessor:
    template: Any
    tokenizer: Any
    processor: Any
    data_args: Any
    selection_config: SelectiveTrainingConfig

    def _response_end_ids(self) -> list[int]:
        elements = self.template.format_assistant.apply(content="")
        return self.template._convert_elements_to_ids(self.tokenizer, elements)

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
        for turn_index, (_source_ids, target_ids) in enumerate(encoded_pairs):
            if len(target_ids) < len(response_end_ids) or target_ids[-len(response_end_ids) :] != response_end_ids:
                raise ValueError(
                    f"sample_id={sample_id} turn={turn_index} does not end with the expected template tokens"
                )
            content_lengths.append(len(target_ids) - len(response_end_ids))

        full_sequence_count = len(initial_ids) + sum(
            len(source_ids) + len(target_ids) for source_ids, target_ids in encoded_pairs
        )
        if self.template.efficient_eos:
            full_sequence_count += 1
        full_response_count = sum(content_lengths)

        input_ids = list(initial_ids)
        labels = list(initial_labels)
        total_length = len(input_ids) + (1 if self.template.efficient_eos else 0)
        retained_label_positions: list[list[int]] = []
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

            source_label = source_ids if self.data_args.train_on_prompt else [IGNORE_INDEX] * source_len
            target_label = list(target_ids)
            target_start = len(labels) + source_len
            retained_content = min(content_lengths[turn_index], target_len)
            retained_label_positions.append(
                [target_start + local_index for local_index in range(retained_content)]
            )

            input_ids += source_ids + target_ids
            labels += source_label + target_label

        if self.template.efficient_eos:
            input_ids.append(self.tokenizer.eos_token_id)
            labels.append(self.tokenizer.eos_token_id)

        truncated_sequence_count = len(input_ids)
        truncated_response_count = sum(len(items) for items in retained_label_positions)
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
        if full_contract:
            score_offset = 0
            for turn_index, content_length in enumerate(content_lengths):
                if turn_index < len(retained_label_positions):
                    for local_index, label_position in enumerate(retained_label_positions[turn_index]):
                        score_index_to_label_position[score_offset + local_index] = label_position
                score_offset += content_length
        else:
            score_index = 0
            for positions in retained_label_positions:
                for label_position in positions:
                    score_index_to_label_position[score_index] = label_position
                    score_index += 1

        retained_indices = sorted(score_index_to_label_position)
        ignored_indices = select_ignored_score_indices(
            scores,
            retained_indices,
            exponent=self.selection_config.normalization_exponent,
            ignore_rate=self.selection_config.ignore_rate,
            sample_id=sample_id,
            zero_scores_only=self.selection_config.ignore_zero_scores_only,
            train_zero_scores_only=self.selection_config.train_zero_scores_only,
            train_nonpositive_scores_only=self.selection_config.train_nonpositive_scores_only,
        )
        for score_index in ignored_indices:
            labels[score_index_to_label_position[score_index]] = IGNORE_INDEX
        return input_ids, labels, len(retained_indices), len(ignored_indices)

    def preprocess_dataset(self, examples: dict[str, list[Any]]) -> dict[str, list[Any]]:
        model_inputs: dict[str, list[Any]] = defaultdict(list)
        for index in range(len(examples["_prompt"])):
            sample_id = str(examples["_selection_sample_id"][index])
            if len(examples["_prompt"][index]) % 2 != 1 or len(examples["_response"][index]) != 1:
                raise ValueError(f"sample_id={sample_id} is not a valid supervised conversation")
            input_ids, labels, retained_count, ignored_count = self._encode_example(
                prompt=examples["_prompt"][index],
                response=examples["_response"][index],
                system=examples["_system"][index],
                tools=examples["_tools"][index],
                images=examples["_images"][index] or [],
                videos=examples["_videos"][index] or [],
                audios=examples["_audios"][index] or [],
                sample_id=sample_id,
                expected_sequence_count=int(examples["_selection_sequence_token_count"][index]),
                expected_response_count=int(examples["_selection_response_token_count"][index]),
                scores=examples["_selection_token_scores"][index],
            )
            model_inputs["input_ids"].append(input_ids)
            model_inputs["attention_mask"].append([1] * len(input_ids))
            model_inputs["labels"].append(labels)
            model_inputs["images"].append(examples["_images"][index])
            model_inputs["videos"].append(examples["_videos"][index])
            model_inputs["audios"].append(examples["_audios"][index])
            model_inputs[SELECTION_STAT_COLUMNS[0]].append(retained_count)
            model_inputs[SELECTION_STAT_COLUMNS[1]].append(ignored_count)
        return model_inputs


def _build_scored_aligned_dataset(
    dataset_name: str,
    score_path: str,
    model_args: Any,
    data_args: Any,
    training_args: Any,
) -> Any:
    from llamafactory.data.converter import get_dataset_converter
    from llamafactory.data.parser import get_dataset_list

    dataset_attr = get_dataset_list([dataset_name], data_args.dataset_dir)[0]
    if dataset_attr.load_from != "file":
        raise ValueError("Selective training currently requires local file datasets.")
    if dataset_attr.num_samples is not None:
        raise ValueError("Selective training does not support dataset_info num_samples sampling.")
    source_path = Path(data_args.dataset_dir) / dataset_attr.dataset_name
    raw_dataset = _load_local_json_dataset(
        source_path.resolve(), data_args.preprocessing_num_workers, model_args.cache_dir
    )
    resolved_score_path = Path(score_path).expanduser().resolve()
    scored = _attach_scores(
        raw_dataset,
        resolved_score_path,
        data_args.preprocessing_num_workers,
        model_args.cache_dir,
    )
    # max_samples limits the aligned scored examples, not the original source
    # rows. Applying it before score attachment incorrectly makes valid sparse
    # or full-file sample_ids look out of range.
    if data_args.max_samples is not None:
        scored = scored.select(range(min(data_args.max_samples, len(scored))))
    converter = get_dataset_converter(dataset_attr.formatting, dataset_attr, data_args)
    column_names = scored.column_names
    return scored.map(
        _convert_scored_example,
        fn_kwargs={"converter": converter},
        batched=False,
        remove_columns=column_names,
        num_proc=data_args.preprocessing_num_workers,
        load_from_cache_file=(not data_args.overwrite_cache) or (training_args.local_process_index != 0),
        desc=f"Aligning selective dataset {dataset_name}",
    )


def get_selective_dataset(
    template: Any,
    model_args: Any,
    data_args: Any,
    training_args: Any,
    stage: str,
    tokenizer: Any,
    processor: Any = None,
    *,
    selection_config: SelectiveTrainingConfig,
) -> dict[str, Any]:
    from datasets import DatasetDict
    from llamafactory.data.data_utils import get_dataset_module, merge_dataset, split_dataset

    if stage != "sft" or not training_args.do_train:
        raise ValueError("Selective training is only implemented for SFT training.")
    if data_args.streaming:
        raise ValueError("Selective training does not support streaming datasets.")
    if data_args.packing or data_args.neat_packing:
        raise ValueError("Selective training currently requires packing=false.")
    if data_args.mask_history:
        raise ValueError("Selective training currently requires mask_history=false.")
    if not data_args.ignore_pad_token_for_loss:
        raise ValueError("Selective training requires ignore_pad_token_for_loss=true.")
    if data_args.tokenized_path is not None:
        raise ValueError("Selective training currently manages tokenization and does not accept tokenized_path.")
    if data_args.eval_dataset is not None:
        raise ValueError("Explicit eval_dataset is not supported; val_size splitting is supported.")

    dataset_names = list(data_args.dataset or [])
    score_paths = selection_config.score_paths(dataset_names)
    with training_args.main_process_first(
        desc="build selective dataset",
        local=(not data_args.data_shared_file_system),
    ):
        aligned_datasets = [
            _build_scored_aligned_dataset(
                dataset_name,
                score_paths[dataset_name],
                model_args,
                data_args,
                training_args,
            )
            for dataset_name in dataset_names
        ]
        aligned = merge_dataset(aligned_datasets, data_args, seed=training_args.seed)
        train_dict, eval_dict = split_dataset(aligned, None, data_args, seed=training_args.seed)
        selective_processor = SelectiveSupervisedProcessor(
            template=template,
            tokenizer=tokenizer,
            processor=processor,
            data_args=data_args,
            selection_config=selection_config,
        )
        for split_name, split_dataset_value in {**train_dict, **eval_dict}.items():
            column_names = split_dataset_value.column_names
            processed = split_dataset_value.map(
                selective_processor.preprocess_dataset,
                batched=True,
                batch_size=data_args.preprocessing_batch_size,
                remove_columns=column_names,
                num_proc=data_args.preprocessing_num_workers,
                load_from_cache_file=(not data_args.overwrite_cache)
                or (training_args.local_process_index != 0),
                desc=f"Tokenizing selective {split_name} dataset",
            )
            retained_count = sum(processed[SELECTION_STAT_COLUMNS[0]])
            ignored_count = sum(processed[SELECTION_STAT_COLUMNS[1]])
            if training_args.local_process_index == 0:
                ignored_rate = ignored_count / retained_count if retained_count else 0.0
                print(
                    f"Selective {split_name}: ignored {ignored_count}/{retained_count} "
                    f"retained response content tokens ({ignored_rate:.2%}).",
                    flush=True,
                )
            processed = processed.remove_columns(list(SELECTION_STAT_COLUMNS))
            if split_name == "train":
                train_dict[split_name] = processed
            else:
                eval_dict[split_name] = processed

        dataset_dict = DatasetDict({**train_dict, **eval_dict})
        return get_dataset_module(dataset_dict)


def install_selective_dataset(selection_config: SelectiveTrainingConfig) -> None:
    """Patch only the SFT workflow's dataset entrypoint."""
    from functools import partial
    import llamafactory.train.sft.workflow as workflow

    workflow.get_dataset = partial(get_selective_dataset, selection_config=selection_config)
