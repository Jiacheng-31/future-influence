"""Shared encoding, model loading, and multi-GPU sharding for token scoring."""

from __future__ import annotations

import argparse
import heapq
import json
import os
import re
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Sequence


SUPPORTED_TEMPLATES = ("gemma3", "gemma4", "llama3", "mimo", "qwen3", "qwen3_nothink")
GEMMA4_RESPONSE_FORMAT_CONTRACT = "gemma4_native_thought_channel_v1"
GEMMA4_THOUGHT_OPEN = "<|channel>thought\n"
GEMMA4_THOUGHT_CLOSE = "<channel|>"
FINAL_OUTPUT_KEYS = (
    "sample_id",
    "sequence_token_count",
    "response_token_count",
    "baseline_response_target_nll_sum",
    "token_scores",
)
ERROR_OUTPUT_KEYS = (
    "source_index",
    "line_number",
    "sample_id",
    "error_type",
    "error",
)


@dataclass(frozen=True)
class EncodedSample:
    source_index: int
    token_ids: list[int]
    # Original response content only; template-added end markers are excluded.
    response_positions: list[int]
    # Supervised response targets, including template-added end markers.
    response_target_positions: list[int]


class EmptyResponseError(ValueError):
    """The source sample contains an empty assistant turn and should be skipped."""


class SequenceTooLongError(ValueError):
    """The encoded sample does not fit under the configured scoring limit."""


def _template_from_training_config(model_dir: Path) -> str | None:
    root = Path(__file__).resolve().parents[2]
    names = [model_dir.name]
    if model_dir.name.startswith("checkpoint-"):
        names.append(model_dir.parent.name)
    for name in names:
        for suffix in ("yaml", "yml"):
            for group in ("baseline", "selective"):
                path = root / "configs" / group / f"{name}.{suffix}"
                if not path.is_file():
                    continue
                match = re.search(r"(?m)^template:\s*([^\s#]+)", path.read_text(encoding="utf-8"))
                if match and match.group(1) in SUPPORTED_TEMPLATES:
                    return match.group(1)
    return None


def resolve_template(requested: str, model_dir: Path, input_path: Path) -> str:
    if requested != "auto":
        return requested
    metadata_path = model_dir / "training_template.json"
    if metadata_path.is_file():
        value = json.loads(metadata_path.read_text(encoding="utf-8")).get("training_template")
        if value in SUPPORTED_TEMPLATES:
            return str(value)
    configured = _template_from_training_config(model_dir)
    if configured is not None:
        return configured

    config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    model_type = str(config.get("model_type", "")).lower()
    if "llama" in model_type:
        return "llama3"
    if model_type == "gemma3":
        return "gemma3"
    if "gemma4" in model_type:
        return "gemma4"
    if model_type == "mimo":
        return "mimo"
    if "qwen" in model_type:
        hint = f"{model_dir} {input_path}".lower()
        return "qwen3" if any(tag in hint for tag in ("openr1", "openmath", "reason")) else "qwen3_nothink"
    raise ValueError("cannot infer the training template; pass --template explicitly")


def _string_field(record: dict[str, Any], requested: str | None, candidates: Sequence[str], kind: str) -> str:
    fields = (requested,) if requested else candidates
    for field in fields:
        if field is not None and isinstance(record.get(field), str):
            return str(record[field])
    raise ValueError(f"cannot find a string {kind} field; tried {list(fields)}")


def extract_conversation(record: dict[str, Any], args: argparse.Namespace) -> tuple[list[dict[str, str]], str | None]:
    messages = record.get(args.messages_field)
    record_system = record.get(args.system_field)
    if record_system is not None and not isinstance(record_system, str):
        raise ValueError(f"{args.system_field!r} must be a string")

    if isinstance(messages, list) and messages:
        system_parts: list[str] = []
        conversation: list[dict[str, str]] = []
        for message in messages:
            if not isinstance(message, dict):
                raise ValueError("every message must be an object")
            role = message.get("role")
            content = message.get("content")
            if not isinstance(content, str):
                raise ValueError("every message content must be a string")
            if role == "system":
                if conversation:
                    raise ValueError("system messages are only supported before the first user message")
                system_parts.append(content)
            elif role in {"user", "assistant"}:
                conversation.append({"role": str(role), "content": content})
            else:
                raise ValueError(f"unsupported message role: {role!r}")
        system = "\n\n".join(system_parts) if system_parts else record_system
    else:
        prompt = _string_field(record, args.prompt_field, ("prompt", "problem", "instruction"), "prompt")
        response = _string_field(
            record,
            args.response_field,
            ("response", "generated_solution", "output"),
            "response",
        )
        conversation = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response},
        ]
        system = record_system

    if not conversation or len(conversation) % 2:
        raise ValueError("conversation must contain complete user/assistant pairs")
    for index, message in enumerate(conversation):
        expected = "user" if index % 2 == 0 else "assistant"
        if message["role"] != expected:
            raise ValueError(f"message {index} has role {message['role']!r}; expected {expected!r}")
    return conversation, system


def _encode_piece(tokenizer: Any, text: str) -> list[int]:
    return [int(token_id) for token_id in tokenizer.encode(text, add_special_tokens=False)]


def _infer_seqlen(source_len: int, target_len: int, cutoff_len: int) -> tuple[int, int]:
    """Mirror LLaMA-Factory's supervised ``infer_seqlen`` exactly."""
    if target_len * 2 < cutoff_len:
        max_target_len = cutoff_len
    elif source_len * 2 < cutoff_len:
        max_target_len = cutoff_len - source_len
    else:
        max_target_len = int(cutoff_len * (target_len / (source_len + target_len)))

    new_target_len = min(max_target_len, target_len)
    max_source_len = max(cutoff_len - new_target_len, 0)
    new_source_len = min(max_source_len, source_len)
    return new_source_len, new_target_len


def _validate_gemma4_response(response_text: str) -> None:
    """Reject the Qwen/MiMo thought grammar before token scoring.

    Gemma-4's reasoning template only recognizes its native channel markers.
    Silently accepting ``<think>`` here would recreate LLaMA-Factory's old
    malformed target (an empty native block followed by literal Qwen tags).
    """

    if "<think>" in response_text or "</think>" in response_text:
        raise ValueError(
            "Gemma-4 score input uses Qwen/MiMo <think> tags; materialize "
            "gemma4_native_thought_channel_v1 first"
        )
    if (
        not response_text.startswith(GEMMA4_THOUGHT_OPEN)
        or response_text.count(GEMMA4_THOUGHT_OPEN) != 1
        or response_text.count(GEMMA4_THOUGHT_CLOSE) != 1
    ):
        raise ValueError(
            "Gemma-4 score input must contain exactly one native "
            "<|channel>thought\\n...<channel|> response block"
        )


def encode_record(
    source_index: int,
    record: dict[str, Any],
    tokenizer: Any,
    template: str,
    args: argparse.Namespace,
) -> EncodedSample:
    conversation, system = extract_conversation(record, args)
    if template == "mimo" and not system:
        # Match LLaMA-Factory's registered ``mimo`` template exactly.  The
        # model-native tokenizer template uses the same default system text.
        system = "You are a helpful assistant."
    if any(message["role"] == "assistant" and not message["content"].strip() for message in conversation):
        raise EmptyResponseError("assistant response is empty")
    prefix_ids: list[int] = []
    if template in ("gemma3", "gemma4", "llama3") and tokenizer.bos_token_id is not None:
        prefix_ids.append(int(tokenizer.bos_token_id))
    if template == "gemma4" and not system:
        # Match LLaMA-Factory's registered Gemma-4 reasoning template.
        system = "You are a helpful assistant."
    if system:
        if template == "llama3":
            prefix_ids += _encode_piece(
                tokenizer,
                f"<|start_header_id|>system<|end_header_id|>\n\n{system}<|eot_id|>",
            )
        elif template == "gemma4":
            prefix_ids += _encode_piece(tokenizer, f"<|turn>system\n<|think|>{system}<turn|>\n")
        elif template != "gemma3":
            prefix_ids += _encode_piece(tokenizer, f"<|im_start|>system\n{system}<|im_end|>\n")

    encoded_pairs: list[tuple[list[int], list[int], int]] = []
    for turn in range(0, len(conversation), 2):
        user_text = conversation[turn]["content"]
        response_text = conversation[turn + 1]["content"]
        if template == "llama3":
            prompt_text = (
                f"<|start_header_id|>user<|end_header_id|>\n\n{user_text}<|eot_id|>"
                "<|start_header_id|>assistant<|end_header_id|>\n\n"
            )
            response_end_text = "<|eot_id|>"
        elif template == "gemma3":
            # Gemma has no system role. LLaMA-Factory's Llama2Template fuses
            # format_system into the content of the first user turn.
            if turn == 0 and system:
                user_text = f"{system}\n\n{user_text}"
            prompt_text = f"<start_of_turn>user\n{user_text}<end_of_turn>\n<start_of_turn>model\n"
            response_end_text = "<end_of_turn>\n"
        elif template == "gemma4":
            prompt_text = f"<|turn>user\n{user_text}<turn|>\n<|turn>model\n"
            _validate_gemma4_response(response_text)
            response_end_text = "<turn|>\n"
        else:
            # qwen3_nothink intentionally follows LLaMA-Factory's registered
            # qwen3_nothink format exactly. Unlike Qwen3's model-native chat
            # template, it does not inject an empty <think>...</think> block.
            prompt_text = f"<|im_start|>user\n{user_text}<|im_end|>\n<|im_start|>assistant\n"
            if template in ("mimo", "qwen3") and "<think>" not in response_text and "</think>" not in response_text:
                response_text = f"<think>\n\n</think>\n\n{response_text}"
            response_end_text = "<|im_end|>\n"

        source_ids = _encode_piece(tokenizer, prompt_text)
        if turn == 0:
            source_ids = prefix_ids + source_ids
        response_ids = _encode_piece(tokenizer, response_text + response_end_text)
        response_end_ids = _encode_piece(tokenizer, response_end_text)
        if not response_end_ids or response_ids[-len(response_end_ids) :] != response_end_ids:
            raise ValueError("cannot isolate the template-added response end tokens")
        response_content_length = len(response_ids) - len(response_end_ids)
        encoded_pairs.append((source_ids, response_ids, response_content_length))

    token_ids: list[int] = []
    response_positions: list[int] = []
    response_target_positions: list[int] = []
    truncate_to_length = getattr(args, "truncate_to_length", None)
    for source_ids, target_ids, response_content_length in encoded_pairs:
        if truncate_to_length is not None:
            remaining = truncate_to_length - len(token_ids)
            if remaining <= 0:
                break
            source_len, target_len = _infer_seqlen(len(source_ids), len(target_ids), remaining)
            source_ids = source_ids[:source_len]
            target_ids = target_ids[:target_len]

        token_ids += source_ids
        target_start = len(token_ids)
        token_ids += target_ids
        retained_content_length = min(response_content_length, len(target_ids))
        response_positions += list(range(target_start, target_start + retained_content_length))
        response_target_positions += list(range(target_start, target_start + len(target_ids)))

    if not token_ids or not response_target_positions:
        raise ValueError("tokenized sequence or supervised response targets are empty")
    if response_positions and response_positions[0] == 0:
        raise ValueError("the first response token has no preceding context")
    if len(token_ids) >= args.max_model_len:
        raise SequenceTooLongError(
            f"tokenized length {len(token_ids)} leaves no room for one scoring token "
            f"under --max-model-len={args.max_model_len}"
        )
    return EncodedSample(source_index, token_ids, response_positions, response_target_positions)


def add_common_arguments(
    parser: argparse.ArgumentParser,
    *,
    include_score_transform: bool = True,
) -> None:
    parser.add_argument("--input", required=True, type=Path, help="Input JSONL.")
    parser.add_argument("--model-dir", required=True, type=Path, help="Local Hugging Face model directory.")
    parser.add_argument("--output", required=True, type=Path, help="Output JSONL path.")
    parser.add_argument("--template", choices=("auto", *SUPPORTED_TEMPLATES), default="auto")
    parser.add_argument("--messages-field", default="messages")
    parser.add_argument("--prompt-field", default=None)
    parser.add_argument("--response-field", default=None)
    parser.add_argument("--system-field", default="system")
    parser.add_argument("--id-field", default=None)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--progress-interval",
        type=int,
        default=1,
        help="Print each rank's progress every N completed records.",
    )
    parser.add_argument(
        "--data-parallel-size",
        type=int,
        default=8,
        help="Number of one-model-per-GPU JSONL shards.",
    )
    parser.add_argument("--max-model-len", type=int, default=32768)
    parser.add_argument(
        "--truncate-to-length",
        type=int,
        default=None,
        help=(
            "Apply LLaMA-Factory supervised truncation before scoring. Use the "
            "training cutoff here so score metadata and token scores describe "
            "the exact sequence consumed by training."
        ),
    )
    parser.add_argument(
        "--activation-checkpoint-min-length",
        type=int,
        default=8192,
        help=(
            "Enable decoder activation checkpointing at this sequence length; "
            "use 1 to force it for every non-empty sample."
        ),
    )
    parser.add_argument(
        "--lm-head-chunk-size",
        type=int,
        default=1024,
        help="Number of supervised target positions per FP32 LM-head loss chunk.",
    )
    parser.add_argument(
        "--cuda-cache-cleanup-interval",
        type=int,
        default=0,
        help=(
            "Call CUDA empty_cache after every N successfully scored samples per rank; "
            "0 disables it. This is useful for architectures whose variable-length "
            "activation shapes fragment the caching allocator."
        ),
    )
    parser.add_argument("--dtype", choices=("auto", "bfloat16", "float16", "float32"), default="bfloat16")
    if include_score_transform:
        parser.add_argument(
            "--score-transform",
            choices=("signed", "positive"),
            default="signed",
            help=(
                "Store the original signed score, or reproduce the legacy "
                "max(0, score) clipping. Use signed for lossless scoring data."
            ),
        )
    parser.add_argument(
        "--device",
        default="auto",
        help="Use auto for multi-GPU; an explicit device is only supported with --data-parallel-size=1.",
    )
    parser.add_argument(
        "--attn-implementation",
        choices=("eager", "sdpa", "flash_attention_2", "flex_attention"),
        default=None,
    )
    parser.add_argument("--trust-remote-code", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")


def validate_common_args(args: argparse.Namespace) -> None:
    args.input = args.input.expanduser().resolve()
    args.model_dir = args.model_dir.expanduser().resolve()
    args.output = args.output.expanduser().resolve()
    if not args.input.is_file():
        raise FileNotFoundError(f"input JSONL not found: {args.input}")
    if args.input.suffix.lower() != ".jsonl":
        raise ValueError("--input must be a .jsonl file")
    if not (args.model_dir / "config.json").is_file():
        raise FileNotFoundError(f"model config not found: {args.model_dir / 'config.json'}")
    if args.start_index < 0:
        raise ValueError("--start-index must be non-negative")
    if args.limit is not None and args.limit < 0:
        raise ValueError("--limit must be non-negative")
    if args.max_model_len < 2:
        raise ValueError("--max-model-len must be at least 2")
    if args.truncate_to_length is not None:
        if args.truncate_to_length < 2:
            raise ValueError("--truncate-to-length must be at least 2")
        if args.truncate_to_length >= args.max_model_len:
            raise ValueError("--truncate-to-length must be smaller than --max-model-len")
    if args.activation_checkpoint_min_length < 1:
        raise ValueError("--activation-checkpoint-min-length must be positive")
    if args.lm_head_chunk_size < 1:
        raise ValueError("--lm-head-chunk-size must be positive")
    if args.cuda_cache_cleanup_interval < 0:
        raise ValueError("--cuda-cache-cleanup-interval must be non-negative")
    if args.data_parallel_size < 1:
        raise ValueError("--data-parallel-size must be positive")
    if args.progress_interval < 1:
        raise ValueError("--progress-interval must be positive")
    if args.data_parallel_size > 1 and args.device != "auto":
        raise ValueError("an explicit --device requires --data-parallel-size=1")


def count_input_records(path: Path) -> int:
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for _line in handle)


def selected_end(total: int, start: int, limit: int | None) -> int:
    if start >= total:
        return total
    return total if limit is None else min(total, start + limit)


def iter_rank_records(
    path: Path,
    rank: int,
    world_size: int,
    start: int,
    end: int,
) -> Iterator[tuple[int, dict[str, Any]]]:
    """Yield one modulo shard without parsing records assigned to other ranks."""
    with path.open("r", encoding="utf-8") as handle:
        for source_index, line in enumerate(handle):
            if source_index >= end:
                break
            if source_index < start or source_index % world_size != rank:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{source_index + 1}") from exc
            if not isinstance(record, dict):
                raise TypeError(f"JSONL value is not an object at {path}:{source_index + 1}")
            yield source_index, record


def iter_rank_record_entries(
    path: Path,
    rank: int,
    world_size: int,
    start: int,
    end: int,
) -> Iterator[tuple[int, dict[str, Any] | None, Exception | None]]:
    """Yield assigned lines while keeping malformed JSONL rows recoverable."""
    with path.open("r", encoding="utf-8") as handle:
        for source_index, line in enumerate(handle):
            if source_index >= end:
                break
            if source_index < start or source_index % world_size != rank:
                continue
            try:
                if not line.strip():
                    raise ValueError("blank JSONL line")
                record = json.loads(line)
                if not isinstance(record, dict):
                    raise TypeError("JSONL value is not an object")
            except Exception as exc:
                yield source_index, None, exc
                continue
            yield source_index, record, None


def iter_result_lines(
    path: Path,
    expected_method: str,
    *,
    allow_public_sample_id: bool = False,
) -> Iterator[tuple[int, str]]:
    """Validate an internal part file and yield its source indices."""
    previous = -1
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise ValueError(f"blank output line: {path}:{line_number}")
            value = json.loads(line)
            if not isinstance(value, dict):
                raise TypeError(f"output value is not an object: {path}:{line_number}")
            source_index = value.get("_source_index", value.get("source_index"))
            is_public_merged_record = source_index is None
            # Merged public outputs intentionally omit the private source-index
            # field.  With the default ID policy, sample_id is the decimal source
            # index and is sufficient to resume an interrupted merged cache.
            if source_index is None and allow_public_sample_id:
                sample_id = value.get("sample_id")
                if isinstance(sample_id, str) and sample_id.isdecimal():
                    source_index = int(sample_id)
            if not isinstance(source_index, int) or source_index <= previous:
                raise ValueError(f"source_index is not strictly increasing: {path}:{line_number}")
            method = value.get("_score_method", value.get("score_method"))
            # The final public cache strips private provenance fields. Its
            # sidecar manifest carries the score method and is validated in
            # prepare_parts before an interrupted run is resumed.
            if method != expected_method and not (
                method is None and is_public_merged_record and allow_public_sample_id
            ):
                raise ValueError(f"score_method mismatch: {path}:{line_number}")
            previous = source_index
            yield source_index, line


def error_output_path(output_path: Path) -> Path:
    return output_path.with_name(f"{output_path.stem}.errors{output_path.suffix}")


def iter_error_lines(path: Path) -> Iterator[tuple[int, str]]:
    """Validate an error file and yield source indices in ascending order."""
    previous = -1
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise ValueError(f"blank error line: {path}:{line_number}")
            value = json.loads(line)
            if not isinstance(value, dict) or tuple(value) != ERROR_OUTPUT_KEYS:
                raise ValueError(f"invalid error entry: {path}:{line_number}")
            source_index = value["source_index"]
            if not isinstance(source_index, int) or source_index <= previous:
                raise ValueError(f"error source_index is not strictly increasing: {path}:{line_number}")
            if value["line_number"] != source_index + 1:
                raise ValueError(f"error line_number mismatch: {path}:{line_number}")
            previous = source_index
            yield source_index, line


def completed_rank_indices(
    output_path: Path,
    part_path: Path,
    expected_method: str,
    rank: int,
    world_size: int,
    error_output: Path | None = None,
    error_part: Path | None = None,
    allow_public_sample_id: bool = False,
) -> set[int]:
    completed: set[int] = set()
    for path in (output_path, part_path):
        if not path.exists():
            continue
        for source_index, _line in iter_result_lines(
            path,
            expected_method,
            allow_public_sample_id=allow_public_sample_id and path == output_path,
        ):
            if source_index % world_size == rank:
                completed.add(source_index)
            elif path == part_path:
                raise ValueError(f"part file {part_path} contains source_index={source_index} for the wrong rank")
    for path in (error_output, error_part):
        if path is None or not path.exists():
            continue
        for source_index, _line in iter_error_lines(path):
            if source_index % world_size == rank:
                completed.add(source_index)
            elif path == error_part:
                raise ValueError(
                    f"error part file {error_part} contains source_index={source_index} for the wrong rank"
                )
    return completed


def resolve_device(name: str):
    import torch

    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def resolve_dtype(name: str):
    import torch

    if name == "auto":
        return "auto"
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[name]


def install_gamma4_unified_scoring_kernels() -> tuple[type[Any], type[Any]]:
    """Install memory-safe fused kernels before Gemma4 Unified is built.

    The native Unified RMSNorm materializes the complete activation in fp32.
    That temporary is particularly expensive for long-sequence gradient
    scoring because checkpoint recomputation happens while the rest of the
    backward graph is resident.  Liger's Gemma4 implementations preserve the
    Unified parameter names and numerical semantics while fusing those
    intermediates.  Replacing the module globals before ``from_pretrained``
    makes every decoder layer use the fused implementations without rewriting
    a loaded model or its state dict.
    """
    try:
        from liger_kernel.transformers.rms_norm import LigerRMSNormForGemma4
        from liger_kernel.transformers.tiled_mlp import LigerTiledGEGLUMLP
        from transformers.models.gemma4_unified import modeling_gemma4_unified
    except ImportError as exc:
        raise RuntimeError(
            "gemma4_unified gradient scoring requires the installed Liger "
            "Gemma4 RMSNorm and GEGLU kernels"
        ) from exc

    class LigerTiledGEGLUMLPForGemma4Unified(LigerTiledGEGLUMLP):
        """Tiled GEGLU with the Unified ``(config, layer_idx)`` signature."""

        def __init__(self, config: Any, layer_idx: int | None = None):
            # Auto-tiling uses ceil(sequence_length / hidden_size), which is
            # only 3--5 shards for this 12B model.  That leaves less than one
            # allocation of headroom on an H100 during Value-gate backward.
            # Sixteen shards preserves the token-wise MLP computation while
            # bounding its recomputation workspace for every sequence shape.
            super().__init__(config, num_shards=16)
            first_shared = config.num_hidden_layers - config.num_kv_shared_layers
            is_shared = layer_idx is not None and layer_idx >= first_shared > 0
            if config.use_double_wide_mlp and is_shared:
                import torch.nn as nn

                self.intermediate_size = config.intermediate_size * 2
                self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
                self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
                self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)

    modeling_gemma4_unified.Gemma4UnifiedRMSNorm = LigerRMSNormForGemma4
    modeling_gemma4_unified.Gemma4UnifiedTextMLP = LigerTiledGEGLUMLPForGemma4Unified
    return LigerRMSNormForGemma4, LigerTiledGEGLUMLPForGemma4Unified


def disable_unused_gamma4_shared_kv_retention(model: Any) -> int:
    """Drop unused full-length K/V retention when the model has no shared layers.

    Gemma4 Unified marks the final layer of each attention type as a K/V writer
    even when ``num_kv_shared_layers`` is zero.  No layer can consume those
    tensors in that configuration, but gradient-checkpoint closures retain the
    mutable dictionary and therefore keep hundreds of MiB alive through
    backward.  Disabling only those dead writes is mathematically exact.
    """
    config = getattr(model, "config", None)
    text_config = getattr(config, "text_config", config)
    if getattr(text_config, "num_kv_shared_layers", None) != 0:
        return 0

    disabled = 0
    for module in model.modules():
        attention = getattr(module, "self_attn", None)
        if attention is None or not hasattr(attention, "store_full_length_kv"):
            continue
        if getattr(attention, "is_kv_shared_layer", False):
            raise RuntimeError(
                "Gemma4 reports num_kv_shared_layers=0 but contains a shared-KV layer"
            )
        if attention.store_full_length_kv:
            attention.store_full_length_kv = False
            disabled += 1
    return disabled


def load_model(args: argparse.Namespace):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = resolve_device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(
        str(args.model_dir),
        trust_remote_code=args.trust_remote_code,
        use_fast=True,
    )
    model_kwargs: dict[str, Any] = {
        "torch_dtype": resolve_dtype(args.dtype),
        "trust_remote_code": args.trust_remote_code,
        "low_cpu_mem_usage": True,
    }
    if args.attn_implementation is not None:
        model_kwargs["attn_implementation"] = args.attn_implementation
    config_data = json.loads((args.model_dir / "config.json").read_text(encoding="utf-8"))
    is_gamma4_unified = config_data.get("model_type") == "gemma4_unified"
    gamma4_kernel_classes: tuple[type[Any], type[Any]] | None = None
    if is_gamma4_unified:
        gamma4_kernel_classes = install_gamma4_unified_scoring_kernels()
        try:
            from transformers import AutoModelForMultimodalLM
        except ImportError as exc:
            raise RuntimeError(
                "gemma4_unified requires the Transformers version declared by "
                "the checkpoint (5.10.0.dev0 or a compatible newer build)"
            ) from exc
        model_class = AutoModelForMultimodalLM
        if args.attn_implementation == "flex_attention":
            from transformers.integrations.flex_attention import flex_attention_forward
            from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS, flex_attention_mask
            from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

            def gamma4_flex_attention_forward(*forward_args, **forward_kwargs):
                # The default head_dim=512 FlexAttention kernel exceeds H100
                # shared memory. Use the same tested conservative tiling as the
                # Gamma4 LLaMAFactory training path.
                forward_kwargs.setdefault(
                    "kernel_options",
                    {
                        "BLOCK_M": 32,
                        "BLOCK_N": 32,
                        "num_stages": 1,
                        "num_warps": 4,
                        "bwd_BLOCK_M1": 32,
                        "bwd_BLOCK_N1": 32,
                        "bwd_BLOCK_M2": 32,
                        "bwd_BLOCK_N2": 32,
                    },
                )
                return flex_attention_forward(*forward_args, **forward_kwargs)

            safe_attention_name = "gamma4_flex_attention"
            ALL_ATTENTION_FUNCTIONS.register(safe_attention_name, gamma4_flex_attention_forward)
            # Transformers dispatches attention math and mask construction via
            # separate registries. Without this registration the custom name
            # silently loses the causal mask and value-gradient/NLL scores are
            # computed with access to future response tokens.
            ALL_MASK_ATTENTION_FUNCTIONS.register(safe_attention_name, flex_attention_mask)
            model_kwargs["attn_implementation"] = safe_attention_name
    else:
        model_class = AutoModelForCausalLM

    if device.type == "cuda":
        # Loading eight CPU copies concurrently can exhaust host RAM and leave
        # every GPU idle while safetensors are materialized. Each spawned rank
        # sees only its assigned GPU, so dispatch the full model there directly.
        model_kwargs["device_map"] = {"": str(device)}
        model = model_class.from_pretrained(str(args.model_dir), **model_kwargs)
    else:
        model = model_class.from_pretrained(str(args.model_dir), **model_kwargs).to(device)
    if gamma4_kernel_classes is not None:
        rmsnorm_class, mlp_class = gamma4_kernel_classes
        rmsnorm_count = sum(isinstance(module, rmsnorm_class) for module in model.modules())
        mlp_count = sum(isinstance(module, mlp_class) for module in model.modules())
        unused_shared_kv_disabled = disable_unused_gamma4_shared_kv_retention(model)
        if rmsnorm_count == 0 or mlp_count == 0:
            raise RuntimeError(
                "Gemma4 Unified fused scoring kernels were not installed in the loaded model: "
                f"rmsnorm_count={rmsnorm_count}, mlp_count={mlp_count}"
            )
        print(
            f"Gemma4 Unified fused scoring kernels active: "
            f"rmsnorm={rmsnorm_count}, tiled_mlp={mlp_count}, "
            f"mlp_shards=16, unused_shared_kv_disabled={unused_shared_kv_disabled}",
            flush=True,
        )
    model.eval()
    return model, tokenizer, device


def encode_sample(
    source_index: int,
    record: dict[str, Any],
    tokenizer: Any,
    args: argparse.Namespace,
    template: str,
) -> EncodedSample:
    return encode_record(source_index, record, tokenizer, template, args)


def sample_header(
    sample: EncodedSample,
    record: dict[str, Any],
    args: argparse.Namespace,
    score_definition: str,
) -> dict[str, Any]:
    del score_definition
    sample_id = str(record.get(args.id_field, sample.source_index)) if args.id_field else str(sample.source_index)
    return {
        "sample_id": sample_id,
        "sequence_token_count": len(sample.token_ids),
        "response_token_count": len(sample.response_positions),
    }


def token_info(tokenizer: Any, token_id: int, response_index: int, sequence_position: int) -> dict[str, Any]:
    return {
        "response_token_index": response_index,
        "sequence_token_index": sequence_position,
        "token_id": int(token_id),
        "token_text": tokenizer.decode([int(token_id)], clean_up_tokenization_spaces=False),
    }


def sequence_nll(model: Any, input_ids: Any) -> Any:
    """Return per-position NLLs; column k predicts input_ids[:, k + 1]."""
    import torch.nn.functional as F

    attention_mask = input_ids.new_ones(input_ids.shape)
    outputs = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
    logits = outputs.logits
    return F.cross_entropy(
        logits[:, :-1, :].contiguous().transpose(1, 2).float(),
        input_ids[:, 1:].contiguous(),
        reduction="none",
    )


def response_nlls(losses: Any, response_positions: list[int]) -> Any:
    if not response_positions:
        raise ValueError("sample has no response positions")
    return losses[0, [position - 1 for position in response_positions]].float().cpu()


def token_metrics(
    tokenizer: Any,
    sample: EncodedSample,
    response_index: int,
    baseline_losses: Any,
    deleted_losses: Any,
) -> dict[str, Any]:
    """Compare future response targets after deleting one sequence token."""
    import math

    position = sample.response_positions[response_index]
    future_positions = [item for item in sample.response_target_positions if item > position]
    info = token_info(tokenizer, sample.token_ids[position], response_index, position)
    info["token_nll"] = float(baseline_losses[0, position - 1].item())
    info["future_response_target_count"] = len(future_positions)
    if not future_positions:
        # Defensive fallback for a malformed template without an end target.
        info.update(
            baseline_future_response_target_nll_sum=0.0,
            deleted_future_response_target_nll_sum=0.0,
            future_response_target_nll_delta_sum=0.0,
            positive_future_response_target_nll_delta_sum=0.0,
            baseline_future_response_target_nll_mean=None,
            deleted_future_response_target_nll_mean=None,
            future_response_target_nll_delta_mean=None,
            positive_future_response_target_nll_delta_mean=None,
        )
        return info

    baseline_values = [baseline_losses[0, position - 1].item() for position in future_positions]
    # Deleting position p shifts every later target q to q-1, whose NLL lives
    # at q-2 because NLL column k predicts sequence position k+1.
    deleted_values = [deleted_losses[0, position - 2].item() for position in future_positions]
    baseline_sum = math.fsum(float(value) for value in baseline_values)
    deleted_sum = math.fsum(float(value) for value in deleted_values)
    influence = deleted_sum - baseline_sum
    info.update(
        baseline_future_response_target_nll_sum=baseline_sum,
        deleted_future_response_target_nll_sum=deleted_sum,
        future_response_target_nll_delta_sum=influence,
        positive_future_response_target_nll_delta_sum=max(0.0, influence),
        baseline_future_response_target_nll_mean=baseline_sum / len(future_positions),
        deleted_future_response_target_nll_mean=deleted_sum / len(future_positions),
        future_response_target_nll_delta_mean=influence / len(future_positions),
        positive_future_response_target_nll_delta_mean=max(0.0, influence / len(future_positions)),
    )
    return info


def deletion_token_score(
    sample: EncodedSample,
    response_index: int,
    baseline_losses: Any,
    deleted_losses: Any,
) -> float:
    """Return signed deleted future-target NLL minus baseline NLL."""
    import math

    position = sample.response_positions[response_index]
    future_positions = [item for item in sample.response_target_positions if item > position]
    if not future_positions:
        return 0.0
    baseline_sum = math.fsum(float(baseline_losses[0, item - 1].item()) for item in future_positions)
    deleted_sum = math.fsum(float(deleted_losses[0, item - 2].item()) for item in future_positions)
    return deleted_sum - baseline_sum


def make_input(sample: EncodedSample, device: Any):
    import torch

    return torch.tensor([sample.token_ids], dtype=torch.long, device=device)


def visible_gpus(world_size: int) -> list[str]:
    configured = os.environ.get("CUDA_VISIBLE_DEVICES")
    devices = [item.strip() for item in configured.split(",") if item.strip()] if configured else []
    if not devices:
        devices = [str(index) for index in range(world_size)]
    if len(devices) < world_size:
        raise ValueError(
            f"--data-parallel-size={world_size}, but CUDA_VISIBLE_DEVICES exposes only {len(devices)} GPU(s)"
        )
    return devices[:world_size]


def worker_devices(args: argparse.Namespace) -> list[str | None]:
    if args.device != "auto":
        return [None]
    return visible_gpus(args.data_parallel_size)


def expected_rank_count(start: int, end: int, rank: int, world_size: int) -> int:
    first = start + ((rank - start) % world_size)
    if first >= end:
        return 0
    return 1 + (end - 1 - first) // world_size


def manifest_value(
    args: argparse.Namespace,
    template: str,
    total: int,
    end: int,
    scorer_kind: str,
    score_method: str,
) -> dict[str, Any]:
    stat = args.input.stat()
    value = {
        "input": str(args.input),
        "input_size": stat.st_size,
        "input_mtime_ns": stat.st_mtime_ns,
        "model_dir": str(args.model_dir),
        "template": template,
        "messages_field": args.messages_field,
        "prompt_field": args.prompt_field,
        "response_field": args.response_field,
        "system_field": args.system_field,
        "id_field": args.id_field,
        "scorer_kind": scorer_kind,
        "score_method": score_method,
        "data_parallel_size": args.data_parallel_size,
        "dtype": args.dtype,
        "device": args.device,
        "attn_implementation": args.attn_implementation,
        "trust_remote_code": args.trust_remote_code,
        "max_model_len": args.max_model_len,
        "truncate_to_length": getattr(args, "truncate_to_length", None),
        "start_index": args.start_index,
        "end_index": end,
        "total_records": total,
    }
    if template == "gemma4":
        # Do not resume a cache whose values were derived from Qwen-style
        # thought tags accidentally passed through Gemma's native template.
        value["response_format_contract"] = GEMMA4_RESPONSE_FORMAT_CONTRACT
    config_path = args.model_dir / "config.json"
    if args.attn_implementation == "flex_attention" and config_path.is_file():
        try:
            is_gamma4 = json.loads(config_path.read_text(encoding="utf-8")).get("model_type") == "gemma4_unified"
        except (OSError, json.JSONDecodeError):
            is_gamma4 = False
        if is_gamma4:
            # This contract deliberately invalidates every Gemma score cache
            # produced before the custom FlexAttention mask was registered.
            value["attention_mask_contract"] = "gamma4_flex_causal_v1"
    score_transform = getattr(args, "score_transform", None)
    if score_transform is not None:
        value["score_transform"] = score_transform
    return value


def _portable_workspace_path(value: Any) -> tuple[str, ...] | None:
    """Return a mount-independent identity for paths inside the shared workspace."""
    if not isinstance(value, str):
        return None
    parts = Path(value).parts
    for marker in ("token-selection-github", "token-selection", "model", "Qwen"):
        if marker in parts:
            return tuple(parts[parts.index(marker) :])
    return None


def manifest_mismatches(
    actual: dict[str, Any],
    expected: dict[str, Any],
) -> dict[str, tuple[Any, Any]]:
    """Return semantic mismatches while tolerating shared-mount aliases."""
    mismatched = {
        key: (actual.get(key), expected_value)
        for key, expected_value in expected.items()
        if actual.get(key) != expected_value
    }
    for key in ("input", "model_dir"):
        if key not in mismatched:
            continue
        old_value, new_value = mismatched[key]
        old_identity = _portable_workspace_path(old_value)
        new_identity = _portable_workspace_path(new_value)
        if old_identity is not None and old_identity == new_identity:
            mismatched.pop(key)
    return mismatched


def validate_resume_manifest(
    actual: dict[str, Any],
    expected: dict[str, Any],
    path: Path,
) -> None:
    """Reject semantic changes but allow relocation and execution-backend changes."""
    mismatched = manifest_mismatches(actual, expected)
    incompatible = {
        key: value for key, value in mismatched.items() if key != "attn_implementation"
    }
    if incompatible:
        raise ValueError(f"resume settings do not match {path}: {incompatible}")
    if "attn_implementation" in mismatched:
        old_backend, new_backend = mismatched["attn_implementation"]
        print(
            f"resuming with attention backend changed from "
            f"{old_backend!r} to {new_backend!r}: {path}",
            flush=True,
        )


def prepare_parts(
    args: argparse.Namespace,
    template: str,
    total: int,
    end: int,
    scorer_kind: str,
    score_method: str,
) -> Path:
    part_dir = Path(f"{args.output}.parts")
    errors_path = error_output_path(args.output)
    if args.overwrite:
        args.output.unlink(missing_ok=True)
        errors_path.unlink(missing_ok=True)
        output_manifest_path(args.output).unlink(missing_ok=True)
        if part_dir.exists():
            shutil.rmtree(part_dir)

    if args.output.exists():
        public_manifest_path = output_manifest_path(args.output)
        if not public_manifest_path.is_file():
            raise ValueError(f"existing output has no manifest: {public_manifest_path}")
        public_manifest = json.loads(public_manifest_path.read_text(encoding="utf-8"))
        expected_public_manifest = manifest_value(args, template, total, end, scorer_kind, score_method)
        # The attention backend is an execution detail, not part of either
        # scoring definition. Permit switching away from a backend that OOMed
        # after producing a valid ordered prefix. Shared storage is mounted at
        # different absolute prefixes across worker pools, so input/model paths
        # are compared by stable workspace-relative identity; input size and
        # mtime plus every scoring-semantic field must still match exactly.
        validate_resume_manifest(
            public_manifest,
            expected_public_manifest,
            public_manifest_path,
        )
        # Validate before loading any model or spawning workers.
        for _source_index, _line in iter_result_lines(
            args.output,
            score_method,
            allow_public_sample_id=args.id_field is None,
        ):
            pass
    if errors_path.exists():
        for _source_index, _line in iter_error_lines(errors_path):
            pass

    expected_manifest = manifest_value(args, template, total, end, scorer_kind, score_method)
    manifest_path = part_dir / "manifest.json"
    if part_dir.exists():
        if not manifest_path.is_file():
            raise ValueError(f"part directory has no manifest: {part_dir}")
        actual_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        validate_resume_manifest(actual_manifest, expected_manifest, manifest_path)
    else:
        part_dir.mkdir(parents=True)
        manifest_path.write_text(
            json.dumps(expected_manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    return part_dir


def output_manifest_path(output_path: Path) -> Path:
    return output_path.with_name(f"{output_path.stem}.manifest.json")


def write_output_manifest(
    args: argparse.Namespace,
    template: str,
    total: int,
    end: int,
    scorer_kind: str,
    score_method: str,
    output_records: int,
    error_records: int,
) -> None:
    """Persist scoring provenance after temporary resume metadata is removed."""
    value = manifest_value(args, template, total, end, scorer_kind, score_method)
    if scorer_kind == "token_nll":
        score_semantics = "per_response_content_token_negative_log_likelihood"
    else:
        score_semantics = (
            "signed_first_order_response_target_nll_change"
            if args.score_transform == "signed"
            else "positive_clipped_first_order_response_target_nll_change"
        )
    value.update(
        output_records=output_records,
        error_records=error_records,
        score_semantics=score_semantics,
    )
    path = output_manifest_path(args.output)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def resolve_score_one(scorer_kind: str) -> Any:
    if scorer_kind == "value_gradient":
        from offline_token_fast_fix_score import score_one
    elif scorer_kind == "value_ablation":
        from offline_token_value_ablation_score import score_one
    elif scorer_kind == "token_nll":
        entropy_dir = Path(__file__).resolve().parent.parent / "entropy"
        if str(entropy_dir) not in sys.path:
            sys.path.insert(0, str(entropy_dir))
        from offline_token_entropy_score import score_one
    else:
        raise ValueError(f"unknown scorer kind: {scorer_kind}")
    return score_one


def worker_main(
    rank: int,
    gpu: str | None,
    part_path: Path,
    error_part_path: Path,
    template: str,
    scorer_kind: str,
    score_method: str,
    start: int,
    end: int,
    args: argparse.Namespace,
) -> None:
    import traceback

    if gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu
        args.device = "cuda:0"

    errors_path = error_output_path(args.output)
    completed = completed_rank_indices(
        args.output,
        part_path,
        score_method,
        rank,
        args.data_parallel_size,
        errors_path,
        error_part_path,
        allow_public_sample_id=args.id_field is None,
    )
    selected_completed = sum(1 for index in completed if start <= index < end)
    expected = expected_rank_count(start, end, rank, args.data_parallel_size)
    if selected_completed >= expected:
        print(f"[rank {rank}] shard already complete ({expected} records)", flush=True)
        return

    score_one = resolve_score_one(scorer_kind)
    print(f"[rank {rank}] loading model on device={args.device}", flush=True)
    model, tokenizer, device = load_model(args)
    print(f"[rank {rank}] model loaded on device={device}", flush=True)
    written = 0
    failed = 0
    skipped_data_errors = 0
    skipped_empty_response = 0
    skipped_sequence_too_long = 0
    error_part_path.parent.mkdir(parents=True, exist_ok=True)
    with (
        part_path.open("a", encoding="utf-8") as output_handle,
        error_part_path.open("a", encoding="utf-8") as error_handle,
    ):
        for source_index, record, read_error in iter_rank_record_entries(
            args.input,
            rank,
            args.data_parallel_size,
            start,
            end,
        ):
            if source_index in completed:
                continue
            sample_id = (
                str(record.get(args.id_field, source_index))
                if record is not None and args.id_field
                else str(source_index)
            )
            try:
                if read_error is not None:
                    raise read_error
                assert record is not None
                sample = encode_sample(source_index, record, tokenizer, args, template)
            except Exception as exc:
                skipped_data_errors += 1
                if isinstance(exc, EmptyResponseError):
                    skipped_empty_response += 1
                elif isinstance(exc, SequenceTooLongError):
                    skipped_sequence_too_long += 1
                error_entry = {
                    "source_index": source_index,
                    "line_number": source_index + 1,
                    "sample_id": sample_id,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
                error_handle.write(json.dumps(error_entry, ensure_ascii=False, allow_nan=False) + "\n")
                error_handle.flush()
                print(
                    f"[rank {rank} data_error] source_index={source_index} "
                    f"line_number={source_index + 1} sample_id={sample_id}: "
                    f"{type(exc).__name__}: {exc}",
                    file=sys.stderr,
                    flush=True,
                )
                rank_progress = selected_completed + written + skipped_data_errors
                progress_percent = 100.0 * rank_progress / expected
                progress_interval = getattr(args, "progress_interval", 1)
                if rank_progress % progress_interval == 0 or rank_progress == expected:
                    print(
                        f"[rank {rank}] progress={rank_progress}/{expected} "
                        f"({progress_percent:.1f}%) source_index={source_index} skipped_data_error=1",
                        flush=True,
                    )
                continue

            try:
                result = score_one(
                    model=model,
                    tokenizer=tokenizer,
                    device=device,
                    sample=sample,
                    record=record,
                    args=args,
                    score_definition=score_method,
                )
                stored_result = {
                    "_source_index": source_index,
                    "_score_method": score_method,
                    **result,
                }
                output_handle.write(json.dumps(stored_result, ensure_ascii=False, allow_nan=False) + "\n")
                output_handle.flush()
                written += 1
                rank_progress = selected_completed + written + skipped_data_errors
                progress_percent = 100.0 * rank_progress / expected
                progress_interval = getattr(args, "progress_interval", 1)
                if rank_progress % progress_interval == 0 or rank_progress == expected:
                    print(
                        f"[rank {rank}] progress={rank_progress}/{expected} "
                        f"({progress_percent:.1f}%) source_index={source_index} "
                        f"response_tokens={len(sample.response_positions)}",
                        flush=True,
                    )
                cleanup_interval = getattr(args, "cuda_cache_cleanup_interval", 0)
                if (
                    cleanup_interval > 0
                    and written % cleanup_interval == 0
                    and device.type == "cuda"
                ):
                    import gc
                    import torch

                    gc.collect()
                    torch.cuda.empty_cache()
            except Exception as exc:
                failed += 1
                print(
                    f"[rank {rank} runtime_error] source_index={source_index} "
                    f"line_number={source_index + 1} sample_id={sample_id}: {exc}",
                    file=sys.stderr,
                    flush=True,
                )
                if args.fail_fast:
                    traceback.print_exc()
                    raise
    print(
        f"[rank {rank}] done: written={written}, skipped_existing={selected_completed}, "
        f"skipped_data_errors={skipped_data_errors}, "
        f"skipped_empty_response={skipped_empty_response}, "
        f"skipped_sequence_too_long={skipped_sequence_too_long}, failed={failed}",
        flush=True,
    )


def run_workers(
    args: argparse.Namespace,
    part_dir: Path,
    template: str,
    scorer_kind: str,
    score_method: str,
    start: int,
    end: int,
) -> None:
    import multiprocessing as mp

    devices = worker_devices(args)
    context = mp.get_context("spawn")
    processes: list[mp.Process] = []
    for rank in range(args.data_parallel_size):
        if expected_rank_count(start, end, rank, args.data_parallel_size) == 0:
            continue
        process = context.Process(
            target=worker_main,
            args=(
                rank,
                devices[rank],
                part_dir / f"rank-{rank:02d}.jsonl",
                part_dir / "errors" / f"rank-{rank:02d}.jsonl",
                template,
                scorer_kind,
                score_method,
                start,
                end,
                args,
            ),
            name=f"token-score-dp-{rank}",
        )
        process.start()
        processes.append(process)

    while processes:
        failed = next((process for process in processes if process.exitcode not in (None, 0)), None)
        if failed is not None:
            for process in processes:
                if process.is_alive():
                    process.terminate()
            for process in processes:
                process.join()
            raise RuntimeError(
                f"{failed.name} exited with code {failed.exitcode}; part files were kept for resume"
            )
        if all(process.exitcode == 0 for process in processes):
            break
        time.sleep(1)
    for process in processes:
        process.join()


def merge_results(args: argparse.Namespace, part_dir: Path, score_method: str) -> int:
    """K-way merge existing output and rank parts without holding results in memory."""
    sources = ([args.output] if args.output.exists() else []) + sorted(part_dir.glob("rank-*.jsonl"))
    iterators = [iter_result_lines(path, score_method) for path in sources]
    heap: list[tuple[int, int, str]] = []
    for source_id, iterator in enumerate(iterators):
        try:
            source_index, line = next(iterator)
        except StopIteration:
            continue
        heapq.heappush(heap, (source_index, source_id, line))

    temporary = Path(f"{args.output}.tmp")
    merged = 0
    pending_index: int | None = None
    pending_line: str | None = None
    try:
        with temporary.open("w", encoding="utf-8") as output_handle:
            while heap:
                source_index, source_id, line = heapq.heappop(heap)
                if pending_index is None:
                    pending_index, pending_line = source_index, line
                elif source_index == pending_index:
                    if json.loads(line) != json.loads(pending_line):
                        raise ValueError(f"conflicting results for source_index={source_index}")
                else:
                    output_handle.write(final_output_line(pending_line))
                    merged += 1
                    pending_index, pending_line = source_index, line

                try:
                    next_index, next_line = next(iterators[source_id])
                except StopIteration:
                    continue
                heapq.heappush(heap, (next_index, source_id, next_line))

            if pending_line is not None:
                output_handle.write(final_output_line(pending_line))
                merged += 1
            output_handle.flush()
            os.fsync(output_handle.fileno())
        temporary.replace(args.output)
    finally:
        for iterator in iterators:
            iterator.close()
        temporary.unlink(missing_ok=True)
    return merged


def final_output_line(part_line: str) -> str:
    """Strip temporary sharding metadata and enforce the compact public schema."""
    value = json.loads(part_line)
    value.pop("_source_index", None)
    value.pop("_score_method", None)
    if tuple(value) != FINAL_OUTPUT_KEYS:
        raise ValueError(f"unexpected final output keys: {list(value)}")
    return json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n"


def merge_errors(args: argparse.Namespace, part_dir: Path) -> int:
    """Merge recoverable per-line data errors into one ordered JSONL file."""
    output = error_output_path(args.output)
    error_part_dir = part_dir / "errors"
    sources = ([output] if output.exists() else []) + sorted(error_part_dir.glob("rank-*.jsonl"))
    iterators = [iter_error_lines(path) for path in sources]
    heap: list[tuple[int, int, str]] = []
    for source_id, iterator in enumerate(iterators):
        try:
            source_index, line = next(iterator)
        except StopIteration:
            continue
        heapq.heappush(heap, (source_index, source_id, line))

    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    merged = 0
    pending_index: int | None = None
    pending_line: str | None = None
    try:
        with temporary.open("w", encoding="utf-8") as output_handle:
            while heap:
                source_index, source_id, line = heapq.heappop(heap)
                if pending_index is None:
                    pending_index, pending_line = source_index, line
                elif source_index == pending_index:
                    if json.loads(line) != json.loads(pending_line):
                        raise ValueError(f"conflicting errors for source_index={source_index}")
                else:
                    output_handle.write(pending_line)
                    merged += 1
                    pending_index, pending_line = source_index, line

                try:
                    next_index, next_line = next(iterators[source_id])
                except StopIteration:
                    continue
                heapq.heappush(heap, (next_index, source_id, next_line))

            if pending_line is not None:
                output_handle.write(pending_line)
                merged += 1
            output_handle.flush()
            os.fsync(output_handle.fileno())
        temporary.replace(output)
    finally:
        for iterator in iterators:
            iterator.close()
        temporary.unlink(missing_ok=True)
    return merged


def write_results(args: argparse.Namespace, scorer_kind: str, score_method: str) -> None:
    validate_common_args(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    template = resolve_template(args.template, args.model_dir, args.input)
    total = count_input_records(args.input)
    end = selected_end(total, args.start_index, args.limit)
    part_dir = prepare_parts(args, template, total, end, scorer_kind, score_method)

    selected = max(0, end - args.start_index)
    print(
        f"scoring {selected} records with template={template}, "
        f"data_parallel_size={args.data_parallel_size}",
        flush=True,
    )
    if selected:
        run_workers(
            args,
            part_dir,
            template,
            scorer_kind,
            score_method,
            args.start_index,
            end,
        )
    merged = merge_results(args, part_dir, score_method)
    merged_errors = merge_errors(args, part_dir)
    write_output_manifest(
        args,
        template,
        total,
        end,
        scorer_kind,
        score_method,
        merged,
        merged_errors,
    )
    shutil.rmtree(part_dir)
    print(f"wrote {merged} ordered records to {args.output}", flush=True)
    print(
        f"recorded {merged_errors} skipped data lines in {error_output_path(args.output)}",
        flush=True,
    )
