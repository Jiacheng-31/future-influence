#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "token_selection" / "scoring"
sys.path.insert(0, str(SCRIPT_DIR))
ENTROPY_DIR = Path(__file__).resolve().parents[1] / "token_selection" / "entropy"
sys.path.insert(0, str(ENTROPY_DIR))

from offline_token_fast_fix_score import score_one as fast_fix_score  # noqa: E402
import offline_token_fast_fix_score as fast_score_module  # noqa: E402
from offline_token_model_utils import response_token_losses  # noqa: E402
from offline_token_value_ablation_score import score_one as value_ablation_score  # noqa: E402
from offline_token_entropy_score import score_one as entropy_score  # noqa: E402
import offline_token_score_common as score_common  # noqa: E402
from offline_token_score_common import (  # noqa: E402
    EncodedSample,
    encode_record,
    expected_rank_count,
    iter_rank_records,
    merge_results,
)


class FakeTokenizer:
    bos_token_id = 1

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        assert not add_special_tokens
        return [ord(character) + 10 for character in text]

    def convert_ids_to_tokens(self, token_id: int) -> str:
        return f"token-{token_id}"

    def decode(self, token_ids: list[int], clean_up_tokenization_spaces: bool = False) -> str:
        del clean_up_tokenization_spaces
        return " ".join(str(token_id) for token_id in token_ids)


class TinySelfAttention(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, hidden):
        query = self.q_proj(hidden)
        key = self.k_proj(hidden)
        value = self.v_proj(hidden)
        scores = query @ key.transpose(-1, -2) / hidden.shape[-1] ** 0.5
        length = hidden.shape[1]
        causal_mask = torch.triu(
            torch.ones(length, length, dtype=torch.bool, device=hidden.device),
            diagonal=1,
        )
        weights = torch.softmax(scores.masked_fill(causal_mask, -torch.inf), dim=-1)
        return self.o_proj(weights @ value)


class TinyDecoderLayer(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.self_attn = TinySelfAttention(hidden_size)

    def forward(self, hidden):
        return torch.tanh(hidden + self.self_attn(hidden))


class TinyBackbone(nn.Module):
    def __init__(self, vocab_size: int = 16, hidden_size: int = 8) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        self.layers = nn.ModuleList([TinyDecoderLayer(hidden_size)])
        self.forward_count = 0

    def get_input_embeddings(self) -> nn.Module:
        return self.embed_tokens

    def forward(self, input_ids=None, attention_mask=None, use_cache=False, inputs_embeds=None):
        del attention_mask, use_cache
        self.forward_count += 1
        if inputs_embeds is None:
            hidden = self.embed_tokens(input_ids)
        else:
            hidden = inputs_embeds
        for layer in self.layers:
            hidden = layer(hidden)
        return SimpleNamespace(last_hidden_state=hidden)


class TinyCausalLM(nn.Module):
    def __init__(self, vocab_size: int = 16, hidden_size: int = 8) -> None:
        super().__init__()
        self.model = TinyBackbone(vocab_size, hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size)

    @property
    def forward_count(self) -> int:
        return self.model.forward_count

    def get_base_model(self) -> nn.Module:
        # Mirrors Transformers causal-LM classes where this API can return the
        # wrapper itself; gradient scoring must still select `.model`.
        return self

    def get_input_embeddings(self) -> nn.Module:
        return self.model.embed_tokens

    def get_output_embeddings(self) -> nn.Module:
        return self.lm_head

    @property
    def is_gradient_checkpointing(self) -> bool:
        return getattr(self, "_gradient_checkpointing", False)

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None) -> None:
        self._gradient_checkpointing = True
        self._gradient_checkpointing_kwargs = gradient_checkpointing_kwargs

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        use_cache=False,
        logits_to_keep=0,
        inputs_embeds=None,
    ):
        hidden = self.model(
            input_ids,
            attention_mask=attention_mask,
            use_cache=use_cache,
            inputs_embeds=inputs_embeds,
        ).last_hidden_state
        if isinstance(logits_to_keep, torch.Tensor):
            hidden = hidden[:, logits_to_keep, :]
        return SimpleNamespace(logits=self.lm_head(hidden))


def score_args(**values) -> argparse.Namespace:
    return argparse.Namespace(id_field=None, **values)


class OfflineTokenScoresTest(unittest.TestCase):
    def setUp(self) -> None:
        self.sample = EncodedSample(0, [1, 2, 3, 4, 5], [2, 3], [2, 3, 4])
        self.tokenizer = FakeTokenizer()

    def test_value_ablation_uses_baseline_plus_one_forward_per_token(self) -> None:
        model = TinyCausalLM()
        result = value_ablation_score(
            model=model,
            tokenizer=self.tokenizer,
            device=torch.device("cpu"),
            sample=self.sample,
            record={},
            args=score_args(ablated_gate_value=0.9),
            score_definition="value_ablation",
        )
        self.assertEqual(model.forward_count, 1 + len(self.sample.response_positions))
        self.assertTrue(all(not parameter.requires_grad for parameter in model.parameters()))
        self.assertTrue(all(parameter.grad is None for parameter in model.parameters()))
        self.assertEqual(len(result["token_scores"]), len(self.sample.response_positions))
        self.assertTrue(all(isinstance(score, float) and math.isfinite(score) for score in result["token_scores"]))
        self.assertEqual(self.sample.response_positions[-1], 3)
        self.assertIn(4, self.sample.response_target_positions)

    def test_response_losses_apply_model_final_logit_softcap(self) -> None:
        torch.manual_seed(11)
        model = TinyCausalLM()
        model.config = SimpleNamespace(
            text_config=SimpleNamespace(final_logit_softcapping=2.0)
        )
        input_ids = torch.tensor([[1, 2, 3, 4]])
        hidden = model.model(input_ids=input_ids).last_hidden_state
        positions = torch.tensor([2, 3])

        actual = response_token_losses(
            model,
            hidden,
            input_ids,
            positions,
            chunk_size=1,
        )
        logits = model.lm_head(hidden[0, positions - 1]).float()
        capped = torch.tanh(logits / 2.0) * 2.0
        expected = torch.nn.functional.cross_entropy(
            capped,
            input_ids[0, positions],
            reduction="none",
        )
        uncapped = torch.nn.functional.cross_entropy(
            logits,
            input_ids[0, positions],
            reduction="none",
        )
        torch.testing.assert_close(actual, expected)
        self.assertFalse(torch.allclose(actual, uncapped))

    def test_fast_fix_value_gate_uses_one_forward_and_one_scalar_per_token(self) -> None:
        model = TinyCausalLM()
        result = fast_fix_score(
            model=model,
            tokenizer=self.tokenizer,
            device=torch.device("cpu"),
            sample=self.sample,
            record={},
            args=score_args(),
            score_definition="value_gradient",
        )
        self.assertEqual(model.forward_count, 1)
        self.assertTrue(all(not parameter.requires_grad for parameter in model.parameters()))
        self.assertTrue(all(parameter.grad is None for parameter in model.parameters()))
        self.assertEqual(len(result["token_scores"]), len(self.sample.response_positions))
        self.assertTrue(all(isinstance(score, float) and math.isfinite(score) for score in result["token_scores"]))

    def test_entropy_score_records_exact_content_token_nlls(self) -> None:
        model = TinyCausalLM()
        input_ids = torch.tensor([self.sample.token_ids], dtype=torch.long)
        target_positions = torch.tensor(self.sample.response_target_positions, dtype=torch.long)
        with torch.no_grad():
            hidden = model.model(input_ids=input_ids).last_hidden_state
            logits = model.lm_head(hidden[0, target_positions - 1, :]).float()
            expected_all = torch.nn.functional.cross_entropy(
                logits,
                input_ids[0, target_positions],
                reduction="none",
            )
        model.model.forward_count = 0
        result = entropy_score(
            model=model,
            tokenizer=self.tokenizer,
            device=torch.device("cpu"),
            sample=self.sample,
            record={},
            args=score_args(lm_head_chunk_size=1),
            score_definition="response_content_token_negative_log_likelihood_v1",
        )
        self.assertEqual(model.forward_count, 1)
        self.assertEqual(len(result["token_scores"]), len(self.sample.response_positions))
        self.assertTrue(all(math.isfinite(score) and score >= 0.0 for score in result["token_scores"]))
        loss_by_position = dict(zip(self.sample.response_target_positions, expected_all.tolist()))
        torch.testing.assert_close(
            torch.tensor(result["token_scores"]),
            torch.tensor([loss_by_position[position] for position in self.sample.response_positions]),
        )
        self.assertAlmostEqual(
            result["baseline_response_target_nll_sum"],
            math.fsum(expected_all.tolist()),
            places=6,
        )

    def test_fast_fix_preserves_negative_scores(self) -> None:
        model = TinyCausalLM()
        gradient = torch.tensor([[0.0, 0.0, -1.25, 2.5, 0.0]])
        with patch("torch.autograd.grad", return_value=(gradient,)):
            result = fast_fix_score(
                model=model,
                tokenizer=self.tokenizer,
                device=torch.device("cpu"),
                sample=self.sample,
                record={},
                args=score_args(),
                score_definition="value_gradient",
            )
        self.assertEqual(result["token_scores"], [1.25, -2.5])

    def test_fast_fix_can_reproduce_legacy_positive_clipping(self) -> None:
        model = TinyCausalLM()
        gradient = torch.tensor([[0.0, 0.0, -1.25, 2.5, 0.0]])
        with patch("torch.autograd.grad", return_value=(gradient,)):
            result = fast_fix_score(
                model=model,
                tokenizer=self.tokenizer,
                device=torch.device("cpu"),
                sample=self.sample,
                record={},
                args=score_args(score_transform="positive"),
                score_definition="value_gradient",
            )
        self.assertEqual(result["token_scores"], [1.25, 0.0])

    def test_value_ablation_preserves_negative_scores(self) -> None:
        model = TinyCausalLM()
        with patch(
            "offline_token_value_ablation_score.gated_forward_losses",
            side_effect=(
                torch.tensor([1.0, 2.0, 3.0]),
                torch.tensor([1.0, 2.0]),
                torch.tensor([4.0]),
            ),
        ):
            result = value_ablation_score(
                model=model,
                tokenizer=self.tokenizer,
                device=torch.device("cpu"),
                sample=self.sample,
                record={},
                args=score_args(ablated_gate_value=0.9),
                score_definition="value_ablation",
            )
        self.assertEqual(result["token_scores"], [-2.0, 1.0])

    def test_fast_fix_low_memory_controls_are_applied(self) -> None:
        model = TinyCausalLM()
        original = fast_score_module.response_token_losses
        with patch.object(fast_score_module, "response_token_losses", wraps=original) as mocked:
            fast_fix_score(
                model=model,
                tokenizer=self.tokenizer,
                device=torch.device("cpu"),
                sample=self.sample,
                record={},
                args=score_args(
                    activation_checkpoint_min_length=1,
                    lm_head_chunk_size=3,
                ),
                score_definition="value_gradient",
            )
        self.assertTrue(model.is_gradient_checkpointing)
        self.assertEqual(model._gradient_checkpointing_kwargs, {"use_reentrant": False})
        self.assertEqual(mocked.call_args.kwargs["chunk_size"], 3)

    def test_gamma4_unified_scoring_kernels_replace_constructor_globals(self) -> None:
        try:
            from liger_kernel.transformers.rms_norm import LigerRMSNormForGemma4
            from liger_kernel.transformers.tiled_mlp import LigerTiledGEGLUMLP
            from transformers.models.gemma4_unified import modeling_gemma4_unified
        except ImportError:
            self.skipTest("Gemma4 Unified/Liger kernels are unavailable in this environment")

        old_rmsnorm = modeling_gemma4_unified.Gemma4UnifiedRMSNorm
        old_mlp = modeling_gemma4_unified.Gemma4UnifiedTextMLP
        try:
            rmsnorm_class, mlp_class = score_common.install_gamma4_unified_scoring_kernels()
            self.assertIs(rmsnorm_class, LigerRMSNormForGemma4)
            self.assertTrue(issubclass(mlp_class, LigerTiledGEGLUMLP))
            self.assertIs(modeling_gemma4_unified.Gemma4UnifiedRMSNorm, LigerRMSNormForGemma4)
            self.assertIs(modeling_gemma4_unified.Gemma4UnifiedTextMLP, mlp_class)

            config = SimpleNamespace(
                hidden_size=8,
                intermediate_size=16,
                num_hidden_layers=4,
                num_kv_shared_layers=0,
                use_double_wide_mlp=False,
            )
            module = mlp_class(config, layer_idx=0)
            self.assertEqual(module.num_shards, 16)
            self.assertEqual(
                set(module.state_dict()),
                {"gate_proj.weight", "up_proj.weight", "down_proj.weight"},
            )
        finally:
            modeling_gemma4_unified.Gemma4UnifiedRMSNorm = old_rmsnorm
            modeling_gemma4_unified.Gemma4UnifiedTextMLP = old_mlp

    def test_gamma4_zero_shared_layers_disable_dead_kv_retention(self) -> None:
        class Attention(nn.Module):
            def __init__(self, store: bool) -> None:
                super().__init__()
                self.store_full_length_kv = store
                self.is_kv_shared_layer = False

        class Layer(nn.Module):
            def __init__(self, store: bool) -> None:
                super().__init__()
                self.self_attn = Attention(store)

        class Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.config = SimpleNamespace(
                    text_config=SimpleNamespace(num_kv_shared_layers=0)
                )
                self.layers = nn.ModuleList([Layer(False), Layer(True), Layer(True)])

        model = Model()
        disabled = score_common.disable_unused_gamma4_shared_kv_retention(model)
        self.assertEqual(disabled, 2)
        self.assertFalse(any(layer.self_attn.store_full_length_kv for layer in model.layers))

    def test_value_gradient_matches_small_fp32_finite_difference(self) -> None:
        torch.manual_seed(7)
        model = TinyCausalLM()
        fast_result = fast_fix_score(
            model=model,
            tokenizer=self.tokenizer,
            device=torch.device("cpu"),
            sample=self.sample,
            record={},
            args=score_args(),
            score_definition="value_gradient",
        )
        gate_value = 0.99
        finite_result = value_ablation_score(
            model=model,
            tokenizer=self.tokenizer,
            device=torch.device("cpu"),
            sample=self.sample,
            record={},
            args=score_args(ablated_gate_value=gate_value),
            score_definition="value_ablation",
        )
        normalized_finite = torch.tensor(finite_result["token_scores"]) / (1.0 - gate_value)
        torch.testing.assert_close(
            normalized_finite,
            torch.tensor(fast_result["token_scores"]),
            rtol=0.08,
            atol=2e-4,
        )

    def test_template_end_tokens_are_targets_but_not_scored_tokens(self) -> None:
        args = argparse.Namespace(
            messages_field="messages",
            prompt_field=None,
            response_field=None,
            system_field="system",
            max_model_len=10000,
        )
        sample = encode_record(
            0,
            {"prompt": "question", "response": "answer"},
            self.tokenizer,
            "llama3",
            args,
        )
        self.assertEqual(len(sample.response_positions), len("answer"))
        self.assertGreater(len(sample.response_target_positions), len(sample.response_positions))
        self.assertGreater(sample.response_target_positions[-1], sample.response_positions[-1])

    def test_gemma3_fuses_system_text_into_first_user_turn(self) -> None:
        args = argparse.Namespace(
            messages_field="messages",
            prompt_field=None,
            response_field=None,
            system_field="system",
            max_model_len=10000,
        )
        sample = encode_record(
            0,
            {"system": "system", "prompt": "question", "response": "answer"},
            self.tokenizer,
            "gemma3",
            args,
        )
        expected = (
            [self.tokenizer.bos_token_id]
            + self.tokenizer.encode(
                "<start_of_turn>user\nsystem\n\nquestion<end_of_turn>\n"
                "<start_of_turn>model\nanswer<end_of_turn>\n",
                add_special_tokens=False,
            )
        )
        self.assertEqual(sample.token_ids, expected)
        self.assertEqual(len(sample.response_positions), len("answer"))

    def test_training_cutoff_uses_llamafactory_truncation_before_scoring(self) -> None:
        common = dict(
            messages_field="messages",
            prompt_field=None,
            response_field=None,
            system_field="system",
            max_model_len=10000,
        )
        record = {"prompt": "question", "response": "r" * 200}
        full = encode_record(0, record, self.tokenizer, "gemma3", argparse.Namespace(**common))
        cutoff = len(full.token_ids) - 20
        truncated = encode_record(
            0,
            record,
            self.tokenizer,
            "gemma3",
            argparse.Namespace(**common, truncate_to_length=cutoff),
        )
        self.assertEqual(len(truncated.token_ids), cutoff)
        self.assertLess(len(truncated.response_positions), len(full.response_positions))
        self.assertEqual(
            truncated.response_positions,
            list(range(truncated.response_positions[0], truncated.response_positions[-1] + 1)),
        )

    def test_gemma4_rejects_qwen_thought_markers(self) -> None:
        args = argparse.Namespace(
            messages_field="messages",
            prompt_field=None,
            response_field=None,
            system_field="system",
            max_model_len=10000,
        )
        with self.assertRaisesRegex(ValueError, "Qwen/MiMo <think>"):
            encode_record(
                0,
                {"prompt": "question", "response": "<think>reasoning</think>answer"},
                self.tokenizer,
                "gemma4",
                args,
            )

    def test_gemma4_accepts_exactly_one_native_thought_block(self) -> None:
        args = argparse.Namespace(
            messages_field="messages",
            prompt_field=None,
            response_field=None,
            system_field="system",
            max_model_len=10000,
        )
        sample = encode_record(
            0,
            {
                "prompt": "question",
                "response": "<|channel>thought\nreasoning<channel|>answer",
            },
            self.tokenizer,
            "gemma4",
            args,
        )
        self.assertGreater(len(sample.response_positions), 0)
        self.assertGreater(len(sample.response_target_positions), len(sample.response_positions))

    def test_empty_response_is_skipped(self) -> None:
        args = argparse.Namespace(
            messages_field="messages",
            prompt_field=None,
            response_field=None,
            system_field="system",
            max_model_len=10000,
        )
        with self.assertRaises(score_common.EmptyResponseError):
            encode_record(0, {"prompt": "question", "response": ""}, self.tokenizer, "llama3", args)

    def test_sequence_over_configured_limit_is_skipped(self) -> None:
        args = argparse.Namespace(
            messages_field="messages",
            prompt_field=None,
            response_field=None,
            system_field="system",
            max_model_len=10,
        )
        with self.assertRaises(score_common.SequenceTooLongError):
            encode_record(0, {"prompt": "question", "response": "answer"}, self.tokenizer, "llama3", args)

    def test_modulo_shards_cover_selected_range_once(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "input.jsonl"
            path.write_text("".join(json.dumps({"value": index}) + "\n" for index in range(10)))
            shards = [
                [index for index, _record in iter_rank_records(path, rank, 3, start=2, end=9)]
                for rank in range(3)
            ]
            self.assertEqual(shards, [[3, 6], [4, 7], [2, 5, 8]])
            self.assertEqual([expected_rank_count(2, 9, rank, 3) for rank in range(3)], [2, 2, 3])
            self.assertEqual(sorted(index for shard in shards for index in shard), list(range(2, 9)))

    def test_resume_manifest_accepts_shared_mount_aliases(self) -> None:
        actual = {
            "input": "/ernie/zhangxinyue/ernie/token-selection/data/input.jsonl",
            "input_size": 123,
            "input_mtime_ns": 456,
            "model_dir": "/ernie/zhangxinyue/ernie/model/gamma-4-12B",
            "template": "gemma4",
            "attn_implementation": "sdpa",
        }
        expected = {
            **actual,
            "input": "/zhangxinyue/zhangxinyue/ernie/token-selection/data/input.jsonl",
            "model_dir": "/zhangxinyue/zhangxinyue/ernie/model/gamma-4-12B",
            "attn_implementation": "flex_attention",
        }
        score_common.validate_resume_manifest(actual, expected, Path("manifest.json"))

    def test_resume_manifest_rejects_different_data_or_model(self) -> None:
        actual = {
            "input": "/ernie/zhangxinyue/ernie/token-selection/data/input.jsonl",
            "input_size": 123,
            "input_mtime_ns": 456,
            "model_dir": "/ernie/zhangxinyue/ernie/model/gamma-4-12B",
        }
        changed_data = {
            **actual,
            "input": "/zhangxinyue/zhangxinyue/ernie/token-selection/data/input.jsonl",
            "input_size": 124,
        }
        with self.assertRaisesRegex(ValueError, "input_size"):
            score_common.validate_resume_manifest(actual, changed_data, Path("manifest.json"))

        changed_model = {
            **actual,
            "model_dir": "/zhangxinyue/zhangxinyue/ernie/model/another-model",
        }
        with self.assertRaisesRegex(ValueError, "model_dir"):
            score_common.validate_resume_manifest(actual, changed_model, Path("manifest.json"))

    def test_rank_parts_merge_in_source_order(self) -> None:
        method = "test_method"

        def line(index: int) -> str:
            return json.dumps(
                {
                    "_source_index": index,
                    "_score_method": method,
                    "sample_id": str(index),
                    "sequence_token_count": 5,
                    "response_token_count": 2,
                    "baseline_response_target_nll_sum": 1.0,
                    "token_scores": [0.1, 0.2],
                }
            ) + "\n"

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "scores.jsonl"
            part_dir = root / "scores.jsonl.parts"
            part_dir.mkdir()
            (part_dir / "rank-00.jsonl").write_text(line(0) + line(2) + line(4), encoding="utf-8")
            (part_dir / "rank-01.jsonl").write_text(line(1) + line(3) + line(5), encoding="utf-8")

            merged = merge_results(argparse.Namespace(output=output), part_dir, method)

            self.assertEqual(merged, 6)
            values = [json.loads(item) for item in output.read_text(encoding="utf-8").splitlines()]
            self.assertEqual([value["sample_id"] for value in values], [str(index) for index in range(6)])
            self.assertTrue(all(set(value) == set(score_common.FINAL_OUTPUT_KEYS) for value in values))

    def test_worker_writes_only_its_gpu_shard(self) -> None:
        method = "test_method"

        def fake_score_one(**kwargs):
            sample = kwargs["sample"]
            return {
                "sample_id": str(sample.source_index),
                "sequence_token_count": len(sample.token_ids),
                "response_token_count": len(sample.response_positions),
                "baseline_response_target_nll_sum": 1.0,
                "token_scores": [0.0] * len(sample.response_positions),
            }

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_path = root / "input.jsonl"
            input_path.write_text(
                "".join(json.dumps({"prompt": "p", "response": "r"}) + "\n" for _ in range(5)),
                encoding="utf-8",
            )
            output_path = root / "scores.jsonl"
            part_path = root / "rank-01.jsonl"
            error_part_path = root / "errors" / "rank-01.jsonl"
            args = argparse.Namespace(
                input=input_path,
                output=output_path,
                data_parallel_size=2,
                device="cpu",
                id_field=None,
                messages_field="messages",
                prompt_field=None,
                response_field=None,
                system_field="system",
                max_model_len=10000,
                fail_fast=True,
            )
            with (
                patch.object(score_common, "load_model", return_value=(object(), self.tokenizer, torch.device("cpu"))),
                patch.object(score_common, "resolve_score_one", return_value=fake_score_one),
            ):
                score_common.worker_main(
                    rank=1,
                    gpu=None,
                    part_path=part_path,
                    error_part_path=error_part_path,
                    template="llama3",
                    scorer_kind="test",
                    score_method=method,
                    start=0,
                    end=5,
                    args=args,
                )

            values = [json.loads(item) for item in part_path.read_text(encoding="utf-8").splitlines()]
            self.assertEqual([value["_source_index"] for value in values], [1, 3])

    def test_worker_skips_bad_data_line_and_records_unambiguous_line_number(self) -> None:
        method = "test_method"

        def fake_score_one(**kwargs):
            sample = kwargs["sample"]
            return {
                "sample_id": str(sample.source_index),
                "sequence_token_count": len(sample.token_ids),
                "response_token_count": len(sample.response_positions),
                "baseline_response_target_nll_sum": 1.0,
                "token_scores": [0.0] * len(sample.response_positions),
            }

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_path = root / "input.jsonl"
            input_path.write_text(
                json.dumps({"prompt": "p", "response": "r"})
                + "\n"
                + json.dumps(
                    {
                        "messages": [
                            {"role": "user", "content": "p"},
                            {"role": "assistant", "content": "r"},
                            {"role": "assistant", "content": "orphan"},
                        ]
                    }
                )
                + "\n"
                + json.dumps({"prompt": "p2", "response": "r2"})
                + "\n",
                encoding="utf-8",
            )
            output_path = root / "scores.jsonl"
            part_path = root / "rank-00.jsonl"
            error_part_path = root / "errors" / "rank-00.jsonl"
            args = argparse.Namespace(
                input=input_path,
                output=output_path,
                data_parallel_size=1,
                device="cpu",
                id_field=None,
                messages_field="messages",
                prompt_field=None,
                response_field=None,
                system_field="system",
                max_model_len=10000,
                progress_interval=1,
                fail_fast=True,
            )
            with (
                patch.object(score_common, "load_model", return_value=(object(), self.tokenizer, torch.device("cpu"))),
                patch.object(score_common, "resolve_score_one", return_value=fake_score_one),
            ):
                score_common.worker_main(
                    rank=0,
                    gpu=None,
                    part_path=part_path,
                    error_part_path=error_part_path,
                    template="llama3",
                    scorer_kind="test",
                    score_method=method,
                    start=0,
                    end=3,
                    args=args,
                )

            results = [json.loads(item) for item in part_path.read_text(encoding="utf-8").splitlines()]
            errors = [json.loads(item) for item in error_part_path.read_text(encoding="utf-8").splitlines()]
            self.assertEqual([value["_source_index"] for value in results], [0, 2])
            self.assertEqual(errors[0]["source_index"], 1)
            self.assertEqual(errors[0]["line_number"], 2)
            self.assertIn("complete user/assistant pairs", errors[0]["error"])


if __name__ == "__main__":
    unittest.main()
