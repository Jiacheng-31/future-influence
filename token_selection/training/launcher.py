#!/usr/bin/env python3
"""Launch standard LLaMAFactory SFT with score-derived selective labels."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from algorithm import SelectiveTrainingConfig
from dataset import install_selective_dataset


# LLaMAFactory's qwen3_nothink training template is plain ChatML: generation
# begins directly after ``<|im_start|>assistant\n``.  Recent upstream Qwen
# tokenizer templates instead inject an empty ``<think>`` block when callers
# pass ``enable_thinking=False``.  Persisting that upstream template with a
# completed no-think run makes native inference disagree with the training
# token sequence.
QWEN3_NOTHINK_CHAT_TEMPLATE = """{%- for message in messages %}
    {%- if message.role in ['system', 'user', 'assistant'] %}
        {{- '<|im_start|>' + message.role + '\\n' + message.content + '<|im_end|>\\n' }}
    {%- endif %}
{%- endfor %}
{%- if add_generation_prompt %}
    {{- '<|im_start|>assistant\\n' }}
{%- endif %}
"""


def load_config(argv: list[str]) -> tuple[dict[str, Any], SelectiveTrainingConfig]:
    if not argv:
        raise ValueError("Usage: launcher.py CONFIG.yaml [key=value ...]")
    config_path = Path(argv[0])
    if config_path.suffix.lower() not in {".yaml", ".yml"}:
        raise ValueError("The selective launcher requires a YAML config.")
    config = OmegaConf.merge(OmegaConf.load(config_path), OmegaConf.from_cli(argv[1:]))
    raw = OmegaConf.to_container(config, resolve=True)
    if not isinstance(raw, dict):
        raise TypeError("Training config must be a mapping.")
    selection_values = raw.pop("selective_training", None)
    if selection_values is not None and not isinstance(selection_values, dict):
        raise TypeError("selective_training must be a mapping.")
    return raw, SelectiveTrainingConfig.from_dict(selection_values)


def persist_no_think_chat_template(training_config: dict[str, Any]) -> None:
    """Keep completed Qwen no-think checkpoints compatible with training."""
    if training_config.get("template") != "qwen3_nothink":
        return
    output_dir = training_config.get("output_dir")
    if not isinstance(output_dir, str) or not output_dir:
        raise ValueError("qwen3_nothink training requires a non-empty output_dir")
    path = Path(output_dir) / "chat_template.jinja"
    path.write_text(QWEN3_NOTHINK_CHAT_TEMPLATE, encoding="utf-8")


def main() -> None:
    training_config, selection_config = load_config(sys.argv[1:])
    install_selective_dataset(selection_config)
    from llamafactory.train.tuner import run_exp

    run_exp(args=training_config)
    persist_no_think_chat_template(training_config)


if __name__ == "__main__":
    main()
