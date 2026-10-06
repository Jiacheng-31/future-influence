#!/usr/bin/env python3
"""Launch think-only selective SFT without changing the existing launcher."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from launcher import persist_no_think_chat_template
from think_algorithm import ThinkSelectiveTrainingConfig
from think_dataset import install_think_selective_dataset


def load_config(argv: list[str]) -> tuple[dict[str, Any], ThinkSelectiveTrainingConfig]:
    if not argv:
        raise ValueError("Usage: think_launcher.py CONFIG.yaml [key=value ...]")
    config_path = Path(argv[0])
    if config_path.suffix.lower() not in {".yaml", ".yml"}:
        raise ValueError("The think-selective launcher requires a YAML config.")
    config = OmegaConf.merge(OmegaConf.load(config_path), OmegaConf.from_cli(argv[1:]))
    raw = OmegaConf.to_container(config, resolve=True)
    if not isinstance(raw, dict):
        raise TypeError("Training config must be a mapping.")
    selection_values = raw.pop("selective_training", None)
    if selection_values is not None and not isinstance(selection_values, dict):
        raise TypeError("selective_training must be a mapping.")
    return raw, ThinkSelectiveTrainingConfig.from_dict(selection_values)


def main() -> None:
    training_config, selection_config = load_config(sys.argv[1:])
    install_think_selective_dataset(selection_config)
    from llamafactory.train.tuner import run_exp

    run_exp(args=training_config)
    persist_no_think_chat_template(training_config)


if __name__ == "__main__":
    main()
