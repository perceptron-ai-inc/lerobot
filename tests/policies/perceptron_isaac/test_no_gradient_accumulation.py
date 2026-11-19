"""ISAAC declines accumulation without changing the stock trainer's other policies."""

import sys
from pathlib import Path
from typing import Literal

import pytest

from lerobot.configs.default import DatasetConfig, EMAConfig
from lerobot.configs.train import TrainPipelineConfig
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.perceptron_isaac.configuration_perceptron_isaac import PerceptronIsaacConfig


@pytest.mark.parametrize("policy_type", ["isaac", "act"])
@pytest.mark.parametrize("accumulation_steps", [1, 2, 8])
def test_accumulation_preflight_is_isaac_specific(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    policy_type: Literal["isaac", "act"],
    accumulation_steps: int,
) -> None:
    """Reject ISAAC accumulation before allocation without restricting ordinary policies."""
    config_type = PerceptronIsaacConfig if policy_type == "isaac" else ACTConfig
    config = TrainPipelineConfig(
        dataset=DatasetConfig(repo_id="fixture/not-downloaded", video_backend="pyav"),
        policy=config_type(device="cpu", push_to_hub=False),
        output_dir=tmp_path / "run",
        steps=16,
        save_checkpoint=False,
        ema=EMAConfig(enable=False),
        env_eval_freq=0,
        eval_steps=0,
    )
    config.accelerator.gradient_accumulation.steps = accumulation_steps
    config.parallelism.resolve(1)
    monkeypatch.setattr(sys, "argv", ["lerobot-train"])
    if policy_type == "isaac" and accumulation_steps > 1:
        with pytest.raises(ValueError, match=r"(?i)isaac.*gradient.accumulation"):
            config.validate()
    else:
        config.validate()
    assert config.output_dir is not None
    assert not config.output_dir.exists()
