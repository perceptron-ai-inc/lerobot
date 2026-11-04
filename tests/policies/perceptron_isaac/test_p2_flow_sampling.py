"""P2.19: one unsaved flow trajectory; explicit counts and RNG order remain authoritative."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

pytest.importorskip("transformers", reason="requires the parent-owned ISAAC CPU environment")

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.perceptron_isaac.configuration_perceptron_isaac import PerceptronIsaacConfig
from lerobot.policies.perceptron_isaac.modeling_qwen35_vla import build_action_expert_head
from tests.policies.perceptron_isaac.test_action_expert_contract import _dit_contract


def test_unsaved_default_samples_one_trajectory() -> None:
    config = PerceptronIsaacConfig(device="cpu")
    assert config.num_flow_samples == 1


def test_missing_serialized_sample_count_uses_one_trajectory(tmp_path: Path) -> None:
    PerceptronIsaacConfig(device="cpu").save_pretrained(tmp_path)
    config_path = tmp_path / "config.json"
    raw = json.loads(config_path.read_text())
    raw.pop("num_flow_samples", None)
    config_path.write_text(json.dumps(raw))

    loaded = PreTrainedConfig.from_pretrained(tmp_path, local_files_only=True)

    assert isinstance(loaded, PerceptronIsaacConfig)
    assert loaded.num_flow_samples == 1


@pytest.mark.parametrize("count", [1, 3, 4])
def test_explicit_serialized_sample_count_and_cli_override_remain_authoritative(
    tmp_path: Path, count: int
) -> None:
    PerceptronIsaacConfig(device="cpu", num_flow_samples=count).save_pretrained(tmp_path)
    loaded = PreTrainedConfig.from_pretrained(tmp_path, local_files_only=True)
    overridden = PreTrainedConfig.from_pretrained(
        tmp_path, local_files_only=True, cli_overrides=["--num_flow_samples=2"]
    )
    assert isinstance(loaded, PerceptronIsaacConfig)
    assert isinstance(overridden, PerceptronIsaacConfig)
    assert loaded.num_flow_samples == count
    assert overridden.num_flow_samples == 2


def test_explicit_real_dit_averaging_preserves_trajectory_order_and_rng() -> None:
    torch.manual_seed(123)
    head = build_action_expert_head(_dit_contract(), vlm_dim=16).eval()
    context = torch.randn(2, 3, 16)
    mask = torch.tensor([[1, 1, 0], [1, 0, 0]])
    with torch.no_grad():
        # Randomize the zero-initialized output projection so this is not only a noise test.
        for parameter in head.parameters():
            parameter.uniform_(-0.15, 0.15)
        torch.manual_seed(456)
        averaged = head.sample(context, mask, num_steps=2, num_flow_samples=3)
        averaged_rng = torch.get_rng_state().clone()
        torch.manual_seed(456)
        trajectories = [head.sample(context, mask, num_steps=2, num_flow_samples=1) for _ in range(3)]
        sequential_rng = torch.get_rng_state().clone()

    assert not torch.equal(trajectories[0], trajectories[1])
    torch.testing.assert_close(averaged, torch.stack(trajectories).mean(dim=0), rtol=0, atol=0)
    assert torch.equal(averaged_rng, sequential_rng)
