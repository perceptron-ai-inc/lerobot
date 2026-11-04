"""P2.02 control: remove model metadata counts, not real training-state counters."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
from accelerate import Accelerator

pytest.importorskip("transformers", reason="requires the parent-owned ISAAC CPU environment")

from lerobot.common.train_utils import resume_after_prepare, resume_before_prepare, save_checkpoint
from lerobot.configs.train import TRAIN_CONFIG_NAME, TrainPipelineConfig
from lerobot.utils.constants import PRETRAINED_MODEL_DIR
from tests.policies.perceptron_isaac.test_checkpoint_resume_paths import (
    _NativeRun,
    native_run,  # noqa: F401 -- real reduced-native policy fixture with owned cleanup
)


def test_model_count_removal_preserves_adam_scheduler_and_moved_resume_step(
    native_run: _NativeRun,  # noqa: F811 -- pytest injects the imported fixture
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    optimizer = torch.optim.AdamW(native_run.policy.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
    for _ in range(2):
        for parameter in native_run.policy.parameters():
            parameter.grad = torch.full_like(parameter, 0.25)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        scheduler.step()
    expected_optimizer = optimizer.state_dict()
    expected_scheduler = scheduler.state_dict()
    assert expected_optimizer["state"]
    assert scheduler.last_epoch == 2
    saved = tmp_path / "checkpoint"
    save_checkpoint(
        saved,
        step=17,
        cfg=native_run.config,
        policy=native_run.policy,
        optimizer=optimizer,
        scheduler=scheduler,
    )
    moved = tmp_path / "moved-checkpoint"
    saved.rename(moved)
    native_run.source.rename(tmp_path / "retired-source")
    config_path = moved / PRETRAINED_MODEL_DIR / TRAIN_CONFIG_NAME
    monkeypatch.setattr(sys, "argv", ["lerobot-train", f"--config_path={config_path}", "--resume=true"])
    resumed = TrainPipelineConfig.from_pretrained(config_path, cli_args=["--resume=true"])
    resumed.parallelism.resolve(1)
    resumed.validate()
    assert resumed.checkpoint_path == moved

    fresh_optimizer = torch.optim.AdamW(native_run.policy.parameters(), lr=0.9)
    fresh_scheduler = torch.optim.lr_scheduler.StepLR(fresh_optimizer, step_size=1, gamma=0.5)
    accelerator = Accelerator(cpu=True, mixed_precision="no")
    prepared_policy, prepared_optimizer = accelerator.prepare(native_run.policy, fresh_optimizer)

    assert resume_before_prepare(resumed) == 17
    resume_after_prepare(resumed, accelerator, prepared_policy, prepared_optimizer, fresh_scheduler)

    restored = prepared_optimizer.state_dict()
    assert restored["param_groups"] == expected_optimizer["param_groups"]
    assert restored["state"].keys() == expected_optimizer["state"].keys()
    for parameter_id, expected_state in expected_optimizer["state"].items():
        actual_state = restored["state"][parameter_id]
        assert actual_state.keys() == expected_state.keys()
        assert actual_state["step"].item() == 2
        for name, expected in expected_state.items():
            actual = actual_state[name]
            assert actual.dtype == expected.dtype
            assert torch.equal(
                actual.reshape(-1).contiguous().view(torch.uint8),
                expected.reshape(-1).contiguous().view(torch.uint8),
            )
    assert fresh_scheduler.state_dict() == expected_scheduler
    assert fresh_scheduler.last_epoch == 2
    # Resumed optimizer progress continues from 2, independently of global checkpoint step 17.
    for parameter in prepared_policy.parameters():
        parameter.grad = torch.full_like(parameter, 0.25)
    prepared_optimizer.step()
    fresh_scheduler.step()
    assert all(state["step"].item() == 3 for state in prepared_optimizer.state_dict()["state"].values())
    assert fresh_scheduler.last_epoch == 3
