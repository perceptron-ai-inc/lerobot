"""Shared lifecycle controls must not leak into the external PEFT save API."""

from pathlib import Path

import torch
from safetensors.torch import load_file

from lerobot.common.train_utils import save_checkpoint
from lerobot.configs.default import DatasetConfig, PeftConfig
from lerobot.configs.train import TrainPipelineConfig
from lerobot.utils.constants import PRETRAINED_MODEL_DIR
from tests.fixtures.dummy_checkpoint_policy import DummyCheckpointConfig, DummyCheckpointPolicy


class _StrictExternalSavePolicy(DummyCheckpointPolicy):
    """Simulate only PEFT's save call surface; serialize a real dummy policy, not PEFT adapters."""

    def save_pretrained(self, save_directory: str | Path) -> str | None:
        # No **kwargs: an internal finalize_package argument must never reach this branch.
        return super().save_pretrained(save_directory)


def test_peft_checkpoint_does_not_forward_internal_finalization_keyword(tmp_path: Path) -> None:
    """Control for the external branch while native HubMixin saves gain deferral."""
    policy = _StrictExternalSavePolicy(DummyCheckpointConfig(device="cpu"))
    cfg = TrainPipelineConfig(
        dataset=DatasetConfig(repo_id="lerobot/dummy"),
        policy=policy.config,
        peft=PeftConfig(),
    )
    cfg.parallelism.resolve(1)

    save_checkpoint(
        tmp_path,
        step=7,
        cfg=cfg,
        policy=policy,
        optimizer=torch.optim.Adam(policy.parameters()),
    )

    saved = tmp_path / PRETRAINED_MODEL_DIR
    weights = load_file(saved / "model.safetensors")
    torch.testing.assert_close(weights["net.weight"], policy.net.weight)
    assert (saved / "config.json").is_file()
    assert (saved / "train_config.json").is_file()
