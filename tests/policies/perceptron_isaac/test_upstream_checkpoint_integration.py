"""Isaac checkpoint saves under upstream LeRobot's collective, format-selecting checkpoint writer.

Upstream ``save_checkpoint`` calls ``policy.save_pretrained`` on EVERY rank when the policy is
sharded (the full state dict is gathered through a collective), and skips it entirely for the
DCP-only ``checkpoint_format``. These tests pin how the Isaac policy behaves on both paths.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import lerobot.distributed.checkpoint as distributed_checkpoint
import lerobot.distributed.utils as distributed_utils

pytest.importorskip("transformers", reason="transformers is required (install lerobot[perceptron_isaac])")

from lerobot.policies.perceptron_isaac.modeling_perceptron_isaac import PerceptronIsaacPolicy  # noqa: E402
from tests.policies.perceptron_isaac.test_checkpoint_lifecycle import _source_package  # noqa: E402


class _GatherRecorder:
    """Stands in for the collective full-state-dict gather and records who reached it."""

    def __init__(self, result: dict) -> None:
        self.result = result
        self.calls: list[object] = []

    def __call__(self, module: object) -> dict:
        self.calls.append(module)
        return self.result


def test_non_main_rank_joins_the_gather_and_then_writes_nothing(tmp_path: Path, monkeypatch) -> None:
    gather = _GatherRecorder(result={})
    monkeypatch.setattr(distributed_checkpoint, "full_model_state_dict", gather)
    monkeypatch.setattr(distributed_utils, "is_main_process", lambda: False)
    policy = PerceptronIsaacPolicy(_source_package(tmp_path / "source"))
    save_directory = tmp_path / "pretrained_model"
    save_directory.mkdir()

    PerceptronIsaacPolicy._save_pretrained(policy, save_directory)

    assert gather.calls == [policy], "every rank must enter the collective gather"
    assert list(save_directory.iterdir()) == []


def test_finalize_refuses_a_dcp_only_checkpoint_that_skipped_the_isaac_package(tmp_path: Path) -> None:
    pretrained_dir = tmp_path / "pretrained_model"
    (pretrained_dir / "pytorch_model_fsdp_0").mkdir(parents=True)
    (pretrained_dir / "config.json").write_text("{}\n")
    policy = PerceptronIsaacPolicy(_source_package(tmp_path / "source"))

    with pytest.raises(RuntimeError, match="checkpoint_format=safetensors"):
        PerceptronIsaacPolicy.finalize_pretrained_package(policy, pretrained_dir)


def test_fsdp2_wrap_units_are_the_trained_set_and_name_real_classes() -> None:
    """Upstream resolves `_fsdp_wrap_modules` by class name; the names must track the code."""
    from lerobot.policies.perceptron_isaac import modeling_qwen35_vla, modeling_qwen36_moe

    assert PerceptronIsaacPolicy._fsdp_wrap_modules == ["GenesisQwen36DecoderLayer", "ActionExpertBlock"]
    assert isinstance(modeling_qwen36_moe.GenesisQwen36DecoderLayer, type)
    assert isinstance(modeling_qwen35_vla.ActionExpertBlock, type)
