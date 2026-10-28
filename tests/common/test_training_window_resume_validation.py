"""Defensive replay metadata/state edges; parent owns offline CPU execution."""

from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest
import torch
from accelerate import Accelerator

from lerobot.common.train_utils import load_training_metadata, load_training_window, save_training_state
from lerobot.common.training_window import TrainingWindowCheckpoint
from lerobot.configs.default import DatasetConfig
from lerobot.configs.train import TrainPipelineConfig
from lerobot.utils.constants import TRAINING_STATE_DIR, TRAINING_STEP
from lerobot.utils.io_utils import load_json, write_json
from lerobot.utils.random_utils import serialize_rng_state


def _config(checkpoint: Path) -> TrainPipelineConfig:
    config = TrainPipelineConfig(dataset=DatasetConfig(repo_id="fixture/replay-validation"), steps=8)
    config.parallelism.resolve(1)
    config.checkpoint_path = checkpoint
    return config


def _boundary(accelerator: Accelerator) -> TrainingWindowCheckpoint:
    return TrainingWindowCheckpoint(
        0,
        0,
        accelerator.num_processes,
        accelerator.process_index,
        accelerator.gradient_accumulation_steps,
        None,
        serialize_rng_state(),
        {},
    )


def test_window_writer_records_actual_divisor_not_configuration(tmp_path: Path) -> None:
    accelerator = Accelerator(cpu=True)
    config = _config(tmp_path)
    config.accelerator.gradient_accumulation.steps = 3
    save_training_state(tmp_path, 0, config, accelerator=accelerator, training_window=_boundary(accelerator))
    metadata = load_training_metadata(tmp_path / TRAINING_STATE_DIR)
    assert metadata["grad_accum_steps"] == accelerator.gradient_accumulation_steps == 1
    assert metadata["training_window_version"] == 1
    restored = load_training_window(config, accelerator, 0)
    assert restored is not None and restored.window is None


@pytest.mark.parametrize("version", [None, True, "1", 0, 2])
def test_invalid_window_metadata_marker_is_not_historical_absence(tmp_path: Path, version: Any) -> None:
    accelerator = Accelerator(cpu=True)
    config = _config(tmp_path)
    save_training_state(tmp_path, 0, config, accelerator=accelerator, training_window=_boundary(accelerator))
    path = tmp_path / TRAINING_STATE_DIR / TRAINING_STEP
    metadata = load_json(path)
    metadata["training_window_version"] = version
    write_json(metadata, path)
    with pytest.raises(ValueError, match="Training window resume"):
        load_training_window(config, accelerator, 0)


def test_invalid_private_rng_is_refused_without_changing_live_rng() -> None:
    accelerator = Accelerator(cpu=True)
    state = deepcopy(_boundary(accelerator).state_dict())
    state["continuation_rng"]["torch_rng_state"] = torch.zeros(1, dtype=torch.uint8)
    before = serialize_rng_state()
    with pytest.raises((ValueError, RuntimeError)):
        TrainingWindowCheckpoint.from_state_dict(state)
    after = serialize_rng_state()
    assert before.keys() == after.keys()
    for name in before:
        assert torch.equal(before[name], after[name]), name


def test_invalid_private_augmentation_generator_is_refused() -> None:
    accelerator = Accelerator(cpu=True)
    state = _boundary(accelerator).state_dict()
    state["augmentation"] = {"generator_state": torch.tensor([1.0])}
    with pytest.raises(ValueError, match="augmentation"):
        TrainingWindowCheckpoint.from_state_dict(state)
