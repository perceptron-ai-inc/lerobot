"""Supplementary local resume guards; parent owns all runtime verification."""

from pathlib import Path

import pytest

from lerobot.common.train_utils import load_training_metadata, resume_before_prepare, save_training_state
from lerobot.configs.default import DatasetConfig
from lerobot.configs.train import TrainPipelineConfig
from lerobot.utils.constants import RNG_STATE, TRAINING_STATE_DIR, TRAINING_STEP
from lerobot.utils.io_utils import write_json


@pytest.mark.parametrize("cadence", [None, True, "2", 0, -1, 2.5])
def test_present_invalid_saved_cadence_is_not_historical_absence(tmp_path: Path, cadence: object) -> None:
    write_json({"step": 3, "grad_accum_steps": cadence}, tmp_path / TRAINING_STEP)
    with pytest.raises(ValueError, match="grad_accum_steps"):
        load_training_metadata(tmp_path)


def test_phase_one_local_rng_read_error_keeps_original_exception(tmp_path: Path) -> None:
    config = TrainPipelineConfig(dataset=DatasetConfig(repo_id="fixture/resume-cadence"))
    config.parallelism.resolve(1)
    config.checkpoint_path = tmp_path
    save_training_state(tmp_path, 3, config)
    (tmp_path / TRAINING_STATE_DIR / RNG_STATE).unlink()
    with pytest.raises(FileNotFoundError):
        resume_before_prepare(config)
