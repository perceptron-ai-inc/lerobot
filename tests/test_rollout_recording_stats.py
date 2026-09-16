"""Recording destinations must not replace checkpoint-owned inference normalization."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from lerobot.policies.perceptron_isaac.processor_perceptron_isaac import (
    make_perceptron_isaac_pre_post_processors,
    make_perceptron_isaac_pre_post_processors_from_pretrained,
)
from tests.policies.perceptron_isaac.test_checkpoint_lifecycle import _config, _write_stats
from tests.test_rollout import _rollout_context_harness


@pytest.mark.parametrize("recording", ["none", "empty", "resumed"])
def test_rollout_recording_preserves_saved_isaac_normalization(monkeypatch, tmp_path, recording):
    from lerobot.rollout import EpisodicStrategyConfig
    from tests.mocks.mock_robot import MockRobot, MockRobotConfig

    stats_path = tmp_path / "isaac_stats.json"
    _write_stats(stats_path)
    config = _config(native_stats_path=str(stats_path), apply_offset_norm=False)
    pre, post = make_perceptron_isaac_pre_post_processors(config)
    pre.save_pretrained(tmp_path)
    post.save_pretrained(tmp_path)
    expected = post(torch.zeros(1, 2))
    robot = MockRobot(MockRobotConfig(n_motors=2))
    context_module, cfg = _rollout_context_harness(monkeypatch, robot)
    monkeypatch.setattr(context_module, "validate_visual_features_consistency", lambda *args: None)
    monkeypatch.setattr(context_module, "create_inference_engine", lambda *args, **kwargs: MagicMock())

    def load_real_processors(**kwargs):
        return make_perceptron_isaac_pre_post_processors_from_pretrained(
            config, tmp_path, dataset_stats=kwargs.get("dataset_stats")
        )

    monkeypatch.setattr(context_module, "make_pre_post_processors", load_real_processors)
    if recording != "none":
        cfg.strategy = EpisodicStrategyConfig()
        cfg.resume = recording == "resumed"
        cfg.dataset = SimpleNamespace(
            repo_id="synthetic/rollout_contract",
            root=tmp_path / "recording",
            fps=20,
            video=False,
            video_encoding_batch_size=1,
            rgb_encoder=None,
            depth_encoder=None,
            streaming_encoding=False,
            encoder_queue_maxsize=30,
            encoder_threads=None,
            num_image_writer_processes=0,
            num_image_writer_threads_per_camera=0,
            single_task="synthetic",
            stamp_repo_id=lambda: None,
        )
        stats = (
            None
            if recording == "empty"
            else {
                key: {"q01": np.array([20.0, 30.0]), "q99": np.array([40.0, 50.0])}
                for key in ("action", "observation.state")
            }
        )
        destination = SimpleNamespace(
            meta=SimpleNamespace(stats=stats), repo_id="synthetic/rollout_contract", num_episodes=0
        )
        monkeypatch.setattr(context_module.LeRobotDataset, "create", lambda *args, **kwargs: destination)
        monkeypatch.setattr(context_module.LeRobotDataset, "resume", lambda *args, **kwargs: destination)
    try:
        context = context_module.build_rollout_context(cfg, MagicMock())
        torch.testing.assert_close(context.policy.postprocessor(torch.zeros(1, 2)), expected, rtol=0, atol=0)
    finally:
        if robot.is_connected:
            robot.disconnect()
