"""Focused P2 data regressions; original RED preserved in data-red-frozen.patch."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from lerobot.lerobot_types import TransitionKey
from lerobot.policies.perceptron_isaac.isaac_stats import (
    isaac_stats_from_dict,
    isaac_stats_to_state_dict,
    normalize_isaac_actions,
    unnormalize_isaac_actions,
)
from lerobot.policies.perceptron_isaac.processor_perceptron_isaac import (
    PERCEPTRON_ISAAC_KEPT_SAMPLE_INDICES_KEY,
    PERCEPTRON_ISAAC_RENDER_META_KEY,
    PERCEPTRON_ISAAC_STREAM_KEY,
    PerceptronIsaacActionUnnormalizeProcessorStep,
)
from lerobot.policies.perceptron_isaac.tensor_stream import TextType
from lerobot.processor.converters import batch_to_transition
from lerobot.utils.constants import ACTION
from tests.policies.perceptron_isaac._p2_data_fixtures import (
    action_event,
    data_batch,
    data_config,
    data_pack,
    data_stats_payload,
)


@pytest.mark.parametrize("clip", [True, False], ids=["clipped", "unclipped"])
@pytest.mark.parametrize("raw,upper", [(25.0, 100.0), (0.8, 0.5), (1.3, 1.0), (-0.2, 1.0)])
def test_raw_gripper_flow_and_all_channel_fast_use_different_units(
    tmp_path: Path, clip: bool, raw: float, upper: float
) -> None:
    config = data_config(tmp_path, data_stats_payload(gripper_upper=upper, passthrough=True))
    config.train_clip_normalized_actions = clip
    pack, fast = data_pack(config)
    batch = data_batch()
    batch[ACTION] = torch.tensor([[[3.0, raw], [0.5, raw]]])
    out = pack(batch_to_transition(batch))
    complementary = out[TransitionKey.COMPLEMENTARY_DATA]
    stream = complementary[PERCEPTRON_ISAAC_STREAM_KEY]
    assert complementary[PERCEPTRON_ISAAC_KEPT_SAMPLE_INDICES_KEY] == [0]
    raw_action = batch[ACTION].numpy()[0]
    all_quantiles = raw_action / np.array([1.0 + 1e-6, upper + 1e-6]) * 2.0 - 1.0
    expected_flow = all_quantiles.copy()
    expected_flow[:, 1] = np.clip(raw, 0.0, upper) if clip else raw
    if clip:
        expected_flow[:, 0] = np.clip(expected_flow[:, 0], -10.0, 10.0)
    np.testing.assert_allclose(action_event(stream).tags["action_target"], expected_flow, atol=1e-6)
    assert len(fast.inputs) == 1
    expected_fast = np.clip(all_quantiles, -1.0, 1.0) if clip else all_quantiles
    np.testing.assert_allclose(fast.inputs[0][0], expected_fast, atol=1e-6)
    assert action_event(stream).tags["action_is_pad"] == [False, False]
    assert action_event(stream, TextType.action).tags["action_is_pad"] == [False, False]
    assert complementary[PERCEPTRON_ISAAC_RENDER_META_KEY]["anchor_timestamp_seconds"] == [0.15]


def test_raw_gripper_quantile_outlier_is_rejected_even_when_physical_value_is_small(tmp_path: Path) -> None:
    config = data_config(tmp_path, data_stats_payload(gripper_upper=0.01, passthrough=True))
    pack, fast = data_pack(config)
    batch = data_batch()
    batch[ACTION] = torch.tensor([[[0.5, 0.8], [0.5, 0.8]]])
    transition = pack(batch_to_transition(batch))
    complementary = transition[TransitionKey.COMPLEMENTARY_DATA]
    assert complementary[PERCEPTRON_ISAAC_STREAM_KEY].collective_only
    assert complementary[PERCEPTRON_ISAAC_KEPT_SAMPLE_INDICES_KEY] == []
    assert not fast.inputs, "Outlier filtering precedes FAST encoding."

    # Local emptiness is now deferred to the window, but quantile filtering still applies.
    batch[ACTION] = torch.tensor([[[0.5, 0.005], [0.5, 0.005]]])
    retained = pack(batch_to_transition(batch))[TransitionKey.COMPLEMENTARY_DATA]
    assert not retained[PERCEPTRON_ISAAC_STREAM_KEY].collective_only
    assert retained[PERCEPTRON_ISAAC_KEPT_SAMPLE_INDICES_KEY] == [0]
    assert fast.inputs


def test_default_normalized_gripper_remains_quantile_flow_and_fast(tmp_path: Path) -> None:
    payload = data_stats_payload()
    pack, fast = data_pack(data_config(tmp_path, payload))
    batch = data_batch()
    out = pack(batch_to_transition(batch))
    stream = out[TransitionKey.COMPLEMENTARY_DATA][PERCEPTRON_ISAAC_STREAM_KEY]
    expected = normalize_isaac_actions(batch[ACTION].numpy()[0], isaac_stats_from_dict(payload).action)
    np.testing.assert_allclose(action_event(stream).tags["action_target"], expected)
    np.testing.assert_allclose(fast.inputs[0][0], expected)


@pytest.mark.parametrize("clip", [True, False], ids=["robot-bounds", "unclipped-control"])
@pytest.mark.parametrize("collapsed", [False, True], ids=["quantile-bounds", "minmax-fallback"])
def test_raw_output_clipping_uses_effective_robot_bounds_not_normalized_cap(
    clip: bool, collapsed: bool
) -> None:
    payload = data_stats_payload(passthrough=True)
    payload["action"]["q01"] = [0.0, 10.0]
    if collapsed:
        payload["action"].update(
            {"q01": [0.0, 0.0], "q99": [1.0, 0.0], "min": [0.0, 10.0], "max": [1.0, 100.0]}
        )
    stats = isaac_stats_from_dict(payload)
    predictions = np.array([[0.0, -5.0], [0.0, 25.0], [0.0, 120.0]], dtype=np.float32)
    original = predictions.copy()
    robot = unnormalize_isaac_actions(predictions, stats.action, clip=clip, clip_normalized_max=10.0)
    np.testing.assert_array_equal(robot[:, 1], [10.0, 25.0, 100.0] if clip else original[:, 1])
    np.testing.assert_array_equal(predictions, original)
    post = PerceptronIsaacActionUnnormalizeProcessorStep(clip_action_pose=False, clip_normalized_max=10.0)
    post.load_state_dict(isaac_stats_to_state_dict(stats))
    if clip:
        out = post(batch_to_transition({ACTION: torch.from_numpy(predictions.copy())}))
        np.testing.assert_array_equal(out[TransitionKey.ACTION].numpy()[:, 1], [10.0, 25.0, 100.0])
