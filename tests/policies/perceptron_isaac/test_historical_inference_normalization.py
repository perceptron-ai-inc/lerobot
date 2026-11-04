"""Historical serving overrides must not relabel source stats or alter training clipping."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from lerobot.lerobot_types import TransitionKey
from lerobot.policies.perceptron_isaac.configuration_perceptron_isaac import PerceptronIsaacConfig
from lerobot.policies.perceptron_isaac.isaac_stats import (
    isaac_stats_from_dict,
    isaac_stats_to_dict,
    normalize_isaac_proprio,
)
from lerobot.policies.perceptron_isaac.processor_perceptron_isaac import (
    PERCEPTRON_ISAAC_STREAM_KEY,
    PerceptronIsaacActionUnnormalizeProcessorStep,
    PerceptronIsaacMharmonyPackProcessorStep,
)
from lerobot.policies.perceptron_isaac.tensor_stream import VectorType
from lerobot.processor.converters import batch_to_transition
from lerobot.utils.constants import ACTION, OBS_STATE
from tests.policies.perceptron_isaac._p2_data_fixtures import (
    StatsPayload,
    data_batch,
    data_config,
    data_pack,
    data_stats_payload,
)
from tests.policies.perceptron_isaac.test_p2_proprio_profiles import _profile_payload


def _profiled_payload() -> StatsPayload:
    payload = data_stats_payload(passthrough=True)
    payload["_profile"] = _profile_payload(payload, 1.0)
    return payload


@pytest.mark.parametrize("clip", [True, False])
def test_quantile_normalization_can_bypass_only_the_clamp(clip: bool) -> None:
    payload = _profiled_payload()
    stats = isaac_stats_from_dict(payload)
    raw = np.array([1.4, 250.0], dtype=np.float32)
    before = raw.copy()
    out = normalize_isaac_proprio(raw, stats.proprio, profile=stats.profile, clip_normalized=clip)
    expected = 1.0 if clip else 1.4 / (1.0 + 1e-6) * 2 - 1
    np.testing.assert_allclose(out, [expected, 250.0], rtol=1e-5)
    np.testing.assert_array_equal(raw, before)
    assert isaac_stats_to_dict(stats)["_profile"] == payload["_profile"]
    assert payload["_profile"]["clip_normalized_proprio"] is True


@pytest.mark.parametrize("training", [False, True], ids=["inference", "training-control"])
@pytest.mark.parametrize("clip", [False, True])
def test_pack_and_portable_state_keep_override_inference_only(
    tmp_path: Path, training: bool, clip: bool
) -> None:
    payload = _profiled_payload()
    config = replace(data_config(tmp_path, payload), inference_clip_normalized_proprio=clip)
    pack, fast = data_pack(config)
    batch = data_batch()
    batch[OBS_STATE] = torch.tensor([[[1.4, 250.0]]])
    batch[ACTION] = torch.tensor([[[0.5, 0.25], [0.75, 0.75]]])
    if not training:
        del batch[ACTION]
    transition = batch_to_transition(batch)
    out = pack(transition)
    serialized = pack.get_config()
    assert serialized.get("inference_clip_normalized_proprio", True) is clip
    restored = PerceptronIsaacMharmonyPackProcessorStep(**serialized)
    restored.load_state_dict(pack.state_dict())
    restored._fast_processor = fast
    for result in (out, restored(transition)):
        complementary = result[TransitionKey.COMPLEMENTARY_DATA]
        assert isinstance(complementary, dict)
        stream = complementary[PERCEPTRON_ISAAC_STREAM_KEY]
        vector = next(event for event in stream.streams[0].events if event.type == VectorType.vector)
        expected = 1.0 if training or clip else 1.4 / (1.0 + 1e-6) * 2 - 1
        np.testing.assert_allclose(vector.data.float().numpy().reshape(-1), [expected, 250.0], rtol=1e-5)
    assert pack._stats is not None and pack._stats.profile is not None
    assert isaac_stats_to_dict(pack._stats)["_profile"] == payload["_profile"]


def test_legacy_pack_config_still_defaults_to_clipping(tmp_path: Path) -> None:
    pack, _ = data_pack(data_config(tmp_path, _profiled_payload()))
    pack._ensure_renderer()
    serialized = pack.get_config()
    serialized.pop("inference_clip_normalized_proprio", None)
    restored = PerceptronIsaacMharmonyPackProcessorStep(**serialized)
    restored.load_state_dict(pack.state_dict())
    batch = data_batch()
    batch[OBS_STATE] = torch.tensor([[[1.4, 250.0]]])
    del batch[ACTION]
    complementary = restored(batch_to_transition(batch))[TransitionKey.COMPLEMENTARY_DATA]
    assert isinstance(complementary, dict)
    stream = complementary[PERCEPTRON_ISAAC_STREAM_KEY]
    vector = next(event for event in stream.streams[0].events if event.type == VectorType.vector)
    np.testing.assert_allclose(vector.data.float().numpy().reshape(-1), [1.0, 250.0])


@pytest.mark.parametrize("bad", [0, 1, "false", None])
def test_inference_override_rejects_non_boolean_values(bad: Any) -> None:
    with pytest.raises(ValueError, match="inference_clip_normalized_proprio"):
        PerceptronIsaacConfig(inference_clip_normalized_proprio=bad)
    with pytest.raises(ValueError, match="inference_clip_normalized_proprio"):
        PerceptronIsaacMharmonyPackProcessorStep(inference_clip_normalized_proprio=bad)


@pytest.mark.parametrize("cap", [1.0, 10.0])
def test_action_postprocessor_uses_its_own_saved_cap(tmp_path: Path, cap: float) -> None:
    config = data_config(tmp_path, data_stats_payload())
    step = PerceptronIsaacActionUnnormalizeProcessorStep(
        stats_path=config.native_stats_path, clip_normalized_max=cap, clip_action_pose=False
    )
    normalized = torch.tensor([[2.0, -2.0]])
    out = step(batch_to_transition({ACTION: normalized}))[TransitionKey.ACTION]
    clipped = np.clip(normalized.numpy(), -cap, cap)
    expected = (clipped + 1) * 0.5 * np.array([1.0 + 1e-6, 100.0 + 1e-6], dtype=np.float32)
    assert isinstance(out, torch.Tensor)
    np.testing.assert_allclose(out.numpy(), expected, rtol=1e-5, atol=1e-5)
