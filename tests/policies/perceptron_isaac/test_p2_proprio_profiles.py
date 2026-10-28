"""Focused P2 data regressions; original RED preserved in data-red-frozen.patch."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from lerobot.lerobot_types import TransitionKey
from lerobot.policies.perceptron_isaac.isaac_stats import (
    isaac_stats_from_dict,
    isaac_stats_from_state_dict,
    isaac_stats_to_dict,
    isaac_stats_to_state_dict,
)
from lerobot.policies.perceptron_isaac.processor_perceptron_isaac import (
    PERCEPTRON_ISAAC_STREAM_KEY,
)
from lerobot.policies.perceptron_isaac.tensor_stream import VectorType
from lerobot.processor.converters import batch_to_transition
from lerobot.utils.constants import ACTION, OBS_STATE
from tests.policies.perceptron_isaac._p2_data_fixtures import (
    ProfilePayload,
    StatsPayload,
    action_event,
    data_batch,
    data_config,
    data_pack,
    data_stats_payload,
)


def _profile_payload(payload: StatsPayload, cap: float) -> ProfilePayload:
    """Synthetic post-authentication metadata with all six Genesis profile fields.

    This exercises retention/consumption, not whole-export SHA authentication.
    """
    action_bytes = json.dumps(payload["action"], sort_keys=True).encode()
    proprio_bytes = json.dumps(payload["proprio"], sort_keys=True).encode()
    return {
        "profile_id": "test/p2-data-profile",
        "profile_sha256": hashlib.sha256(b"test/p2-data-profile").hexdigest(),
        "action_stats_sha256": hashlib.sha256(action_bytes).hexdigest(),
        "proprio_stats_sha256": hashlib.sha256(proprio_bytes).hexdigest(),
        "clip_normalized_proprio": True,
        "proprio_clip_normalized_max": cap,
    }


@pytest.mark.parametrize("cap", [1.0, 0.75])
@pytest.mark.parametrize("raw_joint,raw_gripper", [(1.4, 250.0), (-0.4, -25.0)])
@pytest.mark.parametrize("profiled", [True, False], ids=["profile", "unprofiled-control"])
def test_profile_clip_reaches_rendered_proprio_without_clipping_raw_units(
    tmp_path: Path, cap: float, raw_joint: float, raw_gripper: float, profiled: bool
) -> None:
    payload = data_stats_payload(passthrough=True)
    if profiled:
        payload["_profile"] = _profile_payload(payload, cap)
    pack, _ = data_pack(data_config(tmp_path, payload))
    batch = data_batch()
    batch[OBS_STATE] = torch.tensor([[[raw_joint, raw_gripper]]])
    # Keep action passthrough values below the baseline outlier threshold so a
    # P2.06 failure cannot masquerade as the proprio-profile regression.
    batch[ACTION] = torch.tensor([[[0.5, 0.25], [0.75, 0.75]]])
    out = pack(batch_to_transition(batch))
    stream = out[TransitionKey.COMPLEMENTARY_DATA][PERCEPTRON_ISAAC_STREAM_KEY]
    vectors = [event for event in stream.streams[0].events if event.type == VectorType.vector]
    assert len(vectors) == 1
    quantile_joint = raw_joint / (1.0 + 1e-6) * 2.0 - 1.0
    expected_joint = np.clip(quantile_joint, -cap, cap) if profiled else quantile_joint
    np.testing.assert_allclose(vectors[0].data.float().numpy().reshape(-1), [expected_joint, raw_gripper])
    assert action_event(stream).tags["action_is_pad"] == [False, False]


def test_profile_identity_and_clip_survive_json_and_processor_tensor_state(tmp_path: Path) -> None:
    payload = data_stats_payload(passthrough=True)
    payload["_profile"] = _profile_payload(payload, 0.75)
    stats = isaac_stats_from_dict(payload)
    # Retain all identity fields, not just a boolean or the human-readable profile label.
    assert isaac_stats_to_dict(stats)["_profile"] == payload["_profile"]
    restored = isaac_stats_from_state_dict(isaac_stats_to_state_dict(stats))
    assert isaac_stats_to_dict(restored)["_profile"] == payload["_profile"]
    pack, _ = data_pack(data_config(tmp_path, payload))
    pack.load_state_dict(isaac_stats_to_state_dict(restored))
    batch = data_batch()
    batch[OBS_STATE] = torch.tensor([[[1.4, 250.0]]])
    batch[ACTION] = torch.tensor([[[0.5, 0.25], [0.75, 0.75]]])
    out = pack(batch_to_transition(batch))
    stream = out[TransitionKey.COMPLEMENTARY_DATA][PERCEPTRON_ISAAC_STREAM_KEY]
    vector = next(event for event in stream.streams[0].events if event.type == VectorType.vector)
    np.testing.assert_allclose(vector.data.float().numpy().reshape(-1), [0.75, 250.0])


@pytest.mark.parametrize("cap", [0.0, -1.0, float("nan"), float("inf")])
def test_invalid_profile_clip_is_not_silently_dropped(cap: float) -> None:
    payload = data_stats_payload()
    payload["_profile"] = _profile_payload(payload, cap)
    with pytest.raises(ValueError, match="(?i)profile|clip"):
        isaac_stats_from_dict(payload)


def test_malformed_profile_identity_is_not_silently_dropped() -> None:
    payload = data_stats_payload()
    payload["_profile"] = _profile_payload(payload, 1.0)
    payload["_profile"]["profile_sha256"] = "not-a-sha256"
    with pytest.raises(ValueError, match="(?i)profile|sha256"):
        isaac_stats_from_dict(payload)
