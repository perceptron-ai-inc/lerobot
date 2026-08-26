#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from unittest.mock import MagicMock

import numpy as np
import pytest

from lerobot.teleoperators.bi_yam_leader_plain import (
    BiYamLeaderPlain,
    BiYamLeaderPlainConfig,
    YamLeaderClient,
)


def _leader() -> BiYamLeaderPlain:
    return BiYamLeaderPlain(BiYamLeaderPlainConfig())


def _observation(offset: float, *, physical_gripper: bool) -> dict[str, np.ndarray]:
    observation = {
        "joint_pos": np.arange(6, dtype=np.float64) + offset,
        "io_inputs": np.asarray([1.0]),
    }
    if physical_gripper:
        observation["gripper_pos"] = np.asarray([0.25])
    return observation


def test_feature_schema_is_canonical_before_and_after_connect():
    leader = _leader()
    expected = [
        *(f"left_joint_{index}.pos" for index in range(6)),
        "left_gripper.pos",
        *(f"right_joint_{index}.pos" for index in range(6)),
        "right_gripper.pos",
    ]
    features_before_connect = leader.action_features
    for arm in (leader.left_arm, leader.right_arm):
        arm.connect = MagicMock()
        arm.num_dofs = MagicMock(return_value=7)

    leader.connect()

    assert list(features_before_connect) == expected
    assert leader.action_features is features_before_connect


@pytest.mark.parametrize(("left_dofs", "right_dofs"), [(6, 7), (7, 6)])
def test_connect_rejects_noncanonical_arm_dofs(left_dofs: int, right_dofs: int):
    leader = _leader()
    for arm, dofs in ((leader.left_arm, left_dofs), (leader.right_arm, right_dofs)):
        arm.connect = MagicMock()
        arm.disconnect = MagicMock()
        arm.num_dofs = MagicMock(return_value=dofs)

    with pytest.raises(RuntimeError, match="requires exactly 7 DOFs per arm"):
        leader.connect()

    leader.left_arm.disconnect.assert_called_once_with()
    leader.right_arm.disconnect.assert_called_once_with()


def test_get_action_uses_one_observation_rpc_per_arm():
    leader = _leader()
    left_observation = _observation(0.0, physical_gripper=False)
    right_observation = _observation(10.0, physical_gripper=True)
    leader.left_arm.get_observations = MagicMock(return_value=left_observation)
    leader.right_arm.get_observations = MagicMock(return_value=right_observation)
    leader.left_arm.get_gripper_from_encoder = MagicMock(return_value=0.0)
    leader.right_arm.get_gripper_from_encoder = MagicMock(return_value=1.0)

    action = leader.get_action()

    leader.left_arm.get_observations.assert_called_once_with()
    leader.right_arm.get_observations.assert_called_once_with()
    leader.left_arm.get_gripper_from_encoder.assert_called_once_with(left_observation)
    leader.right_arm.get_gripper_from_encoder.assert_not_called()
    assert list(action) == list(leader.action_features)
    assert action["left_gripper.pos"] == 0.0
    assert action["right_gripper.pos"] == 0.25


def test_encoder_parsing_reuses_supplied_observation_without_rpc():
    client = YamLeaderClient(port=5001)
    client._client = MagicMock()
    observation = _observation(0.0, physical_gripper=False)

    assert client.get_gripper_from_encoder(observation) == 0.0
    client._client.get_observations.assert_not_called()
