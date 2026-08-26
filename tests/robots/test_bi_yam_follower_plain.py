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

from lerobot.robots.bi_yam_follower_plain import BiYamFollowerPlain, BiYamFollowerPlainConfig
from lerobot.robots.config import RobotConfig


def _robot(*, record_torques: bool = False) -> BiYamFollowerPlain:
    robot = BiYamFollowerPlain(BiYamFollowerPlainConfig(record_torques=record_torques))
    for arm in (robot.left_arm, robot.right_arm):
        arm.get_joint_pos = MagicMock(return_value=np.zeros(7, dtype=np.float64))
        arm.command_joint_pos = MagicMock()
    return robot


def _action(value: float) -> dict[str, float]:
    return dict.fromkeys(_robot().action_features, value)


def test_official_robot_type_resolves_to_plain_follower():
    config = BiYamFollowerPlainConfig()

    assert config.type == "bi_yam_follower"
    assert RobotConfig.get_choice_class("bi_yam_follower") is BiYamFollowerPlainConfig
    assert RobotConfig.get_choice_class("bi_yam_follower_plain") is BiYamFollowerPlainConfig
    assert _robot().name == "bi_yam_follower"


def test_full_action_is_sent_once_without_interpolation():
    robot = _robot()
    sent = robot.send_action(_action(2.0))

    robot.left_arm.get_joint_pos.assert_not_called()
    robot.right_arm.get_joint_pos.assert_not_called()
    robot.left_arm.command_joint_pos.assert_called_once()
    robot.right_arm.command_joint_pos.assert_called_once()
    np.testing.assert_allclose(robot.left_arm.command_joint_pos.call_args.args[0], np.full(7, 2.0))
    np.testing.assert_allclose(robot.right_arm.command_joint_pos.call_args.args[0], np.full(7, 2.0))
    assert sent == _action(2.0)


def test_partial_action_preserves_plain_direct_send_behavior():
    robot = _robot()
    action = {"left_joint_0.pos": 0.5}

    robot.send_action(action)

    robot.left_arm.get_joint_pos.assert_not_called()
    robot.right_arm.get_joint_pos.assert_not_called()
    np.testing.assert_allclose(robot.left_arm.command_joint_pos.call_args.args[0], np.asarray([0.5]))
    robot.right_arm.command_joint_pos.assert_not_called()


def test_observation_uses_one_rpc_per_arm_and_preserves_position_torque_keys():
    robot = _robot(record_torques=True)
    for arm, offset in ((robot.left_arm, 0.0), (robot.right_arm, 10.0)):
        arm.get_observations = MagicMock(
            return_value={
                "joint_pos": np.arange(6, dtype=np.float64) + offset,
                "gripper_pos": np.asarray([0.25]),
                "joint_eff": np.arange(6, dtype=np.float64) + offset + 20.0,
                "gripper_eff": np.asarray([0.5]),
            }
        )

    observation = robot.get_observation()

    robot.left_arm.get_observations.assert_called_once_with()
    robot.right_arm.get_observations.assert_called_once_with()
    expected_positions = list(robot.observation_features)
    expected_efforts = [
        *(f"left_joint_{index}.eff" for index in range(6)),
        "left_gripper.eff",
        *(f"right_joint_{index}.eff" for index in range(6)),
        "right_gripper.eff",
    ]
    assert list(observation) == [
        *expected_positions[:7],
        *expected_efforts[:7],
        *expected_positions[7:],
        *expected_efforts[7:],
    ]


def test_feature_schema_is_canonical_before_and_after_connect():
    robot = _robot()
    expected = [
        *(f"left_joint_{index}.pos" for index in range(6)),
        "left_gripper.pos",
        *(f"right_joint_{index}.pos" for index in range(6)),
        "right_gripper.pos",
    ]
    features_before_connect = robot.action_features
    for arm in (robot.left_arm, robot.right_arm):
        arm.connect = MagicMock()
        arm.num_dofs = MagicMock(return_value=7)

    robot.connect()

    assert list(features_before_connect) == expected
    assert robot.action_features is features_before_connect
    assert list(robot.observation_features) == expected


@pytest.mark.parametrize(("left_dofs", "right_dofs"), [(6, 7), (7, 6)])
def test_connect_rejects_noncanonical_arm_dofs(left_dofs: int, right_dofs: int):
    robot = _robot()
    for arm, dofs in ((robot.left_arm, left_dofs), (robot.right_arm, right_dofs)):
        arm.connect = MagicMock()
        arm.disconnect = MagicMock()
        arm.num_dofs = MagicMock(return_value=dofs)

    with pytest.raises(RuntimeError, match="requires exactly 7 DOFs per arm"):
        robot.connect()

    robot.left_arm.disconnect.assert_called_once_with()
    robot.right_arm.disconnect.assert_called_once_with()
