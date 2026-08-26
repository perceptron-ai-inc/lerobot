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

from __future__ import annotations

import time
from unittest.mock import MagicMock

import numpy as np
import pytest

from lerobot.robots.bi_yam_follower import BiYamFollower, BiYamFollowerConfig, YamArmClient
from lerobot.utils.yam_portal import FOLLOWER_PORTAL_SCHEMA


def _caps(value: float = 0.2, side: str = "left") -> dict[str, float]:
    # Dict-form caps use the robot's own key namespace ({side}_joint_0..5.pos +
    # {side}_gripper.pos), matching send_action and observation_features.
    return dict.fromkeys(BiYamFollower._arm_keys(side), value)


def _config(*, torque_enabled: bool = True) -> BiYamFollowerConfig:
    return BiYamFollowerConfig(
        require_portal_readiness=False,
        torque_enabled=torque_enabled,
        left_arm_max_relative_target=_caps(side="left") if torque_enabled else None,
        right_arm_max_relative_target=_caps(side="right") if torque_enabled else None,
    )


def _observation(offset: float = 0.0) -> dict[str, np.ndarray]:
    return {
        "joint_pos": np.arange(6, dtype=np.float64) + offset,
        "gripper_pos": np.asarray([0.5 + offset]),
        "joint_eff": np.zeros(7),
    }


def _health() -> dict:
    return {
        "schema": FOLLOWER_PORTAL_SCHEMA,
        "dofs": 7,
        "position_bounds": np.asarray([[-10.0, 10.0]] * 7),
        "torque_enabled": True,
        "command_watchdog_seconds": 4.0,
    }


def _robot(*, torque_enabled: bool = True) -> BiYamFollower:
    robot = BiYamFollower(_config(torque_enabled=torque_enabled))
    for arm, offset in ((robot.left_arm, 0.0), (robot.right_arm, 1.0)):
        arm._client = object()
        arm._health = _health()
        arm.get_observations = MagicMock(return_value=_observation(offset))
        arm.command_joint_pos = MagicMock()
        arm.zero_torque = MagicMock()
        arm.disconnect = MagicMock()
    robot._left_dofs = robot._right_dofs = 7
    return robot


def _action(value: float = 1.0) -> dict[str, float]:
    return {key: value for side in ("left", "right") for key in BiYamFollower._arm_keys(side)}


def test_guarded_config_requires_readiness_and_motion_caps():
    with pytest.raises(ValueError, match="portal_readiness_file"):
        BiYamFollowerConfig()
    with pytest.raises(ValueError, match="requires explicit left and right"):
        BiYamFollowerConfig(require_portal_readiness=False, torque_enabled=True)


def test_canonical_interface_is_exact_left_then_right_fourteen_dofs():
    robot = _robot()
    expected = [
        *(f"left_joint_{index}.pos" for index in range(6)),
        "left_gripper.pos",
        *(f"right_joint_{index}.pos" for index in range(6)),
        "right_gripper.pos",
    ]

    assert list(robot.action_features) == expected
    assert list(robot.observation_features) == expected
    assert list(robot.get_observation()) == expected


@pytest.mark.parametrize(
    "mutation",
    [
        lambda action: action.pop("left_joint_3.pos"),
        lambda action: action.update({"right_gripper.pos": float("nan")}),
        lambda action: action.update({"left_joint_0.pos": [0.0]}),
    ],
)
def test_invalid_action_sends_neither_arm(mutation):
    robot = _robot()
    action = _action()
    mutation(action)

    with pytest.raises(ValueError):
        robot.send_action(action)

    robot.left_arm.command_joint_pos.assert_not_called()
    robot.right_arm.command_joint_pos.assert_not_called()


def test_relative_caps_apply_to_both_arms_before_send():
    robot = _robot()

    sent = robot.send_action(_action(9.0))

    expected_left = np.concatenate((np.arange(6), [0.5])) + 0.2
    expected_right = np.concatenate((np.arange(6) + 1.0, [1.5])) + 0.2
    np.testing.assert_allclose(robot.left_arm.command_joint_pos.call_args.args[0], expected_left)
    np.testing.assert_allclose(robot.right_arm.command_joint_pos.call_args.args[0], expected_right)
    assert list(sent) == list(robot.action_features)


def test_physical_bound_violation_sends_neither_arm():
    robot = _robot()
    robot.right_arm._health["position_bounds"][2] = [-0.5, 0.5]

    with pytest.raises(ValueError, match="physical position bounds"):
        robot.send_action(_action(1.0))

    robot.left_arm.command_joint_pos.assert_not_called()
    robot.right_arm.command_joint_pos.assert_not_called()


def test_prediction_only_session_refuses_motion():
    robot = _robot(torque_enabled=False)

    with pytest.raises(RuntimeError, match="prediction-only"):
        robot.send_action(_action())

    robot.left_arm.command_joint_pos.assert_not_called()
    robot.right_arm.command_joint_pos.assert_not_called()


def test_second_arm_command_failure_zeros_both_portals():
    robot = _robot()
    robot.right_arm.command_joint_pos.side_effect = TimeoutError("ack timeout")

    with pytest.raises(TimeoutError, match="ack timeout"):
        robot.send_action(_action())

    robot.left_arm.zero_torque.assert_called_once()
    robot.right_arm.zero_torque.assert_called_once()


class _Future:
    def __init__(self, value=None, error=None):
        self.value = value
        self.error = error
        self.timeouts = []

    def result(self, timeout):
        self.timeouts.append(timeout)
        if self.error:
            raise self.error
        return self.value


class _FakePortal:
    def __init__(self, envelopes):
        self.envelopes = iter(envelopes)

    def get_observations_guarded(self):
        return _Future(next(self.envelopes))


def _envelope(sequence: int, *, age: float = 0.0):
    return {
        "schema": FOLLOWER_PORTAL_SCHEMA,
        "sequence": sequence,
        "server_time_unix_ns": time.time_ns() - int(age * 1e9),
        "observation": _observation(),
    }


def test_arm_client_rejects_stale_sequence_and_state_age():
    client = YamArmClient(1234, "localhost", 0.4, 0.25)
    client._client = _FakePortal([_envelope(1), _envelope(1)])
    client._health = _health()
    client.get_observations()
    with pytest.raises(RuntimeError, match="sequence is stale"):
        client.get_observations()

    client._client = _FakePortal([_envelope(2, age=0.5)])
    with pytest.raises(RuntimeError, match="observation is stale"):
        client.get_observations()


def test_arm_client_wraps_rpc_timeout_and_uses_deadline():
    client = YamArmClient(1234, "localhost", 0.4, 0.25)
    future = _Future(error=TimeoutError("late"))
    client._client = MagicMock()
    client._client.health.return_value = future

    with pytest.raises(RuntimeError, match="exceeded 0.400s"):
        client._rpc("health")

    assert future.timeouts == [0.4]


def test_arm_client_send_failure_is_bounded_not_hung():
    """A dropped Portal must raise from the send itself, not block the control loop."""
    client = YamArmClient(1234, "localhost", 0.4, 0.25)
    client._client = MagicMock()
    client._client.command_joint_pos_guarded.side_effect = ConnectionError("socket is gone")

    with pytest.raises(RuntimeError, match="command_joint_pos_guarded"):
        client._rpc("command_joint_pos_guarded", np.zeros(7))


def test_arm_client_rejects_unknown_rpc_names():
    """portal.Client synthesizes any attribute, so the name must be checked against the schema."""
    client = YamArmClient(1234, "localhost", 0.4, 0.25)
    client._client = MagicMock()

    with pytest.raises(RuntimeError, match="does not implement required RPC"):
        client._rpc("command_joint_pos")  # not the guarded name
    client._client.command_joint_pos.assert_not_called()


def test_arm_client_rejects_stale_robot_state_age():
    """The wrapper clock stays current even when i2rt's control thread has died."""
    client = YamArmClient(1234, "localhost", 0.4, 0.25)
    envelope = _envelope(1)
    envelope["robot_state_age_seconds"] = 5.0
    client._client = _FakePortal([envelope])
    client._health = _health()

    with pytest.raises(RuntimeError, match="joint state is stale"):
        client.get_observations()


def test_disconnect_closes_the_portal_socket():
    """Dropping the reference leaves portal auto-reconnecting and re-sending queued motion."""
    client = YamArmClient(1234, "localhost", 0.4, 0.25)
    fake = MagicMock()
    fake.zero_torque.return_value = _Future({"schema": FOLLOWER_PORTAL_SCHEMA, "torque_enabled": False})
    client._client = fake
    client._health = _health()

    client.disconnect(disable_torque=True)

    fake.close.assert_called_once()
    assert client._client is None


def test_disconnect_requests_zero_torque_on_both_arms():
    robot = _robot()

    robot.disconnect()

    robot.left_arm.disconnect.assert_called_once_with(disable_torque=True)
    robot.right_arm.disconnect.assert_called_once_with(disable_torque=True)


def test_camera_connect_failure_disconnects_earlier_cameras_and_arms():
    robot = BiYamFollower(_config())
    for arm in (robot.left_arm, robot.right_arm):
        arm.connect = MagicMock()
        arm.disconnect = MagicMock()
        arm._health = _health()
    first_camera = MagicMock()
    second_camera = MagicMock()
    second_camera.connect.side_effect = RuntimeError("camera unavailable")
    robot.cameras = {"first": first_camera, "second": second_camera}

    with pytest.raises(RuntimeError, match="camera unavailable"):
        robot.connect()

    first_camera.disconnect.assert_called_once_with()
    second_camera.disconnect.assert_not_called()
    robot.left_arm.disconnect.assert_called_once_with(disable_torque=True)
    robot.right_arm.disconnect.assert_called_once_with(disable_torque=True)
    assert robot.is_connected is False


def test_relative_caps_use_the_robot_key_namespace():
    """Dict caps written with the keys the robot actually exposes must be accepted,
    and the historical joint_0..6.pos namespace (which silently made joint_6 the
    gripper cap) must be rejected with a message naming the real keys."""
    caps = BiYamFollower._relative_caps(_caps(side="left"), side="left", label="left_arm_max_relative_target")
    assert caps is not None and caps.shape == (7,)

    with pytest.raises(ValueError, match="left_joint_0.pos"):
        BiYamFollower._relative_caps(
            {f"joint_{index}.pos": 0.2 for index in range(7)},
            side="left",
            label="left_arm_max_relative_target",
        )

    # Scalar float caps stay supported.
    scalar = BiYamFollower._relative_caps(0.1, side="right", label="right_arm_max_relative_target")
    assert scalar is not None and (scalar == 0.1).all()
