# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

import pytest

pytest.importorskip("grpc")

from lerobot.async_inference.configs import PolicyServerConfig, RobotClientConfig
from lerobot.async_inference.helpers import RemotePolicyConfig
from tests.mocks.mock_robot import MockRobotConfig


def test_policy_server_config_dict_round_trip() -> None:
    config = PolicyServerConfig(host="0.0.0.0", port=9000, fps=20, obs_queue_timeout=2.5)

    restored = PolicyServerConfig.from_dict(config.to_dict())

    assert restored == config
    assert "environment_dt" not in config.to_dict()


def test_robot_client_config_dict_round_trip_preserves_all_declared_fields() -> None:
    config = RobotClientConfig(
        policy_type="test",
        robot=MockRobotConfig(),
        actions_per_chunk=12,
        pretrained_name_or_path="checkpoint",
        duration=4.0,
        dispatch_actions=False,
    )

    payload = config.to_dict()
    restored = RobotClientConfig.from_dict(payload)

    assert isinstance(restored.robot, MockRobotConfig)
    assert restored.to_dict() == payload
    assert payload["duration"] == 4.0
    assert payload["dispatch_actions"] is False


def test_remote_policy_config_carries_control_fps() -> None:
    config = RemotePolicyConfig(
        policy_type="test",
        pretrained_name_or_path="checkpoint",
        lerobot_features={},
        actions_per_chunk=12,
        fps=20.0,
    )

    assert config.fps == 20.0


@pytest.mark.parametrize("fps", [0.0, float("nan"), float("inf")])
def test_async_configs_require_finite_positive_fps(fps: float) -> None:
    with pytest.raises(ValueError, match="fps must be finite and positive"):
        PolicyServerConfig(fps=fps)
    with pytest.raises(ValueError, match="fps must be finite and positive"):
        RobotClientConfig(
            policy_type="test",
            robot=MockRobotConfig(),
            actions_per_chunk=1,
            pretrained_name_or_path="checkpoint",
            fps=fps,
        )
