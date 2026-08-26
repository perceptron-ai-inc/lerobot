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

import math
from dataclasses import dataclass, field
from pathlib import Path

from lerobot.cameras import CameraConfig

from ..config import RobotConfig


@RobotConfig.register_subclass("bi_yam_follower_guarded")
@dataclass
class BiYamFollowerConfig(RobotConfig):
    """Guarded local-Portal configuration for two seven-DOF YAM followers."""

    left_arm_port: int = 1235
    right_arm_port: int = 1234
    server_host: str = "localhost"

    # A token emitted and refreshed by lerobot-yam-portals. It binds the
    # supervisor and child process identities, ports, and post-launch CAN feedback.
    portal_readiness_file: Path | None = None
    require_portal_readiness: bool = True
    readiness_max_age_seconds: float = 3.0

    rpc_timeout_seconds: float = 1.0
    max_observation_age_seconds: float = 0.25
    command_watchdog_seconds: float = 4.0

    # Torque-off prediction is the default. Motion requires an explicit opt-in
    # and per-joint delta caps on both arms.
    torque_enabled: bool = False
    disable_torque_on_disconnect: bool = True
    left_arm_max_relative_target: float | dict[str, float] | None = None
    right_arm_max_relative_target: float | dict[str, float] | None = None

    cameras: dict[str, CameraConfig] = field(default_factory=dict)

    def __post_init__(self) -> None:
        super().__post_init__()
        if not (1 <= self.left_arm_port <= 65535 and 1 <= self.right_arm_port <= 65535):
            raise ValueError("YAM Portal ports must be in [1, 65535].")
        if self.left_arm_port == self.right_arm_port:
            raise ValueError("YAM left and right Portal ports must be distinct.")
        for name in (
            "readiness_max_age_seconds",
            "rpc_timeout_seconds",
            "max_observation_age_seconds",
            "command_watchdog_seconds",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive, got {value!r}.")
        if self.require_portal_readiness and self.portal_readiness_file is None:
            raise ValueError(
                "Guarded YAM requires portal_readiness_file from lerobot-yam-portals; "
                "set require_portal_readiness=false only for isolated fake-Portal tests."
            )
        if self.torque_enabled and (
            self.left_arm_max_relative_target is None or self.right_arm_max_relative_target is None
        ):
            raise ValueError(
                "Torque-enabled YAM rollout requires explicit left and right max-relative targets."
            )
