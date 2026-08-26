#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

"""Plain-portal bimanual YAM leader (teleoperator).

Ported verbatim from suveene-lerobot-client e0bf4a54 — the leader class the
hardware-validated YAM teleop/recording setup ran (plain i2rt Portal reads,
teaching-handle encoder button as gripper, no readiness gating). Only the
class/config names and the draccus registry name differ; this repository's
``bi_yam_leader`` is the guarded-Portal variant.
"""

import logging
from functools import cached_property
from typing import TYPE_CHECKING

import numpy as np

from lerobot.utils.import_utils import _portal_available, require_package

if TYPE_CHECKING or _portal_available:
    import portal
else:
    portal = None

from ..teleoperator import Teleoperator
from .config_bi_yam_leader_plain import BiYamLeaderPlainConfig

logger = logging.getLogger(__name__)

_YAM_ARM_DOF_COUNT = 7
_YAM_ARM_JOINT_COUNT = _YAM_ARM_DOF_COUNT - 1


class YamLeaderClient:
    """Client interface for a single Yam leader arm using the portal RPC framework."""

    def __init__(self, port: int, host: str = "localhost"):
        """
        Initialize the Yam leader arm client.

        Args:
            port: Server port for the leader arm
            host: Server host address
        """
        self.port = port
        self.host = host
        self._client = None

    def connect(self):
        """Connect to the leader arm server."""
        require_package("portal", extra="yam")
        logger.info(f"Connecting to Yam leader arm server at {self.host}:{self.port}")
        self._client = portal.Client(f"{self.host}:{self.port}")
        logger.info(f"Successfully connected to Yam leader arm server at {self.host}:{self.port}")

    def disconnect(self):
        """Disconnect from the leader arm server."""
        if self._client is not None:
            logger.info(f"Disconnecting from Yam leader arm server at {self.host}:{self.port}")
            self._client = None

    @property
    def is_connected(self) -> bool:
        """Check if the client is connected."""
        return self._client is not None

    def num_dofs(self) -> int:
        """Get the number of degrees of freedom."""
        if self._client is None:
            raise RuntimeError("Client not connected")
        return self._client.num_dofs().result()

    def get_joint_pos(self) -> np.ndarray:
        """Get current joint positions from the leader arm."""
        if self._client is None:
            raise RuntimeError("Client not connected")
        return self._client.get_joint_pos().result()

    def get_observations(self) -> dict[str, np.ndarray]:
        """Get current observations including joint positions, velocities, etc."""
        if self._client is None:
            raise RuntimeError("Client not connected")
        return self._client.get_observations().result()

    def get_gripper_from_encoder(self, observation: dict[str, np.ndarray] | None = None) -> float:
        """
        Try to get gripper state from teaching handle encoder button.
        Returns a value between 0 (closed) and 1 (open).
        Falls back to 1.0 (open) if not available.
        """
        if self._client is None:
            raise RuntimeError("Client not connected")
        try:
            # Try to get encoder state if the server exposes it
            # This requires custom method in the i2rt server
            obs = observation if observation is not None else self._client.get_observations().result()
            # Check if encoder button data is available in observations
            # The encoder button state might be in io_inputs or similar field
            if "io_inputs" in obs:
                # Button pressed = closed gripper (0), not pressed = open (1)
                return 0.0 if obs["io_inputs"][0] > 0.5 else 1.0
            return 1.0  # Default to open if no encoder data
        except Exception:
            return 1.0  # Default to open on any error


class BiYamLeaderPlain(Teleoperator):
    """
    Bimanual Yam Arms leader (teleoperator) using the i2rt library.

    This teleoperator reads joint positions from two Yam leader arms (with teaching handles)
    and provides them as actions for the follower robot.

    Expected setup:
    - Two Yam leader arms connected via CAN interfaces with teaching handles
    - Server processes running for each leader arm in read-only mode
    - Left leader arm server on port 5002 (default)
    - Right leader arm server on port 5001 (default)

    Note: You'll need to run separate server processes for the leader arms.
    You can modify the i2rt minimum_gello.py script to create read-only
    servers that just expose the leader arm state without trying to control
    a follower.
    """

    config_class = BiYamLeaderPlainConfig
    name = "bi_yam_leader_plain"

    def __init__(self, config: BiYamLeaderPlainConfig):
        super().__init__(config)
        self.config = config

        # Create clients for left and right leader arms
        self.left_arm = YamLeaderClient(port=config.left_arm_port, host=config.server_host)
        self.right_arm = YamLeaderClient(port=config.right_arm_port, host=config.server_host)

    @cached_property
    def action_features(self) -> dict[str, type]:
        """Define the canonical 6-joint-plus-gripper feature contract for both arms."""
        features = {f"left_joint_{i}.pos": float for i in range(_YAM_ARM_JOINT_COUNT)}
        features["left_gripper.pos"] = float
        features.update({f"right_joint_{i}.pos": float for i in range(_YAM_ARM_JOINT_COUNT)})
        features["right_gripper.pos"] = float
        return features

    @cached_property
    def feedback_features(self) -> dict[str, type]:
        """Yam leader arms don't support feedback."""
        return {}

    @property
    def is_connected(self) -> bool:
        """Check if both leader arms are connected."""
        return self.left_arm.is_connected and self.right_arm.is_connected

    def connect(self, calibrate: bool = True) -> None:
        """
        Connect to both leader arm servers.

        Args:
            calibrate: Not used for Yam arms (kept for API compatibility)
        """
        logger.info("Connecting to bimanual Yam leader arms")

        # Connect to leader arm servers
        self.left_arm.connect()
        self.right_arm.connect()

        left_dofs = self.left_arm.num_dofs()
        right_dofs = self.right_arm.num_dofs()
        if left_dofs != _YAM_ARM_DOF_COUNT or right_dofs != _YAM_ARM_DOF_COUNT:
            self.left_arm.disconnect()
            self.right_arm.disconnect()
            raise RuntimeError(
                "Plain bimanual YAM leader requires exactly 7 DOFs per arm "
                f"(left={left_dofs}, right={right_dofs})."
            )

        logger.info("Left leader arm DOFs: %d, Right leader arm DOFs: %d", left_dofs, right_dofs)
        logger.info("Successfully connected to bimanual Yam leader arms")

    @property
    def is_calibrated(self) -> bool:
        """Yam leader arms don't require calibration in the lerobot sense."""
        return self.is_connected

    def calibrate(self) -> None:
        """Yam leader arms don't require calibration in the lerobot sense."""
        pass

    def configure(self) -> None:
        """Configure the teleoperator (not needed for Yam leader arms)."""
        pass

    def setup_motors(self) -> None:
        """Setup motors (not needed for Yam leader arms)."""
        pass

    def get_action(self) -> dict[str, float]:
        """
        Get action from both leader arms by reading their current joint positions.

        For teaching handles (no physical gripper), we try to read encoder button state
        to control the gripper, falling back to fully open if not available.

        Returns:
            Dictionary with joint positions for both arms (including gripper)
        """
        action_dict: dict[str, float] = {}

        left_obs = self.left_arm.get_observations()
        self._populate_arm_action(action_dict, "left", self.left_arm, left_obs)

        right_obs = self.right_arm.get_observations()
        self._populate_arm_action(action_dict, "right", self.right_arm, right_obs)

        return action_dict

    @staticmethod
    def _populate_arm_action(
        action: dict[str, float],
        side: str,
        arm: YamLeaderClient,
        observation: dict[str, np.ndarray],
    ) -> None:
        joint_positions = np.asarray(observation["joint_pos"]).reshape(-1)
        if joint_positions.shape != (_YAM_ARM_JOINT_COUNT,):
            raise RuntimeError(
                f"Plain YAM {side} leader observation must contain 6 joint positions; "
                f"got shape {joint_positions.shape}."
            )
        for index, position in enumerate(joint_positions):
            action[f"{side}_joint_{index}.pos"] = float(position)

        if "gripper_pos" in observation:
            gripper_values = np.asarray(observation["gripper_pos"]).reshape(-1)
            if gripper_values.shape != (1,):
                raise RuntimeError(
                    f"Plain YAM {side} leader gripper observation must have shape (1,); "
                    f"got {gripper_values.shape}."
                )
            gripper_position = float(gripper_values[0])
        else:
            gripper_position = arm.get_gripper_from_encoder(observation)
        action[f"{side}_gripper.pos"] = gripper_position

    def send_feedback(self, feedback: dict[str, float]) -> None:
        """
        Send feedback to leader arms (not supported for Yam teaching handles).

        Args:
            feedback: Dictionary with feedback values (ignored)
        """
        # Yam teaching handles are passive devices and don't support feedback
        pass

    def disconnect(self) -> None:
        """Disconnect from both leader arms."""
        logger.info("Disconnecting from bimanual Yam leader arms")

        self.left_arm.disconnect()
        self.right_arm.disconnect()

        logger.info("Disconnected from bimanual Yam leader arms")
