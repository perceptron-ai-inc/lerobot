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

import logging
from functools import cached_property
from typing import TYPE_CHECKING, Any

import numpy as np

from lerobot.cameras.camera import Camera
from lerobot.cameras.utils import make_cameras_from_configs
from lerobot.lerobot_types import RobotAction, RobotObservation
from lerobot.utils.decorators import check_if_not_connected
from lerobot.utils.import_utils import _portal_available, require_package
from lerobot.utils.yam_portal import (
    FOLLOWER_PORTAL_SCHEMA,
    LOCAL_HOSTS,
    GuardedYamPortalClient,
    finite_vector,
    validate_guarded_observation_envelope,
    validate_yam_portal_readiness,
)

if TYPE_CHECKING or _portal_available:
    import portal
else:
    portal = None

from ..robot import Robot
from .config_bi_yam_follower import BiYamFollowerConfig

logger = logging.getLogger(__name__)


class YamArmClient(GuardedYamPortalClient):
    """Timeout-bounded client for one guarded YAM Portal."""

    # ``portal.Client.__getattr__`` synthesizes a callable for any attribute, so a missing
    # RPC cannot be detected with ``getattr(..., None)``; validate against the schema instead.
    RPC_NAMES = frozenset({"health", "get_observations_guarded", "command_joint_pos_guarded", "zero_torque"})
    PORTAL_SCHEMA = FOLLOWER_PORTAL_SCHEMA

    def __init__(self, port: int, host: str, rpc_timeout_seconds: float, max_observation_age_seconds: float):
        super().__init__(port, host, rpc_timeout_seconds, max_observation_age_seconds)
        self._last_command_sequence = -1

    def connect(self) -> None:
        require_package("portal", extra="yam")
        # autoconn=False makes every send fail fast on a dropped socket and stops portal from
        # silently re-sending queued motion commands after a reconnect.
        self._connect_client(portal.Client(f"{self.host}:{self.port}", autoconn=False))

    def _validate_health(self, health: dict[str, Any]) -> None:
        dofs = np.asarray(health.get("dofs")).item()
        if not isinstance(dofs, (int, np.integer)) or isinstance(dofs, bool) or int(dofs) != 7:
            raise RuntimeError(f"YAM Portal must expose exactly 7 DOFs, got {dofs!r}.")

    def _reset_sequences(self) -> None:
        super()._reset_sequences()
        self._last_command_sequence = -1

    def get_observations(self) -> dict[str, np.ndarray]:
        envelope = self._rpc("get_observations_guarded")
        sequence = validate_guarded_observation_envelope(
            envelope,
            schema=self.PORTAL_SCHEMA,
            observation_label="YAM Portal",
            timestamp_subject="observation envelope",
            last_sequence=self._last_observation_sequence,
            max_age_seconds=self.max_observation_age_seconds,
        )
        # The envelope clock is stamped by the Portal wrapper at service time, so it stays
        # current even when i2rt's control thread has died. Validate the robot-owned age too.
        robot_state_age = envelope.get("robot_state_age_seconds")
        if robot_state_age is not None:
            robot_state_age = float(robot_state_age)
            if not np.isfinite(robot_state_age) or robot_state_age > self.max_observation_age_seconds:
                raise RuntimeError(
                    f"YAM robot joint state is stale (age={robot_state_age:.3f}s, "
                    f"limit={self.max_observation_age_seconds:.3f}s)."
                )
        observation = envelope.get("observation")
        if not isinstance(observation, dict):
            raise RuntimeError("YAM Portal observation payload must be a dictionary.")
        self._last_observation_sequence = sequence
        return observation

    def get_joint_pos(self) -> np.ndarray:
        observation = self.get_observations()
        joint_pos = np.asarray(observation.get("joint_pos"), dtype=np.float64)
        gripper_pos = np.asarray(observation.get("gripper_pos"), dtype=np.float64)
        return np.concatenate((joint_pos, gripper_pos))

    def command_joint_pos(self, joint_pos: np.ndarray) -> None:
        ack = self._rpc("command_joint_pos_guarded", joint_pos)
        if not isinstance(ack, dict) or ack.get("schema") != FOLLOWER_PORTAL_SCHEMA:
            raise RuntimeError("YAM Portal command acknowledgement has an incompatible schema.")
        sequence = ack.get("sequence")
        if not isinstance(sequence, int) or sequence <= self._last_command_sequence:
            raise RuntimeError(
                f"YAM Portal command acknowledgement is stale: {sequence!r} <= {self._last_command_sequence}."
            )
        self._last_command_sequence = sequence

    def zero_torque(self) -> None:
        result = self._rpc("zero_torque")
        if result != {"schema": FOLLOWER_PORTAL_SCHEMA, "torque_enabled": False}:
            raise RuntimeError(f"YAM Portal rejected zero-torque transition: {result!r}.")

    def disconnect(self, *, disable_torque: bool) -> None:
        error = None
        if self._client is not None and disable_torque:
            try:
                self.zero_torque()
            except Exception as exc:  # disconnect both arms even if one Portal failed
                error = exc
        self._close_client()
        if error is not None:
            raise RuntimeError(
                f"Failed to place YAM Portal {self.host}:{self.port} in zero torque."
            ) from error


class BiYamFollower(Robot):
    """Bimanual YAM follower with a canonical left-then-right 14-DOF interface."""

    config_class = BiYamFollowerConfig
    name = "bi_yam_follower_guarded"

    def __init__(self, config: BiYamFollowerConfig):
        super().__init__(config)
        self.config = config
        self.left_arm = YamArmClient(
            config.left_arm_port,
            config.server_host,
            config.rpc_timeout_seconds,
            config.max_observation_age_seconds,
        )
        self.right_arm = YamArmClient(
            config.right_arm_port,
            config.server_host,
            config.rpc_timeout_seconds,
            config.max_observation_age_seconds,
        )
        self.cameras = make_cameras_from_configs(config.cameras)
        self._left_dofs: int | None = None
        self._right_dofs: int | None = None

    @staticmethod
    def _arm_keys(side: str) -> list[str]:
        return [*(f"{side}_joint_{index}.pos" for index in range(6)), f"{side}_gripper.pos"]

    @property
    def _motors_ft(self) -> dict[str, type]:
        return {key: float for side in ("left", "right") for key in self._arm_keys(side)}

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        return {name: (config.height, config.width, 3) for name, config in self.config.cameras.items()}

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        return {**self._motors_ft, **self._cameras_ft}

    @cached_property
    def action_features(self) -> dict[str, type]:
        return self._motors_ft

    @property
    def is_connected(self) -> bool:
        return (
            self.left_arm.is_connected
            and self.right_arm.is_connected
            and all(camera.is_connected for camera in self.cameras.values())
        )

    def _validate_readiness(self) -> None:
        if not self.config.require_portal_readiness:
            return
        if self.config.server_host not in LOCAL_HOSTS:
            raise RuntimeError("Guarded YAM readiness is local-only; use a loopback server_host.")
        validate_yam_portal_readiness(
            self.config.portal_readiness_file,
            required_portals={
                "can_follower_l": {
                    "role": "follower",
                    "side": "left",
                    "port": self.config.left_arm_port,
                },
                "can_follower_r": {
                    "role": "follower",
                    "side": "right",
                    "port": self.config.right_arm_port,
                },
            },
            max_age_seconds=self.config.readiness_max_age_seconds,
        )

    def connect(self, calibrate: bool = True) -> None:
        del calibrate
        if self.is_connected:
            raise RuntimeError("BiYamFollower is already connected.")
        self._validate_readiness()
        connected: list[YamArmClient] = []
        connected_cameras: list[Camera] = []
        try:
            for arm in (self.left_arm, self.right_arm):
                arm.connect()
                connected.append(arm)
            self._left_dofs = int(self.left_arm.health["dofs"])
            self._right_dofs = int(self.right_arm.health["dofs"])
            for side, arm in (("left", self.left_arm), ("right", self.right_arm)):
                if float(arm.health.get("command_watchdog_seconds", -1.0)) != float(
                    self.config.command_watchdog_seconds
                ):
                    raise RuntimeError(f"{side} YAM Portal command-watchdog contract mismatch.")
                if self.config.torque_enabled and arm.health.get("torque_enabled") is not True:
                    raise RuntimeError(
                        f"{side} YAM Portal is already in zero torque; restart it before motion."
                    )
            if not self.config.torque_enabled:
                self.left_arm.zero_torque()
                self.right_arm.zero_torque()
            for camera in self.cameras.values():
                camera.connect()
                connected_cameras.append(camera)
        except Exception:
            for camera in reversed(connected_cameras):
                try:
                    camera.disconnect()
                except Exception:
                    logger.exception("Failed to disconnect a YAM camera after connect failure")
            for arm in connected:
                try:
                    arm.disconnect(disable_torque=True)
                except Exception:
                    logger.exception("Failed to zero a YAM arm after connect failure")
            raise

    @property
    def is_calibrated(self) -> bool:
        return self.is_connected

    def calibrate(self) -> None:
        return None

    def configure(self) -> None:
        return None

    def setup_motors(self) -> None:
        return None

    @staticmethod
    def _relative_caps(value: float | dict[str, float] | None, *, side: str, label: str) -> np.ndarray | None:
        if value is None:
            return None
        if isinstance(value, dict):
            # Dict-form caps use the same key namespace as the rest of this robot
            # (`{side}_joint_0..5.pos` + `{side}_gripper.pos`), so the gripper cap is an
            # explicit key instead of a silent seventh "joint", and the names match what
            # `send_action` validates and `observation_features` exposes.
            expected_keys = BiYamFollower._arm_keys(side)
            if set(value) != set(expected_keys):
                raise ValueError(
                    f"{label} must contain exactly the {side} arm keys {expected_keys}; got {sorted(value)}."
                )
            raw = [value[key] for key in expected_keys]
        else:
            raw = [value] * 7
        caps = finite_vector(raw, size=7, label=label)
        if bool((caps <= 0).any()):
            raise ValueError(f"{label} values must be positive.")
        return caps

    @staticmethod
    def _physical_bounds(health: dict[str, Any], *, side: str) -> np.ndarray:
        bounds = np.asarray(health.get("position_bounds"), dtype=np.float64)
        if (
            bounds.shape != (7, 2)
            or not np.isfinite(bounds).all()
            or bool((bounds[:, 0] >= bounds[:, 1]).any())
        ):
            raise RuntimeError(f"{side} YAM Portal returned invalid physical position bounds.")
        return bounds

    def _arm_observation(self, arm: YamArmClient, *, side: str) -> np.ndarray:
        observation = arm.get_observations()
        joints = finite_vector(observation.get("joint_pos"), size=6, label=f"{side} joints")
        gripper = finite_vector(observation.get("gripper_pos"), size=1, label=f"{side} gripper")
        return np.concatenate((joints, gripper))

    @check_if_not_connected
    def get_observation(self) -> RobotObservation:
        self._validate_readiness()
        observation: RobotObservation = {}
        for side, arm in (("left", self.left_arm), ("right", self.right_arm)):
            values = self._arm_observation(arm, side=side)
            observation.update(zip(self._arm_keys(side), values, strict=True))
        for name, camera in self.cameras.items():
            observation[name] = camera.async_read()
        return observation

    def _fail_closed(self) -> None:
        for arm in (self.left_arm, self.right_arm):
            try:
                arm.zero_torque()
            except Exception:
                logger.exception("YAM fail-closed zero-torque RPC failed")

    @check_if_not_connected
    def send_action(self, action: RobotAction) -> RobotAction:
        if not self.config.torque_enabled:
            raise RuntimeError("YAM torque is disabled; prediction-only sessions cannot send actions.")
        if not isinstance(action, dict):
            raise TypeError(f"BiYAM action must be a dictionary, got {type(action).__name__}.")
        self._validate_readiness()
        expected_keys = {key for side in ("left", "right") for key in self._arm_keys(side)}
        if set(action) != expected_keys:
            missing = sorted(expected_keys - set(action))
            extra = sorted(set(action) - expected_keys)
            raise ValueError(f"BiYAM action keys must match the canonical 14 exactly; {missing=}, {extra=}.")

        targets: dict[str, np.ndarray] = {}
        currents: dict[str, np.ndarray] = {}
        caps: dict[str, np.ndarray] = {}
        bounds: dict[str, np.ndarray] = {}
        for side, arm, cap_config in (
            ("left", self.left_arm, self.config.left_arm_max_relative_target),
            ("right", self.right_arm, self.config.right_arm_max_relative_target),
        ):
            keys = self._arm_keys(side)
            missing = [key for key in keys if key not in action]
            if missing:
                raise ValueError(f"{side} YAM action is missing required targets: {missing}.")
            targets[side] = finite_vector([action[key] for key in keys], size=7, label=f"{side} target")
            caps[side] = self._relative_caps(cap_config, side=side, label=f"{side}_arm_max_relative_target")
            bounds[side] = self._physical_bounds(arm.health, side=side)

        # Validate both live states and both final commands before sending either arm.
        for side, arm in (("left", self.left_arm), ("right", self.right_arm)):
            currents[side] = self._arm_observation(arm, side=side)
            target = targets[side]
            physical = bounds[side]
            if bool(((target < physical[:, 0]) | (target > physical[:, 1])).any()):
                raise ValueError(f"{side} YAM target exceeds Portal-owned physical position bounds.")
            targets[side] = np.clip(target, currents[side] - caps[side], currents[side] + caps[side])

        try:
            self.left_arm.command_joint_pos(targets["left"])
            self.right_arm.command_joint_pos(targets["right"])
        except Exception:
            self._fail_closed()
            raise
        return {
            key: float(targets[side][index])
            for side in ("left", "right")
            for index, key in enumerate(self._arm_keys(side))
        }

    def disconnect(self) -> None:
        errors = []
        for arm in (self.left_arm, self.right_arm):
            try:
                arm.disconnect(disable_torque=self.config.disable_torque_on_disconnect)
            except Exception as exc:
                errors.append(exc)
        for camera in self.cameras.values():
            try:
                camera.disconnect()
            except Exception as exc:
                errors.append(exc)
        if errors:
            raise RuntimeError(f"BiYAM disconnect had {len(errors)} failure(s).") from errors[0]
