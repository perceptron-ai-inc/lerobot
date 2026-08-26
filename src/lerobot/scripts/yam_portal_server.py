#!/usr/bin/env python3

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

"""Guarded Portal RPC server for one seven-DOF YAM follower."""

from __future__ import annotations

import argparse
import logging
import os
import signal
import threading
import time
from typing import TYPE_CHECKING, Any

import numpy as np

from lerobot.utils.import_utils import _i2rt_available
from lerobot.utils.yam_portal import (
    FOLLOWER_PORTAL_SCHEMA,
    LEADER_PORTAL_SCHEMA,
    finite_vector,
    validate_yam_robot_state_live,
)

if TYPE_CHECKING or _i2rt_available:
    from i2rt.robots.get_robot import get_yam_robot
    from i2rt.robots.utils import GripperType

logger = logging.getLogger(__name__)

# The guarded RPCs are unauthenticated, so the servers must not be reachable off-host.
DEFAULT_PORTAL_BIND_HOST = "127.0.0.1"

# i2rt normalizes the gripper to [0, 1] in both directions: ``get_observations`` reports it
# through ``to_command_joint_pos_space`` and ``command_joint_pos`` maps it back with
# ``to_robot_joint_pos_space``. The raw ``gripper_limits`` from ``get_robot_info`` are in
# motor space (radians) and must never be used to bound a command.
GRIPPER_COMMAND_BOUNDS = (0.0, 1.0)


class GuardedYamPortalServer:
    """Add freshness, command acknowledgements, bounds, and zero torque to i2rt."""

    def __init__(
        self,
        robot: Any,
        port: int,
        *,
        command_watchdog_seconds: float = 4.0,
        max_robot_state_age_seconds: float = 0.5,
        host: str = DEFAULT_PORTAL_BIND_HOST,
        portal_module=None,
        start_watchdog: bool = True,
        fatal_watchdog=None,
    ):
        if portal_module is None:
            import portal as portal_module

        self._robot = robot
        self._lock = threading.Lock()
        self._observation_sequence = 0
        self._command_sequence = 0
        self._torque_enabled = True
        self._last_command_monotonic: float | None = None
        self._command_watchdog_seconds = float(command_watchdog_seconds)
        self._max_robot_state_age_seconds = float(max_robot_state_age_seconds)
        self._fatal_watchdog = fatal_watchdog or (lambda: os.kill(os.getpid(), signal.SIGTERM))
        if not np.isfinite(self._command_watchdog_seconds) or self._command_watchdog_seconds <= 0:
            raise ValueError("command_watchdog_seconds must be finite and positive.")
        if not np.isfinite(self._max_robot_state_age_seconds) or self._max_robot_state_age_seconds <= 0:
            raise ValueError("max_robot_state_age_seconds must be finite and positive.")
        # Bind loopback by default: the mutating RPCs below carry no credential, and every
        # supported client is required to be local anyway.
        self._server = portal_module.Server(int(port), host=str(host))
        self._server.bind("health", self.health)
        self._server.bind("get_observations_guarded", self.get_observations_guarded)
        self._server.bind("command_joint_pos_guarded", self.command_joint_pos_guarded)
        self._server.bind("zero_torque", self.zero_torque)
        if start_watchdog:
            threading.Thread(target=self._watchdog_loop, name="yam-command-watchdog", daemon=True).start()

    def _position_bounds(self) -> np.ndarray:
        info = self._robot.get_robot_info()
        joint_limits = np.asarray(info.get("joint_limits"), dtype=np.float64)
        # Required for its own sake: a YAM whose gripper never calibrated exposes no limits,
        # and its normalized command space would be meaningless.
        gripper_limits = np.asarray(info.get("gripper_limits"), dtype=np.float64).reshape(-1)
        if joint_limits.shape != (6, 2) or gripper_limits.shape != (2,):
            raise RuntimeError("i2rt YAM must expose six joint bounds and one gripper bound.")
        if not np.isfinite(gripper_limits).all() or float(gripper_limits[0]) == float(gripper_limits[1]):
            raise RuntimeError("i2rt YAM gripper limits must be finite and non-degenerate.")
        # The gripper is commanded in i2rt's normalized space, not the motor space the raw
        # limits are expressed in, so bound it at [0, 1] rather than at ``gripper_limits``.
        gripper_bounds = np.asarray([GRIPPER_COMMAND_BOUNDS], dtype=np.float64)
        bounds = np.concatenate((joint_limits, gripper_bounds), axis=0)
        if not np.isfinite(bounds).all() or bool((bounds[:, 0] >= bounds[:, 1]).any()):
            raise RuntimeError("i2rt YAM returned invalid position bounds.")
        return bounds

    def health(self) -> dict[str, Any]:
        with self._lock:
            torque_enabled = self._torque_enabled
        return {
            "schema": FOLLOWER_PORTAL_SCHEMA,
            "dofs": int(np.asarray(self._robot.num_dofs()).item()),
            "position_bounds": self._position_bounds(),
            "torque_enabled": torque_enabled,
            "command_watchdog_seconds": self._command_watchdog_seconds,
            "server_time_unix_ns": time.time_ns(),
        }

    def get_observations_guarded(self) -> dict[str, Any]:
        robot_state_age = validate_yam_robot_state_live(
            self._robot, max_age_seconds=self._max_robot_state_age_seconds
        )
        observation = self._robot.get_observations()
        if not isinstance(observation, dict):
            raise RuntimeError("i2rt YAM observation must be a dictionary.")
        with self._lock:
            self._observation_sequence += 1
            sequence = self._observation_sequence
        return {
            "schema": FOLLOWER_PORTAL_SCHEMA,
            "sequence": sequence,
            "server_time_unix_ns": time.time_ns(),
            "robot_state_age_seconds": robot_state_age,
            "observation": observation,
        }

    def command_joint_pos_guarded(self, joint_pos: Any) -> dict[str, Any]:
        target = finite_vector(joint_pos, size=7, label="YAM target")
        validate_yam_robot_state_live(self._robot, max_age_seconds=self._max_robot_state_age_seconds)
        bounds = self._position_bounds()
        if bool(((target < bounds[:, 0]) | (target > bounds[:, 1])).any()):
            raise ValueError("YAM target exceeds server-owned physical position bounds.")
        with self._lock:
            if not self._torque_enabled:
                raise RuntimeError("YAM Portal is in zero-torque mode; restart it before motion.")
            self._robot.command_joint_pos(target)
            self._command_sequence += 1
            self._last_command_monotonic = time.monotonic()
            sequence = self._command_sequence
        return {
            "schema": FOLLOWER_PORTAL_SCHEMA,
            "sequence": sequence,
            "server_time_unix_ns": time.time_ns(),
        }

    def zero_torque(self) -> dict[str, Any]:
        with self._lock:
            if self._torque_enabled:
                self._robot.zero_torque_mode()
            self._torque_enabled = False
        return {"schema": FOLLOWER_PORTAL_SCHEMA, "torque_enabled": False}

    def _check_command_watchdog(self, now: float | None = None) -> bool:
        """Latch zero torque after a post-motion command gap. Returns True when tripped."""
        with self._lock:
            if not self._torque_enabled or self._last_command_monotonic is None:
                return False
            elapsed = (time.monotonic() if now is None else now) - self._last_command_monotonic
            if elapsed <= self._command_watchdog_seconds:
                return False
            self._robot.zero_torque_mode()
            self._torque_enabled = False
            return True

    def _watchdog_loop(self) -> None:
        interval = min(0.1, self._command_watchdog_seconds / 4.0)
        while True:
            try:
                self._check_command_watchdog()
            except Exception:
                logger.exception("YAM command watchdog could not enter zero torque; terminating Portal")
                self._fatal_watchdog()
                return
            time.sleep(interval)

    def serve(self) -> None:
        self._server.start()


class GuardedYamLeaderPortalServer:
    """Read-only Portal for one six-joint YAM leader plus teaching-handle encoder."""

    def __init__(
        self,
        robot: Any,
        port: int,
        *,
        max_robot_state_age_seconds: float = 0.5,
        host: str = DEFAULT_PORTAL_BIND_HOST,
        portal_module=None,
    ):
        if portal_module is None:
            import portal as portal_module

        self._robot = robot
        self._lock = threading.Lock()
        self._observation_sequence = 0
        self._max_robot_state_age_seconds = float(max_robot_state_age_seconds)
        if not np.isfinite(self._max_robot_state_age_seconds) or self._max_robot_state_age_seconds <= 0:
            raise ValueError("max_robot_state_age_seconds must be finite and positive.")
        self._server = portal_module.Server(int(port), host=str(host))
        self._server.bind("health", self.health)
        self._server.bind("get_observations_guarded", self.get_observations_guarded)

    def _leader_observation(self) -> dict[str, np.ndarray]:
        observation = self._robot.get_observations()
        if not isinstance(observation, dict):
            raise RuntimeError("i2rt YAM leader observation must be a dictionary.")
        joints = finite_vector(observation.get("joint_pos"), size=6, label="YAM leader joints")
        encoder_states = self._robot.motor_chain.get_same_bus_device_states()
        if not isinstance(encoder_states, (list, tuple)) or len(encoder_states) != 1:
            raise RuntimeError("YAM leader must expose exactly one teaching-handle encoder.")
        encoder = encoder_states[0]
        encoder_position = np.asarray(getattr(encoder, "position", None), dtype=np.float64)
        if encoder_position.size != 1 or not np.isfinite(encoder_position).all():
            raise RuntimeError("YAM teaching-handle encoder position must be one finite scalar.")
        gripper = np.clip(1.0 - float(encoder_position.item()), 0.0, 1.0)
        io_inputs = np.asarray(getattr(encoder, "io_inputs", None), dtype=np.float64).reshape(-1)
        if io_inputs.size < 1 or not np.isfinite(io_inputs).all():
            raise RuntimeError("YAM teaching-handle encoder inputs must be finite and non-empty.")
        return {
            "joint_pos": joints,
            "gripper_pos": np.asarray([gripper], dtype=np.float64),
            "io_inputs": io_inputs,
        }

    def health(self) -> dict[str, Any]:
        # Reading once proves that both the motor chain and same-bus encoder are live.
        validate_yam_robot_state_live(self._robot, max_age_seconds=self._max_robot_state_age_seconds)
        self._leader_observation()
        return {
            "schema": LEADER_PORTAL_SCHEMA,
            "dofs": 7,
            "read_only": True,
            "encoder_available": True,
            "server_time_unix_ns": time.time_ns(),
        }

    def get_observations_guarded(self) -> dict[str, Any]:
        robot_state_age = validate_yam_robot_state_live(
            self._robot, max_age_seconds=self._max_robot_state_age_seconds
        )
        observation = self._leader_observation()
        with self._lock:
            self._observation_sequence += 1
            sequence = self._observation_sequence
        return {
            "schema": LEADER_PORTAL_SCHEMA,
            "sequence": sequence,
            "server_time_unix_ns": time.time_ns(),
            "robot_state_age_seconds": robot_state_age,
            "observation": observation,
        }

    def serve(self) -> None:
        self._server.start()


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve one guarded YAM follower over Portal.")
    parser.add_argument("--can-channel", required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--role", choices=("follower", "leader"), default="follower")
    parser.add_argument("--gripper")
    parser.add_argument("--command-watchdog-seconds", type=float, default=4.0)
    parser.add_argument(
        "--host",
        default=DEFAULT_PORTAL_BIND_HOST,
        help="Bind address. Defaults to loopback; the guarded RPCs are unauthenticated.",
    )
    args = parser.parse_args()

    if not _i2rt_available:
        raise ImportError(
            "'i2rt' is required to drive a YAM over CAN but is not installed. It lives in the "
            "`yam-hardware` dependency group (PEP 735 groups are not part of any extra, so "
            "`uv sync --extra all` does not pull it). Install with: "
            "uv sync --extra yam --group yam-hardware"
        )

    gripper = args.gripper or ("yam_teaching_handle" if args.role == "leader" else "linear_4310")
    robot = get_yam_robot(
        channel=args.can_channel,
        gripper_type=GripperType.from_string_name(gripper),
    )
    if args.role == "leader":
        server = GuardedYamLeaderPortalServer(robot, args.port, host=args.host)
    else:
        server = GuardedYamPortalServer(
            robot,
            args.port,
            command_watchdog_seconds=args.command_watchdog_seconds,
            host=args.host,
        )

    def request_shutdown(_signum, _frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, request_shutdown)
    signal.signal(signal.SIGHUP, request_shutdown)
    try:
        server.serve()
    except KeyboardInterrupt:
        pass
    finally:
        robot.close()


if __name__ == "__main__":
    main()
