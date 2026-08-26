#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

from __future__ import annotations

from functools import cached_property
from typing import TYPE_CHECKING, Any

import numpy as np

from lerobot.lerobot_types import RobotAction
from lerobot.utils.decorators import check_if_not_connected
from lerobot.utils.import_utils import _portal_available, require_package
from lerobot.utils.yam_portal import (
    LEADER_PORTAL_SCHEMA,
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

from ..teleoperator import Teleoperator
from .config_bi_yam_leader import BiYamLeaderConfig


class YamLeaderClient(GuardedYamPortalClient):
    """Timeout-bounded client for one read-only guarded leader Portal."""

    # ``portal.Client.__getattr__`` synthesizes a callable for any attribute, so a missing RPC
    # cannot be detected with ``getattr(..., None)``; validate against the schema instead.
    RPC_NAMES = frozenset({"health", "get_observations_guarded"})
    PORTAL_SCHEMA = LEADER_PORTAL_SCHEMA
    PORTAL_LABEL = "YAM leader Portal"

    def connect(self) -> None:
        require_package("portal", extra="yam")
        self._connect_client(portal.Client(f"{self.host}:{self.port}", autoconn=False))

    def _validate_health(self, health: dict[str, Any]) -> None:
        if health.get("dofs") != 7 or health.get("read_only") is not True:
            raise RuntimeError("YAM leader Portal must expose exactly seven read-only DOFs.")
        if health.get("encoder_available") is not True:
            raise RuntimeError("YAM leader Portal did not authenticate its teaching-handle encoder.")

    def get_joint_pos(self) -> np.ndarray:
        envelope = self._rpc("get_observations_guarded")
        sequence = validate_guarded_observation_envelope(
            envelope,
            schema=self.PORTAL_SCHEMA,
            observation_label="YAM leader",
            timestamp_subject="observation",
            last_sequence=self._last_observation_sequence,
            max_age_seconds=self.max_observation_age_seconds,
        )
        observation = envelope.get("observation")
        if not isinstance(observation, dict):
            raise RuntimeError("YAM leader observation payload must be a dictionary.")
        joints = finite_vector(observation.get("joint_pos"), size=6, label="YAM leader joints")
        gripper = finite_vector(observation.get("gripper_pos"), size=1, label="YAM leader teaching handle")
        if gripper.item() < 0.0 or gripper.item() > 1.0:
            raise RuntimeError("YAM leader teaching-handle value must be in [0, 1].")
        self._last_observation_sequence = sequence
        return np.concatenate((joints, gripper))

    def disconnect(self) -> None:
        self._close_client()


class BiYamLeader(Teleoperator):
    """Read-only bimanual YAM teaching handles in canonical left-then-right order."""

    config_class = BiYamLeaderConfig
    name = "bi_yam_leader"

    def __init__(self, config: BiYamLeaderConfig):
        super().__init__(config)
        self.config = config
        self.left_arm = YamLeaderClient(
            config.left_arm_port,
            config.server_host,
            config.rpc_timeout_seconds,
            config.max_observation_age_seconds,
        )
        self.right_arm = YamLeaderClient(
            config.right_arm_port,
            config.server_host,
            config.rpc_timeout_seconds,
            config.max_observation_age_seconds,
        )

    @staticmethod
    def _arm_keys(side: str) -> list[str]:
        return [*(f"{side}_joint_{index}.pos" for index in range(6)), f"{side}_gripper.pos"]

    @cached_property
    def action_features(self) -> dict[str, type]:
        return {key: float for side in ("left", "right") for key in self._arm_keys(side)}

    @cached_property
    def feedback_features(self) -> dict[str, type]:
        return {}

    @property
    def is_connected(self) -> bool:
        return self.left_arm.is_connected and self.right_arm.is_connected

    def _validate_readiness(self) -> None:
        if not self.config.require_portal_readiness:
            return
        if self.config.server_host not in LOCAL_HOSTS:
            raise RuntimeError("Guarded YAM readiness is local-only; use a loopback server_host.")
        validate_yam_portal_readiness(
            self.config.portal_readiness_file,
            required_portals={
                "can_leader_l": {
                    "role": "leader",
                    "side": "left",
                    "port": self.config.left_arm_port,
                },
                "can_leader_r": {
                    "role": "leader",
                    "side": "right",
                    "port": self.config.right_arm_port,
                },
            },
            max_age_seconds=self.config.readiness_max_age_seconds,
        )

    def connect(self, calibrate: bool = True) -> None:
        del calibrate
        if self.is_connected:
            raise RuntimeError("BiYamLeader is already connected.")
        self._validate_readiness()
        connected = []
        try:
            for arm in (self.left_arm, self.right_arm):
                arm.connect()
                connected.append(arm)
        except Exception:
            for arm in connected:
                arm.disconnect()
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

    @check_if_not_connected
    def get_action(self) -> RobotAction:
        self._validate_readiness()
        action: RobotAction = {}
        for side, arm in (("left", self.left_arm), ("right", self.right_arm)):
            values = arm.get_joint_pos()
            action.update(zip(self._arm_keys(side), values, strict=True))
        return {key: float(value) for key, value in action.items()}

    @check_if_not_connected
    def send_feedback(self, feedback: dict[str, Any]) -> None:
        if feedback:
            raise RuntimeError("The guarded YAM leader Portal is read-only and rejects feedback motion.")

    def disconnect(self) -> None:
        self.left_arm.disconnect()
        self.right_arm.disconnect()
