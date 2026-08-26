#!/usr/bin/env python

from __future__ import annotations

import time
from unittest.mock import MagicMock

import numpy as np
import pytest

from lerobot.teleoperators.bi_yam_leader import BiYamLeader, BiYamLeaderConfig, YamLeaderClient
from lerobot.utils.yam_portal import LEADER_PORTAL_SCHEMA


class _Future:
    def __init__(self, value):
        self.value = value
        self.timeouts = []

    def result(self, timeout):
        self.timeouts.append(timeout)
        return self.value


class _FakePortal:
    def __init__(self, envelopes):
        self.envelopes = iter(envelopes)

    def get_observations_guarded(self):
        return _Future(next(self.envelopes))


def _envelope(sequence: int, *, age: float = 0.0):
    return {
        "schema": LEADER_PORTAL_SCHEMA,
        "sequence": sequence,
        "server_time_unix_ns": time.time_ns() - int(age * 1e9),
        "observation": {
            "joint_pos": np.arange(6, dtype=np.float64),
            "gripper_pos": np.asarray([0.25]),
            "io_inputs": np.asarray([1.0]),
        },
    }


def _leader() -> BiYamLeader:
    leader = BiYamLeader(BiYamLeaderConfig(require_portal_readiness=False))
    for arm, offset in ((leader.left_arm, 0.0), (leader.right_arm, 10.0)):
        arm._client = object()
        arm._health = {"schema": LEADER_PORTAL_SCHEMA, "dofs": 7, "read_only": True}
        arm.get_joint_pos = MagicMock(
            return_value=np.concatenate((np.arange(6, dtype=np.float64) + offset, [0.5]))
        )
    return leader


def test_leader_config_requires_full_supervisor_readiness():
    with pytest.raises(ValueError, match="--mode=full"):
        BiYamLeaderConfig()


def test_leader_interface_is_exact_canonical_fourteen_and_read_only():
    leader = _leader()
    expected = [
        *(f"left_joint_{index}.pos" for index in range(6)),
        "left_gripper.pos",
        *(f"right_joint_{index}.pos" for index in range(6)),
        "right_gripper.pos",
    ]

    assert list(leader.action_features) == expected
    assert list(leader.get_action()) == expected
    assert leader.feedback_features == {}
    with pytest.raises(RuntimeError, match="read-only"):
        leader.send_feedback({"left_joint_0.pos": 0.0})


def test_leader_client_rejects_stale_sequence_age_and_gripper_range():
    client = YamLeaderClient(5001, "localhost", 0.4, 0.25)
    client._client = _FakePortal([_envelope(1), _envelope(1)])
    client._health = {"schema": LEADER_PORTAL_SCHEMA}
    assert client.get_joint_pos().shape == (7,)
    with pytest.raises(RuntimeError, match="sequence is stale"):
        client.get_joint_pos()

    client._client = _FakePortal([_envelope(2, age=0.5)])
    with pytest.raises(RuntimeError, match="observation is stale"):
        client.get_joint_pos()

    bad = _envelope(3)
    bad["observation"]["gripper_pos"] = np.asarray([1.1])
    client._client = _FakePortal([bad])
    with pytest.raises(RuntimeError, match=r"must be in \[0, 1\]"):
        client.get_joint_pos()
