from __future__ import annotations

import json
import stat
import subprocess
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

from lerobot.robots.bi_yam_follower.bi_yam_follower import YamArmClient
from lerobot.scripts import lerobot_yam_portals as supervisor
from lerobot.scripts.yam_portal_server import GuardedYamLeaderPortalServer, GuardedYamPortalServer
from lerobot.teleoperators.bi_yam_leader.bi_yam_leader import YamLeaderClient
from lerobot.utils.yam_portal import (
    FOLLOWER_PORTAL_SCHEMA,
    LEADER_PORTAL_SCHEMA,
    READINESS_SCHEMA,
    validate_yam_portal_readiness,
)


class _Process:
    def __init__(self, pid: int, returncode=None):
        self.pid = pid
        self.returncode = returncode

    def poll(self):
        return self.returncode


class _Robot:
    def __init__(self):
        self.commands = []
        self.zero_calls = 0

    def num_dofs(self):
        return np.asarray(7)

    def get_robot_info(self):
        return {
            "joint_limits": np.asarray([[-2.0, 2.0]] * 6),
            "gripper_limits": np.asarray([1.0, 0.0]),
        }

    def get_observations(self):
        return {
            "joint_pos": np.zeros(6),
            "gripper_pos": np.zeros(1),
            "joint_eff": np.zeros(7),
        }

    def command_joint_pos(self, target):
        self.commands.append(target)

    def zero_torque_mode(self):
        self.zero_calls += 1


class _LeaderRobot:
    def __init__(self):
        encoder = SimpleNamespace(position=np.asarray(0.25), io_inputs=np.asarray([1.0]))
        self.motor_chain = SimpleNamespace(get_same_bus_device_states=lambda: [encoder])

    def get_observations(self):
        return {"joint_pos": np.arange(6, dtype=np.float64)}


class _PortalModule:
    class Server:
        def __init__(self, port, host=""):
            self.port = port
            self.host = host
            self.bindings = {}

        def bind(self, name, function):
            self.bindings[name] = function

        def start(self):
            return None


def test_guarded_server_validates_bounds_acknowledges_and_latches_zero_torque():
    robot = _Robot()
    server = GuardedYamPortalServer(robot, 1234, portal_module=_PortalModule, start_watchdog=False)

    assert server.health()["position_bounds"].shape == (7, 2)
    assert server.get_observations_guarded()["sequence"] == 1
    assert server.get_observations_guarded()["sequence"] == 2
    assert server.command_joint_pos_guarded(np.zeros(7))["sequence"] == 1
    with pytest.raises(ValueError, match="physical position bounds"):
        server.command_joint_pos_guarded(np.full(7, 3.0))

    assert server.zero_torque()["torque_enabled"] is False
    assert robot.zero_calls == 1
    with pytest.raises(RuntimeError, match="zero-torque mode"):
        server.command_joint_pos_guarded(np.zeros(7))


def test_guarded_servers_bind_loopback_by_default():
    """The guarded RPCs carry no credential, so the sockets must not be reachable off-host."""
    follower = GuardedYamPortalServer(_Robot(), 1234, portal_module=_PortalModule, start_watchdog=False)
    leader = GuardedYamLeaderPortalServer(_LeaderRobot(), 5001, portal_module=_PortalModule)

    assert follower._server.host == "127.0.0.1"
    assert leader._server.host == "127.0.0.1"


def test_gripper_is_bounded_in_normalized_space_not_motor_space():
    """i2rt commands the gripper in [0, 1]; ``gripper_limits`` is motor space (radians)."""
    robot = _Robot()
    robot.get_robot_info = lambda: {
        "joint_limits": np.asarray([[-2.0, 2.0]] * 6),
        # Realistic detected motor travel: wide enough to swallow any normalized command.
        "gripper_limits": np.asarray([0.0, -2.7]),
    }
    server = GuardedYamPortalServer(robot, 1234, portal_module=_PortalModule, start_watchdog=False)

    np.testing.assert_allclose(server._position_bounds()[6], [0.0, 1.0])
    target = np.zeros(7)
    target[6] = 2.5  # 250% open — inside motor travel, far outside the command space
    with pytest.raises(ValueError, match="physical position bounds"):
        server.command_joint_pos_guarded(target)
    target[6] = 1.0
    assert server.command_joint_pos_guarded(target)["sequence"] == 1


def test_degenerate_gripper_limits_are_rejected():
    robot = _Robot()
    robot.get_robot_info = lambda: {
        "joint_limits": np.asarray([[-2.0, 2.0]] * 6),
        "gripper_limits": np.asarray([0.5, 0.5]),
    }
    server = GuardedYamPortalServer(robot, 1234, portal_module=_PortalModule, start_watchdog=False)
    with pytest.raises(RuntimeError, match="non-degenerate"):
        server._position_bounds()


@pytest.mark.parametrize("role", ["follower", "leader"])
def test_frozen_robot_state_fails_closed_for_both_server_roles(role):
    """Neither server role may mask a dead i2rt thread with a fresh wrapper clock."""
    if role == "follower":
        robot = _Robot()
        robot.motor_chain = SimpleNamespace(running=False)
        server = GuardedYamPortalServer(robot, 1234, portal_module=_PortalModule, start_watchdog=False)
    else:
        robot = _LeaderRobot()
        robot.motor_chain.running = False
        server = GuardedYamLeaderPortalServer(robot, 5001, portal_module=_PortalModule)

    with pytest.raises(RuntimeError, match="motor chain has stopped"):
        server.get_observations_guarded()
    if role == "follower":
        with pytest.raises(RuntimeError, match="motor chain has stopped"):
            server.command_joint_pos_guarded(np.zeros(7))
        assert robot.commands == []
    else:
        with pytest.raises(RuntimeError, match="motor chain has stopped"):
            server.health()


def test_stale_robot_joint_state_fails_closed_and_fresh_state_is_reported():
    import time as _time

    robot = _Robot()
    robot.motor_chain = SimpleNamespace(running=True)
    robot._joint_state = SimpleNamespace(timestamp=_time.time() - 5.0)
    server = GuardedYamPortalServer(
        robot,
        1234,
        max_robot_state_age_seconds=0.5,
        portal_module=_PortalModule,
        start_watchdog=False,
    )
    with pytest.raises(RuntimeError, match="joint state is stale"):
        server.get_observations_guarded()

    robot._joint_state = SimpleNamespace(timestamp=_time.time())
    envelope = server.get_observations_guarded()
    assert envelope["robot_state_age_seconds"] is not None
    assert envelope["robot_state_age_seconds"] < 0.5


def test_guarded_server_command_gap_watchdog_latches_zero_torque():
    robot = _Robot()
    server = GuardedYamPortalServer(
        robot,
        1234,
        command_watchdog_seconds=0.5,
        portal_module=_PortalModule,
        start_watchdog=False,
    )
    server.command_joint_pos_guarded(np.zeros(7))
    last_command = server._last_command_monotonic

    assert server._check_command_watchdog(last_command + 0.49) is False
    assert server._check_command_watchdog(last_command + 0.51) is True
    assert robot.zero_calls == 1
    assert server.health()["torque_enabled"] is False


def test_stale_leader_joint_state_fails_closed_and_fresh_state_is_reported():
    import time as _time

    robot = _LeaderRobot()
    robot.motor_chain.running = True
    robot._joint_state = SimpleNamespace(timestamp=_time.time() - 5.0)
    server = GuardedYamLeaderPortalServer(
        robot,
        5001,
        max_robot_state_age_seconds=0.5,
        portal_module=_PortalModule,
    )
    with pytest.raises(RuntimeError, match="joint state is stale"):
        server.get_observations_guarded()

    robot._joint_state = SimpleNamespace(timestamp=_time.time())
    envelope = server.get_observations_guarded()
    assert envelope["robot_state_age_seconds"] is not None
    assert envelope["robot_state_age_seconds"] < 0.5


@pytest.mark.parametrize(
    ("client_cls", "error_label", "health"),
    [
        (YamArmClient, "YAM Portal", {"schema": FOLLOWER_PORTAL_SCHEMA, "dofs": 7}),
        (
            YamLeaderClient,
            "YAM leader Portal",
            {"schema": LEADER_PORTAL_SCHEMA, "dofs": 7, "read_only": True, "encoder_available": True},
        ),
    ],
)
def test_guarded_clients_share_connect_close_and_rpc_timeout_mechanics(client_cls, error_label, health):
    client = client_cls(1234, "localhost", 0.4, 0.25)
    portal_client = MagicMock()
    portal_client.connect.return_value = True
    portal_client.health.return_value.result.return_value = health

    client._connect_client(portal_client)
    assert client.is_connected
    portal_client.health.return_value.result.side_effect = TimeoutError("late")
    with pytest.raises(RuntimeError, match=rf"{error_label} RPC.*exceeded 0.400s"):
        client._rpc("health")
    client._close_client()

    portal_client.connect.assert_called_once_with(timeout=0.4)
    portal_client.health.return_value.result.assert_called_with(timeout=0.4)
    portal_client.close.assert_called_once_with()
    assert not client.is_connected


def test_portal_server_main_names_the_yam_hardware_group_when_i2rt_is_missing(monkeypatch):
    """The module must import without i2rt; only ``main`` may require it, by name."""
    from lerobot.scripts import yam_portal_server

    monkeypatch.setattr(yam_portal_server, "_i2rt_available", False)
    monkeypatch.setattr(
        "sys.argv", ["yam_portal_server", "--can-channel", "can_follower_r", "--port", "6001"]
    )
    with pytest.raises(ImportError, match="uv sync --extra yam --group yam-hardware"):
        yam_portal_server.main()


def test_guarded_leader_server_is_read_only_and_requires_encoder():
    server = GuardedYamLeaderPortalServer(_LeaderRobot(), 5001, portal_module=_PortalModule)

    assert set(server._server.bindings) == {"health", "get_observations_guarded"}
    assert server.health()["read_only"] is True
    envelope = server.get_observations_guarded()
    np.testing.assert_allclose(envelope["observation"]["gripper_pos"], [0.75])

    broken = _LeaderRobot()
    broken.motor_chain.get_same_bus_device_states = lambda: []
    broken_server = GuardedYamLeaderPortalServer(broken, 5002, portal_module=_PortalModule)
    with pytest.raises(RuntimeError, match="exactly one"):
        broken_server.health()


def test_dead_child_fails_supervisor_closed():
    processes = [_Process(101), _Process(102, returncode=7)]

    with pytest.raises(RuntimeError, match="can_follower_l.*status 7"):
        supervisor.assert_portals_alive(processes)


def test_post_launch_feedback_uses_motor_can_ids(monkeypatch):
    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(returncode=0, stderr="")

    monkeypatch.setattr(supervisor.subprocess, "run", fake_run)
    supervisor.verify_motor_feedback([_Process(101), _Process(102)])

    expected_filter = "011:7FF,012:7FF,013:7FF,014:7FF,015:7FF,016:7FF,017:7FF"
    assert [call[0] for call in calls] == [
        ["candump", "-n", "1", f"can_follower_r,{expected_filter}"],
        ["candump", "-n", "1", f"can_follower_l,{expected_filter}"],
    ]


def test_missing_feedback_fails_closed(monkeypatch):
    def timeout(*_args, **_kwargs):
        raise subprocess.TimeoutExpired("candump", 5)

    monkeypatch.setattr(supervisor.subprocess, "run", timeout)
    with pytest.raises(RuntimeError, match="No required YAM CAN feedback"):
        supervisor.verify_motor_feedback([_Process(101), _Process(102)])


def test_readiness_is_atomic_private_and_process_bound(tmp_path, monkeypatch):
    path = tmp_path / "ready.json"
    processes = [_Process(101), _Process(102)]
    monkeypatch.setattr(supervisor.os, "getpid", lambda: 99)
    monkeypatch.setattr(supervisor, "process_start_ticks", lambda pid: pid + 1000)

    result = supervisor.write_readiness(path, processes)

    assert result == path
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    payload = json.loads(path.read_text())
    assert payload["schema"] == READINESS_SCHEMA
    assert payload["mode"] == "follower_only"
    assert payload["portals"]["can_follower_r"] == {
        "pid": 101,
        "start_ticks": 1101,
        "port": 1234,
        "role": "follower",
        "side": "right",
    }
    assert payload["can_feedback_verified_after_launch"] == [
        "can_follower_r",
        "can_follower_l",
    ]
    assert not list(tmp_path.glob("*.tmp"))


def test_readiness_validator_checks_exact_contract_and_all_process_identities(tmp_path, monkeypatch):
    from lerobot.utils import yam_portal

    path = tmp_path / "ready.json"
    processes = [_Process(101), _Process(102)]
    monkeypatch.setattr(supervisor.os, "getpid", lambda: 99)
    monkeypatch.setattr(supervisor, "process_start_ticks", lambda pid: pid + 1000)
    supervisor.write_readiness(path, processes)
    monkeypatch.setattr(yam_portal, "process_start_ticks", lambda pid: pid + 1000)

    payload = validate_yam_portal_readiness(
        path,
        required_portals={
            "can_follower_l": {"role": "follower", "side": "left", "port": 1235},
            "can_follower_r": {"role": "follower", "side": "right", "port": 1234},
        },
        max_age_seconds=3.0,
    )
    assert payload["supervisor_pid"] == 99

    with pytest.raises(RuntimeError, match="does not match 9999"):
        validate_yam_portal_readiness(
            path,
            required_portals={"can_follower_l": {"role": "follower", "side": "left", "port": 9999}},
            max_age_seconds=3.0,
        )


def test_full_supervisor_verifies_leader_motor_and_encoder_feedback(monkeypatch):
    calls = []

    def fake_run(command, **_kwargs):
        calls.append(command)
        return SimpleNamespace(returncode=0, stderr="")

    monkeypatch.setattr(supervisor.subprocess, "run", fake_run)
    processes = [_Process(100 + index) for index in range(4)]
    supervisor.verify_motor_feedback(processes, supervisor.FULL_CONFIGS)

    assert [call[-1] for call in calls[-4:]] == [
        "can_leader_r,011:7FF,012:7FF,013:7FF,014:7FF,015:7FF,016:7FF",
        "can_leader_r,50E:7FF",
        "can_leader_l,011:7FF,012:7FF,013:7FF,014:7FF,015:7FF,016:7FF",
        "can_leader_l,50E:7FF",
    ]


def test_stop_portals_terminates_then_kills_only_timeout(monkeypatch):
    stopped = MagicMock()
    stopped.poll.return_value = 0
    running = MagicMock()
    running.poll.return_value = None
    running.wait.side_effect = [subprocess.TimeoutExpired("portal", 5), None]

    supervisor.stop_portals([stopped, running])

    stopped.terminate.assert_not_called()
    running.terminate.assert_called_once()
    running.kill.assert_called_once()
