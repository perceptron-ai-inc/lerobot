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

"""Contract tests for the official MolmoAct2 YAM async-inference command.

These tests pin the guide-shaped argv, handshake payload, HF-original config
builder, policy cache, rename-map injection, and client stop()/must-go
semantics. They are offline and GPU-free.
"""

from __future__ import annotations

import pickle
import threading

import pytest
import torch

pytest.importorskip("grpc")
pytest.importorskip("serial", reason="pyserial is required (install lerobot[hardware])")
pytest.importorskip("datasets", reason="datasets is required (install lerobot[dataset])")

from tests.async_inference._test_utils import (
    FakeGrpcContext,
    make_policy_setup_request,
    stop_robot_client_fast,
)

# The reference deployment launches async inference under `prlimit --stack=256M` because
# draccus' recursive decode of the full robot/camera registry overflows the
# default 8 MB stack. Tests reproduce that environment by running deep parses
# inside a big-stack thread.
_BIG_STACK_BYTES = 256 * 1024 * 1024

# Representative argv shaped like the MolmoAct2 YAM guide.
VALIDATED_ARGV = [
    "--server_address",
    "localhost:8000",
    "--robot.type",
    "bi_yam_follower",
    "--robot.left_arm_port",
    "1235",
    "--robot.right_arm_port",
    "1234",
    "--robot.cameras",
    '{"right": {"type": "intelrealsense", "serial_number_or_name": "100000000001", '
    '"width": 640, "height": 360, "fps": 30},'
    ' "left": {"type": "intelrealsense", "serial_number_or_name": "100000000002", '
    '"width": 640, "height": 360, "fps": 30},'
    ' "top": {"type": "intelrealsense", "serial_number_or_name": "100000000003", '
    '"width": 640, "height": 360, "fps": 30}}',
    "--task",
    "stack the cups",
    "--policy_type",
    "molmoact2",
    "--pretrained_name_or_path=",
    "--policy_device",
    "cuda",
    "--actions_per_chunk",
    "30",
    "--chunk_size_threshold",
    "0.0",
    "--aggregate_fn_name",
    "weighted_average",
    "--debug_visualize_queue_size",
    "False",
    "--policy_config_overrides",
    '["--checkpoint_path=/path/to/molmoact-yam-checkpoint",'
    ' "--norm_tag=yam_dual_molmoact2", "--inference_action_mode=continuous",'
    ' "--normalize_gripper=false"]',
]

VALIDATED_OVERRIDES = [
    "--checkpoint_path=/path/to/molmoact-yam-checkpoint",
    "--norm_tag=yam_dual_molmoact2",
    "--inference_action_mode=continuous",
    "--normalize_gripper=false",
]


def _run_in_big_stack_thread(fn):
    """Run `fn` in a thread with the deployment's 256 MB stack; re-raise errors."""
    result: dict = {}

    def runner():
        try:
            result["value"] = fn()
        except BaseException as e:  # noqa: BLE001
            result["error"] = e

    old_size = threading.stack_size(_BIG_STACK_BYTES)
    try:
        thread = threading.Thread(target=runner)
        thread.start()
        thread.join()
    finally:
        threading.stack_size(old_size)

    if "error" in result:
        raise result["error"]
    return result["value"]


# -----------------------------------------------------------------------------
# Handshake payload and configs
# -----------------------------------------------------------------------------


def test_validated_argv_roundtrip():
    """The representative guide argv must parse into RobotClientConfig."""
    import draccus

    import lerobot.async_inference.robot_client  # noqa: F401  (registers robot/camera configs)
    from lerobot.async_inference.configs import RobotClientConfig

    cfg = _run_in_big_stack_thread(lambda: draccus.parse(RobotClientConfig, args=VALIDATED_ARGV))

    assert cfg.pretrained_name_or_path == ""
    assert cfg.policy_type == "molmoact2"
    assert cfg.actions_per_chunk == 30
    assert cfg.chunk_size_threshold == 0.0
    assert cfg.aggregate_fn_name == "weighted_average"
    assert cfg.policy_config_overrides == VALIDATED_OVERRIDES
    assert cfg.rename_map == {}
    assert cfg.robot.type == "bi_yam_follower"
    assert cfg.robot.left_arm_port == 1235
    assert cfg.robot.right_arm_port == 1234
    assert set(cfg.robot.cameras) == {"right", "left", "top"}
    assert all(c.width == 640 and c.height == 360 and c.fps == 30 for c in cfg.robot.cameras.values())
    assert cfg.fps == 30
    assert abs(cfg.environment_dt - 1 / 30) < 1e-9
    # weighted_average must blend 0.3*old + 0.7*new
    assert cfg.aggregate_fn(torch.tensor(0.0), torch.tensor(1.0)).item() == pytest.approx(0.7)
    # to_dict must round-trip the deployment-relevant fields
    d = cfg.to_dict()
    assert d["policy_config_overrides"] == VALIDATED_OVERRIDES
    assert d["rename_map"] == {}


def test_config_requires_checkpoint_or_overrides():
    """Empty pretrained path is only valid when overrides are provided."""
    from lerobot.async_inference.configs import RobotClientConfig

    class _DummyRobotConfig:
        pass

    with pytest.raises(ValueError, match="policy_config_overrides"):
        RobotClientConfig(policy_type="molmoact2", robot=_DummyRobotConfig(), actions_per_chunk=30)

    cfg = RobotClientConfig(
        policy_type="molmoact2",
        robot=_DummyRobotConfig(),
        actions_per_chunk=30,
        policy_config_overrides=["--checkpoint_path=/x"],
    )
    assert cfg.pretrained_name_or_path == ""


def test_remote_policy_config_pickle_roundtrip():
    """The handshake payload fields must survive pickling (the wire format)."""
    from lerobot.async_inference.helpers import RemotePolicyConfig

    payload = RemotePolicyConfig(
        "molmoact2",
        "",
        {"observation.state": {"dtype": "float32", "shape": (14,), "names": None}},
        30,
        "cuda",
        rename_map={"a": "b"},
        policy_config_overrides=list(VALIDATED_OVERRIDES),
    )
    restored = pickle.loads(pickle.dumps(payload))  # nosec
    assert restored.policy_type == "molmoact2"
    assert restored.pretrained_name_or_path == ""
    assert restored.actions_per_chunk == 30
    assert restored.device == "cuda"
    assert restored.rename_map == {"a": "b"}
    assert restored.policy_config_overrides == VALIDATED_OVERRIDES


def test_map_robot_keys_includes_action_features():
    """The handshake features must include the robot's action feature (the
    server sizes HF-original output_features from it — suveene e0bf4a54)."""
    from lerobot.async_inference.helpers import map_robot_keys_to_lerobot_features
    from lerobot.robots import make_robot_from_config
    from lerobot.utils.constants import ACTION
    from tests.mocks.mock_robot import MockRobotConfig

    robot = make_robot_from_config(MockRobotConfig())
    features = map_robot_keys_to_lerobot_features(robot)

    assert ACTION in features, "handshake payload must carry the action feature"
    assert features[ACTION]["shape"] == (len(robot.action_features),)
    assert "observation.state" in features


def test_supported_policy_and_robot_lists():
    from lerobot.async_inference.constants import SUPPORTED_POLICIES, SUPPORTED_ROBOTS

    assert "molmoact2" in SUPPORTED_POLICIES
    assert "bi_yam_follower" in SUPPORTED_ROBOTS


# -----------------------------------------------------------------------------
# HF-original config builder and rename-map injection
# -----------------------------------------------------------------------------


def test_build_policy_config_from_overrides_molmoact2():
    """The validated override strings must build a MolmoAct2 config with the
    deployment contract applied (catches draccus-version drift on parsing)."""
    from lerobot.async_inference.policy_server import PolicyServer

    def build():
        return PolicyServer._build_policy_config_from_overrides("molmoact2", list(VALIDATED_OVERRIDES))

    config = _run_in_big_stack_thread(build)

    assert type(config).__name__ == "MolmoAct2Config"
    assert config.checkpoint_path == "/path/to/molmoact-yam-checkpoint"
    assert config.norm_tag == "yam_dual_molmoact2"
    assert config.action_mode == "both"
    assert config.inference_action_mode == "continuous"
    assert config.normalize_gripper is False
    # The guide-shaped YAM path keeps frame transforms disabled.
    assert getattr(config, "joint_signs", None) is None
    assert getattr(config, "joint_offsets", None) is None


def test_inject_rename_map_replaces_step_map():
    from lerobot.async_inference.policy_server import _inject_rename_map
    from lerobot.processor.rename_processor import RenameObservationsProcessorStep

    step = RenameObservationsProcessorStep(rename_map={"old": "stale"})

    class _Pipeline:
        steps = [object(), step]

    _inject_rename_map(_Pipeline(), {"observation.images.cam": "observation.images.top"})
    assert step.rename_map == {"observation.images.cam": "observation.images.top"}


# -----------------------------------------------------------------------------
# Policy cache across SendPolicyInstructions
# -----------------------------------------------------------------------------


_FAKE_CONTEXT = FakeGrpcContext("test-peer")


def _policy_setup_request(**kwargs):
    defaults = {
        "policy_type": "molmoact2",
        "pretrained_name_or_path": "",
        "lerobot_features": {
            "observation.state": {"dtype": "float32", "shape": (14,), "names": ["j"] * 14},
            "action": {"dtype": "float32", "shape": (14,), "names": ["j"] * 14},
        },
        "actions_per_chunk": 30,
        "device": "cpu",
        "rename_map": {},
        "policy_config_overrides": list(VALIDATED_OVERRIDES),
    }
    defaults.update(kwargs)
    return make_policy_setup_request(**defaults)


@pytest.fixture()
def policy_server(monkeypatch):
    """PolicyServer with the model-construction boundary faked out."""
    from lerobot.async_inference import policy_server as ps_module
    from lerobot.async_inference.configs import PolicyServerConfig
    from lerobot.processor.rename_processor import RenameObservationsProcessorStep

    server = ps_module.PolicyServer(PolicyServerConfig(host="localhost", port=9999))

    load_calls = {"count": 0}

    class _FakeConfig:
        device = "cpu"
        input_features = {}
        output_features = {}

    class _FakePolicy:
        config = _FakeConfig()

        def __init__(self, config):
            load_calls["count"] += 1
            self.config = config

        def to(self, *args, **kwargs):
            return self

    class _FakePipeline:
        def __init__(self):
            self.steps = [RenameObservationsProcessorStep(rename_map={})]

    monkeypatch.setattr(ps_module, "get_policy_class", lambda policy_type: _FakePolicy)
    monkeypatch.setattr(
        ps_module.PolicyServer,
        "_build_policy_config_from_overrides",
        staticmethod(lambda policy_type, overrides: _FakeConfig()),
    )
    monkeypatch.setattr(
        ps_module, "make_pre_post_processors", lambda *a, **k: (_FakePipeline(), _FakePipeline())
    )

    server._load_calls = load_calls
    yield server


def test_policy_cache_reuses_identical_signature(policy_server):
    """Identical (type, checkpoint, overrides, device) must not reload; the
    cache must survive Ready()/_reset_server(); rename_map changes must patch
    the cached preprocessor in place (suveene 3521a748 / 8c6ae2f5)."""
    ctx = _FAKE_CONTEXT

    policy_server.SendPolicyInstructions(_policy_setup_request(), ctx)
    assert policy_server._load_calls["count"] == 1

    # Same signature again -> cache hit, no reload
    policy_server.SendPolicyInstructions(_policy_setup_request(), ctx)
    assert policy_server._load_calls["count"] == 1

    # A new client session (Ready) must not evict the cache
    policy_server.Ready(None, ctx)
    policy_server.SendPolicyInstructions(_policy_setup_request(), ctx)
    assert policy_server._load_calls["count"] == 1

    # rename_map is excluded from the cache key but patched into the pipeline
    policy_server.SendPolicyInstructions(_policy_setup_request(rename_map={"x": "y"}), ctx)
    assert policy_server._load_calls["count"] == 1
    rename_step = policy_server.preprocessor.steps[0]
    assert rename_step.rename_map == {"x": "y"}


def test_policy_cache_reloads_on_signature_change(policy_server):
    ctx = _FAKE_CONTEXT

    policy_server.SendPolicyInstructions(_policy_setup_request(), ctx)
    assert policy_server._load_calls["count"] == 1

    # Different overrides -> different signature -> reload
    changed = VALIDATED_OVERRIDES[:-1] + ["--normalize_gripper=true"]
    policy_server.SendPolicyInstructions(_policy_setup_request(policy_config_overrides=changed), ctx)
    assert policy_server._load_calls["count"] == 2

    # Different device -> reload
    policy_server.SendPolicyInstructions(
        _policy_setup_request(policy_config_overrides=changed, device="cpu:1"), ctx
    )
    assert policy_server._load_calls["count"] == 3


def test_unsupported_policy_type_rejected(policy_server):
    with pytest.raises(ValueError, match="not supported"):
        policy_server.SendPolicyInstructions(_policy_setup_request(policy_type="not_a_policy"), _FAKE_CONTEXT)


# -----------------------------------------------------------------------------
# Client stop()/must-go semantics
# -----------------------------------------------------------------------------


@pytest.fixture()
def fast_client():
    """RobotClient on a mock robot, with the slow home move stubbed for speed."""
    from lerobot.async_inference.configs import RobotClientConfig
    from lerobot.async_inference.robot_client import RobotClient
    from tests.mocks.mock_robot import MockRobotConfig

    config = RobotClientConfig(
        robot=MockRobotConfig(),
        server_address="localhost:9999",
        policy_type="molmoact2",
        pretrained_name_or_path="",
        policy_config_overrides=["--checkpoint_path=/x"],
        actions_per_chunk=30,
        chunk_size_threshold=0.0,
    )
    client = RobotClient(config)
    yield client
    stop_robot_client_fast(client)


def test_stop_ordering(fast_client):
    """stop() must clear the queue, home the robot, then disconnect and close
    the channel — in that order (suveene 69a53fcc/9f2c9581)."""
    from lerobot.async_inference.helpers import TimedAction

    calls = []

    fast_client.action_queue.put(TimedAction(timestamp=0.0, timestep=0, action=torch.zeros(3)))
    fast_client.config.shutdown_home_position = dict.fromkeys(fast_client.robot.action_features, 0.0)
    fast_client._slow_move_to_position = lambda *a, **k: calls.append("home_move")

    original_disconnect = fast_client.robot.disconnect
    fast_client.robot.disconnect = lambda: (calls.append("disconnect"), original_disconnect())[1]
    original_close = fast_client.channel.close
    fast_client.channel.close = lambda: (calls.append("close"), original_close())[1]

    fast_client.stop()

    assert fast_client.action_queue.qsize() == 0, "queue must be cleared on stop"
    assert calls == ["home_move", "disconnect", "close"]
    assert not fast_client.running


def test_stop_without_robot_specific_home_disconnects_immediately(fast_client):
    calls = []
    fast_client._slow_move_to_position = lambda *a, **k: calls.append("home_move")
    original_disconnect = fast_client.robot.disconnect
    fast_client.robot.disconnect = lambda: (calls.append("disconnect"), original_disconnect())[1]

    fast_client.stop()

    assert calls == ["disconnect"]


def test_stop_rejects_home_target_with_wrong_robot_keys(fast_client):
    calls = []
    fast_client.config.shutdown_home_position = {"left_joint_0.pos": 0.0}
    fast_client._slow_move_to_position = lambda *a, **k: calls.append("home_move")
    original_disconnect = fast_client.robot.disconnect
    fast_client.robot.disconnect = lambda: (calls.append("disconnect"), original_disconnect())[1]

    fast_client.stop()

    assert calls == ["disconnect"]


def test_stop_skips_robot_when_already_disconnected(fast_client):
    calls = []
    fast_client._slow_move_to_position = lambda *a, **k: calls.append("home_move")
    fast_client._robot_disconnected = True

    fast_client.stop()

    assert calls == [], "home move and disconnect must be skipped"
    fast_client.robot.disconnect()  # clean up the still-connected mock


def test_action_chunk_size_startup_sentinel(fast_client):
    """Before the first chunk (action_chunk_size == -1) the client must be
    ready to stream observations: 0 / -1 == -0.0 <= 0.0. Porting a
    div-by-zero 'fix' here would change validated startup behavior."""
    assert fast_client.action_chunk_size == -1
    assert fast_client._ready_to_send_observation() is True
    fast_client._robot_disconnected = True
    fast_client.robot.disconnect()


def test_must_go_cycle(fast_client):
    """At most one must-go observation per chunk cycle: set -> consumed by the
    first empty-queue send -> re-armed only when a chunk arrives."""
    from lerobot.async_inference.helpers import TimedAction

    sent = []

    class _RecordingStub:
        def SendObservations(self, iterator):  # noqa: N802
            chunks = list(iterator)
            sent.append(pickle.loads(b"".join(c.data for c in chunks)))  # nosec
            return None

    fast_client.stub = _RecordingStub()

    # First observation with an empty queue consumes the must-go flag
    fast_client.control_loop_observation(task="t")
    assert sent[-1].must_go is True
    assert not fast_client.must_go.is_set()

    # Next observation (still empty queue) is not must-go
    fast_client.control_loop_observation(task="t")
    assert sent[-1].must_go is False

    # A chunk arrives: aggregation + must_go.set() (receive_actions body)
    actions = [TimedAction(timestamp=1.0 + i / 30, timestep=i, action=torch.zeros(3)) for i in range(30)]
    fast_client._aggregate_action_queues(actions, fast_client.config.aggregate_fn)
    fast_client.must_go.set()

    # Queue non-empty -> observation goes out with must_go False
    fast_client.control_loop_observation(task="t")
    assert sent[-1].must_go is False
    assert fast_client.must_go.is_set(), "flag survives until an empty-queue send"

    # Drain the queue; the next send is must-go again
    fast_client._clear_action_queue()
    fast_client.control_loop_observation(task="t")
    assert sent[-1].must_go is True
    assert not fast_client.must_go.is_set()

    fast_client._robot_disconnected = True
    fast_client.robot.disconnect()


def test_plain_follower_deployment_contract():
    """The official robot type must resolve to the plain Portal feature contract."""
    import draccus

    import lerobot.async_inference.robot_client  # noqa: F401  (registers robot/camera configs)
    from lerobot.async_inference.configs import RobotClientConfig
    from lerobot.async_inference.constants import SUPPORTED_ROBOTS
    from lerobot.async_inference.helpers import map_robot_keys_to_lerobot_features
    from lerobot.robots.bi_yam_follower_plain import BiYamFollowerPlain

    cfg = _run_in_big_stack_thread(lambda: draccus.parse(RobotClientConfig, args=VALIDATED_ARGV))
    assert cfg.robot.type == "bi_yam_follower"
    assert cfg.robot.record_torques is False
    assert "bi_yam_follower" in SUPPORTED_ROBOTS

    robot = BiYamFollowerPlain(cfg.robot)  # offline: no connect
    expected_keys = (
        [f"left_joint_{i}.pos" for i in range(6)]
        + ["left_gripper.pos"]
        + [f"right_joint_{i}.pos" for i in range(6)]
        + ["right_gripper.pos"]
    )
    assert list(robot.action_features) == expected_keys
    observation_keys = list(robot.observation_features)
    assert observation_keys[:14] == expected_keys
    assert set(observation_keys[14:]) == {"right", "left", "top"}
    assert all(robot.observation_features[cam] == (360, 640, 3) for cam in ("right", "left", "top"))
    assert not robot.is_connected

    features = map_robot_keys_to_lerobot_features(robot)
    assert features["observation.state"]["shape"] == (14,)
    assert features["action"]["shape"] == (14,)
    assert {key for key in features if key.startswith("observation.images.")} == {
        "observation.images.right",
        "observation.images.left",
        "observation.images.top",
    }
