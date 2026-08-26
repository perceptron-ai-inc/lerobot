"""Fast-path (policy-cache hit) contract for SendPolicyInstructions.

The saved-checkpoint cache key is (policy_type, pretrained_name_or_path,
cli_overrides, device, client_fps) and deliberately omits `lerobot_features`
(an unhashable dict) and
`rename_map`. Two defects lived on that path: the cache hit returned before the only
checkpoint-contract check (`validate_async_features`), and it injected the client's
rename map unconditionally, so a default `{}` on reconnect erased a checkpoint's
packaged rename map. These tests pin the fixed behaviour through the real
`PolicyServer.SendPolicyInstructions`, with only the model-construction boundary
faked out.
"""

import pytest

pytest.importorskip("grpc")

from lerobot.processor.rename_processor import RenameObservationsProcessorStep
from tests.async_inference._test_utils import FakeGrpcContext, make_policy_setup_request

_PACKAGED_RENAME_MAP = {"observation.images.image2": "observation.images.left"}
_FAKE_CONTEXT = FakeGrpcContext("fast-path-test")


def _features(state_dim=14, *, state_names=None, action_names=None):
    state_names = state_names or ["j"] * state_dim
    action_names = action_names or ["j"] * 14
    return {
        "observation.state": {"dtype": "float32", "shape": (state_dim,), "names": state_names},
        "action": {"dtype": "float32", "shape": (14,), "names": action_names},
    }


def _setup_request(**kwargs):
    defaults = {
        "policy_type": "perceptron_isaac",
        "pretrained_name_or_path": "fake/checkpoint",
        "lerobot_features": _features(),
        "actions_per_chunk": 30,
        "device": "cpu",
        "rename_map": {},
        "policy_config_overrides": [],
    }
    defaults.update(kwargs)
    return make_policy_setup_request(**defaults)


@pytest.fixture()
def server(monkeypatch):
    from lerobot.async_inference import policy_server as ps_module
    from lerobot.async_inference.configs import PolicyServerConfig

    server = ps_module.PolicyServer(PolicyServerConfig(host="localhost", port=9998))
    load_calls = {"count": 0}
    validate_calls: list[dict] = []

    class _FakeConfig:
        device = "cpu"
        use_peft = False
        input_features: dict = {}
        output_features: dict = {}

    class _FakePolicy:
        def __init__(self):
            self.config = _FakeConfig()

        def to(self, *args, **kwargs):
            return self

        def reset(self):
            return None

        def validate_async_features(self, lerobot_features):
            validate_calls.append(lerobot_features)
            state_shape = tuple(lerobot_features["observation.state"]["shape"])
            if state_shape != (14,):
                raise ValueError(
                    "perceptron_isaac checkpoint contract validation failed: "
                    f"observation.state shape {state_shape} != (14,)"
                )

    class _FakePipeline:
        def __init__(self, rename_map):
            self.steps = [RenameObservationsProcessorStep(rename_map=dict(rename_map))]

    def _fake_load_saved_policy(policy_class, policy_config, checkpoint):
        load_calls["count"] += 1
        return _FakePolicy()

    class _FakePreTrainedConfig:
        @staticmethod
        def from_pretrained(path, cli_overrides=None, **kwargs):
            return _FakeConfig()

    monkeypatch.setattr(ps_module, "PreTrainedConfig", _FakePreTrainedConfig)
    monkeypatch.setattr(ps_module, "_load_saved_policy", _fake_load_saved_policy)
    monkeypatch.setattr(ps_module, "get_policy_class", lambda policy_type: _FakePolicy)
    # The loaded pipeline carries a checkpoint-packaged rename map, as
    # perceptron_isaac's saved preprocessor does. Mutable so tests can model a
    # different checkpoint being loaded on a cache miss.
    packaged_source = {"current": dict(_PACKAGED_RENAME_MAP)}
    monkeypatch.setattr(
        ps_module,
        "make_pre_post_processors",
        lambda *args, **kwargs: (_FakePipeline(packaged_source["current"]), _FakePipeline({})),
    )

    server._load_calls = load_calls
    server._validate_calls = validate_calls
    server._packaged_source = packaged_source
    server._fake_config_type = _FakeConfig
    yield server


def test_fast_path_still_runs_the_checkpoint_contract_validation(server):
    """A cache hit must not skip validate_async_features: the handshake can change
    between sessions while the cache key stays identical."""
    ctx = _FAKE_CONTEXT

    server.SendPolicyInstructions(_setup_request(), ctx)
    assert server._load_calls["count"] == 1
    assert len(server._validate_calls) == 1

    # Identical signature -> cache hit, no reload, but validation STILL runs.
    server.SendPolicyInstructions(_setup_request(), ctx)
    assert server._load_calls["count"] == 1
    assert len(server._validate_calls) == 2

    # A reconnect with a different robot contract must fail at readiness, not
    # mid-episode with a live arm.
    with pytest.raises(ValueError, match="contract validation failed"):
        server.SendPolicyInstructions(_setup_request(lerobot_features=_features(state_dim=6)), ctx)
    assert server._load_calls["count"] == 1


def test_policy_setup_rejects_checkpoint_fps_before_loading_weights(server):
    server._fake_config_type.target_fps = server.config.fps + 1

    with pytest.raises(ValueError, match="checkpoint target_fps"):
        server.SendPolicyInstructions(_setup_request(fps=server.config.fps), _FAKE_CONTEXT)

    assert server._load_calls["count"] == 0


def test_policy_setup_rejects_three_frame_history_before_loading_weights(server):
    config = server._fake_config_type
    config.type = "perceptron_isaac"
    config.n_obs_steps = 3
    config.n_action_steps = 30
    config.num_settle_steps = 0

    with pytest.raises(NotImplementedError, match="n_obs_steps=3.*synchronous"):
        server.SendPolicyInstructions(_setup_request(), _FAKE_CONTEXT)

    assert server._load_calls["count"] == 0


def test_policy_setup_rejects_wrong_joint_order_on_handshake(server):
    joint_names = [f"joint_{index}.pos" for index in range(14)]
    config = server._fake_config_type
    config.strict_hardware_feature_contract = True
    config.state_feature_names = joint_names
    config.action_feature_names = joint_names
    config.camera_order = ()

    good_features = _features(state_names=joint_names, action_names=joint_names)
    server.SendPolicyInstructions(
        _setup_request(lerobot_features=good_features),
        _FAKE_CONTEXT,
    )
    assert server._load_calls["count"] == 1

    wrong_state_order = joint_names.copy()
    wrong_state_order[0], wrong_state_order[1] = wrong_state_order[1], wrong_state_order[0]
    with pytest.raises(ValueError, match="state order"):
        server.SendPolicyInstructions(
            _setup_request(
                lerobot_features=_features(
                    state_names=wrong_state_order,
                    action_names=joint_names,
                )
            ),
            _FAKE_CONTEXT,
        )

    # Saved-checkpoint handshakes validate changing robot features on the cache
    # path; the bad ordering must not trigger a model reload first.
    assert server._load_calls["count"] == 1


def test_fast_path_preserves_the_packaged_rename_map(server):
    ctx = _FAKE_CONTEXT

    server.SendPolicyInstructions(_setup_request(), ctx)
    assert server.preprocessor.steps[0].rename_map == _PACKAGED_RENAME_MAP
    assert server.rename_map == _PACKAGED_RENAME_MAP

    # Second session with the client default {} must NOT erase the packaged map
    # (this presented as a flaky reconnect on the SO100 ISAAC line).
    server.SendPolicyInstructions(_setup_request(), ctx)
    assert server.preprocessor.steps[0].rename_map == _PACKAGED_RENAME_MAP
    assert server.rename_map == _PACKAGED_RENAME_MAP

    # An explicit client map still patches the cached preprocessor in place.
    server.SendPolicyInstructions(_setup_request(rename_map={"x": "y"}), ctx)
    assert server.preprocessor.steps[0].rename_map == {"x": "y"}
    assert server.rename_map == {"x": "y"}

    # A later session sending the default {} must get the PACKAGED map back --
    # not the previous client's injected map (which would silently rename another
    # robot's cameras), and not an erased map.
    server.SendPolicyInstructions(_setup_request(), ctx)
    assert server.preprocessor.steps[0].rename_map == _PACKAGED_RENAME_MAP
    assert server.rename_map == _PACKAGED_RENAME_MAP


def test_hf_original_mode_reloads_when_the_robot_contract_changes(monkeypatch):
    """HF-original mode derives the policy config's features from the handshake, so a
    different robot must miss the cache and rebuild instead of reusing a policy shaped
    by the FIRST robot's features."""
    from lerobot.async_inference import policy_server as ps_module
    from lerobot.async_inference.configs import PolicyServerConfig

    server = ps_module.PolicyServer(PolicyServerConfig(host="localhost", port=9997))
    load_calls = {"count": 0}

    class _FakeConfig:
        device = "cpu"
        use_peft = False

        def __init__(self):
            self.input_features = {}
            self.output_features = {}

    class _FakePolicy:
        def __init__(self, config):
            load_calls["count"] += 1
            self.config = config

        def to(self, *args, **kwargs):
            return self

        def reset(self):
            return None

    monkeypatch.setattr(ps_module, "get_policy_class", lambda policy_type: _FakePolicy)
    monkeypatch.setattr(
        ps_module.PolicyServer,
        "_build_policy_config_from_overrides",
        staticmethod(lambda policy_type, overrides: _FakeConfig()),
    )
    monkeypatch.setattr(ps_module, "dataset_to_policy_features", lambda features: {})
    monkeypatch.setattr(
        ps_module,
        "make_pre_post_processors",
        lambda *args, **kwargs: (
            type("P", (), {"steps": [RenameObservationsProcessorStep(rename_map={})]})(),
            type("P", (), {"steps": []})(),
        ),
    )

    ctx = _FAKE_CONTEXT
    server.SendPolicyInstructions(_setup_request(pretrained_name_or_path=""), ctx)
    assert load_calls["count"] == 1

    # Same robot -> cache hit.
    server.SendPolicyInstructions(_setup_request(pretrained_name_or_path=""), ctx)
    assert load_calls["count"] == 1

    # Different robot contract -> features are part of the config here, so reload.
    server.SendPolicyInstructions(
        _setup_request(pretrained_name_or_path="", lerobot_features=_features(state_dim=6)), ctx
    )
    assert load_calls["count"] == 2


def test_hf_original_config_tempfile_is_cleaned_up(monkeypatch):
    from pathlib import Path

    from lerobot.async_inference import policy_server as ps_module

    seen = {}

    def fake_parse(config_class, config_path, args):
        seen["path"] = Path(config_path)
        assert seen["path"].is_file()
        return object()

    monkeypatch.setattr(ps_module.draccus, "parse", fake_parse)
    monkeypatch.setattr(ps_module.PreTrainedConfig, "get_choice_class", classmethod(lambda cls, t: object))

    ps_module.PolicyServer._build_policy_config_from_overrides("act", [])
    assert not seen["path"].exists()


def test_initial_load_with_a_client_map_still_snapshots_the_packaged_map(server):
    """The packaged-map snapshot must be taken BEFORE the client map is injected on the
    slow path; otherwise a later {} session restores the first client's map instead of
    the checkpoint's."""
    ctx = _FAKE_CONTEXT

    server.SendPolicyInstructions(_setup_request(rename_map={"x": "y"}), ctx)
    assert server.preprocessor.steps[0].rename_map == {"x": "y"}

    server.SendPolicyInstructions(_setup_request(), ctx)
    assert server.preprocessor.steps[0].rename_map == _PACKAGED_RENAME_MAP


def test_cache_miss_refreshes_the_packaged_map_snapshot(server):
    """After a reload of a different checkpoint (cache miss), {} sessions must restore
    THAT checkpoint's packaged map, not the previous checkpoint's."""
    ctx = _FAKE_CONTEXT

    server.SendPolicyInstructions(_setup_request(), ctx)
    assert server.preprocessor.steps[0].rename_map == _PACKAGED_RENAME_MAP

    new_packaged = {"observation.images.image2": "observation.images.top"}
    server._packaged_source["current"] = new_packaged
    # Different overrides -> cache miss -> full reload picks up the new package.
    server.SendPolicyInstructions(
        _setup_request(policy_config_overrides=["--num_flow_samples=1"], rename_map={"x": "y"}), ctx
    )
    assert server.preprocessor.steps[0].rename_map == {"x": "y"}

    # {} on the new signature restores the NEW packaged map.
    server.SendPolicyInstructions(_setup_request(policy_config_overrides=["--num_flow_samples=1"]), ctx)
    assert server.preprocessor.steps[0].rename_map == new_packaged
