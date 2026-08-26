"""Serving-stats single-source contract.

The serving split-brain this file guards against: at online single-step inference the
saved preprocessor pack step skips rendering and the policy-owned online render step
used to independently path-load ``config.native_stats_path`` -- which, for PEFT
finetunes, pointed at the BASE package's stats. These tests pin the fixed contract:

- the pack step hands its restored stats to the policy through the batch,
- the policy adopts them for the online renderer AND the settle branch,
- the online step receives the full config contract (gripper mask, fps check, ...),
- a package whose serialized stats disagree with its config path fails loudly at load,
- checkpoint saves export checkpoint-local stats with a relative config path.
"""

import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

pytest.importorskip("transformers", reason="transformers is required (install lerobot[perceptron_isaac])")

from lerobot.configs import FeatureType, PolicyFeature
from lerobot.lerobot_types import TransitionKey
from lerobot.policies.perceptron_isaac.configuration_perceptron_isaac import PerceptronIsaacConfig
from lerobot.policies.perceptron_isaac.isaac_stats import (
    IsaacNormalizationStats,
    IsaacStatsBlock,
    isaac_stats_value_differences,
    load_isaac_stats,
    normalize_isaac_actions,
    normalize_isaac_proprio,
)
from lerobot.policies.perceptron_isaac.modeling_perceptron_isaac import PerceptronIsaacPolicy
from lerobot.policies.perceptron_isaac.processor_perceptron_isaac import (
    PERCEPTRON_ISAAC_RENDER_META_KEY,
    PERCEPTRON_ISAAC_SERVING_STATS_KEY,
    PERCEPTRON_ISAAC_STREAM_KEY,
    PerceptronIsaacMharmonyPackProcessorStep,
    make_perceptron_isaac_pre_post_processors,
    make_perceptron_isaac_pre_post_processors_from_pretrained,
)
from lerobot.utils.constants import ACTION, OBS_STATE

_ACTION_NAMES = [f"joint_{i}.pos" for i in range(6)] + ["gripper.pos"]
_STATE_NAMES = [f"joint_{i}.pos" for i in range(7)] + ["gripper.pos"]


def _write_stats_json(path, *, shift: float = 0.0, action_dim: int = 7, proprio_dim: int = 8):
    payload = {
        "action": {
            "q01": [-1.0 + shift] * action_dim,
            "q99": [1.0 + shift] * action_dim,
            "min": [-1.5 + shift] * action_dim,
            "max": [1.5 + shift] * action_dim,
        },
        "proprio": {
            "q01": [-1.0 + shift] * proprio_dim,
            "q99": [1.0 + shift] * proprio_dim,
            "min": [-1.5 + shift] * proprio_dim,
            "max": [1.5 + shift] * proprio_dim,
        },
        "action_dim": action_dim,
        "proprio_dim": proprio_dim,
        "action_horizon": 30,
        "target_fps": 20.0,
        "schema": "flow_matching_stats_v1",
    }
    path.write_text(json.dumps(payload))


def _finetune_stats(*, shift: float = 0.3, action_dim: int = 7, proprio_dim: int = 8):
    """Stats with values distinct from `_write_stats_json(shift=0.0)` plus gripper masks."""
    action_mask = np.array([True] * (action_dim - 1) + [False])
    proprio_mask = np.array([True] * (proprio_dim - 1) + [False])
    return IsaacNormalizationStats(
        action=IsaacStatsBlock(
            q01=np.full(action_dim, -1.0 + shift, dtype=np.float32),
            q99=np.full(action_dim, 1.0 + shift, dtype=np.float32),
            mask=action_mask,
        ),
        proprio=IsaacStatsBlock(
            q01=np.full(proprio_dim, -1.0 + shift, dtype=np.float32),
            q99=np.full(proprio_dim, 1.0 + shift, dtype=np.float32),
            mask=proprio_mask,
        ),
        target_fps=20.0,
        action_horizon=30,
        schema="lerobot_dataset_quantiles",
        profile_id="synthetic/base#finetune_dataset",
        profile_scope="finetune_dataset",
        validation_status="dataset_stats",
    )


def _features():
    return {
        "observation.images.image": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 256, 256)),
        "observation.images.wrist_image": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 256, 256)),
        OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(8,)),
    }


def _config(tmp_path, **overrides):
    stats_path = tmp_path / "base_stats.json"
    if not stats_path.exists():
        _write_stats_json(stats_path)
    kwargs = {
        "device": "cpu",
        "native_stats_path": str(stats_path),
        "n_obs_steps": 1,
        "input_features": _features(),
        "output_features": {ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(7,))},
    }
    kwargs.update(overrides)
    return PerceptronIsaacConfig(**kwargs)


class _FakeStream:
    def to(self, *args, **kwargs):
        return self


class _FakeModel:
    def sample_action(self, stream, **kwargs):
        return torch.zeros(1, 30, 7)

    def parameters(self):
        return iter(())


def _single_frame_transition(task="pick"):
    return {
        TransitionKey.OBSERVATION: {
            "observation.images.image": np.zeros((1, 3, 16, 16), dtype=np.float32),
            "observation.images.wrist_image": np.zeros((1, 3, 16, 16), dtype=np.float32),
            OBS_STATE: np.zeros((1, 8), dtype=np.float32),
        },
        TransitionKey.ACTION: None,
        TransitionKey.REWARD: 0.0,
        TransitionKey.DONE: False,
        TransitionKey.TRUNCATED: False,
        TransitionKey.INFO: {},
        TransitionKey.COMPLEMENTARY_DATA: {"task": task},
    }


def test_pack_step_hands_restored_stats_to_the_policy_on_skip():
    step = PerceptronIsaacMharmonyPackProcessorStep(
        enabled=True,
        camera_order=["image", "wrist_image"],
        image_size=(16, 16),
        n_obs_steps=1,
        device="cpu",
        dtype="float32",
    )
    stats = _finetune_stats()
    step._stats = stats

    out = step(_single_frame_transition())

    complementary = out[TransitionKey.COMPLEMENTARY_DATA]
    assert complementary[PERCEPTRON_ISAAC_RENDER_META_KEY]["render_skipped"] is True
    assert complementary[PERCEPTRON_ISAAC_SERVING_STATS_KEY] is stats


def test_pack_step_skip_without_loaded_stats_attaches_nothing():
    step = PerceptronIsaacMharmonyPackProcessorStep(
        enabled=True,
        camera_order=["image", "wrist_image"],
        image_size=(16, 16),
        n_obs_steps=1,
        device="cpu",
        dtype="float32",
    )
    out = step(_single_frame_transition())
    complementary = out[TransitionKey.COMPLEMENTARY_DATA]
    assert PERCEPTRON_ISAAC_SERVING_STATS_KEY not in complementary


def test_policy_adopts_serialized_stats_and_online_step_matches_train_normalization(tmp_path, monkeypatch):
    """Issue-1 durable fix: the online renderer sources the checkpoint's serialized stats.

    Adoption must survive reset() (the async server rebuilds the online step every
    Ready()) and must beat the config-path stats, which point at the base package.
    """
    cfg = _config(tmp_path)
    policy = PerceptronIsaacPolicy(cfg)
    policy._isaac_model = _FakeModel()
    ft_stats = _finetune_stats()

    policy.predict_action_chunk(
        {
            PERCEPTRON_ISAAC_STREAM_KEY: _FakeStream(),
            PERCEPTRON_ISAAC_SERVING_STATS_KEY: ft_stats,
            "task": "pick",
        }
    )

    assert policy._stats is ft_stats
    step = policy._ensure_online_render_step()
    assert step._stats is ft_stats

    policy.reset()  # per-Ready() reset rebuilds the online step
    step_after_reset = policy._ensure_online_render_step()
    assert step_after_reset._stats is ft_stats

    probe = np.linspace(-0.5, 0.5, 8).astype(np.float32)
    online = normalize_isaac_proprio(probe, step_after_reset._stats.proprio)
    base = normalize_isaac_proprio(probe, load_isaac_stats(cfg.native_stats_path).proprio)
    assert not np.allclose(online, base)
    # The gripper dimension rides through raw (mask), unlike a maskless base load.
    assert online[-1] == probe[-1]

    # The adopted stats must reach the actual stream builder, not just sit on the
    # step: stub the renderer class and capture what _ensure_renderer constructs.
    import lerobot.policies.perceptron_isaac.processor_perceptron_isaac as processor_module

    captured = {}

    class _CapturingRenderer:
        def __init__(self, metadata, stats):
            captured["stats"] = stats

    monkeypatch.setattr(processor_module, "IsaacNativeMharmonyRenderer", _CapturingRenderer)
    step_after_reset._ensure_renderer()
    assert captured["stats"] is ft_stats


def test_settle_branch_normalizes_idle_actions_with_adopted_stats(tmp_path):
    cfg = _config(tmp_path, num_settle_steps=1)
    policy = PerceptronIsaacPolicy(cfg)
    ft_stats = _finetune_stats()

    batch = {
        "observation.images.image": torch.zeros(1, 3, 256, 256),
        "observation.images.wrist_image": torch.zeros(1, 3, 256, 256),
        OBS_STATE: torch.zeros(1, 8),
        PERCEPTRON_ISAAC_SERVING_STATS_KEY: ft_stats,
        "task": "pick",
    }
    action = policy.select_action(batch)

    idle = np.zeros((1, 7), dtype=np.float32)
    idle[:, -1] = float(cfg.settle_gripper)
    expected = normalize_isaac_actions(idle, ft_stats.action)
    np.testing.assert_allclose(action.numpy(), expected)
    base_stats = load_isaac_stats(cfg.native_stats_path)
    assert not np.allclose(action.numpy(), normalize_isaac_actions(idle, base_stats.action))


def test_online_render_step_receives_the_full_config_contract(tmp_path):
    cfg = _config(
        tmp_path,
        normalize_gripper=False,
        action_feature_names=list(_ACTION_NAMES),
        state_feature_names=list(_STATE_NAMES),
        target_fps=20.0,
        fast_clip_normalized_max=5.0,
        image_preprocessing="letterbox",
        normalization_profile_id="synthetic/base",
        normalization_profile_scope="base",
        normalization_validation_status="validated",
    )
    policy = PerceptronIsaacPolicy(cfg)
    step = policy._ensure_online_render_step()

    assert step.normalize_gripper is False
    assert step.action_feature_names == _ACTION_NAMES
    assert step.state_feature_names == _STATE_NAMES
    assert step.target_fps == 20.0
    assert step.fast_clip_normalized_max == 5.0
    assert step.image_preprocessing == "letterbox"
    assert step.normalization_profile_id == "synthetic/base"
    assert step.normalization_profile_scope == "base"
    assert step.normalization_validation_status == "validated"

    # The path-load leg now derives the gripper mask instead of silently normalizing
    # dimensions the checkpoint expects raw.
    state = step.state_dict()
    assert "proprio.mask" in state and not bool(state["proprio.mask"][-1])
    assert "action.mask" in state and not bool(state["action.mask"][-1])


def test_pretrained_load_fails_loud_when_config_path_stats_diverge(tmp_path):
    cfg = _config(tmp_path)
    preprocessor, postprocessor = make_perceptron_isaac_pre_post_processors(cfg)
    package = tmp_path / "package"
    package.mkdir()
    preprocessor.save_pretrained(package)
    postprocessor.save_pretrained(package)

    # Sanity: matching path stats load cleanly.
    make_perceptron_isaac_pre_post_processors_from_pretrained(cfg, str(package))

    # Repoint history: the config path now serves different quantiles than the
    # package serialized -- the deployed-finetune split-brain, reproduced.
    _write_stats_json(tmp_path / "base_stats.json", shift=0.3)
    with pytest.raises(RuntimeError, match="split-brained"):
        make_perceptron_isaac_pre_post_processors_from_pretrained(cfg, str(package))

    # Training RESUME is exempt: the trainer passes dataset_meta without
    # dataset_stats, the serialized pipeline stats are authoritative there, and the
    # resumed train_config still records the launch-time (base) stats path -- the
    # guard must not kill a healthy finetune resume.
    make_perceptron_isaac_pre_post_processors_from_pretrained(
        cfg, str(package), dataset_meta=SimpleNamespace(fps=20.0)
    )


def test_pretrained_load_fails_loud_on_internally_inconsistent_package(tmp_path):
    from safetensors.torch import save_file

    from lerobot.policies.perceptron_isaac.isaac_stats import isaac_stats_to_state_dict

    cfg = _config(tmp_path)
    preprocessor, postprocessor = make_perceptron_isaac_pre_post_processors(cfg)
    package = tmp_path / "package"
    package.mkdir()
    preprocessor.save_pretrained(package)
    postprocessor.save_pretrained(package)

    # Tamper the postprocessor's serialized stats so pre and post disagree.
    save_file(
        isaac_stats_to_state_dict(_finetune_stats(shift=0.5)),
        str(package / "policy_postprocessor_step_0_perceptron_isaac_action_unnormalize.safetensors"),
    )
    with pytest.raises(RuntimeError, match="internally inconsistent"):
        make_perceptron_isaac_pre_post_processors_from_pretrained(cfg, str(package))


def test_missing_config_path_stats_falls_back_to_serialized(tmp_path, caplog):
    """Portable packages keep loading after the source stats file disappears."""
    import logging

    cfg = _config(tmp_path)
    preprocessor, postprocessor = make_perceptron_isaac_pre_post_processors(cfg)
    package = tmp_path / "package"
    package.mkdir()
    preprocessor.save_pretrained(package)
    postprocessor.save_pretrained(package)

    (tmp_path / "base_stats.json").unlink()
    with caplog.at_level(logging.WARNING):
        pre, _post = make_perceptron_isaac_pre_post_processors_from_pretrained(cfg, str(package))
    assert any("does not exist" in record.message for record in caplog.records)
    pack = next(step for step in pre.steps if isinstance(step, PerceptronIsaacMharmonyPackProcessorStep))
    assert pack._stats is not None


def test_config_save_exports_checkpoint_local_stats_with_relative_path(tmp_path):
    cfg = _config(
        tmp_path,
        normalize_gripper=False,
        action_feature_names=list(_ACTION_NAMES),
        state_feature_names=list(_STATE_NAMES),
        normalization_profile_id="synthetic/base",
        normalization_profile_scope="base",
        normalization_validation_status="validated",
    )
    dataset_stats = {
        "action": {
            "q01": np.full(7, -0.7, dtype=np.float32),
            "q99": np.full(7, 0.7, dtype=np.float32),
            "min": np.full(7, -0.9, dtype=np.float32),
            "max": np.full(7, 0.9, dtype=np.float32),
        },
        "observation.state": {
            "q01": np.full(8, -0.7, dtype=np.float32),
            "q99": np.full(8, 0.7, dtype=np.float32),
            "min": np.full(8, -0.9, dtype=np.float32),
            "max": np.full(8, 0.9, dtype=np.float32),
        },
    }
    preprocessor, _postprocessor = make_perceptron_isaac_pre_post_processors(
        cfg, dataset_stats=dataset_stats, dataset_meta=SimpleNamespace(fps=20.0)
    )

    # Installed dataset stats sync the step-level JSON labels (issue 6: saved step
    # configs must stop advertising the base lineage).
    pack = next(
        step for step in preprocessor.steps if isinstance(step, PerceptronIsaacMharmonyPackProcessorStep)
    )
    pack_config = pack.get_config()
    assert pack_config["normalization_profile_scope"] == "finetune_dataset"
    assert pack_config["normalization_validation_status"] == "dataset_stats"
    assert pack_config["normalization_profile_id"] == "synthetic/base#finetune_dataset"

    exported_stats = cfg._export_native_stats
    assert exported_stats.profile_scope == "finetune_dataset"
    # De-collision: the finetune stats must not reuse the base profile_id verbatim.
    assert exported_stats.profile_id == "synthetic/base#finetune_dataset"

    save_dir = tmp_path / "checkpoint"
    save_dir.mkdir()
    original_path = cfg.native_stats_path
    cfg.save_pretrained(save_dir)

    saved = json.loads((save_dir / "config.json").read_text())
    assert saved["native_stats_path"] == "isaac_stats.json"
    assert saved["normalization_profile_scope"] == "finetune_dataset"
    assert saved["normalization_validation_status"] == "dataset_stats"
    # The live config is restored after the save (training continues on it).
    assert cfg.native_stats_path == original_path
    assert cfg.normalization_profile_scope == "base"

    reloaded = load_isaac_stats(save_dir / "isaac_stats.json")
    assert not isaac_stats_value_differences(reloaded, exported_stats)
    # The mask must be embedded: the loading step's normalize_gripper flag reads True
    # on every path-load, so a maskless JSON would silently normalize the grippers.
    assert reloaded.proprio.mask is not None and not bool(reloaded.proprio.mask[-1])

    # Issue-3 contract: PEFT loaders resolve the relative path against the ADAPTER
    # root through the public hook, and an already-absolute path is left untouched.
    fresh = _config(tmp_path, native_stats_path="isaac_stats.json")
    PerceptronIsaacPolicy.resolve_checkpoint_config_paths(fresh, save_dir)
    assert fresh.native_stats_path == str(save_dir / "isaac_stats.json")
    PerceptronIsaacPolicy.resolve_checkpoint_config_paths(fresh, tmp_path)
    assert fresh.native_stats_path == str(save_dir / "isaac_stats.json")


def _write_suite_tables(tmp_path) -> tuple:
    """A per-suite routing pair sitting where a training launch would leave it."""
    host = tmp_path / "training_host"
    host.mkdir(parents=True, exist_ok=True)
    stats = host / "suite_stats.json"
    stats.write_text(
        json.dumps(
            {
                "libero_spatial": {
                    "schema": "flow_matching_stats_v1",
                    "target_fps": 20.0,
                    "action_horizon": 30,
                    "action_dim": 7,
                    "proprio_dim": 8,
                    "action": {"q01": [-1.0] * 7, "q99": [1.0] * 7},
                    "proprio": {"q01": [0.916] * 8, "q99": [1.286] * 8},
                }
            }
        )
    )
    mapping = host / "suite_by_task_index.json"
    mapping.write_text(json.dumps({"30": "libero_spatial"}))
    return stats, mapping


def test_config_save_carries_per_suite_tables_checkpoint_local(tmp_path):
    """All eight path-typed fields resolve checkpoint-local, not six of them.

    `suite_stats_path` / `suite_by_task_index_path` arrive as absolute training-host
    paths from `--policy.suite_stats_path`, so without this every multi-suite LIBERO
    finetune ships two paths that are stale the moment the package moves.
    """
    suite_stats, suite_map = _write_suite_tables(tmp_path)
    cfg = _config(
        tmp_path,
        suite_stats_path=str(suite_stats),
        suite_by_task_index_path=str(suite_map),
    )
    save_dir = tmp_path / "checkpoint"
    save_dir.mkdir()
    cfg.save_pretrained(save_dir)

    saved = json.loads((save_dir / "config.json").read_text())
    assert saved["suite_stats_path"] == "isaac_suite_stats.json"
    assert saved["suite_by_task_index_path"] == "isaac_suite_by_task_index.json"
    # Carried, not merely renamed: the tables outlive the training host.
    assert json.loads((save_dir / "isaac_suite_stats.json").read_text()) == json.loads(
        suite_stats.read_text()
    )
    assert json.loads((save_dir / "isaac_suite_by_task_index.json").read_text()) == {"30": "libero_spatial"}
    # The live config is restored after the save (training continues on it).
    assert cfg.suite_stats_path == str(suite_stats)
    assert cfg.suite_by_task_index_path == str(suite_map)

    # And the loader resolves them back against the checkpoint root, like every other
    # sidecar -- including through the PEFT hook, which runs against the ADAPTER root.
    fresh = _config(
        tmp_path,
        native_stats_path="isaac_stats.json",
        suite_stats_path="isaac_suite_stats.json",
        suite_by_task_index_path="isaac_suite_by_task_index.json",
    )
    PerceptronIsaacPolicy.resolve_checkpoint_config_paths(fresh, save_dir)
    assert fresh.suite_stats_path == str(save_dir / "isaac_suite_stats.json")
    assert fresh.suite_by_task_index_path == str(save_dir / "isaac_suite_by_task_index.json")


def test_config_save_keeps_unreadable_suite_tables_declared(tmp_path):
    """A save must never fail, and must never silently drop the routing pair.

    Nulling the pair would disable per-suite routing on the next resume without a word;
    a stale path still fails loudly at the first render.
    """
    cfg = _config(
        tmp_path,
        suite_stats_path="/gone/suite_stats.json",
        suite_by_task_index_path="/gone/suite_by_task_index.json",
    )
    save_dir = tmp_path / "checkpoint"
    save_dir.mkdir()
    cfg.save_pretrained(save_dir)

    saved = json.loads((save_dir / "config.json").read_text())
    assert saved["suite_stats_path"] == "/gone/suite_stats.json"
    assert saved["suite_by_task_index_path"] == "/gone/suite_by_task_index.json"
    assert not (save_dir / "isaac_suite_stats.json").exists()
