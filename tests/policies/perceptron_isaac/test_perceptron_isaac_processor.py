from __future__ import annotations

import ast
import hashlib
import json
import logging
import shutil
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

# modeling_qwen35_vla imports transformers at module scope; the base fast_tests tier does not
# install it (see .github/workflows/fast_tests.yml).
pytest.importorskip("transformers", reason="transformers is required (install lerobot[perceptron_isaac])")

import lerobot.policies.perceptron_isaac.mharmony_native as mharmony_native
from lerobot.configs import FeatureType, PolicyFeature
from lerobot.lerobot_types import TransitionKey
from lerobot.policies import get_policy_class, make_policy_config, make_pre_post_processors
from lerobot.policies.perceptron_isaac.configuration_perceptron_isaac import PerceptronIsaacConfig
from lerobot.policies.perceptron_isaac.isaac_stats import (
    IsaacNormalizationStats,
    IsaacStatsBlock,
    apply_gripper_passthrough,
    isaac_stats_from_state_dict,
    isaac_stats_to_state_dict,
    load_isaac_stats,
    normalize_isaac_actions,
    normalize_isaac_proprio,
    unnormalize_isaac_actions,
)
from lerobot.policies.perceptron_isaac.mharmony_adapter import import_mharmony_module
from lerobot.policies.perceptron_isaac.mharmony_contract import SUPPORTED_MHARMONY_VERSION
from lerobot.policies.perceptron_isaac.mharmony_native import (
    IsaacActionOutlierError,
    IsaacMharmonyRenderMetadata,
    IsaacNativeMharmonyRenderer,
    build_isaac_mharmony_content_plan,
    load_native_render_metadata,
)
from lerobot.policies.perceptron_isaac.modeling_perceptron_isaac import PerceptronIsaacPolicy
from lerobot.policies.perceptron_isaac.modeling_qwen35_vla import (
    MolmoActExpertHead,
    Qwen35VLAConfig,
    load_sharded_safetensors_into_model,
)
from lerobot.policies.perceptron_isaac.processor_perceptron_isaac import (
    PERCEPTRON_ISAAC_ANCHOR_TIMESTAMPS_KEY,
    PERCEPTRON_ISAAC_KEPT_SAMPLE_INDICES_KEY,
    PERCEPTRON_ISAAC_RENDER_META_KEY,
    PERCEPTRON_ISAAC_STREAM_KEY,
    PerceptronIsaacActionFrameTransformStep,
    PerceptronIsaacActionUnnormalizeProcessorStep,
    PerceptronIsaacMharmonyPackProcessorStep,
    PerceptronIsaacRenderProcessorStep,
    PerceptronIsaacStateFrameTransformStep,
    PerceptronIsaacTrainingActionFrameTransformStep,
    make_perceptron_isaac_pre_post_processors,
    make_perceptron_isaac_pre_post_processors_from_pretrained,
    normalize_perceptron_isaac_task_text,
)
from lerobot.processor import PolicyProcessorPipeline, batch_to_transition, transition_to_batch
from lerobot.utils.constants import ACTION, OBS_STATE


class FakeTensorStream:
    def __init__(self, payload, sequence_length: int = 1):
        self.payload = payload
        self.sequence_length = sequence_length
        self.to_calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    @property
    def shape(self):
        return (1, self.sequence_length)

    def to(self, *args, **kwargs):
        self.to_calls.append((args, kwargs))
        return self


def test_perceptron_isaac_uses_v06_convention_based_factories(tmp_path):
    stats_path = tmp_path / "source_stats.json"
    _write_native_stats(stats_path)
    cfg = make_policy_config(
        "perceptron_isaac",
        device="cpu",
        n_obs_steps=1,
        native_stats_path=str(stats_path),
        input_features=_features(),
        output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(7,))},
    )

    assert isinstance(cfg, PerceptronIsaacConfig)
    assert get_policy_class("perceptron_isaac") is PerceptronIsaacPolicy

    preprocessor, postprocessor = make_pre_post_processors(cfg)
    preprocessor.save_pretrained(tmp_path)
    postprocessor.save_pretrained(tmp_path)
    loaded_preprocessor, loaded_postprocessor = make_pre_post_processors(
        cfg,
        pretrained_path=str(tmp_path),
    )

    assert any(
        isinstance(step, PerceptronIsaacMharmonyPackProcessorStep) for step in loaded_preprocessor.steps
    )
    assert any(
        isinstance(step, PerceptronIsaacActionUnnormalizeProcessorStep) for step in loaded_postprocessor.steps
    )


def test_peft_retained_processor_root_survives_adapter_source_replacement(tmp_path):
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    stats_path = adapter / "isaac_stats.json"
    _write_native_stats(stats_path)
    cfg = make_policy_config(
        "perceptron_isaac",
        device="cpu",
        n_obs_steps=1,
        native_stats_path=str(stats_path),
        input_features=_features(),
        output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(7,))},
    )
    preprocessor, postprocessor = make_pre_post_processors(cfg)
    preprocessor.save_pretrained(adapter)
    postprocessor.save_pretrained(adapter)
    assert list(adapter.glob("*.safetensors"))

    # Match the path records installed by policy loading: config sidecars belong
    # to the adapter even though the processor JSON currently contains absolutes.
    cfg.native_stats_path = "isaac_stats.json"
    PerceptronIsaacPolicy._resolve_checkpoint_local_paths(cfg, adapter)
    policy = PerceptronIsaacPolicy(cfg)
    policy.retain_pretrained_processor_assets(adapter)
    retained_root = Path(cfg._pretrained_processor_root)

    adapter.rename(tmp_path / "retired-adapter")
    loaded_preprocessor, loaded_postprocessor = make_pre_post_processors(
        cfg,
        pretrained_path=str(adapter),
    )

    assert retained_root.is_dir()
    assert list(retained_root.glob("*.safetensors"))
    assert any(
        isinstance(step, PerceptronIsaacMharmonyPackProcessorStep) for step in loaded_preprocessor.steps
    )
    assert any(
        isinstance(step, PerceptronIsaacActionUnnormalizeProcessorStep) for step in loaded_postprocessor.steps
    )


def test_per_suite_normalization_survives_loading_from_a_package(tmp_path):
    """--policy.suite_stats_path must reach the loaded step, not just the policy config.

    Regression: the pack step is rebuilt from the package's saved JSON, where these paths are
    absent. When they were not re-applied the step kept suite_stats_path=None, per-suite routing
    silently switched off, and every sample was normalized with the single set of quantiles baked
    into the package. A multi-suite run then trained entirely in one suite's frame and scored well
    on that suite alone -- indistinguishable from a policy that simply failed to learn the others.
    """
    stats_path = tmp_path / "source_stats.json"
    _write_native_stats(stats_path)

    suite_stats = tmp_path / "suite_stats.json"
    suite_map = tmp_path / "suite_by_task_index.json"
    suite_stats.write_text(json.dumps({"suite_a": {}, "suite_b": {}}))
    suite_map.write_text(json.dumps({"0": "suite_a", "1": "suite_b"}))

    base = make_policy_config(
        "perceptron_isaac",
        device="cpu",
        n_obs_steps=1,
        native_stats_path=str(stats_path),
        input_features=_features(),
        output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(7,))},
    )
    preprocessor, postprocessor = make_pre_post_processors(base)
    preprocessor.save_pretrained(tmp_path)
    postprocessor.save_pretrained(tmp_path)

    # The saved package carries no suite fields -- that is the condition being guarded.
    saved = json.loads((tmp_path / "policy_preprocessor.json").read_text())
    packed = next(s for s in saved["steps"] if "mharmony_pack" in str(s.get("registry_name")))
    assert packed["config"].get("suite_stats_path") is None

    base.suite_stats_path = str(suite_stats)
    base.suite_by_task_index_path = str(suite_map)
    loaded, _ = make_pre_post_processors(base, pretrained_path=str(tmp_path))

    pack = next(s for s in loaded.steps if isinstance(s, PerceptronIsaacMharmonyPackProcessorStep))
    assert pack.suite_stats_path == str(suite_stats)
    assert pack.suite_by_task_index_path == str(suite_map)


def test_from_pretrained_factory_rejects_half_configured_suite_normalization(tmp_path):
    """A lone --policy.suite_stats_path must fail loudly on the package path too.

    The fresh-construction path raises in the pack step's __post_init__; the loaded-package
    path re-applies the pair only when both are set, so a single path used to silently
    disable per-suite routing -- every sample normalized by the package's single quantile set.
    """
    config = make_policy_config(
        "perceptron_isaac",
        device="cpu",
        n_obs_steps=1,
        suite_stats_path=str(tmp_path / "suite_stats.json"),
        input_features=_features(),
        output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(7,))},
    )

    with pytest.raises(ValueError, match="suite_by_task_index_path together"):
        make_perceptron_isaac_pre_post_processors_from_pretrained(
            config, pretrained_path=str(tmp_path / "does_not_exist")
        )


def test_dataset_declared_fps_reaches_pack_step_on_both_factory_paths(tmp_path):
    stats_path = tmp_path / "source_stats.json"
    _write_native_stats(stats_path)
    cfg = make_policy_config(
        "perceptron_isaac",
        device="cpu",
        n_obs_steps=1,
        native_stats_path=str(stats_path),
        input_features=_features(),
        output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(7,))},
    )
    meta = SimpleNamespace(fps=30)

    fresh, _ = make_pre_post_processors(cfg, dataset_meta=meta)
    pack = next(s for s in fresh.steps if isinstance(s, PerceptronIsaacMharmonyPackProcessorStep))
    assert pack._dataset_declared_fps == 30.0

    preprocessor, postprocessor = make_pre_post_processors(cfg)
    preprocessor.save_pretrained(tmp_path)
    postprocessor.save_pretrained(tmp_path)
    loaded, _ = make_pre_post_processors(cfg, pretrained_path=str(tmp_path), dataset_meta=meta)
    pack = next(s for s in loaded.steps if isinstance(s, PerceptronIsaacMharmonyPackProcessorStep))
    assert pack._dataset_declared_fps == 30.0

    # Inference loads carry no dataset: the cross-check must stay disarmed.
    inference, _ = make_pre_post_processors(cfg, pretrained_path=str(tmp_path))
    pack = next(s for s in inference.steps if isinstance(s, PerceptronIsaacMharmonyPackProcessorStep))
    assert pack._dataset_declared_fps is None


def test_task_normalization_config_reaches_loaded_pack_step(tmp_path):
    stats_path = tmp_path / "source_stats.json"
    _write_native_stats(stats_path)
    cfg = make_policy_config(
        "perceptron_isaac",
        device="cpu",
        n_obs_steps=1,
        normalize_task_text=False,
        native_stats_path=str(stats_path),
        input_features=_features(),
        output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(7,))},
    )
    preprocessor, postprocessor = make_pre_post_processors(cfg)
    preprocessor.save_pretrained(tmp_path)
    postprocessor.save_pretrained(tmp_path)

    cfg.normalize_task_text = True
    loaded, _ = make_pre_post_processors(cfg, pretrained_path=str(tmp_path))
    pack = next(step for step in loaded.steps if isinstance(step, PerceptronIsaacMharmonyPackProcessorStep))

    assert pack.normalize_language is True


def test_sharded_loader_streams_and_validates_index(tmp_path):
    from safetensors.torch import save_file

    model = torch.nn.Sequential(torch.nn.Linear(2, 3), torch.nn.Linear(3, 1))
    expected = {key: tensor.detach().clone() + 1 for key, tensor in model.state_dict().items()}
    keys = sorted(expected)
    shard_names = ["model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"]
    save_file({key: expected[key] for key in keys[:2]}, tmp_path / shard_names[0])
    save_file({key: expected[key] for key in keys[2:]}, tmp_path / shard_names[1])
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {"total_size": sum(t.numel() * t.element_size() for t in expected.values())},
                "weight_map": {key: shard_names[0] if key in keys[:2] else shard_names[1] for key in keys},
            }
        )
    )

    missing, unexpected = load_sharded_safetensors_into_model(model, tmp_path)

    assert missing == []
    assert unexpected == []
    for key, tensor in model.state_dict().items():
        torch.testing.assert_close(tensor, expected[key])


def test_sharded_loader_rejects_index_mismatch(tmp_path):
    from safetensors.torch import save_file

    model = torch.nn.Linear(2, 1)
    save_file(model.state_dict(), tmp_path / "model-00001-of-00001.safetensors")
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {},
                "weight_map": {"weight": "model-00001-of-00001.safetensors"},
            }
        )
    )

    with pytest.raises(ValueError, match="index mismatch"):
        load_sharded_safetensors_into_model(model, tmp_path)


class FakeNativeRenderer:
    def __init__(self):
        self.build_calls = []
        self.build_training_calls = []
        self.collate_calls = []
        self.metadata = SimpleNamespace(target_fps=20.0, action_conditioning=False)
        self.stats = SimpleNamespace(target_fps=20.0)
        self.sequence_length = 1

    def build(self, **kwargs):
        self.build_calls.append(kwargs)
        return FakeTensorStream({"mode": "build", **kwargs}, self.sequence_length)

    def build_training(self, **kwargs):
        self.build_training_calls.append(kwargs)
        if float(np.asarray(kwargs["action_chunk"])[0, 0]) == -999.0:
            raise IsaacActionOutlierError("synthetic outlier")
        return FakeTensorStream({"mode": "train", **kwargs})

    def collate(self, streams):
        streams = list(streams)
        self.collate_calls.append(streams)
        return FakeTensorStream({"mode": "collate", "streams": streams})


class FakePackedStream:
    def __init__(self):
        self.to_calls = []

    def to(self, *args, **kwargs):
        self.to_calls.append((args, kwargs))
        return self


class FakeIsaacModel:
    def __init__(self, value: float = 0.0):
        self.config = SimpleNamespace(
            vision_config=SimpleNamespace(patch_size=16, spatial_merge_size=2, temporal_patch_size=2)
        )
        self.value = value
        self.sample_calls: list[dict[str, object]] = []

    def sample_action(self, stream, **kwargs):
        self.sample_calls.append({"stream": stream, **kwargs})
        action_dim = int(kwargs["action_dim"])
        horizon = int(kwargs["num_action_steps"])
        return torch.full((1, horizon, action_dim), float(self.value), dtype=torch.float32)


class FakeMixedPrecisionIsaacModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Keep the first parameter BF16 so the policy builds a BF16 input stream,
        # while the action expert represents deliberately retained FP32 master weights.
        self.compute_anchor = torch.nn.Parameter(torch.ones(1, dtype=torch.bfloat16), requires_grad=False)
        self.action_expert = torch.nn.Linear(1, 1, dtype=torch.float32)
        self.sample_dtype = None

    def sample_action(self, stream, **kwargs):
        horizon = int(kwargs["num_action_steps"])
        action_dim = int(kwargs["action_dim"])
        inputs = torch.ones((1, horizon, 1), dtype=torch.bfloat16)
        values = self.action_expert(inputs)
        self.sample_dtype = values.dtype
        return values.expand(1, horizon, action_dim)


class FakeFlowAveragingIsaacModel(FakeIsaacModel):
    def sample_action(self, stream, **kwargs):
        self.sample_calls.append({"stream": stream, **kwargs})
        action_dim = int(kwargs["action_dim"])
        horizon = int(kwargs["num_action_steps"])
        num_flow_samples = int(kwargs.get("num_flow_samples", 1))
        samples = torch.arange(1, num_flow_samples + 1, dtype=torch.float32).view(num_flow_samples, 1, 1, 1)
        samples = samples.expand(num_flow_samples, 1, horizon, action_dim)
        return samples.mean(dim=0)


class FakeRandomFlowIsaacModel(FakeIsaacModel):
    def sample_action(self, stream, **kwargs):
        self.sample_calls.append({"stream": stream, **kwargs})
        action_dim = int(kwargs["action_dim"])
        horizon = int(kwargs["num_action_steps"])
        num_flow_samples = int(kwargs.get("num_flow_samples", 1))
        samples = [torch.randn(1, horizon, action_dim) for _ in range(num_flow_samples)]
        return torch.stack(samples, dim=0).mean(dim=0)


class ZeroVelocityActionExpert:
    def forward_with_context(self, x, t_tensor, *, context):
        return torch.zeros_like(x)


def _features():
    return {
        "observation.images.image": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 256, 256)),
        "observation.images.image2": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 256, 256)),
        OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(8,)),
    }


def _install_fake_native_renderer(
    step: PerceptronIsaacRenderProcessorStep, monkeypatch: pytest.MonkeyPatch
) -> FakeNativeRenderer:
    renderer = FakeNativeRenderer()
    # Test-double injection: production currently infers this field as None.
    monkeypatch.setattr(step, "_stream_builder", renderer)
    return renderer


def _write_native_stats(
    path: Path,
    *,
    action_dim: int = 7,
    proprio_dim: int = 8,
    action_q01: float = -1.0,
    action_q99: float = 1.0,
    profile_id: str | None = None,
    profile_scope: str | None = None,
    validation_status: str | None = None,
) -> None:
    path.write_text(
        json.dumps(
            {
                "schema": "flow_matching_stats_v1",
                "target_fps": 20.0,
                "action_horizon": 30,
                "action_dim": action_dim,
                "proprio_dim": proprio_dim,
                "profile_id": profile_id,
                "profile_scope": profile_scope,
                "validation_status": validation_status,
                "action": {
                    "q01": [action_q01] * action_dim,
                    "q99": [action_q99] * action_dim,
                    "min": [action_q01] * action_dim,
                    "max": [action_q99] * action_dim,
                },
                "proprio": {
                    "q01": [-1.0] * proprio_dim,
                    "q99": [1.0] * proprio_dim,
                    "min": [-1.0] * proprio_dim,
                    "max": [1.0] * proprio_dim,
                },
            }
        )
    )


def test_molmoact_expert_averages_flow_samples_with_one_context_build():
    head = MolmoActExpertHead.__new__(MolmoActExpertHead)
    torch.nn.Module.__init__(head)
    head.num_inference_steps = 1
    head.action_dim = 1
    head.action_horizon = 1
    head.clean_at_0 = False
    head.action_expert = ZeroVelocityActionExpert()
    context_calls = []

    def _fake_context(*args, **kwargs):
        context_calls.append((args, kwargs))
        return object()

    head._build_single_context = _fake_context
    torch.manual_seed(123)
    actual = head.sample(
        torch.zeros(1, 2, 4),
        num_steps=1,
        num_action_steps=1,
        action_dim=1,
        num_flow_samples=4,
    )
    torch.manual_seed(123)
    expected = torch.stack([torch.randn(1, 1, 1) for _ in range(4)], dim=0).mean(dim=0)

    assert len(context_calls) == 1
    torch.testing.assert_close(actual, expected)


def test_qwen35_vla_config_forces_untied_embeddings():
    cfg = Qwen35VLAConfig(tie_word_embeddings=True)

    assert cfg.tie_word_embeddings is False


def test_task_text_normalization_for_isaac_prompts():
    assert normalize_perceptron_isaac_task_text(" Task: Pick up the mug!!! ") == "pick up the mug"
    assert normalize_perceptron_isaac_task_text("The task is to Open the drawer.") == "open the drawer"


def test_image_key_fallback_maps_libero_image2_to_wrist_image():
    step = PerceptronIsaacMharmonyPackProcessorStep(
        camera_order=["image", "wrist_image"],
        image_keys=["observation.images.image", "observation.images.wrist_image"],
        allow_image_key_fallback=True,
    )
    observation = {
        "observation.images.image": torch.zeros(1, 3, 16, 16),
        "observation.images.image2": torch.zeros(1, 3, 16, 16),
    }

    assert step._resolve_image_keys(observation) == [
        "observation.images.image",
        "observation.images.image2",
    ]


def test_image_extraction_accepts_nhwc_timestep_window():
    step = PerceptronIsaacMharmonyPackProcessorStep(
        camera_order=["image"],
        image_keys=["observation.images.image"],
        image_size=(16, 16),
        allow_image_key_fallback=False,
    )
    observation = {"observation.images.image": np.zeros((3, 16, 16, 3), dtype=np.uint8)}

    images = step._extract_image_tensor(observation, batch_size=1)

    assert images.shape == (1, 3, 1, 16, 16, 3)


def test_render_registry_configs_round_trip_without_schema_drift(tmp_path):
    render_config = {
        "enabled": False,
        "inference_backend": "native_mharmony",
        "config_toml_path": None,
        "stats_path": None,
        "camera_order": ["image", "wrist_image"],
        "image_size": [256, 256],
        "image_preprocessing": "stretch",
        "image_keys": [],
        "allow_image_key_fallback": True,
        "normalize_language": True,
        "n_obs_steps": 3,
        "chunk_size": 30,
        "objective": "flow",
        "patch_size": 16,
        "max_num_patches": None,
        "min_num_patches": None,
        "pixel_shuffle_scale": 2,
        "temporal_patch_size": 2,
        "device": "cpu",
        "dtype": "bfloat16",
        "stream_key": "perceptron_isaac_stream",
        "metadata_key": "perceptron_isaac_render_meta",
        "target_fps": None,
        "normalization_profile_id": None,
        "normalization_profile_scope": None,
        "normalization_validation_status": None,
        "suite_stats_path": None,
        "suite_by_task_index_path": None,
    }
    pack_config = {
        **render_config,
        "native_render_metadata_path": "/tmp/native_render.json",
        "native_stats_path": "/tmp/native_stats.json",
        "fast_processor_path": None,
        "fast_processor_tree_sha256": None,
        "mharmony_version": SUPPORTED_MHARMONY_VERSION,
        "action_dim": 7,
        "proprio_dim": 8,
        "vector_max_states": 128,
        "dataset_name": "libero",
        "robot_type": "generic",
        "action_conditioning": False,
        "action_conditioning_role": "user",
        "mistake_conditioning": False,
        "render_metadata": None,
        "train_clip_normalized_actions": True,
        "clip_normalized_max": 10.0,
        "fast_clip_normalized_max": 1.0,
        "train_skip_outlier_threshold": 20.0,
        "train_max_sequence_length": 4096,
        "normalize_gripper": True,
        "action_feature_names": None,
        "state_feature_names": None,
    }
    pipeline = PolicyProcessorPipeline(
        steps=[
            PerceptronIsaacRenderProcessorStep(**render_config),
            PerceptronIsaacMharmonyPackProcessorStep(**pack_config),
        ],
        name="test_processor",
        to_transition=batch_to_transition,
        to_output=transition_to_batch,
    )
    pipeline.save_pretrained(tmp_path, config_filename="processor.json")

    loaded = PolicyProcessorPipeline.from_pretrained(
        tmp_path,
        config_filename="processor.json",
        to_transition=batch_to_transition,
        to_output=transition_to_batch,
    )

    config = json.loads((tmp_path / "processor.json").read_text())
    assert config["steps"] == [
        {"registry_name": "perceptron_isaac_render", "config": render_config},
        {"registry_name": "perceptron_isaac_mharmony_pack", "config": pack_config},
    ]
    assert [type(step) for step in loaded.steps] == [
        PerceptronIsaacRenderProcessorStep,
        PerceptronIsaacMharmonyPackProcessorStep,
    ]
    assert [step.get_config() for step in loaded.steps] == [render_config, pack_config]


def test_native_backend_factory_uses_mharmony_pack_step(tmp_path):
    stats_path = tmp_path / "native_stats.json"
    _write_native_stats(stats_path)
    cfg = PerceptronIsaacConfig(
        device="cpu",
        native_render_metadata_path="/tmp/native_render.json",
        native_stats_path=str(stats_path),
        mharmony_version=SUPPORTED_MHARMONY_VERSION,
        input_features=_features(),
        output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(7,))},
    )
    preprocessor, postprocessor = make_perceptron_isaac_pre_post_processors(cfg)

    native_step = next(
        step for step in preprocessor.steps if isinstance(step, PerceptronIsaacMharmonyPackProcessorStep)
    )
    action_step = next(
        step
        for step in postprocessor.steps
        if isinstance(step, PerceptronIsaacActionUnnormalizeProcessorStep)
    )

    assert native_step.enabled is True
    assert native_step.inference_backend == "native_mharmony"
    assert native_step.native_render_metadata_path == "/tmp/native_render.json"
    assert native_step.native_stats_path == str(stats_path)
    assert native_step.mharmony_version == SUPPORTED_MHARMONY_VERSION
    assert action_step.stats_path == str(stats_path)
    assert action_step.gripper_binary_to_signed is False


def test_legacy_processor_mharmony_marker_is_upgraded() -> None:
    with pytest.warns(FutureWarning, match="genesis-in-tree"):
        step = PerceptronIsaacMharmonyPackProcessorStep(mharmony_version="genesis-in-tree")

    assert step.mharmony_version == SUPPORTED_MHARMONY_VERSION


def test_unsafe_normalization_profile_fails_closed_in_config_and_processor(tmp_path):
    with pytest.raises(ValueError, match="unsafe for closed-loop"):
        PerceptronIsaacConfig(
            device="cpu",
            normalization_validation_status="unsafe_closed_loop_ablation",
            input_features=_features(),
            output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(7,))},
        )

    stats_path = tmp_path / "unsafe_stats.json"
    _write_native_stats(
        stats_path,
        profile_id="ablation/rss180",
        profile_scope="output_denormalization_ablation",
        validation_status="unsafe_closed_loop_ablation",
    )
    step = PerceptronIsaacMharmonyPackProcessorStep(native_stats_path=str(stats_path))
    with pytest.raises(ValueError, match="unsafe for closed-loop"):
        step._ensure_renderer()


def test_serialized_processor_retains_normalization_safety_metadata(tmp_path):
    stats_path = tmp_path / "stats.json"
    _write_native_stats(
        stats_path,
        profile_id="production/yam",
        profile_scope="yam",
        validation_status="validated",
    )
    cfg = PerceptronIsaacConfig(
        device="cpu",
        native_stats_path=str(stats_path),
        normalization_profile_id="production/yam",
        normalization_profile_scope="yam",
        normalization_validation_status="validated",
        input_features=_features(),
        output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(7,))},
    )
    preprocessor, postprocessor = make_perceptron_isaac_pre_post_processors(cfg)
    preprocessor.save_pretrained(tmp_path)
    postprocessor.save_pretrained(tmp_path)

    pre_config = json.loads((tmp_path / "policy_preprocessor.json").read_text())
    pack_config = next(
        entry["config"]
        for entry in pre_config["steps"]
        if entry["registry_name"] == "perceptron_isaac_mharmony_pack"
    )
    post_config = json.loads((tmp_path / "policy_postprocessor.json").read_text())
    action_config = next(
        entry["config"]
        for entry in post_config["steps"]
        if entry["registry_name"] == "perceptron_isaac_action_unnormalize"
    )
    assert pack_config["normalization_validation_status"] == "validated"
    assert action_config["normalization_profile_id"] == "production/yam"

    preprocessor.save_pretrained(tmp_path, config_filename="preprocessor.json")
    config = json.loads((tmp_path / "preprocessor.json").read_text())
    native_entries = [
        entry for entry in config["steps"] if entry.get("registry_name") == "perceptron_isaac_mharmony_pack"
    ]
    assert len(native_entries) == 1
    assert native_entries[0]["config"]["enabled"] is True
    assert native_entries[0]["config"]["native_stats_path"] is None
    assert native_entries[0]["state_file"].endswith(".safetensors")

    postprocessor.save_pretrained(tmp_path, config_filename="postprocessor.json")
    post_config = json.loads((tmp_path / "postprocessor.json").read_text())
    action_entries = [
        entry
        for entry in post_config["steps"]
        if entry.get("registry_name") == "perceptron_isaac_action_unnormalize"
    ]
    assert len(action_entries) == 1
    assert action_entries[0]["config"]["stats_path"] is None
    assert action_entries[0]["state_file"].endswith(".safetensors")


def test_saved_processors_are_portable_without_source_stats(tmp_path):
    source_dir = tmp_path / "source"
    package_dir = tmp_path / "package"
    moved_dir = tmp_path / "moved" / "package"
    source_dir.mkdir()
    stats_path = source_dir / "stats.json"
    _write_native_stats(stats_path)
    cfg = PerceptronIsaacConfig(
        device="cpu",
        native_stats_path=str(stats_path),
        n_obs_steps=1,
        input_features=_features(),
        output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(7,))},
    )
    preprocessor, postprocessor = make_pre_post_processors(cfg)
    preprocessor.save_pretrained(package_dir)
    postprocessor.save_pretrained(package_dir)
    stats_path.unlink()
    moved_dir.parent.mkdir()
    shutil.copytree(package_dir, moved_dir)

    loaded_preprocessor, loaded_postprocessor = make_pre_post_processors(
        cfg,
        pretrained_path=str(moved_dir),
    )
    pack_step = next(
        step
        for step in loaded_preprocessor.steps
        if isinstance(step, PerceptronIsaacMharmonyPackProcessorStep)
    )
    action_step = next(
        step
        for step in loaded_postprocessor.steps
        if isinstance(step, PerceptronIsaacActionUnnormalizeProcessorStep)
    )

    assert pack_step.native_stats_path is None
    assert pack_step._stats.action_dim == 7
    assert action_step.stats_path is None
    assert action_step._stats.proprio_dim == 8


def test_preprocessor_returns_rendered_stream_as_policy_batch_key(monkeypatch):
    cfg = PerceptronIsaacConfig(
        device="cpu",
        native_stats_path="/tmp/stats.json",
        input_features=_features(),
        output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(7,))},
    )
    preprocessor, _ = make_perceptron_isaac_pre_post_processors(cfg)
    render_step = next(
        step for step in preprocessor.steps if isinstance(step, PerceptronIsaacMharmonyPackProcessorStep)
    )
    fake_renderer = _install_fake_native_renderer(render_step, monkeypatch)

    batch = {
        "observation.images.image": torch.zeros(3, 256, 256),
        "observation.images.image2": torch.ones(3, 256, 256),
        OBS_STATE: torch.zeros(8),
        "task": "Instruction: Pick Up The Mug.",
        "timestamp": torch.tensor(0.0),
    }

    out = preprocessor(batch)

    assert PERCEPTRON_ISAAC_STREAM_KEY in out
    assert PERCEPTRON_ISAAC_RENDER_META_KEY in out
    assert out[PERCEPTRON_ISAAC_STREAM_KEY].payload["prompt"] == "pick up the mug"
    assert fake_renderer.build_calls[-1]["patch_size"] == 16


def test_batch_to_transition_passes_frame_index_and_timestamp():
    transition = batch_to_transition(
        {
            "observation.images.image": torch.zeros(3, 256, 256),
            OBS_STATE: torch.zeros(8),
            "frame_index": torch.tensor(2),
            "timestamp": torch.tensor(0.2),
        }
    )

    complementary = transition[TransitionKey.COMPLEMENTARY_DATA]
    # The frame counter has to reach the processors: converted LIBERO datasets misdeclare their
    # fps, so model time is derived from frame_index rather than the stored timestamp.
    assert float(complementary["frame_index"]) == pytest.approx(2)
    assert float(complementary["timestamp"]) == pytest.approx(0.2)


def test_native_pack_anchor_uses_frame_index_over_misdeclared_dataset_timestamp(monkeypatch):
    """A dataset that misdeclares its fps must not shift training-time model time.

    ``allenai/MolmoAct2-LIBERO-Dataset`` and ``lerobot/libero_spatial_image`` both declare
    ``fps=10`` for 20 Hz robosuite demos, so their stored ``timestamp`` is twice the real
    elapsed time. Online rollout clocks itself at ``frame_index / target_fps``; if training
    trusted the stored timestamp the two would disagree by 2x.
    """
    step = PerceptronIsaacMharmonyPackProcessorStep(
        enabled=True,
        camera_order=["image", "wrist_image"],
        image_size=(16, 16),
        n_obs_steps=1,
        device="cpu",
        dtype="float32",
    )
    fake_renderer = _install_fake_native_renderer(step, monkeypatch)
    assert fake_renderer.metadata.target_fps == 20.0

    transition = {
        TransitionKey.OBSERVATION: {
            "observation.images.image": torch.zeros(1, 1, 3, 16, 16),
            "observation.images.image2": torch.ones(1, 1, 3, 16, 16),
            OBS_STATE: torch.zeros(1, 1, 8),
        },
        TransitionKey.ACTION: None,
        TransitionKey.REWARD: 0.0,
        TransitionKey.DONE: False,
        TransitionKey.TRUNCATED: False,
        TransitionKey.INFO: {},
        TransitionKey.COMPLEMENTARY_DATA: {
            "task": "Task: Pick Up The Mug.",
            "frame_index": torch.tensor([4]),
            # What a 10-fps-declared dataset stores for frame 4 of a 20 Hz recording.
            "timestamp": torch.tensor([0.4]),
        },
    }

    out = step(transition)
    metadata = out[TransitionKey.COMPLEMENTARY_DATA][PERCEPTRON_ISAAC_RENDER_META_KEY]

    # 4 / 20 Hz = 0.2 s of real elapsed time, not the 0.4 s the dataset claims.
    np.testing.assert_allclose(metadata["anchor_timestamp_seconds"], [0.2])
    np.testing.assert_allclose([fake_renderer.build_calls[-1]["anchor_timestamp_seconds"]], [0.2])


def test_native_pack_build_path_passes_external_timestamp_from_explicit_window(monkeypatch):
    step = PerceptronIsaacMharmonyPackProcessorStep(
        enabled=True,
        camera_order=["image", "wrist_image"],
        image_size=(16, 16),
        n_obs_steps=3,
        device="cpu",
        dtype="float32",
    )
    fake_renderer = _install_fake_native_renderer(step, monkeypatch)

    transition = {
        TransitionKey.OBSERVATION: {
            "observation.images.image": torch.stack(
                [torch.full((3, 16, 16), idx / 10.0) for idx in range(3)], dim=0
            ).unsqueeze(0),
            "observation.images.image2": torch.stack(
                [torch.full((3, 16, 16), (idx + 1) / 10.0) for idx in range(3)], dim=0
            ).unsqueeze(0),
            OBS_STATE: torch.stack([torch.full((8,), float(idx)) for idx in range(3)], dim=0).unsqueeze(0),
        },
        TransitionKey.ACTION: None,
        TransitionKey.REWARD: 0.0,
        TransitionKey.DONE: False,
        TransitionKey.TRUNCATED: False,
        TransitionKey.INFO: {},
        TransitionKey.COMPLEMENTARY_DATA: {
            "task": "Task: Pick Up The Mug.",
            "timestamp": torch.tensor([0.2]),
        },
    }
    out = step(transition)

    metadata = out[TransitionKey.COMPLEMENTARY_DATA][PERCEPTRON_ISAAC_RENDER_META_KEY]
    stream = out[TransitionKey.COMPLEMENTARY_DATA][PERCEPTRON_ISAAC_STREAM_KEY]

    assert metadata["task"] == ["pick up the mug"]
    assert metadata["image_keys"] == ["observation.images.image", "observation.images.image2"]
    np.testing.assert_allclose(metadata["anchor_timestamp_seconds"], [0.2])
    np.testing.assert_allclose([fake_renderer.build_calls[-1]["anchor_timestamp_seconds"]], [0.2])
    assert metadata["n_obs_steps"] == 3
    assert stream.payload["prompt"] == "pick up the mug"
    assert len(stream.payload["observation_window"]) == 3
    assert fake_renderer.build_calls[-1]["patch_size"] == 16

    step.reset()
    assert not hasattr(step, "_image_queue")
    assert not hasattr(step, "_state_queue")


def test_native_pack_skips_live_single_step_without_anchor(monkeypatch):
    step = PerceptronIsaacMharmonyPackProcessorStep(
        enabled=True,
        camera_order=["image", "wrist_image"],
        image_size=(16, 16),
        n_obs_steps=3,
        device="cpu",
        dtype="float32",
    )
    fake_renderer = _install_fake_native_renderer(step, monkeypatch)

    out = step(
        {
            TransitionKey.OBSERVATION: {
                "observation.images.image": torch.zeros(1, 3, 16, 16),
                "observation.images.image2": torch.ones(1, 3, 16, 16),
                OBS_STATE: torch.zeros(1, 8),
            },
            TransitionKey.ACTION: None,
            TransitionKey.REWARD: 0.0,
            TransitionKey.DONE: False,
            TransitionKey.TRUNCATED: False,
            TransitionKey.INFO: {},
            TransitionKey.COMPLEMENTARY_DATA: {"task": "Task: Pick Up The Mug."},
        }
    )
    metadata = out[TransitionKey.COMPLEMENTARY_DATA][PERCEPTRON_ISAAC_RENDER_META_KEY]

    assert PERCEPTRON_ISAAC_STREAM_KEY not in out[TransitionKey.COMPLEMENTARY_DATA]
    assert metadata["render_skipped"] is True
    assert metadata["skip_reason"] == "online_single_step_without_anchor"
    assert metadata["anchor_timestamp_seconds"] is None
    assert fake_renderer.build_calls == []


def test_native_pack_rejects_multi_environment_batch(monkeypatch):
    step = PerceptronIsaacMharmonyPackProcessorStep(
        enabled=True,
        camera_order=["image", "wrist_image"],
        image_size=(16, 16),
        n_obs_steps=3,
        device="cpu",
        dtype="float32",
    )
    _install_fake_native_renderer(step, monkeypatch)

    transition = {
        TransitionKey.OBSERVATION: {
            "observation.images.image": torch.zeros(2, 3, 16, 16),
            "observation.images.image2": torch.ones(2, 3, 16, 16),
            OBS_STATE: torch.zeros(2, 8),
        },
        TransitionKey.ACTION: None,
        TransitionKey.REWARD: 0.0,
        TransitionKey.DONE: False,
        TransitionKey.TRUNCATED: False,
        TransitionKey.INFO: {},
        TransitionKey.COMPLEMENTARY_DATA: {"task": ["Pick up the mug", "Open the drawer"]},
    }

    with pytest.raises(NotImplementedError, match="batch_size=1.*batch_size=2"):
        step(transition)


def test_native_policy_online_rollout_uses_policy_clock(monkeypatch, tmp_path):
    stats_path = tmp_path / "stats.json"
    _write_native_stats(stats_path)

    def _cfg(**kwargs):
        return PerceptronIsaacConfig(
            device="cpu",
            native_stats_path=str(stats_path),
            n_obs_steps=3,
            n_action_steps=12,
            chunk_size=30,
            input_features=_features(),
            output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(7,))},
            **kwargs,
        )

    def _batch(idx: int):
        return {
            "observation.images.image": torch.full((3, 256, 256), idx / 10.0),
            "observation.images.wrist_image": torch.full((3, 256, 256), (idx + 1) / 10.0),
            OBS_STATE: torch.full((8,), float(idx)),
            "task": "Task: Pick Up The Mug.",
        }

    policy = PerceptronIsaacPolicy(_cfg(num_settle_steps=0))
    policy._isaac_model = FakeIsaacModel()
    fake_renderer = _install_fake_native_renderer(policy._ensure_online_render_step(), monkeypatch)

    policy.select_action(_batch(0))
    # Genesis floors the anchor at (k-1)/fps so the k rendered observation timestamps stay
    # distinct; with n_obs_steps=3 at 20 fps that is 0.1s, not 0.0s.
    np.testing.assert_allclose([fake_renderer.build_calls[-1]["anchor_timestamp_seconds"]], [0.1])
    first_window = fake_renderer.build_calls[-1]["observation_window"]
    assert len(first_window) == 3
    assert [float(frame["proprio"][0]) for frame in first_window] == [0.0, 0.0, 0.0]

    for idx in range(1, 13):
        policy.select_action(_batch(idx))

    anchors = [call["anchor_timestamp_seconds"] for call in fake_renderer.build_calls]
    np.testing.assert_allclose(anchors, [0.1, 0.6])
    assert fake_renderer.build_calls[-1]["observation_window"][-1]["proprio"][0] == 12.0

    policy.reset()
    assert len(policy._online_image_queue) == 0
    assert len(policy._online_state_queue) == 0
    assert policy._online_render_step is None
    assert policy._frame_index == -1
    assert policy._last_external_timestamp_seconds is None

    settled_policy = PerceptronIsaacPolicy(_cfg(num_settle_steps=10))
    settled_policy._isaac_model = FakeIsaacModel()
    settled_renderer = _install_fake_native_renderer(settled_policy._ensure_online_render_step(), monkeypatch)
    for idx in range(11):
        settled_policy.select_action(_batch(idx))

    assert len(settled_renderer.build_calls) == 1
    # Settling happens inside the environment reset for Genesis, and its model clock counts
    # only executed post-settle steps, so the first chunk anchors at (k-1)/fps regardless of
    # num_settle_steps rather than carrying a +num_settle_steps/fps offset.
    np.testing.assert_allclose([settled_renderer.build_calls[-1]["anchor_timestamp_seconds"]], [0.1])
    settled_window = settled_renderer.build_calls[-1]["observation_window"]
    assert [float(frame["proprio"][0]) for frame in settled_window] == [10.0, 10.0, 10.0]
    assert settled_policy._settle_index == 10
    assert settled_policy._frame_index == 0


def test_online_renderer_inherits_conditioning_contract(tmp_path):
    stats_path = tmp_path / "stats.json"
    _write_native_stats(stats_path)
    config = PerceptronIsaacConfig(
        device="cpu",
        native_stats_path=str(stats_path),
        action_conditioning=True,
        action_conditioning_role="user",
        mistake_conditioning=True,
        input_features=_features(),
        output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(7,))},
    )

    step = PerceptronIsaacPolicy(config)._ensure_online_render_step()

    assert step.action_conditioning is True
    assert step.action_conditioning_role == "user"
    assert step.mistake_conditioning is True
    step._ensure_renderer()
    assert step._stream_builder.metadata.action_conditioning is True
    assert step._stream_builder.metadata.action_conditioning_role == "user"
    assert step._stream_builder.metadata.mistake_conditioning is True


def test_native_policy_requires_and_resets_executed_action_history(monkeypatch, tmp_path):
    stats_path = tmp_path / "stats.json"
    _write_native_stats(stats_path)
    config = PerceptronIsaacConfig(
        device="cpu",
        native_stats_path=str(stats_path),
        n_obs_steps=3,
        n_action_steps=1,
        chunk_size=30,
        action_conditioning=True,
        clip_normalized_max=10.0,
        fast_clip_normalized_max=1.0,
        input_features=_features(),
        output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(7,))},
    )

    def batch(index: int) -> dict[str, object]:
        return {
            "observation.images.image": torch.full((3, 256, 256), index / 10.0),
            "observation.images.wrist_image": torch.full((3, 256, 256), (index + 1) / 10.0),
            OBS_STATE: torch.full((8,), float(index)),
            "task": "pick up the mug",
        }

    policy = PerceptronIsaacPolicy(config)
    policy._isaac_model = FakeIsaacModel()
    render_step = policy._ensure_online_render_step()
    renderer = _install_fake_native_renderer(render_step, monkeypatch)
    renderer.metadata.action_conditioning = True
    render_step._fast_processor = object()

    policy.select_action(batch(0))
    assert [frame.get("previous_action") for frame in renderer.build_calls[-1]["observation_window"]] == [
        None,
        None,
        None,
    ]

    with pytest.raises(RuntimeError, match="record_executed_action"):
        policy.select_action(batch(1))
    assert policy._frame_index == 0

    with pytest.raises(ValueError, match="finite"):
        policy.record_executed_action(np.full(7, np.nan, dtype=np.float32))

    executed_action = np.full(7, 2.0, dtype=np.float32)
    policy.record_executed_action(executed_action)
    policy.select_action(batch(1))
    expected = normalize_isaac_actions(executed_action[None, :], load_isaac_stats(stats_path).action)[0]
    expected = np.clip(expected, -config.fast_clip_normalized_max, config.fast_clip_normalized_max)
    np.testing.assert_allclose(
        renderer.build_calls[-1]["observation_window"][-1]["previous_action"],
        expected,
    )

    policy.reset()
    assert list(policy._online_action_queue) == []
    assert policy._pending_executed_action is None
    first_step = policy._ensure_online_render_step()
    first_renderer = _install_fake_native_renderer(first_step, monkeypatch)
    first_renderer.metadata.action_conditioning = True
    first_step._fast_processor = object()
    policy.select_action(batch(0))


def test_native_policy_external_timestamp_is_monotonic_and_resettable(monkeypatch, tmp_path):
    stats_path = tmp_path / "stats.json"
    _write_native_stats(stats_path)
    cfg = PerceptronIsaacConfig(
        device="cpu",
        native_stats_path=str(stats_path),
        n_obs_steps=1,
        n_action_steps=1,
        chunk_size=30,
        num_settle_steps=0,
        input_features=_features(),
        output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(7,))},
    )
    policy = PerceptronIsaacPolicy(cfg)
    policy._isaac_model = FakeIsaacModel()
    fake_renderer = _install_fake_native_renderer(policy._ensure_online_render_step(), monkeypatch)

    first = {
        "observation.images.image": torch.zeros(3, 256, 256),
        "observation.images.wrist_image": torch.zeros(3, 256, 256),
        OBS_STATE: torch.zeros(8),
        "task": "pick up the mug",
        "timestamp_seconds": torch.tensor(0.25),
    }
    policy.select_action(first)
    assert fake_renderer.build_calls[-1]["anchor_timestamp_seconds"] == pytest.approx(0.25)

    with pytest.raises(ValueError, match="monotonic"):
        policy.select_action({**first, "timestamp_seconds": torch.tensor(0.2)})

    policy.reset()
    policy._isaac_model = FakeIsaacModel()
    reset_renderer = _install_fake_native_renderer(policy._ensure_online_render_step(), monkeypatch)
    policy.select_action({**first, "timestamp_seconds": torch.tensor(0.0)})
    assert reset_renderer.build_calls[-1]["anchor_timestamp_seconds"] == pytest.approx(0.0)


def test_native_pack_accepts_policy_owned_explicit_anchor(monkeypatch):
    step = PerceptronIsaacMharmonyPackProcessorStep(
        enabled=True,
        camera_order=["image", "wrist_image"],
        image_size=(16, 16),
        n_obs_steps=3,
        device="cpu",
        dtype="float32",
    )
    fake_renderer = _install_fake_native_renderer(step, monkeypatch)
    transition = {
        TransitionKey.OBSERVATION: {
            "observation.images.image": torch.zeros(1, 3, 3, 16, 16),
            "observation.images.image2": torch.ones(1, 3, 3, 16, 16),
            OBS_STATE: torch.zeros(1, 3, 8),
        },
        TransitionKey.ACTION: None,
        TransitionKey.REWARD: 0.0,
        TransitionKey.DONE: False,
        TransitionKey.TRUNCATED: False,
        TransitionKey.INFO: {},
        TransitionKey.COMPLEMENTARY_DATA: {
            "task": "Task: Pick Up The Mug.",
            PERCEPTRON_ISAAC_ANCHOR_TIMESTAMPS_KEY: [0.6],
        },
    }

    out = step(transition)
    metadata = out[TransitionKey.COMPLEMENTARY_DATA][PERCEPTRON_ISAAC_RENDER_META_KEY]

    np.testing.assert_allclose(metadata["anchor_timestamp_seconds"], [0.6])
    np.testing.assert_allclose([fake_renderer.build_calls[-1]["anchor_timestamp_seconds"]], [0.6])


def test_so100_101_joint_frame_steps_preserve_math_and_serialization(tmp_path):
    signs = [1.0, -1.0, 1.0, 1.0, 1.0, 1.0]
    offsets = [0.0, 90.0, 90.0, 0.0, 0.0, 0.0]
    current = torch.tensor([[10.0, 20.0, 30.0, 40.0, 50.0, 60.0]])
    observation_transition = {
        TransitionKey.OBSERVATION: {OBS_STATE: current},
        TransitionKey.ACTION: None,
        TransitionKey.REWARD: 0.0,
        TransitionKey.DONE: False,
        TransitionKey.TRUNCATED: False,
        TransitionKey.INFO: {},
        TransitionKey.COMPLEMENTARY_DATA: {},
    }
    state_step = PerceptronIsaacStateFrameTransformStep(signs, offsets)
    action_step = PerceptronIsaacActionFrameTransformStep(signs, offsets)

    model_state = state_step(observation_transition)[TransitionKey.OBSERVATION][OBS_STATE]
    action_transition = {**observation_transition, TransitionKey.ACTION: model_state}
    recovered = action_step(action_transition)[TransitionKey.ACTION]

    torch.testing.assert_close(
        model_state,
        torch.tensor([[10.0, 70.0, 120.0, 40.0, 50.0, 60.0]]),
    )
    torch.testing.assert_close(recovered, current)

    frame_config = {"joint_signs": signs, "joint_offsets": offsets}
    registry_names = [
        "perceptron_isaac_state_frame_transform",
        "perceptron_isaac_training_action_frame_transform",
        "perceptron_isaac_action_frame_transform",
    ]
    step_types = [
        PerceptronIsaacStateFrameTransformStep,
        PerceptronIsaacTrainingActionFrameTransformStep,
        PerceptronIsaacActionFrameTransformStep,
    ]
    pipeline = PolicyProcessorPipeline(
        steps=[step_type(**frame_config) for step_type in step_types],
        name="joint_frames",
        to_transition=batch_to_transition,
        to_output=transition_to_batch,
    )
    pipeline.save_pretrained(tmp_path, config_filename="joint_frames.json")
    loaded = PolicyProcessorPipeline.from_pretrained(
        tmp_path,
        config_filename="joint_frames.json",
        to_transition=batch_to_transition,
        to_output=transition_to_batch,
    )
    serialized_steps = json.loads((tmp_path / "joint_frames.json").read_text())["steps"]

    assert [step["registry_name"] for step in serialized_steps] == registry_names
    assert [step["config"] for step in serialized_steps] == [frame_config] * 3
    assert [type(step) for step in loaded.steps] == step_types
    assert [step.get_config() for step in loaded.steps] == [frame_config] * 3


def test_so100_training_state_and_action_use_the_same_model_frame():
    signs = [1.0, -1.0, 1.0, 1.0, 1.0, 1.0]
    offsets = [0.0, 90.0, 90.0, 0.0, 0.0, 0.0]
    arm = torch.tensor([[[10.0, 20.0, 30.0, 40.0, 50.0, 60.0]]])
    transition = {
        TransitionKey.OBSERVATION: {OBS_STATE: arm.clone()},
        TransitionKey.ACTION: arm.clone(),
        TransitionKey.COMPLEMENTARY_DATA: {},
    }

    state_out = PerceptronIsaacStateFrameTransformStep(signs, offsets)(transition)
    both_out = PerceptronIsaacTrainingActionFrameTransformStep(signs, offsets)(state_out)
    expected = torch.tensor([[[10.0, 70.0, 120.0, 40.0, 50.0, 60.0]]])

    torch.testing.assert_close(both_out[TransitionKey.OBSERVATION][OBS_STATE], expected)
    torch.testing.assert_close(both_out[TransitionKey.ACTION], expected)


def test_loaded_legacy_base_inserts_training_action_transform_before_pack(tmp_path):
    stats_path = tmp_path / "stats.json"
    _write_native_stats(stats_path)
    cfg = PerceptronIsaacConfig(
        device="cpu",
        native_stats_path=str(stats_path),
        joint_signs=[1.0, -1.0, 1.0, 1.0, 1.0, 1.0],
        joint_offsets=[0.0, 90.0, 90.0, 0.0, 0.0, 0.0],
        input_features=_features(),
        output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(7,))},
    )
    preprocessor, postprocessor = make_perceptron_isaac_pre_post_processors(cfg)
    preprocessor.steps = [
        step
        for step in preprocessor.steps
        if not isinstance(step, PerceptronIsaacTrainingActionFrameTransformStep)
    ]
    preprocessor.save_pretrained(tmp_path)
    postprocessor.save_pretrained(tmp_path)

    loaded, _ = make_pre_post_processors(cfg, pretrained_path=str(tmp_path))
    names = [type(step) for step in loaded.steps]

    assert names.count(PerceptronIsaacTrainingActionFrameTransformStep) == 1
    assert names.index(PerceptronIsaacTrainingActionFrameTransformStep) < names.index(
        PerceptronIsaacMharmonyPackProcessorStep
    )


def test_joint_frame_config_rejects_noninvertible_signs():
    with pytest.raises(ValueError, match=r"only \+1 or -1"):
        PerceptronIsaacConfig(
            device="cpu",
            joint_signs=[1.0, 0.0],
            joint_offsets=[0.0, 90.0],
            input_features=_features(),
            output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(7,))},
        )


def test_native_pack_renders_batched_training_actions_with_explicit_dataset_time(monkeypatch):
    step = PerceptronIsaacMharmonyPackProcessorStep(
        enabled=True,
        camera_order=["image", "wrist_image"],
        image_size=(16, 16),
        n_obs_steps=3,
        chunk_size=30,
        action_dim=7,
        device="cpu",
        dtype="float32",
    )
    renderer = _install_fake_native_renderer(step, monkeypatch)
    step._fast_processor = object()
    action = torch.arange(2 * 30 * 7, dtype=torch.float32).reshape(2, 30, 7)
    action[0, 0, 0] = -999.0
    action_is_pad = torch.zeros(2, 30, dtype=torch.bool)
    action_is_pad[1, -2:] = True
    transition = {
        TransitionKey.OBSERVATION: {
            "observation.images.image": torch.zeros(2, 3, 3, 16, 16),
            "observation.images.image2": torch.ones(2, 3, 3, 16, 16),
            OBS_STATE: torch.zeros(2, 3, 8),
        },
        TransitionKey.ACTION: action,
        TransitionKey.REWARD: 0.0,
        TransitionKey.DONE: False,
        TransitionKey.TRUNCATED: False,
        TransitionKey.INFO: {},
        TransitionKey.COMPLEMENTARY_DATA: {
            "task": ["Pick Place", "Open Drawer"],
            "timestamp": torch.tensor([0.0, 0.05]),
            "action_is_pad": action_is_pad,
        },
    }

    out = step(transition)
    complementary = out[TransitionKey.COMPLEMENTARY_DATA]

    assert complementary[PERCEPTRON_ISAAC_KEPT_SAMPLE_INDICES_KEY] == [1]
    assert complementary[PERCEPTRON_ISAAC_STREAM_KEY].payload["mode"] == "collate"
    assert len(renderer.build_training_calls) == 2
    assert renderer.build_training_calls[1]["prompt"] == "open drawer"
    assert renderer.build_training_calls[1]["anchor_timestamp_seconds"] == pytest.approx(0.05)
    assert renderer.build_training_calls[1]["action_is_pad"] == [False] * 28 + [True, True]
    assert len(renderer.collate_calls) == 1
    assert len(renderer.collate_calls[0]) == 1


def test_real_preprocessor_promotes_retained_indices_to_weighted_trainer(monkeypatch):
    """The filtered-row protocol must survive the real pipeline conversion boundary."""
    pytest.importorskip("accelerate", reason="accelerate is required for the trainer boundary")
    pytest.importorskip("datasets", reason="datasets is required to import lerobot_train")

    from accelerate import Accelerator

    from lerobot.scripts.lerobot_train import update_policy

    cfg = PerceptronIsaacConfig(
        device="cpu",
        n_obs_steps=3,
        native_stats_path="/tmp/stats.json",
        input_features=_features(),
        output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(7,))},
    )
    preprocessor, _ = make_perceptron_isaac_pre_post_processors(cfg)
    render_step = next(
        step for step in preprocessor.steps if isinstance(step, PerceptronIsaacMharmonyPackProcessorStep)
    )
    _install_fake_native_renderer(render_step, monkeypatch)
    render_step._fast_processor = object()

    action = torch.zeros(3, 30, 7)
    action[0, 0, 0] = -999.0
    processed = preprocessor(
        {
            "observation.images.image": torch.zeros(3, 3, 3, 256, 256),
            "observation.images.image2": torch.ones(3, 3, 3, 256, 256),
            OBS_STATE: torch.zeros(3, 3, 8),
            ACTION: action,
            "task": ["drop", "keep one", "keep two"],
            "timestamp": torch.tensor([0.0, 0.05, 0.1]),
        }
    )

    # transition_to_batch promotes complementary data back to policy-batch keys.
    assert processed[PERCEPTRON_ISAAC_KEPT_SAMPLE_INDICES_KEY] == [1, 2]

    class BoundaryPolicy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(0.0))

        def forward(self, batch, reduction="mean"):
            assert batch[PERCEPTRON_ISAAC_KEPT_SAMPLE_INDICES_KEY] == [1, 2]
            per_sample = (self.weight - self.weight.new_tensor([1.0, 3.0])).square()
            loss = per_sample if reduction == "none" else per_sample.mean()
            return loss, {"boundary_loss": per_sample.mean().detach()}

    class OriginalBatchWeighter:
        @staticmethod
        def compute_batch_weights(_batch):
            return torch.tensor([1000.0, 1.0, 3.0]), {}

    class Tracker:
        @staticmethod
        def update_metrics(_metrics):
            return None

    accelerator = Accelerator(cpu=True)
    policy = BoundaryPolicy()
    optimizer = torch.optim.SGD(policy.parameters(), lr=0.1)
    policy, optimizer = accelerator.prepare(policy, optimizer)

    update_policy(
        Tracker(),
        policy,
        processed,
        optimizer,
        10.0,
        accelerator,
        sample_weighter=OriginalBatchWeighter(),
    )

    # Rows 1 and 2 survive. Their weighted gradient is
    # (1 * -2 + 3 * -6) / 4 = -5, so SGD(lr=.1) moves the weight to .5.
    assert accelerator.unwrap_model(policy).weight.item() == pytest.approx(0.5)


def test_native_pack_training_requires_explicit_dataset_timestamp(monkeypatch):
    step = PerceptronIsaacMharmonyPackProcessorStep(
        enabled=True,
        camera_order=["image", "wrist_image"],
        image_size=(16, 16),
        n_obs_steps=3,
        chunk_size=30,
        action_dim=7,
        device="cpu",
        dtype="float32",
    )
    _install_fake_native_renderer(step, monkeypatch)
    step._fast_processor = object()
    transition = {
        TransitionKey.OBSERVATION: {
            "observation.images.image": torch.zeros(1, 3, 3, 16, 16),
            "observation.images.image2": torch.ones(1, 3, 3, 16, 16),
            OBS_STATE: torch.zeros(1, 3, 8),
        },
        TransitionKey.ACTION: torch.zeros(1, 30, 7),
        TransitionKey.REWARD: 0.0,
        TransitionKey.DONE: False,
        TransitionKey.TRUNCATED: False,
        TransitionKey.INFO: {},
        TransitionKey.COMPLEMENTARY_DATA: {"task": ["Pick Place"]},
    }

    with pytest.raises(ValueError, match="explicit episode-relative dataset timestamp"):
        step(transition)


class ThresholdAwareFakeNativeRenderer(FakeNativeRenderer):
    """Honors skip_outlier_threshold like the real renderer and gives per-prompt lengths."""

    def __init__(self, lengths_by_prompt=None):
        super().__init__()
        self.lengths_by_prompt = lengths_by_prompt or {}

    def build_training(self, **kwargs):
        self.build_training_calls.append(kwargs)
        outlier = float(np.asarray(kwargs["action_chunk"])[0, 0]) == -999.0
        if outlier and float(kwargs["skip_outlier_threshold"]) > 0.0:
            raise IsaacActionOutlierError("synthetic outlier")
        length = self.lengths_by_prompt.get(kwargs["prompt"], 1)
        return FakeTensorStream({"mode": "train", **kwargs}, length)


def _training_step(**step_kwargs):
    step = PerceptronIsaacMharmonyPackProcessorStep(
        enabled=True,
        camera_order=["image", "wrist_image"],
        image_size=(16, 16),
        n_obs_steps=3,
        chunk_size=30,
        action_dim=7,
        device="cpu",
        dtype="float32",
        **step_kwargs,
    )
    step._fast_processor = object()
    return step


def _training_transition(action, tasks, **complementary):
    batch = action.shape[0]
    return {
        TransitionKey.OBSERVATION: {
            "observation.images.image": torch.zeros(batch, 3, 3, 16, 16),
            "observation.images.image2": torch.ones(batch, 3, 3, 16, 16),
            OBS_STATE: torch.zeros(batch, 3, 8),
        },
        TransitionKey.ACTION: action,
        TransitionKey.REWARD: 0.0,
        TransitionKey.DONE: False,
        TransitionKey.TRUNCATED: False,
        TransitionKey.INFO: {},
        TransitionKey.COMPLEMENTARY_DATA: {
            "task": tasks,
            "action_is_pad": torch.zeros(batch, 30, dtype=torch.bool),
            **complementary,
        },
    }


def test_frame_index_anchor_rejects_genuine_dataset_fps_mismatch(monkeypatch):
    """A correctly-declared 30 Hz dataset under a 20 Hz contract must not silently dilate."""
    step = _training_step(dataset_name="molmoact2_so100_101")
    _install_fake_native_renderer(step, monkeypatch)  # contract target_fps=20
    step._dataset_declared_fps = 30.0

    transition = _training_transition(torch.zeros(1, 30, 7), ["pick"], frame_index=torch.tensor([4]))
    with pytest.raises(ValueError, match="scaled episode clock"):
        step(transition)


def test_frame_index_anchor_warns_once_for_misdeclared_libero_fps(monkeypatch, caplog):
    """LIBERO conversions misdeclare fps; the contract clock wins, but loudly."""
    step = _training_step(dataset_name="libero")
    renderer = _install_fake_native_renderer(step, monkeypatch)  # contract target_fps=20
    step._dataset_declared_fps = 10.0

    transition = _training_transition(torch.zeros(1, 30, 7), ["pick"], frame_index=torch.tensor([4]))
    with caplog.at_level(logging.WARNING):
        step(transition)
        step(_training_transition(torch.zeros(1, 30, 7), ["pick"], frame_index=torch.tensor([5])))

    np.testing.assert_allclose([renderer.build_training_calls[0]["anchor_timestamp_seconds"]], [4 / 20])
    warnings = [rec for rec in caplog.records if "misdeclare" in rec.getMessage()]
    assert len(warnings) == 1


def test_frame_index_anchor_accepts_matching_dataset_fps(monkeypatch):
    step = _training_step(dataset_name="molmoact2_so100_101")
    renderer = _install_fake_native_renderer(step, monkeypatch)  # contract target_fps=20
    step._dataset_declared_fps = 20.0

    step(_training_transition(torch.zeros(1, 30, 7), ["pick"], frame_index=torch.tensor([4])))

    np.testing.assert_allclose([renderer.build_training_calls[0]["anchor_timestamp_seconds"]], [4 / 20])


def test_all_outlier_microbatch_fails_loud_single_process(monkeypatch):
    """Without collectives to keep in lockstep, an all-outlier microbatch must abort."""
    step = _training_step()
    renderer = _install_fake_native_renderer(step, monkeypatch)
    action = torch.zeros(1, 30, 7)
    action[0, 0, 0] = -999.0

    with pytest.raises(IsaacActionOutlierError, match="normalization stats"):
        step(_training_transition(action, ["pick"], timestamp=torch.tensor([0.0])))

    # Single-process keeps the renderer's own fail-loud over-length check.
    assert renderer.build_training_calls[0]["max_sequence_length"] == 4096


def _mock_distributed_batch_validation(monkeypatch, *, peer_has_no_usable_samples=False):
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_backend", lambda: "gloo")

    def all_reduce(status, op=None):
        if peer_has_no_usable_samples:
            status.zero_()

    monkeypatch.setattr(torch.distributed, "all_reduce", all_reduce)


def test_all_outlier_microbatch_fails_collectively_distributed(monkeypatch):
    _mock_distributed_batch_validation(monkeypatch)
    step = _training_step()
    renderer = ThresholdAwareFakeNativeRenderer()
    step._stream_builder = renderer
    action = torch.zeros(2, 30, 7)
    action[:, 0, 0] = -999.0

    with pytest.raises(IsaacActionOutlierError, match="distributed rank has no usable samples"):
        step(_training_transition(action, ["pick", "place"], timestamp=torch.tensor([0.0, 0.05])))

    # The threshold is never disabled just to keep collectives alive.
    assert all(call["skip_outlier_threshold"] > 0.0 for call in renderer.build_training_calls)
    assert renderer.collate_calls == []


def test_distributed_over_length_sample_is_skipped_not_fatal(monkeypatch):
    """One over-length sample on one rank must not abort mid-accumulation (NCCL hang)."""
    _mock_distributed_batch_validation(monkeypatch)
    step = _training_step(train_max_sequence_length=5)
    renderer = ThresholdAwareFakeNativeRenderer(lengths_by_prompt={"pick": 10, "place": 1})
    step._stream_builder = renderer

    out = step(
        _training_transition(torch.zeros(2, 30, 7), ["pick", "place"], timestamp=torch.tensor([0.0, 0.05]))
    )

    complementary = out[TransitionKey.COMPLEMENTARY_DATA]
    assert complementary[PERCEPTRON_ISAAC_KEPT_SAMPLE_INDICES_KEY] == [1]
    # The length gate moved out of build_training so over-length cannot raise there.
    assert renderer.build_training_calls[0]["max_sequence_length"] == 2**31 - 1


def test_distributed_all_over_length_microbatch_fails_collectively(monkeypatch):
    _mock_distributed_batch_validation(monkeypatch)
    step = _training_step(train_max_sequence_length=5)
    renderer = ThresholdAwareFakeNativeRenderer(lengths_by_prompt={"pick": 10, "place": 11})
    step._stream_builder = renderer

    with pytest.raises(IsaacActionOutlierError, match="distributed rank has no usable samples"):
        step(
            _training_transition(
                torch.zeros(2, 30, 7),
                ["pick", "place"],
                timestamp=torch.tensor([0.0, 0.05]),
            )
        )


def test_distributed_valid_rank_fails_before_loss_collectives_when_peer_is_invalid(monkeypatch):
    _mock_distributed_batch_validation(monkeypatch, peer_has_no_usable_samples=True)
    step = _training_step()
    renderer = ThresholdAwareFakeNativeRenderer()
    step._stream_builder = renderer

    with pytest.raises(IsaacActionOutlierError, match="distributed rank has no usable samples"):
        step(
            _training_transition(
                torch.zeros(1, 30, 7),
                ["pick"],
                timestamp=torch.tensor([0.0]),
            )
        )

    assert renderer.collate_calls == []


def test_render_processor_conforms_resolution_and_rejects_undecidable_layouts():
    """Off-size frames now conform via the saved geometry contract instead of raising;
    only a layout whose channel axis cannot be identified still fails loudly."""
    step = PerceptronIsaacMharmonyPackProcessorStep(
        camera_order=["image"],
        image_keys=["observation.images.image"],
        image_size=(256, 256),
        allow_image_key_fallback=False,
    )
    conformed = step._extract_image_tensor(
        {"observation.images.image": torch.zeros(1, 3, 16, 16)}, batch_size=1
    )
    assert conformed.shape == (1, 1, 1, 3, 256, 256)

    with pytest.raises(ValueError, match="training resolution"):
        step._extract_image_tensor({"observation.images.image": torch.zeros(1, 2, 16, 16)}, batch_size=1)


def test_native_mharmony_pack_builds_native_renderer(tmp_path):
    stats_path = tmp_path / "stats.json"
    _write_native_stats(stats_path)
    step = PerceptronIsaacMharmonyPackProcessorStep(
        enabled=True,
        inference_backend="native_mharmony",
        camera_order=["image", "wrist_image"],
        image_size=(256, 256),
        native_stats_path=str(stats_path),
    )

    step._ensure_renderer()

    assert isinstance(step._stream_builder, IsaacNativeMharmonyRenderer)
    assert step._stats.target_fps == 20.0


def test_local_isaac_stats_loads_and_unnormalizes_actions(tmp_path):
    stats_path = tmp_path / "stats.json"
    stats_path.write_text(
        """
        {
          "schema": "flow_matching_stats_v1",
          "target_fps": 20.0,
          "action_horizon": 30,
          "action_dim": 2,
          "proprio_dim": 2,
          "action": {"q01": [0.0, 0.0], "q99": [1.0, 0.0], "min": [-1.0, -2.0], "max": [2.0, 2.0]},
          "proprio": {"q01": [-1.0, -1.0], "q99": [1.0, 1.0], "min": [-1.0, -1.0], "max": [1.0, 1.0]}
        }
        """
    )

    stats = load_isaac_stats(stats_path)
    actions = unnormalize_isaac_actions(np.asarray([[0.0, 1.0]], dtype=np.float32), stats.action)

    assert stats.action_horizon == 30
    assert stats.action_dim == 2
    np.testing.assert_allclose(actions, np.asarray([[0.5, 2.0]], dtype=np.float32), rtol=1e-6)


def test_local_isaac_stats_collapsed_dims_degrade_to_identity(tmp_path):
    stats_path = tmp_path / "stats.json"
    stats_path.write_text(
        json.dumps(
            {
                "schema": "flow_matching_stats_v1",
                "target_fps": 20.0,
                "action_horizon": 30,
                "action_dim": 2,
                "proprio_dim": 1,
                "action": {"q01": [5.0, 0.0], "q99": [5.0, 2.0], "min": [5.0, 0.0], "max": [5.0, 2.0]},
                "proprio": {"q01": [3.0], "q99": [3.0], "min": [3.0], "max": [3.0]},
            }
        )
    )

    stats = load_isaac_stats(stats_path)
    # Genesis keeps the collapsed quantile box and normalizes through the eps denominator, so
    # a raw value sitting at the constant maps to -1.0 -- not 0.0. Feeding 0.0 where training
    # saw -1.0 is a full-unit shift on that input dimension.
    at_constant = normalize_isaac_actions(np.asarray([[5.0, 1.0]], dtype=np.float32), stats.action)
    np.testing.assert_allclose(at_constant, [[-1.0, 0.0]], atol=1e-5)

    actions = unnormalize_isaac_actions(np.asarray([[0.75, 0.0]], dtype=np.float32), stats.action)
    np.testing.assert_allclose(actions, [[5.0, 1.0]], atol=1e-6)


def test_isaac_stats_state_dict_round_trips_empty_string_labels():
    from lerobot.policies.perceptron_isaac.isaac_stats import (
        IsaacNormalizationStats,
        IsaacStatsBlock,
        isaac_stats_from_state_dict,
        isaac_stats_to_state_dict,
    )

    block = IsaacStatsBlock(
        q01=np.zeros(2, dtype=np.float32),
        q99=np.ones(2, dtype=np.float32),
        minimum=None,
        maximum=None,
    )
    stats = IsaacNormalizationStats(
        action=block,
        proprio=block,
        target_fps=30.0,
        action_horizon=30,
        schema="flow_matching_stats_v1",
        profile_id="",
        profile_scope="",
        validation_status="",
    )

    # Empty-string labels must serialize (torch.frombuffer rejects zero-length buffers);
    # a crash here loses the checkpoint save at the end of a training run.
    state = isaac_stats_to_state_dict(stats)
    restored = isaac_stats_from_state_dict(state)

    assert restored.profile_id == ""
    assert restored.profile_scope == ""
    assert restored.validation_status == ""
    assert restored.schema == "flow_matching_stats_v1"


def test_policy_reset_clears_flow_seed_index_and_has_no_policy_side_renderer(tmp_path):
    stats_path = tmp_path / "stats.json"
    _write_native_stats(stats_path)
    cfg = PerceptronIsaacConfig(
        device="cpu",
        native_stats_path=str(stats_path),
        input_features=_features(),
        output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(7,))},
    )
    policy = PerceptronIsaacPolicy(cfg)
    policy._isaac_model = FakeIsaacModel()
    policy._flow_seed_index = 12
    policy._frame_index = 3
    policy._last_external_timestamp_seconds = 0.15
    policy._online_image_queue.append(np.zeros((2, 3, 16, 16), dtype=np.uint8))
    policy._online_state_queue.append(np.zeros((8,), dtype=np.float32))
    policy._online_render_step = PerceptronIsaacMharmonyPackProcessorStep()

    policy.reset()

    assert policy._flow_seed_index == -1
    assert policy._frame_index == -1
    assert policy._last_external_timestamp_seconds is None
    assert len(policy._online_image_queue) == 0
    assert len(policy._online_state_queue) == 0
    assert policy._online_render_step is None
    policy._ensure_renderer()
    assert policy._stats.target_fps == 20.0
    assert policy._native_render_metadata.robot_type == "generic"
    assert not hasattr(policy, "_stream_builder")


def test_native_policy_without_stream_requires_raw_online_observation(tmp_path):
    stats_path = tmp_path / "stats.json"
    _write_native_stats(stats_path)
    cfg = PerceptronIsaacConfig(
        device="cpu",
        native_stats_path=str(stats_path),
        input_features=_features(),
        output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(7,))},
    )
    policy = PerceptronIsaacPolicy(cfg)
    policy._isaac_model = FakeIsaacModel()

    with pytest.raises(ValueError, match="at least one image observation"):
        policy.predict_action_chunk({"task": "pick up the mug"})


def test_saved_pipelines_reload_with_trainer_supplied_normalizer_overrides(tmp_path):
    """lerobot-train always passes normalizer overrides; ISAAC has no such step.

    ``normalization_mapping`` is IDENTITY throughout because the ISAAC steps carry the
    checkpoint's own quantile stats, so the pipeline contains no normalizer_processor and
    ``from_pretrained`` would otherwise reject the unused override keys -- breaking
    ``lerobot-train --policy.path=<isaac package>`` for every ISAAC package.
    """
    from lerobot.policies.perceptron_isaac.processor_perceptron_isaac import (
        make_perceptron_isaac_pre_post_processors_from_pretrained,
    )

    stats_path = tmp_path / "stats.json"
    _write_native_stats(stats_path)
    cfg = PerceptronIsaacConfig(
        device="cpu",
        native_stats_path=str(stats_path),
        input_features=_features(),
        output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(7,))},
    )
    package = tmp_path / "package"
    preprocessor, postprocessor = make_perceptron_isaac_pre_post_processors(cfg)
    preprocessor.save_pretrained(package)
    postprocessor.save_pretrained(package)

    reloaded_pre, reloaded_post = make_perceptron_isaac_pre_post_processors_from_pretrained(
        config=cfg,
        pretrained_path=str(package),
        preprocessor_overrides={
            "device_processor": {"device": "cpu"},
            # The two the shared trainer always sends and ISAAC does not have:
            "normalizer_processor": {"features": {}, "norm_map": {}, "stats": {}},
        },
        postprocessor_overrides={
            "device_processor": {"device": "cpu"},
            "unnormalizer_processor": {"features": {}, "norm_map": {}, "stats": {}},
        },
    )

    assert reloaded_pre is not None
    assert reloaded_post is not None
    step_names = {type(step).__name__ for step in reloaded_pre.steps}
    assert not any("Normalizer" in name for name in step_names)


def test_native_policy_loads_backbone_with_policy_chunk_size(tmp_path, monkeypatch):
    stats_path = tmp_path / "stats.json"
    _write_native_stats(stats_path)
    calls = []

    import lerobot.policies.perceptron_isaac.modeling_qwen35_vla as qwen35_vla

    def fake_load_qwen35_vla_from_hf(hf_dir, **kwargs):
        calls.append({"hf_dir": hf_dir, **kwargs})
        return FakeIsaacModel(), SimpleNamespace()

    monkeypatch.setattr(qwen35_vla, "load_qwen35_vla_from_hf", fake_load_qwen35_vla_from_hf)
    cfg = PerceptronIsaacConfig(
        device="cpu",
        hf_model_path=str(tmp_path / "hf"),
        native_stats_path=str(stats_path),
        chunk_size=50,
        n_action_steps=30,
        num_inference_steps=12,
        # fp32 storage defers the load, so pin bf16 storage to exercise the eager path.
        train_storage_fp32=False,
        input_features=_features(),
        output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(7,))},
    )

    policy = PerceptronIsaacPolicy(cfg)

    assert policy._isaac_model is not None
    assert len(calls) == 1
    # Serving knobs travel as overrides; the full expert dict is resolved inside
    # load_qwen35_vla_from_hf so a checkpoint-stamped contract stays authoritative.
    assert "action_expert" not in calls[0]
    overrides = calls[0]["action_expert_overrides"]
    assert overrides["action_horizon"] == 50
    assert overrides["num_inference_steps"] == 12


def _patch_mk1_checkpoint_load(monkeypatch, loader):
    contract = SimpleNamespace(
        artifact=SimpleNamespace(artifact_kind="trained_policy", trained_steps=1),
        vector_encoder=SimpleNamespace(max_states=128),
        action_expert=SimpleNamespace(action_dim=64, action_horizon=64),
        validate_coord_reserved_token_groups=lambda groups: None,
    )
    import lerobot.policies.perceptron_isaac.mk1_checkpoint_contract as checkpoint_contract
    import lerobot.policies.perceptron_isaac.modeling_mk1_vla as mk1_vla

    monkeypatch.setattr(
        checkpoint_contract,
        "validate_mk1_checkpoint",
        lambda model_dir, **kwargs: (contract, SimpleNamespace()),
    )
    monkeypatch.setattr(mk1_vla, "load_mk1_vla_from_hf", loader)
    return contract


def test_native_policy_mk1_loader_is_checkpoint_driven(tmp_path, monkeypatch):
    stats_path = tmp_path / "stats.json"
    _write_native_stats(stats_path)
    hf_dir = tmp_path / "hf"
    calls = []

    def fake_load_mk1_vla_from_hf(model_dir, **kwargs):
        calls.append({"model_dir": model_dir, **kwargs})
        return FakeIsaacModel(), SimpleNamespace(), contract

    contract = _patch_mk1_checkpoint_load(monkeypatch, fake_load_mk1_vla_from_hf)
    cfg = PerceptronIsaacConfig(
        device="cpu",
        hf_model_path=str(hf_dir),
        native_stats_path=str(stats_path),
        chunk_size=50,
        n_action_steps=30,
        num_inference_steps=12,
        apply_offset_norm=False,
        mk1_model_import_sha256="0" * 64,
        train_storage_fp32=False,
        input_features=_features(),
        output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(7,))},
    )

    policy = PerceptronIsaacPolicy(cfg)

    assert policy._isaac_model is not None
    assert calls == [
        {
            "model_dir": hf_dir,
            "dtype": torch.bfloat16,
            "device": "cpu",
            "allow_test_only_reduced_geometry": False,
            "attention_backend": "torch_sdpa_v1",
            "route_reduction": "stable_token_segment_sum_v1",
            "allowed_storage_dtypes": frozenset({"BF16"}),
        }
    ]


def test_neutral_debug_package_loads_but_cannot_emit_actions(tmp_path):
    stats_path = tmp_path / "stats.json"
    _write_native_stats(stats_path)
    config = PerceptronIsaacConfig(
        device="cpu",
        native_stats_path=str(stats_path),
        artifact_kind="neutral_debug",
        trained_steps=0,
        input_features=_features(),
        output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(7,))},
    )
    policy = PerceptronIsaacPolicy(config)

    with pytest.raises(RuntimeError, match="rejects neutral_debug"):
        policy.predict_action_chunk({PERCEPTRON_ISAAC_STREAM_KEY: FakePackedStream()})
    with pytest.raises(RuntimeError, match="rejects neutral_debug"):
        policy.select_action({})


@pytest.mark.parametrize("trained_steps", [None, 1, -1, True, 1.5])
def test_neutral_debug_config_requires_exact_zero_training_steps(trained_steps):
    with pytest.raises(ValueError, match="trained_steps"):
        PerceptronIsaacConfig(
            artifact_kind="neutral_debug",
            trained_steps=trained_steps,
            input_features=_features(),
            output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(7,))},
        )


@pytest.mark.parametrize("trained_steps", [None, 0, -1, True, 1.5])
def test_trained_policy_config_requires_positive_training_steps(trained_steps):
    with pytest.raises(ValueError, match="trained_steps"):
        PerceptronIsaacConfig(
            artifact_kind="trained_policy",
            trained_steps=trained_steps,
            input_features=_features(),
            output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(7,))},
        )


@pytest.mark.parametrize(
    ("artifact_kind", "trained_steps"),
    [("trained_policy", 2), ("neutral_debug", 0)],
)
def test_native_policy_rejects_spoofed_outer_artifact_identity_before_load(
    tmp_path, monkeypatch, artifact_kind, trained_steps
):
    stats_path = tmp_path / "stats.json"
    _write_native_stats(stats_path)
    load_calls = []
    _patch_mk1_checkpoint_load(monkeypatch, lambda *args, **kwargs: load_calls.append((args, kwargs)))

    with pytest.raises(RuntimeError, match="artifact identity mismatch"):
        PerceptronIsaacPolicy(
            PerceptronIsaacConfig(
                device="cpu",
                hf_model_path=str(tmp_path / "hf"),
                native_stats_path=str(stats_path),
                artifact_kind=artifact_kind,
                trained_steps=trained_steps,
                mk1_model_import_sha256="0" * 64,
                train_storage_fp32=False,
                input_features=_features(),
                output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(7,))},
            )
        )

    assert load_calls == []


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("vector_max_states", 64, "vector width mismatch"),
        ("max_action_dim", 63, "maximum action width mismatch"),
        ("max_action_horizon", 63, "maximum action horizon mismatch"),
        ("action_dim", 65, "action_dim.*max|action_dim exceeds"),
        ("chunk_size", 65, "chunk_size.*max|chunk_size exceeds"),
        ("apply_offset_norm", True, "forbid"),
    ],
)
def test_native_policy_mk1_rejects_outer_contract_drift_before_load(
    tmp_path, monkeypatch, field, value, message
):
    stats_path = tmp_path / "stats.json"
    _write_native_stats(stats_path)
    hf_dir = tmp_path / "hf"
    load_calls = []
    _patch_mk1_checkpoint_load(
        monkeypatch,
        lambda *args, **kwargs: load_calls.append((args, kwargs)),
    )
    config_kwargs = {
        "device": "cpu",
        "hf_model_path": str(hf_dir),
        "native_stats_path": str(stats_path),
        "chunk_size": 50,
        "n_action_steps": 30,
        "action_dim": 7,
        "vector_max_states": 128,
        "apply_offset_norm": False,
        "mk1_model_import_sha256": "0" * 64,
        "train_storage_fp32": False,
        "input_features": _features(),
        "output_features": {ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(7,))},
    }
    config_kwargs[field] = value

    with pytest.raises((RuntimeError, ValueError), match=message):
        PerceptronIsaacPolicy(PerceptronIsaacConfig(**config_kwargs))

    assert not load_calls


def test_training_parameter_freeze_is_applied_once(tmp_path, monkeypatch):
    """Freezing must not re-apply per step: FSDP parameters are not leaf tensors.

    ``forward`` calls ``_configure_training_parameters`` on every training step. Once FSDP
    flat-shards the module the parameters are views, and ``requires_grad_`` raises "you can
    only change requires_grad flags of leaf variables" — which killed every rank and made
    multi-GPU training of this policy impossible.
    """
    stats_path = tmp_path / "stats.json"
    _write_native_stats(stats_path)
    calls = []

    import lerobot.policies.perceptron_isaac.modeling_qwen35_vla as qwen35_vla

    monkeypatch.setattr(
        qwen35_vla, "load_qwen35_vla_from_hf", lambda hf_dir, **kw: (MagicMock(), SimpleNamespace())
    )
    monkeypatch.setattr(qwen35_vla, "setup_qwen35_vla_for_training", lambda *a, **k: calls.append("setup"))
    cfg = PerceptronIsaacConfig(
        device="cpu",
        hf_model_path=str(tmp_path / "hf"),
        native_stats_path=str(stats_path),
        freeze_input_embeddings=False,
        input_features=_features(),
        output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(7,))},
    )
    policy = PerceptronIsaacPolicy(cfg)
    policy._load_backbone(training=True)
    assert calls == ["setup"]

    # Every later call (one per training step) must be a no-op.
    policy._configure_training_parameters()
    policy._configure_training_parameters()
    assert calls == ["setup"]


def test_fp32_training_storage_defers_backbone_load_then_loads_lazily(tmp_path, monkeypatch):
    """Genesis finetunes with fp32 parameter storage; never materialize bf16 weights first."""
    stats_path = tmp_path / "stats.json"
    _write_native_stats(stats_path)
    calls = []

    import lerobot.policies.perceptron_isaac.modeling_qwen35_vla as qwen35_vla

    def fake_load_qwen35_vla_from_hf(hf_dir, **kwargs):
        calls.append({"hf_dir": hf_dir, **kwargs})
        return MagicMock(), SimpleNamespace()

    monkeypatch.setattr(qwen35_vla, "load_qwen35_vla_from_hf", fake_load_qwen35_vla_from_hf)
    monkeypatch.setattr(qwen35_vla, "setup_qwen35_vla_for_training", lambda *a, **k: None)
    cfg = PerceptronIsaacConfig(
        device="cpu",
        hf_model_path=str(tmp_path / "hf"),
        native_stats_path=str(stats_path),
        input_features=_features(),
        output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(7,))},
    )
    assert cfg.train_storage_fp32 is True

    policy = PerceptronIsaacPolicy(cfg)
    assert policy._isaac_model is None
    assert calls == []

    policy._load_backbone(training=True)
    assert len(calls) == 1
    assert calls[0]["dtype"] is torch.float32


def test_native_mk1_dense_training_is_rejected_before_checkpoint_or_model_allocation(
    tmp_path, monkeypatch
):
    stats_path = tmp_path / "stats.json"
    _write_native_stats(stats_path)
    hf_dir = tmp_path / "hf"
    calls = []

    def fake_load_mk1_vla_from_hf(model_dir, **kwargs):
        calls.append({"model_dir": model_dir, **kwargs})
        return MagicMock(), SimpleNamespace(), contract

    contract = _patch_mk1_checkpoint_load(monkeypatch, fake_load_mk1_vla_from_hf)
    cfg = PerceptronIsaacConfig(
        device="cpu",
        hf_model_path=str(hf_dir),
        native_stats_path=str(stats_path),
        apply_offset_norm=False,
        mk1_model_import_sha256="0" * 64,
        input_features=_features(),
        output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(7,))},
    )
    assert cfg.train_storage_fp32 is True
    assert not cfg.train_expert_only and not cfg.use_peft

    policy = PerceptronIsaacPolicy(cfg)
    assert policy._isaac_model is None
    assert calls == []

    with pytest.raises(RuntimeError, match="dense full-parameter training is unsupported"):
        policy._load_backbone(training=True)
    assert calls == []

    policy._isaac_model = MagicMock()
    with pytest.raises(RuntimeError, match="dense full-parameter training is unsupported"):
        policy.forward({})


def test_native_mk1_bounded_trainable_set_reaches_the_checkpoint_load(tmp_path, monkeypatch):
    """The refusal is scoped to dense: expert-only training gets past the guard to the loader."""
    stats_path = tmp_path / "stats.json"
    _write_native_stats(stats_path)
    hf_dir = tmp_path / "hf"
    calls = []

    def fake_load_mk1_vla_from_hf(model_dir, **kwargs):
        calls.append({"model_dir": model_dir, **kwargs})
        return MagicMock(), SimpleNamespace(), contract

    contract = _patch_mk1_checkpoint_load(monkeypatch, fake_load_mk1_vla_from_hf)
    cfg = PerceptronIsaacConfig(
        device="cpu",
        hf_model_path=str(hf_dir),
        native_stats_path=str(stats_path),
        apply_offset_norm=False,
        mk1_model_import_sha256="0" * 64,
        train_expert_only=True,
        input_features=_features(),
        output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(7,))},
    )

    policy = PerceptronIsaacPolicy(cfg)
    policy._configure_training_parameters = lambda: None
    policy._load_backbone(training=True)

    assert len(calls) == 1
    # The frozen 32.2B MoE experts stay BF16, which is what keeps the production
    # grouped_mm dispatch in use during training.
    assert calls[0]["dtype"] is torch.bfloat16


def test_expert_only_training_keeps_frozen_base_bf16_and_promotes_trainables(tmp_path, monkeypatch):
    stats_path = tmp_path / "stats.json"
    _write_native_stats(stats_path)
    model = torch.nn.Module()
    model.vlm = torch.nn.Linear(2, 2)
    model.action_expert = torch.nn.Linear(2, 2)
    calls = []

    import lerobot.policies.perceptron_isaac.modeling_qwen35_vla as qwen35_vla

    def fake_load_qwen35_vla_from_hf(hf_dir, **kwargs):
        calls.append({"hf_dir": hf_dir, **kwargs})
        model.to(dtype=kwargs["dtype"])
        return model, SimpleNamespace()

    def fake_setup(loaded_model, **kwargs):
        for name, parameter in loaded_model.named_parameters():
            parameter.requires_grad_("action_expert" in name)

    monkeypatch.setattr(qwen35_vla, "load_qwen35_vla_from_hf", fake_load_qwen35_vla_from_hf)
    monkeypatch.setattr(qwen35_vla, "setup_qwen35_vla_for_training", fake_setup)
    cfg = PerceptronIsaacConfig(
        device="cpu",
        hf_model_path=str(tmp_path / "hf"),
        native_stats_path=str(stats_path),
        train_expert_only=True,
        freeze_input_embeddings=False,
        input_features=_features(),
        output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(7,))},
    )

    policy = PerceptronIsaacPolicy(cfg)
    policy._load_backbone(training=True)

    assert calls[0]["dtype"] is torch.bfloat16
    assert model.vlm.weight.requires_grad is False
    assert model.vlm.weight.dtype is torch.bfloat16
    assert model.action_expert.weight.requires_grad is True
    assert model.action_expert.weight.dtype is torch.float32


def test_isaac_compute_dtype_is_explicitly_bfloat16():
    assert PerceptronIsaacConfig().dtype == "bfloat16"
    with pytest.raises(ValueError, match="require dtype='bfloat16'"):
        PerceptronIsaacConfig(dtype="float32")


def test_native_policy_consumes_processor_stream_and_returns_normalized_actions(tmp_path):
    stats_path = tmp_path / "stats.json"
    _write_native_stats(stats_path, action_q01=0.0, action_q99=2.0)
    cfg = PerceptronIsaacConfig(
        device="cpu",
        native_stats_path=str(stats_path),
        input_features=_features(),
        output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(7,))},
    )
    policy = PerceptronIsaacPolicy(cfg)
    policy._isaac_model = FakeIsaacModel(value=0.0)
    stream = FakePackedStream()

    actions = policy.predict_action_chunk({PERCEPTRON_ISAAC_STREAM_KEY: stream, "task": "pick up the mug"})

    assert tuple(actions.shape) == (1, 30, 7)
    np.testing.assert_allclose(actions.numpy(), np.zeros((1, 30, 7)), atol=1e-6)
    assert stream.to_calls
    assert policy._isaac_model.sample_calls[0]["stream"] is stream
    assert policy._isaac_model.sample_calls[0]["num_flow_samples"] == 4


def test_native_policy_rejects_non_finite_normalized_actions(tmp_path):
    stats_path = tmp_path / "stats.json"
    _write_native_stats(stats_path)
    cfg = PerceptronIsaacConfig(
        device="cpu",
        native_stats_path=str(stats_path),
        input_features=_features(),
        output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(7,))},
    )
    policy = PerceptronIsaacPolicy(cfg)
    policy._isaac_model = FakeIsaacModel(value=float("nan"))

    with pytest.raises(RuntimeError, match="non-finite normalized actions"):
        policy.predict_action_chunk(
            {PERCEPTRON_ISAAC_STREAM_KEY: FakePackedStream(), "task": "pick up the mug"}
        )


def test_native_policy_autocasts_bf16_compute_with_fp32_master_weights(tmp_path):
    stats_path = tmp_path / "stats.json"
    _write_native_stats(stats_path)
    cfg = PerceptronIsaacConfig(
        device="cpu",
        native_stats_path=str(stats_path),
        train_storage_fp32=True,
        input_features=_features(),
        output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(7,))},
    )
    policy = PerceptronIsaacPolicy(cfg)
    policy._isaac_model = FakeMixedPrecisionIsaacModel()

    actions = policy.predict_action_chunk({PERCEPTRON_ISAAC_STREAM_KEY: FakePackedStream(), "task": "pick"})

    assert policy._isaac_model.action_expert.weight.dtype == torch.float32
    assert policy._isaac_model.sample_dtype == torch.bfloat16
    assert actions.dtype == torch.float32


def test_native_policy_requests_model_side_flow_sample_averaging(tmp_path):
    stats_path = tmp_path / "stats.json"
    _write_native_stats(stats_path)
    cfg = PerceptronIsaacConfig(
        device="cpu",
        native_stats_path=str(stats_path),
        num_flow_samples=3,
        input_features=_features(),
        output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(7,))},
    )
    policy = PerceptronIsaacPolicy(cfg)
    policy._isaac_model = FakeFlowAveragingIsaacModel()

    actions = policy.predict_action_chunk({PERCEPTRON_ISAAC_STREAM_KEY: FakePackedStream(), "task": "pick"})

    assert len(policy._isaac_model.sample_calls) == 1
    assert policy._isaac_model.sample_calls[0]["num_flow_samples"] == 3
    np.testing.assert_allclose(actions.numpy(), np.full((1, 30, 7), 2.0, dtype=np.float32))


def test_native_policy_uses_configured_flow_seed_base_without_environment(monkeypatch, tmp_path):
    stats_path = tmp_path / "stats.json"
    _write_native_stats(stats_path)

    def _policy():
        cfg = PerceptronIsaacConfig(
            device="cpu",
            native_stats_path=str(stats_path),
            num_flow_samples=4,
            flow_seed_base=1234,
            input_features=_features(),
            output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(7,))},
        )
        policy = PerceptronIsaacPolicy(cfg)
        policy._isaac_model = FakeRandomFlowIsaacModel()
        return policy

    monkeypatch.delenv("ISAAC_FLOW_SEED_BASE", raising=False)
    first = _policy().predict_action_chunk({PERCEPTRON_ISAAC_STREAM_KEY: FakePackedStream(), "task": "pick"})
    second = _policy().predict_action_chunk({PERCEPTRON_ISAAC_STREAM_KEY: FakePackedStream(), "task": "pick"})

    torch.testing.assert_close(first, second)


def test_native_action_postprocessor_unnormalizes_and_clips_pose(tmp_path):
    stats_path = tmp_path / "stats.json"
    _write_native_stats(stats_path, action_q01=0.0, action_q99=4.0)
    step = PerceptronIsaacActionUnnormalizeProcessorStep(
        stats_path=str(stats_path),
        clip_normalized_max=10.0,
        clip_action_pose=True,
    )
    transition = {
        TransitionKey.OBSERVATION: None,
        TransitionKey.ACTION: torch.zeros(1, 7),
        TransitionKey.REWARD: 0.0,
        TransitionKey.DONE: False,
        TransitionKey.TRUNCATED: False,
        TransitionKey.INFO: {},
        TransitionKey.COMPLEMENTARY_DATA: {},
    }

    out = step(transition)

    assert out is not transition
    expected = np.asarray([[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0]], dtype=np.float32)
    np.testing.assert_allclose(out[TransitionKey.ACTION].numpy(), expected, atol=1e-6)


def test_native_action_postprocessor_can_remap_binary_gripper(tmp_path):
    stats_path = tmp_path / "stats.json"
    _write_native_stats(stats_path, action_q01=0.0, action_q99=1.0)
    step = PerceptronIsaacActionUnnormalizeProcessorStep(
        stats_path=str(stats_path),
        clip_action_pose=False,
        gripper_binary_to_signed=True,
    )
    transition = {
        TransitionKey.OBSERVATION: None,
        TransitionKey.ACTION: torch.tensor(
            [
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0],
            ]
        ),
        TransitionKey.REWARD: 0.0,
        TransitionKey.DONE: False,
        TransitionKey.TRUNCATED: False,
        TransitionKey.INFO: {},
        TransitionKey.COMPLEMENTARY_DATA: {},
    }

    out = step(transition)[TransitionKey.ACTION].numpy()

    # Tolerance covers the +eps term Genesis carries in the unnormalize denominator.
    np.testing.assert_allclose(out[:, -1], [-1.0, 1.0], atol=1e-5)


def test_native_settle_action_is_normalized_for_postprocessor(tmp_path):
    stats_path = tmp_path / "stats.json"
    _write_native_stats(stats_path, action_q01=-2.0, action_q99=2.0)
    cfg = PerceptronIsaacConfig(
        device="cpu",
        native_stats_path=str(stats_path),
        num_settle_steps=1,
        settle_gripper=-1.0,
        input_features=_features(),
        output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(7,))},
    )
    policy = PerceptronIsaacPolicy(cfg)
    post = PerceptronIsaacActionUnnormalizeProcessorStep(stats_path=str(stats_path), clip_action_pose=True)

    normalized_idle = policy.select_action({PERCEPTRON_ISAAC_STREAM_KEY: FakePackedStream()})
    robot_transition = post(
        {
            TransitionKey.OBSERVATION: None,
            TransitionKey.ACTION: normalized_idle,
            TransitionKey.REWARD: 0.0,
            TransitionKey.DONE: False,
            TransitionKey.TRUNCATED: False,
            TransitionKey.INFO: {},
            TransitionKey.COMPLEMENTARY_DATA: {},
        }
    )
    robot_idle = robot_transition[TransitionKey.ACTION]

    np.testing.assert_allclose(normalized_idle.numpy(), [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.5]], atol=1e-6)
    np.testing.assert_allclose(robot_idle.numpy(), [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0]], atol=1e-6)


def test_policy_training_requires_rendered_stream_and_checkpoint(tmp_path):
    stats_path = tmp_path / "stats.json"
    _write_native_stats(stats_path)
    cfg = PerceptronIsaacConfig(
        device="cpu",
        native_stats_path=str(stats_path),
        input_features=_features(),
        output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(7,))},
    )
    policy = PerceptronIsaacPolicy(cfg)

    with pytest.raises(ValueError, match="requires the native preprocessor output"):
        policy.forward({})
    with pytest.raises(RuntimeError, match="hf_model_path is required"):
        policy.get_optim_params()


def test_checkpoint_local_sidecar_paths_resolve_relative_to_checkpoint(tmp_path):
    (tmp_path / "hf_model").mkdir()
    (tmp_path / "stats.json").write_text("{}")
    (tmp_path / "native_render.json").write_text("{}")
    cfg = PerceptronIsaacConfig(
        device="cpu",
        hf_model_path="hf_model",
        stats_path="stats.json",
        native_render_metadata_path="native_render.json",
        native_stats_path="stats.json",
        input_features=_features(),
        output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(7,))},
    )

    PerceptronIsaacPolicy._resolve_checkpoint_local_paths(cfg, tmp_path)

    assert cfg.hf_model_path == str(tmp_path / "hf_model")
    assert cfg.stats_path == str(tmp_path / "stats.json")
    assert cfg.native_render_metadata_path == str(tmp_path / "native_render.json")
    assert cfg.native_stats_path == str(tmp_path / "stats.json")


@pytest.mark.parametrize("unsafe_path", ["../outside.json", "/tmp/outside.json"])
def test_checkpoint_local_sidecar_paths_reject_escape_and_absolute_paths(tmp_path, unsafe_path):
    outside = tmp_path.parent / "outside.json"
    outside.write_text("{}")
    cfg = PerceptronIsaacConfig(
        device="cpu",
        native_stats_path=unsafe_path,
        input_features=_features(),
        output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(7,))},
    )

    with pytest.raises(RuntimeError, match="must be relative|escapes the package root"):
        PerceptronIsaacPolicy._resolve_checkpoint_local_paths(cfg, tmp_path)


def test_checkpoint_local_sidecar_paths_reject_missing_asset(tmp_path):
    cfg = PerceptronIsaacConfig(
        device="cpu",
        native_stats_path="missing.json",
        input_features=_features(),
        output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(7,))},
    )

    with pytest.raises(RuntimeError, match="does not exist"):
        PerceptronIsaacPolicy._resolve_checkpoint_local_paths(cfg, tmp_path)


def test_checkpoint_local_model_directory_rejects_symlink_escape(tmp_path):
    outside = tmp_path.parent / f"{tmp_path.name}-outside-model"
    outside.mkdir()
    (tmp_path / "hf_model").symlink_to(outside, target_is_directory=True)
    cfg = PerceptronIsaacConfig(
        device="cpu",
        hf_model_path="hf_model",
        input_features=_features(),
        output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(7,))},
    )

    with pytest.raises(RuntimeError, match="must reference a directory"):
        PerceptronIsaacPolicy._resolve_checkpoint_local_paths(cfg, tmp_path)


def test_from_pretrained_resolves_hub_snapshot_and_loads_weights(tmp_path, monkeypatch):
    from safetensors.torch import save_file

    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    (snapshot / "hf_model").mkdir()
    (snapshot / "stats.json").write_text("{}")
    adapter = snapshot / "isaac_deployment_adapter.json"
    adapter.write_text("{}")
    save_file({"_isaac_model.weight": torch.full((1, 1), 7.0)}, snapshot / "model.safetensors")

    snapshot_calls = []

    def _snapshot_download(**kwargs):
        snapshot_calls.append(kwargs)
        return str(snapshot)

    def _load_backbone(policy, *, training=False):
        # A dense finetune resume must materialize FP32 trainable storage before
        # applying the saved outer state; BF16-first loading loses master-weight bits.
        assert training is True
        policy._isaac_model = torch.nn.Linear(1, 1, bias=False)

    monkeypatch.setattr("huggingface_hub.snapshot_download", _snapshot_download)
    monkeypatch.setattr(PerceptronIsaacPolicy, "_load_backbone", _load_backbone)
    monkeypatch.setattr(PerceptronIsaacPolicy, "_verify_packaged_contract_digests", lambda *_args: None)

    cfg = PerceptronIsaacConfig(
        device="cpu",
        hf_model_path="hf_model",
        native_stats_path="stats.json",
        deployment_adapter_sha256=hashlib.sha256(adapter.read_bytes()).hexdigest(),
        input_features=_features(),
        output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(7,))},
    )
    policy = PerceptronIsaacPolicy.from_pretrained(
        "perceptron/example",
        config=cfg,
        revision="test-revision",
        local_files_only=True,
    )

    assert snapshot_calls == [
        {
            "repo_id": "perceptron/example",
            "revision": "test-revision",
            "cache_dir": None,
            "force_download": False,
            "token": None,
            "local_files_only": True,
        }
    ]
    assert cfg.hf_model_path == str(snapshot / "hf_model")
    assert cfg.native_stats_path == str(snapshot / "stats.json")
    torch.testing.assert_close(policy._isaac_model.weight, torch.full((1, 1), 7.0))


def test_deprecated_bridge_config_fields_are_ignored_and_stats_promoted():
    cfg = PerceptronIsaacConfig(
        device="cpu",
        config_toml_path="/legacy/config.toml",
        stats_path="/checkpoint/stats.json",
        per_suite_stats_dir="/legacy/persuite",
        input_features=_features(),
        output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(7,))},
    )

    assert cfg.config_toml_path is None
    assert cfg.per_suite_stats_dir is None
    assert cfg.stats_path == "/checkpoint/stats.json"
    assert cfg.native_stats_path == "/checkpoint/stats.json"


def test_native_mharmony_metadata_validates_against_config(tmp_path):
    cfg = PerceptronIsaacConfig(
        device="cpu",
        input_features=_features(),
        output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(7,))},
    )
    metadata = IsaacMharmonyRenderMetadata.from_config(cfg, stats_fps=20.0)
    metadata.validate_for_config(cfg)
    metadata_path = tmp_path / "render_metadata.json"
    metadata_path.write_text(json.dumps(metadata.to_json_dict()))
    cfg.native_render_metadata_path = str(metadata_path)

    loaded = load_native_render_metadata(cfg)

    assert loaded.schema == "perceptron_isaac_mharmony_render_v1"
    assert loaded.camera_order == ["image", "wrist_image"]
    assert loaded.image_size == [256, 256]
    assert loaded.target_fps == 20.0
    assert loaded.dataset_name == "libero"
    assert loaded.robot_type == "generic"


def test_native_mharmony_metadata_rejects_package_fps_mismatch():
    cfg = PerceptronIsaacConfig(
        device="cpu",
        target_fps=30.0,
        input_features=_features(),
        output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(7,))},
    )
    metadata = IsaacMharmonyRenderMetadata.from_config(cfg, stats_fps=20.0)

    with pytest.raises(ValueError, match="target_fps"):
        metadata.validate_for_config(cfg)


def test_inference_render_rejects_stream_over_checkpoint_token_budget(monkeypatch):
    step = PerceptronIsaacMharmonyPackProcessorStep(
        image_size=(16, 16),
        n_obs_steps=1,
        device="cpu",
        dtype="float32",
        train_max_sequence_length=4,
    )
    fake_renderer = _install_fake_native_renderer(step, monkeypatch)
    fake_renderer.sequence_length = 5

    with pytest.raises(ValueError, match="checkpoint-trained max_sequence_length=4"):
        step._render_streams(
            windows=[[{"images": {}, "proprio": np.zeros(8, dtype=np.float32)}]],
            tasks=["pick up the mug"],
            actions=None,
            complementary={},
            anchors=[0.0],
        )


def test_native_mharmony_content_plan_matches_libero_prompt_contract(tmp_path):
    stats_path = tmp_path / "stats.json"
    _write_native_stats(stats_path)
    stats = load_isaac_stats(stats_path)
    metadata = IsaacMharmonyRenderMetadata(
        dataset_name="libero",
        robot_type="generic",
        target_fps=20.0,
    )
    observation_window = [
        {
            "images": {
                "image": np.zeros((4, 4, 3), dtype=np.uint8),
                "wrist_image": np.ones((4, 4, 3), dtype=np.uint8),
            },
            "proprio": np.zeros(8, dtype=np.float32),
        }
        for _ in range(3)
    ]

    plan = build_isaac_mharmony_content_plan(
        observation_window=observation_window,
        prompt="pick up the mug",
        metadata=metadata,
        stats=stats,
        anchor_timestamp_seconds=0.1,
    )

    expected_preamble = "\n".join(
        [
            "pick up the mug",
            "Configuration:",
            "form: Flow",
            "fps: 20",
            "control_mode: ee",
            "camera_views: [primary, wrist]",
        ]
    )
    assert plan.preamble == expected_preamble
    assert [item["kind"] for item in plan.user_content] == [
        "text",
        "timestamp",
        "image",
        "image",
        "timestamp",
        "image",
        "image",
        "timestamp",
        "image",
        "image",
        "vector",
    ]
    np.testing.assert_allclose(plan.observation_timestamps_seconds, [0.0, 0.05, 0.1])
    timestamp_items = [item for item in plan.user_content if item["kind"] == "timestamp"]
    assert [item["text"] for item in timestamp_items] == [
        "<0.00 seconds>",
        "<0.05 seconds>",
        "<0.10 seconds>",
    ]
    image_items = [item for item in plan.user_content if item["kind"] == "image"]
    assert [item["camera"] for item in image_items] == ["image", "wrist_image"] * 3
    assert [item["media_id"] for item in image_items] == [f"image_{index}" for index in range(6)]
    vector_item = next(item for item in plan.user_content if item["kind"] == "vector")
    np.testing.assert_allclose(vector_item["values"], np.zeros(8), atol=1e-5)
    action_tags = plan.assistant_content[0]["tags"]
    assert action_tags["token_group"] == "flow_action"
    assert action_tags["genesis_text_type"] == "action_c"
    assert action_tags["obs_window"] == [0, 2]
    assert action_tags["action_window"] == [2, 31]
    assert action_tags["action_indices"][0] == 2
    assert action_tags["action_indices"][-1] == 31
    assert action_tags["action_horizon"] == 30
    assert action_tags["action_dim"] == 7
    assert np.asarray(action_tags["action_target"]).shape == (30, 7)

    episode_start = build_isaac_mharmony_content_plan(
        observation_window=observation_window,
        prompt="pick up the mug",
        metadata=metadata,
        stats=stats,
        anchor_timestamp_seconds=0.0,
    )
    assert episode_start.anchor_timestamp_seconds == 0.0
    np.testing.assert_allclose(episode_start.observation_timestamps_seconds, [0.0, 0.0, 0.0])


def _libero_content_plan_window() -> list[dict[str, object]]:
    return [
        {
            "images": {
                "image": np.zeros((4, 4, 3), dtype=np.uint8),
                "wrist_image": np.ones((4, 4, 3), dtype=np.uint8),
            },
            "proprio": np.zeros(8, dtype=np.float32),
        }
        for _ in range(3)
    ]


def test_native_mharmony_places_historical_fast_actions_before_next_observation(tmp_path):
    stats_path = tmp_path / "stats.json"
    _write_native_stats(stats_path)
    plan = build_isaac_mharmony_content_plan(
        observation_window=_libero_content_plan_window(),
        prompt="pick up the mug",
        metadata=IsaacMharmonyRenderMetadata(action_conditioning=True),
        stats=load_isaac_stats(stats_path),
        anchor_timestamp_seconds=0.1,
        historical_fast_action_tokens=[None, [17, 18], [19]],
    )

    kinds = [item["kind"] for item in plan.user_content]
    assert kinds == [
        "text",
        "timestamp",
        "image",
        "image",
        "tokens",
        "timestamp",
        "image",
        "image",
        "tokens",
        "timestamp",
        "image",
        "image",
        "vector",
    ]
    action_items = [item for item in plan.user_content if item["kind"] == "tokens"]
    assert [item["tokens"] for item in action_items] == [[17, 18], [19]]
    assert [item["tags"]["action_conditioning_observation_window"] for item in action_items] == [
        [0, 1],
        [1, 2],
    ]
    assert all(item["tags"]["token_group"] == "fast_action" for item in action_items)


def test_native_mharmony_encodes_normalized_executed_actions_as_fast_history():
    class RecordingFastProcessor:
        def __init__(self):
            self.actions = []

        def __call__(self, actions):
            self.actions.append(np.asarray(actions))
            return [[23, 24]]

    processor = RecordingFastProcessor()
    window = _libero_content_plan_window()
    window[1]["previous_action"] = np.arange(7, dtype=np.float32)

    tokens = mharmony_native.encode_isaac_historical_fast_actions(
        observation_window=window,
        metadata=IsaacMharmonyRenderMetadata(action_conditioning=True),
        fast_processor=processor,
    )

    assert tokens == [None, [23, 24], None]
    assert len(processor.actions) == 1
    np.testing.assert_allclose(processor.actions[0], np.arange(7, dtype=np.float32).reshape(1, 1, 7))


def test_native_mharmony_renders_mistake_false_when_checkpoint_trained_with_support(tmp_path):
    stats_path = tmp_path / "stats.json"
    _write_native_stats(stats_path)
    plan = build_isaac_mharmony_content_plan(
        observation_window=_libero_content_plan_window(),
        prompt="pick up the mug",
        metadata=IsaacMharmonyRenderMetadata(mistake_conditioning=True),
        stats=load_isaac_stats(stats_path),
        anchor_timestamp_seconds=0.1,
    )

    assert "mistake: false" in plan.preamble.splitlines()


def test_native_mharmony_metadata_rejects_prompt_contract_mismatch():
    cfg = PerceptronIsaacConfig(
        device="cpu",
        input_features=_features(),
        output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(7,))},
    )
    metadata = IsaacMharmonyRenderMetadata(camera_order=["wrist_image", "image"])

    with pytest.raises(ValueError, match="camera_order"):
        metadata.validate_for_config(cfg)


def test_native_mharmony_metadata_requires_robot_type():
    cfg = PerceptronIsaacConfig(
        device="cpu",
        input_features=_features(),
        output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(7,))},
    )
    metadata = IsaacMharmonyRenderMetadata(robot_type=None)

    with pytest.raises(ValueError, match="robot_type"):
        metadata.validate_for_config(cfg)


def test_native_mharmony_import_helper_allows_only_mharmony_namespace():
    with pytest.raises(ImportError, match="mharmony"):
        import_mharmony_module("genesis.inference.flow_matching.observation")
    with pytest.raises(ImportError, match="mharmony"):
        import_mharmony_module("mharmony_anything")


def test_perceptron_isaac_forbids_all_genesis_imports():
    root = Path(__file__).parents[3]
    isaac_root = root / "src/lerobot/policies/perceptron_isaac"
    offenders = []
    for path in isaac_root.rglob("*.py"):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                names = [node.module] if node.module else []
            elif isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            else:
                names = []
            for module_name in names:
                if module_name.startswith("genesis"):
                    offenders.append(f"{path}:{node.lineno}:{module_name}")

            if not isinstance(node, ast.Call) or not node.args:
                continue
            function_name = None
            if isinstance(node.func, ast.Attribute):
                function_name = node.func.attr
            elif isinstance(node.func, ast.Name):
                function_name = node.func.id
            if function_name not in {"import_module", "__import__"}:
                continue
            first_arg = node.args[0]
            if not isinstance(first_arg, ast.Constant) or not isinstance(first_arg.value, str):
                continue
            module_name = first_arg.value
            if module_name.startswith("genesis"):
                offenders.append(f"{path}:{node.lineno}:{module_name}")

    assert offenders == []


def _suite_stats_block(*, proprio_q01: float, proprio_q99: float) -> dict:
    return {
        "schema": "flow_matching_stats_v1",
        "target_fps": 20.0,
        "action_horizon": 10,
        "action_dim": 7,
        "proprio_dim": 8,
        "action": {"q01": [-1.0] * 7, "q99": [1.0] * 7},
        "proprio": {"q01": [proprio_q01] * 8, "q99": [proprio_q99] * 8},
    }


def test_training_normalization_is_routed_per_libero_suite(monkeypatch, tmp_path):
    """Each sample must be normalized by its own suite's quantiles, keyed on task_index.

    LIBERO suites are not interchangeable: end-effector height spans [0.916, 1.286] in
    libero_spatial but [0.010, 0.337] in libero_object. Those ranges are disjoint, so applying
    one suite's stats to another drives proprio far outside [-1, 1] rather than filling it.
    """
    stats_path = tmp_path / "suite_stats.json"
    stats_path.write_text(
        json.dumps(
            {
                "libero_spatial": _suite_stats_block(proprio_q01=0.916, proprio_q99=1.286),
                "libero_object": _suite_stats_block(proprio_q01=0.010, proprio_q99=0.337),
            }
        )
    )
    map_path = tmp_path / "suite_by_task_index.json"
    # 30 -> spatial and 20 -> object, matching MolmoAct2-LIBERO's 4 x 10 block layout.
    map_path.write_text(json.dumps({"30": "libero_spatial", "20": "libero_object"}))

    step = PerceptronIsaacMharmonyPackProcessorStep(
        enabled=True,
        camera_order=["image", "wrist_image"],
        image_size=(16, 16),
        n_obs_steps=1,
        chunk_size=10,
        device="cpu",
        dtype="float32",
        suite_stats_path=str(stats_path),
        suite_by_task_index_path=str(map_path),
    )
    renderer = _install_fake_native_renderer(step, monkeypatch)
    step._fast_processor = object()

    def _obs(n):
        return {
            "observation.images.image": torch.zeros(n, 1, 3, 16, 16),
            "observation.images.image2": torch.ones(n, 1, 3, 16, 16),
            OBS_STATE: torch.zeros(n, 1, 8),
        }

    step(
        {
            TransitionKey.OBSERVATION: _obs(2),
            TransitionKey.ACTION: torch.zeros(2, 10, 7),
            TransitionKey.REWARD: 0.0,
            TransitionKey.DONE: False,
            TransitionKey.TRUNCATED: False,
            TransitionKey.INFO: {},
            TransitionKey.COMPLEMENTARY_DATA: {
                "task": ["pick up the black bowl", "pick up the ketchup"],
                "frame_index": torch.tensor([4, 4]),
                "task_index": torch.tensor([30, 20]),
            },
        }
    )

    assert len(renderer.build_training_calls) == 2
    spatial_stats = renderer.build_training_calls[0]["stats"]
    object_stats = renderer.build_training_calls[1]["stats"]
    assert spatial_stats is not None and object_stats is not None
    # Routed by task_index, not by position or by the builder's single default.
    assert spatial_stats.proprio.q01[0] == pytest.approx(0.916)
    assert object_stats.proprio.q01[0] == pytest.approx(0.010)


def test_multi_suite_normalization_fails_fast_for_serving(tmp_path):
    """A multi-suite training package has no single correct serving normalization frame."""
    step = PerceptronIsaacMharmonyPackProcessorStep(
        enabled=True,
        camera_order=["image", "wrist_image"],
        image_size=(16, 16),
        n_obs_steps=1,
        chunk_size=10,
        device="cpu",
        dtype="float32",
        suite_stats_path=str(tmp_path / "suite_stats.json"),
        suite_by_task_index_path=str(tmp_path / "suite_by_task_index.json"),
    )

    with pytest.raises(RuntimeError, match="training-only.*single-suite serving package"):
        step(
            {
                TransitionKey.OBSERVATION: {
                    "observation.images.image": torch.zeros(1, 1, 3, 16, 16),
                    "observation.images.image2": torch.ones(1, 1, 3, 16, 16),
                    OBS_STATE: torch.zeros(1, 1, 8),
                },
                TransitionKey.ACTION: None,
                TransitionKey.REWARD: 0.0,
                TransitionKey.DONE: False,
                TransitionKey.TRUNCATED: False,
                TransitionKey.INFO: {},
                TransitionKey.COMPLEMENTARY_DATA: {"task": "pick"},
            }
        )


def test_per_suite_normalization_requires_both_paths(tmp_path):
    with pytest.raises(ValueError, match="suite_stats_path and suite_by_task_index_path together"):
        PerceptronIsaacMharmonyPackProcessorStep(
            enabled=True,
            suite_stats_path=str(tmp_path / "only_stats.json"),
        )


def test_per_suite_table_must_agree_with_the_checkpoint_geometry(tmp_path):
    """Per-suite blocks replace the package's own stats, so their geometry must still match.

    Differing quantiles are the point -- LIBERO suites are not interchangeable -- but a
    block whose action_horizon is not the chunk the model emits means the run trains on a
    contract the checkpoint cannot serve. Caught at the table load, on the training render
    path, which is per-suite routing's only consumer.
    """
    block = _suite_stats_block(proprio_q01=0.9, proprio_q99=1.3)
    block["action_horizon"] = 16
    stats_path = tmp_path / "suite_stats.json"
    stats_path.write_text(json.dumps({"libero_spatial": block}))
    map_path = tmp_path / "suite_by_task_index.json"
    map_path.write_text(json.dumps({"30": "libero_spatial"}))

    step = PerceptronIsaacMharmonyPackProcessorStep(
        enabled=True,
        camera_order=["image", "wrist_image"],
        image_size=(16, 16),
        n_obs_steps=1,
        chunk_size=10,
        device="cpu",
        dtype="float32",
        suite_stats_path=str(stats_path),
        suite_by_task_index_path=str(map_path),
    )
    with pytest.raises(ValueError, match=r"action_horizon 16 != chunk_size 10"):
        step._ensure_suite_stats()


def test_per_suite_normalization_rejects_unmapped_task_index(monkeypatch, tmp_path):
    stats_path = tmp_path / "suite_stats.json"
    stats_path.write_text(
        json.dumps({"libero_spatial": _suite_stats_block(proprio_q01=0.9, proprio_q99=1.3)})
    )
    map_path = tmp_path / "suite_by_task_index.json"
    map_path.write_text(json.dumps({"30": "libero_spatial"}))

    step = PerceptronIsaacMharmonyPackProcessorStep(
        enabled=True,
        camera_order=["image", "wrist_image"],
        image_size=(16, 16),
        n_obs_steps=1,
        chunk_size=10,
        device="cpu",
        dtype="float32",
        suite_stats_path=str(stats_path),
        suite_by_task_index_path=str(map_path),
    )
    _install_fake_native_renderer(step, monkeypatch)
    step._fast_processor = object()

    # An unmapped suite must fail loudly rather than silently borrow the wrong quantiles.
    with pytest.raises(ValueError, match="no LIBERO suite mapping"):
        step(
            {
                TransitionKey.OBSERVATION: {
                    "observation.images.image": torch.zeros(1, 1, 3, 16, 16),
                    "observation.images.image2": torch.ones(1, 1, 3, 16, 16),
                    OBS_STATE: torch.zeros(1, 1, 8),
                },
                TransitionKey.ACTION: torch.zeros(1, 10, 7),
                TransitionKey.REWARD: 0.0,
                TransitionKey.DONE: False,
                TransitionKey.TRUNCATED: False,
                TransitionKey.INFO: {},
                TransitionKey.COMPLEMENTARY_DATA: {
                    "task": ["something else"],
                    "frame_index": torch.tensor([0]),
                    "task_index": torch.tensor([7]),
                },
            }
        )


def test_letterbox_geometry_conform_matches_camera_primitive():
    """Off-size frames conform via the shared letterbox primitive, both memory layouts."""
    from lerobot.transforms import letterbox_resize_image

    step = PerceptronIsaacMharmonyPackProcessorStep(
        camera_order=["image"],
        image_keys=["observation.images.image"],
        image_size=(256, 256),
        image_preprocessing="letterbox",
        allow_image_key_fallback=False,
    )
    rng = np.random.default_rng(0)
    frame = rng.integers(0, 256, size=(480, 640, 3), dtype=np.uint8)

    hwc = step._extract_image_tensor({"observation.images.image": frame[None, ...]}, batch_size=1)
    assert hwc.shape == (1, 1, 1, 256, 256, 3)
    reference = (
        letterbox_resize_image(torch.from_numpy(frame).permute(2, 0, 1), (256, 256)).permute(1, 2, 0).numpy()
    )
    np.testing.assert_array_equal(hwc[0, 0, 0], reference)
    # 480x640 letterboxed into 256x256 lands a 192-row band with 32 pad rows above and below.
    assert not hwc[0, 0, 0, :32].any() and not hwc[0, 0, 0, -32:].any()

    chw = step._extract_image_tensor(
        {"observation.images.image": np.moveaxis(frame, -1, 0)[None, ...].astype(np.float32) / 255.0},
        batch_size=1,
    )
    assert chw.shape == (1, 1, 1, 3, 256, 256)


def test_stretch_geometry_conform_and_contract_size_passthrough():
    step = PerceptronIsaacMharmonyPackProcessorStep(
        camera_order=["image"],
        image_keys=["observation.images.image"],
        image_size=(256, 256),
        allow_image_key_fallback=False,
    )
    assert step.image_preprocessing == "stretch"
    off_size = np.zeros((480, 640, 3), dtype=np.uint8)
    stretched = step._extract_image_tensor({"observation.images.image": off_size[None, ...]}, batch_size=1)
    assert stretched.shape == (1, 1, 1, 256, 256, 3)

    rng = np.random.default_rng(1)
    exact = rng.integers(0, 256, size=(256, 256, 3), dtype=np.uint8)
    passthrough = step._extract_image_tensor({"observation.images.image": exact[None, ...]}, batch_size=1)
    # Contract-size input must pass through byte-identical: the serving parity gates feed
    # camera-resized frames and may not observe any resample.
    np.testing.assert_array_equal(passthrough[0, 0, 0], exact)


def test_image_preprocessing_round_trips_and_rejects_unknown_modes():
    step = PerceptronIsaacMharmonyPackProcessorStep(image_preprocessing="letterbox")
    assert step.get_config()["image_preprocessing"] == "letterbox"
    with pytest.raises(ValueError, match="image_preprocessing"):
        PerceptronIsaacMharmonyPackProcessorStep(image_preprocessing="crop")


def test_dataset_stats_override_transforms_quantiles():
    from lerobot.policies.perceptron_isaac.processor_perceptron_isaac import _dataset_stats_to_isaac_stats

    config = PerceptronIsaacConfig(
        action_dim=2,
        proprio_dim=2,
        chunk_size=4,
        n_action_steps=4,
        target_fps=30.0,
        joint_signs=[-1.0, 1.0],
        joint_offsets=[10.0, 2.0],
    )
    stats = _dataset_stats_to_isaac_stats(
        {
            "action": {"q01": torch.tensor([1.0, 2.0]), "q99": torch.tensor([3.0, 4.0])},
            "observation.state": {"q01": torch.tensor([5.0, 6.0]), "q99": torch.tensor([7.0, 8.0])},
        },
        config,
        dataset_fps=20.0,
    )
    torch.testing.assert_close(torch.from_numpy(stats.action.q01), torch.tensor([7.0, 4.0]))
    torch.testing.assert_close(torch.from_numpy(stats.action.q99), torch.tensor([9.0, 6.0]))
    # The package contract wins over a dataset's declared rate. Converted LIBERO datasets
    # are known to declare 10 Hz for demonstrations recorded at 20 Hz.
    assert stats.target_fps == 30.0
    assert stats.action_horizon == 4


def test_dataset_stats_fall_back_to_dataset_fps_without_contract_fps():
    from lerobot.policies.perceptron_isaac.processor_perceptron_isaac import _dataset_stats_to_isaac_stats

    config = PerceptronIsaacConfig(
        action_dim=2,
        proprio_dim=2,
        chunk_size=4,
        n_action_steps=4,
        target_fps=None,
        joint_signs=[1.0, 1.0],
        joint_offsets=[0.0, 0.0],
    )
    stats = _dataset_stats_to_isaac_stats(
        {
            "action": {"q01": torch.tensor([1.0, 2.0]), "q99": torch.tensor([3.0, 4.0])},
            "observation.state": {"q01": torch.tensor([5.0, 6.0]), "q99": torch.tensor([7.0, 8.0])},
        },
        config,
        dataset_fps=20.0,
    )

    assert stats.target_fps == 20.0


# --- gripper passthrough (MolmoAct2 `normalize_gripper: false` parity) ---------------------

_GRIPPER_NAMES = (
    [f"left_joint_{i}.pos" for i in range(6)]
    + ["left_gripper.pos"]
    + [f"right_joint_{i}.pos" for i in range(6)]
    + ["right_gripper.pos"]
)


def _bi_yam_stats() -> IsaacNormalizationStats:
    """A 14-dim bundle whose grippers span [0, 1] and whose joints span [-1, 1]."""
    q01 = np.array([-1.0] * 6 + [0.05] + [-1.0] * 6 + [0.05], dtype=np.float32)
    q99 = np.array([1.0] * 6 + [0.95] + [1.0] * 6 + [0.95], dtype=np.float32)
    block = IsaacStatsBlock(q01=q01, q99=q99, minimum=q01 - 0.5, maximum=q99 + 0.5)
    return IsaacNormalizationStats(
        action=block,
        proprio=block,
        target_fps=30.0,
        action_horizon=30,
        schema="flow_matching_stats_v1",
        profile_id="molmoact2_bimanualyam/test/Flow",
        profile_scope="test",
        validation_status="validated",
    )


def _masked_stats() -> IsaacNormalizationStats:
    return apply_gripper_passthrough(
        _bi_yam_stats(),
        action_feature_names=_GRIPPER_NAMES,
        state_feature_names=_GRIPPER_NAMES,
    )


def test_gripper_passthrough_leaves_gripper_dims_in_physical_units():
    base, masked = _bi_yam_stats(), _masked_stats()
    state = np.linspace(0.1, 0.9, 14).astype(np.float32)
    grip = [6, 13]
    joints = [i for i in range(14) if i not in grip]

    normalized = normalize_isaac_proprio(state, base.proprio)
    passthrough = normalize_isaac_proprio(state, masked.proprio)

    assert np.allclose(passthrough[grip], state[grip])
    # The gripper actually moves under normalization, so this is not a vacuous assertion.
    assert not np.allclose(normalized[grip], state[grip])
    assert np.allclose(passthrough[joints], normalized[joints])


def test_gripper_passthrough_applies_to_action_chunks_and_survives_unnormalize():
    base, masked = _bi_yam_stats(), _masked_stats()
    chunk = np.tile(np.linspace(0.1, 0.9, 14, dtype=np.float32), (4, 1))
    grip = [6, 13]

    normalized = normalize_isaac_actions(chunk, masked.action)
    assert np.allclose(normalized[:, grip], chunk[:, grip])
    assert np.allclose(normalized[:, [0]], normalize_isaac_actions(chunk, base.action)[:, [0]])

    restored = unnormalize_isaac_actions(normalized, masked.action, clip=True, clip_normalized_max=1.0)
    # Gripper columns are already inside the clip, so the passthrough round-trips exactly.
    assert np.allclose(restored[:, grip], chunk[:, grip])
    assert np.allclose(restored, chunk, atol=1e-4)


def test_gripper_mask_survives_the_checkpoint_round_trip():
    """Eval must inherit the mask from the checkpoint, not re-derive it from a serving flag."""
    masked = _masked_stats()
    restored = isaac_stats_from_state_dict(isaac_stats_to_state_dict(masked))
    for block in ("action", "proprio"):
        original = getattr(masked, block).mask
        assert np.array_equal(getattr(restored, block).mask, original)
        assert [i for i, keep in enumerate(original) if not keep] == [6, 13]


def test_unmasked_stats_emit_no_mask_key():
    """Checkpoints written before the mask existed must round-trip unchanged."""
    state = isaac_stats_to_state_dict(_bi_yam_stats())
    assert not [key for key in state if key.endswith(".mask")]
    assert isaac_stats_from_state_dict(state).action.mask is None


def test_all_true_gripper_mask_is_normalized_away():
    block = _bi_yam_stats().action
    same = IsaacStatsBlock(q01=block.q01, q99=block.q99, mask=np.ones(14, dtype=bool))
    assert same.mask is None


@pytest.mark.parametrize(
    ("names", "match"),
    [
        ([f"joint_{i}.pos" for i in range(14)], "No gripper dimension"),
        (_GRIPPER_NAMES[:13], "cover 13 dims"),
    ],
)
def test_gripper_passthrough_rejects_unusable_feature_names(names, match):
    with pytest.raises(ValueError, match=match):
        apply_gripper_passthrough(_bi_yam_stats(), action_feature_names=names, state_feature_names=names)


_ARM_NAMES = [f"joint_{i}.pos" for i in range(6)] + ["gripper.pos"]


def _dummy_dataset_stats(dim: int = 7):
    q01 = torch.linspace(-1.0, 1.0, dim)
    q99 = q01 + 2.0
    block = {"q01": q01, "q99": q99, "min": q01 - 1.0, "max": q99 + 1.0}
    return {"action": dict(block), OBS_STATE: dict(block)}


def _serialized_masks(*pipelines):
    found = {}
    for label, pipe in pipelines:
        for step in pipe.steps:
            for key, value in (getattr(step, "state_dict", lambda: {})() or {}).items():
                if key.endswith(".mask"):
                    found[f"{label}:{key}"] = [i for i, keep in enumerate(value.bool().tolist()) if not keep]
    return found


@pytest.mark.parametrize("normalize_gripper", [True, False])
def test_gripper_passthrough_reaches_the_dataset_stats_path(normalize_gripper):
    """Finetuning builds stats from dataset quantiles, not from a stats path.

    `_install_dataset_stats` assigns `step._stats` directly, so a mask applied only in the
    path-load branch never reaches training or the saved checkpoint: the config would
    record normalize_gripper=False while the grippers were normalized anyway. This covers
    the branch `lerobot_train` actually takes.
    """
    cfg = PerceptronIsaacConfig(
        device="cpu",
        action_dim=7,
        proprio_dim=7,
        input_features={
            "observation.images.image": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 256, 256)),
            OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(7,)),
        },
        output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(7,))},
        action_feature_names=list(_ARM_NAMES),
        state_feature_names=list(_ARM_NAMES),
        normalize_gripper=normalize_gripper,
    )
    preprocessor, postprocessor = make_perceptron_isaac_pre_post_processors(
        cfg, dataset_stats=_dummy_dataset_stats()
    )
    masks = _serialized_masks(("pre", preprocessor), ("post", postprocessor))
    if normalize_gripper:
        assert masks == {}
    else:
        assert masks == {
            "pre:action.mask": [6],
            "pre:proprio.mask": [6],
            "post:action.mask": [6],
            "post:proprio.mask": [6],
        }


@pytest.mark.parametrize("load_saved_processors", [False, True])
def test_dataset_state_feature_names_reach_gripper_masks(tmp_path, load_saved_processors):
    """Non-strict fine-tunes derive state names from dataset metadata on both factory paths."""
    dataset_stats = _dummy_dataset_stats()
    dataset_meta = SimpleNamespace(
        fps=20.0,
        features={OBS_STATE: {"names": {"motors": list(_ARM_NAMES)}}},
    )
    cfg = PerceptronIsaacConfig(
        device="cpu",
        action_dim=7,
        proprio_dim=7,
        normalize_gripper=False,
        input_features={
            "observation.images.image": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 256, 256)),
            OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(7,)),
        },
        output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(7,))},
        action_feature_names=list(_ARM_NAMES),
        state_feature_names=None,
    )

    if load_saved_processors:
        base_cfg = PerceptronIsaacConfig(
            device="cpu",
            action_dim=7,
            proprio_dim=7,
            input_features=cfg.input_features,
            output_features=cfg.output_features,
            action_feature_names=list(_ARM_NAMES),
        )
        base_pre, base_post = make_perceptron_isaac_pre_post_processors(base_cfg, dataset_stats=dataset_stats)
        base_pre.save_pretrained(tmp_path)
        base_post.save_pretrained(tmp_path)
        preprocessor, postprocessor = make_perceptron_isaac_pre_post_processors_from_pretrained(
            cfg,
            str(tmp_path),
            dataset_stats=dataset_stats,
            dataset_meta=dataset_meta,
        )
    else:
        preprocessor, postprocessor = make_perceptron_isaac_pre_post_processors(
            cfg,
            dataset_stats=dataset_stats,
            dataset_meta=dataset_meta,
        )

    assert cfg.state_feature_names == _ARM_NAMES
    assert _serialized_masks(("pre", preprocessor), ("post", postprocessor)) == {
        "pre:action.mask": [6],
        "pre:proprio.mask": [6],
        "post:action.mask": [6],
        "post:proprio.mask": [6],
    }
