"""P1.03: portable checkpoint configs must not borrow or mutate live asset paths.

These CPU fixtures exercise safetensors save/resume, not rendering, FSDP or a full
training job. Only backbone materialization uses the existing reduced-geometry
native loader; checkpoint writing, draccus, resume resolution and factory routing
remain real. RED execution belongs to the parent integration environment.
"""

import json
import sys
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import draccus
import numpy as np
import pytest
import torch
from accelerate import Accelerator
from safetensors.torch import load_file

pytest.importorskip("transformers")

from lerobot.common.train_utils import resume_after_prepare, resume_before_prepare, save_checkpoint
from lerobot.configs.default import DatasetConfig
from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.train import TRAIN_CONFIG_NAME, TrainPipelineConfig
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.factory import make_policy
from lerobot.policies.perceptron_isaac.checkpoint_integrity import canonical_json_sha256, file_sha256
from lerobot.policies.perceptron_isaac.configuration_perceptron_isaac import (
    SUITE_BY_TASK_INDEX_EXPORT_FILENAME,
    SUITE_STATS_EXPORT_FILENAME,
    PerceptronIsaacConfig,
)
from lerobot.policies.perceptron_isaac.isaac_stats import IsaacNormalizationStats, IsaacStatsBlock
from lerobot.policies.perceptron_isaac.modeling_perceptron_isaac import PerceptronIsaacPolicy
from lerobot.rewards.classifier.configuration_classifier import RewardClassifierConfig
from lerobot.utils.constants import ACTION, OBS_STATE, PRETRAINED_MODEL_DIR
from tests.policies.perceptron_isaac.test_checkpoint_lifecycle import (
    _load_native_fixture,
    _source_package as tiny_source_package,
)


@dataclass(frozen=True)
class _LiveConfigPaths:
    hf_model_path: str | None
    native_stats_path: str | None
    native_render_metadata_path: str | None
    fast_processor_path: str | None
    pretrained_path: Path | None

    @classmethod
    def capture(cls, config: PerceptronIsaacConfig) -> "_LiveConfigPaths":
        return cls(
            config.hf_model_path,
            config.native_stats_path,
            config.native_render_metadata_path,
            config.fast_processor_path,
            config.pretrained_path,
        )


@dataclass(frozen=True)
class _NativeRun:
    source: Path
    config: TrainPipelineConfig
    policy: PerceptronIsaacPolicy
    optimizer: torch.optim.SGD


@dataclass(frozen=True)
class _DatasetMetadata:
    features: dict[str, dict[str, Any]]
    stats: dict[str, dict[str, torch.Tensor]] = field(default_factory=dict)
    fps: int = 20


@pytest.fixture
def native_run(tmp_path: Path) -> Iterator[_NativeRun]:
    source = tmp_path / "original-package"
    config = tiny_source_package(source)
    config.pretrained_path = source
    config.push_to_hub = False
    config.camera_order = ("image",)
    policy = PerceptronIsaacPolicy(config)
    policy._isaac_model = _load_native_fixture(config)
    optimizer = torch.optim.SGD(policy.parameters(), lr=0.123, momentum=0.9)
    for parameter in policy.parameters():
        parameter.grad = torch.full_like(parameter, 0.25)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    with torch.no_grad():
        policy._isaac_model.lm_head.weight.fill_(1.2500001192092896)
        policy._isaac_model.model.vector_embedding[0].weight.fill_(-0.0)

    # Model weights are already materialized; live sidecars now have a private owner.
    policy._retain_verified_runtime_assets(source)
    assert policy._verified_runtime_assets is not None
    assert policy._verified_runtime_assets.root != source
    train_config = TrainPipelineConfig(
        dataset=DatasetConfig(repo_id="fixture/unused"),
        policy=config,
        output_dir=tmp_path / "run",
    )
    train_config.parallelism.resolve(1)
    try:
        yield _NativeRun(source, train_config, policy, optimizer)
    finally:
        if policy._verified_runtime_assets is not None:
            policy._verified_runtime_assets.cleanup()
        for snapshot in policy._verified_external_runtime_assets:
            snapshot.cleanup()


def _assert_portable_asset_paths(payload: Mapping[str, Any]) -> None:
    expected = {
        "hf_model_path": "hf_model",
        "native_stats_path": "isaac_stats.json",
        "native_render_metadata_path": "native_render_metadata.json",
        "fast_processor_path": "fast_processor",
    }
    assert {name: payload[name] for name in expected} == expected


def _assert_tensor_bits(actual: torch.Tensor, expected: torch.Tensor, label: str) -> None:
    assert actual.dtype == expected.dtype, label
    assert actual.shape == expected.shape, label
    # Floating equality alone would silently accept a lost negative-zero sign bit.
    assert torch.equal(actual.reshape(-1).view(torch.uint8), expected.reshape(-1).view(torch.uint8)), label


def _assert_native_inventory(package: Path, expected: Mapping[str, torch.Tensor]) -> None:
    model_root = package / "hf_model"
    index = json.loads((model_root / "model.safetensors.index.json").read_text())
    actual: dict[str, torch.Tensor] = {}
    for shard_name in sorted(set(index["weight_map"].values())):
        shard = load_file(model_root / shard_name)
        assert not actual.keys() & shard.keys()
        assert shard.keys() == {key for key, owner in index["weight_map"].items() if owner == shard_name}
        actual.update(shard)
    assert actual.keys() == index["weight_map"].keys() == expected.keys()
    for name, tensor in expected.items():
        _assert_tensor_bits(actual[name], tensor, name)


def test_save_checkpoint_nested_policy_uses_portable_asset_paths(
    native_run: _NativeRun, tmp_path: Path
) -> None:
    checkpoint = tmp_path / "checkpoint"
    original_paths = _LiveConfigPaths.capture(native_run.policy.config)
    tokenizer_bytes = (native_run.source / "hf_model" / "tokenizer.json").read_bytes()
    source_identity = native_run.policy.config.mk1_source_model_import_sha256

    save_checkpoint(
        checkpoint, step=17, cfg=native_run.config, policy=native_run.policy, optimizer=native_run.optimizer
    )

    package = checkpoint / PRETRAINED_MODEL_DIR
    standalone = json.loads((package / "config.json").read_text())
    nested = json.loads((package / TRAIN_CONFIG_NAME).read_text())["policy"]
    _assert_portable_asset_paths(standalone)
    _assert_portable_asset_paths(nested)
    assert standalone["type"] == nested["type"] == "perceptron_isaac"
    assert standalone["pretrained_path"] == nested["pretrained_path"] == str(native_run.source)
    assert standalone["mk1_model_import_sha256"] is nested["mk1_model_import_sha256"] is None
    assert standalone["mk1_source_model_import_sha256"] == nested["mk1_source_model_import_sha256"]
    assert nested["mk1_source_model_import_sha256"] == source_identity
    manifest_path = package / "mk1_trained_package_manifest.json"
    canonical = dict(standalone, mk1_trained_package_manifest_sha256=None)
    assert canonical == dict(nested, mk1_trained_package_manifest_sha256=None)
    assert standalone["mk1_trained_package_manifest_sha256"] == file_sha256(manifest_path)
    assert json.loads(manifest_path.read_text())["config_sha256"] == canonical_json_sha256(canonical)
    assert (package / "hf_model" / "tokenizer.json").read_bytes() == tokenizer_bytes
    assert json.loads(tokenizer_bytes)["added_tokens"], "fixture must retain reserved token identities"
    assert _LiveConfigPaths.capture(native_run.policy.config) == original_paths


@pytest.mark.parametrize("resume_lr", [None, 0.00012345], ids=["saved-policy", "cli-override"])
def test_moved_checkpoint_resumes_native_weights_and_optimizer(
    native_run: _NativeRun, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, resume_lr: float | None
) -> None:
    checkpoint = tmp_path / "checkpoint"
    original_paths = _LiveConfigPaths.capture(native_run.policy.config)
    tokenizer_bytes = (native_run.source / "hf_model" / "tokenizer.json").read_bytes()
    assert native_run.policy._isaac_model is not None
    expected_weights = {
        name: tensor.detach().clone() for name, tensor in native_run.policy._isaac_model.state_dict().items()
    }
    probe = expected_weights["lm_head.weight"]
    assert probe.dtype == torch.float32
    assert not torch.equal(probe, probe.to(torch.bfloat16).float())
    assert torch.signbit(expected_weights["model.vector_embedding.0.weight"]).all()
    save_checkpoint(
        checkpoint, step=17, cfg=native_run.config, policy=native_run.policy, optimizer=native_run.optimizer
    )
    _assert_native_inventory(checkpoint / PRETRAINED_MODEL_DIR, expected_weights)

    native_run.source.rename(tmp_path / "retired-original-package")
    moved = tmp_path / "moved-checkpoint"
    checkpoint.rename(moved)
    assert (moved / PRETRAINED_MODEL_DIR / "hf_model" / "tokenizer.json").read_bytes() == tokenizer_bytes
    saved_config = moved / PRETRAINED_MODEL_DIR / TRAIN_CONFIG_NAME
    cli_args = ["--resume=true"]
    if resume_lr is not None:
        cli_args.append(f"--policy.optimizer_lr={resume_lr}")
    monkeypatch.setattr(sys, "argv", ["lerobot-train", f"--config_path={saved_config}", *cli_args])
    resumed = TrainPipelineConfig.from_pretrained(saved_config, cli_args=cli_args)
    resumed.parallelism.resolve(1)
    resumed.validate()
    assert resumed.checkpoint_path == moved
    assert isinstance(resumed.policy, PerceptronIsaacConfig)
    package_config = json.loads((moved / PRETRAINED_MODEL_DIR / "config.json").read_text())
    assert (
        resumed.policy.mk1_trained_package_manifest_sha256
        == package_config["mk1_trained_package_manifest_sha256"]
    )
    assert resumed.policy.optimizer_lr == (package_config["optimizer_lr"] if resume_lr is None else resume_lr)
    assert resumed.policy.pretrained_path == native_run.source
    assert not resumed.policy.pretrained_path.exists()

    def load_tiny_backbone(policy: PerceptronIsaacPolicy, *, training: bool = False) -> None:
        # Same native F32 loader as the source fixture, without production-size/CPU guards.
        policy._isaac_model = _load_native_fixture(policy.config)

    monkeypatch.setattr(PerceptronIsaacPolicy, "_load_backbone", load_tiny_backbone)
    metadata = _DatasetMetadata(
        features={
            "observation.images.image": {
                "dtype": "image",
                "shape": (256, 256, 3),
                "names": ["height", "width", "channels"],
            },
            OBS_STATE: {"dtype": "float32", "shape": (2,), "names": ["joint0", "joint1"]},
            ACTION: {"dtype": "float32", "shape": (2,), "names": ["joint0", "joint1"]},
        }
    )
    reloaded = make_policy(
        resumed.policy,
        ds_meta=metadata,
        pretrained_path=resumed.checkpoint_path / PRETRAINED_MODEL_DIR,
    )
    assert isinstance(reloaded, PerceptronIsaacPolicy)
    assert reloaded.config.pretrained_path == native_run.source
    assert Path(reloaded.config._checkpoint_source_root) == moved / PRETRAINED_MODEL_DIR
    reloaded._load_backbone(training=True)
    assert reloaded._isaac_model is not None
    actual_weights = reloaded._isaac_model.state_dict()
    assert actual_weights.keys() == expected_weights.keys()
    for name, tensor in expected_weights.items():
        _assert_tensor_bits(actual_weights[name], tensor, name)
    assert _LiveConfigPaths.capture(native_run.policy.config) == original_paths

    fresh_optimizer = torch.optim.SGD(reloaded.parameters(), lr=0.999, momentum=0.1)
    accelerator = Accelerator(cpu=True, mixed_precision="no")
    reloaded, fresh_optimizer = accelerator.prepare(reloaded, fresh_optimizer)
    assert resume_before_prepare(resumed) == 17
    resume_after_prepare(resumed, accelerator, reloaded, fresh_optimizer, None)
    original_state = native_run.optimizer.state_dict()
    restored_state = fresh_optimizer.state_dict()
    assert restored_state["param_groups"] == original_state["param_groups"]
    assert restored_state["state"].keys() == original_state["state"].keys()
    assert original_state["state"], "optimizer fixture must contain real momentum state"
    for parameter_id, state in original_state["state"].items():
        assert restored_state["state"][parameter_id].keys() == state.keys()
        for name, tensor in state.items():
            _assert_tensor_bits(restored_state["state"][parameter_id][name], tensor, name)


def test_checkpoint_serializers_never_temporarily_mutate_live_asset_paths(
    native_run: _NativeRun, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original_paths = _LiveConfigPaths.capture(native_run.policy.config)
    observations: list[tuple[str, _LiveConfigPaths]] = []
    real_encode = draccus.encode
    real_dump = draccus.dump

    def observe_encode(value: Any, *args: Any, **kwargs: Any) -> Any:
        observations.append(("encode", _LiveConfigPaths.capture(native_run.policy.config)))
        return real_encode(value, *args, **kwargs)

    def observe_dump(value: Any, *args: Any, **kwargs: Any) -> Any:
        observations.append(("dump", _LiveConfigPaths.capture(native_run.policy.config)))
        return real_dump(value, *args, **kwargs)

    monkeypatch.setattr(draccus, "encode", observe_encode)
    monkeypatch.setattr(draccus, "dump", observe_dump)
    checkpoint = tmp_path / "checkpoint"
    save_checkpoint(
        checkpoint, step=17, cfg=native_run.config, policy=native_run.policy, optimizer=native_run.optimizer
    )

    assert {boundary for boundary, _ in observations} == {"encode", "dump"}
    for boundary, observed_paths in observations:
        assert observed_paths == original_paths, f"live config asset paths changed at draccus.{boundary}"
    assert _LiveConfigPaths.capture(native_run.policy.config) == original_paths
    package = checkpoint / PRETRAINED_MODEL_DIR
    _assert_portable_asset_paths(json.loads((package / "config.json").read_text()))
    _assert_portable_asset_paths(json.loads((package / TRAIN_CONFIG_NAME).read_text())["policy"])


def test_policy_and_nested_config_share_export_overrides_without_live_changes(tmp_path: Path) -> None:
    config = PerceptronIsaacConfig(
        device="cpu",
        native_stats_path="/runtime/base-stats.json",
        stats_path="/runtime/legacy-stats.json",
        mk1_model_import_sha256="1" * 64,
        normalization_profile_id="base",
    )
    config._export_mk1_model_import_sha256 = None
    config._export_mk1_source_model_import_sha256 = "1" * 64
    config._export_native_stats = IsaacNormalizationStats(
        action=IsaacStatsBlock(q01=np.full(7, -1.0), q99=np.full(7, 1.0)),
        proprio=IsaacStatsBlock(q01=np.full(8, -1.0), q99=np.full(8, 1.0)),
        target_fps=20.0,
        action_horizon=30,
        schema="flow_matching_stats_v1",
        profile_id="finetune",
        profile_scope="finetune_dataset",
        validation_status="dataset_stats",
    )
    original = draccus.encode(config, PreTrainedConfig)
    train = TrainPipelineConfig(dataset=DatasetConfig(repo_id="fixture/unused"), policy=config)

    portable = config.to_dict()
    assert train.to_dict()["policy"] == portable
    assert not list(tmp_path.iterdir()), "to_dict must not export sidecars"
    assert portable["mk1_model_import_sha256"] is None
    assert portable["mk1_source_model_import_sha256"] == "1" * 64
    assert portable["native_stats_path"] == "isaac_stats.json"
    assert portable["stats_path"] is None
    assert portable["normalization_profile_id"] == "finetune"
    assert portable["normalization_profile_scope"] == "finetune_dataset"
    assert portable["normalization_validation_status"] == "dataset_stats"
    config.save_pretrained(tmp_path)
    train.save_pretrained(tmp_path)
    assert json.loads((tmp_path / "config.json").read_text()) == portable
    assert json.loads((tmp_path / TRAIN_CONFIG_NAME).read_text())["policy"] == portable
    assert (tmp_path / "isaac_stats.json").is_file()
    assert draccus.encode(config, PreTrainedConfig) == original


@pytest.mark.parametrize("target", ["policy", "reward_model", "none"])
def test_train_config_serialization_preserves_other_targets(tmp_path: Path, target: str) -> None:
    policy = ACTConfig(device="cpu", pretrained_path=tmp_path / "base") if target == "policy" else None
    reward = RewardClassifierConfig(pretrained_path="fixture/reward") if target == "reward_model" else None
    config = TrainPipelineConfig(
        dataset=DatasetConfig(repo_id="fixture/unused"), policy=policy, reward_model=reward
    )
    expected = draccus.encode(config)

    assert config.to_dict() == expected
    config.save_pretrained(tmp_path)
    assert json.loads((tmp_path / TRAIN_CONFIG_NAME).read_text()) == expected
    loaded = TrainPipelineConfig.from_pretrained(tmp_path)
    assert loaded.to_dict() == expected
    if policy is not None:
        policy.save_pretrained(tmp_path)
        assert json.loads((tmp_path / "config.json").read_text()) == expected["policy"]


@pytest.mark.parametrize("complete", [True, False], ids=["portable-pair", "missing-table-control"])
def test_nested_policy_uses_the_same_suite_table_paths_as_its_package(tmp_path: Path, complete: bool) -> None:
    source = tmp_path / "training-host"
    source.mkdir()
    stats = source / "stats.json"
    tasks = source / "tasks.json"
    stats.write_text('{"fixture": "stats"}\n')
    if complete:
        tasks.write_text('{"fixture": "tasks"}\n')
    config = PerceptronIsaacConfig(
        device="cpu", suite_stats_path=str(stats), suite_by_task_index_path=str(tasks)
    )
    train = TrainPipelineConfig(dataset=DatasetConfig(repo_id="fixture/unused"), policy=config)
    original = draccus.encode(config, PreTrainedConfig)
    destination = tmp_path / "saved"
    destination.mkdir()
    portable = config.to_dict()
    assert not list(destination.iterdir()), "pure serialization must not write sidecars"
    config.save_pretrained(destination)
    train.save_pretrained(destination)
    package_policy = json.loads((destination / "config.json").read_text())
    nested_policy = json.loads((destination / TRAIN_CONFIG_NAME).read_text())["policy"]
    paths = ("suite_stats_path", "suite_by_task_index_path")
    assert {name: nested_policy[name] for name in paths} == {name: package_policy[name] for name in paths}
    assert {name: portable[name] for name in paths} == {name: package_policy[name] for name in paths}
    if complete:
        assert package_policy["suite_stats_path"] == SUITE_STATS_EXPORT_FILENAME
        assert package_policy["suite_by_task_index_path"] == SUITE_BY_TASK_INDEX_EXPORT_FILENAME
        assert (destination / SUITE_STATS_EXPORT_FILENAME).read_bytes() == stats.read_bytes()
        assert (destination / SUITE_BY_TASK_INDEX_EXPORT_FILENAME).read_bytes() == tasks.read_bytes()
    else:
        assert package_policy["suite_stats_path"] == str(stats)
        assert package_policy["suite_by_task_index_path"] == str(tasks)
        assert not (destination / SUITE_STATS_EXPORT_FILENAME).exists()
    assert draccus.encode(config, PreTrainedConfig) == original
