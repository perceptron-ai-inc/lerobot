from __future__ import annotations

import gc
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from safetensors.torch import save_file

pytest.importorskip("transformers")

from lerobot.configs import FeatureType, PolicyFeature
from lerobot.policies.perceptron_isaac.checkpoint_integrity import file_sha256
from lerobot.policies.perceptron_isaac.configuration_perceptron_isaac import PerceptronIsaacConfig
from lerobot.policies.perceptron_isaac.modeling_mk1_vla import load_mk1_vla_from_hf
from lerobot.policies.perceptron_isaac.modeling_perceptron_isaac import PerceptronIsaacPolicy
from lerobot.policies.perceptron_isaac.processor_perceptron_isaac import (
    make_perceptron_isaac_pre_post_processors,
)
from lerobot.policies.perceptron_isaac.trained_package import finalize_canonical_trained_package
from lerobot.utils.constants import ACTION, OBS_STATE


def _config(**overrides) -> PerceptronIsaacConfig:
    values: dict[str, Any] = {
        "device": "cpu",
        "n_obs_steps": 1,
        "chunk_size": 2,
        "n_action_steps": 2,
        "action_dim": 2,
        "proprio_dim": 2,
        "train_samples_per_chunk": 1,
        "freeze_input_embeddings": False,
        "input_features": {
            "observation.images.image": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 256, 256)),
            OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(2,)),
        },
        "output_features": {ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(2,))},
    }
    values.update(overrides)
    return PerceptronIsaacConfig(**values)


def _source_package(root: Path) -> PerceptronIsaacConfig:
    from tests.policies.perceptron_isaac.test_mk1_checkpoint_contract import (
        _write_portable_isaac05_checkpoint,
    )

    root.mkdir(parents=True)
    _write_portable_isaac05_checkpoint(root / "hf_model")
    source_identity = root / "hf_model" / "source_mk1_model_import.json"
    source_identity.write_text('{"fixture": "native-source-identity"}\n')
    (root / "fast_processor").mkdir()
    (root / "fast_processor" / "processing_action_tokenizer.py").write_text("class ActionTokenizer: pass\n")
    (root / "native_render_metadata.json").write_text("{}\n")
    (root / "isaac_stats.json").write_text("{}\n")
    config = _config(
        hf_model_path="hf_model",
        native_render_metadata_path="native_render_metadata.json",
        native_stats_path="isaac_stats.json",
        fast_processor_path="fast_processor",
        fast_processor_tree_sha256="0" * 64,
        apply_offset_norm=False,
        vector_max_states=4,
        max_action_dim=4,
        max_action_horizon=3,
        mk1_source_model_import_sha256=file_sha256(source_identity),
    )
    (root / "isaac_deployment_adapter.json").write_text("{}\n")
    config.deployment_adapter_sha256 = file_sha256(root / "isaac_deployment_adapter.json")
    config.save_pretrained(root)
    (root / "policy_preprocessor.json").write_text(
        json.dumps(
            {
                "steps": [
                    {
                        "config": {
                            "config_toml_path": "/stale/config.toml",
                            "fast_processor_path": str(root / "fast_processor"),
                            "native_render_metadata_path": str(root / "native_render_metadata.json"),
                            "native_stats_path": str(root / "isaac_stats.json"),
                            "stats_path": "/stale/stats.json",
                            "suite_stats_path": "/stale/suite-stats.json",
                            "suite_by_task_index_path": "/stale/suite-map.json",
                        }
                    }
                ]
            },
            indent=2,
        )
        + "\n"
    )
    finalize_canonical_trained_package(
        config,
        root,
        mk1_source_model_import_sha256=config.mk1_source_model_import_sha256,
    )
    PerceptronIsaacPolicy._resolve_checkpoint_local_paths(config, root)
    return config


def _load_native_fixture(config: PerceptronIsaacConfig) -> torch.nn.Module:
    assert config.hf_model_path is not None
    model, _, _ = load_mk1_vla_from_hf(
        config.hf_model_path,
        dtype=torch.float32,
        device="cpu",
        allow_test_only_reduced_geometry=True,
        allowed_storage_dtypes=frozenset({"F32"}),
        load_intent="training",
    )
    return model


def _write_stats(path: Path, *, action_dim: int = 2, proprio_dim: int = 2) -> None:
    path.write_text(
        json.dumps(
            {
                "schema": "flow_matching_stats_v1",
                "target_fps": 20.0,
                "action_horizon": 2,
                "action_dim": action_dim,
                "proprio_dim": proprio_dim,
                "action": {
                    "q01": [-1.0] * action_dim,
                    "q99": [1.0] * action_dim,
                    "min": [-1.0] * action_dim,
                    "max": [1.0] * action_dim,
                },
                "proprio": {
                    "q01": [-1.0] * proprio_dim,
                    "q99": [1.0] * proprio_dim,
                    "min": [-1.0] * proprio_dim,
                    "max": [1.0] * proprio_dim,
                },
            }
        )
        + "\n"
    )


def test_native_manifest_hashes_actual_bytes_and_rejects_changed_inventory(tmp_path: Path) -> None:
    source = tmp_path / "native"
    config = _source_package(source)
    manifest_path = source / "mk1_trained_package_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    expected = {
        path.relative_to(source).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in source.rglob("*")
        if path.is_file() and path not in {source / "config.json", manifest_path}
    }
    assert manifest["files_sha256"] == expected
    PerceptronIsaacPolicy._verify_trained_package_manifest(config, source)
    (source / "isaac_stats.json").write_text('{"tampered": true}\n')
    with pytest.raises(RuntimeError, match="digest mismatch for isaac_stats.json"):
        PerceptronIsaacPolicy._verify_trained_package_manifest(config, source)


def test_dense_save_is_self_contained_after_move_and_preserves_fp32_trainables(monkeypatch, tmp_path) -> None:
    source = tmp_path / "source"
    config = _source_package(source)
    policy = PerceptronIsaacPolicy(config)
    policy._isaac_model = _load_native_fixture(config)
    with torch.no_grad():
        policy._isaac_model.lm_head.weight.fill_(1.2500001192092896)
        policy._isaac_model.model.vector_embedding[0].weight.fill_(-0.0)
    expected_state = {name: tensor.clone() for name, tensor in policy._isaac_model.state_dict().items()}

    saved = tmp_path / "saved"
    policy.save_pretrained(saved)
    (saved / "isaac_deployment_adapter.json").write_text("{}\n")
    config.deployment_adapter_sha256 = file_sha256(saved / "isaac_deployment_adapter.json")
    policy.finalize_pretrained_package(saved)
    assert not list(saved.glob("model*.safetensors"))
    assert [path.name for path in saved.joinpath("hf_model").glob("model*.safetensors")] == [
        "model-00001-of-00001.safetensors"
    ]
    assert (saved / "hf_model" / "model.safetensors.index.json").is_file()
    moved = tmp_path / "moved"
    saved.rename(moved)
    source.rename(tmp_path / "retired-source")

    serialized = json.loads((moved / "config.json").read_text())
    assert serialized["apply_offset_norm"] is False
    assert config.apply_offset_norm is False
    assert serialized["hf_model_path"] == "hf_model"
    assert serialized["native_render_metadata_path"] == "native_render_metadata.json"
    assert serialized["native_stats_path"] == "isaac_stats.json"
    assert serialized["fast_processor_path"] == "fast_processor"
    assert (moved / "fast_processor" / "processing_action_tokenizer.py").is_file()
    assert (moved / "hf_model" / "source_mk1_model_import.json").is_file()
    assert serialized["mk1_source_model_import_sha256"] == config.mk1_source_model_import_sha256
    processor_config = json.loads((moved / "policy_preprocessor.json").read_text())["steps"][0]["config"]
    assert processor_config == {
        "config_toml_path": None,
        "fast_processor_path": "fast_processor",
        "native_render_metadata_path": "native_render_metadata.json",
        "native_stats_path": "isaac_stats.json",
        "stats_path": None,
        "suite_by_task_index_path": None,
        "suite_stats_path": None,
    }

    load_modes: list[bool] = []

    def fake_load_backbone(self, *, training=False):
        load_modes.append(training)
        assert self.config.apply_offset_norm is False
        self._isaac_model = _load_native_fixture(self.config)

    monkeypatch.setattr(PerceptronIsaacPolicy, "_load_backbone", fake_load_backbone)
    reloaded = PerceptronIsaacPolicy.from_pretrained(moved, strict=True)
    reloaded._load_backbone(training=True)

    assert load_modes == [True]
    assert reloaded._isaac_model is not None
    assert reloaded._isaac_model.lm_head.weight.dtype == torch.float32
    expected = torch.full_like(reloaded._isaac_model.lm_head.weight, 1.2500001192092896)
    torch.testing.assert_close(reloaded._isaac_model.lm_head.weight, expected, rtol=0, atol=0)
    assert torch.signbit(reloaded._isaac_model.model.vector_embedding[0].weight).all()
    actual_state = reloaded._isaac_model.state_dict()
    assert actual_state.keys() == expected_state.keys()
    for name, expected_tensor in expected_state.items():
        actual_tensor = actual_state[name]
        assert actual_tensor.dtype == expected_tensor.dtype
        assert actual_tensor.shape == expected_tensor.shape
        assert torch.equal(
            actual_tensor.contiguous().view(torch.uint8), expected_tensor.contiguous().view(torch.uint8)
        )


@pytest.mark.parametrize("train_storage_fp32", [True, False], ids=["lazy", "eager"])
def test_from_pretrained_retains_sidecars_without_rebinding_weight_source(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, train_storage_fp32: bool
) -> None:
    source = tmp_path / "source"
    config = _source_package(source)
    config.train_storage_fp32 = train_storage_fp32
    (source / "hf_model" / "vocab.json").write_text('{"token": 0}\n')
    (source / "policy_inference_recipe.json").write_text('{"fixture": "recipe"}\n')
    provenance = (source / "hf_model" / "source_mk1_model_import.json").read_text()
    finalize_canonical_trained_package(
        config, source, mk1_source_model_import_sha256=config.mk1_source_model_import_sha256
    )
    monkeypatch.delenv("QWEN35_VOCAB_PATH", raising=False)
    load_modes: list[bool] = []

    # Only the reduced-geometry loader is substituted; package loading and ownership are real.
    def load_fixture(self: PerceptronIsaacPolicy, *, training: bool = False) -> None:
        load_modes.append(training)
        self._isaac_model = _load_native_fixture(self.config)

    monkeypatch.setattr(PerceptronIsaacPolicy, "_load_backbone", load_fixture)
    policy = PerceptronIsaacPolicy.from_pretrained(source, local_files_only=True, strict=True)
    assert load_modes == ([] if train_storage_fp32 else [False])
    assert policy.config.hf_model_path is not None
    assert Path(policy.config.hf_model_path) == source / "hf_model"
    assert list(Path(policy.config.hf_model_path).glob("model*.safetensors"))
    assert policy._verified_runtime_assets is not None
    assert not list(policy._verified_runtime_assets.root.rglob("*.safetensors"))
    assert not list(policy._verified_runtime_assets.root.rglob("model.safetensors.index.json"))
    assert policy.config.native_stats_path is not None
    assert not Path(policy.config.native_stats_path).is_relative_to(source)
    retained_vocab = Path(os.environ["QWEN35_VOCAB_PATH"])
    assert not retained_vocab.is_relative_to(source)

    if train_storage_fp32:
        policy._load_backbone(training=True)
    assert policy._isaac_model is not None
    expected = policy._isaac_model.lm_head.weight.detach().clone()
    (source / "isaac_stats.json").write_text('{"changed": true}\n')
    (source / "hf_model" / "vocab.json").write_text('{"changed": true}\n')
    (source / "native_render_metadata.json").write_text('{"changed": true}\n')
    (source / "policy_inference_recipe.json").write_text('{"changed": true}\n')
    (source / "hf_model" / "source_mk1_model_import.json").write_text('{"changed": true}\n')
    (source / "fast_processor" / "processing_action_tokenizer.py").write_text("# changed\n")
    source.rename(tmp_path / "retired-source")
    assert Path(policy.config.native_stats_path).read_text() == "{}\n"
    assert retained_vocab.read_text() == '{"token": 0}\n'

    saved = tmp_path / "saved"
    policy.save_pretrained(saved)
    PerceptronIsaacPolicy._verify_saved_package(saved)
    assert (saved / "isaac_stats.json").read_text() == "{}\n"
    assert (saved / "hf_model" / "vocab.json").read_text() == '{"token": 0}\n'
    assert (saved / "fast_processor" / "processing_action_tokenizer.py").read_text() == (
        "class ActionTokenizer: pass\n"
    )
    assert (saved / "native_render_metadata.json").read_text() == "{}\n"
    assert (saved / "policy_inference_recipe.json").read_text() == '{"fixture": "recipe"}\n'
    assert (saved / "hf_model" / "source_mk1_model_import.json").read_text() == provenance
    reloaded = PerceptronIsaacPolicy.from_pretrained(saved, local_files_only=True, strict=True)
    if reloaded._isaac_model is None:
        reloaded._load_backbone(training=True)
    assert reloaded._isaac_model is not None
    torch.testing.assert_close(reloaded._isaac_model.lm_head.weight, expected, rtol=0, atol=0)


@pytest.mark.parametrize("load_from_package", [False, True], ids=["constructor", "from_pretrained"])
def test_native_save_never_copies_source_weight_files(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, load_from_package: bool
) -> None:
    from lerobot.policies.perceptron_isaac import checkpoint_integrity

    source = tmp_path / "source"
    config = _source_package(source)
    copied_weights: list[Path] = []
    copyfile = checkpoint_integrity.shutil.copyfile

    def record_copy(
        src: str | os.PathLike[str],
        dst: str | os.PathLike[str],
        *,
        follow_symlinks: bool = True,
    ) -> str | os.PathLike[str]:
        if Path(src).suffix == ".safetensors" or Path(src).name == "model.safetensors.index.json":
            copied_weights.append(Path(src))
        return copyfile(src, dst, follow_symlinks=follow_symlinks)

    monkeypatch.setattr(checkpoint_integrity.shutil, "copyfile", record_copy)
    policy = (
        PerceptronIsaacPolicy.from_pretrained(source, local_files_only=True, strict=True)
        if load_from_package
        else PerceptronIsaacPolicy(config)
    )
    policy._isaac_model = _load_native_fixture(policy.config)
    policy.save_pretrained(tmp_path / "saved")

    assert copied_weights == [], f"Source weights were unnecessarily snapshotted: {copied_weights}"
    assert list((source / "hf_model").glob("model*.safetensors"))
    assert (source / "mk1_trained_package_manifest.json").is_file()


def test_verified_runtime_sidecars_survive_source_mutation_without_weight_snapshot(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    source = tmp_path / "source"
    config = _source_package(source)
    (source / "hf_model" / "vocab.json").write_text('{"token": 0}\n')
    monkeypatch.delenv("QWEN35_VOCAB_PATH", raising=False)
    policy = PerceptronIsaacPolicy(config)
    policy._isaac_model = _load_native_fixture(config)

    policy._retain_verified_runtime_assets(source)
    assert config.native_stats_path is not None
    assert config.hf_model_path is not None
    retained_stats = Path(config.native_stats_path)
    retained_vocab = Path(os.environ["QWEN35_VOCAB_PATH"])
    (source / "isaac_stats.json").write_text('{"changed": true}\n')
    (source / "hf_model" / "vocab.json").write_text('{"changed": true}\n')

    assert Path(config._checkpoint_source_root) == source
    assert retained_stats.parent != source
    assert retained_stats.read_text() == "{}\n"
    assert retained_vocab.read_text() == '{"token": 0}\n'
    assert Path(config.hf_model_path) == source / "hf_model"
    assert list(Path(config.hf_model_path).glob("model*.safetensors"))
    assert policy._verified_runtime_assets is not None
    assert not list(policy._verified_runtime_assets.root.rglob("model*.safetensors"))

    saved = tmp_path / "saved"
    policy.save_pretrained(saved)
    assert (saved / "isaac_stats.json").read_text() == "{}\n"
    assert (saved / "hf_model" / "vocab.json").read_text() == '{"token": 0}\n'


def test_verified_runtime_assets_preserve_adapter_owned_sidecars(monkeypatch, tmp_path) -> None:
    base = tmp_path / "base"
    _source_package(base)
    (base / "hf_model" / "vocab.json").write_text('{"owner": "base"}\n')
    (base / "isaac_stats.json").write_text('{"owner": "base"}\n')

    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "isaac_stats.json").write_text('{"owner": "adapter"}\n')
    (adapter / "native_render_metadata.json").write_text('{"owner": "adapter"}\n')
    (adapter / "fast_processor").mkdir()
    (adapter / "fast_processor" / "processing_action_tokenizer.py").write_text("# adapter processor\n")
    config = _config(
        hf_model_path="hf_model",
        native_render_metadata_path="native_render_metadata.json",
        native_stats_path="isaac_stats.json",
        fast_processor_path="fast_processor",
        fast_processor_tree_sha256="0" * 64,
    )

    PerceptronIsaacPolicy._resolve_checkpoint_local_paths(
        config, adapter, allow_missing_relative=True, record_source_root=False
    )
    PerceptronIsaacPolicy._resolve_checkpoint_local_paths(config, base)
    monkeypatch.delenv("QWEN35_VOCAB_PATH", raising=False)
    policy = PerceptronIsaacPolicy(config)
    policy._retain_verified_runtime_assets(base)

    (base / "hf_model" / "vocab.json").write_text('{"changed": true}\n')
    (adapter / "isaac_stats.json").write_text('{"changed": true}\n')
    (adapter / "native_render_metadata.json").write_text('{"changed": true}\n')
    (adapter / "fast_processor" / "processing_action_tokenizer.py").write_text("# changed\n")

    assert config.hf_model_path is not None
    assert config.native_stats_path is not None
    assert config.native_render_metadata_path is not None
    assert config.fast_processor_path is not None
    assert Path(config.hf_model_path) == base / "hf_model"
    assert list(Path(config.hf_model_path).glob("model*.safetensors"))
    retained_vocab = Path(os.environ["QWEN35_VOCAB_PATH"])
    assert retained_vocab.read_text() == '{"owner": "base"}\n'
    assert not retained_vocab.is_relative_to(base)
    assert Path(config.native_stats_path).read_text() == '{"owner": "adapter"}\n'
    assert Path(config.native_render_metadata_path).read_text() == '{"owner": "adapter"}\n'
    assert Path(config.fast_processor_path, "processing_action_tokenizer.py").read_text() == (
        "# adapter processor\n"
    )
    assert Path(config.native_stats_path).parent not in {base, adapter}
    assert policy._verified_runtime_assets is not None
    assert Path(config.native_stats_path).parent != policy._verified_runtime_assets.root
    assert retained_vocab.is_relative_to(policy._verified_runtime_assets.root)
    assert len(policy._verified_external_runtime_assets) == 1
    assert Path(config.native_stats_path).is_relative_to(policy._verified_external_runtime_assets[0].root)
    assert not list(policy._verified_runtime_assets.root.rglob("model*.safetensors"))


def test_native_base_save_verifies_fresh_source_config_and_retains_sidecars(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    source = tmp_path / "source"
    config = _source_package(source)
    policy = PerceptronIsaacPolicy(config)
    policy._isaac_model = _load_native_fixture(config)
    verified: list[Path] = []
    verify_contract = PerceptronIsaacPolicy._verify_packaged_contract_digests

    def verify_fresh(source_config: PerceptronIsaacConfig, source_root: Path | None) -> None:
        assert source_config is not config
        assert source_root is not None
        assert source_root == source
        assert source_config.hf_model_path is not None
        assert Path(source_config.hf_model_path) == source / "hf_model"
        assert config.native_stats_path == str(source / "isaac_stats.json")
        verify_contract(source_config, source_root)
        verified.append(source_root)

    monkeypatch.setattr(
        PerceptronIsaacPolicy, "_verify_packaged_contract_digests", staticmethod(verify_fresh)
    )

    policy.save_pretrained(tmp_path / "first-save")
    assert verified == [source]
    assert policy._verified_runtime_assets is not None
    retained_root = policy._verified_runtime_assets.root
    assert retained_root != source
    assert (retained_root / "hf_model" / "config.json").is_file()
    assert not list(retained_root.rglob("*.safetensors"))
    assert not list(retained_root.rglob("model.safetensors.index.json"))
    assert config.native_stats_path is not None
    assert config.native_stats_path == str(retained_root / "isaac_stats.json")
    assert Path(config.native_stats_path).read_text() == "{}\n"
    assert config.hf_model_path == str(source / "hf_model")
    assert list((source / "hf_model").glob("model*.safetensors"))

    retired_source = tmp_path / "retired-source"
    source.rename(retired_source)
    policy.save_pretrained(tmp_path / "second-save")
    assert verified == [source]
    assert policy._verified_runtime_assets.root == retained_root
    assert (retained_root / "isaac_stats.json").read_text() == "{}\n"
    assert (tmp_path / "second-save" / "isaac_stats.json").read_text() == "{}\n"

    del policy
    gc.collect()
    assert not retained_root.exists()
    assert list((retired_source / "hf_model").glob("model*.safetensors"))
    assert (retired_source / "mk1_trained_package_manifest.json").is_file()


@pytest.mark.parametrize("asset", ["isaac_stats.json", "hf_model/model-00001-of-00002.safetensors"])
def test_native_save_rejects_changed_source_before_retaining_sidecars(tmp_path: Path, asset: str) -> None:
    from lerobot.policies.perceptron_isaac.mk1_checkpoint_contract import Mk1CheckpointContractError

    source = tmp_path / "source"
    config = _source_package(source)
    policy = PerceptronIsaacPolicy(config)
    changed = source / asset
    changed.write_bytes(changed.read_bytes() + b"tampered")

    with (
        # A malformed shard can fail inventory validation before full digest verification.
        pytest.raises(
            (RuntimeError, Mk1CheckpointContractError),
            match="digest mismatch|could not inspect safetensors shard",
        ),
        policy._verified_checkpoint_source(source),
    ):
        pytest.fail("Changed source must not become a verified checkpoint source.")

    assert policy._verified_runtime_assets is None
    assert config.native_stats_path == str(source / "isaac_stats.json")


def test_native_vocab_replaces_prior_env_with_retained_snapshot(monkeypatch, tmp_path) -> None:
    source = tmp_path / "source"
    config = _source_package(source)
    (source / "hf_model" / "vocab.json").write_text('{"owner": "source"}\n')
    verified_model = tmp_path / "verified" / "hf_model"
    verified_model.mkdir(parents=True)
    (verified_model / "vocab.json").write_text('{"owner": "verified"}\n')
    config.hf_model_path = str(verified_model)
    monkeypatch.setenv("QWEN35_VOCAB_PATH", str(source / "hf_model" / "vocab.json"))

    PerceptronIsaacPolicy._set_checkpoint_local_vocab_path(config)
    (source / "hf_model" / "vocab.json").unlink()

    vocab_path = Path(os.environ["QWEN35_VOCAB_PATH"])
    assert vocab_path == verified_model / "vocab.json"
    assert vocab_path.read_text() == '{"owner": "verified"}\n'


@pytest.mark.parametrize("layout", ["single", "sharded", "unindexed_shard", "index_only"])
@pytest.mark.parametrize("has_inner_model", [False, True])
def test_public_load_rejects_unsupported_outer_weights(tmp_path, layout, has_inner_model) -> None:
    from tests.policies.perceptron_isaac.test_portable_model_repo import _native_portable_package

    root, config, _ = _native_portable_package(tmp_path)
    if not has_inner_model:
        config.hf_model_path = None
    config.save_pretrained(root)
    first = "model-00001-of-00002.safetensors"
    second = "model-00002-of-00002.safetensors"
    if layout == "single":
        save_file({"_isaac_model.weight": torch.full((1, 2), 3.0)}, root / "model.safetensors")
    elif layout != "index_only":
        save_file({"_isaac_model.weight": torch.full((1, 2), 3.0)}, root / first)
        if layout == "sharded":
            save_file({"_isaac_model.bias": torch.tensor([4.0])}, root / second)
    if layout in {"sharded", "index_only"}:
        (root / "model.safetensors.index.json").write_text(
            json.dumps(
                {
                    "metadata": {"total_size": 12},
                    "weight_map": {"_isaac_model.weight": first, "_isaac_model.bias": second},
                }
            )
        )

    with pytest.raises(RuntimeError, match="Outer.*unsupported.*hf_model"):
        PerceptronIsaacPolicy.from_pretrained(root, local_files_only=True)


def test_original_package_processors_bind_retained_instance(tmp_path) -> None:
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.perceptron_isaac.mharmony_native import IsaacMharmonyRenderMetadata
    from lerobot.policies.perceptron_isaac.processor_perceptron_isaac import (
        PerceptronIsaacMharmonyPackProcessorStep,
    )
    from lerobot.policies.perceptron_isaac.trained_package import rewrite_trained_processor_paths

    source = tmp_path / "source"
    source.mkdir()
    _write_stats(source / "isaac_stats.json")
    (source / "fast_processor").mkdir()
    fast_file = source / "fast_processor" / "processing_action_tokenizer.py"
    fast_file.write_text("# synthetic inert FAST sidecar\n")
    config = _config(
        native_stats_path=str(source / "isaac_stats.json"),
        fast_processor_path=str(source / "fast_processor"),
        fast_processor_tree_sha256="0" * 64,
    )
    pre, post = make_perceptron_isaac_pre_post_processors(config)
    pre.save_pretrained(source)
    post.save_pretrained(source)
    rewrite_trained_processor_paths(source)
    (source / "native_render_metadata.json").write_text(
        json.dumps(IsaacMharmonyRenderMetadata.from_config(config).to_json_dict())
    )
    payload = json.loads((source / "policy_preprocessor.json").read_text())
    for step in payload["steps"]:
        if "fast_processor_path" in step["config"]:
            step["config"]["native_render_metadata_path"] = "native_render_metadata.json"
            step["config"]["native_stats_path"] = "isaac_stats.json"
    (source / "policy_preprocessor.json").write_text(json.dumps(payload))
    before = {
        str(path.relative_to(source)): file_sha256(path) for path in source.rglob("*") if path.is_file()
    }
    policies = []
    retained_paths: list[str] = []
    for index in range(2):
        instance_config = _config(
            native_stats_path="isaac_stats.json",
            native_render_metadata_path="native_render_metadata.json",
            fast_processor_path="fast_processor",
            fast_processor_tree_sha256="0" * 64,
        )
        PerceptronIsaacPolicy._resolve_checkpoint_local_paths(instance_config, source)
        policy = PerceptronIsaacPolicy(instance_config)
        policies.append(policy)
        policy._retain_verified_runtime_assets(source)
        pre, post = make_pre_post_processors(instance_config, pretrained_path=str(source))
        pack = next(step for step in pre.steps if isinstance(step, PerceptronIsaacMharmonyPackProcessorStep))
        assert pack.fast_processor_path == instance_config.fast_processor_path
        assert pack.native_stats_path == instance_config.native_stats_path
        assert pack.native_render_metadata_path == instance_config.native_render_metadata_path
        assert pack.fast_processor_tree_sha256 == "0" * 64
        assert pack._stats is not None
        assert list(pack._stats.action.q01) == [-1.0, -1.0]
        assert isinstance(pack.fast_processor_path, str)
        retained_paths.append(pack.fast_processor_path)
        destination = tmp_path / f"saved-{index}"
        assert policy._verified_runtime_assets is not None
        policy._copy_checkpoint_package(policy._verified_runtime_assets.root, destination)
        pre.save_pretrained(destination)
        post.save_pretrained(destination)
        rewrite_trained_processor_paths(destination)
        assert (
            file_sha256(destination / "fast_processor" / fast_file.name)
            == before[str(fast_file.relative_to(source))]
        )
        assert file_sha256(destination / "isaac_stats.json") == before["isaac_stats.json"]
        assert (
            file_sha256(destination / "native_render_metadata.json") == before["native_render_metadata.json"]
        )
    assert retained_paths[0] != retained_paths[1]
    assert before == {
        str(path.relative_to(source)): file_sha256(path) for path in source.rglob("*") if path.is_file()
    }


@pytest.mark.parametrize(
    "invalid", ["record", "relative", "root", "absolute", "configured", "traversal", "symlink"]
)
def test_resolve_retained_processor_path_rejects_invalid_provenance(tmp_path, invalid: str) -> None:
    from lerobot.policies.perceptron_isaac.processor_perceptron_isaac import (
        _resolve_retained_processor_path,
    )

    root = tmp_path / "retained"
    root.mkdir()
    asset = root / "isaac_stats.json"
    asset.write_text("{}\n")
    raw = asset.name
    configured = str(asset)
    record = {"relative": raw, "absolute": configured, "root": str(root)}
    if invalid == "record":
        record = {"relative": raw}
    elif invalid == "relative":
        record["relative"] = "unrelated.json"
    elif invalid == "root":
        record["root"] = str(tmp_path / "unrelated")
    elif invalid == "absolute":
        record["absolute"] = str(tmp_path / "unrelated.json")
    elif invalid == "configured":
        configured = str(tmp_path / "unrelated.json")
    elif invalid == "traversal":
        raw = "../secret.json"
        configured = str(root / raw)
        record = {"relative": raw, "absolute": configured, "root": str(root)}
    elif invalid == "symlink":
        secret = tmp_path / "secret.json"
        secret.write_text("{}\n")
        link = root / "linked.json"
        link.symlink_to(secret)
        raw = link.name
        configured = str(link)
        record = {"relative": raw, "absolute": configured, "root": str(root)}
    with pytest.raises(RuntimeError, match="retained|Retained"):
        _resolve_retained_processor_path(raw, configured, record, root)


@pytest.mark.parametrize("retention", ["none", "runtime"])
def test_factory_restoration_keeps_checkpoint_stats_over_destination_metadata(
    tmp_path, retention: str
) -> None:
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.perceptron_isaac.processor_perceptron_isaac import (
        PerceptronIsaacMharmonyPackProcessorStep,
    )

    source = tmp_path / "source"
    source.mkdir()
    _write_stats(source / "isaac_stats.json")
    config = _config(native_stats_path=str(source / "isaac_stats.json"))
    pre, post = make_pre_post_processors(config)
    probe = torch.full((1, 2), 0.5)
    expected = post(probe.clone())
    pre.save_pretrained(source)
    post.save_pretrained(source)
    # Older packages may retain a relative fallback alongside inlined stats.
    payload = json.loads((source / "policy_preprocessor.json").read_text())
    for step in payload["steps"]:
        if "native_stats_path" in step["config"]:
            step["config"]["native_stats_path"] = "isaac_stats.json"
    (source / "policy_preprocessor.json").write_text(json.dumps(payload))
    if retention != "none":
        config.native_stats_path = "isaac_stats.json"
        PerceptronIsaacPolicy._resolve_checkpoint_local_paths(config, source)
        policy = PerceptronIsaacPolicy(config)
        policy._retain_verified_runtime_assets(source)
    destination_stats = {
        key: {"q01": torch.full((2,), 10.0), "q99": torch.full((2,), 20.0)} for key in (ACTION, OBS_STATE)
    }
    # Resume passes metadata, not the explicit fine-tune dataset_stats override.
    loaded_pre, loaded_post = make_pre_post_processors(
        config,
        pretrained_path=str(source),
        dataset_meta=SimpleNamespace(fps=20, stats=destination_stats, features={}),
    )
    pack = next(s for s in loaded_pre.steps if isinstance(s, PerceptronIsaacMharmonyPackProcessorStep))
    assert pack.native_stats_path == config.native_stats_path
    assert pack._stats is not None
    assert list(pack._stats.action.q01) == [-1.0, -1.0]
    assert list(pack._stats.action.q99) == [1.0, 1.0]
    assert list(pack._stats.proprio.q01) == [-1.0, -1.0]
    torch.testing.assert_close(loaded_post(probe.clone()), expected)
    torch.testing.assert_close(destination_stats[ACTION]["q01"], torch.full((2,), 10.0))


@pytest.mark.parametrize("hostile", ["traversal", "symlink"])
def test_factory_processor_sidecars_reject_hostile_paths_at_package_boundary(tmp_path, hostile: str) -> None:
    from lerobot.policies.factory import make_pre_post_processors

    source = tmp_path / "source"
    source.mkdir()
    secret = tmp_path / "secret-fast"
    secret.mkdir()
    marker = secret / "processing_action_tokenizer.py"
    marker.write_text("# inert outside-package marker\n")
    config = _config()
    pre, post = make_pre_post_processors(config)
    pre.save_pretrained(source)
    post.save_pretrained(source)
    raw = "../secret-fast"
    if hostile == "symlink":
        (source / "linked-fast").symlink_to(secret, target_is_directory=True)
        raw = "linked-fast"
    payload = json.loads((source / "policy_preprocessor.json").read_text())
    for step in payload["steps"]:
        if "fast_processor_path" in step["config"]:
            step["config"]["fast_processor_path"] = raw
    (source / "policy_preprocessor.json").write_text(json.dumps(payload))
    with pytest.raises(RuntimeError, match="escapes its package"):
        make_pre_post_processors(config, pretrained_path=str(source))
    assert marker.read_text() == "# inert outside-package marker\n"
