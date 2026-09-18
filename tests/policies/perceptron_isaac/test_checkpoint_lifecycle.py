from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import load_file, save_file

import lerobot.policies.peft as peft_loader
import lerobot.policies.pretrained as pretrained_module
from lerobot.configs import FeatureType, PolicyFeature, PreTrainedConfig
from lerobot.policies.perceptron_isaac.checkpoint_integrity import file_sha256
from lerobot.policies.perceptron_isaac.configuration_perceptron_isaac import PerceptronIsaacConfig
from lerobot.policies.perceptron_isaac.modeling_perceptron_isaac import PerceptronIsaacPolicy
from lerobot.policies.perceptron_isaac.modeling_qwen35_vla import load_qwen35_vla_from_hf
from lerobot.policies.perceptron_isaac.processor_perceptron_isaac import (
    make_perceptron_isaac_pre_post_processors,
    make_perceptron_isaac_pre_post_processors_from_pretrained,
)
from lerobot.policies.perceptron_isaac.qwen35_checkpoint import QWEN35_WEIGHT_CONVENTION_FILE
from lerobot.policies.perceptron_isaac.trained_package import stage_trained_processor_assets
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.utils.constants import ACTION, OBS_STATE


def _config(**overrides) -> PerceptronIsaacConfig:
    values = {
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
    (root / "hf_model").mkdir(parents=True)
    (root / "hf_model" / "config.json").write_text('{"model_type": "qwen3_5"}\n')
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
    )
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
    PerceptronIsaacPolicy._resolve_checkpoint_local_paths(config, root)
    return config


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


def test_dense_save_is_self_contained_after_move_and_preserves_fp32_trainables(monkeypatch, tmp_path) -> None:
    source = tmp_path / "source"
    config = _source_package(source)
    policy = PerceptronIsaacPolicy(config)
    policy._isaac_model = torch.nn.Linear(2, 1)
    with torch.no_grad():
        policy._isaac_model.weight.fill_(1.25)
        policy._isaac_model.bias.fill_(-0.5)

    saved = tmp_path / "saved"
    policy.save_pretrained(saved)
    (saved / "isaac_deployment_adapter.json").write_text("{}\n")
    config.deployment_adapter_sha256 = file_sha256(saved / "isaac_deployment_adapter.json")
    policy.finalize_pretrained_package(saved)
    assert not list(saved.glob("model*.safetensors"))
    assert [path.name for path in saved.joinpath("hf_model").glob("model*.safetensors")] == [
        "model.safetensors"
    ]
    moved = tmp_path / "moved"
    saved.rename(moved)
    source.rename(tmp_path / "retired-source")

    serialized = json.loads((moved / "config.json").read_text())
    assert serialized["apply_offset_norm"] is False
    assert config.apply_offset_norm is True
    assert serialized["hf_model_path"] == "hf_model"
    assert serialized["native_render_metadata_path"] == "native_render_metadata.json"
    assert serialized["native_stats_path"] == "isaac_stats.json"
    assert serialized["fast_processor_path"] == "fast_processor"
    assert (moved / "fast_processor" / "processing_action_tokenizer.py").is_file()
    assert (moved / "hf_model" / QWEN35_WEIGHT_CONVENTION_FILE).is_file()
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

    standalone_hf_model = tmp_path / "standalone-canonical-hf-model"
    shutil.copytree(moved / "hf_model", standalone_hf_model)
    with pytest.raises(RuntimeError, match="Detached canonical Qwen3.5 hf_model"):
        load_qwen35_vla_from_hf(standalone_hf_model)

    load_modes: list[bool] = []

    def fake_load_backbone(self, *, training=False):
        load_modes.append(training)
        assert self.config.apply_offset_norm is False
        self._isaac_model = torch.nn.Linear(2, 1)
        self._isaac_model.load_state_dict(load_file(Path(self.config.hf_model_path) / "model.safetensors"))

    monkeypatch.setattr(PerceptronIsaacPolicy, "_load_backbone", fake_load_backbone)
    reloaded = PerceptronIsaacPolicy.from_pretrained(moved, strict=True)
    reloaded._load_backbone(training=True)

    assert load_modes == [True]
    assert reloaded._isaac_model.weight.dtype == torch.float32
    torch.testing.assert_close(reloaded._isaac_model.weight, torch.full((1, 2), 1.25))
    torch.testing.assert_close(reloaded._isaac_model.bias, torch.tensor([-0.5]))


def test_verified_runtime_sidecars_survive_source_mutation_without_weight_snapshot(tmp_path) -> None:
    source = tmp_path / "source"
    config = _source_package(source)
    (source / "hf_model" / "vocab.json").write_text('{"token": 0}\n')
    save_file({"weight": torch.ones(1)}, source / "hf_model" / "model.safetensors")
    policy = PerceptronIsaacPolicy(config)
    policy._isaac_model = torch.nn.Linear(2, 1)

    policy._retain_verified_runtime_assets(source)
    retained_stats = Path(config.native_stats_path)
    retained_vocab = Path(config.hf_model_path) / "vocab.json"
    (source / "isaac_stats.json").write_text('{"changed": true}\n')
    (source / "hf_model" / "vocab.json").write_text('{"changed": true}\n')

    assert Path(config._checkpoint_source_root) == source
    assert retained_stats.parent != source
    assert retained_stats.read_text() == "{}\n"
    assert retained_vocab.read_text() == '{"token": 0}\n'
    assert not list(Path(config.hf_model_path).glob("model*.safetensors"))

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

    PerceptronIsaacPolicy.resolve_checkpoint_config_paths(config, adapter)
    PerceptronIsaacPolicy._resolve_checkpoint_local_paths(config, base)
    monkeypatch.delenv("QWEN35_VOCAB_PATH", raising=False)
    policy = PerceptronIsaacPolicy(config)
    policy._retain_verified_runtime_assets(base)

    (base / "hf_model" / "vocab.json").write_text('{"changed": true}\n')
    (adapter / "isaac_stats.json").write_text('{"changed": true}\n')
    (adapter / "native_render_metadata.json").write_text('{"changed": true}\n')
    (adapter / "fast_processor" / "processing_action_tokenizer.py").write_text("# changed\n")

    assert Path(config.hf_model_path, "vocab.json").read_text() == '{"owner": "base"}\n'
    assert Path(config.native_stats_path).read_text() == '{"owner": "adapter"}\n'
    assert Path(config.native_render_metadata_path).read_text() == '{"owner": "adapter"}\n'
    assert Path(config.fast_processor_path, "processing_action_tokenizer.py").read_text() == (
        "# adapter processor\n"
    )
    assert Path(config.native_stats_path).parent not in {base, adapter}
    assert Path(config.hf_model_path).parent != Path(config.native_stats_path).parent


@pytest.mark.parametrize("unsafe_kind", ["absolute", "traversal"])
def test_processor_retention_rejects_untrusted_serialized_sidecars(tmp_path, unsafe_kind) -> None:
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    secret = tmp_path / "secret.json"
    secret.write_text('{"secret": true}\n')
    raw = str(secret) if unsafe_kind == "absolute" else "../secret.json"
    (adapter / "policy_preprocessor.json").write_text(
        json.dumps({"steps": [{"config": {"native_stats_path": raw}}]})
    )
    policy = PerceptronIsaacPolicy(_config())

    with pytest.raises(RuntimeError, match="untrusted|escapes"):
        policy.retain_pretrained_processor_assets(adapter)

    assert secret.read_text() == '{"secret": true}\n'


def test_processor_retention_accepts_trusted_base_owned_absolute_sidecar(tmp_path) -> None:
    base = tmp_path / "base"
    base.mkdir()
    (base / "isaac_stats.json").write_text('{"owner": "base"}\n')
    config = _config(native_stats_path="isaac_stats.json")
    PerceptronIsaacPolicy._resolve_checkpoint_local_paths(config, base)

    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "policy_preprocessor.json").write_text(
        json.dumps(
            {
                "steps": [
                    {
                        "config": {"native_stats_path": str(base / "isaac_stats.json")},
                    }
                ]
            }
        )
    )
    policy = PerceptronIsaacPolicy(config)

    policy.retain_pretrained_processor_assets(adapter)
    retained_stats = Path(config.native_stats_path)
    retained_config = Path(config._pretrained_processor_root) / "policy_preprocessor.json"
    base.rename(tmp_path / "retired-base")
    adapter.rename(tmp_path / "retired-adapter")

    assert retained_stats.read_text() == '{"owner": "base"}\n'
    serialized = json.loads(retained_config.read_text())
    assert serialized["steps"][0]["config"]["native_stats_path"] == str(retained_stats)


@pytest.mark.parametrize("unsafe_kind", ["absolute", "traversal"])
def test_peft_asset_staging_rejects_untrusted_serialized_sidecars(tmp_path, unsafe_kind) -> None:
    package = tmp_path / "adapter"
    package.mkdir()
    retained = tmp_path / "retained"
    retained.mkdir()
    secret = tmp_path / "secret.json"
    secret.write_text('{"secret": true}\n')
    raw = str(secret) if unsafe_kind == "absolute" else "../secret.json"
    (package / "policy_preprocessor.json").write_text(
        json.dumps({"steps": [{"config": {"native_stats_path": raw}}]})
    )
    config = _config()
    config._pretrained_processor_root = str(retained)

    with pytest.raises(RuntimeError, match="untrusted|escapes"):
        stage_trained_processor_assets(package, config)

    assert not (package / "isaac_stats.json").exists()


def test_qwen_local_base_save_verifies_a_fresh_snapshot_config(monkeypatch, tmp_path) -> None:
    source = tmp_path / "source"
    config = _source_package(source)
    config.qwen35_trained_package_manifest_sha256 = "0" * 64
    config.save_pretrained(source)
    policy = PerceptronIsaacPolicy(config)
    policy._retain_verified_runtime_assets(source)
    live_model_path = Path(config.hf_model_path)
    verified: list[Path] = []

    def verify_fresh(snapshot_config, snapshot_root):
        assert snapshot_config is not config
        model_path = Path(snapshot_config.hf_model_path)
        assert model_path.is_relative_to(snapshot_root)
        assert model_path != live_model_path
        verified.append(Path(snapshot_root))

    monkeypatch.setattr(policy, "_verify_qwen35_trained_package_manifest", verify_fresh)

    with policy._verified_checkpoint_source(source) as stable_source:
        assert stable_source == verified[0]
        assert (stable_source / "hf_model" / "config.json").is_file()

    assert verified


def test_qwen_backbone_redirects_lazy_sidecars_before_releasing_verified_model(monkeypatch, tmp_path) -> None:
    import lerobot.policies.perceptron_isaac.modeling_qwen35_vla as qwen35_vla

    source = tmp_path / "source"
    config = _source_package(source)
    (source / "hf_model" / "vocab.json").write_text('{"token": 0}\n')
    policy = PerceptronIsaacPolicy(config)

    class VerifiedHandle:
        root = source
        cleaned = False

        def cleanup(self):
            self.cleaned = True

    verified = VerifiedHandle()
    config._qwen35_verified_package = verified

    def load_verified(handle, **_kwargs):
        assert handle is verified
        (source / "isaac_stats.json").write_text('{"changed": true}\n')
        (source / "native_render_metadata.json").write_text('{"changed": true}\n')
        handle.cleanup()
        return torch.nn.Linear(2, 1), SimpleNamespace()

    monkeypatch.setattr(qwen35_vla, "_load_qwen35_vla_from_verified_package", load_verified)

    policy._load_backbone()

    assert verified.cleaned
    assert Path(config.native_stats_path).read_text() == "{}\n"
    assert Path(config.native_render_metadata_path).read_text() == "{}\n"
    assert Path(config.native_stats_path).parent != source
    assert not hasattr(config, "_qwen35_verified_package")


def test_lazy_qwen_vocab_replaces_prior_env_with_open_verified_snapshot(monkeypatch, tmp_path) -> None:
    source = tmp_path / "source"
    config = _source_package(source)
    (source / "hf_model" / "vocab.json").write_text('{"owner": "source"}\n')
    verified_model = tmp_path / "verified" / "hf_model"
    verified_model.mkdir(parents=True)
    (verified_model / "vocab.json").write_text('{"owner": "verified"}\n')
    config._qwen35_verified_package = SimpleNamespace(hf_model_path=verified_model)
    monkeypatch.setenv("QWEN35_VOCAB_PATH", str(source / "hf_model" / "vocab.json"))

    PerceptronIsaacPolicy._set_checkpoint_local_vocab_path(config)
    (source / "hf_model" / "vocab.json").unlink()

    vocab_path = Path(os.environ["QWEN35_VOCAB_PATH"])
    assert vocab_path == verified_model / "vocab.json"
    assert vocab_path.read_text() == '{"owner": "verified"}\n'


def test_hub_upload_stages_nested_checkpoint_assets(monkeypatch, tmp_path) -> None:
    source = tmp_path / "source"
    config = _source_package(source)
    (source / "runtime.log").write_text("authenticated staged asset\n")
    config.repo_id = "user/policy"
    config.private = True
    policy = PerceptronIsaacPolicy(config)
    policy._isaac_model = torch.nn.Linear(2, 1)

    upload: dict[str, object] = {}

    class FakeApi:
        def create_repo(self, **kwargs):
            return SimpleNamespace(repo_id=kwargs["repo_id"])

        def upload_folder(self, **kwargs):
            folder = Path(kwargs["folder_path"])
            upload["files"] = {
                path.relative_to(folder).as_posix() for path in folder.rglob("*") if path.is_file()
            }
            upload["allow_patterns"] = kwargs.get("allow_patterns")
            upload["ignore_patterns"] = kwargs.get("ignore_patterns")
            upload["delete_patterns"] = kwargs.get("delete_patterns")
            return SimpleNamespace(repo_url=SimpleNamespace(url="https://huggingface.co/user/policy"))

    class FakeCard:
        def save(self, path):
            Path(path).write_text("model card\n")

    class FakeTrainConfig:
        dataset = SimpleNamespace(repo_id="user/dataset")

        def save_pretrained(self, path):
            (Path(path) / "train_config.json").write_text("{}\n")

    class FakeProcessor:
        def __init__(self, filename):
            self.filename = filename

        def save_pretrained(self, path):
            (Path(path) / self.filename).write_text("{}\n")

    monkeypatch.setattr(pretrained_module, "HfApi", FakeApi)
    monkeypatch.setattr(policy, "generate_model_card", lambda *args, **kwargs: FakeCard())
    finalized_snapshots: list[set[str]] = []
    original_finalize = policy.finalize_pretrained_package

    def record_finalize(path):
        root = Path(path)
        finalized_snapshots.append(
            {candidate.relative_to(root).as_posix() for candidate in root.rglob("*") if candidate.is_file()}
        )
        original_finalize(path)

    monkeypatch.setattr(policy, "finalize_pretrained_package", record_finalize)

    policy.push_model_to_hub(
        FakeTrainConfig(),
        preprocessor=FakeProcessor("policy_preprocessor.json"),
        postprocessor=FakeProcessor("policy_postprocessor.json"),
    )

    uploaded_files = upload["files"]
    assert isinstance(uploaded_files, set)
    assert "hf_model/config.json" in uploaded_files
    assert "fast_processor/processing_action_tokenizer.py" in uploaded_files
    assert "policy_preprocessor.json" in uploaded_files
    assert "policy_postprocessor.json" in uploaded_files
    assert "runtime.log" in uploaded_files
    assert {"policy_preprocessor.json", "policy_postprocessor.json"}.issubset(finalized_snapshots[-1])
    assert upload["allow_patterns"] is None
    assert upload["ignore_patterns"] is None
    assert upload["delete_patterns"] == "*"


def test_local_peft_package_embeds_base_and_loads_after_source_is_moved(monkeypatch, tmp_path) -> None:
    source = tmp_path / "source"
    config = _source_package(source)
    config.pretrained_path = source
    config.use_peft = True
    policy = PerceptronIsaacPolicy(config)
    save_file({"base": torch.ones(1)}, source / "hf_model" / "model.safetensors")
    policy._retain_verified_runtime_assets(source)

    adapter = tmp_path / "adapter"
    adapter.mkdir()
    config.save_pretrained(adapter)
    (adapter / "adapter_config.json").write_text(
        json.dumps(
            {
                "base_model_name_or_path": str(source),
                "revision": "local-revision-that-must-not-survive",
            }
        )
    )
    save_file({"adapter": torch.ones(1)}, adapter / "adapter_model.safetensors")
    policy.finalize_pretrained_package(adapter)

    finalized = json.loads((adapter / "adapter_config.json").read_text())
    assert finalized["base_model_name_or_path"] == "base_model"
    assert finalized["revision"] is None
    assert (adapter / "base_model" / "fast_processor" / "processing_action_tokenizer.py").is_file()
    assert (adapter / "base_model" / "hf_model" / "model.safetensors").is_file()

    moved = tmp_path / "moved-adapter"
    adapter.rename(moved)
    source.rename(tmp_path / "retired-source")
    loaded_config = PreTrainedConfig.from_pretrained(moved)
    peft_config = SimpleNamespace(base_model_name_or_path="base_model", revision=None)
    monkeypatch.setattr(peft_loader, "require_package", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        peft_loader,
        "PeftConfig",
        SimpleNamespace(from_pretrained=lambda path: peft_config),
    )
    monkeypatch.setattr(
        peft_loader,
        "PeftModel",
        SimpleNamespace(from_pretrained=lambda base, path, **kwargs: base),
    )

    def fake_load_backbone(self, *, training=False):
        assert training is False
        self._isaac_model = torch.nn.Linear(2, 1)

    monkeypatch.setattr(PerceptronIsaacPolicy, "_load_backbone", fake_load_backbone)
    reloaded = peft_loader.load_peft_policy(PerceptronIsaacPolicy, loaded_config, moved)

    assert isinstance(reloaded, PerceptronIsaacPolicy)
    assert Path(reloaded.config.hf_model_path).is_relative_to(moved / "base_model")


def test_outer_sharded_weights_are_loaded_only_from_the_index(tmp_path) -> None:
    policy = PerceptronIsaacPolicy(_config())
    policy._isaac_model = torch.nn.Linear(2, 1)
    first = "model-00001-of-00002.safetensors"
    second = "model-00002-of-00002.safetensors"
    save_file({"_isaac_model.weight": torch.full((1, 2), 3.0)}, tmp_path / first)
    save_file({"_isaac_model.bias": torch.tensor([4.0])}, tmp_path / second)
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {"total_size": 12},
                "weight_map": {
                    "_isaac_model.weight": first,
                    "_isaac_model.bias": second,
                },
            }
        )
    )

    PerceptronIsaacPolicy._maybe_load_finetuned_weights(policy, tmp_path, strict=True)

    torch.testing.assert_close(policy._isaac_model.weight, torch.full((1, 2), 3.0))
    torch.testing.assert_close(policy._isaac_model.bias, torch.tensor([4.0]))

    save_file({"_isaac_model.weight": torch.zeros(1, 2)}, tmp_path / "model-00003-of-00003.safetensors")
    with pytest.raises(RuntimeError, match="unreferenced"):
        PerceptronIsaacPolicy._maybe_load_finetuned_weights(policy, tmp_path, strict=True)


def test_peft_materializes_frozen_backbone_in_bf16_and_promotes_only_trainables(
    monkeypatch,
) -> None:
    class TinyPeftBackbone(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.frozen = torch.nn.Parameter(torch.ones(1, dtype=torch.bfloat16), requires_grad=False)
            self.lora_A = torch.nn.Parameter(torch.ones(1, dtype=torch.bfloat16), requires_grad=False)

    policy = PerceptronIsaacPolicy(_config())
    load_modes: list[bool] = []

    def fake_load_backbone(*, training=False):
        load_modes.append(training)
        policy._isaac_model = TinyPeftBackbone()

    policy._load_backbone = fake_load_backbone
    monkeypatch.setattr(
        PreTrainedPolicy,
        "wrap_with_peft",
        lambda self, peft_config=None, peft_cli_overrides=None: self,
    )

    wrapped = policy.wrap_with_peft()

    assert wrapped is policy
    assert load_modes == [False]
    assert policy._isaac_model.frozen.dtype == torch.bfloat16
    assert not policy._isaac_model.frozen.requires_grad
    assert policy._isaac_model.lora_A.dtype == torch.float32
    assert policy._isaac_model.lora_A.requires_grad


def test_generic_peft_config_records_base_reference_and_revision(monkeypatch) -> None:
    policy = PerceptronIsaacPolicy(
        _config(pretrained_path=Path("user/base-policy"), pretrained_revision="base-sha")
    )
    policy._isaac_model = torch.nn.Linear(1, 1)
    peft_config = SimpleNamespace(base_model_name_or_path=None, revision=None)
    monkeypatch.setattr(pretrained_module, "require_package", lambda *args, **kwargs: None)
    monkeypatch.setattr(policy, "_validate_peft_config", lambda config: None)
    monkeypatch.setattr(pretrained_module, "get_peft_model", lambda model, config: model)

    wrapped = PreTrainedPolicy.wrap_with_peft(policy, peft_config=peft_config)

    assert wrapped is policy
    assert peft_config.base_model_name_or_path == "user/base-policy"
    assert peft_config.revision == "base-sha"
    assert policy.config._embed_base_on_save is False
    assert policy.config._peft_base_reference == "user/base-policy"
    assert policy.config._peft_base_revision == "base-sha"


def test_fresh_remote_peft_pins_resolved_base_commit(monkeypatch, tmp_path) -> None:
    commit = "a" * 40
    snapshot = tmp_path / "models--org--base-policy" / "snapshots" / commit
    snapshot.mkdir(parents=True)
    policy = PerceptronIsaacPolicy(_config(pretrained_path="org/base-policy", pretrained_revision="main"))
    policy.config._checkpoint_source_root = str(snapshot)
    policy._isaac_model = torch.nn.Linear(1, 1)
    peft_config = SimpleNamespace(base_model_name_or_path=None, revision=None)
    monkeypatch.setattr(pretrained_module, "require_package", lambda *args, **kwargs: None)
    monkeypatch.setattr(policy, "_validate_peft_config", lambda config: None)
    monkeypatch.setattr(pretrained_module, "get_peft_model", lambda model, config: model)

    PreTrainedPolicy.wrap_with_peft(policy, peft_config=peft_config)

    assert peft_config.revision == commit
    assert policy.config._peft_base_revision == commit


def test_fresh_peft_records_local_base_for_embedding(monkeypatch, tmp_path) -> None:
    base = tmp_path / "base"
    base.mkdir()
    policy = PerceptronIsaacPolicy(_config(pretrained_path=base, pretrained_revision=None))
    policy._isaac_model = torch.nn.Linear(1, 1)
    peft_config = SimpleNamespace(base_model_name_or_path=None, revision=None)
    monkeypatch.setattr(pretrained_module, "require_package", lambda *args, **kwargs: None)
    monkeypatch.setattr(policy, "_validate_peft_config", lambda config: None)
    monkeypatch.setattr(pretrained_module, "get_peft_model", lambda model, config: model)

    PreTrainedPolicy.wrap_with_peft(policy, peft_config=peft_config)

    assert policy.config._embed_base_on_save is True
    assert policy.config._peft_base_reference == str(base)


def test_remote_peft_finalization_preserves_hub_base_without_embedding(tmp_path) -> None:
    config = _config(pretrained_path="org/base-policy", pretrained_revision="base-sha", use_peft=True)
    config._embed_base_on_save = False
    config._peft_base_reference = "org/base-policy"
    config._peft_base_revision = "base-sha"
    policy = PerceptronIsaacPolicy(config)
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    config.save_pretrained(adapter)
    (adapter / "adapter_config.json").write_text(
        json.dumps({"base_model_name_or_path": "/cache/base-snapshot", "revision": None})
    )
    save_file({"adapter": torch.ones(1)}, adapter / "adapter_model.safetensors")

    policy.finalize_pretrained_package(adapter)

    saved = json.loads((adapter / "adapter_config.json").read_text())
    assert saved["base_model_name_or_path"] == "org/base-policy"
    assert saved["revision"] == "base-sha"
    assert not (adapter / "base_model").exists()


def test_remote_peft_finalization_stages_real_processor_runtime_assets(tmp_path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    _write_stats(source / "isaac_stats.json")
    (source / "fast_processor").mkdir()
    (source / "fast_processor" / "processing_action_tokenizer.py").write_text("# retained\n")
    config = _config(
        pretrained_path="org/base-policy",
        pretrained_revision="base-sha",
        use_peft=True,
        native_stats_path="isaac_stats.json",
        fast_processor_path="fast_processor",
        fast_processor_tree_sha256="0" * 64,
    )
    PerceptronIsaacPolicy._resolve_checkpoint_local_paths(config, source)
    preprocessor, postprocessor = make_perceptron_isaac_pre_post_processors(config)
    preprocessor.save_pretrained(source)
    postprocessor.save_pretrained(source)
    policy = PerceptronIsaacPolicy(config)
    policy.retain_pretrained_processor_assets(source)
    source.rename(tmp_path / "retired-source")
    preprocessor, postprocessor = make_perceptron_isaac_pre_post_processors_from_pretrained(
        config,
        str(config._pretrained_processor_root),
    )

    config._embed_base_on_save = False
    config._peft_base_reference = "org/base-policy"
    config._peft_base_revision = "base-sha"
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    config.save_pretrained(adapter)
    preprocessor.save_pretrained(adapter)
    postprocessor.save_pretrained(adapter)
    (adapter / "adapter_config.json").write_text(
        json.dumps({"base_model_name_or_path": "/cache/base-snapshot", "revision": None})
    )
    save_file({"adapter": torch.ones(1)}, adapter / "adapter_model.safetensors")

    policy.finalize_pretrained_package(adapter)

    saved_adapter = json.loads((adapter / "adapter_config.json").read_text())
    assert saved_adapter == {"base_model_name_or_path": "org/base-policy", "revision": "base-sha"}
    assert (adapter / "fast_processor" / "processing_action_tokenizer.py").read_text() == "# retained\n"
    assert (adapter / "isaac_stats.json").is_file()
    assert not (adapter / "base_model").exists()
    serialized_preprocessor = json.loads((adapter / "policy_preprocessor.json").read_text())
    assert "fast_processor" in json.dumps(serialized_preprocessor)
    for filename in ("policy_preprocessor.json", "policy_postprocessor.json"):
        serialized = json.loads((adapter / filename).read_text())
        assert str(tmp_path) not in json.dumps(serialized)

    moved = tmp_path / "moved-adapter"
    adapter.rename(moved)
    loaded_config = PreTrainedConfig.from_pretrained(moved)
    PerceptronIsaacPolicy._resolve_checkpoint_local_paths(loaded_config, moved)
    loaded_preprocessor, loaded_postprocessor = make_perceptron_isaac_pre_post_processors_from_pretrained(
        loaded_config,
        str(moved),
    )
    assert loaded_preprocessor is not None
    assert loaded_postprocessor is not None


@pytest.mark.parametrize("retention", ["runtime", "processors"])
def test_original_package_processors_bind_retained_instance(tmp_path, retention: str) -> None:
    from lerobot.policies.factory import make_pre_post_processors
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
    (source / "native_render_metadata.json").write_text("{}\n")
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
        if retention == "runtime":
            policy._retain_verified_runtime_assets(source)
        else:
            policy.retain_pretrained_processor_assets(source)
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
        destination = tmp_path / f"staged-{index}"
        destination.mkdir()
        pre.save_pretrained(destination)
        post.save_pretrained(destination)
        stage_trained_processor_assets(destination, instance_config)
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


@pytest.mark.parametrize("retention", ["none", "runtime", "processors"])
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
        if retention == "runtime":
            policy._retain_verified_runtime_assets(source)
        else:
            policy.retain_pretrained_processor_assets(source)
    destination_stats = {
        key: {"q01": torch.full((2,), 10.0), "q99": torch.full((2,), 20.0)} for key in (ACTION, OBS_STATE)
    }
    # Resume passes metadata, not the explicit fine-tune dataset_stats override.
    loaded_pre, loaded_post = make_pre_post_processors(
        config,
        pretrained_path=str(source),
        dataset_meta=SimpleNamespace(fps=20, stats=destination_stats),
    )
    pack = next(s for s in loaded_pre.steps if isinstance(s, PerceptronIsaacMharmonyPackProcessorStep))
    assert pack.native_stats_path == config.native_stats_path
    assert pack._stats is not None
    assert list(pack._stats.action.q01) == [-1.0, -1.0]
    assert list(pack._stats.action.q99) == [1.0, 1.0]
    assert list(pack._stats.proprio.q01) == [-1.0, -1.0]
    torch.testing.assert_close(loaded_post(probe.clone()), expected)
    torch.testing.assert_close(destination_stats[ACTION]["q01"], torch.full((2,), 10.0))


@pytest.mark.parametrize("hostile", ["traversal", "symlink", "forged_absolute"])
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
    elif hostile == "forged_absolute":
        raw = str(secret)
        # Matching a caller-supplied config path is not verified provenance.
        config.fast_processor_path = raw
    payload = json.loads((source / "policy_preprocessor.json").read_text())
    for step in payload["steps"]:
        if "fast_processor_path" in step["config"]:
            step["config"]["fast_processor_path"] = raw
    (source / "policy_preprocessor.json").write_text(json.dumps(payload))
    if hostile == "forged_absolute":
        # Legacy absolutes remain untouched by restoration; staging is the trust boundary.
        pre, post = make_pre_post_processors(config, pretrained_path=str(source))
        destination = tmp_path / "destination"
        destination.mkdir()
        pre.save_pretrained(destination)
        post.save_pretrained(destination)
        with pytest.raises(RuntimeError, match="untrusted absolute"):
            stage_trained_processor_assets(destination, config)
        assert not (destination / "fast_processor").exists()
    else:
        with pytest.raises(RuntimeError, match="escapes its package"):
            make_pre_post_processors(config, pretrained_path=str(source))
    assert marker.read_text() == "# inert outside-package marker\n"
