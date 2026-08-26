from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest
import torch
from safetensors.torch import save_file

import lerobot.scripts.lerobot_isaac_export_bf16 as export_module
from lerobot.configs import PreTrainedConfig
from lerobot.policies.perceptron_isaac import modeling_qwen35_vla
from lerobot.policies.perceptron_isaac.checkpoint_import import (
    MK1_TRAINED_PACKAGE_MANIFEST_FILENAME,
)
from lerobot.policies.perceptron_isaac.checkpoint_integrity import file_sha256
from lerobot.policies.perceptron_isaac.configuration_perceptron_isaac import (
    NATIVE_STATS_EXPORT_FILENAME,
    SUITE_BY_TASK_INDEX_EXPORT_FILENAME,
    SUITE_STATS_EXPORT_FILENAME,
    PerceptronIsaacConfig,
)
from lerobot.policies.perceptron_isaac.modeling_perceptron_isaac import PerceptronIsaacPolicy
from lerobot.policies.perceptron_isaac.qwen35_checkpoint import QWEN35_WEIGHT_CONVENTION_FILE
from lerobot.policies.perceptron_isaac.trained_package import (
    QWEN35_TRAINED_PACKAGE_MANIFEST_FILENAME,
    finalize_canonical_trained_package,
)
from lerobot.scripts.lerobot_isaac_export_bf16 import export_isaac_bf16_checkpoint
from tests.policies.perceptron_isaac.test_mk1_checkpoint_contract import (
    _write_checkpoint as _write_mk1_checkpoint,
)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _write_package_fixture(
    tmp_path: Path,
    *,
    suite_paths: tuple[str | None, str | None],
    family: str = "mk1",
) -> tuple[Path, Path]:
    policy_dir = tmp_path / "checkpoint" / "pretrained_model"
    policy_dir.mkdir(parents=True)
    family_identity = (
        {
            "mk1_model_import_sha256": "b" * 64,
            "mk1_trained_package_manifest_sha256": "c" * 64,
        }
        if family == "mk1"
        else {}
    )
    config = PerceptronIsaacConfig(
        device="cpu",
        native_stats_path=NATIVE_STATS_EXPORT_FILENAME,
        native_render_metadata_path="native_render_metadata.json",
        fast_processor_path="fast_processor",
        fast_processor_tree_sha256="a" * 64,
        suite_stats_path=suite_paths[0],
        suite_by_task_index_path=suite_paths[1],
        **family_identity,
    )
    config._save_pretrained(policy_dir)
    _write_json(policy_dir / "adapter_config.json", {"base_model_name_or_path": "ignored"})
    (policy_dir / "adapter_model.safetensors").write_bytes(b"adapter")
    _write_json(policy_dir / NATIVE_STATS_EXPORT_FILENAME, {"owner": "fine-tune"})

    processor_payload = {
        "steps": [
            {
                "config": {
                    "config_toml_path": "/training-host/config.toml",
                    "fast_processor_path": "/base/fast_processor",
                    "native_render_metadata_path": "/base/native_render_metadata.json",
                    "native_stats_path": "/base/isaac_stats.json",
                    "stats_path": "/training-host/stats.json",
                    "suite_by_task_index_path": "/training-host/suite-map.json",
                    "suite_stats_path": "/training-host/suite-stats.json",
                }
            }
        ]
    }
    _write_json(policy_dir / "policy_preprocessor.json", processor_payload)
    _write_json(policy_dir / "policy_postprocessor.json", processor_payload)

    base_root = tmp_path / "resolved-base-revision"
    base_hf = base_root / "hf_model"
    if family == "mk1":
        base_root.mkdir()
        _write_mk1_checkpoint(base_hf)
    else:
        base_hf.mkdir(parents=True)
        _write_json(base_hf / "config.json", {"model_type": "qwen3_5"})
    (base_hf / "tokenizer.json").write_text("base-revision-tokenizer\n")
    _write_json(base_hf / "mk1_model_import.json", {"stale": "base manifest"})
    (base_root / "fast_processor").mkdir()
    (base_root / "fast_processor" / "processor.py").write_text("# base revision\n")
    _write_json(base_root / "native_render_metadata.json", {"owner": "base-revision"})
    _write_json(base_root / NATIVE_STATS_EXPORT_FILENAME, {"owner": "base"})
    _write_json(base_root / "isaac_deployment_adapter.json", {"owner": "base-revision"})
    config.deployment_adapter_sha256 = file_sha256(base_root / "isaac_deployment_adapter.json")
    config._save_pretrained(policy_dir)
    return policy_dir, base_root


class _FakeBackbone(torch.nn.Module):
    def __init__(self, mk1_source: Path | None = None) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(1))
        self.mk1_source = mk1_source

    def save_pretrained(
        self,
        output: Path,
        *,
        safe_serialization: bool,
        max_shard_size: str,
    ) -> None:
        assert safe_serialization
        assert max_shard_size
        output.mkdir(parents=True)
        if self.mk1_source is not None:
            from safetensors.torch import load_file

            runtime_state = {
                key: value
                for shard in sorted(self.mk1_source.glob("model-*.safetensors"))
                for key, value in load_file(shard).items()
                if not key.startswith("mtp.")
            }
            save_file(runtime_state, output / "model.safetensors")
            _write_json(output / "config.json", {"written_by": "merged MK1 model"})
            return
        save_file({"weight": self.weight.detach().cpu()}, output / "model.safetensors")
        _write_json(output / "config.json", {"written_by": "merged model"})


def _mock_model_loading(
    monkeypatch,
    base_root: Path,
    *,
    base_reference: str = "org/isaac-base",
    base_revision: str | None = "base-revision",
    resolved_base_revision: str = "a" * 40,
    mock_base_resolution: bool = True,
) -> tuple[list[tuple[str, str | None]], list[Any]]:
    resolve_calls: list[tuple[str, str | None]] = []
    load_configs: list[Any] = []
    adapter_config = SimpleNamespace(
        base_model_name_or_path=base_reference,
        revision=base_revision,
    )

    def resolve(
        _adapter_root: Path,
        reference: str,
        *,
        revision: str | None,
        adapter_is_local: bool,
    ) -> tuple[Path, bool]:
        assert adapter_is_local
        resolve_calls.append((reference, revision))
        return base_root, False

    def load(reference: str, *, config: PerceptronIsaacConfig, revision: str | None = None, **_: Any):
        resolve_calls.append((f"weights:{reference}", revision))
        load_configs.append(config)
        return object()

    mk1_source = base_root / "hf_model"
    merged = SimpleNamespace(
        _isaac_model=_FakeBackbone(
            mk1_source if (mk1_source / "model.safetensors.index.json").is_file() else None
        )
    )
    adapted = SimpleNamespace(merge_and_unload=lambda *, safe_merge: merged)
    peft = ModuleType("peft")
    peft.PeftConfig = SimpleNamespace(from_pretrained=lambda *_args, **_kwargs: adapter_config)
    peft.PeftModel = SimpleNamespace(from_pretrained=lambda *_args, **_kwargs: adapted)
    monkeypatch.setitem(sys.modules, "peft", peft)
    if mock_base_resolution:
        monkeypatch.setattr(export_module, "_resolve_peft_base_snapshot", resolve)
    monkeypatch.setattr(
        export_module,
        "hub_snapshot_revision",
        lambda snapshot: resolved_base_revision if Path(snapshot) == base_root else pytest.fail("wrong base"),
    )
    monkeypatch.setattr(PerceptronIsaacPolicy, "from_pretrained", staticmethod(load))
    return resolve_calls, load_configs


def _assert_processor_paths_are_relative(package: Path) -> None:
    path_keys = {
        "config_toml_path",
        "fast_processor_path",
        "native_render_metadata_path",
        "native_stats_path",
        "stats_path",
        "suite_by_task_index_path",
        "suite_stats_path",
    }

    def check(value: Any) -> None:
        if isinstance(value, list):
            for item in value:
                check(item)
        elif isinstance(value, dict):
            for key, item in value.items():
                if key in path_keys and item:
                    path = Path(str(item))
                    assert not path.is_absolute()
                    assert ".." not in path.parts
                check(item)

    for name in ("policy_preprocessor.json", "policy_postprocessor.json"):
        check(json.loads((package / name).read_text()))


def test_export_uses_adapter_stats_pinned_base_and_writes_relocatable_package(tmp_path, monkeypatch):
    policy_dir, base_root = _write_package_fixture(
        tmp_path,
        suite_paths=(SUITE_STATS_EXPORT_FILENAME, SUITE_BY_TASK_INDEX_EXPORT_FILENAME),
    )
    _write_json(policy_dir / SUITE_STATS_EXPORT_FILENAME, {"libero_spatial": {"stats": True}})
    _write_json(policy_dir / SUITE_BY_TASK_INDEX_EXPORT_FILENAME, {"0": "libero_spatial"})
    resolve_calls, load_configs = _mock_model_loading(monkeypatch, base_root)

    output = export_isaac_bf16_checkpoint(policy_dir, tmp_path / "export", device="cpu")

    assert resolve_calls[0] == ("org/isaac-base", "base-revision")
    loaded_base = Path(resolve_calls[1][0].removeprefix("weights:"))
    assert loaded_base != base_root
    assert resolve_calls[1][1] is None
    assert not loaded_base.exists()
    loaded_stats_path = Path(load_configs[0].native_stats_path)
    assert loaded_stats_path.name == NATIVE_STATS_EXPORT_FILENAME
    assert loaded_stats_path.parent != policy_dir
    assert json.loads((output / NATIVE_STATS_EXPORT_FILENAME).read_text()) == {"owner": "fine-tune"}
    assert (output / "hf_model" / "tokenizer.json").read_text() == "base-revision-tokenizer\n"
    assert json.loads((output / "native_render_metadata.json").read_text()) == {"owner": "base-revision"}
    assert not (output / "hf_model" / "mk1_model_import.json").exists()

    saved = json.loads((output / "config.json").read_text())
    assert saved["native_stats_path"] == NATIVE_STATS_EXPORT_FILENAME
    assert saved["suite_stats_path"] == SUITE_STATS_EXPORT_FILENAME
    assert saved["suite_by_task_index_path"] == SUITE_BY_TASK_INDEX_EXPORT_FILENAME
    assert saved["mk1_model_import_sha256"] is None
    assert saved["mk1_source_model_import_sha256"] == "b" * 64
    assert saved["mk1_trained_package_manifest_sha256"] is not None
    assert (output / MK1_TRAINED_PACKAGE_MANIFEST_FILENAME).is_file()
    provenance = json.loads((output / "isaac_export_provenance.json").read_text())
    assert provenance["source_base_revision"] == "a" * 40
    for field in (
        "hf_model_path",
        "native_render_metadata_path",
        "native_stats_path",
        "fast_processor_path",
        "suite_stats_path",
        "suite_by_task_index_path",
    ):
        assert not Path(saved[field]).is_absolute()
    _assert_processor_paths_are_relative(output)
    assert json.loads((output / SUITE_STATS_EXPORT_FILENAME).read_text()) == {
        "libero_spatial": {"stats": True}
    }
    assert json.loads((output / SUITE_BY_TASK_INDEX_EXPORT_FILENAME).read_text()) == {"0": "libero_spatial"}
    for name in ("policy_preprocessor.json", "policy_postprocessor.json"):
        step = json.loads((output / name).read_text())["steps"][0]["config"]
        assert step["suite_stats_path"] == SUITE_STATS_EXPORT_FILENAME
        assert step["suite_by_task_index_path"] == SUITE_BY_TASK_INDEX_EXPORT_FILENAME

    reloaded = PreTrainedConfig.from_pretrained(output)
    assert isinstance(reloaded, PerceptronIsaacConfig)
    PerceptronIsaacPolicy._resolve_checkpoint_local_paths(reloaded, output)
    assert PerceptronIsaacPolicy._config_declares_mk1(reloaded)
    PerceptronIsaacPolicy._verify_packaged_contract_digests(reloaded, output)


def test_export_resolves_embedded_base_from_relocated_adapter(tmp_path, monkeypatch):
    policy_dir, source_base = _write_package_fixture(tmp_path, suite_paths=(None, None))
    shutil.copytree(source_base, policy_dir / "base_model")
    shutil.rmtree(source_base)
    moved_checkpoint = tmp_path / "relocated" / "checkpoint"
    moved_checkpoint.parent.mkdir()
    shutil.move(str(policy_dir.parent), moved_checkpoint)
    moved_policy = moved_checkpoint / "pretrained_model"
    embedded_base = moved_policy / "base_model"
    resolve_calls, _load_configs = _mock_model_loading(
        monkeypatch,
        embedded_base,
        base_reference="base_model",
        base_revision="local-revision",
        mock_base_resolution=False,
    )
    monkeypatch.setattr(
        export_module,
        "hub_snapshot_revision",
        lambda _snapshot: pytest.fail("embedded bases must not be treated as Hub snapshots"),
    )

    output = export_isaac_bf16_checkpoint(moved_policy, tmp_path / "export", device="cpu")

    loaded_base = Path(resolve_calls[0][0].removeprefix("weights:"))
    assert loaded_base != embedded_base.resolve()
    assert resolve_calls[0][1] is None
    assert not loaded_base.exists()
    assert output.is_dir()
    assert not source_base.exists()
    provenance = json.loads((output / "isaac_export_provenance.json").read_text())
    assert provenance["source_base_revision"] == "local-revision"


@pytest.mark.parametrize("declared_revision", [None, "main"])
def test_export_records_resolved_remote_base_commit(tmp_path, monkeypatch, declared_revision):
    policy_dir, base_root = _write_package_fixture(
        tmp_path,
        suite_paths=(None, None),
        family="qwen",
    )
    resolved_commit = "d" * 40
    resolve_calls, _load_configs = _mock_model_loading(
        monkeypatch,
        base_root,
        base_revision=declared_revision,
        resolved_base_revision=resolved_commit,
    )

    output = export_isaac_bf16_checkpoint(policy_dir, tmp_path / "export", device="cpu")

    assert resolve_calls[0] == ("org/isaac-base", declared_revision)
    provenance = json.loads((output / "isaac_export_provenance.json").read_text())
    assert provenance["source_base"] == "org/isaac-base"
    assert provenance["source_base_revision"] == resolved_commit


def test_export_uses_one_private_base_snapshot_after_source_mutation(tmp_path, monkeypatch):
    policy_dir, base_root = _write_package_fixture(tmp_path, suite_paths=(None, None))
    _mock_model_loading(monkeypatch, base_root)
    loaded_from: list[Path] = []

    def load(reference: str, *, config: PerceptronIsaacConfig, revision: str | None = None, **_: Any):
        stable_base = Path(reference)
        loaded_from.append(stable_base)
        assert stable_base != base_root
        assert revision is None
        assert (stable_base / "hf_model" / "tokenizer.json").read_text() == "base-revision-tokenizer\n"
        (base_root / "hf_model" / "tokenizer.json").write_text("mutated tokenizer\n")
        _write_json(base_root / "native_render_metadata.json", {"owner": "mutated base"})
        for shard in (base_root / "hf_model").glob("model-*.safetensors"):
            shard.write_bytes(b"mutated after snapshot")
        return object()

    def load_adapter(_base_policy, _adapter_path: str, **_kwargs):
        merged = SimpleNamespace(_isaac_model=_FakeBackbone(loaded_from[0] / "hf_model"))
        return SimpleNamespace(merge_and_unload=lambda *, safe_merge: merged)

    monkeypatch.setattr(PerceptronIsaacPolicy, "from_pretrained", staticmethod(load))
    sys.modules["peft"].PeftModel = SimpleNamespace(from_pretrained=load_adapter)

    output = export_isaac_bf16_checkpoint(policy_dir, tmp_path / "export", device="cpu")

    assert len(loaded_from) == 1
    assert not loaded_from[0].exists()
    assert (output / "hf_model" / "tokenizer.json").read_text() == "base-revision-tokenizer\n"
    assert json.loads((output / "native_render_metadata.json").read_text()) == {"owner": "base-revision"}
    assert (base_root / "hf_model" / "tokenizer.json").read_text() == "mutated tokenizer\n"


def test_export_hashes_the_same_stable_adapter_bytes_it_merges(tmp_path, monkeypatch):
    policy_dir, base_root = _write_package_fixture(tmp_path, suite_paths=(None, None))
    original_digest = file_sha256(policy_dir / "adapter_model.safetensors")
    _mock_model_loading(monkeypatch, base_root)
    merged = SimpleNamespace(_isaac_model=_FakeBackbone(base_root / "hf_model"))

    def load_adapter(_base_policy, adapter_path: str, **_kwargs):
        stable_adapter = Path(adapter_path) / "adapter_model.safetensors"
        assert Path(adapter_path) != policy_dir
        assert file_sha256(stable_adapter) == original_digest
        (policy_dir / "adapter_model.safetensors").write_bytes(b"changed after snapshot")
        return SimpleNamespace(merge_and_unload=lambda *, safe_merge: merged)

    sys.modules["peft"].PeftModel = SimpleNamespace(from_pretrained=load_adapter)

    output = export_isaac_bf16_checkpoint(policy_dir, tmp_path / "export", device="cpu")

    provenance = json.loads((output / "isaac_export_provenance.json").read_text())
    assert provenance["source_adapter_sha256"] == original_digest
    assert file_sha256(policy_dir / "adapter_model.safetensors") != original_digest


def test_export_clears_unreadable_suite_pair_everywhere(tmp_path, monkeypatch):
    policy_dir, base_root = _write_package_fixture(
        tmp_path,
        suite_paths=("missing-suite-stats.json", "missing-suite-map.json"),
    )
    _mock_model_loading(monkeypatch, base_root)

    output = export_isaac_bf16_checkpoint(policy_dir, tmp_path / "export", device="cpu")

    saved = json.loads((output / "config.json").read_text())
    assert saved["suite_stats_path"] is None
    assert saved["suite_by_task_index_path"] is None
    for name in ("policy_preprocessor.json", "policy_postprocessor.json"):
        step = json.loads((output / name).read_text())["steps"][0]["config"]
        assert step["suite_stats_path"] is None
        assert step["suite_by_task_index_path"] is None
        assert step["stats_path"] is None
        assert step["config_toml_path"] is None
    assert not (output / SUITE_STATS_EXPORT_FILENAME).exists()
    assert not (output / SUITE_BY_TASK_INDEX_EXPORT_FILENAME).exists()


def test_export_failure_never_publishes_partial_output(tmp_path, monkeypatch):
    policy_dir, base_root = _write_package_fixture(tmp_path, suite_paths=(None, None))
    (policy_dir / NATIVE_STATS_EXPORT_FILENAME).unlink()
    (base_root / NATIVE_STATS_EXPORT_FILENAME).unlink()
    _mock_model_loading(monkeypatch, base_root)
    output = tmp_path / "export"

    with pytest.raises(FileNotFoundError, match="normalization stats"):
        export_isaac_bf16_checkpoint(policy_dir, output, device="cpu")

    assert not output.exists()
    assert not list(tmp_path.glob(".export.tmp-*"))


def test_export_rejects_adapter_stats_symlink_that_escapes_package(tmp_path, monkeypatch):
    policy_dir, base_root = _write_package_fixture(tmp_path, suite_paths=(None, None))
    adapter_stats = policy_dir / NATIVE_STATS_EXPORT_FILENAME
    adapter_stats.unlink()
    outside_stats = tmp_path / "outside-stats.json"
    _write_json(outside_stats, {"owner": "outside"})
    adapter_stats.symlink_to(outside_stats)
    _mock_model_loading(monkeypatch, base_root)
    output = tmp_path / "export"

    with pytest.raises(RuntimeError, match="not a contained regular file"):
        export_isaac_bf16_checkpoint(policy_dir, output, device="cpu")

    assert not output.exists()


def test_export_rejects_absolute_adapter_stats_path(tmp_path, monkeypatch):
    policy_dir, base_root = _write_package_fixture(
        tmp_path,
        suite_paths=(None, None),
        family="qwen",
    )
    (policy_dir / NATIVE_STATS_EXPORT_FILENAME).unlink()
    outside_stats = tmp_path / "outside-stats.json"
    _write_json(outside_stats, {"owner": "outside"})
    config = json.loads((policy_dir / "config.json").read_text())
    config["native_stats_path"] = str(outside_stats)
    _write_json(policy_dir / "config.json", config)
    _mock_model_loading(monkeypatch, base_root)
    output = tmp_path / "export"

    with pytest.raises(RuntimeError, match="checkpoint-local path must be package-relative"):
        export_isaac_bf16_checkpoint(policy_dir, output, device="cpu")

    assert not output.exists()


@pytest.mark.parametrize("absolute_field", ["suite_stats_path", "suite_by_task_index_path"])
def test_export_rejects_absolute_suite_table_path(tmp_path, monkeypatch, absolute_field):
    policy_dir, base_root = _write_package_fixture(
        tmp_path,
        suite_paths=(SUITE_STATS_EXPORT_FILENAME, SUITE_BY_TASK_INDEX_EXPORT_FILENAME),
        family="qwen",
    )
    _write_json(policy_dir / SUITE_STATS_EXPORT_FILENAME, {"libero_spatial": {"stats": True}})
    _write_json(policy_dir / SUITE_BY_TASK_INDEX_EXPORT_FILENAME, {"0": "libero_spatial"})
    outside_table = tmp_path / f"outside-{absolute_field}.json"
    _write_json(outside_table, {"outside": True})
    config = json.loads((policy_dir / "config.json").read_text())
    config[absolute_field] = str(outside_table)
    _write_json(policy_dir / "config.json", config)
    _mock_model_loading(monkeypatch, base_root)
    output = tmp_path / "export"

    with pytest.raises(RuntimeError, match="checkpoint-local path must be package-relative"):
        export_isaac_bf16_checkpoint(policy_dir, output, device="cpu")

    assert not output.exists()


def test_qwen_export_is_exhaustively_bound_and_rejects_tampering(tmp_path, monkeypatch):
    policy_dir, base_root = _write_package_fixture(
        tmp_path,
        suite_paths=(None, None),
        family="qwen",
    )
    _mock_model_loading(monkeypatch, base_root)
    output = export_isaac_bf16_checkpoint(policy_dir, tmp_path / "export", device="cpu")

    saved = json.loads((output / "config.json").read_text())
    assert saved["apply_offset_norm"] is False
    assert saved["qwen35_trained_package_manifest_sha256"] is not None
    manifest = json.loads((output / QWEN35_TRAINED_PACKAGE_MANIFEST_FILENAME).read_text())
    assert manifest["model_storage"] == "canonical_inner"
    assert {
        "hf_model/config.json",
        "hf_model/model.safetensors",
        f"hf_model/{QWEN35_WEIGHT_CONVENTION_FILE}",
        "fast_processor/processor.py",
        "isaac_export_provenance.json",
        NATIVE_STATS_EXPORT_FILENAME,
        "native_render_metadata.json",
        "policy_preprocessor.json",
        "policy_postprocessor.json",
    }.issubset(manifest["files_sha256"])

    loaded_from: list[Path] = []

    def fake_load(stable_hf_model: str | Path, **_kwargs):
        loaded_from.append(Path(stable_hf_model))
        return object(), object()

    monkeypatch.setattr(modeling_qwen35_vla, "_load_qwen35_vla_from_hf", fake_load)
    modeling_qwen35_vla.load_qwen35_vla_from_hf(output / "hf_model", apply_offset_norm=False)
    assert loaded_from and loaded_from[0] != output / "hf_model"
    assert not loaded_from[0].exists()

    detached_hf_model = tmp_path / "standalone-exported-hf-model"
    shutil.copytree(output / "hf_model", detached_hf_model)
    with pytest.raises(RuntimeError, match="Detached canonical Qwen3.5 hf_model"):
        modeling_qwen35_vla.load_qwen35_vla_from_hf(detached_hf_model)

    def verify(package: Path) -> None:
        config = PreTrainedConfig.from_pretrained(package)
        PerceptronIsaacPolicy._resolve_checkpoint_local_paths(config, package)
        PerceptronIsaacPolicy._verify_packaged_contract_digests(config, package)

    def tamper_config(package: Path) -> None:
        config_path = package / "config.json"
        payload = json.loads(config_path.read_text())
        payload["chunk_size"] += 1
        _write_json(config_path, payload)

    def strip_outer_manifest(package: Path) -> None:
        config_path = package / "config.json"
        payload = json.loads(config_path.read_text())
        payload["qwen35_trained_package_manifest_sha256"] = None
        _write_json(config_path, payload)
        (package / QWEN35_TRAINED_PACKAGE_MANIFEST_FILENAME).unlink()
        (package / "policy_preprocessor.json").write_text('{"tampered": true}\n')

    mutations = {
        "config": tamper_config,
        "fast": lambda package: (package / "fast_processor" / "processor.py").write_text("# tampered\n"),
        "extra_shard": lambda package: save_file(
            {"extra": torch.ones(1)}, package / "hf_model" / "model-00002-of-00002.safetensors"
        ),
        "extra_index": lambda package: _write_json(
            package / "hf_model" / "model.safetensors.index.json",
            {"weight_map": {"weight": "model.safetensors"}},
        ),
        "duplicate_config_key": lambda package: (package / "config.json").write_text(
            (package / "config.json")
            .read_text()
            .replace("{\n", '{\n  "qwen35_trained_package_manifest_sha256": null,\n', 1)
        ),
        "stripped_outer_manifest": strip_outer_manifest,
    }
    for name, mutate in mutations.items():
        tampered = tmp_path / f"tampered-{name}"
        shutil.copytree(output, tampered)
        mutate(tampered)
        with pytest.raises((RuntimeError, ValueError)):
            verify(tampered)


def test_qwen_export_rejects_unconsumed_hf_model_entry(tmp_path, monkeypatch):
    policy_dir, base_root = _write_package_fixture(
        tmp_path,
        suite_paths=(None, None),
        family="qwen",
    )
    (base_root / "hf_model" / "unexpected.bin").write_bytes(b"not consumed by the Qwen loader")
    _mock_model_loading(monkeypatch, base_root)
    output = tmp_path / "export"

    with pytest.raises(RuntimeError, match=r"unsupported entries.*unexpected\.bin"):
        export_isaac_bf16_checkpoint(policy_dir, output, device="cpu")

    assert not output.exists()


def test_qwen_canonical_verifier_binds_configured_deployment_adapter_digest(tmp_path, monkeypatch):
    policy_dir, base_root = _write_package_fixture(
        tmp_path,
        suite_paths=(None, None),
        family="qwen",
    )
    _mock_model_loading(monkeypatch, base_root)
    output = export_isaac_bf16_checkpoint(policy_dir, tmp_path / "export", device="cpu")
    config = PreTrainedConfig.from_pretrained(output)
    assert isinstance(config, PerceptronIsaacConfig)
    config.deployment_adapter_sha256 = "0" * 64
    finalize_canonical_trained_package(config, output, family="qwen35")

    rebound = PreTrainedConfig.from_pretrained(output)
    assert isinstance(rebound, PerceptronIsaacConfig)
    PerceptronIsaacPolicy._resolve_checkpoint_local_paths(rebound, output)
    with pytest.raises(RuntimeError, match="deployment_adapter_sha256 mismatch"):
        PerceptronIsaacPolicy._verify_qwen35_trained_package_manifest(rebound, output)
