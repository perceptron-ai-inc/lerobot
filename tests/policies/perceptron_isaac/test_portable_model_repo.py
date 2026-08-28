from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from torch import nn

from lerobot.configs import FeatureType, PolicyFeature
from lerobot.policies.perceptron_isaac.configuration_perceptron_isaac import PerceptronIsaacConfig
from lerobot.policies.perceptron_isaac.modeling_perceptron_isaac import PerceptronIsaacPolicy
from lerobot.utils.constants import ACTION, OBS_STATE

AutoModelForCausalLM = pytest.importorskip("transformers").AutoModelForCausalLM


def _write_portable_repo(path: Path) -> Path:
    path.mkdir()
    config = {
        "model_type": "isaac_0_5",
        "architectures": ["Isaac05ForConditionalGeneration"],
        "auto_map": {
            "AutoConfig": "configuration_isaac05.Isaac05Config",
            "AutoModelForCausalLM": "modeling_isaac05.Isaac05ForConditionalGeneration",
            "AutoProcessor": "processing_isaac05.Isaac05Processor",
        },
    }
    (path / "config.json").write_text(json.dumps(config, sort_keys=True) + "\n")
    return path


@pytest.fixture(autouse=True)
def _fail_legacy_loader(monkeypatch: pytest.MonkeyPatch) -> None:
    import lerobot.policies.perceptron_isaac.modeling_qwen35_vla as legacy_model

    monkeypatch.setattr(
        legacy_model,
        "load_qwen35_vla_from_hf",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("legacy loader called")),
    )


def _policy(model_path: Path, *, device: str = "cpu") -> PerceptronIsaacPolicy:
    config = PerceptronIsaacConfig(
        device=device,
        hf_model_path=str(model_path),
        n_obs_steps=1,
        chunk_size=2,
        n_action_steps=2,
        action_dim=2,
        proprio_dim=2,
        input_features={
            "observation.images.image": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 256, 256)),
            OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(2,)),
        },
        output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(2,))},
    )
    return PerceptronIsaacPolicy(config)


def test_policy_loads_portable_repo_through_transformers_autoclass(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_path = _write_portable_repo(tmp_path / "model")
    loaded_model = nn.Linear(2, 2)
    call: dict[str, object] = {}

    def load_auto(path: Path, **kwargs: object) -> nn.Module:
        call.update(path=Path(path), **kwargs)
        return loaded_model

    monkeypatch.setattr(AutoModelForCausalLM, "from_pretrained", load_auto)
    policy = _policy(model_path)

    policy._load_backbone()

    assert policy._isaac_model is loaded_model
    assert call == {
        "path": model_path,
        "trust_remote_code": True,
        "local_files_only": True,
        "dtype": torch.bfloat16,
        "device_map": {"": "cpu"},
        "low_cpu_mem_usage": True,
        "attn_implementation": "sdpa",
    }


def test_policy_allows_direct_parent_portable_model_repo(tmp_path: Path) -> None:
    model_path = _write_portable_repo(tmp_path / "artifact")
    policy_path = model_path / "lerobot_policy"
    policy_path.mkdir()
    config = PerceptronIsaacConfig(hf_model_path="..")

    PerceptronIsaacPolicy._resolve_checkpoint_local_paths(config, policy_path)

    assert config.hf_model_path == str(model_path)


def test_policy_allows_portable_parent_fast_processor(tmp_path: Path) -> None:
    model_path = _write_portable_repo(tmp_path / "artifact")
    fast_processor_path = model_path / "fast_processor_pinned"
    fast_processor_path.mkdir()
    policy_path = model_path / "lerobot_policy"
    policy_path.mkdir()
    config = PerceptronIsaacConfig(
        hf_model_path="..",
        fast_processor_path="../fast_processor_pinned",
        fast_processor_tree_sha256="0" * 64,
    )

    PerceptronIsaacPolicy._resolve_checkpoint_local_paths(config, policy_path)

    assert config.fast_processor_path == str(fast_processor_path)


def test_portable_package_verifies_adapter_without_legacy_manifest(tmp_path: Path) -> None:
    model_path = _write_portable_repo(tmp_path / "artifact")
    policy_path = model_path / "lerobot_policy"
    policy_path.mkdir()
    adapter_path = policy_path / "isaac_deployment_adapter.json"
    adapter_path.write_text("{}\n")
    import hashlib

    config = PerceptronIsaacConfig(
        hf_model_path="..",
        deployment_adapter_sha256=hashlib.sha256(adapter_path.read_bytes()).hexdigest(),
    )
    PerceptronIsaacPolicy._resolve_checkpoint_local_paths(config, policy_path)

    PerceptronIsaacPolicy._verify_packaged_contract_digests(config, policy_path)


def test_policy_keeps_portable_training_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_path = _write_portable_repo(tmp_path / "model")
    called = False

    def load_auto(*args: object, **kwargs: object) -> nn.Module:
        nonlocal called
        called = True
        return nn.Linear(2, 2)

    monkeypatch.setattr(AutoModelForCausalLM, "from_pretrained", load_auto)
    policy = _policy(model_path)

    with pytest.raises(RuntimeError, match="Isaac-0.5 training is not supported yet"):
        policy._load_backbone(training=True)
    assert not called
