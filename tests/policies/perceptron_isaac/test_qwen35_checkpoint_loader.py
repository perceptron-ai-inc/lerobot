"""Native inventory checks and explicit rejection of generic Qwen packages."""

import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from lerobot.configs import FeatureType, PolicyFeature
from lerobot.policies.perceptron_isaac.configuration_perceptron_isaac import PerceptronIsaacConfig
from lerobot.policies.perceptron_isaac.modeling_perceptron_isaac import PerceptronIsaacPolicy
from lerobot.policies.perceptron_isaac.qwen35_checkpoint import safetensors_checkpoint_layout


@pytest.mark.parametrize("entry_point", ["from_pretrained", "backbone"])
def test_policy_rejects_generic_qwen_packages_before_model_loading(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, entry_point: str
) -> None:
    pytest.importorskip("transformers")
    pytest.importorskip("mharmony")
    model = tmp_path / "hf_model"
    model.mkdir()
    (model / "config.json").write_text('{"model_type": "qwen3_5"}\n')
    config = PerceptronIsaacConfig(
        device="cpu",
        hf_model_path="hf_model",
        input_features={
            "observation.images.image": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 256, 256)),
        },
    )
    config.save_pretrained(tmp_path)
    # No model tensors exist: rejection must occur before model construction or weight reads.
    with pytest.raises(RuntimeError, match="supports only native MK1/Isaac-0.5 checkpoints"):
        if entry_point == "from_pretrained":
            PerceptronIsaacPolicy.from_pretrained(tmp_path, config=config, local_files_only=True)
        else:
            config.hf_model_path = str(model)
            PerceptronIsaacPolicy(config)._load_backbone()


@pytest.mark.parametrize("payload", ['{"outer": {"key": 1, "key": 2}}', "[]", "{broken", "\\udcff"])
def test_policy_json_boundary_rejects_invalid_objects_with_context(tmp_path: Path, payload: str) -> None:
    from lerobot.policies.perceptron_isaac.modeling_perceptron_isaac import _load_json_object

    path = tmp_path / "package.json"
    path.write_bytes(payload.encode("utf-8", errors="surrogateescape"))
    with pytest.raises(RuntimeError, match="test package") as failure:
        _load_json_object(path, "test package")
    assert isinstance(failure.value.__cause__, ValueError)


def _write_indexed_checkpoint(root: Path) -> None:
    shard_names = ["model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"]
    save_file({"weight": torch.ones(1)}, root / shard_names[0])
    save_file({"bias": torch.zeros(1)}, root / shard_names[1])
    (root / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": dict(zip(("weight", "bias"), shard_names, strict=True))})
    )


def test_single_safetensors_rejects_added_shard_override(tmp_path: Path) -> None:
    save_file({"weight": torch.ones(1)}, tmp_path / "model.safetensors")
    save_file({"weight": torch.ones(1)}, tmp_path / "model-00001-of-00001.safetensors")

    with pytest.raises(ValueError, match="both model.safetensors and sharded model files"):
        safetensors_checkpoint_layout(tmp_path)


def test_indexed_safetensors_rejects_unindexed_shard_file(tmp_path: Path) -> None:
    _write_indexed_checkpoint(tmp_path)
    save_file({"unused": torch.zeros(1)}, tmp_path / "model-99999-of-99999.safetensors")

    with pytest.raises(ValueError, match="unindexed shard files"):
        safetensors_checkpoint_layout(tmp_path)


def test_safetensors_index_rejects_duplicate_tensor_keys(tmp_path: Path) -> None:
    (tmp_path / "model.safetensors.index.json").write_text(
        '{"weight_map": {"weight": "model-00001-of-00001.safetensors", '
        '"weight": "model-00001-of-00001.safetensors", '
        '"bias": "model-00001-of-00001.safetensors"}}'
    )

    with pytest.raises(ValueError, match="duplicate key 'weight'"):
        safetensors_checkpoint_layout(tmp_path)
