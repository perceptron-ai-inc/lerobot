"""Raw exports remain loadable, but saving their advanced layout is deferred.

Keep containment, snapshot ownership and native F32 storage contracts intact.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("transformers", reason="transformers is required (install lerobot[perceptron_isaac])")

from lerobot.policies.perceptron_isaac.configuration_perceptron_isaac import PerceptronIsaacConfig
from lerobot.policies.perceptron_isaac.isaac_checkpoint_contract import (
    IsaacCheckpointContractError,
    finalize_isaac_runtime_checkpoint_layout,
)
from lerobot.policies.perceptron_isaac.modeling_perceptron_isaac import PerceptronIsaacPolicy

PACKAGE_DIRECTORY_NAME = "lerobot_policy"


def _write_raw_export(root: Path) -> Path:
    """Build inert raw-export assets and return their nested policy package."""
    root.mkdir(parents=True)
    (root / "config.json").write_text('{"model_type": "isaac_0_5"}\n')
    (root / "model.safetensors.index.json").write_text('{"weight_map": {}}\n')
    (root / "model-00001-of-00002.safetensors").write_bytes(b"stale-shard-one")
    (root / "model-00002-of-00002.safetensors").write_bytes(b"stale-shard-two")
    fast_processor = root / "fast_processor_pinned"
    fast_processor.mkdir()
    (fast_processor / "tokenizer.json").write_text("{}\n")
    package = root / PACKAGE_DIRECTORY_NAME
    package.mkdir()
    (package / "config.json").write_text(
        json.dumps(
            {
                "type": "perceptron_isaac",
                "hf_model_path": "..",
                "fast_processor_path": "../fast_processor_pinned",
            }
        )
        + "\n"
    )
    return package


def _config(*, hf_model_path: str, fast_processor_path: str | None = None) -> PerceptronIsaacConfig:
    return PerceptronIsaacConfig(
        device="cpu",
        hf_model_path=hf_model_path,
        fast_processor_path=fast_processor_path,
        fast_processor_tree_sha256="0" * 64 if fast_processor_path is not None else None,
    )


def _unstarted_policy(config: PerceptronIsaacConfig, save_root: Path) -> PerceptronIsaacPolicy:
    """Exercise save preflight without allocating a backbone or connecting CUDA."""
    policy = PerceptronIsaacPolicy.__new__(PerceptronIsaacPolicy)
    policy.config = config
    policy._canonical_inner_save_root = str(save_root.resolve())
    return policy


def _empty_model_dir(root: Path) -> None:
    model_root = root / "hf_model"
    model_root.mkdir(parents=True)
    (model_root / "config.json").write_text('{"model_type": "isaac_0_5"}\n')


@pytest.mark.parametrize("entry_point", ["save", "finalize"])
@pytest.mark.parametrize("layout", ["nested", "outside"])
def test_raw_export_saving_is_rejected_before_collectives_or_writes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, entry_point: str, layout: str
) -> None:
    import lerobot.distributed.checkpoint as checkpoint

    package = _write_raw_export(tmp_path / "export")
    config = _config(hf_model_path="..")
    PerceptronIsaacPolicy._resolve_checkpoint_local_paths(config, package)
    destination = tmp_path / "saved"
    if entry_point == "finalize":
        _empty_model_dir(destination)
    if layout == "outside":
        # A package that points at model weights it does not hold cannot be copied and rewritten.
        source = tmp_path / "package"
        source.mkdir()
        config.hf_model_path = str(tmp_path / "export")
        config._checkpoint_source_root = str(source)
    before = {p.relative_to(destination): p.read_bytes() for p in destination.rglob("*") if p.is_file()}
    policy = _unstarted_policy(config, destination)

    def forbidden_gather(*args: object, **kwargs: object) -> None:
        pytest.fail("Unsupported raw export save reached collective state gathering")

    monkeypatch.setattr(checkpoint, "full_model_state_dict", forbidden_gather)
    with pytest.raises(RuntimeError, match="Raw/portable-export saving is unsupported"):
        if entry_point == "save":
            policy._save_pretrained(destination)
        else:
            policy.finalize_pretrained_package(destination)
    after = {p.relative_to(destination): p.read_bytes() for p in destination.rglob("*") if p.is_file()}
    assert after == before


def test_resolve_refuses_dotdot_when_the_parent_is_not_an_export(tmp_path: Path) -> None:
    package = tmp_path / "bare" / PACKAGE_DIRECTORY_NAME
    package.mkdir(parents=True)
    config = _config(hf_model_path="..")
    with pytest.raises(RuntimeError, match="escapes the package root"):
        PerceptronIsaacPolicy._resolve_checkpoint_local_paths(config, package)


def test_resolve_refuses_a_sideways_escape_out_of_an_export(tmp_path: Path) -> None:
    package = _write_raw_export(tmp_path / "export")
    (tmp_path / "export" / "elsewhere").mkdir()
    config = _config(hf_model_path="../elsewhere")
    with pytest.raises(RuntimeError, match="escapes the package root"):
        PerceptronIsaacPolicy._resolve_checkpoint_local_paths(config, package)


def test_finalize_reads_the_vla_block_of_the_saved_config(tmp_path: Path) -> None:
    model_root = tmp_path / "model"
    model_root.mkdir()
    (model_root / "config.json").write_text(
        json.dumps(
            {
                "model_type": "isaac_0_5",
                "isaac05_vla": {"mtp": {"present": False, "physical_layers": 0, "rollout_steps": 0}},
                "text_config": {"mtp_num_hidden_layers": 0},
            }
        )
    )
    (model_root / "model.safetensors.index.json").write_text('{"weight_map": {}}\n')
    finalize_isaac_runtime_checkpoint_layout(model_root)


def test_finalize_still_refuses_a_config_with_no_vla_block(tmp_path: Path) -> None:
    model_root = tmp_path / "no_vla"
    model_root.mkdir()
    (model_root / "config.json").write_text('{"model_type": "isaac_0_5", "text_config": {}}\n')
    with pytest.raises(IsaacCheckpointContractError, match="missing its MTP contract"):
        finalize_isaac_runtime_checkpoint_layout(model_root)
