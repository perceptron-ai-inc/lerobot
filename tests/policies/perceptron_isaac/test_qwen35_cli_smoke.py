"""Offline full CLI smoke: synthetic images and random reduced Qwen, real renderer and training.

Requires the pinned Qwen vocabulary and FAST snapshot already cached, plus the Isaac
training extras. No downloads or simulator. Each case executes three optimizer updates:
two uninterrupted, then one resumed from a relocated step-one checkpoint.
"""

import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file

from lerobot.configs import FeatureType, PolicyFeature
from lerobot.optim.optimizers import load_optimizer_state_dict
from lerobot.policies.perceptron_isaac.checkpoint_integrity import file_sha256
from lerobot.policies.perceptron_isaac.fast_processor import (
    DEFAULT_FAST_PROCESSOR_TREE_SHA256,
    materialize_pinned_fast_processor_snapshot,
    resolve_pinned_fast_processor_snapshot,
)
from lerobot.policies.perceptron_isaac.mharmony_native import (
    IsaacMharmonyRenderMetadata,
    build_isaac_training_proprio_contract,
)
from lerobot.policies.perceptron_isaac.modeling_perceptron_isaac import PerceptronIsaacPolicy
from lerobot.policies.perceptron_isaac.processor_perceptron_isaac import (
    make_perceptron_isaac_pre_post_processors,
)
from tests.policies.perceptron_isaac.test_checkpoint_lifecycle import _write_stats
from tests.policies.perceptron_isaac.test_qwen35_training_lifecycle import (
    reduced_qwen35_policy,
    synthetic_qwen35_dataset,
)


def _rendered_package(root: Path) -> Path:
    """Package genuine reduced weights and digest-checked cached renderer assets."""
    qwen = Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface")) / (
        "hub/models--Qwen--Qwen3.5-0.8B/snapshots/2fc06364715b967f1860aea9cf38778875588b17"
    )
    for filename in ("config.json", "vocab.json"):
        if not (qwen / filename).is_file():
            raise FileNotFoundError(f"Required pinned offline Qwen asset: {qwen / filename}")
    fast_source = resolve_pinned_fast_processor_snapshot(local_files_only=True)
    seed = root / "seed"
    metadata = IsaacMharmonyRenderMetadata(
        camera_order=["image", "wrist_image"],
        camera_views=["primary", "wrist"],
        image_size=[256, 256],
        n_obs_steps=1,
        action_horizon=2,
        action_dim=7,
        proprio_dim=8,
        vector_max_states=8,
        target_fps=20.0,
        robot_type="libero",
        dataset_name="synthetic/qwen35",
        # This hash identifies only this synthetic prompt fixture, not external provenance.
        training_proprio_contract=build_isaac_training_proprio_contract(
            robot_type="libero",
            dataset_name="synthetic/qwen35",
            proprio_dim=8,
            contract_version=1,
            contract_hash=hashlib.sha256(b"synthetic CLI fixture; not production provenance").hexdigest(),
        ),
        max_num_patches=256,
    )
    base_vocab = json.loads((qwen / "config.json").read_text())["text_config"]["vocab_size"]
    vocab_size = base_vocab + sum(group["size"] for group in metadata.mharmony_reserved_token_groups)
    initial = reduced_qwen35_policy(seed, vocab_size=vocab_size, action_dim=7, proprio_dim=8)
    initial.get_optim_params()
    initial.config.camera_order = ("image", "wrist_image")
    initial.config.input_features["observation.images.wrist_image"] = PolicyFeature(
        type=FeatureType.VISUAL, shape=(3, 256, 256)
    )
    initial.config.robot_type = "libero"
    initial.config.dataset_name = "synthetic/qwen35"
    initial.config.render_max_num_patches = 256
    initial.config.num_inference_steps = 2
    initial.config.num_settle_steps = 0
    initial.config.num_flow_samples = 1
    initial.config.target_fps = 20.0
    initial.config.native_stats_path = "isaac_stats.json"
    initial.config.native_render_metadata_path = "native_render_metadata.json"
    initial.config.fast_processor_path = "fast_processor"
    initial.config.fast_processor_tree_sha256 = DEFAULT_FAST_PROCESSOR_TREE_SHA256
    _write_stats(seed / "isaac_stats.json", action_dim=7, proprio_dim=8)
    (seed / "native_render_metadata.json").write_text(json.dumps(metadata.to_json_dict()))
    materialize_pinned_fast_processor_snapshot(fast_source, seed / "fast_processor")
    adapter = seed / "isaac_deployment_adapter.json"
    adapter.write_text(json.dumps({"fixture": "synthetic random reduced Qwen; not deployment-qualified"}))
    initial.config.hf_model_path = "hf_model"
    initial.config.save_pretrained(seed)
    PerceptronIsaacPolicy.resolve_checkpoint_config_paths(initial.config, seed)
    package = root / "package"
    initial.save_pretrained(package, finalize=False)
    pre, post = make_perceptron_isaac_pre_post_processors(initial.config)
    pre.save_pretrained(package)
    post.save_pretrained(package)
    initial.config.deployment_adapter_sha256 = file_sha256(adapter)
    initial.finalize_pretrained_package(package)
    print(
        f"Fixture: vocab={vocab_size}, hidden=32, layers=1, vision=256x256, cameras=2, state=8, action=2x7, CPU BF16"
    )
    return package


def _run_train(log_dir: Path, *args: str) -> None:
    log_dir.mkdir()
    command = [sys.executable, "-m", "lerobot.scripts.lerobot_train", *args]
    (log_dir / "command.json").write_text(json.dumps(command, indent=2))
    started = time.time()
    (log_dir / "started.json").write_text(json.dumps({"unix_time": started}))
    with (log_dir / "stdout.log").open("w") as stdout, (log_dir / "stderr.log").open("w") as stderr:
        result = subprocess.run(command, stdout=stdout, stderr=stderr, check=False, timeout=600)
    (log_dir / "result.json").write_text(
        json.dumps(
            {
                "exit_code": result.returncode,
                "elapsed_seconds": time.time() - started,
            }
        )
    )
    assert result.returncode == 0, (log_dir / "stderr.log").read_text()


def _checkpoint(run: Path, step: int) -> Path:
    matches = [
        p.parent.parent
        for p in (run / "checkpoints").glob("*/training_state/training_step.json")
        if p.parent.parent.name != "last" and json.loads(p.read_text())["step"] == step
    ]
    assert len(matches) == 1, matches
    return matches[0]


def _load_matching_finite_tensors(
    left: Path, right: Path
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    expected, actual = load_file(left), load_file(right)
    assert expected and actual.keys() == expected.keys()
    for name, value in expected.items():
        assert actual[name].shape == value.shape, name
        assert torch.isfinite(value).all() and torch.isfinite(actual[name]).all(), name
    return expected, actual


def _assert_training_state(checkpoint: Path, step: int) -> None:
    state = checkpoint / "training_state"
    assert json.loads((state / "training_step.json").read_text())["step"] == step
    moments = load_optimizer_state_dict(state)["state"]
    steps = [values["step"].item() for values in moments.values()]
    assert steps and set(steps) == {step}
    for field in ("exp_avg", "exp_avg_sq"):
        values = [entry[field] for entry in moments.values()]
        assert values and all(torch.isfinite(value).all() for value in values)
        assert any(torch.count_nonzero(value) > 0 for value in values)
    groups = json.loads((state / "optimizer_param_groups.json").read_text())
    assert groups
    scheduler = json.loads((state / "scheduler_state.json").read_text())
    assert scheduler["last_epoch"] == step
    print(
        f"Checkpoint step={step}; optimizer parameter states={len(moments)}; groups={groups}; scheduler={scheduler}"
    )


@pytest.mark.parametrize("use_peft", [False, True], ids=["dense", "peft"])
def test_qwen35_cli_train_relocated_resume(tmp_path: Path, use_peft: bool) -> None:
    """Native CLI owns parsing, datasets, processors, forward, updates, save and resume."""
    dataset_root = tmp_path / "dataset"
    dataset = synthetic_qwen35_dataset(
        dataset_root, action_dim=7, proprio_dim=8, camera_order=("image", "wrist_image")
    )
    assert dataset.num_frames == 2 and dataset.num_episodes == 1
    package = _rendered_package(tmp_path)
    initial_weights = load_file(package / "hf_model/model.safetensors")
    run = tmp_path / "train"
    arguments = [
        f"--policy.path={package}",
        "--policy.device=cpu",
        "--policy.push_to_hub=false",
        "--dataset.repo_id=synthetic/qwen35",
        f"--dataset.root={dataset_root}",
        "--dataset.video_backend=pyav",
        f"--output_dir={run}",
        "--steps=2",
        "--save_freq=1",
        "--batch_size=1",
        "--num_workers=0",
        "--env_eval_freq=0",
        "--log_freq=1",
        "--wandb.enable=false",
        "--use_policy_training_preset=false",
        "--optimizer.type=adamw",
        "--optimizer.lr=0.002",
        "--scheduler.type=cosine_decay_with_warmup",
        "--scheduler.num_warmup_steps=0",
        "--scheduler.num_decay_steps=4",
        "--scheduler.peak_lr=0.002",
        "--scheduler.decay_lr=0.0001",
    ]
    if use_peft:
        arguments.append("--peft.r=2")
    _run_train(tmp_path / "initial-cli", *arguments)
    first = _checkpoint(run, 1)
    _assert_training_state(first, 1)
    portable = tmp_path / "relocated-checkpoint"
    shutil.copytree(first, portable)
    original_files = {p.relative_to(first): file_sha256(p) for p in first.rglob("*") if p.is_file()}
    copied_files = {p.relative_to(portable): file_sha256(p) for p in portable.rglob("*") if p.is_file()}
    assert original_files and copied_files == original_files
    # Keep originals, but make every original policy/checkpoint absolute path unavailable.
    package.rename(tmp_path / "retired-package")
    (tmp_path / "seed").rename(tmp_path / "retired-seed")
    run.rename(tmp_path / "retired-run")
    assert not package.exists() and not (tmp_path / "seed").exists() and not run.exists()
    final = _checkpoint(tmp_path / "retired-run", 2)
    resumed_run = tmp_path / "resumed"
    _run_train(
        tmp_path / "resume-cli",
        f"--config_path={portable / 'pretrained_model/train_config.json'}",
        "--resume=true",
        f"--output_dir={resumed_run}",
    )
    resumed = _checkpoint(resumed_run, 2)
    resumed_log = (tmp_path / "resume-cli/stderr.log").read_text()
    resumed_updates = re.findall(r"\bstep:(\d+).*?\bloss:(\S+)", resumed_log)
    assert [int(step) for step, _ in resumed_updates] == [2], resumed_log
    assert all(math.isfinite(float(loss)) for _, loss in resumed_updates), resumed_updates
    for checkpoint in (final, resumed):
        _assert_training_state(checkpoint, 2)
    for name in ("optimizer_param_groups.json", "scheduler_state.json", "training_step.json"):
        assert json.loads((resumed / "training_state" / name).read_text()) == json.loads(
            (final / "training_state" / name).read_text()
        )
    # Native fixed-batch tests require exact restoration; fresh CLI trajectories need valid updates.
    _load_matching_finite_tensors(
        portable / "training_state/optimizer_state.safetensors",
        resumed / "training_state/optimizer_state.safetensors",
    )
    expected_package = final / "pretrained_model"
    resumed_package = resumed / "pretrained_model"
    weight_files = sorted(p.relative_to(expected_package) for p in expected_package.rglob("*.safetensors"))
    assert weight_files
    assert weight_files == sorted(p.relative_to(resumed_package) for p in resumed_package.rglob("*.safetensors"))
    for relative in weight_files:
        _load_matching_finite_tensors(expected_package / relative, resumed_package / relative)
    for checkpoint in (portable, final, resumed):
        saved_package = checkpoint / "pretrained_model"
        config = json.loads((saved_package / "config.json").read_text())
        for name in (
            "hf_model_path",
            "native_stats_path",
            "native_render_metadata_path",
            "fast_processor_path",
        ):
            assert not Path(config[name]).is_absolute(), name
            assert (saved_package / config[name]).exists(), name
        assert (saved_package / "qwen35_trained_package_manifest.json").is_file()
        assert (saved_package / "fast_processor/processing_action_tokenizer.py").is_file()
        saved_train = json.loads((saved_package / "train_config.json").read_text())
        assert saved_train["optimizer"]["lr"] == 0.002
        assert saved_train["scheduler"]["num_decay_steps"] == 4
    for updated_package in (expected_package, resumed_package):
        before_weights, trained_weights = _load_matching_finite_tensors(
            portable / "pretrained_model/hf_model/model.safetensors",
            updated_package / "hf_model/model.safetensors",
        )
        assert trained_weights.keys() == initial_weights.keys()
        if use_peft:
            # Both CLI outputs keep the base frozen; only serialized trainables may change.
            for name, value in trained_weights.items():
                torch.testing.assert_close(value, initial_weights[name].to(value.dtype), rtol=0, atol=0, msg=name)
            before, after = _load_matching_finite_tensors(
                portable / "pretrained_model/adapter_model.safetensors",
                updated_package / "adapter_model.safetensors",
            )
            changed = [name for name in after if not torch.equal(before[name], after[name])]
            assert any("lora_B" in name for name in changed), changed
            assert any("action_expert" in name for name in changed), changed
        else:
            changed = [
                name for name, value in trained_weights.items() if not torch.equal(value, before_weights[name])
            ]
            for component in ("visual", "action_expert", "language_model"):
                assert any(component in name for name in changed), (component, changed)
        print(
            f"PASS mode={'peft' if use_peft else 'dense'}; package={updated_package}; "
            f"changed step1-to2 tensors={len(changed)}; finite state; exact groups/scheduler/step"
        )


def test_reduced_qwen35_cli_helpers_preserve_default_geometry(tmp_path: Path) -> None:
    """The CLI geometry extension must not change existing two-dimensional fixtures."""
    dataset = synthetic_qwen35_dataset(tmp_path / "dataset")
    torch.testing.assert_close(dataset[0]["action"], torch.tensor([1.0, 2.0]), rtol=0, atol=0)
    torch.testing.assert_close(dataset[1]["action"], torch.tensor([3.0, 4.0]), rtol=0, atol=0)
    assert dataset.meta.camera_keys == ["observation.images.image"]
    assert dataset[0]["observation.state"].shape == (2,)
    policy = reduced_qwen35_policy(tmp_path / "seed")
    assert policy.config.action_dim == policy.config.proprio_dim == policy.config.vector_max_states == 2
    assert policy.config.input_features["observation.state"].shape == (2,)
    assert policy.config.output_features["action"].shape == (2,)
    inner = json.loads((tmp_path / "seed/hf_model/config.json").read_text())
    assert inner["text_config"]["vocab_size"] == 32
    assert inner["vector_max_states"] == 2
    assert inner["action_expert"]["action_dim"] == 2
