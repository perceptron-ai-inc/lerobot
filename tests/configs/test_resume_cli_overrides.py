"""Resume transports native CLI spellings without replacing checkpoint-owned policy/state."""

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest
import torch

from lerobot.common.train_utils import load_training_state, save_training_state
from lerobot.configs import parser
from lerobot.configs.default import DatasetConfig
from lerobot.configs.train import TrainPipelineConfig
from lerobot.optim.optimizers import AdamWConfig
from lerobot.policies.act.configuration_act import ACTConfig


@dataclass(frozen=True)
class SavedResumeCheckpoint:
    path: Path
    optimizer_moment: torch.Tensor


@pytest.fixture
def moved_resume_checkpoint(tmp_path: Path) -> SavedResumeCheckpoint:
    original = tmp_path / "original"
    package = original / "pretrained_model"
    policy = ACTConfig(device="cuda", push_to_hub=False, optimizer_lr=0.001, optimizer_weight_decay=0.011)
    policy.save_pretrained(package)
    policy.optimizer_lr = 0.009
    policy.optimizer_weight_decay = 0.099
    policy.pretrained_path = Path("/retired/runtime/policy")
    config = TrainPipelineConfig(
        dataset=DatasetConfig(repo_id="synthetic/resume-cli"),
        policy=policy,
        optimizer=AdamWConfig(lr=0.004),
        steps=2,
        resume=True,
    )
    config.save_pretrained(package)
    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer = torch.optim.AdamW([parameter], lr=0.004)
    parameter.square().sum().backward()
    optimizer.step()
    save_training_state(original, 1, optimizer)
    expected_moment = optimizer.state[parameter]["exp_avg"].clone()
    moved = tmp_path / "moved"
    original.rename(moved)
    return SavedResumeCheckpoint(moved, expected_moment)


@pytest.mark.parametrize("spelling", ["equals", "two_tokens"])
@pytest.mark.parametrize("caller", ["native_wrapper", "supplied_cli_args"])
def test_resume_preserves_explicit_cli_tokens_and_checkpoint_ownership(
    moved_resume_checkpoint, monkeypatch, caller, spelling
):
    moved = moved_resume_checkpoint.path
    package = moved / "pretrained_model"
    target = package / "train_config.json"
    # Isolate the already-parsed YAML override registry, not either real config loader.
    monkeypatch.setattr(parser, "get_yaml_overrides", lambda name: ["--device=cuda", "--optimizer_lr=0.002"])
    overrides = (
        ["--policy.device=cpu", "--policy.optimizer_lr=0.003"]
        if spelling == "equals"
        else ["--policy.device", "cpu", "--policy.optimizer_lr", "0.003"]
    )
    args = ["--resume=true", *overrides]
    original_args = tuple(args)
    if caller == "native_wrapper":
        monkeypatch.setattr(sys, "argv", ["lerobot-train", f"--config_path={target}", *args])

        @parser.wrap()
        def validate_resume(cfg: TrainPipelineConfig):
            cfg.validate()
            return cfg

        loaded = validate_resume()
    else:
        # The public loader's supplied list, not another invocation's ambient argv, owns overrides.
        monkeypatch.setattr(
            sys,
            "argv",
            ["other-command", f"--config_path={target}", "--policy.device=cuda", "--policy.optimizer_lr=0.5"],
        )
        loaded = TrainPipelineConfig.from_pretrained(target, cli_args=args)
        loaded.validate()
    assert tuple(args) == original_args
    assert loaded.policy.device == "cpu"
    assert loaded.policy.optimizer_lr == 0.003
    assert loaded.policy.optimizer_weight_decay == 0.011
    assert loaded.policy.pretrained_path == package
    assert loaded.checkpoint_path == moved
    assert loaded.optimizer.lr == 0.004
    restored_parameter = torch.nn.Parameter(torch.tensor([0.0]))
    restored = torch.optim.AdamW([restored_parameter], lr=0.9)
    step, restored, _ = load_training_state(loaded.checkpoint_path, restored, None)
    assert step == 1 and restored.param_groups[0]["lr"] == 0.004
    torch.testing.assert_close(
        restored.state[restored_parameter]["exp_avg"],
        moved_resume_checkpoint.optimizer_moment,
        rtol=0,
        atol=0,
    )


@pytest.mark.parametrize(
    "args,expected",
    [
        (
            [
                "--policy.type",
                "act",
                "--policy.path",
                "/package",
                "--policy.optimizer_lr",
                "-0.01",
                "--steps",
                "2",
            ],
            ["--optimizer_lr", "-0.01"],
        ),
        (
            [
                "--policy.type=act",
                "--policy.path=/package",
                "--policy.device",
                "cpu",
                "--dataset.repo_id=x/y",
            ],
            ["--device", "cpu"],
        ),
        (
            ["--policy.input_features", '{"state": "raw value"}', "--policy.device=cpu"],
            ["--input_features", '{"state": "raw value"}', "--device=cpu"],
        ),
    ],
)
def test_nested_cli_overrides_preserve_values_but_exclude_choice_and_path(args, expected):
    assert parser.get_cli_overrides("policy", args=args) == expected


@pytest.mark.parametrize("cli_args,expected_lr", [([], 0.002), (None, 0.5)])
def test_resume_argument_provenance_is_instance_local_and_not_serialized(
    moved_resume_checkpoint, tmp_path, monkeypatch, cli_args, expected_lr
):
    package = moved_resume_checkpoint.path / "pretrained_model"
    monkeypatch.setattr(parser, "get_yaml_overrides", lambda name: ["--optimizer_lr=0.002"])
    monkeypatch.setattr(sys, "argv", ["other-command", "--policy.optimizer_lr=0.5"])
    first = TrainPipelineConfig.from_pretrained(package, cli_args=cli_args)
    second = TrainPipelineConfig.from_pretrained(package, cli_args=["--policy.optimizer_lr=0.003"])
    # Neither a second load nor a subsequent argv change may overwrite the first parse's provenance.
    monkeypatch.setattr(sys, "argv", ["another-command", "--policy.optimizer_lr=0.9"])
    second.validate()
    first.validate()
    assert first.policy.optimizer_lr == expected_lr
    assert second.policy.optimizer_lr == 0.003
    assert first.checkpoint_path == second.checkpoint_path == moved_resume_checkpoint.path
    saved = tmp_path / "resaved"
    first.save_pretrained(saved)
    original_json = json.loads((package / "train_config.json").read_text())
    resaved_json = json.loads((saved / "train_config.json").read_text())
    assert resaved_json.keys() == original_json.keys()
    assert not any(key.startswith("_") for key in resaved_json)


@pytest.mark.parametrize("ambient_field", ["policy", "reward_model"])
@pytest.mark.parametrize("cli_args", [[], ["--policy.optimizer_lr=0.003"]])
def test_public_resume_ignores_ambient_pretrained_path_selection(
    moved_resume_checkpoint, tmp_path, monkeypatch, ambient_field, cli_args
):
    from lerobot.rewards.classifier.configuration_classifier import RewardClassifierConfig

    other = tmp_path / "other"
    if ambient_field == "policy":
        ACTConfig(device="cpu", push_to_hub=False, optimizer_lr=0.6).save_pretrained(other)
    else:
        RewardClassifierConfig(learning_rate=0.6).save_pretrained(other)
    package = moved_resume_checkpoint.path / "pretrained_model"
    monkeypatch.setattr(sys, "argv", ["other-command", f"--{ambient_field}.path={other}"])
    loaded = TrainPipelineConfig.from_pretrained(package, cli_args=cli_args)
    loaded.validate()
    assert loaded.reward_model is None
    assert loaded.policy.pretrained_path == package
    assert loaded.policy.optimizer_lr == (0.003 if cli_args else 0.001)
    assert loaded.checkpoint_path == moved_resume_checkpoint.path
    assert loaded.optimizer.lr == 0.004


@pytest.mark.parametrize("field", ["policy", "reward_model"])
def test_native_wrapper_retains_explicit_pretrained_path_precedence(tmp_path, monkeypatch, field):
    from lerobot.rewards.classifier.configuration_classifier import RewardClassifierConfig

    package = tmp_path / "checkpoint/pretrained_model"
    config = TrainPipelineConfig(
        dataset=DatasetConfig(repo_id="synthetic/path-override"), resume=True, optimizer=AdamWConfig(lr=0.004)
    )
    other_policy = tmp_path / "other-policy"
    ACTConfig(device="cpu", push_to_hub=False, optimizer_lr=0.6).save_pretrained(other_policy)
    if field == "policy":
        config.policy = ACTConfig(device="cpu", push_to_hub=False)
        overrides = [f"--policy.path={other_policy}", "--policy.optimizer_lr=0.007"]
    else:
        config.reward_model = RewardClassifierConfig(learning_rate=0.004)
        other_reward = tmp_path / "other-reward"
        RewardClassifierConfig(learning_rate=0.6).save_pretrained(other_reward)
        # Reward paths retain priority over policy paths, as in the original caller.
        overrides = [
            f"--policy.path={other_policy}",
            f"--reward_model.path={other_reward}",
            "--reward_model.learning_rate=0.007",
        ]
    config.save_pretrained(package)
    monkeypatch.setattr(
        sys, "argv", ["lerobot-train", f"--config_path={package / 'train_config.json'}", *overrides]
    )

    @parser.wrap()
    def validate_resume(cfg: TrainPipelineConfig):
        cfg.validate()
        return cfg

    loaded = validate_resume()
    if field == "policy":
        assert loaded.policy.pretrained_path == other_policy
        assert loaded.policy.optimizer_lr == 0.007
    else:
        assert loaded.policy is None
        assert loaded.reward_model.pretrained_path == str(other_reward)
        assert loaded.reward_model.learning_rate == 0.007
    assert loaded.optimizer.lr == 0.004


@pytest.mark.parametrize("inspectable", [True, False])
def test_native_wrapper_keeps_legacy_loader_arguments(tmp_path, monkeypatch, inspectable):
    @dataclass
    class LegacyConfig:
        value: int = 0

        @classmethod
        def __get_path_fields__(cls):
            return ["policy"]

        @classmethod
        def from_pretrained(cls, path, *, cli_args, **kwargs):
            assert path == str(target)
            assert cli_args == ["--value=3"]
            assert kwargs == {}  # **kwargs alone is not an opt-in to new parser context.
            return cls(value=3)

    target = tmp_path / "legacy.json"
    target.write_text('{"value": 1}')
    argv = [
        "legacy",
        f"--config_path={target}",
        "--policy.path=/inert-local-path",
        "--policy.device=cpu",
        "--value=3",
    ]
    original_argv = tuple(argv)
    monkeypatch.setattr(sys, "argv", argv)
    if not inspectable:

        def unavailable_signature(callable):
            raise ValueError("legacy signature unavailable")

        monkeypatch.setattr(parser.inspect, "signature", unavailable_signature)

    @parser.wrap()
    def parse_legacy(cfg: LegacyConfig):
        return cfg

    assert parse_legacy().value == 3
    assert tuple(sys.argv) == original_argv
