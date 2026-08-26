"""Tests for policy.path support in YAML config files (issue #2957)."""

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from lerobot.configs import parser
from lerobot.configs.parser import (
    _config_explicit_fields,
    _config_path_args,
    _flatten_to_cli_args,
    extract_path_fields_from_config,
    get_path_arg,
    get_yaml_overrides,
)


@pytest.fixture(autouse=True)
def clear_parser_override_registries():
    parser._clear_config_overrides()
    yield
    parser._clear_config_overrides()


def _write_config(tmp_path: Path, config: dict | str, suffix: str = ".yaml") -> Path:
    config_path = tmp_path / f"config{suffix}"
    if suffix in (".yaml", ".yml"):
        content = yaml.safe_dump(config)
    elif suffix == ".json":
        content = json.dumps(config)
    elif isinstance(config, str):
        content = config
    else:
        raise ValueError(f"Unsupported config suffix: {suffix}")
    config_path.write_text(content)
    return config_path


def test_extract_path_fields_from_yaml(tmp_path):
    """Test that policy.path is extracted from a YAML config and the policy block
    is removed entirely (siblings are captured separately as cli_overrides)."""
    config = {
        "dataset": {"repo_id": "lerobot/pusht"},
        "policy": {"type": "smolvla", "path": "lerobot/smolvla_base", "push_to_hub": False},
    }
    config_path = _write_config(tmp_path, config)

    cleaned_path = extract_path_fields_from_config(config_path, ["policy"])

    # Path should be extracted and stored
    assert _config_path_args["policy"] == "lerobot/smolvla_base"

    # Cleaned config should not have the policy block at all -- draccus must not
    # try to decode it as PreTrainedConfig; the actual config comes from
    # from_pretrained(path) with the captured overrides applied on top.
    with open(cleaned_path) as f:
        cleaned = yaml.safe_load(f)
    assert "policy" not in cleaned

    # Original dataset should be untouched
    assert cleaned["dataset"]["repo_id"] == "lerobot/pusht"

    # Sibling overrides (excluding type/path) captured for from_pretrained.
    overrides = get_yaml_overrides("policy")
    assert any("push_to_hub=false" in o for o in overrides)


def test_extract_path_fields_from_json(tmp_path):
    """Test that policy.path is extracted from a JSON config and the policy
    block is removed entirely."""
    config = {
        "policy": {"type": "act", "path": "some/local/path"},
    }
    config_path = _write_config(tmp_path, config, ".json")

    cleaned_path = extract_path_fields_from_config(config_path, ["policy"])

    assert _config_path_args["policy"] == "some/local/path"

    with open(cleaned_path) as f:
        cleaned = json.load(f)
    assert "policy" not in cleaned


def test_extract_no_path_returns_original(tmp_path):
    """Test that configs without path fields are returned unchanged."""
    config = {
        "dataset": {"repo_id": "lerobot/pusht"},
        "policy": {"type": "smolvla"},
    }
    config_path = _write_config(tmp_path, config)

    result = extract_path_fields_from_config(config_path, ["policy"])

    assert result == config_path
    assert "policy" not in _config_path_args


def test_extract_toml_returns_original(tmp_path):
    """TOML is not rewritten by the YAML/JSON path extractor."""
    config_path = _write_config(
        tmp_path,
        'policy = { path = "lerobot/smolvla_base" }',
        ".toml",
    )

    result = extract_path_fields_from_config(config_path, ["policy"])

    assert result == config_path
    assert "policy" not in _config_path_args


def test_extract_removes_empty_field(tmp_path):
    """Test that the field dict is removed entirely if path was the only key."""
    config = {
        "dataset": {"repo_id": "lerobot/pusht"},
        "policy": {"path": "lerobot/smolvla_base"},
    }
    config_path = _write_config(tmp_path, config)

    cleaned_path = extract_path_fields_from_config(config_path, ["policy"])

    assert _config_path_args["policy"] == "lerobot/smolvla_base"

    with open(cleaned_path) as f:
        cleaned = yaml.safe_load(f)
    assert "policy" not in cleaned


def test_get_path_arg_fallback():
    """Test that get_path_arg falls back to _config_path_args when CLI has no path."""
    _config_path_args["policy"] = "lerobot/smolvla_base"

    # No CLI args with --policy.path
    result = get_path_arg("policy", args=[])
    assert result == "lerobot/smolvla_base"


def test_get_path_arg_cli_takes_precedence():
    """Test that CLI --policy.path takes precedence over YAML config path."""
    _config_path_args["policy"] = "yaml/path"

    result = get_path_arg("policy", args=["--policy.path=cli/path"])
    assert result == "cli/path"


def test_yaml_overrides_captured(tmp_path):
    """Test that non-path policy fields are captured as CLI-style overrides."""
    config = {
        "policy": {"path": "lerobot/smolvla_base", "lr": 1e-4, "batch_size": 32},
    }
    config_path = _write_config(tmp_path, config)

    extract_path_fields_from_config(config_path, ["policy"])

    overrides = get_yaml_overrides("policy")
    assert "--lr=0.0001" in overrides or any("lr=" in o for o in overrides)
    assert any("batch_size=32" in o for o in overrides)


def test_yaml_overrides_excludes_type_and_path(tmp_path):
    """Test that type and path fields are not included in YAML overrides."""
    config = {
        "policy": {"path": "lerobot/smolvla_base", "type": "smolvla", "lr": 5e-5},
    }
    config_path = _write_config(tmp_path, config)

    extract_path_fields_from_config(config_path, ["policy"])

    overrides = get_yaml_overrides("policy")
    assert not any("path=" in o for o in overrides)
    assert not any("type=" in o for o in overrides)
    assert any("lr=" in o for o in overrides)


def test_get_yaml_overrides_empty_when_path_only(tmp_path):
    """Test that get_yaml_overrides returns [] when policy had only a path field."""
    config = {
        "policy": {"path": "lerobot/smolvla_base"},
    }
    config_path = _write_config(tmp_path, config)

    extract_path_fields_from_config(config_path, ["policy"])

    assert get_yaml_overrides("policy") == []


def test_flatten_bool_values():
    """Test that boolean values are serialized as lowercase strings for draccus."""
    d = {"push_to_hub": True, "use_rabc": False, "lr": 0.001, "name": "test"}
    args = _flatten_to_cli_args(d)
    assert "--push_to_hub=true" in args
    assert "--use_rabc=false" in args
    assert "--lr=0.001" in args
    assert "--name=test" in args


def test_flatten_none_values_skipped():
    """Test that None values are not included in flattened args."""
    d = {"lr": 0.001, "path_override": None, "name": "test"}
    args = _flatten_to_cli_args(d)
    assert any("lr=" in a for a in args)
    assert any("name=" in a for a in args)
    assert not any("path_override" in a for a in args)


def test_flatten_nested_with_bools():
    """Test that bools in nested dicts are handled correctly."""
    d = {"optimizer": {"use_warmup": True, "lr": 0.01}}
    args = _flatten_to_cli_args(d)
    assert "--optimizer.use_warmup=true" in args
    assert "--optimizer.lr=0.01" in args


def test_extract_removes_field_with_siblings_and_no_type(tmp_path):
    """Regression: when policy.path has siblings but no type:, the entire policy
    block must still be removed from the cleaned config. Otherwise draccus tries
    to decode the leftover dict as PreTrainedConfig and crashes on the missing
    type discriminator.
    """
    config = {
        "dataset": {"repo_id": "lerobot/pusht"},
        "policy": {
            "path": "lerobot/smolvla_base",
            "n_action_steps": 10,
            "dtype": "bfloat16",
        },
    }
    config_path = _write_config(tmp_path, config)

    cleaned_path = extract_path_fields_from_config(config_path, ["policy"])

    with open(cleaned_path) as f:
        cleaned = yaml.safe_load(f) or {}
    assert "policy" not in cleaned, "policy block should be fully removed when path is present"
    assert cleaned["dataset"]["repo_id"] == "lerobot/pusht"
    assert _config_path_args["policy"] == "lerobot/smolvla_base"
    overrides = get_yaml_overrides("policy")
    assert any("n_action_steps=10" in o for o in overrides)
    assert any("dtype=bfloat16" in o for o in overrides)


@dataclass
class _DummyNested:
    foo: int = 0


@dataclass
class _DummyConfig:
    nested: _DummyNested = field(default_factory=_DummyNested)
    other: str = "default"

    @classmethod
    def __get_path_fields__(cls):
        return ["nested"]


def test_wrap_uses_cleaned_config_for_draccus_parse(tmp_path):
    """Regression: wrap() updates config_path_cli to point at the cleaned temp
    file but must propagate that to the draccus.parse fallback branch. Without
    the fix, cli_args still contains --config_path=<original> and draccus reads
    the original YAML with `path:` still in it, crashing on the unknown field.
    """
    config = {
        "nested": {"path": "some/checkpoint", "foo": 42},
        "other": "set-via-yaml",
    }
    config_path = _write_config(tmp_path, config)

    captured: dict = {}

    @parser.wrap()
    def main(cfg: _DummyConfig) -> _DummyConfig:
        captured["cfg"] = cfg
        return cfg

    with patch.object(sys, "argv", ["prog", f"--config_path={config_path}"]):
        main()

    assert captured["cfg"].other == "set-via-yaml"
    assert _config_path_args["nested"] == "some/checkpoint"
    # Cleaned config dropped `nested:` entirely; defaults stand for this wrapper
    # class (a real PreTrainedConfig would now load the checkpoint and apply
    # the captured yaml_overrides via from_pretrained()).
    assert captured["cfg"].nested.foo == 0


@dataclass
class _ExplicitNested:
    fps: int = 10
    task_ids: list[int] | None = None


@dataclass
class _ExplicitConfig:
    env: _ExplicitNested = field(default_factory=_ExplicitNested)
    steps: int = 5

    @classmethod
    def __get_path_fields__(cls):
        return ["policy"]


def test_config_file_defaults_are_not_explicit_overrides(tmp_path):
    """A config file dumped by a previous run serializes every default; those must not
    count as deliberate user choices, or checkpoint-owned adoption paths hard-error on
    fields nobody set."""
    config = {"env": {"fps": 10, "task_ids": None}, "steps": 5}
    config_path = _write_config(tmp_path, config)

    extract_path_fields_from_config(config_path, ["policy"], _ExplicitConfig)

    assert parser.get_explicit_override_fields("env") == set()
    assert "steps" not in _config_explicit_fields


def test_config_file_non_defaults_are_explicit_including_lists(tmp_path):
    """Non-default config-file values are explicit — including list-valued fields, which
    the CLI-flattening path used to drop silently."""
    config = {"env": {"fps": 99, "task_ids": [1, 2]}}
    config_path = _write_config(tmp_path, config)

    extract_path_fields_from_config(config_path, ["policy"], _ExplicitConfig)

    assert {"fps", "task_ids"} <= parser.get_explicit_override_fields("env")


def test_wrap_second_parse_starts_clean(tmp_path):
    """A second wrapped parse in the same process — even one passing no --config_path —
    must not inherit the first parse's path args or explicit-override registries."""
    config = {
        "nested": {"path": "run1/checkpoint"},
        "other": "set-via-yaml",
    }
    config_path = _write_config(tmp_path, config)

    @parser.wrap()
    def main(cfg: _DummyConfig) -> _DummyConfig:
        return cfg

    with patch.object(sys, "argv", ["prog", f"--config_path={config_path}"]):
        main()
    assert get_path_arg("nested") == "run1/checkpoint"
    assert "other" in parser.get_explicit_override_fields("other")

    with patch.object(sys, "argv", ["prog", "--other=run2"]):
        cfg2 = main()
    assert cfg2.other == "run2"
    assert get_path_arg("nested") is None
    assert parser.get_explicit_override_fields("other") == set()


def test_wrap_clears_stale_registries_for_presupplied_config():
    """A programmatically built config must not inherit a previous CLI parse's
    registry entries: the pre-supplied-cfg branch performs no parse, so before the
    fix a stale explicit env.fps override could hard-fail reconcile_policy_contract
    on a config that never mentioned fps."""
    from dataclasses import dataclass

    from lerobot.configs import parser

    @dataclass
    class _TinyConfig:
        x: int = 0

    # Pollute the registries as an earlier --config_path parse would.
    _config_explicit_fields["env"] = {"fps"}
    parser._config_path_args["policy"] = "lerobot/some_policy"
    parser._config_yaml_overrides["env"] = {"fps": 25}

    @parser.wrap()
    def main(cfg: _TinyConfig):
        return (
            parser.get_explicit_override_fields("env"),
            parser.get_path_arg("policy"),
        )

    explicit, path_arg = main(_TinyConfig())
    assert explicit == set()
    assert path_arg is None
