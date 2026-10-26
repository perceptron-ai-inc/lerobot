from __future__ import annotations

import importlib
import tomllib
from pathlib import Path

import pytest


def test_perceptron_isaac_extra_installs_standalone_mharmony() -> None:
    pyproject = Path(__file__).resolve().parents[3] / "pyproject.toml"
    project = tomllib.loads(pyproject.read_text(encoding="utf-8"))["project"]

    assert (
        "mharmony[qwen35] @ git+https://github.com/perceptron-ai-inc/mharmony.git"
        "@9a57efbf8ada2d73641fc40081f4e4fa6ff2ab92"
    ) in project["optional-dependencies"].get("perceptron_isaac", [])


def test_mharmony_boundary_does_not_import_genesis(monkeypatch) -> None:
    from lerobot.policies.perceptron_isaac import mharmony_adapter
    from lerobot.policies.perceptron_isaac.mharmony_contract import SUPPORTED_MHARMONY_VERSION

    imported: list[str] = []

    def record_import(module_name: str):
        imported.append(module_name)
        return object()

    monkeypatch.setattr(
        mharmony_adapter,
        "_validate_installed_mharmony_version",
        lambda: SUPPORTED_MHARMONY_VERSION,
    )
    monkeypatch.setattr(importlib, "import_module", record_import)

    mharmony_adapter.load_mharmony()

    assert imported == ["mharmony"]


def test_policy_config_exposes_causal_conditioning_contract() -> None:
    from lerobot.policies.perceptron_isaac.configuration_perceptron_isaac import PerceptronIsaacConfig

    config = PerceptronIsaacConfig(action_conditioning=True, mistake_conditioning=True)

    assert config.action_conditioning is True
    assert config.action_conditioning_role == "user"
    assert config.mistake_conditioning is True


@pytest.mark.parametrize("raw_scale", [0, 0.0, False, True, -1, 1.5, "2"])
def test_pixel_shuffle_scale_rejects_explicit_invalid_values(raw_scale: object) -> None:
    from lerobot.policies.perceptron_isaac.mharmony_adapter import _pixel_shuffle_scale

    with pytest.raises(ValueError, match="pixel_shuffle_scale"):
        _pixel_shuffle_scale({"config": {"image": {"pixel_shuffle_scale": raw_scale}}})


@pytest.mark.parametrize(
    ("rendered", "expected"),
    [
        ({}, 1),
        ({"config": {}}, 1),
        ({"config": {"image": {}}}, 1),
        ({"config": {"image": {"pixel_shuffle_scale": None}}}, 1),
        ({"config": {"image": {"pixel_shuffle_scale": 1}}}, 1),
        ({"config": {"image": {"pixel_shuffle_scale": 2}}}, 2),
        ({"config": {"image": {"pixel_shuffle_scale": 2.0}}}, 2),
    ],
)
def test_pixel_shuffle_scale_defaults_only_when_absent(rendered: dict[str, object], expected: int) -> None:
    from lerobot.policies.perceptron_isaac.mharmony_adapter import _pixel_shuffle_scale

    assert _pixel_shuffle_scale(rendered) == expected
