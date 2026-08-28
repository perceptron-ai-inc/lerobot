from __future__ import annotations

import tomllib
from pathlib import Path

from lerobot.policies.perceptron_isaac import mharmony_adapter


def test_perceptron_isaac_extra_installs_standalone_mharmony() -> None:
    pyproject = Path(__file__).resolve().parents[3] / "pyproject.toml"
    project = tomllib.loads(pyproject.read_text(encoding="utf-8"))["project"]

    assert "mharmony[qwen35]==0.1.0" in project["optional-dependencies"]["perceptron_isaac"]


def test_mharmony_boundary_does_not_import_genesis(monkeypatch) -> None:
    imported: list[str] = []

    def record_import(module_name: str):
        imported.append(module_name)
        return object()

    monkeypatch.setattr(mharmony_adapter.importlib, "import_module", record_import)

    mharmony_adapter.load_mharmony()

    assert imported == ["mharmony"]


def test_policy_config_exposes_causal_conditioning_contract() -> None:
    from lerobot.policies.perceptron_isaac.configuration_perceptron_isaac import PerceptronIsaacConfig

    config = PerceptronIsaacConfig(action_conditioning=True, mistake_conditioning=True)

    assert config.action_conditioning is True
    assert config.action_conditioning_role == "user"
    assert config.mistake_conditioning is True
