"""P2.13 static recipe consistency only; not training provenance or runtime qualification.

Pins follow the parent-supplied executed-stack authority, not the generic lock.
Static consistency does not establish a complete historical freeze or runtime qualification.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
CONSTRAINTS = "constraints/perceptron_isaac-cu128.txt"


@dataclass(frozen=True)
class StackPin:
    package: str
    version: str


def _stack_pins() -> tuple[StackPin, ...]:
    text = (ROOT / CONSTRAINTS).read_text()
    return tuple(
        StackPin(package, version)
        for package, version in re.findall(
            r"^(torch|torchvision|accelerate|transformers)==([^\s#]+)$", text, re.M
        )
    )


def test_isaac_ci_install_selects_the_declared_scoped_stack() -> None:
    workflow = (ROOT / ".github/workflows/fast_tests.yml").read_text()
    pins = _stack_pins()
    assert {pin.package for pin in pins} == {"torch", "torchvision", "accelerate", "transformers"}
    lock_installs_isaac = re.search(r"uv sync[^\n]*--locked[^\n]*--extra perceptron_isaac", workflow)
    if lock_installs_isaac is not None:
        lock = (ROOT / "uv.lock").read_text()
        for pin in pins:
            locked_versions = re.findall(
                rf'^name = "{re.escape(pin.package)}"\nversion = "([^"\n]+)"$', lock, re.M
            )
            assert pin.version in locked_versions, (
                f"Static recipe discrepancy: ISAAC CI uses the lock, but {CONSTRAINTS} selects "
                f"{pin.package}=={pin.version}; lock versions are {locked_versions}. "
                "Align to the executed training receipt; this is not runtime RED evidence."
            )
    else:
        assert CONSTRAINTS in workflow, "ISAAC CI must select the scoped recipe it claims to test"


def test_scoped_recipe_and_documented_versions_agree_without_selecting_new_versions() -> None:
    readme = (ROOT / "src/lerobot/policies/perceptron_isaac/README.md").read_text()
    pins = _stack_pins()
    assert len(pins) == 4
    for pin in pins:
        assert pin.version in readme, f"Documented ISAAC recipe omits {pin.package}=={pin.version}"


def test_scoped_recipe_keeps_authorized_builds_and_immutable_sdk_source() -> None:
    assert {pin.package: pin.version for pin in _stack_pins()} == {
        "torch": "2.10.0+cu128",
        "torchvision": "0.25.0+cu128",
        "accelerate": "1.15.0",
        "transformers": "5.5.4",
    }
    project = (ROOT / "pyproject.toml").read_text()
    assert (
        "mharmony[qwen35] @ git+https://github.com/perceptron-ai-inc/mharmony.git@"
        "9a57efbf8ada2d73641fc40081f4e4fa6ff2ab92"
    ) in project
    workflow = (ROOT / ".github/workflows/fast_tests.yml").read_text()
    assert 'assert torch.version.cuda == "12.8"' in workflow
    assert ".venv-isaac/bin/python -m pytest" in workflow
