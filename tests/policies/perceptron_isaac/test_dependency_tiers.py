"""Missing ISAAC extras must skip only dependency-backed test cases."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize(
    ("test_file", "blocked", "required", "passed", "skipped"),
    [
        # Transformers requires tokenizers; the base tier omits both, not a broken partial install.
        pytest.param(
            "test_fast_processor.py",
            ("tokenizers", "transformers", "peft"),
            (),
            7,
            21,
            id="fast-base-tier",
        ),
        # Tamper/digest rejection needs artifact data, but never reaches Transformers.
        pytest.param(
            "test_fast_processor.py",
            ("transformers", "peft"),
            ("tokenizers",),
            8,
            20,
            id="fast-no-transformers",
        ),
        # Loading and security checks do not call the SciPy-backed codec.
        pytest.param(
            "test_fast_processor.py", ("scipy",), ("tokenizers", "transformers"), 9, 19, id="fast-no-scipy"
        ),
        pytest.param(
            "test_hardware_feature_order.py",
            ("transformers", "peft"),
            (),
            20,
            4,
            id="hardware-no-transformers",
        ),
        pytest.param("test_hardware_feature_order.py", ("mharmony",), (), 20, 4, id="hardware-no-mharmony"),
    ],
)
def test_missing_extra_preserves_dependency_free_cases(
    test_file: str, blocked: tuple[str, ...], required: tuple[str, ...], passed: int, skipped: int
) -> None:
    for dependency in required:
        pytest.importorskip(dependency, reason="isolated missing-extra check needs the other FAST extras")
    # A fresh interpreter prevents import caches from hiding a missing extra.
    script = """
import importlib.abc
import importlib.util
import sys
from collections.abc import Sequence
from importlib.machinery import ModuleSpec
from types import ModuleType

blocked = set(sys.argv[1].split(","))
original_find_spec = importlib.util.find_spec

def find_spec(name: str, package: str | None = None) -> ModuleSpec | None:
    if name.split(".")[0] in blocked:
        return None
    return original_find_spec(name, package)

importlib.util.find_spec = find_spec

class MissingOptionalDependency(importlib.abc.MetaPathFinder):
    def find_spec(
        self,
        fullname: str,
        path: Sequence[str] | None = None,
        target: ModuleType | None = None,
    ) -> ModuleSpec | None:
        if fullname.split(".")[0] in blocked:
            raise ModuleNotFoundError(f"Blocked optional dependency: {fullname}", name=fullname)

sys.meta_path.insert(0, MissingOptionalDependency())

import pytest

class CaseCounts:
    def __init__(self) -> None:
        self.passed = 0
        self.skipped = 0

    def pytest_runtest_logreport(self, report: pytest.TestReport) -> None:
        if report.when == "call" and report.passed:
            self.passed += 1
        if report.skipped:
            self.skipped += 1

counts = CaseCounts()
result = pytest.main([sys.argv[2], "-q", "--tb=short"], plugins=[counts])
assert result == 0, f"Missing extras must not produce failures or errors: {result}"
assert (counts.passed, counts.skipped) == (int(sys.argv[3]), int(sys.argv[4])), (
    "Dependency-free coverage must run; skip only dependency-backed cases",
    counts.passed,
    counts.skipped,
)
"""
    test_path = Path(__file__).with_name(test_file)
    result = subprocess.run(
        [sys.executable, "-c", script, ",".join(blocked), str(test_path), str(passed), str(skipped)],
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert result.returncode == 0, result.stdout + result.stderr
