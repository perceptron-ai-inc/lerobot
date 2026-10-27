"""Base installs register Isaac lazily and explain each missing optional dependency."""

import subprocess
import sys

import pytest

from lerobot.utils.import_utils import _transformers_available

_MISSING_IMPORTS = """
import importlib.abc
import importlib.util
import sys

blocked = set(sys.argv[1].split(","))
original_find_spec = importlib.util.find_spec

def find_spec(name, *args, **kwargs):
    if name.split(".")[0] in blocked:
        return None
    return original_find_spec(name, *args, **kwargs)

importlib.util.find_spec = find_spec

class MissingOptionalDependency(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in blocked:
            raise ModuleNotFoundError(fullname)

sys.meta_path.insert(0, MissingOptionalDependency())
"""

_MISSING_EXTRA = (
    _MISSING_IMPORTS
    + """
missing = sys.argv[2]

from lerobot.policies import PerceptronIsaacConfig, make_policy_config
assert isinstance(make_policy_config("perceptron_isaac"), PerceptronIsaacConfig)
from lerobot.policies.perceptron_isaac import PerceptronIsaacPolicy
assert not blocked.intersection(sys.modules), "importing the policy must keep extras lazy"

try:
    PerceptronIsaacPolicy(PerceptronIsaacConfig(device="cpu"))
except ImportError as exc:
    assert "lerobot[perceptron_isaac]" in str(exc), str(exc)
    assert missing in str(exc), str(exc)
else:
    raise AssertionError(f"construction must explain the missing {missing} extra")
"""
)


@pytest.mark.parametrize(
    ("blocked", "missing"),
    [
        pytest.param(("transformers", "peft"), "transformers", id="transformers"),
        pytest.param(
            ("mharmony",),
            "mharmony",
            id="mharmony",
            marks=pytest.mark.skipif(not _transformers_available, reason="requires Transformers first"),
        ),
    ],
)
def test_policy_extra_is_optional(blocked: tuple[str, ...], missing: str) -> None:
    result = subprocess.run(
        [sys.executable, "-c", _MISSING_EXTRA, ",".join(blocked), missing],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
