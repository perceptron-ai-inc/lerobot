"""Base installs must register Isaac without importing optional Transformers."""

import subprocess
import sys


def test_policy_registration_without_transformers() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import importlib.abc
import importlib.util
import sys

original_find_spec = importlib.util.find_spec
blocked = {"transformers", "peft"}
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

from lerobot.policies import PerceptronIsaacConfig, make_policy_config
assert isinstance(make_policy_config("perceptron_isaac"), PerceptronIsaacConfig)
assert "transformers" not in sys.modules
from lerobot.policies.perceptron_isaac import PerceptronIsaacPolicy
try:
    PerceptronIsaacPolicy(PerceptronIsaacConfig(device="cpu"))
except ImportError as exc:
    assert "lerobot[perceptron_isaac]" in str(exc)
else:
    raise AssertionError("policy construction must explain the missing extra")
""",
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
