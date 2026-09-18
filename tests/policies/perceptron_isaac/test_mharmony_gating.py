"""Isaac must follow the documented optional-dependency gating convention for mharmony.

`docs/source/bring_your_own_policies.mdx` requires the availability flag to live in
`lerobot.utils.import_utils` and the constructor to call `require_package`, so that
importing the policy package works without the extra and constructing the policy
fails with an ImportError that names the extra.
"""

import subprocess
import sys

_BLOCK_MHARMONY_AND_CONSTRUCT = """
import importlib.abc
import importlib.util
import sys

original_find_spec = importlib.util.find_spec
blocked = {"mharmony"}


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

import lerobot.policies  # noqa: E402
from lerobot.policies import PerceptronIsaacConfig  # noqa: E402
from lerobot.policies.perceptron_isaac import PerceptronIsaacPolicy  # noqa: E402

assert "mharmony" not in sys.modules, "importing the policy must not import mharmony"

try:
    PerceptronIsaacPolicy(PerceptronIsaacConfig(device="cpu"))
except ImportError as exc:
    assert "lerobot[perceptron_isaac]" in str(exc), str(exc)
    assert "mharmony" in str(exc), str(exc)
else:
    raise AssertionError("policy construction without mharmony must raise ImportError")
"""


def test_mharmony_availability_flag_lives_in_import_utils() -> None:
    from lerobot.utils import import_utils

    assert isinstance(import_utils._mharmony_available, bool)


def test_policy_package_imports_and_construction_fails_clearly_without_mharmony() -> None:
    result = subprocess.run(
        [sys.executable, "-c", _BLOCK_MHARMONY_AND_CONSTRUCT],
        capture_output=True,
        text=True,
        timeout=300,
    )

    assert result.returncode == 0, result.stdout + result.stderr
