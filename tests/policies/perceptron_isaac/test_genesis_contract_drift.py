"""Drift canary: fail when the genesis contract moves past what this repo implements.

The ISAAC importer and the vendored action-expert loader implement the genesis
policy-inference-recipe and DiT-expert contracts by value. Genesis has drifted before
without any signal here (schema 2 -> 7, four new observation keys, the ``dit`` expert
stamp), so this test reads the authoritative constants straight out of a local genesis
checkout and compares them with what this repo pins. It parses the genesis sources with
``ast`` -- importing genesis would drag in its full dependency stack.

Skipped when no genesis checkout is available (CI); on developer machines set
``GENESIS_REPO_PATH`` or keep a checkout at ``~/genesis-main`` / ``~/genesis``.
"""

import ast
import os
from pathlib import Path

import pytest

pytest.importorskip("transformers", reason="transformers is required (install lerobot[perceptron_isaac])")

from lerobot.policies.perceptron_isaac import checkpoint_import as ci, modeling_qwen35_vla as vla


def _genesis_root() -> Path | None:
    candidates = [
        os.environ.get("GENESIS_REPO_PATH"),
        Path.home() / "genesis-main",
        Path.home() / "genesis",
    ]
    for candidate in candidates:
        if candidate and (Path(candidate) / "genesis" / "core").is_dir():
            return Path(candidate)
    return None


GENESIS_ROOT = _genesis_root()

pytestmark = pytest.mark.skipif(
    GENESIS_ROOT is None,
    reason="no local genesis checkout (set GENESIS_REPO_PATH to enable the drift canary)",
)


def _eval_node(node: ast.expr, env: dict[str, object]) -> object:
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name):
        return env[node.id]
    if isinstance(node, ast.Set):
        return {_eval_node(item, env) for item in node.elts}
    if isinstance(node, ast.Tuple):
        return tuple(_eval_node(item, env) for item in node.elts)
    if isinstance(node, ast.List):
        return [_eval_node(item, env) for item in node.elts]
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "frozenset":
        return frozenset(_eval_node(node.args[0], env)) if node.args else frozenset()
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
        left = _eval_node(node.left, env)
        right = _eval_node(node.right, env)
        return set(left) | set(right)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        return _eval_node(node.left, env) + _eval_node(node.right, env)  # type: ignore[operator]
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult):
        return _eval_node(node.left, env) * _eval_node(node.right, env)  # type: ignore[operator]
    raise ValueError(f"unsupported node {ast.dump(node)[:80]}")


def _module_constants(path: Path) -> dict[str, object]:
    env: dict[str, object] = {}
    for statement in ast.parse(path.read_text()).body:
        targets: list[ast.expr] = []
        value: ast.expr | None = None
        if isinstance(statement, ast.Assign):
            targets, value = statement.targets, statement.value
        elif isinstance(statement, ast.AnnAssign) and statement.value is not None:
            targets, value = [statement.target], statement.value
        for target in targets:
            if isinstance(target, ast.Name) and value is not None:
                try:
                    env[target.id] = _eval_node(value, env)
                except (ValueError, KeyError):
                    continue
    return env


@pytest.fixture(scope="module")
def genesis_recipe_constants() -> dict[str, object]:
    return _module_constants(GENESIS_ROOT / "genesis/core/robotics/inference_recipe.py")


def test_recipe_schema_version_is_implemented(genesis_recipe_constants):
    version = genesis_recipe_constants["POLICY_INFERENCE_RECIPE_SCHEMA_VERSION"]
    assert version in ci._SUPPORTED_RECIPE_SCHEMA_VERSIONS, (
        f"genesis bumped POLICY_INFERENCE_RECIPE_SCHEMA_VERSION to {version}; this repo "
        f"implements {sorted(ci._SUPPORTED_RECIPE_SCHEMA_VERSIONS)}. Diff the genesis "
        "recipe contract and extend checkpoint_import.py before importing new exports."
    )


def test_recipe_key_sets_match_genesis(genesis_recipe_constants):
    genesis = genesis_recipe_constants
    assert set(genesis["_TOP_LEVEL_KEYS"]) == set(ci._RECIPE_KEYS)
    assert set(genesis["_RECIPE_KEYS"]) == set(ci._RECIPE_RECORD_KEYS)
    assert set(genesis["_LOWERING_KEYS"]) == set(ci._LOWERING_KEYS)
    assert set(genesis["_SELECTION_KEYS"]) == set(ci._SELECTION_KEYS)
    assert set(genesis["_OBSERVATION_KEYS"]) == set(ci._OBSERVATION_KEYS_V7), (
        "genesis changed the canonical observation key set; update _OBSERVATION_KEYS_V7 "
        "and the conditioning validation in checkpoint_import.py."
    )
    assert set(genesis["_FLOW_ACTION_KEYS"]) == set(ci._FLOW_ACTION_WIRE_KEYS)
    assert set(genesis["_FAST_ACTION_KEYS"]) == set(ci._FAST_ACTION_WIRE_KEYS)
    assert set(genesis["_FLOW_FAST_CONDITIONING_KEYS"]) == set(ci._FLOW_FAST_CONDITIONING_WIRE_KEYS)


def test_dit_action_expert_contract_matches_genesis():
    constants = _module_constants(GENESIS_ROOT / "genesis/core/rtc.py")
    assert (
        constants["DIT_ACTION_EXPERT_CONFIG_SCHEMA_VERSION"] == vla.GENESIS_DIT_ACTION_EXPERT_SCHEMA_VERSION
    ), "genesis bumped the DiT expert contract schema; update modeling_qwen35_vla.py."
    assert tuple(constants["DIT_ACTION_EXPERT_CONFIG_V1_FIELDS"]) == tuple(
        vla.GENESIS_DIT_ACTION_EXPERT_V1_FIELDS
    ), (
        "genesis changed the DiT expert contract fields; update "
        "GENESIS_DIT_ACTION_EXPERT_V1_FIELDS and re-verify sampling parity."
    )


def test_sibling_schema_pins_still_match_genesis():
    manifest = _module_constants(GENESIS_ROOT / "genesis/core/robotics/contract.py")
    normalization = _module_constants(GENESIS_ROOT / "genesis/core/robotics/normalization_artifact.py")
    artifacts = _module_constants(GENESIS_ROOT / "genesis/core/robotics/checkpoint_artifacts.py")
    # checkpoint_import.py pins these inline (see _validate_manifest / _validate_normalization).
    assert manifest["POLICY_STATE_CHECKPOINT_MANIFEST_SCHEMA_VERSION"] == 5
    assert normalization["POLICY_NORMALIZATION_ARTIFACT_SCHEMA_VERSION"] == 2
    assert (
        artifacts["POLICY_STATE_DCP_IDENTITY_SCHEMA_VERSION"] == ci.POLICY_STATE_DCP_IDENTITY_SCHEMA_VERSION
    )
