"""Canonical-path load contracts for a raw Isaac-0.5 export.

A raw export nests the LeRobot policy package in ``lerobot_policy/`` and keeps its
documented assets (``fast_processor_pinned/``, ``isaac_stats.json``,
``policy_normalization.json``) at the export root. That is the layout
``lerobot_eval`` hands to ``make_pre_post_processors(pretrained_path=...)`` and to
``make_policy``, so resolution, the escape guard, stats discovery and the settle
branch all have to agree about which directory holds the assets.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file

pytest.importorskip("transformers", reason="transformers is required (install lerobot[perceptron_isaac])")

from lerobot.configs import FeatureType, PolicyFeature
from lerobot.policies import make_pre_post_processors
from lerobot.policies.perceptron_isaac.configuration_perceptron_isaac import (
    PerceptronIsaacConfig,
)
from lerobot.policies.perceptron_isaac.isaac_stats import (
    isaac_stats_from_state_dict,
    isaac_stats_value_differences,
    load_isaac_stats,
)
from lerobot.policies.perceptron_isaac.mharmony_native import (
    IsaacMharmonyRenderMetadata,
    load_native_isaac_stats,
    load_native_render_metadata,
    read_recipe_reserved_token_groups,
)
from lerobot.policies.perceptron_isaac.modeling_perceptron_isaac import (
    PerceptronIsaacPolicy,
)
from lerobot.policies.perceptron_isaac.processor_perceptron_isaac import (
    PerceptronIsaacMharmonyPackProcessorStep,
    make_perceptron_isaac_pre_post_processors,
)
from lerobot.utils.constants import ACTION, OBS_STATE

PACK_REGISTRY_NAME = "perceptron_isaac_mharmony_pack"
PACK_STATE_FILENAME = "policy_preprocessor_step_4_perceptron_isaac_mharmony_pack.safetensors"
# Same value the shipped export stores, because fast_processor_pinned/ is a sibling of
# lerobot_policy/ rather than a child of it.
SIBLING_FAST_PROCESSOR_PATH = "../fast_processor_pinned"
# Synthetic stand-in for the Genesis policy-state contract identity; only its format
# matters to the hydration path.
POLICY_STATE_CONTRACT_HASH = "3d5a0c1f9b7e4a2d8c6b5f4e3a2d1c0b9a8f7e6d5c4b3a291807f6e5d4c3b2a1"


def _features() -> dict[str, PolicyFeature]:
    return {
        "observation.images.image": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 256, 256)),
        "observation.images.image2": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 256, 256)),
        OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(8,)),
    }


def _write_isaac_stats(path: Path, *, action_q99: float = 1.0) -> None:
    path.write_text(
        json.dumps(
            {
                "schema": "flow_matching_stats_v1",
                "target_fps": 20.0,
                "action_horizon": 30,
                "action_dim": 7,
                "proprio_dim": 8,
                "action": {
                    "q01": [-1.0] * 7,
                    "q99": [action_q99] * 7,
                    "min": [-1.0] * 7,
                    "max": [action_q99] * 7,
                },
                "proprio": {
                    "q01": [-1.0] * 8,
                    "q99": [1.0] * 8,
                    "min": [-1.0] * 8,
                    "max": [1.0] * 8,
                },
            }
        )
    )


def _write_raw_isaac05_export(root: Path) -> Path:
    """Build the shipped raw-export layout and return its nested policy package."""
    root.mkdir(parents=True)
    (root / "config.json").write_text(json.dumps({"model_type": "isaac_0_5"}, sort_keys=True) + "\n")
    fast_processor = root / "fast_processor_pinned"
    fast_processor.mkdir()
    (fast_processor / "processor_config.json").write_text("{}\n")
    _write_isaac_stats(root / "isaac_stats.json")
    policy_dir = root / "lerobot_policy"
    policy_dir.mkdir()
    return policy_dir


def _write_policy_normalization(root: Path) -> None:
    root.joinpath("policy_normalization.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "entries": [
                    {
                        "policy_state_dataset": "libero",
                        "objective": "Flow",
                        "normalization_scope": "libero_spatial",
                        "policy_state_contract_version": 1,
                        "policy_state_contract_hash": POLICY_STATE_CONTRACT_HASH,
                    }
                ],
            }
        )
    )


def _config(
    policy_dir: Path,
    *,
    native_stats_path: str | None = None,
    fast_processor_path: str | None = None,
    fast_processor_tree_sha256: str | None = None,
    num_settle_steps: int = 0,
) -> PerceptronIsaacConfig:
    config = PerceptronIsaacConfig(
        device="cpu",
        chunk_size=30,
        n_action_steps=30,
        target_fps=20.0,
        policy_state_dataset="libero",
        native_stats_path=native_stats_path,
        fast_processor_path=fast_processor_path,
        fast_processor_tree_sha256=fast_processor_tree_sha256,
        num_settle_steps=num_settle_steps,
        input_features=_features(),
        output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(7,))},
    )
    # What draccus fills in from --policy.path, and what lerobot_eval.py hands to
    # make_pre_post_processors.
    config.pretrained_path = policy_dir
    return config


def _save_packaged_pipelines(
    policy_dir: Path,
    *,
    stats_path: Path,
    pack_overrides: dict[str, object],
) -> None:
    """Write the packaged pipeline pair the way an export ships it."""
    export_config = _config(policy_dir, native_stats_path=str(stats_path))
    preprocessor, postprocessor = make_perceptron_isaac_pre_post_processors(export_config)
    preprocessor.save_pretrained(policy_dir)
    postprocessor.save_pretrained(policy_dir)
    serialized_path = policy_dir / "policy_preprocessor.json"
    payload = json.loads(serialized_path.read_text())
    steps = [step for step in payload["steps"] if step["registry_name"] == PACK_REGISTRY_NAME]
    assert len(steps) == 1, "the packaged preprocessor must hold exactly one pack step"
    steps[0]["config"].update(pack_overrides)
    serialized_path.write_text(json.dumps(payload))


def _pack_step(preprocessor) -> PerceptronIsaacMharmonyPackProcessorStep:
    packs = [
        step for step in preprocessor.steps if isinstance(step, PerceptronIsaacMharmonyPackProcessorStep)
    ]
    assert len(packs) == 1
    return packs[0]


def test_raw_export_resolves_sibling_fast_processor_asset(tmp_path: Path) -> None:
    """Item 2: the packaged sibling-relative asset path must load, not raise."""
    root = tmp_path / "isaac_0_5-export"
    policy_dir = _write_raw_isaac05_export(root)
    _save_packaged_pipelines(
        policy_dir,
        stats_path=root / "isaac_stats.json",
        pack_overrides={"fast_processor_path": SIBLING_FAST_PROCESSOR_PATH},
    )

    preprocessor, _ = make_pre_post_processors(_config(policy_dir), pretrained_path=str(policy_dir))

    # The loader keeps the serialized spelling it resolved against the export root, so
    # compare the target rather than the literal string.
    resolved = Path(_pack_step(preprocessor).fast_processor_path)
    assert resolved.resolve() == (root / "fast_processor_pinned").resolve()
    assert resolved.is_dir()


def test_raw_export_still_refuses_assets_outside_the_export_root(tmp_path: Path) -> None:
    """Item 2: the escape guard survives; only the export root is in bounds."""
    root = tmp_path / "isaac_0_5-export"
    policy_dir = _write_raw_isaac05_export(root)
    (tmp_path / "outside_fast_processor").mkdir()
    _save_packaged_pipelines(
        policy_dir,
        stats_path=root / "isaac_stats.json",
        pack_overrides={"fast_processor_path": "../../outside_fast_processor"},
    )

    with pytest.raises(RuntimeError, match="escapes its package"):
        make_pre_post_processors(_config(policy_dir), pretrained_path=str(policy_dir))


def test_plain_package_still_refuses_sibling_assets(tmp_path: Path) -> None:
    """Item 2: a package that is not a raw Isaac-0.5 export keeps the strict root."""
    policy_dir = tmp_path / "package"
    policy_dir.mkdir()
    (tmp_path / "fast_processor_pinned").mkdir()
    stats_path = tmp_path / "isaac_stats.json"
    _write_isaac_stats(stats_path)
    _save_packaged_pipelines(
        policy_dir,
        stats_path=stats_path,
        pack_overrides={"fast_processor_path": SIBLING_FAST_PROCESSOR_PATH},
    )

    with pytest.raises(RuntimeError, match="escapes its package"):
        make_pre_post_processors(_config(policy_dir), pretrained_path=str(policy_dir))


def test_raw_export_hydrates_training_contract_from_root_policy_normalization(tmp_path: Path) -> None:
    """Item 5: policy_normalization.json lives at the export root, not in the package."""
    root = tmp_path / "isaac_0_5-export"
    policy_dir = _write_raw_isaac05_export(root)
    _write_policy_normalization(root)
    render_metadata = IsaacMharmonyRenderMetadata(
        dataset_name="libero",
        robot_type="libero",
        proprio_dim=8,
        action_dim=7,
        action_horizon=30,
        target_fps=20.0,
    )
    assert render_metadata.training_proprio_contract is None, "this is the triggering condition"
    _save_packaged_pipelines(
        policy_dir,
        stats_path=root / "isaac_stats.json",
        pack_overrides={
            "fast_processor_path": SIBLING_FAST_PROCESSOR_PATH,
            "render_metadata": render_metadata.to_json_dict(),
        },
    )

    preprocessor, _ = make_pre_post_processors(_config(policy_dir), pretrained_path=str(policy_dir))

    contract = _pack_step(preprocessor).render_metadata["training_proprio_contract"]
    assert contract is not None
    assert contract["contract_hash"] == POLICY_STATE_CONTRACT_HASH
    assert contract["target_dim"] == 8


def test_raw_export_resolves_stats_from_config_alone(tmp_path: Path) -> None:
    """Item 3: a config with no stats field finds the conventional export sidecar."""
    root = tmp_path / "isaac_0_5-export"
    policy_dir = _write_raw_isaac05_export(root)
    config = _config(policy_dir)
    assert config.native_stats_path is None and config.stats_path is None

    stats = load_native_isaac_stats(config)

    assert config.resolve_native_stats_path() == str(root / "isaac_stats.json")
    assert stats.action_dim == 7
    assert stats.proprio_dim == 8
    assert stats.target_fps == 20.0


def test_explicit_native_stats_path_overrides_the_export_sidecar(tmp_path: Path) -> None:
    """Item 3: the declared field stays authoritative."""
    root = tmp_path / "isaac_0_5-export"
    policy_dir = _write_raw_isaac05_export(root)
    override = tmp_path / "override_stats.json"
    _write_isaac_stats(override, action_q99=0.5)
    config = _config(policy_dir, native_stats_path=str(override))

    stats = load_native_isaac_stats(config)

    assert config.resolve_native_stats_path() == str(override)
    assert float(stats.action.q99[0]) == pytest.approx(0.5)


def test_config_resolved_stats_match_the_packaged_state_file(tmp_path: Path) -> None:
    """Item 3 parity: config-resolved stats equal the packaged processor stats."""
    root = tmp_path / "isaac_0_5-export"
    policy_dir = _write_raw_isaac05_export(root)
    _save_packaged_pipelines(
        policy_dir,
        stats_path=root / "isaac_stats.json",
        pack_overrides={"fast_processor_path": SIBLING_FAST_PROCESSOR_PATH},
    )
    config = _config(policy_dir)

    resolved_stats = load_isaac_stats(config.resolve_native_stats_path())
    packaged_stats = isaac_stats_from_state_dict(load_file(str(policy_dir / PACK_STATE_FILENAME)))

    assert isaac_stats_value_differences(resolved_stats, packaged_stats) == []
    assert resolved_stats.action_dim == packaged_stats.action_dim
    assert resolved_stats.proprio_dim == packaged_stats.proprio_dim
    assert resolved_stats.action_horizon == packaged_stats.action_horizon
    assert resolved_stats.schema == packaged_stats.schema
    assert resolved_stats.target_fps == packaged_stats.target_fps


def test_settle_branch_rejects_a_missing_renderer_asset_on_the_first_call(tmp_path: Path) -> None:
    """Item 6: a misconfigured renderer fails on call #0, not after the settle window."""
    root = tmp_path / "isaac_0_5-export"
    policy_dir = _write_raw_isaac05_export(root)
    config = _config(
        policy_dir,
        fast_processor_path=str(root / "absent_fast_processor"),
        fast_processor_tree_sha256=POLICY_STATE_CONTRACT_HASH,
        num_settle_steps=40,
    )
    policy = PerceptronIsaacPolicy(config)

    with pytest.raises(RuntimeError, match="fast_processor_path"):
        policy.select_action({})


def test_settle_branch_keeps_serving_canned_actions_for_a_correct_policy(tmp_path: Path) -> None:
    """Item 6: settle semantics are unchanged when the prerequisites hold."""
    root = tmp_path / "isaac_0_5-export"
    policy_dir = _write_raw_isaac05_export(root)
    config = _config(
        policy_dir,
        fast_processor_path=str(root / "fast_processor_pinned"),
        fast_processor_tree_sha256=POLICY_STATE_CONTRACT_HASH,
        num_settle_steps=40,
    )
    policy = PerceptronIsaacPolicy(config)

    action = policy.select_action({})

    assert action.shape == (1, 7)
    assert torch.isfinite(action).all()
    assert policy._settle_index == 1


# What the shipped step-100000 export declares in
# policy_inference_recipe.json -> recipes[0].rendering.reserved_token_groups: the FAST pool
# first at 249321 and the coord block second at 248320, i.e. descending offsets.
SHIPPED_RESERVED_TOKEN_GROUPS = [
    {"name": None, "offset": 249321, "size": 2048, "tokenizer": "physical-intelligence/fast"},
    {"name": "coord", "offset": 248320, "size": 1001, "tokenizer": None},
]


def _write_inference_recipe(root: Path, groups: list[dict]) -> Path:
    path = root / "policy_inference_recipe.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "recipes": [
                    {
                        "rendering": {
                            "mharmony_encoding": "QWEN35_HARMONY",
                            "reserved_token_groups": groups,
                            "system_default_hints": [],
                        }
                    }
                ],
            }
        )
    )
    return path


def test_raw_export_render_metadata_takes_reserved_groups_from_the_recipe(tmp_path: Path) -> None:
    """Item 4: the checkpoint's own recipe, not the config-derived default, names the blocks."""
    root = tmp_path / "isaac_0_5-export"
    policy_dir = _write_raw_isaac05_export(root)
    _write_inference_recipe(root, SHIPPED_RESERVED_TOKEN_GROUPS)
    config = _config(policy_dir)
    assert config.native_render_metadata_path is None

    metadata = load_native_render_metadata(config)

    assert config.resolve_native_recipe_path() == str(root / "policy_inference_recipe.json")
    assert metadata.mharmony_reserved_token_groups == SHIPPED_RESERVED_TOKEN_GROUPS
    coord_groups = [group for group in metadata.mharmony_reserved_token_groups if group["name"] == "coord"]
    assert len(coord_groups) == 1
    assert (coord_groups[0]["offset"], coord_groups[0]["size"]) == (248320, 1001)
    fast_groups = [
        group
        for group in metadata.mharmony_reserved_token_groups
        if group["tokenizer"] == "physical-intelligence/fast"
    ]
    assert len(fast_groups) == 1
    # The config-derived default omits the offset, which mHarmony bump-allocates to 248087.
    assert (fast_groups[0]["offset"], fast_groups[0]["size"]) == (249321, 2048)


def test_export_without_a_recipe_keeps_the_config_derived_groups(tmp_path: Path) -> None:
    """Item 4: no recipe beside the package means today's behaviour, unchanged."""
    root = tmp_path / "isaac_0_5-export"
    policy_dir = _write_raw_isaac05_export(root)
    config = _config(policy_dir)

    metadata = load_native_render_metadata(config)

    assert config.resolve_native_recipe_path() is None
    assert metadata.mharmony_reserved_token_groups == [
        {"tokenizer": "physical-intelligence/fast", "size": 2048}
    ]


def test_declared_render_metadata_path_still_wins_over_the_recipe(tmp_path: Path) -> None:
    """Item 4: an importer-written package keeps pointing at its own sidecar."""
    root = tmp_path / "isaac_0_5-export"
    policy_dir = _write_raw_isaac05_export(root)
    _write_inference_recipe(root, SHIPPED_RESERVED_TOKEN_GROUPS)
    config = _config(policy_dir)
    sidecar_groups = [{"name": "coord", "offset": 248320, "size": 1001, "tokenizer": None}]
    sidecar = tmp_path / "native_render_metadata.json"
    sidecar.write_text(
        json.dumps(
            {
                **IsaacMharmonyRenderMetadata.from_config(config).to_json_dict(),
                "mharmony_reserved_token_groups": sidecar_groups,
            }
        )
    )
    config.native_render_metadata_path = str(sidecar)

    metadata = load_native_render_metadata(config)

    assert metadata.mharmony_reserved_token_groups == sidecar_groups


@pytest.mark.parametrize(
    ("groups", "match"),
    [
        (
            [
                {"name": "coord", "offset": 248320, "size": 1001, "tokenizer": None},
                {"name": "coord", "offset": 250000, "size": 1001, "tokenizer": None},
            ],
            "exactly one",
        ),
        (
            [{"name": None, "size": 2048, "tokenizer": "physical-intelligence/fast"}],
            "explicit offset",
        ),
        ([{"name": "coord", "offset": 248320, "size": "1001", "tokenizer": None}], "must be an integer"),
        ([{"name": "coord", "offset": 248320, "size": 0, "tokenizer": None}], "must be >= 1"),
        (
            [
                {"name": "coord", "offset": 248320, "size": 1001, "tokenizer": None},
                {"name": None, "offset": 248400, "size": 2048, "tokenizer": "physical-intelligence/fast"},
            ],
            "overlap",
        ),
        ([], "non-empty list"),
        (["coord"], "must be an object"),
        ([{"name": "coord", "offset": 248320, "size": 1001, "extra": 1}], "unsupported keys"),
    ],
)
def test_ambiguous_recipe_reserved_groups_are_refused(tmp_path: Path, groups, match: str) -> None:
    """Item 4: a malformed or ambiguous group set never falls back to the default."""
    root = tmp_path / "isaac_0_5-export"
    policy_dir = _write_raw_isaac05_export(root)
    recipe = _write_inference_recipe(root, groups)
    config = _config(policy_dir)

    with pytest.raises(ValueError, match=match):
        read_recipe_reserved_token_groups(recipe)
    with pytest.raises(ValueError, match=match):
        load_native_render_metadata(config)


def test_recipe_groups_stay_subject_to_the_checkpoint_cross_check(tmp_path: Path) -> None:
    """Item 4: the recipe is read verbatim so config.json remains the independent check.

    The reserved groups are reused from the canonical MK1 contract test rather than
    rebuilt, because the cross-check being exercised is that contract's own rule.
    """
    from lerobot.policies.perceptron_isaac.mk1_checkpoint_contract import (
        Mk1CheckpointContract,
        Mk1CheckpointContractError,
    )
    from tests.policies.perceptron_isaac.test_mk1_checkpoint_contract import _config as _mk1_config

    root = tmp_path / "isaac_0_5-export"
    policy_dir = _write_raw_isaac05_export(root)
    wrong_coord_range = [
        {"name": None, "offset": 249321, "size": 2048, "tokenizer": "physical-intelligence/fast"},
        {"name": "coord", "offset": 247000, "size": 1001, "tokenizer": None},
    ]
    _write_inference_recipe(root, wrong_coord_range)
    config = _config(policy_dir)
    contract = Mk1CheckpointContract.parse_allowlisted(_mk1_config(test_geometry=False))

    metadata = load_native_render_metadata(config)

    assert metadata.mharmony_reserved_token_groups == wrong_coord_range
    with pytest.raises(Mk1CheckpointContractError, match="disagrees"):
        contract.validate_coord_reserved_token_groups(metadata.mharmony_reserved_token_groups)

    good_root = tmp_path / "good-export"
    good_policy_dir = _write_raw_isaac05_export(good_root)
    _write_inference_recipe(good_root, SHIPPED_RESERVED_TOKEN_GROUPS)
    accepted = load_native_render_metadata(_config(good_policy_dir))

    contract.validate_coord_reserved_token_groups(accepted.mharmony_reserved_token_groups)


RETAINED_TREE_SHA256 = "0" * 64


def _retained_record(*, relative: str, root: Path, absolute: Path) -> dict[str, str]:
    """Reproduce the retention-record shape ``_resolve_checkpoint_local_paths`` writes."""
    return {"absolute": str(absolute), "relative": relative, "root": str(root)}


def test_raw_export_resolves_a_retained_sibling_fast_processor_record(tmp_path: Path) -> None:
    """lerobot_eval order: make_policy resolves and records the sibling asset first, so the
    processor factory has to honour that record instead of re-refusing the '..' spelling."""
    root = tmp_path / "isaac_0_5-export"
    policy_dir = _write_raw_isaac05_export(root)
    _save_packaged_pipelines(
        policy_dir,
        stats_path=root / "isaac_stats.json",
        pack_overrides={"fast_processor_path": SIBLING_FAST_PROCESSOR_PATH},
    )
    config = _config(
        policy_dir,
        fast_processor_path=SIBLING_FAST_PROCESSOR_PATH,
        fast_processor_tree_sha256=RETAINED_TREE_SHA256,
    )
    # The real producer of the record, called exactly as the policy load path calls it.
    PerceptronIsaacPolicy._resolve_checkpoint_local_paths(config, policy_dir)
    record = config._checkpoint_relative_paths["fast_processor_path"]
    assert record["relative"] == SIBLING_FAST_PROCESSOR_PATH
    assert Path(record["absolute"]) == (root / "fast_processor_pinned").resolve()

    preprocessor, _ = make_pre_post_processors(config, pretrained_path=str(policy_dir))

    resolved = Path(_pack_step(preprocessor).fast_processor_path)
    assert resolved.resolve() == (root / "fast_processor_pinned").resolve()
    assert resolved.is_dir()


def test_retained_record_still_refuses_a_sibling_asset_in_a_plain_package(tmp_path: Path) -> None:
    """Guard: outside the raw-export layout the package root stays the only bound."""
    policy_dir = tmp_path / "package"
    policy_dir.mkdir()
    sibling = tmp_path / "fast_processor_pinned"
    sibling.mkdir()
    stats_path = tmp_path / "isaac_stats.json"
    _write_isaac_stats(stats_path)
    _save_packaged_pipelines(
        policy_dir,
        stats_path=stats_path,
        pack_overrides={"fast_processor_path": SIBLING_FAST_PROCESSOR_PATH},
    )
    config = _config(
        policy_dir,
        fast_processor_path=str(sibling.resolve()),
        fast_processor_tree_sha256=RETAINED_TREE_SHA256,
    )
    config._checkpoint_relative_paths = {
        "fast_processor_path": _retained_record(
            relative=SIBLING_FAST_PROCESSOR_PATH, root=policy_dir, absolute=sibling.resolve()
        )
    }

    with pytest.raises(RuntimeError, match="escapes its package"):
        make_pre_post_processors(config, pretrained_path=str(policy_dir))


def test_retained_record_still_refuses_a_target_outside_the_export_root(tmp_path: Path) -> None:
    """Guard: a record cannot buy passage out of the raw export's own root."""
    root = tmp_path / "isaac_0_5-export"
    policy_dir = _write_raw_isaac05_export(root)
    outside = tmp_path / "outside_fast_processor"
    outside.mkdir()
    _save_packaged_pipelines(
        policy_dir,
        stats_path=root / "isaac_stats.json",
        pack_overrides={"fast_processor_path": "../../outside_fast_processor"},
    )
    config = _config(
        policy_dir,
        fast_processor_path=str(outside.resolve()),
        fast_processor_tree_sha256=RETAINED_TREE_SHA256,
    )
    config._checkpoint_relative_paths = {
        "fast_processor_path": _retained_record(
            relative="../../outside_fast_processor", root=policy_dir, absolute=outside.resolve()
        )
    }

    with pytest.raises(RuntimeError, match="escapes its package"):
        make_pre_post_processors(config, pretrained_path=str(policy_dir))


def test_retained_record_that_disagrees_with_the_config_is_refused(tmp_path: Path) -> None:
    """Guard: the record must still describe the value the config actually carries."""
    root = tmp_path / "isaac_0_5-export"
    policy_dir = _write_raw_isaac05_export(root)
    _save_packaged_pipelines(
        policy_dir,
        stats_path=root / "isaac_stats.json",
        pack_overrides={"fast_processor_path": SIBLING_FAST_PROCESSOR_PATH},
    )
    config = _config(
        policy_dir,
        fast_processor_path=str((root / "fast_processor_pinned").resolve()),
        fast_processor_tree_sha256=RETAINED_TREE_SHA256,
    )
    config._checkpoint_relative_paths = {
        "fast_processor_path": _retained_record(
            relative=SIBLING_FAST_PROCESSOR_PATH,
            root=policy_dir,
            absolute=root / "some_other_directory",
        )
    }

    with pytest.raises(RuntimeError, match="Invalid retained processor path record"):
        make_pre_post_processors(config, pretrained_path=str(policy_dir))
