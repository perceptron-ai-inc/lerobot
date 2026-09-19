from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
import torch
import torch.distributed.checkpoint as dcp

import lerobot.policies.perceptron_isaac.checkpoint_import as checkpoint_import
import lerobot.policies.perceptron_isaac.fast_processor as fast_processor
from lerobot.configs import PreTrainedConfig
from lerobot.policies import make_pre_post_processors
from lerobot.policies.perceptron_isaac.checkpoint_import import (
    CONDITIONING_DISABLED_DEPLOYMENT,
    ISAAC_DEPLOYMENT_ADAPTER_FILENAME,
    POLICY_INFERENCE_RECIPE_FILENAME,
    POLICY_NORMALIZATION_FILENAME,
    POLICY_STATE_CONTRACT_FILENAME,
    POLICY_STATE_DCP_IDENTITY_BYTES,
    POLICY_STATE_DCP_IDENTITY_KEY,
    POLICY_STATE_IDENTITY_FILENAME,
    IsaacCheckpointImportError,
    IsaacConditioningDeployment,
    authenticate_isaac_checkpoint,
    build_dcp_identity,
    canonical_sha256,
    convert_hf_weights_once,
    file_sha256,
    import_authenticated_isaac_checkpoint,
    materialize_qwen35_mharmony_vocab,
    validate_isaac_deployment_adapter,
)
from lerobot.policies.perceptron_isaac.configuration_perceptron_isaac import PerceptronIsaacConfig
from lerobot.policies.perceptron_isaac.fast_processor import fast_processor_tree_identity
from lerobot.policies.perceptron_isaac.processor_perceptron_isaac import (
    PerceptronIsaacActionUnnormalizeProcessorStep,
    PerceptronIsaacMharmonyPackProcessorStep,
)
from lerobot.policies.perceptron_isaac.qwen35_checkpoint import QWEN35_IMPORT_PROVENANCE_SCHEMA
from tests.policies.perceptron_isaac.test_mk1_checkpoint_contract import (
    _config as _mk1_config,
    _write_checkpoint as _write_mk1_checkpoint,
)

_CONTRACT_FILENAMES = (
    POLICY_STATE_CONTRACT_FILENAME,
    POLICY_NORMALIZATION_FILENAME,
    POLICY_INFERENCE_RECIPE_FILENAME,
)


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _write_qwen35_tokenizer_fixture(path: Path) -> None:
    _write_json(path, {"version": "1", "model": {"vocab": {"!": 0, '"': 1}}})


def _write_reduced_mk1_tokenizer_fixture(path: Path) -> None:
    _write_json(
        path,
        {
            "version": "1",
            "model": {"vocab": {"!": 0, "<|endoftext|>": 1}},
            "added_tokens": [
                {"content": "<|vision_start|>", "id": 2, "special": True},
                {"content": "<|vision_end|>", "id": 3, "special": True},
                {"content": "<|image_pad|>", "id": 4, "special": True},
                {"content": "<|video_pad|>", "id": 5, "special": True},
            ],
        },
    )


def _write_minimal_hf_inference_fixture(hf_export: Path, *, weight: float = 1.5) -> None:
    from safetensors.torch import save_file

    _write_json(hf_export / "config.json", {"model_type": "qwen3_5"})
    _write_qwen35_tokenizer_fixture(hf_export / "tokenizer.json")
    save_file(
        {"model.language_model.norm.weight": torch.full((2,), weight)},
        hf_export / "model.safetensors",
    )


def _write_fast_processor_fixture(tmp_path: Path) -> Path:
    artifact = tmp_path / "fast_processor_source"
    artifact.mkdir()
    _write_json(artifact / "config.json", {"processor_class": "SyntheticFastProcessor"})
    (artifact / "processing_action_tokenizer.py").write_text("# synthetic pinned remote code\n")
    return artifact


def _allow_synthetic_fast_artifact(monkeypatch, artifact: Path) -> None:
    monkeypatch.setattr(
        fast_processor,
        "DEFAULT_FAST_PROCESSOR_TREE_SHA256",
        fast_processor_tree_identity(artifact)[0],
    )
    monkeypatch.setattr(
        checkpoint_import,
        "DEFAULT_FAST_PROCESSOR_TREE_SHA256",
        fast_processor_tree_identity(artifact)[0],
    )
    monkeypatch.setattr(checkpoint_import, "QWEN35_BASE_VOCAB_SIZE", 2)


def _make_contracts() -> tuple[dict, dict, dict]:
    digest_a = "a" * 64
    digest_b = "b" * 64
    dataset = "cloud/isaac_yam"
    scope = "yam"
    action_layout = [
        *(f"left_joint_{index}.pos" for index in range(6)),
        "left_gripper.pos",
        *(f"right_joint_{index}.pos" for index in range(6)),
        "right_gripper.pos",
    ]
    deployment_profile = {
        "robot_type": "bi_yam",
        "camera": {
            "observation_keys": ["image.top", "image.left", "image.right"],
            "prompt_views": ["external_high", "left_wrist", "right_wrist"],
        },
        "state_action_schema": {
            "name": "joint_gripper_14",
            "control_mode": None,
            "components": action_layout,
        },
        "normalization": None,
    }
    manifest = {
        "schema_version": 5,
        "policy_state_contract_version": 1,
        "robotics_config_sha256": digest_a,
        "training_dataset_expression_sha256": digest_b,
        "datasets": {
            dataset: {
                "contract_hash": digest_a,
                "contract_version": 1,
                "deployment_profile_hash": canonical_sha256(deployment_profile),
                "mode": "components",
                "normalization_identities": [{"normalization_scope": scope, "objective": "Flow"}],
                "target_dim": 14,
            }
        },
    }
    manifest_digest = canonical_sha256(manifest)
    identity = {
        "policy_state_dataset": dataset,
        "normalization_scope": scope,
        "objective": "Flow",
    }
    entry = {
        **identity,
        "schema": "joint_gripper_14",
        "action": {
            "q01": [-1.0] * 14,
            "q99": [1.0] * 14,
        },
        "proprio": {
            "q01": [-1.0] * 14,
            "q99": [1.0] * 14,
        },
        "action_dim": 14,
        "proprio_dim": 14,
        "action_horizon": 30,
        "target_fps": 30.0,
        "policy_state_contract_hash": digest_a,
        "policy_state_contract_version": 1,
        "action_normalized": True,
        "proprio_normalized": True,
        "action_normalization_eps": 1e-6,
        "proprio_normalization_eps": 1e-6,
        "clip_normalized_actions": True,
        "clip_normalized_max": 10.0,
        "action_representation": "absolute",
        "relative_exclude_joints": [],
        "stats_key": "action",
    }
    entry["stats_sha256"] = canonical_sha256(entry)
    normalization = {
        "schema_version": 2,
        "policy_state_manifest_sha256": manifest_digest,
        "entries_sha256": canonical_sha256([entry]),
        "entries": [entry],
        "coverage": {"complete": True, "observed": [identity], "missing": []},
    }
    recipe_record = {
        **identity,
        "lowering_transform": "trajectory_flow_matching_pipeline",
        "lowering": {
            "layout": "pi_fast",
            "selection": {
                "observations_per_action": 1,
                "observation_window_seconds": None,
                "action_window_seconds": None,
                "action_anchor": "random",
                "action_anchor_index": None,
                "observation_delta_indices": [0],
                "action_delta_indices": list(range(30)),
            },
            "observation": {
                "inline_media_bytes": True,
                "materialize_media": True,
                "emit_vector_observations": True,
                "emit_observation_timestamps": True,
                "timestamp_precision": None,
                "last_state_only": True,
                "include_depth_images": False,
                "include_reasoning_trace": False,
                "include_scene_description": False,
                "emit_scene_prediction_target": False,
                "scene_description_probability": 1.0,
                "reasoning_max_points": 8,
                "depth_key_substrings": ["depth"],
                "clip_window_seconds": 0.1,
            },
            "action_wire": {
                "kind": "flow_action",
                "stats_key": "action",
                "require_stats": True,
                "normalize": True,
                "eps": 1e-6,
                "clip_normalized_actions": True,
                "clip_normalized_max": 10.0,
                "action_representation": "absolute",
                "relative_exclude_joints": [],
                "drop_action_payload": True,
                "token_group_name": "flow_action",
                "marker_token_id": 0,
            },
        },
        "transform_chain": [
            {
                "resolution_phase": "post_resolve",
                "policy": "folded",
                "config": {"name": "sample_trajectory_lowering", "p": 1.0},
            }
        ],
        "deployment_profile": deployment_profile,
        "model_config_sha256": digest_b,
        "rendering": {
            "mharmony_encoding": "QWEN35_HARMONY",
            "reserved_token_groups": [
                {
                    "name": None,
                    "offset": 248087,
                    "size": 2048,
                    "tokenizer": "physical-intelligence/fast",
                }
            ],
            "system_default_hints": [],
        },
        "image_preprocess": {
            "patch_size": 16,
            "pixel_shuffle_scale_factor": 2,
            "temporal_patch_size": 2,
            "max_num_patches": 576,
            "min_num_patches": None,
        },
    }
    recipe = {
        "schema_version": 2,
        "policy_state_manifest_sha256": manifest_digest,
        "recipes_sha256": canonical_sha256([recipe_record]),
        "recipes": [recipe_record],
    }
    return manifest, normalization, recipe


def _upgrade_recipe_to_schema7(recipe: dict, **observation_overrides) -> dict:
    upgraded = json.loads(json.dumps(recipe))
    upgraded["schema_version"] = 7
    for record in upgraded["recipes"]:
        observation = record["lowering"]["observation"]
        observation.update(
            {
                "action_conditioning_probability": 0.0,
                "action_conditioning_role": "user",
                "mistake_conditioning_probability": 0.0,
                "mistake_conditioning_override": None,
                **observation_overrides,
            }
        )
    upgraded["recipes_sha256"] = canonical_sha256(upgraded["recipes"])
    return upgraded


def _write_contract_bundle(
    hf_export: Path,
    dcp_checkpoint: Path,
    contracts: tuple[dict, dict, dict],
    *,
    valid_identity: bool = True,
) -> bytes:
    identity = build_dcp_identity(*contracts)
    if not valid_identity:
        identity = bytes(len(identity))
    dcp.save(
        {POLICY_STATE_DCP_IDENTITY_KEY: torch.tensor(tuple(identity), dtype=torch.uint8)},
        storage_writer=dcp.FileSystemWriter(dcp_checkpoint, per_thread_copy_ahead=0),
    )
    for root in (hf_export, dcp_checkpoint):
        for filename, value in zip(_CONTRACT_FILENAMES, contracts, strict=True):
            _write_json(root / filename, value)
    return identity


def _write_deployment_adapter(path: Path, contract_root: Path, **overrides) -> Path:
    payload = {
        "schema": "perceptron_isaac_deployment_adapter_v1",
        "contract_sha256": {
            filename: file_sha256(contract_root / filename) for filename in _CONTRACT_FILENAMES
        },
        "policy_state_dataset": "cloud/isaac_yam",
        "normalization_scope": "yam",
        "objective": "Flow",
        "render_dataset_name": "molmoact2_bimanualyam",
        "robot_type": "bi_yam",
        "control_mode": "joint",
        "image_size": [360, 640],
        "camera_order": ["top", "left", "right"],
        "camera_views": ["external_high", "left_wrist", "right_wrist"],
        "n_action_steps": 30,
        "num_inference_steps": 10,
        "num_flow_samples": 1,
        "clip_action_pose": False,
        "gripper_binary_to_signed": False,
        "num_settle_steps": 0,
        "settle_gripper": -1.0,
        "normalize_task_text": False,
        "joint_signs": None,
        "joint_offsets": None,
        "normalization_profile_id": "synthetic/yam",
        "normalization_profile_scope": "source_dataset",
        "normalization_validation_status": "validated",
        "adapter_validation_status": "reviewed_offline",
        "provenance": {"source": "synthetic importer fixture"},
    }
    payload.update(overrides)
    _write_json(path, payload)
    return path


def _make_platform_contracts(
    platform: str, *, legacy_generic_profile: bool = False
) -> tuple[tuple[dict, dict, dict], dict]:
    manifest, normalization, recipe = _make_contracts()
    if platform == "so100_so101":
        components = [
            "main_shoulder_pan",
            "main_shoulder_lift",
            "main_elbow_flex",
            "main_wrist_flex",
            "main_wrist_roll",
            "main_gripper",
        ]
        spec = {
            "dataset": "molmoact2_so100_101",
            "scope": "so100_fixture",
            "robot_type": "so100_so101",
            "control_mode": "joint",
            "schema_name": "joint_gripper_6",
            "components": components,
            "camera": None,
            "camera_order": ["top", "side"],
            "camera_views": ["external_high", "external_side"],
            "image_size": [256, 256],
            "action_dim": 6,
            "proprio_dim": 6,
            "target_fps": 30.0,
            "render_dataset_name": "molmoact2_so100_101",
            "clip_action_pose": False,
            "num_settle_steps": 0,
            "normalize_task_text": False,
            "joint_signs": [1.0, -1.0, 1.0, 1.0, 1.0, 1.0],
            "joint_offsets": [0.0, 90.0, 90.0, 0.0, 0.0, 0.0],
        }
    elif platform == "libero":
        spec = {
            "dataset": "libero",
            "scope": "libero_spatial",
            "robot_type": "libero",
            "control_mode": "ee",
            "schema_name": "gripper_7",
            "components": ["world_vector", "rotation_delta", "gripper_closedness_action"],
            "camera": {
                "observation_keys": ["image", "wrist_image"],
                "prompt_views": ["primary", "wrist"],
            },
            "camera_order": ["image", "wrist_image"],
            "camera_views": ["primary", "wrist"],
            "image_size": [256, 256],
            "action_dim": 7,
            "proprio_dim": 8,
            "target_fps": 20.0,
            "render_dataset_name": "libero",
            "clip_action_pose": True,
            # 10 LiberoEnv reset no-ops + 40 policy idle frames == the Genesis reference's 50.
            "num_settle_steps": 40,
            "normalize_task_text": True,
            "joint_signs": None,
            "joint_offsets": None,
        }
    else:
        raise ValueError(f"Unsupported synthetic platform {platform!r}.")

    profile = {
        "robot_type": "generic" if platform == "libero" and legacy_generic_profile else spec["robot_type"],
        "camera": spec["camera"],
        "state_action_schema": {
            "name": spec["schema_name"],
            "control_mode": (
                None if platform == "libero" and legacy_generic_profile else spec["control_mode"]
            ),
            "components": spec["components"],
        },
        "normalization": None,
    }
    dataset_record = next(iter(manifest["datasets"].values()))
    dataset_record["deployment_profile_hash"] = canonical_sha256(profile)
    dataset_record["normalization_identities"] = [{"normalization_scope": spec["scope"], "objective": "Flow"}]
    dataset_record["target_dim"] = spec["proprio_dim"]
    manifest["datasets"] = {spec["dataset"]: dataset_record}
    manifest_digest = canonical_sha256(manifest)

    identity = {
        "policy_state_dataset": spec["dataset"],
        "normalization_scope": spec["scope"],
        "objective": "Flow",
    }
    entry = normalization["entries"][0]
    entry.update(identity)
    entry["action"] = {
        "q01": [-1.0] * spec["action_dim"],
        "q99": [1.0] * spec["action_dim"],
    }
    entry["proprio"] = {
        "q01": [-1.0] * spec["proprio_dim"],
        "q99": [1.0] * spec["proprio_dim"],
    }
    entry["action_dim"] = spec["action_dim"]
    entry["proprio_dim"] = spec["proprio_dim"]
    entry["schema"] = spec["schema_name"]
    entry["target_fps"] = spec["target_fps"]
    entry["stats_sha256"] = canonical_sha256(
        {key: value for key, value in entry.items() if key != "stats_sha256"}
    )
    normalization.update(
        {
            "policy_state_manifest_sha256": manifest_digest,
            "entries_sha256": canonical_sha256([entry]),
            "coverage": {"complete": True, "observed": [identity], "missing": []},
        }
    )

    recipe_record = recipe["recipes"][0]
    recipe_record.update(identity)
    recipe_record["deployment_profile"] = profile
    recipe.update(
        {
            "policy_state_manifest_sha256": manifest_digest,
            "recipes_sha256": canonical_sha256([recipe_record]),
        }
    )
    return (manifest, normalization, recipe), spec


def _write_platform_authenticated_fixture(
    tmp_path: Path, platform: str, *, legacy_generic_profile: bool = False
) -> tuple[Path, Path, Path]:
    hf_export = tmp_path / f"{platform}_hf_export"
    dcp_checkpoint = tmp_path / f"{platform}_dcp"
    hf_export.mkdir()
    _write_minimal_hf_inference_fixture(hf_export)
    contracts, spec = _make_platform_contracts(
        platform,
        legacy_generic_profile=legacy_generic_profile,
    )
    _write_contract_bundle(hf_export, dcp_checkpoint, contracts)
    adapter_path = _write_deployment_adapter(
        tmp_path / f"{platform}_{ISAAC_DEPLOYMENT_ADAPTER_FILENAME}",
        hf_export,
        policy_state_dataset=spec["dataset"],
        normalization_scope=spec["scope"],
        render_dataset_name=spec["render_dataset_name"],
        robot_type=spec["robot_type"],
        control_mode=spec["control_mode"],
        image_size=spec["image_size"],
        camera_order=spec["camera_order"],
        camera_views=spec["camera_views"],
        clip_action_pose=spec["clip_action_pose"],
        num_settle_steps=spec["num_settle_steps"],
        normalize_task_text=spec["normalize_task_text"],
        joint_signs=spec["joint_signs"],
        joint_offsets=spec["joint_offsets"],
        normalization_profile_id=f"synthetic/{platform}",
    )
    return hf_export, dcp_checkpoint, adapter_path


def _write_authenticated_fixture(
    tmp_path: Path,
    *,
    contracts: tuple[dict, dict, dict] | None = None,
    valid_identity: bool = True,
    with_hf_identity: bool = False,
) -> tuple[Path, Path, Path]:
    hf_export = tmp_path / "hf_export"
    dcp_checkpoint = tmp_path / "dcp"
    hf_export.mkdir()
    _write_minimal_hf_inference_fixture(hf_export)
    contracts = contracts or _make_contracts()
    identity = _write_contract_bundle(
        hf_export,
        dcp_checkpoint,
        contracts,
        valid_identity=valid_identity,
    )
    if with_hf_identity:
        _write_hf_identity(hf_export, identity)
    adapter_path = _write_deployment_adapter(tmp_path / ISAAC_DEPLOYMENT_ADAPTER_FILENAME, hf_export)
    return hf_export, dcp_checkpoint, adapter_path


def _write_hf_identity(
    hf_export: Path,
    identity: bytes,
    *,
    dtype: torch.dtype = torch.uint8,
    extra_key: bool = False,
) -> Path:
    from safetensors.torch import save_file

    tensors = {POLICY_STATE_DCP_IDENTITY_KEY: torch.tensor(tuple(identity), dtype=dtype)}
    if extra_key:
        tensors["unexpected"] = torch.zeros(1, dtype=torch.uint8)
    output = hf_export / POLICY_STATE_IDENTITY_FILENAME
    save_file(tensors, output)
    return output


def _write_hf_authenticated_fixture(tmp_path: Path, *, contracts=None) -> tuple[Path, Path, Path]:
    return _write_authenticated_fixture(tmp_path, contracts=contracts, with_hf_identity=True)


def test_authenticated_import_contract_matches_dcp_identity(tmp_path):
    hf_export, dcp_checkpoint, _adapter_path = _write_authenticated_fixture(tmp_path)

    contracts = authenticate_isaac_checkpoint(hf_export, dcp_checkpoint)

    assert contracts.contract_authentication == "dcp_authenticated"
    assert len(contracts.dcp_identity_sha256) == 64
    assert set(contracts.source_hashes) == {
        POLICY_STATE_CONTRACT_FILENAME,
        POLICY_NORMALIZATION_FILENAME,
        POLICY_INFERENCE_RECIPE_FILENAME,
        "config.json",
        "tokenizer.json",
        "model.safetensors",
    }
    assert len(contracts.source_artifact_manifest_sha256) == 64
    assert len(contracts.authenticated_import_identity_sha256) == 64


def test_authenticated_import_accepts_current_genesis_recipe_schema7(tmp_path):
    manifest, normalization, recipe = _make_contracts()
    recipe = _upgrade_recipe_to_schema7(recipe)
    hf_export, dcp_checkpoint, _adapter_path = _write_authenticated_fixture(
        tmp_path,
        contracts=(manifest, normalization, recipe),
    )

    contracts = authenticate_isaac_checkpoint(hf_export, dcp_checkpoint)

    assert contracts.inference_recipe["schema_version"] == 7


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("action_conditioning_probability", 1.1, "conditioning probabilities"),
        ("action_conditioning_role", "system", "must be 'user' or 'agent'"),
        ("mistake_conditioning_override", True, "requires mistake_conditioning_probability"),
    ],
)
def test_authenticated_import_rejects_invalid_schema7_conditioning(tmp_path, field, value, match):
    manifest, normalization, recipe = _make_contracts()
    recipe = _upgrade_recipe_to_schema7(recipe, **{field: value})
    hf_export, dcp_checkpoint, _adapter_path = _write_authenticated_fixture(
        tmp_path,
        contracts=(manifest, normalization, recipe),
    )

    with pytest.raises(IsaacCheckpointImportError, match=match):
        authenticate_isaac_checkpoint(hf_export, dcp_checkpoint)


def test_authenticated_import_identity_binds_model_weights(tmp_path):
    from safetensors.torch import save_file

    hf_export, dcp_checkpoint, _adapter_path = _write_authenticated_fixture(tmp_path)
    original = authenticate_isaac_checkpoint(hf_export, dcp_checkpoint)

    save_file(
        {"model.language_model.norm.weight": torch.full((2,), 9.5)},
        hf_export / "model.safetensors",
    )
    mutated = authenticate_isaac_checkpoint(hf_export, dcp_checkpoint)

    assert mutated.contract_authentication == original.contract_authentication == "dcp_authenticated"
    assert mutated.dcp_identity_sha256 == original.dcp_identity_sha256
    assert mutated.source_hashes["model.safetensors"] != original.source_hashes["model.safetensors"]
    assert mutated.source_artifact_manifest_sha256 != original.source_artifact_manifest_sha256
    assert mutated.authenticated_import_identity_sha256 != original.authenticated_import_identity_sha256


def test_authenticated_import_accepts_hf_identity_sidecar_without_dcp(tmp_path):
    hf_export, _dcp_checkpoint, _adapter_path = _write_hf_authenticated_fixture(tmp_path)
    identity_path = hf_export / POLICY_STATE_IDENTITY_FILENAME

    contracts = authenticate_isaac_checkpoint(hf_export)

    assert contracts.contract_authentication == "hf_identity_authenticated"
    assert contracts.source_hashes[POLICY_STATE_IDENTITY_FILENAME] == file_sha256(identity_path)
    assert contracts.artifact_paths[POLICY_STATE_IDENTITY_FILENAME] == identity_path


def test_authenticated_import_rejects_hf_identity_mismatch(tmp_path):
    hf_export, _dcp_checkpoint, _adapter_path = _write_authenticated_fixture(tmp_path)
    _write_hf_identity(hf_export, bytes(POLICY_STATE_DCP_IDENTITY_BYTES))

    with pytest.raises(IsaacCheckpointImportError, match="HF policy-state identity does not authenticate"):
        authenticate_isaac_checkpoint(hf_export)


def test_authenticated_import_rejects_hf_identity_extra_tensor(tmp_path):
    hf_export, _dcp_checkpoint, _adapter_path = _write_authenticated_fixture(tmp_path)
    manifest, normalization, recipe = _make_contracts()
    _write_hf_identity(
        hf_export,
        build_dcp_identity(manifest, normalization, recipe),
        extra_key=True,
    )

    with pytest.raises(IsaacCheckpointImportError, match="must contain exactly"):
        authenticate_isaac_checkpoint(hf_export)


@pytest.mark.parametrize(
    ("identity", "dtype"),
    [
        (bytes(POLICY_STATE_DCP_IDENTITY_BYTES - 1), torch.uint8),
        (bytes(POLICY_STATE_DCP_IDENTITY_BYTES), torch.int16),
    ],
)
def test_authenticated_import_rejects_hf_identity_abi_drift(tmp_path, identity, dtype):
    hf_export, _dcp_checkpoint, _adapter_path = _write_authenticated_fixture(tmp_path)
    _write_hf_identity(hf_export, identity, dtype=dtype)

    with pytest.raises(IsaacCheckpointImportError, match=r"HF identity must be uint8\[100\]"):
        authenticate_isaac_checkpoint(hf_export)


def test_authenticated_import_rejects_dcp_identity_mismatch(tmp_path):
    hf_export, dcp_checkpoint, _adapter_path = _write_authenticated_fixture(tmp_path, valid_identity=False)

    with pytest.raises(IsaacCheckpointImportError, match="does not authenticate"):
        authenticate_isaac_checkpoint(hf_export, dcp_checkpoint)


def test_authenticated_import_rejects_converter_sidecar_drift(tmp_path):
    hf_export, dcp_checkpoint, _adapter_path = _write_authenticated_fixture(tmp_path)
    recipe_path = hf_export / POLICY_INFERENCE_RECIPE_FILENAME
    recipe = json.loads(recipe_path.read_text())
    recipe["recipes"][0]["deployment_profile"]["robot_type"] = "not-yam"
    _write_json(recipe_path, recipe)

    with pytest.raises(IsaacCheckpointImportError, match="differs from its DCP source"):
        authenticate_isaac_checkpoint(hf_export, dcp_checkpoint)


def test_authenticated_import_accepts_non_fast_reserved_group_for_flow_recipe(tmp_path):
    manifest, normalization, recipe = _make_contracts()
    recipe["recipes"][0]["rendering"]["reserved_token_groups"] = [
        {
            "name": "coord",
            "offset": 248320,
            "size": 1001,
        }
    ]
    recipe["recipes_sha256"] = canonical_sha256(recipe["recipes"])
    hf_export, dcp_checkpoint, _adapter_path = _write_authenticated_fixture(
        tmp_path,
        contracts=(manifest, normalization, recipe),
    )

    authenticate_isaac_checkpoint(hf_export, dcp_checkpoint)


def test_authenticated_import_accepts_shipped_fast_before_coord_group_order(tmp_path):
    """The real Isaac-0.5 export lists its FAST group before its lower-offset coord group.

    Those two blocks are exactly adjacent -- coord [248320, 249321) and FAST [249321, 251369) --
    so nothing overlaps; only the JSON list order is descending. mHarmony binds each group to its
    own declared offset regardless of position, so list order carries no meaning, and the shipped
    checkpoint was trained with precisely these ids (isaac05_coord_tokens / isaac05_fast_tokens in
    its own config.json, and MK1_FAST_TOKEN_OFFSET in mk1_checkpoint_contract).
    """
    manifest, normalization, recipe = _make_contracts()
    recipe["recipes"][0]["rendering"]["reserved_token_groups"] = [
        {
            "name": None,
            "offset": 249321,
            "size": 2048,
            "tokenizer": "physical-intelligence/fast",
        },
        {"name": "coord", "offset": 248320, "size": 1001},
    ]
    recipe["recipes_sha256"] = canonical_sha256(recipe["recipes"])
    hf_export, dcp_checkpoint, _adapter_path = _write_authenticated_fixture(
        tmp_path,
        contracts=(manifest, normalization, recipe),
    )

    authenticate_isaac_checkpoint(hf_export, dcp_checkpoint)


def test_authenticated_import_rejects_overlapping_reserved_groups(tmp_path):
    """Overlapping reserved ranges alias token ids, so they must stay rejected in any list order."""
    manifest, normalization, recipe = _make_contracts()
    recipe["recipes"][0]["rendering"]["reserved_token_groups"] = [
        {
            "name": None,
            "offset": 249321,
            "size": 2048,
            "tokenizer": "physical-intelligence/fast",
        },
        {"name": "coord", "offset": 249000, "size": 1001},
    ]
    recipe["recipes_sha256"] = canonical_sha256(recipe["recipes"])
    hf_export, dcp_checkpoint, _adapter_path = _write_authenticated_fixture(
        tmp_path,
        contracts=(manifest, normalization, recipe),
    )

    with pytest.raises(IsaacCheckpointImportError, match="non-overlapping"):
        authenticate_isaac_checkpoint(hf_export, dcp_checkpoint)


def test_authenticated_import_rejects_deployment_profile_hash_drift(tmp_path):
    hf_export, dcp_checkpoint, _adapter_path = _write_authenticated_fixture(tmp_path)
    recipe_path = hf_export / POLICY_INFERENCE_RECIPE_FILENAME
    recipe = json.loads(recipe_path.read_text())
    recipe["recipes"][0]["deployment_profile"]["camera"]["prompt_views"][0] = "wrong_view"
    recipe["recipes_sha256"] = canonical_sha256(recipe["recipes"])
    _write_json(recipe_path, recipe)
    _write_json(dcp_checkpoint / POLICY_INFERENCE_RECIPE_FILENAME, recipe)

    with pytest.raises(IsaacCheckpointImportError, match="deployment_profile_hash"):
        authenticate_isaac_checkpoint(hf_export, dcp_checkpoint)


def test_authenticated_import_rejects_normalization_state_action_schema_drift(tmp_path):
    hf_export, dcp_checkpoint, _adapter_path = _write_authenticated_fixture(tmp_path)
    normalization_path = hf_export / POLICY_NORMALIZATION_FILENAME
    normalization = json.loads(normalization_path.read_text())
    entry = normalization["entries"][0]
    entry["schema"] = "wrong_schema"
    entry["stats_sha256"] = canonical_sha256(
        {key: value for key, value in entry.items() if key != "stats_sha256"}
    )
    normalization["entries_sha256"] = canonical_sha256(normalization["entries"])
    _write_json(normalization_path, normalization)
    _write_json(dcp_checkpoint / POLICY_NORMALIZATION_FILENAME, normalization)

    with pytest.raises(IsaacCheckpointImportError, match="state/action schemas disagree"):
        authenticate_isaac_checkpoint(hf_export, dcp_checkpoint)


def test_deployment_adapter_is_bound_to_exact_contract_files(tmp_path):
    hf_export, dcp_checkpoint, adapter_path = _write_authenticated_fixture(tmp_path)
    contracts = authenticate_isaac_checkpoint(hf_export, dcp_checkpoint)
    adapter = json.loads(adapter_path.read_text())
    adapter["contract_sha256"][POLICY_NORMALIZATION_FILENAME] = "0" * 64
    _write_json(adapter_path, adapter)

    with pytest.raises(IsaacCheckpointImportError, match="different policy_normalization.json"):
        validate_isaac_deployment_adapter(
            adapter_path,
            contracts=contracts,
            policy_state_dataset="cloud/isaac_yam",
            normalization_scope="yam",
            objective="Flow",
        )


def test_deployment_adapter_specializes_legacy_generic_libero_profile(tmp_path):
    hf_export, dcp_checkpoint, adapter_path = _write_platform_authenticated_fixture(
        tmp_path,
        "libero",
        legacy_generic_profile=True,
    )
    contracts = authenticate_isaac_checkpoint(hf_export, dcp_checkpoint)

    adapter = validate_isaac_deployment_adapter(
        adapter_path,
        contracts=contracts,
        policy_state_dataset="libero",
        normalization_scope="libero_spatial",
        objective="Flow",
    )

    assert adapter.payload["robot_type"] == "libero"


def _strict_yam_config(**overrides) -> PerceptronIsaacConfig:
    layout = [
        *(f"left_joint_{index}.pos" for index in range(6)),
        "left_gripper.pos",
        *(f"right_joint_{index}.pos" for index in range(6)),
        "right_gripper.pos",
    ]
    values = {
        "n_obs_steps": 1,
        "chunk_size": 30,
        "n_action_steps": 30,
        "action_dim": 14,
        "proprio_dim": 14,
        "num_inference_steps": 10,
        "num_flow_samples": 1,
        "image_size": (360, 640),
        "camera_order": ("top", "left", "right"),
        "dataset_name": "molmoact2_bimanualyam",
        "robot_type": "bi_yam",
        "control_mode": "joint",
        "clip_action_pose": False,
        "gripper_binary_to_signed": False,
        "num_settle_steps": 0,
        "normalize_task_text": False,
        "apply_offset_norm": False,
        "target_fps": 30.0,
        "action_feature_names": layout,
        "state_feature_names": layout,
        "strict_hardware_feature_contract": True,
    }
    values.update(overrides)
    return PerceptronIsaacConfig(**values)


def test_strict_yam_contract_rejects_unsafe_adapter_defaults():
    with pytest.raises(ValueError, match="clip_action_pose=True"):
        _strict_yam_config(clip_action_pose=True)


def test_strict_deployment_rejects_unknown_platform():
    with pytest.raises(ValueError, match="supported robot_type"):
        _strict_yam_config(robot_type="bi_yam_typo")


def test_current_post3279_converter_contracts_and_identity_authenticate():
    source = Path(__file__).parents[3] / "checkpoint/isaac_yam_post3279/step-60000"
    if not source.is_dir():
        pytest.skip("local post3279 converted checkpoint is unavailable")
    assert not (source / "policy_contracts_dummy_provenance.json").exists()
    provenance = json.loads((source / "policy_contracts_s3_provenance.json").read_text())
    assert provenance["contract_authentication"] == "hf_identity_authenticated"
    assert provenance["converter_output"] is True
    assert provenance["validation"] == {
        "authorized_identities": 1474,
        "normalization_entries": 1474,
        "production_schema_hash_coverage": "passed",
        "recipes": 1474,
        "selected_identity_present": True,
    }
    for filename, expected_hash in provenance["contract_sha256"].items():
        assert file_sha256(source / filename) == expected_hash
    identity_path = source / POLICY_STATE_IDENTITY_FILENAME
    assert provenance["policy_state_identity"]["file_sha256"] == file_sha256(identity_path)
    adapter_path = source / ISAAC_DEPLOYMENT_ADAPTER_FILENAME
    assert provenance["deployment_adapter"]["sha256"] == file_sha256(adapter_path)
    contracts = authenticate_isaac_checkpoint(source)
    assert contracts.contract_authentication == "hf_identity_authenticated"
    assert contracts.dcp_identity_sha256 == provenance["policy_state_identity"]["payload_sha256"]


def test_current_post3279_tokenizer_materializes_mharmony_vocab(tmp_path):
    source = Path(__file__).parents[3] / "checkpoint/isaac_yam_post3279/step-60000"
    if not source.is_dir():
        pytest.skip("local post3279 converted checkpoint is unavailable")

    output = tmp_path / "vocab.json"
    record = materialize_qwen35_mharmony_vocab(
        source / "tokenizer.json",
        output,
        source_vocab_path=source / "vocab.json",
    )
    vocab = json.loads(output.read_text())

    assert record["token_count"] == checkpoint_import.QWEN35_BASE_VOCAB_SIZE
    assert len(vocab) == checkpoint_import.QWEN35_BASE_VOCAB_SIZE
    assert set(vocab.values()) == set(range(checkpoint_import.QWEN35_BASE_VOCAB_SIZE))


def test_hf_import_applies_rmsnorm_correction_once(tmp_path, monkeypatch):
    from safetensors.torch import load_file, save_file

    source = tmp_path / "raw_hf"
    output = tmp_path / "imported_hf"
    source.mkdir()
    _write_json(source / "config.json", {"model_type": "qwen3_5"})
    _write_qwen35_tokenizer_fixture(source / "tokenizer.json")
    monkeypatch.setattr(checkpoint_import, "QWEN35_BASE_VOCAB_SIZE", 2)
    corrected_key = "model.language_model.layers.0.input_layernorm.weight"
    q_norm_key = "model.language_model.layers.0.self_attn.q_norm.weight"
    gated_key = "model.language_model.layers.0.linear_attn.norm.weight"
    visual_key = "model.visual.blocks.0.norm1.weight"
    raw_norm = torch.tensor([0.6728749871253967, 0.39779573678970337])
    save_file(
        {
            corrected_key: raw_norm,
            q_norm_key: torch.full((2,), 1.25),
            gated_key: torch.full((2,), 2.0),
            visual_key: torch.full((2,), 3.0),
        },
        source / "model.safetensors",
    )

    provenance = convert_hf_weights_once(source, output)
    converted = load_file(output / "model.safetensors")

    torch.testing.assert_close(
        converted[corrected_key],
        raw_norm.to(dtype=torch.bfloat16).sub(1.0),
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        converted[q_norm_key],
        torch.full((2,), 1.25).to(dtype=torch.bfloat16).sub(1.0),
        rtol=0.0,
        atol=0.0,
    )
    assert converted[corrected_key].dtype == torch.bfloat16
    torch.testing.assert_close(converted[gated_key], torch.full((2,), 2.0))
    torch.testing.assert_close(converted[visual_key], torch.full((2,), 3.0))
    assert provenance["converted_key_count"] == 2
    assert provenance["schema"] == "perceptron_isaac_rmsnorm_conversion_v2"
    assert provenance["conversion_dtype"] == "bfloat16"
    assert provenance["qwen35_vocab"]["token_count"] == 2
    assert json.loads((output / "vocab.json").read_text()) == {"!": 0, '"': 1}
    assert (output / "rmsnorm_conversion.json").is_file()

    with pytest.raises(IsaacCheckpointImportError, match="already marked RMSNorm-converted"):
        convert_hf_weights_once(output, tmp_path / "double_converted")


def test_mk1_hf_import_preserves_zero_centered_weights_byte_for_byte(tmp_path, monkeypatch):
    source = tmp_path / "mk1_hf"
    output = tmp_path / "imported_mk1_hf"
    _write_mk1_checkpoint(source)
    raw_config = json.loads((source / "config.json").read_text())
    raw_config["genesis_artifact"] = {
        "schema_version": 1,
        "artifact_kind": "neutral_debug",
        "trained_steps": 0,
    }
    _write_json(source / "config.json", raw_config)
    _write_reduced_mk1_tokenizer_fixture(source / "tokenizer.json")
    monkeypatch.setattr(checkpoint_import, "QWEN35_BASE_VOCAB_SIZE", 2)
    source_digests = {shard.name: file_sha256(shard) for shard in sorted(source.glob("model-*.safetensors"))}

    with pytest.raises(IsaacCheckpointImportError, match="test-only"):
        convert_hf_weights_once(source, tmp_path / "production_dispatch")
    provenance = checkpoint_import._copy_mk1_hf_weights_once(
        source,
        output,
        allow_test_only_reduced_geometry=True,
    )

    assert {
        shard.name: file_sha256(shard) for shard in sorted(output.glob("model-*.safetensors"))
    } == source_digests
    assert provenance["schema"] == "perceptron_isaac_mk1_model_import_v2"
    assert provenance["algorithm"] == "byte_for_byte_safetensors_copy"
    assert provenance["rmsnorm_weight_convention"] == "zero_centered_1_plus_weight"
    assert provenance["artifact"] == {
        "schema_version": 1,
        "artifact_kind": "neutral_debug",
        "trained_steps": 0,
    }
    assert set(provenance["files_sha256"]) == {
        "config.json",
        "model-00001-of-00002.safetensors",
        "model-00002-of-00002.safetensors",
        "model.safetensors.index.json",
        "tokenizer.json",
        "vocab.json",
    }
    assert provenance["files_sha256"]["config.json"] == provenance["source_files_sha256"]["config.json"]
    assert (output / "mk1_model_import.json").is_file()
    assert not (output / "rmsnorm_conversion.json").exists()
    contracts = checkpoint_import.AuthenticatedIsaacContracts(
        manifest={},
        normalization={},
        inference_recipe={},
        dcp_identity_sha256="0" * 64,
        source_hashes={
            path.name: file_sha256(path)
            for path in checkpoint_import._authenticated_inference_source_paths(source)
        },
    )
    checkpoint_import._verify_mk1_import_copy_binding(source, provenance, contracts)


def test_mk1_hf_import_reuses_authenticated_source_hashes(tmp_path, monkeypatch):
    source = tmp_path / "mk1_hf"
    output = tmp_path / "imported_mk1_hf"
    _write_mk1_checkpoint(source)
    raw_config = json.loads((source / "config.json").read_text())
    raw_config["genesis_artifact"] = {
        "schema_version": 1,
        "artifact_kind": "neutral_debug",
        "trained_steps": 0,
    }
    _write_json(source / "config.json", raw_config)
    _write_reduced_mk1_tokenizer_fixture(source / "tokenizer.json")
    monkeypatch.setattr(checkpoint_import, "QWEN35_BASE_VOCAB_SIZE", 2)

    source_paths = checkpoint_import._authenticated_inference_source_paths(source)
    real_file_sha256 = checkpoint_import.file_sha256
    authenticated_hashes = {path.name: real_file_sha256(path) for path in source_paths}
    hashed_paths = []

    def output_only_sha256(path):
        resolved = Path(path).resolve()
        assert not resolved.is_relative_to(source.resolve())
        hashed_paths.append(resolved)
        return real_file_sha256(path)

    monkeypatch.setattr(checkpoint_import, "file_sha256", output_only_sha256)

    provenance = checkpoint_import._copy_mk1_hf_weights_once(
        source,
        output,
        authenticated_source_hashes=authenticated_hashes,
        allow_test_only_reduced_geometry=True,
    )

    assert hashed_paths
    assert all(path.is_relative_to(output.resolve()) for path in hashed_paths)
    assert provenance["source_files_sha256"] == authenticated_hashes


def test_mk1_import_rejects_transient_source_swap_restored_after_copy(tmp_path, monkeypatch):
    """Copied bytes must match authentication, not merely the source's final state."""
    from safetensors.torch import load_file, save_file

    source = tmp_path / "mk1_hf"
    output = tmp_path / "imported_mk1_hf"
    _write_mk1_checkpoint(source)
    raw_config = json.loads((source / "config.json").read_text())
    raw_config["genesis_artifact"] = {
        "schema_version": 1,
        "artifact_kind": "trained_policy",
        "trained_steps": 10,
    }
    _write_json(source / "config.json", raw_config)
    _write_reduced_mk1_tokenizer_fixture(source / "tokenizer.json")
    monkeypatch.setattr(checkpoint_import, "QWEN35_BASE_VOCAB_SIZE", 2)

    source_paths = checkpoint_import._authenticated_inference_source_paths(source)
    authenticated_hashes = {path.name: file_sha256(path) for path in source_paths}
    contracts = checkpoint_import.AuthenticatedIsaacContracts(
        manifest={},
        normalization={},
        inference_recipe={},
        dcp_identity_sha256="0" * 64,
        source_hashes=authenticated_hashes,
    )
    shard = sorted(source.glob("model-*.safetensors"))[0]
    original_bytes = shard.read_bytes()
    tensors = load_file(shard)
    first_key = sorted(tensors)[0]
    tensors[first_key] = tensors[first_key].clone().fill_(1)
    save_file(tensors, shard)

    provenance = checkpoint_import._copy_mk1_hf_weights_once(
        source,
        output,
        allow_test_only_reduced_geometry=True,
    )
    shard.write_bytes(original_bytes)
    assert file_sha256(shard) == authenticated_hashes[shard.name]

    with pytest.raises(IsaacCheckpointImportError, match="unauthenticated source bytes"):
        checkpoint_import._verify_mk1_import_copy_binding(source, provenance, contracts)


def test_authenticated_mk1_import_finalizes_trained_package_manifest(tmp_path, monkeypatch):
    from lerobot.policies.perceptron_isaac.modeling_perceptron_isaac import PerceptronIsaacPolicy

    hf_export, dcp_checkpoint, adapter_path = _write_authenticated_fixture(tmp_path)
    output = tmp_path / "mk1_trained_package"
    fast_artifact = _write_fast_processor_fixture(tmp_path)
    _allow_synthetic_fast_artifact(monkeypatch, fast_artifact)
    original_convert = checkpoint_import.convert_hf_weights_once

    def synthetic_mk1_convert(source, destination, **kwargs):
        provenance = original_convert(source, destination, **kwargs)
        _write_json(Path(destination) / checkpoint_import.MK1_MODEL_IMPORT_FILENAME, {"synthetic": True})
        return provenance

    monkeypatch.setattr(
        checkpoint_import,
        "_validate_trained_mk1_import_source",
        lambda _source: (
            object(),
            {"schema_version": 1, "artifact_kind": "trained_policy", "trained_steps": 10},
        ),
    )
    monkeypatch.setattr(checkpoint_import, "convert_hf_weights_once", synthetic_mk1_convert)
    monkeypatch.setattr(checkpoint_import, "_verify_mk1_import_copy_binding", lambda *_args: None)
    monkeypatch.setattr(checkpoint_import, "_validate_mk1_recipe_coord_contract", lambda *_args: None)
    monkeypatch.setattr(
        checkpoint_import,
        "_derive_mk1_policy_fields",
        lambda *_args, **_kwargs: {
            "artifact_kind": "trained_policy",
            "trained_steps": 10,
            "vector_max_states": 128,
            "max_state_dim": 128,
            "max_action_dim": 64,
            "max_action_horizon": 64,
        },
    )

    import_authenticated_isaac_checkpoint(
        hf_export,
        dcp_checkpoint,
        output,
        policy_state_dataset="cloud/isaac_yam",
        normalization_scope="yam",
        deployment_adapter_path=adapter_path,
        fast_processor_source=fast_artifact,
        allow_fast_remote_code=True,
    )

    config = PreTrainedConfig.from_pretrained(output)
    manifest_path = output / checkpoint_import.MK1_TRAINED_PACKAGE_MANIFEST_FILENAME
    manifest = json.loads(manifest_path.read_text())
    assert config.mk1_trained_package_manifest_sha256 == file_sha256(manifest_path)
    assert manifest["mk1_model_import_sha256"] == config.mk1_model_import_sha256
    assert "policy_postprocessor.json" in manifest["files_sha256"]
    assert any(path.startswith("fast_processor/") for path in manifest["files_sha256"])
    assert all(not path.startswith("hf_model/") for path in manifest["files_sha256"])
    PerceptronIsaacPolicy._resolve_checkpoint_local_paths(config, output)
    PerceptronIsaacPolicy._verify_trained_package_manifest(config, output.resolve())


def _write_reduced_trained_mk1_package(tmp_path, monkeypatch):
    from safetensors.torch import save_file

    import lerobot.policies.perceptron_isaac.mk1_checkpoint_contract as contract_module
    import lerobot.policies.perceptron_isaac.modeling_perceptron_isaac as policy_module

    package = tmp_path / "package"
    package.mkdir()
    source = tmp_path / "source_model"
    contract = _write_mk1_checkpoint(source)
    raw_config = json.loads((source / "config.json").read_text())
    raw_config["genesis_artifact"] = {
        "schema_version": 1,
        "artifact_kind": "trained_policy",
        "trained_steps": 10,
    }
    _write_json(source / "config.json", raw_config)
    _write_reduced_mk1_tokenizer_fixture(source / "tokenizer.json")
    monkeypatch.setattr(checkpoint_import, "QWEN35_BASE_VOCAB_SIZE", 2)
    monkeypatch.setattr(policy_module, "_MK1_BASE_VOCAB_SIZE", 2)
    checkpoint_import._copy_mk1_hf_weights_once(
        source,
        package / "hf_model",
        allow_test_only_reduced_geometry=True,
    )
    monkeypatch.setattr(contract_module, "read_mk1_config", lambda _path: contract)

    adapter = package / ISAAC_DEPLOYMENT_ADAPTER_FILENAME
    _write_json(adapter, {"synthetic": True})
    _write_json(package / "policy_postprocessor.json", {"schema": "synthetic_postprocessor_v1"})
    save_file(
        {"action_q01": torch.zeros(contract.action_expert.action_dim)},
        package / "policy_postprocessor_step_0_perceptron_isaac_action_unnormalize.safetensors",
    )
    config = PerceptronIsaacConfig(
        hf_model_path=str(package / "hf_model"),
        artifact_kind="trained_policy",
        trained_steps=10,
        action_dim=contract.action_expert.action_dim,
        max_action_dim=contract.action_expert.action_dim,
        chunk_size=contract.action_expert.action_horizon,
        max_action_horizon=contract.action_expert.action_horizon,
        n_action_steps=contract.action_expert.action_horizon,
        proprio_dim=contract.vector_encoder.max_states,
        max_state_dim=contract.vector_encoder.max_states,
        vector_max_states=contract.vector_encoder.max_states,
        deployment_adapter_sha256=file_sha256(adapter),
        mk1_model_import_sha256=file_sha256(package / "hf_model" / "mk1_model_import.json"),
    )
    _refresh_trained_package_manifest(package, config)
    return package, config, contract


def _refresh_trained_package_manifest(package: Path, config: PerceptronIsaacConfig) -> None:
    # Production configs are relocatable; direct verifier tests use the
    # post-resolution absolute path held by from_pretrained.
    config.hf_model_path = "hf_model"
    config._save_pretrained(package)
    assert config.mk1_model_import_sha256 is not None
    config.mk1_trained_package_manifest_sha256 = checkpoint_import._write_mk1_trained_package_manifest(
        package,
        trained_steps=config.trained_steps,
        mk1_model_import_sha256=config.mk1_model_import_sha256,
    )
    config._save_pretrained(package)
    config.hf_model_path = str(package / "hf_model")


def test_trained_mk1_import_record_verifies_exhaustive_model_inventory(tmp_path, monkeypatch):
    from lerobot.policies.perceptron_isaac.modeling_perceptron_isaac import PerceptronIsaacPolicy

    package, config, _contract = _write_reduced_trained_mk1_package(tmp_path, monkeypatch)

    PerceptronIsaacPolicy._verify_packaged_contract_digests(config, package)
    reloaded_config = PreTrainedConfig.from_pretrained(package)
    PerceptronIsaacPolicy._resolve_checkpoint_local_paths(reloaded_config, package)
    PerceptronIsaacPolicy._verify_packaged_contract_digests(reloaded_config, package)

    rogue = package / "hf_model" / "unlisted.bin"
    rogue.write_bytes(b"unlisted")
    with pytest.raises(RuntimeError, match="file inventory"):
        PerceptronIsaacPolicy._verify_packaged_contract_digests(config, package)


def test_trained_mk1_package_rejects_unbound_root_weight_override(tmp_path, monkeypatch):
    from lerobot.policies.perceptron_isaac.modeling_perceptron_isaac import PerceptronIsaacPolicy

    package, config, _contract = _write_reduced_trained_mk1_package(tmp_path, monkeypatch)
    (package / "model.safetensors").write_bytes(b"unbound override")

    with pytest.raises(RuntimeError, match="file inventory"):
        PerceptronIsaacPolicy._verify_packaged_contract_digests(config, package)


def test_trained_mk1_dense_save_binds_one_inner_weight_inventory_after_move(tmp_path, monkeypatch):
    from lerobot.configs import FeatureType, PolicyFeature
    from lerobot.policies.perceptron_isaac.modeling_mk1_vla import load_mk1_vla_from_hf
    from lerobot.policies.perceptron_isaac.modeling_perceptron_isaac import PerceptronIsaacPolicy
    from lerobot.utils.constants import ACTION, OBS_STATE

    package, config, _contract = _write_reduced_trained_mk1_package(tmp_path, monkeypatch)
    config.hf_model_path = "hf_model"
    PerceptronIsaacPolicy._resolve_checkpoint_local_paths(config, package)
    config.input_features = {
        "observation.images.image": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 256, 256)),
        OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(config.proprio_dim,)),
    }
    config.output_features = {ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(config.action_dim,))}
    policy = PerceptronIsaacPolicy(config)
    policy._isaac_model, _, _ = load_mk1_vla_from_hf(
        package / "hf_model",
        dtype=torch.bfloat16,
        device="cpu",
        allow_test_only_reduced_geometry=True,
    )
    expected_state = {key: value.clone() for key, value in policy._isaac_model.state_dict().items()}

    saved = tmp_path / "saved-mk1"
    policy.save_pretrained(saved)
    manifest = json.loads((saved / checkpoint_import.MK1_TRAINED_PACKAGE_MANIFEST_FILENAME).read_text())
    assert manifest["model_storage"] == "canonical_inner"
    assert "hf_model/model.safetensors.index.json" in manifest["files_sha256"]
    assert "hf_model/model-00001-of-00001.safetensors" in manifest["files_sha256"]
    assert not list(saved.glob("model*.safetensors"))
    assert not (saved / "hf_model" / checkpoint_import.MK1_MODEL_IMPORT_FILENAME).exists()
    assert (saved / "hf_model" / "source_mk1_model_import.json").is_file()
    PerceptronIsaacPolicy._verify_saved_package(saved)

    # Checkpoint/Hub writers serialize processors after policy weights. The finalizer
    # must rebind that last writer's bytes before the package can move or upload.
    postprocessor = saved / "policy_postprocessor.json"
    postprocessor.write_text('{"schema": "updated_after_policy_save"}\n')
    with pytest.raises(RuntimeError, match="digest mismatch for policy_postprocessor.json"):
        PerceptronIsaacPolicy._verify_saved_package(saved)
    policy.finalize_pretrained_package(saved)
    finalized_manifest = json.loads(
        (saved / checkpoint_import.MK1_TRAINED_PACKAGE_MANIFEST_FILENAME).read_text()
    )
    assert finalized_manifest["files_sha256"][postprocessor.name] == file_sha256(postprocessor)
    PerceptronIsaacPolicy._verify_saved_package(saved)

    moved = tmp_path / "moved-mk1"
    saved.rename(moved)
    package.rename(tmp_path / "retired-mk1")

    reloaded = PerceptronIsaacPolicy.from_pretrained(moved, strict=True)
    reloaded_model, _, reloaded_contract = load_mk1_vla_from_hf(
        reloaded.config.hf_model_path,
        dtype=torch.bfloat16,
        device="cpu",
        allow_test_only_reduced_geometry=True,
    )
    assert reloaded_contract.mtp.present is False
    assert reloaded_model.state_dict().keys() == expected_state.keys()
    for key, expected in expected_state.items():
        torch.testing.assert_close(reloaded_model.state_dict()[key], expected)


def test_local_mk1_peft_package_verifies_embedded_base_after_source_is_moved(tmp_path, monkeypatch):
    from safetensors.torch import save_file

    import lerobot.policies.peft as peft_loader
    from lerobot.configs import FeatureType, PolicyFeature
    from lerobot.policies.perceptron_isaac.modeling_perceptron_isaac import PerceptronIsaacPolicy
    from lerobot.utils.constants import ACTION, OBS_STATE

    package, config, _contract = _write_reduced_trained_mk1_package(tmp_path, monkeypatch)
    config.hf_model_path = "hf_model"
    PerceptronIsaacPolicy._resolve_checkpoint_local_paths(config, package)
    config.input_features = {
        "observation.images.image": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 256, 256)),
        OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(config.proprio_dim,)),
    }
    config.output_features = {ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(config.action_dim,))}
    config.pretrained_path = package
    config.use_peft = True
    policy = PerceptronIsaacPolicy(config)
    adapter = tmp_path / "mk1-adapter"
    adapter.mkdir()
    config.save_pretrained(adapter)
    _write_json(
        adapter / "adapter_config.json",
        {"base_model_name_or_path": str(package), "revision": "local-source"},
    )
    save_file({"adapter": torch.ones(1)}, adapter / "adapter_model.safetensors")

    policy.finalize_pretrained_package(adapter)
    PerceptronIsaacPolicy._verify_saved_package(adapter / "base_model")
    moved = tmp_path / "moved-mk1-adapter"
    adapter.rename(moved)
    package.rename(tmp_path / "retired-mk1-base")

    class FakePeftConfig:
        @classmethod
        def from_pretrained(cls, path):
            raw = json.loads((Path(path) / "adapter_config.json").read_text())
            return type("AdapterConfig", (), raw)()

    monkeypatch.setattr(peft_loader, "require_package", lambda *args, **kwargs: None)
    monkeypatch.setattr(peft_loader, "PeftConfig", FakePeftConfig)
    monkeypatch.setattr(
        peft_loader,
        "PeftModel",
        type("FakePeftModel", (), {"from_pretrained": staticmethod(lambda base, path, **kwargs: base)}),
    )

    def fake_load_backbone(self, *, training=False):
        assert training is False
        self._isaac_model = torch.nn.Linear(2, 1)

    monkeypatch.setattr(PerceptronIsaacPolicy, "_load_backbone", fake_load_backbone)
    loaded_config = PreTrainedConfig.from_pretrained(moved)
    reloaded = peft_loader.load_peft_policy(PerceptronIsaacPolicy, loaded_config, moved)

    assert Path(reloaded.config.hf_model_path).is_relative_to(moved / "base_model")


@pytest.mark.parametrize(
    "filename",
    [
        "policy_postprocessor.json",
        "policy_postprocessor_step_0_perceptron_isaac_action_unnormalize.safetensors",
    ],
)
def test_trained_mk1_package_manifest_rejects_tampered_processor_asset(
    tmp_path,
    monkeypatch,
    filename,
):
    from lerobot.policies.perceptron_isaac.modeling_perceptron_isaac import PerceptronIsaacPolicy

    package, config, _contract = _write_reduced_trained_mk1_package(tmp_path, monkeypatch)
    target = package / filename
    target.write_bytes(target.read_bytes() + b"tampered")

    with pytest.raises(RuntimeError, match=rf"digest mismatch for {filename}"):
        PerceptronIsaacPolicy._verify_packaged_contract_digests(config, package)


def test_trained_mk1_package_manifest_rejects_missing_file(tmp_path, monkeypatch):
    from lerobot.policies.perceptron_isaac.modeling_perceptron_isaac import PerceptronIsaacPolicy

    package, config, _contract = _write_reduced_trained_mk1_package(tmp_path, monkeypatch)
    (package / "policy_postprocessor.json").unlink()

    with pytest.raises(RuntimeError, match="file inventory"):
        PerceptronIsaacPolicy._verify_packaged_contract_digests(config, package)


def test_trained_mk1_package_manifest_rejects_extra_file(tmp_path, monkeypatch):
    from lerobot.policies.perceptron_isaac.modeling_perceptron_isaac import PerceptronIsaacPolicy

    package, config, _contract = _write_reduced_trained_mk1_package(tmp_path, monkeypatch)
    (package / "unlisted.json").write_text("{}\n")

    with pytest.raises(RuntimeError, match="file inventory"):
        PerceptronIsaacPolicy._verify_packaged_contract_digests(config, package)


@pytest.mark.parametrize("filename", [".gitattributes", "README.md"])
def test_trained_mk1_package_manifest_allows_only_inert_hub_metadata(
    tmp_path,
    monkeypatch,
    filename,
):
    from lerobot.policies.perceptron_isaac.modeling_perceptron_isaac import PerceptronIsaacPolicy

    package, config, _contract = _write_reduced_trained_mk1_package(tmp_path, monkeypatch)
    metadata = package / filename
    metadata.write_text("publisher-generated metadata\n")

    # Hub metadata may be added after import and is never runtime input.
    PerceptronIsaacPolicy._verify_packaged_contract_digests(config, package)
    # Re-finalizing a package that already contains the metadata also excludes
    # it from the authenticated runtime/data inventory.
    _refresh_trained_package_manifest(package, config)
    manifest = json.loads((package / checkpoint_import.MK1_TRAINED_PACKAGE_MANIFEST_FILENAME).read_text())
    assert filename not in manifest["files_sha256"]
    metadata.write_text("metadata may change without runtime authority\n")
    PerceptronIsaacPolicy._verify_packaged_contract_digests(config, package)


@pytest.mark.parametrize("filename", [".gitattributes", "README.md"])
def test_trained_mk1_package_rejects_non_file_hub_metadata(tmp_path, monkeypatch, filename):
    from lerobot.policies.perceptron_isaac.modeling_perceptron_isaac import PerceptronIsaacPolicy

    package, config, _contract = _write_reduced_trained_mk1_package(tmp_path, monkeypatch)
    (package / filename).mkdir()

    with pytest.raises(RuntimeError, match="Hub metadata must be a regular package file"):
        PerceptronIsaacPolicy._verify_packaged_contract_digests(config, package)
    with pytest.raises(IsaacCheckpointImportError, match="Hub metadata must be a regular root file"):
        _refresh_trained_package_manifest(package, config)


@pytest.mark.parametrize("filename", [".gitattributes", "README.md"])
def test_trained_mk1_package_rejects_local_hub_metadata_symlink(tmp_path, monkeypatch, filename):
    from lerobot.policies.perceptron_isaac.modeling_perceptron_isaac import PerceptronIsaacPolicy

    package, config, _contract = _write_reduced_trained_mk1_package(tmp_path, monkeypatch)
    (package / filename).symlink_to(package / ISAAC_DEPLOYMENT_ADAPTER_FILENAME)

    with pytest.raises(RuntimeError, match="immutable Hub blob link"):
        PerceptronIsaacPolicy._verify_packaged_contract_digests(config, package)


def test_trained_mk1_package_manifest_rejects_config_mutation(tmp_path, monkeypatch):
    from lerobot.policies.perceptron_isaac.modeling_perceptron_isaac import PerceptronIsaacPolicy

    package, config, _contract = _write_reduced_trained_mk1_package(tmp_path, monkeypatch)
    config_path = package / "config.json"
    raw_config = json.loads(config_path.read_text())
    raw_config["num_flow_samples"] += 1
    _write_json(config_path, raw_config)

    with pytest.raises(RuntimeError, match="canonical config digest mismatch"):
        PerceptronIsaacPolicy._verify_packaged_contract_digests(config, package)


@pytest.mark.parametrize(
    "filename",
    [
        "config.json",
        "tokenizer.json",
        "vocab.json",
        "model.safetensors.index.json",
        "model-00001-of-00002.safetensors",
    ],
)
def test_trained_mk1_import_record_rejects_tampered_model_asset(tmp_path, monkeypatch, filename):
    from lerobot.policies.perceptron_isaac.modeling_perceptron_isaac import PerceptronIsaacPolicy

    package, config, _contract = _write_reduced_trained_mk1_package(tmp_path, monkeypatch)
    target = package / "hf_model" / filename
    target.write_bytes(target.read_bytes() + b"tampered")

    with pytest.raises(RuntimeError, match="digest mismatch"):
        PerceptronIsaacPolicy._verify_packaged_contract_digests(config, package)


def test_trained_mk1_import_record_schema_is_strict(tmp_path, monkeypatch):
    from lerobot.policies.perceptron_isaac.modeling_perceptron_isaac import PerceptronIsaacPolicy

    package, config, _contract = _write_reduced_trained_mk1_package(tmp_path, monkeypatch)
    record_path = package / "hf_model" / "mk1_model_import.json"
    record = json.loads(record_path.read_text())
    record["unchecked"] = True
    _write_json(record_path, record)
    config.mk1_model_import_sha256 = file_sha256(record_path)
    _refresh_trained_package_manifest(package, config)

    with pytest.raises(RuntimeError, match="fields do not match"):
        PerceptronIsaacPolicy._verify_packaged_contract_digests(config, package)


def test_trained_mk1_import_record_rejects_duplicate_json_keys(tmp_path, monkeypatch):
    from lerobot.policies.perceptron_isaac.modeling_perceptron_isaac import PerceptronIsaacPolicy

    package, config, _contract = _write_reduced_trained_mk1_package(tmp_path, monkeypatch)
    record_path = package / "hf_model" / "mk1_model_import.json"
    record_text = record_path.read_text()
    record_path.write_text(
        record_text.replace(
            '"schema": "perceptron_isaac_mk1_model_import_v2",',
            '"schema": "perceptron_isaac_mk1_model_import_v2",\n'
            '  "schema": "perceptron_isaac_mk1_model_import_v2",',
            1,
        )
    )
    config.mk1_model_import_sha256 = file_sha256(record_path)
    _refresh_trained_package_manifest(package, config)

    with pytest.raises(RuntimeError, match="duplicate key"):
        PerceptronIsaacPolicy._verify_packaged_contract_digests(config, package)


def test_trained_mk1_import_record_binds_index_to_shard_records(tmp_path, monkeypatch):
    from lerobot.policies.perceptron_isaac.modeling_perceptron_isaac import PerceptronIsaacPolicy

    package, config, _contract = _write_reduced_trained_mk1_package(tmp_path, monkeypatch)
    record_path = package / "hf_model" / "mk1_model_import.json"
    record = json.loads(record_path.read_text())
    record["shards"][0]["tensor_count"] += 1
    _write_json(record_path, record)
    config.mk1_model_import_sha256 = file_sha256(record_path)
    _refresh_trained_package_manifest(package, config)

    with pytest.raises(RuntimeError, match="shard record disagrees"):
        PerceptronIsaacPolicy._verify_packaged_contract_digests(config, package)


def test_trained_mk1_import_record_accepts_hf_snapshot_blob_symlinks(tmp_path, monkeypatch):
    from lerobot.policies.perceptron_isaac.modeling_perceptron_isaac import PerceptronIsaacPolicy

    package, config, _contract = _write_reduced_trained_mk1_package(tmp_path, monkeypatch)
    (package / ".gitattributes").write_text("*.safetensors filter=lfs\n")
    (package / "README.md").write_text("# MK1\n")
    repository = tmp_path / "cache" / "models--org--mk1"
    blobs = repository / "blobs"
    snapshot = repository / "snapshots" / "revision"
    blobs.mkdir(parents=True)
    snapshot.mkdir(parents=True)
    for source in sorted(package.rglob("*")):
        relative = source.relative_to(package)
        destination = snapshot / relative
        if source.is_dir():
            destination.mkdir(exist_ok=True)
            continue
        blob = blobs / f"{len(list(blobs.iterdir())):04d}-{source.name}"
        shutil.copy2(source, blob)
        destination.symlink_to(blob)

    config.hf_model_path = str(snapshot / "hf_model")
    PerceptronIsaacPolicy._verify_packaged_contract_digests(config, snapshot)


def test_mk1_import_rejects_coordinate_recipe_drift_before_copy(tmp_path):
    source = tmp_path / "mk1_hf"
    source.mkdir()
    _write_json(source / "config.json", _mk1_config(test_geometry=False))
    recipe = {
        "rendering": {
            "reserved_token_groups": [
                {"name": "coord", "offset": 248320, "size": 1001},
                {
                    "name": None,
                    "offset": 249321,
                    "size": 2048,
                    "tokenizer": "physical-intelligence/fast",
                },
            ]
        }
    }

    checkpoint_import._validate_mk1_recipe_coord_contract(source, recipe)
    recipe["rendering"]["reserved_token_groups"][0]["offset"] = 248321
    with pytest.raises(IsaacCheckpointImportError, match="coordinate/rendering contract mismatch"):
        checkpoint_import._validate_mk1_recipe_coord_contract(source, recipe)


def test_mk1_import_derives_checkpoint_maxima_and_artifact_identity() -> None:
    from lerobot.policies.perceptron_isaac.mk1_checkpoint_contract import Mk1CheckpointContract

    contract = Mk1CheckpointContract.parse_allowlisted(
        _mk1_config(test_geometry=True),
        allow_test_only_reduced_geometry=True,
    )
    artifact = {"schema_version": 1, "artifact_kind": "trained_policy", "trained_steps": 12}

    fields = checkpoint_import._derive_mk1_policy_fields(
        contract,
        artifact,
        served_action_dim=3,
        served_proprio_dim=2,
        served_action_horizon=2,
    )

    assert fields == {
        "artifact_kind": "trained_policy",
        "trained_steps": 12,
        "vector_max_states": 4,
        "max_state_dim": 4,
        "max_action_dim": 4,
        "max_action_horizon": 3,
    }


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"served_action_dim": 5}, "action width"),
        ({"served_proprio_dim": 5}, "proprio width"),
        ({"served_action_horizon": 4}, "action horizon"),
    ],
)
def test_mk1_import_rejects_served_geometry_above_checkpoint_maxima(overrides, message) -> None:
    from lerobot.policies.perceptron_isaac.mk1_checkpoint_contract import Mk1CheckpointContract

    contract = Mk1CheckpointContract.parse_allowlisted(
        _mk1_config(test_geometry=True),
        allow_test_only_reduced_geometry=True,
    )
    geometry = {
        "served_action_dim": 4,
        "served_proprio_dim": 4,
        "served_action_horizon": 3,
        **overrides,
    }
    with pytest.raises(IsaacCheckpointImportError, match=message):
        checkpoint_import._derive_mk1_policy_fields(
            contract,
            {"schema_version": 1, "artifact_kind": "trained_policy", "trained_steps": 12},
            **geometry,
        )


def test_production_mk1_import_requires_exact_trained_artifact_marker() -> None:
    config = _mk1_config(test_geometry=False)
    config["genesis_artifact"] = {
        "schema_version": 1,
        "artifact_kind": "trained_policy",
        "trained_steps": 100_000,
    }
    assert (
        checkpoint_import._validate_mk1_artifact_metadata(config, require_trained=True)
        == config["genesis_artifact"]
    )

    config["genesis_artifact"]["artifact_kind"] = "neutral_debug"
    config["genesis_artifact"]["trained_steps"] = 0
    with pytest.raises(IsaacCheckpointImportError, match="rejects neutral_debug"):
        checkpoint_import._validate_mk1_artifact_metadata(config, require_trained=True)

    config["genesis_artifact"] = {
        "schema_version": 1,
        "artifact_kind": "trained_policy",
        "trained_steps": 100_000,
        "unchecked": True,
    }
    with pytest.raises(IsaacCheckpointImportError, match="keys do not match"):
        checkpoint_import._validate_mk1_artifact_metadata(config, require_trained=True)


def test_hf_import_rejects_unknown_checkpoint_norm_convention(tmp_path):
    source = tmp_path / "unknown_hf"
    source.mkdir()
    _write_json(
        source / "config.json",
        {
            "model_type": "qwen3_5_moe",
            "genesis_vla": {
                "backbone_family": "mk1_qwen3_6_moe",
                "rmsnorm_weight_convention": "guess_from_tensors",
            },
        },
    )

    with pytest.raises(IsaacCheckpointImportError, match="Unsupported genesis_vla"):
        convert_hf_weights_once(source, tmp_path / "output")


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        ({"action_representation": "relative"}, "action_representation"),
        ({"relative_exclude_joints": ["left_gripper.pos"]}, "relative_exclude_joints"),
        ({"action_normalization_eps": 1e-8}, "action_normalization_eps"),
        ({"proprio_normalization_eps": 1e-8}, "proprio_normalization_eps"),
    ],
)
def test_import_rejects_normalization_the_runtime_cannot_reproduce(tmp_path, monkeypatch, mutation, match):
    """The native stats runtime only serves absolute actions at a fixed epsilon.

    These contracts validate cleanly, so without an explicit gate the importer would emit a
    package that unnormalizes deltas and commands them as absolute joint targets.
    """
    hf_export, _dcp_checkpoint, adapter_path = _write_authenticated_fixture(tmp_path)
    manifest, normalization, recipe = _make_contracts()
    entry = dict(normalization["entries"][0])
    entry.update(mutation)
    entry.pop("stats_sha256", None)
    entry["stats_sha256"] = canonical_sha256(entry)
    normalization = {
        **normalization,
        "entries": [entry],
        "entries_sha256": canonical_sha256([entry]),
    }
    # Keep the recipe's action wire consistent with the entry so the internal cross-check
    # passes and the runtime-capability gate is what rejects the checkpoint.
    wire = dict(recipe["recipes"][0]["lowering"]["action_wire"])
    if "action_representation" in mutation:
        wire["action_representation"] = mutation["action_representation"]
    if "relative_exclude_joints" in mutation:
        wire["relative_exclude_joints"] = mutation["relative_exclude_joints"]
    if "action_normalization_eps" in mutation:
        wire["eps"] = mutation["action_normalization_eps"]
    recipe_record = {
        **recipe["recipes"][0],
        "lowering": {**recipe["recipes"][0]["lowering"], "action_wire": wire},
    }
    recipe = {
        **recipe,
        "recipes": [recipe_record],
        "recipes_sha256": canonical_sha256([recipe_record]),
    }
    for filename, value in (
        (POLICY_NORMALIZATION_FILENAME, normalization),
        (POLICY_INFERENCE_RECIPE_FILENAME, recipe),
    ):
        _write_json(hf_export / filename, value)
    _write_hf_identity(hf_export, build_dcp_identity(manifest, normalization, recipe))

    fast_artifact = _write_fast_processor_fixture(tmp_path)
    _allow_synthetic_fast_artifact(monkeypatch, fast_artifact)
    with pytest.raises(IsaacCheckpointImportError, match=match):
        import_authenticated_isaac_checkpoint(
            hf_export,
            None,
            tmp_path / "isaac_lerobot_rejected",
            policy_state_dataset="cloud/isaac_yam",
            normalization_scope="yam",
            deployment_adapter_path=adapter_path,
            fast_processor_source=fast_artifact,
            allow_fast_remote_code=True,
        )


def _make_mixed_capability_contracts() -> tuple[dict, dict, dict]:
    """Authorize one deployable identity plus an identity trained without proprio normalization.

    This is the shape of the real Genesis export: ``policy_normalization.json`` publishes one
    entry per (dataset, scope, objective) that was ever trained -- the step-100000 bundle carries
    3004 over 725 datasets -- and a handful of them declare ``proprio_normalized=false``. Those
    entries are authenticated, sorted and digest-bound like every other entry; they are simply not
    servable by the native runtime, and are never the identity being deployed.
    """
    manifest, normalization, recipe = _make_contracts()
    compliant_entry = normalization["entries"][0]
    compliant_recipe = recipe["recipes"][0]
    compliant_identity = {
        key: compliant_entry[key] for key in ("policy_state_dataset", "normalization_scope", "objective")
    }
    unsupported_identity = {
        "policy_state_dataset": "cloud/isaac_partial",
        "normalization_scope": "partial",
        "objective": "Flow",
    }
    # Genesis nulls a channel's epsilon and quantile block when it did not normalize it.
    unsupported_entry = {
        **compliant_entry,
        **unsupported_identity,
        "proprio_normalized": False,
        "proprio_normalization_eps": None,
        "proprio": None,
    }
    unsupported_entry["stats_sha256"] = canonical_sha256(
        {key: value for key, value in unsupported_entry.items() if key != "stats_sha256"}
    )
    unsupported_recipe = {**compliant_recipe, **unsupported_identity}
    manifest["datasets"][unsupported_identity["policy_state_dataset"]] = {
        **manifest["datasets"][compliant_identity["policy_state_dataset"]],
        "normalization_identities": [
            {
                "normalization_scope": unsupported_identity["normalization_scope"],
                "objective": unsupported_identity["objective"],
            }
        ],
    }
    manifest_digest = canonical_sha256(manifest)
    # Entries and recipes are both ordered by the identity tuple, and "cloud/isaac_partial"
    # sorts before "cloud/isaac_yam", so the unsupported entry is validated first -- exactly
    # like index 1158 of the real bundle.
    entries = [unsupported_entry, compliant_entry]
    recipes = [unsupported_recipe, compliant_recipe]
    normalization.update(
        {
            "policy_state_manifest_sha256": manifest_digest,
            "entries": entries,
            "entries_sha256": canonical_sha256(entries),
            "coverage": {
                "complete": True,
                "observed": [unsupported_identity, compliant_identity],
                "missing": [],
            },
        }
    )
    recipe.update(
        {
            "policy_state_manifest_sha256": manifest_digest,
            "recipes": recipes,
            "recipes_sha256": canonical_sha256(recipes),
        }
    )
    return manifest, normalization, recipe


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        ({"proprio_normalized": False}, "must be null when proprio_normalized is false"),
        ({"action_normalized": False}, "must be null when action_normalized is false"),
        ({"proprio_normalization_eps": None}, "proprio_normalization_eps must be a finite number"),
        ({"action": None}, "action must be an object"),
    ],
    ids=[
        "unnormalized_proprio_still_ships_stats",
        "unnormalized_action_still_ships_stats",
        "normalized_proprio_without_eps",
        "normalized_action_without_stats",
    ],
)
def test_authentication_requires_normalization_fields_to_match_their_flags(tmp_path, mutation, match):
    """Genesis publishes a channel's epsilon and quantile block exactly when it normalized it.

    Both directions must fail closed: a normalized channel without its epsilon or stats is
    unservable, and an unnormalized channel that still ships them is a malformed artifact.
    """
    manifest, normalization, recipe = _make_contracts()
    entry = {**normalization["entries"][0], **mutation}
    entry["stats_sha256"] = canonical_sha256(
        {key: value for key, value in entry.items() if key != "stats_sha256"}
    )
    normalization = {
        **normalization,
        "entries": [entry],
        "entries_sha256": canonical_sha256([entry]),
    }
    hf_export, dcp_checkpoint, _adapter_path = _write_authenticated_fixture(
        tmp_path, contracts=(manifest, normalization, recipe)
    )

    with pytest.raises(IsaacCheckpointImportError, match=match):
        authenticate_isaac_checkpoint(hf_export, dcp_checkpoint)


def test_authentication_accepts_bundle_whose_unselected_entry_is_unservable(tmp_path):
    """An unrelated unservable entry must not veto authenticating a compliant identity."""
    hf_export, dcp_checkpoint, adapter_path = _write_authenticated_fixture(
        tmp_path, contracts=_make_mixed_capability_contracts()
    )

    contracts = authenticate_isaac_checkpoint(hf_export, dcp_checkpoint)

    assert [entry["policy_state_dataset"] for entry in contracts.normalization["entries"]] == [
        "cloud/isaac_partial",
        "cloud/isaac_yam",
    ]
    adapter = validate_isaac_deployment_adapter(
        adapter_path,
        contracts=contracts,
        policy_state_dataset="cloud/isaac_yam",
        normalization_scope="yam",
        objective="Flow",
    )

    assert adapter.normalization_entry["normalization_scope"] == "yam"
    assert adapter.normalization_entry["action_normalized"] is True
    assert adapter.normalization_entry["proprio_normalized"] is True


def test_deployment_adapter_rejects_selected_entry_without_proprio_normalization(tmp_path):
    """The capability rule still fails closed for the identity actually being deployed."""
    hf_export, dcp_checkpoint, _compliant_adapter = _write_authenticated_fixture(
        tmp_path, contracts=_make_mixed_capability_contracts()
    )
    adapter_path = _write_deployment_adapter(
        tmp_path / f"partial_{ISAAC_DEPLOYMENT_ADAPTER_FILENAME}",
        hf_export,
        policy_state_dataset="cloud/isaac_partial",
        normalization_scope="partial",
        normalization_profile_id="synthetic/partial",
    )
    contracts = authenticate_isaac_checkpoint(hf_export, dcp_checkpoint)

    with pytest.raises(IsaacCheckpointImportError, match="proprio_normalized"):
        validate_isaac_deployment_adapter(
            adapter_path,
            contracts=contracts,
            policy_state_dataset="cloud/isaac_partial",
            normalization_scope="partial",
            objective="Flow",
        )


def _camera_profile(observation_keys: list[str], prompt_views: list[str]) -> dict:
    """A deployment profile that differs from the fixture only in its camera arity."""
    return {
        "robot_type": "franka",
        "camera": {"observation_keys": observation_keys, "prompt_views": prompt_views},
        "state_action_schema": {
            "name": "gripper_7",
            "control_mode": "ee",
            "components": ["world_vector", "rotation_delta", "gripper_closedness_action"],
        },
        "normalization": None,
    }


def _validate_camera_profile(observation_keys: list[str], prompt_views: list[str]) -> None:
    profile = _camera_profile(observation_keys, prompt_views)
    checkpoint_import._validate_deployment_profile(
        profile,
        manifest_record={"deployment_profile_hash": canonical_sha256(profile)},
        location="inference_recipe.recipes[1148].deployment_profile",
    )


def test_deployment_profile_accepts_cameras_sharing_one_deduplicated_prompt_view():
    # The real berkeley_cable_routing record: wrist225_image and wrist45_image share the
    # mount label "wrist", which genesis deduplicates into a single prompt view.
    _validate_camera_profile(
        ["image", "top_image", "wrist225_image", "wrist45_image"],
        ["primary", "external_high", "wrist"],
    )


def test_deployment_profile_rejects_more_prompt_views_than_cameras():
    with pytest.raises(IsaacCheckpointImportError, match="more prompt views than camera"):
        _validate_camera_profile(
            ["image", "top_image", "wrist225_image", "wrist45_image"],
            ["primary", "external_high", "wrist", "left_wrist", "right_wrist"],
        )


def test_deployment_profile_rejects_duplicate_prompt_views():
    with pytest.raises(IsaacCheckpointImportError, match="prompt_views must not contain duplicates"):
        _validate_camera_profile(
            ["image", "top_image", "wrist225_image", "wrist45_image"],
            ["primary", "wrist", "wrist"],
        )


def _validate_normalization_profile(normalization: dict | None) -> None:
    profile = _camera_profile(["image", "wrist_image"], ["primary", "wrist"])
    profile["normalization"] = normalization
    checkpoint_import._validate_deployment_profile(
        profile,
        manifest_record={"deployment_profile_hash": canonical_sha256(profile)},
        location="inference_recipe.recipes[2754].deployment_profile",
    )


def test_deployment_profile_accepts_boolean_normalization_switches():
    # The real nitrogen_shuffled record: genesis emits the switches as booleans.
    _validate_normalization_profile({"action": False, "proprio": False})


def test_deployment_profile_accepts_null_normalization_switches():
    _validate_normalization_profile({"action": None, "proprio": None})


def test_deployment_profile_rejects_string_normalization_switches():
    with pytest.raises(IsaacCheckpointImportError, match="normalization.action must be a boolean"):
        _validate_normalization_profile({"action": "q01_q99", "proprio": False})


def test_authenticated_import_records_a_declared_conditioning_free_deployment(tmp_path, monkeypatch):
    manifest, normalization, recipe = _make_contracts()
    hf_export, dcp_checkpoint, adapter_path = _write_hf_authenticated_fixture(
        tmp_path, contracts=(manifest, normalization, recipe)
    )
    output = tmp_path / "isaac_lerobot_conditioning_free"

    fast_artifact = _write_fast_processor_fixture(tmp_path)
    _allow_synthetic_fast_artifact(monkeypatch, fast_artifact)
    import_authenticated_isaac_checkpoint(
        hf_export,
        dcp_checkpoint,
        output,
        policy_state_dataset="cloud/isaac_yam",
        normalization_scope="yam",
        deployment_adapter_path=adapter_path,
        fast_processor_source=fast_artifact,
        allow_fast_remote_code=True,
        conditioning_deployment=CONDITIONING_DISABLED_DEPLOYMENT,
    )

    provenance = json.loads((output / "isaac_import_provenance.json").read_text())
    assert provenance["schema"] == QWEN35_IMPORT_PROVENANCE_SCHEMA
    assert provenance["conditioning_deployment"] == {
        "renders_action_conditioning": False,
        "renders_mistake_conditioning": False,
    }


def test_authenticated_import_builds_movable_lerobot_package(tmp_path, monkeypatch):
    manifest, normalization, recipe = _make_contracts()
    hf_export, dcp_checkpoint, adapter_path = _write_hf_authenticated_fixture(
        tmp_path, contracts=(manifest, normalization, recipe)
    )
    identity_path = hf_export / POLICY_STATE_IDENTITY_FILENAME
    output = tmp_path / "isaac_lerobot"

    fast_artifact = _write_fast_processor_fixture(tmp_path)
    _allow_synthetic_fast_artifact(monkeypatch, fast_artifact)
    result = import_authenticated_isaac_checkpoint(
        hf_export,
        dcp_checkpoint,
        output,
        policy_state_dataset="cloud/isaac_yam",
        normalization_scope="yam",
        deployment_adapter_path=adapter_path,
        fast_processor_source=fast_artifact,
        allow_fast_remote_code=True,
    )

    assert result.output_path == output
    config = PreTrainedConfig.from_pretrained(output)
    assert config.type == "perceptron_isaac"
    assert config.n_obs_steps == 1
    assert config.action_dim == 14
    assert config.proprio_dim == 14
    assert config.image_size == (360, 640)
    assert config.camera_order == ("top", "left", "right")
    assert config.dataset_name == "molmoact2_bimanualyam"
    assert config.policy_state_dataset == "cloud/isaac_yam"
    assert config.robot_type == "bi_yam"
    assert config.control_mode == "joint"
    assert config.num_flow_samples == 1
    assert config.clip_action_pose is False
    assert config.num_settle_steps == 0
    assert config.normalize_task_text is False
    assert config.allow_image_key_fallback is False
    assert config.action_feature_names == [
        *(f"left_joint_{index}.pos" for index in range(6)),
        "left_gripper.pos",
        *(f"right_joint_{index}.pos" for index in range(6)),
        "right_gripper.pos",
    ]
    assert config.state_feature_names == config.action_feature_names
    assert config.strict_hardware_feature_contract is True
    assert config.target_fps == 30.0
    assert config.apply_offset_norm is False
    assert len(config.deployment_adapter_sha256) == 64
    assert (output / ISAAC_DEPLOYMENT_ADAPTER_FILENAME).is_file()
    assert config.fast_processor_path == "fast_processor"
    assert len(config.fast_processor_tree_sha256) == 64
    assert (output / "hf_model/rmsnorm_conversion.json").is_file()
    assert json.loads((output / "hf_model/vocab.json").read_text()) == {"!": 0, '"': 1}
    assert fast_processor_tree_identity(output / "fast_processor")[0] == config.fast_processor_tree_sha256
    provenance = json.loads((output / "isaac_import_provenance.json").read_text())
    assert provenance["schema"] == QWEN35_IMPORT_PROVENANCE_SCHEMA
    # No declaration was required for this recipe, which the record states as null rather
    # than as a conditioning-free declaration the caller never made.
    assert provenance["conditioning_deployment"] is None
    assert provenance["contract_authentication"] == "hf_and_dcp_identity_authenticated"
    assert provenance["policy_state_identity_source"] == "hf_and_dcp_identity_authenticated"
    assert len(provenance["source_artifact_manifest_sha256"]) == 64
    assert len(provenance["authenticated_import_identity_sha256"]) == 64
    assert provenance["source_hashes"]["model.safetensors"] == file_sha256(hf_export / "model.safetensors")
    assert file_sha256(output / POLICY_STATE_IDENTITY_FILENAME) == file_sha256(identity_path)
    assert provenance["fast_processor"]["tree_sha256"] == config.fast_processor_tree_sha256
    assert provenance["deployment_adapter"]["sha256"] == config.deployment_adapter_sha256
    stats = json.loads((output / "isaac_stats.json").read_text())
    assert stats["schema"] == "flow_matching_stats_v1"
    assert stats["state_action_schema"] == "joint_gripper_14"

    preprocessor_config = json.loads((output / "policy_preprocessor.json").read_text())
    pack_config = next(
        step
        for step in preprocessor_config["steps"]
        if step["registry_name"] == "perceptron_isaac_mharmony_pack"
    )
    assert pack_config["config"]["native_stats_path"] is None
    assert pack_config["config"]["native_render_metadata_path"] is None
    assert pack_config["config"]["fast_processor_path"] == "fast_processor"
    assert pack_config["config"]["fast_processor_tree_sha256"] == config.fast_processor_tree_sha256
    assert pack_config["config"]["render_metadata"]["robot_type"] == "bi_yam"
    assert pack_config["config"]["render_metadata"]["image_size"] == [360, 640]
    prompt_contract = pack_config["config"]["render_metadata"]["training_proprio_contract"]
    assert prompt_contract["dataset_name"] == "molmoact2_bimanualyam"
    assert prompt_contract["contract_hash"] == normalization["entries"][0]["policy_state_contract_hash"]
    assert [component["name"] for component in prompt_contract["components"]] == [
        "left_arm_joints",
        "left_gripper",
        "right_arm_joints",
        "right_gripper",
    ]

    moved = tmp_path / "relocated" / "isaac_lerobot"
    moved.parent.mkdir()
    shutil.copytree(output, moved)
    moved_preprocessor = json.loads((moved / "policy_preprocessor.json").read_text())
    moved_pack_config = next(
        step
        for step in moved_preprocessor["steps"]
        if step["registry_name"] == "perceptron_isaac_mharmony_pack"
    )
    moved_pack_config["config"]["render_metadata"].pop("training_proprio_contract")
    _write_json(moved / "policy_preprocessor.json", moved_preprocessor)
    moved_config = PreTrainedConfig.from_pretrained(moved)
    preprocessor, postprocessor = make_pre_post_processors(
        moved_config,
        pretrained_path=str(moved),
    )
    pack_step = next(
        step for step in preprocessor.steps if isinstance(step, PerceptronIsaacMharmonyPackProcessorStep)
    )
    action_step = next(
        step
        for step in postprocessor.steps
        if isinstance(step, PerceptronIsaacActionUnnormalizeProcessorStep)
    )
    assert pack_step._stats.action_dim == 14
    assert pack_step.fast_processor_path == str(moved / "fast_processor")
    assert fast_processor_tree_identity(pack_step.fast_processor_path)[0] == config.fast_processor_tree_sha256
    assert pack_step.render_metadata["n_obs_steps"] == 1
    assert (
        pack_step.render_metadata["training_proprio_contract"]["contract_hash"]
        == (normalization["entries"][0]["policy_state_contract_hash"])
    )
    assert action_step._stats.proprio_dim == 14


def test_packaged_deployment_adapter_digest_is_enforced_at_load(tmp_path, monkeypatch):
    """The recorded digest was previously only format-checked, never compared."""
    from lerobot.policies.perceptron_isaac.modeling_perceptron_isaac import PerceptronIsaacPolicy

    hf_export, dcp_checkpoint, adapter_path = _write_hf_authenticated_fixture(tmp_path)
    output = tmp_path / "isaac_lerobot_digest"
    fast_artifact = _write_fast_processor_fixture(tmp_path)
    _allow_synthetic_fast_artifact(monkeypatch, fast_artifact)
    import_authenticated_isaac_checkpoint(
        hf_export,
        dcp_checkpoint,
        output,
        policy_state_dataset="cloud/isaac_yam",
        normalization_scope="yam",
        deployment_adapter_path=adapter_path,
        fast_processor_source=fast_artifact,
        allow_fast_remote_code=True,
    )
    config = PreTrainedConfig.from_pretrained(output)
    packaged_adapter = output / ISAAC_DEPLOYMENT_ADAPTER_FILENAME
    assert packaged_adapter.is_file()
    original_adapter = packaged_adapter.read_bytes()

    # Unmodified package verifies cleanly.
    PerceptronIsaacPolicy._verify_packaged_contract_digests(config, output)

    downgraded = tmp_path / "downgraded-import"
    shutil.copytree(output, downgraded)
    downgraded_config = json.loads((downgraded / "config.json").read_text())
    downgraded_config["qwen35_trained_package_manifest_sha256"] = None
    _write_json(downgraded / "config.json", downgraded_config)
    (downgraded / "qwen35_trained_package_manifest.json").unlink()
    (downgraded / "policy_preprocessor.json").write_text('{"tampered": true}\n')
    stripped_config = PreTrainedConfig.from_pretrained(downgraded)
    PerceptronIsaacPolicy._resolve_checkpoint_local_paths(stripped_config, downgraded)
    with pytest.raises(RuntimeError, match="require the exhaustive outer"):
        PerceptronIsaacPolicy._verify_packaged_contract_digests(stripped_config, downgraded)

    # Editing the reviewed adapter after import must be detected.
    payload = json.loads(packaged_adapter.read_text())
    payload["render_dataset_name"] = "tampered"
    _write_json(packaged_adapter, payload)
    with pytest.raises(RuntimeError, match="digest mismatch for isaac_deployment_adapter.json"):
        PerceptronIsaacPolicy._verify_packaged_contract_digests(config, output)

    # Restore the reviewed adapter, then prove that a structurally unrelated
    # converted-weight mutation is independently covered by import provenance.
    packaged_adapter.write_bytes(original_adapter)
    provenance = json.loads((output / "isaac_import_provenance.json").read_text())
    shard_name = provenance["rmsnorm_conversion"]["shards"][0]["filename"]
    shard_path = output / "hf_model" / shard_name
    shard_path.write_bytes(shard_path.read_bytes() + b"tampered")
    with pytest.raises(RuntimeError, match="model shard digest mismatch"):
        PerceptronIsaacPolicy._verify_packaged_contract_digests(config, output)


def test_legacy_bf16_export_requires_exhaustive_outer_manifest(tmp_path):
    from lerobot.policies.perceptron_isaac.modeling_perceptron_isaac import PerceptronIsaacPolicy

    output = tmp_path / "bf16_export"
    hf_dir = output / "hf_model"
    hf_dir.mkdir(parents=True)
    shard = hf_dir / "model-00001-of-00001.safetensors"
    shard.write_bytes(b"portable-bf16-weights")
    _write_json(
        output / "isaac_export_provenance.json",
        {
            "schema": "perceptron_isaac_bf16_deployment_export_v1",
            "model_shards": [{"filename": shard.name, "sha256": file_sha256(shard)}],
        },
    )
    config = PerceptronIsaacConfig()

    with pytest.raises(RuntimeError, match="require the exhaustive outer"):
        PerceptronIsaacPolicy._verify_packaged_contract_digests(config, output)


def test_bf16_export_dtype_verifier_rejects_fp32_tensors(tmp_path):
    from safetensors.torch import save_file

    from lerobot.scripts.lerobot_isaac_export_bf16 import _verify_bf16_model_shards

    bf16_shard = tmp_path / "bf16.safetensors"
    save_file({"weight": torch.ones(2, dtype=torch.bfloat16)}, bf16_shard)
    _verify_bf16_model_shards([bf16_shard])

    fp32_shard = tmp_path / "fp32.safetensors"
    save_file({"weight": torch.ones(2, dtype=torch.float32)}, fp32_shard)
    with pytest.raises(RuntimeError, match="has dtype F32, expected BF16"):
        _verify_bf16_model_shards([fp32_shard])


def test_loader_refuses_to_apply_offset_norm_twice(tmp_path):
    """rmsnorm_conversion.json marks an already-corrected directory; the -1 is not idempotent."""
    from lerobot.policies.perceptron_isaac.modeling_qwen35_vla import (
        _assert_offset_norm_matches_marker,
    )

    converted = tmp_path / "converted"
    converted.mkdir()
    # A raw (unconverted) export has no marker, so the correction is expected.
    _assert_offset_norm_matches_marker(converted, apply_offset_norm=True)

    _write_json(converted / "rmsnorm_conversion.json", {"schema": "perceptron_isaac_rmsnorm_conversion_v2"})
    _assert_offset_norm_matches_marker(converted, apply_offset_norm=False)
    with pytest.raises(RuntimeError, match="already in the native unit-offset convention"):
        _assert_offset_norm_matches_marker(converted, apply_offset_norm=True)


def test_authenticated_import_uses_stable_snapshot_when_source_changes(tmp_path, monkeypatch):
    from safetensors.torch import save_file

    hf_export, dcp_checkpoint, adapter_path = _write_hf_authenticated_fixture(tmp_path)
    fast_artifact = _write_fast_processor_fixture(tmp_path)
    _allow_synthetic_fast_artifact(monkeypatch, fast_artifact)
    original_convert = checkpoint_import.convert_hf_weights_once
    original_source_digest = file_sha256(hf_export / "model.safetensors")

    def replace_source_after_conversion(source_path, output_path, **kwargs):
        assert Path(source_path) != hf_export
        result = original_convert(source_path, output_path, **kwargs)
        save_file(
            {"model.language_model.norm.weight": torch.full((2,), 9.5)},
            hf_export / "model.safetensors",
        )
        return result

    monkeypatch.setattr(checkpoint_import, "convert_hf_weights_once", replace_source_after_conversion)
    output = tmp_path / "replaced_source_package"

    import_authenticated_isaac_checkpoint(
        hf_export,
        dcp_checkpoint,
        output,
        policy_state_dataset="cloud/isaac_yam",
        normalization_scope="yam",
        deployment_adapter_path=adapter_path,
        fast_processor_source=fast_artifact,
        allow_fast_remote_code=True,
    )

    provenance = json.loads((output / "isaac_import_provenance.json").read_text())
    assert provenance["source_hashes"]["model.safetensors"] == original_source_digest
    assert file_sha256(hf_export / "model.safetensors") != original_source_digest


@pytest.mark.parametrize("platform", ["so100_so101", "libero"])
def test_authenticated_import_builds_strict_cross_platform_package(tmp_path, monkeypatch, platform):
    hf_export, dcp_checkpoint, adapter_path = _write_platform_authenticated_fixture(tmp_path, platform)
    fast_artifact = _write_fast_processor_fixture(tmp_path)
    _allow_synthetic_fast_artifact(monkeypatch, fast_artifact)
    _contracts, spec = _make_platform_contracts(platform)
    output = tmp_path / f"{platform}_lerobot"

    import_authenticated_isaac_checkpoint(
        hf_export,
        dcp_checkpoint,
        output,
        policy_state_dataset=spec["dataset"],
        normalization_scope=spec["scope"],
        deployment_adapter_path=adapter_path,
        fast_processor_source=fast_artifact,
        allow_fast_remote_code=True,
    )

    config = PreTrainedConfig.from_pretrained(output)
    assert config.robot_type == platform
    assert config.control_mode == spec["control_mode"]
    assert config.dataset_name == spec["render_dataset_name"]
    assert config.action_dim == spec["action_dim"]
    assert config.proprio_dim == spec["proprio_dim"]
    assert config.image_size == tuple(spec["image_size"])
    assert config.camera_order == tuple(spec["camera_order"])
    assert config.target_fps == spec["target_fps"]
    render_metadata = json.loads((output / "native_render_metadata.json").read_text())
    assert render_metadata["action_layout"] == spec["components"]
    prompt_contract = render_metadata["training_proprio_contract"]
    assert prompt_contract["dataset_name"] == spec["render_dataset_name"]
    assert prompt_contract["target_dim"] == spec["proprio_dim"]
    expected_prompt_components = {
        "so100_so101": ["arm_joints", "gripper"],
        "libero": ["end_effector_position", "end_effector_orientation", "gripper"],
    }
    assert [component["name"] for component in prompt_contract["components"]] == (
        expected_prompt_components[platform]
    )
    if platform == "so100_so101":
        assert config.strict_hardware_feature_contract is True
        assert config.strict_environment_feature_contract is False
        hardware_names = [
            "shoulder_pan.pos",
            "shoulder_lift.pos",
            "elbow_flex.pos",
            "wrist_flex.pos",
            "wrist_roll.pos",
            "gripper.pos",
        ]
        assert config.action_feature_names == hardware_names
        assert config.state_feature_names == hardware_names
        assert config.joint_signs == spec["joint_signs"]
        assert config.joint_offsets == spec["joint_offsets"]
        from lerobot.rollout.context import _validate_strict_hardware_feature_contract

        observation_features = {
            **dict.fromkeys(hardware_names, float),
            "top": (256, 256, 3),
            "side": (256, 256, 3),
        }
        _validate_strict_hardware_feature_contract(
            config,
            observation_features,
            dict.fromkeys(hardware_names, float),
            {},
        )
    else:
        assert config.strict_hardware_feature_contract is False
        assert config.strict_environment_feature_contract is True
        assert config.action_feature_names is None
        assert config.state_feature_names is None
        assert render_metadata["action_layout"] == [
            "world_vector",
            "rotation_delta",
            "gripper_closedness_action",
        ]


def test_qwen35_vocab_rejects_drift_from_tokenizer(tmp_path, monkeypatch):
    tokenizer = tmp_path / "tokenizer.json"
    source_vocab = tmp_path / "source_vocab.json"
    _write_qwen35_tokenizer_fixture(tokenizer)
    _write_json(source_vocab, {"!": 1, '"': 0})
    monkeypatch.setattr(checkpoint_import, "QWEN35_BASE_VOCAB_SIZE", 2)

    with pytest.raises(IsaacCheckpointImportError, match="differs from tokenizer.json"):
        materialize_qwen35_mharmony_vocab(
            tokenizer,
            tmp_path / "vocab.json",
            source_vocab_path=source_vocab,
        )


# --- Genesis recipe schema v7 (conditioning-era contract) ---


def test_authenticate_accepts_schema_v7_recipe_with_inert_conditioning(tmp_path):
    manifest, normalization, recipe = _make_contracts()
    recipe = _upgrade_recipe_to_schema7(recipe)
    hf_export, _dcp_checkpoint, _adapter_path = _write_hf_authenticated_fixture(
        tmp_path, contracts=(manifest, normalization, recipe)
    )

    contracts = authenticate_isaac_checkpoint(hf_export)

    assert contracts.contract_authentication == "hf_identity_authenticated"
    assert contracts.inference_recipe["schema_version"] == 7


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"action_conditioning_probability": 1.0}, "does not implement"),
        ({"mistake_conditioning_probability": 0.25}, "does not implement"),
        (
            {"action_conditioning_probability": 1.0, "action_conditioning_role": "agent"},
            "does not implement",
        ),
        ({"action_conditioning_role": "agent"}, "must be 'user'"),
        ({"mistake_conditioning_override": False}, "must be.*null"),
    ],
)
def test_authenticate_rejects_v7_recipes_requiring_conditioning(tmp_path, overrides, match):
    manifest, normalization, recipe = _make_contracts()
    recipe = _upgrade_recipe_to_schema7(recipe, **overrides)
    hf_export, _dcp_checkpoint, _adapter_path = _write_hf_authenticated_fixture(
        tmp_path, contracts=(manifest, normalization, recipe)
    )

    with pytest.raises(IsaacCheckpointImportError, match=match):
        authenticate_isaac_checkpoint(hf_export)


def test_authenticate_rejects_v7_flow_wire_conditioning_keys(tmp_path):
    """A conditioning-enabled Flow wire must die on a targeted error, not a generic one."""
    manifest, normalization, recipe = _make_contracts()
    recipe = _upgrade_recipe_to_schema7(recipe)
    recipe["recipes"][0]["lowering"]["action_wire"].update(
        {
            "conditioning_tokenizer_name_or_path": "physical-intelligence/fast",
            "conditioning_tokenizer_revision": "abc",
            "conditioning_processor_artifact": {"source": "hub"},
            "conditioning_token_group_key": "fast",
            "conditioning_token_group_name": "fast",
        }
    )
    recipe["recipes_sha256"] = canonical_sha256(recipe["recipes"])
    hf_export, _dcp_checkpoint, _adapter_path = _write_hf_authenticated_fixture(
        tmp_path, contracts=(manifest, normalization, recipe)
    )

    with pytest.raises(IsaacCheckpointImportError, match="conditioning keys"):
        authenticate_isaac_checkpoint(hf_export)


def test_authenticate_rejects_unknown_recipe_schema_version(tmp_path):
    manifest, normalization, recipe = _make_contracts()
    recipe = dict(recipe)
    recipe["schema_version"] = 8
    hf_export, _dcp_checkpoint, _adapter_path = _write_hf_authenticated_fixture(
        tmp_path, contracts=(manifest, normalization, recipe)
    )

    with pytest.raises(
        IsaacCheckpointImportError, match="Unsupported policy inference recipe schema version"
    ):
        authenticate_isaac_checkpoint(hf_export)


def test_authenticate_rejects_conditioning_keys_on_a_v2_recipe(tmp_path):
    """v2 exports never carried the conditioning fields; exact-match still holds."""
    manifest, normalization, recipe = _make_contracts()
    recipe = json.loads(json.dumps(recipe))
    recipe["recipes"][0]["lowering"]["observation"]["action_conditioning_probability"] = 0.0
    recipe["recipes_sha256"] = canonical_sha256(recipe["recipes"])
    hf_export, _dcp_checkpoint, _adapter_path = _write_hf_authenticated_fixture(
        tmp_path, contracts=(manifest, normalization, recipe)
    )

    with pytest.raises(IsaacCheckpointImportError, match="unexpected="):
        authenticate_isaac_checkpoint(hf_export)


# --- Conditioning-trained recipes served conditioning-free (genesis schema v7) ---
#
# isaac_0_5-step-100000 was trained with action_conditioning_probability=0.5 and
# mistake_conditioning_probability=0.8075 on every one of its 3028 recipes. Genesis samples
# both gates per training sample (genesis/core/datasets/augment/trajectory.py:5714-5718 and
# :3950-3971) and renders nothing extra when a gate misses (:4080-4092, and
# genesis/core/datasets/augment/robotics/lowering.py:494-495 omits the "mistake:" line when
# the value is None), so for any probability strictly below 1 the unconditioned prompt is a
# trained mode that LeRobot reproduces byte-for-byte. Genesis nonetheless forces both ON at
# its own serving entry point (genesis/core/robotics/inference_recipe.py:335-348,
# genesis/inference/flow_matching/observation.py:445-456), which is a Genesis serving
# preference rather than a property of the checkpoint.


def _conditioning_trained_recipe(recipe: dict, **overrides) -> dict:
    """A v7 recipe with the Bernoulli conditioning support isaac_0_5-step-100000 ships."""
    return _upgrade_recipe_to_schema7(
        recipe,
        **{
            "action_conditioning_probability": 0.5,
            "mistake_conditioning_probability": 0.8075,
            **overrides,
        },
    )


def _attach_flow_conditioning_wire_keys(recipe: dict) -> dict:
    """Attach the five conditioning_* keys genesis puts on every conditioning-trained Flow wire."""
    for record in recipe["recipes"]:
        record["lowering"]["action_wire"].update(
            {
                "conditioning_tokenizer_name_or_path": "physical-intelligence/fast",
                "conditioning_tokenizer_revision": "ec4d7aa71691cac0b8bed6942be45684db2110f4",
                "conditioning_processor_artifact": {"source": "huggingface_hub"},
                "conditioning_token_group_key": "token_group",
                "conditioning_token_group_name": "fast_action",
            }
        )
    recipe["recipes_sha256"] = canonical_sha256(recipe["recipes"])
    return recipe


def test_authenticate_accepts_conditioning_trained_recipe_for_a_conditioning_free_deployment(tmp_path):
    manifest, normalization, recipe = _make_contracts()
    recipe = _conditioning_trained_recipe(recipe)
    hf_export, _dcp_checkpoint, _adapter_path = _write_hf_authenticated_fixture(
        tmp_path, contracts=(manifest, normalization, recipe)
    )

    contracts = authenticate_isaac_checkpoint(
        hf_export, conditioning_deployment=CONDITIONING_DISABLED_DEPLOYMENT
    )

    observation = contracts.inference_recipe["recipes"][0]["lowering"]["observation"]
    assert observation["action_conditioning_probability"] == 0.5
    assert observation["mistake_conditioning_probability"] == 0.8075
    assert contracts.conditioning_deployment == CONDITIONING_DISABLED_DEPLOYMENT


def test_authenticate_accepts_flow_conditioning_wire_keys_for_a_conditioning_free_deployment(tmp_path):
    manifest, normalization, recipe = _make_contracts()
    recipe = _attach_flow_conditioning_wire_keys(_conditioning_trained_recipe(recipe))
    hf_export, _dcp_checkpoint, _adapter_path = _write_hf_authenticated_fixture(
        tmp_path, contracts=(manifest, normalization, recipe)
    )

    contracts = authenticate_isaac_checkpoint(
        hf_export, conditioning_deployment=CONDITIONING_DISABLED_DEPLOYMENT
    )

    wire = contracts.inference_recipe["recipes"][0]["lowering"]["action_wire"]
    assert wire["conditioning_token_group_name"] == "fast_action"


def test_authenticate_still_refuses_conditioning_trained_recipe_without_a_declaration(tmp_path):
    """No declaration keeps the historical refusal verbatim; the flag is not a default."""
    manifest, normalization, recipe = _make_contracts()
    recipe = _conditioning_trained_recipe(recipe)
    hf_export, _dcp_checkpoint, _adapter_path = _write_hf_authenticated_fixture(
        tmp_path, contracts=(manifest, normalization, recipe)
    )

    with pytest.raises(IsaacCheckpointImportError, match="does not implement"):
        authenticate_isaac_checkpoint(hf_export)


@pytest.mark.parametrize(
    "overrides",
    [
        {"action_conditioning_probability": 1.0},
        {"mistake_conditioning_probability": 1.0},
    ],
)
def test_authenticate_refuses_always_conditioned_training_even_conditioning_free(tmp_path, overrides):
    """probability >= 1 short-circuits the genesis Bernoulli gate: no unconditioned mode exists."""
    manifest, normalization, recipe = _make_contracts()
    recipe = _conditioning_trained_recipe(recipe, **overrides)
    hf_export, _dcp_checkpoint, _adapter_path = _write_hf_authenticated_fixture(
        tmp_path, contracts=(manifest, normalization, recipe)
    )

    with pytest.raises(IsaacCheckpointImportError, match="no unconditioned training mode"):
        authenticate_isaac_checkpoint(hf_export, conditioning_deployment=CONDITIONING_DISABLED_DEPLOYMENT)


@pytest.mark.parametrize(
    "deployment",
    [
        IsaacConditioningDeployment(renders_action_conditioning=True, renders_mistake_conditioning=False),
        IsaacConditioningDeployment(renders_action_conditioning=False, renders_mistake_conditioning=True),
    ],
)
def test_authenticate_refuses_a_deployment_that_claims_to_render_conditioning(tmp_path, deployment):
    """The import path never sets config.action_conditioning/mistake_conditioning."""
    manifest, normalization, recipe = _make_contracts()
    recipe = _conditioning_trained_recipe(recipe)
    hf_export, _dcp_checkpoint, _adapter_path = _write_hf_authenticated_fixture(
        tmp_path, contracts=(manifest, normalization, recipe)
    )

    with pytest.raises(IsaacCheckpointImportError, match="needs a conditioning-free deployment"):
        authenticate_isaac_checkpoint(hf_export, conditioning_deployment=deployment)


@pytest.mark.parametrize(
    "deployment",
    [
        None,
        IsaacConditioningDeployment(renders_action_conditioning=True, renders_mistake_conditioning=False),
    ],
)
def test_authenticate_still_refuses_flow_conditioning_wire_keys_undeclared(tmp_path, deployment):
    """Probabilities stay at zero so the wire-key rule, not the probability gate, is the one probed."""
    manifest, normalization, recipe = _make_contracts()
    recipe = _attach_flow_conditioning_wire_keys(_upgrade_recipe_to_schema7(recipe))
    hf_export, _dcp_checkpoint, _adapter_path = _write_hf_authenticated_fixture(
        tmp_path, contracts=(manifest, normalization, recipe)
    )

    with pytest.raises(IsaacCheckpointImportError, match="conditioning keys"):
        authenticate_isaac_checkpoint(hf_export, conditioning_deployment=deployment)


def test_isaac_import_cli_declares_the_conditioning_free_deployment(tmp_path, monkeypatch):
    """The CLI is the only surface that can declare it: the adapter is immutable and hash-bound."""
    from lerobot.scripts import lerobot_isaac_import

    captured: dict[str, object] = {}

    def _record_import(*args: object, **kwargs: object):
        captured.update(kwargs)
        return checkpoint_import.ImportedIsaacPackage(
            output_path=tmp_path / "package",
            policy_state_dataset="cloud/isaac_yam",
            normalization_scope="libero_spatial",
            objective="Flow",
            provenance={},
        )

    monkeypatch.setattr(lerobot_isaac_import, "import_authenticated_isaac_checkpoint", _record_import)
    monkeypatch.setattr(
        "sys.argv",
        [
            "lerobot_isaac_import",
            "--hf-export",
            str(tmp_path / "export"),
            "--output",
            str(tmp_path / "package"),
            "--policy-state-dataset",
            "cloud/isaac_yam",
            "--normalization-scope",
            "libero_spatial",
            "--deployment-adapter",
            str(tmp_path / "adapter.json"),
            "--conditioning-free-deployment",
        ],
    )

    lerobot_isaac_import.main()

    assert captured["conditioning_deployment"] == CONDITIONING_DISABLED_DEPLOYMENT


def test_isaac_import_cli_leaves_the_declaration_unset_by_default(tmp_path, monkeypatch):
    from lerobot.scripts import lerobot_isaac_import

    captured: dict[str, object] = {}

    def _record_import(*args: object, **kwargs: object):
        captured.update(kwargs)
        return checkpoint_import.ImportedIsaacPackage(
            output_path=tmp_path / "package",
            policy_state_dataset="cloud/isaac_yam",
            normalization_scope="libero_spatial",
            objective="Flow",
            provenance={},
        )

    monkeypatch.setattr(lerobot_isaac_import, "import_authenticated_isaac_checkpoint", _record_import)
    monkeypatch.setattr(
        "sys.argv",
        [
            "lerobot_isaac_import",
            "--hf-export",
            str(tmp_path / "export"),
            "--output",
            str(tmp_path / "package"),
            "--policy-state-dataset",
            "cloud/isaac_yam",
            "--normalization-scope",
            "libero_spatial",
            "--deployment-adapter",
            str(tmp_path / "adapter.json"),
        ],
    )

    lerobot_isaac_import.main()

    assert captured["conditioning_deployment"] is None
