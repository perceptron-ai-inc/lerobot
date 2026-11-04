"""Source-bound prompt identity is distinct from strict physical LIBERO deployment."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from lerobot.configs import FeatureType, PolicyFeature
from lerobot.policies.perceptron_isaac.configuration_perceptron_isaac import PerceptronIsaacConfig
from lerobot.policies.perceptron_isaac.mharmony_native import (
    IsaacMharmonyRenderMetadata,
    _format_robotics_configuration_text,
)


def _source_profile() -> dict:
    return {
        "policy_state_dataset": "libero",
        "normalization_scope": "libero_10",
        "objective": "Flow",
        "deployment_profile": {
            "robot_type": "franka",
            "camera": {"observation_keys": ["image", "wrist_image"], "prompt_views": ["primary", "wrist"]},
            "state_action_schema": {
                "name": "gripper_7",
                "control_mode": None,
                "components": ["world_vector", "rotation_delta", "gripper_closedness_action"],
            },
        },
    }


def _physical_profile(tmp_path: Path) -> PerceptronIsaacConfig:
    model = tmp_path / "hf_model"
    model.mkdir()
    (model / "policy_inference_recipe.json").write_text(json.dumps({"recipes": [_source_profile()]}))
    return PerceptronIsaacConfig(
        artifact_kind="trained_policy",
        hf_model_path=str(model),
        robot_type="libero",
        control_mode="ee",
        dataset_name="libero",
        policy_state_dataset="libero",
        normalization_profile_scope="libero_10",
        strict_environment_feature_contract=True,
        action_dim=7,
        proprio_dim=8,
        n_obs_steps=1,
        chunk_size=10,
        n_action_steps=8,
        image_size=(256, 256),
        camera_order=("image", "wrist_image"),
        clip_action_pose=True,
        gripper_binary_to_signed=False,
        normalize_task_text=True,
        target_fps=20.0,
        num_settle_steps=0,
        render_min_num_patches=784,
        render_max_num_patches=784,
        input_features={
            "observation.images.image": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 256, 256)),
            "observation.images.wrist_image": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 256, 256)),
            "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(8,)),
        },
        output_features={"action": PolicyFeature(type=FeatureType.ACTION, shape=(7,))},
    )


def _render_profile() -> IsaacMharmonyRenderMetadata:
    return IsaacMharmonyRenderMetadata(
        robot_type="franka",
        control_mode=None,
        dataset_name="libero",
        image_size=[256, 256],
        camera_order=["image", "wrist_image"],
        camera_views=["primary", "wrist"],
        action_layout=["world_vector", "rotation_delta", "gripper_closedness_action"],
        n_obs_steps=1,
        action_horizon=10,
        action_dim=7,
        proprio_dim=8,
        min_num_patches=784,
        max_num_patches=784,
        patch_size=16,
        pixel_shuffle_scale=2,
        image_resize_size=None,
    )


def test_source_bound_prompt_preserves_franka_null_control_and_physical_strictness(tmp_path: Path) -> None:
    config = _physical_profile(tmp_path)
    metadata = _render_profile()
    metadata.validate_for_config(config)
    assert config.strict_environment_feature_contract is True
    assert config.robot_type == "libero" and config.control_mode == "ee"
    text = _format_robotics_configuration_text(metadata, fps=20.0)
    assert "robot_type: franka" in text
    assert "robot_type: libero" not in text and "control_mode:" not in text
    assert metadata.image_resize_size is None
    assert metadata.min_num_patches == metadata.max_num_patches == 784


def test_source_bound_prompt_survives_metadata_json_roundtrip(tmp_path: Path) -> None:
    config = _physical_profile(tmp_path)
    path = tmp_path / "render.json"
    path.write_text(json.dumps(_render_profile().to_json_dict()))
    loaded = IsaacMharmonyRenderMetadata.from_json_file(path)
    loaded.validate_for_config(config)
    assert loaded.robot_type == "franka" and loaded.control_mode is None
    assert loaded.image_resize_size is None


def test_existing_identical_libero_prompt_needs_no_source_alias(tmp_path: Path) -> None:
    config = _physical_profile(tmp_path)
    assert config.hf_model_path is not None
    (Path(config.hf_model_path) / "policy_inference_recipe.json").unlink()
    metadata = IsaacMharmonyRenderMetadata.from_config(config)
    metadata.validate_for_config(config)


@pytest.mark.parametrize(
    "defect",
    ["missing_recipe", "wrong_robot", "wrong_control", "wrong_schema", "wrong_dataset", "wrong_scope"],
)
def test_source_bound_prompt_refuses_unrelated_or_missing_contracts(tmp_path: Path, defect: str) -> None:
    config = _physical_profile(tmp_path)
    recipe = _source_profile()
    assert config.hf_model_path is not None
    path = Path(config.hf_model_path) / "policy_inference_recipe.json"
    if defect == "missing_recipe":
        path.unlink()
    else:
        if defect == "wrong_robot":
            recipe["deployment_profile"]["robot_type"] = "different_robot"
        elif defect == "wrong_control":
            recipe["deployment_profile"]["state_action_schema"]["control_mode"] = "joint"
        elif defect == "wrong_schema":
            recipe["deployment_profile"]["state_action_schema"]["name"] = "joint_gripper_14"
        elif defect == "wrong_dataset":
            recipe["policy_state_dataset"] = "different_dataset"
        else:
            recipe["normalization_scope"] = "libero_spatial"
        path.write_text(json.dumps({"recipes": [recipe]}))
    with pytest.raises(ValueError):
        _render_profile().validate_for_config(config)
