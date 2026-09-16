"""Strict hardware training must not interpret permuted joints positionally."""

from types import SimpleNamespace

import pytest

from lerobot.policies.factory import make_policy
from lerobot.policies.perceptron_isaac.configuration_perceptron_isaac import PerceptronIsaacConfig
from lerobot.policies.perceptron_isaac.processor_perceptron_isaac import (
    make_perceptron_isaac_pre_post_processors,
    make_perceptron_isaac_pre_post_processors_from_pretrained,
)


def hardware_config(robot_type: str) -> PerceptronIsaacConfig:
    camera_order: tuple[str, ...]
    joint_signs: list[float] | None
    joint_offsets: list[float] | None
    if robot_type == "so100_so101":
        names = [
            "shoulder_pan.pos",
            "shoulder_lift.pos",
            "elbow_flex.pos",
            "wrist_flex.pos",
            "wrist_roll.pos",
            "gripper.pos",
        ]
        camera_order = ("top", "side")
        image_size = (256, 256)
        joint_signs = [1.0, -1.0, 1.0, 1.0, 1.0, 1.0]
        joint_offsets = [0.0, 90.0, 90.0, 0.0, 0.0, 0.0]
        dataset_name = "molmoact2_so100_101"
    else:
        names = [
            *(f"left_joint_{i}.pos" for i in range(6)),
            "left_gripper.pos",
            *(f"right_joint_{i}.pos" for i in range(6)),
            "right_gripper.pos",
        ]
        camera_order = ("top", "left", "right")
        image_size = (360, 640)
        joint_signs = None
        joint_offsets = None
        dataset_name = "molmoact2_bimanualyam"
    return PerceptronIsaacConfig(
        device="cpu",
        robot_type=robot_type,
        dataset_name=dataset_name,
        control_mode="joint",
        action_dim=len(names),
        proprio_dim=len(names),
        chunk_size=30,
        n_action_steps=30,
        num_inference_steps=10,
        num_flow_samples=1,
        clip_action_pose=False,
        gripper_binary_to_signed=False,
        num_settle_steps=0,
        apply_offset_norm=False,
        target_fps=30.0,
        strict_hardware_feature_contract=True,
        action_feature_names=names.copy(),
        state_feature_names=names.copy(),
        camera_order=camera_order,
        image_size=image_size,
        joint_signs=joint_signs,
        joint_offsets=joint_offsets,
        normalize_task_text=robot_type == "so100_so101",
    )


def hardware_dataset(config: PerceptronIsaacConfig) -> SimpleNamespace:
    assert config.action_feature_names is not None
    assert config.state_feature_names is not None
    features = {
        "action": {
            "dtype": "float32",
            "shape": (config.action_dim,),
            "names": config.action_feature_names.copy(),
        },
        "observation.state": {
            "dtype": "float32",
            "shape": (config.proprio_dim,),
            "names": config.state_feature_names.copy(),
        },
    }
    for camera in config.camera_order:
        features[f"observation.images.{camera}"] = {
            "dtype": "image",
            "shape": (*config.image_size, 3),
            "names": ["height", "width", "channels"],
        }
    return SimpleNamespace(features=features, stats={}, fps=30.0)


@pytest.mark.parametrize("robot_type", ["so100_so101", "bi_yam"])
@pytest.mark.parametrize("feature", ["action", "observation.state"])
@pytest.mark.parametrize("caller", ["policy", "fresh_processors", "restored_processors"])
def test_strict_hardware_rejects_permuted_dataset_joints(robot_type, feature, caller, tmp_path) -> None:
    config = hardware_config(robot_type)
    dataset = hardware_dataset(config)
    if caller == "restored_processors":
        pre, post = make_perceptron_isaac_pre_post_processors(config, dataset_meta=dataset)
        pre.save_pretrained(tmp_path)
        post.save_pretrained(tmp_path)
    names = dataset.features[feature]["names"]
    names[0], names[1] = names[1], names[0]
    with pytest.raises(ValueError, match="ISAAC dataset feature order"):
        if caller == "policy":
            make_policy(config, ds_meta=dataset)
        elif caller == "fresh_processors":
            make_perceptron_isaac_pre_post_processors(config, dataset_meta=dataset)
        else:
            make_perceptron_isaac_pre_post_processors_from_pretrained(config, tmp_path, dataset_meta=dataset)


@pytest.mark.parametrize("semantic_aliases", [False, True])
def test_strict_so100_accepts_ordered_hardware_names_and_approved_aliases(semantic_aliases) -> None:
    config = hardware_config("so100_so101")
    dataset = hardware_dataset(config)
    if semantic_aliases:
        for feature in ("action", "observation.state"):
            dataset.features[feature]["names"] = {
                "motors": [
                    "main_shoulder_pan",
                    "main_shoulder_lift",
                    "main_elbow_flex",
                    "main_wrist_flex",
                    "main_wrist_roll",
                    "main_gripper",
                ]
            }
    policy = make_policy(config, ds_meta=dataset)
    assert policy.config.action_feature_names == config.state_feature_names
    assert policy.config.action_feature_names[0] == "shoulder_pan.pos"


@pytest.mark.parametrize("caller", ["fresh_processors", "restored_processors"])
def test_strict_hardware_processors_reject_missing_feature_metadata(caller, tmp_path) -> None:
    config = hardware_config("so100_so101")
    dataset = hardware_dataset(config)
    if caller == "restored_processors":
        pre, post = make_perceptron_isaac_pre_post_processors(config, dataset_meta=dataset)
        pre.save_pretrained(tmp_path)
        post.save_pretrained(tmp_path)
    dataset.features = None
    with pytest.raises(ValueError, match="ISAAC dataset feature order"):
        if caller == "fresh_processors":
            make_perceptron_isaac_pre_post_processors(config, dataset_meta=dataset)
        else:
            make_perceptron_isaac_pre_post_processors_from_pretrained(config, tmp_path, dataset_meta=dataset)


@pytest.mark.parametrize("feature", ["action", "observation.state"])
@pytest.mark.parametrize("names", [None, [], ["unknown"]])
def test_strict_hardware_rejects_missing_or_unknown_joint_names(feature, names) -> None:
    config = hardware_config("so100_so101")
    dataset = hardware_dataset(config)
    dataset.features[feature]["names"] = names
    with pytest.raises(ValueError, match="ISAAC dataset feature order"):
        config.set_dataset_feature_metadata(dataset.features)
