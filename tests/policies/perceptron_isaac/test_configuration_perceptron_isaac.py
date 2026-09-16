import json

import pytest

from lerobot.configs import FeatureType, PolicyFeature
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.perceptron_isaac.configuration_perceptron_isaac import PerceptronIsaacConfig
from lerobot.policies.perceptron_isaac.mharmony_contract import SUPPORTED_MHARMONY_VERSION
from lerobot.utils.constants import ACTION, OBS_STATE


def test_image_preprocessing_defaults_to_legacy_stretch() -> None:
    assert PerceptronIsaacConfig().image_preprocessing == "stretch"


def test_mharmony_version_defaults_to_supported_public_release() -> None:
    assert PerceptronIsaacConfig().mharmony_version == SUPPORTED_MHARMONY_VERSION


def test_legacy_mharmony_marker_is_explicitly_upgraded() -> None:
    with pytest.warns(FutureWarning, match="genesis-in-tree"):
        config = PerceptronIsaacConfig(mharmony_version="genesis-in-tree")

    assert config.mharmony_version == SUPPORTED_MHARMONY_VERSION


def test_unknown_mharmony_version_is_rejected() -> None:
    with pytest.raises(ValueError, match=r"supports mharmony==0\.1\.0"):
        PerceptronIsaacConfig(mharmony_version="0.2.0")


def test_image_preprocessing_round_trips_in_policy_package(tmp_path) -> None:
    config = PerceptronIsaacConfig(image_preprocessing="letterbox")

    config.save_pretrained(tmp_path)
    loaded = PreTrainedConfig.from_pretrained(tmp_path)

    assert json.loads((tmp_path / "config.json").read_text())["image_preprocessing"] == "letterbox"
    assert loaded.image_preprocessing == "letterbox"


def test_image_preprocessing_rejects_unknown_geometry() -> None:
    with pytest.raises(ValueError, match="image_preprocessing"):
        PerceptronIsaacConfig(image_preprocessing="crop")


def test_serving_camera_roles_round_trip_physical_roles_separately_from_legacy_slots(tmp_path) -> None:
    config = PerceptronIsaacConfig(
        camera_order=("top", "side"),
        serving_camera_roles={"side": "top", "wrist": "side"},
    )

    config.save_pretrained(tmp_path)
    loaded = PreTrainedConfig.from_pretrained(tmp_path)

    assert loaded.camera_order == ("top", "side")
    assert loaded.serving_camera_roles == {"side": "top", "wrist": "side"}


@pytest.mark.parametrize(
    "roles",
    [
        {"side": "top"},
        {"side": "top", "wrist": "top"},
        {"side": "top", "wrist": "unknown"},
    ],
)
def test_serving_camera_roles_reject_ambiguous_or_unknown_slots(roles) -> None:
    with pytest.raises(ValueError, match="serving_camera_roles"):
        PerceptronIsaacConfig(camera_order=("top", "side"), serving_camera_roles=roles)


def test_training_action_frame_contract_rejects_legacy_version() -> None:
    with pytest.raises(ValueError, match="training_action_frame_contract_version=2"):
        PerceptronIsaacConfig(training_action_frame_contract_version=1)


@pytest.mark.parametrize("n_action_steps", [0, -1])
def test_config_rejects_nonpositive_action_steps(n_action_steps) -> None:
    with pytest.raises(ValueError, match="n_action_steps must be positive"):
        PerceptronIsaacConfig(device="cpu", n_action_steps=n_action_steps)


@pytest.mark.parametrize(
    "field_name",
    [
        "fast_processor_tree_sha256",
        "deployment_adapter_sha256",
        "qwen35_trained_package_manifest_sha256",
        "mk1_model_import_sha256",
        "mk1_source_model_import_sha256",
        "mk1_trained_package_manifest_sha256",
    ],
)
def test_config_rejects_noncanonical_checkpoint_digests(field_name) -> None:
    values = {field_name: "A" * 64}
    if field_name == "fast_processor_tree_sha256":
        values["fast_processor_path"] = "FAST"

    with pytest.raises(ValueError, match=rf"{field_name} must be a lowercase SHA-256 digest"):
        PerceptronIsaacConfig(device="cpu", **values)


def test_config_rejects_proprio_wider_than_vector_encoder() -> None:
    with pytest.raises(
        ValueError, match="proprio_dim must be positive and no greater than vector_max_states"
    ):
        PerceptronIsaacConfig(device="cpu", proprio_dim=129, vector_max_states=128)


@pytest.mark.parametrize(
    ("features", "expected_error"),
    [
        (
            {
                "input_features": {
                    OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(9,)),
                }
            },
            "observation.state feature width must match proprio_dim=8",
        ),
        (
            {
                "output_features": {
                    ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(6,)),
                }
            },
            "action feature width must match action_dim=7",
        ),
    ],
    ids=["state", "action"],
)
def test_config_rejects_mismatched_declared_feature_widths(features, expected_error) -> None:
    with pytest.raises(ValueError, match=expected_error):
        PerceptronIsaacConfig(device="cpu", **features)


@pytest.mark.parametrize(
    ("feature_group", "key", "feature"),
    [
        ("input_features", OBS_STATE, PolicyFeature(type=FeatureType.STATE, shape=(9,))),
        ("output_features", ACTION, PolicyFeature(type=FeatureType.ACTION, shape=(6,))),
    ],
    ids=["state", "action"],
)
def test_validate_features_rejects_widths_mutated_after_config_load(feature_group, key, feature) -> None:
    config = PerceptronIsaacConfig(
        device="cpu",
        input_features={
            "observation.images.image": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 16, 16))
        },
    )
    getattr(config, feature_group)[key] = feature

    with pytest.raises(ValueError, match="feature width must match"):
        config.validate_features()


def test_config_declares_single_environment_eval_limit() -> None:
    assert PerceptronIsaacConfig(device="cpu").max_eval_batch_size == 1
