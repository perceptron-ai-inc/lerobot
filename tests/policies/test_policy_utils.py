import pytest

from lerobot.configs import FeatureType, PolicyFeature
from lerobot.policies.eo1.configuration_eo1 import EO1Config
from lerobot.policies.molmoact2.configuration_molmoact2 import MolmoAct2Config
from lerobot.policies.perceptron_isaac.configuration_perceptron_isaac import PerceptronIsaacConfig
from lerobot.utils.constants import ACTION, OBS_STATE


@pytest.mark.parametrize(
    ("config", "state_dim", "action_dim"),
    [
        pytest.param(PerceptronIsaacConfig(proprio_dim=11, action_dim=7), 11, 7, id="perceptron_isaac"),
        pytest.param(EO1Config(vlm_config={}, max_state_dim=13, max_action_dim=9), 13, 9, id="eo1"),
        pytest.param(MolmoAct2Config(), 0, 32, id="molmoact2"),
    ],
)
def test_vla_configs_share_feature_contract_without_changing_policy_specifics(
    config,
    state_dim: int,
    action_dim: int,
) -> None:
    with pytest.raises(ValueError):
        config.validate_features()

    config.input_features["observation.images.camera"] = PolicyFeature(
        type=FeatureType.VISUAL,
        shape=(3, 16, 16),
    )
    config.validate_features()

    assert config.input_features[OBS_STATE].shape == (state_dim,)
    assert config.output_features[ACTION].shape == (action_dim,)


def test_vla_feature_contract_preserves_existing_state_and_action_features() -> None:
    config = MolmoAct2Config(
        input_features={
            "observation.images.camera": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 16, 16)),
            OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(4,)),
        },
        output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(6,))},
    )

    config.validate_features()

    assert config.input_features[OBS_STATE].shape == (4,)
    assert config.output_features[ACTION].shape == (6,)
