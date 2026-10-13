from __future__ import annotations

import torch

from lerobot.configs import FeatureType, PolicyFeature
from lerobot.policies.perceptron_isaac.configuration_perceptron_isaac import PerceptronIsaacConfig
from lerobot.policies.perceptron_isaac.loss_perceptron_isaac import (
    ISAAC_FLOW_LOSS_MASK_KEY,
    ISAAC_FLOW_NOISE_KEY,
    ISAAC_FLOW_TIMESTEPS_KEY,
)
from lerobot.policies.perceptron_isaac.modeling_perceptron_isaac import PerceptronIsaacPolicy
from lerobot.policies.perceptron_isaac.processor_perceptron_isaac import (
    PERCEPTRON_ISAAC_KEPT_SAMPLE_INDICES_KEY,
    PERCEPTRON_ISAAC_STREAM_KEY,
)
from lerobot.utils.constants import ACTION, OBS_STATE
from tests.policies.perceptron_isaac._fixtures import TinyIsaacModel, flow_stream


def _config() -> PerceptronIsaacConfig:
    return PerceptronIsaacConfig(
        device="cpu",
        n_obs_steps=1,
        chunk_size=2,
        n_action_steps=2,
        action_dim=2,
        proprio_dim=2,
        train_samples_per_chunk=1,
        freeze_input_embeddings=False,
        input_features={
            "observation.images.image": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 256, 256)),
            OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(2,)),
        },
        output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(2,))},
    )


def _policy_with_tiny_model() -> PerceptronIsaacPolicy:
    policy = PerceptronIsaacPolicy(_config())
    policy._isaac_model = TinyIsaacModel()
    policy._configure_training_parameters = lambda: None
    return policy


def test_policy_forward_runs_native_joint_loss_and_returns_scalar_metrics():
    policy = _policy_with_tiny_model()
    loss, metrics = policy.forward(
        {
            PERCEPTRON_ISAAC_STREAM_KEY: flow_stream(),
            ISAAC_FLOW_TIMESTEPS_KEY: torch.tensor([[0.5]]),
            ISAAC_FLOW_NOISE_KEY: torch.zeros(1, 1, 2, 2),
        }
    )

    assert loss.ndim == 0
    assert metrics.keys() >= {"loss", "text_loss", "flow_loss", "flow_mean_timestep"}
    assert all(isinstance(value, float) for value in metrics.values())
    loss.backward()
    assert policy._isaac_model.action_expert.scale.grad is not None
    assert policy._isaac_model.context_seed.grad is not None


def test_policy_forward_aligns_explicit_flow_fixtures_after_outlier_skip():
    policy = _policy_with_tiny_model()
    loss, _ = policy.forward(
        {
            PERCEPTRON_ISAAC_STREAM_KEY: flow_stream(),
            PERCEPTRON_ISAAC_KEPT_SAMPLE_INDICES_KEY: [1],
            # The rejected sample deliberately contains a non-scalar tau grid. It must be removed
            # before the native loss validates the retained sample.
            ISAAC_FLOW_TIMESTEPS_KEY: torch.tensor([[[[0.1], [0.2]], [[0.5], [0.5]]]]),
            ISAAC_FLOW_NOISE_KEY: torch.zeros(1, 2, 2, 2),
            ISAAC_FLOW_LOSS_MASK_KEY: torch.ones(1, 2, 2, 2),
        }
    )

    assert torch.isfinite(loss)


def test_policy_optimizer_groups_use_checkpoint_recipe_learning_rates():
    policy = _policy_with_tiny_model()

    groups = policy.get_optim_params()

    assert [group["lr"] for group in groups] == [1e-5, 5e-6, 5e-5]
    parameter_ids = [id(parameter) for group in groups for parameter in group["params"]]
    assert len(parameter_ids) == len(set(parameter_ids))
    assert set(parameter_ids) == {
        id(parameter) for parameter in policy._isaac_model.parameters() if parameter.requires_grad
    }
    optimizer = policy.config.get_optimizer_preset()
    scheduler = policy.config.get_scheduler_preset()
    assert optimizer.eps == 1e-6
    assert optimizer.weight_decay == 0.0
    assert scheduler.num_warmup_steps == 200
