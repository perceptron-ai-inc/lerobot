"""Standalone, trainer and direct eval callers must honor stateful policy limits."""

import pytest
import torch

from lerobot.configs.default import DatasetConfig, EvalConfig
from lerobot.configs.eval import EvalPipelineConfig
from lerobot.configs.train import TrainPipelineConfig
from lerobot.envs.configs import LiberoEnv
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.perceptron_isaac.configuration_perceptron_isaac import PerceptronIsaacConfig


@pytest.mark.parametrize("caller", ["standalone", "trainer"])
@pytest.mark.parametrize("batch_size,parallel", [(1, 2), (2, 1)])
def test_isaac_eval_rejects_explicit_unsupported_concurrency(caller, batch_size, parallel, tmp_path):
    policy = PerceptronIsaacConfig(device="cpu", push_to_hub=False)
    env = LiberoEnv(task="libero_spatial", max_parallel_tasks=parallel)
    evaluation = EvalConfig(batch_size=batch_size, n_episodes=2)
    with pytest.raises(ValueError, match="perceptron_isaac supports"):
        if caller == "standalone":
            EvalPipelineConfig(policy=policy, env=env, eval=evaluation, output_dir=tmp_path)
        else:
            cfg = TrainPipelineConfig(
                dataset=DatasetConfig(repo_id="synthetic/eval"),
                policy=policy,
                env=env,
                eval=evaluation,
                output_dir=tmp_path / "run",
            )
            cfg.validate()


def test_trainer_caps_automatic_batch_and_retains_generic_parallelism(monkeypatch, tmp_path):
    monkeypatch.setattr(EvalConfig, "_auto_batch_size", lambda self: 2)
    cfg = TrainPipelineConfig(
        dataset=DatasetConfig(repo_id="synthetic/eval"),
        policy=PerceptronIsaacConfig(device="cpu", push_to_hub=False),
        env=LiberoEnv(task="libero_spatial"),
        eval=EvalConfig(n_episodes=2),
        output_dir=tmp_path / "isaac",
    )
    cfg.validate()
    assert cfg.eval.batch_size == 1
    generic = EvalPipelineConfig(
        policy=ACTConfig(device="cpu", push_to_hub=False),
        env=LiberoEnv(task="libero_spatial", max_parallel_tasks=2),
        eval=EvalConfig(batch_size=2, n_episodes=2),
        output_dir=tmp_path / "generic",
    )
    assert generic.eval.batch_size == 2
    assert generic.env.max_parallel_tasks == 2


def test_direct_eval_rejects_shared_stateful_policy_parallelism():
    from lerobot.scripts.lerobot_eval import eval_policy_all

    policy = torch.nn.Linear(1, 1)
    policy.config = PerceptronIsaacConfig(device="cpu", push_to_hub=False)
    with pytest.raises(ValueError, match="perceptron_isaac supports"):
        eval_policy_all({}, policy, None, None, None, None, n_episodes=1, max_parallel_tasks=2)


def test_trainer_env_context_checks_policy_limits_before_allocating_envs(monkeypatch):
    from lerobot.scripts import lerobot_train

    cfg = TrainPipelineConfig(
        dataset=DatasetConfig(repo_id="synthetic/eval"),
        policy=PerceptronIsaacConfig(device="cpu", push_to_hub=False),
        env=LiberoEnv(task="libero_spatial"),
        eval=EvalConfig(batch_size=2, n_episodes=2),
    )

    def forbidden_env_allocation(*args, **kwargs):
        raise AssertionError("unsupported eval reached environment allocation")

    monkeypatch.setattr(lerobot_train, "make_env", forbidden_env_allocation)
    with pytest.raises(ValueError, match="perceptron_isaac supports"), lerobot_train._make_eval_envs(cfg):
        raise AssertionError("unsupported eval entered caller body")


def test_explicit_unsupported_batch_is_not_hidden_by_episode_count():
    evaluation = EvalConfig(batch_size=2, n_episodes=1)
    with pytest.raises(ValueError, match="perceptron_isaac supports"):
        evaluation.reconcile_policy_limits(PerceptronIsaacConfig(device="cpu"))


@pytest.mark.parametrize("caller", ["rollout", "eval_policy"])
def test_direct_single_task_callers_reject_batch_before_reset(caller):
    from types import SimpleNamespace

    from lerobot.scripts import lerobot_eval
    from tests.test_eval_recording import ConstantPolicy

    policy = ConstantPolicy()
    policy.config = PerceptronIsaacConfig(device="cpu")
    env = SimpleNamespace(num_envs=2)
    with pytest.raises(ValueError, match="perceptron_isaac supports"):
        if caller == "rollout":
            lerobot_eval.rollout(env, policy, None, None, None, None)
        else:
            lerobot_eval.eval_policy(env, policy, None, None, None, None, n_episodes=1)
