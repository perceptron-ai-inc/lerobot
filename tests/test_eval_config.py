from types import SimpleNamespace

import pytest

from lerobot.configs.default import EvalConfig
from lerobot.configs.eval import EvalPipelineConfig


def _policy(*, max_eval_batch_size):
    return SimpleNamespace(type="test", max_eval_batch_size=max_eval_batch_size)


def test_eval_pipeline_caps_auto_batch_size_to_policy_limit(monkeypatch, tmp_path):
    monkeypatch.setattr(EvalConfig, "_auto_batch_size", lambda _self: 8)
    eval_config = EvalConfig(batch_size=0, n_episodes=8)

    cfg = EvalPipelineConfig(
        env=None,
        eval=eval_config,
        policy=_policy(max_eval_batch_size=1),
        output_dir=tmp_path,
        job_name="test",
    )

    assert cfg.eval.batch_size == 1


def test_eval_pipeline_rejects_explicit_batch_size_above_policy_limit(tmp_path):
    with pytest.raises(ValueError, match="test supports eval.batch_size at most 1; got 2"):
        EvalPipelineConfig(
            env=None,
            eval=EvalConfig(batch_size=2, n_episodes=8),
            policy=_policy(max_eval_batch_size=1),
            output_dir=tmp_path,
            job_name="test",
        )


def test_eval_pipeline_leaves_unlimited_policy_batch_size_unchanged(tmp_path):
    cfg = EvalPipelineConfig(
        env=None,
        eval=EvalConfig(batch_size=4, n_episodes=8),
        policy=_policy(max_eval_batch_size=None),
        output_dir=tmp_path,
        job_name="test",
    )

    assert cfg.eval.batch_size == 4
