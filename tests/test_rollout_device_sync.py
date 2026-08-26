from unittest.mock import MagicMock

from tests.mocks.mock_robot import MockRobot, MockRobotConfig
from tests.test_rollout import _rollout_context_harness


def test_rollout_device_override_reaches_policy_load_and_restored_processor(monkeypatch):
    robot = MockRobot(MockRobotConfig(n_motors=2))
    rollout_context, cfg = _rollout_context_harness(monkeypatch, robot)
    cfg.policy.device = "cpu"
    cfg.device = "cuda:7"

    policy = MagicMock(type="mock")
    policy.to.return_value = policy
    captured = {}

    def load_policy(policy_config):
        captured["load_device"] = policy_config.device
        policy.config = policy_config
        return policy

    def make_processors(**kwargs):
        captured["processor_config_device"] = kwargs["policy_cfg"].device
        captured["processor_override_device"] = kwargs["preprocessor_overrides"]["device_processor"]["device"]
        return MagicMock(steps=[]), MagicMock(steps=[])

    monkeypatch.setattr(rollout_context, "_load_pretrained_policy", load_policy)
    monkeypatch.setattr(rollout_context, "make_pre_post_processors", make_processors)
    monkeypatch.setattr(rollout_context, "create_inference_engine", lambda *args, **kwargs: MagicMock())
    monkeypatch.setattr(rollout_context, "validate_visual_features_consistency", lambda *args: None)

    context = rollout_context.build_rollout_context(cfg, MagicMock())

    assert captured == {
        "load_device": "cuda:7",
        "processor_config_device": "cuda:7",
        "processor_override_device": "cuda:7",
    }
    policy.to.assert_called_once_with("cuda:7")
    context.hardware.robot_wrapper.inner.disconnect()
