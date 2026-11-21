"""A checkpoint's declared RTC prefix budget survives a training stage that ran without RTC.

The action-expert recipe declares the inference prefix budget and every stage carries it forward
unchanged. A training stage records its own RTC setting separately, so an RTC-off fine-tune of an
RTC-capable source keeps the budget as an inherited declaration that nothing has validated on the
new weights. A checkpoint with no declared or inherited budget keeps rejecting trained prefixes.

The lifecycle tests run the public load, training-setup, save, finalize and reload path on the
reduced fixture geometry; only the FSDP2 topology and the fixture geometry gate are simulated.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest
import torch

pytest.importorskip("transformers", reason="requires the existing perceptron_isaac test environment")

from lerobot.policies.chunk import ExecutionMode
from lerobot.policies.perceptron_isaac.configuration_perceptron_isaac import PerceptronIsaacConfig
from lerobot.policies.perceptron_isaac.modeling_perceptron_isaac import PerceptronIsaacPolicy
from lerobot.policies.perceptron_isaac.rtc import RtcCapability, resolve_rtc_capability
from lerobot.policies.perceptron_isaac.trained_package import finalize_canonical_trained_package
from tests.policies.perceptron_isaac.test_checkpoint_lifecycle import _config, _source_package
from tests.policies.perceptron_isaac.test_trained_storage_contract import (
    _render_metadata,
    _set_trained_values,
    reduced_loader as _reduced_loader,
)

reduced_loader = _reduced_loader


@pytest.mark.parametrize(
    ("action", "prediction_steps", "final_stage", "budget", "inherited"),
    [
        pytest.param({"rtc_max_delay_steps": 30, "rtc_probability": 0.5}, 60, None, 30, False, id="source"),
        pytest.param(
            {"rtc_max_delay_steps": 30, "rtc_probability": 0.5}, 60, 0, 30, True, id="rtc_off_stage"
        ),
        pytest.param({"rtc_max_delay_steps": 30, "rtc_probability": None}, 60, None, 30, False, id="no_prob"),
        pytest.param({"rtc_max_delay_steps": 30, "rtc_probability": 0.5}, 8, 0, 7, True, id="short_chunk"),
        pytest.param({"rtc_max_delay_steps": 30, "rtc_probability": 0.0}, 60, 0, 0, False, id="zero_prob"),
        pytest.param({"rtc_max_delay_steps": 0, "rtc_probability": None}, 60, 0, 0, False, id="no_budget"),
        pytest.param({"type": "molmoact"}, 60, None, 0, False, id="legacy_block"),
        pytest.param(None, 60, 0, 0, False, id="no_recipe"),
    ],
)
def test_capability_separates_declared_budget_from_the_final_stage(
    action: dict | None, prediction_steps: int, final_stage: int | None, budget: int, inherited: bool
) -> None:
    capability = resolve_rtc_capability(
        action, prediction_steps=prediction_steps, final_stage_max_delay_steps=final_stage
    )
    assert capability.prefix_budget == budget
    assert capability.inherited is inherited
    assert capability.final_stage_max_delay_steps == final_stage
    assert capability.declared_max_delay_steps == (
        0 if action is None else action.get("rtc_max_delay_steps", 0)
    )


def test_capability_rejects_malformed_recipe_values() -> None:
    with pytest.raises(ValueError, match="rtc_max_delay_steps must be an integer"):
        resolve_rtc_capability(
            {"rtc_max_delay_steps": True}, prediction_steps=4, final_stage_max_delay_steps=None
        )
    with pytest.raises(ValueError, match="rtc_probability must be a number"):
        resolve_rtc_capability(
            {"rtc_max_delay_steps": 2, "rtc_probability": "0.5"},
            prediction_steps=4,
            final_stage_max_delay_steps=None,
        )
    with pytest.raises(ValueError, match="rtc_final_stage_max_delay_steps"):
        _config(rtc_final_stage_max_delay_steps=-1)


def _write_action_recipe(package: Path, *, max_delay: int, probability: float | None) -> None:
    (package / "hf_model").mkdir(parents=True, exist_ok=True)
    (package / "hf_model" / "config.json").write_text(
        json.dumps({"action_expert": {"rtc_max_delay_steps": max_delay, "rtc_probability": probability}})
    )


def test_config_reads_the_budget_before_weights_load_and_the_recorded_stage(tmp_path: Path) -> None:
    package = tmp_path / "package"
    _write_action_recipe(package, max_delay=3, probability=0.5)
    config = _config(chunk_size=6, n_action_steps=2, hf_model_path="hf_model")
    assert config.rtc_capability() == RtcCapability(0, 0, None, None), "no package to read yet"
    config.pretrained_path = package
    assert config.rtc_capability() == RtcCapability(3, 3, 0.5, None)
    assert config.rtc_training_max_delay == 3
    assert not config.rtc_capability().inherited
    # A stage that trained without RTC marks the budget inherited and leaves it unchanged.
    config.rtc_final_stage_max_delay_steps = 0
    assert config.rtc_capability() == RtcCapability(3, 3, 0.5, 0)
    assert config.rtc_capability().inherited
    assert config.rtc_training_max_delay == 3


def _trained_policy(root: Path, *, max_delay: int, probability: float | None) -> PerceptronIsaacPolicy:
    """Load a source package, then run the one-time training setup of an RTC-off stage."""
    config = _source_package(root)
    model_config_path = root / "hf_model" / "config.json"
    raw = json.loads(model_config_path.read_text())
    for action in (raw["action_expert"], raw["isaac05_vla"]["action_expert"]):
        action.update(rtc_max_delay_steps=max_delay, rtc_probability=probability)
    model_config_path.write_text(json.dumps(raw))
    config.freeze_input_embeddings = True
    render_metadata_path = root / "native_render_metadata.json"
    _render_metadata(config, render_metadata_path)
    # Chunk inference is offered only to recipes that render no episode-time tokens.
    render_metadata = json.loads(render_metadata_path.read_text())
    render_metadata["emit_observation_timestamps"] = False
    render_metadata_path.write_text(json.dumps(render_metadata))
    finalize_canonical_trained_package(
        config, root, mk1_source_model_import_sha256=config.mk1_source_model_import_sha256
    )
    policy = PerceptronIsaacPolicy.from_pretrained(root, local_files_only=True)
    assert policy.config.rtc_final_stage_max_delay_steps is None, "a source package records no stage"
    policy._load_backbone()
    policy._retain_verified_runtime_assets(root)
    policy._configure_training_parameters()
    return policy


def _save_and_reload(policy: PerceptronIsaacPolicy, saved: Path) -> PerceptronIsaacPolicy:
    _set_trained_values(policy, "mixed")
    policy.save_pretrained(saved, finalize_package=False)
    policy.finalize_pretrained_package(saved)
    inference_config = PerceptronIsaacConfig.from_pretrained(saved)
    inference_config.train_storage_fp32 = False
    return PerceptronIsaacPolicy.from_pretrained(
        saved, config=inference_config, local_files_only=True, strict=True
    )


def test_rtc_off_finetune_of_an_rtc_capable_source_inherits_the_budget(
    tmp_path: Path, reduced_loader: list, caplog: pytest.LogCaptureFixture
) -> None:
    source = tmp_path / "source"
    policy = _trained_policy(source, max_delay=12, probability=0.5)
    source_model_config = (source / "hf_model" / "config.json").read_bytes()
    source_identity = policy.config.mk1_source_model_import_sha256
    assert policy.config.flow_rtc_max_delay_steps == 0
    saved = tmp_path / "saved"

    reloaded = _save_and_reload(policy, saved)

    # The declared budget and its source provenance are carried forward byte for byte.
    assert (saved / "hf_model" / "config.json").read_bytes() == source_model_config
    saved_config = json.loads((saved / "config.json").read_text())
    assert saved_config["mk1_source_model_import_sha256"] == source_identity
    # The stage records that it trained without RTC, separately from that declaration.
    assert saved_config["rtc_final_stage_max_delay_steps"] == 0
    assert saved_config["flow_rtc_max_delay_steps"] == 0
    # The saved render metadata keeps the recipe's timestamp flag for later stages.
    saved_render_metadata = json.loads((saved / "native_render_metadata.json").read_text())
    assert saved_render_metadata["emit_observation_timestamps"] is False

    capability = reloaded.rtc_capability()
    # chunk_size=2 in the fixture, so one row can be pinned while one is still generated.
    assert capability == RtcCapability(1, 12, 0.5, 0)
    assert capability.inherited
    assert reloaded.supports_rtc()
    with caplog.at_level(logging.WARNING):
        spec = reloaded.chunk_inference_spec()
    assert spec.modes == (ExecutionMode.CHUNK, ExecutionMode.RTC_TRAINED)
    assert spec.training_max_delay == 1
    assert "inherited from the source checkpoint" in caplog.text
    assert "unvalidated" in caplog.text

    # The reloaded sampler still admits a prefix within the inherited budget.
    assert reloaded._isaac_model is not None
    expert = reloaded._isaac_model.model.action_expert.float()
    assert expert.args.rtc_max_delay_steps == 12
    prefix = torch.full((1, 1, expert.action_dim), 0.25)
    actions = expert.sample(
        torch.zeros(1, 3, expert.vlm_dim), num_action_steps=2, action_prefix=prefix, prefix_length=1
    )
    torch.testing.assert_close(actions[:, :1], prefix, rtol=0, atol=0)


def test_checkpoint_without_a_declared_or_inherited_budget_rejects_trained_prefixes(
    tmp_path: Path, reduced_loader: list
) -> None:
    policy = _trained_policy(tmp_path / "source", max_delay=0, probability=None)
    reloaded = _save_and_reload(policy, tmp_path / "saved")

    capability = reloaded.rtc_capability()
    assert capability == RtcCapability(0, 0, None, 0)
    assert not capability.inherited
    assert not reloaded.supports_rtc()
    assert reloaded.chunk_inference_spec().modes == (ExecutionMode.CHUNK,)
    assert reloaded.chunk_inference_spec().training_max_delay == 0

    assert reloaded._isaac_model is not None
    expert = reloaded._isaac_model.model.action_expert.float()
    with pytest.raises(ValueError, match="maximum supported RTC prefix 0"):
        expert.sample(
            torch.zeros(1, 3, expert.vlm_dim),
            num_action_steps=2,
            action_prefix=torch.zeros(1, 1, expert.action_dim),
            prefix_length=1,
        )
