"""Native DiT metadata, retired RTC refusal, and ordinary MolmoAct compatibility."""

import pytest
import torch

pytest.importorskip("transformers", reason="transformers is required (install lerobot[perceptron_isaac])")

from lerobot.policies.perceptron_isaac.modeling_qwen35_vla import build_action_expert_head

_VLM_DIM = 16


def _dit_contract(**overrides):
    """A genesis-stamped DiT contract (schema v1, all 27 payload fields), tiny geometry."""
    contract = {
        "type": "dit",
        "schema_version": 1,
        "action_dim": 4,
        "action_horizon": 6,
        "num_layers": 1,
        "hidden_dim": 16,
        "num_heads": 2,
        "mlp_ratio": 1.0,
        "num_inference_steps": 2,
        "timestep_sampling_alpha": 1.5,
        "timestep_sampling_beta": 1.0,
        "timestep_sampling_scale": 0.999,
        "timestep_sampling_offset": 0.001,
        "train_samples_per_chunk": 1,
        "timestep_embed_dim": 8,
        "rtc_max_delay_steps": 0,
        "rtc_probability": None,
        "rtc_delay_sampling": "uniform",
        "rtc_poisson_mean": 5.0,
        "mask_padded_action_rows": False,
        "drop_action_dim_overflow": False,
        "ffn_multiple_of": 8,
        "qk_norm": True,
        "qk_norm_eps": 1e-6,
        "rope": True,
        "context_layer_norm": True,
        "causal_attn": False,
        "k_batched_cross_attn": True,
        "k_batched_cross_attn_backend": "flash_gqa",
    }
    contract.update(overrides)
    return contract


def _molmoact_cfg(**overrides):
    cfg = {
        "type": "molmoact",
        "action_dim": 4,
        "action_horizon": 6,
        "num_layers": 1,
        "hidden_dim": 16,
        "num_heads": 2,
        "mlp_ratio": 1.0,
        "num_inference_steps": 2,
        "timestep_embed_dim": 8,
        "ffn_multiple_of": 8,
    }
    cfg.update(overrides)
    return cfg


def test_genesis_dit_contract_builds_the_vendored_head():
    head = build_action_expert_head(_dit_contract(), vlm_dim=_VLM_DIM)
    assert type(head).__name__ == "DiTActionExpertHead"
    reference = build_action_expert_head(_molmoact_cfg(), vlm_dim=_VLM_DIM)
    assert set(head.state_dict()) == set(reference.state_dict())
    assert head.args.rtc_max_delay_steps == 0


def test_dit_contract_schema_drift_fails_loud():
    with pytest.raises(ValueError, match="schema_version"):
        build_action_expert_head(_dit_contract(schema_version=2), vlm_dim=_VLM_DIM)
    with pytest.raises(ValueError, match="unexpected fields"):
        build_action_expert_head(_dit_contract(new_genesis_field=1), vlm_dim=_VLM_DIM)


def test_non_dit_config_with_schema_version_is_rejected():
    with pytest.raises(ValueError, match="schema_version"):
        build_action_expert_head(_molmoact_cfg(schema_version=1), vlm_dim=_VLM_DIM)


def test_unknown_expert_type_still_fails_loud():
    with pytest.raises(ValueError, match="unknown action_expert type"):
        build_action_expert_head({"type": "genesis-flow-v2"}, vlm_dim=_VLM_DIM)


@pytest.mark.parametrize("kind", ["dit", "molmoact"])
@pytest.mark.parametrize("budget", [0, 2])
def test_source_rtc_budget_never_enables_execution(kind, budget):
    config = (
        _dit_contract(rtc_max_delay_steps=budget)
        if kind == "dit"
        else _molmoact_cfg(rtc_max_delay_steps=budget)
    )
    head = build_action_expert_head(config, vlm_dim=_VLM_DIM)
    with pytest.raises(ValueError, match="RTC"):
        head.sample(torch.zeros(1, 3, _VLM_DIM), action_prefix=torch.zeros(1, 2, 4), prefix_length=2)


def test_dit_contract_declaring_rtc_training_is_supported():
    head = build_action_expert_head(_dit_contract(rtc_max_delay_steps=4), vlm_dim=_VLM_DIM)
    assert head.args.rtc_max_delay_steps == 4
    with pytest.raises(ValueError, match="requires rtc_max_delay_steps"):
        build_action_expert_head(_dit_contract(rtc_max_delay_steps=0, rtc_probability=0.5), vlm_dim=_VLM_DIM)


def test_ordinary_dit_matches_legacy_values_and_rng():
    head = build_action_expert_head(
        _dit_contract(rtc_max_delay_steps=2, rtc_probability=0.5), vlm_dim=_VLM_DIM
    ).eval()
    legacy = build_action_expert_head(_molmoact_cfg(), vlm_dim=_VLM_DIM).eval()
    with torch.no_grad():
        for parameter in head.parameters():
            parameter.uniform_(-0.2, 0.2)
    legacy.load_state_dict(head.state_dict(), strict=True)
    vlm = torch.randn(2, 3, _VLM_DIM)
    vlm_mask = torch.tensor([[1, 1, 0], [1, 0, 0]])
    x = torch.randn(2, 6, 4)
    times = torch.tensor([0.25, 0.75])
    torch.testing.assert_close(head(vlm, vlm_mask, x, times), legacy(vlm, vlm_mask, x, times), rtol=0, atol=0)
    torch.manual_seed(99)
    actual = head.sample(vlm, vlm_mask)
    actual_rng = torch.get_rng_state()
    torch.manual_seed(99)
    expected = legacy.sample(vlm, vlm_mask)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert torch.equal(actual_rng, torch.get_rng_state())
