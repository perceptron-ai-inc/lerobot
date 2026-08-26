"""Action-expert contract handling: genesis 'dit' aliasing, RTC gating, file authority."""

import pytest
import torch

pytest.importorskip("transformers", reason="transformers is required (install lerobot[perceptron_isaac])")

from lerobot.policies.perceptron_isaac.modeling_qwen35_vla import (
    DEFAULT_MOLMOACT_EXPERT_CFG,
    MolmoActExpertHead,
    build_action_expert_head,
    resolve_action_expert_config,
)

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
    assert isinstance(head, MolmoActExpertHead)
    reference = build_action_expert_head(_molmoact_cfg(), vlm_dim=_VLM_DIM)
    assert set(head.state_dict()) == set(reference.state_dict())
    assert head.rtc_max_delay_steps == 0


def test_dit_contract_schema_drift_fails_loud():
    with pytest.raises(ValueError, match="schema_version"):
        build_action_expert_head(_dit_contract(schema_version=2), vlm_dim=_VLM_DIM)
    with pytest.raises(ValueError, match="unknown fields"):
        build_action_expert_head(_dit_contract(new_genesis_field=1), vlm_dim=_VLM_DIM)


def test_non_dit_config_with_schema_version_is_rejected():
    with pytest.raises(ValueError, match="schema_version"):
        build_action_expert_head(_molmoact_cfg(schema_version=1), vlm_dim=_VLM_DIM)


def test_unknown_expert_type_still_fails_loud():
    with pytest.raises(ValueError, match="unknown action_expert type"):
        build_action_expert_head({"type": "genesis-flow-v2"}, vlm_dim=_VLM_DIM)


def test_sample_rejects_prefix_beyond_declared_rtc_capability():
    """The checkpoint contract's rtc_max_delay_steps gates prefix inpainting.

    Genesis gates serve-time RTC on the checkpoint's own trained budget; a non-RTC
    expert (budget 0) must reject any pinned prefix instead of silently inpainting.
    """
    head = build_action_expert_head(_dit_contract(), vlm_dim=_VLM_DIM)
    vlm_activations = torch.zeros(1, 3, _VLM_DIM)
    prefix = torch.zeros(1, 2, 4)
    with pytest.raises(ValueError, match="rtc_max_delay_steps"):
        head.sample(vlm_activations, action_prefix=prefix, prefix_length=2)


def test_sample_accepts_prefix_within_an_explicitly_configured_budget():
    """A developer-supplied molmoact config may opt in to a prefix budget; a genesis
    contract may not (see the rejection test below) because the vendored sampler's
    single-flow-time conditioning differs from genesis's RTC regime."""
    head = build_action_expert_head(_molmoact_cfg(rtc_max_delay_steps=2), vlm_dim=_VLM_DIM)
    vlm_activations = torch.zeros(1, 3, _VLM_DIM)
    prefix = torch.zeros(1, 2, 4)
    actions = head.sample(vlm_activations, action_prefix=prefix, prefix_length=2)
    assert actions.shape == (1, 6, 4)


def test_dit_contract_declaring_rtc_training_is_rejected():
    with pytest.raises(ValueError, match="cannot faithfully execute"):
        build_action_expert_head(_dit_contract(rtc_max_delay_steps=4), vlm_dim=_VLM_DIM)
    with pytest.raises(ValueError, match="cannot faithfully execute"):
        build_action_expert_head(_dit_contract(rtc_max_delay_steps=0, rtc_probability=0.5), vlm_dim=_VLM_DIM)


def test_resolver_defaults_to_molmoact_when_checkpoint_has_no_contract():
    resolved = resolve_action_expert_config(None)
    assert resolved == DEFAULT_MOLMOACT_EXPERT_CFG
    assert resolved is not DEFAULT_MOLMOACT_EXPERT_CFG  # defensive copy


def test_resolver_honors_the_checkpoint_stamped_contract():
    contract = _dit_contract()
    resolved = resolve_action_expert_config(
        contract, action_expert_overrides={"action_horizon": 6, "num_inference_steps": 12}
    )
    assert resolved["type"] == "dit"
    assert resolved["num_inference_steps"] == 12
    assert resolved["rtc_max_delay_steps"] == 0


def test_resolver_rejects_wholesale_override_of_a_stamped_contract():
    with pytest.raises(ValueError, match="authoritative action_expert contract"):
        resolve_action_expert_config(_dit_contract(), action_expert=_molmoact_cfg())


def test_resolver_rejects_horizon_contradicting_the_stamped_contract():
    with pytest.raises(ValueError, match="disagree on trained geometry"):
        resolve_action_expert_config(_dit_contract(), action_expert_overrides={"action_horizon": 50})


def test_resolver_applies_serving_overrides_to_the_default():
    resolved = resolve_action_expert_config(
        None, action_expert_overrides={"action_horizon": 50, "num_inference_steps": 12}
    )
    assert resolved["action_horizon"] == 50
    assert resolved["num_inference_steps"] == 12
    assert resolved["type"] == "molmoact"


def _stub_loader_boundaries(monkeypatch, vla, file_block, captured):
    from types import SimpleNamespace

    class _StubConfig:
        def __init__(self, action_expert):
            self.action_expert = action_expert

    def fake_config_from_pretrained(path, **kwargs):
        # The loader must NOT pass an action_expert kwarg: HF from_pretrained applies
        # kwargs as wholesale setattr, which would clobber a checkpoint-stamped block.
        assert "action_expert" not in kwargs, "loader clobbers the file's action_expert block"
        return _StubConfig(file_block)

    class _StubModel:
        def __init__(self, config):
            captured["config"] = config
            self._embeddings = torch.nn.Embedding(2, 2)
            self.lm_head = torch.nn.Linear(2, 2, bias=False)
            self.model = SimpleNamespace(language_model=None)

        def get_input_embeddings(self):
            return self._embeddings

        def to(self, *args, **kwargs):
            return self

        def eval(self):
            return self

    monkeypatch.setattr(vla.Qwen35VLAConfig, "from_pretrained", staticmethod(fake_config_from_pretrained))
    monkeypatch.setattr(vla, "Qwen35VLAForActionGeneration", _StubModel)
    monkeypatch.setattr(vla, "load_sharded_safetensors_into_model", lambda *a, **k: None)
    monkeypatch.setattr(vla, "_assert_offset_norm_matches_marker", lambda *a, **k: None)


def test_loader_honors_a_file_stamped_contract_end_to_end(monkeypatch):
    """Pin the glue: load_qwen35_vla_from_hf must let a checkpoint-stamped block win.

    Reverting to the historical `action_expert=<default dict>` kwarg (the silent-
    discard bug) fails this test twice: the from_pretrained stub rejects the kwarg,
    and the model would be built from the default instead of the dit contract.
    """
    import lerobot.policies.perceptron_isaac.modeling_qwen35_vla as vla

    captured: dict = {}
    _stub_loader_boundaries(monkeypatch, vla, _dit_contract(), captured)

    _, config = vla.load_qwen35_vla_from_hf(
        "/nonexistent/checkpoint",
        action_expert_overrides={"num_inference_steps": 12},
        apply_offset_norm=False,
    )
    assert captured["config"] is config
    assert config.action_expert["type"] == "dit"
    assert config.action_expert["num_inference_steps"] == 12
    assert config.action_expert["hidden_dim"] == _dit_contract()["hidden_dim"]

    with pytest.raises(ValueError, match="authoritative action_expert contract"):
        vla.load_qwen35_vla_from_hf(
            "/nonexistent/checkpoint", action_expert=_molmoact_cfg(), apply_offset_norm=False
        )


def test_loader_falls_back_to_the_default_for_legacy_checkpoints(monkeypatch):
    import lerobot.policies.perceptron_isaac.modeling_qwen35_vla as vla

    captured: dict = {}
    _stub_loader_boundaries(monkeypatch, vla, None, captured)

    _, config = vla.load_qwen35_vla_from_hf(
        "/nonexistent/checkpoint",
        action_expert_overrides={"action_horizon": 30, "num_inference_steps": 10},
        apply_offset_norm=False,
    )
    expected = dict(DEFAULT_MOLMOACT_EXPERT_CFG)
    expected.update({"action_horizon": 30, "num_inference_steps": 10})
    assert config.action_expert == expected
