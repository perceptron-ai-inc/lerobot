"""P2.11: training recomputes F32 RoPE while serving retains checkpoint BF16 numerics."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import pytest
import torch
from torch import Tensor

pytest.importorskip("transformers", reason="requires lerobot[perceptron_isaac]")

from lerobot.policies.perceptron_isaac.mk1_checkpoint_contract import Mk1CheckpointContract
from lerobot.policies.perceptron_isaac.modeling_mk1_vla import (
    Mk1Qwen36VLAForActionGeneration,
    _build_config,
    load_mk1_vla_from_hf,
)
from lerobot.policies.perceptron_isaac.modeling_qwen35_vla import setup_qwen35_vla_for_training
from tests.policies.perceptron_isaac.test_mk1_checkpoint_contract import (
    _config,
    _write_portable_isaac05_checkpoint,
)


def _mrope_oracle(positions: Tensor, frequencies: Tensor, sections: list[int]) -> tuple[Tensor, Tensor]:
    phases = positions.float().unsqueeze(-1) * frequencies.view(1, 1, 1, -1)
    interleaved = phases[0].clone()
    for axis, offset in enumerate((1, 2), start=1):
        slots = slice(offset, sections[axis] * 3, 3)
        interleaved[..., slots] = phases[axis, ..., slots]
    embedding = torch.cat((interleaved, interleaved), dim=-1)
    return embedding.cos(), embedding.sin()


@pytest.mark.parametrize("intent", ["training", "inference"])
def test_native_load_intent_rederives_training_rope_but_keeps_bf16_serving(
    tmp_path: Path,
    intent: Literal["training", "inference"],
) -> None:
    raw = _config(test_geometry=True, mtp_present=False)
    # Default tiny fixture has only frequency 1.0 and cannot expose BF16 precision loss.
    raw["text_config"]["head_dim"] = 12
    raw["text_config"]["rope_parameters"]["mrope_section"] = [1, 1, 1]
    root = tmp_path / "native"
    _, _, value = _write_portable_isaac05_checkpoint(root, config=raw)
    before = {
        path.relative_to(root).as_posix(): path.read_bytes() for path in root.rglob("*") if path.is_file()
    }
    dtype = torch.float32 if intent == "training" else torch.bfloat16
    model, config, contract = load_mk1_vla_from_hf(
        root,
        dtype=dtype,
        device="cpu",
        allow_test_only_reduced_geometry=True,
        allowed_storage_dtypes=frozenset({"F32"}),
        load_intent=intent,
    )
    rotary = model.model.language_model.rotary_emb
    parameters = config.text_config.rope_parameters
    dimension = int(config.text_config.head_dim * parameters["partial_rotary_factor"])
    fresh = 1.0 / (float(parameters["rope_theta"]) ** (torch.arange(0, dimension, 2).float() / dimension))
    quantized = fresh.bfloat16().float()
    assert not torch.equal(fresh, quantized), "fixture must distinguish recomputation from a cast"
    expected = fresh if intent == "training" else quantized
    assert rotary.inv_freq.dtype == dtype
    torch.testing.assert_close(rotary.inv_freq.float(), expected, rtol=0, atol=0)
    positions = torch.tensor([[[0, 12345]], [[17, 23456]], [[29, 32123]]])
    # BF16 activations under mixed precision must still use the F32 training frequency table.
    hidden = torch.zeros(1, 2, config.text_config.hidden_size, dtype=torch.bfloat16)
    cos, sin = rotary(hidden, positions)
    expected_cos, expected_sin = _mrope_oracle(positions, expected, parameters["mrope_section"])
    torch.testing.assert_close(cos, expected_cos.bfloat16(), rtol=0, atol=0)
    torch.testing.assert_close(sin, expected_sin.bfloat16(), rtol=0, atol=0)
    assert rotary.state_dict() == {}
    assert set(model.state_dict()) == set(contract.expected_tensor_shapes())
    for parameter in model.parameters():
        assert parameter.dtype == dtype
        torch.testing.assert_close(
            parameter, torch.full_like(parameter, value.to(dtype).item()), rtol=0, atol=0
        )
    after = {
        path.relative_to(root).as_posix(): path.read_bytes() for path in root.rglob("*") if path.is_file()
    }
    assert after == before


def test_training_setup_recomputes_bf16_source_rope_without_replacing_parameters() -> None:
    raw = _config(test_geometry=True, mtp_present=False)
    raw["text_config"]["head_dim"] = 12
    raw["text_config"]["rope_parameters"]["mrope_section"] = [1, 1, 1]
    contract = Mk1CheckpointContract.parse_allowlisted(raw, allow_test_only_reduced_geometry=True)
    model = Mk1Qwen36VLAForActionGeneration(_build_config(raw, contract)).bfloat16().eval()
    rotary = model.model.language_model.rotary_emb
    assert rotary.inv_freq.dtype == torch.bfloat16
    parameters = tuple(model.parameters())
    identities = tuple(id(parameter) for parameter in parameters)
    storage = tuple(parameter.data_ptr() for parameter in parameters)
    bits = [parameter.detach().view(torch.int16).clone() for parameter in parameters]
    optimizer = torch.optim.SGD(parameters, lr=0.01)
    setup_qwen35_vla_for_training(model, train_samples_per_chunk=2)
    rope = raw["text_config"]["rope_parameters"]
    dimension = int(raw["text_config"]["head_dim"] * rope["partial_rotary_factor"])
    fresh = 1.0 / (float(rope["rope_theta"]) ** (torch.arange(0, dimension, 2).float() / dimension))
    assert rotary.inv_freq.dtype == torch.float32
    torch.testing.assert_close(rotary.inv_freq, fresh, rtol=0, atol=0)
    assert tuple(id(parameter) for parameter in model.parameters()) == identities
    assert tuple(parameter.data_ptr() for parameter in model.parameters()) == storage
    assert tuple(id(parameter) for parameter in optimizer.param_groups[0]["params"]) == identities
    for parameter, original in zip(model.parameters(), bits, strict=True):
        assert parameter.dtype == torch.bfloat16
        assert torch.equal(parameter.detach().view(torch.int16), original)
    assert rotary.state_dict() == {}
