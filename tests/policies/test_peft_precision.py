"""Shared adapter loading must preserve FP32 saved tensors over a BF16 base."""

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import load_file, save_file

from lerobot.policies.peft import load_peft_policy


class SavedBfloat16Policy(torch.nn.Module):
    """Small on-disk base implementing the policy loader interface, with real linear layers."""

    def __init__(self) -> None:
        super().__init__()
        self.projection = torch.nn.Linear(2, 2, bias=False, dtype=torch.bfloat16)
        self.head = torch.nn.Linear(2, 1, bias=False, dtype=torch.bfloat16)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.head(self.projection(inputs).to(self.head.weight.dtype))

    @classmethod
    def from_pretrained(cls, pretrained_name_or_path: str, **kwargs) -> "SavedBfloat16Policy":
        model = cls()
        model.load_state_dict(load_file(Path(pretrained_name_or_path) / "model.safetensors"))
        return model


@pytest.mark.parametrize("head_dtype", [torch.bfloat16, torch.float16, torch.float32, torch.float64])
@pytest.mark.parametrize("is_trainable", [False, True])
def test_load_peft_policy_preserves_saved_modules_to_save_precision(
    tmp_path: Path, is_trainable: bool, head_dtype: torch.dtype
) -> None:
    peft = pytest.importorskip("peft")
    base = tmp_path / "base"
    base.mkdir()
    model = SavedBfloat16Policy()
    save_file(model.state_dict(), base / "model.safetensors")
    adapted = peft.get_peft_model(
        model,
        peft.LoraConfig(
            r=1,
            target_modules=["projection"],
            modules_to_save=["head"],
        ),
    )
    adapted.peft_config["default"].base_model_name_or_path = str(base)
    with torch.no_grad():
        for name, parameter in adapted.named_parameters():
            if parameter.requires_grad:
                parameter.data = parameter.data.to(head_dtype if "modules_to_save" in name else torch.float32)
                parameter.fill_(0.123456)
    frozen = {name: p.clone() for name, p in adapted.named_parameters() if not p.requires_grad}
    saved = {key: value.clone() for key, value in peft.get_peft_model_state_dict(adapted).items()}
    adapter = tmp_path / "adapter"
    adapted.save_pretrained(adapter)
    reloaded = load_peft_policy(SavedBfloat16Policy, SimpleNamespace(), adapter, is_trainable=is_trainable)
    actual = peft.get_peft_model_state_dict(reloaded)
    assert saved.keys() == actual.keys()
    for key, expected in saved.items():
        torch.testing.assert_close(actual[key], expected, rtol=0, atol=0, msg=key)
    assert all(
        not parameter.is_meta and parameter.device.type == "cpu" for parameter in reloaded.parameters()
    )
    assert reloaded.base_model.model.projection.base_layer.weight.dtype == torch.bfloat16
    for name, parameter in reloaded.named_parameters():
        if "lora_" in name:
            assert parameter.requires_grad == is_trainable
        if "modules_to_save" in name:
            assert parameter.requires_grad == is_trainable
        if name in frozen:
            assert not parameter.requires_grad
            torch.testing.assert_close(parameter, frozen[name], rtol=0, atol=0)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        prediction = reloaded(torch.ones(1, 2))
    assert torch.isfinite(prediction).all()
    if is_trainable:
        optimizer = torch.optim.AdamW(p for p in reloaded.parameters() if p.requires_grad)
        prediction.float().square().sum().backward()
        optimizer.step()
        assert any(not torch.equal(actual[key], value) for key, value in saved.items())
