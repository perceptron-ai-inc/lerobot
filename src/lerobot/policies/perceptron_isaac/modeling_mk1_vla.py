"""Config-driven MK1 Qwen3.6 null-MoE composition for the existing ISAAC VLA shell."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Literal

import torch
import transformers
from accelerate import init_empty_weights
from accelerate.utils import set_module_tensor_to_device
from safetensors import safe_open
from torch import nn
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
    Qwen3_5MoePreTrainedModel,
    Qwen3_5MoeRMSNormGated,
    torch_causal_conv1d_update,
    torch_chunk_gated_delta_rule,
    torch_recurrent_gated_delta_rule,
)

from .mk1_checkpoint_contract import (
    Mk1CheckpointContract,
    SafetensorsInventory,
    read_and_validate_safetensors_index,
    read_mk1_config_data,
    validate_mk1_tokenizer,
)
from .modeling_qwen35_vla import (
    Qwen35VLAForActionGeneration,
    Qwen35VLAModel,
    build_action_expert_head,
    build_vector_encoder,
)
from .modeling_qwen36_moe import (
    DETERMINISTIC_ROUTE_REDUCTION,
    GenesisQwen36Model,
    Mk1Qwen36MoeConfig,
)

TORCH_SDPA_ATTENTION_BACKEND = "torch_sdpa_v1"
QUALIFIED_TRANSFORMERS_VERSION = "5.5.4"
QUALIFIED_TORCH_CUDA_ABI = "2.10.0+cu128"
_QUALIFIED_CUDA_VERSION = "12.8"
_QUALIFIED_CUDA_CAPABILITY = (9, 0)
_QUALIFIED_CUDA_DEVICE_PREFIX = "NVIDIA H100"
_QUALIFIED_COMMON_EXECUTION_DEFAULTS: dict[str, Any] = {
    "output_attentions": False,
    "output_hidden_states": False,
    "return_dict": True,
    "chunk_size_feed_forward": 0,
    "is_encoder_decoder": False,
}
_QUALIFIED_TEXT_EXECUTION_DEFAULTS: dict[str, Any] = {
    "use_cache": True,
    "output_router_logits": False,
    # Genesis Qwen3 routing normalizes selected top-k probabilities. The
    # custom null router owns this behavior, but retain the exported semantic
    # explicitly for config inspection and any inherited Transformers path.
    "norm_topk_prob": True,
}


class Mk1Qwen36VLAConfig(Mk1Qwen36MoeConfig):
    """MK1 backbone config with the two already-reviewed PR #3154 attachments."""

    def __init__(
        self,
        *,
        vector_max_states: int,
        action_expert: dict[str, Any],
        **kwargs: Any,
    ) -> None:
        kwargs["tie_word_embeddings"] = False
        super().__init__(**kwargs)
        self.tie_word_embeddings = False
        self.vector_max_states = int(vector_max_states)
        self.action_expert = dict(action_expert)


class Mk1Qwen36VLAModel(GenesisQwen36Model):
    """The existing VLA modality shell around the null-aware MK1 backbone."""

    config_class = Mk1Qwen36VLAConfig

    def __init__(self, config: Mk1Qwen36VLAConfig) -> None:
        super().__init__(config)
        hidden = int(config.text_config.hidden_size)
        self.vector_embedding = build_vector_encoder(config.vector_max_states, hidden)
        self.action_expert = build_action_expert_head(config.action_expert, vlm_dim=hidden)

    # These methods are the reviewed PR #3154 VLA shell. Assigning the method
    # descriptors keeps one implementation of rendering, mRoPE, masking, and
    # final-hidden semantics while changing only the backbone base class.
    _embed_text = Qwen35VLAModel._embed_text
    _embed_vector = Qwen35VLAModel._embed_vector
    _embed_vision = Qwen35VLAModel._embed_vision
    embed_stream = Qwen35VLAModel.embed_stream
    forward = Qwen35VLAModel.forward


class Mk1Qwen36VLAForActionGeneration(Qwen3_5MoePreTrainedModel):
    """MK1 backbone plus the unchanged vector encoder, lm head, and MolmoAct2 DiT."""

    config_class = Mk1Qwen36VLAConfig
    _no_split_modules = ["GenesisQwen36DecoderLayer", "Qwen3_5MoeVisionBlock", "ActionExpertBlock"]

    def __init__(self, config: Mk1Qwen36VLAConfig) -> None:
        super().__init__(config)
        self.model = Mk1Qwen36VLAModel(config)
        self.lm_head = nn.Linear(
            config.text_config.hidden_size,
            config.text_config.vocab_size,
            bias=False,
        )
        self.post_init()

    get_input_embeddings = Qwen35VLAForActionGeneration.get_input_embeddings
    action_expert = Qwen35VLAForActionGeneration.action_expert
    sample_action = Qwen35VLAForActionGeneration.sample_action
    train_forward = Qwen35VLAForActionGeneration.train_forward


def _build_config(
    raw: dict[str, Any],
    contract: Mk1CheckpointContract,
) -> Mk1Qwen36VLAConfig:
    normalized = copy.deepcopy(raw)
    for key, value in _QUALIFIED_COMMON_EXECUTION_DEFAULTS.items():
        normalized.setdefault(key, value)
    for section_name in ("text_config", "vision_config"):
        section = normalized.get(section_name)
        if not isinstance(section, dict):
            raise ValueError(f"MK1 VLA {section_name} must be a JSON object.")
        for key, value in _QUALIFIED_COMMON_EXECUTION_DEFAULTS.items():
            section.setdefault(key, value)
    for key, value in _QUALIFIED_TEXT_EXECUTION_DEFAULTS.items():
        normalized["text_config"].setdefault(key, value)

    action_expert = dict(normalized["genesis_vla"]["action_expert"])
    # Sampler steps are runtime input to sample_action(), not module geometry.
    action_expert.pop("num_inference_steps", None)
    config = Mk1Qwen36VLAConfig(
        vector_max_states=contract.vector_encoder.max_states,
        action_expert=action_expert,
        **normalized,
    )
    if config.tie_word_embeddings or config.text_config.tie_word_embeddings:
        raise ValueError("MK1 VLA requires untied input embeddings and lm_head.")
    return config


def _runtime_expected_keys(contract: Mk1CheckpointContract) -> set[str]:
    excluded = contract.excluded_runtime_prefixes
    return {
        key
        for key in contract.expected_tensor_shapes()
        if not any(key.startswith(prefix) for prefix in excluded)
    }


def _configure_attention_backend(config: Mk1Qwen36VLAConfig, attention_backend: str) -> None:
    if attention_backend != TORCH_SDPA_ATTENTION_BACKEND:
        raise ValueError(
            f"Unsupported MK1 attention backend {attention_backend!r}; "
            f"expected {TORCH_SDPA_ATTENTION_BACKEND!r}."
        )
    # This is a runtime-only selection made after the checkpoint-owned config
    # has passed validation. It is never written back into config.json.
    config._attn_implementation = "sdpa"
    config.text_config._attn_implementation = "sdpa"
    config.vision_config._attn_implementation = "sdpa"


def _configure_route_reduction(config: Mk1Qwen36VLAConfig, route_reduction: str) -> None:
    if route_reduction != DETERMINISTIC_ROUTE_REDUCTION:
        raise ValueError(
            f"Unsupported MK1 route reduction {route_reduction!r}; "
            f"expected {DETERMINISTIC_ROUTE_REDUCTION!r}."
        )
    config.text_config._mk1_route_reduction = route_reduction


def _require_qualified_transformers_abi() -> None:
    if transformers.__version__ != QUALIFIED_TRANSFORMERS_VERSION:
        raise RuntimeError(
            "MK1 native inference requires the parity-qualified Transformers ABI "
            f"{QUALIFIED_TRANSFORMERS_VERSION}, got {transformers.__version__}."
        )


def _require_qualified_torch_cuda_abi(
    contract: Mk1CheckpointContract,
    device: str | torch.device,
) -> None:
    """Fail closed on the CUDA ABI used for production parity qualification."""
    if (
        torch.device(device).type != "cuda"
        or contract.test_only_reduced_geometry
        or contract.artifact.artifact_kind == "neutral_debug"
    ):
        return

    torch_version = str(torch.__version__)
    cuda_version = torch.version.cuda
    if torch_version != QUALIFIED_TORCH_CUDA_ABI or cuda_version != _QUALIFIED_CUDA_VERSION:
        installed_abi = f"{torch_version} (CUDA {cuda_version or 'unavailable'})"
        raise RuntimeError(
            "MK1 trained_policy CUDA inference requires the parity-qualified PyTorch ABI "
            f"{QUALIFIED_TORCH_CUDA_ABI}, got {installed_abi}. Neutral debug artifacts and "
            "explicit test-only reduced geometry are not production qualification."
        )
    try:
        capability = torch.cuda.get_device_capability(device)
    except (AssertionError, RuntimeError, ValueError) as exc:
        raise RuntimeError("Cannot inspect the CUDA device for MK1 runtime qualification.") from exc
    if capability != _QUALIFIED_CUDA_CAPABILITY:
        raise RuntimeError(
            "MK1 trained_policy CUDA inference requires the parity-qualified Hopper SM90 device, "
            f"got compute capability {capability[0]}.{capability[1]}."
        )
    device_name = torch.cuda.get_device_name(device)
    if not device_name.startswith(_QUALIFIED_CUDA_DEVICE_PREFIX):
        raise RuntimeError(
            "MK1 trained_policy CUDA inference requires the parity-qualified NVIDIA H100 family, "
            f"got {device_name!r}."
        )


def _stream_checkpoint_tensors(
    model: nn.Module,
    inventory: SafetensorsInventory,
    *,
    expected_keys: set[str],
    device: str | torch.device,
    dtype: torch.dtype,
) -> None:
    loaded: set[str] = set()
    for shard in inventory.shards:
        inventory.verify_shard_identity(shard)
        with safe_open(shard, framework="pt", device="cpu") as handle:
            inventory.verify_shard_identity(shard)
            for key in sorted(handle.keys()):
                if key not in expected_keys:
                    continue
                value = handle.get_tensor(key)
                set_module_tensor_to_device(
                    model,
                    key,
                    device,
                    value=value,
                    dtype=dtype,
                    clear_cache=False,
                )
                loaded.add(key)
                del value
            inventory.verify_shard_identity(shard)
        inventory.verify_shard_identity(shard)
    missing = sorted(expected_keys - loaded)
    if missing:
        raise RuntimeError(f"MK1 streamed load missed {len(missing)} tensors; first={missing[:10]}.")


def _place_config_buffers(model: nn.Module, device: str | torch.device) -> None:
    """Move nonpersistent, config-derived buffers after empty-weight construction."""
    for name, buffer in list(model.named_buffers()):
        if buffer.device.type == "meta":
            raise RuntimeError(f"MK1 config-derived buffer {name!r} remained on meta.")
        if buffer.device != torch.device(device):
            set_module_tensor_to_device(
                model,
                name,
                device,
                value=buffer,
                dtype=buffer.dtype,
                clear_cache=False,
            )


def _select_gdn_backend(model: nn.Module, device: str | torch.device) -> None:
    """Keep optional CUDA-only GDN extensions out of CPU debug execution."""
    if torch.device(device).type == "cuda":
        return
    for module in model.modules():
        if not hasattr(module, "causal_conv1d_fn"):
            continue
        module.causal_conv1d_fn = None
        module.causal_conv1d_update = torch_causal_conv1d_update
        module.chunk_gated_delta_rule = torch_chunk_gated_delta_rule
        module.recurrent_gated_delta_rule = torch_recurrent_gated_delta_rule
        module.norm = Qwen3_5MoeRMSNormGated(module.head_v_dim, eps=module.layer_norm_epsilon)


def load_mk1_vla_from_hf(
    model_dir: str | Path,
    *,
    dtype: torch.dtype = torch.bfloat16,
    device: str | torch.device = "cpu",
    allow_test_only_reduced_geometry: bool = False,
    attention_backend: Literal["torch_sdpa_v1"] = TORCH_SDPA_ATTENTION_BACKEND,
    route_reduction: Literal["stable_token_segment_sum_v1"] = DETERMINISTIC_ROUTE_REDUCTION,
    allowed_storage_dtypes: frozenset[str] = frozenset({"BF16"}),
) -> tuple[Mk1Qwen36VLAForActionGeneration, Mk1Qwen36VLAConfig, Mk1CheckpointContract]:
    """Validate, construct once on meta, and stream an MK1 VLA using the qualified SDPA backend."""
    _require_qualified_transformers_abi()
    root = Path(model_dir)
    raw, contract = read_mk1_config_data(
        root,
        allow_test_only_reduced_geometry=allow_test_only_reduced_geometry,
    )
    _require_qualified_torch_cuda_abi(contract, device)
    inventory = read_and_validate_safetensors_index(root)
    contract.validate_tensor_inventory(inventory, allowed_storage_dtypes=allowed_storage_dtypes)
    validate_mk1_tokenizer(root, contract)
    config = _build_config(raw, contract)
    _configure_attention_backend(config, attention_backend)
    _configure_route_reduction(config, route_reduction)

    with init_empty_weights(include_buffers=False):
        model = Mk1Qwen36VLAForActionGeneration(config)
        _select_gdn_backend(model, device)
    expected_keys = _runtime_expected_keys(contract)
    model_keys = set(model.state_dict())
    if model_keys != expected_keys:
        missing = sorted(expected_keys - model_keys)
        unexpected = sorted(model_keys - expected_keys)
        raise RuntimeError(
            "MK1 model/checkpoint ABI mismatch before loading: "
            f"missing_model_keys={missing[:10]}, unexpected_model_keys={unexpected[:10]}."
        )
    if model.lm_head.weight is model.get_input_embeddings().weight:
        raise RuntimeError("MK1 VLA checkpoint requires untied lm_head and input embeddings.")

    _stream_checkpoint_tensors(
        model,
        inventory,
        expected_keys=expected_keys,
        device=device,
        dtype=dtype,
    )
    _place_config_buffers(model, device)
    meta_parameters = [
        name for name, parameter in model.named_parameters() if parameter.device.type == "meta"
    ]
    meta_buffers = [name for name, buffer in model.named_buffers() if buffer.device.type == "meta"]
    if meta_parameters or meta_buffers:
        raise RuntimeError(
            f"MK1 load left tensors on meta: parameters={meta_parameters[:10]}, buffers={meta_buffers[:10]}."
        )
    model.eval()
    return model, config, contract


__all__ = [
    "Mk1Qwen36VLAConfig",
    "Mk1Qwen36VLAForActionGeneration",
    "Mk1Qwen36VLAModel",
    "QUALIFIED_TORCH_CUDA_ABI",
    "QUALIFIED_TRANSFORMERS_VERSION",
    "TORCH_SDPA_ATTENTION_BACKEND",
    "load_mk1_vla_from_hf",
]
