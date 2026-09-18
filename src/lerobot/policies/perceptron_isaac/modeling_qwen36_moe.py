"""Native Qwen3.6 null-MoE components for the MK1 Fast backbone.

Qwen3.6 uses the Hugging Face Qwen3.5-MoE weight ABI, but Genesis extends its
router with logical null experts.  A checkpoint stores one physical null row
and expands it to many equal logical routes at runtime.  Selected null routes
skip expert compute; the remaining real routes are renormalized before the
independently gated shared expert is added.

The classes in this module deliberately reuse the Transformers attention,
GatedDeltaNet, vision, cache, RMSNorm, rotary, and fused-expert layouts.  The
only new numerical path is null-aware routing and dispatch.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Literal, NamedTuple

import torch
from torch import nn
from torch.nn import functional
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.models.qwen3_5_moe import modeling_qwen3_5_moe as qwen35_modeling
from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import (
    Qwen3_5MoeConfig,
    Qwen3_5MoeTextConfig,
)
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
    Qwen3_5MoeDecoderLayer,
    Qwen3_5MoeExperts,
    Qwen3_5MoeGatedDeltaNet,
    Qwen3_5MoeMLP,
    Qwen3_5MoeModel,
    Qwen3_5MoePreTrainedModel,
    Qwen3_5MoeRMSNorm,
    Qwen3_5MoeTextModel,
    Qwen3_5MoeVisionModel,
)

_NULL_EXPERT_SEMANTICS = "skip_compute_renormalize_real_routes"
_SIGMOID_GATED_SHARED_EXPERT = "sigmoid_gated_additive"
_ROUTER_CONTRACT_VERSION = 1
DETERMINISTIC_ROUTE_REDUCTION = "stable_token_segment_sum_v1"
GENESIS_ROTARY_PRECISION = "checkpoint_bf16_inv_freq_fp32_phase_v1"


class _ExplicitRepeatKvSdpaResult(NamedTuple):
    output: torch.Tensor
    repeated_key: torch.Tensor
    repeated_value: torch.Tensor
    raw_output: torch.Tensor


def _genesis_layout_repeat_kv(hidden_states: torch.Tensor, num_repetitions: int) -> torch.Tensor:
    """Repeat KV heads in Genesis' [batch, length, heads, dim] layout."""
    batch, sequence_length, num_key_value_heads, head_dim = hidden_states.shape
    if num_repetitions == 1:
        return hidden_states
    return (
        hidden_states.unsqueeze(3)
        .expand(batch, sequence_length, num_key_value_heads, num_repetitions, head_dim)
        .reshape(batch, sequence_length, num_key_value_heads * num_repetitions, head_dim)
    )


def _sdpa_with_repeated_kv(
    module: nn.Module,
    query: torch.Tensor,
    repeated_key: torch.Tensor,
    repeated_value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    *,
    dropout: float,
    scaling: float | None,
    is_causal: bool | None = None,
) -> _ExplicitRepeatKvSdpaResult:
    causal = is_causal if is_causal is not None else bool(getattr(module, "is_causal", True))
    causal = query.shape[2] > 1 and attention_mask is None and causal
    if torch.jit.is_tracing() and isinstance(causal, torch.Tensor):
        causal = bool(causal.item())
    raw_output = functional.scaled_dot_product_attention(
        query,
        repeated_key,
        repeated_value,
        attn_mask=attention_mask,
        dropout_p=dropout,
        scale=scaling,
        is_causal=causal,
    )
    return _ExplicitRepeatKvSdpaResult(
        output=raw_output.transpose(1, 2).contiguous(),
        repeated_key=repeated_key,
        repeated_value=repeated_value,
        raw_output=raw_output,
    )


def _explicit_repeat_kv_sdpa_components(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    *,
    dropout: float = 0.0,
    scaling: float | None = None,
    is_causal: bool | None = None,
) -> _ExplicitRepeatKvSdpaResult:
    num_key_value_groups = getattr(module, "num_key_value_groups", None)
    if not isinstance(num_key_value_groups, int) or num_key_value_groups <= 0:
        raise ValueError("MK1 SDPA requires a positive integer num_key_value_groups")
    repeated_key = qwen35_modeling.repeat_kv(key, num_key_value_groups)
    repeated_value = qwen35_modeling.repeat_kv(value, num_key_value_groups)
    return _sdpa_with_repeated_kv(
        module,
        query,
        repeated_key=repeated_key,
        repeated_value=repeated_value,
        attention_mask=attention_mask,
        dropout=dropout,
        scaling=scaling,
        is_causal=is_causal,
    )


def explicit_repeat_kv_sdpa(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    *,
    dropout: float = 0.0,
    scaling: float | None = None,
    is_causal: bool | None = None,
) -> torch.Tensor:
    """Run the qualified MK1 SDPA path after materializing grouped KV heads."""
    return _explicit_repeat_kv_sdpa_components(
        module,
        query,
        key,
        value,
        attention_mask,
        dropout=dropout,
        scaling=scaling,
        is_causal=is_causal,
    ).output


def deterministic_token_segment_sum(
    base_output: torch.Tensor,
    token_indices: torch.Tensor,
    routed_output: torch.Tensor,
) -> torch.Tensor:
    """Add routed rows by token without repeated-index CUDA atomics."""
    if token_indices.ndim != 1 or routed_output.ndim != 2 or base_output.ndim != 2:
        raise ValueError("deterministic route reduction requires base [N,D], indices [R], and routes [R,D]")
    if routed_output.shape != (token_indices.numel(), base_output.shape[1]):
        raise ValueError("route rows and token indices must match the base output width")
    if token_indices.numel() == 0:
        return base_output.clone()
    if token_indices.dtype != torch.long or token_indices.device != base_output.device:
        raise ValueError("route token indices must be int64 on the base output device")
    if routed_output.device != base_output.device or routed_output.dtype != base_output.dtype:
        raise ValueError("route rows must match the base output device and dtype")

    order = torch.argsort(token_indices, stable=True)
    sorted_indices = token_indices[order]
    sorted_routes = routed_output[order]
    unique_indices, counts = torch.unique_consecutive(sorted_indices, return_counts=True)
    reduced = torch.segment_reduce(sorted_routes, "sum", lengths=counts)
    output = base_output.clone()
    output[unique_indices] = output[unique_indices] + reduced
    return output


def _required_metadata_value(
    metadata: Mapping[str, Any],
    key: str,
    expected_type: type | tuple[type, ...],
) -> Any:
    value = metadata.get(key)
    if not isinstance(value, expected_type) or (isinstance(value, bool) and expected_type is not bool):
        raise ValueError(f"genesis_moe.{key} has invalid value {value!r}")
    return value


@dataclass(frozen=True, slots=True)
class GenesisNullMoeContract:
    """Validated, artifact-owned null-routing geometry and behavior."""

    num_real_experts: int
    num_null_experts: int
    top_k: int
    route_scale: float

    @property
    def physical_router_outputs(self) -> int:
        return self.num_real_experts + 1

    @property
    def logical_router_outputs(self) -> int:
        return self.num_real_experts + self.num_null_experts

    @classmethod
    def from_metadata(cls, metadata: Mapping[str, Any]) -> GenesisNullMoeContract:
        """Parse the versioned Genesis metadata, rejecting unsupported variants."""

        version = _required_metadata_value(metadata, "router_contract_version", int)
        if version != _ROUTER_CONTRACT_VERSION:
            raise ValueError(
                f"genesis_moe.router_contract_version must be {_ROUTER_CONTRACT_VERSION}, got {version}"
            )

        num_real = _required_metadata_value(metadata, "num_real_experts", int)
        num_null = _required_metadata_value(metadata, "num_null_experts", int)
        top_k = _required_metadata_value(metadata, "top_k", int)
        route_scale = float(_required_metadata_value(metadata, "route_scale", (int, float)))
        if num_real <= 0 or num_null <= 0:
            raise ValueError("genesis_moe requires positive real and null expert counts")
        if top_k <= 0 or top_k > num_real + num_null:
            raise ValueError(f"genesis_moe.top_k must be in [1, {num_real + num_null}], got {top_k}")
        if not math.isfinite(route_scale) or route_scale <= 0:
            raise ValueError(f"genesis_moe.route_scale must be finite and positive, got {route_scale}")

        contract = cls(
            num_real_experts=num_real,
            num_null_experts=num_null,
            top_k=top_k,
            route_scale=route_scale,
        )
        exact_values: dict[str, object] = {
            "router_contract": [num_real, num_null],
            "physical_router_outputs": contract.physical_router_outputs,
            "logical_router_outputs": contract.logical_router_outputs,
            "shared_null_router_row": True,
            "score_func": "softmax",
            "route_norm": True,
            "score_before_experts": False,
            "null_expert_semantics": _NULL_EXPERT_SEMANTICS,
            "shared_expert_mode": _SIGMOID_GATED_SHARED_EXPERT,
            "uses_expert_bias": False,
        }
        for key, expected in exact_values.items():
            actual = metadata.get(key)
            if actual != expected:
                raise ValueError(f"genesis_moe.{key} must be {expected!r}, got {actual!r}")
        return contract

    @classmethod
    def from_text_config(cls, config: Qwen3_5MoeTextConfig) -> GenesisNullMoeContract:
        metadata = getattr(config, "genesis_moe", None)
        if not isinstance(metadata, Mapping):
            raise ValueError("MK1 Qwen3.6 text_config requires a genesis_moe object")
        contract = cls.from_metadata(metadata)
        if config.num_experts != contract.num_real_experts:
            raise ValueError(
                "text_config.num_experts must match genesis_moe.num_real_experts: "
                f"{config.num_experts} != {contract.num_real_experts}"
            )
        if config.num_experts_per_tok != contract.top_k:
            raise ValueError(
                "text_config.num_experts_per_tok must match genesis_moe.top_k: "
                f"{config.num_experts_per_tok} != {contract.top_k}"
            )
        if config.shared_expert_intermediate_size != config.moe_intermediate_size:
            raise ValueError(
                "MK1 Qwen3.6 requires exactly one shared expert: "
                "shared_expert_intermediate_size must equal moe_intermediate_size"
            )
        return contract


class Mk1Qwen36MoeConfig(Qwen3_5MoeConfig):
    """Qwen3.5-MoE ABI config that requires the complete MK1 null contract."""

    def __post_init__(self, **kwargs: Any) -> None:
        super().__post_init__(**kwargs)
        root_metadata = getattr(self, "genesis_moe", None)
        text_metadata = getattr(self.text_config, "genesis_moe", None)
        if not isinstance(root_metadata, Mapping) or not isinstance(text_metadata, Mapping):
            raise ValueError("MK1 Qwen3.6 config requires matching root and text_config genesis_moe objects")
        if dict(root_metadata) != dict(text_metadata):
            raise ValueError("MK1 Qwen3.6 root and text_config genesis_moe objects must be identical")
        GenesisNullMoeContract.from_text_config(self.text_config)


class GenesisNullRoutingOutput(NamedTuple):
    """Router values retained for numerical parity tests and compact dispatch."""

    physical_logits: torch.Tensor
    logical_probabilities: torch.Tensor
    selected_experts: torch.Tensor
    real_route_weights: torch.Tensor


class GenesisNullTopKRouter(nn.Linear):
    """Compact physical projection with deterministic logical null routing.

    Subclassing ``nn.Linear`` preserves the exported ``gate.weight`` state key.
    ``_router_contract`` is nonpersistent because the Genesis exporter consumes
    the DCP buffer and records it in ``config.json`` instead of the HF shards.
    """

    def __init__(self, config: Qwen3_5MoeTextConfig) -> None:
        contract = GenesisNullMoeContract.from_text_config(config)
        super().__init__(
            config.hidden_size,
            contract.physical_router_outputs,
            bias=False,
        )
        self.contract = contract
        self.top_k = contract.top_k
        self.num_experts = contract.num_real_experts
        self.num_null_experts = contract.num_null_experts
        self.num_logical_experts = contract.logical_router_outputs
        self.register_buffer(
            "_router_contract",
            torch.tensor([self.num_experts, self.num_null_experts], dtype=torch.int64),
            persistent=False,
        )

    def route(self, hidden_states: torch.Tensor) -> GenesisNullRoutingOutput:
        hidden_states = hidden_states.reshape(-1, self.in_features)
        physical_logits = functional.linear(hidden_states, self.weight)
        real_logits = physical_logits[:, : self.num_experts]
        shared_null_logit = physical_logits[:, self.num_experts :]
        logical_logits = torch.cat(
            [real_logits, shared_null_logit.expand(-1, self.num_null_experts)],
            dim=-1,
        )

        # Genesis performs scoring in fp32.
        logical_probabilities = functional.softmax(logical_logits, dtype=torch.float32, dim=-1)
        # The logical tensor already exists for the router ABI and auxiliary
        # outputs.  One stable sort is faster than two candidate sorts at the
        # production 256-real/256-null geometry while retaining exact Genesis
        # tie behavior.
        selected_experts = self._expanded_stable_topk(logical_probabilities)
        selected_probabilities = logical_probabilities.gather(-1, selected_experts)

        real_route_mask = selected_experts < self.num_experts
        selected_real_probabilities = torch.where(
            real_route_mask,
            selected_probabilities,
            torch.zeros_like(selected_probabilities),
        )
        real_mass = selected_real_probabilities.sum(dim=-1, keepdim=True)
        # Avoid a hidden 0/0 branch in autograd.  All-null tokens stay exactly zero.
        safe_real_mass = torch.where(real_mass > 0, real_mass, torch.ones_like(real_mass))
        real_route_weights = selected_real_probabilities / safe_real_mass
        real_route_weights = real_route_weights * self.contract.route_scale

        return GenesisNullRoutingOutput(
            physical_logits=physical_logits,
            logical_probabilities=logical_probabilities,
            selected_experts=selected_experts,
            real_route_weights=real_route_weights,
        )

    def _expanded_stable_topk(self, logical_probabilities: torch.Tensor) -> torch.Tensor:
        """Exact logical-slot selector used by the runtime and parity tests."""

        return torch.argsort(
            logical_probabilities,
            dim=-1,
            descending=True,
            stable=True,
        )[:, : self.top_k].contiguous()

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return the same tuple shape as the stock Qwen3.5-MoE router."""

        routing = self.route(hidden_states)
        return routing.logical_probabilities, routing.real_route_weights, routing.selected_experts


class GenesisNullMoeBranches(NamedTuple):
    """Separated routed/shared branches used by the parity ladder."""

    output: torch.Tensor
    routed_output: torch.Tensor
    shared_expert_output: torch.Tensor
    routing: GenesisNullRoutingOutput


class _CompactRealRoutes(NamedTuple):
    token_indices: torch.Tensor
    expert_indices: torch.Tensor
    expert_scores: torch.Tensor
    offsets: torch.Tensor


class GenesisNullDispatchTrace(NamedTuple):
    """Exact route values at the two dispatch normalization boundaries."""

    logical_probabilities: torch.Tensor
    selected_experts: torch.Tensor
    selected_route_weights: torch.Tensor
    dispatch_token_indices: torch.Tensor
    dispatch_expert_indices: torch.Tensor
    dispatch_route_weights: torch.Tensor


class GenesisNullSparseMoeBlock(nn.Module):
    """Qwen fused real experts with Genesis null-route dropping."""

    def __init__(self, config: Qwen3_5MoeTextConfig) -> None:
        super().__init__()
        self.contract = GenesisNullMoeContract.from_text_config(config)
        self.gate = GenesisNullTopKRouter(config)
        self.experts = Qwen3_5MoeExperts(config)
        self.shared_expert = Qwen3_5MoeMLP(
            config,
            intermediate_size=config.shared_expert_intermediate_size,
        )
        self.shared_expert_gate = nn.Linear(config.hidden_size, 1, bias=False)
        route_reduction = getattr(config, "_mk1_route_reduction", DETERMINISTIC_ROUTE_REDUCTION)
        if route_reduction != DETERMINISTIC_ROUTE_REDUCTION:
            raise ValueError(
                f"Unsupported MK1 route reduction {route_reduction!r}; "
                f"expected {DETERMINISTIC_ROUTE_REDUCTION!r}."
            )
        # Grouped GEMMs remain unchanged; only repeated-token accumulation uses
        # the stable segment reducer qualified by the Step-0 parity workflow.
        self.deterministic_route_reduction = True
        self.dispatch_trace_observer: Callable[[GenesisNullDispatchTrace], None] | None = None

    @property
    def _is_production_geometry(self) -> bool:
        return (
            self.contract.num_real_experts == 256
            and self.contract.num_null_experts == 256
            and self.contract.top_k == 8
        )

    @staticmethod
    def _grouped_mm_operator() -> Any | None:
        if hasattr(functional, "grouped_mm"):
            return functional.grouped_mm
        return getattr(torch, "_grouped_mm", None)

    def _supports_grouped_mm(self, hidden_states: torch.Tensor) -> bool:
        if hidden_states.device.type != "cuda" or hidden_states.dtype is not torch.bfloat16:
            return False
        if (
            self.experts.gate_up_proj.device != hidden_states.device
            or self.experts.down_proj.device != hidden_states.device
        ):
            return False
        if (
            self.experts.gate_up_proj.dtype is not torch.bfloat16
            or self.experts.down_proj.dtype is not torch.bfloat16
        ):
            return False
        # CUDA grouped GEMM requires row strides aligned to 16 bytes for BF16.
        if hidden_states.shape[-1] % 8 != 0 or self.experts.intermediate_dim % 8 != 0:
            return False
        if self._grouped_mm_operator() is None:
            return False
        major, _ = torch.cuda.get_device_capability(hidden_states.device)
        return major >= 8

    def dispatch_backend(self, hidden_states: torch.Tensor) -> Literal["grouped_mm", "eager"]:
        """Keep production inference grouped; allow FP32-storage eager training."""

        if self._supports_grouped_mm(hidden_states):
            return "grouped_mm"
        if self._is_production_geometry and hidden_states.device.type == "cuda":
            # Check module mode, not grad mode: checkpointed training can run under no_grad.
            # BF16 activations need BF16 autocast for the FP32-weight eager linears.
            if (
                self.training
                and self.experts.gate_up_proj.dtype is torch.float32
                and self.experts.down_proj.dtype is torch.float32
                and self.experts.gate_up_proj.device == hidden_states.device
                and self.experts.down_proj.device == hidden_states.device
                and (
                    hidden_states.dtype is torch.float32
                    or (
                        hidden_states.dtype is torch.bfloat16
                        and torch.is_autocast_enabled(hidden_states.device.type)
                        and torch.get_autocast_dtype(hidden_states.device.type) is torch.bfloat16
                    )
                )
            ):
                return "eager"
            raise RuntimeError(
                "Production MK1 null-MoE CUDA inference requires BF16 torch grouped_mm on SM80 or newer; "
                f"got dtype={hidden_states.dtype}, capability={torch.cuda.get_device_capability(hidden_states.device)}"
            )
        return "eager"

    def _compact_real_routes(
        self,
        hidden_states: torch.Tensor,
        routing: GenesisNullRoutingOutput,
    ) -> _CompactRealRoutes:
        num_tokens, top_k = routing.selected_experts.shape
        token_indices = torch.arange(num_tokens, device=hidden_states.device).unsqueeze(1).expand(-1, top_k)
        real_mask = routing.selected_experts < self.contract.num_real_experts
        expert_indices = routing.selected_experts[real_mask]
        token_indices = token_indices[real_mask]

        # Reproduce Genesis's two normalization stages and their operation
        # order.  The first normalizes all selected logical routes; after null
        # routes are removed, the second renormalizes the real-route prefix.
        selected_probabilities = routing.logical_probabilities.gather(-1, routing.selected_experts)
        selected_scores = selected_probabilities / (selected_probabilities.sum(dim=-1, keepdim=True) + 1e-20)
        selected_scores = selected_scores * self.contract.route_scale
        expert_scores = selected_scores[real_mask]

        order = torch.argsort(expert_indices, stable=True)
        expert_indices = expert_indices[order]
        token_indices = token_indices[order]
        expert_scores = expert_scores[order]
        counts = torch.bincount(expert_indices, minlength=self.contract.num_real_experts)
        offsets = torch.cumsum(counts, dim=0, dtype=torch.int32)

        unscaled_scores = expert_scores / self.contract.route_scale
        if self.deterministic_route_reduction:
            selected_unscaled_scores = selected_scores / self.contract.route_scale
            real_mass = torch.where(
                real_mask,
                selected_unscaled_scores,
                torch.zeros_like(selected_unscaled_scores),
            ).sum(dim=-1)
        else:
            real_mass = unscaled_scores.new_zeros(num_tokens)
            real_mass.index_add_(0, token_indices, unscaled_scores)
        expert_scores = (
            unscaled_scores / real_mass.clamp_min(1e-6)[token_indices]
        ) * self.contract.route_scale
        routes = _CompactRealRoutes(token_indices, expert_indices, expert_scores, offsets)
        if self.dispatch_trace_observer is not None:
            self.dispatch_trace_observer(
                GenesisNullDispatchTrace(
                    logical_probabilities=routing.logical_probabilities,
                    selected_experts=routing.selected_experts,
                    selected_route_weights=selected_scores,
                    dispatch_token_indices=routes.token_indices,
                    dispatch_expert_indices=routes.expert_indices,
                    dispatch_route_weights=routes.expert_scores,
                )
            )
        return routes

    def _dispatch_real_experts_eager(
        self,
        hidden_states: torch.Tensor,
        routes: _CompactRealRoutes,
        shared_output: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        routed_output = torch.zeros_like(hidden_states)
        # Autocast may make the shared branch BF16 while route accumulation is FP32.
        combined_output = shared_output.to(dtype=routed_output.dtype, copy=True)
        start = 0
        deterministic_token_indices: list[torch.Tensor] = []
        deterministic_outputs: list[torch.Tensor] = []
        # One device-to-host transfer avoids one synchronization per active expert.
        for expert_index, end in enumerate(routes.offsets.tolist()):
            if start == end:
                continue
            token_indices = routes.token_indices[start:end]
            current_state = hidden_states[token_indices]
            gate, up = functional.linear(current_state, self.experts.gate_up_proj[expert_index]).chunk(
                2, dim=-1
            )
            swiglu = self.experts.act_fn(gate) * up

            # Match Genesis BF16 operation order: score after SwiGLU and before
            # this expert's down projection.
            expert_scores = routes.expert_scores[start:end]
            weighted_swiglu = (swiglu.float() * expert_scores[:, None]).to(swiglu.dtype)
            current_output = functional.linear(weighted_swiglu, self.experts.down_proj[expert_index])
            current_output = current_output.to(routed_output.dtype)
            if self.deterministic_route_reduction:
                deterministic_token_indices.append(token_indices)
                deterministic_outputs.append(current_output)
            else:
                routed_output.index_add_(0, token_indices, current_output)
                # Genesis scatters route rows directly into the shared-expert base.
                # Keeping that accumulation order avoids an extra BF16 rounding step.
                combined_output.index_add_(0, token_indices, current_output)
            start = end
        if deterministic_outputs:
            all_token_indices = torch.cat(deterministic_token_indices)
            all_outputs = torch.cat(deterministic_outputs)
            routed_output = deterministic_token_segment_sum(routed_output, all_token_indices, all_outputs)
            combined_output = deterministic_token_segment_sum(combined_output, all_token_indices, all_outputs)
        return routed_output, combined_output

    def _dispatch_real_experts_grouped_mm(
        self,
        hidden_states: torch.Tensor,
        routes: _CompactRealRoutes,
        shared_output: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        routed_output = torch.zeros_like(hidden_states)
        combined_output = shared_output.clone()
        if routes.token_indices.numel() == 0:
            return routed_output, combined_output

        grouped_mm = self._grouped_mm_operator()
        if grouped_mm is None:  # pragma: no cover - guarded by dispatch_backend
            raise RuntimeError("torch grouped_mm disappeared after backend selection")
        routed_input = hidden_states[routes.token_indices]
        gate_up = grouped_mm(
            routed_input,
            self.experts.gate_up_proj.transpose(-2, -1),
            offs=routes.offsets,
        )
        gate, up = gate_up.chunk(2, dim=-1)
        swiglu = self.experts.act_fn(gate) * up
        weighted_swiglu = (swiglu.float() * routes.expert_scores[:, None]).to(swiglu.dtype)
        current_output = grouped_mm(
            weighted_swiglu,
            self.experts.down_proj.transpose(-2, -1),
            offs=routes.offsets,
        ).to(routed_output.dtype)
        if self.deterministic_route_reduction:
            routed_output = deterministic_token_segment_sum(
                routed_output, routes.token_indices, current_output
            )
            combined_output = deterministic_token_segment_sum(
                combined_output, routes.token_indices, current_output
            )
        else:
            routed_output.index_add_(0, routes.token_indices, current_output)
            combined_output.index_add_(0, routes.token_indices, current_output)
        return routed_output, combined_output

    def _dispatch_real_experts(
        self,
        hidden_states: torch.Tensor,
        routing: GenesisNullRoutingOutput,
        shared_output: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        routes = self._compact_real_routes(hidden_states, routing)
        if self.dispatch_backend(hidden_states) == "grouped_mm":
            return self._dispatch_real_experts_grouped_mm(hidden_states, routes, shared_output)
        return self._dispatch_real_experts_eager(hidden_states, routes, shared_output)

    def _shared_expert_output(self, hidden_states: torch.Tensor) -> torch.Tensor:
        shared_output = self.shared_expert(hidden_states)
        gate_input = hidden_states.to(self.shared_expert_gate.weight.dtype)
        shared_gate = torch.sigmoid(self.shared_expert_gate(gate_input).float())
        return shared_output * shared_gate.to(shared_output.dtype)

    def forward_with_branches(self, hidden_states: torch.Tensor) -> GenesisNullMoeBranches:
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        flattened = hidden_states.reshape(-1, hidden_dim)
        routing = self.gate.route(flattened)
        shared_output = self._shared_expert_output(flattened)
        routed_output, combined_output = self._dispatch_real_experts(flattened, routing, shared_output)
        output = combined_output.reshape(batch_size, sequence_length, hidden_dim)
        return GenesisNullMoeBranches(
            output=output,
            routed_output=routed_output.reshape(batch_size, sequence_length, hidden_dim),
            shared_expert_output=shared_output.reshape(batch_size, sequence_length, hidden_dim),
            routing=routing,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.forward_with_branches(hidden_states).output


class _DeviceAwareRMSNormGated(qwen35_modeling.Qwen3_5MoeRMSNormGated):
    """One state-compatible norm that dispatches by its input device."""

    def __init__(self, hidden_size: int, *, eps: float, activation: str) -> None:
        super().__init__(hidden_size, eps=eps)
        self.eps = eps
        self.activation = activation
        self.register_parameter("bias", None)

    def forward(self, hidden_states: torch.Tensor, gate: torch.Tensor | None = None) -> torch.Tensor:
        if gate is None:
            raise ValueError("MK1 GatedDeltaNet normalization requires a gate tensor")
        if hidden_states.device.type == "cuda" and qwen35_modeling.FusedRMSNormGated is not None:
            # The FLA module's forward only depends on the attributes defined
            # above. Calling it unbound keeps one norm.weight state key while
            # retaining the exact stock CUDA kernel.
            return qwen35_modeling.FusedRMSNormGated.forward(self, hidden_states, gate)
        return super().forward(hidden_states, gate)


class GenesisQwen36GatedDeltaNet(Qwen3_5MoeGatedDeltaNet):
    """Stock GatedDeltaNet ABI with input-device-aware kernel dispatch.

    Transformers selects optional FLA and causal-conv kernels at import and
    construction time. When those packages are installed, the stock module
    consequently creates a CUDA norm even inside a CPU or meta construction
    context and later sends CPU tensors to CUDA-only functions. This subclass
    retains the stock forward implementation and parameter names, but binds
    thin dispatchers that select the same stock fast kernels only for CUDA
    inputs and the same stock Torch fallbacks otherwise.
    """

    def __init__(self, config: Qwen3_5MoeTextConfig, layer_idx: int) -> None:
        # Qwen3_5MoeGatedDeltaNet.__init__ explicitly places the optional fused
        # norm on the current CUDA device, which breaks CPU and meta contexts.
        nn.Module.__init__(self)
        self.hidden_size = config.hidden_size
        self.num_v_heads = config.linear_num_value_heads
        self.num_k_heads = config.linear_num_key_heads
        self.head_k_dim = config.linear_key_head_dim
        self.head_v_dim = config.linear_value_head_dim
        self.key_dim = self.head_k_dim * self.num_k_heads
        self.value_dim = self.head_v_dim * self.num_v_heads

        self.conv_kernel_size = config.linear_conv_kernel_dim
        self.layer_idx = layer_idx
        self.activation = config.hidden_act
        self.act = qwen35_modeling.ACT2FN[config.hidden_act]
        self.layer_norm_epsilon = config.rms_norm_eps
        self.conv_dim = self.key_dim * 2 + self.value_dim
        self.conv1d = nn.Conv1d(
            in_channels=self.conv_dim,
            out_channels=self.conv_dim,
            bias=False,
            kernel_size=self.conv_kernel_size,
            groups=self.conv_dim,
            padding=self.conv_kernel_size - 1,
        )
        self.dt_bias = nn.Parameter(torch.ones(self.num_v_heads))
        self.A_log = nn.Parameter(torch.log(torch.empty(self.num_v_heads).uniform_(0, 16)))
        self.norm = _DeviceAwareRMSNormGated(
            self.head_v_dim,
            eps=self.layer_norm_epsilon,
            activation=self.activation,
        )
        self.out_proj = nn.Linear(self.value_dim, self.hidden_size, bias=False)

        # The inherited stock forward calls these attributes. The wrappers do
        # not alter arguments, cache updates, masks, or fast-kernel behavior.
        self.causal_conv1d_fn = self._causal_conv1d
        self.causal_conv1d_update = self._causal_conv1d_update
        self.chunk_gated_delta_rule = self._chunk_gated_delta_rule
        self.recurrent_gated_delta_rule = self._recurrent_gated_delta_rule

        self.in_proj_qkv = nn.Linear(self.hidden_size, self.conv_dim, bias=False)
        self.in_proj_z = nn.Linear(self.hidden_size, self.value_dim, bias=False)
        self.in_proj_b = nn.Linear(self.hidden_size, self.num_v_heads, bias=False)
        self.in_proj_a = nn.Linear(self.hidden_size, self.num_v_heads, bias=False)

    def _causal_conv1d(
        self,
        *,
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None,
        activation: str,
        seq_idx: torch.Tensor | None,
    ) -> torch.Tensor:
        if x.device.type == "cuda" and qwen35_modeling.causal_conv1d_fn is not None:
            return qwen35_modeling.causal_conv1d_fn(
                x=x,
                weight=weight,
                bias=bias,
                activation=activation,
                seq_idx=seq_idx,
            )
        output = functional.conv1d(
            x,
            weight.unsqueeze(1),
            bias,
            padding=self.conv_kernel_size - 1,
            groups=self.conv_dim,
        )
        return functional.silu(output[:, :, : x.shape[-1]])

    @staticmethod
    def _causal_conv1d_update(
        hidden_states: torch.Tensor,
        conv_state: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None = None,
        activation: str | None = None,
    ) -> torch.Tensor:
        if hidden_states.device.type == "cuda" and qwen35_modeling.causal_conv1d_update is not None:
            return qwen35_modeling.causal_conv1d_update(
                hidden_states,
                conv_state,
                weight,
                bias,
                activation,
            )
        return qwen35_modeling.torch_causal_conv1d_update(
            hidden_states,
            conv_state,
            weight,
            bias,
            activation,
        )

    @staticmethod
    def _chunk_gated_delta_rule(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if query.device.type == "cuda" and qwen35_modeling.chunk_gated_delta_rule is not None:
            return qwen35_modeling.chunk_gated_delta_rule(query, key, value, **kwargs)
        return qwen35_modeling.torch_chunk_gated_delta_rule(query, key, value, **kwargs)

    @staticmethod
    def _recurrent_gated_delta_rule(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if query.device.type == "cuda" and qwen35_modeling.fused_recurrent_gated_delta_rule is not None:
            return qwen35_modeling.fused_recurrent_gated_delta_rule(query, key, value, **kwargs)
        return qwen35_modeling.torch_recurrent_gated_delta_rule(query, key, value, **kwargs)


class GenesisQwen36AttentionTrace(NamedTuple):
    """Non-mutating component boundaries for cross-runtime attention parity."""

    query_projection: torch.Tensor
    query_gate: torch.Tensor
    query_norm: torch.Tensor
    key_projection: torch.Tensor
    key_norm: torch.Tensor
    value_projection: torch.Tensor
    rotary_query: torch.Tensor
    rotary_key: torch.Tensor
    sdpa_query: torch.Tensor
    repeated_key: torch.Tensor
    repeated_value: torch.Tensor
    raw_sdpa_output: torch.Tensor
    post_transpose_output: torch.Tensor
    gated_output: torch.Tensor
    token_mixer: torch.Tensor


class GenesisQwen36Attention(qwen35_modeling.Qwen3_5MoeAttention):
    """Qwen3.6 text attention using the Genesis-qualified explicit-KV SDPA path."""

    def __init__(self, config: Qwen3_5MoeTextConfig, layer_idx: int) -> None:
        super().__init__(config, layer_idx)
        self.trace_observer: Callable[[GenesisQwen36AttentionTrace], None] | None = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
        past_key_values: Any | None = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, None]:
        del kwargs
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states, gate = torch.chunk(
            self.q_proj(hidden_states).view(*input_shape, -1, self.head_dim * 2),
            2,
            dim=-1,
        )
        gate = gate.reshape(*input_shape, -1)

        query_projection = query_states.view(hidden_shape)
        key_projection = self.k_proj(hidden_states).view(hidden_shape)
        value_projection = self.v_proj(hidden_states).view(hidden_shape)
        query_norm = self.q_norm(query_projection)
        key_norm = self.k_norm(key_projection)
        cos, sin = position_embeddings
        rotary_query, rotary_key = qwen35_modeling.apply_rotary_pos_emb(
            query_norm,
            key_norm,
            cos,
            sin,
            unsqueeze_dim=2,
        )
        sdpa_query = rotary_query.transpose(1, 2)
        if past_key_values is not None:
            cached_key, cached_value = past_key_values.update(
                rotary_key.transpose(1, 2),
                value_projection.transpose(1, 2),
                self.layer_idx,
            )
            repeated_key = qwen35_modeling.repeat_kv(cached_key, self.num_key_value_groups)
            repeated_value = qwen35_modeling.repeat_kv(cached_value, self.num_key_value_groups)
        else:
            repeated_key = _genesis_layout_repeat_kv(
                rotary_key,
                self.num_key_value_groups,
            ).transpose(1, 2)
            repeated_value = _genesis_layout_repeat_kv(
                value_projection,
                self.num_key_value_groups,
            ).transpose(1, 2)

        sdpa = _sdpa_with_repeated_kv(
            self,
            sdpa_query,
            repeated_key,
            repeated_value,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
        )
        post_transpose_output = sdpa.output
        attention_output = post_transpose_output.reshape(*input_shape, -1).contiguous()
        gated_output = attention_output * torch.sigmoid(gate)
        token_mixer = self.o_proj(gated_output)
        if self.trace_observer is not None:
            self.trace_observer(
                GenesisQwen36AttentionTrace(
                    query_projection=query_projection,
                    query_gate=gate,
                    query_norm=query_norm,
                    key_projection=key_projection,
                    key_norm=key_norm,
                    value_projection=value_projection,
                    rotary_query=rotary_query,
                    rotary_key=rotary_key,
                    sdpa_query=sdpa_query,
                    repeated_key=sdpa.repeated_key,
                    repeated_value=sdpa.repeated_value,
                    raw_sdpa_output=sdpa.raw_output,
                    post_transpose_output=post_transpose_output,
                    gated_output=gated_output,
                    token_mixer=token_mixer,
                )
            )
        return token_mixer, None


class GenesisQwen36TextRotaryEmbedding(nn.Module):
    """Genesis mRoPE with model-dtype inverse-frequency quantization.

    Genesis registers ``inv_freq`` on the backbone before casting the complete
    model to its inference dtype. Its phase calculation then promotes that
    already-quantized table back to FP32. Keeping this derived buffer
    nonpersistent preserves the HF checkpoint tensor ABI, while the loader
    places it using the same dtype as the checkpoint parameters.
    """

    precision_contract = GENESIS_ROTARY_PRECISION

    def __init__(self, config: Qwen3_5MoeTextConfig) -> None:
        super().__init__()
        rope_parameters = config.rope_parameters
        if rope_parameters.get("rope_type") != "default":
            raise ValueError("MK1 Genesis rotary precision only supports default RoPE")
        head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
        rotary_dim = int(head_dim * rope_parameters.get("partial_rotary_factor", 1.0))
        if rotary_dim <= 0 or rotary_dim % 2 != 0:
            raise ValueError(f"MK1 rotary dimension must be positive and even, got {rotary_dim}")
        theta = float(rope_parameters["rope_theta"])
        inv_freq = 1.0 / (theta ** (torch.arange(0, rotary_dim, 2, dtype=torch.float32) / rotary_dim))
        # The trained Genesis checkpoint owns this BF16 quantization regardless
        # of an optional FP32 debug runtime requested by the caller.
        self.register_buffer("inv_freq", inv_freq.to(torch.bfloat16), persistent=False)
        self.mrope_section = list(rope_parameters.get("mrope_section", [11, 11, 10]))
        if sum(self.mrope_section) != rotary_dim // 2:
            raise ValueError(
                "MK1 mrope_section must sum to half the rotary dimension: "
                f"sum={sum(self.mrope_section)}, rotary_dim={rotary_dim}"
            )

    @torch.no_grad()
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if position_ids.ndim == 2:
            position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)
        if position_ids.ndim != 3 or position_ids.shape[0] != 3:
            raise ValueError(
                f"MK1 position_ids must have shape [3, batch, length], got {tuple(position_ids.shape)}"
            )

        # Match Genesis precompute_cos_sin_3d exactly: the model-dtype table is
        # promoted to FP32 before phase construction, then trig results are cast
        # back to the activation dtype before rotary multiplication.
        inv_freq = self.inv_freq.to(device=position_ids.device, dtype=torch.float32)
        phases = position_ids.float().unsqueeze(-1) * inv_freq.view(1, 1, 1, -1)
        interleaved = phases[0].clone()
        for axis, offset in enumerate((1, 2), start=1):
            length = int(self.mrope_section[axis]) * 3
            interleaved[..., slice(offset, length, 3)] = phases[axis, ..., slice(offset, length, 3)]
        embedding = torch.cat((interleaved, interleaved), dim=-1)
        return embedding.cos().to(hidden_states.dtype), embedding.sin().to(hidden_states.dtype)


class GenesisQwen36VisionAttention(qwen35_modeling.Qwen3_5MoeVisionAttention):
    """Packed vision attention using the same explicit-KV SDPA primitive."""

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rotary_pos_emb: torch.Tensor | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        del rotary_pos_emb, kwargs
        if position_embeddings is None:
            raise ValueError("MK1 vision attention requires precomputed position embeddings")
        sequence_length = hidden_states.shape[0]
        query_states, key_states, value_states = (
            self.qkv(hidden_states)
            .reshape(sequence_length, 3, self.num_heads, -1)
            .permute(1, 0, 2, 3)
            .unbind(0)
        )
        cos, sin = position_embeddings
        query_states, key_states = qwen35_modeling.apply_rotary_pos_emb_vision(
            query_states,
            key_states,
            cos,
            sin,
        )
        query_states = query_states.transpose(0, 1).unsqueeze(0)
        key_states = key_states.transpose(0, 1).unsqueeze(0)
        value_states = value_states.transpose(0, 1).unsqueeze(0)

        lengths = (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()
        if not lengths or any(length <= 0 for length in lengths) or sum(lengths) != sequence_length:
            raise ValueError("MK1 vision cu_seqlens must describe positive chunks covering every token")
        splits = [torch.split(tensor, lengths, dim=2) for tensor in (query_states, key_states, value_states)]
        outputs = [
            explicit_repeat_kv_sdpa(
                self,
                query,
                key,
                value,
                None,
                dropout=0.0 if not self.training else self.attention_dropout,
                scaling=self.scaling,
                is_causal=False,
            )
            for query, key, value in zip(*splits, strict=True)
        ]
        attention_output = torch.cat(outputs, dim=1)
        return self.proj(attention_output.reshape(sequence_length, -1).contiguous())


class GenesisQwen36VisionModel(Qwen3_5MoeVisionModel):
    """Qwen3.5 vision stack with Genesis-order positional interpolation."""

    def fast_pos_embed_interpolate(self, grid_thw: torch.Tensor) -> torch.Tensor:
        grid_thw_list = grid_thw.tolist()
        grid_ts = [row[0] for row in grid_thw_list]
        grid_hs = [row[1] for row in grid_thw_list]
        grid_ws = [row[2] for row in grid_thw_list]
        device = self.pos_embed.weight.device

        idx_list = [[] for _ in range(4)]
        weight_list = [[] for _ in range(4)]
        for _t, height, width in grid_thw_list:
            height_indices = torch.linspace(0, self.num_grid_per_side - 1, height)
            width_indices = torch.linspace(0, self.num_grid_per_side - 1, width)
            height_floor = height_indices.int()
            width_floor = width_indices.int()
            height_ceil = (height_floor + 1).clip(max=self.num_grid_per_side - 1)
            width_ceil = (width_floor + 1).clip(max=self.num_grid_per_side - 1)
            height_delta = height_indices - height_floor
            width_delta = width_indices - width_floor
            base_height = height_floor * self.num_grid_per_side
            base_height_ceil = height_ceil * self.num_grid_per_side

            indices = [
                (base_height[None].T + width_floor[None]).flatten(),
                (base_height[None].T + width_ceil[None]).flatten(),
                (base_height_ceil[None].T + width_floor[None]).flatten(),
                (base_height_ceil[None].T + width_ceil[None]).flatten(),
            ]
            weights = [
                ((1 - height_delta)[None].T * (1 - width_delta)[None]).flatten(),
                ((1 - height_delta)[None].T * width_delta[None]).flatten(),
                (height_delta[None].T * (1 - width_delta)[None]).flatten(),
                (height_delta[None].T * width_delta[None]).flatten(),
            ]
            for index in range(4):
                idx_list[index].extend(indices[index].tolist())
                weight_list[index].extend(weights[index].tolist())

        index_tensor = torch.tensor(idx_list, dtype=torch.long, device=device)
        weight_tensor = torch.tensor(
            weight_list,
            dtype=self.pos_embed.weight.dtype,
            device=device,
        )
        interpolated = self.pos_embed(index_tensor).to(device) * weight_tensor[:, :, None]
        # Genesis performs one BF16 reduction; chained additions round at three
        # different boundaries and diverge at production image resolutions.
        patch_pos_embeds = interpolated.sum(dim=0)
        patch_pos_embeds = patch_pos_embeds.split(
            [height * width for height, width in zip(grid_hs, grid_ws, strict=True)]
        )

        permuted = []
        merge_size = self.config.spatial_merge_size
        for pos_embed, frames, height, width in zip(
            patch_pos_embeds,
            grid_ts,
            grid_hs,
            grid_ws,
            strict=True,
        ):
            pos_embed = pos_embed.repeat(frames, 1)
            pos_embed = (
                pos_embed.view(
                    frames,
                    height // merge_size,
                    merge_size,
                    width // merge_size,
                    merge_size,
                    -1,
                )
                .permute(0, 1, 3, 2, 4, 5)
                .flatten(0, 4)
            )
            permuted.append(pos_embed)
        return torch.cat(permuted)


class GenesisQwen36DecoderLayer(Qwen3_5MoeDecoderLayer):
    """Stock Qwen hybrid token mixer with the Genesis null-MoE block."""

    def __init__(self, config: Qwen3_5MoeTextConfig, layer_idx: int) -> None:
        # Avoid constructing and then discarding a stock sparse-MoE block.
        GradientCheckpointingLayer.__init__(self)
        GenesisNullMoeContract.from_text_config(config)
        self.hidden_size = config.hidden_size
        self.layer_type = config.layer_types[layer_idx]
        if self.layer_type == "linear_attention":
            self.linear_attn = GenesisQwen36GatedDeltaNet(config, layer_idx)
        elif self.layer_type == "full_attention":
            self.self_attn = GenesisQwen36Attention(config, layer_idx)
        else:
            raise ValueError(f"Unsupported MK1 layer type {self.layer_type!r} at index {layer_idx}")
        self.mlp = GenesisNullSparseMoeBlock(config)
        self.input_layernorm = Qwen3_5MoeRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3_5MoeRMSNorm(config.hidden_size, eps=config.rms_norm_eps)


class GenesisQwen36TextModel(Qwen3_5MoeTextModel):
    """Qwen3.6 hybrid text stack constructed directly with null-aware layers."""

    _no_split_modules = ["GenesisQwen36DecoderLayer"]

    def __init__(self, config: Qwen3_5MoeTextConfig) -> None:
        Qwen3_5MoePreTrainedModel.__init__(self, config)
        GenesisNullMoeContract.from_text_config(config)
        full_attention_interval = getattr(config, "full_attention_interval", 4)
        if isinstance(full_attention_interval, bool) or not isinstance(full_attention_interval, int):
            raise ValueError("MK1 Qwen3.6 full_attention_interval must be an integer")
        if full_attention_interval <= 0:
            raise ValueError("MK1 Qwen3.6 full_attention_interval must be positive")
        expected_layer_types = [
            "full_attention" if (layer_index + 1) % full_attention_interval == 0 else "linear_attention"
            for layer_index in range(config.num_hidden_layers)
        ]
        if config.layer_types != expected_layer_types:
            raise ValueError(
                f"MK1 Qwen3.6 requires full attention every {full_attention_interval} layers "
                "and linear attention elsewhere"
            )
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)
        self.layers = nn.ModuleList(
            [GenesisQwen36DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = Qwen3_5MoeRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = GenesisQwen36TextRotaryEmbedding(config)
        self.gradient_checkpointing = False
        self.post_init()


class GenesisQwen36Model(Qwen3_5MoeModel):
    """Multimodal Qwen3.6 backbone exposing the existing Qwen VLA interface."""

    _no_split_modules = ["GenesisQwen36DecoderLayer", "Qwen3_5MoeVisionBlock"]

    def __init__(self, config: Mk1Qwen36MoeConfig) -> None:
        Qwen3_5MoePreTrainedModel.__init__(self, config)
        GenesisNullMoeContract.from_text_config(config.text_config)
        self.visual = GenesisQwen36VisionModel._from_config(config.vision_config)
        for block in self.visual.blocks:
            block.attn = GenesisQwen36VisionAttention(config.vision_config)
        self.language_model = GenesisQwen36TextModel._from_config(config.text_config)
        self.rope_deltas = None
        self.post_init()


__all__ = [
    "GenesisNullMoeBranches",
    "GenesisNullMoeContract",
    "GenesisNullDispatchTrace",
    "GenesisNullRoutingOutput",
    "GenesisNullSparseMoeBlock",
    "GenesisNullTopKRouter",
    "GenesisQwen36DecoderLayer",
    "GenesisQwen36Attention",
    "GenesisQwen36AttentionTrace",
    "GenesisQwen36GatedDeltaNet",
    "GenesisQwen36Model",
    "GenesisQwen36TextModel",
    "GenesisQwen36TextRotaryEmbedding",
    "GenesisQwen36VisionAttention",
    "GenesisQwen36VisionModel",
    "Mk1Qwen36MoeConfig",
    "GENESIS_ROTARY_PRECISION",
    "deterministic_token_segment_sum",
    "explicit_repeat_kv_sdpa",
]
