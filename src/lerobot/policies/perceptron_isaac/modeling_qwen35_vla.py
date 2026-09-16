# ruff: noqa
"""Local HF model definition for the Isaac Qwen3.5-VL flow-matching VLA.

This wraps the native transformers ``Qwen3_5Model`` by composition and adds the
Isaac proprio encoder plus MolmoAct2 flow-matching action expert. The model
loads converted HF checkpoints with keys for ``lm_head``, ``model.language_model``,
``model.visual``, ``model.vector_embedding``, and ``model.action_expert``.

``sample_action(tensor_stream)`` runs the VLM once over a local TensorStream,
conditions the action expert on pre-action context, and integrates the flow
expert into a normalized ``[B, H, action_dim]`` chunk. TensorStream, MRoPE, and
mask utilities are vendored locally for the LeRobot native eval path.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from numbers import Real
from typing import Any, Literal, Optional
from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5Config
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5Model, Qwen3_5PreTrainedModel, Qwen3_5RMSNorm

# --- Genesis TensorStream layout / mrope / context-mask utilities ---
from .checkpoint_integrity import load_json_object
from .rtc import (
    DIT_ACTION_EXPERT_CONFIG_SCHEMA_VERSION,
    DIT_ACTION_EXPERT_CONFIG_V1_FIELDS,
    ActionExpertStepModulation,
    integrate_rtc_euler,
    materialize_rtc_action_prefix,
    prepare_rtc_conditioning,
    project_rtc_modulation,
    resolve_rtc_action_prefix,
)
from .qwen35_checkpoint import (
    QWEN35_CONVERSION_PROVENANCE_FILE,
    QWEN35_IMPORT_PROVENANCE_FILE,
    QWEN35_PACKAGED_SOURCE_FILENAMES,
    QWEN35_SOURCE_CONVERSION_PROVENANCE_FILE,
    QWEN35_WEIGHT_CONVENTION_FILE,
    _snapshot_verified_qwen35_imported_package,
    _snapshot_verified_qwen35_trained_package,
    _VerifiedQwen35Package,
    _verified_qwen35_model_path,
    safetensors_checkpoint_layout,
    verify_qwen35_weight_convention,
)
from .tensor_stream import TensorStream, TextType, VisionType, group_streams
from .tensor_stream_utils import (
    build_action_context_mask,
    compute_mrope_pos_tensor,
    first_event_start_indices,
    modality_mask,
    reconstruct_tensor_stream_from_compact_dict,
)

# The continuous-action marker type that delimits where the action chunk begins. Genesis sets
# FLOW_ACTION_TEXT_TYPE = TextType.action_c (value 15); the expert never attends to action tokens.
FLOW_ACTION_TEXT_TYPE = TextType.action_c


def normalize_vector_tokens(vector_tokens: torch.Tensor, max_states: int | None) -> torch.Tensor:
    """Normalize vector tokens to 2D and enforce a fixed width if max_states is set."""
    if vector_tokens.dim() < 1:
        raise ValueError("vector tokens must have at least one dimension")
    if vector_tokens.dim() == 1:
        vector_tokens = vector_tokens.unsqueeze(0)
    elif vector_tokens.dim() > 2:
        vector_tokens = vector_tokens.reshape(-1, vector_tokens.shape[-1])
    if max_states is None:
        return vector_tokens
    if max_states <= 0:
        raise ValueError("max_states must be >= 1")
    if vector_tokens.shape[-1] < max_states:
        return F.pad(vector_tokens, (0, max_states - vector_tokens.shape[-1]))
    if vector_tokens.shape[-1] > max_states:
        # Genesis stream_embedding raises here rather than dropping trailing state dims; a
        # partially-fed proprio encoder produces plausible but wrong actions with no error.
        raise ValueError(
            f"vector width {vector_tokens.shape[-1]} exceeds the configured model budget "
            f"{max_states}: silent truncation is not allowed."
        )
    return vector_tokens


class ModelOutput:
    """Minimal local training-output container retained for API compatibility."""

    def __init__(self, final_activations, final_embedding, additional_info=None, flow_matching_expert=None):
        self.final_activations = final_activations
        self.final_embedding = final_embedding
        self.additional_info = additional_info if additional_info is not None else {}
        self.flow_matching_expert = flow_matching_expert
        self._logits = None

    @property
    def logits(self):
        if self._logits is None:
            self._logits = self.final_embedding(self.final_activations)
        return self._logits


def expand_single_frame_patches_to_temporal_tubelets(
    hidden_states: torch.Tensor,
    *,
    in_channels: int,
    patch_size: int,
    temporal_patch_size: int,
) -> torch.Tensor:
    """Expand ``[N, C*P*P]`` per-frame patch rows to Qwen-temporal ``[N, C*T*P*P]`` rows.

    Vendored verbatim from ``genesis/core/models/perceptron/vision/qwen35.py``. The genesis
    rendering emits per-frame ``C*P*P`` patches, but the native ``Qwen3_5VisionPatchEmbed`` expects
    pre-tiled ``C*T*P*P`` rows (it does ``.view(-1, C, T, P, P)``). This tiles along the temporal axis
    so the same packed patches feed the native Conv3d.
    """
    if hidden_states.ndim != 2:
        raise ValueError(f"hidden_states must be rank-2 [N, D], got shape={tuple(hidden_states.shape)}")
    patch_dim = int(in_channels * patch_size * patch_size)
    if hidden_states.shape[-1] != patch_dim:
        raise ValueError(
            f"single-frame patch rows must have width C*P*P; expected {patch_dim}, got {hidden_states.shape[-1]}"
        )
    return (
        hidden_states.view(-1, in_channels, 1, patch_size, patch_size)
        .expand(-1, -1, temporal_patch_size, -1, -1)
        .reshape(-1, in_channels * temporal_patch_size * patch_size * patch_size)
    )


def build_vector_encoder(vector_max_states: int, hidden: int) -> nn.Module:
    """Proprio encoder mirroring genesis ``build_vector_encoder`` (2-Linear SiLU MLP, no bias).

    Loads ``model.vector_embedding.0.weight`` ``[hidden, vector_max_states]`` and
    ``model.vector_embedding.2.weight`` ``[hidden, hidden]``.
    """
    return nn.Sequential(
        nn.Linear(vector_max_states, hidden, bias=False),
        nn.SiLU(),
        nn.Linear(hidden, hidden, bias=False),
    )


@torch.no_grad()
def apply_qwen35_offset_norm_correction(language_model: nn.Module) -> int:
    """Convert genesis (standard) RMSNorm weights to the native Qwen3.5 unit-offset convention.

    Native ``Qwen3_5RMSNorm`` computes ``x_normed * (1 + weight)`` (weight init = 0), whereas genesis
    trains a standard ``Qwen2RMSNorm`` (``x_normed * weight``, weight ~ 1). ``convert_genesis_qwen35_to_hf``
    writes the genesis weights verbatim, so native would apply ``(1 + w)`` instead of ``w`` — a per-channel
    direction change that compounds across layers into a garbage final state. Subtracting
    1.0 makes native compute ``(1 + (w - 1)) = w``. Only ``Qwen3_5RMSNorm`` modules are touched — the gated
    ``Qwen3_5RMSNormGated`` (a different class, standard convention) and the vision LayerNorms are untouched.
    Idempotency is the caller's responsibility (apply exactly once, right after loading raw genesis weights).
    """
    corrected = 0
    for module in language_model.modules():
        if isinstance(module, Qwen3_5RMSNorm):
            module.weight.sub_(1.0)
            corrected += 1
    return corrected


# === Action expert (vendored MolmoAct2 DiT + pluggable head) ===
@dataclass
class MolmoAct2ActionExpertConfig:
    """Plain-dataclass mirror of MolmoAct2's action-expert config (defaults = config.json)."""

    hidden_size: int = 768
    num_layers: int = 36
    num_heads: int = 8
    max_action_dim: int = 32
    max_action_horizon: int = 30
    mlp_ratio: float = 4.0
    ffn_multiple_of: int = 256
    timestep_embed_dim: int = 256
    attn_dropout: float = 0.0
    dropout: float = 0.0
    qk_norm: bool = True
    qk_norm_eps: float = 1e-6
    rope: bool = True
    context_layer_norm: bool = True
    causal_attn: bool = False


def _broadcast_action_condition(condition: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
    if condition.dim() == actions.dim() - 1:
        return condition.unsqueeze(1)
    if condition.dim() == actions.dim():
        return condition
    raise ValueError(
        f"Action conditioning must be [B,D] or [B,H,D]; got {tuple(condition.shape)} "
        f"for actions {tuple(actions.shape)}."
    )


def _modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    shift = _broadcast_action_condition(shift, x)
    scale = _broadcast_action_condition(scale, x)
    return x * (1 + scale) + shift


def _round_up_multiple(value: int, multiple_of: int) -> int:
    if multiple_of <= 0:
        return value
    return int(math.ceil(value / multiple_of) * multiple_of)


def _init_linear(linear: nn.Linear, *, zero: bool = False, scale: float = 1.0) -> None:
    if zero:
        nn.init.zeros_(linear.weight)
    else:
        nn.init.xavier_uniform_(linear.weight)
        if scale != 1.0:
            with torch.no_grad():
                linear.weight.mul_(scale)
    if linear.bias is not None:
        nn.init.zeros_(linear.bias)


@dataclass
class ActionExpertContext:
    kv_contexts: Sequence[tuple[torch.Tensor, torch.Tensor]]
    cross_mask: torch.Tensor | None
    self_mask: torch.Tensor | None
    valid_action: torch.Tensor | None
    rope_cache: tuple[torch.Tensor, torch.Tensor] | None = None


class ActionExpertRMSNorm(nn.Module):
    def __init__(
        self,
        size: int,
        *,
        eps: float = 1e-6,
        elementwise_affine: bool = False,
        device=None,
    ) -> None:
        super().__init__()
        self.size = size
        self.eps = eps
        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(size, device=device))
        else:
            self.register_parameter("weight", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.autocast(enabled=False, device_type=x.device.type):
            dtype = x.dtype
            x_float = x.to(torch.float32)
            variance = x_float.pow(2).mean(dim=-1, keepdim=True)
            out = x_float * torch.rsqrt(variance + self.eps)
            out = out.to(dtype)
        if self.weight is not None:
            out = out * self.weight
        return out

    def reset_parameters(self) -> None:
        if self.weight is not None:
            nn.init.ones_(self.weight)


class ActionExpertRotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, base: float = 10000.0) -> None:
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError("RoPE requires an even head_dim.")
        self.head_dim = head_dim
        self.base = base

    def build_cache(
        self,
        *,
        seq_len: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        half_dim = self.head_dim // 2
        inv_freq = 1.0 / (
            self.base ** (torch.arange(0, half_dim, device=device, dtype=torch.float32) / max(half_dim, 1))
        )
        positions = torch.arange(seq_len, device=device, dtype=torch.float32)
        freqs = torch.outer(positions, inv_freq)
        cos = freqs.cos().to(dtype=dtype).view(1, 1, seq_len, half_dim)
        sin = freqs.sin().to(dtype=dtype).view(1, 1, seq_len, half_dim)
        return cos, sin

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        *,
        rope_cache: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if rope_cache is None:
            rope_cache = self.build_cache(seq_len=q.shape[-2], device=q.device, dtype=q.dtype)
        cos, sin = rope_cache
        half_dim = self.head_dim // 2

        def _apply(x: torch.Tensor) -> torch.Tensor:
            x1, x2 = x[..., :half_dim], x[..., half_dim:]
            return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)

        return _apply(q), _apply(k)


class ActionExpertSelfAttention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        *,
        attn_dropout: float = 0.0,
        proj_dropout: float = 0.0,
        qk_norm: bool = True,
        qk_norm_eps: float = 1e-6,
        use_rope: bool = True,
    ) -> None:
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads")
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.attn_dropout = attn_dropout
        self.q_norm = ActionExpertRMSNorm(self.head_dim, eps=qk_norm_eps) if qk_norm else None
        self.k_norm = ActionExpertRMSNorm(self.head_dim, eps=qk_norm_eps) if qk_norm else None
        self.rope = ActionExpertRotaryEmbedding(self.head_dim) if use_rope else None
        self.qkv = nn.Linear(hidden_size, hidden_size * 3)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        self.out_drop = nn.Dropout(proj_dropout)

    def _apply_qk_norm(self, q: torch.Tensor, k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.q_norm is None or self.k_norm is None:
            return q, k
        return self.q_norm(q), self.k_norm(k)

    def _attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        attn_mask: torch.Tensor | None = None,
        is_causal: bool = False,
    ) -> torch.Tensor:
        dropout_p = self.attn_dropout if self.training else 0.0
        out = F.scaled_dot_product_attention(
            q.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
            attn_mask=attn_mask,
            dropout_p=dropout_p,
            is_causal=is_causal,
        )
        return out.transpose(1, 2).contiguous()

    def forward(
        self,
        x: torch.Tensor,
        *,
        attn_mask: torch.Tensor | None = None,
        is_causal: bool = False,
        rope_cache: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        bsz, seq_len, _ = x.shape
        qkv = self.qkv(x).view(bsz, seq_len, 3, self.num_heads, self.head_dim)
        q = qkv[:, :, 0].transpose(1, 2)
        k = qkv[:, :, 1].transpose(1, 2)
        v = qkv[:, :, 2].contiguous()
        q, k = self._apply_qk_norm(q, k)
        if self.rope is not None:
            q, k = self.rope(q, k, rope_cache=rope_cache)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        out = self._attention(q, k, v, attn_mask=attn_mask, is_causal=is_causal)
        out = out.reshape(bsz, seq_len, self.hidden_size)
        return self.out_drop(self.out_proj(out))


class ActionExpertCrossAttention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        *,
        attn_dropout: float = 0.0,
        proj_dropout: float = 0.0,
        qk_norm: bool = True,
        qk_norm_eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads")
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.attn_dropout = attn_dropout
        self.q_norm = ActionExpertRMSNorm(self.head_dim, eps=qk_norm_eps) if qk_norm else None
        self.k_norm = ActionExpertRMSNorm(self.head_dim, eps=qk_norm_eps) if qk_norm else None
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        self.out_drop = nn.Dropout(proj_dropout)

    def _as_heads(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 4:
            if x.shape[2] == self.num_heads:
                return x
            if x.shape[1] == self.num_heads:
                return x.transpose(1, 2).contiguous()
            raise ValueError(f"Unexpected cross-attention KV shape {tuple(x.shape)}")
        if x.dim() != 3:
            raise ValueError(f"Expected 3D/4D cross-attention KV, got {tuple(x.shape)}")
        bsz, seq_len, _ = x.shape
        return x.view(bsz, seq_len, self.num_heads, self.head_dim)

    def _attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        dropout_p = self.attn_dropout if self.training else 0.0
        out = F.scaled_dot_product_attention(
            q.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
            attn_mask=attn_mask,
            dropout_p=dropout_p,
            is_causal=False,
        )
        return out.transpose(1, 2).contiguous()

    def forward(
        self,
        x: torch.Tensor,
        *,
        kv_k: torch.Tensor,
        kv_v: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        bsz, tgt_len, _ = x.shape
        q = self.q_proj(x).view(bsz, tgt_len, self.num_heads, self.head_dim)
        k = self._as_heads(kv_k)
        v = self._as_heads(kv_v)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        if self.q_norm is not None:
            q = self.q_norm(q)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        out = self._attention(q, k, v, attn_mask=attn_mask)
        out = out.reshape(bsz, tgt_len, self.hidden_size)
        return self.out_drop(self.out_proj(out))


class ActionExpertMLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        *,
        mlp_ratio: float,
        multiple_of: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        inner_dim = _round_up_multiple(int(hidden_size * mlp_ratio), multiple_of)
        self.up_proj = nn.Linear(hidden_size, inner_dim)
        self.gate_proj = nn.Linear(hidden_size, inner_dim)
        self.down_proj = nn.Linear(inner_dim, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.silu(self.gate_proj(x)) * self.up_proj(x)
        x = self.dropout(x)
        x = self.down_proj(x)
        return self.dropout(x)


class ActionExpertModulation(nn.Module):
    def __init__(self, hidden_size: int, num_chunks: int) -> None:
        super().__init__()
        self.act = nn.SiLU()
        self.linear = nn.Linear(hidden_size, num_chunks * hidden_size)

    def forward(self, conditioning: torch.Tensor) -> torch.Tensor:
        return self.linear(self.act(conditioning))


class ActionExpertBlock(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        *,
        mlp_ratio: float,
        ffn_multiple_of: int,
        attn_dropout: float = 0.0,
        dropout: float = 0.0,
        qk_norm: bool = True,
        qk_norm_eps: float = 1e-6,
        rope: bool = True,
    ) -> None:
        super().__init__()
        self.self_norm = ActionExpertRMSNorm(hidden_size, eps=1e-6)
        self.cross_norm = ActionExpertRMSNorm(hidden_size, eps=1e-6)
        self.ff_norm = ActionExpertRMSNorm(hidden_size, eps=1e-6)
        self.self_attn = ActionExpertSelfAttention(
            hidden_size,
            num_heads,
            attn_dropout=attn_dropout,
            proj_dropout=dropout,
            qk_norm=qk_norm,
            qk_norm_eps=qk_norm_eps,
            use_rope=rope,
        )
        self.cross_attn = ActionExpertCrossAttention(
            hidden_size,
            num_heads,
            attn_dropout=attn_dropout,
            proj_dropout=dropout,
            qk_norm=qk_norm,
            qk_norm_eps=qk_norm_eps,
        )
        self.mlp = ActionExpertMLP(
            hidden_size,
            mlp_ratio=mlp_ratio,
            multiple_of=ffn_multiple_of,
            dropout=dropout,
        )
        self.modulation = ActionExpertModulation(hidden_size, 9)

    def forward(
        self,
        x: torch.Tensor,
        conditioning: torch.Tensor,
        *,
        cross_kv: tuple[torch.Tensor, torch.Tensor],
        self_attn_mask: torch.Tensor | None = None,
        attn_mask: torch.Tensor | None = None,
        is_causal: bool = False,
        modulation: tuple[torch.Tensor, ...] | None = None,
        rope_cache: tuple[torch.Tensor, torch.Tensor] | None = None,
        rtc_suffix_conditioning: torch.Tensor | None = None,
        rtc_prefix_conditioning: torch.Tensor | None = None,
        rtc_prefix_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if rtc_suffix_conditioning is not None:
            assert rtc_prefix_conditioning is not None
            assert rtc_prefix_mask is not None
            if modulation is not None:
                raise ValueError("precomputed modulation and RTC conditioning are mutually exclusive.")
            modulation = project_rtc_modulation(
                rtc_suffix_conditioning,
                rtc_prefix_conditioning,
                rtc_prefix_mask,
                modulation=self.modulation,
                chunks=9,
            )
        elif modulation is None:
            modulation = self.modulation(conditioning).chunk(9, dim=-1)
        (
            shift_msa,
            scale_msa,
            gate_msa,
            shift_mca,
            scale_mca,
            gate_mca,
            shift_mlp,
            scale_mlp,
            gate_mlp,
        ) = modulation
        x = x + _broadcast_action_condition(gate_msa, x) * self.self_attn(
            _modulate(self.self_norm(x), shift_msa, scale_msa),
            attn_mask=self_attn_mask,
            is_causal=is_causal,
            rope_cache=rope_cache,
        )
        x = x + _broadcast_action_condition(gate_mca, x) * self.cross_attn(
            _modulate(self.cross_norm(x), shift_mca, scale_mca),
            kv_k=cross_kv[0],
            kv_v=cross_kv[1],
            attn_mask=attn_mask,
        )
        x = x + _broadcast_action_condition(gate_mlp, x) * self.mlp(_modulate(self.ff_norm(x), shift_mlp, scale_mlp))
        return x


class ActionExpertFinalLayer(nn.Module):
    def __init__(self, hidden_size: int, output_dim: int) -> None:
        super().__init__()
        self.norm = ActionExpertRMSNorm(hidden_size, eps=1e-6)
        self.modulation = ActionExpertModulation(hidden_size, 2)
        self.linear = nn.Linear(hidden_size, output_dim)

    def forward(
        self,
        x: torch.Tensor,
        conditioning: torch.Tensor,
        *,
        modulation: tuple[torch.Tensor, torch.Tensor] | None = None,
        rtc_suffix_conditioning: torch.Tensor | None = None,
        rtc_prefix_conditioning: torch.Tensor | None = None,
        rtc_prefix_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if rtc_suffix_conditioning is not None:
            assert rtc_prefix_conditioning is not None
            assert rtc_prefix_mask is not None
            if modulation is not None:
                raise ValueError("precomputed modulation and RTC conditioning are mutually exclusive.")
            modulation = project_rtc_modulation(
                rtc_suffix_conditioning,
                rtc_prefix_conditioning,
                rtc_prefix_mask,
                modulation=self.modulation,
                chunks=2,
            )
        elif modulation is None:
            modulation = self.modulation(conditioning).chunk(2, dim=-1)
        shift, scale = modulation
        return self.linear(_modulate(self.norm(x), shift, scale))


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        timestep_shape = timesteps.shape
        timesteps = timesteps.reshape(-1)
        half_dim = self.dim // 2
        freq = torch.exp(
            torch.arange(half_dim, device=timesteps.device, dtype=timesteps.dtype)
            * (-math.log(10000.0) / max(half_dim - 1, 1))
        )
        args = timesteps[:, None] * freq[None, :]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb.reshape(*timestep_shape, self.dim)


class ActionExpert(nn.Module):
    """Modern MolmoAct2 action expert, embedded for HF remote-code inference."""

    def __init__(
        self,
        config: MolmoAct2ActionExpertConfig,
        *,
        llm_dim: int,
        llm_kv_dim: int,
        llm_num_layers: int,
        device=None,
    ):
        super().__init__()
        if config.num_layers != llm_num_layers:
            raise ValueError(
                "MolmoAct2 HF action expert supports only per-layer conditioning with one "
                f"action block per LLM layer (action={config.num_layers}, llm={llm_num_layers})."
            )
        self.config = config
        self.hidden_size = config.hidden_size
        self.llm_dim = llm_dim
        self.llm_kv_dim = llm_kv_dim
        self.action_head_dim = config.hidden_size // config.num_heads

        self.time_embed = nn.Sequential(
            SinusoidalTimeEmbedding(config.timestep_embed_dim),
            nn.Linear(config.timestep_embed_dim, config.hidden_size, device=device),
            nn.SiLU(),
            nn.Linear(config.hidden_size, config.hidden_size, device=device),
        )
        self.action_embed = nn.Linear(config.max_action_dim, config.hidden_size, device=device)
        self.context_k_proj = nn.Linear(self.llm_kv_dim, config.hidden_size, bias=False, device=device)
        self.context_v_proj = nn.Linear(self.llm_kv_dim, config.hidden_size, bias=False, device=device)
        self.context_norm = (
            ActionExpertRMSNorm(config.hidden_size, eps=1e-6) if config.context_layer_norm else nn.Identity()
        )
        self._modulation_cache_key: tuple[Any, ...] | None = None
        self._modulation_cache_value: Sequence[ActionExpertStepModulation] | None = None
        self.blocks = nn.ModuleList(
            [
                ActionExpertBlock(
                    config.hidden_size,
                    config.num_heads,
                    mlp_ratio=config.mlp_ratio,
                    ffn_multiple_of=config.ffn_multiple_of,
                    attn_dropout=config.attn_dropout,
                    dropout=config.dropout,
                    qk_norm=config.qk_norm,
                    qk_norm_eps=config.qk_norm_eps,
                    rope=config.rope,
                )
                for _ in range(config.num_layers)
            ]
        )
        self.final_layer = ActionExpertFinalLayer(config.hidden_size, config.max_action_dim)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self.time_embed.modules():
            if isinstance(module, nn.Linear):
                _init_linear(module)
        _init_linear(self.action_embed)
        _init_linear(self.context_k_proj)
        _init_linear(self.context_v_proj)
        if isinstance(self.context_norm, ActionExpertRMSNorm):
            self.context_norm.reset_parameters()
        residual_scale = (2 * max(self.config.num_layers, 1)) ** -0.5
        for block in self.blocks:
            _init_linear(block.self_attn.qkv)
            _init_linear(block.self_attn.out_proj, scale=residual_scale)
            _init_linear(block.cross_attn.q_proj)
            _init_linear(block.cross_attn.out_proj, scale=residual_scale)
            _init_linear(block.mlp.up_proj)
            _init_linear(block.mlp.gate_proj)
            _init_linear(block.mlp.down_proj, scale=residual_scale)
            _init_linear(block.modulation.linear, zero=True)
            block.self_norm.reset_parameters()
            block.cross_norm.reset_parameters()
            block.ff_norm.reset_parameters()
            if block.self_attn.q_norm is not None:
                block.self_attn.q_norm.reset_parameters()
            if block.self_attn.k_norm is not None:
                block.self_attn.k_norm.reset_parameters()
            if block.cross_attn.q_norm is not None:
                block.cross_attn.q_norm.reset_parameters()
            if block.cross_attn.k_norm is not None:
                block.cross_attn.k_norm.reset_parameters()
        self.final_layer.norm.reset_parameters()
        _init_linear(self.final_layer.modulation.linear, zero=True)
        _init_linear(self.final_layer.linear, zero=True)

    def _reshape_hidden_to_heads(self, x: torch.Tensor) -> torch.Tensor:
        return x.view(x.shape[0], x.shape[1], self.config.num_heads, self.action_head_dim)

    def _time_conditioning(self, timesteps: torch.Tensor) -> torch.Tensor:
        conditioning = self.time_embed[0](timesteps)
        first_linear = self.time_embed[1]
        if isinstance(first_linear, nn.Linear):
            conditioning = conditioning.to(dtype=first_linear.weight.dtype)
        for module in list(self.time_embed.children())[1:]:
            conditioning = module(conditioning)
        return conditioning

    def prepare_rtc_conditioning(
        self,
        base_timesteps: torch.Tensor,
        prefix_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return prepare_rtc_conditioning(
            base_timesteps,
            prefix_mask,
            time_conditioning=self._time_conditioning,
        )

    def _project_kv_tensor(self, x: torch.Tensor, proj: nn.Linear) -> torch.Tensor:
        flat = self.context_norm(proj(x))
        return self._reshape_hidden_to_heads(flat)

    def _prepare_kv_context(
        self,
        encoder_kv_states: Sequence[tuple[torch.Tensor, torch.Tensor]],
    ) -> Sequence[tuple[torch.Tensor, torch.Tensor]]:
        if len(encoder_kv_states) != len(self.blocks):
            raise ValueError(
                f"Expected {len(self.blocks)} KV layers for per-layer conditioning, got {len(encoder_kv_states)}."
            )
        kv_contexts = []
        for block, (k_in, v_in) in zip(self.blocks, encoder_kv_states, strict=False):
            k_ctx = self._project_kv_tensor(k_in, self.context_k_proj)
            v_ctx = self._project_kv_tensor(v_in, self.context_v_proj)
            k_norm = block.cross_attn.k_norm
            if k_norm is not None:
                k_ctx = k_norm(k_ctx.transpose(1, 2)).transpose(1, 2)
            kv_contexts.append((k_ctx, v_ctx))
        return kv_contexts

    @staticmethod
    def _build_cross_attention_mask(
        encoder_attention_mask: torch.Tensor | None,
        batch_size: int,
        dtype: torch.dtype,
    ) -> torch.Tensor | None:
        if encoder_attention_mask is None:
            return None
        mask = encoder_attention_mask[:, None, None, :].to(dtype=dtype)
        return (1.0 - mask) * torch.finfo(dtype).min

    def _build_self_attention_mask(
        self,
        action_attention_mask: torch.Tensor | None,
        seq_len: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor | None:
        mask = None
        if action_attention_mask is not None:
            valid = action_attention_mask.to(device=device, dtype=torch.bool)
            key_mask = (~valid)[:, None, None, :].to(dtype=dtype)
            mask = key_mask * torch.finfo(dtype).min
        if self.config.causal_attn:
            causal = torch.ones(seq_len, seq_len, device=device, dtype=torch.bool).triu(diagonal=1)
            causal = causal.unsqueeze(0).unsqueeze(0).to(dtype=dtype) * torch.finfo(dtype).min
            mask = causal if mask is None else mask + causal
        return mask

    def prepare_context(
        self,
        *,
        encoder_kv_states: Sequence[tuple[torch.Tensor, torch.Tensor]],
        encoder_attention_mask: torch.Tensor | None = None,
        action_attention_mask: torch.Tensor | None = None,
        state_embeddings: torch.Tensor | None = None,
        batch_size: int,
        seq_len: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> ActionExpertContext:
        if state_embeddings is not None:
            raise ValueError(
                "MolmoAct2 HF action expert supports only discrete state tokens. "
                "Continuous state embeddings are not supported."
            )
        valid_action = None
        if action_attention_mask is not None:
            valid_action = action_attention_mask.to(device=device, dtype=dtype).unsqueeze(-1)
        rope_cache = None
        if len(self.blocks) > 0 and self.blocks[0].self_attn.rope is not None:
            rope_cache = self.blocks[0].self_attn.rope.build_cache(
                seq_len=seq_len,
                device=device,
                dtype=dtype,
            )
        kv_contexts = self._prepare_kv_context(encoder_kv_states)
        cross_mask = self._build_cross_attention_mask(
            encoder_attention_mask,
            batch_size,
            dtype,
        )
        self_mask = self._build_self_attention_mask(action_attention_mask, seq_len, device, dtype)
        return ActionExpertContext(
            kv_contexts=kv_contexts,
            cross_mask=cross_mask,
            self_mask=self_mask,
            valid_action=valid_action,
            rope_cache=rope_cache,
        )

    def prepare_modulation_cache(
        self,
        timesteps: Sequence[torch.Tensor],
    ) -> Sequence[ActionExpertStepModulation]:
        cache = []
        for _idx, step_t in enumerate(timesteps):
            conditioning = self._time_conditioning(step_t)
            block_modulations = []
            for block in self.blocks:
                block_modulations.append(tuple(block.modulation(conditioning).chunk(9, dim=-1)))
            final_modulation = tuple(self.final_layer.modulation(conditioning).chunk(2, dim=-1))
            cache.append(
                ActionExpertStepModulation(
                    conditioning=conditioning,
                    block_modulations=block_modulations,
                    final_modulation=final_modulation,
                )
            )
        return cache

    def get_or_prepare_modulation_cache(
        self,
        timesteps: Sequence[torch.Tensor],
        *,
        cache_key: tuple[Any, ...] | None = None,
    ) -> Sequence[ActionExpertStepModulation]:
        if self.training or cache_key is None:
            return self.prepare_modulation_cache(timesteps)
        if self._modulation_cache_key == cache_key and self._modulation_cache_value is not None:
            return self._modulation_cache_value
        cached = self.prepare_modulation_cache(timesteps)
        self._modulation_cache_key = cache_key
        self._modulation_cache_value = cached
        return cached

    def forward_with_context(
        self,
        actions: torch.Tensor,
        timesteps: torch.Tensor,
        *,
        context: ActionExpertContext,
        modulation: ActionExpertStepModulation | None = None,
        rtc_conditioning: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        bsz, seq_len, _ = actions.shape
        if seq_len > self.config.max_action_horizon:
            raise ValueError(
                f"Action sequence length {seq_len} exceeds configured max_action_horizon={self.config.max_action_horizon}"
            )
        if rtc_conditioning is not None:
            if modulation is not None:
                raise ValueError("precomputed modulation and RTC conditioning are mutually exclusive.")
            rtc_suffix_conditioning, rtc_prefix_conditioning, rtc_prefix_mask = rtc_conditioning
            conditioning = rtc_suffix_conditioning
            block_modulations = [None] * len(self.blocks)
            final_modulation = None
        elif modulation is None:
            rtc_suffix_conditioning = rtc_prefix_conditioning = rtc_prefix_mask = None
            conditioning = self._time_conditioning(timesteps)
            block_modulations: Sequence[tuple[torch.Tensor, ...] | None] = [None] * len(self.blocks)
            final_modulation = None
        else:
            rtc_suffix_conditioning = rtc_prefix_conditioning = rtc_prefix_mask = None
            conditioning = modulation.conditioning
            block_modulations = modulation.block_modulations
            final_modulation = modulation.final_modulation
        x = self.action_embed(actions)
        if context.valid_action is not None:
            x = x * context.valid_action
        for _idx, (block, kv_context, block_modulation) in enumerate(
            zip(self.blocks, context.kv_contexts, block_modulations, strict=False)
        ):
            x = block(
                x,
                conditioning,
                cross_kv=kv_context,
                self_attn_mask=context.self_mask,
                attn_mask=context.cross_mask,
                is_causal=self.config.causal_attn,
                modulation=block_modulation,
                rope_cache=context.rope_cache,
                rtc_suffix_conditioning=rtc_suffix_conditioning,
                rtc_prefix_conditioning=rtc_prefix_conditioning,
                rtc_prefix_mask=rtc_prefix_mask,
            )
            if context.valid_action is not None:
                x = x * context.valid_action
        out = self.final_layer(
            x,
            conditioning,
            modulation=final_modulation,
            rtc_suffix_conditioning=rtc_suffix_conditioning,
            rtc_prefix_conditioning=rtc_prefix_conditioning,
            rtc_prefix_mask=rtc_prefix_mask,
        )
        if context.valid_action is not None:
            out = out * context.valid_action
        return out

    def forward(
        self,
        actions: torch.Tensor,
        timesteps: torch.Tensor,
        *,
        encoder_kv_states: Sequence[tuple[torch.Tensor, torch.Tensor]],
        encoder_attention_mask: torch.Tensor | None = None,
        action_attention_mask: torch.Tensor | None = None,
        state_embeddings: torch.Tensor | None = None,
    ) -> torch.Tensor:
        bsz, seq_len, _ = actions.shape
        context = self.prepare_context(
            encoder_kv_states=encoder_kv_states,
            encoder_attention_mask=encoder_attention_mask,
            action_attention_mask=action_attention_mask,
            state_embeddings=state_embeddings,
            batch_size=bsz,
            seq_len=seq_len,
            device=actions.device,
            dtype=actions.dtype,
        )
        return self.forward_with_context(actions, timesteps, context=context)


# ---------------------------------------------------------------------------
# Pluggable action-expert head. The vendored ActionExpert (above) is driven from
# the backbone's FINAL-layer activations as a single shared cross-attention
# context (clean-at-1), ported from genesis MolmoActFlowExpert. Geometry comes
# from IsaacConfig.action_expert; weights nest as action_expert.action_expert.*
# (mirrors genesis flow_matching_expert.action_expert.* -> pure converter prefix swap).
# ---------------------------------------------------------------------------


class ActionExpertHead(nn.Module):
    """Interface for a pluggable continuous-action expert head.

    Implementations build their own flow/horizon/context-wiring internally and expose:
      - ``from_config(action_expert_cfg: dict, vlm_dim: int)`` constructor,
      - ``action_dim`` / ``action_horizon`` ints,
      - ``sample(vlm_activations, vlm_mask, ...) -> [B, H, action_dim]``.
    Swapping experts is a config flip + checkpoint; the VLM and ``sample_action`` are
    expert-agnostic. (GenesisFlowExpertHead, clean-at-0, is a future drop-in.)
    """

    expert_type: str = "base"
    action_dim: int = 0
    action_horizon: int = 0

    @torch.no_grad()
    def sample(self, vlm_activations, vlm_mask=None, **kwargs):  # pragma: no cover - interface
        raise NotImplementedError


@dataclass
class MolmoActExpertTrainArgs:
    """Genesis-free duck-type of ``MolmoActFlowExpertArgs`` exposing the training-time fields the
    genesis flow loss reads via ``getattr(expert.args, ...)``. Attached to ``MolmoActExpertHead`` and
    tuned by ``setup_qwen35_vla_for_training``; the inference path never reads it. Defaults mirror
    ``FlowMatchingExpertArgs`` (Beta(1.5,1.0) timesteps) so a freshly-loaded head is loss-ready."""

    action_dim: int
    action_horizon: int
    hidden_dim: int
    clean_at_0: bool = False
    timestep_sampling_alpha: float = 1.5
    timestep_sampling_beta: float = 1.0
    timestep_sampling_scale: float = 0.999
    timestep_sampling_offset: float = 0.001
    train_samples_per_chunk: int = 1
    dual_timestep_ratio: float = 0.0
    rtc_max_delay_steps: int = 0
    rtc_delay_sampling: str = "uniform"
    mask_padded_action_rows: bool = False
    drop_action_dim_overflow: bool = False


class MolmoActExpertHead(ActionExpertHead):
    """Vendored MolmoAct2 ActionExpert wired to a single shared final-layer context.

    Ported from genesis ``MolmoActFlowExpert``: context_k/v_proj project vlm_dim->hidden
    (the genesis-wired projections ARE the trained weights), every other weight loads
    verbatim. Convention is clean-at-1 (``clean_at_0=False``: feed ``1 - tau``).
    """

    expert_type = "molmoact"

    def __init__(self, action_expert_cfg: dict, vlm_dim: int) -> None:
        super().__init__()
        c = dict(action_expert_cfg)
        self.vlm_dim = int(vlm_dim)
        self.action_dim = int(c["action_dim"])
        self.action_horizon = int(c["action_horizon"])
        self.num_inference_steps = int(c.get("num_inference_steps", 10))
        self.clean_at_0 = bool(c.get("clean_at_0", False))
        # RTC capability declared by the checkpoint contract (genesis rtc_max_delay_steps):
        # the maximum pinned-prefix length the expert was trained to inpaint. 0 = not
        # RTC-trained; sample() rejects any action prefix beyond this budget.
        self.rtc_max_delay_steps = int(c.get("rtc_max_delay_steps", 0) or 0)
        cfg = MolmoAct2ActionExpertConfig(
            hidden_size=int(c["hidden_dim"]),
            num_layers=int(c["num_layers"]),
            num_heads=int(c["num_heads"]),
            max_action_dim=self.action_dim,
            max_action_horizon=self.action_horizon,
            mlp_ratio=float(c.get("mlp_ratio", 4.0)),
            ffn_multiple_of=int(c.get("ffn_multiple_of", 256)),
            timestep_embed_dim=int(c.get("timestep_embed_dim", 256)),
            attn_dropout=0.0,
            dropout=0.0,
            qk_norm=bool(c.get("qk_norm", True)),
            qk_norm_eps=float(c.get("qk_norm_eps", 1e-6)),
            rope=bool(c.get("rope", True)),
            context_layer_norm=bool(c.get("context_layer_norm", True)),
            causal_attn=bool(c.get("causal_attn", False)),
        )
        # llm_kv_dim=vlm_dim => context_k/v_proj are vlm_dim->hidden; one shared context fed
        # to all blocks, so llm_num_layers is nominal (satisfies the one-block-per-layer assert).
        self.action_expert = ActionExpert(
            cfg, llm_dim=vlm_dim, llm_kv_dim=vlm_dim, llm_num_layers=cfg.num_layers
        )
        self.hidden_dim = int(c["hidden_dim"])
        # Training-time args duck-typing genesis MolmoActFlowExpertArgs (read by the genesis flow
        # loss). setup_qwen35_vla_for_training overrides the train-only knobs; inference ignores it.
        self.args = MolmoActExpertTrainArgs(
            action_dim=self.action_dim,
            action_horizon=self.action_horizon,
            hidden_dim=self.hidden_dim,
            clean_at_0=self.clean_at_0,
        )

    def _flow_time(self, tau):
        # Genesis tau is the noise level (1=noise); MolmoAct2 is clean-at-1, so feed 1-tau.
        return tau if self.clean_at_0 else (1.0 - tau)

    def _build_single_context(
        self, vlm_activations, vlm_mask, action_mask, *, seq_len, batch_size, device, dtype
    ):
        ae = self.action_expert
        encoder_attention_mask = None if vlm_mask is None else vlm_mask.to(device=device, dtype=dtype)
        action_attention_mask = None if action_mask is None else action_mask.to(device=device)
        k_base = ae._project_kv_tensor(vlm_activations, ae.context_k_proj)  # noqa: SLF001
        v_ctx = ae._project_kv_tensor(vlm_activations, ae.context_v_proj)  # noqa: SLF001
        kv_contexts = []
        for block in ae.blocks:
            k_ctx = k_base
            k_norm = block.cross_attn.k_norm
            if k_norm is not None:
                k_ctx = k_norm(k_ctx.transpose(1, 2)).transpose(1, 2)
            kv_contexts.append((k_ctx, v_ctx))
        cross_mask = ae._build_cross_attention_mask(encoder_attention_mask, batch_size, dtype)  # noqa: SLF001
        self_mask = ae._build_self_attention_mask(action_attention_mask, seq_len, device, dtype)  # noqa: SLF001
        valid_action = None
        if action_attention_mask is not None:
            valid_action = action_attention_mask.to(dtype=dtype).unsqueeze(-1)
        rope_cache = None
        if len(ae.blocks) > 0 and ae.blocks[0].self_attn.rope is not None:
            rope_cache = ae.blocks[0].self_attn.rope.build_cache(seq_len=seq_len, device=device, dtype=dtype)
        return ActionExpertContext(
            kv_contexts=kv_contexts,
            cross_mask=cross_mask,
            self_mask=self_mask,
            valid_action=valid_action,
            rope_cache=rope_cache,
        )

    # -- training timestep sampling: genesis Beta(1.5,1.0) noise level (== MolmoAct2's t-dist) --
    def sample_timesteps(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        a = self.args
        beta = torch.distributions.Beta(
            torch.tensor(a.timestep_sampling_alpha, device=device, dtype=torch.float32),
            torch.tensor(a.timestep_sampling_beta, device=device, dtype=torch.float32),
        ).sample((batch_size,))
        tau = beta * a.timestep_sampling_scale + a.timestep_sampling_offset
        return tau.to(device=device, dtype=dtype)

    # -- horizon embedding unused (MolmoAct2 positions come from RoPE); zeros placeholder --
    def horizon_embeddings(
        self,
        horizons: int | Sequence[int],
        *,
        h_max: int | None = None,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        real = [horizons] if isinstance(horizons, int) else list(horizons)
        h = max(real) if h_max is None else h_max
        return torch.zeros(len(real), h, self.hidden_dim, device=device, dtype=dtype)

    def forward(
        self,
        vlm_activations: torch.Tensor,
        vlm_mask: torch.Tensor | None,
        x_tau: torch.Tensor,
        tau: torch.Tensor,
        horizon_emb: torch.Tensor | None = None,
        action_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Differentiable single-step velocity prediction (training). Mirrors genesis
        ``MolmoActFlowExpert.forward``: build the shared final-layer context once, run the expert at the
        clean-at-1 flow time. ``x_tau`` is ``[B,H,D]`` or ``[K,B,H,D]`` (K MC samples looped against the
        shared context to bound memory). ``horizon_emb`` is ignored (positions come from RoPE)."""
        del horizon_emb
        if x_tau.dim() == 4:
            sample_count, batch_size, horizon, _ = x_tau.shape
        elif x_tau.dim() == 3:
            sample_count, batch_size, horizon = 1, x_tau.shape[0], x_tau.shape[1]
        else:
            raise ValueError(f"x_tau must be [B,H,D] or [K,B,H,D]; got {tuple(x_tau.shape)}")
        context = self._build_single_context(
            vlm_activations,
            vlm_mask,
            action_mask,
            seq_len=horizon,
            batch_size=batch_size,
            device=x_tau.device,
            dtype=x_tau.dtype,
        )
        if x_tau.dim() == 3:
            return self.action_expert.forward_with_context(x_tau, self._flow_time(tau), context=context)
        outs = [
            self.action_expert.forward_with_context(x_tau[k], self._flow_time(tau[k]), context=context)
            for k in range(sample_count)
        ]
        return torch.stack(outs, dim=0)

    @torch.no_grad()
    def sample(
        self,
        vlm_activations,
        vlm_mask=None,
        num_steps=None,
        num_action_steps=None,
        action_dim=None,
        action_prefix=None,
        prefix_length=None,
        num_flow_samples: int = 1,
    ):
        num_steps = num_steps or self.num_inference_steps
        horizon = num_action_steps or self.action_horizon
        full_dim = self.action_dim
        out_dim = full_dim if action_dim is None else int(action_dim)
        if out_dim > full_dim:
            raise ValueError(f"action_dim={out_dim} exceeds expert action_dim={full_dim}")
        batch_size = vlm_activations.shape[0]
        device = vlm_activations.device
        model_dtype = vlm_activations.dtype
        state_dtype = torch.float32

        context = self._build_single_context(
            vlm_activations,
            vlm_mask,
            action_mask=None,
            seq_len=horizon,
            batch_size=batch_size,
            device=device,
            dtype=model_dtype,
        )

        dim_mask = None
        if out_dim < full_dim:
            dim_mask = torch.zeros(1, 1, full_dim, dtype=state_dtype, device=device)
            dim_mask[:, :, :out_dim] = 1.0

        prefix_tensor = None
        prefix_mask = None
        if action_prefix is not None:
            if prefix_length is None:
                lengths = torch.full((batch_size,), action_prefix.shape[1], device=device, dtype=torch.long)
            elif isinstance(prefix_length, int):
                lengths = torch.full((batch_size,), prefix_length, device=device, dtype=torch.long)
            else:
                lengths = prefix_length.to(device=device, dtype=torch.long)
            prefix_source_dim = action_prefix.shape[-1]
            max_prefix = int(lengths.max().item())
            # Capability gate mirroring genesis's serve-time RTC guard: a pinned prefix
            # is inpainting the expert must have been trained for. Without this, any
            # caller could pin rows on a non-RTC expert and get silently wrong chunks.
            if max_prefix > self.rtc_max_delay_steps:
                raise ValueError(
                    f"action_prefix pins {max_prefix} rows, but this expert's checkpoint "
                    f"contract declares rtc_max_delay_steps={self.rtc_max_delay_steps}; "
                    "it was not trained to inpaint a prefix of that length."
                )
            prefix_tensor = torch.zeros(batch_size, horizon, full_dim, dtype=state_dtype, device=device)
            prefix_tensor[:, :max_prefix, :prefix_source_dim] = action_prefix[:, :max_prefix].to(
                device=device, dtype=state_dtype
            )
            if dim_mask is not None:
                prefix_tensor = prefix_tensor * dim_mask
            prefix_mask = torch.arange(horizon, device=device).view(1, horizon, 1) < lengths.view(
                batch_size, 1, 1
            )

        def _apply_prefix(state):
            if prefix_mask is None or prefix_tensor is None:
                return state
            return torch.where(prefix_mask, prefix_tensor, state)

        clean_at_0 = self.clean_at_0
        dt = -1.0 / num_steps if clean_at_0 else 1.0 / num_steps

        def _sample_once() -> torch.Tensor:
            x = torch.randn(batch_size, horizon, full_dim, dtype=state_dtype, device=device)
            if dim_mask is not None:
                x = x * dim_mask
            x = _apply_prefix(x)
            for step in range(num_steps):
                tau_value = 1.0 - step / num_steps  # noise level, 1 -> 1/N
                flow_time = tau_value if clean_at_0 else (1.0 - tau_value)
                t_tensor = torch.full((batch_size,), flow_time, dtype=model_dtype, device=device)
                v = self.action_expert.forward_with_context(x.to(model_dtype), t_tensor, context=context).to(
                    state_dtype
                )
                if dim_mask is not None:
                    v = v * dim_mask
                if prefix_mask is not None:
                    v = torch.where(prefix_mask, torch.zeros_like(v), v)
                x = x + dt * v
                if dim_mask is not None:
                    x = x * dim_mask
                x = _apply_prefix(x)
            return x

        sample_count = max(1, int(num_flow_samples or 1))
        if sample_count == 1:
            return _sample_once()
        return torch.stack([_sample_once() for _ in range(sample_count)], dim=0).mean(dim=0)


@dataclass
class DiTActionExpertArgs:
    """Configuration for the sole Isaac05 continuous-action DiT expert.

    Architecture defaults mirror the released MolmoAct2 ActionExpert. Isaac05 uses a
    catalog-wide action width and can extend the horizon without changing checkpoint
    parameter shapes. The objective is always MolmoAct2's clean-at-1 flow convention.
    """

    action_dim: int = 64
    action_horizon: int = 30
    num_layers: int = 36
    hidden_dim: int = 768
    num_heads: int = 8
    mlp_ratio: float = 4.0
    num_inference_steps: int = 10
    timestep_sampling_alpha: float = 1.5
    timestep_sampling_beta: float = 1.0
    timestep_sampling_scale: float = 0.999
    timestep_sampling_offset: float = 0.001
    train_samples_per_chunk: int = 1
    timestep_embed_dim: int = 256
    rtc_max_delay_steps: int = 0
    rtc_probability: float | None = None
    rtc_delay_sampling: Literal["uniform", "exponential", "poisson"] = "uniform"
    rtc_poisson_mean: float = 5.0
    mask_padded_action_rows: bool = False
    # action_dim=64 spans the catalog (max 54), so no chunk overflows. Keep the
    # loud-fail safety net (no silent drops). The upstream pretrained 32 dims
    # load into the first 32; dims 32..63 are fresh-init and learned during adaptation.
    drop_action_dim_overflow: bool = False
    ffn_multiple_of: int = 256
    qk_norm: bool = True
    qk_norm_eps: float = 1e-6
    rope: bool = True
    context_layer_norm: bool = True
    causal_attn: bool = False
    # WS4: batch the K flow samples in one pass by folding K into
    # cross-attention QUERY heads (GQA), keeping the VLM context K/V at batch B (not K*B). Removes the
    # serial per-sample loop. K=1 is unchanged either way. Default True (validated in the 4B VLA run).
    k_batched_cross_attn: bool = True
    # "flash_gqa" (default): FA3 GQA over FA3-varlen ("CrossVarLen") — context K/V stays flat with K (the memory
    #   win), the production path; bf16/fp16 only, so it transparently falls back to sdpa_gqa for fp32/CPU (see
    #   ActionExpertCrossAttention.forward). "sdpa_gqa": SDPA(enable_gqa) — mask-correct fp32/CPU reference that
    #   materializes K/V (memory grows with K). Both are proven equal to the serial loop (see the k-batched tests).
    k_batched_cross_attn_backend: str = "flash_gqa"

    def __post_init__(self) -> None:
        integer_fields = (
            ("action_dim", self.action_dim),
            ("action_horizon", self.action_horizon),
            ("num_layers", self.num_layers),
            ("hidden_dim", self.hidden_dim),
            ("num_heads", self.num_heads),
            ("num_inference_steps", self.num_inference_steps),
            ("train_samples_per_chunk", self.train_samples_per_chunk),
            ("timestep_embed_dim", self.timestep_embed_dim),
            ("rtc_max_delay_steps", self.rtc_max_delay_steps),
            ("ffn_multiple_of", self.ffn_multiple_of),
        )
        for field_name, value in integer_fields:
            if not isinstance(value, int) or isinstance(value, bool):
                raise ValueError(f"{field_name} must be an int.")
        boolean_fields = (
            ("mask_padded_action_rows", self.mask_padded_action_rows),
            ("drop_action_dim_overflow", self.drop_action_dim_overflow),
            ("qk_norm", self.qk_norm),
            ("rope", self.rope),
            ("context_layer_norm", self.context_layer_norm),
            ("causal_attn", self.causal_attn),
            ("k_batched_cross_attn", self.k_batched_cross_attn),
        )
        for field_name, value in boolean_fields:
            if not isinstance(value, bool):
                raise ValueError(f"{field_name} must be a bool.")
        numeric_fields = (
            ("mlp_ratio", self.mlp_ratio),
            ("qk_norm_eps", self.qk_norm_eps),
            ("timestep_sampling_alpha", self.timestep_sampling_alpha),
            ("timestep_sampling_beta", self.timestep_sampling_beta),
            ("timestep_sampling_scale", self.timestep_sampling_scale),
            ("timestep_sampling_offset", self.timestep_sampling_offset),
            ("rtc_poisson_mean", self.rtc_poisson_mean),
        )
        for field_name, value in numeric_fields:
            if not isinstance(value, Real) or isinstance(value, bool) or not math.isfinite(float(value)):
                raise ValueError(f"{field_name} must be a finite number.")
        if self.rtc_probability is not None and (
            not isinstance(self.rtc_probability, Real)
            or isinstance(self.rtc_probability, bool)
            or not math.isfinite(float(self.rtc_probability))
        ):
            raise ValueError("rtc_probability must be None or a finite number in [0, 1].")
        if self.hidden_dim < 1 or self.num_heads < 1 or self.timestep_embed_dim < 1:
            raise ValueError("hidden_dim, num_heads, and timestep_embed_dim must be >= 1.")
        if self.hidden_dim % self.num_heads != 0:
            raise ValueError(f"hidden_dim ({self.hidden_dim}) must be divisible by num_heads ({self.num_heads}).")
        if self.action_dim < 1 or self.action_horizon < 1:
            raise ValueError("action_dim and action_horizon must be >= 1.")
        if self.num_layers < 1 or self.num_inference_steps < 1:
            raise ValueError("num_layers and num_inference_steps must be >= 1.")
        if self.timestep_sampling_alpha <= 0 or self.timestep_sampling_beta <= 0:
            raise ValueError("Beta distribution parameters must be positive.")
        if self.mlp_ratio <= 0 or self.qk_norm_eps <= 0:
            raise ValueError("mlp_ratio and qk_norm_eps must be positive.")
        if self.timestep_sampling_scale <= 0 or self.timestep_sampling_offset < 0:
            raise ValueError("timestep sampling scale must be positive and offset must be non-negative.")
        if self.timestep_sampling_offset + self.timestep_sampling_scale > 1:
            raise ValueError("timestep_sampling_offset + timestep_sampling_scale must be <= 1.")
        if self.train_samples_per_chunk < 1:
            raise ValueError("train_samples_per_chunk must be >= 1.")
        if self.rtc_max_delay_steps < 0:
            raise ValueError("rtc_max_delay_steps must be >= 0.")
        if self.rtc_probability is not None and not 0.0 <= self.rtc_probability <= 1.0:
            raise ValueError("rtc_probability must be None or in [0, 1].")
        if self.rtc_max_delay_steps == 0 and self.rtc_probability not in (None, 0.0):
            raise ValueError("rtc_probability > 0 requires rtc_max_delay_steps > 0.")
        if self.rtc_delay_sampling not in ("uniform", "exponential", "poisson"):
            raise ValueError("rtc_delay_sampling must be 'uniform', 'exponential', or 'poisson'.")
        if not math.isfinite(self.rtc_poisson_mean) or self.rtc_poisson_mean <= 0:
            raise ValueError("rtc_poisson_mean must be finite and > 0.")
        if self.ffn_multiple_of < 1:
            raise ValueError("ffn_multiple_of must be >= 1.")
        if self.k_batched_cross_attn_backend not in ("flash_gqa", "sdpa_gqa"):
            raise ValueError("k_batched_cross_attn_backend must be 'flash_gqa' or 'sdpa_gqa'.")

    def to_action_expert_config(self) -> MolmoAct2ActionExpertConfig:
        return MolmoAct2ActionExpertConfig(
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            max_action_dim=self.action_dim,
            max_action_horizon=self.action_horizon,
            mlp_ratio=self.mlp_ratio,
            ffn_multiple_of=self.ffn_multiple_of,
            timestep_embed_dim=self.timestep_embed_dim,
            attn_dropout=0.0,
            dropout=0.0,
            qk_norm=self.qk_norm,
            qk_norm_eps=self.qk_norm_eps,
            rope=self.rope,
            context_layer_norm=self.context_layer_norm,
            causal_attn=self.causal_attn,
        )


def _validate_action_expert_contract(action_expert_cfg: dict[str, Any]) -> DiTActionExpertArgs:
    schema_version = action_expert_cfg.get("schema_version")
    if schema_version != DIT_ACTION_EXPERT_CONFIG_SCHEMA_VERSION:
        raise ValueError(
            "action_expert metadata must carry "
            f"schema_version={DIT_ACTION_EXPERT_CONFIG_SCHEMA_VERSION}; got {schema_version!r}. "
            "Re-export the checkpoint with the current converter."
        )
    expert_type = action_expert_cfg.get("type")
    if expert_type != "dit":
        raise ValueError(f"only the 'dit' action expert is supported; got {expert_type!r}")
    required = DIT_ACTION_EXPERT_CONFIG_V1_FIELDS
    missing = [key for key in required if key not in action_expert_cfg]
    if missing:
        raise ValueError(f"action_expert metadata is missing required fields: {missing}.")
    unexpected = sorted(set(action_expert_cfg) - set(required) - {"schema_version", "type"})
    if unexpected:
        raise ValueError(f"action_expert metadata has unexpected fields for schema v1: {unexpected}.")
    values = {key: action_expert_cfg[key] for key in required}
    try:
        return DiTActionExpertArgs(**values)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid action_expert metadata: {exc}") from exc


class DiTActionExpertHead(MolmoActExpertHead):
    """Isaac05 DiT/RTC extension of the shared MolmoAct geometry.

    Bounded port from Isaac05's owned modeling_qwen35_vla.py (DiTActionExpertHead),
    itself derived from MolmoAct2. State keys and clean-at-1 operations are retained.
    The outer legacy action_expert_type label does not select the RTC semantics.
    Training prefix sampling/loss orchestration is not implemented by this head.
    """

    expert_type = "dit"

    def __init__(self, action_expert_cfg: dict[str, Any], vlm_dim: int) -> None:
        args = _validate_action_expert_contract(action_expert_cfg)
        super().__init__(action_expert_cfg, vlm_dim)
        self.args = args

    @torch.no_grad()
    def sample(
        self,
        vlm_activations: torch.Tensor,
        vlm_mask: torch.Tensor | None = None,
        num_steps: int | None = None,
        num_action_steps: int | None = None,
        action_dim: int | None = None,
        action_prefix: torch.Tensor | None = None,
        prefix_length: int | Sequence[int] | torch.Tensor | None = None,
        num_flow_samples: int = 1,
        allow_ood_rtc_prefix: bool = False,
    ) -> torch.Tensor:
        num_steps = self.num_inference_steps if num_steps is None else int(num_steps)
        horizon = self.action_horizon if num_action_steps is None else int(num_action_steps)
        if num_steps < 1:
            raise ValueError(f"num_steps must be >= 1; got {num_steps}.")
        if horizon < 1 or horizon > self.action_horizon:
            raise ValueError(f"num_action_steps must be in [1, {self.action_horizon}]; got {horizon}.")
        full_dim = self.action_dim
        batch_size = vlm_activations.shape[0]
        device = vlm_activations.device
        model_dtype = vlm_activations.dtype
        state_dtype = torch.float32

        resolved_prefix = resolve_rtc_action_prefix(
            action_prefix=action_prefix,
            prefix_length=prefix_length,
            action_dim=action_dim,
            batch_size=batch_size,
            action_horizon=horizon,
            expert_action_dim=full_dim,
            rtc_max_delay_steps=self.args.rtc_max_delay_steps,
            rtc_probability=self.args.rtc_probability,
            device=device,
            allow_ood=bool(allow_ood_rtc_prefix),
        )
        out_dim = resolved_prefix.output_dim

        context = self._build_single_context(
            vlm_activations,
            vlm_mask,
            action_mask=None,
            seq_len=horizon,
            batch_size=batch_size,
            device=device,
            dtype=model_dtype,
        )

        dim_mask = None
        if out_dim < full_dim:
            dim_mask = torch.zeros(1, 1, full_dim, dtype=state_dtype, device=device)
            dim_mask[:, :, :out_dim] = 1.0

        prefix_tensor, prefix_mask = materialize_rtc_action_prefix(
            resolved_prefix,
            action_prefix,
            batch_size=batch_size,
            action_horizon=horizon,
            expert_action_dim=full_dim,
            device=device,
            dtype=state_dtype,
            dim_mask=dim_mask,
        )

        def velocity_fn(state: torch.Tensor, flow_time: float):
            # Preserve the released checkpoint path exactly when RTC is not requested:
            # scalar [B] timesteps and no per-action conditioning tensors.
            t_tensor = torch.full((batch_size,), flow_time, dtype=model_dtype, device=device)
            rtc_conditioning = None
            if prefix_mask is not None:
                rtc_conditioning = self.action_expert.prepare_rtc_conditioning(
                    t_tensor,
                    prefix_mask.squeeze(-1),
                )
            return self.action_expert.forward_with_context(
                state.to(model_dtype),
                t_tensor,
                context=context,
                rtc_conditioning=rtc_conditioning,
            ).to(state_dtype)

        def sample_once() -> torch.Tensor:
            x = torch.randn(batch_size, horizon, full_dim, dtype=state_dtype, device=device)
            return integrate_rtc_euler(
                x,
                num_steps=num_steps,
                velocity_fn=velocity_fn,
                prefix_tensor=prefix_tensor,
                prefix_mask=prefix_mask,
                dim_mask=dim_mask,
            )

        sample_count = max(1, int(num_flow_samples or 1))
        if sample_count == 1:
            return sample_once()
        return torch.stack([sample_once() for _ in range(sample_count)], dim=0).mean(dim=0)


ACTION_EXPERT_HEADS = {
    MolmoActExpertHead.expert_type: MolmoActExpertHead,
    DiTActionExpertHead.expert_type: DiTActionExpertHead,
}


def build_action_expert_head(action_expert_cfg, vlm_dim):
    """Construct the action-expert head selected by ``action_expert_cfg['type']``.

    Schema-v1 DiT metadata selects the native per-row clean-time RTC head.
    Legacy MolmoAct metadata retains its existing scalar-time sampling path.
    """
    if action_expert_cfg is None:
        return None
    cfg = dict(action_expert_cfg)
    head_type = cfg.get("type", "molmoact")
    if head_type != "dit" and "schema_version" in cfg:
        raise ValueError(
            f"action_expert type {head_type!r} does not define a schema_version "
            f"contract; got schema_version={cfg['schema_version']!r}."
        )
    if head_type not in ACTION_EXPERT_HEADS:
        raise ValueError(f"unknown action_expert type {head_type!r}; known: {sorted(ACTION_EXPERT_HEADS)}")
    return ACTION_EXPERT_HEADS[head_type](cfg, vlm_dim=vlm_dim)


# === Config ===


class Qwen35VLAConfig(Qwen3_5Config):
    """Native Qwen3.5-VL config extended with the two VLA blocks.

    Adds ``vector_max_states`` (proprio input width, padded to this dim before the encoder) and
    ``action_expert`` (the flow-matching head config dict consumed by ``build_action_expert_head``).
    Everything else (text_config / vision_config / token ids) is inherited from ``Qwen3_5Config``.
    """

    model_type = "qwen3_5_vla"

    def __init__(
        self,
        vector_max_states: int = 128,
        action_expert: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        # The VLA keeps lm_head and input embeddings as distinct checkpoint tensors; tying them can
        # overwrite eval-critical token embeddings when both keys load into shared storage.
        kwargs["tie_word_embeddings"] = False
        super().__init__(**kwargs)
        self.tie_word_embeddings = False
        self.vector_max_states = vector_max_states
        self.action_expert = action_expert


# The molmoact flavor (qwen3_5_2b_moevit_ps16_vla_fm_molmoact) expert geometry — matches the converted
# checkpoint's model.action_expert.action_expert.* shapes. clean_at_0=False => clean-at-1 (feed 1-tau).
DEFAULT_MOLMOACT_EXPERT_CFG: dict[str, Any] = {
    "type": "molmoact",
    "action_dim": 64,
    "action_horizon": 30,
    "num_layers": 36,
    "hidden_dim": 768,
    "num_heads": 8,
    "mlp_ratio": 4.0,
    "num_inference_steps": 10,
    "timestep_embed_dim": 256,
    "clean_at_0": False,
    "ffn_multiple_of": 256,
    "qk_norm": True,
    "qk_norm_eps": 1e-6,
    "rope": True,
    "context_layer_norm": True,
    "causal_attn": False,
}


# === Inner model: native Qwen3.5-VL + proprio encoder + action expert ===


class Qwen35VLAModel(Qwen3_5Model):
    """Native ``Qwen3_5Model`` (``visual`` + ``language_model``) plus ``vector_embedding`` + ``action_expert``.

    State-dict keys reproduce the converted checkpoint exactly: ``visual.*``, ``language_model.*``,
    ``vector_embedding.{0,2}.weight``, ``action_expert.action_expert.*`` (all under the outer ``model.`` prefix).
    """

    config_class = Qwen35VLAConfig

    def __init__(self, config: Qwen35VLAConfig) -> None:
        super().__init__(config)
        hidden = config.text_config.hidden_size
        self.vector_embedding = build_vector_encoder(config.vector_max_states, hidden)
        self.action_expert = build_action_expert_head(config.action_expert, vlm_dim=hidden)

    # -- per-modality embedders --

    def _embed_text(self, token_ids: torch.Tensor) -> torch.Tensor:
        h = self.language_model.embed_tokens(token_ids)
        # Text events are shaped (..., 1); squeeze the singleton index dim.
        if h.dim() >= 2 and h.size(-2) == 1:
            h = h[..., 0, :]
        return h

    def _embed_vector(self, vector_tokens: torch.Tensor) -> torch.Tensor:
        vt = normalize_vector_tokens(vector_tokens, self.config.vector_max_states)
        return self.vector_embedding(vt)

    def _embed_vision(self, patches: torch.Tensor, grid_thw: torch.Tensor) -> torch.Tensor:
        vcfg = self.config.vision_config
        # The InferenceStreamBuilder already patchifies to temporal tubelets (C*T*P*P) when
        # temporal_patch_size>1, so the stream payload is the native-expected width. Only tile if
        # the renderer emitted per-frame (C*P*P) rows — mirrors genesis Qwen35MoeVisionPatchEmbed.
        per_frame_dim = vcfg.in_channels * vcfg.patch_size * vcfg.patch_size
        if patches.shape[-1] == per_frame_dim:
            patches = expand_single_frame_patches_to_temporal_tubelets(
                patches,
                in_channels=vcfg.in_channels,
                patch_size=vcfg.patch_size,
                temporal_patch_size=vcfg.temporal_patch_size,
            )
        # Native Qwen3_5Model.get_image_features -> BaseModelOutputWithPooling; pooler_output is a
        # per-image tuple of merged (out_hidden_size) embeds. Concatenate back to stream order.
        feats = self.get_image_features(patches, image_grid_thw=grid_thw)
        return torch.cat(list(feats.pooler_output), dim=0)

    def embed_stream(self, tensor_stream: TensorStream) -> torch.Tensor:
        """Embed each modality in place and compact -> ``[B, T, D]`` interleaved embeddings.

        Mirrors genesis ``balanced_embed_stream`` / sglang ``embed_and_interleave``: group events by
        modality, embed each group with the matching encoder, write back, then ``compact()`` (events
        are stored in sequence order, so the concat IS the interleave).
        """
        flat_stream = tensor_stream.flat_stream()
        per_modality_stream = group_streams(flat_stream, group_fn=lambda ev: ev.type, schedule=False)
        per_modality_compact = {k: v.compact() for k, v in per_modality_stream.items()}

        # Per-event spatial grids for vision events: dims(virtual=False) == [T, H, W].
        grids: dict[Any, list[list[int]]] = defaultdict(list)
        for stream in tensor_stream.streams:
            for event in stream:
                grids[event.type].append(event.dims(virtual=False))

        embedded: dict[Any, torch.Tensor] = {}
        for stype, payload in per_modality_compact.items():
            mod_name = stype.modality.__name__
            if mod_name == "VisionType":
                grid_thw = torch.tensor(grids[stype], dtype=torch.long, device=tensor_stream.device)
                embedded[stype] = self._embed_vision(payload, grid_thw)
            elif mod_name == "VectorType":
                embedded[stype] = self._embed_vector(payload)
            else:
                embedded[stype] = self._embed_text(payload)

        embedded_ts = reconstruct_tensor_stream_from_compact_dict(tensor_stream, embedded)
        return embedded_ts.compact()  # [B, T, D]

    def forward(self, tensor_stream: TensorStream, **kwargs: Any) -> BaseModelOutputWithPast:  # type: ignore[override]
        """Run the VLM over the rendered stream and return post-final-norm activations.

        Replicates genesis ``PerceptronTransformer.forward``: interleave -> MRoPE positions ->
        next-token-prediction truncation ``[:, :-1]`` -> native hybrid decoder -> final norm.
        ``last_hidden_state`` has length ``L_model = stream_len - 1`` (= genesis ``final_activations``).
        """
        inputs_embeds = self.embed_stream(tensor_stream)  # [B, L, D]

        # MRoPE positions with genesis's "1-D rotation equivalence": only image tokens keep their
        # (t, h, w) grid; every non-spatial token (text / proprio / action / timestamp) collapses to
        # (t, t, t). Mirrors PerceptronTransformer.compute_position_embeddings. Skipping this is
        # silently wrong: raw (t, h, w) on text tokens compounds into a garbage final state.
        pos = compute_mrope_pos_tensor(tensor_stream)  # [B, L, 3]
        mod = modality_mask(tensor_stream)  # [B, L]
        not_spatial = ~((mod == VisionType.I.value) | (mod == VisionType.P.value))
        pos = pos.clone()
        pos[not_spatial] = pos[not_spatial][..., 0:1].expand(-1, pos.shape[-1])
        position_ids = pos.permute(2, 0, 1).contiguous()  # [3, B, L] (native mrope format)

        # Next-token-prediction truncation, exactly as genesis model.forward (h, pos = h[:, :-1], pos[:, :-1]).
        inputs_embeds = inputs_embeds[:, :-1]
        position_ids = position_ids[:, :, :-1]

        # Padding-aware key mask: 1 for real tokens, 0 for TextType.padding. The training collate
        # right-pads variable-length per-sample streams to a common length; batch=1 inference has no
        # padding so this is all-ones (identical to the prior behavior). Truncate `mod` to L_model to
        # match the [:, :-1] NTP shift.
        attention_mask = (mod != TextType.padding.value).to(torch.long)[:, :-1]
        out = self.language_model(
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            attention_mask=attention_mask,
            use_cache=False,
        )
        return BaseModelOutputWithPast(last_hidden_state=out.last_hidden_state)


# === Top-level model: inner VLA model + lm_head + sample_action ===


class Qwen35VLAForActionGeneration(Qwen3_5PreTrainedModel):
    """Top-level VLA model: ``model`` (Qwen35VLAModel) + ``lm_head``; exposes ``sample_action``.

    State-dict keys: ``model.*`` + ``lm_head.weight`` — exactly the converted checkpoint. ``lm_head`` is
    kept for future scene-generation (the trained checkpoint is ``include_scene_description=true``); it is
    unused by ``sample_action`` (which only needs the backbone activations).
    """

    config_class = Qwen35VLAConfig

    def __init__(self, config: Qwen35VLAConfig) -> None:
        super().__init__(config)
        self.model = Qwen35VLAModel(config)
        self.lm_head = nn.Linear(config.text_config.hidden_size, config.text_config.vocab_size, bias=False)
        self.post_init()

    def get_input_embeddings(self) -> nn.Module:
        return self.model.language_model.embed_tokens

    @property
    def action_expert(self) -> nn.Module:
        return self.model.action_expert

    @torch.no_grad()
    def sample_action(
        self,
        tensor_stream: TensorStream,
        *,
        num_steps: int | None = None,
        action_dim: int | None = None,
        num_action_steps: int | None = None,
        action_prefix: torch.Tensor | None = None,
        prefix_length: int | None = None,
        num_flow_samples: int = 1,
    ) -> torch.Tensor:
        """Integrate the flow expert into a ``[B, num_action_steps or H, action_dim]`` chunk.

        Numerically mirrors genesis ``PerceptronTransformer.sample_action``: one VLM forward over the
        stream, take the post-norm final activations (already ``[:, :-1]``-truncated by ``forward``),
        build the pre-action context mask (excludes action-marker positions), and run the expert's
        clean-at-1 Euler ODE. Output is in normalized action space (caller unnormalizes).
        """
        expert = self.model.action_expert
        if expert is None:
            raise RuntimeError(
                "sample_action requires config.action_expert to be set (this is a VLM-only checkpoint)."
            )
        output = self.model(tensor_stream)
        final = output.last_hidden_state  # [B, L_model, D]
        bsz, l_model = final.shape[0], final.shape[1]
        device = final.device

        action_start = first_event_start_indices(
            tensor_stream, FLOW_ACTION_TEXT_TYPE, fallback_start=int(l_model)
        )
        vlm_mask, _ = build_action_context_mask(
            tensor_stream,
            action_batch_indices=list(range(bsz)),
            action_start_indices=action_start,
            l_model=l_model,
            device=device,
        )
        actions = expert.sample(
            vlm_activations=final,
            vlm_mask=vlm_mask,
            num_steps=num_steps,
            num_action_steps=num_action_steps,
            action_dim=action_dim,
            action_prefix=action_prefix,
            prefix_length=prefix_length,
            num_flow_samples=num_flow_samples,
        )
        if action_dim is not None and action_dim < expert.action_dim:
            actions = actions[:, :, :action_dim].contiguous()
        return actions

    def train_forward(self, tensor_stream: TensorStream) -> Any:
        """Return differentiable native activations and heads for the policy's joint training loss.

        Evaluation uses ``sample_action``; training consumes this output through
        ``perceptron_isaac_training_loss`` without a separate backbone implementation.
        """
        out = self.model(tensor_stream)
        return ModelOutput(
            final_activations=out.last_hidden_state,
            final_embedding=self.lm_head,
            flow_matching_expert=self.model.action_expert,
        )


# === Loader for the convert_genesis_qwen35_to_hf.py output ===


def load_sharded_safetensors_into_model(
    model: nn.Module,
    checkpoint_dir: str | Path,
    *,
    strict: bool = True,
) -> tuple[list[str], list[str]]:
    """Load a safetensors checkpoint while retaining at most one shard.

    When an index exists it is authoritative: every advertised key must be in
    its advertised shard and no unindexed tensor may be present. This catches
    partial or accidentally mixed checkpoint directories before inference.
    """
    from safetensors.torch import load_file

    root = Path(checkpoint_dir)
    shard_paths, advertised_by_shard = safetensors_checkpoint_layout(root)

    expected_keys = set(model.state_dict())
    loaded_keys: set[str] = set()
    unexpected_keys: set[str] = set()
    for shard_path in shard_paths:
        if not shard_path.is_file():
            raise FileNotFoundError(f"Safetensors index references missing shard {shard_path}.")
        shard = load_file(str(shard_path))
        shard_keys = set(shard)
        if advertised_by_shard is not None:
            advertised = advertised_by_shard[shard_path.name]
            missing_from_shard = advertised - shard_keys
            unindexed_in_shard = shard_keys - advertised
            if missing_from_shard or unindexed_in_shard:
                raise ValueError(
                    f"Safetensors index mismatch for {shard_path.name}: "
                    f"missing={sorted(missing_from_shard)} unindexed={sorted(unindexed_in_shard)}."
                )
        duplicates = loaded_keys.intersection(shard_keys)
        if duplicates:
            raise ValueError(
                f"Duplicate tensor keys across safetensors shards; first: {sorted(duplicates)[:10]}."
            )
        _, shard_unexpected = model.load_state_dict(shard, strict=False)
        unexpected_keys.update(shard_unexpected)
        loaded_keys.update(shard_keys)
        del shard

    missing_keys = sorted(expected_keys - loaded_keys)
    unexpected = sorted(unexpected_keys | (loaded_keys - expected_keys))
    if strict and (missing_keys or unexpected):
        raise RuntimeError(
            "ISAAC checkpoint key coverage mismatch: "
            f"missing={len(missing_keys)} (first={missing_keys[:10]}), "
            f"unexpected={len(unexpected)} (first={unexpected[:10]})."
        )
    return missing_keys, unexpected


def _assert_offset_norm_matches_marker(hf_path: Path, *, apply_offset_norm: bool) -> None:
    """Cross-check the offset-norm flag against checkpoint-owned convention markers.

    ``rmsnorm_conversion.json`` is written only into directories whose RMSNorm weights have
    already been converted to the native unit-offset convention. Canonical trained checkpoints
    archive that import record and carry an explicit weight-convention marker. Subtracting one
    again turns every language-model norm gain into ``w - 1 ~ 0``, which produces garbage
    activations with no error, so refuse the load instead.
    """
    convention_marker = hf_path / QWEN35_WEIGHT_CONVENTION_FILE
    if convention_marker.exists() or convention_marker.is_symlink():
        verify_qwen35_weight_convention(hf_path)
    markers = (
        hf_path / QWEN35_CONVERSION_PROVENANCE_FILE,
        hf_path / QWEN35_SOURCE_CONVERSION_PROVENANCE_FILE,
        convention_marker,
    )
    already_converted = any(marker.exists() or marker.is_symlink() for marker in markers)
    if already_converted and apply_offset_norm:
        marker_names = ", ".join(marker.name for marker in markers if marker.exists() or marker.is_symlink())
        raise RuntimeError(
            f"{hf_path} carries {marker_names}, so its RMSNorm weights are already in the native "
            "unit-offset convention. Loading with apply_offset_norm=True would subtract one "
            "twice and corrupt every language-model norm; pass apply_offset_norm=False."
        )


def resolve_action_expert_config(
    file_expert_cfg: dict[str, Any] | None,
    *,
    action_expert: dict[str, Any] | None = None,
    action_expert_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Resolve the effective action-expert config for a checkpoint load.

    Precedence: a checkpoint-stamped block (``file_expert_cfg``, genesis post-#3402
    exports) is authoritative and may not be wholesale-replaced; an explicit
    ``action_expert`` dict is accepted only for checkpoints without one; otherwise
    ``DEFAULT_MOLMOACT_EXPERT_CFG`` applies. ``action_expert_overrides`` carries
    serving knobs (action_horizon / num_inference_steps) merged on top -- except that
    a horizon contradicting a checkpoint-stamped contract fails loudly, because that
    means the policy config and checkpoint disagree on trained geometry.
    """
    if action_expert is not None:
        if file_expert_cfg is not None:
            raise ValueError(
                "The checkpoint's config.json carries an authoritative action_expert "
                "contract; refusing to override it wholesale. Pass "
                "action_expert_overrides for serving knobs instead."
            )
        resolved = dict(action_expert)
    elif file_expert_cfg is not None:
        resolved = dict(file_expert_cfg)
    else:
        resolved = dict(DEFAULT_MOLMOACT_EXPERT_CFG)
    if action_expert_overrides:
        file_horizon = resolved.get("action_horizon")
        wanted_horizon = action_expert_overrides.get("action_horizon")
        if (
            file_expert_cfg is not None
            and wanted_horizon is not None
            and file_horizon is not None
            and int(wanted_horizon) != int(file_horizon)
        ):
            raise ValueError(
                f"Policy config implies action_horizon={wanted_horizon}, but the "
                f"checkpoint's action_expert contract declares {file_horizon}; the "
                "policy config and checkpoint disagree on trained geometry."
            )
        resolved = {**resolved, **action_expert_overrides}
    return resolved


def load_qwen35_vla_from_hf(
    hf_dir: str | Path,
    *,
    vector_max_states: int = 128,
    action_expert: dict[str, Any] | None = None,
    action_expert_overrides: dict[str, Any] | None = None,
    dtype: torch.dtype = torch.float32,
    device: str = "cpu",
    strict: bool = True,
    apply_offset_norm: bool = True,
) -> tuple[Qwen35VLAForActionGeneration, Qwen35VLAConfig]:
    """Load raw upstream HF weights or a stable, fully verified imported package snapshot."""
    hf_path = Path(hf_dir)
    package_root = hf_path.parent
    load_options = {
        "vector_max_states": vector_max_states,
        "action_expert": action_expert,
        "action_expert_overrides": action_expert_overrides,
        "dtype": dtype,
        "device": device,
        "strict": strict,
        "apply_offset_norm": apply_offset_norm,
    }

    from .trained_package import QWEN35_TRAINED_PACKAGE_MANIFEST_FILENAME

    if (package_root / QWEN35_TRAINED_PACKAGE_MANIFEST_FILENAME).is_file():

        def verify_trained_snapshot(snapshot_root: Path) -> tuple[str | Path, bool]:
            from lerobot.configs import PreTrainedConfig

            from .modeling_perceptron_isaac import PerceptronIsaacPolicy

            saved_config = PreTrainedConfig.from_pretrained(snapshot_root)
            PerceptronIsaacPolicy._resolve_checkpoint_local_paths(saved_config, snapshot_root)
            PerceptronIsaacPolicy._verify_qwen35_trained_package_manifest(saved_config, snapshot_root)
            return saved_config.hf_model_path, bool(saved_config.apply_offset_norm)

        verified = _snapshot_verified_qwen35_trained_package(
            package_root,
            hf_model_dir=hf_path,
            verify_snapshot=verify_trained_snapshot,
        )
        return _load_qwen35_vla_from_verified_package(verified, **load_options)

    has_import_provenance = (package_root / QWEN35_IMPORT_PROVENANCE_FILE).is_file()
    has_conversion_marker = (hf_path / QWEN35_CONVERSION_PROVENANCE_FILE).is_file()
    parent_config_path = package_root / "config.json"
    has_policy_config = False
    if parent_config_path.is_file():
        try:
            has_policy_config = load_json_object(parent_config_path).get("type") == "perceptron_isaac"
        except ValueError as exc:
            raise RuntimeError(f"Cannot inspect parent policy config: {exc}") from exc
    has_package_identity = (package_root / "isaac_deployment_adapter.json").is_file() or any(
        (package_root / filename).is_file() for filename in QWEN35_PACKAGED_SOURCE_FILENAMES
    )
    if has_import_provenance or has_conversion_marker or has_package_identity or has_policy_config:
        verified = _snapshot_verified_qwen35_imported_package(
            package_root,
            hf_model_dir=hf_path,
        )
        return _load_qwen35_vla_from_verified_package(verified, **load_options)
    detached_markers = [
        name
        for name in (
            QWEN35_SOURCE_CONVERSION_PROVENANCE_FILE,
            QWEN35_WEIGHT_CONVENTION_FILE,
        )
        if (hf_path / name).exists() or (hf_path / name).is_symlink()
    ]
    if detached_markers:
        if QWEN35_WEIGHT_CONVENTION_FILE in detached_markers:
            verify_qwen35_weight_convention(hf_path)
        raise RuntimeError(
            "Detached canonical Qwen3.5 hf_model cannot be loaded without its authenticated "
            f"outer trained package; found markers={detached_markers}. Load the package root instead."
        )
    return _load_qwen35_vla_from_hf(hf_path, **load_options)


def _load_qwen35_vla_from_verified_package(
    verified: _VerifiedQwen35Package,
    *,
    vector_max_states: int,
    action_expert: dict[str, Any] | None,
    action_expert_overrides: dict[str, Any] | None,
    dtype: torch.dtype,
    device: str,
    strict: bool,
    apply_offset_norm: bool,
) -> tuple[Qwen35VLAForActionGeneration, Qwen35VLAConfig]:
    """Consume one exact-root verified snapshot and release it after model materialization."""
    try:
        hf_model_path = _verified_qwen35_model_path(verified)
        if verified.apply_offset_norm != apply_offset_norm:
            raise RuntimeError(
                "Verified Qwen3.5 package apply_offset_norm disagrees with the loader request."
            )
        return _load_qwen35_vla_from_hf(
            hf_model_path,
            vector_max_states=vector_max_states,
            action_expert=action_expert,
            action_expert_overrides=action_expert_overrides,
            dtype=dtype,
            device=device,
            strict=strict,
            apply_offset_norm=apply_offset_norm,
        )
    finally:
        verified.cleanup()


def _load_qwen35_vla_from_hf(
    hf_dir: str | Path,
    *,
    vector_max_states: int,
    action_expert: dict[str, Any] | None,
    action_expert_overrides: dict[str, Any] | None,
    dtype: torch.dtype,
    device: str,
    strict: bool,
    apply_offset_norm: bool,
) -> tuple[Qwen35VLAForActionGeneration, Qwen35VLAConfig]:
    """Instantiate ``Qwen35VLAForActionGeneration`` and load the converter's safetensors shards.

    Historical converters write a *native* ``Qwen3_5Config`` (model_type ``qwen3_5``, no
    action_expert block), for which ``DEFAULT_MOLMOACT_EXPERT_CFG`` supplies the expert
    geometry. Genesis post-#3402 exports stamp an authoritative ``action_expert`` block
    (type "dit" + schema_version) into config.json; that block is honored, never
    silently replaced. ``action_expert_overrides`` carries serving knobs
    (action_horizon / num_inference_steps) applied on top of whichever base wins.
    Weights load with ``strict=False`` (the VLM/expert/proprio keys must all match; we
    assert nothing is missing).
    """
    hf_path = Path(hf_dir).resolve()
    # Load WITHOUT an action_expert kwarg so a checkpoint-stamped block survives as
    # the authoritative contract (an HF from_pretrained kwarg would wholesale-replace
    # it -- exactly the silent-discard failure this loader used to have).
    config = Qwen35VLAConfig.from_pretrained(
        hf_path,
        vector_max_states=vector_max_states,
        tie_word_embeddings=False,
    )
    config.action_expert = resolve_action_expert_config(
        getattr(config, "action_expert", None),
        action_expert=action_expert,
        action_expert_overrides=action_expert_overrides,
    )

    # Construct parameters directly in their requested storage dtype. This
    # avoids a second full fp32 model allocation during inference and matches
    # the validated Accelerate loader's cast-before-RMSNorm-correction order.
    original_default_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        model = Qwen35VLAForActionGeneration(config)
    finally:
        torch.set_default_dtype(original_default_dtype)
    if model.lm_head.weight.data_ptr() == model.get_input_embeddings().weight.data_ptr():
        raise RuntimeError("Qwen35VLA checkpoint load requires untied lm_head and input embeddings.")

    load_sharded_safetensors_into_model(model, hf_path, strict=strict)

    # convert_genesis_qwen35_to_hf writes standard (genesis) RMSNorm weights; native Qwen3.5 applies
    # the unit-offset (1 + weight) convention, so correct the language-model norms once here. A
    # finetuned-and-resaved checkpoint is ALREADY in native convention — load it with
    # apply_offset_norm=False (re-applying the -1 would corrupt every language-model norm).
    #
    # The importer drops rmsnorm_conversion.json next to converted shards precisely so this is
    # detectable: the correction is not idempotent, so honor the marker over the flag rather
    # than silently subtracting one twice.
    _assert_offset_norm_matches_marker(Path(hf_path), apply_offset_norm=apply_offset_norm)
    if apply_offset_norm:
        apply_qwen35_offset_norm_correction(model.model.language_model)

    model = model.to(device=device, dtype=dtype)
    model.eval()
    return model, config


def setup_qwen35_vla_for_training(
    model: Qwen35VLAForActionGeneration,
    *,
    train_expert_only: bool = False,
    train_samples_per_chunk: int = 4,
    rtc_max_delay_steps: int = 0,
    rtc_delay_sampling: str = "uniform",
    dual_timestep_ratio: float = 0.0,
    mask_padded_action_rows: bool = True,
) -> Qwen35VLAForActionGeneration:
    """Put a loaded VLA into training mode and set the action expert's genesis-flow-loss args.

    - ``model.train()`` + ``requires_grad`` (``train_expert_only`` freezes everything but
      ``model.action_expert.*``; otherwise the whole VLA trains).
    - tunes ``expert.args`` (the train-only knobs the genesis flow loss reads) to the finetune recipe;
      defaults keep RTC disabled for the HF MolmoAct wrapper while retaining 4 MC samples and pad masking.

    Does NOT touch the offset-norm correction (applied once at load).
    """
    expert = model.model.action_expert
    if expert is not None and hasattr(expert, "args"):
        expert.args.train_samples_per_chunk = int(train_samples_per_chunk)
        expert.args.rtc_max_delay_steps = int(rtc_max_delay_steps)
        expert.args.rtc_delay_sampling = str(rtc_delay_sampling)
        expert.args.dual_timestep_ratio = float(dual_timestep_ratio)
        expert.args.mask_padded_action_rows = bool(mask_padded_action_rows)
    model.train()
    if train_expert_only:
        for name, param in model.named_parameters():
            param.requires_grad_(name.startswith("model.action_expert."))
        if not any(p.requires_grad for p in model.parameters()):
            raise RuntimeError("train_expert_only=True but found no model.action_expert.* parameters")
    else:
        model.requires_grad_(True)
    return model
