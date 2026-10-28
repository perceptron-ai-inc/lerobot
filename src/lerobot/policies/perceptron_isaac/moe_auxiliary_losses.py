"""Genesis logical-null router statistics and global training auxiliaries.

Dispatch counts sum top-k lanes. One physical null row represents multiple
logical slots. Only forwarded positions matter, not text or action loss masks.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn

from .loss_perceptron_isaac import _global_objective_mean, _global_sum_detached


@dataclass(frozen=True)
class RouterAuxiliaryStatistics:
    physical_logits: Tensor
    logical_probabilities: Tensor
    selected_experts: Tensor
    num_real_experts: int
    num_null_experts: int


class GenesisNullRouterAuxiliary(nn.Module):
    """Parameter-free output for Transformers' maintained per-forward output recorder.

    Serving does no auxiliary tensor work and keeps no references between calls.
    Training materializes statistics only after the complete backbone forward.
    """

    def __init__(self, num_real_experts: int, num_null_experts: int) -> None:
        super().__init__()
        assert num_real_experts > 0 and num_null_experts > 0
        self.num_real_experts = num_real_experts
        self.num_null_experts = num_null_experts

    def forward(
        self, physical_logits: Tensor, logical_probabilities: Tensor, selected_experts: Tensor
    ) -> RouterAuxiliaryStatistics:
        return RouterAuxiliaryStatistics(
            physical_logits,
            logical_probabilities,
            selected_experts,
            self.num_real_experts,
            self.num_null_experts,
        )


def materialize_router_auxiliaries(
    layers: tuple[RouterAuxiliaryStatistics, ...],
) -> dict[str, list[Tensor]]:
    """Reduce dispatch counts before forming each layer's differentiable global loss."""
    result: dict[str, list[Tensor]] = {
        "load_balancing_loss": [],
        "router_z_loss": [],
        "router_dispatch_counts": [],
        "router_token_count": [],
    }
    for layer in layers:
        logical_experts = layer.num_real_experts + layer.num_null_experts
        local_counts = torch.zeros(logical_experts, device=layer.selected_experts.device, dtype=torch.float32)
        local_counts.scatter_add_(
            0,
            layer.selected_experts.detach().reshape(-1),
            torch.ones(
                layer.selected_experts.numel(), device=layer.selected_experts.device, dtype=torch.float32
            ),
        )
        counts = _global_sum_detached(local_counts)
        logits = layer.physical_logits.float()
        local_token_count = logits.new_tensor(float(logits.shape[0]))
        token_count = _global_sum_detached(local_token_count)
        if not bool(token_count > 0):
            raise ValueError("MoE auxiliaries require at least one forwarded token globally.")
        balance_sum = logical_experts * ((counts / token_count) * layer.logical_probabilities.sum(0)).sum()
        # The shared null row contributes num_null_experts copies to the partition.
        partition_logits = torch.cat(
            (
                logits[:, : layer.num_real_experts],
                logits[:, layer.num_real_experts :] + math.log(layer.num_null_experts),
            ),
            dim=-1,
        )
        z_sum = partition_logits.logsumexp(-1).square().sum()
        balance, _, _ = _global_objective_mean(balance_sum, local_token_count, global_denominator=token_count)
        z, _, _ = _global_objective_mean(z_sum, local_token_count, global_denominator=token_count)
        result["load_balancing_loss"].append(balance)
        result["router_z_loss"].append(z)
        result["router_dispatch_counts"].append(counts)
        result["router_token_count"].append(token_count)
    return result
