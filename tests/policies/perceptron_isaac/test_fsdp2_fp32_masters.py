"""FP32-master verification under live FSDP2 (2-process gloo on CPU, bf16 MixedPrecisionPolicy).

The check runs inside the wrapped forward, as it does in ``PerceptronIsaacPolicy.forward``. At
that point the root unit's parameters are swapped to their BF16 all-gathered compute views and
child units are still sharded, so it must read the stored shard masters, not the module's
attributes. Covers root- and child-owned trainables, the BF16 root compute view seen while the
check passes, genuine BF16 masters in the root and in a child unit, a trainable that no unit owns
(FSDP2 ``ignored_params``), a frozen BF16 parameter, and zero coverage.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn

pytest.importorskip("transformers", reason="transformers is required (install lerobot[perceptron_isaac])")

from lerobot.policies.perceptron_isaac import modeling_perceptron_isaac  # noqa: E402
from lerobot.policies.perceptron_isaac.modeling_perceptron_isaac import (  # noqa: E402
    FullShardTopology,
    PerceptronIsaacPolicy,
    inspect_fsdp2_trainable_masters,
)

WORLD_SIZE = 2


class _Block(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(8, 8)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin(x)


class _ToyPolicy(nn.Module):
    """Root-owned trainable head, one child FSDP unit, and a frozen BF16 root parameter."""

    def __init__(self) -> None:
        super().__init__()
        self.head = nn.Linear(8, 8)
        self.block = _Block()
        self.frozen_table = nn.Parameter(torch.zeros(4, 8, dtype=torch.bfloat16), requires_grad=False)
        self.config = SimpleNamespace(train_storage_fp32=True)
        self._fp32_masters_verified = False
        self.outcome: dict = {}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # What the module's own attributes hold at check time: the root's are its compute views.
        attribute_dtypes = {name: str(param.dtype) for name, param in self.named_parameters()}
        report = inspect_fsdp2_trainable_masters(self)
        errors: list[str | None] = []
        for _attempt in range(2):
            try:
                PerceptronIsaacPolicy._verify_fp32_masters_once(self)
                errors.append(None)
            except RuntimeError as exc:
                errors.append(str(exc))
        self.outcome = {
            "error": errors[0],
            "retry_error": errors[1],
            "units": report.units,
            "covered": sorted(report.trainable_param_names),
            "attribute_dtypes": attribute_dtypes,
        }
        return self.block(self.head(x))


def _fsdp2_topology() -> FullShardTopology:
    return FullShardTopology(
        world_size=WORLD_SIZE, fsdp_version=2, sharding_strategy="reshard_after_forward=True"
    )


def _worker(rank: int, case: str, init_file: str, out_dir: str) -> None:
    dist.init_process_group("gloo", init_method=f"file://{init_file}", rank=rank, world_size=WORLD_SIZE)
    try:
        from torch.distributed.device_mesh import init_device_mesh
        from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard

        modeling_perceptron_isaac.detect_full_shard_topology = _fsdp2_topology
        torch.manual_seed(0)
        policy = _ToyPolicy()
        ignored: set[nn.Parameter] = set()
        if case == "bf16_child_master":
            policy.block.to(torch.bfloat16)
        if case == "bf16_root_master":
            policy.head.to(torch.bfloat16)
        if case == "uncovered_trainable":
            # A trainable root parameter that FSDP2 is told not to manage: no unit owns it.
            policy.side = nn.Parameter(torch.zeros(8))
            ignored = {policy.side}
        if case != "unsharded":
            mesh = init_device_mesh("cpu", (WORLD_SIZE,))
            mp_policy = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
            fully_shard(policy.block, mesh=mesh, mp_policy=mp_policy)
            # accelerate shards the root with reshard_after_forward left at its False default.
            fully_shard(
                policy, mesh=mesh, mp_policy=mp_policy, reshard_after_forward=False, ignored_params=ignored
            )
        policy(torch.randn(2, 8)).float().sum().backward()
        Path(out_dir, f"rank{rank}.json").write_text(json.dumps(policy.outcome))
    finally:
        dist.destroy_process_group()


def _run(case: str, tmp_path: Path) -> list[dict]:
    mp.spawn(_worker, args=(case, str(tmp_path / "init"), str(tmp_path)), nprocs=WORLD_SIZE, join=True)
    return [json.loads((tmp_path / f"rank{rank}.json").read_text()) for rank in range(WORLD_SIZE)]


@pytest.fixture(autouse=True)
def _cpu_threads(monkeypatch):
    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    yield


def test_fp32_masters_pass_inside_the_mixed_precision_forward_with_root_and_child_coverage(tmp_path):
    for outcome in _run("fp32_masters", tmp_path):
        assert outcome["error"] is None
        assert outcome["units"] == 2
        assert {"head.weight", "head.bias", "block.lin.weight", "block.lin.bias"} == set(outcome["covered"])


def test_the_check_passes_while_the_root_attributes_are_bf16_compute_views(tmp_path):
    """The pass must come from the FP32 masters: the attributes it would otherwise read are BF16."""
    for outcome in _run("fp32_masters", tmp_path):
        assert outcome["error"] is None
        assert outcome["attribute_dtypes"]["head.weight"] == "torch.bfloat16"
        assert outcome["attribute_dtypes"]["head.bias"] == "torch.bfloat16"


def test_a_bf16_trainable_shard_master_is_rejected(tmp_path):
    for outcome in _run("bf16_child_master", tmp_path):
        assert outcome["error"] is not None
        assert "block.lin.weight" in outcome["error"]
        assert "FP32" in outcome["error"]
        assert outcome["retry_error"] == outcome["error"], "a failed check must not pass on retry"


def test_a_bf16_root_owned_trainable_master_is_rejected(tmp_path):
    for outcome in _run("bf16_root_master", tmp_path):
        assert outcome["error"] is not None
        assert "head.weight" in outcome["error"]
        assert "FP32" in outcome["error"]
        assert outcome["retry_error"] == outcome["error"], "a failed check must not pass on retry"


def test_a_trainable_owned_by_no_unit_is_rejected_even_when_the_rest_are_covered(tmp_path):
    for outcome in _run("uncovered_trainable", tmp_path):
        assert {"head.weight", "head.bias", "block.lin.weight", "block.lin.bias"} == set(outcome["covered"])
        assert outcome["error"] is not None
        assert "not owned by any" in outcome["error"]
        assert "side" in outcome["error"]
        assert outcome["retry_error"] == outcome["error"], "a failed check must not pass on retry"


def test_zero_shard_coverage_under_an_fsdp2_topology_is_rejected(tmp_path):
    for outcome in _run("unsharded", tmp_path):
        assert outcome["covered"] == []
        assert outcome["error"] is not None
        assert "no FSDP2" in outcome["error"]
