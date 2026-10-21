"""Native F32 initialization of the Isaac-0.5 backbone for dense FSDP2 training (CPU and Gloo only).

Dense FP32 FSDP2 training of an F32-storage Isaac-0.5 package must:
  R1  hold every trainable parameter as the exact F32 source bits (no BF16 round trip);
  R2  hold them as views of the package's own shard-file mappings (no per-rank anonymous copy),
      never writing the read-only source;
  R3  keep bit-exact FP32 masters in both root- and child-owned FSDP2 units after accelerate's
      real ``fsdp2_prepare_model``;
  R4  leave nothing for accelerate's FSDP2 upcast to copy (it upcasts every trainable that is not
      FP32 before ``fully_shard``, accelerate 1.15 ``utils/fsdp_utils.py`` ``fsdp2_prepare_model``);
  R5  select GDN kernels and gate the production CUDA ABI on the RUN device, even though the
      backbone is materialized in host memory;
  R6  let accelerate's real FSDP2 remap (``Accelerator._prepare_fsdp2``, keyed by each parameter's
      pre-preparation ``data_ptr``) rebind the optimizer built before preparation -- as
      ``lerobot_train`` builds it -- so each original parameter maps once to its own prepared
      shard, every group keeps its hyperparameters, and the optimizer state covers every shard.

"Bits" means the raw F32 bit patterns (an int32 view), so -0.0 and +0.0 differ.

What CPU tests cannot prove: the CUDA branch of FSDP2's ``_move_states_to_device`` (source reading
only), fused-kernel availability or numerics on a GPU, or memory accounting on a cluster node.
Everything else that does not change is pinned by the controls at the end.
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
import subprocess
import sys
import traceback
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

pytest.importorskip("transformers", reason="transformers is required (install lerobot[perceptron_isaac])")

from accelerate import Accelerator  # noqa: E402

from lerobot.policies.perceptron_isaac import modeling_mk1_vla, modeling_perceptron_isaac  # noqa: E402
from lerobot.policies.perceptron_isaac.modeling_perceptron_isaac import (  # noqa: E402
    FullShardTopology,
    PerceptronIsaacPolicy,
    inspect_fsdp2_trainable_masters,
)
from lerobot.policies.perceptron_isaac.modeling_qwen36_moe import GenesisQwen36GatedDeltaNet  # noqa: E402
from tests.policies.perceptron_isaac.test_portable_model_repo import (  # noqa: E402
    _install_distributed_state,
    _native_portable_package,
    _StubAcceleratorState,
    _StubFsdpPlugin,
)

SOURCE_SEED = 20261006
WORLD_SIZE = 2
LINUX_PROC = Path("/proc/self/maps").exists()
# Worker stages, in order. A worker that stops short of the stage a test needs is INFRASTRUCTURE.
STAGES = ("init", "loaded", "prepared", "checked", "stepped")
# accelerate's own remap failures (accelerator.py _prepare_fsdp2, utils/fsdp_utils.py
# fsdp2_switch_optimizer_parameters): an assertion about the remap, not infrastructure.
REMAP_ERRORS = ("couldn't be switched", "missing after FSDP2 wrapping", "FSDP2 mapping failed")


# --------------------------------------------------------------------------- bit comparison


def _f32_bits(tensor: torch.Tensor) -> torch.Tensor:
    """The raw IEEE-754 bit patterns of an F32 tensor, as int32 (so -0.0 and +0.0 differ)."""
    assert tensor.dtype is torch.float32, f"bit comparison needs F32, got {tensor.dtype}"
    return tensor.detach().contiguous().view(torch.int32)


def _same_f32_bits(actual: torch.Tensor, expected: torch.Tensor) -> bool:
    """Whether ``actual`` is F32 with exactly ``expected``'s shape and bit patterns."""
    return (
        actual.dtype is torch.float32
        and actual.shape == expected.shape
        and torch.equal(_f32_bits(actual), _f32_bits(expected))
    )


NEGATIVE_ZERO_BITS = int(_f32_bits(torch.tensor([-0.0]))[0])


# --------------------------------------------------------------------------- fixture helpers


def _randomize_f32_sources(model_dir: Path) -> dict[str, torch.Tensor]:
    """Rewrite each F32 source tensor with seeded values BF16 cannot represent; return them.

    Every tensor of three or more elements also starts with -0.0, +0.0, which numeric equality
    cannot tell apart. Shapes, dtypes and therefore the index total_size and the artifact
    tensor_bytes are unchanged.
    """
    from safetensors.torch import load_file, save_file

    generator = torch.Generator().manual_seed(SOURCE_SEED)
    sources: dict[str, torch.Tensor] = {}
    for shard in sorted(model_dir.glob("model-*.safetensors")):
        fresh = {}
        for key, tensor in sorted(load_file(shard).items()):
            assert tensor.dtype is torch.float32, f"fixture source {key} is not F32"
            value = 0.12345679 + 0.01 * torch.randn(tensor.shape, generator=generator)
            if value.numel() >= 3:  # signed zeros, still beside values BF16 cannot represent
                value.view(-1)[:2] = torch.tensor([-0.0, 0.0])
            fresh[key] = value
        save_file(fresh, shard)
        sources.update({key: value.clone() for key, value in fresh.items()})
    lossy = [key for key, value in sources.items() if _same_f32_bits(value.bfloat16().float(), value)]
    assert not lossy, f"sources must not be BF16-representable: {lossy[:5]}"
    signed = [key for key, value in sources.items() if bool((_f32_bits(value) == NEGATIVE_ZERO_BITS).any())]
    assert len(signed) > len(sources) // 2, f"too few sources carry -0.0: {len(signed)} of {len(sources)}"
    return sources


def _package_digests(model_dir: Path) -> dict[str, str]:
    """SHA-256 of every package file outside the policy subdirectory."""
    return {
        str(path.relative_to(model_dir)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(model_dir.rglob("*"))
        if path.is_file() and "lerobot_policy" not in path.relative_to(model_dir).parts
    }


def _make_read_only(model_dir: Path) -> None:
    for path in model_dir.rglob("*"):
        if path.is_file() and "lerobot_policy" not in path.relative_to(model_dir).parts:
            path.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)


def _f32_package(tmp_path: Path):
    """The reduced portable Isaac-0.5 F32 package with randomized sources, dense FP32 config."""
    policy_path, config, _ = _native_portable_package(tmp_path)
    model_dir = policy_path.parent
    sources = _randomize_f32_sources(model_dir)
    config.train_expert_only = False
    config.use_peft = False
    config.train_storage_fp32 = True
    # Full-parameter scope, as the port35 launch sets it (--policy.freeze_input_embeddings=false).
    config.freeze_input_embeddings = False
    return policy_path, config, model_dir, sources


def _live_fsdp2(monkeypatch: pytest.MonkeyPatch) -> None:
    from accelerate.utils.dataclasses import DistributedType

    _install_distributed_state(
        monkeypatch,
        world_size=16,
        state=_StubAcceleratorState(
            distributed_type=DistributedType.FSDP,
            fsdp_plugin=_StubFsdpPlugin(fsdp_version=2, sharding_strategy=None, reshard_after_forward=True),
        ),
    )


def _fsdp2_dense_policy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, run_device: str = "cpu", read_only: bool = False
):
    """Load the backbone the way lerobot_train does under live FSDP2: get_optim_params before prepare."""
    policy_path, config, model_dir, sources = _f32_package(tmp_path)
    if read_only:
        _make_read_only(model_dir)
    policy = PerceptronIsaacPolicy.from_pretrained(policy_path, config=config, local_files_only=True)
    policy.config.device = run_device
    _live_fsdp2(monkeypatch)
    policy.get_optim_params()
    return policy, model_dir, sources


class _LoaderStoppedError(Exception):
    """Raised by the recording loader once it has recorded a call it was told not to run."""


def _record_loader_calls(monkeypatch: pytest.MonkeyPatch, *, stop: bool = False) -> list[dict]:
    """Record the keyword arguments the policy passes to the MK1 loader (then load, or stop)."""
    real_loader = modeling_mk1_vla.load_mk1_vla_from_hf
    calls: list[dict] = []

    def recording_loader(*args: object, **kwargs: object):
        calls.append(dict(kwargs))
        if stop:
            raise _LoaderStoppedError("recorded")
        return real_loader(*args, **kwargs)

    monkeypatch.setattr(modeling_mk1_vla, "load_mk1_vla_from_hf", recording_loader)
    return calls


def _gdn_modules(model: torch.nn.Module) -> list[GenesisQwen36GatedDeltaNet]:
    modules = [module for module in model.modules() if isinstance(module, GenesisQwen36GatedDeltaNet)]
    assert modules, "the reduced fixture must contain gated-delta-net layers"
    return modules


def _uses_torch_gdn_fallbacks(module: torch.nn.Module) -> bool:
    return (
        module.causal_conv1d_update is modeling_mk1_vla.torch_causal_conv1d_update
        and module.chunk_gated_delta_rule is modeling_mk1_vla.torch_chunk_gated_delta_rule
    )


def _file_mappings(paths: set[str]) -> list[tuple[int, int]]:
    """Address ranges of this process's mappings of the given files (/proc/self/maps)."""
    ranges = []
    for line in Path("/proc/self/maps").read_text().splitlines():
        fields = line.split(maxsplit=5)
        if len(fields) == 6 and os.path.realpath(fields[5]) in paths:
            start, end = (int(bound, 16) for bound in fields[0].split("-"))
            ranges.append((start, end))
    return ranges


# --------------------------------------------------------------------------- R1 numeric preservation


def test_fsdp2_dense_training_holds_the_exact_f32_source_bits(tmp_path, monkeypatch) -> None:
    """R1: every source tensor is a trainable FP32 parameter equal bit-for-bit to its source."""
    policy, _, sources = _fsdp2_dense_policy(tmp_path, monkeypatch)

    params = dict(policy._isaac_model.named_parameters())
    assert set(params) == set(sources), "the dense trainable set is the whole source inventory"
    assert all(param.requires_grad for param in params.values()), "dense: nothing is frozen"
    wrong_dtype = sorted(name for name, param in params.items() if param.dtype is not torch.float32)
    assert not wrong_dtype, f"trainables not held as F32: {len(wrong_dtype)}, first {wrong_dtype[:3]}"
    unequal = sorted(name for name, param in params.items() if not _same_f32_bits(param, sources[name]))
    assert not unequal, f"trainables differ from their F32 source bits: {len(unequal)}, first {unequal[:3]}"
    upcast = [name for name, p in policy.named_parameters() if p.requires_grad and p.dtype != torch.float32]
    assert upcast == [], f"accelerate would copy {len(upcast)} parameters to FP32, first {upcast[:3]}"


def test_the_bit_comparison_tells_signed_zeros_apart() -> None:
    """Control: torch.equal calls -0.0 and +0.0 equal; the F32 bit comparison R1/R3/R6 use does not."""
    negative, positive = torch.tensor([-0.0, 1.0]), torch.tensor([0.0, 1.0])

    assert torch.equal(negative, positive)
    assert not _same_f32_bits(negative, positive)
    assert _same_f32_bits(negative, negative.clone())


# --------------------------------------------------------------------------- R2 no copy, read-only source


@pytest.mark.skipif(not LINUX_PROC, reason="needs /proc/self/maps")
def test_fsdp2_dense_training_parameters_are_views_of_the_package_shard_files(tmp_path, monkeypatch) -> None:
    """R2 (deterministic): every parameter's bytes lie inside a mapping of one of the shard files."""
    policy, model_dir, _ = _fsdp2_dense_policy(tmp_path, monkeypatch, read_only=True)
    shards = {os.path.realpath(path) for path in model_dir.glob("model-*.safetensors")}
    ranges = _file_mappings(shards)

    outside = []
    for name, param in policy._isaac_model.named_parameters():
        start = param.data_ptr()
        end = start + param.numel() * param.element_size()
        if not any(low <= start and end <= high for low, high in ranges):
            outside.append(name)
    assert ranges, "no shard file is mapped: the parameters were copied out of the package"
    assert not outside, f"parameters held outside the shard mappings: {len(outside)}, first {outside[:3]}"


def test_fsdp2_dense_training_never_writes_the_read_only_package(tmp_path, monkeypatch) -> None:
    """R2 control: load, then write into a parameter in place; the 0444 package bytes are unchanged."""
    policy_path, config, model_dir, _ = _f32_package(tmp_path)
    _make_read_only(model_dir)
    before = _package_digests(model_dir)
    policy = PerceptronIsaacPolicy.from_pretrained(policy_path, config=config, local_files_only=True)
    _live_fsdp2(monkeypatch)
    policy.get_optim_params()

    with torch.no_grad():
        for param in policy._isaac_model.parameters():
            param.add_(1.0)

    assert _package_digests(model_dir) == before
    shard_modes = {stat.S_IMODE(path.stat().st_mode) for path in model_dir.glob("model-*.safetensors")}
    assert shard_modes == {0o444}


_RSS_PROBE = r"""
import json, sys
from pathlib import Path
import torch
from accelerate import init_empty_weights
from lerobot.policies.perceptron_isaac.mk1_checkpoint_contract import read_and_validate_safetensors_index
from lerobot.policies.perceptron_isaac.modeling_mk1_vla import _stream_checkpoint_tensors

def anon():
    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith("RssAnon:"):
            return int(line.split()[1]) * 1024
    raise RuntimeError("no RssAnon")

def module(numel):
    with init_empty_weights(include_buffers=False):
        holder = torch.nn.Module()
        holder.w = torch.nn.Parameter(torch.empty(numel))
    return holder

root, numel = Path(sys.argv[1]), int(sys.argv[3])
dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16}[sys.argv[2]]
inventory = read_and_validate_safetensors_index(root)
warm = module(numel)  # warm-up: the same primitive and code paths, F32 (file-backed), then released
_stream_checkpoint_tensors(warm, inventory, expected_keys={"w"}, device="cpu", dtype=torch.float32)
float(warm.w.sum())
del warm
target = module(numel)
before = anon()
_stream_checkpoint_tensors(target, inventory, expected_keys={"w"}, device="cpu", dtype=dtype)
float(target.w.sum())
print(json.dumps({"anon_delta": anon() - before, "f32_bytes": numel * 4}))
"""


def _rss_package(root: Path, numel: int) -> None:
    from safetensors.torch import save_file

    root.mkdir()
    save_file({"w": torch.randn(numel)}, str(root / "model-00001-of-00001.safetensors"))
    index = {"metadata": {"total_size": numel * 4}, "weight_map": {"w": "model-00001-of-00001.safetensors"}}
    (root / "model.safetensors.index.json").write_text(json.dumps(index))


def _fresh_process_anon_delta(root: Path, dtype: str, numel: int) -> dict:
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1"}
    result = subprocess.run(
        [sys.executable, "-c", _RSS_PROBE, str(root), dtype, str(numel)],
        capture_output=True,
        text=True,
        timeout=300,
        env=env,
        check=False,
    )
    assert result.returncode == 0, result.stderr[-2000:]
    return json.loads(result.stdout.strip().splitlines()[-1])


@pytest.mark.skipif(not LINUX_PROC, reason="needs /proc/self/status")
def test_the_f32_streaming_primitive_adds_no_anonymous_copy_in_a_fresh_process(tmp_path) -> None:
    """R2 (isolated characterization of the unchanged loader primitive, with a copying control).

    A fresh process per measurement, warmed up through the same primitive. The F32 stream must add
    under 10% of the tensor bytes; the BF16 stream -- the pre-candidate MK1 call -- must add at least
    40% (a BF16 copy is 50%), proving the measurement detects a copy.
    """
    numel = 16 * 1024 * 1024  # 64 MiB of F32: well above allocator noise
    root = tmp_path / "rss"
    _rss_package(root, numel)

    native = _fresh_process_anon_delta(root, "float32", numel)
    copying = _fresh_process_anon_delta(root, "bfloat16", numel)

    assert copying["anon_delta"] >= 0.4 * copying["f32_bytes"], f"positive control did not copy: {copying}"
    assert native["anon_delta"] < 0.1 * native["f32_bytes"], f"F32 stream copied: {native}"


# --------------------------------------------------------------------------- R5 run-device kernels and ABI


def test_fsdp2_host_build_keeps_the_run_device_gdn_kernels(tmp_path, monkeypatch) -> None:
    """R5 (behavior): built in host memory for a CUDA run, the GDN layers keep their own kernels.

    CPU-only wiring proof; whether the fused kernels exist and match numerically on a GPU is not
    tested here.
    """
    policy, _, _ = _fsdp2_dense_policy(tmp_path, monkeypatch, run_device="cuda")

    swapped = [module for module in _gdn_modules(policy._isaac_model) if _uses_torch_gdn_fallbacks(module)]
    assert not swapped, f"{len(swapped)} GDN layers were switched to the torch CPU fallbacks"


def test_fsdp2_policy_passes_the_run_device_to_the_loader(tmp_path, monkeypatch) -> None:
    """R5 (wiring): host materialization device and the run device reach the loader separately."""
    policy_path, config, _, _ = _f32_package(tmp_path)
    policy = PerceptronIsaacPolicy.from_pretrained(policy_path, config=config, local_files_only=True)
    policy.config.device = "cuda"
    calls = _record_loader_calls(monkeypatch)
    _live_fsdp2(monkeypatch)

    policy.get_optim_params()

    assert [(call["device"], call.get("runtime_device")) for call in calls] == [("cpu", "cuda")]
    assert (calls[0]["dtype"], calls[0].get("load_intent")) == (torch.float32, "training")


def _production_loader_spies(monkeypatch, tmp_path):
    from tests.policies.perceptron_isaac.test_modeling_mk1_vla import _mk1_config, _production_mk1_contract

    raw = _mk1_config(test_geometry=True)
    contract = _production_mk1_contract(raw)
    reads: list[object] = []

    def unexpected_read(*args, **kwargs):
        reads.append(args)
        raise AssertionError("safetensors must not be read before the qualified Torch ABI gate")

    monkeypatch.setattr(modeling_mk1_vla, "read_mk1_config_data", lambda *a, **k: (raw, contract))
    monkeypatch.setattr(modeling_mk1_vla, "read_and_validate_safetensors_index", unexpected_read)
    monkeypatch.setattr(modeling_mk1_vla.torch, "__version__", "2.11.0+cu128")
    monkeypatch.setattr(modeling_mk1_vla.torch.version, "cuda", "12.8")
    return reads


def test_host_materialization_for_a_cuda_run_refuses_an_unqualified_production_abi(tmp_path, monkeypatch):
    """R5 (production negative): materialized on cpu for a cuda run, a wrong Torch ABI is refused
    before any safetensors read. The neutral/reduced bypass does not apply: the contract is production.
    """
    reads = _production_loader_spies(monkeypatch, tmp_path)

    with pytest.raises(RuntimeError, match="requires.*parity-qualified"):
        modeling_mk1_vla.load_mk1_vla_from_hf(
            tmp_path / "config-is-mocked",
            dtype=torch.float32,
            device="cpu",
            runtime_device="cuda",
            allowed_storage_dtypes=frozenset({"F32"}),
            load_intent="training",
        )
    assert reads == []


# --------------------------------------------------------------------------- R3 real fsdp2_prepare_model


class _Fsdp2PrepareHost(SimpleNamespace):
    """Stands in for ``Accelerator`` in ``_prepare_fsdp2``: the plain fields it reads, plus accelerate's
    own parameter walk (``_get_named_parameters``, which keys the remap by ``data_ptr``)."""

    _get_named_parameters = Accelerator._get_named_parameters


def _prepare_one_unwrapped(obj: object, first_pass: bool = False) -> object:
    """Stands in for ``Accelerator._prepare_one``: no AcceleratedOptimizer wrapper, no scheduler."""
    return obj


def _join_live_fsdp2_rank(rank: int, init_file: str) -> None:
    """Join the Gloo group as one rank of a live FSDP2 bf16 run; accelerate's state is real."""
    from accelerate import PartialState

    dist.init_process_group(
        "gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=WORLD_SIZE,
        timeout=timedelta(seconds=120),
    )
    # MULTI_CPU over the Gloo group above, as fsdp2_prepare_model's logging requires.
    os.environ.update(
        RANK=str(rank), WORLD_SIZE=str(WORLD_SIZE), LOCAL_RANK=str(rank), LOCAL_WORLD_SIZE=str(WORLD_SIZE)
    )
    state = PartialState(cpu=True, backend="gloo")
    assert state.num_processes == WORLD_SIZE and state.process_index == rank
    modeling_perceptron_isaac.detect_full_shard_topology = lambda: FullShardTopology(
        world_size=WORLD_SIZE, fsdp_version=2, sharding_strategy="reshard_after_forward=True"
    )
    modeling_perceptron_isaac.read_accelerate_mixed_precision = lambda: "bf16"


def _fsdp2_accelerator_fields(
    host: type[SimpleNamespace] = SimpleNamespace, **members: object
) -> SimpleNamespace:
    """The accelerator fields accelerate's FSDP2 preparation reads, with LeRobot's own plugin."""
    from accelerate import PartialState
    from accelerate.utils.dataclasses import DynamoBackend
    from torch.distributed.device_mesh import init_device_mesh

    from lerobot.configs.accelerator import FSDPConfig

    plugin = FSDPConfig().build_plugin()
    plugin.set_mixed_precision("bf16")
    return host(
        state=SimpleNamespace(fsdp_plugin=plugin, dynamo_plugin=SimpleNamespace(backend=DynamoBackend.NO)),
        torch_device_mesh=init_device_mesh("cpu", (WORLD_SIZE,), mesh_dim_names=("dp_shard_cp",)),
        parallelism_config=SimpleNamespace(fsdp_dim_names=("dp_shard_cp",)),
        mixed_precision="bf16",
        is_main_process=PartialState().is_main_process,
        device=torch.device("cpu"),
        **members,
    )


def _gathered(param: torch.Tensor) -> torch.Tensor:
    """The full value of a (possibly sharded) parameter; a collective, so call it in rank order."""
    from torch.distributed.tensor import DTensor

    return param.full_tensor() if isinstance(param, DTensor) else param.detach()


def _prepare_worker(rank: int, policy_path: str, config, sources_path: str, init_file: str, out_dir: str):
    outcome: dict = {"stage": "init"}
    _join_live_fsdp2_rank(rank, init_file)
    try:
        from accelerate.utils.fsdp_utils import fsdp2_prepare_model
        from torch.distributed.fsdp import FSDPModule
        from torch.distributed.tensor import DTensor

        from lerobot.distributed.factory import set_fsdp_wrap_modules

        sources = torch.load(sources_path)
        policy = PerceptronIsaacPolicy.from_pretrained(policy_path, config=config, local_files_only=True)
        policy.get_optim_params()
        outcome["stage"] = "loaded"

        accelerator = _fsdp2_accelerator_fields()
        set_fsdp_wrap_modules(accelerator, policy)
        prepared = fsdp2_prepare_model(accelerator, policy)  # accelerate's real FSDP2 preparation
        outcome["stage"] = "prepared"

        child_prefixes = [
            name + "." for name, module in prepared.named_modules() if name and isinstance(module, FSDPModule)
        ]
        report = inspect_fsdp2_trainable_masters(prepared)
        try:
            PerceptronIsaacPolicy._verify_fp32_masters_once(prepared)
            outcome["verify_error"] = None
        except RuntimeError as exc:
            outcome["verify_error"] = str(exc)[:300]
        root_trainables = child_trainables = 0
        local_dtypes: set[str] = set()
        mismatched: list[str] = []
        for name, param in prepared.named_parameters():  # same order on every rank: collectives align
            if not param.requires_grad:
                continue
            if any(name.startswith(prefix) for prefix in child_prefixes):
                child_trainables += 1
            else:
                root_trainables += 1
            local = param.to_local() if isinstance(param, DTensor) else param
            local_dtypes.add(str(local.dtype))
            full = _gathered(param)
            key = name.removeprefix("_isaac_model.")
            if key not in sources or not _same_f32_bits(full, sources[key]):
                mismatched.append(name)
        outcome.update(
            stage="checked",
            units=report.units,
            covered=len(report.trainable_param_names),
            uncovered=list(report.uncovered_trainable_names),
            low_precision=list(report.low_precision),
            root_trainables=root_trainables,
            child_trainables=child_trainables,
            trainables=root_trainables + child_trainables,
            local_dtypes=sorted(local_dtypes),
            mismatched_count=len(mismatched),
            mismatched_first=mismatched[:3],
            source_count=len(sources),
        )
    except Exception:  # noqa: BLE001 -- recorded so the parent can tell infrastructure from assertion
        outcome["error"] = traceback.format_exc()[-3000:]
    finally:
        Path(out_dir, f"rank{rank}.json").write_text(json.dumps(outcome))
        dist.destroy_process_group()


def _remap_worker(rank: int, policy_path: str, config, sources_path: str, init_file: str, out_dir: str):
    """R6: lerobot_train's order, then accelerate's real FSDP2 branch of prepare (``_prepare_fsdp2``)."""
    outcome: dict = {"stage": "init"}
    _join_live_fsdp2_rank(rank, init_file)
    try:
        from accelerate.utils.fsdp_utils import fsdp2_canonicalize_names
        from torch.distributed.tensor import DTensor

        from lerobot.distributed.factory import set_fsdp_wrap_modules

        sources = torch.load(sources_path)
        policy = PerceptronIsaacPolicy.from_pretrained(policy_path, config=config, local_files_only=True)
        # make_optimizer_and_scheduler with the policy preset, on the unsharded parameters.
        optimizer = policy.config.get_optimizer_preset().build(policy.get_optim_params())
        outcome["stage"] = "loaded"

        accelerator = _fsdp2_accelerator_fields(
            _Fsdp2PrepareHost, fp8_backend=None, _models=[], _prepare_one=_prepare_one_unwrapped
        )
        set_fsdp_wrap_modules(accelerator, policy)
        names = {id(param): name for name, param in policy.named_parameters()}
        layout = [[names[id(param)] for param in group["params"]] for group in optimizer.param_groups]
        hyper_before = [{k: v for k, v in group.items() if k != "params"} for group in optimizer.param_groups]
        # The same walk _prepare_fsdp2 keys its remap with, taken just before it runs.
        old_pointers = list(
            fsdp2_canonicalize_names(accelerator._get_named_parameters(policy, drop_refs=True)).values()
        )
        outcome.update(
            old_count=len(old_pointers),
            old_unique=len(set(old_pointers)),
            old_zero=old_pointers.count(0),
            source_count=len(sources),
        )

        prepared, optimizer = Accelerator._prepare_fsdp2(accelerator, policy, optimizer)
        outcome["stage"] = "prepared"

        new_named = fsdp2_canonicalize_names(dict(prepared.named_parameters()))
        trainable_ids = {id(param) for param in new_named.values() if param.requires_grad}
        groups = optimizer.param_groups
        slots = [
            (param, name)
            for group, row in zip(groups, layout, strict=True)
            for param, name in zip(group["params"], row, strict=False)
        ]
        flat_ids = [id(param) for param, _ in slots]
        outcome.update(
            group_sizes=[len(group["params"]) for group in groups],
            layout_sizes=[len(row) for row in layout],
            misplaced=[name for param, name in slots if param is not new_named.get(name)][:3],
            misplaced_count=sum(param is not new_named.get(name) for param, name in slots),
            each_once=len(set(flat_ids)) == len(flat_ids),
            covers_trainables=set(flat_ids) == trainable_ids,
            trainables=len(trainable_ids),
            not_f32_shards=[
                name
                for param, name in slots
                if not (isinstance(param, DTensor) and param.to_local().dtype is torch.float32)
            ][:3],
            hyper_unchanged=[{k: v for k, v in group.items() if k != "params"} for group in groups]
            == hyper_before,
        )
        mismatched = [
            name
            for param, name in slots
            if not _same_f32_bits(_gathered(param), sources[name.removeprefix("_isaac_model.")])
        ]
        outcome.update(stage="checked", mismatched_count=len(mismatched), mismatched_first=mismatched[:3])

        for param, _ in slots:
            param.grad = torch.zeros_like(param)
        optimizer.step()
        outcome["stage"] = "stepped"

        state_ids = {id(param) for param in optimizer.state}
        state_errors = [
            f"{name}:{key}"
            for param, name in slots
            for key, value in optimizer.state.get(param, {}).items()
            if isinstance(value, torch.Tensor)
            and value.dim() > 0
            and not (
                isinstance(value, DTensor)
                and value.placements == param.placements
                and value.to_local().shape == param.to_local().shape
            )
        ]
        outcome.update(
            state_count=len(state_ids),
            state_covers=state_ids == set(flat_ids),
            state_keys=sorted({key for entry in optimizer.state.values() for key in entry}),
            state_error_count=len(state_errors),
            state_errors=state_errors[:3],
        )
    except Exception:  # noqa: BLE001 -- recorded so the parent can tell infrastructure from assertion
        outcome["error"] = traceback.format_exc()[-3000:]
    finally:
        Path(out_dir, f"rank{rank}.json").write_text(json.dumps(outcome))
        dist.destroy_process_group()


def _run_ranks(worker, tmp_path: Path) -> dict:
    """Two Gloo ranks run ``worker`` on the same read-only F32 package."""
    policy_path, config, model_dir, sources = _f32_package(tmp_path)
    _make_read_only(model_dir)
    before = _package_digests(model_dir)
    sources_path = tmp_path / "sources.pt"
    torch.save(sources, sources_path)
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("OMP_NUM_THREADS", "1")
        patch.setenv("CUDA_VISIBLE_DEVICES", "")
        mp.spawn(
            worker,
            args=(str(policy_path), config, str(sources_path), str(tmp_path / "init"), str(tmp_path)),
            nprocs=WORLD_SIZE,
            join=True,
        )
    ranks = [json.loads((tmp_path / f"rank{rank}.json").read_text()) for rank in range(WORLD_SIZE)]
    return {"ranks": ranks, "source_unchanged": _package_digests(model_dir) == before}


@pytest.fixture(scope="module")
def prepared_outcome(tmp_path_factory) -> dict:
    """Two Gloo ranks load the package, then run accelerate's fsdp2_prepare_model."""
    return _run_ranks(_prepare_worker, tmp_path_factory.mktemp("fsdp2_prepare"))


@pytest.fixture(scope="module")
def remapped_outcome(tmp_path_factory) -> dict:
    """Two Gloo ranks build the optimizer, then run accelerate's _prepare_fsdp2 and one step."""
    return _run_ranks(_remap_worker, tmp_path_factory.mktemp("fsdp2_remap"))


def _require_harness(outcome: dict, needed: str = "checked") -> None:
    """Infrastructure gate: a worker that stopped before ``needed`` says nothing about the assertion."""
    for rank, row in enumerate(outcome["ranks"]):
        stage = row.get("stage")
        assert stage in STAGES and STAGES.index(stage) >= STAGES.index(needed), (
            f"INFRASTRUCTURE rank {rank} stopped at {stage} (needs {needed}): " + str(row.get("error"))
        )


def test_prepared_fsdp2_units_cover_root_and_child_trainables_with_fp32_masters(prepared_outcome) -> None:
    """R3 control: harness works; every trainable is covered, root- and child-owned, FP32 shards."""
    _require_harness(prepared_outcome)
    for row in prepared_outcome["ranks"]:
        assert row["verify_error"] is None
        assert row["uncovered"] == [] and row["low_precision"] == []
        assert row["covered"] == row["trainables"] == row["source_count"]
        assert row["root_trainables"] > 0 and row["child_trainables"] > 0
        assert row["local_dtypes"] == ["torch.float32"]
    assert prepared_outcome["source_unchanged"]


def test_prepared_fsdp2_masters_equal_the_f32_source_bits(prepared_outcome) -> None:
    """R3: after accelerate's real preparation, every gathered master equals its F32 source."""
    _require_harness(prepared_outcome)
    for row in prepared_outcome["ranks"]:
        assert row["mismatched_count"] == 0, (
            f"{row['mismatched_count']} of {row['trainables']} masters differ from the F32 source, "
            f"first {row['mismatched_first']}"
        )


# --------------------------------------------------------------------------- R6 optimizer remap


def test_accelerate_fsdp2_remap_silently_merges_parameters_that_share_an_old_pointer() -> None:
    """Control for R6: accelerate keys its remap by each parameter's old ``data_ptr``, so parameters at
    one address -- meta tensors are all at 0 -- are silently rebound to a single new parameter. R6
    therefore requires unique, nonzero old pointers.
    """
    from accelerate.utils.fsdp_utils import fsdp2_switch_optimizer_parameters

    with torch.device("meta"):
        old = torch.nn.Linear(2, 2)
    old_pointers = _Fsdp2PrepareHost(fp8_backend=None)._get_named_parameters(old, drop_refs=True)
    new = dict(torch.nn.Linear(2, 2).named_parameters())
    mapping = {pointer: new[name] for name, pointer in old_pointers.items()}  # as _prepare_fsdp2 builds it
    optimizer = SimpleNamespace(
        param_groups=[{"params": [SimpleNamespace(data_ptr=p) for p in old_pointers.values()]}]
    )

    fsdp2_switch_optimizer_parameters(optimizer, mapping)  # accelerate's own switch: no error

    assert set(old_pointers.values()) == {0}
    rebound = optimizer.param_groups[0]["params"]
    assert len(rebound) == 2 and rebound[0] is rebound[1], "two parameters were rebound to one"


def _require_no_remap_error(outcome: dict) -> None:
    for rank, row in enumerate(outcome["ranks"]):
        error = str(row.get("error") or "")
        assert not any(marker in error for marker in REMAP_ERRORS), (
            f"rank {rank}: accelerate's remap failed: {error}"
        )


def test_optimizer_remap_maps_each_original_parameter_once_to_its_prepared_shard(remapped_outcome) -> None:
    """R6 (characterization): every original name, at its own nonzero pointer, is rebound to exactly
    its own prepared FP32 shard, in its own group, with the group hyperparameters intact."""
    _require_no_remap_error(remapped_outcome)
    _require_harness(remapped_outcome, "checked")
    for row in remapped_outcome["ranks"]:
        assert row["old_count"] == row["old_unique"] == row["source_count"] and row["old_zero"] == 0, row
        assert row["group_sizes"] == row["layout_sizes"] and sum(row["group_sizes"]) == row["source_count"]
        assert row["misplaced_count"] == 0, f"rebound to another parameter: {row['misplaced']}"
        assert row["each_once"] and row["covers_trainables"] and row["trainables"] == row["source_count"]
        assert row["not_f32_shards"] == [] and row["hyper_unchanged"]
    assert remapped_outcome["source_unchanged"]


def test_remapped_optimizer_state_covers_every_prepared_shard(remapped_outcome) -> None:
    """R6 (characterization): after one step, the AdamW state is keyed by exactly the remapped shards,
    each moment sharded like its parameter."""
    _require_no_remap_error(remapped_outcome)
    _require_harness(remapped_outcome, "stepped")
    for row in remapped_outcome["ranks"]:
        assert row["state_covers"] and row["state_count"] == row["source_count"]
        assert {"exp_avg", "exp_avg_sq", "step"} <= set(row["state_keys"])
        assert row["state_error_count"] == 0, f"state not sharded like its parameter: {row['state_errors']}"


def test_remapped_optimizer_parameters_hold_the_f32_source_bits(remapped_outcome) -> None:
    """R6: the parameters the optimizer will update are the exact F32 source bits."""
    _require_no_remap_error(remapped_outcome)
    _require_harness(remapped_outcome, "checked")
    for row in remapped_outcome["ranks"]:
        assert row["mismatched_count"] == 0, (
            f"{row['mismatched_count']} of {row['source_count']} optimizer parameters differ from the F32 "
            f"source, first {row['mismatched_first']}"
        )


# --------------------------------------------------------------------------- controls (unchanged)


def test_loader_default_keeps_the_cpu_abi_exemption(tmp_path, monkeypatch) -> None:
    """Control: without a run device, a cpu load stays exempt from the CUDA ABI gate."""
    reads = _production_loader_spies(monkeypatch, tmp_path)

    with pytest.raises(AssertionError, match="safetensors must not be read"):
        modeling_mk1_vla.load_mk1_vla_from_hf(tmp_path / "config-is-mocked", device="cpu")
    assert len(reads) == 1, "the cpu load passed the ABI gate and reached the safetensors read"


def test_loader_default_cuda_load_still_refuses_an_unqualified_abi(tmp_path, monkeypatch) -> None:
    """Control: a cuda load without a separate run device is gated exactly as before."""
    reads = _production_loader_spies(monkeypatch, tmp_path)

    with pytest.raises(RuntimeError, match="requires.*parity-qualified"):
        modeling_mk1_vla.load_mk1_vla_from_hf(tmp_path / "config-is-mocked", device="cuda")
    assert reads == []


def test_loader_default_cpu_load_keeps_the_torch_gdn_fallbacks(tmp_path) -> None:
    """Control: a plain cpu load (no run device) still runs GDN on the torch CPU fallbacks."""
    from tests.policies.perceptron_isaac.test_mk1_checkpoint_contract import (
        _write_portable_isaac05_checkpoint,
    )

    root = tmp_path / "portable"
    _write_portable_isaac05_checkpoint(root)
    model, _, _ = modeling_mk1_vla.load_mk1_vla_from_hf(
        root,
        dtype=torch.float32,
        device="cpu",
        allow_test_only_reduced_geometry=True,
        allowed_storage_dtypes=frozenset({"F32"}),
        load_intent="training",
    )

    assert all(_uses_torch_gdn_fallbacks(module) for module in _gdn_modules(model))


def test_inference_load_is_unchanged(tmp_path, monkeypatch) -> None:
    """Control: inference loads BF16 with inference intent on the run device."""
    policy_path, config, _, _ = _f32_package(tmp_path)
    policy = PerceptronIsaacPolicy.from_pretrained(policy_path, config=config, local_files_only=True)
    calls = _record_loader_calls(monkeypatch)

    policy._load_backbone(training=False)

    call = calls[0]
    assert (call["dtype"], call.get("load_intent", "inference"), call["device"]) == (
        torch.bfloat16,
        "inference",
        "cpu",
    )
    assert call.get("runtime_device", call["device"]) == call["device"]
    assert {p.dtype for p in policy._isaac_model.parameters()} == {torch.bfloat16}


@pytest.mark.parametrize(("train_expert_only", "use_peft"), [(True, False), (False, True)])
def test_deferred_training_modes_refuse_before_backbone_load(
    tmp_path, monkeypatch, train_expert_only: bool, use_peft: bool
) -> None:
    policy_path, config, _, _ = _f32_package(tmp_path)
    policy = PerceptronIsaacPolicy.from_pretrained(policy_path, config=config, local_files_only=True)
    config.train_expert_only = train_expert_only
    config.use_peft = use_peft
    calls = _record_loader_calls(monkeypatch)
    _live_fsdp2(monkeypatch)

    with pytest.raises(RuntimeError, match="PEFT and expert-only training are unsupported"):
        policy.get_optim_params()

    assert calls == []


def test_fsdp1_training_refuses_before_backbone_load(tmp_path, monkeypatch) -> None:
    from accelerate.utils.dataclasses import DistributedType
    from torch.distributed.fsdp import ShardingStrategy

    policy_path, config, _, _ = _f32_package(tmp_path)
    policy = PerceptronIsaacPolicy.from_pretrained(policy_path, config=config, local_files_only=True)
    calls = _record_loader_calls(monkeypatch)
    _install_distributed_state(
        monkeypatch,
        world_size=16,
        state=_StubAcceleratorState(
            distributed_type=DistributedType.FSDP,
            fsdp_plugin=_StubFsdpPlugin(
                fsdp_version=1,
                sharding_strategy=ShardingStrategy.FULL_SHARD,
                reshard_after_forward=True,
            ),
            mixed_precision="bf16",
        ),
    )

    with pytest.raises(RuntimeError, match="FSDP1 training is unsupported"):
        policy.get_optim_params()

    assert calls == []


def test_bf16_storage_package_fsdp2_training_load_is_unchanged(tmp_path, monkeypatch) -> None:
    """Control: a BF16-storage MK1 package keeps the BF16 load (no lossless F32 source exists).

    Only the storage classification the load decision reads is stubbed to BF16; the preflight still
    validates the fixture's real F32 bytes. The loader call is recorded and stopped.
    """
    from lerobot.policies.perceptron_isaac import mk1_checkpoint_contract

    policy_path, config, _, _ = _f32_package(tmp_path)
    policy = PerceptronIsaacPolicy.from_pretrained(policy_path, config=config, local_files_only=True)
    real_validate = mk1_checkpoint_contract.validate_mk1_checkpoint
    monkeypatch.setattr(
        mk1_checkpoint_contract,
        "validate_mk1_checkpoint",
        lambda root, **kwargs: real_validate(
            root, **{**kwargs, "allowed_storage_dtypes": frozenset({"F32"})}
        ),
    )
    monkeypatch.setattr(modeling_perceptron_isaac, "_package_storage_dtypes", lambda _: frozenset({"BF16"}))
    calls = _record_loader_calls(monkeypatch, stop=True)
    _live_fsdp2(monkeypatch)

    with pytest.raises(_LoaderStoppedError, match="recorded"):
        policy.get_optim_params()

    assert (calls[0]["dtype"], calls[0].get("load_intent", "inference")) == (torch.bfloat16, "inference")
    assert calls[0]["allowed_storage_dtypes"] == frozenset({"BF16"})
