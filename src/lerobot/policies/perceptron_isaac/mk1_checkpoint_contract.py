"""Allocation-free validation for the MK1 Fast VLA checkpoint contract.

This module treats ``hf_model/config.json`` and the safetensors index as data.
It does not import Transformers, construct a model, or read tensor payloads.
Only the single MK1/PR-3154 export ABI is accepted for production. A reduced
geometry is available solely for explicitly marked CPU test fixtures.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

MK1_MODEL_TYPE = "qwen3_5_moe"
MK1_TEXT_MODEL_TYPE = "qwen3_5_moe_text"
MK1_ARCHITECTURE = "GenesisQwen3_5MoeForConditionalGeneration"
MK1_VLA_SCHEMA_VERSION = 1
MK1_BACKBONE_FAMILY = "mk1_qwen3_6_moe"
MK1_STATE_DICT_SCHEMA = "pr3154_v1"
MK1_RMSNORM_CONVENTION = "zero_centered_1_plus_weight"
MK1_TEST_GEOMETRY_MARKER = "genesis_test_only_reduced_geometry"
MK1_SAFETENSORS_INDEX = "model.safetensors.index.json"
MK1_FAST_TOKENIZER = "physical-intelligence/fast"
MK1_FAST_TOKEN_OFFSET = 249321
MK1_FAST_TOKEN_SIZE = 2048
_MAX_JSON_FILE_BYTES = 64 * 1024 * 1024
_MAX_SAFETENSORS_HEADER_BYTES = 64 * 1024 * 1024

_SHARD_NAME = re.compile(r"model-(\d{5})-of-(\d{5})\.safetensors")
_NON_MODEL_SAFETENSORS_SIDECARS = {"policy_state_identity.safetensors"}
_CUSTOM_MOE_KEYS = {
    "router_contract_version",
    "router_contract",
    "num_real_experts",
    "num_null_experts",
    "physical_router_outputs",
    "logical_router_outputs",
    "shared_null_router_row",
    "top_k",
    "score_func",
    "route_norm",
    "route_scale",
    "score_before_experts",
    "null_expert_semantics",
    "shared_expert_mode",
    "uses_expert_bias",
}
_VLA_KEYS = {
    "schema_version",
    "backbone_family",
    "state_dict_schema",
    "rmsnorm_weight_convention",
    "vector_encoder",
    "action_expert",
    "mtp",
}
_VECTOR_KEYS = {"type", "max_states", "hidden_dim", "output_dim", "bias"}
_ACTION_KEYS = {
    "type",
    "action_dim",
    "action_horizon",
    "num_layers",
    "hidden_dim",
    "num_heads",
    "mlp_ratio",
    "timestep_embed_dim",
    "clean_at_0",
    "ffn_multiple_of",
    "qk_norm",
    "qk_norm_eps",
    "rope",
    "context_layer_norm",
    "causal_attn",
}
_MTP_KEYS = {"present", "physical_layers", "rollout_steps", "action_runtime"}
_COORD_ENABLED_KEYS = {"enabled", "offset", "size"}
_COORD_DISABLED_KEYS = {"enabled"}
_ROPE_KEYS = {
    "mrope_interleaved",
    "mrope_section",
    "partial_rotary_factor",
    "rope_theta",
    "rope_type",
}
_ARTIFACT_IDENTITY_KEYS = {"schema_version", "artifact_kind", "trained_steps"}


class Mk1CheckpointContractError(ValueError):
    """The checkpoint cannot be interpreted as the allowlisted MK1 VLA ABI."""


@dataclass(frozen=True)
class Mk1MoeContract:
    num_real_experts: int
    num_null_experts: int
    physical_router_outputs: int
    logical_router_outputs: int
    top_k: int
    route_scale: float


@dataclass(frozen=True)
class Mk1TextContract:
    hidden_size: int
    num_hidden_layers: int
    vocab_size: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    moe_intermediate_size: int
    shared_expert_intermediate_size: int
    layer_types: tuple[str, ...]
    full_attention_interval: int
    linear_conv_kernel_dim: int
    linear_key_head_dim: int
    linear_num_key_heads: int
    linear_num_value_heads: int
    linear_value_head_dim: int


@dataclass(frozen=True)
class Mk1VisionContract:
    depth: int
    hidden_size: int
    intermediate_size: int
    num_heads: int
    num_position_embeddings: int
    out_hidden_size: int
    in_channels: int
    patch_size: int
    spatial_merge_size: int
    temporal_patch_size: int


@dataclass(frozen=True)
class Mk1CoordTokenContract:
    enabled: bool
    offset: int | None = None
    size: int | None = None


@dataclass(frozen=True)
class Mk1SpecialTokenContract:
    bos_token_id: int
    eos_token_id: int
    image_token_id: int
    video_token_id: int
    vision_start_token_id: int
    vision_end_token_id: int


@dataclass(frozen=True)
class Mk1VectorEncoderContract:
    max_states: int
    hidden_dim: int
    output_dim: int


@dataclass(frozen=True)
class Mk1ActionExpertContract:
    action_dim: int
    action_horizon: int
    num_layers: int
    hidden_dim: int
    num_heads: int
    mlp_ratio: float
    timestep_embed_dim: int
    ffn_multiple_of: int

    @property
    def intermediate_size(self) -> int:
        unrounded = int(self.hidden_dim * self.mlp_ratio)
        return int(math.ceil(unrounded / self.ffn_multiple_of) * self.ffn_multiple_of)


@dataclass(frozen=True)
class Mk1MtpContract:
    present: bool
    physical_layers: int
    rollout_steps: int

    @property
    def excluded_runtime_prefixes(self) -> tuple[str, ...]:
        return ("mtp.",) if self.present else ()


@dataclass(frozen=True)
class Mk1ArtifactContract:
    schema_version: int
    artifact_kind: str
    trained_steps: int


@dataclass(frozen=True)
class TensorHeader:
    key: str
    shard: str
    shape: tuple[int, ...]
    dtype: str


@dataclass(frozen=True)
class SafetensorsShardIdentity:
    """Filesystem identity pinned while a safetensors shard is validated."""

    path: Path
    resolved_path: Path
    device: int
    inode: int
    size: int
    mtime_ns: int
    ctime_ns: int


@dataclass(frozen=True)
class SafetensorsInventory:
    index_path: Path
    shards: tuple[Path, ...]
    shard_identities: Mapping[Path, SafetensorsShardIdentity]
    tensors: Mapping[str, TensorHeader]

    @property
    def keys(self) -> frozenset[str]:
        return frozenset(self.tensors)

    def verify_shard_identity(self, shard: Path) -> None:
        """Reject a shard that changed since its headers were validated."""
        expected = self.shard_identities.get(shard)
        if expected is None:
            _fail(f"safetensors shard {shard} is not part of the validated inventory")
        actual = _capture_shard_identity(shard)
        if actual != expected:
            _fail(
                f"safetensors shard {shard.name} changed after validation: "
                f"expected={_format_shard_identity(expected)}, "
                f"actual={_format_shard_identity(actual)}"
            )


@dataclass(frozen=True)
class Mk1CheckpointContract:
    artifact: Mk1ArtifactContract
    text: Mk1TextContract
    vision: Mk1VisionContract
    moe: Mk1MoeContract
    special_tokens: Mk1SpecialTokenContract
    coord_tokens: Mk1CoordTokenContract
    vector_encoder: Mk1VectorEncoderContract
    action_expert: Mk1ActionExpertContract
    mtp: Mk1MtpContract
    test_only_reduced_geometry: bool = False

    @property
    def excluded_runtime_prefixes(self) -> tuple[str, ...]:
        return self.mtp.excluded_runtime_prefixes

    @property
    def requires_fast_reserved_token_group(self) -> bool:
        return not self.test_only_reduced_geometry and self.artifact.artifact_kind == "trained_policy"

    @classmethod
    def parse_allowlisted(
        cls,
        raw_config: Mapping[str, Any],
        *,
        allow_test_only_reduced_geometry: bool = False,
    ) -> Mk1CheckpointContract:
        """Parse config data without invoking checkpoint-supplied Python.

        Reduced geometry requires two independent opt-ins: the serialized test
        marker and this call-site flag. The normal production entry point can
        therefore never accept a tiny checkpoint accidentally.
        """
        root = _object(raw_config, "config")
        test_marker = root.get(MK1_TEST_GEOMETRY_MARKER, False)
        if test_marker is not False and test_marker is not True:
            _fail(f"{MK1_TEST_GEOMETRY_MARKER} must be a boolean")
        if test_marker and not allow_test_only_reduced_geometry:
            _fail("reduced checkpoint geometry is test-only and was not explicitly enabled")
        test_geometry = bool(test_marker)

        if "apply_offset_norm" in root:
            _fail("MK1 config must not contain the legacy apply_offset_norm override")
        if root.get("model_type") != MK1_MODEL_TYPE:
            _fail(f"model_type must be {MK1_MODEL_TYPE!r}")
        if root.get("architectures") != [MK1_ARCHITECTURE]:
            _fail(f"architectures must be [{MK1_ARCHITECTURE!r}]")
        if root.get("tie_word_embeddings") is not False:
            _fail("tie_word_embeddings must be false")

        artifact = _parse_artifact(_object_field(root, "genesis_artifact"))
        _validate_standard_runtime_fields(root, path="config")

        text_raw = _object_field(root, "text_config")
        vision_raw = _object_field(root, "vision_config")
        _validate_standard_runtime_fields(text_raw, path="text_config")
        _validate_standard_runtime_fields(vision_raw, path="vision_config")
        root_moe = _object_field(root, "genesis_moe")
        text_moe = _object_field(text_raw, "genesis_moe")
        if root_moe != text_moe:
            _fail("root and text_config genesis_moe blocks must be identical")
        moe = _parse_moe(root_moe, test_geometry=test_geometry)

        mtp_raw = _object_field(_object_field(root, "genesis_vla"), "mtp")
        mtp = _parse_mtp(mtp_raw)
        text = _parse_text(text_raw, moe=moe, mtp=mtp, test_geometry=test_geometry)
        vision = _parse_vision(vision_raw, text=text, test_geometry=test_geometry)
        special_tokens = _parse_special_tokens(root, text_raw, text=text, test_geometry=test_geometry)
        coord = _parse_coord(_object_field(root, "genesis_coord_tokens"), text, test_geometry=test_geometry)
        vector, action = _parse_vla(
            _object_field(root, "genesis_vla"), text=text, mtp=mtp, test_geometry=test_geometry
        )

        return cls(
            artifact=artifact,
            text=text,
            vision=vision,
            moe=moe,
            special_tokens=special_tokens,
            coord_tokens=coord,
            vector_encoder=vector,
            action_expert=action,
            mtp=mtp,
            test_only_reduced_geometry=test_geometry,
        )

    def expected_tensor_shapes(self) -> dict[str, tuple[int, ...]]:
        """Return the exact PR-3154 state-dict inventory derived from config."""
        shapes: dict[str, tuple[int, ...]] = {}

        def add(key: str, *shape: int) -> None:
            if key in shapes:  # pragma: no cover - programmer invariant
                raise AssertionError(f"duplicate expected MK1 tensor {key}")
            shapes[key] = tuple(shape)

        text = self.text
        moe = self.moe
        hidden = text.hidden_size
        add("model.language_model.embed_tokens.weight", text.vocab_size, hidden)
        add("model.language_model.norm.weight", hidden)
        add("lm_head.weight", text.vocab_size, hidden)

        for layer_index, layer_type in enumerate(text.layer_types):
            prefix = f"model.language_model.layers.{layer_index}"
            _add_decoder_layer_shapes(add, prefix, text=text, moe=moe, layer_type=layer_type)

        vision = self.vision
        vh = vision.hidden_size
        vi = vision.intermediate_size
        add(
            "model.visual.patch_embed.proj.weight",
            vh,
            vision.in_channels,
            vision.temporal_patch_size,
            vision.patch_size,
            vision.patch_size,
        )
        add("model.visual.patch_embed.proj.bias", vh)
        add("model.visual.pos_embed.weight", vision.num_position_embeddings, vh)
        for block_index in range(vision.depth):
            prefix = f"model.visual.blocks.{block_index}"
            add(f"{prefix}.norm1.weight", vh)
            add(f"{prefix}.norm1.bias", vh)
            add(f"{prefix}.norm2.weight", vh)
            add(f"{prefix}.norm2.bias", vh)
            add(f"{prefix}.attn.qkv.weight", 3 * vh, vh)
            add(f"{prefix}.attn.qkv.bias", 3 * vh)
            add(f"{prefix}.attn.proj.weight", vh, vh)
            add(f"{prefix}.attn.proj.bias", vh)
            add(f"{prefix}.mlp.linear_fc1.weight", vi, vh)
            add(f"{prefix}.mlp.linear_fc1.bias", vi)
            add(f"{prefix}.mlp.linear_fc2.weight", vh, vi)
            add(f"{prefix}.mlp.linear_fc2.bias", vh)
        merged = vh * vision.spatial_merge_size**2
        add("model.visual.merger.norm.weight", vh)
        add("model.visual.merger.norm.bias", vh)
        add("model.visual.merger.linear_fc1.weight", merged, merged)
        add("model.visual.merger.linear_fc1.bias", merged)
        add("model.visual.merger.linear_fc2.weight", vision.out_hidden_size, merged)
        add("model.visual.merger.linear_fc2.bias", vision.out_hidden_size)

        vector = self.vector_encoder
        add("model.vector_embedding.0.weight", vector.hidden_dim, vector.max_states)
        add("model.vector_embedding.2.weight", vector.output_dim, vector.hidden_dim)
        _add_action_expert_shapes(add, self.action_expert, context_dim=hidden)

        if self.mtp.present:
            add("mtp.fc.weight", hidden, 2 * hidden)
            add("mtp.norm.weight", hidden)
            add("mtp.pre_fc_norm_embedding.weight", hidden)
            add("mtp.pre_fc_norm_hidden.weight", hidden)
            for layer_index in range(self.mtp.physical_layers):
                _add_decoder_layer_shapes(
                    add,
                    f"mtp.layers.{layer_index}",
                    text=text,
                    moe=moe,
                    layer_type="full_attention",
                )
        return shapes

    def validate_tensor_inventory(
        self,
        inventory: SafetensorsInventory,
        *,
        allowed_storage_dtypes: frozenset[str] = frozenset({"BF16"}),
    ) -> None:
        """Require exact key, shape, and explicitly allowed dtype coverage before allocation."""
        expected = self.expected_tensor_shapes()
        actual_keys = set(inventory.tensors)
        missing = sorted(set(expected) - actual_keys)
        unexpected = sorted(actual_keys - set(expected))
        if missing or unexpected:
            _fail(
                "safetensors key coverage mismatch: "
                f"missing={len(missing)} (first={missing[:8]}), "
                f"unexpected={len(unexpected)} (first={unexpected[:8]})"
            )
        bad_shapes = [
            (key, inventory.tensors[key].shape, shape)
            for key, shape in expected.items()
            if inventory.tensors[key].shape != shape
        ]
        if bad_shapes:
            key, actual, wanted = bad_shapes[0]
            _fail(f"tensor {key!r} has shape {actual}, expected {wanted}")
        bad_dtypes = sorted(
            key for key, header in inventory.tensors.items() if header.dtype not in allowed_storage_dtypes
        )
        if bad_dtypes:
            first = bad_dtypes[0]
            expected = "/".join(sorted(allowed_storage_dtypes))
            _fail(f"tensor {first!r} has dtype {inventory.tensors[first].dtype}, expected {expected}")

    def validate_coord_reserved_token_groups(self, groups: Any) -> None:
        """Cross-check the checkpoint coordinate range against mHarmony metadata.

        Genesis identifies this reserved block by the exact ``name="coord"``
        marker.  A tokenizer label is neither required nor sufficient: changing
        the name disables the coordinate-token render path even if the numeric
        range happens to match.
        """
        if not isinstance(groups, list) or not groups:
            _fail("mHarmony reserved-token groups must be a non-empty list")
        coord_groups = [
            group for group in groups if isinstance(group, Mapping) and group.get("name") == "coord"
        ]
        if not self.coord_tokens.enabled:
            if coord_groups:
                _fail(
                    "mHarmony reserved-token groups must not contain name='coord' "
                    "when coordinates are disabled"
                )
        elif len(coord_groups) != 1:
            _fail("mHarmony reserved-token groups must contain exactly one name='coord' entry")

        fast_groups = [
            group
            for group in groups
            if isinstance(group, Mapping) and group.get("tokenizer") == MK1_FAST_TOKENIZER
        ]
        if self.requires_fast_reserved_token_group and len(fast_groups) != 1:
            _fail(
                "production trained_policy mHarmony reserved-token groups must contain exactly one "
                f"{MK1_FAST_TOKENIZER} entry"
            )
        if not self.requires_fast_reserved_token_group and len(fast_groups) > 1:
            _fail(f"mHarmony reserved-token groups must contain at most one {MK1_FAST_TOKENIZER} entry")
        if fast_groups and not self.test_only_reduced_geometry:
            fast = fast_groups[0]
            if (fast.get("offset"), fast.get("size")) != (MK1_FAST_TOKEN_OFFSET, MK1_FAST_TOKEN_SIZE):
                _fail(
                    "production MK1 FAST reserved-token range must be "
                    f"offset {MK1_FAST_TOKEN_OFFSET} with size {MK1_FAST_TOKEN_SIZE}"
                )

        if not self.coord_tokens.enabled:
            return
        coord = coord_groups[0]
        offset = coord.get("offset")
        size = coord.get("size")
        if not isinstance(offset, int) or isinstance(offset, bool):
            _fail("mHarmony coord reserved-token offset must be an integer")
        if not isinstance(size, int) or isinstance(size, bool):
            _fail("mHarmony coord reserved-token size must be an integer")
        if (offset, size) != (self.coord_tokens.offset, self.coord_tokens.size):
            _fail(
                "mHarmony coord reserved-token range disagrees with genesis_coord_tokens: "
                f"groups=({offset}, {size}), checkpoint=({self.coord_tokens.offset}, {self.coord_tokens.size})"
            )


def read_mk1_config_data(
    model_dir: str | Path,
    *,
    allow_test_only_reduced_geometry: bool = False,
) -> tuple[dict[str, Any], Mk1CheckpointContract]:
    """Read strict config JSON once and return both its data and contract."""
    root = Path(model_dir).resolve()
    raw = _read_json_object(_local_file(root, "config.json"))
    contract = Mk1CheckpointContract.parse_allowlisted(
        raw,
        allow_test_only_reduced_geometry=allow_test_only_reduced_geometry,
    )
    return raw, contract


def read_mk1_config(
    model_dir: str | Path,
    *,
    allow_test_only_reduced_geometry: bool = False,
) -> Mk1CheckpointContract:
    """Read and parse an inner HF config without model construction."""
    _raw, contract = read_mk1_config_data(
        model_dir,
        allow_test_only_reduced_geometry=allow_test_only_reduced_geometry,
    )
    return contract


def validate_mk1_tokenizer(
    model_dir: str | Path,
    contract: Mk1CheckpointContract,
) -> None:
    """Require tokenizer IDs to agree with special and coordinate declarations."""
    root = Path(model_dir).resolve()
    tokenizer = _read_json_object(_local_file(root, "tokenizer.json"))
    model = tokenizer.get("model")
    base_vocab = model.get("vocab") if isinstance(model, Mapping) else None
    if not isinstance(base_vocab, Mapping) or not base_vocab:
        _fail("tokenizer.json model.vocab must be a non-empty object")

    by_token: dict[str, int] = {}
    by_id: dict[int, str] = {}
    added_special_tokens: set[str] = set()

    def add(token: Any, token_id: Any, *, location: str) -> None:
        if not isinstance(token, str) or not token:
            _fail(f"{location} token content must be a non-empty string")
        if not isinstance(token_id, int) or isinstance(token_id, bool) or token_id < 0:
            _fail(f"{location} token ID must be a non-negative integer")
        if token_id >= contract.text.vocab_size:
            _fail(f"{location} token ID {token_id} exceeds the checkpoint vocabulary")
        if token in by_token and by_token[token] != token_id:
            _fail(f"tokenizer maps token {token!r} to multiple IDs")
        if token_id in by_id and by_id[token_id] != token:
            _fail(f"tokenizer maps ID {token_id} to multiple tokens")
        by_token[token] = token_id
        by_id[token_id] = token

    for token, token_id in base_vocab.items():
        add(token, token_id, location="tokenizer.json model.vocab")
    added_tokens = tokenizer.get("added_tokens")
    if not isinstance(added_tokens, list):
        _fail("tokenizer.json added_tokens must be a list")
    for index, item in enumerate(added_tokens):
        if not isinstance(item, Mapping):
            _fail(f"tokenizer.json added_tokens[{index}] must be an object")
        add(item.get("content"), item.get("id"), location=f"tokenizer.json added_tokens[{index}]")
        if item.get("special") is True:
            added_special_tokens.add(item["content"])

    special = contract.special_tokens
    required = {
        "<|endoftext|>": special.eos_token_id,
        "<|vision_start|>": special.vision_start_token_id,
        "<|vision_end|>": special.vision_end_token_id,
        "<|image_pad|>": special.image_token_id,
        "<|video_pad|>": special.video_token_id,
    }
    if special.bos_token_id != special.eos_token_id:
        _fail("MK1 bos_token_id and eos_token_id must share <|endoftext|>")
    for token, expected_id in required.items():
        if by_token.get(token) != expected_id:
            _fail(f"tokenizer special token {token!r} has ID {by_token.get(token)!r}, expected {expected_id}")
        if token not in base_vocab and token not in added_special_tokens:
            _fail(f"tokenizer token {token!r} is not marked special")

    if contract.coord_tokens.enabled:
        coord_start = contract.coord_tokens.offset
        coord_size = contract.coord_tokens.size
        if coord_start is None or coord_size is None:  # pragma: no cover - parser invariant
            raise AssertionError("enabled coordinate contract is missing its range")
        coord_end = coord_start + coord_size
        collisions = sorted(token_id for token_id in by_id if coord_start <= token_id < coord_end)
        if collisions:
            _fail(f"tokenizer assigns explicit tokens inside the reserved coordinate range: {collisions[:8]}")

    if contract.requires_fast_reserved_token_group:
        fast_end = MK1_FAST_TOKEN_OFFSET + MK1_FAST_TOKEN_SIZE
        collisions = sorted(token_id for token_id in by_id if MK1_FAST_TOKEN_OFFSET <= token_id < fast_end)
        if collisions:
            _fail(f"tokenizer assigns explicit tokens inside the reserved FAST range: {collisions[:8]}")


def read_and_validate_safetensors_index(model_dir: str | Path) -> SafetensorsInventory:
    """Inspect an authoritative shard index and safetensors headers only."""
    from safetensors import safe_open

    root = Path(model_dir)
    if not root.is_dir():
        _fail(f"model directory does not exist: {root}")
    root = root.resolve()
    index_path = _local_file(root, MK1_SAFETENSORS_INDEX)
    index = _read_json_object(index_path)
    unknown_index_keys = set(index) - {"metadata", "weight_map"}
    if unknown_index_keys:
        _fail(f"safetensors index has unsupported fields: {sorted(unknown_index_keys)}")
    metadata = index.get("metadata")
    if not isinstance(metadata, dict) or set(metadata) != {"total_size"}:
        _fail("safetensors index metadata must contain exactly total_size")
    advertised_size = metadata["total_size"]
    if not isinstance(advertised_size, int) or isinstance(advertised_size, bool) or advertised_size < 0:
        _fail("safetensors index metadata.total_size must be a non-negative integer")
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        _fail("safetensors index must contain a non-empty weight_map object")

    advertised_by_shard: dict[str, set[str]] = {}
    for key, shard_name in weight_map.items():
        if not isinstance(key, str) or not key or key.strip() != key or "\x00" in key:
            _fail("safetensors weight_map contains an invalid tensor key")
        if not isinstance(shard_name, str) or _SHARD_NAME.fullmatch(shard_name) is None:
            _fail(f"safetensors weight_map contains an invalid shard name: {shard_name!r}")
        advertised_by_shard.setdefault(shard_name, set()).add(key)
    _validate_shard_sequence(set(advertised_by_shard))

    indexed_names = set(advertised_by_shard)
    on_disk_names = {
        path.name for path in root.glob("*.safetensors") if path.name not in _NON_MODEL_SAFETENSORS_SIDECARS
    }
    if on_disk_names != indexed_names:
        _fail(
            "safetensors shard set does not match the index: "
            f"missing={sorted(indexed_names - on_disk_names)}, unindexed={sorted(on_disk_names - indexed_names)}"
        )

    tensors: dict[str, TensorHeader] = {}
    missing_from_shards: set[str] = set()
    unindexed_in_shards: set[str] = set()
    duplicates: set[str] = set()
    shard_paths: list[Path] = []
    shard_identities: dict[Path, SafetensorsShardIdentity] = {}
    for shard_name in sorted(advertised_by_shard):
        shard_path = root / shard_name
        if not _is_local_or_hf_snapshot_file(root, shard_path):
            _fail(f"safetensors index references a missing or escaping shard: {shard_name}")
        identity = _capture_shard_identity(shard_path)
        try:
            with shard_path.open("rb") as stream:
                header_prefix = stream.read(8)
            if len(header_prefix) != 8:
                _fail(f"safetensors shard {shard_name} is shorter than its header prefix")
            header_size = int.from_bytes(header_prefix, byteorder="little", signed=False)
            if header_size > _MAX_SAFETENSORS_HEADER_BYTES:
                _fail(
                    f"safetensors shard {shard_name} header exceeds the "
                    f"{_MAX_SAFETENSORS_HEADER_BYTES}-byte safety limit"
                )
            if header_size + 8 > shard_path.stat().st_size:
                _fail(f"safetensors shard {shard_name} declares a truncated header")
        except Mk1CheckpointContractError:
            raise
        except OSError as exc:
            _fail(f"could not inspect safetensors shard {shard_name}: {exc}")
        _verify_shard_identity(identity)
        shard_paths.append(shard_path)
        try:
            _verify_shard_identity(identity)
            with safe_open(str(shard_path), framework="pt", device="cpu") as handle:
                _verify_shard_identity(identity)
                actual_keys = set(handle.keys())
                duplicates.update(actual_keys.intersection(tensors))
                advertised = advertised_by_shard[shard_name]
                missing_from_shards.update(advertised - actual_keys)
                unindexed_in_shards.update(actual_keys - advertised)
                for key in actual_keys:
                    tensor_slice = handle.get_slice(key)
                    tensors.setdefault(
                        key,
                        TensorHeader(
                            key=key,
                            shard=shard_name,
                            shape=tuple(tensor_slice.get_shape()),
                            dtype=str(tensor_slice.get_dtype()),
                        ),
                    )
                _verify_shard_identity(identity)
            _verify_shard_identity(identity)
        except Mk1CheckpointContractError:
            raise
        except Exception as exc:
            _fail(f"could not inspect safetensors shard {shard_name}: {exc}")
        shard_identities[shard_path] = identity
    if duplicates:
        _fail(f"duplicate tensor keys across safetensors shards: {sorted(duplicates)[:8]}")
    if missing_from_shards or unindexed_in_shards:
        _fail(
            "safetensors index/header mismatch: "
            f"missing={sorted(missing_from_shards)[:8]}, unindexed={sorted(unindexed_in_shards)[:8]}"
        )
    header_size = sum(_tensor_payload_size(header) for header in tensors.values())
    if advertised_size != header_size:
        _fail(
            "safetensors index metadata.total_size does not match tensor headers: "
            f"advertised={advertised_size}, headers={header_size}"
        )
    return SafetensorsInventory(
        index_path=index_path,
        shards=tuple(shard_paths),
        shard_identities=shard_identities,
        tensors=tensors,
    )


def validate_mk1_checkpoint(
    model_dir: str | Path,
    *,
    allow_test_only_reduced_geometry: bool = False,
    allowed_storage_dtypes: frozenset[str] = frozenset({"BF16"}),
) -> tuple[Mk1CheckpointContract, SafetensorsInventory]:
    """Validate config and tensor headers before any module is allocated."""
    _raw, contract = read_mk1_config_data(
        model_dir,
        allow_test_only_reduced_geometry=allow_test_only_reduced_geometry,
    )
    inventory = read_and_validate_safetensors_index(model_dir)
    contract.validate_tensor_inventory(inventory, allowed_storage_dtypes=allowed_storage_dtypes)
    return contract, inventory


def finalize_mk1_runtime_checkpoint_layout(model_dir: str | Path) -> None:
    """Drop source-only MTP metadata and ensure one authoritative sharded-model index."""
    from safetensors import safe_open

    root = Path(model_dir)
    config_path = root / "config.json"
    raw = _read_json_object(config_path)
    genesis_vla = raw.get("genesis_vla")
    text_config = raw.get("text_config")
    mtp = genesis_vla.get("mtp") if isinstance(genesis_vla, dict) else None
    if not isinstance(mtp, dict) or not isinstance(text_config, dict):
        _fail("MK1 runtime checkpoint config is missing its MTP contract")
    if mtp.get("present") is True:
        mtp.update({"present": False, "physical_layers": 0, "rollout_steps": 0})
        text_config["mtp_num_hidden_layers"] = 0
        Mk1CheckpointContract.parse_allowlisted(raw, allow_test_only_reduced_geometry=True)
        temporary_config = config_path.with_suffix(".json.tmp")
        temporary_config.write_text(json.dumps(raw, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        temporary_config.replace(config_path)

    index_path = root / MK1_SAFETENSORS_INDEX
    single_path = root / "model.safetensors"
    if index_path.is_file():
        return
    if not single_path.is_file():
        _fail("MK1 runtime checkpoint save produced neither an index nor model.safetensors")
    shard_name = "model-00001-of-00001.safetensors"
    shard_path = root / shard_name
    weight_map: dict[str, str] = {}
    total_size = 0
    with safe_open(single_path, framework="pt", device="cpu") as reader:
        for key in reader.keys():  # noqa: SIM118 - safe_open is not iterable
            tensor_slice = reader.get_slice(key)
            header = TensorHeader(
                key=key,
                shard=shard_name,
                shape=tuple(tensor_slice.get_shape()),
                dtype=str(tensor_slice.get_dtype()),
            )
            total_size += _tensor_payload_size(header)
            weight_map[key] = shard_name
    if not weight_map:
        _fail("MK1 runtime checkpoint save produced no model tensors")
    single_path.replace(shard_path)
    temporary_index = index_path.with_suffix(".json.tmp")
    temporary_index.write_text(
        json.dumps(
            {"metadata": {"total_size": total_size}, "weight_map": weight_map},
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary_index.replace(index_path)


def _parse_moe(raw: Mapping[str, Any], *, test_geometry: bool) -> Mk1MoeContract:
    _require_exact_keys(raw, _CUSTOM_MOE_KEYS, "genesis_moe")
    real = _integer(raw, "num_real_experts")
    null = _integer(raw, "num_null_experts")
    physical = _integer(raw, "physical_router_outputs")
    logical = _integer(raw, "logical_router_outputs")
    top_k = _integer(raw, "top_k")
    route_scale = _number(raw, "route_scale")
    router_contract = raw.get("router_contract")
    if not isinstance(router_contract, list) or router_contract != [real, null]:
        _fail("genesis_moe.router_contract must equal [num_real_experts, num_null_experts]")
    fixed = {
        "router_contract_version": 1,
        "shared_null_router_row": True,
        "score_func": "softmax",
        "route_norm": True,
        "score_before_experts": False,
        "null_expert_semantics": "skip_compute_renormalize_real_routes",
        "shared_expert_mode": "sigmoid_gated_additive",
        "uses_expert_bias": False,
    }
    _require_values(raw, fixed, "genesis_moe")
    if real <= 0 or null != real:
        _fail("genesis_moe requires equal positive real and logical-null expert counts")
    if physical != real + 1 or logical != real + null:
        _fail("genesis_moe compact physical/logical router dimensions are inconsistent")
    if not 0 < top_k <= logical:
        _fail("genesis_moe.top_k is outside the logical router range")
    if route_scale != 1.0:
        _fail("genesis_moe.route_scale must be 1.0")
    if not test_geometry and (real, null, physical, logical, top_k) != (256, 256, 257, 512, 8):
        _fail("production MK1 requires the 256-real/256-null compact router and top-k 8")
    return Mk1MoeContract(real, null, physical, logical, top_k, route_scale)


def _parse_artifact(raw: Mapping[str, Any]) -> Mk1ArtifactContract:
    """Parse the exporter-owned training-state identity.

    Neutral Step-0 exports may carry additional hash/provenance fields. Those
    fields are inert metadata; the three versioned identity fields below are
    the complete behavior and deployment authority contract.
    """
    missing = _ARTIFACT_IDENTITY_KEYS - set(raw)
    if missing:
        _fail(f"genesis_artifact is missing required identity fields: {sorted(missing)}")
    schema_version = _integer(raw, "schema_version", "genesis_artifact")
    if schema_version != 1:
        _fail(f"genesis_artifact.schema_version must be 1, got {schema_version}")
    artifact_kind = raw.get("artifact_kind")
    if artifact_kind not in {"trained_policy", "neutral_debug"}:
        _fail("genesis_artifact.artifact_kind must be 'trained_policy' or 'neutral_debug'")
    trained_steps = _integer(raw, "trained_steps", "genesis_artifact")
    if artifact_kind == "trained_policy" and trained_steps <= 0:
        _fail("genesis_artifact trained_policy requires positive trained_steps")
    if artifact_kind == "neutral_debug" and trained_steps != 0:
        _fail("genesis_artifact neutral_debug requires trained_steps=0")
    if artifact_kind == "trained_policy" and set(raw) != _ARTIFACT_IDENTITY_KEYS:
        _fail("trained genesis_artifact fields do not match schema version 1")
    return Mk1ArtifactContract(schema_version, artifact_kind, trained_steps)


def _validate_standard_runtime_fields(raw: Mapping[str, Any], *, path: str) -> None:
    """Pin optional Transformers flags whose defaults can change execution."""
    common = {
        "output_attentions": False,
        "output_hidden_states": False,
        "return_dict": True,
        "chunk_size_feed_forward": 0,
        "is_encoder_decoder": False,
    }
    expected = dict(common)
    if path == "config":
        expected.update({"dtype": "bfloat16"})
    elif path == "text_config":
        expected.update(
            {
                "dtype": "bfloat16",
                "use_cache": True,
                "attention_dropout": 0.0,
                "output_router_logits": False,
                "norm_topk_prob": True,
            }
        )
    for key, wanted in expected.items():
        if key in raw and not _same_json_value(raw[key], wanted):
            _fail(f"{path}.{key} must be {wanted!r}, got {raw[key]!r}")
    for forbidden in ("use_return_dict", "_attn_implementation", "attn_implementation"):
        if forbidden in raw:
            _fail(f"{path}.{forbidden} must not be serialized in an MK1 checkpoint")


def _parse_text(
    raw: Mapping[str, Any],
    *,
    moe: Mk1MoeContract,
    mtp: Mk1MtpContract,
    test_geometry: bool,
) -> Mk1TextContract:
    if raw.get("model_type") != MK1_TEXT_MODEL_TYPE:
        _fail(f"text_config.model_type must be {MK1_TEXT_MODEL_TYPE!r}")
    if raw.get("tie_word_embeddings") is not False:
        _fail("text_config.tie_word_embeddings must be false")
    hidden = _integer(raw, "hidden_size", "text_config")
    layers = _integer(raw, "num_hidden_layers", "text_config")
    vocab = _integer(raw, "vocab_size", "text_config")
    heads = _integer(raw, "num_attention_heads", "text_config")
    kv_heads = _integer(raw, "num_key_value_heads", "text_config")
    head_dim = _integer(raw, "head_dim", "text_config")
    moe_intermediate = _integer(raw, "moe_intermediate_size", "text_config")
    shared_intermediate = _integer(raw, "shared_expert_intermediate_size", "text_config")
    interval = _integer(raw, "full_attention_interval", "text_config")
    conv_kernel = _integer(raw, "linear_conv_kernel_dim", "text_config")
    linear_key_dim = _integer(raw, "linear_key_head_dim", "text_config")
    linear_key_heads = _integer(raw, "linear_num_key_heads", "text_config")
    linear_value_heads = _integer(raw, "linear_num_value_heads", "text_config")
    linear_value_dim = _integer(raw, "linear_value_head_dim", "text_config")
    layer_types_raw = raw.get("layer_types")
    if not isinstance(layer_types_raw, list) or not all(isinstance(item, str) for item in layer_types_raw):
        _fail("text_config.layer_types must be a string list")
    layer_types = tuple(layer_types_raw)
    if len(layer_types) != layers:
        _fail("text_config.layer_types length must equal num_hidden_layers")
    if interval <= 0:
        _fail("text_config.full_attention_interval must be positive")
    expected_layers = tuple(
        "full_attention" if (index + 1) % interval == 0 else "linear_attention" for index in range(layers)
    )
    if layer_types != expected_layers:
        _fail("text_config.layer_types does not match full_attention_interval")
    if hidden <= 0 or heads <= 0 or kv_heads <= 0 or head_dim <= 0 or heads * head_dim <= 0:
        _fail("text_config attention geometry must be positive")
    if moe_intermediate <= 0 or shared_intermediate <= 0:
        _fail("text_config expert widths must be positive")
    if _integer(raw, "num_experts", "text_config") != moe.num_real_experts:
        _fail("text_config.num_experts must equal genesis_moe.num_real_experts")
    if _integer(raw, "num_experts_per_tok", "text_config") != moe.top_k:
        _fail("text_config.num_experts_per_tok must equal genesis_moe.top_k")
    if _integer(raw, "mtp_num_hidden_layers", "text_config") != mtp.physical_layers:
        _fail("text_config.mtp_num_hidden_layers must equal genesis_vla.mtp.physical_layers")
    fixed = {
        "attention_bias": False,
        "attention_dropout": 0.0,
        "attn_output_gate": True,
        "dtype": "bfloat16",
        "hidden_act": "silu",
        "mamba_ssm_dtype": "float32",
        "mtp_use_dedicated_embeddings": False,
        "rms_norm_eps": 1e-6,
        "use_cache": True,
    }
    _require_values(raw, fixed, "text_config")
    rope = _object_field(raw, "rope_parameters", "text_config")
    _require_exact_keys(rope, _ROPE_KEYS, "text_config.rope_parameters")
    partial = _number(raw, "partial_rotary_factor", "text_config")
    _require_values(
        rope,
        {
            "mrope_interleaved": True,
            "partial_rotary_factor": partial,
            "rope_type": "default",
        },
        "text_config.rope_parameters",
    )
    mrope = rope.get("mrope_section")
    if (
        not isinstance(mrope, list)
        or len(mrope) != 3
        or not all(isinstance(value, int) and not isinstance(value, bool) and value >= 0 for value in mrope)
    ):
        _fail("text_config.rope_parameters.mrope_section must be three non-negative integers")
    rotary_half = head_dim * partial / 2
    if not rotary_half.is_integer() or sum(mrope) != int(rotary_half):
        _fail("text_config mRoPE sections do not span half the partial rotary dimension")
    rope_theta = _number(rope, "rope_theta", "text_config.rope_parameters")
    max_positions = _integer(raw, "max_position_embeddings", "text_config")
    if max_positions <= 0:
        _fail("text_config.max_position_embeddings must be positive")

    production = {
        "hidden_size": 2048,
        "num_hidden_layers": 40,
        "vocab_size": 256279,
        "num_attention_heads": 16,
        "num_key_value_heads": 2,
        "head_dim": 256,
        "moe_intermediate_size": 512,
        "shared_expert_intermediate_size": 512,
        "full_attention_interval": 4,
        "linear_conv_kernel_dim": 4,
        "linear_key_head_dim": 128,
        "linear_num_key_heads": 16,
        "linear_num_value_heads": 32,
        "linear_value_head_dim": 128,
        "max_position_embeddings": 262144,
        "partial_rotary_factor": 0.25,
    }
    if not test_geometry:
        _require_values(raw, production, "text_config")
        if mrope != [11, 11, 10] or rope_theta != 10_000_000:
            _fail("production MK1 requires mRoPE [11, 11, 10] and rope theta 10000000")
    elif (
        min(
            hidden,
            layers,
            vocab,
            heads,
            kv_heads,
            head_dim,
            conv_kernel,
            linear_key_dim,
            linear_key_heads,
            linear_value_heads,
            linear_value_dim,
        )
        <= 0
    ):
        _fail("test-only text geometry values must be positive")
    return Mk1TextContract(
        hidden,
        layers,
        vocab,
        heads,
        kv_heads,
        head_dim,
        moe_intermediate,
        shared_intermediate,
        layer_types,
        interval,
        conv_kernel,
        linear_key_dim,
        linear_key_heads,
        linear_value_heads,
        linear_value_dim,
    )


def _parse_vision(raw: Mapping[str, Any], *, text: Mk1TextContract, test_geometry: bool) -> Mk1VisionContract:
    fields = {
        name: _integer(raw, name, "vision_config")
        for name in (
            "depth",
            "hidden_size",
            "intermediate_size",
            "num_heads",
            "num_position_embeddings",
            "out_hidden_size",
            "in_channels",
            "patch_size",
            "spatial_merge_size",
            "temporal_patch_size",
        )
    }
    if min(fields.values()) <= 0:
        _fail("vision_config geometry values must be positive")
    _require_values(
        raw,
        {
            "deepstack_visual_indexes": [],
            "hidden_act": "gelu_pytorch_tanh",
            "model_type": MK1_MODEL_TYPE,
        },
        "vision_config",
    )
    if fields["out_hidden_size"] != text.hidden_size:
        _fail("vision_config.out_hidden_size must equal text_config.hidden_size")
    if fields["hidden_size"] % fields["num_heads"]:
        _fail("vision_config.hidden_size must be divisible by num_heads")
    if not test_geometry:
        _require_values(
            raw,
            {
                "depth": 27,
                "hidden_size": 1152,
                "intermediate_size": 4304,
                "num_heads": 16,
                "num_position_embeddings": 2304,
                "out_hidden_size": 2048,
                "in_channels": 3,
                "patch_size": 16,
                "spatial_merge_size": 2,
                "temporal_patch_size": 2,
            },
            "vision_config",
        )
    return Mk1VisionContract(**fields)


def _parse_coord(
    raw: Mapping[str, Any], text: Mk1TextContract, *, test_geometry: bool
) -> Mk1CoordTokenContract:
    enabled = raw.get("enabled")
    if enabled is False:
        _require_exact_keys(raw, _COORD_DISABLED_KEYS, "genesis_coord_tokens")
        return Mk1CoordTokenContract(enabled=False)
    if enabled is not True:
        _fail("genesis_coord_tokens.enabled must be a boolean")
    _require_exact_keys(raw, _COORD_ENABLED_KEYS, "genesis_coord_tokens")
    offset = _integer(raw, "offset", "genesis_coord_tokens")
    size = _integer(raw, "size", "genesis_coord_tokens")
    if offset < 0 or size <= 0 or offset + size > text.vocab_size:
        _fail("coordinate token range lies outside the exported vocabulary")
    if not test_geometry and (offset, size) != (248320, 1001):
        _fail("production MK1 coordinate range must be offset 248320 with size 1001")
    return Mk1CoordTokenContract(enabled=True, offset=offset, size=size)


def _parse_special_tokens(
    root: Mapping[str, Any],
    text_raw: Mapping[str, Any],
    *,
    text: Mk1TextContract,
    test_geometry: bool,
) -> Mk1SpecialTokenContract:
    values = {
        "bos_token_id": _integer(text_raw, "bos_token_id", "text_config"),
        "eos_token_id": _integer(text_raw, "eos_token_id", "text_config"),
        "image_token_id": _integer(root, "image_token_id", "config"),
        "video_token_id": _integer(root, "video_token_id", "config"),
        "vision_start_token_id": _integer(root, "vision_start_token_id", "config"),
        "vision_end_token_id": _integer(root, "vision_end_token_id", "config"),
    }
    if text_raw.get("pad_token_id", object()) is not None:
        _fail("text_config.pad_token_id must be null")
    if any(token_id < 0 or token_id >= text.vocab_size for token_id in values.values()):
        _fail("Qwen special-token IDs must lie inside the exported vocabulary")
    if not test_geometry:
        expected = {
            "bos_token_id": 248044,
            "eos_token_id": 248044,
            "image_token_id": 248056,
            "video_token_id": 248057,
            "vision_start_token_id": 248053,
            "vision_end_token_id": 248054,
        }
        for key, wanted in expected.items():
            if values[key] != wanted:
                _fail(f"production MK1 {key} must be {wanted}, got {values[key]}")
    return Mk1SpecialTokenContract(**values)


def _parse_vla(
    raw: Mapping[str, Any], *, text: Mk1TextContract, mtp: Mk1MtpContract, test_geometry: bool
) -> tuple[Mk1VectorEncoderContract, Mk1ActionExpertContract]:
    _require_exact_keys(raw, _VLA_KEYS, "genesis_vla")
    _require_values(
        raw,
        {
            "schema_version": MK1_VLA_SCHEMA_VERSION,
            "backbone_family": MK1_BACKBONE_FAMILY,
            "state_dict_schema": MK1_STATE_DICT_SCHEMA,
            "rmsnorm_weight_convention": MK1_RMSNORM_CONVENTION,
        },
        "genesis_vla",
    )
    if _object_field(raw, "mtp", "genesis_vla") != {
        "present": mtp.present,
        "physical_layers": mtp.physical_layers,
        "rollout_steps": mtp.rollout_steps,
        "action_runtime": "exclude",
    }:
        _fail("genesis_vla.mtp changed while parsing")  # pragma: no cover - defensive

    vector_raw = _object_field(raw, "vector_encoder", "genesis_vla")
    _require_exact_keys(vector_raw, _VECTOR_KEYS, "genesis_vla.vector_encoder")
    _require_values(
        vector_raw,
        {"type": "linear_silu_linear", "bias": False},
        "genesis_vla.vector_encoder",
    )
    max_states = _integer(vector_raw, "max_states", "genesis_vla.vector_encoder")
    hidden_dim = _integer(vector_raw, "hidden_dim", "genesis_vla.vector_encoder")
    output_dim = _integer(vector_raw, "output_dim", "genesis_vla.vector_encoder")
    if min(max_states, hidden_dim, output_dim) <= 0:
        _fail("vector encoder dimensions must be positive")
    if hidden_dim != text.hidden_size or output_dim != text.hidden_size:
        _fail("vector encoder hidden/output widths must equal the backbone hidden size")
    if not test_geometry and max_states != 128:
        _fail("production MK1 vector encoder max_states must be 128")
    vector = Mk1VectorEncoderContract(max_states, hidden_dim, output_dim)

    action_raw = _object_field(raw, "action_expert", "genesis_vla")
    _require_exact_keys(action_raw, _ACTION_KEYS, "genesis_vla.action_expert")
    _require_values(
        action_raw,
        {
            "type": "molmoact",
            "clean_at_0": False,
            "qk_norm": True,
            "qk_norm_eps": 1e-6,
            "rope": True,
            "context_layer_norm": True,
            "causal_attn": False,
        },
        "genesis_vla.action_expert",
    )
    action = Mk1ActionExpertContract(
        action_dim=_integer(action_raw, "action_dim", "genesis_vla.action_expert"),
        action_horizon=_integer(action_raw, "action_horizon", "genesis_vla.action_expert"),
        num_layers=_integer(action_raw, "num_layers", "genesis_vla.action_expert"),
        hidden_dim=_integer(action_raw, "hidden_dim", "genesis_vla.action_expert"),
        num_heads=_integer(action_raw, "num_heads", "genesis_vla.action_expert"),
        mlp_ratio=_number(action_raw, "mlp_ratio", "genesis_vla.action_expert"),
        timestep_embed_dim=_integer(action_raw, "timestep_embed_dim", "genesis_vla.action_expert"),
        ffn_multiple_of=_integer(action_raw, "ffn_multiple_of", "genesis_vla.action_expert"),
    )
    if (
        min(
            action.action_dim,
            action.action_horizon,
            action.num_layers,
            action.hidden_dim,
            action.num_heads,
            action.timestep_embed_dim,
            action.ffn_multiple_of,
        )
        <= 0
        or action.mlp_ratio <= 0
    ):
        _fail("action expert dimensions must be positive")
    if action.hidden_dim % action.num_heads:
        _fail("action expert hidden_dim must be divisible by num_heads")
    if not test_geometry and action != Mk1ActionExpertContract(64, 64, 36, 768, 8, 4.0, 256, 256):
        _fail("production MK1 action expert geometry does not match the 36-layer MolmoAct contract")
    return vector, action


def _parse_mtp(raw: Mapping[str, Any]) -> Mk1MtpContract:
    _require_exact_keys(raw, _MTP_KEYS, "genesis_vla.mtp")
    _require_values(raw, {"action_runtime": "exclude"}, "genesis_vla.mtp")
    present = raw.get("present")
    if not isinstance(present, bool):
        _fail("genesis_vla.mtp.present must be a boolean")
    physical = _integer(raw, "physical_layers", "genesis_vla.mtp")
    rollout = _integer(raw, "rollout_steps", "genesis_vla.mtp")
    expected = (1, 3) if present else (0, 0)
    if (physical, rollout) != expected:
        state = "present=true" if present else "present=false"
        _fail(f"MK1 MTP {state} requires physical_layers/rollout_steps={expected}")
    return Mk1MtpContract(present, physical, rollout)


def _add_decoder_layer_shapes(add, prefix: str, *, text, moe, layer_type: str) -> None:
    hidden = text.hidden_size
    add(f"{prefix}.input_layernorm.weight", hidden)
    add(f"{prefix}.post_attention_layernorm.weight", hidden)
    add(f"{prefix}.mlp.gate.weight", moe.physical_router_outputs, hidden)
    add(
        f"{prefix}.mlp.experts.gate_up_proj",
        moe.num_real_experts,
        2 * text.moe_intermediate_size,
        hidden,
    )
    add(
        f"{prefix}.mlp.experts.down_proj",
        moe.num_real_experts,
        hidden,
        text.moe_intermediate_size,
    )
    shared = text.shared_expert_intermediate_size
    add(f"{prefix}.mlp.shared_expert.gate_proj.weight", shared, hidden)
    add(f"{prefix}.mlp.shared_expert.up_proj.weight", shared, hidden)
    add(f"{prefix}.mlp.shared_expert.down_proj.weight", hidden, shared)
    add(f"{prefix}.mlp.shared_expert_gate.weight", 1, hidden)
    if layer_type == "full_attention":
        query_width = text.num_attention_heads * text.head_dim
        kv_width = text.num_key_value_heads * text.head_dim
        add(f"{prefix}.self_attn.q_proj.weight", 2 * query_width, hidden)
        add(f"{prefix}.self_attn.k_proj.weight", kv_width, hidden)
        add(f"{prefix}.self_attn.v_proj.weight", kv_width, hidden)
        add(f"{prefix}.self_attn.o_proj.weight", hidden, query_width)
        add(f"{prefix}.self_attn.q_norm.weight", text.head_dim)
        add(f"{prefix}.self_attn.k_norm.weight", text.head_dim)
    elif layer_type == "linear_attention":
        key_width = text.linear_num_key_heads * text.linear_key_head_dim
        value_width = text.linear_num_value_heads * text.linear_value_head_dim
        qkv_width = 2 * key_width + value_width
        add(f"{prefix}.linear_attn.A_log", text.linear_num_value_heads)
        add(f"{prefix}.linear_attn.dt_bias", text.linear_num_value_heads)
        add(f"{prefix}.linear_attn.conv1d.weight", qkv_width, 1, text.linear_conv_kernel_dim)
        add(f"{prefix}.linear_attn.in_proj_a.weight", text.linear_num_value_heads, hidden)
        add(f"{prefix}.linear_attn.in_proj_b.weight", text.linear_num_value_heads, hidden)
        add(f"{prefix}.linear_attn.in_proj_qkv.weight", qkv_width, hidden)
        add(f"{prefix}.linear_attn.in_proj_z.weight", value_width, hidden)
        add(f"{prefix}.linear_attn.norm.weight", text.linear_value_head_dim)
        add(f"{prefix}.linear_attn.out_proj.weight", hidden, value_width)
    else:  # pragma: no cover - config parser prevents this
        raise AssertionError(f"unknown MK1 layer type {layer_type}")


def _add_action_expert_shapes(add, action: Mk1ActionExpertContract, *, context_dim: int) -> None:
    prefix = "model.action_expert.action_expert"
    hidden = action.hidden_dim
    intermediate = action.intermediate_size
    add(f"{prefix}.time_embed.1.weight", hidden, action.timestep_embed_dim)
    add(f"{prefix}.time_embed.1.bias", hidden)
    add(f"{prefix}.time_embed.3.weight", hidden, hidden)
    add(f"{prefix}.time_embed.3.bias", hidden)
    add(f"{prefix}.action_embed.weight", hidden, action.action_dim)
    add(f"{prefix}.action_embed.bias", hidden)
    add(f"{prefix}.context_k_proj.weight", hidden, context_dim)
    add(f"{prefix}.context_v_proj.weight", hidden, context_dim)
    for block_index in range(action.num_layers):
        block = f"{prefix}.blocks.{block_index}"
        add(f"{block}.self_attn.qkv.weight", 3 * hidden, hidden)
        add(f"{block}.self_attn.qkv.bias", 3 * hidden)
        add(f"{block}.self_attn.out_proj.weight", hidden, hidden)
        add(f"{block}.self_attn.out_proj.bias", hidden)
        add(f"{block}.cross_attn.q_proj.weight", hidden, hidden)
        add(f"{block}.cross_attn.q_proj.bias", hidden)
        add(f"{block}.cross_attn.out_proj.weight", hidden, hidden)
        add(f"{block}.cross_attn.out_proj.bias", hidden)
        add(f"{block}.mlp.up_proj.weight", intermediate, hidden)
        add(f"{block}.mlp.up_proj.bias", intermediate)
        add(f"{block}.mlp.gate_proj.weight", intermediate, hidden)
        add(f"{block}.mlp.gate_proj.bias", intermediate)
        add(f"{block}.mlp.down_proj.weight", hidden, intermediate)
        add(f"{block}.mlp.down_proj.bias", hidden)
        add(f"{block}.modulation.linear.weight", 9 * hidden, hidden)
        add(f"{block}.modulation.linear.bias", 9 * hidden)
    add(f"{prefix}.final_layer.modulation.linear.weight", 2 * hidden, hidden)
    add(f"{prefix}.final_layer.modulation.linear.bias", 2 * hidden)
    add(f"{prefix}.final_layer.linear.weight", action.action_dim, hidden)
    add(f"{prefix}.final_layer.linear.bias", action.action_dim)


def _read_json_object(path: Path) -> dict[str, Any]:
    if not path.is_file():
        _fail(f"required JSON file does not exist: {path}")
    try:
        size = path.stat().st_size
    except OSError as exc:
        _fail(f"could not stat JSON object {path}: {exc}")
    if size > _MAX_JSON_FILE_BYTES:
        _fail(f"JSON object {path} exceeds the {_MAX_JSON_FILE_BYTES}-byte safety limit")

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                _fail(f"duplicate JSON key {key!r} in {path}")
            result[key] = value
        return result

    try:
        value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=reject_duplicates)
    except Mk1CheckpointContractError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        _fail(f"could not read JSON object {path}: {exc}")
    return dict(_object(value, str(path)))


def _local_file(root: Path, name: str) -> Path:
    path = root / name
    if not _is_local_or_hf_snapshot_file(root, path):
        _fail(f"required file is missing or escapes the model directory: {name}")
    return path


def _is_local_or_hf_snapshot_file(root: Path, path: Path) -> bool:
    """Allow regular local files and Hugging Face's immutable snapshot links."""
    try:
        resolved = path.resolve(strict=True)
    except (OSError, RuntimeError):
        return False
    if not resolved.is_file():
        return False
    if resolved.parent == root:
        return True
    if not path.is_symlink():
        return False
    for ancestor in path.parents:
        if ancestor.parent.name != "snapshots":
            continue
        blob_root = (ancestor.parent.parent / "blobs").resolve()
        return resolved.is_relative_to(blob_root)
    return False


def _capture_shard_identity(path: Path) -> SafetensorsShardIdentity:
    """Capture both the logical path and its current referent without reading payloads."""
    try:
        resolved = path.resolve(strict=True)
        stat = path.stat()
        resolved_stat = resolved.stat()
    except (OSError, RuntimeError) as exc:
        _fail(f"could not stat safetensors shard {path.name}: {exc}")
    followed = (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
    resolved_values = (
        resolved_stat.st_dev,
        resolved_stat.st_ino,
        resolved_stat.st_size,
        resolved_stat.st_mtime_ns,
        resolved_stat.st_ctime_ns,
    )
    if followed != resolved_values:
        _fail(f"safetensors shard {path.name} changed while resolving its path")
    return SafetensorsShardIdentity(
        path=path,
        resolved_path=resolved,
        device=stat.st_dev,
        inode=stat.st_ino,
        size=stat.st_size,
        mtime_ns=stat.st_mtime_ns,
        ctime_ns=stat.st_ctime_ns,
    )


def _verify_shard_identity(expected: SafetensorsShardIdentity) -> None:
    actual = _capture_shard_identity(expected.path)
    if actual != expected:
        _fail(
            f"safetensors shard {expected.path.name} changed during validation: "
            f"expected={_format_shard_identity(expected)}, "
            f"actual={_format_shard_identity(actual)}"
        )


def _format_shard_identity(identity: SafetensorsShardIdentity) -> str:
    return (
        f"target={identity.resolved_path},dev={identity.device},ino={identity.inode},"
        f"size={identity.size},mtime_ns={identity.mtime_ns},ctime_ns={identity.ctime_ns}"
    )


def _tensor_payload_size(header: TensorHeader) -> int:
    bytes_per_element = {
        "BOOL": 1,
        "U8": 1,
        "I8": 1,
        "F8_E4M3": 1,
        "F8_E5M2": 1,
        "I16": 2,
        "U16": 2,
        "F16": 2,
        "BF16": 2,
        "I32": 4,
        "U32": 4,
        "F32": 4,
        "I64": 8,
        "U64": 8,
        "F64": 8,
    }.get(header.dtype)
    if bytes_per_element is None:
        _fail(f"tensor {header.key!r} uses an unsupported safetensors dtype {header.dtype!r}")
    return math.prod(header.shape) * bytes_per_element


def _validate_shard_sequence(names: set[str]) -> None:
    parsed = [_SHARD_NAME.fullmatch(name) for name in names]
    assert all(match is not None for match in parsed)
    totals = {int(match.group(2)) for match in parsed if match is not None}
    if len(totals) != 1:
        _fail("safetensors shard names disagree on the total shard count")
    total = totals.pop()
    indices = {int(match.group(1)) for match in parsed if match is not None}
    if total != len(names) or indices != set(range(1, total + 1)):
        _fail("safetensors shard sequence is incomplete or inconsistent")


def _object(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        _fail(f"{path} must be a JSON object")
    return value


def _object_field(raw: Mapping[str, Any], key: str, path: str = "config") -> Mapping[str, Any]:
    return _object(raw.get(key), f"{path}.{key}")


def _integer(raw: Mapping[str, Any], key: str, path: str = "genesis_moe") -> int:
    value = raw.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        _fail(f"{path}.{key} must be an integer")
    return value


def _number(raw: Mapping[str, Any], key: str, path: str = "genesis_moe") -> float:
    value = raw.get(key)
    if not isinstance(value, int | float) or isinstance(value, bool) or not math.isfinite(value):
        _fail(f"{path}.{key} must be a finite number")
    return float(value)


def _require_exact_keys(raw: Mapping[str, Any], expected: set[str], path: str) -> None:
    actual = set(raw)
    if actual != expected:
        _fail(
            f"{path} fields do not match the supported schema: "
            f"missing={sorted(expected - actual)}, unexpected={sorted(actual - expected)}"
        )


def _require_values(raw: Mapping[str, Any], expected: Mapping[str, Any], path: str) -> None:
    for key, wanted in expected.items():
        actual = raw.get(key)
        if not _same_json_value(actual, wanted):
            _fail(f"{path}.{key} must be {wanted!r}, got {actual!r}")


def _same_json_value(actual: Any, expected: Any) -> bool:
    """Compare schema literals without Python's bool/int equality aliasing."""
    if isinstance(expected, bool):
        return actual is expected
    if isinstance(expected, int):
        return isinstance(actual, int) and not isinstance(actual, bool) and actual == expected
    if isinstance(expected, float):
        return (
            isinstance(actual, int | float)
            and not isinstance(actual, bool)
            and math.isfinite(actual)
            and float(actual) == expected
        )
    if isinstance(expected, list):
        return (
            isinstance(actual, list)
            and len(actual) == len(expected)
            and all(_same_json_value(item, wanted) for item, wanted in zip(actual, expected, strict=True))
        )
    return type(actual) is type(expected) and actual == expected


def _fail(message: str) -> None:
    raise Mk1CheckpointContractError(message)
