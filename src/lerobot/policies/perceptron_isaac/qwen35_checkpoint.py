"""Safetensors layout validation shared by native ISAAC checkpoint readers."""

from __future__ import annotations

from pathlib import Path

from .checkpoint_integrity import load_json_object

SAFETENSORS_SINGLE_FILE = "model.safetensors"
SAFETENSORS_INDEX_FILE = "model.safetensors.index.json"


def safetensors_checkpoint_layout(
    checkpoint_dir: str | Path,
) -> tuple[list[Path], dict[str, set[str]] | None]:
    """Resolve one unambiguous safetensors layout and validate its authoritative index."""
    root = Path(checkpoint_dir)
    index_path = root / SAFETENSORS_INDEX_FILE
    single_path = root / SAFETENSORS_SINGLE_FILE
    discovered_shards = sorted(root.glob("model-*.safetensors"))

    if index_path.is_file():
        if single_path.is_file():
            raise ValueError(
                f"Checkpoint {root} contains both {SAFETENSORS_SINGLE_FILE} and an indexed sharded model."
            )
        index = load_json_object(index_path)
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise ValueError(f"Invalid or empty safetensors weight_map in {index_path}.")
        advertised_by_shard: dict[str, set[str]] = {}
        for key, shard_name in weight_map.items():
            shard_path = Path(shard_name) if isinstance(shard_name, str) else Path()
            if (
                not isinstance(key, str)
                or not key
                or not isinstance(shard_name, str)
                or not shard_name
                or shard_path.name != shard_name
                or not shard_name.startswith("model-")
                or not shard_name.endswith(".safetensors")
            ):
                raise ValueError(f"Invalid safetensors weight_map entry in {index_path}.")
            advertised_by_shard.setdefault(shard_name, set()).add(key)

        advertised_names = set(advertised_by_shard)
        discovered_names = {path.name for path in discovered_shards}
        missing = sorted(advertised_names - discovered_names)
        unindexed = sorted(discovered_names - advertised_names)
        if missing or unindexed:
            raise ValueError(
                f"Safetensors index shard inventory mismatch in {root}: "
                f"missing={missing}, unindexed shard files={unindexed}."
            )
        return [root / name for name in sorted(advertised_names)], advertised_by_shard

    if single_path.is_file():
        if discovered_shards:
            raise ValueError(
                f"Checkpoint {root} contains both model.safetensors and sharded model files: "
                f"{[path.name for path in discovered_shards]}."
            )
        return [single_path], None
    if discovered_shards:
        return discovered_shards, None
    raise FileNotFoundError(f"No safetensors shards found in {root}.")
