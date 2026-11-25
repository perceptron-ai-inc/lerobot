"""Package-relative paths for the processor sidecars of a saved Perceptron ISAAC checkpoint."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .checkpoint_integrity import load_json_object

PROCESSOR_PACKAGE_PATHS = {
    "fast_processor_path": ("fast_processor", True),
    "native_render_metadata_path": ("native_render_metadata.json", False),
    "native_stats_path": ("isaac_stats.json", False),
    "suite_stats_path": ("isaac_suite_stats.json", False),
    "suite_by_task_index_path": ("isaac_suite_by_task_index.json", False),
}


def _processor_configs(root: Path) -> list[Path]:
    return [
        path
        for path in sorted(root.iterdir())
        if path.is_file()
        and path.name.startswith(("policy_preprocessor", "policy_postprocessor"))
        and path.suffix == ".json"
    ]


def rewrite_trained_processor_paths(package_root: str | Path) -> None:
    """Rewrite serialized ISAAC processor sidecars to package-relative paths in place."""
    root = Path(package_root)
    suite_available = all(
        (root / filename).is_file()
        for filename in (
            "isaac_suite_stats.json",
            "isaac_suite_by_task_index.json",
        )
    )

    def rewrite(value: Any, *, filename: str) -> Any:
        if isinstance(value, list):
            return [rewrite(item, filename=filename) for item in value]
        if not isinstance(value, dict):
            return value
        rewritten = {key: rewrite(item, filename=filename) for key, item in value.items()}
        for key in ("config_toml_path", "stats_path"):
            if key in rewritten:
                rewritten[key] = None
        for key, (relative, is_directory) in PROCESSOR_PACKAGE_PATHS.items():
            if key not in rewritten:
                continue
            if key.startswith("suite_"):
                rewritten[key] = relative if suite_available else None
                continue
            if rewritten[key] in (None, ""):
                continue
            candidate = root / relative
            available = candidate.is_dir() if is_directory else candidate.is_file()
            if not available:
                raise RuntimeError(
                    f"Serialized processor {filename} declares {key}, but package asset {relative} is missing."
                )
            rewritten[key] = relative
        return rewritten

    for config_path in _processor_configs(root):
        try:
            payload = load_json_object(config_path)
        except ValueError as exc:
            raise RuntimeError(f"Cannot rewrite serialized processor paths: {exc}") from exc
        rewritten = rewrite(payload, filename=config_path.name)
        temporary = config_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(rewritten, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        temporary.replace(config_path)
