"""Version contract between Perceptron ISAAC and standalone mharmony."""

from __future__ import annotations

import warnings

SUPPORTED_MHARMONY_VERSION = "0.1.0"
LEGACY_MHARMONY_VERSION_MARKER = "genesis-in-tree"


def normalize_mharmony_version_marker(version: str | None) -> str:
    """Resolve checkpoint metadata to the supported standalone package version.

    Early native ISAAC packages recorded ``genesis-in-tree`` because mharmony
    had not been released separately yet. Version 0.1.0 is the compatibility
    extraction of that contract, so those packages remain readable and are
    upgraded in memory. Any other version is rejected instead of being guessed.
    """
    if version is None:
        return SUPPORTED_MHARMONY_VERSION
    if version == LEGACY_MHARMONY_VERSION_MARKER:
        warnings.warn(
            "Legacy mharmony marker 'genesis-in-tree' maps to standalone mharmony==0.1.0; "
            "re-save the package to persist the public package version.",
            FutureWarning,
            stacklevel=2,
        )
        return SUPPORTED_MHARMONY_VERSION
    if version != SUPPORTED_MHARMONY_VERSION:
        raise ValueError(
            f"Perceptron ISAAC supports mharmony=={SUPPORTED_MHARMONY_VERSION}, got {version!r}."
        )
    return version
