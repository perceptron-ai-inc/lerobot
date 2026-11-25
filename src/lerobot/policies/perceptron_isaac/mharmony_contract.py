"""Version contract between Perceptron ISAAC and standalone mharmony."""

from __future__ import annotations

SUPPORTED_MHARMONY_VERSION = "0.1.0"


def normalize_mharmony_version_marker(version: str | None) -> str:
    """Resolve checkpoint metadata to the supported standalone package version.

    A missing version means the supported one. Any other version is rejected instead of
    being guessed.
    """
    if version is None:
        return SUPPORTED_MHARMONY_VERSION
    if version != SUPPORTED_MHARMONY_VERSION:
        raise ValueError(
            f"Perceptron ISAAC supports mharmony=={SUPPORTED_MHARMONY_VERSION}, got {version!r}."
        )
    return version
