"""Isolate the SDK vocabulary environment between independent ISAAC tests."""

import os
from collections.abc import Iterator

import pytest


@pytest.fixture(autouse=True)
def restore_qwen35_vocab_environment() -> Iterator[None]:
    """Package loading intentionally selects a process-global SDK vocabulary path."""
    original = os.environ.get("QWEN35_VOCAB_PATH")
    try:
        yield
    finally:
        if original is None:
            os.environ.pop("QWEN35_VOCAB_PATH", None)
        else:
            os.environ["QWEN35_VOCAB_PATH"] = original
