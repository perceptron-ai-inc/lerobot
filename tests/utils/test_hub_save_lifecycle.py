"""HubMixin owns standalone package finalization and rejects deferred publication."""

from pathlib import Path
from unittest.mock import Mock

import pytest

from lerobot.utils.hub import HubMixin


class _FilePackage(HubMixin):
    """A non-policy package with real local serialization and a finalizer-owned sidecar."""

    def __init__(self) -> None:
        self.finalizations: list[Path] = []

    def _save_pretrained(self, save_directory: Path) -> None:
        (save_directory / "payload.txt").write_text("saved payload\n")

    def finalize_pretrained_package(self, save_directory: Path) -> None:
        assert (save_directory / "payload.txt").read_text() == "saved payload\n"
        self.finalizations.append(save_directory)
        (save_directory / "finalized.txt").write_text("complete package\n")


def test_shared_hub_save_finalizes_non_policy_package_once(tmp_path: Path) -> None:
    package = _FilePackage()
    saved = tmp_path / "standalone"

    result = package.save_pretrained(saved)

    assert package.finalizations == [saved]
    assert (saved / "finalized.txt").read_text() == "complete package\n"
    assert result is None


def test_deferred_save_rejects_publication_before_any_writes_or_upload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    package = _FilePackage()
    saved = tmp_path / "must-not-exist"
    # Replace only the upload boundary; no real network call can occur, even before the guard exists.
    push = Mock(return_value="offline-upload-result")
    monkeypatch.setattr(package, "push_to_hub", push)

    with pytest.raises(ValueError, match="finaliz"):
        package.save_pretrained(
            saved,
            finalize_package=False,
            push_to_hub=True,
            repo_id="test/package",
        )

    push.assert_not_called()
    assert not saved.exists()
    assert package.finalizations == []
