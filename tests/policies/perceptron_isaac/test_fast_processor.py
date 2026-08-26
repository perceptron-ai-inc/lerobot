from __future__ import annotations

import pytest

from lerobot.policies.perceptron_isaac.fast_processor import (
    DEFAULT_FAST_PROCESSOR_TREE_SHA256,
    encode_fast_action_tokens,
    fast_processor_tree_identity,
    materialize_fast_processor_artifact,
    materialize_pinned_fast_processor_snapshot,
)


def test_fast_processor_artifact_materialization_is_movable_and_hash_stable(tmp_path):
    source = tmp_path / "hub_snapshot"
    source.mkdir()
    blob = tmp_path / "blob"
    blob.write_text("processor code")
    (source / "processing_action_tokenizer.py").symlink_to(blob)
    (source / "processor_config.json").write_text("{}")

    artifact = materialize_fast_processor_artifact(source, tmp_path / "packaged_fast")

    assert not (artifact.path / "processing_action_tokenizer.py").is_symlink()
    assert artifact.file_count == 2
    assert fast_processor_tree_identity(artifact.path) == (artifact.tree_sha256, 2)


def test_fast_processor_identity_changes_on_tamper(tmp_path):
    artifact = tmp_path / "fast"
    artifact.mkdir()
    config = artifact / "processor_config.json"
    config.write_text("{}")
    before, _ = fast_processor_tree_identity(artifact)
    config.write_text('{"tampered": true}')
    after, _ = fast_processor_tree_identity(artifact)
    assert before != after


def test_fast_processor_identity_accepts_immutable_hub_snapshot_links(tmp_path):
    repository = tmp_path / "models--org--fast"
    blob_root = repository / "blobs"
    snapshot = repository / "snapshots" / "revision"
    fast_root = snapshot / "fast_processor"
    blob_root.mkdir(parents=True)
    fast_root.mkdir(parents=True)
    blob = blob_root / "processor-config"
    blob.write_text("{}")
    (fast_root / "processor_config.json").symlink_to(blob)

    linked_identity = fast_processor_tree_identity(fast_root)
    materialized = tmp_path / "materialized"
    materialized.mkdir()
    (materialized / "processor_config.json").write_text("{}")

    assert linked_identity == fast_processor_tree_identity(materialized)


def test_pinned_fast_snapshot_materialization_excludes_repository_metadata(tmp_path):
    source = tmp_path / "hub_snapshot"
    source.mkdir()
    for filename in (
        "processing_action_tokenizer.py",
        "processor_config.json",
        "special_tokens_map.json",
        "tokenizer.json",
        "tokenizer_config.json",
    ):
        (source / filename).write_text(filename)
    (source / "README.md").write_text("repository metadata")
    (source / ".gitattributes").write_text("repository metadata")

    artifact = materialize_pinned_fast_processor_snapshot(source, tmp_path / "packaged_fast")

    assert artifact.file_count == 5
    assert not (artifact.path / "README.md").exists()
    assert not (artifact.path / ".gitattributes").exists()


def test_fast_tokens_must_fit_reserved_group():
    assert encode_fast_action_tokens(lambda _: [[1, 7]], object(), reserved_pool_size=8) == [1, 7]
    with pytest.raises(ValueError, match="must be in"):
        encode_fast_action_tokens(lambda _: [[8]], object(), reserved_pool_size=8)


def test_reviewed_fast_tree_identity_is_immutable():
    assert DEFAULT_FAST_PROCESSOR_TREE_SHA256 == (
        "127eb029e5acb8242c7bc2c8efcea6c5dd6ffb565353b5c689b8c8ac55f33a8a"
    )
