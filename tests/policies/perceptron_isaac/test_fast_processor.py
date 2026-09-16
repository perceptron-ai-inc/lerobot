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


@pytest.fixture
def tiny_fast_artifact(tmp_path, monkeypatch):
    """Generate BPE DATA directly; never fit or execute artifact Python."""
    import json

    from tokenizers import Tokenizer, decoders, models

    from lerobot.policies.perceptron_isaac import fast_processor as fast

    root = tmp_path / "tiny-fast"
    root.mkdir()
    vocabulary = {chr(i): i for i in range(128)}
    vocabulary["AA"] = 128
    tokenizer = Tokenizer(models.BPE(vocab=vocabulary, merges=[("A", "A")]))
    tokenizer.decoder = decoders.Fuse()
    tokenizer.save(str(root / "tokenizer.json"))
    (root / "processing_action_tokenizer.py").write_text(
        'raise RuntimeError("ARTIFACT PYTHON MUST NEVER RUN")\n'
    )
    (root / "processor_config.json").write_text(
        json.dumps(
            {
                "processor_class": "UniversalActionProcessor",
                "auto_map": {"AutoProcessor": "processing_action_tokenizer.UniversalActionProcessor"},
                "scale": 10,
                "min_token": -65,
                "vocab_size": 129,
                "time_horizon": None,
                "action_dim": None,
            }
        )
    )
    (root / "tokenizer_config.json").write_text(
        json.dumps(
            {
                "clean_up_tokenization_spaces": True,
                "model_max_length": 512,
            }
        )
    )
    (root / "special_tokens_map.json").write_text("{}")
    digest, _ = fast.fast_processor_tree_identity(root)
    # Explicit fixture identity boundary; production pin remains fixed in source.
    monkeypatch.setattr(fast, "DEFAULT_FAST_PROCESSOR_TREE_SHA256", digest)

    def reject_checkpoint_code(*args, **kwargs):
        raise AssertionError("NATIVE_FAST_NO_CHECKPOINT_CODE")

    # Module-local compile interception runs BEFORE compile/exec of any artifact.
    # Do not patch builtins: normal dependency imports must continue to work.
    monkeypatch.setitem(fast.__dict__, "compile", reject_checkpoint_code)
    monkeypatch.setitem(fast.__dict__, "exec", reject_checkpoint_code)
    return root, digest


def test_fast_processor_never_executes_checkpoint_python(tiny_fast_artifact):
    from lerobot.policies.perceptron_isaac.fast_processor import load_fast_action_processor

    root, digest = tiny_fast_artifact
    processor = load_fast_action_processor(root, expected_tree_sha256=digest)
    assert type(processor).__module__ == "lerobot.policies.perceptron_isaac.fast_processor"
    assert processor.bpe_tokenizer.clean_up_tokenization_spaces is True
    assert processor.bpe_tokenizer.model_max_length == 512
    assert not (root / "__pycache__").exists()


@pytest.fixture
def native_fast(tiny_fast_artifact):
    from lerobot.policies.perceptron_isaac.fast_processor import load_fast_action_processor

    root, digest = tiny_fast_artifact
    return load_fast_action_processor(root, expected_tree_sha256=digest)


def test_fast_processor_analytic_dct_bpe_and_inverse(native_fast):
    import numpy as np

    actions = np.array([[np.sqrt(2) / 10, 0], [0, 0]])
    # DCT-II along time gives [[.1, 0], [.1, 0]], shifted characters BABA.
    assert native_fast(actions) == [[66, 65, 66, 65]]
    np.testing.assert_allclose(native_fast.decode([[66, 65, 66, 65]]), actions[None], atol=1e-15)
    # The real BPE merges AA, proving this is not a character-ID stand-in.
    assert native_fast(np.zeros((2, 2))) == [[128, 128]]
    np.testing.assert_array_equal(native_fast.decode([[128, 128]]), np.zeros((1, 2, 2)))


def test_fast_processor_quantization_clipping_and_batch(native_fast):
    import numpy as np

    # Real FFT coverage uses coefficients away from half ties: floating DCT
    # arithmetic need not preserve a mathematical tie bit-for-bit.
    actions = np.array([[[0.04, 0.16, 0.24, -0.04, -7]]])
    # Rounded coefficients: 0, 2, 2, 0, -70; shifted negatives clamp to NUL.
    assert native_fast(actions) == [[65, 67, 67, 65, 0]]
    decoded = native_fast.decode([[65, 67, 67, 65, 0]])
    np.testing.assert_allclose(decoded, [[[0, 0.2, 0.2, 0, -6.5]]])
    assert native_fast(np.zeros((2, 1, 2))) == [[128], [128]]


@pytest.mark.parametrize(
    ("direction", "rounded"),
    [
        (-float("inf"), [-3, -2, -1, 0, 1, 2]),
        (None, [-2, -2, 0, 0, 2, 2]),
        (float("inf"), [-2, -1, 0, 1, 2, 3]),
    ],
)
def test_fast_processor_nearest_even_and_adjacent_boundaries(native_fast, monkeypatch, direction, rounded):
    import numpy as np
    import scipy.fft

    # Inject only the FFT result to isolate exact binary half ties and their
    # immediate neighbors. Separate analytic/real-FFT tests retain end-to-end coverage.
    coefficients = np.array([-2.5, -1.5, -0.5, 0.5, 1.5, 2.5]).reshape(6, 1, 1)
    if direction is not None:
        coefficients = np.nextafter(coefficients, direction)
    actions = np.zeros_like(coefficients)

    def exact_dct(values: np.ndarray, *, axis: int, norm: str) -> np.ndarray:
        np.testing.assert_array_equal(values, actions)
        assert axis == 1 and norm == "ortho"
        return coefficients.copy()

    monkeypatch.setitem(scipy.fft.__dict__, "dct", exact_dct)
    native_fast.scale = 1
    # One character per batch row avoids the fixture's AA merge.
    assert native_fast(actions) == [[value - native_fast.min_token] for value in rounded]


def test_fast_processor_geometry_precedence_and_persistent_decode_override(native_fast):
    import numpy as np

    with pytest.raises(ValueError, match="requires geometry"):
        native_fast.decode([[128]])
    native_fast(np.zeros((1, 2)))
    assert native_fast.decode([[128]]).shape == (1, 1, 2)
    assert native_fast.decode([[128]], time_horizon=2, action_dim=1).shape == (1, 2, 1)
    native_fast(np.zeros((3, 1)))
    assert native_fast.decode([[128]]).shape == (1, 2, 1)
    assert native_fast.called_time_horizon == 2
    with pytest.raises(ValueError, match="positive integer"):
        native_fast.decode([[128]], time_horizon=0)


def test_fast_processor_constructor_geometry_and_cleanup(native_fast):
    import numpy as np

    from lerobot.policies.perceptron_isaac.fast_processor import NativeFastActionProcessor

    processor = NativeFastActionProcessor(
        native_fast.bpe_tokenizer, min_token=-65, time_horizon=1, action_dim=2
    )
    processor(np.zeros((3, 1)))
    assert processor.decode([[128]]).shape == (1, 1, 2)
    # Default native tokenizer cleanup changes 'A .' to 'A.'.
    assert processor.bpe_tokenizer.decode([65, 32, 46]) == "A."
    np.testing.assert_allclose(processor.decode([[65, 32, 46]]), [[[0, -1.9]]])
    for scale in (0, -1, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="positive and finite"):
            NativeFastActionProcessor(processor.bpe_tokenizer, scale=scale)


@pytest.mark.parametrize("row", [[], [65], [65, 65, 65], [-1]])
def test_fast_processor_malformed_decode_returns_zero_coefficients(native_fast, row):
    import numpy as np

    np.testing.assert_array_equal(
        native_fast.decode([row], time_horizon=1, action_dim=2), np.zeros((1, 1, 2))
    )


@pytest.mark.parametrize("actions", [0, [1], [[[]]], [[float("nan")]], [[float("inf")]], [[1j]], [["x"]]])
def test_fast_processor_rejects_malformed_actions(native_fast, actions):
    import numpy as np

    with pytest.raises(ValueError, match="FAST actions"):
        native_fast(np.asarray(actions))


def test_fast_processor_unicode_and_empty_decode_boundaries(native_fast):
    import numpy as np

    with pytest.raises(ValueError, match="Unicode"):
        native_fast(np.array([[1e7]]))
    with pytest.raises(ValueError, match="at least one"):
        native_fast.decode([], time_horizon=1, action_dim=1)


def test_fast_processor_loading_rejects_tamper_and_unreviewed_digest(tiny_fast_artifact):
    from lerobot.policies.perceptron_isaac.fast_processor import load_fast_action_processor

    root, digest = tiny_fast_artifact
    with pytest.raises(ValueError, match="reviewed pinned revision"):
        load_fast_action_processor(root, expected_tree_sha256="f" * 64)
    (root / "processing_action_tokenizer.py").write_text("changed inert data")
    with pytest.raises(ValueError, match="identity mismatch"):
        load_fast_action_processor(root, expected_tree_sha256=digest)


def test_fast_processor_rejects_uncontained_symlinks(tmp_path):
    root = tmp_path / "fast"
    root.mkdir()
    external = tmp_path / "external"
    external.write_text("{}")
    (root / "processor_config.json").symlink_to(external)
    with pytest.raises(ValueError, match="symlinks"):
        fast_processor_tree_identity(root)
    link = tmp_path / "root-link"
    link.symlink_to(root, target_is_directory=True)
    with pytest.raises(ValueError, match="real directory"):
        fast_processor_tree_identity(link)
