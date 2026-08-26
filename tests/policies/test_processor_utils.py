from __future__ import annotations

import numpy as np
import pytest
import torch

from lerobot.policies.processor_utils import (
    coerce_batch_text,
    coerce_rgb_uint8,
    flatten_feature_names,
    to_numpy,
)


def test_rgb_coercion_preserves_layout_options_and_singleton_handling():
    channel_first = np.arange(3 * 2 * 5, dtype=np.uint8).reshape(3, 2, 5)
    view = coerce_rgb_uint8(channel_first, shape_error="bad {shape}")
    contiguous = coerce_rgb_uint8(channel_first, shape_error="bad {shape}", contiguous=True)
    squeezed = coerce_rgb_uint8(
        channel_first[None, None],
        shape_error="bad {shape}",
        squeeze_leading_singletons=True,
    )

    assert to_numpy(channel_first) is channel_first
    assert view.dtype == np.uint8
    assert view.shape == (2, 5, 3)
    assert not view.flags.c_contiguous
    assert contiguous.flags.c_contiguous
    np.testing.assert_array_equal(squeezed, view)
    with pytest.raises(ValueError, match=r"bad \(1, 1, 3, 2, 5\)"):
        coerce_rgb_uint8(channel_first[None, None], shape_error="bad {shape}")


def test_rgb_coercion_preserves_scaling_and_numpy_identity():
    source = np.asarray([[0.0, 0.5, 1.0]], dtype=np.float32)
    np.testing.assert_array_equal(
        coerce_rgb_uint8(source, shape_error="bad {shape}"),
        np.asarray([[[0, 0, 0], [127, 127, 127], [255, 255, 255]]], dtype=np.uint8),
    )


def test_batch_text_coercion_broadcasts_and_keeps_policy_callback_explicit():
    assert coerce_batch_text(torch.tensor(7), 2) == ["7", "7"]
    assert coerce_batch_text(np.asarray(["pick", "place"]), 2, normalizer=str.upper) == ["PICK", "PLACE"]
    with pytest.raises(ValueError, match=r"Expected 3 Isaac task strings, got 2\."):
        coerce_batch_text(["pick", "place"], 3, item_label="Isaac task strings")


def test_feature_names_flatten_groups_in_declaration_order():
    grouped = {"left": ["l1", "l2"], "unused": None, "right": "r1"}
    assert flatten_feature_names(grouped) == ["l1", "l2", "r1"]
    assert flatten_feature_names([]) == []
    assert flatten_feature_names([], empty_as_none=True) is None
