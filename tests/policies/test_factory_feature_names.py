"""Dataset feature names must survive the trip into a policy config.

A LeRobot dataset expresses a feature's ``names`` either as a flat list or as a mapping from
motor group to names. Flattening the mapping form incorrectly produced a one-element list of
group keys, which silently mislabels the action dimensions for every policy that consumes them
-- and makes a finetuned checkpoint unloadable for any policy that validates the length against
its action dimension.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from lerobot.policies.factory import _flatten_feature_names, _set_action_feature_names_from_dataset
from lerobot.utils.constants import ACTION


@pytest.mark.parametrize(
    ("names", "expected"),
    [
        (None, None),
        (["x", "y", "z"], ["x", "y", "z"]),
        # The form used by lerobot/libero_spatial_image and friends.
        (
            {"motors": ["x", "y", "z", "roll", "pitch", "yaw", "gripper"]},
            ["x", "y", "z", "roll", "pitch", "yaw", "gripper"],
        ),
        # Multiple motor groups concatenate in declaration order.
        (
            {"left": ["l1", "l2"], "right": ["r1", "r2"]},
            ["l1", "l2", "r1", "r2"],
        ),
        # A degenerate single-string group is still a name, not an iterable of characters.
        ({"gripper": "grip"}, ["grip"]),
    ],
)
def test_feature_names_flatten_to_per_dimension_names(names, expected):
    assert _flatten_feature_names(names) == expected


def test_grouped_names_are_not_collapsed_to_group_keys():
    """The regression itself: ``list({"motors": [...]})`` returns ``["motors"]``."""
    grouped = {"motors": ["x", "y", "z", "roll", "pitch", "yaw", "gripper"]}

    assert _flatten_feature_names(grouped) != list(grouped)
    assert len(_flatten_feature_names(grouped)) == 7


def test_strict_hardware_action_names_are_not_replaced_by_dataset_aliases():
    hardware_names = ["shoulder_pan.pos", "shoulder_lift.pos"]
    cfg = SimpleNamespace(action_feature_names=hardware_names.copy(), strict_hardware_feature_contract=True)

    _set_action_feature_names_from_dataset(
        cfg, {ACTION: {"names": ["main_shoulder_pan", "main_shoulder_lift"]}}
    )

    assert cfg.action_feature_names == hardware_names


def test_non_strict_policy_still_adopts_dataset_action_names():
    cfg = SimpleNamespace(action_feature_names=["old"], strict_hardware_feature_contract=False)

    _set_action_feature_names_from_dataset(cfg, {ACTION: {"names": ["new_a", "new_b"]}})

    assert cfg.action_feature_names == ["new_a", "new_b"]
