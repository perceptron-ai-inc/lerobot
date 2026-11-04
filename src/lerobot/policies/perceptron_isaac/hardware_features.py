"""Reviewed semantic aliases for the authenticated SO100/SO101 wire layout."""


def so100_hardware_feature_aliases() -> dict[str, str]:
    """Map legacy semantic joint names to hardware names, without reordering values."""
    return {
        "main_shoulder_pan": "shoulder_pan.pos",
        "main_shoulder_lift": "shoulder_lift.pos",
        "main_elbow_flex": "elbow_flex.pos",
        "main_wrist_flex": "wrist_flex.pos",
        "main_wrist_roll": "wrist_roll.pos",
        "main_gripper": "gripper.pos",
    }
