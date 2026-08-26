# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass

from ..configs import CameraConfig, ColorMode, Cv2Rotation, OutputResizeMode


@CameraConfig.register_subclass("intelrealsense")
@dataclass
class RealSenseCameraConfig(CameraConfig):
    """Configuration class for Intel RealSense cameras.

    This class provides specialized configuration options for Intel RealSense cameras,
    including support for depth sensing and device identification via serial number or name.

    Example configurations for Intel RealSense D405:
    ```python
    # Basic configurations
    RealSenseCameraConfig("0123456789", 30, 1280, 720)  # 1280x720 @ 30FPS
    RealSenseCameraConfig("0123456789", 60, 640, 480)  # 640x480 @ 60FPS

    # Advanced configurations
    RealSenseCameraConfig("0123456789", 30, 640, 480, use_depth=True)  # With depth sensing
    RealSenseCameraConfig("0123456789", 30, 640, 480, rotation=Cv2Rotation.ROTATE_90)  # With 90° rotation
    ```

    Attributes:
        fps: Requested frames per second for the color stream.
        width: Requested frame width in pixels for the color stream.
        height: Requested frame height in pixels for the color stream.
        serial_number_or_name: Unique serial number or human-readable name to identify the camera.
        color_mode: Color mode for image output (RGB or BGR). Defaults to RGB.
        use_rgb: Whether to enable the color stream. Defaults to True.
        use_depth: Whether to enable depth stream. Defaults to False.
        rotation: Image rotation setting (0°, 90°, 180°, or 270°). Defaults to no rotation.
        warmup_s: Time reading frames before returning from connect (in seconds)
        startup_timeout_s: Maximum wait for the first frame. RealSense devices on
            constrained USB links may need substantially longer than their later
            per-frame interval.
        exposure: Manual exposure value for the color sensor. When set, auto-exposure is
            disabled and this fixed value is used. Valid ranges are camera-model specific
            and reported if the value is rejected. Defaults to None (leave unchanged).
        gain: Manual gain value for the color sensor. When set, auto-exposure is disabled
            and this fixed gain is used, which also freezes exposure at its current value
            when no exposure is configured. Valid ranges are camera-model specific and
            reported if the value is rejected. Defaults to None (leave unchanged).
        white_balance: Manual white balance value for the color sensor. When set, auto
            white balance is disabled and this fixed value is used. Valid ranges are
            camera-model specific and reported if the value is rejected. Defaults to None
            (leave unchanged).
        output_width: Optional width returned by the camera after capture and rotation.
            This does not change the sensor mode requested by ``width``/``height``.
        output_height: Optional height returned by the camera after capture and rotation.
            Must be set together with ``output_width``.
        output_resize_mode: ``stretch`` matches the target dimensions directly; ``letterbox``
            preserves aspect ratio and center-pads with black.

    Note:
        - Either name or serial_number must be specified.
        - At least one of `use_rgb` or `use_depth` must be enabled.
        - Depth stream configuration (if enabled) will use the same FPS as the color stream.
        - The actual resolution and FPS may be adjusted by the camera to the nearest supported mode.
        - For `fps`, `width` and `height`, either all of them need to be set, or none of them.
    """

    serial_number_or_name: str
    color_mode: ColorMode = ColorMode.RGB
    use_rgb: bool = True
    use_depth: bool = False
    rotation: Cv2Rotation = Cv2Rotation.NO_ROTATION
    warmup_s: int = 1
    startup_timeout_s: float = 10.0
    exposure: int | None = None
    gain: int | None = None
    white_balance: int | None = None
    output_width: int | None = None
    output_height: int | None = None
    output_resize_mode: OutputResizeMode = OutputResizeMode.STRETCH

    def __post_init__(self) -> None:
        self.color_mode = ColorMode(self.color_mode)
        self.rotation = Cv2Rotation(self.rotation)
        self.output_resize_mode = OutputResizeMode(self.output_resize_mode)
        self.output_resize_mode.validate_dimensions(self.output_width, self.output_height)

        if self.startup_timeout_s <= 0:
            raise ValueError("`startup_timeout_s` must be positive.")

        if not self.use_rgb and not self.use_depth:
            raise ValueError("At least one of `use_rgb` or `use_depth` must be enabled.")

        manual_color_options = {
            "exposure": self.exposure,
            "gain": self.gain,
            "white_balance": self.white_balance,
        }
        configured_color_options = [name for name, value in manual_color_options.items() if value is not None]
        if configured_color_options and not self.use_rgb:
            raise ValueError(
                "Manual color sensor options require `use_rgb=True`. "
                f"Configured options: {configured_color_options}."
            )

        values = (self.fps, self.width, self.height)
        if any(v is not None for v in values) and any(v is None for v in values):
            raise ValueError(
                "For `fps`, `width` and `height`, either all of them need to be set, or none of them."
            )
