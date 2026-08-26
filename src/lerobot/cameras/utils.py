#!/usr/bin/env python

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

from typing import Any, cast

from numpy.typing import NDArray  # type: ignore  # TODO: add type stubs for numpy.typing

from lerobot.transforms import letterbox_resize_image
from lerobot.utils.import_utils import make_device_from_device_class

from .camera import Camera
from .configs import CameraConfig, Cv2Rotation, OutputResizeMode


def make_cameras_from_configs(camera_configs: dict[str, CameraConfig]) -> dict[str, Camera]:
    cameras: dict[str, Camera] = {}

    for key, cfg in camera_configs.items():
        # TODO(Steven): Consider just using the make_device_from_device_class for all types
        if cfg.type == "opencv":
            from .opencv import OpenCVCamera

            cameras[key] = OpenCVCamera(cfg)

        elif cfg.type == "intelrealsense":
            from .realsense.camera_realsense import RealSenseCamera

            cameras[key] = RealSenseCamera(cfg)

        elif cfg.type == "reachy2_camera":
            from .reachy2_camera.reachy2_camera import Reachy2Camera

            cameras[key] = Reachy2Camera(cfg)

        elif cfg.type == "zmq":
            from .zmq.camera_zmq import ZMQCamera

            cameras[key] = ZMQCamera(cfg)

        else:
            try:
                cameras[key] = cast(Camera, make_device_from_device_class(cfg))
            except Exception as e:
                raise ValueError(f"Error creating camera {key} with config {cfg}: {e}") from e

    return cameras


def get_cv2_rotation(rotation: Cv2Rotation) -> int | None:
    import cv2  # type: ignore  # TODO: add type stubs for OpenCV

    if rotation == Cv2Rotation.ROTATE_90:
        return int(cv2.ROTATE_90_CLOCKWISE)
    elif rotation == Cv2Rotation.ROTATE_180:
        return int(cv2.ROTATE_180)
    elif rotation == Cv2Rotation.ROTATE_270:
        return int(cv2.ROTATE_90_COUNTERCLOCKWISE)
    else:
        return None


def resize_camera_frame(
    image: NDArray[Any],
    width: int,
    height: int,
    *,
    is_depth: bool = False,
    resize_mode: OutputResizeMode | str = OutputResizeMode.STRETCH,
) -> NDArray[Any]:
    """Resize an HWC color frame or HW depth frame with the selected geometry contract.

    ``stretch`` uses Torchvision's exact two-dimensional resize. ``letterbox`` calls
    the same centered aspect-preserving primitive exposed to dataset transforms.
    Depth maps use nearest-neighbor interpolation to avoid inventing range values.
    """
    if image.shape[:2] == (height, width):
        return image

    import numpy as np  # type: ignore  # TODO: add type stubs for numpy
    import torch
    from torchvision.transforms import InterpolationMode
    from torchvision.transforms.v2 import functional as tvf

    contiguous = np.ascontiguousarray(image)
    if contiguous.ndim == 2:
        tensor = torch.from_numpy(contiguous).unsqueeze(0)
    elif contiguous.ndim == 3:
        tensor = torch.from_numpy(contiguous).permute(2, 0, 1)
    else:
        raise ValueError(f"Expected an HW or HWC camera frame, got shape {contiguous.shape}.")

    interpolation = InterpolationMode.NEAREST_EXACT if is_depth else InterpolationMode.BILINEAR
    mode = OutputResizeMode(resize_mode)
    if mode is OutputResizeMode.LETTERBOX:
        resized = letterbox_resize_image(
            tensor,
            (height, width),
            interpolation=interpolation,
            antialias=not is_depth,
        )
    else:
        resized = tvf.resize_image(
            tensor,
            [height, width],
            interpolation=interpolation,
            antialias=not is_depth,
        )
    if contiguous.ndim == 2:
        return resized.squeeze(0).numpy()
    return resized.permute(1, 2, 0).numpy()
