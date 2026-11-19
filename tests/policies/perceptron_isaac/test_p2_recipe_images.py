"""P2.04 recipe ingestion and real image-grid routing, without weight allocation.

The fixture uses the frozen Genesis recipes[0].image_preprocess schema. It is not
an authenticated model export: full manifest/SHA admission remains in P1 tests.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, NotRequired, TypedDict

import pytest

from lerobot.lerobot_types import TransitionKey
from lerobot.policies.perceptron_isaac.configuration_perceptron_isaac import PerceptronIsaacConfig
from lerobot.policies.perceptron_isaac.mharmony_native import (
    IsaacMharmonyRenderMetadata,
    load_native_render_metadata,
)
from lerobot.policies.perceptron_isaac.processor_perceptron_isaac import PERCEPTRON_ISAAC_STREAM_KEY
from lerobot.policies.perceptron_isaac.tensor_stream import TextType, VisionType
from lerobot.processor.converters import batch_to_transition
from tests.policies.perceptron_isaac._p2_data_fixtures import (
    action_event,
    data_batch,
    data_config,
    data_pack,
    data_stats_payload,
)


class ImageRecipe(TypedDict):
    patch_size: int
    pixel_shuffle_scale_factor: int
    temporal_patch_size: int
    max_num_patches: int | None
    min_num_patches: int | None
    image_resize_size: NotRequired[list[int]]


def _recipe_groups() -> list[dict[str, Any]]:
    # Shipped order is descending offsets. Neither pool may be bump-allocated.
    return [
        {"name": None, "offset": 249321, "size": 2048, "tokenizer": "physical-intelligence/fast"},
        {"name": "coord", "offset": 248320, "size": 1001, "tokenizer": None},
    ]


def _image_recipe(*, bounded: bool, fixed: bool) -> ImageRecipe:
    recipe: ImageRecipe = {
        "patch_size": 16,
        "pixel_shuffle_scale_factor": 2,
        "temporal_patch_size": 2,
        "max_num_patches": 784 if bounded else 576,
        "min_num_patches": 784 if bounded else None,
    }
    if fixed:
        recipe["image_resize_size"] = [448, 448]
    return recipe


def _recipe_config(
    tmp_path: Path,
    recipe: ImageRecipe | None,
    *,
    image_size: tuple[int, int] = (360, 640),
) -> PerceptronIsaacConfig:
    config = data_config(tmp_path, data_stats_payload(), image_size=image_size)
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "isaac_0_5"}))
    package = tmp_path / "lerobot_policy"
    package.mkdir()
    config.pretrained_path = package
    if recipe is not None:
        (tmp_path / "policy_inference_recipe.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "recipes": [
                        {
                            "rendering": {
                                "mharmony_encoding": "QWEN35_HARMONY",
                                "reserved_token_groups": _recipe_groups(),
                                "system_default_hints": [],
                            },
                            "image_preprocess": recipe,
                        }
                    ],
                }
            )
        )
    return config


def test_recipe_image_budget_is_loaded_without_moving_reserved_pools(tmp_path: Path) -> None:
    config = _recipe_config(tmp_path, _image_recipe(bounded=True, fixed=True))
    metadata = load_native_render_metadata(config)
    assert metadata.mharmony_reserved_token_groups == _recipe_groups()
    assert (metadata.max_num_patches, metadata.min_num_patches) == (784, 784)
    assert (metadata.patch_size, metadata.pixel_shuffle_scale, metadata.temporal_patch_size) == (16, 2, 2)
    assert metadata.image_size == [360, 640], "Camera input geometry is not the model's fixed resize."


@pytest.mark.parametrize(
    "image_size,recipe,grid,tokens",
    [
        pytest.param((360, 640), None, (18, 32), 144, id="no-recipe-default-control"),
        pytest.param(
            (256, 256), _image_recipe(bounded=False, fixed=False), (16, 16), 64, id="libero-control"
        ),
        pytest.param(
            (256, 256), _image_recipe(bounded=True, fixed=False), (28, 28), 196, id="libero-196-budget"
        ),
        pytest.param(
            (360, 640), _image_recipe(bounded=True, fixed=False), (20, 36), 180, id="bounds-not-square"
        ),
        pytest.param((360, 640), _image_recipe(bounded=True, fixed=True), (28, 28), 196, id="fixed-square"),
    ],
)
def test_recipe_image_geometry_reaches_factory_pack_and_real_renderer(
    tmp_path: Path,
    image_size: tuple[int, int],
    recipe: ImageRecipe | None,
    grid: tuple[int, int],
    tokens: int,
) -> None:
    config = _recipe_config(tmp_path, recipe, image_size=image_size)
    # Follow the imported-package path: resolve recipe once, then carry the real
    # metadata sidecar through the maintained factory (not a patched renderer).
    sidecar = tmp_path / "native_render_metadata.json"
    sidecar.write_text(json.dumps(load_native_render_metadata(config).to_json_dict()))
    config.native_render_metadata_path = str(sidecar)
    pack, _ = data_pack(config)
    out = pack(batch_to_transition(data_batch(image_size=image_size)))
    stream = out[TransitionKey.COMPLEMENTARY_DATA][PERCEPTRON_ISAAC_STREAM_KEY]
    images = [event for event in stream.streams[0].events if event.type == VisionType.I]
    assert len(images) == 1
    assert images[0].dims_real == [1, *grid]
    assert images[0].num_tokens() == tokens
    assert images[0].dims_virtual == [1, grid[0] // 2, grid[1] // 2]
    if recipe is not None:
        assert pack._ensure_renderer().metadata.mharmony_reserved_token_groups == _recipe_groups()
        assert action_event(stream, TextType.action).data.reshape(-1).tolist() == [249328, 249329]
    else:
        assert pack._ensure_renderer().metadata.mharmony_reserved_token_groups == [
            {"tokenizer": "physical-intelligence/fast", "size": 2048}
        ]
        assert action_event(stream, TextType.action).data.reshape(-1).tolist() == [248094, 248095]
    assert action_event(stream).tags["action_is_pad"] == [False, False]


def test_raw_recipe_reaches_factory_without_an_intermediate_sidecar(tmp_path: Path) -> None:
    config = _recipe_config(tmp_path, _image_recipe(bounded=True, fixed=True))
    pack, _ = data_pack(config)
    out = pack(batch_to_transition(data_batch(image_size=(360, 640))))
    stream = out[TransitionKey.COMPLEMENTARY_DATA][PERCEPTRON_ISAAC_STREAM_KEY]
    image = next(event for event in stream.streams[0].events if event.type == VisionType.I)
    assert image.dims_real == [1, 28, 28]
    assert action_event(stream, TextType.action).data.reshape(-1).tolist() == [249328, 249329]


def test_retained_assets_do_not_discover_a_recipe_in_the_original_package(tmp_path: Path) -> None:
    config = _recipe_config(tmp_path, _image_recipe(bounded=True, fixed=True))
    retained = tmp_path / "retained"
    retained.mkdir()
    # Existing policy-loader handoff, not a new test-only production hook.
    config._pretrained_processor_root = str(retained)
    assert config.resolve_native_recipe_path() is None
    metadata = load_native_render_metadata(config)
    assert metadata.max_num_patches == 576
    assert metadata.image_resize_size is None
    assert metadata.mharmony_reserved_token_groups == [
        {"tokenizer": "physical-intelligence/fast", "size": 2048}
    ]


def test_explicit_render_sidecar_keeps_precedence_over_recipe(tmp_path: Path) -> None:
    config = _recipe_config(tmp_path, _image_recipe(bounded=True, fixed=True))
    declared = IsaacMharmonyRenderMetadata.from_config(config).to_json_dict()
    declared.update(max_num_patches=256, min_num_patches=64)
    sidecar = tmp_path / "native_render_metadata.json"
    sidecar.write_text(json.dumps(declared))
    config.native_render_metadata_path = str(sidecar)
    metadata = load_native_render_metadata(config)
    assert metadata.to_json_dict() == declared


def test_runtime_image_override_cannot_silently_conflict_with_recipe(tmp_path: Path) -> None:
    config = _recipe_config(tmp_path, _image_recipe(bounded=True, fixed=False))
    config.render_max_num_patches = 576
    config.render_min_num_patches = 576
    with pytest.raises(ValueError, match="(?i)recipe|preprocess|patch"):
        load_native_render_metadata(config)


@pytest.mark.parametrize("resize", [[0, 448], [-1, 448], [448], [448, 448, 448]])
def test_invalid_recipe_fixed_resize_is_rejected_instead_of_ignored(
    tmp_path: Path, resize: list[int]
) -> None:
    recipe = _image_recipe(bounded=True, fixed=True)
    recipe["image_resize_size"] = resize
    config = _recipe_config(tmp_path, recipe)
    with pytest.raises(ValueError, match="(?i)resize|image"):
        load_native_render_metadata(config)


def _write_recipe_image_declarations(tmp_path: Path, images: Sequence[Mapping[str, Any] | None]) -> None:
    path = tmp_path / "policy_inference_recipe.json"
    payload = json.loads(path.read_text())
    rendering = payload["recipes"][0]["rendering"]
    payload["recipes"] = [
        {"rendering": rendering, **({"image_preprocess": image} if image is not None else {})}
        for image in images
    ]
    path.write_text(json.dumps(payload))


@pytest.mark.parametrize(
    "images",
    [
        pytest.param([None, _image_recipe(bounded=True, fixed=True)], id="missing-first"),
        pytest.param([_image_recipe(bounded=True, fixed=True), None], id="missing-second"),
        pytest.param([None, {"unsupported": 1}], id="invalid-second"),
        pytest.param(
            [
                _image_recipe(bounded=True, fixed=True),
                {**_image_recipe(bounded=True, fixed=True), "patch_size": 16.0},
            ],
            id="invalid-equal-second-number",
        ),
        pytest.param(
            [_image_recipe(bounded=True, fixed=True), _image_recipe(bounded=False, fixed=False)],
            id="different-declarations",
        ),
    ],
)
def test_image_preprocessing_agreement_checks_every_recipe(
    tmp_path: Path, images: list[dict[str, Any] | None]
) -> None:
    config = _recipe_config(tmp_path, _image_recipe(bounded=True, fixed=True))
    _write_recipe_image_declarations(tmp_path, images)
    with pytest.raises(ValueError, match="(?i)ambiguous|agree|recipe"):
        load_native_render_metadata(config)


@pytest.mark.parametrize("present", [False, True], ids=["all-default", "all-explicit"])
def test_agreeing_image_preprocessing_recipes_retain_defaults_or_declared_resize(
    tmp_path: Path, present: bool
) -> None:
    config = _recipe_config(tmp_path, _image_recipe(bounded=True, fixed=True))
    image = _image_recipe(bounded=True, fixed=True) if present else None
    _write_recipe_image_declarations(tmp_path, [image, image])
    metadata = load_native_render_metadata(config)
    assert metadata.image_resize_size == ([448, 448] if present else None)
    assert metadata.max_num_patches == (784 if present else 576)
