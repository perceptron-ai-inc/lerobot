"""P2.03/.04 authenticated render assets survive source retirement and native saves.

Only the external SDK token-serialization leaf is substituted: the tiny native
vocabulary is not a Qwen tokenizer. Package authentication, native weights,
retention, processor state, recipe admission and PNG image admission stay real.
"""

from __future__ import annotations

import base64
import io
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from lerobot.configs import FeatureType, PolicyFeature
from lerobot.lerobot_types import TransitionKey
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.perceptron_isaac.isaac_stats import isaac_stats_to_dict, load_isaac_stats
from lerobot.policies.perceptron_isaac.mharmony_adapter import load_mharmony
from lerobot.policies.perceptron_isaac.mharmony_native import (
    IsaacMharmonyContentPlan,
    _plan_item_to_mh_content,
)
from lerobot.policies.perceptron_isaac.modeling_perceptron_isaac import PerceptronIsaacPolicy
from lerobot.policies.perceptron_isaac.processor_perceptron_isaac import (
    PERCEPTRON_ISAAC_STREAM_KEY,
    PerceptronIsaacMharmonyPackProcessorStep,
)
from lerobot.policies.perceptron_isaac.tensor_stream import VectorType
from lerobot.policies.perceptron_isaac.trained_package import (
    finalize_canonical_trained_package,
    rewrite_trained_processor_paths,
)
from lerobot.processor.converters import batch_to_transition
from lerobot.utils.constants import ACTION, OBS_STATE
from tests.policies.perceptron_isaac._p2_data_fixtures import data_batch, data_stats_payload
from tests.policies.perceptron_isaac._rollout_fixtures import sdk_plans as _sdk_plans
from tests.policies.perceptron_isaac.test_checkpoint_lifecycle import _source_package
from tests.policies.perceptron_isaac.test_checkpoint_resume_paths import (
    _assert_native_inventory,
    _assert_tensor_bits,
)
from tests.policies.perceptron_isaac.test_p2_proprio_profiles import _profile_payload
from tests.policies.perceptron_isaac.test_p2_recipe_images import _image_recipe

sdk_plans = _sdk_plans


def _authenticated_source(root: Path, *, fixed_resize: bool, profiled: bool) -> None:
    config = _source_package(root)
    # Use the public neutral-debug geometry permission, never a patched validator/loader.
    model_config = root / "hf_model" / "config.json"
    raw = json.loads(model_config.read_text())
    raw["isaac05_artifact"].update(artifact_kind="neutral_debug", trained_steps=0)
    model_config.write_text(json.dumps(raw))
    config.artifact_kind = "neutral_debug"
    config.pretrained_path = root
    config.camera_order = ("image",)
    config.image_size = (360, 640)
    config.target_fps = 20.0
    config.input_features["observation.images.image"] = PolicyFeature(
        type=FeatureType.VISUAL, shape=(3, 360, 640)
    )
    config.native_render_metadata_path = None
    (root / "native_render_metadata.json").unlink()
    payload = data_stats_payload(passthrough=True)
    if profiled:
        payload["_profile"] = _profile_payload(payload, 0.75)
    (root / "isaac_stats.json").write_text(json.dumps(payload))

    # Maintained factory contract for older serialized pipelines without inlined
    # render metadata. Recipe discovery must happen after authenticated retention.
    pre, post = make_pre_post_processors(config)
    pre.save_pretrained(root)
    post.save_pretrained(root)
    rewrite_trained_processor_paths(root)
    image = _image_recipe(bounded=True, fixed=fixed_resize)
    (root / "policy_inference_recipe.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "recipes": [
                    {
                        "rendering": {
                            "mharmony_encoding": "QWEN35_HARMONY",
                            # Descending offsets match this tiny checkpoint's authenticated pools.
                            "reserved_token_groups": [
                                {
                                    "name": None,
                                    "offset": 28,
                                    "size": 4,
                                    "tokenizer": "physical-intelligence/fast",
                                },
                                {"name": "coord", "offset": 24, "size": 4, "tokenizer": None},
                            ],
                            "system_default_hints": [],
                        },
                        "image_preprocess": image,
                    }
                ],
            }
        )
    )
    finalize_canonical_trained_package(
        config, root, mk1_source_model_import_sha256=config.mk1_source_model_import_sha256
    )


def _files(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes() for path in root.rglob("*") if path.is_file()
    }


def _assert_restored_render(
    policy: PerceptronIsaacPolicy,
    package: Path,
    plans: list[IsaacMharmonyContentPlan],
    *,
    fixed_resize: bool,
    profiled: bool,
) -> None:
    pre, _ = make_pre_post_processors(policy.config, pretrained_path=str(package))
    pack = next(step for step in pre.steps if isinstance(step, PerceptronIsaacMharmonyPackProcessorStep))
    assert pack.native_stats_path is None, "Stats must come from serialized processor tensor state."
    assert policy.config.native_stats_path == str(package / "isaac_stats.json")
    assert pack._stats is not None, "Processor tensor-state restoration must load real stats."
    stats = isaac_stats_to_dict(pack._stats)
    assert stats["proprio"]["mask"] == [True, False]
    if profiled:
        source_stats = json.loads((package / "isaac_stats.json").read_text())
        assert stats["_profile"] == source_stats["_profile"]
    else:
        assert "_profile" not in stats
    renderer = pack._ensure_renderer()
    metadata = renderer.metadata
    recipe = json.loads((package / "policy_inference_recipe.json").read_text())["recipes"][0]
    assert metadata.mharmony_reserved_token_groups == recipe["rendering"]["reserved_token_groups"]
    assert (metadata.max_num_patches, metadata.min_num_patches) == (784, 784)
    assert (metadata.patch_size, metadata.pixel_shuffle_scale, metadata.temporal_patch_size) == (16, 2, 2)
    assert metadata.image_size == [360, 640]
    assert metadata.image_resize_size == ([448, 448] if fixed_resize else None)

    # Inference avoids substituting FAST or requiring a real tokenizer for a four-token pool.
    batch = data_batch(image_size=(360, 640))
    del batch[ACTION], batch["action_is_pad"]
    batch[OBS_STATE] = torch.tensor([[[1.4, 250.0]]])
    pack.dtype = "float32"
    before = len(plans)
    out = pack(batch_to_transition(batch))
    assert len(plans) == before + 1
    stream = out[TransitionKey.COMPLEMENTARY_DATA][PERCEPTRON_ISAAC_STREAM_KEY]
    vector = next(event for event in stream.streams[0].events if event.type == VectorType.vector)
    expected_joint = 0.75 if profiled else 1.4 / (1.0 + 1e-6) * 2.0 - 1.0
    np.testing.assert_allclose(vector.data.float().numpy().reshape(-1), [expected_joint, 250.0])

    # sdk_plans skips media serialization too. Explicitly run the untouched production
    # image admission on its real plan, so fixed resize cannot pass on metadata alone.
    images = [item for item in plans[-1].user_content if item["kind"] == "image"]
    assert len(images) == 1
    assert images[0]["image"].shape == (360, 640, 3)
    content = _plan_item_to_mh_content(load_mharmony(), images[0], encoding=None, metadata=metadata)
    with Image.open(io.BytesIO(base64.b64decode(content.media_ref.bytes_b64))) as admitted:
        assert admitted.size == ((448, 448) if fixed_resize else (640, 360))
        assert admitted.mode == "RGB"


@pytest.mark.parametrize("fixed_resize", [True, False], ids=["fixed-448", "patch-bounds-only"])
@pytest.mark.parametrize("profiled", [True, False], ids=["authenticated-profile", "unprofiled-control"])
def test_authenticated_render_profile_survives_retire_save_move_restore(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    sdk_plans: list[IsaacMharmonyContentPlan],
    fixed_resize: bool,
    profiled: bool,
) -> None:
    monkeypatch.delenv("QWEN35_VOCAB_PATH", raising=False)
    source = tmp_path / "source"
    _authenticated_source(source, fixed_resize=fixed_resize, profiled=profiled)
    original = _files(source)
    policy = PerceptronIsaacPolicy.from_pretrained(source, local_files_only=True, strict=True)
    assert policy._verified_tensor_storage is not None
    assert policy._verified_runtime_assets is not None
    retained = policy._verified_runtime_assets.root
    assert retained != source
    # Processor tensor state is required metadata, not a copied model payload.
    assert list(retained.glob("policy_*_step_*.safetensors"))
    assert not list((retained / "hf_model").rglob("*.safetensors"))
    assert policy.config.hf_model_path == str(source / "hf_model")
    assert policy.config.resolve_native_recipe_path() == str(retained / "policy_inference_recipe.json")
    policy._load_backbone()
    assert policy._isaac_model is not None
    # Keep P1's exact F32/signed-zero save sentinels while using real native weights.
    policy._isaac_model.float()
    with torch.no_grad():
        policy._isaac_model.lm_head.weight.fill_(1.2500001192092896)
        policy._isaac_model.model.vector_embedding[0].weight.fill_(-0.0)
    expected = {name: tensor.detach().clone() for name, tensor in policy._isaac_model.state_dict().items()}
    assert _files(source) == original, "Loading must not rewrite authenticated source bytes."
    retired = tmp_path / "retired-source"
    source.rename(retired)
    assert not source.exists()
    _assert_restored_render(policy, retained, sdk_plans, fixed_resize=fixed_resize, profiled=profiled)

    saved = tmp_path / "saved"
    policy.save_pretrained(saved, finalize_package=False)
    policy.finalize_pretrained_package(saved)
    _assert_native_inventory(saved, expected)
    for relative in (
        "policy_inference_recipe.json",
        "hf_model/tokenizer.json",
        "hf_model/config.json",
        "hf_model/source_mk1_model_import.json",
        "fast_processor/processing_action_tokenizer.py",
    ):
        assert (saved / relative).read_bytes() == original[relative]
    # Normal serialization makes optional None defaults explicit; compare domain
    # values while the source-byte checks below still prohibit source rewrites.
    expected_stats = isaac_stats_to_dict(load_isaac_stats(retired / "isaac_stats.json"))
    assert isaac_stats_to_dict(load_isaac_stats(saved / "isaac_stats.json")) == expected_stats
    moved = tmp_path / "moved"
    saved.rename(moved)
    before_reload = _files(moved)
    reloaded = PerceptronIsaacPolicy.from_pretrained(moved, local_files_only=True, strict=True)
    assert reloaded._verified_tensor_storage is not None
    assert reloaded.config.mk1_source_model_import_sha256 == policy.config.mk1_source_model_import_sha256
    reloaded._load_backbone()
    assert reloaded._isaac_model is not None
    for name, tensor in reloaded._isaac_model.state_dict().items():
        _assert_tensor_bits(tensor, expected[name].bfloat16(), name)
    assert reloaded._verified_runtime_assets is not None
    moved.rename(tmp_path / "retired-reload-source")
    _assert_restored_render(
        reloaded,
        reloaded._verified_runtime_assets.root,
        sdk_plans,
        fixed_resize=fixed_resize,
        profiled=profiled,
    )
    assert _files(tmp_path / "retired-reload-source") == before_reload
    assert _files(retired) == original


@pytest.mark.parametrize(
    "asset", ["policy_inference_recipe.json", "isaac_stats.json", "mk1_trained_package_manifest.json"]
)
def test_authenticated_render_source_rejects_byte_tamper(tmp_path: Path, asset: str) -> None:
    source = tmp_path / "source"
    _authenticated_source(source, fixed_resize=True, profiled=True)
    # Positive control: this exact fixture authenticates before a byte-only change.
    admitted = PerceptronIsaacPolicy.from_pretrained(source, local_files_only=True, strict=True)
    assert admitted._verified_tensor_storage is not None
    assert admitted._verified_runtime_assets is not None
    changed = source / asset
    changed.write_bytes(changed.read_bytes() + b"\n")
    expected = "manifest digest mismatch" if asset.startswith("mk1_") else f"digest mismatch for {asset}"
    with pytest.raises(RuntimeError, match=expected):
        PerceptronIsaacPolicy.from_pretrained(source, local_files_only=True, strict=True)
