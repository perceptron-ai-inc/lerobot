from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from lerobot.policies.perceptron_isaac.fast_processor import (
    fast_processor_tree_identity,
    load_fast_action_processor,
    materialize_pinned_fast_processor_snapshot,
    resolve_pinned_fast_processor_snapshot,
)
from lerobot.policies.perceptron_isaac.isaac_stats import (
    IsaacNormalizationStats,
    IsaacStatsBlock,
)
from lerobot.policies.perceptron_isaac.mharmony_adapter import load_mharmony_encoding_cache
from lerobot.policies.perceptron_isaac.mharmony_native import (
    GENESIS_TEXT_TYPE_ACTION,
    GENESIS_TEXT_TYPE_ACTION_C,
    GENESIS_TEXT_TYPE_TAG,
    IsaacMharmonyRenderMetadata,
    IsaacNativeMharmonyRenderer,
    build_isaac_mharmony_content_plan,
    build_isaac_training_proprio_contract,
)
from lerobot.policies.perceptron_isaac.tensor_stream import TextType


def _stats():
    bounds = IsaacStatsBlock(q01=np.zeros(2), q99=np.ones(2))
    return IsaacNormalizationStats(
        schema="flow_matching_stats_v1",
        action=bounds,
        proprio=bounds,
        action_horizon=2,
        target_fps=20.0,
    )


def _metadata():
    return IsaacMharmonyRenderMetadata(
        camera_order=["image"],
        camera_views=["primary"],
        image_size=[2, 2],
        n_obs_steps=1,
        action_horizon=2,
        action_dim=2,
        proprio_dim=2,
        target_fps=20.0,
    )


def test_joint_plan_places_flow_marker_immediately_before_fast_span():
    action = np.asarray([[0.25, -0.5], [0.75, 1.0]], dtype=np.float32)
    plan = build_isaac_mharmony_content_plan(
        observation_window=[
            {"images": {"image": np.zeros((2, 2, 3), dtype=np.uint8)}, "proprio": np.zeros(2)}
        ],
        prompt="pick up the block",
        metadata=_metadata(),
        stats=_stats(),
        anchor_timestamp_seconds=0.0,
        normalized_action_target=action,
        fast_action_tokens=[7, 8],
        action_is_pad=[False, True],
    )

    assert len(plan.assistant_content) == 2
    marker, fast = plan.assistant_content
    assert marker["tags"][GENESIS_TEXT_TYPE_TAG] == GENESIS_TEXT_TYPE_ACTION_C
    assert fast["tags"][GENESIS_TEXT_TYPE_TAG] == GENESIS_TEXT_TYPE_ACTION
    assert fast["tokens"] == [7, 8]
    np.testing.assert_allclose(marker["tags"]["action_target"], action)
    for key in ("action_index", "action_window", "action_indices", "action_is_pad"):
        assert marker["tags"][key] == fast["tags"][key]
    assert marker["tags"]["objective_form"] == "Flow"
    assert fast["tags"]["objective_form"] == "FAST"


def _footer_rewrite_for(terminal: bool | None) -> bool:
    """Capture the footer decision ``build_training`` hands to the renderer."""
    from lerobot.policies.perceptron_isaac.mharmony_native import IsaacNativeMharmonyRenderer

    captured = {}
    renderer = IsaacNativeMharmonyRenderer.__new__(IsaacNativeMharmonyRenderer)
    renderer.metadata = _metadata()
    renderer.stats = _stats()

    def fake_render(plan, **kwargs):
        captured.update(kwargs)
        return SimpleNamespace(shape=(1, 8))

    renderer._render_plan_to_tensor_stream = fake_render
    kwargs = {} if terminal is None else {"terminal": terminal}
    renderer.build_training(
        observation_window=[
            {"images": {"image": np.zeros((2, 2, 3), dtype=np.uint8)}, "proprio": np.zeros(2)}
        ],
        prompt="pick up the block",
        action_chunk=np.zeros((2, 2), dtype=np.float32),
        fast_processor=_StubFastProcessor(),
        anchor_timestamp_seconds=0.0,
        **kwargs,
    )
    return captured["rewrite_final_assistant_footer"]


class _StubFastProcessor:
    def __call__(self, actions, **kwargs):
        return [[1, 2]]


def test_fast_and_flow_targets_are_clipped_independently():
    """Genesis lowers FAST and Flow through separate transforms with independent caps.

    ``fast_tokenizer`` defaults to ``clip_normalized_max=1.0`` while
    ``attach_flow_matching_action_target`` takes the recipe's value (10.0 for the shipped
    LIBERO identity) — the ``fastcap1_flowcap10`` run naming. Sharing a single clipped array
    feeds the tokenizer an action distribution Genesis never trains it on: on
    lerobot/libero_spatial_image, |normalized| reaches 1.99, so the 1.0 cap binds on ~1.4% of
    values across ~9.4% of frames while the 10.0 cap never binds at all.
    """
    from lerobot.policies.perceptron_isaac.mharmony_native import IsaacNativeMharmonyRenderer

    seen = {}
    renderer = IsaacNativeMharmonyRenderer.__new__(IsaacNativeMharmonyRenderer)
    renderer.metadata = _metadata()
    renderer.stats = _stats()
    renderer._render_plan_to_tensor_stream = lambda plan, **kw: SimpleNamespace(shape=(1, 8))

    class _CapturingFastProcessor:
        def __call__(self, actions, **kwargs):
            seen["fast_input"] = np.asarray(actions).copy()
            return [[1, 2]]

    # q01=-1/q99=1 stats put raw ±3 well outside the FAST cap and inside the flow cap.
    renderer.build_training(
        observation_window=[
            {"images": {"image": np.zeros((2, 2, 3), dtype=np.uint8)}, "proprio": np.zeros(2)}
        ],
        prompt="pick up the block",
        action_chunk=np.asarray([[3.0, -3.0], [0.25, -0.5]], dtype=np.float32),
        fast_processor=_CapturingFastProcessor(),
        clip_normalized_max=10.0,
        fast_clip_normalized_max=1.0,
        anchor_timestamp_seconds=0.0,
    )

    fast_input = seen["fast_input"]
    assert np.abs(fast_input).max() <= 1.0 + 1e-6, "FAST input must honor its own (tighter) cap"
    # The flow target keeps the wider cap, so it retains magnitude the FAST input lost.
    assert np.abs(fast_input).max() < 3.0


def test_non_terminal_training_chunks_supervise_the_return_footer():
    """Genesis sets message_stop_token="return" for every chunk short of the terminal step.

    ``_action_completion_metadata`` (core/datasets/augment/trajectory.py) attaches the token
    whenever the chunk is non-terminal, and ``render_document_to_stream`` then swaps the
    rendered ``<|end|>`` footer for ``<|return|>``. Genesis's LeRobot-native training builder
    emits a window that never reaches the terminal step, so this is the default.
    """
    assert _footer_rewrite_for(None) is True
    assert _footer_rewrite_for(False) is True


def test_terminal_training_chunks_keep_the_end_footer():
    assert _footer_rewrite_for(True) is False


def test_joint_plan_exactly_matches_genesis_final_prompt_metadata():
    bounds = IsaacStatsBlock(q01=np.zeros(14), q99=np.ones(14))
    stats = IsaacNormalizationStats(
        schema="flow_matching_stats_v1",
        action=bounds,
        proprio=bounds,
        action_horizon=2,
        target_fps=30.0,
    )
    contract_hash = "a" * 64
    metadata = IsaacMharmonyRenderMetadata(
        camera_order=["top", "left", "right"],
        camera_views=["external_high", "left_wrist", "right_wrist"],
        image_size=[2, 2],
        n_obs_steps=1,
        action_horizon=2,
        action_dim=14,
        proprio_dim=14,
        target_fps=30.0,
        dataset_name="molmoact2_bimanualyam",
        robot_type="bi_yam",
        training_proprio_contract=build_isaac_training_proprio_contract(
            robot_type="bi_yam",
            dataset_name="molmoact2_bimanualyam",
            proprio_dim=14,
            contract_version=1,
            contract_hash=contract_hash,
        ),
    )
    plan = build_isaac_mharmony_content_plan(
        observation_window=[
            {
                "images": {camera: np.zeros((2, 2, 3), dtype=np.uint8) for camera in metadata.camera_order},
                "proprio": np.zeros(14, dtype=np.float32),
            }
        ],
        prompt="place the block",
        metadata=metadata,
        stats=stats,
        anchor_timestamp_seconds=0.0,
        normalized_action_target=np.zeros((2, 14), dtype=np.float32),
        fast_action_tokens=[7, 8],
        action_is_pad=[False, True],
    )

    images = [item for item in plan.user_content if item["kind"] == "image"]
    assert [item["media_id"] for item in images] == ["image_0", "image_1", "image_2"]
    vector = next(item for item in plan.user_content if item["kind"] == "vector")
    vector_metadata = vector["metadata"]
    assert vector_metadata["source_state_keys"] == ["state"] * 4
    assert vector_metadata["source_state_dims"] == [6, 1, 6, 1]
    assert vector_metadata["source_state_slices"] == [[0, 6], [6, 7], [7, 13], [13, 14]]
    assert vector_metadata["source_state_raw_layout"] == [{"key": "state", "dim": 14}]
    assert vector_metadata["proprio_collapsed"] is True
    assert vector_metadata["proprio_stats_key"] == "state"
    assert vector_metadata["proprio_original_dim"] == 14
    assert vector_metadata["proprio_target_dim"] == 14
    assert vector_metadata["proprio_padding_applied"] is False
    assert vector_metadata["proprio_truncated"] is False
    assert vector_metadata["policy_state_contract_version"] == 1
    assert vector_metadata["policy_state_contract_hash"] == contract_hash
    assert vector_metadata["policy_state_excluded"] == []

    flow, fast = plan.assistant_content
    shared_keys = {
        "token_group",
        GENESIS_TEXT_TYPE_TAG,
        "action_index",
        "trajectory_action",
        "objective_form",
        "obs_window",
        "action_window",
        "action_indices",
        "action_delta_indices",
        "action_is_pad",
        "action_horizon",
        "action_dim",
        "normalized",
    }
    assert set(flow["tags"]) == shared_keys | {"action_target"}
    assert set(fast["tags"]) == shared_keys | {"tokenizer"}
    assert not {
        "terminal",
        "message_stop_token",
        "observation_delta_indices",
        "observation_is_pad",
        "requested_action_delta_indices",
        "stats_key",
        "flow_matching_target",
    } & set(flow["tags"])


def test_real_cached_fast_and_mharmony_render_joint_training_stream(tmp_path):
    pytest.importorskip("mharmony")
    pytest.importorskip("genesis.data.mharmony.types")
    try:
        source = resolve_pinned_fast_processor_snapshot(local_files_only=True)
    except Exception as exc:  # pragma: no cover - cache is host-specific
        pytest.skip(f"pinned FAST processor is not cached: {exc}")
    artifact = materialize_pinned_fast_processor_snapshot(source, tmp_path / "fast_processor")
    identity_before = fast_processor_tree_identity(artifact.path)
    processor = load_fast_action_processor(
        artifact.path,
        expected_tree_sha256=artifact.tree_sha256,
    )
    assert fast_processor_tree_identity(artifact.path) == identity_before
    assert not (artifact.path / "__pycache__").exists()
    bounds = IsaacStatsBlock(q01=-np.ones(2), q99=np.ones(2))
    stats = IsaacNormalizationStats(
        schema="flow_matching_stats_v1",
        action=bounds,
        proprio=bounds,
        action_horizon=2,
        target_fps=20.0,
    )
    metadata = IsaacMharmonyRenderMetadata(
        camera_order=["image"],
        camera_views=["primary"],
        image_size=[256, 256],
        n_obs_steps=1,
        action_horizon=2,
        action_dim=2,
        proprio_dim=2,
        target_fps=20.0,
    )
    renderer = IsaacNativeMharmonyRenderer(metadata=metadata, stats=stats)

    stream = renderer.build_training(
        observation_window=[
            {
                "images": {"image": np.zeros((256, 256, 3), dtype=np.uint8)},
                "proprio": np.zeros(2, dtype=np.float32),
            }
        ],
        prompt="pick up the block",
        action_chunk=np.asarray([[0.25, -0.5], [0.75, 1.0]], dtype=np.float32),
        action_is_pad=[False, True],
        fast_processor=processor,
        max_num_patches=256,
        device="cpu",
        dtype=torch.float32,
        anchor_timestamp_seconds=0.0,
    )

    events = stream.streams[0].events
    flow = [event for event in events if event.type == TextType.action_c]
    fast = [event for event in events if event.type == TextType.action]
    assert stream.shape[0] == 1
    assert len(flow) == 1
    assert len(fast) == 1
    assert flow[0].tags["action_is_pad"] == [False, True]
    assert int(fast[0].data.min()) >= 248087

    encoding = load_mharmony_encoding_cache().get_harmony_encoding(
        metadata.mharmony_encoding_name,
        reserved_token_groups=metadata.mharmony_reserved_token_groups,
    )
    end_id = encoding.encode("<|end|>", allowed_special={"<|end|>"})[0]
    return_id = encoding.encode("<|return|>", allowed_special={"<|return|>"})[0]

    def final_control_id(rendered_stream):
        controls = [
            int(event.data.reshape(-1)[-1])
            for event in rendered_stream.streams[0].events
            if event.type == TextType.control and event.data.numel()
        ]
        assert controls
        return controls[-1]

    # Genesis supervises <|return|> for any chunk short of the trajectory's terminal step, and
    # its LeRobot-native builder never reaches that step, so the default render carries
    # <|return|>. Only an explicitly terminal chunk keeps <|end|>.
    assert final_control_id(stream) == return_id

    terminal_stream = renderer.build_training(
        observation_window=[
            {
                "images": {"image": np.zeros((256, 256, 3), dtype=np.uint8)},
                "proprio": np.zeros(2, dtype=np.float32),
            }
        ],
        prompt="pick up the block",
        action_chunk=np.asarray([[0.25, -0.5], [0.75, 1.0]], dtype=np.float32),
        action_is_pad=[False, True],
        fast_processor=processor,
        max_num_patches=256,
        device="cpu",
        dtype=torch.float32,
        anchor_timestamp_seconds=0.0,
        terminal=True,
    )
    assert final_control_id(terminal_stream) == end_id

    inference_stream = renderer.build(
        observation_window=[
            {
                "images": {"image": np.zeros((256, 256, 3), dtype=np.uint8)},
                "proprio": np.zeros(2, dtype=np.float32),
            }
        ],
        prompt="pick up the block",
        max_num_patches=256,
        device="cpu",
        dtype=torch.float32,
        anchor_timestamp_seconds=0.0,
    )
    assert final_control_id(inference_stream) == return_id
