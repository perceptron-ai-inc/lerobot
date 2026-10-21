from __future__ import annotations

import sys
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any, TypedDict

import numpy as np
import pytest
import torch

from lerobot.policies.perceptron_isaac import mharmony_adapter, mharmony_native
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
from lerobot.policies.perceptron_isaac.mharmony_adapter import (
    load_mharmony_encoding,
    rendered_stream_to_local_tensor_stream,
)
from lerobot.policies.perceptron_isaac.mharmony_native import (
    GENESIS_TEXT_TYPE_ACTION,
    GENESIS_TEXT_TYPE_ACTION_C,
    GENESIS_TEXT_TYPE_TAG,
    IsaacMharmonyContentPlan,
    IsaacMharmonyRenderMetadata,
    IsaacNativeMharmonyRenderer,
    build_isaac_mharmony_content_plan,
    build_isaac_training_proprio_contract,
)
from lerobot.policies.perceptron_isaac.tensor_stream import (
    Event,
    ModalityType,
    Stream,
    TensorStream,
    TextType,
    VectorType,
    VisionType,
    create_stream,
    group_streams,
)
from lerobot.policies.perceptron_isaac.tensor_stream_utils import (
    build_action_context_mask,
    compute_mrope_pos_tensor,
    compute_tensor_stream_metadata,
    interleave_partial_mask,
    set_data,
)


class _RenderInputs(TypedDict):
    observation_window: list[dict[str, Any]]
    prompt: str
    max_num_patches: int
    device: str
    dtype: torch.dtype
    anchor_timestamp_seconds: float


class _ActionInputs(TypedDict):
    action_chunk: np.ndarray
    action_is_pad: list[bool]
    fast_processor: Any


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


def _footer_rewrite_for(terminal: bool | None, monkeypatch: pytest.MonkeyPatch) -> bool:
    """Capture the footer decision ``build_training`` hands to the renderer."""
    from lerobot.policies.perceptron_isaac.mharmony_native import IsaacNativeMharmonyRenderer

    captured = {}
    renderer = IsaacNativeMharmonyRenderer.__new__(IsaacNativeMharmonyRenderer)
    renderer.metadata = _metadata()
    renderer.stats = _stats()

    def fake_render(plan, **kwargs):
        captured.update(kwargs)
        return SimpleNamespace(shape=(1, 8))

    monkeypatch.setattr(renderer, "_render_plan_to_tensor_stream", fake_render)
    kwargs: dict[str, Any] = {} if terminal is None else {"terminal": terminal}
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


def test_non_terminal_training_chunks_supervise_the_return_footer(monkeypatch: pytest.MonkeyPatch) -> None:
    """Genesis sets message_stop_token="return" for every chunk short of the terminal step.

    ``_action_completion_metadata`` (core/datasets/augment/trajectory.py) attaches the token
    whenever the chunk is non-terminal, and ``render_document_to_stream`` then swaps the
    rendered ``<|end|>`` footer for ``<|return|>``. Genesis's LeRobot-native training builder
    emits a window that never reaches the terminal step, so this is the default.
    """
    assert _footer_rewrite_for(None, monkeypatch) is True
    assert _footer_rewrite_for(False, monkeypatch) is True


def test_terminal_training_chunks_keep_the_end_footer(monkeypatch: pytest.MonkeyPatch) -> None:
    assert _footer_rewrite_for(True, monkeypatch) is False


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


def test_public_rendered_stream_lowering_preserves_isaac_contract():
    image = np.arange(4 * 2 * 3, dtype=np.float32)
    rendered = {
        "stream": {
            "events": [
                {
                    "time": [0.0, 0.0],
                    "modality": "role_marker",
                    "role": "user",
                    "tags": {},
                    "data": {"Tokens": [101, 102]},
                },
                {
                    "time": [0.0, 0.5],
                    "modality": "text",
                    "role": "user",
                    "tags": {},
                    "data": {"Tokens": [11, 12, 13]},
                },
                {
                    "time": [0.5, 0.5],
                    "modality": "text",
                    "role": "developer",
                    "tags": {GENESIS_TEXT_TYPE_TAG: "timestamp"},
                    "data": {"Tokens": [21]},
                },
                {
                    "time": [0.5, 0.5],
                    "modality": "vector",
                    "role": "tool",
                    "tags": {"name": "state"},
                    "data": {"Vector": {"data": [0.25, -0.5, 0.75], "shape": [3]}},
                },
                {
                    "time": [0.5, 0.5],
                    "modality": "image_frame",
                    "role": "assistant",
                    "tags": {"media_ref_id": "image_0"},
                    "data": {"Image": {"data": image, "shape": [4, 2, 3]}},
                },
                {
                    "time": [0.5, 0.5],
                    "modality": "text",
                    "role": "assistant",
                    "channel": "final",
                    "tags": {
                        GENESIS_TEXT_TYPE_TAG: GENESIS_TEXT_TYPE_ACTION_C,
                        "action_target": [[0.25, -0.5]],
                        "action_is_pad": [False],
                    },
                    "data": {"Tokens": [248_087]},
                },
                {
                    "time": [0.5, 0.5],
                    "modality": "action",
                    "role": "assistant",
                    "channel": "final",
                    "tags": {"token_group": "fast_action"},
                    "data": {"Tokens": [248_088, 248_089]},
                },
                {
                    "time": [0.5, 0.5],
                    "modality": "special_token",
                    "role": "assistant",
                    "channel": "final",
                    "tags": {},
                    "data": {"Tokens": [248_000]},
                },
            ]
        },
        "config": {"image": {"pixel_shuffle_scale": 2}},
    }

    tensor_stream = rendered_stream_to_local_tensor_stream(
        rendered,
        device="cpu",
        dtype=torch.float32,
    )
    stream = tensor_stream.streams[0]

    assert stream.priority == [
        TextType.padding,
        TextType.timestamp,
        TextType.text,
        TextType.action,
        TextType.action_c,
        TextType.control,
        VectorType.vector,
        VisionType.I,
        VisionType.P,
    ]
    assert [event.type for event in stream.events] == [
        TextType.control,
        TextType.text,
        TextType.timestamp,
        VectorType.vector,
        VisionType.I,
        TextType.action_c,
        TextType.action,
        TextType.control,
    ]
    assert [event.role for event in stream.events] == [
        "user",
        "user",
        "system",
        "tool_output",
        "agent",
        "agent",
        "agent",
        "agent",
    ]

    text_event = stream.events[1]
    assert text_event.data.shape == (3, 1)
    assert text_event.dims_virtual == [3, 1]
    assert text_event.dims_real == [3, 1]
    assert text_event.idx_range == (0, 3)

    vector_event = stream.events[3]
    assert vector_event.data.shape == (1, 3)
    assert vector_event.dims_virtual == [1]
    assert vector_event.dims_real == [1, 3]
    assert vector_event.idx_range == (0, 1)
    assert vector_event.tags == {"name": "state"}

    image_event = stream.events[4]
    assert image_event.data.shape == (8, 3)
    assert image_event.dims_real == [1, 4, 2]
    assert image_event.dims_virtual == [1, 2, 1]
    assert image_event.idx_range == (0, 2)
    assert image_event.tags == {"media_ref_id": "image_0"}

    flow_event = stream.events[5]
    assert flow_event.tags["channel"] == "final"
    assert flow_event.tags["action_target"] == [[0.25, -0.5]]
    assert flow_event.tags["action_is_pad"] == [False]
    assert all(event.data.dtype == torch.long for event in stream.events if isinstance(event.type, TextType))
    assert vector_event.data.dtype == torch.float32
    assert image_event.data.dtype == torch.float32


def test_public_rendered_stream_lowering_rejects_invalid_pixel_shuffle_grid():
    rendered = {
        "stream": {
            "events": [
                {
                    "time": [0.0, 0.0],
                    "modality": "image_frame",
                    "role": "user",
                    "tags": {},
                    "data": {"Image": {"data": np.zeros(4 * 2 * 3), "shape": [4, 2, 3]}},
                }
            ]
        },
        "config": {"image": {"pixel_shuffle_scale": 3}},
    }

    with pytest.raises(ValueError, match="not divisible"):
        rendered_stream_to_local_tensor_stream(rendered, device="cpu", dtype=torch.float32)


def test_stream_construction_keeps_render_order_and_owns_event_lists() -> None:
    late = Event.from_text_tokens(torch.tensor([11]), time=(3.0, 3.0))
    early = Event.from_text_tokens(torch.tensor([12]), time=(0.0, 0.0), type=TextType.control)
    last = Event.from_text_tokens(torch.tensor([13]), time=(1.0, 1.0))
    events = [late, early, last]
    priority: list[ModalityType] = [TextType.control, TextType.text]
    stream = create_stream(events, priority)
    events.clear()
    assert stream.events == [late, early, last]
    grouped = group_streams(stream, lambda event: event.type)
    assert list(grouped) == [TextType.text, TextType.control]
    assert grouped[TextType.text].events == [late, last]
    assert grouped[TextType.control].events == [early]
    assert grouped[TextType.text].priority == priority
    grouped[TextType.text].events.clear()
    assert stream.events == [late, early, last]
    mapped = stream.map(lambda _event: {})
    assert all(left is right for left, right in zip(mapped.events, stream.events, strict=True))
    flat = TensorStream([stream]).flat_stream()
    flat.events.clear()
    assert stream.events == [late, early, last]
    short = TensorStream([create_stream([early], priority)])
    padded = IsaacNativeMharmonyRenderer.collate([TensorStream([stream]), short])
    assert padded.shape == (2, 3)
    assert padded.streams[0].events == [late, early, last]
    assert padded.streams[1].events[0] is early
    assert padded.streams[1].events[-1].type == TextType.padding
    torch.testing.assert_close(padded.streams[1].events[-1].data, torch.zeros(2, 1, dtype=torch.long))
    assert short.streams[0].events == [early]


@pytest.mark.parametrize("dtype", [None, torch.float64])
def test_stream_transfer_preserves_tokens_padding_float_values_and_autograd(
    dtype: torch.dtype | None,
) -> None:
    text_events = [
        Event.from_text_tokens(torch.tensor([11, 12], dtype=torch.int32), time=(0.0, 0.0), type=kind)
        for kind in (TextType.text, TextType.control, TextType.timestamp, TextType.padding, TextType.action)
    ]
    vector = torch.tensor([[-0.0, 1.5]], requires_grad=True)
    image = torch.tensor([[2.0, -3.0]], requires_grad=True)
    float_events = [
        Event(data=data, time=(0.0, 0.0), type=kind, dims_virtual=[1], dims_real=[1], idx_range=(0, 1))
        for data, kind in [(vector, VectorType.vector), (image, VisionType.I)]
    ]
    stream = TensorStream([Stream(text_events + float_events, [])])
    assert all(event.data.shape == (2, 1) for event in text_events)
    assert stream.to("cpu", dtype=dtype, non_blocking=False) is stream
    for event in text_events:
        torch.testing.assert_close(event.data, torch.tensor([[11], [12]], dtype=torch.long))
        assert event.idx_range == (0, 2)
    for event, expected in zip(float_events, ([[-0.0, 1.5]], [[2.0, -3.0]]), strict=True):
        torch.testing.assert_close(event.data, torch.tensor(expected, dtype=dtype or torch.float32))
        assert event.data.requires_grad
    assert torch.signbit(float_events[0].data[0, 0])
    (float_events[0].data.sum() + float_events[1].data.sum()).backward()
    torch.testing.assert_close(vector.grad, torch.ones_like(vector))
    torch.testing.assert_close(image.grad, torch.ones_like(image))


def test_stream_metadata_preserves_partial_tokens_roles_context_and_mrope() -> None:
    events = [
        Event.from_text_tokens(
            torch.tensor(tokens),
            time=(0.0, 0.0),
            type=kind,
            role=role,
            tags={"source_stream_name": source},
        )
        for tokens, kind, role, source in [
            ([10, 11, 12], TextType.text, "user", "a"),
            ([99], TextType.control, "user", "a"),
            ([0], TextType.action_c, "assistant", "a"),
            ([7], TextType.action, "assistant", "a"),
            ([6], TextType.text, "user", "b"),
            ([0], TextType.action_c, "assistant", "b"),
        ]
    ]
    events[0].slice_tokens(1, 3)
    stream = TensorStream([Stream(events, list(TextType))])
    metadata = compute_tensor_stream_metadata(stream)

    torch.testing.assert_close(metadata.modality_mask, torch.tensor([[8, 8, 12, 15, 14, 8, 15]]))
    torch.testing.assert_close(metadata.role_mask, torch.tensor([[1, 1, 1, 0, 0, 1, 0]]))
    torch.testing.assert_close(
        interleave_partial_mask(stream, TextType.text, metadata),
        torch.tensor([False, True, True, True]),
    )
    tokens, valid = set_data(stream, (TextType.text, TextType.control), roles=("user",), metadata=metadata)
    torch.testing.assert_close(tokens, torch.tensor([[11, 12, 99, 0, 0, 6, 0]]))
    torch.testing.assert_close(valid, torch.tensor([[True, True, True, False, False, True, False]]))
    context, fallback_count = build_action_context_mask(
        stream,
        metadata,
        action_batch_indices=[0, 0, 0],
        action_start_indices=[3, 6, 6],
        action_provenance_keys=[("a", None, None), ("b", None, None), None],
    )
    torch.testing.assert_close(
        context,
        torch.tensor(
            [
                [True, True, True, False, False, False, False],
                [False, False, False, False, False, True, False],
                [True, True, True, False, False, True, False],
            ]
        ),
    )
    assert fallback_count == 1
    torch.testing.assert_close(
        compute_mrope_pos_tensor(stream),
        torch.tensor([[[1, 0, 0], [2, 0, 0], [3, 3, 3], [4, 4, 4], [5, 5, 5], [6, 6, 6], [7, 7, 7]]]),
    )
    assert metadata.to("cpu") is metadata
    moved = metadata.to("meta")
    assert moved.modality_mask.device.type == "meta"
    assert moved.role_mask.device.type == "meta"
    assert moved.partial_mask(TextType.text).device.type == "meta"


@pytest.fixture
def encoding_load_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[list[tuple[str, list[dict[str, Any]] | None]]]:
    calls: list[tuple[str, list[dict[str, Any]] | None]] = []

    class _FakeMharmony:
        @staticmethod
        def load_harmony_encoding(name: str, *, reserved_token_groups: list[dict[str, Any]] | None) -> object:
            calls.append((name, reserved_token_groups))
            return object()

    mharmony_adapter._load_mharmony_encoding_cached.cache_clear()
    monkeypatch.delenv("QWEN35_VOCAB_PATH", raising=False)
    monkeypatch.setattr(mharmony_adapter, "load_mharmony", lambda: _FakeMharmony)
    try:
        yield calls
    finally:
        mharmony_adapter._load_mharmony_encoding_cached.cache_clear()


def test_public_mharmony_encoding_loader_caches_equivalent_group_configs(
    encoding_load_calls: list[tuple[str, list[dict[str, Any]] | None]],
) -> None:
    first = load_mharmony_encoding(
        "QWEN35_HARMONY",
        reserved_token_groups=[{"size": 2048, "tokenizer": "physical-intelligence/fast"}],
    )
    second = load_mharmony_encoding(
        "QWEN35_HARMONY",
        reserved_token_groups=[{"tokenizer": "physical-intelligence/fast", "size": 2048}],
    )
    assert first is second
    assert encoding_load_calls == [
        (
            "QWEN35_HARMONY",
            [{"size": 2048, "tokenizer": "physical-intelligence/fast"}],
        )
    ]


def test_qwen35_encoding_cache_is_scoped_to_resolved_checkpoint_vocab(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    encoding_load_calls: list[tuple[str, list[dict[str, Any]] | None]],
) -> None:
    first_vocab = tmp_path / "first" / "vocab.json"
    second_vocab = tmp_path / "second" / "vocab.json"
    monkeypatch.setenv("QWEN35_VOCAB_PATH", str(first_vocab.parent / ".." / "first/vocab.json"))
    first = load_mharmony_encoding("QWEN35_HARMONY")
    assert load_mharmony_encoding("QWEN35_HARMONY") is first
    monkeypatch.setenv("QWEN35_VOCAB_PATH", str(second_vocab))
    second = load_mharmony_encoding("QWEN35_HARMONY")
    assert second is not first
    assert encoding_load_calls == [("QWEN35_HARMONY", None), ("QWEN35_HARMONY", None)]


def test_public_mharmony_version_must_match_the_pinned_distribution(monkeypatch):
    mharmony_adapter._validate_installed_mharmony_version.cache_clear()
    monkeypatch.setattr(mharmony_adapter.importlib.metadata, "version", lambda _name: "0.2.0")
    try:
        with pytest.raises(RuntimeError, match=r"requires mharmony==0\.1\.0, but 0\.2\.0 is installed"):
            mharmony_adapter.load_mharmony()
    finally:
        mharmony_adapter._validate_installed_mharmony_version.cache_clear()


def test_installed_mharmony_exposes_the_supported_public_contract():
    pytest.importorskip("mharmony")

    mharmony_adapter.assert_mharmony_available()


def test_qwen35_processor_factory_uses_public_root_export(monkeypatch):
    calls = []
    processor = object()

    def factory(**kwargs):
        calls.append(kwargs)
        return processor

    monkeypatch.setattr(
        mharmony_adapter,
        "load_mharmony",
        lambda: SimpleNamespace(create_qwen35_image_processor=factory),
    )

    assert mharmony_adapter.create_qwen35_image_processor(patch_size=16) is processor
    assert calls == [{"patch_size": 16}]


def test_pinned_renderer_serializes_without_mutating_the_final_assistant_message() -> None:
    mharmony = pytest.importorskip("mharmony")
    encoding = load_mharmony_encoding("QWEN35_HARMONY")
    conversation = mharmony.Conversation(
        messages=[
            mharmony.Message(
                author=mharmony.Author(role=mharmony.Role("user")),
                content=[mharmony.TextContent(text="pick up the block")],
            ),
            mharmony.Message(
                author=mharmony.Author(role=mharmony.Role("assistant")),
                content=[mharmony.TokensContent(tokens=[0])],
                channel="final",
            ),
        ]
    )
    before = conversation.to_json()
    rendered = encoding.render_conversation_multimodal_with_processors(conversation)
    assert conversation.to_json() == before
    footer = rendered["stream"]["events"][-1]
    assert footer["role"] == "assistant"
    assert footer["channel"] == "final"
    assert footer["modality"] == "special_token"
    assert footer["data"]["Tokens"] == encoding.encode("<|end|>", allowed_special={"<|end|>"})


def test_native_renderer_constructs_the_public_conversation_directly(monkeypatch):
    class _Model:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    class _Role:
        def __init__(self, value):
            self.value = value

    class _TextContent(_Model):
        pass

    class _TokensContent(_Model):
        pass

    class _ImageContent(_Model):
        pass

    class _VectorContent(_Model):
        pass

    fake_mharmony = SimpleNamespace(
        Role=_Role,
        Author=_Model,
        Message=_Model,
        Conversation=_Model,
        TextContent=_TextContent,
        TokensContent=_TokensContent,
        MediaRef=_Model,
        ImageContent=_ImageContent,
        VectorContent=_VectorContent,
    )
    seen = {}

    class _Encoding:
        @staticmethod
        def encode_timestamps_qwen(timestamps, *, precision):
            assert timestamps == [1.25]
            assert precision == 2
            return [[42]]

        @staticmethod
        def render_conversation_multimodal_with_processors(
            conversation,
            *,
            preprocess_config,
            media_processors,
        ):
            seen["conversation"] = conversation
            seen["preprocess_config"] = preprocess_config
            seen["media_processors"] = media_processors
            return {
                "stream": {
                    "events": [
                        {
                            "time": [0.0, 0.0],
                            "modality": "text",
                            "role": "assistant",
                            "channel": "final",
                            "tags": {GENESIS_TEXT_TYPE_TAG: GENESIS_TEXT_TYPE_ACTION_C},
                            "data": {"Tokens": [248_087]},
                        }
                    ]
                },
                "config": {"image": {"pixel_shuffle_scale": 2}},
            }

    processor = object()
    monkeypatch.setattr(mharmony_native, "load_mharmony", lambda: fake_mharmony)
    monkeypatch.setattr(mharmony_native, "load_mharmony_encoding", lambda *args, **kwargs: _Encoding())
    monkeypatch.setattr(mharmony_native, "create_qwen35_image_processor", lambda **kwargs: processor)

    plan = IsaacMharmonyContentPlan(
        preamble="prompt",
        anchor_timestamp_seconds=1.25,
        observation_timestamps_seconds=[1.25],
        user_content=[
            {"kind": "text", "text": "prompt"},
            {
                "kind": "timestamp",
                "normalized_seconds": 1.25,
                "precision": 2,
                "tags": {GENESIS_TEXT_TYPE_TAG: "timestamp"},
            },
            {
                "kind": "image",
                "media_id": "image_0",
                "image": np.zeros((2, 2, 3), dtype=np.uint8),
                "metadata": {"camera": "primary"},
            },
            {
                "kind": "vector",
                "values": [0.25, -0.5],
                "shape": [2],
                "dtype": "float32",
                "name": "state",
            },
        ],
        assistant_content=[
            {
                "kind": "tokens",
                "tokens": [0],
                "tags": {GENESIS_TEXT_TYPE_TAG: GENESIS_TEXT_TYPE_ACTION_C},
            }
        ],
    )
    renderer = IsaacNativeMharmonyRenderer(metadata=_metadata(), stats=_stats())
    stream = renderer._render_plan_to_tensor_stream(
        plan,
        device="cpu",
        dtype=torch.float32,
        patch_size=16,
        max_num_patches=4,
        min_num_patches=None,
        pixel_shuffle_scale=2,
        temporal_patch_size=1,
        rewrite_final_assistant_footer=False,
    )

    conversation = seen["conversation"]
    assert isinstance(conversation, _Model)
    assert [message.author.role.value for message in conversation.messages] == ["user", "assistant"]
    assert [type(content) for content in conversation.messages[0].content] == [
        _TextContent,
        _TokensContent,
        _ImageContent,
        _VectorContent,
    ]
    assert isinstance(conversation.messages[1].content[0], _TokensContent)
    assert conversation.messages[1].channel == "final"
    assert seen["media_processors"] == {"image": processor}
    assert seen["preprocess_config"]["pixel_shuffle_scale"] == 2
    assert stream.streams[0].events[0].type == TextType.action_c
    assert stream.streams[0].events[0].role == "agent"


def test_cached_renderer_does_not_skip_fast_resolver_defects(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("mharmony")

    def broken_resolver(*, local_files_only: bool) -> Path:
        assert local_files_only
        raise ValueError("FAST resolver defect")

    monkeypatch.setattr(
        sys.modules[__name__],
        "resolve_pinned_fast_processor_snapshot",
        broken_resolver,
    )
    with pytest.raises(ValueError, match="FAST resolver defect"):
        try:
            test_real_cached_fast_and_mharmony_render_joint_training_stream(tmp_path, monkeypatch)
        except pytest.skip.Exception as exc:
            pytest.fail(f"A resolver defect must fail, not skip: {exc}")


def test_real_cached_fast_and_mharmony_render_joint_training_stream(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("mharmony")
    monkeypatch.delenv("QWEN35_VOCAB_PATH", raising=False)
    try:
        source = resolve_pinned_fast_processor_snapshot(local_files_only=True)
    except FileNotFoundError as exc:  # pragma: no cover - cache is host-specific
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

    common_inputs: _RenderInputs = {
        "observation_window": [
            {
                "images": {"image": np.zeros((256, 256, 3), dtype=np.uint8)},
                "proprio": np.zeros(2, dtype=np.float32),
            }
        ],
        "prompt": "pick up the block",
        "max_num_patches": 256,
        "device": "cpu",
        "dtype": torch.float32,
        "anchor_timestamp_seconds": 0.0,
    }
    action_inputs: _ActionInputs = {
        "action_chunk": np.asarray([[0.25, -0.5], [0.75, 1.0]], dtype=np.float32),
        "action_is_pad": [False, True],
        "fast_processor": processor,
    }
    stream = renderer.build_training(**common_inputs, **action_inputs)

    events = stream.streams[0].events
    flow = [event for event in events if event.type == TextType.action_c]
    fast = [event for event in events if event.type == TextType.action]
    assert stream.shape[0] == 1
    assert len(flow) == 1
    assert len(fast) == 1
    assert flow[0].tags["action_is_pad"] == [False, True]
    assert int(fast[0].data.min()) >= 248087

    encoding = load_mharmony_encoding(
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

    terminal_stream = renderer.build_training(**common_inputs, **action_inputs, terminal=True)
    assert final_control_id(terminal_stream) == end_id

    inference_stream = renderer.build(**common_inputs)
    assert final_control_id(inference_stream) == return_id
