# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

import lerobot.policies.factory as policy_factory
import lerobot.policies.peft as peft_loader


def test_load_peft_policy_keeps_adapter_and_base_revisions_separate(monkeypatch):
    adapter_snapshot = "/cache/adapter-snapshot"
    base_snapshot = "/cache/base-snapshot"
    config = SimpleNamespace()
    base_policy = torch.nn.Linear(1, 1)
    policy_from_pretrained = MagicMock(return_value=base_policy)
    policy_class = SimpleNamespace(from_pretrained=policy_from_pretrained)
    peft_config = SimpleNamespace(base_model_name_or_path="user/base-policy", revision="base-sha")
    config_loader = MagicMock(return_value=peft_config)
    adapted_policy = torch.nn.Linear(1, 1)
    model_loader = MagicMock(return_value=adapted_policy)
    require_package = MagicMock()
    monkeypatch.setattr(peft_loader, "require_package", require_package)
    monkeypatch.setattr(peft_loader, "PeftConfig", SimpleNamespace(from_pretrained=config_loader))
    monkeypatch.setattr(peft_loader, "PeftModel", SimpleNamespace(from_pretrained=model_loader))
    resolve_snapshot = MagicMock(side_effect=[Path(adapter_snapshot), Path(base_snapshot)])
    monkeypatch.setattr(peft_loader, "resolve_hub_snapshot", resolve_snapshot)
    base_commit = "b" * 40
    monkeypatch.setattr(peft_loader, "hub_snapshot_revision", lambda _snapshot: base_commit)

    policy = peft_loader.load_peft_policy(
        policy_class,
        config,
        "user/adapter",
        adapter_revision="adapter-sha",
        is_trainable=True,
        policy_kwargs={"dataset_stats": {"action": {}}, "dataset_meta": "meta"},
    )

    assert policy is adapted_policy
    require_package.assert_called_once_with("peft", extra="peft")
    assert resolve_snapshot.call_args_list == [
        (("user/adapter",), {"revision": "adapter-sha"}),
        (("user/base-policy",), {"revision": "base-sha", "force_remote": True}),
    ]
    config_loader.assert_called_once_with(adapter_snapshot)
    policy_from_pretrained.assert_called_once_with(
        pretrained_name_or_path=base_snapshot,
        config=config,
        revision=None,
        dataset_stats={"action": {}},
        dataset_meta="meta",
    )
    model_loader.assert_called_once_with(
        base_policy,
        adapter_snapshot,
        config=peft_config,
        is_trainable=True,
    )
    assert config._embed_base_on_save is False
    assert config._peft_base_reference == "user/base-policy"
    assert config._peft_base_revision == base_commit
    assert peft_config.revision == base_commit


def test_load_peft_policy_resolves_local_sibling_base_and_adapter_sidecars(monkeypatch, tmp_path):
    adapter = tmp_path / "run" / "pretrained_model"
    base = tmp_path / "step-65000-lerobot"
    adapter.mkdir(parents=True)
    base.mkdir()
    config = SimpleNamespace()
    resolve_paths = MagicMock()
    base_policy = object()
    policy_from_pretrained = MagicMock(return_value=base_policy)
    policy_class = SimpleNamespace(
        from_pretrained=policy_from_pretrained,
        resolve_checkpoint_config_paths=resolve_paths,
    )
    peft_config = SimpleNamespace(
        base_model_name_or_path="../../step-65000-lerobot",
        revision="base-sha",
    )
    adapted = torch.nn.Linear(1, 1)
    model_loader = MagicMock(return_value=adapted)
    monkeypatch.setattr(peft_loader, "require_package", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        peft_loader,
        "PeftConfig",
        SimpleNamespace(from_pretrained=MagicMock(return_value=peft_config)),
    )
    monkeypatch.setattr(peft_loader, "PeftModel", SimpleNamespace(from_pretrained=model_loader))
    monkeypatch.setattr(peft_loader, "resolve_hub_snapshot", lambda reference, **kwargs: Path(reference))

    result = peft_loader.load_peft_policy(policy_class, config, adapter)

    assert result is adapted
    resolve_paths.assert_called_once_with(config, adapter)
    policy_from_pretrained.assert_called_once_with(
        pretrained_name_or_path=str(base.resolve()),
        config=config,
        revision=None,
    )
    model_loader.assert_called_once_with(base_policy, str(adapter), config=peft_config)
    assert config._embed_base_on_save is True


def test_load_local_peft_policy_accepts_single_component_hub_base_and_pins_revision(monkeypatch, tmp_path):
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    commit = "a" * 40
    base_snapshot = tmp_path / "hub" / "models--gpt2" / "snapshots" / commit
    base_snapshot.mkdir(parents=True)
    config = SimpleNamespace()
    base_policy = object()
    policy_from_pretrained = MagicMock(return_value=base_policy)
    policy_class = SimpleNamespace(from_pretrained=policy_from_pretrained)
    peft_config = SimpleNamespace(base_model_name_or_path="gpt2", revision=None)
    adapted = torch.nn.Linear(1, 1)
    model_loader = MagicMock(return_value=adapted)
    resolve_snapshot = MagicMock(side_effect=[adapter, base_snapshot])
    monkeypatch.setattr(peft_loader, "require_package", lambda *args, **kwargs: None)
    monkeypatch.setattr(peft_loader, "resolve_hub_snapshot", resolve_snapshot)
    monkeypatch.setattr(
        peft_loader,
        "PeftConfig",
        SimpleNamespace(from_pretrained=MagicMock(return_value=peft_config)),
    )
    monkeypatch.setattr(peft_loader, "PeftModel", SimpleNamespace(from_pretrained=model_loader))

    result = peft_loader.load_peft_policy(policy_class, config, adapter)

    assert result is adapted
    assert resolve_snapshot.call_args_list == [
        ((str(adapter),), {"revision": None}),
        (("gpt2",), {"revision": None, "force_remote": True}),
    ]
    policy_from_pretrained.assert_called_once_with(
        pretrained_name_or_path=str(base_snapshot),
        config=config,
        revision=None,
    )
    model_loader.assert_called_once_with(base_policy, str(adapter), config=peft_config)
    assert config._embed_base_on_save is False
    assert config._peft_base_reference == "gpt2"
    assert config._peft_base_revision == commit
    assert peft_config.revision == commit


@pytest.mark.parametrize("base_reference", ["./missing", "../missing", "../../missing"])
def test_load_local_peft_policy_rejects_missing_relative_base_without_hub_lookup(
    monkeypatch, tmp_path, base_reference
):
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    config = SimpleNamespace()
    resolve_snapshot = MagicMock(return_value=adapter)
    monkeypatch.setattr(peft_loader, "require_package", lambda *args, **kwargs: None)
    monkeypatch.setattr(peft_loader, "resolve_hub_snapshot", resolve_snapshot)
    monkeypatch.setattr(
        peft_loader,
        "PeftConfig",
        SimpleNamespace(
            from_pretrained=MagicMock(
                return_value=SimpleNamespace(base_model_name_or_path=base_reference, revision=None)
            )
        ),
    )

    with pytest.raises(ValueError, match="contained checkpoint directory or valid Hub repo ID"):
        peft_loader.load_peft_policy(SimpleNamespace(), config, adapter)

    resolve_snapshot.assert_called_once_with(str(adapter), revision=None)


def test_load_local_peft_policy_never_resolves_missing_embedded_base_from_hub(monkeypatch, tmp_path):
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    config = SimpleNamespace()
    resolve_snapshot = MagicMock(return_value=adapter)
    monkeypatch.setattr(peft_loader, "require_package", lambda *args, **kwargs: None)
    monkeypatch.setattr(peft_loader, "resolve_hub_snapshot", resolve_snapshot)
    monkeypatch.setattr(
        peft_loader,
        "PeftConfig",
        SimpleNamespace(
            from_pretrained=MagicMock(
                return_value=SimpleNamespace(base_model_name_or_path="base_model", revision=None)
            )
        ),
    )

    with pytest.raises(ValueError, match="embedded base_model directory is missing"):
        peft_loader.load_peft_policy(SimpleNamespace(), config, adapter)

    resolve_snapshot.assert_called_once_with(str(adapter), revision=None)


@pytest.mark.parametrize("base_reference", ["/host/private/base", "../../outside"])
def test_remote_peft_adapter_rejects_host_filesystem_base(monkeypatch, tmp_path, base_reference):
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    config = SimpleNamespace()
    monkeypatch.setattr(peft_loader, "require_package", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        peft_loader,
        "resolve_hub_snapshot",
        MagicMock(return_value=snapshot),
    )
    monkeypatch.setattr(
        peft_loader,
        "PeftConfig",
        SimpleNamespace(
            from_pretrained=MagicMock(
                return_value=SimpleNamespace(base_model_name_or_path=base_reference, revision=None)
            )
        ),
    )

    with pytest.raises(ValueError, match="remote PEFT adapter cannot reference"):
        peft_loader.load_peft_policy(SimpleNamespace(), config, "user/adapter")


def test_load_peft_policy_resolves_remote_adapter_sidecars_before_base(monkeypatch, tmp_path):
    snapshot = tmp_path / "snapshots" / "adapter-sha"
    snapshot.mkdir(parents=True)
    (snapshot / "isaac_stats.json").write_text("{}")
    config = SimpleNamespace()
    resolution_order: list[tuple[Path, str | None]] = []

    def resolve_paths(policy_config, root):
        root = Path(root)
        resolution_order.append((root, getattr(policy_config, "native_stats_path", None)))
        candidate = root / str(policy_config.native_stats_path)
        if candidate.exists():
            policy_config.native_stats_path = str(candidate)

    def load_base(**kwargs):
        resolution_order.append((Path("base-loader"), kwargs["config"].native_stats_path))
        return "base"

    policy_class = SimpleNamespace(
        from_pretrained=MagicMock(side_effect=load_base),
        resolve_checkpoint_config_paths=resolve_paths,
    )
    peft_config = SimpleNamespace(base_model_name_or_path="user/base", revision="base-sha")
    monkeypatch.setattr(peft_loader, "require_package", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        peft_loader,
        "resolve_hub_snapshot",
        MagicMock(return_value=snapshot),
    )
    monkeypatch.setattr(
        peft_loader,
        "PeftConfig",
        SimpleNamespace(from_pretrained=MagicMock(return_value=peft_config)),
    )
    adapted = torch.nn.Linear(1, 1)
    monkeypatch.setattr(
        peft_loader,
        "PeftModel",
        SimpleNamespace(from_pretrained=MagicMock(return_value=adapted)),
    )
    monkeypatch.setattr(peft_loader, "hub_snapshot_revision", lambda _snapshot: "b" * 40)
    config.native_stats_path = "isaac_stats.json"

    result = peft_loader.load_peft_policy(
        policy_class,
        config,
        "user/adapter",
        adapter_revision="adapter-sha",
    )

    assert result is adapted
    assert resolution_order == [
        (snapshot, "isaac_stats.json"),
        (Path("base-loader"), str(snapshot / "isaac_stats.json")),
    ]


def test_load_peft_policy_retains_adapter_processor_assets_before_loading_adapter(monkeypatch, tmp_path):
    adapter = tmp_path / "adapter"
    base = tmp_path / "base"
    adapter.mkdir()
    base.mkdir()
    config = SimpleNamespace()
    call_order: list[tuple[str, object]] = []
    base_policy = SimpleNamespace(
        retain_pretrained_processor_assets=lambda root: call_order.append(("retain", root))
    )
    policy_class = SimpleNamespace(from_pretrained=MagicMock(return_value=base_policy))
    peft_config = SimpleNamespace(base_model_name_or_path=str(base), revision=None)
    monkeypatch.setattr(peft_loader, "require_package", lambda *args, **kwargs: None)
    monkeypatch.setattr(peft_loader, "resolve_hub_snapshot", lambda reference, **kwargs: Path(reference))
    monkeypatch.setattr(
        peft_loader,
        "PeftConfig",
        SimpleNamespace(from_pretrained=MagicMock(return_value=peft_config)),
    )
    adapted = torch.nn.Linear(1, 1)
    monkeypatch.setattr(
        peft_loader,
        "PeftModel",
        SimpleNamespace(
            from_pretrained=lambda policy, adapter_path, **kwargs: (
                call_order.append(("load", Path(adapter_path))),
                adapted,
            )[1]
        ),
    )

    result = peft_loader.load_peft_policy(policy_class, config, adapter)

    assert result is adapted
    assert call_order == [("retain", adapter), ("load", adapter)]


def test_make_policy_delegates_peft_loading(monkeypatch):
    cfg = SimpleNamespace(
        type="mock",
        device="cpu",
        pretrained_path="user/adapter",
        pretrained_revision="adapter-sha",
        use_peft=True,
        input_features={},
        output_features={},
    )
    dataset_meta = SimpleNamespace(features={}, stats={})

    policy_class = SimpleNamespace()
    monkeypatch.setattr(policy_factory, "get_policy_class", lambda _: policy_class)
    monkeypatch.setattr(policy_factory, "dataset_to_policy_features", lambda _: {})
    monkeypatch.setattr(policy_factory, "validate_visual_features_consistency", lambda *args: None)
    adapted_policy = torch.nn.Linear(1, 1)
    load_peft = MagicMock(return_value=adapted_policy)
    monkeypatch.setattr(policy_factory, "load_peft_policy", load_peft)

    policy = policy_factory.make_policy(cfg, ds_meta=dataset_meta)

    assert policy is adapted_policy
    load_peft.assert_called_once_with(
        policy_class,
        cfg,
        "user/adapter",
        adapter_revision="adapter-sha",
        is_trainable=True,
        policy_kwargs={"config": cfg, "dataset_stats": dataset_meta.stats, "dataset_meta": dataset_meta},
    )
