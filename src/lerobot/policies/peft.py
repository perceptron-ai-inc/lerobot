# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Shared PEFT checkpoint loading for policy frontends."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from huggingface_hub.errors import HFValidationError
from huggingface_hub.utils import validate_repo_id

from lerobot.utils.hub import hub_snapshot_revision, resolve_hub_snapshot
from lerobot.utils.import_utils import _peft_available, require_package

if TYPE_CHECKING or _peft_available:
    from peft import PeftConfig, PeftModel
    from peft.utils.other import ModulesToSaveWrapper
    from peft.utils.save_and_load import load_peft_weights, set_peft_model_state_dict
else:
    PeftConfig = None
    PeftModel = None


def _resolve_peft_base_snapshot(
    adapter_root: str | Path,
    base_reference: str | Path,
    *,
    revision: str | None,
    adapter_is_local: bool,
) -> tuple[Path, bool]:
    """Resolve one adapter base without allowing the process CWD to shadow it."""
    adapter_root = Path(adapter_root).resolve()
    original_reference = str(base_reference)
    base_path = Path(original_reference)
    if not adapter_is_local and (base_path.is_absolute() or ".." in base_path.parts):
        raise ValueError("A remote PEFT adapter cannot reference a filesystem path outside its snapshot.")
    if base_path.is_absolute():
        if not base_path.is_dir():
            raise ValueError(f"PEFT adapter local base checkpoint does not exist: {base_path}.")
        return base_path.resolve(), True

    local_base = (adapter_root / base_path).resolve()
    if local_base.is_dir():
        if not adapter_is_local and not local_base.is_relative_to(adapter_root):
            raise ValueError("A remote PEFT adapter cannot reference a base outside its snapshot.")
        return local_base, True
    if original_reference == "base_model":
        raise ValueError("PEFT adapter embedded base_model directory is missing or invalid.")
    try:
        validate_repo_id(original_reference)
    except HFValidationError as exc:
        raise ValueError(
            "A PEFT adapter base must be a contained checkpoint directory or valid Hub repo ID."
        ) from exc
    return (
        resolve_hub_snapshot(original_reference, revision=revision, force_remote=True),
        False,
    )


def load_peft_policy(
    policy_class: type,
    policy_config: Any,
    adapter_path: str | Path,
    *,
    adapter_revision: str | None = None,
    is_trainable: bool = False,
    policy_kwargs: dict[str, Any] | None = None,
):
    """Load a policy base and adapter while keeping their revisions independent."""
    require_package("peft", extra="peft")
    adapter_reference = str(adapter_path)
    adapter_was_local = Path(adapter_reference).is_dir()
    adapter_root = resolve_hub_snapshot(adapter_reference, revision=adapter_revision)
    adapter_reference = str(adapter_root)
    peft_config = PeftConfig.from_pretrained(adapter_reference)
    base_reference = peft_config.base_model_name_or_path
    if not base_reference:
        raise ValueError("PEFT adapter config does not identify a base policy checkpoint")

    base_revision = getattr(peft_config, "revision", None)
    original_base_reference = str(base_reference)
    base_snapshot, embed_base_on_save = _resolve_peft_base_snapshot(
        adapter_root,
        original_base_reference,
        revision=base_revision,
        adapter_is_local=adapter_was_local,
    )
    if not embed_base_on_save:
        base_revision = hub_snapshot_revision(base_snapshot)
        peft_config.revision = base_revision
    base_reference = str(base_snapshot)

    policy_config._embed_base_on_save = embed_base_on_save
    policy_config._peft_base_reference = original_base_reference
    policy_config._peft_base_revision = base_revision

    resolve_config_paths = getattr(policy_class, "resolve_checkpoint_config_paths", None)
    if resolve_config_paths is not None:
        resolve_config_paths(policy_config, adapter_root)

    base_kwargs = dict(policy_kwargs or {})
    base_kwargs.update(
        pretrained_name_or_path=base_reference,
        config=policy_config,
        revision=None,
    )
    policy = policy_class.from_pretrained(**base_kwargs)
    retain_processor_assets = getattr(policy, "retain_pretrained_processor_assets", None)
    if retain_processor_assets is not None:
        retain_processor_assets(adapter_root)

    adapter_kwargs: dict[str, Any] = {"config": peft_config}
    if is_trainable:
        adapter_kwargs["is_trainable"] = True
    adapted_policy = PeftModel.from_pretrained(policy, adapter_reference, **adapter_kwargs)
    _restore_peft_saved_module_precision(adapted_policy, adapter_root)
    return adapted_policy


def _restore_peft_saved_module_precision(model: PeftModel, adapter_root: Path) -> None:
    """Recover saved module precision before optimizers capture adapter-owned parameters."""
    saved_modules = [
        (name, module) for name, module in model.named_modules() if isinstance(module, ModulesToSaveWrapper)
    ]
    if not saved_modules:
        return
    # Ordinary PEFT loading copies saved heads into the base dtype. Meta/assign
    # loading is not a substitute: PEFT then rounds LoRA tensors to the base dtype.
    weights = load_peft_weights(str(adapter_root), device="cpu", local_files_only=True)
    promoted = False
    for name, module in saved_modules:
        tensors = module.state_dict(keep_vars=True)
        for saved_key, loaded_key in module.adapter_state_dict_load_map("default").items():
            saved = weights[f"{name}.{saved_key}"]
            target = tensors[loaded_key]
            if torch.promote_types(target.dtype, saved.dtype) != target.dtype:
                # Restore only this saved copy, including intentional FP16/FP64
                # storage. Never cast original_module or the frozen backbone.
                target.data = target.data.to(dtype=saved.dtype)
                promoted = True
    if promoted:
        set_peft_model_state_dict(model, weights, adapter_name="default", low_cpu_mem_usage=False)
