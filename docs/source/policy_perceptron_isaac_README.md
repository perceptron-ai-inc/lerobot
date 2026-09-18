# Perceptron ISAAC

`policy.type=perceptron_isaac` is a first-class LeRobot v0.6 policy for ISAAC
Qwen3.5-VLA flow-matching checkpoints. Inference, finetuning, LIBERO eval, and
local robot rollout use the normal LeRobot policy and processor APIs. There is
no Genesis inference server and no ISAAC branch in LeRobot's policy factory,
training loop, eval loop, or rollout dispatcher.

MolmoAct2 is outside this integration.

For the maintained installation and fine-tuning tutorial, see
[Perceptron Isaac](./perceptron_isaac.mdx).

## Native surface

- `PerceptronIsaacConfig`, `PerceptronIsaacPolicy`, and convention-discovered
  pre/postprocessor factories.
- Local Qwen3.5-VLA backbone and action expert inference.
- Native NTP, FAST, and flow-matching training loss. Training fixtures can
  inject the sampled flow timesteps, noise, and loss mask directly.
- Policy-owned online observation history and episode-relative frame clock.
  Processors are stateless and never synthesize or retain model time.
- Authenticated one-time checkpoint import, including streamed shard
  conversion, one-time RMSNorm correction, inline processor state, and a
  hash-pinned FAST processor artifact.
- `lerobot-eval` for LIBERO and synchronous `lerobot-rollout` for SO100,
  SO101, and bimanual YAM.
- A strict Genesis MDS v2 to `LeRobotDataset` converter for preserved YAM data.
- Self-authenticating inference/training parity artifacts with exact fixture
  checks and tolerance-based numerical/gradient checks.

## External package boundary

The `perceptron_isaac` extra declares `mharmony[qwen35]==0.1.0`.
`[tool.uv.sources]` pins mharmony to Git revision
`9a57efbf8ada2d73641fc40081f4e4fa6ff2ab92`; locked `uv sync` is the supported
repository installation path. Availability of the same version as a public
PyPI distribution is a separate packaging concern: plain pip does not honor
uv's Git source mapping. Model, FAST and dataset artifacts are separate from
package installation and must satisfy their pinned identity contracts.

A sibling checkout, custom `PYTHONPATH`, or direct import from
`genesis.data.mharmony` is a release blocker. The package boundary and pinned
versions are recorded in
[`docs/PERCEPTRON_ISAAC_PROVENANCE.md`](../PERCEPTRON_ISAAC_PROVENANCE.md).

## Install

From this repository:

```bash
uv sync --locked --extra perceptron_isaac
uv run python -c "from lerobot.policies.perceptron_isaac import PerceptronIsaacPolicy"
```

The base extra installs the declared model and mharmony dependencies and supports
CPU development. Image-rendering tests additionally require the pinned local FAST
processor artifacts; installing dependencies does not supply model weights.
Production NVIDIA inference will additionally require the CUDA kernels:

```bash
uv sync --locked --extra perceptron_isaac_cuda
```

The CUDA extra currently targets Linux; it includes `perceptron_isaac` and adds
`flash-linear-attention==0.4.1` plus `causal-conv1d==1.6.0`.

MK1 Fast production inference has a dedicated, narrower qualified runtime:
PyTorch `2.10.0+cu128`, Transformers `5.5.4`, the `torch_sdpa_v1` attention
backend, and an NVIDIA H100. The project-default PyTorch 2.11 environment is
useful for CPU development, neutral-debug packages, and explicitly reduced test
fixtures, but it is not the v12 action-parity runtime. A full `trained_policy`
checkpoint loaded onto CUDA fails before tensor streaming or model allocation
unless PyTorch reports the exact 2.10.0 release and CUDA 12.8. Provision that
ABI in the deployment environment in addition to installing
`perceptron_isaac_cuda`; the CUDA extra does not override the repository-wide
PyTorch selection.

Locked uv installation resolves the declared Git source. Developer checkouts and
untracked local wheels are not supported installation steps.

For bimanual YAM support on Linux:

```bash
uv sync --locked --extra perceptron_isaac --extra yam --group yam-hardware
```

The `yam` extra installs the published Portal dependency; the development-only
`yam-hardware` group pins the reviewed i2rt revision because PyPI metadata cannot
carry that direct Git dependency. `uv.lock` also constrains the
scikit-build-core version required by i2rt's pinned ruckig build.

## Import a checkpoint once

Raw converted HF exports are not deployable. A deployable directory is created
only after the three Genesis contract JSONs have been checked against the
converter-emitted `policy_state_identity.safetensors` sidecar or the DCP-owned
identity tensor.

The two are not equally strong, and the recorded provenance says which was used:

- **`dcp_authenticated`** — checked against the DCP checkpoint's own identity
  tensor, which the HF export cannot forge. Note that supplying
  `--dcp-checkpoint` unpickles that checkpoint's metadata before anything is
  compared, so point it only at your own training output.
- **`hf_identity_authenticated`** — sidecar only. The sidecar ships in the same
  directory as the JSONs it binds and its payload is exactly
  `build_dcp_identity()` of those JSONs, so anyone who can write the export can
  regenerate a passing sidecar. This proves internal consistency, not provenance.

Model weights are not covered by either check: the import binds the contract
JSONs and detects mid-import mutation of the shards, but nothing compares the
weights against an external source-of-truth digest.

```bash
uv run lerobot-isaac-import \
  --hf-export /path/to/raw-hf-export \
  --deployment-adapter /path/to/isaac_deployment_adapter.json \
  --output /path/to/isaac-lerobot-package \
  --policy-state-dataset cloud/isaac_yam \
  --normalization-scope yam \
  --allow-fast-remote-code
```

Pass `--dcp-checkpoint /path/to/dcp-checkpoint` as an optional independent
cross-check. If both identity sources are present, the importer requires both
to match the contract JSONs exactly.

The importer validates the complete nested Genesis schemas and the separately
reviewed deployment adapter, which is bound to the exact three contract-file
hashes. It then streams model shards, applies the Qwen3.5 RMSNorm correction
once, copies immutable model/tokenizer assets, records provenance, and saves
portable LeRobot processor pipelines. Imported packages disable camera fallback
and carry strict state, action, camera, FPS, safety, joint-frame, and
normalization-profile metadata.

The checkpoint is the source of truth for policy semantics. Published packages
must include their immutable source revision, contract hashes, selected
normalization scope, and reproducible validation results in the model card.

## Config field recipe

`PerceptronIsaacConfig()` constructed from its dataclass defaults does not describe any
shipped checkpoint. The defaults are a neutral starting point for a new fine-tune;
28 Perceptron-specific fields default to `None`, and several scalar defaults differ from
the values the released LIBERO package carries (`chunk_size` 30 vs 50, `n_action_steps`
30 vs 8, `num_flow_samples` 4 vs 1, `flow_seed_base` `None` vs 20260826, `num_settle_steps`
0 vs 40, `target_fps` `None` vs 20.0). `_validate_known_deployment_contract` /
`_require_platform_values` (`configuration_perceptron_isaac.py:657-770`) only *validate* a
declared platform config; they never supply one. Every value below therefore comes from the
checkpoint package, not from the policy code.

Reference package for the "raw export" column: the Isaac-0.5 LIBERO export
(`<root>/lerobot_policy/config.json`, `model_type: isaac_0_5` at `<root>/config.json`).
Keys that a config.json omits simply fall back to the dataclass default at load.

### Where a raw Isaac-0.5 export keeps its values

| Asset | Path | Read by |
| --- | --- | --- |
| Policy config | `<root>/lerobot_policy/config.json` | `PerceptronIsaacConfig` |
| Normalization stats | `<root>/isaac_stats.json` | `resolve_native_stats_path` (`configuration_perceptron_isaac.py:518-534`) |
| Inference recipe | `<root>/policy_inference_recipe.json` | `resolve_native_recipe_path` (`:536-548`), consumed by `load_native_render_metadata` (`mharmony_native.py:510-526`) |
| Pinned FAST processor | `<root>/fast_processor_pinned/` | `fast_processor_path` (`../fast_processor_pinned`), rebased by `_resolve_checkpoint_local_paths` (`modeling_perceptron_isaac.py:2383-2430`) |
| Deployment adapter | `<root>/isaac_deployment_adapter.json` | digest-verified against `deployment_adapter_sha256` (`modeling_perceptron_isaac.py:1699-1712`) |
| Backbone weights | `<root>` (`hf_model_path: ".."`) | `_load_backbone` |

The deployment adapter is the reviewed source of the serving knobs (it carries
`flow_seed_base`, `num_flow_samples`, `num_inference_steps`, `n_action_steps`,
`num_settle_steps`, `settle_gripper`, `robot_type`, `control_mode`, `policy_state_dataset`,
camera and normalization metadata). At load time its **digest** is checked; the effective
values are the ones already written into `config.json`. A config that disagrees with the
adapter is not silently corrected.

### Observation and action geometry

| Field | What it is | Raw export value / source | Auto-resolved? | If absent |
| --- | --- | --- | --- | --- |
| `chunk_size` | Flow chunk length the model predicts | `50`, written in config.json | No | Default `30`; wrong horizon for this package |
| `n_action_steps` | Actions served per chunk before a refill | `8`, config.json (adapter agrees) | No | Default `30`; executes a stale 30-step chunk |
| `action_dim` | Real action width | `7`, config.json | No | Default `7` |
| `max_action_dim` | Padded model action width | `64`, config.json | No | Default `64` |
| `max_action_horizon` | Upper bound on `chunk_size` | `64`, config.json | No | Default `64` |
| `proprio_dim` | Real proprio width | `8`, config.json | No | Default `8` |
| `max_state_dim` | Declared max state width | `128`, config.json | No | Default `128` |
| `vector_max_states` | Proprio padding width into the vector encoder | `128`, config.json | No | Default `128` |

### Inference, flow sampling and settling

| Field | What it is | Raw export value / source | Auto-resolved? | If absent |
| --- | --- | --- | --- | --- |
| `inference_backend` | Serving backend; only `native_mharmony` exists | `native_mharmony` | No | Default is the only accepted value |
| `action_expert_type` | Expert head (`molmoact`/`dit` alias the same head) | `molmoact` | No | Default `molmoact` |
| `num_inference_steps` | Euler steps in the flow ODE | `10`, config.json + adapter | No | Default `10` |
| `num_flow_samples` | Flow samples averaged per chunk | `1`, config.json + adapter | No | Default `4`; averages 4 samples this package never used |
| `flow_seed_base` | Base seed for flow noise. **The only knob that varies flow noise on the policy path.** `predict_action_chunk` runs the sampler inside `torch.random.fork_rng(...)` and reseeds the device RNG with `flow_seed_base + _flow_seed_index` (`modeling_perceptron_isaac.py:1447-1464`), so `torch.manual_seed` and any ambient global seeding have **no effect** on the sampled chunk. Successive calls advance `_flow_seed_index`; `reset()` returns it to index 0 (`:764`), so the first chunk after every reset repeats bitwise. | `20260826`, config.json + adapter | No | Default `None` → falls back to `ISAAC_FLOW_SEED_BASE`, else no reseed at all and flow noise follows ambient global RNG |
| `clip_normalized_max` | Flow-target clip (checkpoint-owned, from the recipe's `flow_action` wire) | `10.0` | No | Default `10.0` |
| `fast_clip_normalized_max` | FAST-target clip; must match the base recipe, not the library default | `1.0` | No | Default `1.0`, correct only for the LIBERO/Genesis lineage |
| `clip_action_pose` | Clip OSC pose dims to [-1,1] before the env | `true`, pinned by the LIBERO contract (`:732`) | No | Default `true` |
| `num_settle_steps` | Idle actions emitted at episode start | `40`, config.json + adapter. Deliberately **not** pinned by the LIBERO contract (`:734-738`) | No | Default `0`: no settle window, and calls that this package expects to be canned no-ops become live forwards |
| `settle_gripper` | Gripper value held during settling | `-1.0`, config.json + adapter | No | Default `-1.0` |
| `rtc_prefix_length` | Real-time-chunking prefix; unsupported | `0`; `__post_init__` rejects non-zero (`:410-411`) | No | Default `0` is the only accepted value |
| `native_require_processor_stream` | Require processor-owned mharmony packing | `true` | No | Default `true` |
| `mharmony_version` | Exact runtime covered by the render-parity contract | `0.1.0` | Normalized by `normalize_mharmony_version_marker` (`:305`) | Default is the supported marker |

### Vision, cameras and rendering

| Field | What it is | Raw export value / source | Auto-resolved? | If absent |
| --- | --- | --- | --- | --- |
| `image_size` | Render geometry | `[256,256]`, pinned by the LIBERO contract (`:730`) | No | Default `(256,256)` |
| `image_preprocessing` | `stretch` or `letterbox` before the renderer | `stretch` | No | Default `stretch` (backward-compatible) |
| `camera_order` | Ordered camera slots | `["image","wrist_image"]`, pinned by the LIBERO contract (`:731`) | No | Default matches LIBERO |
| `serving_camera_roles` | Physical role → checkpoint slot map | `null` (LIBERO needs none) | No | `None` means no remap; only SO100-style packages set it |
| `allow_image_key_fallback` | Compatibility image-key fallback | `false` (authenticated import sets false) | No | Default `true` is permissive; an imported package should carry `false` |
| `render_processor_enabled` | Processor-side render step toggle | `false` | No | Default `false` |
| `render_patch_size` | mharmony patch size | `16` | No | Default `16` |
| `render_pixel_shuffle_scale` | Pixel-shuffle factor | `2` | No | Default `2` |
| `render_temporal_patch_size` | Temporal patch size | `2` | No | Default `2` |
| `render_max_num_patches` | Max patch budget | `576`, config.json | No | `None` → metadata/default budget |
| `render_min_num_patches` | Min patch budget | `null` | No | `None` → metadata/default budget |

Reserved-token group layout is **not** a config field: it is read from
`<root>/policy_inference_recipe.json` when `native_render_metadata_path` is unset
(`mharmony_native.py:517-522`).

### Normalization

| Field | What it is | Raw export value / source | Auto-resolved? | If absent |
| --- | --- | --- | --- | --- |
| `normalize_gripper` | q01/q99-normalize gripper dims | `true` (Genesis convention) | No | Default `true` |
| `gripper_binary_to_signed` | Remap {0,1} gripper to [-1,+1] at serving | `false`, pinned by the LIBERO contract (`:733`) | No | Default `false` |
| `normalization_mapping` | LeRobot normalizer modes; all IDENTITY because the ISAAC processors own normalization | all `IDENTITY` | No | Default is already all-IDENTITY |
| `normalization_profile_id` | Label of the stats profile | `null` in this export | Written only by `_save_pretrained` when the pipeline recorded `_export_native_stats` (`:583-594`) | `None` passes `require_closed_loop_safe_isaac_profile` as an unlabelled profile; no stats are inferred from it |
| `normalization_profile_scope` | Scope of that profile (e.g. per-suite) | `null` in this export | Same export path as above | Same as above |
| `normalization_validation_status` | Validation status of that profile | `null` in this export | Same export path as above | Same as above |

### Joint-frame convention adapter

| Field | What it is | Raw export value / source | Auto-resolved? | If absent |
| --- | --- | --- | --- | --- |
| `joint_signs` | Per-joint sign for the old/current calibration remap | `null`, pinned to `null` by the LIBERO contract (`:741`) | No | `None` means no remap; must be set together with `joint_offsets` (`:388-391`) |
| `joint_offsets` | Per-joint offset for the same remap | `null` (`:742`) | No | Same pairing rule |
| `training_action_frame_contract_version` | Training-side joint-frame contract version | `2` | No | Only `2` is accepted (`:399-403`) |

### Checkpoint identity and paths

| Field | What it is | Raw export value / source | Auto-resolved? | If absent |
| --- | --- | --- | --- | --- |
| `hf_model_path` | Backbone HF directory | `".."` (the export root), rebased to an absolute path by `_resolve_checkpoint_local_paths` (`modeling_perceptron_isaac.py:2406-2430`) | Yes, relative→absolute against the package root | `None` → no backbone to load |
| `artifact_kind` | `trained_policy` or `neutral_debug` | `trained_policy` | No | Default `trained_policy`, which requires positive `trained_steps` (`:343-344`) |
| `trained_steps` | Declared training steps | `100000` | No | Default `1`; `neutral_debug` must declare `0` (`:341-342`) |
| `apply_offset_norm` | Apply the Qwen3.5 RMSNorm correction at load | `false` (already corrected in this export) | No | Default `true` would double-apply the correction on an already-corrected export |
| `config_toml_path` | Deprecated bridge field | `null`; `__post_init__` force-clears it (`:408`) | Cleared unconditionally | Never used by the native path |
| `stats_path` | Deprecated alias for `native_stats_path` | `null`; migrated into `native_stats_path` when set (`:406-407`) | Yes, migrated | Falls through to `native_stats_path` resolution |
| `per_suite_stats_dir` | Deprecated bridge field | `null`; force-cleared (`:409`) | Cleared unconditionally | Never used |
| `native_stats_path` | q01/q99 normalization stats | `null` in config.json; **resolved to `<root>/isaac_stats.json`** by `resolve_native_stats_path` via `checkpoint_asset_roots` (`:502-534`) | **Yes**, since the raw-export load path landed | Neither declared nor discoverable → `load_native_isaac_stats` raises (`mharmony_native.py:545-552`) |
| `native_render_metadata_path` | Importer-written render metadata sidecar | `null`; this export ships no such file and does not need one | No, but its absence is handled: metadata is built with `IsaacMharmonyRenderMetadata.from_config` plus reserved-token groups from the export recipe (`mharmony_native.py:510-526`) | Falls back to config-derived metadata + recipe groups; only an importer-written package sets it |
| `suite_stats_path` | Per-suite training stats table | `null` (single-scope inference) | Copied checkpoint-local at save (`:611-655`) | Must be set together with `suite_by_task_index_path`; unset means single-scope normalization |
| `suite_by_task_index_path` | Task-index → suite routing table | `null` | Same save-time export | Same pairing rule |
| `fast_processor_path` | Pinned local FAST processor tree | `"../fast_processor_pinned"`, rebased against the export root (`modeling_perceptron_isaac.py:2412-2430`) | Relative→absolute only; **not discovered** when unset | `None` is accepted for inference, but ISAAC training refuses to build FAST targets without it (`processor_perceptron_isaac.py:1073-1079`) |
| `fast_processor_tree_sha256` | Digest of that tree | `127eb029…f33a8a` | No | Must be set/unset together with `fast_processor_path` (`:418-425`) |
| `deployment_adapter_sha256` | Digest of the reviewed deployment adapter | `dd91c009…776d64` | No | A portable Isaac-0.5 package without it is rejected (`modeling_perceptron_isaac.py:1699-1712`) |
| `qwen35_trained_package_manifest_sha256` | Digest of the Qwen3.5 trained-package manifest | key absent from this export → `None` | No | `None` skips the manifest cross-check; importer-written packages carry it |
| `mk1_model_import_sha256` | MK1 import digest | key absent → `None` | No | MK1-only; unused for a Qwen3.5 export |
| `mk1_source_model_import_sha256` | MK1 source import digest | key absent → `None` | No | MK1-only |
| `mk1_trained_package_manifest_sha256` | MK1 trained-package manifest digest | key absent → `None` | No | MK1-only; a trained MK1 package requires the adapter digest (`:1906-1912`) |

### Serving metadata and deployment contract

| Field | What it is | Raw export value / source | Auto-resolved? | If absent |
| --- | --- | --- | --- | --- |
| `include_scene_description` | Scene text in the prompt | `false`; `true` is rejected (`:380-383`) | No | Default `false` |
| `normalize_task_text` | Normalize task strings | `true`, pinned by the LIBERO contract (`:739`) | No | Default `true` |
| `action_conditioning` | Feed executed actions back as history | `false` | No | Default `false`; when `true`, rollout must call `record_executed_action` |
| `action_conditioning_role` | Chat role used for that history | `user` (only accepted value, `:384-387`) | No | Default `user` |
| `mistake_conditioning` | Mistake-conditioning prompt variant | `false` | No | Default `false` |
| `dataset_name` | Render dataset identity | `libero`, pinned by the LIBERO contract (`:726`) | No | Default `libero` happens to match; other platforms must set it |
| `policy_state_dataset` | Policy-state table selector | `libero`, config.json + adapter | No | `None` means no policy-state selection |
| `robot_type` | Platform selector for the strict contract | `libero` | No | Default `generic`, which `_validate_known_deployment_contract` rejects under strict mode (`:746-750`) |
| `control_mode` | `ee` or `joint` | `ee`, pinned by the LIBERO contract (`:727`) | No | Default `joint` is wrong for LIBERO |
| `target_fps` | Checkpoint-owned control rate | `20.0` in config.json; pinned by the LIBERO contract (`:743`). The adapter leaves it `null` | Render metadata inherits the stats FPS when the field is `None` (`mharmony_native.py:523-524`) | `None` with no stats FPS means no declared rate; LIBERO eval adopts this value, so a wrong one desynchronizes sim and policy |
| `action_feature_names` | Ordered action feature layout | `null` (LIBERO is not a strict-hardware platform) | No | Required only when `strict_hardware_feature_contract` is set (`:471-476`) |
| `state_feature_names` | Ordered state feature layout | `null` | No | Same rule |
| `strict_hardware_feature_contract` | Enforce hardware joint order | `false` | No | Default `false`; robot rollout packages set `true` |
| `strict_environment_feature_contract` | Enforce the platform contract above | `true` in this export | No | Default `false` skips `_validate_known_deployment_contract` entirely, so a mis-specified LIBERO config loads silently |

### Native training fields

These are read by the training path only; serving ignores them.

| Field | What it is | Raw export value / source | Auto-resolved? | If absent |
| --- | --- | --- | --- | --- |
| `objective` | Loss objective selector | `mixed` | No | Default `mixed` |
| `loss_plan` | Enabled loss terms | `text_ntp_fast_flow_action` | No | Default is the full plan |
| `train_expert_only` | Freeze VLM + vision, train the expert only | `false` | No | Default `false` = dense training |
| `flow_matching_detach_vlm_activations` | Knowledge-insulation switch | `false` | No | Default `false` |
| `exclude_non_agent_roles` | Drop non-agent chat roles from the NTP loss | `true` | No | Default `true` |
| `train_samples_per_chunk` | Flow samples drawn per chunk in training | `8` | No | Default `8`, must be ≥1 (`:412-413`) |
| `train_clip_normalized_actions` | Clip normalized action targets | `true` | No | Default `true` |
| `train_skip_outlier_threshold` | Skip samples beyond this normalized magnitude | `20.0` | No | Default `20.0`, must be ≥0 (`:414-415`) |
| `train_max_sequence_length` | Token budget per training sample | `4096` | No | Default `4096`, must be ≥1 (`:416-417`) |
| `flow_rtc_max_delay_steps` | RTC delay sampling; unimplemented | `0`; non-zero rejected (`:426-430`) | No | `0` is the only accepted value |
| `flow_rtc_delay_sampling` | RTC delay distribution | `uniform` (inert while the above is 0) | No | Default `uniform` |
| `flow_dual_timestep_ratio` | Dual-timestep sampling; unimplemented | `0.0`; non-zero rejected (`:431-435`) | No | `0.0` is the only accepted value |
| `flow_mask_padded_action_rows` | Mask padded terminal action rows | `true` | No | Default `true` |
| `softmax_auxiliary_loss_scale` | Auxiliary softmax loss weight | `0.0001` | No | Default `0.0001`, must be ≥0 (`:436-437`) |
| `freeze_input_embeddings` | Freeze input embeddings while training | `true` | No | Default `true` |
| `dtype` | Compute dtype under autocast | `bfloat16`; anything else is rejected (`:438-442`) | No | `bfloat16` is the only accepted value |
| `train_storage_fp32` | Keep trainable params and Adam moments in FP32 | `true` | No | Default `true` |

### Optimizer and scheduler presets

| Field | What it is | Raw export value / source | Auto-resolved? | If absent |
| --- | --- | --- | --- | --- |
| `optimizer_lr` | Base learning rate | `1e-5` | No | Preset default; a fine-tune recipe should set it deliberately |
| `optimizer_vit_lr` | Vision-tower learning rate | `5e-6` | No | Preset default |
| `optimizer_action_expert_lr` | Action-expert learning rate | `5e-5` | No | Preset default |
| `optimizer_betas` | AdamW betas | `(0.9, 0.95)` | No | Preset default |
| `optimizer_eps` | AdamW epsilon | `1e-6` | No | Preset default |
| `optimizer_weight_decay` | AdamW weight decay | `0.0` | No | Preset default |
| `optimizer_grad_clip_norm` | Gradient clipping norm | `1.0` | No | Preset default |
| `optimizer_warmup_steps` | Warmup steps for the cosine schedule | `200` | No | Preset default |
| `max_train_steps` | Horizon used by the cosine decay schedule | `30000` | No | Preset default; set it to the real run length or the decay curve is wrong |

### Inherited `PreTrainedConfig` fields that still matter here

`n_obs_steps` (`3` in this export; only `1` or `3` are accepted, `:352-356`), `device`
(`cuda`), `use_amp` (`false`), `use_peft` (`false`), `input_features` / `output_features`
(empty in the export; populated from dataset metadata at training time),
`pretrained_path` (set by the loader, and what `checkpoint_asset_roots` uses to find the
sidecars), plus the hub fields `repo_id`, `private`, `tags`, `license`,
`pretrained_revision` (all `null`).


## LIBERO

Install both the policy and simulator extras on Linux:

```bash
uv sync --locked --extra perceptron_isaac --extra libero
```

An imported LIBERO package is evaluated through the standard command:

```bash
uv run lerobot-eval \
  --policy.path=/path/to/isaac-lerobot-package \
  --policy.device=cuda:0 \
  --env.type=libero \
  --env.task=libero_spatial \
  --eval.batch_size=1 \
  --eval.n_episodes=20 \
  --seed=0 \
  --output_dir=/tmp/isaac-libero-spatial
```

Before any environment is created, LIBERO adopts the package's target FPS,
256×256 image geometry, and ordered camera mapping. An explicitly supplied
conflicting FPS, resolution, camera mapping, or camera order is rejected, whether
it was passed on the command line or set in a `--config_path` YAML. Simulator
control frequency and policy frame time therefore use the same checkpoint-owned
rate.

ISAAC online inference currently supports one environment per policy instance.
The evaluator caps its auto-selected batch size to one; an explicit
`--eval.batch_size` greater than one is rejected before environment creation.

## Finetuning

Use the imported policy and normal v0.6 trainer:

```bash
uv run lerobot-train \
  --policy.path=/path/to/isaac-lerobot-package \
  --policy.push_to_hub=false \
  --dataset.repo_id=local/isaac_yam \
  --dataset.root=/path/to/lerobot-dataset \
  --batch_size=1 \
  --gradient_accumulation_steps=8 \
  --steps=1000 \
  --output_dir=outputs/train/isaac-yam
```

The trainer uses generic Accelerator accumulation and records accumulation,
world-size, and effective-batch metadata for sample-correct resume. The native
policy owns the joint objective; the shared trainer contains no ISAAC loss
branch.

Training batches must include episode-relative dataset timestamps. Flow parity
must pass explicit `[K,N,H,1]` timesteps and `[K,N,H,A]` noise tensors rather
than relying on matching RNG consumption order.

## Convert preserved YAM MDS data

```bash
uv run lerobot-isaac-convert-mds \
  --shard /path/to/shard.00000.mds \
  --output-root /path/to/isaac-yam-lerobot \
  --repo-id local/isaac_yam \
  --fps 30
```

The converter validates the exact Genesis MDS v2 schema, three-camera
`top,left,right` order, canonical left-then-right 14D joint/gripper order,
finite vectors, timestamps, video lengths, and FPS. It writes atomically and
records source shard hashes. Dataset quantile verification records every
5,000-bin initial/rebin grid and enforces the per-dimension budget
`E_lerobot + E_source + 1e-6`. Model normalization remains the imported
checkpoint's canonical q01/q99 profile and preserves the ISAAC min/max-fallback
and identity-mask semantics.

## Parity artifacts

Compare an immutable Genesis oracle artifact with a native artifact:

```bash
uv run lerobot-isaac-parity \
  --reference /path/to/genesis-oracle \
  --candidate /path/to/native-result \
  --report /tmp/isaac-parity-report.json
```

Sampled timesteps, noise, and optional loss masks are exact inputs. Losses use
`1e-6 + 1e-5*abs(reference)`. Selected gradients require cosine similarity of
at least `0.9999`, relative L2 error at most `1e-3`, and the declared zero-norm
policy. The reference artifact owns the acceptance policy; a candidate cannot
relax it.

Native training follows Genesis's terminality rule for the supervised assistant
footer: a chunk whose last action step is the trajectory's terminal step keeps
`<|end|>`, and every non-terminal chunk is supervised with `<|return|>`. Genesis's
LeRobot-native builder emits a window that never reaches the terminal step, so
non-terminal is the default. Online inference keeps the post-action `<|return|>`
footer.

## SO100 and SO101

Older checkpoints trained in the pre-v0.5 joint convention must serialize the
following processor fields in their package config:

```text
joint_signs   = [1, -1, 1, 1, 1, 1]
joint_offsets = [0, 90, 90, 0, 0, 0]
```

The input step applies `q_model = signs*q_lerobot + offsets` before rendering;
the output step applies the exact inverse after unnormalization. Signs are
restricted to `+1/-1`, offsets must be finite, and the transform cannot exceed
the state/action width. Normally calibrate the arm using current LeRobot; do
not encode the old convention into calibration.

## YAM rollout safety

`bi_yam_follower_guarded` uses two local guarded Portal processes and exposes exactly
14 positions in this order: left joints 0–5, left gripper, right joints 0–5,
right gripper. `bi_yam_leader` is read-only.

The default YAM configuration is torque-off. Rollout loads and validates the
policy, processors, strict feature contract, CUDA device lock, and warm
predictions before connecting. Prediction-only mode still runs the complete
observation/policy/postprocessor path but never calls `send_action`.

Motion requires all of the following explicit, independently checked inputs:

- fresh supervisor readiness bound to the exact Portal processes and CAN
  interfaces;
- live CAN feedback and compatible guarded RPC schemas;
- fresh monotonic observations and timeout-bounded RPCs;
- `torque_enabled=true`, `execute_actions=true`, and explicit per-joint delta
  caps for both arms;
- an authenticated package with a closed-loop-safe normalization profile.

Both 7D targets are validated before either arm is commanded. Any validation,
RPC, acknowledgement, bounds, or watchdog failure requests zero torque on both
arms. A public model release must document live hardware acceptance separately
from offline software tests.

## Evaluation and acceptance boundaries

Standalone and trainer evaluation enforce the same policy limits: ISAAC permits
one vector slot and one task at a time per policy instance. Automatic batch size
is capped at one; explicit conflicting batch/task parallelism is rejected.
Recorded rollout datasets are output sinks: neither empty nor resumed recording
statistics replace checkpoint-owned inference normalization. LIBERO eval
recordings use the declared slash-free feature mapping, raw camera/state values,
executed actions and the environment control FPS, with one writer across episodes.
`lerobot-record` is teleoperation-only in this fork; policy recording uses
`lerobot-rollout` or `lerobot-eval --eval.recording=true`.

Reduced-model and synthetic-environment contract tests are not a benchmark or a
hardware safety qualification. Production checkpoint, renderer, simulator and
robot acceptance must be recorded separately. Qwen3.5 training is implemented;
MK1/Isaac-0.5 training remains explicitly unsupported. Existing RTC, conditioning,
normalization and checkpoint-path restrictions remain mandatory.
