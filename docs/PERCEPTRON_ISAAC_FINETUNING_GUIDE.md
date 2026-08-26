# Fine-tuning Perceptron Isaac

This guide covers the public path from an exported Perceptron Isaac checkpoint
to a fine-tuned LeRobot policy package. It uses the standard LeRobot importer,
dataset, trainer, checkpoint, and evaluation interfaces.

For source lineage and third-party boundaries, see
[`PERCEPTRON_ISAAC_PROVENANCE.md`](./PERCEPTRON_ISAAC_PROVENANCE.md).

## 1. Install

From a clean checkout:

```bash
uv sync --locked --extra perceptron_isaac --extra training --extra peft
uv run lerobot-isaac-import --help
uv run lerobot-train --help
```

For bimanual YAM hardware, add `--extra yam --group yam-hardware`. For LIBERO
evaluation on Linux, add `--extra libero`.

The supported installation must be self-contained. If imports require a
sibling repository, a custom `PYTHONPATH`, or an untracked local wheel, the
release dependency integration is incomplete. At present the external mharmony
runtime package is not yet declared by the extra, so rendering, training, and
inference are not release-ready from a clean install.

## 2. Export the source checkpoint

Genesis DCP-to-Hugging-Face conversion belongs to the Genesis release, not this
repository. Its output must contain the model files and these authenticated
contract sidecars:

- `policy_state_contracts.json`
- `policy_normalization.json`
- `policy_inference_recipe.json`
- `policy_state_identity.safetensors`, unless the original DCP checkpoint is
  supplied to the importer

Use immutable source revisions and keep the source checkpoint, converter
revision, sidecar hashes, and dataset/license information for the model card.

## 3. Review the deployment adapter

`isaac_deployment_adapter.json` contains deployment choices that are not
owned by the training checkpoint, including camera order, image geometry,
control rate, safety limits, and any joint-frame calibration. Bind it to the
three contract JSON files by SHA-256 and review it for the target robot.

Do not copy an adapter between checkpoints or robot profiles. The importer
validates known YAM, SO100/SO101, and LIBERO contracts and rejects conflicting
geometry or control settings.

## 4. Import a native LeRobot package

```bash
RAW=/path/to/raw-hf-export
ADAPTER=/path/to/isaac_deployment_adapter.json
PACKAGE=/path/to/perceptron-isaac-lerobot

uv run lerobot-isaac-import \
  --hf-export "$RAW" \
  --dcp-checkpoint /path/to/trusted/dcp-checkpoint \
  --deployment-adapter "$ADAPTER" \
  --output "$PACKAGE" \
  --policy-state-dataset <dataset-name> \
  --normalization-scope <scope-from-policy_normalization.json> \
  --allow-fast-remote-code
```

`--dcp-checkpoint` is optional when the export includes the identity
safetensors file. It is stronger when the original checkpoint is available,
but DCP metadata uses Python pickle internally; only open a checkpoint produced
by a trusted training run.

`--allow-fast-remote-code` acknowledges that the hash-pinned FAST processor
contains reviewed executable Python. Omit it until you have reviewed the pinned
revision recorded in the package provenance.

After import, confirm that `config.json` describes the intended robot,
features, camera order, image size, action horizon, control rate, and
normalization profile. All referenced sidecars must be package-relative so the
directory can be moved or uploaded intact.

## 5. Validate the dataset contract

The package and dataset must agree before allocating a long training run.

| Profile      | State/action width | Cameras                | Image size | FPS |
| ------------ | -----------------: | ---------------------- | ---------- | --: |
| Bimanual YAM |            14 / 14 | `top`, `left`, `right` | 360 × 640  |  30 |
| SO100/SO101  |              6 / 6 | `top`, `side`          | 256 × 256  |  30 |
| LIBERO       |              8 / 7 | `image`, `wrist_image` | 256 × 256  |  20 |

Check the dataset metadata for:

- the exact ordered feature names expected by the package;
- episode-relative timestamps and the expected FPS;
- `q01` and `q99` statistics for state and action features; and
- decodable video streams with the backend used for training.

Inspect the actual video codec with `ffprobe`. In particular, AV1 datasets
should use `--dataset.video_backend=pyav` when the installed torchcodec build
cannot decode AV1.

Run a short 10–20 step job with the same image geometry, flow sample count,
precision, and per-device batch size planned for the full run. This catches
feature, decoder, memory, and distributed-training errors early.

## 6. Train with the native LeRobot trainer

The imported package is the policy source of truth. A typical single-node DDP
launch is:

```bash
uv run accelerate launch \
  --num_processes=8 \
  --mixed_precision=bf16 \
  -m lerobot.scripts.lerobot_train \
  --policy.path=/path/to/perceptron-isaac-lerobot \
  --policy.device=cuda \
  --dataset.repo_id=<owner/dataset> \
  --dataset.root=/path/to/lerobot-dataset \
  --dataset.video_backend=pyav \
  --batch_size=2 \
  --gradient_accumulation_steps=1 \
  --ddp_find_unused_parameters=false \
  --steps=20000 \
  --save_checkpoint=true \
  --save_freq=2500 \
  --output_dir=outputs/perceptron-isaac-finetune
```

Adjust batch size, steps, save frequency, and tracking options for the
experiment. Do not copy policy knobs between robot profiles: the imported
configuration owns action width, camera roles, normalization behavior, flow
sampling, and the action-expert contract.

For this policy every trainable parameter is expected to participate in each
forward pass, so `--ddp_find_unused_parameters=false` avoids an unnecessary
autograd traversal. If that assumption is violated, DDP fails near the start
of training rather than silently changing gradients.

The effective global batch is:

```text
batch_size × gradient_accumulation_steps × world_size
```

LeRobot does not automatically rescale learning rates or step counts when the
world size changes.

## 7. Inspect saved checkpoints

A checkpoint is releasable only when it can be loaded independently of the
base package and training host. At minimum, check:

- `config.json`, processor JSON files, processor state, and
  `isaac_stats.json` are present;
- every sidecar path in the config is relative to the checkpoint;
- the saved normalization values match the processor state;
- PEFT checkpoints include both adapter weights and the trained action expert;
  and
- loading still works after copying the checkpoint to a different directory.

For PEFT checkpoints, `adapter_config.json` should list the action expert in
`modules_to_save`. Otherwise the package contains the VLM adapter but omits
the fully trained expert.

Resume from the saved training configuration:

```bash
uv run accelerate launch \
  --num_processes=8 \
  --mixed_precision=bf16 \
  -m lerobot.scripts.lerobot_train \
  --config_path=/path/to/checkpoint/train_config.json \
  --resume=true
```

Do not pass `--policy.path` on the same resume command; it selects a new base
policy instead of the checkpoint-owned resume configuration.

## 8. Evaluate and publish

Use the normal LeRobot evaluator. For example, a LIBERO package can be checked
after installing `--extra libero`:

```bash
uv run lerobot-eval \
  --policy.path=/path/to/checkpoint/pretrained_model \
  --policy.device=cuda \
  --env.type=libero \
  --env.task=libero_spatial \
  --eval.batch_size=1 \
  --eval.n_episodes=20 \
  --seed=0 \
  --output_dir=outputs/eval/perceptron-isaac-libero-spatial
```

Training loss is not a robot-success metric. Select checkpoints using a
reproducible evaluation suite or repeated hardware trials with documented
initial conditions and safety limits.

Before publishing, verify the package from a clean environment, add the model
and dataset provenance to the model card, and ensure no checkpoint references
local paths, private storage, or developer-only dependencies.
