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
