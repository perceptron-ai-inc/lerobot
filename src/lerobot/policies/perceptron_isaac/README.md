# Perceptron ISAAC native runtime (draft)

In-tree `perceptron_isaac` policy and processors for preconverted native MK1
and Isaac-0.5 packages. The offline checkpoint producer is not included.
Generic Qwen checkpoint packages, PEFT, expert-only training and FSDP1 training
are unsupported and fail explicitly. Developer parity artifact tooling is not included.

## Scoped installation

Use a fresh, dedicated Python >=3.12 environment on Linux x86_64. From this
checkout, use the ISAAC constraints instead of generic `uv sync --locked`:

```sh
python -m pip install -c constraints/perceptron_isaac-cu128.txt \
  --extra-index-url https://download.pytorch.org/whl/cu128 \
  '.[training,perceptron_isaac]'
```

Equivalent uv pip installation (in the dedicated activated environment):

```sh
uv pip install -c constraints/perceptron_isaac-cu128.txt \
  --index https://download.pytorch.org/whl/cu128 --index-strategy unsafe-best-match \
  '.[training,perceptron_isaac]'
```

These select Torch 2.10.0+cu128, torchvision 0.25.0+cu128, Accelerate 1.15.0,
Transformers 5.5.4 and immutable mharmony 0.1.0 source. Access to the mharmony
Git repository is required. No flash-attn or kernels extra is required.
The generic upstream lock remains unchanged and does not qualify this extra.
ISAAC CI uses a separate `.venv-isaac` environment and these constraints, not the
generic lock. It checks the Torch/torchvision/CUDA build versions even in CPU tests.
Clean noneditable installation on this new upstream base remains a merge gate.

Historical execution records directly establish Torch 2.10.0+cu128,
torchvision 0.25.0+cu128 and CUDA 12.8, with mharmony named in configuration.
They do not provide a complete historical pip freeze. The current offline CPU
environment uses Python 3.12.7, Transformers 5.5.4, mharmony 0.1.0,
tokenizers 0.22.2 and Accelerate 1.15.0. The SDK source remains pinned to
`9a57efbf8ada2d73641fc40081f4e4fa6ff2ab92`; CPU checks are not new GPU qualification.

## Preconverted package and training entry point

The qualified package is already published; do not run a converter:

```text
s3://perceptron-checkpoints/phil/isaac_0_5-step-100000-fp32-native-lerobot-vocabfix-113238
69 files; 142,967,074,649 bytes
SHA256SUMS: 85d9743e3f46d19cd5291a00607cfe0abc54eb56d1a40a4fbcd20493e293cc9b
```

Stage that complete package without changing its layout or authenticated assets.
Set `NATIVE_POLICY_DIR` to its `lerobot_policy` directory and `DATASET_REPO_ID`
to a compatible dataset. Use standard `lerobot-train` through an Accelerate
FSDP2 launch configuration, not a custom runtime launcher:

```sh
accelerate launch --config_file "$FSDP2_CONFIG" -m lerobot.scripts.lerobot_train \
  --policy.path="$NATIVE_POLICY_DIR" --dataset.repo_id="$DATASET_REPO_ID" \
  --dataset.video_backend=pyav --policy.train_storage_fp32=true \
  --policy.train_expert_only=false --policy.use_peft=false \
  --parallelism.dp_shard=-1 --accelerator.mixed_precision=bf16 \
  --policy.push_to_hub=false --save_checkpoint=false --save_checkpoint_to_hub=false \
  --env_eval_freq=0 --eval_steps=0
```

Native FP32 initialization requires training, F32 package storage, live FSDP2,
FP32 master storage, no PEFT and no expert-only training. It fails closed on
unsupported private Torch interfaces, missing/root-uncovered trainables or
non-FP32 masters. Other load paths retain BF16 behavior. Production MK1's
qualified hardware/ABI is H100 SM90 / CUDA 12.8. Budget for the 143 GB package,
full model host materialization and optimizer state; the mechanics proof used
16 H100s over two hosts, not a laptop or single-GPU recipe.

## Observation conditioning

Historical executed-action conditioning is unsupported in training and inference.
Legacy policy configs, saved processors and render metadata with
`action_conditioning=true` fail explicitly; use `false`. There is no executed-action
recorder lifecycle. Visual history, last-state proprio, mistake conditioning and
FAST action targets/codecs remain supported.

Online evaluation retains each environment's image/state history and returns
`[B, D]` actions from `[B, H, D]` model chunks. Vector environments share the
rollout clock and synchronized chunk refills. The maintained evaluation loop
freezes completed environments and resets the entire group between rollouts;
`policy.reset()` clears queued actions and observation histories together.
Changing the live batch size requires that global reset. Settling emits one idle
row per environment without adding frames to model history or time.

## Training objective and retired RTC

Native DiT uses clean-at-one flow: `x(t)=(1-t)*noise+t*action`, with velocity
`action-noise`.

`moe_auxiliary_losses.py` materializes global logical dispatch counts and router
probabilities for each layer. Balance (0.02) and router z-loss (0.001) average
across layers, independently of the LM softmax penalty (0.0001). Every forwarded
position participates, including padding remaining after NTP truncation.

RTC execution, prefix pinning and dual-time conditioning are retired. Legacy
RTC-off settings remain readable. Released RTC-capable source metadata and
weights remain authenticated and unchanged; source recipe values do not enable
RTC at runtime. Enabled requests fail before sampling or changing rollout clocks.

## Evidence and limits

Job 113283 proved two dense FP32-master FSDP2 steps with all 35,723,506,864
parameters learnable. All 16 ranks recorded before/after same-PID standard-CLI
execution with the importer absent/unfindable/unloaded and no outside LeRobot
modules. That proof was on HF base `8c920c4` plus runtime `9c315c1` and CUT
`ac7c0ec74a19811cc939e99b1fe51bc77437e7e37350936e9d3f30c52f497dd1`,
not this updated HF base `ae2b57f`.

Raw/portable-export saving is deferred and fails before gathering weights or writing
package assets. Source-bound native MK1 saves remain implemented. Focused CPU tests
cover reduced native save/finalize/moved reload, exact FP32 values and signed zero;
that is not production-scale checkpoint qualification.

This draft does not claim current-base multi-GPU qualification, production-scale
save/reload/move, evaluation, convergence or model quality. Keep saving, evaluation
and Hub publishing disabled for the published raw export. The existing checkpoint vocabulary loader
sets process-global `QWEN35_VOCAB_PATH`; do not interleave different package
owners in one process. Instance-bound vocabulary requires a separately reviewed
public mharmony API change; strict vocabulary authentication is not relaxed here.
