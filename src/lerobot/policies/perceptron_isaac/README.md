# Perceptron ISAAC native runtime (draft)

In-tree `perceptron_isaac` policy and processors for preconverted native
Isaac-0.5 packages. The offline checkpoint producer is not included.
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
Set `NATIVE_POLICY_DIR` to its `lerobot_policy` directory, `DATASET_REPO_ID`
to a compatible dataset, and `GPUS` to the number of GPUs on the training node.
Use `torchrun` and set FSDP2 through LeRobot's CLI flags. Set `STEPS` to the
desired training length and use the same value for the learning-rate decay.

```sh
STEPS=100000
torchrun --standalone --nproc_per_node="$GPUS" -m lerobot.scripts.lerobot_train \
  --policy.path="$NATIVE_POLICY_DIR" --dataset.repo_id="$DATASET_REPO_ID" \
  --steps="$STEPS" --policy.max_train_steps="$STEPS" \
  --dataset.video_backend=pyav --policy.train_storage_fp32=true \
  --policy.train_expert_only=false --policy.use_peft=false \
  --parallelism.dp_shard=-1 --accelerator.mixed_precision=bf16 \
  --accelerator.fsdp.reduce_dtype=float32 --accelerator.gradient_accumulation.steps=1 \
  --policy.push_to_hub=false --save_checkpoint=false --save_checkpoint_to_hub=false \
  --env_eval_freq=0 --eval_steps=0
```

Native FP32 initialization requires training, F32 package storage, live FSDP2,
FP32 master storage, no PEFT and no expert-only training. It fails closed on
unsupported private Torch interfaces, missing/root-uncovered trainables or
non-FP32 masters. Other load paths retain BF16 behavior. The
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

## Training objective and RTC

Native DiT uses clean-at-one flow: `x(t)=(1-t)*noise+t*action`, with velocity
`action-noise`. Flow first averages valid weighted elements within each draw,
then draws within each chunk, then chunks equally. Text remains token-normalized.
Each forward normalizes text tokens and Flow chunks across data-parallel ranks,
including unequal local counts after filtering. ISAAC gradient accumulation is
unsupported: set `accelerator.gradient_accumulation.steps=1`. The standard config
preflight rejects larger values before allocating the model or engine. Other
policies retain LeRobot's native gradient accumulation. There is no custom
prepared-window staging or partial-window checkpoint/replay path.

`moe_auxiliary_losses.py` materializes global logical dispatch counts and router
probabilities for each layer. Balance (0.02) and router z-loss (0.001) average
across layers, independently of the LM softmax penalty (0.0001). Every forwarded
position participates, including padding remaining after NTP truncation.

Training load intent and one-time training setup recompute nonpersistent text
RoPE frequencies in F32 from the original formula, including BF16-source training.
They do not replace parameters or widen a rounded BF16 table. Inference retains
the released BF16 frequency table and phase behavior.

The action expert is always the native DiT. Legacy checkpoints, whose action
block says `type="molmoact"`, hold the same network and load into it with RTC off.

Prompts carry an episode-time item before each observation only when the
checkpoint's selected recipe sets `emit_observation_timestamps`. The released YAM
and SO-100/101 recipes do not, and LIBERO does. New packages record the flag in
their render metadata; older packages are resolved from the Flow recipe that
`policy_state_dataset` and `normalization_profile_scope` select in
`hf_model/policy_inference_recipe.json`, and a checkpoint with no declaration
keeps emitting.

RTC inference pins a prefix for checkpoints with a prefix budget. The budget is
the action-expert recipe in `hf_model/config.json` (`rtc_max_delay_steps`, off
when `rtc_probability` is 0), and training stages carry it forward unchanged. A
training stage records its own setting in `rtc_final_stage_max_delay_steps`.
ISAAC training does not sample RTC prefixes, so a fine-tune of an RTC-capable
checkpoint records 0 there and keeps the source budget as an inherited
declaration: prefixes are admitted, a warning says so, and nothing has validated
them on the new weights. A checkpoint with no budget rejects every prefix.

Two callers reach the prefix sampler. Pinned rows use clean flow time 1 and zero
velocity in both.

- Sync (`--inference.type=sync --policy.rtc_prefix_length=P`): each refill pins
  its first `P` rows to the previous chunk's rows
  `[n_action_steps, n_action_steps + P)`. `P` must fit the budget, which is
  checked before the first action is returned, and `n_action_steps + P` must not
  exceed `chunk_size`. Dropping the queued actions, as an instruction change
  does, also drops the carried rows. `P=0`, the default, leaves sampling
  unchanged.
- Asynchronous (`--inference.type=rtc --inference.rtc.mode=trained`): the engine
  passes its measured `inference_delay` and the previous chunk's remaining
  normalized actions; the first `inference_delay` rows become the prefix, and a
  zero delay samples without one. The 60-step YAM and SO-100/101 checkpoints
  declare a budget of 30, which requires `--inference.rtc.execution_horizon=30`.
  Guided RTC is not offered. FP32-storage packages defer their weight load to
  first use; asking for the chunk spec loads them, so the load happens at
  startup and not inside the engine's first deadline.

Asynchronous callers predict once per chunk, so they cannot use the policy's
per-step frame clock. A checkpoint whose recipe emits episode-time tokens sets
`needs_episode_time` in its chunk spec. The engine then supplies each
observation's capture time, relative to the episode's first predicted
observation, as the batch `timestamp`, and the prompt renders that time. This
covers plain asynchronous chunk mode too: LIBERO emits timestamps and has no
RTC budget.

## Evidence and limits

Job 113283 proved two dense FP32-master FSDP2 steps with all 35,723,506,864
parameters learnable. All 16 ranks recorded before/after same-PID standard-CLI
execution with the importer absent/unfindable/unloaded and no outside LeRobot
modules. That proof was on HF base `8c920c4` plus runtime `9c315c1` and CUT
`ac7c0ec74a19811cc939e99b1fe51bc77437e7e37350936e9d3f30c52f497dd1`,
not this updated HF base `ae2b57f`.

Raw/portable-export saving is deferred and fails before gathering weights or writing
package assets. Source-bound native saves remain implemented. Focused CPU tests
cover reduced native save/finalize/moved reload, exact FP32 values and signed zero;
that is not production-scale checkpoint qualification.

Trained RTC ran through `lerobot-rollout` on one H100 against the published 60-step
YAM and SO-100/101 packages on 2026-11-21, with a virtual robot replaying each
package's sample observation. No hardware was driven. The first prediction had no
prefix, later predictions pinned the engine's continuation exactly, and a reset
cleared it. A prefix-conditioned prediction took about 0.95 s for YAM and 0.78 s for
SO-100/101, which is 24 to 29 of the 30 budgeted steps at 30 Hz. With no margin, the
30 Hz runs ended with an exhausted action buffer after 17 and 24 predictions. At
15 Hz both ran 40 s to completion; YAM also needed
`--inference.max_observation_age_s=10`, because its 2.6 s first prediction otherwise
ages the first chunk past the 5 s default.

This draft does not claim current-base multi-GPU qualification, production-scale
save/reload/move, evaluation, convergence or model quality. Keep saving, evaluation
and Hub publishing disabled for the published raw export. The existing checkpoint vocabulary loader
sets process-global `QWEN35_VOCAB_PATH`; do not interleave different package
owners in one process. Instance-bound vocabulary requires a separately reviewed
public mharmony API change; strict vocabulary authentication is not relaxed here.
