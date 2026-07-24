# Vast.ai SO101 Pi0.5 quickstart

This is the shortest safe path from the current local working tree to a single RTX PRO 6000 Blackwell Workstation
Edition Vast instance. The default pipeline mode performs setup, dataset validation, normalization validation, a real
10-step Pi0.5 forward/backward smoke, checkpoint save, and private Hugging Face upload. It does not start full training
unless explicitly confirmed.

The project environment is fixed to Python 3.11. `gym-aloha` pins `mujoco==2.3.7`, which provides a Linux CPython 3.11
wheel but no CPython 3.12 wheel. Python 3.12 therefore falls back to an unnecessary source build and fails with
`MUJOCO_PATH environment variable is not set`. The pipeline installs uv-managed Python 3.11 into
`/workspace/so101_pi05/.venv-py311` and leaves any previous failed `.venv` untouched.

After `uv sync --frozen`, the pipeline does not broadly upgrade Hugging Face packages. The locked compatible set is
`huggingface-hub==0.32.3`, `transformers==4.53.2`, and `torchcodec==0.4.0`; these versions are asserted before any GPU
work. Dataset/model transfers use the Python Hub API, so a newer `hf` CLI is not required.

## 1. Prepare the Vast instance

Use an image with a recent NVIDIA driver, bootstrap Python 3.11/3.12, direct SSH, and sufficient free space under
`/workspace`.
The pipeline will create persistent caches, datasets, assets, checkpoints, and logs under `/workspace/so101_pi05`.

Record the SSH host and port shown by Vast. Do not put SSH keys or Hugging Face tokens in this repository.

## 2. Sync code and existing normalization stats

Run locally:

```bash
cd /home/larry/openpi-kuavo
export VAST_SSH_HOST="<vast-host>"
export VAST_SSH_PORT="<vast-port>"
export VAST_SSH_USER="root"
export VAST_CODE_DIR="/workspace/so101_pi05/openpi-kuavo"
export VAST_ASSETS_DIR="/workspace/so101_pi05/assets"
export SYNC_NORM_STATS=1
scripts/vast/push_code_to_vast.sh
```

The sync includes the current tracked and untracked source files but excludes `.git`, `.venv`, checkpoints, outputs,
W&B files, and caches. A second restricted rsync transfers only files named `norm_stats.json`; it does not transfer the
12 GB base model or any checkpoint.

Re-run the same command after local code changes. It does not use `--delete`, so unknown remote files are preserved.

## 3. Create the private runtime environment file

SSH to the instance, then:

```bash
cd /workspace/so101_pi05/openpi-kuavo
cp scripts/vast/so101_rtxpro6000.env.example /workspace/so101_60.env
chmod 600 /workspace/so101_60.env
editor /workspace/so101_60.env
```

Fill at least:

```bash
export HF_TOKEN="hf_..."
export MODEL_REPO="larryi/<private-so101-60-smoke-model-repo>"
```

The token needs read access to `larryi/so101_grab_blue_pen_60`, access to the gated PaliGemma tokenizer if required, and
write access to the private model repository. The pipeline creates the model repository as private if it does not exist.

The RTX PRO template uses official PyPI on Vast:

```bash
export UV_DEFAULT_INDEX="https://pypi.org/simple"
```

The pipeline fallback is BFSU only when this variable is absent. Frozen packages still follow the exact URLs and hashes
recorded in `uv.lock`.

## 4. Run the mandatory 60-episode smoke

Prefer a persistent terminal such as `tmux` if it is available:

```bash
source /workspace/so101_60.env
cd /workspace/so101_pi05/openpi-kuavo
scripts/vast/run_pi05_pipeline.sh
```

Do not switch to full training unless all of these pass:

- one RTX PRO 6000 is detected with at least 90,000 MiB;
- at least 88,000 MiB is free before JAX starts, so another process cannot silently consume the training budget;
- CUDA compute capability and the CUDA 12.8 PTX toolchain pass;
- JAX 0.5.3 sees exactly one GPU and completes a JIT operation;
- the downloaded dataset is v3.0 with 60 episodes, 10,559 frames, 30 Hz, two expected cameras, and six actions;
- normalization stats have six finite dimensions with valid quantiles;
- 10 real optimizer steps complete;
- the latest Orbax checkpoint is finalized and uploaded privately.

Logs are written under `/workspace/so101_pi05/logs/$RUN_ID` even if the SSH session disconnects.

## 5. Probe batch size, then start the 60-episode run

The single-card path has no FSDP parameter sharding and no gradient accumulation. It retains full-parameter AdamW but
disables EMA, which avoids keeping a second full model parameter tree. JAX preallocates 90% of the card, matching the
upstream OpenPI full-fine-tuning recommendation and avoiding fragmentation during the first optimizer step. Keep all
other settings fixed and use short, distinct smoke runs to probe global batch `1 -> 2 -> 4`. Stop increasing at the first
OOM or unstable compile.

If the first step still reports OOM at batch one, first confirm the launch log shows `memory.free >= 88000 MiB`,
`EMA_DECAY=None`, and `XLA_PYTHON_CLIENT_MEM_FRACTION=0.90`. Do not lower the batch further—it is already minimal. The
next supported option is a multi-GPU host with `FSDP_DEVICES` equal to the number of GPUs; LoRA is intentionally not
enabled by this workflow.

The RTX PRO profile also selects `REMAT_POLICY=dots_with_no_batch_dims_saveable`. This balanced JAX checkpoint policy
lets Gemma and SigLIP retain expensive dot-product outputs for backward instead of recomputing every intermediate. It
uses more of the preallocated pool and can improve step time without changing model parameters or checkpoint format.
Use a new run ID when comparing it with the conservative `nothing_saveable` baseline because each policy produces a
separate XLA compilation. `REMAT_POLICY=none` is available as an experimental maximum-speed profile, but only try it
after the balanced profile completes a full 10-step smoke without approaching OOM.

For full training, edit or export:

```bash
source /workspace/so101_60.env
export PIPELINE_MODE="train"
export CONFIRM_FULL_TRAIN="YES"
export GLOBAL_BATCH_SIZE="<largest-passed-batch>"
export RUN_ID="so101_60_pi05_rtxpro6000_full_v1"
export MODEL_REPO="larryi/<private-so101-60-full-model-repo>"
scripts/vast/run_pi05_pipeline.sh
```

`max_to_keep=1` retains the latest ordinary checkpoint. During an asynchronous save, old and new checkpoints coexist
temporarily, so disk capacity must cover roughly two checkpoints plus the base model, optimizer state, dataset, and
caches.

### Optional Server酱 notification and automatic stop

Put secrets only in `/workspace/so101_60.env` or `/workspace/so101_90.env`:

```bash
export SERVERCHAN_SENDKEY="<ServerChan-SendKey>"
export AUTO_STOP_INSTANCE="1"
export AUTO_STOP_ON_FAILURE="0"
export VAST_INSTANCE_ID="<instance-id>"
export VAST_API_KEY="<scoped-key-with-manage-instances>"
```

Server酱 sends one final summary containing success/failure, run ID, phase, exit code, HF upload status, and model repo.
The SendKey is never printed. Automatic stop is requested only after notification and, by default, only when the entire
pipeline including private HF upload succeeds. A failed run remains online for diagnosis. The Vast CLI runs through
`uvx`, outside the frozen training environment.

## 6. Repeat independently for 90 episodes

Create a separate environment file:

```bash
cp /workspace/so101_60.env /workspace/so101_90.env
chmod 600 /workspace/so101_90.env
```

Change all of the following:

```bash
export ROBOT_TASK="so101_90"
export DATASET_REPO="larryi/so101_grab_blue_pen_90"
export RUN_ID="so101_90_pi05_rtxpro6000_smoke_v1"
export MODEL_REPO="larryi/<private-so101-90-smoke-model-repo>"
export PIPELINE_MODE="smoke"
export CONFIRM_FULL_TRAIN="NO"
unset GLOBAL_BATCH_SIZE
```

Run the smoke, then give the full 90-episode run another new `RUN_ID` and model repository. Never set `RESUME=1` against
the 60-episode checkpoint: both comparisons must initialize from the same official Pi0.5 base.

## 7. Resume after interruption

Use the same task, dataset, run ID, model repository, global batch, and topology:

```bash
export PIPELINE_MODE="train"
export CONFIRM_FULL_TRAIN="YES"
export RESUME="1"
scripts/vast/run_pi05_pipeline.sh
```

If the local run directory is absent, the pipeline downloads the uploaded private Orbax run before resuming. Keep
automatic Vast shutdown disabled until checkpoint upload and resume have both been tested.
