# Vast.ai: Kuavo Pi0.5 JAX training runbook

This runbook prepares a single Vast.ai host with four A100 80 GB GPUs for OpenPI JAX Pi0.5 fine-tuning. The pipeline is
deliberately fail-closed: its default mode runs a real-model 10-step smoke test, and full training requires an explicit
confirmation variable.

## 1. What is transferred

Keep the following on the Vast persistent volume (normally `/workspace`):

- OpenPI source and its exact LeRobot commit;
- the private LeRobot v3 dataset downloaded from Hugging Face;
- the official Pi0.5 JAX base cached from GCS, or an equivalent local `params/` directory;
- PaliGemma `tokenizer.model`;
- full OpenPI normalization stats;
- Orbax checkpoints, optimizer state, JAX compilation cache, and logs.

The pipeline never converts or copies the dataset to LeRobot v2.1. It does not enable depth, LoRA, ROS, or checkpoint
conversion.

## 2. Rent the instance

Recommended initial filter:

- `4 x A100 80 GB` on one host;
- at least 250 GiB free persistent storage (more if keeping several runs);
- CUDA driver visible through `nvidia-smi`;
- direct SSH access and reliable download bandwidth;
- a recent Ubuntu CUDA image with Python 3.11 or 3.12.

The image's Python may be 3.12, but the frozen project environment is always created with uv-managed Python 3.11 at
`$WORK_ROOT/.venv-py311`. This avoids a `mujoco==2.3.7` source build: that release has a CPython 3.11 Linux wheel but no
CPython 3.12 wheel.

The preflight rejects non-A100 GPUs, cards below 79,000 MiB, the wrong GPU count, a non-GPU JAX backend, or a JAX version
other than 0.5.3. Override `REQUIRE_GPU_NAME`/`MIN_GPU_MEMORY_MB` only intentionally.

### Single RTX PRO 6000 Blackwell Workstation profile

Multi-GPU support does not need to be removed for a single-card run. Override only the topology and hardware profile:

```bash
export GPU_IDS=0
export GPU_COUNT=1
export FSDP_DEVICES=1
export SMOKE_GLOBAL_BATCH_SIZE=1
export TRAIN_GLOBAL_BATCH_SIZE=1
export REQUIRE_GPU_NAME="RTX PRO 6000"
export MIN_GPU_MEMORY_MB=90000
export CUDA_NVCC_VERSION=12.8.93
```

The RTX PRO 6000 Blackwell Workstation Edition has 96 GB, but `FSDP_DEVICES=1` means there is no cross-device parameter
sharding. Start with global batch 1 and probe `1 -> 2 -> 4`; OpenPI has no gradient accumulation in this training path.
Blackwell requires a CUDA 12.8-or-newer compiler toolchain, so the A100 default `CUDA_NVCC_VERSION=12.6.85` must not be
used for this profile. The pipeline still pins JAX 0.5.3 and therefore treats the tiny CUDA JIT and real 10-step Pi0.5
smoke as mandatory compatibility gates. Do not start a full run merely because `nvidia-smi` recognizes the card.

## 3. Send the current working tree

From the local workstation, configure the Vast SSH endpoint shown by the instance UI:

```bash
cd /home/larry/openpi-kuavo
export VAST_SSH_HOST="<host>"
export VAST_SSH_PORT="<port>"
export VAST_SSH_USER=root
export VAST_CODE_DIR=/workspace/kuavo_pi05/openpi-kuavo
scripts/vast/push_code_to_vast.sh
```

For the SO101 single-RTX workflow, use `/workspace/so101_pi05` and follow
[`docs/vast_so101_quickstart.md`](vast_so101_quickstart.md). The sync helper can transfer only the already-computed
`norm_stats.json` files into the persistent assets directory while continuing to exclude checkpoints and model weights.

The rsync excludes `.git`, virtual environments, local datasets, checkpoints, caches, and outputs. It does not delete
unknown remote files.

## 4. Configure secrets and locations

On the Vast host:

```bash
cd /workspace/kuavo_pi05/openpi-kuavo
cp scripts/vast/pi05_vast.env.example /workspace/pi05_vast.env
chmod 600 /workspace/pi05_vast.env
editor /workspace/pi05_vast.env
source /workspace/pi05_vast.env
```

Required secret/value fields are `HF_TOKEN` and `MODEL_REPO`. The token needs read access to the private dataset and write
access to the target model repository. Set `MODEL_REPO_PRIVATE=0` for a public output repository; this does not make the
input dataset public. Do not save tokens under the Git working tree.

Supported task selectors:

| Task | Config | Dataset repository | Episodes | Frames | Effective action | Cameras |
|---|---|---|---:|---:|---:|---:|
| TASK1 | `pi05_kuavo` | `larryi/kuavo-task1-345` | 345 | 81,142 | 8 | 2 |
| TASK2 | `pi05_kuavo_task2` | `larryi/kuavo-task2-264` | 264 | 50,042 | 16 | 3 |
| SO101-60 | `pi05_so101_60` | `larryi/so101_grab_blue_pen_60` | 60 | 10,559 | 6 | 2 |
| SO101-90 | `pi05_so101_90` | `larryi/so101_grab_blue_pen_90` | 90 | 17,391 | 6 | 2 |

Change `DATASET_REPO` if the private Hub repository differs. Dataset metadata is checked before any training starts.
Select the row with `ROBOT_TASK=task1`, `task2`, `so101_60`, or `so101_90`. For a fair SO101 comparison, use separate
run IDs/model repositories and initialize both runs from the same Pi0.5 base checkpoint; do not resume the 90-episode run
from the 60-episode checkpoint.

## 5. Run the mandatory real-model smoke

```bash
source /workspace/pi05_vast.env
export PIPELINE_MODE=smoke
export RUN_ID=task1_pi05_a100x4_smoke_v1
scripts/vast/run_pi05_pipeline.sh
```

Stages:

1. create/reuse a persistent uv environment and install the frozen OpenPI dependency graph via the BFSU PyPI mirror;
   the PTX toolchain is explicitly aligned to CUDA 12.6 because JAX 0.5.3 leaves its nvcc wheel otherwise unconstrained;
2. verify four CUDA devices and compile a tiny JAX operation;
3. authenticate Hugging Face, create the output repository, and verify its requested visibility before training;
4. download and validate the LeRobot v3 dataset and PaliGemma tokenizer;
5. prefetch the 12 GB official JAX base before spending time on normalization;
6. compute full, transformed OpenPI norm stats if absent;
7. run 10 real Pi0.5 forward/backward steps using FSDP over four GPUs;
8. save an Orbax checkpoint and upload the resumable checkpoint folder to the selected public/private model repository.

The first run is long even with only 10 steps because it includes dataset/model downloads, full norm stats, parameter
restore, and XLA compilation. Repeated runs reuse all caches.

## 6. Choose the production global batch

OpenPI currently has no gradient-accumulation option. `--batch-size` is the global batch and must be divisible by the JAX
device count. `fsdp_devices=4` creates a `(data=1, fsdp=4)` mesh and shards large model tensors across all four cards.

After the batch-4 smoke succeeds, probe in separate short runs:

```text
global batch: 4 -> 8 -> 16 -> 32
FSDP devices: 4
steps per probe: 10
```

Do not compare this batch directly with the previous PyTorch `12 x 4` result: JAX parameter/optimizer sharding, activation
memory, compilation, and input pipelines differ. The provided production default is global batch 32, but it must be
validated on the selected Vast host.

At global batch 32, 30,000 optimizer steps expose approximately 960,000 samples:

- TASK1: about 11.8 dataset passes (`30000 * 32 / 81142`);
- TASK2: about 19.2 dataset passes (`30000 * 32 / 50042`).

Adjust steps based on validation/open-loop performance rather than matching the PyTorch epoch count blindly.

## 7. Start full training

Use a new public model repository and stable run ID:

```bash
export PIPELINE_MODE=train
export CONFIRM_FULL_TRAIN=YES
export GLOBAL_BATCH_SIZE=32
export NUM_TRAIN_STEPS=30000
export SAVE_INTERVAL=1000
export KEEP_PERIOD=1000000000
export RUN_ID=task1_pi05_a100x4_full_v1
export MODEL_REPO="<user>/kuavo-task1-pi05-a100x4-full-v1"
export MODEL_REPO_PRIVATE=0
scripts/vast/run_pi05_pipeline.sh
```

Set `WANDB_API_KEY` to enable W&B; otherwise OpenPI runs with W&B disabled. Checkpoints are asynchronous in full mode.
The provided large `KEEP_PERIOD` lets Orbax's `max_to_keep=1` retain only the newest finalized checkpoint; lowering it
preserves milestone checkpoints but can consume tens of gigabytes per milestone.

The pipeline never overwrites an existing run by default. Choose a new `RUN_ID`, set `RESUME=1`, or explicitly set
`OVERWRITE=1`; the latter destroys the existing local run directory and should only be used for disposable smoke runs.

## 8. Resume after interruption or on another instance

Use exactly the same config, run ID, batch size, FSDP topology, model repository, and normalization stats:

```bash
export PIPELINE_MODE=train
export CONFIRM_FULL_TRAIN=YES
export RESUME=1
export RUN_ID=task1_pi05_a100x4_full_v1
export MODEL_REPO="<user>/kuavo-task1-pi05-a100x4-full-v1"
scripts/vast/run_pi05_pipeline.sh
```

If the checkpoint is absent locally, the pipeline downloads the private model repository into the expected Orbax run
directory before passing `--resume`. Full `train_state` is required; a params-only checkpoint cannot resume the optimizer.

## 9. Upload and shutdown policy

`AUTO_UPLOAD=1` uploads the latest full Orbax run folder and verifies that repository visibility matches
`MODEL_REPO_PRIVATE`. Logs and the non-secret run manifest are uploaded separately. Upload is also attempted after a
failed training stage. A public repository exposes all checkpoint tensors and uploaded logs to everyone.

Automatic Vast shutdown is disabled by default. Enable it only after smoke testing the upload path:

```bash
export AUTO_STOP_INSTANCE=1
export AUTO_STOP_ON_UPLOAD_FAILURE=1
export VAST_API_KEY="..."
export VAST_INSTANCE_ID="..."
```

If either value is missing, the script deliberately leaves the paid instance running. Failed pipelines also stay online
unless `AUTO_STOP_ON_FAILURE=1` is explicitly set. `AUTO_STOP_ON_UPLOAD_FAILURE=1` is narrower: it stops after an upload
or quota failure without changing the policy for ordinary training failures. The lifecycle CLI runs in an isolated
`uvx` environment and does not modify the frozen OpenPI training dependencies.

For an interactive Kuavo launch, run `scripts/vast/interactive_kuavo_train.sh`. It defaults to the two fixed private
Kuavo datasets, a public model repository, official PyPI on Vast, automatic upload, and an explicit final confirmation.
The launcher explicitly exports `PYTHON_VERSION=3.11` so a Vast image's ambient `PYTHON_VERSION=3.12` cannot override
the frozen project requirement.
On first use it can save HF, W&B, ServerChan, and Vast API credentials to
`/workspace/kuavo_pi05/.secrets/credentials.env`; the directory is mode `700`, the file is mode `600`, and later
interactive launches load it automatically. Set `OPENPI_VAST_CREDENTIALS_FILE` before launch to choose another
persistent location. Explicit non-empty environment variables override loaded values. The credentials file is
plaintext and must never be copied into the Git working tree or a public model repository.

### Cloud pitfalls already handled

- use Python 3.11: `mujoco==2.3.7` has no suitable Python 3.12 wheel and otherwise attempts a failing source build;
- use the frozen lock: `huggingface-hub==0.32.3`, `transformers==4.53.2`, and `torchcodec==0.4.0` are asserted before GPU work;
- use `https://pypi.org/simple` on Vast by default; the BFSU mirror remains a local/manual override only;
- JAX reserves 90% of VRAM, so the dashboard's nearly constant memory is expected and is not the live batch footprint;
- single-GPU full fine-tuning disables EMA and uses the balanced remat policy to avoid the earlier first-step OOM;
- `max_to_keep=1` plus the very large keep period retains only the latest ordinary finalized checkpoint;
- public model output avoids private-model storage quota, while both Kuavo input datasets remain private;
- output-repository permissions and visibility are checked before expensive dataset/model work starts;
- upload/quota failure can stop the instance independently via `AUTO_STOP_ON_UPLOAD_FAILURE=1`;
- lifecycle variables are inherited at process launch, so use the interactive launcher or source the environment before
  starting the pipeline rather than exporting shutdown settings afterward.

### Continuous LR tail when resuming

The optional `LR_TAIL_START_STEP`, `LR_TAIL_DECAY_STEPS`, and `LR_TAIL_DECAY_LR` settings add a second cosine segment
without changing the original 0–30k schedule. For example, resuming a finalized step-44000 checkpoint with
`44000 / 6000 / 2.5e-7` is continuous at `2.5e-6` and reaches `2.5e-7` at step 50000. Set all three values together.
For this branch, set `KEEP_PERIOD=44000` so Orbax preserves the source checkpoint while retaining the latest checkpoint;
disk space must accommodate both.

For one final success/failure notification, set `SERVERCHAN_SENDKEY` in the private environment file. The notification is
sent after any checkpoint upload attempt and before a Vast stop request; the secret URL is never logged.

## 10. Post-training gates

Before considering a run usable:

1. load the uploaded checkpoint with the same config and norm assets;
2. for Kuavo, run `scripts/open_loop_kuavo.py` on held-out episodes; for SO101, compare JAX policy outputs on a held-out
   episode before connecting a robot client;
3. inspect the Streamlit viewer for camera mapping, action drift, gripper behavior, and horizon degradation;
4. confirm output shapes `(50, 8)` for TASK1, `(50, 16)` for TASK2, or `(50, 6)` for SO101;
5. verify no NaN/Inf and review dataset-range violations;
6. retain the complete Orbax checkpoint until training is conclusively finished.

Open-loop success does not authorize ROS or real-robot control; those remain separate later stages.
