# Vast SO-101 quick commands

This file contains examples only. Keep HF, W&B, ServerChan, and Vast credentials in a private file outside the Git
working tree, for example `/workspace/so101_90.env` with mode `600`.

## 1. Sync from the local workstation

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

## 2. Create a private environment file on Vast

```bash
cp /workspace/so101_pi05/openpi-kuavo/scripts/vast/so101_rtxpro6000.env.example /workspace/so101_90.env
chmod 600 /workspace/so101_90.env
editor /workspace/so101_90.env
```

Use placeholders until the private file is open in the editor:

```bash
export HF_TOKEN="<hf-token>"
export MODEL_REPO="larryi/<public-model-repo>"
export MODEL_REPO_PRIVATE="0"
export SERVERCHAN_SENDKEY="<serverchan-sendkey>"
export WANDB_API_KEY="<wandb-api-key>"
export WANDB_PROJECT="openpi-so101"

export ROBOT_TASK="so101_90"
export DATASET_REPO="larryi/so101_grab_blue_pen_90"
export RUN_ID="so101_90_pi05_rtxpro6000_full_v1"
export PIPELINE_MODE="train"
export CONFIRM_FULL_TRAIN="YES"
export GLOBAL_BATCH_SIZE="16"
export NUM_TRAIN_STEPS="30000"

export AUTO_UPLOAD="1"
export AUTO_STOP_INSTANCE="1"
export AUTO_STOP_ON_FAILURE="0"
export AUTO_STOP_ON_UPLOAD_FAILURE="1"
export VAST_INSTANCE_ID="<instance-id>"
export VAST_API_KEY="<vast-api-key>"
```

## 3. Start

```bash
source /workspace/so101_90.env
cd /workspace/so101_pi05/openpi-kuavo
scripts/vast/run_pi05_pipeline.sh
```
