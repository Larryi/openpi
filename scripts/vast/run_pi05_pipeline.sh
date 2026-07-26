#!/usr/bin/env bash
set -Eeuo pipefail
umask 077

: "${HF_TOKEN:?Set HF_TOKEN with private dataset read and model write access}"
: "${MODEL_REPO:?Set the Hugging Face model repository}"
: "${MODEL_REPO_PRIVATE:=1}"
: "${ROBOT_TASK:=${KUAVO_TASK:-task1}}"
: "${WORK_ROOT:=/workspace/kuavo_pi05}"
: "${CODE_DIR:=${WORK_ROOT}/openpi-kuavo}"
: "${PIPELINE_MODE:=smoke}"
: "${CONFIRM_FULL_TRAIN:=NO}"
: "${GPU_IDS:=0,1,2,3}"
: "${GPU_COUNT:=4}"
: "${FSDP_DEVICES:=${GPU_COUNT}}"
: "${SMOKE_GLOBAL_BATCH_SIZE:=${GPU_COUNT}}"
: "${TRAIN_GLOBAL_BATCH_SIZE:=}"
: "${NUM_WORKERS:=8}"
: "${TRAIN_VIDEO_BACKEND:=torchcodec}"
: "${PYTORCH_INDEX_URL:=https://download.pytorch.org/whl/cu128}"
: "${TORCH_VERSION:=2.11.0+cu128}"
: "${TORCHVISION_VERSION:=0.26.0+cu128}"
: "${TORCHCODEC_VERSION:=0.11.1}"
: "${NORM_NUM_WORKERS:=0}"
: "${NORM_BATCH_SIZE:=32}"
: "${NORM_REPO:=${MODEL_REPO}}"
: "${NORM_CACHE_PREFIX:=openpi_norm}"
: "${NUM_TRAIN_STEPS:=30000}"
: "${SMOKE_STEPS:=10}"
: "${SAVE_INTERVAL:=1000}"
: "${LOG_INTERVAL:=20}"
: "${KEEP_PERIOD:=1000000000}"
: "${HF_DOWNLOAD_WORKERS:=16}"
: "${BASE_PARAMS:=gs://openpi-assets/checkpoints/pi05_base/params}"
: "${PALIGEMMA_REPO:=google/paligemma-3b-pt-224}"
: "${OPENPI_PYTHON_VERSION:=3.11}"
: "${CUDA_NVCC_VERSION:=auto}"
: "${RUN_ID:=${ROBOT_TASK}_pi05_gpu${GPU_COUNT}_$(date +%Y%m%d_%H%M%S)}"
: "${AUTO_UPLOAD:=1}"
: "${OVERWRITE:=0}"
: "${RESUME:=0}"
: "${RESUME_REPO:=${MODEL_REPO}}"
: "${AUTO_STOP_INSTANCE:=0}"
: "${AUTO_STOP_ON_FAILURE:=0}"
: "${AUTO_STOP_ON_UPLOAD_FAILURE:=0}"
: "${VAST_INSTANCE_ID:=}"
: "${VAST_API_KEY:=}"
: "${SERVERCHAN_SENDKEY:=}"
: "${MIN_FREE_GB:=50}"
: "${REQUIRE_GPU_NAME:=}"
: "${LR_WARMUP_STEPS:=1000}"
: "${PEAK_LR:=2.5e-5}"
: "${LR_DECAY_STEPS:=${NUM_TRAIN_STEPS}}"
: "${DECAY_LR:=2.5e-6}"
: "${MIN_GPU_MEMORY_MB:=79000}"
: "${MIN_GPU_FREE_MB:=70000}"
: "${EMA_DECAY:=auto}"
: "${REMAT_POLICY:=auto}"
: "${LR_TAIL_START_STEP:=}"
: "${LR_TAIL_DECAY_STEPS:=}"
: "${LR_TAIL_DECAY_LR:=}"
: "${DATASET_MIX_JSON:=}"

for value in MODEL_REPO_PRIVATE AUTO_UPLOAD OVERWRITE RESUME AUTO_STOP_INSTANCE AUTO_STOP_ON_FAILURE AUTO_STOP_ON_UPLOAD_FAILURE; do
  [[ "${!value}" == "0" || "${!value}" == "1" ]] || {
    echo "${value} must be 0 or 1, got ${!value}" >&2
    exit 2
  }
done
case "${TRAIN_VIDEO_BACKEND}" in
  torchcodec|pyav|video_reader) ;;
  *) echo "TRAIN_VIDEO_BACKEND must be torchcodec, pyav, or video_reader" >&2; exit 2 ;;
esac

lr_tail_value_count=0
for value in LR_TAIL_START_STEP LR_TAIL_DECAY_STEPS LR_TAIL_DECAY_LR; do
  [[ -n "${!value}" ]] && lr_tail_value_count=$((lr_tail_value_count + 1))
done
if (( lr_tail_value_count != 0 && lr_tail_value_count != 3 )); then
  echo "LR_TAIL_START_STEP, LR_TAIL_DECAY_STEPS, and LR_TAIL_DECAY_LR must be set together" >&2
  exit 2
fi
if (( lr_tail_value_count == 3 )); then
  [[ "${LR_TAIL_START_STEP}" =~ ^[1-9][0-9]*$ ]] || {
    echo "LR_TAIL_START_STEP must be a positive integer" >&2
    exit 2
  }
  [[ "${LR_TAIL_DECAY_STEPS}" =~ ^[1-9][0-9]*$ ]] || {
    echo "LR_TAIL_DECAY_STEPS must be a positive integer" >&2
    exit 2
  }
  [[ "${LR_TAIL_DECAY_LR}" =~ ^[0-9]+([.][0-9]+)?([eE]-?[0-9]+)?$ ]] || {
    echo "LR_TAIL_DECAY_LR must be a non-negative number" >&2
    exit 2
  }
fi

case "${PIPELINE_MODE}" in
  smoke) GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-${SMOKE_GLOBAL_BATCH_SIZE}}" ;;
  train)
    if [[ -z "${TRAIN_GLOBAL_BATCH_SIZE}" ]]; then
      if (( GPU_COUNT == 1 )); then
        TRAIN_GLOBAL_BATCH_SIZE=16
      else
        TRAIN_GLOBAL_BATCH_SIZE=32
      fi
    fi
    GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-${TRAIN_GLOBAL_BATCH_SIZE}}"
    ;;
  *) echo "PIPELINE_MODE must be smoke or train, got ${PIPELINE_MODE}" >&2; exit 2 ;;
esac

case "${ROBOT_TASK}" in
  task1)
    CONFIG_NAME="pi05_kuavo"
    DATASET_REPO="${DATASET_REPO:-larryi/kuavo-task1-345}"
    DATASET_DIR_NAME="lerobot_task1_345"
    EXPECTED_EPISODES=345
    EXPECTED_FRAMES=81142
    EXPECTED_ACTION_DIM=8
    EXPECTED_FPS=10
    EXPECTED_ROBOT_TYPE="kuavo4pro"
    EXPECTED_CAMERAS="observation.images.head_cam_h,observation.images.wrist_cam_r"
    NORM_ASSET_ID="kuavo_task1"
    ;;
  task2)
    CONFIG_NAME="pi05_kuavo_task2"
    DATASET_REPO="${DATASET_REPO:-larryi/kuavo-task2-264}"
    DATASET_DIR_NAME="lerobot_task2_264"
    EXPECTED_EPISODES=264
    EXPECTED_FRAMES=50042
    EXPECTED_ACTION_DIM=16
    EXPECTED_FPS=10
    EXPECTED_ROBOT_TYPE="kuavo4pro"
    EXPECTED_CAMERAS="observation.images.head_cam_h,observation.images.wrist_cam_l,observation.images.wrist_cam_r"
    NORM_ASSET_ID="kuavo_task2"
    ;;
  so101_60)
    CONFIG_NAME="pi05_so101_60"
    DATASET_REPO="${DATASET_REPO:-larryi/so101_grab_blue_pen_60}"
    DATASET_DIR_NAME="merged_grab_blue_pen_60"
    EXPECTED_EPISODES=60
    EXPECTED_FRAMES=10559
    EXPECTED_ACTION_DIM=6
    EXPECTED_FPS=30
    EXPECTED_ROBOT_TYPE="so101_follower"
    EXPECTED_CAMERAS="observation.images.front,observation.images.wrist"
    NORM_ASSET_ID="so101_grab_blue_pen_60"
    ;;
  so101_90)
    CONFIG_NAME="pi05_so101_90"
    DATASET_REPO="${DATASET_REPO:-larryi/so101_grab_blue_pen_90}"
    DATASET_DIR_NAME="merged_lerobot_dataset_with_dagger30_trimmed"
    EXPECTED_EPISODES=90
    EXPECTED_FRAMES=17391
    EXPECTED_ACTION_DIM=6
    EXPECTED_FPS=30
    EXPECTED_ROBOT_TYPE="so101_follower"
    EXPECTED_CAMERAS="observation.images.front,observation.images.wrist"
    NORM_ASSET_ID="so101_grab_blue_pen_90"
    ;;
  *)
    echo "ROBOT_TASK must be task1, task2, so101_60, or so101_90; got ${ROBOT_TASK}" >&2
    exit 2
    ;;
esac

for value in GPU_COUNT FSDP_DEVICES GLOBAL_BATCH_SIZE NUM_TRAIN_STEPS SMOKE_STEPS SAVE_INTERVAL LOG_INTERVAL KEEP_PERIOD MIN_FREE_GB MIN_GPU_MEMORY_MB; do
  [[ "${!value}" =~ ^[1-9][0-9]*$ ]] || { echo "${value} must be a positive integer" >&2; exit 2; }
done
[[ "${NUM_WORKERS}" =~ ^[0-9]+$ ]] || {
  echo "NUM_WORKERS must be a non-negative integer" >&2
  exit 2
}
[[ "${NORM_NUM_WORKERS}" =~ ^[0-9]+$ ]] || {
  echo "NORM_NUM_WORKERS must be a non-negative integer" >&2
  exit 2
}
[[ "${NORM_BATCH_SIZE}" =~ ^[1-9][0-9]*$ ]] || {
  echo "NORM_BATCH_SIZE must be a positive integer" >&2
  exit 2
}
for value in LR_WARMUP_STEPS LR_DECAY_STEPS; do
  [[ "${!value}" =~ ^[0-9]+$ ]] || {
    echo "${value} must be a non-negative integer" >&2
    exit 2
  }
done
for value in PEAK_LR DECAY_LR; do
  [[ "${!value}" =~ ^[0-9]+([.][0-9]+)?([eE]-?[0-9]+)?$ ]] || {
    echo "${value} must be a non-negative number" >&2
    exit 2
  }
done
[[ "${MIN_GPU_FREE_MB}" =~ ^[0-9]+$ ]] || { echo "MIN_GPU_FREE_MB must be a non-negative integer" >&2; exit 2; }
if [[ "${EMA_DECAY}" == "auto" ]]; then
  # A full-model EMA duplicates a large parameter tree. It is useful on sharded multi-GPU runs but can push
  # single-card full fine-tuning over the memory limit even at batch size one.
  if (( GPU_COUNT == 1 )); then EMA_DECAY="None"; else EMA_DECAY="0.99"; fi
fi
if [[ "${EMA_DECAY}" != "None" && ! "${EMA_DECAY}" =~ ^0(\.[0-9]+)?$ ]]; then
  echo "EMA_DECAY must be auto, None, or a value in [0, 1), got ${EMA_DECAY}" >&2
  exit 2
fi
if [[ "${REMAT_POLICY}" == "auto" ]]; then
  if (( GPU_COUNT == 1 )); then
    REMAT_POLICY="dots_with_no_batch_dims_saveable"
  else
    REMAT_POLICY="nothing_saveable"
  fi
fi
case "${REMAT_POLICY}" in
  nothing_saveable|dots_with_no_batch_dims_saveable|none) ;;
  *) echo "REMAT_POLICY must be nothing_saveable, dots_with_no_batch_dims_saveable, or none" >&2; exit 2 ;;
esac
if (( GLOBAL_BATCH_SIZE % GPU_COUNT != 0 )); then
  echo "GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE} must be divisible by GPU_COUNT=${GPU_COUNT}" >&2
  exit 2
fi
if (( GPU_COUNT % FSDP_DEVICES != 0 )); then
  echo "GPU_COUNT=${GPU_COUNT} must be divisible by FSDP_DEVICES=${FSDP_DEVICES}" >&2
  exit 2
fi
IFS=',' read -r -a selected_gpu_ids <<<"${GPU_IDS}"
if (( ${#selected_gpu_ids[@]} != GPU_COUNT )); then
  echo "GPU_IDS=${GPU_IDS} selects ${#selected_gpu_ids[@]} devices, expected ${GPU_COUNT}" >&2
  exit 2
fi
if [[ "${PIPELINE_MODE}" == "train" && "${CONFIRM_FULL_TRAIN}" != "YES" ]]; then
  echo "Full training requires CONFIRM_FULL_TRAIN=YES" >&2
  exit 2
fi
if [[ ! -f "${CODE_DIR}/scripts/train.py" ]]; then
  echo "OpenPI code not found at ${CODE_DIR}; run push_code_to_vast.sh first" >&2
  exit 3
fi
free_gb="$(df -Pk "${WORK_ROOT}" | awk 'NR == 2 {print int($4 / 1024 / 1024)}')"
if (( free_gb < MIN_FREE_GB )); then
  echo "Only ${free_gb} GiB free under ${WORK_ROOT}; require at least ${MIN_FREE_GB} GiB" >&2
  exit 3
fi

export HF_TOKEN HF_XET_HIGH_PERFORMANCE=1 PYTHONUNBUFFERED=1
export UV_DEFAULT_INDEX="${UV_DEFAULT_INDEX:-https://pypi.org/simple/}"
export UV_HTTP_TIMEOUT="${UV_HTTP_TIMEOUT:-180}"
export HF_HOME="${HF_HOME:-${WORK_ROOT}/hf_cache}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-${WORK_ROOT}/openpi_cache}"
export JAX_COMPILATION_CACHE_DIR="${JAX_COMPILATION_CACHE_DIR:-${WORK_ROOT}/jax_cache}"
# OpenPI full fine-tuning needs more than JAX's default 75% pool and performs several very large allocations on
# the first optimizer step. A fixed 90% pool follows upstream guidance and avoids allocator fragmentation.
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-true}"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.90}"
export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
if [[ -n "${WANDB_API_KEY:-}" ]]; then
  export WANDB_API_KEY WANDB_PROJECT="${WANDB_PROJECT:-openpi-kuavo}"
fi

DATASET_ROOT="${DATASET_ROOT:-${WORK_ROOT}/datasets/${DATASET_DIR_NAME}}"
TOKENIZER_DIR="${TOKENIZER_DIR:-${WORK_ROOT}/models/paligemma-3b-pt-224}"
TOKENIZER_PATH="${TOKENIZER_PATH:-${TOKENIZER_DIR}/tokenizer.model}"
ASSETS_BASE_DIR="${ASSETS_BASE_DIR:-${WORK_ROOT}/assets}"
CHECKPOINT_BASE_DIR="${CHECKPOINT_BASE_DIR:-${WORK_ROOT}/checkpoints}"
LOG_DIR="${WORK_ROOT}/logs/${RUN_ID}"
TRAIN_LOG="${LOG_DIR}/train.log"
PIPELINE_LOG="${LOG_DIR}/pipeline.log"
MANIFEST="${LOG_DIR}/run_manifest.json"
MIX_RESOLVED_FILE="${LOG_DIR}/dataset_mix.resolved.json"
if [[ "${OPENPI_PYTHON_VERSION}" != "3.11" ]]; then
  echo "This frozen OpenPI environment requires OPENPI_PYTHON_VERSION=3.11 because mujoco 2.3.7 has no Python 3.12 wheel" >&2
  exit 2
fi
PYTHON_VERSION="${OPENPI_PYTHON_VERSION}"
export PYTHON_VERSION
VENV="${WORK_ROOT}/.venv-py311"
PYTHON="${VENV}/bin/python"
export UV_PROJECT_ENVIRONMENT="${VENV}"
PIPELINE_PHASE="bootstrap"
TRAIN_STARTED=0
UPLOAD_STATUS="not_started"

mkdir -p "${WORK_ROOT}" "${LOG_DIR}" "${ASSETS_BASE_DIR}" "${CHECKPOINT_BASE_DIR}"
exec > >(tee -a "${PIPELINE_LOG}") 2>&1
echo "Training memory policy: EMA_DECAY=${EMA_DECAY}, REMAT_POLICY=${REMAT_POLICY}, XLA_PYTHON_CLIENT_PREALLOCATE=${XLA_PYTHON_CLIENT_PREALLOCATE}, XLA_PYTHON_CLIENT_MEM_FRACTION=${XLA_PYTHON_CLIENT_MEM_FRACTION}"

retry() {
  local attempts="$1"
  shift
  local attempt=1 rc=0
  while true; do
    if "$@"; then return 0; else rc=$?; fi
    if (( attempt >= attempts )); then return "${rc}"; fi
    echo "Retry ${attempt}/${attempts}: $*" >&2
    sleep $((attempt * 10))
    attempt=$((attempt + 1))
  done
}

upload_run() {
  if [[ "${AUTO_UPLOAD}" != "1" ]]; then
    UPLOAD_STATUS="disabled"
    return 0
  fi
  local run_dir="${CHECKPOINT_BASE_DIR}/${CONFIG_NAME}/${RUN_ID}"
  if [[ ! -d "${run_dir}" ]]; then
    UPLOAD_STATUS="no_checkpoint"
    echo "No checkpoint run to upload: ${run_dir}" >&2
    return 1
  fi
  local latest_step
  latest_step="$(find "${run_dir}" -mindepth 1 -maxdepth 1 -type d -name '[0-9]*' -printf '%f\n' | sort -n | tail -n 1)"
  if [[ -z "${latest_step}" || ! -f "${run_dir}/${latest_step}/_CHECKPOINT_METADATA" ]]; then
    UPLOAD_STATUS="no_checkpoint"
    echo "No finalized Orbax checkpoint to upload under ${run_dir}" >&2
    return 1
  fi
  UPLOAD_STATUS="uploading"
  if MODEL_REPO="${MODEL_REPO}" MODEL_REPO_PRIVATE="${MODEL_REPO_PRIVATE}" RUN_DIR="${run_dir}" \
    LOG_DIR="${LOG_DIR}" MANIFEST="${MANIFEST}" \
    "${PYTHON}" - <<'PY'
import os
from pathlib import Path
from huggingface_hub import HfApi

api = HfApi(token=os.environ["HF_TOKEN"])
repo_id = os.environ["MODEL_REPO"]
private = os.environ["MODEL_REPO_PRIVATE"] == "1"
api.create_repo(repo_id, repo_type="model", private=private, exist_ok=True)
api.update_repo_settings(repo_id, repo_type="model", private=private)
api.upload_large_folder(
    repo_id=repo_id,
    repo_type="model",
    folder_path=os.environ["RUN_DIR"],
)
for local_path in (Path(os.environ["MANIFEST"]), Path(os.environ["LOG_DIR"]) / "train.log"):
    if local_path.is_file():
        api.upload_file(
            repo_id=repo_id,
            repo_type="model",
            path_or_fileobj=local_path,
            path_in_repo=f"logs/{local_path.name}",
        )
actual_private = api.model_info(repo_id).private
assert actual_private is private, (repo_id, actual_private, private)
visibility = "private" if private else "public"
print(f"Uploaded and verified {visibility} model repository: https://huggingface.co/{repo_id}")
PY
  then
    UPLOAD_STATUS="success"
  else
    UPLOAD_STATUS="failed"
    return 1
  fi
}

notify_serverchan() {
  local pipeline_rc="$1"
  [[ -n "${SERVERCHAN_SENDKEY}" ]] || return 0
  SERVERCHAN_SENDKEY="${SERVERCHAN_SENDKEY}" SERVERCHAN_PIPELINE_RC="${pipeline_rc}" \
  SERVERCHAN_PHASE="${PIPELINE_PHASE}" SERVERCHAN_UPLOAD_STATUS="${UPLOAD_STATUS}" \
  SERVERCHAN_HOSTNAME="$(hostname)" PIPELINE_MODE="${PIPELINE_MODE}" ROBOT_TASK="${ROBOT_TASK}" \
  RUN_ID="${RUN_ID}" MODEL_REPO="${MODEL_REPO}" python3 - <<'PY'
import json
import os
import re
import urllib.parse
import urllib.request

key = os.environ["SERVERCHAN_SENDKEY"]
match = re.match(r"^sctp(\d+)t", key)
if match:
    endpoint = f"https://{match.group(1)}.push.ft07.com/send/{key}.send"
else:
    endpoint = f"https://sctapi.ftqq.com/{key}.send"

rc = int(os.environ["SERVERCHAN_PIPELINE_RC"])
status = "成功" if rc == 0 else "失败"
title = f"OpenPI {os.environ['PIPELINE_MODE']} {status}: {os.environ['ROBOT_TASK']}"
description = "\n".join(
    [
        f"- run: `{os.environ['RUN_ID']}`",
        f"- host: `{os.environ['SERVERCHAN_HOSTNAME']}`",
        f"- phase: `{os.environ['SERVERCHAN_PHASE']}`",
        f"- exit code: `{rc}`",
        f"- HF upload: `{os.environ['SERVERCHAN_UPLOAD_STATUS']}`",
        f"- model repo: `{os.environ['MODEL_REPO']}`",
    ]
)
request = urllib.request.Request(
    endpoint,
    data=urllib.parse.urlencode({"title": title, "desp": description}).encode(),
    method="POST",
)
try:
    with urllib.request.urlopen(request, timeout=20) as response:
        payload = json.load(response)
    if payload.get("code") != 0:
        raise RuntimeError("ServerChan returned a non-zero code")
    print("ServerChan notification sent")
except Exception as error:
    # Never print the exception text because HTTP errors may contain the secret URL.
    print(f"ServerChan notification failed: {type(error).__name__}")
PY
}

stop_instance() {
  local pipeline_rc="$1"
  [[ "${AUTO_STOP_INSTANCE}" == "1" ]] || return 0
  if (( pipeline_rc != 0 )); then
    if [[ "${UPLOAD_STATUS}" == "failed" || "${UPLOAD_STATUS}" == "no_checkpoint" ]]; then
      if [[ "${AUTO_STOP_ON_UPLOAD_FAILURE}" != "1" ]]; then
        echo "Upload did not complete; AUTO_STOP_ON_UPLOAD_FAILURE is disabled, leaving instance running" >&2
        return 0
      fi
      echo "Upload did not complete; stopping because AUTO_STOP_ON_UPLOAD_FAILURE=1" >&2
    elif [[ "${AUTO_STOP_ON_FAILURE}" != "1" ]]; then
      echo "Pipeline failed; AUTO_STOP_ON_FAILURE is disabled, leaving instance running" >&2
      return 0
    fi
  fi
  [[ -n "${VAST_API_KEY}" && -n "${VAST_INSTANCE_ID}" ]] || {
    echo "AUTO_STOP_INSTANCE requires VAST_API_KEY and VAST_INSTANCE_ID; leaving instance running" >&2
    return 0
  }
  echo "Requesting Vast instance stop: ${VAST_INSTANCE_ID}"
  VAST_API_KEY="${VAST_API_KEY}" uvx --from vastai vastai stop instance "${VAST_INSTANCE_ID}" --raw
}

on_exit() {
  local rc=$?
  trap - EXIT
  set +e
  echo "Pipeline exit rc=${rc}, phase=${PIPELINE_PHASE}"
  if (( rc != 0 && TRAIN_STARTED == 1 )) && [[ "${UPLOAD_STATUS}" == "not_started" ]]; then
    upload_run
  fi
  notify_serverchan "${rc}"
  stop_instance "${rc}"
  exit "${rc}"
}
trap on_exit EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

PIPELINE_PHASE="create Python environment"
python3 -c 'import sys; assert sys.version_info >= (3, 11), sys.version'
python3 -m pip install --upgrade --index-url "${UV_DEFAULT_INDEX}" uv
retry 3 uv python install "${PYTHON_VERSION}"
if [[ ! -x "${PYTHON}" ]]; then
  uv venv --python "${PYTHON_VERSION}" "${VENV}"
fi
"${PYTHON}" -c 'import sys; assert sys.version_info[:2] == (3, 11), sys.version'
cd "${CODE_DIR}"
environment_digest="$(
  sha256sum pyproject.toml uv.lock | sha256sum | cut -d' ' -f1
)"
environment_stamp="${VENV}/.kuavo-environment-${environment_digest}"
if [[ ! -f "${environment_stamp}" ]]; then
  retry 3 uv sync --python "${PYTHON}" --frozen --no-group dev
else
  echo "Reusing verified OpenPI environment: ${environment_stamp}"
fi
# The frozen lock currently provides CUDA NVCC 12.9. Do not downgrade it: CUDA 12.8+
# is required to compile for Blackwell. An explicit version remains available for
# reproducing an older platform, and is validated against the selected GPU below.
if [[ "${CUDA_NVCC_VERSION}" != "auto" ]]; then
  retry 3 uv pip install --python "${PYTHON}" \
    "nvidia-cuda-nvcc-cu12==${CUDA_NVCC_VERSION}"
fi
# The repository lock still carries the older Torch 2.7 / TorchCodec 0.4 pair.
# The proven local Kuavo training environment uses the CUDA 12.8 Blackwell
# wheels below; TorchCodec 0.11 fixes the spawned-worker decoder crashes and
# restores the original eight-worker input throughput.
retry 3 uv pip install --python "${PYTHON}" \
  --index-url "${PYTORCH_INDEX_URL}" \
  "torch==${TORCH_VERSION}" \
  "torchvision==${TORCHVISION_VERSION}"
retry 3 uv pip install --python "${PYTHON}" \
  --index-url "https://pypi.org/simple" \
  --no-deps \
  "torchcodec==${TORCHCODEC_VERSION}"
CUDA_NVCC_VERSION="$(
  "${PYTHON}" -c \
    'from importlib.metadata import version; print(version("nvidia-cuda-nvcc-cu12"))'
)"
echo "Resolved CUDA NVCC Python package: ${CUDA_NVCC_VERSION}"
"${PYTHON}" - <<'PY'
from importlib.metadata import version

expected = {
    "huggingface-hub": "0.32.3",
    "transformers": "4.53.2",
    "torch": "2.11.0+cu128",
    "torchvision": "0.26.0+cu128",
    "torchcodec": "0.11.1",
}
actual = {package: version(package) for package in expected}
assert actual == expected, (actual, expected)
print("Frozen Python dependency versions passed:", actual)
PY
touch "${environment_stamp}"

PIPELINE_PHASE="CUDA and topology preflight"
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi is required on the Vast host" >&2
  exit 4
fi
nvidia-smi --query-gpu=index,name,memory.total,memory.free,driver_version --format=csv
nvidia-smi topo -m || true
gpu_rows="$(nvidia-smi --query-gpu=index,name,memory.total,memory.free,compute_cap --format=csv,noheader,nounits)"
while IFS=',' read -r gpu_index gpu_name gpu_memory gpu_free compute_cap; do
  gpu_index="${gpu_index// /}"
  [[ ",${GPU_IDS}," == *",${gpu_index},"* ]] || continue
  gpu_name="${gpu_name# }"
  gpu_memory="${gpu_memory// /}"
  gpu_free="${gpu_free// /}"
  compute_cap="${compute_cap// /}"
  if [[ -n "${REQUIRE_GPU_NAME}" && "${gpu_name}" != *"${REQUIRE_GPU_NAME}"* ]]; then
    echo "GPU ${gpu_index} (${gpu_name}) does not match REQUIRE_GPU_NAME=${REQUIRE_GPU_NAME}" >&2
    exit 4
  fi
  if (( gpu_memory < MIN_GPU_MEMORY_MB )); then
    echo "GPU ${gpu_index} (${gpu_name}) has ${gpu_memory} MiB; require ${MIN_GPU_MEMORY_MB} MiB" >&2
    exit 4
  fi
  if (( gpu_free < MIN_GPU_FREE_MB )); then
    echo "GPU ${gpu_index} (${gpu_name}) has only ${gpu_free} MiB free; require ${MIN_GPU_FREE_MB} MiB" >&2
    exit 4
  fi
  compute_major="${compute_cap%%.*}"
  cuda_minor="$(cut -d. -f2 <<<"${CUDA_NVCC_VERSION}")"
  if (( compute_major >= 10 && cuda_minor < 8 )); then
    echo "Blackwell compute capability ${compute_cap} requires CUDA_NVCC_VERSION=12.8 or newer" >&2
    exit 4
  fi
done <<<"${gpu_rows}"
GPU_COUNT="${GPU_COUNT}" "${PYTHON}" - <<'PY'
import os
import jax
import jax.numpy as jnp

assert jax.__version__ == "0.5.3", jax.__version__
devices = jax.devices()
print("JAX devices:", devices)
assert len(devices) == int(os.environ["GPU_COUNT"]), (devices, os.environ["GPU_COUNT"])
assert all(device.platform == "gpu" for device in devices), devices
result = jax.jit(lambda value: value + 1)(jnp.arange(len(devices)))
print("JAX CUDA JIT passed:", result)
PY

PIPELINE_PHASE="authenticate Hugging Face"
MODEL_REPO="${MODEL_REPO}" MODEL_REPO_PRIVATE="${MODEL_REPO_PRIVATE}" "${PYTHON}" - <<'PY'
import os
from huggingface_hub import HfApi

api = HfApi(token=os.environ["HF_TOKEN"])
print("Hugging Face identity:", api.whoami()["name"])
repo_id = os.environ["MODEL_REPO"]
private = os.environ["MODEL_REPO_PRIVATE"] == "1"
api.create_repo(repo_id, repo_type="model", private=private, exist_ok=True)
api.update_repo_settings(repo_id, repo_type="model", private=private)
actual_private = api.model_info(repo_id).private
assert actual_private is private, (repo_id, actual_private, private)
print(f"Hugging Face output repository preflight passed: {repo_id} ({'private' if private else 'public'})")
PY

PIPELINE_PHASE="download dataset and tokenizer"
mkdir -p "${DATASET_ROOT}" "${TOKENIZER_DIR}"
if [[ -n "${DATASET_MIX_JSON}" ]]; then
  DATASET_MIX_JSON="${DATASET_MIX_JSON}" WORK_ROOT="${WORK_ROOT}" \
  MIX_RESOLVED_FILE="${MIX_RESOLVED_FILE}" HF_DOWNLOAD_WORKERS="${HF_DOWNLOAD_WORKERS}" \
  EXPECTED_ACTION_DIM="${EXPECTED_ACTION_DIM}" EXPECTED_FPS="${EXPECTED_FPS}" \
  EXPECTED_ROBOT_TYPE="${EXPECTED_ROBOT_TYPE}" EXPECTED_CAMERAS="${EXPECTED_CAMERAS}" \
  "${PYTHON}" - <<'PY'
import hashlib
import json
import os
from pathlib import Path
import re
from huggingface_hub import HfApi, snapshot_download

sources = json.loads(os.environ["DATASET_MIX_JSON"])
if not isinstance(sources, list) or not sources:
    raise ValueError("DATASET_MIX_JSON must be a non-empty JSON list")
total_weight = sum(float(source["weight"]) for source in sources)
if total_weight <= 0 or any(float(source["weight"]) <= 0 for source in sources):
    raise ValueError("Dataset mixture weights must be positive")
root = Path(os.environ["WORK_ROOT"]) / "datasets" / "mixture"
expected_cameras = set(os.environ["EXPECTED_CAMERAS"].split(","))
resolved = []
api = HfApi(token=os.environ["HF_TOKEN"])
for index, source in enumerate(sources, 1):
    repo_id = str(source["repo_id"])
    safe_name = re.sub(r"[^A-Za-z0-9._-]+", "_", repo_id)
    local_root = root / f"{index:02d}_{safe_name}"
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=local_root,
        max_workers=int(os.environ["HF_DOWNLOAD_WORKERS"]),
        token=os.environ["HF_TOKEN"],
    )
    info = json.loads((local_root / "meta/info.json").read_text())
    assert info["codebase_version"] == "v3.0", (repo_id, info)
    assert info["robot_type"] == os.environ["EXPECTED_ROBOT_TYPE"], (repo_id, info["robot_type"])
    assert info["fps"] == int(os.environ["EXPECTED_FPS"]), (repo_id, info["fps"])
    action_dim = int(os.environ["EXPECTED_ACTION_DIM"])
    assert info["features"]["action"]["shape"] == [action_dim], (repo_id, info["features"]["action"])
    assert info["features"]["observation.state"]["shape"] == [action_dim], (
        repo_id,
        info["features"]["observation.state"],
    )
    assert not any("depth" in key for key in info["features"]), (repo_id, info["features"].keys())
    actual_cameras = {
        key for key in info["features"] if key.startswith("observation.images.")
    }
    assert actual_cameras == expected_cameras, (repo_id, actual_cameras, expected_cameras)
    resolved.append({
        "name": str(source.get("name") or f"source_{index:02d}"),
        "repo_id": repo_id,
        "root": str(local_root),
        "weight": float(source["weight"]) / total_weight,
        "revision": api.dataset_info(repo_id).sha,
    })

output = Path(os.environ["MIX_RESOLVED_FILE"])
output.write_text(json.dumps(resolved, indent=2))
cache_identity = [
    {
        key: source.get(key)
        for key in ("name", "repo_id", "weight", "revision", "episodes")
        if source.get(key) is not None
    }
    for source in resolved
]
digest = hashlib.sha256(
    json.dumps(cache_identity, sort_keys=True, separators=(",", ":")).encode()
).hexdigest()[:10]
(output.parent / "dataset_mix.asset_id").write_text(digest)
print("Weighted dataset mixture ready:")
for source in resolved:
    print(f"  {source['repo_id']} weight={source['weight']:.6f} root={source['root']}")
PY
  export KUAVO_DATASET_MIX_JSON
  KUAVO_DATASET_MIX_JSON="$(<"${MIX_RESOLVED_FILE}")"
  DATASET_ROOT="$("${PYTHON}" -c 'import json,sys; print(json.load(open(sys.argv[1]))[0]["root"])' "${MIX_RESOLVED_FILE}")"
  DATASET_REPO="$("${PYTHON}" -c 'import json,sys; print(json.load(open(sys.argv[1]))[0]["repo_id"])' "${MIX_RESOLVED_FILE}")"
  mix_digest="$(<"${LOG_DIR}/dataset_mix.asset_id")"
  NORM_ASSET_ID="${NORM_ASSET_ID}_mix_${mix_digest}"
  export KUAVO_MIX_ASSET_ID="${NORM_ASSET_ID}"
else
  DATASET_REPO="${DATASET_REPO}" DATASET_ROOT="${DATASET_ROOT}" \
  HF_DOWNLOAD_WORKERS="${HF_DOWNLOAD_WORKERS}" "${PYTHON}" - <<'PY'
import os
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id=os.environ["DATASET_REPO"],
    repo_type="dataset",
    local_dir=os.environ["DATASET_ROOT"],
    max_workers=int(os.environ["HF_DOWNLOAD_WORKERS"]),
    token=os.environ["HF_TOKEN"],
)
PY
  dataset_revision="$(
    DATASET_REPO="${DATASET_REPO}" "${PYTHON}" - <<'PY'
import os
from huggingface_hub import HfApi
print(HfApi(token=os.environ["HF_TOKEN"]).dataset_info(os.environ["DATASET_REPO"]).sha[:10])
PY
  )"
  NORM_ASSET_ID="${NORM_ASSET_ID}_rev_${dataset_revision}"
  export KUAVO_MIX_ASSET_ID="${NORM_ASSET_ID}"
fi
if [[ ! -s "${TOKENIZER_PATH}" ]]; then
  PALIGEMMA_REPO="${PALIGEMMA_REPO}" TOKENIZER_DIR="${TOKENIZER_DIR}" "${PYTHON}" - <<'PY'
import os
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id=os.environ["PALIGEMMA_REPO"],
    local_dir=os.environ["TOKENIZER_DIR"],
    allow_patterns=["tokenizer.model"],
    token=os.environ["HF_TOKEN"],
)
PY
fi

PIPELINE_PHASE="prefetch JAX Pi0.5 base"
BASE_PARAMS="$(BASE_PARAMS="${BASE_PARAMS}" "${PYTHON}" - <<'PY'
import os
from openpi.shared.download import maybe_download

print(maybe_download(os.environ["BASE_PARAMS"]))
PY
)"
[[ -f "${BASE_PARAMS}/_METADATA" ]] || { echo "Invalid JAX params directory: ${BASE_PARAMS}" >&2; exit 5; }
echo "JAX base params ready: ${BASE_PARAMS}"

PIPELINE_PHASE="validate LeRobot v3 dataset"
if [[ -z "${DATASET_MIX_JSON}" ]]; then
  DATASET_ROOT="${DATASET_ROOT}" EXPECTED_EPISODES="${EXPECTED_EPISODES}" \
EXPECTED_FRAMES="${EXPECTED_FRAMES}" EXPECTED_ACTION_DIM="${EXPECTED_ACTION_DIM}" \
EXPECTED_FPS="${EXPECTED_FPS}" EXPECTED_ROBOT_TYPE="${EXPECTED_ROBOT_TYPE}" \
EXPECTED_CAMERAS="${EXPECTED_CAMERAS}" "${PYTHON}" - <<'PY'
import json
import os
from pathlib import Path

root = Path(os.environ["DATASET_ROOT"])
info = json.loads((root / "meta/info.json").read_text())
assert info["codebase_version"] == "v3.0", info
assert info["robot_type"] == os.environ["EXPECTED_ROBOT_TYPE"], info["robot_type"]
assert info["total_episodes"] == int(os.environ["EXPECTED_EPISODES"]), info["total_episodes"]
assert info["total_frames"] == int(os.environ["EXPECTED_FRAMES"]), info["total_frames"]
assert info["fps"] == int(os.environ["EXPECTED_FPS"]), info["fps"]
assert info["features"]["action"]["shape"] == [int(os.environ["EXPECTED_ACTION_DIM"])]
assert info["features"]["observation.state"]["shape"] == [int(os.environ["EXPECTED_ACTION_DIM"])]
assert not any("depth" in key for key in info["features"]), info["features"].keys()
actual_cameras = {key for key in info["features"] if key.startswith("observation.images.")}
expected_cameras = set(os.environ["EXPECTED_CAMERAS"].split(","))
assert actual_cameras == expected_cameras, (actual_cameras, expected_cameras)
print("Dataset validation passed:", info["total_episodes"], info["total_frames"], info["features"].keys())
PY
fi

PIPELINE_PHASE="compute full OpenPI normalization stats"
NORM_FILE="${ASSETS_BASE_DIR}/${CONFIG_NAME}/${NORM_ASSET_ID}/norm_stats.json"
NORM_CACHE_PATH="${NORM_CACHE_PREFIX}/${CONFIG_NAME}/${NORM_ASSET_ID}/norm_stats.json"
echo "OpenPI norm cache target: ${NORM_REPO}/${NORM_CACHE_PATH}"
if [[ ! -s "${NORM_FILE}" ]]; then
  NORM_REPO="${NORM_REPO}" NORM_CACHE_PATH="${NORM_CACHE_PATH}" \
  NORM_FILE="${NORM_FILE}" "${PYTHON}" - <<'PY'
import os
from pathlib import Path
import shutil
from huggingface_hub import hf_hub_download

try:
    cached = hf_hub_download(
        repo_id=os.environ["NORM_REPO"],
        repo_type="model",
        filename=os.environ["NORM_CACHE_PATH"],
        token=os.environ["HF_TOKEN"],
    )
except Exception:
    print("No matching OpenPI norm cache; computing it once.")
else:
    destination = Path(os.environ["NORM_FILE"])
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(cached, destination)
    print(f"Reused OpenPI norm cache: {os.environ['NORM_REPO']}/{os.environ['NORM_CACHE_PATH']}")
PY
fi
if [[ ! -s "${NORM_FILE}" ]]; then
  norm_args=(
    --config-name "${CONFIG_NAME}"
    --dataset-root "${DATASET_ROOT}"
    --tokenizer-path "${TOKENIZER_PATH}"
    --assets-base-dir "${ASSETS_BASE_DIR}"
    --batch-size "${NORM_BATCH_SIZE}"
    --num-workers "${NORM_NUM_WORKERS}"
    --state-action-only
  )
  if ! "${PYTHON}" scripts/compute_norm_stats.py "${norm_args[@]}"; then
    if (( NORM_NUM_WORKERS == 0 )); then
      exit 1
    fi
    echo "Parallel norm-stat loading failed; retrying with num_workers=0" >&2
    norm_args[-1]="0"
    "${PYTHON}" scripts/compute_norm_stats.py "${norm_args[@]}"
  fi
  NORM_REPO="${NORM_REPO}" NORM_CACHE_PATH="${NORM_CACHE_PATH}" \
  NORM_FILE="${NORM_FILE}" MODEL_REPO_PRIVATE="${MODEL_REPO_PRIVATE}" \
  "${PYTHON}" - <<'PY'
import os
from huggingface_hub import HfApi

api = HfApi(token=os.environ["HF_TOKEN"])
api.create_repo(
    os.environ["NORM_REPO"],
    repo_type="model",
    private=os.environ["MODEL_REPO_PRIVATE"] == "1",
    exist_ok=True,
)
api.upload_file(
    repo_id=os.environ["NORM_REPO"],
    repo_type="model",
    path_or_fileobj=os.environ["NORM_FILE"],
    path_in_repo=os.environ["NORM_CACHE_PATH"],
)
print(f"Uploaded reusable OpenPI norm cache: {os.environ['NORM_REPO']}/{os.environ['NORM_CACHE_PATH']}")
PY
fi
[[ -s "${NORM_FILE}" ]] || { echo "Missing norm stats: ${NORM_FILE}" >&2; exit 5; }
NORM_FILE="${NORM_FILE}" EXPECTED_ACTION_DIM="${EXPECTED_ACTION_DIM}" "${PYTHON}" - <<'PY'
import json
import os
from pathlib import Path
import numpy as np

payload = json.loads(Path(os.environ["NORM_FILE"]).read_text())["norm_stats"]
expected_dim = int(os.environ["EXPECTED_ACTION_DIM"])
for key in ("state", "actions"):
    stats = payload[key]
    for field in ("mean", "std", "q01", "q99"):
        values = np.asarray(stats[field], dtype=np.float64)
        assert values.shape == (expected_dim,), (key, field, values.shape)
        assert np.isfinite(values).all(), (key, field)
    assert (np.asarray(stats["std"]) > 1e-8).all(), (key, stats["std"])
    assert (np.asarray(stats["q99"]) > np.asarray(stats["q01"])).all(), key
print("Normalization validation passed:", os.environ["NORM_FILE"])
PY

cat >"${MANIFEST}" <<EOF
{
  "run_id": "${RUN_ID}",
  "task": "${ROBOT_TASK}",
  "config": "${CONFIG_NAME}",
  "dataset_repo": "${DATASET_REPO}",
  "dataset_root": "${DATASET_ROOT}",
  "dataset_mix": ${KUAVO_DATASET_MIX_JSON:-null},
  "base_params": "${BASE_PARAMS}",
  "gpu_ids": "${GPU_IDS}",
  "gpu_count": ${GPU_COUNT},
  "fsdp_devices": ${FSDP_DEVICES},
  "global_batch_size": ${GLOBAL_BATCH_SIZE},
  "num_workers": ${NUM_WORKERS},
  "train_video_backend": "${TRAIN_VIDEO_BACKEND}",
  "torch_version": "${TORCH_VERSION}",
  "torchvision_version": "${TORCHVISION_VERSION}",
  "torchcodec_version": "${TORCHCODEC_VERSION}",
  "ema_decay": "${EMA_DECAY}",
  "remat_policy": "${REMAT_POLICY}",
  "xla_memory_fraction": "${XLA_PYTHON_CLIENT_MEM_FRACTION}",
  "num_train_steps": ${NUM_TRAIN_STEPS},
  "lr_warmup_steps": ${LR_WARMUP_STEPS},
  "peak_lr": "${PEAK_LR}",
  "lr_decay_steps": ${LR_DECAY_STEPS},
  "decay_lr": "${DECAY_LR}",
  "lr_tail_start_step": "${LR_TAIL_START_STEP}",
  "lr_tail_decay_steps": "${LR_TAIL_DECAY_STEPS}",
  "lr_tail_decay_lr": "${LR_TAIL_DECAY_LR}",
  "pipeline_mode": "${PIPELINE_MODE}",
  "model_repo": "${MODEL_REPO}",
  "model_repo_private": ${MODEL_REPO_PRIVATE}
}
EOF

train_args=(
  "${CONFIG_NAME}"
  --exp-name "${RUN_ID}"
  --data.root "${DATASET_ROOT}"
  --data.video-backend "${TRAIN_VIDEO_BACKEND}"
  --data.tokenizer-path "${TOKENIZER_PATH}"
  --data.assets.assets-dir "${ASSETS_BASE_DIR}/${CONFIG_NAME}"
  --data.assets.asset-id "${NORM_ASSET_ID}"
  --weight-loader.params-path "${BASE_PARAMS}"
  --assets-base-dir "${ASSETS_BASE_DIR}"
  --checkpoint-base-dir "${CHECKPOINT_BASE_DIR}"
  --batch-size "${GLOBAL_BATCH_SIZE}"
  --num-workers "${NUM_WORKERS}"
  --fsdp-devices "${FSDP_DEVICES}"
  --save-interval "${SAVE_INTERVAL}"
  --log-interval "${LOG_INTERVAL}"
  --keep-period "${KEEP_PERIOD}"
  --ema-decay "${EMA_DECAY}"
  --model.remat-policy "${REMAT_POLICY}"
  --project-name "${WANDB_PROJECT:-openpi-kuavo}"
)
train_args+=(
  --lr-schedule.warmup-steps "${LR_WARMUP_STEPS}"
  --lr-schedule.peak-lr "${PEAK_LR}"
  --lr-schedule.decay-steps "${LR_DECAY_STEPS}"
  --lr-schedule.decay-lr "${DECAY_LR}"
)
if (( lr_tail_value_count == 3 )); then
  train_args+=(
    --lr-schedule.tail-start-step "${LR_TAIL_START_STEP}"
    --lr-schedule.tail-decay-steps "${LR_TAIL_DECAY_STEPS}"
    --lr-schedule.tail-decay-lr "${LR_TAIL_DECAY_LR}"
  )
fi
if [[ -n "${WANDB_API_KEY:-}" ]]; then
  train_args+=(--wandb-enabled)
else
  train_args+=(--no-wandb-enabled)
fi

PIPELINE_PHASE="train ${PIPELINE_MODE}"
if [[ "${PIPELINE_MODE}" == "smoke" ]]; then
  train_args+=(--num-train-steps "${SMOKE_STEPS}" --no-async-checkpointing)
else
  train_args+=(--num-train-steps "${NUM_TRAIN_STEPS}")
fi
run_dir="${CHECKPOINT_BASE_DIR}/${CONFIG_NAME}/${RUN_ID}"
if [[ "${RESUME}" == "1" ]]; then
  if ! find "${run_dir}" -mindepth 1 -maxdepth 1 -type d -name '[0-9]*' -print -quit 2>/dev/null | grep -q .; then
    PIPELINE_PHASE="download checkpoint for resume"
    mkdir -p "${run_dir}"
    RESUME_REPO="${RESUME_REPO}" RUN_DIR="${run_dir}" HF_DOWNLOAD_WORKERS="${HF_DOWNLOAD_WORKERS}" \
      "${PYTHON}" - <<'PY'
import os
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id=os.environ["RESUME_REPO"],
    repo_type="model",
    local_dir=os.environ["RUN_DIR"],
    max_workers=int(os.environ["HF_DOWNLOAD_WORKERS"]),
    token=os.environ["HF_TOKEN"],
)
PY
  fi
  train_args+=(--resume)
else
  if [[ -d "${run_dir}" ]]; then
    if [[ "${OVERWRITE}" == "1" ]]; then
      train_args+=(--overwrite)
    else
      echo "Checkpoint run already exists: ${run_dir}; set RESUME=1, OVERWRITE=1, or choose a new RUN_ID" >&2
      exit 6
    fi
  fi
fi

TRAIN_STARTED=1
set +e
"${PYTHON}" scripts/train.py "${train_args[@]}" 2>&1 | tee "${TRAIN_LOG}"
train_rc=${PIPESTATUS[0]}
set -e
(( train_rc == 0 )) || exit "${train_rc}"

PIPELINE_PHASE="upload checkpoint"
upload_run
PIPELINE_PHASE="complete"
echo "Pipeline complete: task=${ROBOT_TASK}, mode=${PIPELINE_MODE}, run=${RUN_ID}"
