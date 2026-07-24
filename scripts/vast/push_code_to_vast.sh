#!/usr/bin/env bash
set -Eeuo pipefail

: "${VAST_SSH_HOST:?Set VAST_SSH_HOST}"
: "${VAST_SSH_PORT:?Set VAST_SSH_PORT}"
: "${VAST_SSH_USER:=root}"
: "${VAST_CODE_DIR:=/workspace/kuavo_pi05/openpi-kuavo}"
: "${VAST_ASSETS_DIR:=${VAST_CODE_DIR%/*}/assets}"
: "${SYNC_NORM_STATS:=1}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SSH_ARGS=(-p "${VAST_SSH_PORT}" -o ServerAliveInterval=30 -o ServerAliveCountMax=6)

ssh "${SSH_ARGS[@]}" "${VAST_SSH_USER}@${VAST_SSH_HOST}" "mkdir -p '${VAST_CODE_DIR}'"
rsync -az --info=progress2 \
  -e "ssh ${SSH_ARGS[*]}" \
  --exclude '.git/' \
  --exclude '.venv/' \
  --exclude 'assets/' \
  --exclude 'checkpoints/' \
  --exclude 'outputs/' \
  --exclude 'wandb/' \
  --exclude '__pycache__/' \
  "${REPO_ROOT}/" "${VAST_SSH_USER}@${VAST_SSH_HOST}:${VAST_CODE_DIR}/"

echo "Code synchronized to ${VAST_SSH_USER}@${VAST_SSH_HOST}:${VAST_CODE_DIR}"

if [[ "${SYNC_NORM_STATS}" == "1" && -d "${REPO_ROOT}/assets" ]]; then
  ssh "${SSH_ARGS[@]}" "${VAST_SSH_USER}@${VAST_SSH_HOST}" "mkdir -p '${VAST_ASSETS_DIR}'"
  rsync -az --info=progress2 \
    -e "ssh ${SSH_ARGS[*]}" \
    --include '*/' \
    --include 'norm_stats.json' \
    --exclude '*' \
    "${REPO_ROOT}/assets/" "${VAST_SSH_USER}@${VAST_SSH_HOST}:${VAST_ASSETS_DIR}/"
  echo "Normalization stats synchronized to ${VAST_SSH_USER}@${VAST_SSH_HOST}:${VAST_ASSETS_DIR}"
fi
