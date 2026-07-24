#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${OPENPI_PYTHON:-${REPO_ROOT}/.venv/bin/python}"

if [[ ! -x "${PYTHON}" ]]; then
  echo "OpenPI Python is not executable: ${PYTHON}" >&2
  exit 1
fi

# pip NVIDIA wheels place each CUDA component under nvidia/<component>/lib. This environment may
# expose those wheels through --system-site-packages, so JAX's plugin cannot always locate them itself.
NVIDIA_PATHS="$(${PYTHON} - <<'PY'
import pathlib
import nvidia

paths = []
for namespace_root in nvidia.__path__:
    root = pathlib.Path(namespace_root)
    paths.extend(str(path) for path in root.glob("*/lib") if path.is_dir())
print(":".join(dict.fromkeys(paths)))
PY
)"
if [[ -n "${NVIDIA_PATHS}" ]]; then
  export LD_LIBRARY_PATH="${NVIDIA_PATHS}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi

CUDA_NVCC_BIN="$(${PYTHON} - <<'PY'
import pathlib
import nvidia

for namespace_root in nvidia.__path__:
    candidate = pathlib.Path(namespace_root) / "cuda_nvcc" / "bin"
    if (candidate / "ptxas").is_file():
        print(candidate)
        break
PY
)"
if [[ -n "${CUDA_NVCC_BIN}" ]]; then
  export PATH="${CUDA_NVCC_BIN}:${PATH}"
fi

export UV_DEFAULT_INDEX="${UV_DEFAULT_INDEX:-https://mirrors.bfsu.edu.cn/pypi/web/simple}"
export OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-/tmp/openpi-data}"
export HF_HOME="${HF_HOME:-/tmp/openpi-hf-home}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-/tmp/openpi-hf-datasets}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
export PYTHONPATH="${REPO_ROOT}/src:/home/larry/kuavo_data_challenge/third_party/lerobot/src${PYTHONPATH:+:${PYTHONPATH}}"

cd "${REPO_ROOT}"
if [[ "${1:-}" == "--viewer" ]]; then
  shift
  exec "${PYTHON}" -m streamlit run scripts/open_loop_kuavo_viewer.py \
    --server.address 127.0.0.1 \
    --server.port "${STREAMLIT_PORT:-8501}" \
    "$@"
fi
exec "${PYTHON}" -u scripts/open_loop_kuavo.py "$@"
