#!/usr/bin/env bash
# Create (or update) the `glider` conda environment and verify the GPU is visible.
#
#   bash scripts/setup_conda.sh
#
# Safe to re-run: if the environment already exists it is updated in place.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
ENV_FILE="${REPO_ROOT}/environment.yml"
ENV_NAME="${ENV_NAME:-glider}"

if ! command -v conda >/dev/null 2>&1; then
  echo "error: conda not found on PATH. Install Miniconda first:"
  echo "  wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"
  echo "  bash Miniconda3-latest-Linux-x86_64.sh"
  exit 1
fi

# Make `conda activate` usable inside a non-interactive script.
eval "$(conda shell.bash hook)"

if conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  echo "==> environment '${ENV_NAME}' exists, updating from ${ENV_FILE}"
  conda env update -n "${ENV_NAME}" -f "${ENV_FILE}" --prune
else
  echo "==> creating environment '${ENV_NAME}' from ${ENV_FILE}"
  conda env create -n "${ENV_NAME}" -f "${ENV_FILE}"
fi

conda activate "${ENV_NAME}"

echo
echo "==> environment check"
python - <<'PY'
import torch, numpy, scipy, yaml, matplotlib
print(f"python      : {__import__('sys').version.split()[0]}")
print(f"torch       : {torch.__version__}")
print(f"cuda avail  : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"gpu         : {torch.cuda.get_device_name(0)}")
    print(f"capability  : {torch.cuda.get_device_capability(0)}")
else:
    print("WARNING: CUDA is not visible. Training will fall back to CPU and be slow.")
    print("         Check `nvidia-smi` and that pytorch-cuda matches your driver.")
print(f"numpy       : {numpy.__version__}")
PY

echo
echo "==> running the test suite"
cd "${REPO_ROOT}/code"
pytest -q

echo
echo "Setup complete. Activate with:  conda activate ${ENV_NAME}"
