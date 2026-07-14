#!/usr/bin/env bash
# Create (or update) the `glider` conda environment and verify the GPU is visible.
#
#   bash scripts/setup_conda.sh
#
# Safe to re-run: if the environment already exists it is updated in place.
#
# NOTE on `set -u`: conda's own activation hooks (e.g. Anaconda's
# etc/conda/activate.d/qt-main_activate.sh) reference unbound variables. Running
# them under `set -u` aborts this script before it can do anything, so every conda
# call below is wrapped in `set +u`. Do not add `-u` to the line below.
set -eo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
ENV_FILE="${REPO_ROOT}/environment.yml"
ENV_NAME="${ENV_NAME:-glider}"

die() { echo; echo "ERROR: $*" >&2; exit 1; }

command -v conda >/dev/null 2>&1 || die "conda not found on PATH.
  Install Miniconda:
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
    bash Miniconda3-latest-Linux-x86_64.sh"

[ -f "${ENV_FILE}" ] || die "environment file not found: ${ENV_FILE}"

echo "==> conda   : $(command -v conda)"
echo "==> env file: ${ENV_FILE}"
echo "==> env name: ${ENV_NAME}"

# Disk check up front: a half-written env is worse than no env.
AVAIL_GB=$(df -BG --output=avail "${HOME}" 2>/dev/null | tail -1 | tr -dc '0-9' || echo "")
if [ -n "${AVAIL_GB}" ]; then
  echo "==> free disk on \$HOME: ${AVAIL_GB}GB"
  if [ "${AVAIL_GB}" -lt 8 ]; then
    echo "WARNING: the environment needs roughly 5-7GB (PyTorch + CUDA runtime)."
    echo "         Free space first:  conda clean --all -y"
    printf "Continue anyway? [y/N] "
    read -r reply
    case "${reply}" in [yY]*) ;; *) die "aborted; not enough free disk." ;; esac
  fi
fi

# `set -u` is deliberately NOT enabled (see the header note): conda's activation
# hooks reference unbound variables and would abort the script here.
eval "$(conda shell.bash hook)"

echo
if conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  echo "==> environment '${ENV_NAME}' exists, updating in place"
  conda env update -n "${ENV_NAME}" -f "${ENV_FILE}" --prune \
    || die "conda env update failed (see the output above)."
else
  echo "==> creating environment '${ENV_NAME}' (this downloads ~5GB, be patient)"
  conda env create -n "${ENV_NAME}" -f "${ENV_FILE}" \
    || die "conda env create failed (see the output above).
  Common causes:
    - out of disk space          -> conda clean --all -y, then re-run
    - pytorch-cuda too new for the driver -> set pytorch-cuda=11.8 in environment.yml"
fi

conda activate "${ENV_NAME}" || die "could not activate '${ENV_NAME}'."

echo
echo "==> environment check"
python - <<'PY'
import sys
print(f"python      : {sys.version.split()[0]}")
try:
    import torch
except Exception as e:  # noqa: BLE001
    raise SystemExit(f"FATAL: torch did not import: {e}")
print(f"torch       : {torch.__version__}")
print(f"cuda avail  : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(f"gpu {i}       : {torch.cuda.get_device_name(i)}")
else:
    print()
    print("WARNING: CUDA is NOT visible. Training would fall back to CPU and be slow.")
    print("         Check `nvidia-smi`, and that pytorch-cuda in environment.yml")
    print("         matches your driver (12.1 needs a rather recent driver; else 11.8).")
import numpy
print(f"numpy       : {numpy.__version__}")
PY

echo
echo "==> running the test suite"
cd "${REPO_ROOT}/code"
pytest -q || die "tests failed; do not start a training run until they pass."

echo
echo "Setup complete."
echo "  conda activate ${ENV_NAME}"
echo "  bash scripts/run_tmux.sh"
