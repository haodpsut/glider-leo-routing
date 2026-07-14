#!/usr/bin/env bash
# Launch the full experiment inside a detached tmux session, so it survives SSH drops.
#
#   bash scripts/run_tmux.sh                 # start (or attach to) the run
#   bash scripts/run_tmux.sh --attach        # just attach to the running session
#   bash scripts/run_tmux.sh --status        # print progress without attaching
#   bash scripts/run_tmux.sh --kill          # stop the run
#
# Env vars are passed through to run_full.sh, e.g.
#   SEEDS="1 2 3 4 5" N_EVAL=100 bash scripts/run_tmux.sh
#
# The session runs three panes: the experiment, a live log tail, and nvidia-smi.
#
# `set -u` is deliberately omitted: `conda` is a shell function whose activation
# hooks reference unbound variables, so nounset would abort us on any conda call.
set -eo pipefail

CODE_DIR="$(cd "$(dirname "$0")/.." && pwd)"
SESSION="${SESSION:-glider}"
ENV_NAME="${ENV_NAME:-glider}"
SEEDS="${SEEDS:-1 2 3}"
N_EVAL="${N_EVAL:-50}"
# On a multi-GPU host, pin the run to one device, e.g. CUDA_VISIBLE_DEVICES=1.
# Empty means "use whatever torch picks" (device 0).
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-}"
LOG_DIR="${CODE_DIR}/logs"
LOG_FILE="${LOG_DIR}/run_$(date +%Y%m%d_%H%M%S).log"
LATEST="${LOG_DIR}/latest.log"

command -v tmux >/dev/null 2>&1 || { echo "error: tmux not installed (sudo apt install tmux)"; exit 1; }

case "${1:-}" in
  --attach) exec tmux attach -t "${SESSION}" ;;
  --kill)
      tmux kill-session -t "${SESSION}" 2>/dev/null && echo "killed session '${SESSION}'" \
        || echo "no session '${SESSION}'"
      exit 0 ;;
  --status)
      if [ -f "${LATEST}" ]; then
        echo "=== last 20 lines of ${LATEST} ==="
        tail -20 "${LATEST}"
      else
        echo "no log yet at ${LATEST}"
      fi
      tmux has-session -t "${SESSION}" 2>/dev/null \
        && echo "=== session '${SESSION}' is RUNNING ===" \
        || echo "=== session '${SESSION}' is not running ==="
      exit 0 ;;
esac

if tmux has-session -t "${SESSION}" 2>/dev/null; then
  echo "session '${SESSION}' already running; attaching. (--kill to stop it)"
  exec tmux attach -t "${SESSION}"
fi

# Fail fast, and in the terminal the user is looking at, rather than inside a tmux
# pane the user has to hunt for. A missing env is the single most common cause of a
# run that dies seconds after launch.
if ! command -v conda >/dev/null 2>&1; then
  echo "error: conda not found on PATH. Run: bash scripts/setup_conda.sh"
  exit 1
fi
if ! conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  echo "error: conda environment '${ENV_NAME}' does not exist."
  echo
  echo "Create it first (this also verifies the GPU and runs the tests):"
  echo "    bash scripts/setup_conda.sh"
  echo
  echo "Existing environments:"
  conda env list | sed 's/^/    /'
  exit 1
fi

# A full disk silently breaks training runs hours in. Warn before burning GPU time.
AVAIL_GB=$(df -BG --output=avail . 2>/dev/null | tail -1 | tr -dc '0-9')
if [ -n "${AVAIL_GB}" ] && [ "${AVAIL_GB}" -lt 5 ]; then
  echo "warning: only ${AVAIL_GB}GB free on this filesystem."
  echo "         Checkpoints and logs may fail to write. Free space first:"
  echo "             conda clean --all -y"
  echo
  printf "Continue anyway? [y/N] "
  read -r reply
  case "${reply}" in [yY]*) ;; *) echo "aborted."; exit 1 ;; esac
fi

mkdir -p "${LOG_DIR}"
ln -sf "${LOG_FILE}" "${LATEST}"

# The experiment command. `conda run` avoids needing an interactive shell hook, and
# `stdbuf -oL` keeps the log line-buffered so --status shows live progress.
# PYTHONUNBUFFERED is essential: python block-buffers stdout when piped into tee, so
# without it the per-step progress lines sit in a buffer for minutes and the log pane
# looks frozen. stdbuf alone does not help, since it does not propagate to the python
# grandchild that conda run spawns.
RUN_CMD="cd '${CODE_DIR}' && \
SEEDS='${SEEDS}' N_EVAL='${N_EVAL}' CUDA_VISIBLE_DEVICES='${CUDA_VISIBLE_DEVICES}' \
PYTHONUNBUFFERED=1 \
stdbuf -oL -eL conda run --no-capture-output -n '${ENV_NAME}' \
bash scripts/run_full.sh 2>&1 | tee '${LOG_FILE}'; \
echo; echo '=== run finished (exit=\$?) ==='; echo 'Press any key to close.'; read -n 1"

tmux new-session  -d -s "${SESSION}" -n run "${RUN_CMD}"
tmux split-window -t "${SESSION}:run" -v "tail -f '${LOG_FILE}'"
tmux split-window -t "${SESSION}:run" -h "watch -n 5 nvidia-smi"
tmux select-layout -t "${SESSION}:run" main-horizontal
tmux select-pane   -t "${SESSION}:run.0"

echo "started tmux session '${SESSION}'"
echo "  log      : ${LOG_FILE}"
echo "  attach   : tmux attach -t ${SESSION}     (detach with Ctrl-b then d)"
echo "  status   : bash scripts/run_tmux.sh --status"
echo "  stop     : bash scripts/run_tmux.sh --kill"
echo
echo "Attaching in 2s (Ctrl-b d to detach and leave it running)..."
sleep 2
exec tmux attach -t "${SESSION}"
