#!/usr/bin/env bash
# Full experiment pipeline for the paper (RTX 4090).
#
#   1. Train GLIDER and the no-message-passing ablation ONCE PER SEED on the
#      'medium' constellation. Every reported number is therefore a mean over
#      independently trained models, not a single lucky run.
#   2. Evaluate each trained model in-distribution (medium) and zero-shot on unseen
#      shells (starlink, kuiper, telesat) to test inductive generalisation.
#   3. Sweep the ISL failure rate for the robustness study, per seed.
#   4. Emit LaTeX tables (mean +/- std across seeds) and figures straight from the CSVs.
#
# Results land in results/*.csv; the paper reads paper/tables/*.tex and paper/figs/*.pdf.
set -euo pipefail
cd "$(dirname "$0")/.."

# Keep per-step progress visible when this script's output is piped (tee, tmux, nohup).
export PYTHONUNBUFFERED=1

# The reference baselines (CA-Global, Deflect-Oracle) are iterative Dijkstra on the
# CPU, so evaluation on a 1584-satellite shell is CPU-bound. 30 scenarios per cell is
# plenty given three seeds; raise N_EVAL if you want tighter error bars.
SEEDS=${SEEDS:-"1 2 3"}
N_EVAL=${N_EVAL:-30}
EVAL_SEED=${EVAL_SEED:-1000}   # scenario-sampling seed, held fixed across models
TRAIN_CONSTEL=${TRAIN_CONSTEL:-starlink_shell1}
# STEPS overrides the config's step count. Label generation on a 1584-satellite shell
# is CPU-bound, so the full 4000-step, 3-seed, 2-model sweep takes several hours.
# To answer the decision gate fast first, run one seed with fewer steps, e.g.:
#     SEEDS=1 STEPS=1500 bash scripts/run_tmux.sh
# then inspect results/main_starlink_shell1_s1.csv (GLIDER must beat Deflect-Local).
STEPS_ARG=""
[ -n "${STEPS:-}" ] && STEPS_ARG="--steps ${STEPS}"

mkdir -p results runs

for seed in $SEEDS; do
  echo "=== seed ${seed}: training ==="
  python -m glider.train --config configs/main.yaml          --seed "$seed" $STEPS_ARG --out "runs/main_s${seed}"
  python -m glider.train --config configs/ablation_nomp.yaml --seed "$seed" $STEPS_ARG --out "runs/nomp_s${seed}"

  echo "=== seed ${seed}: in-distribution + zero-shot transfer ==="
  # starlink_shell1 is in-distribution (trained on it); the rest test zero-shot
  # transfer, including the small shells where a learned ranker should have little
  # to add over the free Deflect-Local heuristic.
  for preset in starlink_shell1 kuiper_shell telesat_polar medium; do
    python -m glider.evaluate --config configs/main.yaml --ckpt "runs/main_s${seed}/glider.pt" \
        --presets "$preset" --n "$N_EVAL" --seed "$EVAL_SEED" --run-seed "$seed" \
        --out "results/main_${preset}_s${seed}.csv"
  done

  echo "=== seed ${seed}: ablation (no message passing), in-distribution ==="
  python -m glider.evaluate --config configs/ablation_nomp.yaml --ckpt "runs/nomp_s${seed}/glider.pt" \
      --presets "$TRAIN_CONSTEL" --n "$N_EVAL" --seed "$EVAL_SEED" --run-seed "$seed" \
      --out "results/nomp_${TRAIN_CONSTEL}_s${seed}.csv"

  echo "=== seed ${seed}: ISL failure sweep (starlink) ==="
  for f in 0.00 0.05 0.10 0.15 0.20; do
    python -m glider.evaluate --config configs/main.yaml --ckpt "runs/main_s${seed}/glider.pt" \
        --presets starlink_shell1 --failure-min "$f" --failure-max "$f" \
        --n "$N_EVAL" --seed "$EVAL_SEED" --run-seed "$seed" \
        --out "results/failure_${f}_s${seed}.csv"
  done
done

echo "=== aggregating: tables + figures ==="
python scripts/make_tables.py  --results results --out ../paper/tables
python scripts/make_figures.py --results results --runs runs --out ../paper/figs

echo "full run complete. Rebuild the paper: (cd ../paper && latexmk -pdf main.tex)"
