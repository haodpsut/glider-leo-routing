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

SEEDS=${SEEDS:-"1 2 3"}
N_EVAL=${N_EVAL:-50}
EVAL_SEED=${EVAL_SEED:-1000}   # scenario-sampling seed, held fixed across models

mkdir -p results runs

for seed in $SEEDS; do
  echo "=== seed ${seed}: training ==="
  python -m glider.train --config configs/main.yaml          --seed "$seed" --out "runs/main_s${seed}"
  python -m glider.train --config configs/ablation_nomp.yaml --seed "$seed" --out "runs/nomp_s${seed}"

  echo "=== seed ${seed}: in-distribution + zero-shot transfer ==="
  for preset in medium starlink_shell1 kuiper_shell telesat_polar; do
    python -m glider.evaluate --config configs/main.yaml --ckpt "runs/main_s${seed}/glider.pt" \
        --presets "$preset" --n "$N_EVAL" --seed "$EVAL_SEED" --run-seed "$seed" \
        --out "results/main_${preset}_s${seed}.csv"
  done

  echo "=== seed ${seed}: ablation (no message passing) ==="
  python -m glider.evaluate --config configs/ablation_nomp.yaml --ckpt "runs/nomp_s${seed}/glider.pt" \
      --presets medium --n "$N_EVAL" --seed "$EVAL_SEED" --run-seed "$seed" \
      --out "results/nomp_medium_s${seed}.csv"

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
