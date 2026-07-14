#!/usr/bin/env bash
# Fast end-to-end smoke run: tiny constellation, a few dozen steps, then evaluate.
set -euo pipefail
cd "$(dirname "$0")/.."

python -m glider.train --config configs/smoke.yaml --out runs/smoke
python -m glider.evaluate --config configs/smoke.yaml --ckpt runs/smoke/glider.pt \
    --n 5 --seed 7 --out results/smoke.csv
echo "smoke run complete -> results/smoke.csv"
