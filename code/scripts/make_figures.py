"""Build paper figures from the evaluation CSVs.

Reads results/*.csv (as produced by glider.evaluate) and writes vector PDFs into
the paper figures directory. Every figure is generated purely from the logged
numbers so the paper never contains a hand-drawn result.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

METHOD_LABEL = {"sp": "Shortest-path", "ca_global": "CA-Global", "glider": "GLIDER"}
METHOD_COLOR = {"sp": "#888888", "ca_global": "#1f4e79", "glider": "#c0504d"}


def read_csv(path: str) -> dict[str, list[dict]]:
    rows: dict[str, list[dict]] = defaultdict(list)
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows[r["method"]].append(r)
    return rows


def _mean(rows: list[dict], key: str) -> float:
    vals = [float(r[key]) for r in rows if r[key] not in ("", "inf")]
    vals = [v for v in vals if np.isfinite(v)]
    return float(np.mean(vals)) if vals else float("nan")


CONSTEL_LABEL = {
    "medium": "Medium\n(train)",
    "starlink_shell1": "Starlink-like",
    "kuiper_shell": "Kuiper-like",
    "telesat_polar": "Telesat-like",
}


def _seed_means(paths: list[str], method: str, metric: str) -> list[float]:
    """One mean per trained model (run_seed): average over that model's scenarios."""
    per_seed: dict[str, list[float]] = {}
    for p in paths:
        rows = read_csv(p).get(method, [])
        for r in rows:
            try:
                v = float(r[metric])
            except (ValueError, KeyError):
                continue
            if np.isfinite(v):
                per_seed.setdefault(r.get("run_seed", "0"), []).append(v)
    return [float(np.mean(v)) for v in per_seed.values() if v]


def fig_generalization(results_dir: str, out_dir: str) -> None:
    """Grouped bars: carried fraction per constellation. Error bars = std across seeds."""
    constels = ["medium", "starlink_shell1", "kuiper_shell", "telesat_polar"]
    methods = ["sp", "ca_global", "glider"]
    data: dict[str, dict[str, list[float]]] = {c: {} for c in constels}
    for c in constels:
        paths = glob.glob(os.path.join(results_dir, f"main_{c}_s*.csv"))
        for m in methods:
            data[c][m] = _seed_means(paths, m, "carried_fraction")
    present = [c for c in constels if any(data[c][m] for m in methods)]
    if not present:
        return
    x = np.arange(len(present))
    w = 0.26
    fig, ax = plt.subplots(figsize=(6.6, 3.2))
    for i, m in enumerate(methods):
        means = [np.mean(data[c][m]) if data[c][m] else 0.0 for c in present]
        errs = [np.std(data[c][m]) if len(data[c][m]) > 1 else 0.0 for c in present]
        ax.bar(x + (i - 1) * w, means, w, yerr=errs, capsize=3,
               label=METHOD_LABEL[m], color=METHOD_COLOR[m])
    ax.set_xticks(x)
    ax.set_xticklabels([CONSTEL_LABEL.get(c, c) for c in present], fontsize=8)
    ax.set_ylabel("Carried demand fraction")
    ax.set_ylim(0, 1.05)
    if "medium" in present:
        ax.axvline(0.5, ls=":", c="k", lw=0.8)
    ax.legend(loc="lower right", fontsize=8)
    ax.set_title("In-distribution vs. zero-shot transfer (bars: mean over seeds)", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "generalization.pdf"))
    plt.close(fig)


def fig_failure(results_dir: str, out_dir: str) -> None:
    """Carried fraction vs ISL failure rate, with std-across-seeds error bars."""
    methods = ["sp", "ca_global", "glider"]
    fracs = sorted({
        os.path.basename(p).split("_")[1]
        for p in glob.glob(os.path.join(results_dir, "failure_*_s*.csv"))
    }, key=float)
    if not fracs:
        return
    fig, ax = plt.subplots(figsize=(5.4, 3.2))
    for m in methods:
        xs, ys, es = [], [], []
        for f in fracs:
            paths = glob.glob(os.path.join(results_dir, f"failure_{f}_s*.csv"))
            vals = _seed_means(paths, m, "carried_fraction")
            if not vals:
                continue
            xs.append(float(f))
            ys.append(float(np.mean(vals)))
            es.append(float(np.std(vals)) if len(vals) > 1 else 0.0)
        if xs:
            ax.errorbar(xs, ys, yerr=es, marker="o", capsize=3,
                        label=METHOD_LABEL[m], color=METHOD_COLOR[m])
    ax.set_xlabel("ISL failure rate")
    ax.set_ylabel("Carried demand fraction")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8)
    ax.set_title("Robustness to link churn (Starlink-like)", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "failure.pdf"))
    plt.close(fig)


def fig_training(runs_dir: str, out_dir: str) -> None:
    """Training curves across seeds: next-hop CE (drives the decision) and cost-to-go."""
    paths = sorted(glob.glob(os.path.join(runs_dir, "main_s*", "history.json")))
    if not paths:
        legacy = os.path.join(runs_dir, "main", "history.json")
        paths = [legacy] if os.path.exists(legacy) else []
    if not paths:
        return

    curves_nh, curves_reg, steps_ref = [], [], None
    for p in paths:
        with open(p, encoding="utf-8") as f:
            hist = json.load(f)
        if not hist:
            continue
        steps_ref = [h["step"] for h in hist]
        curves_nh.append([h.get("nh_ce", np.nan) for h in hist])
        curves_reg.append([h.get("reg_loss", h.get("loss", np.nan)) for h in hist])
    if steps_ref is None:
        return

    n = min(len(c) for c in curves_nh)
    steps = steps_ref[:n]
    nh = np.array([c[:n] for c in curves_nh], dtype=float)
    reg = np.array([c[:n] for c in curves_reg], dtype=float)

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.6, 2.7))
    for ax, arr, name, color in (
        (a1, nh, "Next-hop cross-entropy", "#c0504d"),
        (a2, reg, "Cost-to-go SmoothL1 (ms)", "#1f4e79"),
    ):
        mean, std = np.nanmean(arr, axis=0), np.nanstd(arr, axis=0)
        ax.plot(steps, mean, color=color)
        if arr.shape[0] > 1:
            ax.fill_between(steps, mean - std, mean + std, color=color, alpha=0.2)
        ax.set_xlabel("Training step")
        ax.set_ylabel(name, fontsize=8)
    fig.suptitle("GLIDER training (mean $\\pm$ std over seeds)", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "training.pdf"))
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--out", default="../paper/figs")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    fig_generalization(args.results, args.out)
    fig_failure(args.results, args.out)
    fig_training(args.runs, args.out)
    print(f"[glider] figures written to {args.out}")


if __name__ == "__main__":
    main()
