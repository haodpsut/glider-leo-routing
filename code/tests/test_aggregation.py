"""Tests for the seed-aggregation protocol used to produce every paper number.

The contract: aggregate per-seed scenario means first, then report mean and standard
deviation of those per-seed means ACROSS seeds. Deterministic baselines must show
zero spread; the learned policy's spread must reflect variation across independently
trained models.
"""

import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))

import make_tables  # noqa: E402

METRIC_KEYS = [
    "mean_latency_ms", "p95_latency_ms", "max_utilization", "overloaded_edges",
    "carried_fraction", "delivered_flows", "total_flows", "mean_path_hops",
]


def _write_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["method", "run_seed", "constellation", "scenario_idx"] + METRIC_KEYS)
        for r in rows:
            w.writerow(r)


def _row(method, seed, carried, scenario):
    # method, seed, constellation, scenario_idx, then metrics
    return [method, seed, "medium", scenario, 40.0, 60.0, 0.9, 0, carried, 25, 30, 7.0]


def test_per_seed_then_across_seeds(tmp_path):
    results = tmp_path / "results"
    results.mkdir()
    # Three trained models; GLIDER carries 0.80 / 0.84 / 0.88 (per-seed means).
    # Within each seed the scenarios vary around that mean, which must NOT leak
    # into the reported std.
    for seed, g in zip([1, 2, 3], [0.80, 0.84, 0.88]):
        rows = []
        for i, delta in enumerate([-0.05, 0.0, +0.05]):  # scenario spread, mean = g
            rows.append(_row("glider", seed, g + delta, i))
            rows.append(_row("sp", seed, 0.75, i))
        _write_csv(results / f"main_medium_s{seed}.csv", rows)

    glider_rows = make_tables.collect(str(results), "main_medium_s*.csv", "glider")
    mean, std, n = make_tables.mean_std(glider_rows, "carried_fraction")
    assert n == 3
    assert mean == 0.84  # mean of per-seed means
    # std across the per-seed means (0.80, 0.84, 0.88), NOT across scenarios.
    assert np.isclose(std, np.std([0.80, 0.84, 0.88]))

    sp_rows = make_tables.collect(str(results), "main_medium_s*.csv", "sp")
    sp_mean, sp_std, sp_n = make_tables.mean_std(sp_rows, "carried_fraction")
    assert sp_n == 3
    assert np.isclose(sp_mean, 0.75)
    assert np.isclose(sp_std, 0.0)  # deterministic baseline: zero spread across seeds


def test_fmt_renders_pm_only_when_spread():
    assert make_tables.fmt(0.84, 0.0) == "$0.840$"
    assert "\\pm" in make_tables.fmt(0.84, 0.03)
    assert make_tables.fmt(float("nan"), float("nan")) == "---"


def test_tables_written(tmp_path):
    results = tmp_path / "results"
    results.mkdir()
    for seed in [1, 2]:
        rows = [_row("glider", seed, 0.8, 0), _row("sp", seed, 0.75, 0),
                _row("ca_global", seed, 0.85, 0)]
        _write_csv(results / f"main_medium_s{seed}.csv", rows)
    out = tmp_path / "tables"
    make_tables.table_main(str(results), str(out))
    make_tables.table_generalization(str(results), str(out))
    text = (out / "main.tex").read_text(encoding="utf-8")
    assert "\\begin{tabular}" in text and "GLIDER" in text
    assert (out / "generalization.tex").exists()
