"""Evaluation: compare SP, CA-Global and GLIDER across many random scenarios.

Usage:
    python -m glider.evaluate --config configs/starlink.yaml --ckpt runs/main/glider.pt \
        --n 50 --seed 1 --out results/main.csv
"""

from __future__ import annotations

import argparse
import csv
import os
from dataclasses import asdict

import numpy as np
import torch

from .baselines import (
    route_ca_global,
    route_deflect_local,
    route_deflect_oracle,
    route_shortest_path,
)
from .config import load_train_config
from .dataset import ScenarioConfig, sample_scenario
from .model import GLIDER
from .policy import route_glider
from .queueing import RoutingMetrics, evaluate_routing

# The four references that bracket GLIDER:
#   sp             - congestion-oblivious shortest path, the deployed rule.
#   deflect_local  - myopic congestion-greedy deflection, NO learning. The bar that
#                    matters: if GLIDER cannot beat this, the model earns nothing.
#   deflect_oracle - best possible deflection with global load knowledge: the ceiling
#                    of GLIDER's own action space.
#   ca_global      - unrestricted centralized congestion-aware routing.
METHODS = ["sp", "deflect_local", "ca_global", "deflect_oracle", "glider"]
_METRIC_KEYS = [
    "mean_latency_ms", "p95_latency_ms", "max_utilization",
    "overloaded_edges", "carried_fraction", "delivered_flows",
    "total_flows", "mean_path_hops",
]


def load_model(ckpt_path: str, node_dim: int, edge_dim: int, device: torch.device) -> GLIDER:
    # weights_only=True refuses to unpickle arbitrary objects. Our checkpoints hold
    # only tensors plus a dict of primitives, so this is safe and forward-compatible
    # with PyTorch flipping this default.
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    c = ckpt["cfg"]
    model = GLIDER(
        node_dim=node_dim, edge_dim=edge_dim,
        hidden=c["hidden"], num_layers=c["num_layers"],
        geo_dim=1, use_messages=c["use_messages"],
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


def evaluate_methods(
    scen_cfg: ScenarioConfig,
    model: GLIDER | None,
    device: torch.device,
    n_scenarios: int,
    seed: int,
    methods: list[str] | None = None,
) -> dict[str, list[RoutingMetrics]]:
    methods = methods or (
        METHODS if model is not None
        else ["sp", "deflect_local", "ca_global", "deflect_oracle"]
    )
    rng = np.random.default_rng(seed)
    out: dict[str, list[RoutingMetrics]] = {m: [] for m in methods}

    for _ in range(n_scenarios):
        scenario = sample_scenario(scen_cfg, rng)
        for m in methods:
            if m == "sp":
                paths = route_shortest_path(scenario.snapshot, scenario.demands, scenario.qcfg)
            elif m == "deflect_local":
                # Sees exactly the load GLIDER sees, so the only difference is the ranker.
                paths = route_deflect_local(
                    scenario.snapshot, scenario.demands, scenario.qcfg, scenario.warmup_load
                )
            elif m == "ca_global":
                paths = route_ca_global(
                    scenario.snapshot, scenario.demands, scenario.qcfg, iters=scen_cfg.ca_iters
                )
            elif m == "deflect_oracle":
                paths = route_deflect_oracle(
                    scenario.snapshot, scenario.demands, scenario.qcfg,
                    iters=scen_cfg.deflect_iters,
                )
            elif m == "glider":
                assert model is not None
                paths = route_glider(model, scenario, scenario.qcfg, device)
            else:
                raise ValueError(f"unknown method {m}")
            out[m].append(evaluate_routing(scenario.snapshot, paths, scenario.qcfg))
    return out


def aggregate(metrics: list[RoutingMetrics]) -> dict[str, tuple[float, float]]:
    agg: dict[str, tuple[float, float]] = {}
    for key in _METRIC_KEYS:
        vals = np.array([getattr(m, key) for m in metrics], dtype=np.float64)
        finite = vals[np.isfinite(vals)]
        if len(finite) == 0:
            agg[key] = (float("inf"), 0.0)
        else:
            agg[key] = (float(finite.mean()), float(finite.std()))
    return agg


def print_table(results: dict[str, list[RoutingMetrics]]) -> None:
    print("\n=== Aggregate metrics (mean over scenarios) ===")
    header = f"{'method':<12}" + "".join(f"{k:>20}" for k in _METRIC_KEYS)
    print(header)
    for m, mets in results.items():
        agg = aggregate(mets)
        row = f"{m:<12}" + "".join(f"{agg[k][0]:>20.4f}" for k in _METRIC_KEYS)
        print(row)


def write_csv(
    results: dict[str, list[RoutingMetrics]],
    path: str,
    run_seed: int = -1,
    constellation: str = "",
) -> None:
    """Write per-scenario metrics, tagged with the training seed and constellation.

    The ``run_seed`` column identifies which trained model produced the row, so
    downstream aggregation can report mean and standard deviation across seeds.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["method", "run_seed", "constellation", "scenario_idx"] + _METRIC_KEYS)
        for m, mets in results.items():
            for i, met in enumerate(mets):
                d = asdict(met)
                writer.writerow([m, run_seed, constellation, i] + [d[k] for k in _METRIC_KEYS])
    print(f"[glider] wrote {path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default="results/eval.csv")
    ap.add_argument("--presets", nargs="*", default=None, help="override scenario presets")
    ap.add_argument("--failure-min", type=float, default=None)
    ap.add_argument("--failure-max", type=float, default=None)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--run-seed", type=int, default=-1,
                    help="training seed of the evaluated checkpoint (tagged into the CSV)")
    args = ap.parse_args()

    from .train import resolve_device
    from .features import EDGE_FEAT_DIM, NODE_FEAT_DIM

    device = resolve_device(args.device)
    scen_cfg = load_train_config(args.config).scenario
    if args.presets:
        scen_cfg.presets = args.presets
    if args.failure_min is not None:
        scen_cfg.failure_min = args.failure_min
    if args.failure_max is not None:
        scen_cfg.failure_max = args.failure_max

    model = None
    if args.ckpt:
        model = load_model(args.ckpt, NODE_FEAT_DIM, EDGE_FEAT_DIM, device)

    results = evaluate_methods(scen_cfg, model, device, args.n, args.seed)
    print_table(results)
    write_csv(results, args.out, run_seed=args.run_seed,
              constellation=",".join(scen_cfg.presets))


if __name__ == "__main__":
    main()
