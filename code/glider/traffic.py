"""Traffic-demand generation between ground stations.

Demands are generated with a gravity model: the demand between two ground stations
is proportional to the product of their weights (a proxy for served population),
scaled so that the aggregate demand reaches a target load level. A ``load_scale``
knob lets experiments sweep the network from lightly to heavily loaded.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class TrafficConfig:
    load_scale: float = 1.0        # multiplies the whole demand matrix
    base_total_gbps: float = 200.0  # aggregate demand at load_scale == 1.0
    skew: float = 1.0              # exponent on gravity weights (>1 concentrates traffic)


@dataclass
class Demand:
    """A single directed flow between two ground-station nodes (global indices)."""

    src_node: int
    dst_node: int
    rate_gbps: float


def make_gravity_demands(
    num_sats: int,
    gs_weights: np.ndarray,
    cfg: TrafficConfig,
    rng: np.random.Generator,
) -> list[Demand]:
    """Build a list of directed GS->GS demands under a gravity model.

    Args:
        num_sats: Number of satellite nodes (ground-station node ids are offset by
            this amount).
        gs_weights: (num_gs,) non-negative weights per ground station.
        cfg: Traffic configuration.
        rng: Random generator (used for a small multiplicative jitter).
    """
    g = len(gs_weights)
    if g < 2:
        return []
    w = np.asarray(gs_weights, dtype=np.float64) ** cfg.skew
    w = w / w.sum()

    # Gravity matrix; zero diagonal (no self traffic).
    gravity = np.outer(w, w)
    np.fill_diagonal(gravity, 0.0)
    # Multiplicative jitter so repeated draws differ but stay correlated with weights.
    jitter = rng.uniform(0.8, 1.2, size=gravity.shape)
    gravity = gravity * jitter
    np.fill_diagonal(gravity, 0.0)

    total_target = cfg.base_total_gbps * cfg.load_scale
    scale = total_target / max(gravity.sum(), 1e-12)
    matrix = gravity * scale

    demands: list[Demand] = []
    for i in range(g):
        for j in range(g):
            if i == j:
                continue
            rate = float(matrix[i, j])
            if rate <= 0.0:
                continue
            demands.append(Demand(src_node=num_sats + i, dst_node=num_sats + j, rate_gbps=rate))
    return demands
