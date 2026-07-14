"""GLIDER inference: shortest-path-anchored deflection with a learned ranker.

Each node knows the shortest-path potential D(.,d) (plain link-state Dijkstra on
propagation delay, which routers already compute) and restricts its choice to
neighbours that make progress under it. Among those it forwards to the one
minimising ``observed_edge_cost(u,v) + Q_theta(v,d)``.

Delivery is guaranteed by construction: D strictly decreases every hop, so the walk
cannot loop and must reach the destination. The learned model can only change *which*
shortest-ish path is taken, never whether the packet arrives. That is what bounds
this policy below by shortest path, and it is why replacing the unrestricted greedy
walk with this one removes the p^H compounding-error failure mode.
"""

from __future__ import annotations

import numpy as np
import torch

from .baselines import routing_edge_cost, sp_potential, trace_deflect
from .dataset import Scenario
from .features import build_features, geo_term
from .model import GLIDER
from .queueing import QueueConfig


@torch.no_grad()
def glider_scores(
    model: GLIDER,
    scenario: Scenario,
    device: torch.device,
    destinations: list[int],
) -> dict[int, np.ndarray]:
    """Predicted cost-to-go Q_theta(., d) for every node and each destination."""
    snapshot = scenario.snapshot
    feats = build_features(snapshot, scenario.warmup_load)
    h = model.embed(
        torch.from_numpy(feats.node_feat).to(device),
        torch.from_numpy(feats.edge_index).to(device),
        torch.from_numpy(feats.edge_feat).to(device),
    )
    all_nodes = np.arange(snapshot.num_nodes, dtype=np.int64)
    out: dict[int, np.ndarray] = {}
    for dest in destinations:
        dst_idx = np.full(snapshot.num_nodes, dest, dtype=np.int64)
        geo = geo_term(feats.node_pos, all_nodes, dst_idx)
        q = model.cost_to_go(
            h,
            torch.from_numpy(all_nodes).to(device),
            torch.from_numpy(dst_idx).to(device),
            torch.from_numpy(geo).to(device),
        ).cpu().numpy()
        q[dest] = 0.0
        out[dest] = q
    return out


@torch.no_grad()
def route_glider(
    model: GLIDER,
    scenario: Scenario,
    qcfg: QueueConfig,
    device: torch.device,
) -> list[tuple[object, list[int]]]:
    """Route every demand under the learned deflection policy."""
    model.eval()
    snapshot = scenario.snapshot
    destinations = sorted({d.dst_node for d in scenario.demands})
    q_by_dest = glider_scores(model, scenario, device, destinations)

    # Locally observable congestion cost, and the shortest-path progress anchor.
    edge_cost = routing_edge_cost(snapshot, scenario.warmup_load, qcfg)
    pot_by_dest = {d: sp_potential(snapshot, d) for d in destinations}

    demand_paths: list[tuple[object, list[int]]] = []
    for demand in scenario.demands:
        d = demand.dst_node
        path = trace_deflect(
            snapshot, demand.src_node, d,
            pot_by_dest[d], q_by_dest[d], edge_cost, qcfg.max_hops,
        )
        demand_paths.append((demand, path))
    return demand_paths
