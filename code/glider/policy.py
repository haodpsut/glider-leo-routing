"""GLIDER inference: turn a trained model into a distributed greedy routing.

At inference every node forwards toward destination ``d`` by choosing the neighbour
minimising ``observed_edge_cost(u, v) + Q_hat(v, d)``. ``Q_hat`` comes from the GNN
embeddings; the observed edge cost uses the warm-up (locally observable) congestion
state. Only 1-hop neighbour embeddings and the destination embedding are needed per
hop, so the rule is distributed.
"""

from __future__ import annotations

import numpy as np
import torch

from .baselines import routing_edge_cost
from .dataset import Scenario
from .features import build_features, geo_term
from .model import GLIDER
from .queueing import QueueConfig, trace_path


@torch.no_grad()
def predict_cost_to_go_all(
    model: GLIDER,
    scenario: Scenario,
    device: torch.device,
) -> tuple[torch.Tensor, np.ndarray, object]:
    """Return node embeddings, node positions, and the snapshot feature bundle."""
    feats = build_features(scenario.snapshot, scenario.warmup_load)
    node_feat = torch.from_numpy(feats.node_feat).to(device)
    edge_index = torch.from_numpy(feats.edge_index).to(device)
    edge_feat = torch.from_numpy(feats.edge_feat).to(device)
    h = model.embed(node_feat, edge_index, edge_feat)
    return h, feats.node_pos, feats


@torch.no_grad()
def route_glider(
    model: GLIDER,
    scenario: Scenario,
    qcfg: QueueConfig,
    device: torch.device,
) -> list[tuple[object, list[int]]]:
    """Produce (demand, path) routing under the learned greedy policy."""
    model.eval()
    snapshot = scenario.snapshot
    h, node_pos, _feats = predict_cost_to_go_all(model, scenario, device)

    # Observed congestion-aware forwarding cost (locally measurable queue state).
    edge_weight = routing_edge_cost(snapshot, scenario.warmup_load, qcfg)

    all_nodes = np.arange(snapshot.num_nodes, dtype=np.int64)
    destinations = sorted({d.dst_node for d in scenario.demands})

    # Precompute predicted cost-to-go array per destination.
    q_by_dest: dict[int, np.ndarray] = {}
    for dest in destinations:
        dst_idx = np.full(snapshot.num_nodes, dest, dtype=np.int64)
        geo = geo_term(node_pos, all_nodes, dst_idx)
        src_t = torch.from_numpy(all_nodes).to(device)
        dst_t = torch.from_numpy(dst_idx).to(device)
        geo_t = torch.from_numpy(geo).to(device)
        q = model.cost_to_go(h, src_t, dst_t, geo_t).cpu().numpy()
        q[dest] = 0.0
        q_by_dest[dest] = q

    demand_paths: list[tuple[object, list[int]]] = []
    for demand in scenario.demands:
        q = q_by_dest[demand.dst_node]
        path = trace_path(snapshot, demand.src_node, demand.dst_node, q, edge_weight, qcfg.max_hops)
        demand_paths.append((demand, path))
    return demand_paths
