"""Routing evaluation: cost-to-go, greedy forwarding, load aggregation, M/M/1 delay.

The routing abstraction is deliberately uniform. A *policy* assigns, for each
destination, a next-hop at every node. Following next-hops from a demand source
traces a path. Given all paths we aggregate per-directed-edge load, then score the
routing with an M/M/1 flow-delay model plus utilisation / feasibility metrics.

This uniform interface lets shortest-path baselines, the congestion-aware oracle,
and the learned GLIDER policy all be evaluated by the same scorer, so any
performance difference comes from the *forwarding decisions*, not the metric.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass

import numpy as np

from .network import NetworkSnapshot
from .traffic import Demand

INF = float("inf")


@dataclass
class QueueConfig:
    packet_bits: float = 12000.0        # 1500-byte MTU
    # Multiplier on the M/M/1 queue term. 1.0 is the physical value; larger values
    # make queueing visible at moderate load (documented as a sensitivity knob).
    queue_delay_scale: float = 1.0
    max_hops: int = 128                 # loop / TTL guard when tracing paths


def reverse_dijkstra(snapshot: NetworkSnapshot, dest: int, edge_weight: np.ndarray) -> np.ndarray:
    """Shortest cost-to-go from every node to ``dest`` under ``edge_weight``.

    Runs Dijkstra on the reversed graph starting at ``dest``. Returns a
    ``(num_nodes,)`` array of costs (``inf`` where ``dest`` is unreachable).
    """
    dist = np.full(snapshot.num_nodes, INF)
    dist[dest] = 0.0
    visited = np.zeros(snapshot.num_nodes, dtype=bool)
    pq: list[tuple[float, int]] = [(0.0, dest)]
    while pq:
        d, u = heapq.heappop(pq)
        if visited[u]:
            continue
        visited[u] = True
        # Reverse traversal: incoming edges (w -> u) let us relax w via u.
        for w, e in snapshot.in_neighbors[u]:
            if visited[w]:
                continue
            nd = d + edge_weight[e]
            if nd < dist[w]:
                dist[w] = nd
                heapq.heappush(pq, (nd, w))
    return dist


def next_hop_from_costs(
    snapshot: NetworkSnapshot, node: int, cost_to_go: np.ndarray, edge_weight: np.ndarray
) -> tuple[int, int]:
    """Greedy next hop from ``node`` minimising ``w(node,v) + cost_to_go[v]``.

    Returns ``(next_node, edge_id)`` or ``(-1, -1)`` if no finite-cost move exists.
    """
    best_node, best_edge, best_val = -1, -1, INF
    for v, e in snapshot.out_neighbors[node]:
        val = edge_weight[e] + cost_to_go[v]
        if val < best_val:
            best_val, best_node, best_edge = val, v, e
    return best_node, best_edge


def trace_path(
    snapshot: NetworkSnapshot,
    src: int,
    dest: int,
    cost_to_go: np.ndarray,
    edge_weight: np.ndarray,
    max_hops: int,
) -> list[int] | None:
    """Trace a loop-free greedy path src->dest.

    At each hop the packet moves to the unvisited neighbour minimising
    ``edge_weight + cost_to_go``. Already-visited nodes are excluded (a small path
    record carried in the packet header), so the walk cannot loop; it deflects to
    the next-best neighbour instead. Returns None only on a genuine dead end (all
    neighbours visited) or if ``max_hops`` is exceeded.
    """
    if src == dest:
        return [src]
    path = [src]
    visited = {src}
    node = src
    for _ in range(max_hops):
        best_node, best_val = -1, INF
        for v, e in snapshot.out_neighbors[node]:
            if v in visited:
                continue
            val = edge_weight[e] + cost_to_go[v]
            if val < best_val:
                best_val, best_node = val, v
        if best_node < 0:
            return None  # dead end: every neighbour already on the path
        path.append(best_node)
        if best_node == dest:
            return path
        visited.add(best_node)
        node = best_node
    return None


def paths_to_edge_ids(snapshot: NetworkSnapshot, path: list[int]) -> list[int]:
    """Resolve a node path to the list of edge ids it traverses (min-delay parallel edge)."""
    edge_ids: list[int] = []
    for a, b in zip(path[:-1], path[1:]):
        best_e, best_w = -1, INF
        for v, e in snapshot.out_neighbors[a]:
            if v == b and snapshot.prop_delay_ms[e] < best_w:
                best_w, best_e = snapshot.prop_delay_ms[e], e
        if best_e < 0:
            return []  # broken path
        edge_ids.append(best_e)
    return edge_ids


def aggregate_loads(snapshot: NetworkSnapshot, demand_paths: list[tuple[Demand, list[int]]]) -> np.ndarray:
    """Sum demand rates over the edges each demand traverses -> (num_edges,) Gbps."""
    load = np.zeros(snapshot.num_edges)
    for demand, path in demand_paths:
        if path is None or len(path) < 2:
            continue
        for e in paths_to_edge_ids(snapshot, path):
            load[e] += demand.rate_gbps
    return load


def link_latency_ms(snapshot: NetworkSnapshot, load_gbps: np.ndarray, qcfg: QueueConfig) -> np.ndarray:
    """Per-edge latency = propagation + M/M/1 queue term (inf when overloaded)."""
    cap = snapshot.capacity_gbps
    rho = np.divide(load_gbps, cap, out=np.zeros_like(load_gbps), where=cap > 0)
    service_ms = qcfg.packet_bits / (cap * 1e9) * 1000.0  # transmission time of one MTU
    with np.errstate(divide="ignore"):
        queue = qcfg.queue_delay_scale * service_ms / (1.0 - rho)
    latency = snapshot.prop_delay_ms + queue
    latency[rho >= 1.0] = INF
    return latency


@dataclass
class RoutingMetrics:
    mean_latency_ms: float
    p95_latency_ms: float
    max_utilization: float
    overloaded_edges: int
    carried_fraction: float          # share of demand *volume* delivered on feasible paths
    delivered_flows: int
    total_flows: int
    mean_path_hops: float


def evaluate_routing(
    snapshot: NetworkSnapshot,
    demand_paths: list[tuple[Demand, list[int]]],
    qcfg: QueueConfig,
) -> RoutingMetrics:
    """Score a full routing (list of (demand, path)).

    A demand is *delivered* only if it has a valid path on which no edge is
    overloaded (rho < 1). Latency percentiles are computed over delivered demands,
    weighted by rate so heavy flows count proportionally.
    """
    load = aggregate_loads(snapshot, demand_paths)
    lat = link_latency_ms(snapshot, load, qcfg)
    cap = snapshot.capacity_gbps
    util = np.divide(load, cap, out=np.zeros_like(load), where=cap > 0)

    total_vol = sum(d.rate_gbps for d, _ in demand_paths)
    delivered_vol = 0.0
    delivered_flows = 0
    lat_samples: list[float] = []
    weights: list[float] = []
    hop_samples: list[int] = []

    for demand, path in demand_paths:
        if path is None or len(path) < 2:
            continue
        edge_ids = paths_to_edge_ids(snapshot, path)
        if not edge_ids:
            continue
        path_lat = float(lat[edge_ids].sum())
        if not np.isfinite(path_lat):
            continue  # traverses an overloaded edge -> not delivered
        delivered_vol += demand.rate_gbps
        delivered_flows += 1
        lat_samples.append(path_lat)
        weights.append(demand.rate_gbps)
        hop_samples.append(len(path) - 1)

    if lat_samples:
        lat_arr = np.array(lat_samples)
        w_arr = np.array(weights)
        order = np.argsort(lat_arr)
        lat_sorted = lat_arr[order]
        w_sorted = w_arr[order]
        cum = np.cumsum(w_sorted) / w_sorted.sum()
        p95_idx = int(np.searchsorted(cum, 0.95))
        p95_idx = min(p95_idx, len(lat_sorted) - 1)
        mean_lat = float(np.average(lat_arr, weights=w_arr))
        p95_lat = float(lat_sorted[p95_idx])
        mean_hops = float(np.mean(hop_samples))
    else:
        mean_lat = p95_lat = INF
        mean_hops = 0.0

    return RoutingMetrics(
        mean_latency_ms=mean_lat,
        p95_latency_ms=p95_lat,
        max_utilization=float(util.max()) if snapshot.num_edges else 0.0,
        overloaded_edges=int((util >= 1.0).sum()),
        carried_fraction=delivered_vol / total_vol if total_vol > 0 else 0.0,
        delivered_flows=delivered_flows,
        total_flows=len(demand_paths),
        mean_path_hops=mean_hops,
    )
