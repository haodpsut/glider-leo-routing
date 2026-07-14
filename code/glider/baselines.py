"""Routing baselines and the iterative congestion-aware reference (CA-Global).

* ``route_shortest_path`` — congestion-oblivious minimum-propagation-delay routing
  (one reverse-Dijkstra per destination). This mirrors the default ISL routing used
  by Hypatia / DisCoRoute-style systems.
* ``route_ca_global`` — an iterative congestion-aware routing that has access to the
  full instantaneous traffic matrix and link loads. It converges (via the method of
  successive averages) to a load-balanced user equilibrium and serves as a strong,
  centralised *upper reference*; it is NOT distributed. Its converged cost-to-go maps
  are the imitation labels for the learned policy.
"""

from __future__ import annotations

import numpy as np

from .network import NetworkSnapshot
from .queueing import QueueConfig, reverse_dijkstra, trace_path
from .traffic import Demand

_OVERLOAD_PENALTY_MS = 1.0e4


def routing_edge_cost(snapshot: NetworkSnapshot, load_gbps: np.ndarray, qcfg: QueueConfig) -> np.ndarray:
    """Finite per-edge routing cost (average M/M/1 latency with a saturation penalty).

    Unlike :func:`queueing.link_latency_ms` this never returns ``inf`` so that
    Dijkstra can always make progress; overloaded edges are heavily penalised instead.
    """
    cap = snapshot.capacity_gbps
    rho = np.divide(load_gbps, cap, out=np.zeros_like(load_gbps), where=cap > 0)
    rho_c = np.clip(rho, 0.0, 0.999)
    service_ms = qcfg.packet_bits / (cap * 1e9) * 1000.0
    cost = snapshot.prop_delay_ms + qcfg.queue_delay_scale * service_ms / (1.0 - rho_c)
    cost = cost + np.where(rho >= 1.0, _OVERLOAD_PENALTY_MS, 0.0)
    return cost


def _destinations(demands: list[Demand]) -> list[int]:
    return sorted({d.dst_node for d in demands})


def _route_with_weights(
    snapshot: NetworkSnapshot,
    demands: list[Demand],
    edge_weight: np.ndarray,
    qcfg: QueueConfig,
    cost_to_go_cache: dict[int, np.ndarray] | None = None,
) -> list[tuple[Demand, list[int]]]:
    """Trace every demand's path under fixed ``edge_weight`` (one Dijkstra per dest)."""
    cache = cost_to_go_cache if cost_to_go_cache is not None else {}
    for dest in _destinations(demands):
        if dest not in cache:
            cache[dest] = reverse_dijkstra(snapshot, dest, edge_weight)
    out: list[tuple[Demand, list[int]]] = []
    for demand in demands:
        ctg = cache[demand.dst_node]
        path = trace_path(snapshot, demand.src_node, demand.dst_node, ctg, edge_weight, qcfg.max_hops)
        out.append((demand, path))
    return out


def route_shortest_path(
    snapshot: NetworkSnapshot, demands: list[Demand], qcfg: QueueConfig
) -> list[tuple[Demand, list[int]]]:
    """Congestion-oblivious shortest (propagation-delay) routing."""
    weight = snapshot.prop_delay_ms.copy()
    return _route_with_weights(snapshot, demands, weight, qcfg)


def route_ca_global(
    snapshot: NetworkSnapshot,
    demands: list[Demand],
    qcfg: QueueConfig,
    iters: int = 20,
    return_labels: bool = False,
):
    """Iterative congestion-aware routing with global state (MSA on link loads).

    Returns the routing (list of ``(demand, path)``). When ``return_labels`` is True
    also returns ``(edge_weight, cost_to_go)`` where ``cost_to_go`` maps each
    destination node to its converged ``(num_nodes,)`` cost-to-go vector, used as the
    supervised target for GLIDER.
    """
    from .queueing import aggregate_loads

    load = np.zeros(snapshot.num_edges)
    edge_weight = routing_edge_cost(snapshot, load, qcfg)
    demand_paths: list[tuple[Demand, list[int]]] = []
    cost_to_go: dict[int, np.ndarray] = {}

    for k in range(iters):
        edge_weight = routing_edge_cost(snapshot, load, qcfg)
        cost_to_go = {}
        demand_paths = _route_with_weights(snapshot, demands, edge_weight, qcfg, cost_to_go)
        new_load = aggregate_loads(snapshot, demand_paths)
        # Method of successive averages: damp the load update for stable convergence.
        step = 1.0 / (k + 2.0)
        load = load + step * (new_load - load)

    # Final routing on the converged loads.
    edge_weight = routing_edge_cost(snapshot, load, qcfg)
    cost_to_go = {}
    demand_paths = _route_with_weights(snapshot, demands, edge_weight, qcfg, cost_to_go)

    if return_labels:
        return demand_paths, edge_weight, cost_to_go
    return demand_paths
