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
from .queueing import INF, QueueConfig, reverse_dijkstra, trace_path
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


# --------------------------------------------------------------------------------
# Shortest-path-anchored deflection.
#
# Replacing routing wholesale with a greedy walk on a *learned* cost-to-go fails for
# a structural reason: greedy must pick the right neighbour at EVERY hop, so with
# per-hop accuracy p the delivery rate over an H-hop path decays like p^H (p=0.9,
# H=8 delivers only 43%). No amount of training reaches the ~99.4% per-hop accuracy
# that a 95% delivery rate over 8 hops demands.
#
# We therefore anchor on the shortest-path potential D(.,d) (plain reverse Dijkstra
# on propagation delay, exactly what a link-state protocol already computes) and
# restrict every forwarding decision to neighbours that make progress under it:
#
#     A(u,d) = { v in N(u) : D(v,d) < D(u,d) }
#
# D strictly decreases along any such walk, so the walk cannot loop and must reach d
# in finitely many hops: DELIVERY IS GUARANTEED BY CONSTRUCTION, whatever the policy
# does inside A(u,d). The learned part only chooses *which* progressing neighbour to
# take, which is where congestion-awareness lives (a +Grid offers many equal-cost
# shortest paths, so there is a lot of load-balancing freedom without any stretch).
# A policy in this class can never do worse than shortest path by dropping traffic.
# --------------------------------------------------------------------------------


def sp_potential(snapshot: NetworkSnapshot, dest: int) -> np.ndarray:
    """Shortest-path cost-to-go on propagation delay: the deflection anchor D(.,d)."""
    return reverse_dijkstra(snapshot, dest, snapshot.prop_delay_ms)


def trace_deflect(
    snapshot: NetworkSnapshot,
    src: int,
    dest: int,
    potential: np.ndarray,     # D(.,d), the progress anchor
    score: np.ndarray,         # per-node cost-to-go used to rank progressing neighbours
    edge_cost: np.ndarray,     # observed per-edge cost (propagation + queueing)
    max_hops: int,
) -> list[int] | None:
    """Route src->dest choosing, among progressing neighbours, the cheapest by
    ``edge_cost + score``.

    Returns None only if ``dest`` is unreachable from ``src`` (D infinite). Otherwise
    delivery is guaranteed, because every hop strictly decreases the potential.
    """
    if src == dest:
        return [src]
    if not np.isfinite(potential[src]):
        return None
    path = [src]
    node = src
    for _ in range(max_hops):
        best_node, best_val = -1, INF
        d_node = potential[node]
        for v, e in snapshot.out_neighbors[node]:
            if not np.isfinite(potential[v]) or potential[v] >= d_node:
                continue  # not a progressing neighbour
            val = edge_cost[e] + score[v]
            if val < best_val:
                best_val, best_node = val, v
        if best_node < 0:
            return None  # unreachable (cannot happen when potential[node] is finite)
        path.append(best_node)
        if best_node == dest:
            return path
        node = best_node
    return None


def route_deflect(
    snapshot: NetworkSnapshot,
    demands: list[Demand],
    qcfg: QueueConfig,
    score_by_dest: dict[int, np.ndarray],
    load_gbps: np.ndarray,
) -> list[tuple[Demand, list[int]]]:
    """Deflection routing given a per-destination score and an observed load."""
    edge_cost = routing_edge_cost(snapshot, load_gbps, qcfg)
    out: list[tuple[Demand, list[int]]] = []
    pot_cache: dict[int, np.ndarray] = {}
    for demand in demands:
        d = demand.dst_node
        if d not in pot_cache:
            pot_cache[d] = sp_potential(snapshot, d)
        path = trace_deflect(
            snapshot, demand.src_node, d, pot_cache[d], score_by_dest[d], edge_cost, qcfg.max_hops
        )
        out.append((demand, path))
    return out


def route_deflect_local(
    snapshot: NetworkSnapshot,
    demands: list[Demand],
    qcfg: QueueConfig,
    load_gbps: np.ndarray,
) -> list[tuple[Demand, list[int]]]:
    """Myopic congestion-greedy deflection: no learning at all.

    Ranks progressing neighbours by ``observed_edge_cost + D(v,d)``, i.e. it avoids
    the locally congested link and otherwise trusts the shortest-path potential.

    This is the baseline that decides whether a learned ranker is worth anything. On
    a small constellation it already captures essentially the entire deflection
    ceiling, so learning buys nothing there. Its weakness is myopia: it sees only the
    next link's queue, so on a mega-constellation with long paths it walks into
    congestion further downstream, and on some shells it is even WORSE than plain
    shortest path. Anticipating downstream congestion is precisely what a learned
    cost-to-go is for, and beating this baseline is the bar GLIDER must clear.
    """
    dests = _destinations(demands)
    pot = {d: sp_potential(snapshot, d) for d in dests}
    return route_deflect(snapshot, demands, qcfg, pot, load_gbps)


def route_deflect_oracle(
    snapshot: NetworkSnapshot,
    demands: list[Demand],
    qcfg: QueueConfig,
    iters: int = 12,
    return_labels: bool = False,
):
    """Best achievable deflection policy: ranks progressing neighbours with the
    converged CA-Global cost-to-go and full knowledge of the load.

    This is the CEILING of the deflection action space. A learned policy in this
    class cannot beat it, so if this does not beat shortest path there is nothing to
    learn and the whole approach should be abandoned.
    """
    from .queueing import aggregate_loads

    load = np.zeros(snapshot.num_edges)
    paths: list[tuple[Demand, list[int]]] = []
    ctg: dict[int, np.ndarray] = {}
    for k in range(iters):
        weight = routing_edge_cost(snapshot, load, qcfg)
        ctg = {d: reverse_dijkstra(snapshot, d, weight) for d in _destinations(demands)}
        paths = route_deflect(snapshot, demands, qcfg, ctg, load)
        new_load = aggregate_loads(snapshot, paths)
        load = load + (1.0 / (k + 2.0)) * (new_load - load)  # MSA damping

    weight = routing_edge_cost(snapshot, load, qcfg)
    ctg = {d: reverse_dijkstra(snapshot, d, weight) for d in _destinations(demands)}
    paths = route_deflect(snapshot, demands, qcfg, ctg, load)
    if return_labels:
        return paths, ctg, load
    return paths


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
