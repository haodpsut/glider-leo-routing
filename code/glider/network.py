"""Dynamic network snapshots built from a constellation at a given epoch.

A :class:`NetworkSnapshot` is an immutable directed graph over satellite and
ground-station nodes with per-edge propagation delay (ms) and capacity (Gbps).
Snapshots are the unit of routing: baselines and the learned policy both operate
on a single snapshot, and time evolution is handled by generating a sequence of
snapshots at successive epochs.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .constellation import Constellation, SPEED_OF_LIGHT_KM_S


@dataclass
class NetworkConfig:
    isl_capacity_gbps: float = 20.0
    gsl_capacity_gbps: float = 10.0
    min_elevation_deg: float = 25.0
    # Processing/transmission overhead added to every hop, in milliseconds.
    per_hop_overhead_ms: float = 0.0


class NetworkSnapshot:
    """Directed graph snapshot with propagation delay and capacity per edge.

    Nodes ``0..num_sats-1`` are satellites; nodes ``num_sats..num_sats+num_gs-1``
    are ground stations. Every undirected physical link contributes two directed
    edges so that per-direction load can be tracked independently.
    """

    def __init__(
        self,
        num_sats: int,
        num_gs: int,
        edge_index: np.ndarray,       # (2, E) directed
        prop_delay_ms: np.ndarray,    # (E,)
        capacity_gbps: np.ndarray,    # (E,)
        sat_pos_eci: np.ndarray,      # (num_sats, 3) km, for geometric features
        gs_pos_eci: np.ndarray,       # (num_gs, 3) km
        epoch_s: float,
        sat_grid_frac: np.ndarray | None = None,  # (num_sats, 2) fractional plane/slot
    ):
        self.num_sats = num_sats
        self.num_gs = num_gs
        self.num_nodes = num_sats + num_gs
        self.edge_index = edge_index
        self.prop_delay_ms = prop_delay_ms
        self.capacity_gbps = capacity_gbps
        self.sat_pos_eci = sat_pos_eci
        self.gs_pos_eci = gs_pos_eci
        self.epoch_s = epoch_s
        self.sat_grid_frac = (
            sat_grid_frac if sat_grid_frac is not None else np.zeros((num_sats, 2))
        )
        self._build_adjacency()

    def _build_adjacency(self) -> None:
        """Build neighbour lists keyed by source (out) and by target (in) node.

        Each entry is a ``(neighbor_node, edge_id)`` tuple. The in-adjacency is used
        by reverse-Dijkstra cost-to-go computations.
        """
        self.out_neighbors: list[list[tuple[int, int]]] = [[] for _ in range(self.num_nodes)]
        self.in_neighbors: list[list[tuple[int, int]]] = [[] for _ in range(self.num_nodes)]
        for e in range(self.edge_index.shape[1]):
            u = int(self.edge_index[0, e])
            v = int(self.edge_index[1, e])
            self.out_neighbors[u].append((v, e))
            self.in_neighbors[v].append((u, e))

    def node_positions(self) -> np.ndarray:
        """Return (num_nodes, 3) ECI positions (sats then ground stations)."""
        if self.num_gs:
            return np.concatenate([self.sat_pos_eci, self.gs_pos_eci], axis=0)
        return self.sat_pos_eci

    @property
    def num_edges(self) -> int:
        return self.edge_index.shape[1]


def build_snapshot(
    constellation: Constellation,
    net_cfg: NetworkConfig,
    epoch_s: float,
    isl_edges: np.ndarray | None = None,
    rng: np.random.Generator | None = None,
    isl_failure_frac: float = 0.0,
) -> NetworkSnapshot:
    """Materialise a :class:`NetworkSnapshot` at ``epoch_s``.

    Args:
        isl_edges: Precomputed static +Grid edge list (avoids recomputation across
            epochs). If None it is derived from the constellation.
        isl_failure_frac: Fraction of active ISLs to fail (both directions) to model
            link churn / hardware faults. Requires ``rng`` when > 0.
    """
    spec = constellation.spec
    n_sats = spec.num_sats
    n_gs = len(constellation.ground_stations)
    if isl_edges is None:
        isl_edges = constellation.isl_edges()

    sat_pos = constellation.sat_positions_eci(epoch_s)
    gs_pos = constellation.gs_positions_eci(epoch_s)

    active = constellation.active_isl_mask(epoch_s, isl_edges)
    live = isl_edges[active]

    if isl_failure_frac > 0.0 and len(live) > 0:
        if rng is None:
            raise ValueError("isl_failure_frac > 0 requires an rng")
        n_fail = int(round(isl_failure_frac * len(live)))
        if n_fail > 0:
            fail_idx = rng.choice(len(live), size=n_fail, replace=False)
            keep = np.ones(len(live), dtype=bool)
            keep[fail_idx] = False
            live = live[keep]

    src_list: list[int] = []
    dst_list: list[int] = []
    delay_list: list[float] = []
    cap_list: list[float] = []

    overhead = net_cfg.per_hop_overhead_ms

    # Inter-satellite links (bidirectional).
    if len(live) > 0:
        d = np.linalg.norm(sat_pos[live[:, 0]] - sat_pos[live[:, 1]], axis=1)
        delay_ms = d / SPEED_OF_LIGHT_KM_S * 1000.0 + overhead
        for k in range(len(live)):
            u, v = int(live[k, 0]), int(live[k, 1])
            src_list += [u, v]
            dst_list += [v, u]
            delay_list += [float(delay_ms[k]), float(delay_ms[k])]
            cap_list += [net_cfg.isl_capacity_gbps, net_cfg.isl_capacity_gbps]

    # Ground-satellite links (bidirectional). Ground station g -> node index n_sats+g.
    for gi, si, dist in constellation.visible_gsl(epoch_s, net_cfg.min_elevation_deg):
        gnode = n_sats + gi
        delay_ms = dist / SPEED_OF_LIGHT_KM_S * 1000.0 + overhead
        src_list += [gnode, si]
        dst_list += [si, gnode]
        delay_list += [delay_ms, delay_ms]
        cap_list += [net_cfg.gsl_capacity_gbps, net_cfg.gsl_capacity_gbps]

    edge_index = np.array([src_list, dst_list], dtype=np.int64) if src_list else np.zeros((2, 0), np.int64)
    prop_delay = np.array(delay_list, dtype=np.float64)
    capacity = np.array(cap_list, dtype=np.float64)

    return NetworkSnapshot(
        num_sats=n_sats,
        num_gs=n_gs,
        edge_index=edge_index,
        prop_delay_ms=prop_delay,
        capacity_gbps=capacity,
        sat_pos_eci=sat_pos,
        gs_pos_eci=gs_pos,
        epoch_s=epoch_s,
        sat_grid_frac=constellation.sat_grid_fractions(),
    )
