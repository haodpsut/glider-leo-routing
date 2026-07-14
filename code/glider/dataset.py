"""Scenario sampling and supervised-target construction for GLIDER.

A :class:`Scenario` bundles everything needed to both featurise and evaluate a
single routing instance: constellation, snapshot at an epoch, demand set, and the
warm-up (shortest-path) load that a router would observe as its congestion signal.

Training pairs regress GLIDER's predicted cost-to-go onto the converged CA-Global
cost-to-go, using features built from the *observable* warm-up load. This teaches
the model to map observed congestion to load-balanced cost-to-go.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .baselines import route_ca_global, route_shortest_path, routing_edge_cost
from .constellation import Constellation, ConstellationSpec
from .features import SnapshotFeatures, build_features, geo_term
from .network import NetworkConfig, build_snapshot
from .queueing import QueueConfig, aggregate_loads
from .scenarios import PRESETS, default_ground_stations, ground_station_weights
from .traffic import TrafficConfig, make_gravity_demands


@dataclass
class ScenarioConfig:
    presets: list[str] = field(default_factory=lambda: ["medium"])
    n_ground_stations: int = 12
    epoch_max_s: float = 5600.0        # ~ one orbital period
    load_min: float = 0.6
    load_max: float = 1.6
    failure_min: float = 0.0
    failure_max: float = 0.10
    ca_iters: int = 20
    net: NetworkConfig = field(default_factory=NetworkConfig)
    queue: QueueConfig = field(default_factory=QueueConfig)
    traffic_base_gbps: float = 200.0


@dataclass
class Scenario:
    constellation: Constellation
    snapshot: object
    demands: list
    warmup_load: np.ndarray
    qcfg: QueueConfig


def sample_scenario(cfg: ScenarioConfig, rng: np.random.Generator) -> Scenario:
    """Draw a random routing instance (constellation, epoch, traffic, failures)."""
    preset_name = cfg.presets[int(rng.integers(len(cfg.presets)))]
    spec: ConstellationSpec = PRESETS[preset_name]
    gss = default_ground_stations(cfg.n_ground_stations)
    constellation = Constellation(spec, gss)

    epoch = float(rng.uniform(0.0, cfg.epoch_max_s))
    fail = float(rng.uniform(cfg.failure_min, cfg.failure_max))
    snapshot = build_snapshot(
        constellation, cfg.net, epoch, rng=rng, isl_failure_frac=fail
    )

    weights = ground_station_weights(cfg.n_ground_stations)
    tcfg = TrafficConfig(
        load_scale=float(rng.uniform(cfg.load_min, cfg.load_max)),
        base_total_gbps=cfg.traffic_base_gbps,
        skew=1.0,
    )
    demands = make_gravity_demands(spec.num_sats, weights, tcfg, rng)

    sp_paths = route_shortest_path(snapshot, demands, cfg.queue)
    warmup_load = aggregate_loads(snapshot, sp_paths)
    return Scenario(constellation, snapshot, demands, warmup_load, cfg.queue)


MAX_DEGREE = 12  # padding width for next-hop neighbour lists


@dataclass
class TrainingSample:
    feats: SnapshotFeatures
    # Regression pairs: (node, dest) -> CA-Global cost-to-go.
    src_idx: np.ndarray      # (P,)
    dst_idx: np.ndarray      # (P,)
    geo: np.ndarray          # (P, 1)
    target_q: np.ndarray     # (P,)
    # Next-hop imitation: at (node, dest), rank neighbours by obs_c + Q(nbr, dest).
    nh_dst: np.ndarray       # (M,)
    nh_nbr: np.ndarray       # (M, K) neighbour node ids (padded)
    nh_geo: np.ndarray       # (M, K) geo(nbr, dest)
    nh_obs_c: np.ndarray     # (M, K) observed edge cost u->nbr (padded large)
    nh_mask: np.ndarray      # (M, K) bool, valid neighbour
    nh_target: np.ndarray    # (M,) index (in K) of the CA-Global next hop


def training_pairs(
    scenario: Scenario,
    cfg: ScenarioConfig,
    rng: np.random.Generator,
    max_pairs_per_dest: int = 64,
) -> TrainingSample:
    """Build cost-to-go regression pairs and next-hop imitation targets."""
    snap = scenario.snapshot
    _paths, ca_weight, ctg = route_ca_global(
        snap, scenario.demands, cfg.queue, iters=cfg.ca_iters, return_labels=True
    )
    feats = build_features(snap, scenario.warmup_load)
    obs_cost = routing_edge_cost(snap, scenario.warmup_load, cfg.queue)

    src_list: list[int] = []
    dst_list: list[int] = []
    tgt_list: list[float] = []
    nh_dst: list[int] = []
    nh_nbr: list[list[int]] = []
    nh_obs: list[list[float]] = []
    nh_mask: list[list[bool]] = []
    nh_tgt: list[int] = []
    K = MAX_DEGREE
    BIG = 1.0e6

    for dest, costs in ctg.items():
        finite = np.where(np.isfinite(costs))[0]
        finite = finite[finite != dest]
        if len(finite) == 0:
            continue
        if len(finite) > max_pairs_per_dest:
            finite = rng.choice(finite, size=max_pairs_per_dest, replace=False)
        for v in finite:
            src_list.append(int(v))
            dst_list.append(int(dest))
            tgt_list.append(float(costs[v]))

            # Next-hop target: neighbour minimising ca_edge_cost + ca_cost_to_go.
            nbrs = snap.out_neighbors[v]
            best_w, best_val, best_j = -1, np.inf, -1
            row_nbr, row_obs, row_mask = [], [], []
            for j, (w, e) in enumerate(nbrs[:K]):
                cg = costs[w]
                val = ca_weight[e] + cg
                row_nbr.append(w)
                row_obs.append(float(obs_cost[e]))
                row_mask.append(np.isfinite(cg))
                if np.isfinite(cg) and val < best_val:
                    best_val, best_w, best_j = val, w, j
            if best_j < 0:
                continue
            # Pad to K.
            pad = K - len(row_nbr)
            row_nbr += [0] * pad
            row_obs += [BIG] * pad
            row_mask += [False] * pad
            nh_dst.append(int(dest))
            nh_nbr.append(row_nbr)
            nh_obs.append(row_obs)
            nh_mask.append(row_mask)
            nh_tgt.append(best_j)

    src_idx = np.array(src_list, dtype=np.int64)
    dst_idx = np.array(dst_list, dtype=np.int64)
    geo = geo_term(feats.node_pos, src_idx, dst_idx) if len(src_idx) else np.zeros((0, 1), np.float32)
    target_q = np.array(tgt_list, dtype=np.float32)

    nh_dst_a = np.array(nh_dst, dtype=np.int64)
    nh_nbr_a = np.array(nh_nbr, dtype=np.int64) if nh_nbr else np.zeros((0, K), np.int64)
    nh_obs_a = np.array(nh_obs, dtype=np.float32) if nh_obs else np.zeros((0, K), np.float32)
    nh_mask_a = np.array(nh_mask, dtype=bool) if nh_mask else np.zeros((0, K), bool)
    nh_tgt_a = np.array(nh_tgt, dtype=np.int64)
    # geo(nbr, dest) for every padded neighbour.
    if len(nh_dst_a):
        flat_nbr = nh_nbr_a.reshape(-1)
        flat_dst = np.repeat(nh_dst_a, K)
        nh_geo_a = geo_term(feats.node_pos, flat_nbr, flat_dst).reshape(-1, K)
    else:
        nh_geo_a = np.zeros((0, K), np.float32)

    return TrainingSample(
        feats, src_idx, dst_idx, geo, target_q,
        nh_dst_a, nh_nbr_a, nh_geo_a, nh_obs_a, nh_mask_a, nh_tgt_a,
    )
