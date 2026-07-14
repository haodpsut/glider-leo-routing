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

from .baselines import (
    route_ca_global,
    route_deflect_oracle,
    route_shortest_path,
    routing_edge_cost,
    sp_potential,
)
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
    # MSA iterations for DEFLECT-ORACLE, the teacher and the deflection ceiling. 8 is
    # enough for the load to converge and keeps per-step label generation (the CPU
    # bottleneck when training on a mega-constellation) affordable.
    deflect_iters: int = 8
    net: NetworkConfig = field(default_factory=NetworkConfig)
    queue: QueueConfig = field(default_factory=QueueConfig)
    # Aggregate ground-to-ground demand (Gbps) at load_scale == 1.0.
    #
    # This value decides whether the routing problem exists at all, so it is
    # calibrated, not guessed. Demand flows between a fixed set of ground stations,
    # so the bottleneck sits in the mesh around the ground-station access points and
    # does NOT scale with satellite count. Measured SP -> CA-Global carried-demand
    # gap at load_scale 1.0 (three scenarios each):
    #     base=300 : medium +6pp,  starlink  0pp,  kuiper  0pp   <- too light, SP is
    #                already near-optimal and there is nothing to win
    #     base=450 : medium +15pp, starlink +20pp, kuiper +14pp  <- used here
    #     base=550 : medium +7pp   <- so heavy that even CA-Global collapses
    # Setting this too low is what makes shortest path look unbeatable.
    traffic_base_gbps: float = 450.0


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
    """Build cost-to-go regression pairs and next-hop imitation targets.

    Supervision comes from DEFLECT-ORACLE (the best policy inside the same
    progress-restricted action space the learner will act in), not from unrestricted
    CA-Global. Imitating a teacher that can take actions the student cannot is a
    mismatch; here teacher and student share an action space, so the student can in
    principle reach the teacher.

    Next-hop candidates are restricted to *progressing* neighbours
    (D(v,d) < D(u,d)), matching inference exactly.
    """
    snap = scenario.snapshot
    _paths, ctg, oracle_load = route_deflect_oracle(
        snap, scenario.demands, cfg.queue, iters=cfg.deflect_iters, return_labels=True
    )
    oracle_cost = routing_edge_cost(snap, oracle_load, cfg.queue)
    feats = build_features(snap, scenario.warmup_load)
    obs_cost = routing_edge_cost(snap, scenario.warmup_load, cfg.queue)
    pot_cache: dict[int, np.ndarray] = {}

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
        if dest not in pot_cache:
            pot_cache[dest] = sp_potential(snap, dest)
        pot = pot_cache[dest]

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

            # Candidates = progressing neighbours only, exactly as at inference.
            # Target = the one DEFLECT-ORACLE takes (min oracle_cost + oracle ctg).
            best_val, best_j = np.inf, -1
            row_nbr, row_obs, row_mask = [], [], []
            d_v = pot[v]
            j = 0
            for w, e in snap.out_neighbors[v]:
                if j >= K:
                    break
                if not np.isfinite(pot[w]) or pot[w] >= d_v:
                    continue  # not progressing
                cg = costs[w]
                if not np.isfinite(cg):
                    continue
                row_nbr.append(w)
                row_obs.append(float(obs_cost[e]))
                row_mask.append(True)
                val = oracle_cost[e] + cg
                if val < best_val:
                    best_val, best_j = val, j
                j += 1
            # A single candidate carries no decision, so it teaches nothing.
            if best_j < 0 or len(row_nbr) < 2:
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
