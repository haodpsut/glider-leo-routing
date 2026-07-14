"""Feature extraction turning a network snapshot (+ current load) into GNN tensors.

All features are constellation-size agnostic and expressed in normalised units so
that a model trained on one constellation can be applied to another without
retraining. Node and edge features are destination-agnostic; destination
conditioning happens in the readout head via node embeddings and a geometric term.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .network import NetworkSnapshot

# Normalisation scales (km, ms, Gbps). Fixed constants keep features comparable
# across constellations of different size/altitude.
POS_SCALE_KM = 10000.0
DELAY_SCALE_MS = 50.0
GEO_SCALE_KM = 10000.0

NODE_FEAT_DIM = 11  # 3 pos + 2 type + out-util + out-deg + 4 grid sin/cos
EDGE_FEAT_DIM = 3   # norm prop delay + utilisation + norm capacity


@dataclass
class SnapshotFeatures:
    node_feat: np.ndarray     # (N, NODE_FEAT_DIM)
    edge_index: np.ndarray    # (2, E)
    edge_feat: np.ndarray     # (E, EDGE_FEAT_DIM)
    node_pos: np.ndarray      # (N, 3) normalised ECI, for geometric readout term


def build_features(snapshot: NetworkSnapshot, load_gbps: np.ndarray) -> SnapshotFeatures:
    """Construct normalised node/edge feature tensors for one snapshot.

    Args:
        load_gbps: (num_edges,) current per-edge load used for utilisation features.
            Pass zeros for a load-agnostic embedding.
    """
    n = snapshot.num_nodes
    pos = snapshot.node_positions() / POS_SCALE_KM

    cap = snapshot.capacity_gbps
    util = np.divide(load_gbps, cap, out=np.zeros_like(load_gbps), where=cap > 0)
    cap_norm = cap / (cap.max() if snapshot.num_edges and cap.max() > 0 else 1.0)

    # Aggregate per-node signals from outgoing edges.
    mean_out_util = np.zeros(n)
    out_deg = np.zeros(n)
    for u in range(n):
        edges = snapshot.out_neighbors[u]
        out_deg[u] = len(edges)
        if edges:
            mean_out_util[u] = np.mean([util[e] for _v, e in edges])
    max_deg = out_deg.max() if out_deg.max() > 0 else 1.0

    is_sat = np.zeros(n)
    is_sat[: snapshot.num_sats] = 1.0
    is_gs = 1.0 - is_sat

    # Size-agnostic structural coordinates: sin/cos of fractional plane and slot
    # (wrap-aware). Ground stations get zeros. These make +Grid routing direction
    # learnable without leaking the absolute constellation size.
    grid = np.zeros((n, 4), dtype=np.float64)
    if snapshot.num_sats and snapshot.sat_grid_frac is not None:
        pf = 2.0 * np.pi * snapshot.sat_grid_frac[:, 0]
        sf = 2.0 * np.pi * snapshot.sat_grid_frac[:, 1]
        grid[: snapshot.num_sats, 0] = np.sin(pf)
        grid[: snapshot.num_sats, 1] = np.cos(pf)
        grid[: snapshot.num_sats, 2] = np.sin(sf)
        grid[: snapshot.num_sats, 3] = np.cos(sf)

    node_feat = np.concatenate(
        [
            pos,                                   # 3
            is_sat[:, None],                       # 1
            is_gs[:, None],                        # 1
            mean_out_util[:, None],                # 1
            (out_deg / max_deg)[:, None],          # 1
            grid,                                  # 4
        ],
        axis=1,
    ).astype(np.float32)

    edge_feat = np.stack(
        [
            snapshot.prop_delay_ms / DELAY_SCALE_MS,
            util,
            cap_norm,
        ],
        axis=1,
    ).astype(np.float32)

    return SnapshotFeatures(
        node_feat=node_feat,
        edge_index=snapshot.edge_index.astype(np.int64),
        edge_feat=edge_feat,
        node_pos=pos.astype(np.float32),
    )


def geo_term(node_pos: np.ndarray, src_idx: np.ndarray, dst_idx: np.ndarray) -> np.ndarray:
    """Normalised straight-line distance between node pairs, shape (len, 1).

    Used as an explicit geometric cue in the destination-conditioned readout so the
    model does not have to rediscover Euclidean progress from scratch.
    """
    d = np.linalg.norm(node_pos[src_idx] - node_pos[dst_idx], axis=1) * POS_SCALE_KM / GEO_SCALE_KM
    return d.astype(np.float32)[:, None]
