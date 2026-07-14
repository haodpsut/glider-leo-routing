import numpy as np

from glider.constellation import Constellation
from glider.network import NetworkConfig, build_snapshot
from glider.scenarios import PRESETS, default_ground_stations


def _con():
    return Constellation(PRESETS["tiny"], default_ground_stations(6))


def test_snapshot_builds_and_delays_positive():
    con = _con()
    snap = build_snapshot(con, NetworkConfig(), epoch_s=0.0)
    assert snap.num_edges > 0
    assert np.all(snap.prop_delay_ms > 0)
    assert np.all(snap.capacity_gbps > 0)
    assert snap.num_nodes == PRESETS["tiny"].num_sats + 6


def test_edges_are_bidirectional():
    con = _con()
    snap = build_snapshot(con, NetworkConfig(), epoch_s=0.0)
    pairs = set(zip(snap.edge_index[0].tolist(), snap.edge_index[1].tolist()))
    for u, v in pairs:
        assert (v, u) in pairs


def test_in_out_adjacency_consistent():
    con = _con()
    snap = build_snapshot(con, NetworkConfig(), epoch_s=0.0)
    total_out = sum(len(x) for x in snap.out_neighbors)
    total_in = sum(len(x) for x in snap.in_neighbors)
    assert total_out == total_in == snap.num_edges


def test_failures_reduce_edges():
    con = _con()
    rng = np.random.default_rng(0)
    full = build_snapshot(con, NetworkConfig(), 0.0, rng=rng)
    failed = build_snapshot(con, NetworkConfig(), 0.0, rng=rng, isl_failure_frac=0.5)
    assert failed.num_edges < full.num_edges
