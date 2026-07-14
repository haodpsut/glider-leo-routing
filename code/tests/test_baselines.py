import numpy as np

from glider.baselines import route_ca_global, route_shortest_path
from glider.constellation import Constellation
from glider.network import NetworkConfig, build_snapshot
from glider.queueing import QueueConfig, evaluate_routing
from glider.scenarios import PRESETS, default_ground_stations, ground_station_weights
from glider.traffic import TrafficConfig, make_gravity_demands


_NET = NetworkConfig(isl_capacity_gbps=20.0, gsl_capacity_gbps=80.0)


def _heavy_scenario(seed: int = 3, epoch: float = 1200.0):
    spec = PRESETS["medium"]
    con = Constellation(spec, default_ground_stations(12))
    rng = np.random.default_rng(seed)
    snap = build_snapshot(con, _NET, epoch_s=epoch, rng=rng)
    weights = ground_station_weights(12)
    # High load so congestion-oblivious routing is stressed (ISL bottleneck regime).
    tcfg = TrafficConfig(load_scale=1.4, base_total_gbps=220.0)
    demands = make_gravity_demands(spec.num_sats, weights, tcfg, rng)
    return snap, demands, QueueConfig(queue_delay_scale=50.0)


def test_shortest_path_delivers():
    snap, demands, qcfg = _heavy_scenario()
    paths = route_shortest_path(snap, demands, qcfg)
    m = evaluate_routing(snap, paths, qcfg)
    assert m.total_flows == len(demands)
    assert m.delivered_flows > 0


def test_ca_global_balances_load_better_than_sp():
    # Aggregate property across several instances: congestion-aware routing should,
    # on average, lower peak utilisation and deliver at least as much demand volume.
    sp_util, ca_util, sp_cf, ca_cf = [], [], [], []
    for seed in range(4):
        snap, demands, qcfg = _heavy_scenario(seed=seed, epoch=500.0 + 700.0 * seed)
        sp = evaluate_routing(snap, route_shortest_path(snap, demands, qcfg), qcfg)
        ca = evaluate_routing(snap, route_ca_global(snap, demands, qcfg, iters=25), qcfg)
        sp_util.append(sp.max_utilization); ca_util.append(ca.max_utilization)
        sp_cf.append(sp.carried_fraction); ca_cf.append(ca.carried_fraction)
    assert np.mean(ca_util) <= np.mean(sp_util) + 1e-6
    assert np.mean(ca_cf) >= np.mean(sp_cf) - 1e-6


def test_ca_global_labels_shapes():
    snap, demands, qcfg = _heavy_scenario()
    paths, weight, ctg = route_ca_global(snap, demands, qcfg, iters=10, return_labels=True)
    assert weight.shape[0] == snap.num_edges
    dests = {d.dst_node for d in demands}
    assert set(ctg.keys()) == dests
    for _d, costs in ctg.items():
        assert costs.shape[0] == snap.num_nodes
