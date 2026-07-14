"""The guarantee that the whole redesign rests on: deflection always delivers.

Greedy on an unrestricted learned cost-to-go must be right at every hop, so delivery
decays like p^H and no amount of training saves it. Restricting every choice to
neighbours that strictly decrease the shortest-path potential D(.,d) makes the walk
monotone, hence loop free and finite, so the packet arrives regardless of what the
ranking says. These tests pin that property down: it must survive any refactor,
because without it the method is unsound.
"""

import numpy as np

from glider.baselines import (
    route_deflect,
    route_deflect_oracle,
    route_shortest_path,
    sp_potential,
    trace_deflect,
)
from glider.dataset import ScenarioConfig, sample_scenario
from glider.queueing import evaluate_routing, paths_to_edge_ids


def _scenario(seed=0, preset="medium", base=450.0):
    cfg = ScenarioConfig(
        presets=[preset], n_ground_stations=12, ca_iters=10, deflect_iters=8,
        traffic_base_gbps=base, load_min=1.0, load_max=1.0,
        failure_min=0.0, failure_max=0.0,
    )
    cfg.net.isl_capacity_gbps = 20.0
    cfg.net.gsl_capacity_gbps = 80.0
    cfg.queue.queue_delay_scale = 50.0
    return cfg, sample_scenario(cfg, np.random.default_rng(seed))


def test_potential_strictly_decreases_along_path():
    cfg, sc = _scenario()
    snap = sc.snapshot
    rng = np.random.default_rng(1)
    for demand in sc.demands[:10]:
        d = demand.dst_node
        pot = sp_potential(snap, d)
        # An adversarial ranking: random scores. Delivery must NOT depend on it.
        score = rng.uniform(0, 100, size=snap.num_nodes)
        score[d] = 0.0
        edge_cost = rng.uniform(1, 10, size=snap.num_edges)
        path = trace_deflect(snap, demand.src_node, d, pot, score, edge_cost, 256)
        assert path is not None, "deflection failed to deliver under a random ranking"
        assert path[0] == demand.src_node and path[-1] == d
        # The anchor potential must strictly decrease at every hop.
        pots = [pot[n] for n in path]
        assert all(a > b for a, b in zip(pots[:-1], pots[1:])), "potential did not decrease"
        # Strict decrease implies no repeated node.
        assert len(set(path)) == len(path), "path revisited a node"


def test_delivery_is_ranking_independent():
    """Every demand gets a path no matter how bad the learned scores are."""
    cfg, sc = _scenario(seed=3)
    snap = sc.snapshot
    rng = np.random.default_rng(7)
    dests = sorted({d.dst_node for d in sc.demands})
    # Deliberately awful scores (uniform noise, no relation to distance).
    bad = {d: rng.uniform(0, 1000, size=snap.num_nodes) for d in dests}
    for d in dests:
        bad[d][d] = 0.0
    paths = route_deflect(snap, sc.demands, sc.qcfg, bad, np.zeros(snap.num_edges))
    assert len(paths) == len(sc.demands)
    for demand, path in paths:
        assert path is not None, "a demand was dropped despite the progress guarantee"
        assert path[-1] == demand.dst_node
        assert paths_to_edge_ids(snap, path), "path does not resolve to real edges"


def test_deflect_oracle_beats_shortest_path_under_congestion():
    """The ceiling of the restricted action space must actually be worth chasing."""
    sp_c, df_c, sp_u, df_u = [], [], [], []
    for seed in range(3):
        cfg, sc = _scenario(seed=seed)
        sp = evaluate_routing(sc.snapshot, route_shortest_path(sc.snapshot, sc.demands, sc.qcfg), sc.qcfg)
        df = evaluate_routing(sc.snapshot, route_deflect_oracle(sc.snapshot, sc.demands, sc.qcfg, iters=10), sc.qcfg)
        sp_c.append(sp.carried_fraction); df_c.append(df.carried_fraction)
        sp_u.append(sp.max_utilization); df_u.append(df.max_utilization)
    # Congestion-aware deflection carries more demand and lowers peak utilisation.
    assert np.mean(df_c) > np.mean(sp_c) + 0.03
    assert np.mean(df_u) <= np.mean(sp_u) + 1e-6


def test_deflection_does_not_inflate_path_length():
    """Load balancing must come from choosing among near-equal paths, not detours."""
    cfg, sc = _scenario(seed=5)
    sp = evaluate_routing(sc.snapshot, route_shortest_path(sc.snapshot, sc.demands, sc.qcfg), sc.qcfg)
    df = evaluate_routing(sc.snapshot, route_deflect_oracle(sc.snapshot, sc.demands, sc.qcfg, iters=10), sc.qcfg)
    assert df.mean_path_hops <= sp.mean_path_hops * 1.15
