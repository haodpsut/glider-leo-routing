import numpy as np

from glider.network import NetworkSnapshot
from glider.queueing import (
    QueueConfig,
    link_latency_ms,
    next_hop_from_costs,
    reverse_dijkstra,
    trace_path,
)


def _line_graph():
    """0 -> 1 -> 2 -> 3 directed chain with unit delays and shortcut 0->3 (delay 10)."""
    src = [0, 1, 2, 0]
    dst = [1, 2, 3, 3]
    edge_index = np.array([src, dst], dtype=np.int64)
    prop = np.array([1.0, 1.0, 1.0, 10.0])
    cap = np.array([10.0, 10.0, 10.0, 10.0])
    pos = np.zeros((4, 3))
    return NetworkSnapshot(4, 0, edge_index, prop, cap, pos, np.zeros((0, 3)), 0.0)


def test_reverse_dijkstra_distances():
    g = _line_graph()
    d = reverse_dijkstra(g, dest=3, edge_weight=g.prop_delay_ms)
    # 0->1->2->3 costs 3 (cheaper than the length-10 shortcut).
    assert d[3] == 0.0
    assert d[2] == 1.0
    assert d[1] == 2.0
    assert d[0] == 3.0


def test_next_hop_greedy_matches_shortest():
    g = _line_graph()
    d = reverse_dijkstra(g, dest=3, edge_weight=g.prop_delay_ms)
    nxt, _e = next_hop_from_costs(g, 0, d, g.prop_delay_ms)
    assert nxt == 1  # follows the chain, not the expensive shortcut


def test_trace_path():
    g = _line_graph()
    d = reverse_dijkstra(g, dest=3, edge_weight=g.prop_delay_ms)
    path = trace_path(g, 0, 3, d, g.prop_delay_ms, max_hops=16)
    assert path == [0, 1, 2, 3]


def test_link_latency_monotonic_and_overload():
    g = _line_graph()
    qcfg = QueueConfig(queue_delay_scale=1.0)
    low = link_latency_ms(g, np.array([1.0, 1.0, 1.0, 1.0]), qcfg)
    high = link_latency_ms(g, np.array([9.0, 9.0, 9.0, 9.0]), qcfg)
    assert np.all(high[:3] > low[:3])           # more load -> more delay
    over = link_latency_ms(g, np.array([10.0, 0.0, 0.0, 0.0]), qcfg)
    assert not np.isfinite(over[0])             # rho >= 1 -> infinite
