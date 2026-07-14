import numpy as np

from glider.constellation import EARTH_RADIUS_KM, Constellation
from glider.scenarios import PRESETS, default_ground_stations


def test_positions_on_orbit_sphere():
    spec = PRESETS["tiny"]
    con = Constellation(spec)
    pos = con.sat_positions_eci(0.0)
    r = np.linalg.norm(pos, axis=1)
    assert pos.shape == (spec.num_sats, 3)
    # All satellites sit on the orbit-radius sphere at every epoch.
    assert np.allclose(r, spec.orbit_radius_km, rtol=1e-6)


def test_positions_move_over_time():
    con = Constellation(PRESETS["tiny"])
    p0 = con.sat_positions_eci(0.0)
    p1 = con.sat_positions_eci(600.0)
    assert not np.allclose(p0, p1)


def test_orbital_period_reasonable():
    # A 550 km LEO orbit has a period near 95 minutes.
    period_min = PRESETS["starlink_shell1"].period_s / 60.0
    assert 90.0 < period_min < 100.0


def test_isl_edges_degree():
    spec = PRESETS["tiny"]
    con = Constellation(spec)
    edges = con.isl_edges()
    # +Grid on a torus: one intra-plane ring + one inter-plane ring -> 2N undirected
    # edges, giving each satellite degree 4 (prev/next slot, prev/next plane).
    assert edges.shape[1] == 2
    assert len(edges) == 2 * spec.num_sats
    assert edges[:, 0].min() >= 0 and edges[:, 1].max() < spec.num_sats
    deg = np.bincount(edges.reshape(-1), minlength=spec.num_sats)
    assert np.all(deg == 4)


def test_ground_station_altitude():
    con = Constellation(PRESETS["tiny"], default_ground_stations(6))
    gs = con.gs_positions_eci(0.0)
    r = np.linalg.norm(gs, axis=1)
    assert np.allclose(r, EARTH_RADIUS_KM, rtol=1e-6)


def test_visible_gsl_nonempty():
    con = Constellation(PRESETS["medium"], default_ground_stations(8))
    links = con.visible_gsl(0.0, min_elevation_deg=25.0)
    assert len(links) > 0
    for gi, si, dist in links:
        assert dist > 0
