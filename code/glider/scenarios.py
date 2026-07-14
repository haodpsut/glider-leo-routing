"""Constellation presets and ground-station sets used across experiments.

Presets are approximations of published shells (parameters scaled down where noted
in configs) and are only used to define geometry; nothing here depends on
proprietary data. The ``tiny`` preset exists for smoke tests.
"""

from __future__ import annotations

import numpy as np

from .constellation import ConstellationSpec, GroundStation

# A spread of major population centres (name, lat, lon). Weights (a population proxy)
# drive the gravity traffic model.
_CITIES = [
    ("new_york", 40.71, -74.01, 8.4),
    ("los_angeles", 34.05, -118.24, 4.0),
    ("london", 51.51, -0.13, 9.0),
    ("paris", 48.86, 2.35, 2.1),
    ("frankfurt", 50.11, 8.68, 0.75),
    ("moscow", 55.75, 37.62, 12.5),
    ("dubai", 25.20, 55.27, 3.3),
    ("mumbai", 19.08, 72.88, 20.4),
    ("singapore", 1.35, 103.82, 5.9),
    ("tokyo", 35.68, 139.69, 13.9),
    ("sydney", -33.87, 151.21, 5.3),
    ("sao_paulo", -23.55, -46.63, 12.3),
    ("johannesburg", -26.20, 28.05, 5.6),
    ("cairo", 30.04, 31.24, 9.5),
    ("sao_francisco", 37.77, -122.42, 0.87),
    ("seoul", 37.57, 126.98, 9.7),
]


def default_ground_stations(n: int | None = None) -> list[GroundStation]:
    cities = _CITIES if n is None else _CITIES[:n]
    return [GroundStation(name=c[0], lat_deg=c[1], lon_deg=c[2]) for c in cities]


def ground_station_weights(n: int | None = None) -> np.ndarray:
    cities = _CITIES if n is None else _CITIES[:n]
    return np.array([c[3] for c in cities], dtype=np.float64)


# --- Constellation presets ---------------------------------------------------
PRESETS: dict[str, ConstellationSpec] = {
    # Starlink-like shell 1 (72 planes x 22 sats, 53 deg, 550 km).
    "starlink_shell1": ConstellationSpec(
        name="starlink_shell1", altitude_km=550.0, inclination_deg=53.0,
        num_planes=72, sats_per_plane=22, phasing_f=1, isl_polar_cutoff_deg=70.0,
    ),
    # Kuiper-like shell (34 planes x 34 sats, 51.9 deg, 630 km) — different geometry
    # for cross-constellation generalisation tests.
    "kuiper_shell": ConstellationSpec(
        name="kuiper_shell", altitude_km=630.0, inclination_deg=51.9,
        num_planes=34, sats_per_plane=34, phasing_f=1, isl_polar_cutoff_deg=70.0,
    ),
    # Telesat-like polar-ish shell for a third geometry.
    "telesat_polar": ConstellationSpec(
        name="telesat_polar", altitude_km=1015.0, inclination_deg=98.98,
        num_planes=27, sats_per_plane=13, phasing_f=1, isl_polar_cutoff_deg=None,
    ),
    # Small training constellation (keeps oracle labelling fast).
    "medium": ConstellationSpec(
        name="medium", altitude_km=550.0, inclination_deg=53.0,
        num_planes=24, sats_per_plane=16, phasing_f=1, isl_polar_cutoff_deg=70.0,
    ),
    # Tiny constellation for smoke tests (8x8 keeps ground-station coverage complete
    # at a low elevation mask while staying fast to label).
    "tiny": ConstellationSpec(
        name="tiny", altitude_km=550.0, inclination_deg=53.0,
        num_planes=8, sats_per_plane=8, phasing_f=1, isl_polar_cutoff_deg=70.0,
    ),
}
