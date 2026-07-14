"""Walker-delta LEO constellation geometry and +Grid ISL topology.

We model satellites on circular orbits and propagate their positions analytically
(no SGP4 / perturbations) so that a full topology snapshot at any epoch is cheap to
compute. This matches the constellation model used by Hypatia (Kassing et al.,
IMC 2020) for the +Grid inter-satellite-link pattern, while remaining fully
self-contained and dependency-light.

All distances are in kilometres, times in seconds, angles internally in radians.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Physical constants
EARTH_RADIUS_KM = 6371.0
MU_EARTH = 398600.4418  # km^3 / s^2, standard gravitational parameter
SPEED_OF_LIGHT_KM_S = 299792.458


@dataclass
class ConstellationSpec:
    """Walker-delta constellation parameters (notation i: T/P/F).

    Attributes:
        name: Human-readable label (e.g. "starlink_shell1").
        altitude_km: Orbit altitude above the Earth surface.
        inclination_deg: Orbital inclination.
        num_planes: Number of orbital planes (P).
        sats_per_plane: Satellites per plane (S); total T = P * S.
        phasing_f: Walker phasing parameter F in [0, P-1].
        isl_polar_cutoff_deg: Latitude above which inter-plane ISLs are disabled
            (pointing/relative-motion constraint near the poles). None disables the
            cutoff and keeps inter-plane ISLs active everywhere.
    """

    name: str
    altitude_km: float
    inclination_deg: float
    num_planes: int
    sats_per_plane: int
    phasing_f: int = 1
    isl_polar_cutoff_deg: float | None = 70.0

    @property
    def num_sats(self) -> int:
        return self.num_planes * self.sats_per_plane

    @property
    def orbit_radius_km(self) -> float:
        return EARTH_RADIUS_KM + self.altitude_km

    @property
    def mean_motion_rad_s(self) -> float:
        """Angular velocity of a satellite on the circular orbit."""
        return float(np.sqrt(MU_EARTH / self.orbit_radius_km**3))

    @property
    def period_s(self) -> float:
        return float(2.0 * np.pi / self.mean_motion_rad_s)


@dataclass
class GroundStation:
    name: str
    lat_deg: float
    lon_deg: float


class Constellation:
    """Analytic Walker-delta constellation with a +Grid ISL topology.

    Satellite indexing is row-major over (plane, slot): global index
    ``g = plane * sats_per_plane + slot``.
    """

    def __init__(self, spec: ConstellationSpec, ground_stations: list[GroundStation] | None = None):
        self.spec = spec
        self.ground_stations = ground_stations or []
        self._precompute_orbital_elements()

    # ------------------------------------------------------------------ geometry
    def _precompute_orbital_elements(self) -> None:
        s = self.spec
        planes = np.arange(s.num_planes)
        slots = np.arange(s.sats_per_plane)
        pp, ss = np.meshgrid(planes, slots, indexing="ij")  # (P, S)
        self._plane = pp.reshape(-1)
        self._slot = ss.reshape(-1)

        # RAAN of each plane spread over 2*pi (delta constellation).
        raan = 2.0 * np.pi * self._plane / s.num_planes
        # Argument of latitude at epoch 0: in-plane spacing + Walker phasing offset.
        u0 = 2.0 * np.pi * self._slot / s.sats_per_plane
        u0 = u0 + 2.0 * np.pi * s.phasing_f * self._plane / s.num_sats

        self._raan = raan
        self._u0 = u0
        self._inc = np.deg2rad(s.inclination_deg)

    def sat_grid_fractions(self) -> np.ndarray:
        """Return (num_sats, 2) fractional (plane, slot) coordinates in [0, 1).

        These are size-agnostic structural coordinates (which plane, which slot),
        always known on board, and highly predictive of +Grid routing direction.
        """
        s = self.spec
        frac = np.stack(
            [self._plane / s.num_planes, self._slot / s.sats_per_plane], axis=1
        )
        return frac.astype(np.float64)

    def sat_positions_eci(self, t_s: float) -> np.ndarray:
        """Return (num_sats, 3) ECI positions in km at epoch ``t_s`` seconds."""
        s = self.spec
        u = self._u0 + s.mean_motion_rad_s * t_s
        cu, su = np.cos(u), np.sin(u)
        cO, sO = np.cos(self._raan), np.sin(self._raan)
        ci, si = np.cos(self._inc), np.sin(self._inc)
        r = s.orbit_radius_km
        x = r * (cu * cO - su * ci * sO)
        y = r * (cu * sO + su * ci * cO)
        z = r * (su * si)
        return np.stack([x, y, z], axis=1)

    def gs_positions_eci(self, t_s: float) -> np.ndarray:
        """Return (num_gs, 3) ECI positions in km for ground stations at ``t_s``.

        Ground stations rotate with the Earth (sidereal rate). We use a simplified
        Earth rotation rate of 2*pi per sidereal day.
        """
        if not self.ground_stations:
            return np.zeros((0, 3))
        omega_earth = 2.0 * np.pi / 86164.0  # rad/s, sidereal day
        lats = np.deg2rad([g.lat_deg for g in self.ground_stations])
        lons = np.deg2rad([g.lon_deg for g in self.ground_stations]) + omega_earth * t_s
        r = EARTH_RADIUS_KM
        x = r * np.cos(lats) * np.cos(lons)
        y = r * np.cos(lats) * np.sin(lons)
        z = r * np.sin(lats)
        return np.stack([x, y, z], axis=1)

    @staticmethod
    def _geodetic_latitude(pos_eci: np.ndarray) -> np.ndarray:
        """Approximate latitude (deg) from ECI position (spherical Earth)."""
        r = np.linalg.norm(pos_eci, axis=1)
        r = np.where(r == 0.0, 1.0, r)
        return np.rad2deg(np.arcsin(pos_eci[:, 2] / r))

    # ------------------------------------------------------------------ topology
    def isl_edges(self) -> np.ndarray:
        """Return the static +Grid ISL edge list as an (E, 2) array of global indices.

        Each satellite has up to four ISLs: two intra-plane (previous/next slot,
        wrapping within the plane) and two inter-plane (same slot, neighbouring
        planes). Edges are undirected and de-duplicated (u < v).
        """
        s = self.spec
        P, S = s.num_planes, s.sats_per_plane
        edges = set()

        def gid(p, slot):
            return (p % P) * S + (slot % S)

        for p in range(P):
            for slot in range(S):
                a = gid(p, slot)
                # Intra-plane neighbour (next slot, wrap within plane).
                b = gid(p, slot + 1)
                edges.add((min(a, b), max(a, b)))
                # Inter-plane neighbour (next plane, same slot). Planes wrap so that
                # the last plane connects back to the first (seam handled naturally).
                c = gid(p + 1, slot)
                edges.add((min(a, c), max(a, c)))
        return np.array(sorted(edges), dtype=np.int64)

    def active_isl_mask(self, t_s: float, isl_edges: np.ndarray | None = None) -> np.ndarray:
        """Boolean mask over ISL edges that are active at epoch ``t_s``.

        Intra-plane ISLs are always active. Inter-plane ISLs are deactivated when
        either endpoint is above the polar cut-off latitude.
        """
        if isl_edges is None:
            isl_edges = self.isl_edges()
        S = self.spec.sats_per_plane
        same_plane = (isl_edges[:, 0] // S) == (isl_edges[:, 1] // S)
        if self.spec.isl_polar_cutoff_deg is None:
            return np.ones(len(isl_edges), dtype=bool)
        pos = self.sat_positions_eci(t_s)
        lat = np.abs(self._geodetic_latitude(pos))
        cutoff = self.spec.isl_polar_cutoff_deg
        below_cutoff = (lat[isl_edges[:, 0]] <= cutoff) & (lat[isl_edges[:, 1]] <= cutoff)
        return same_plane | below_cutoff

    def visible_gsl(self, t_s: float, min_elevation_deg: float = 25.0) -> list[tuple[int, int, float]]:
        """Ground-station-to-satellite links visible at epoch ``t_s``.

        Returns a list of (gs_local_index, sat_global_index, distance_km) for every
        (ground station, satellite) pair with elevation above ``min_elevation_deg``.
        """
        if not self.ground_stations:
            return []
        sats = self.sat_positions_eci(t_s)          # (N, 3)
        gss = self.gs_positions_eci(t_s)            # (G, 3)
        links: list[tuple[int, int, float]] = []
        for gi in range(len(gss)):
            g = gss[gi]
            up = g / np.linalg.norm(g)              # local zenith direction
            vec = sats - g                          # (N, 3)
            dist = np.linalg.norm(vec, axis=1)
            # Elevation angle above local horizon.
            cos_zenith = (vec @ up) / np.maximum(dist, 1e-9)
            elevation = np.rad2deg(np.arcsin(np.clip(cos_zenith, -1.0, 1.0)))
            visible = np.where(elevation >= min_elevation_deg)[0]
            for si in visible:
                links.append((gi, int(si), float(dist[si])))
        return links
