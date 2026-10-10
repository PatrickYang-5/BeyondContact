"""Fabric-safe jet cone debug visualization."""

from __future__ import annotations

import math

from blower.configs.airflow_cfg import JetCfg, VizCfg


def _cfg_get(obj, name, default):
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _normalize(v: tuple[float, float, float]) -> tuple[float, float, float] | None:
    n = math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])
    if n < 1e-12:
        return None
    return v[0] / n, v[1] / n, v[2] / n


def _cross(a, b):
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def _dot(a, b) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _basis_from_axis(axis: tuple[float, float, float]):
    ref = (0.0, 0.0, 1.0)
    if abs(_dot(axis, ref)) > 0.95:
        ref = (0.0, 1.0, 0.0)
    u = _normalize(_cross(axis, ref))
    if u is None:
        u = (1.0, 0.0, 0.0)
    v = _cross(axis, u)
    return u, v


def _add(a, b):
    return a[0] + b[0], a[1] + b[1], a[2] + b[2]


def _mul(s: float, v):
    return s * v[0], s * v[1], s * v[2]


def _velocity_to_color(vel_ratio: float) -> tuple[float, float, float]:
    x = max(0.0, min(1.0, float(vel_ratio)))
    low = (0.10, 0.45, 1.00)
    mid = (1.00, 0.78, 0.12)
    high = (1.00, 0.18, 0.04)

    if x < 0.5:
        t = x / 0.5
        a, b = low, mid
    else:
        t = (x - 0.5) / 0.5
        a, b = mid, high
    return tuple((1.0 - t) * a[i] + t * b[i] for i in range(3))


def _centerline_velocity(z: float, jet: JetCfg, intensity: float) -> float:
    u0 = float(_cfg_get(jet, "U0", 130.0)) * max(0.0, float(intensity))
    axial = max(0.0, float(z))
    axial_model = str(_cfg_get(jet, "axial_model", "inverse")).lower()

    if axial_model == "inverse":
        d0 = 2.0 * float(_cfg_get(jet, "nozzle_radius", 0.01))
        return min(u0, float(_cfg_get(jet, "B", 5.9)) * u0 * d0 / (axial + float(_cfg_get(jet, "eps", 0.01))))
    if axial_model in ("exponential", "piecewise_exponential"):
        xc = float(_cfg_get(jet, "xc", 0.0))
        decay_k = float(_cfg_get(jet, "decay_k", 0.0))
        if axial <= xc:
            return u0
        return u0 * math.exp(-decay_k * (axial - xc))

    raise ValueError(f"Unsupported jet axial_model: {axial_model!r}")


def _threshold_boundary_radius(
    z: float,
    jet: JetCfg,
    intensity: float,
    velocity_threshold: float,
    sigma_scale: float,
) -> float:
    """Visualization radius for the edge-spread threshold boundary.

    The drawn cone starts from the physical nozzle rim. Downstream, the
    threshold is solved on the spreading width outside the nozzle radius.
    """
    nozzle_radius = float(_cfg_get(jet, "nozzle_radius", 0.01))
    spread_k = float(_cfg_get(jet, "spread_k", 0.08))
    spread_width = spread_k * max(0.0, float(z))
    if velocity_threshold <= 0.0:
        return nozzle_radius + sigma_scale * spread_width

    centerline_velocity = _centerline_velocity(z, jet, intensity)
    if centerline_velocity <= velocity_threshold:
        return 0.0

    ratio = max(1e-12, min(1.0, velocity_threshold / centerline_velocity))
    return nozzle_radius + spread_width * math.sqrt(-2.0 * math.log(ratio))


def _find_threshold_end_distance(
    jet: JetCfg,
    intensity: float,
    velocity_threshold: float,
    start_distance: float,
    max_dist: float,
) -> float:
    """Find where centerline velocity falls to the visualization threshold."""
    if velocity_threshold <= 0.0:
        return max_dist
    if _centerline_velocity(max_dist, jet, intensity) > velocity_threshold:
        return max_dist
    if _centerline_velocity(start_distance, jet, intensity) <= velocity_threshold:
        return start_distance

    lo = start_distance
    hi = max_dist
    for _ in range(48):
        mid = 0.5 * (lo + hi)
        if _centerline_velocity(mid, jet, intensity) > velocity_threshold:
            lo = mid
        else:
            hi = mid
    return hi


class JetConeVisualizer:
    """Draw a lightweight velocity cone using Isaac Sim debug-draw lines.

    This avoids USD mesh/material updates during RL stepping, so it stays usable
    with Fabric enabled and does not interfere with articulation control.
    """

    def __init__(
        self,
        cfg: VizCfg,
        jet_cfg: JetCfg | None = None,
        prim_root: str = "/World/Debug/JetCone",
    ):
        if isinstance(jet_cfg, str):
            jet_cfg = None

        self.cfg = cfg
        self.jet_cfg = jet_cfg if jet_cfg is not None else JetCfg()
        self._frame_count = 0
        self._warned_missing_draw = False
        self._draw = None
        try:
            import isaacsim.util.debug_draw._debug_draw as omni_debug_draw

            self._draw = omni_debug_draw.acquire_debug_draw_interface()
        except Exception:
            self._draw = None

    def update(
        self,
        nozzle_pos: tuple[float, float, float],
        nozzle_dir: tuple[float, float, float],
        intensity: float = 1.0,
    ):
        self.update_many([nozzle_pos], [nozzle_dir], [intensity])

    def update_many(
        self,
        nozzle_positions: list[tuple[float, float, float]],
        nozzle_dirs: list[tuple[float, float, float]],
        intensities: list[float] | None = None,
    ):
        """Draw all cones as one debug-draw batch so envs do not clear each other."""
        if not self.cfg.show_jet_cone:
            return
        if self._draw is None:
            if not self._warned_missing_draw:
                print("[WARN] Jet cone debug draw interface is unavailable; cone visualization is disabled.")
                self._warned_missing_draw = True
            return

        self._frame_count += 1
        update_interval = max(1, int(_cfg_get(self.cfg, "jet_cone_update_interval", 4)))
        if (self._frame_count - 1) % update_interval != 0:
            return

        if intensities is None:
            intensities = [1.0] * len(nozzle_positions)

        starts = []
        ends = []
        colors = []
        sizes = []
        for nozzle_pos, nozzle_dir, intensity in zip(nozzle_positions, nozzle_dirs, intensities):
            cone_starts, cone_ends, cone_colors, cone_sizes = self._build_cone_lines(
                nozzle_pos,
                nozzle_dir,
                intensity,
            )
            starts.extend(cone_starts)
            ends.extend(cone_ends)
            colors.extend(cone_colors)
            sizes.extend(cone_sizes)

        self._draw.clear_lines()
        if starts:
            self._draw.draw_lines(starts, ends, colors, sizes)

    def _build_cone_lines(
        self,
        nozzle_pos: tuple[float, float, float],
        nozzle_dir: tuple[float, float, float],
        intensity: float = 1.0,
    ):
        axis = _normalize(tuple(float(v) for v in nozzle_dir))
        if axis is None:
            return [], [], [], []
        origin = tuple(float(v) for v in nozzle_pos)
        u, v = _basis_from_axis(axis)

        num_bands = max(1, int(_cfg_get(self.cfg, "jet_cone_num_slices", 16)))
        num_segments = max(8, int(_cfg_get(self.cfg, "jet_cone_num_segments", 32)))
        max_dist = float(_cfg_get(self.cfg, "jet_cone_max_distance", 5.0))
        velocity_threshold = float(_cfg_get(self.cfg, "jet_cone_velocity_threshold", 0.5))
        sigma_scale = float(_cfg_get(self.cfg, "jet_cone_sigma_scale", 1.0))
        min_alpha = float(_cfg_get(self.cfg, "jet_cone_min_alpha", 0.015))
        max_alpha = float(_cfg_get(self.cfg, "jet_cone_max_alpha", 0.12))
        start_distance = float(_cfg_get(self.cfg, "jet_cone_start_distance", 0.0))
        alpha_gamma = float(_cfg_get(self.cfg, "jet_cone_alpha_gamma", 0.55))
        thickness = float(_cfg_get(self.cfg, "jet_cone_line_thickness", 2.0))
        use_flow_intensity = bool(_cfg_get(self.cfg, "jet_cone_use_flow_intensity", False))
        visual_intensity = max(0.0, float(intensity)) if use_flow_intensity else 1.0

        jet = self.jet_cfg
        u0_safe = max(float(_cfg_get(jet, "U0", 130.0)), 1e-12)
        max_dist = max(max_dist, start_distance + 1e-4)
        max_dist = _find_threshold_end_distance(
            jet,
            visual_intensity,
            velocity_threshold,
            start_distance,
            max_dist,
        )
        max_dist = max(max_dist, start_distance + 1e-4)

        rings = []
        z_values = []
        for i in range(num_bands + 1):
            t = i / num_bands
            z = start_distance + (max_dist - start_distance) * (t * t)
            radius = _threshold_boundary_radius(z, jet, visual_intensity, velocity_threshold, sigma_scale)
            center = _add(origin, _mul(z, axis))
            ring = []
            for j in range(num_segments):
                theta = 2.0 * math.pi * j / num_segments
                radial = _add(_mul(math.cos(theta) * radius, u), _mul(math.sin(theta) * radius, v))
                ring.append(_add(center, radial))
            rings.append(ring)
            z_values.append(z)

        starts = []
        ends = []
        colors = []
        sizes = []

        for i, ring in enumerate(rings):
            velocity = _centerline_velocity(z_values[i], jet, visual_intensity)
            vel_ratio = max(0.0, min(1.0, velocity / u0_safe))
            color_rgb = _velocity_to_color(vel_ratio)
            alpha = min_alpha + (max_alpha - min_alpha) * (vel_ratio ** alpha_gamma)
            color = (color_rgb[0], color_rgb[1], color_rgb[2], max(0.2, min(1.0, alpha * 6.0)))
            for j in range(num_segments):
                starts.append(ring[j])
                ends.append(ring[(j + 1) % num_segments])
                colors.append(color)
                sizes.append(thickness)

        stride = max(1, num_segments // 12)
        for i in range(len(rings) - 1):
            z_mid = 0.5 * (z_values[i] + z_values[i + 1])
            velocity = _centerline_velocity(z_mid, jet, visual_intensity)
            vel_ratio = max(0.0, min(1.0, velocity / u0_safe))
            color_rgb = _velocity_to_color(vel_ratio)
            color = (color_rgb[0], color_rgb[1], color_rgb[2], 0.85)
            for j in range(0, num_segments, stride):
                starts.append(rings[i][j])
                ends.append(rings[i + 1][j])
                colors.append(color)
                sizes.append(thickness)

        return starts, ends, colors, sizes
