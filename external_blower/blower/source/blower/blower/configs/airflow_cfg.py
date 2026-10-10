"""Airflow configuration classes for the blower extension."""

from __future__ import annotations

try:
    from isaaclab.utils import configclass
except Exception:
    import copy

    def configclass(cls):
        annotations = getattr(cls, '__annotations__', {})
        defaults = {
            name: getattr(cls, name)
            for name in annotations
            if hasattr(cls, name)
        }

        def __init__(self, **kwargs):
            for name in annotations:
                if name in kwargs:
                    value = kwargs.pop(name)
                elif name in defaults:
                    value = copy.deepcopy(defaults[name])
                else:
                    continue
                setattr(self, name, value)
            if kwargs:
                unexpected = ', '.join(kwargs)
                raise TypeError(f"Unexpected config argument(s): {unexpected}")

        cls.__init__ = __init__
        return cls

@configclass
class JetCfg:
    # ------------------------------------------------------------------
    # Parameters used by BOTH axial models.
    # ------------------------------------------------------------------
    # These control the radial Gaussian term:
    #
    #   sigma_r(x) = nozzle_radius + spread_k * x
    #   U(x, r) = Uc(x) * exp(-r^2 / (2 * sigma_r(x)^2))
    #
    # Therefore `spread_k` is still active when axial_model == "exponential".
    nozzle_radius: float = 0.01
    spread_k: float = 0.08

    # Shared centerline velocity scale.
    U0: float = 130.0

    # ------------------------------------------------------------------
    # Select the axial centerline model Uc(x).
    # ------------------------------------------------------------------
    #   "inverse":
    #       Uc(x) = min(U0, B * U0 * 2 * nozzle_radius / (x + eps))
    #
    #   "exponential":
    #       Uc(x) = U0                              for x <= xc
    #       Uc(x) = U0 * exp(-decay_k * (x - xc))   for x > xc
    axial_model: str = 'inverse'

    # ------------------------------------------------------------------
    # Inverse-model-only parameters. Ignored when axial_model == "exponential".
    # ------------------------------------------------------------------
    B: float = 5.9
    eps: float = 0.01

    # ------------------------------------------------------------------
    # Exponential-model-only parameters. Ignored when axial_model == "inverse".
    # ------------------------------------------------------------------
    xc: float = 0.1
    decay_k: float = 1.0

@configclass
class DragCfg:
    rho: float = 1.225
    C_D: float = 1.1

@configclass
class SamplingCfg:
    target_area: float = 6e-5
    min_n: int = 2
    max_n: int = 64
    min_ntheta: int = 8
    max_ntheta: int = 128
    min_nz: int = 2
    max_nz: int = 128
    mesh_oversample_factor: int = 8
    mesh_seed: int = 0

    @property
    def spacing(self) -> float:
        return float((self.target_area if self.target_area > 0 else 1e-12) ** 0.5)

@configclass
class VizCfg:
    show_sample_markers: bool = True
    show_jet_cone: bool = True
    show_laser_ray: bool = True
    max_visualized_envs: int = 16
    visualized_env_ids: tuple[int, ...] = ()

    debug_prim_root: str = '/World/Debug/Airflow'
    """Root USD prim path for all debug visualizations from this blower.

    Override per-blower to avoid prim conflicts when multiple blowers are active in the same scene.
    Example: '/World/Debug/Airflow_blower_franka'
    """

    jet_cone_render_mode: str = "wireframe"
    """Use 'mesh' for project-style translucent shells or 'wireframe' for RL debug lines."""

    jet_cone_num_slices: int = 24
    jet_cone_num_segments: int = 48
    jet_cone_max_distance: float = 5.0
    jet_cone_velocity_threshold: float = 0.5
    jet_cone_sigma_scale: float = 1.0
    jet_cone_min_alpha: float = 0.015
    jet_cone_max_alpha: float = 0.12
    jet_cone_start_distance: float = 0.0
    jet_cone_close_caps: bool = False
    jet_cone_alpha_gamma: float = 0.55
    jet_cone_emissive_strength: float = 0.45
    jet_cone_use_flow_intensity: bool = False
    jet_cone_update_interval: int = 4
    jet_cone_line_thickness: float = 2.0

    sample_marker_radius: float = 0.002
    sample_marker_color: tuple[float, float, float] = (0.2, 0.8, 1.0)

    laser_ray_length: float = 1.5
    laser_ray_radius: float = 0.005
    laser_ray_color: tuple[float, float, float] = (0.0, 0.9, 1.0)
    laser_point_radius: float = 0.015
    laser_point_color: tuple[float, float, float] = (1.0, 0.1, 0.02)
    laser_use_hit_plane: bool = False
    laser_hit_plane_z: float = 0.0
    laser_hit_plane_max_length: float = 5.0
    laser_point_surface_offset: float = 0.0

    show_reaction_force_arrow: bool = False
    reaction_force_arrow_length_per_newton: float = 0.05
    reaction_force_arrow_min_force: float = 1e-4
    reaction_force_arrow_min_length: float = 0.02
    reaction_force_arrow_max_length: float = 0.6
    reaction_force_arrow_shaft_radius: float = 0.008
    reaction_force_arrow_head_radius: float = 0.025
    reaction_force_arrow_head_length: float = 0.06
    reaction_force_arrow_color: tuple[float, float, float] = (1.0, 0.55, 0.05)
