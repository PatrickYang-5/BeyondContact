"""Tensorized airflow physics model for batched multi-environment simulation."""

from __future__ import annotations

import torch

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

from blower.configs.airflow_cfg import DragCfg, JetCfg

@configclass
class AirflowModelCfg:
    jet: JetCfg = JetCfg()
    drag: DragCfg = DragCfg()
    mode: str = 'surface'
    min_axial_distance: float = 0.0
    velocity_cutoff: float = 0.5
    nozzle_local_axis: str = 'x'


def quat_to_axis(quat: torch.Tensor, axis: str = 'x') -> torch.Tensor:
    """Rotate a local cardinal axis by quaternion in (w, x, y, z) convention."""
    w, x, y, z = quat.unbind(-1)

    axis = axis.lower()
    sign = -1.0 if axis.startswith('-') else 1.0
    axis_key = axis[1:] if axis.startswith('-') else axis

    if axis_key == 'x':
        vec = torch.stack([
            1.0 - 2.0 * (y * y + z * z),
            2.0 * (x * y + w * z),
            2.0 * (x * z - w * y),
        ], dim=-1)
    elif axis_key == 'y':
        vec = torch.stack([
            2.0 * (x * y - w * z),
            1.0 - 2.0 * (x * x + z * z),
            2.0 * (y * z + w * x),
        ], dim=-1)
    elif axis_key == 'z':
        vec = torch.stack([
            2.0 * (x * z + w * y),
            2.0 * (y * z - w * x),
            1.0 - 2.0 * (x * x + y * y),
        ], dim=-1)
    else:
        raise ValueError(f'Unsupported local axis: {axis}')

    return sign * vec


def quat_to_forward(quat: torch.Tensor) -> torch.Tensor:
    """Rotate local +X by quaternion in (w, x, y, z) convention."""
    return quat_to_axis(quat, 'x')


def _normalize(v: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    norm = torch.norm(v, dim=-1, keepdim=True)
    return v / norm.clamp(min=eps)


def compute_jet_coordinates(
    nozzle_pos: torch.Tensor,
    nozzle_dir: torch.Tensor,
    points: torch.Tensor,
    eps: float = 1e-9,
):
    if points.dim() == 2:
        points = points.unsqueeze(1)
    if nozzle_pos.dim() == 1:
        nozzle_pos = nozzle_pos.unsqueeze(0)
    if nozzle_dir.dim() == 1:
        nozzle_dir = nozzle_dir.unsqueeze(0)

    nozzle_dir = _normalize(nozzle_dir, eps=eps)
    delta = points - nozzle_pos.unsqueeze(1)
    axial = torch.sum(delta * nozzle_dir.unsqueeze(1), dim=-1)
    radial_vec = delta - axial.unsqueeze(-1) * nozzle_dir.unsqueeze(1)
    radial = torch.norm(radial_vec, dim=-1)
    return axial, radial, radial_vec


def compute_axial_velocity(
    axial: torch.Tensor,
    jet_cfg: JetCfg,
    intensity: torch.Tensor | None = None,
) -> torch.Tensor:
    if intensity is None:
        intensity = torch.ones_like(axial)
    else:
        intensity = intensity.to(dtype=axial.dtype, device=axial.device)
        while intensity.dim() < axial.dim():
            intensity = intensity.unsqueeze(-1)

    U0 = jet_cfg.U0 * intensity
    axial_clamped = torch.clamp(axial, min=0.0)
    axial_model = str(getattr(jet_cfg, 'axial_model', 'inverse')).lower()

    if axial_model == 'inverse':
        d0 = 2.0 * jet_cfg.nozzle_radius
        return torch.minimum(U0, jet_cfg.B * U0 * d0 / (axial_clamped + jet_cfg.eps))
    if axial_model in ('exponential', 'piecewise_exponential'):
        xc = float(getattr(jet_cfg, 'xc', 0.0))
        decay_k = float(getattr(jet_cfg, 'decay_k', 0.0))
        return torch.where(
            axial_clamped <= xc,
            U0,
            U0 * torch.exp(-decay_k * (axial_clamped - xc)),
        )

    raise ValueError(f"Unsupported jet axial_model: {jet_cfg.axial_model!r}")


def compute_jet_velocity(
    axial: torch.Tensor,
    radial: torch.Tensor,
    jet_cfg: JetCfg,
    intensity: torch.Tensor | None = None,
):
    axial_clamped = torch.clamp(axial, min=0.0)
    U_axial = compute_axial_velocity(axial, jet_cfg, intensity=intensity)
    sigma_r = jet_cfg.nozzle_radius + jet_cfg.spread_k * axial_clamped
    sigma_r = sigma_r.clamp(min=1e-12)
    velocity = U_axial * torch.exp(-radial ** 2 / (2.0 * sigma_r ** 2))
    return torch.where(axial < 0.0, torch.zeros_like(velocity), velocity)


def compute_drag_pressure(velocity: torch.Tensor, rho: float, C_D: float) -> torch.Tensor:
    return 0.5 * rho * velocity ** 2 * C_D


def compute_simple_airflow_result(
    nozzle_pos: torch.Tensor,
    nozzle_dir: torch.Tensor,
    target_pos: torch.Tensor,
    sample_normals_world: torch.Tensor,
    sample_areas: torch.Tensor,
    sample_is_omni: torch.Tensor,
    cfg: AirflowModelCfg,
    flow_intensity: torch.Tensor,
) -> dict:
    if target_pos.dim() == 2:
        target_pos = target_pos.unsqueeze(1)
    nozzle_dir = _normalize(nozzle_dir)
    axial, radial, _ = compute_jet_coordinates(nozzle_pos, nozzle_dir, target_pos)
    velocity = compute_jet_velocity(axial, radial, cfg.jet, intensity=flow_intensity)
    valid = (axial >= cfg.min_axial_distance) & (velocity >= cfg.velocity_cutoff)
    velocity = torch.where(valid, velocity, torch.zeros_like(velocity))
    q = compute_drag_pressure(velocity, cfg.drag.rho, cfg.drag.C_D)

    nozzle_dir_expanded = nozzle_dir.unsqueeze(1)
    omni_expanded = sample_is_omni.unsqueeze(-1)
    normal_dot = torch.sum(nozzle_dir_expanded * sample_normals_world, dim=-1)
    front_mask = (normal_dot < 0.0) & ~sample_is_omni
    projected_area = torch.where(
        omni_expanded,
        sample_areas.unsqueeze(0).unsqueeze(-1),
        torch.where(
            front_mask.unsqueeze(-1),
            sample_areas.unsqueeze(0).unsqueeze(-1) * (-normal_dot).unsqueeze(-1),
            torch.zeros_like(normal_dot).unsqueeze(-1),
        ),
    ).sum(dim=1)

    if projected_area.dim() == 2:
        projected_area = projected_area.squeeze(-1)
    force_mag = q.squeeze(-1) * projected_area
    force_vec = force_mag.unsqueeze(-1).unsqueeze(-1) * nozzle_dir_expanded
    torque_vec = torch.zeros_like(force_vec)

    return {
        'hit': force_mag > 0.0,
        'force_vec': force_vec,
        'torque_vec': torque_vec,
        'force_mag': force_mag,
    }


def compute_surface_airflow_result(
    nozzle_pos: torch.Tensor,
    nozzle_dir: torch.Tensor,
    target_pos: torch.Tensor,
    sample_positions_world: torch.Tensor,
    sample_normals_world: torch.Tensor,
    sample_areas: torch.Tensor,
    sample_is_omni: torch.Tensor,
    cfg: AirflowModelCfg,
    flow_intensity: torch.Tensor,
) -> dict:
    nozzle_dir = _normalize(nozzle_dir)
    axial, radial, _ = compute_jet_coordinates(nozzle_pos, nozzle_dir, sample_positions_world)
    velocity = compute_jet_velocity(axial, radial, cfg.jet, intensity=flow_intensity)
    velocity = torch.where(axial >= cfg.min_axial_distance, velocity, torch.zeros_like(velocity))
    q = compute_drag_pressure(velocity, cfg.drag.rho, cfg.drag.C_D)

    nozzle_dir_expanded = nozzle_dir.unsqueeze(1)
    normal_dot = torch.sum(nozzle_dir_expanded * sample_normals_world, dim=-1)
    front_mask = (normal_dot < 0.0) & ~sample_is_omni
    cos_factor = torch.where(
        sample_is_omni,
        torch.ones_like(normal_dot),
        torch.where(front_mask, -normal_dot, torch.zeros_like(normal_dot)),
    )
    force_vec = q.unsqueeze(-1) * sample_areas.unsqueeze(0).unsqueeze(-1) * cos_factor.unsqueeze(-1) * nozzle_dir_expanded
    total_force = force_vec.sum(dim=1, keepdim=True)
    moment_arm = sample_positions_world - target_pos.unsqueeze(1)
    torque_vec = torch.cross(moment_arm, force_vec, dim=-1).sum(dim=1, keepdim=True)

    target_axial, target_radial, _ = compute_jet_coordinates(nozzle_pos, nozzle_dir, target_pos)
    target_velocity = compute_jet_velocity(target_axial, target_radial, cfg.jet, intensity=flow_intensity)
    object_mask = (target_axial >= cfg.min_axial_distance) & (target_velocity >= cfg.velocity_cutoff)
    total_force = torch.where(object_mask.unsqueeze(-1), total_force, torch.zeros_like(total_force))
    torque_vec = torch.where(object_mask.unsqueeze(-1), torque_vec, torch.zeros_like(torque_vec))
    force_mag = torch.norm(total_force, dim=-1).squeeze(-1)

    return {
        'hit': force_mag > 0.0,
        'force_vec': total_force,
        'torque_vec': torque_vec,
        'force_mag': force_mag,
    }


class AirflowModel:
    def __init__(self, cfg: AirflowModelCfg, device: str = 'cuda:0'):
        self.cfg = cfg
        self.device = device

    def compute_force_result(
        self,
        blower_pos: torch.Tensor,
        blower_quat: torch.Tensor,
        target_pos: torch.Tensor,
        sample_positions_world: torch.Tensor,
        sample_normals_world: torch.Tensor,
        sample_areas: torch.Tensor,
        sample_is_omni: torch.Tensor,
        flow_intensity: torch.Tensor,
    ) -> dict:
        blower_pos = blower_pos.to(dtype=torch.float32, device=self.device)
        blower_quat = blower_quat.to(dtype=torch.float32, device=self.device)
        target_pos = target_pos.to(dtype=torch.float32, device=self.device)
        sample_positions_world = sample_positions_world.to(dtype=torch.float32, device=self.device)
        sample_normals_world = sample_normals_world.to(dtype=torch.float32, device=self.device)
        sample_areas = sample_areas.to(dtype=torch.float32, device=self.device)
        sample_is_omni = sample_is_omni.to(dtype=torch.bool, device=self.device)
        flow_intensity = flow_intensity.to(dtype=torch.float32, device=self.device)

        if flow_intensity.dim() == 1:
            flow_intensity = flow_intensity.unsqueeze(-1)

        if self.cfg.mode == 'surface':
            return compute_surface_airflow_result(
                blower_pos,
                quat_to_axis(blower_quat, self.cfg.nozzle_local_axis),
                target_pos,
                sample_positions_world,
                sample_normals_world,
                sample_areas,
                sample_is_omni,
                self.cfg,
                flow_intensity,
            )
        return compute_simple_airflow_result(
            blower_pos,
            quat_to_axis(blower_quat, self.cfg.nozzle_local_axis),
            target_pos,
            sample_normals_world,
            sample_areas,
            sample_is_omni,
            self.cfg,
            flow_intensity,
        )
