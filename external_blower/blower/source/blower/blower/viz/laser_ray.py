"""Laser ray debug visualization for blower direction."""

from __future__ import annotations

import omni.usd
from pxr import Gf, UsdGeom
from blower.configs.airflow_cfg import VizCfg


def _ensure_xform(stage, path: str):
    prim = stage.GetPrimAtPath(path)
    return prim if prim.IsValid() else UsdGeom.Xform.Define(stage, path).GetPrim()


def _get_or_create_op(xformable, op_type):
    for op in xformable.GetOrderedXformOps():
        if op.GetOpType() == op_type:
            return op
    return xformable.AddXformOp(op_type)


def _rotation_to_quat(direction: tuple[float, float, float]) -> Gf.Quatd:
    dx, dy, dz = direction
    length = (dx * dx + dy * dy + dz * dz) ** 0.5
    if length < 1e-9:
        return Gf.Quatd(1.0, 0.0, 0.0, 0.0)
    nx, ny, nz = dx / length, dy / length, dz / length
    z_axis = (0.0, 0.0, 1.0)
    dot = z_axis[0] * nx + z_axis[1] * ny + z_axis[2] * nz
    if dot > 0.9999:
        return Gf.Quatd(1.0, 0.0, 0.0, 0.0)
    if dot < -0.9999:
        return Gf.Quatd(0.0, 1.0, 0.0, 0.0)
    cross = (
        z_axis[1] * nz - z_axis[2] * ny,
        z_axis[2] * nx - z_axis[0] * nz,
        z_axis[0] * ny - z_axis[1] * nx,
    )
    q = Gf.Quatd(1.0 + dot, cross[0], cross[1], cross[2])
    q.Normalize()
    return q


def _normalize(direction: tuple[float, float, float]) -> tuple[float, float, float]:
    dx, dy, dz = direction
    length = (dx * dx + dy * dy + dz * dz) ** 0.5
    if length < 1e-9:
        return 0.0, 0.0, 1.0
    return dx / length, dy / length, dz / length


def _plane_hit_length(
    nozzle_pos: tuple[float, float, float],
    nozzle_dir: tuple[float, float, float],
    plane_z: float,
    max_length: float,
) -> float | None:
    dz = nozzle_dir[2]
    if abs(dz) < 1e-9:
        return None
    hit_length = (plane_z - nozzle_pos[2]) / dz
    if hit_length <= 0.0 or hit_length > max_length:
        return None
    return hit_length


class LaserRayVisualizer:
    def __init__(self, cfg: VizCfg, prim_root: str = '/World/Debug/LaserRay'):
        self.cfg = cfg
        self.stage = omni.usd.get_context().get_stage()
        self.root_path = prim_root
        _ensure_xform(self.stage, prim_root)
        self.cylinder = UsdGeom.Cylinder.Define(self.stage, f'{prim_root}/RayCylinder')
        self.cylinder.CreateRadiusAttr(float(self.cfg.laser_ray_radius))
        self.cylinder.CreateHeightAttr(float(self.cfg.laser_ray_length))
        self.cylinder.CreateDisplayColorAttr([Gf.Vec3f(*[float(c) for c in self.cfg.laser_ray_color])])

        self.point = UsdGeom.Sphere.Define(self.stage, f'{prim_root}/LaserPoint')
        self.point.CreateRadiusAttr(float(self.cfg.laser_point_radius))
        self.point.CreateDisplayColorAttr([Gf.Vec3f(*[float(c) for c in self.cfg.laser_point_color])])
        _get_or_create_op(UsdGeom.Xformable(self.point.GetPrim()), UsdGeom.XformOp.TypeTranslate)

    def update(self, nozzle_pos: tuple[float, float, float], nozzle_dir: tuple[float, float, float], length: float | None = None):
        if not self.cfg.show_laser_ray:
            return
        nozzle_dir = _normalize(nozzle_dir)
        if length is None:
            length = self.cfg.laser_ray_length
        if self.cfg.laser_use_hit_plane:
            hit_length = _plane_hit_length(
                nozzle_pos,
                nozzle_dir,
                self.cfg.laser_hit_plane_z,
                self.cfg.laser_hit_plane_max_length,
            )
            if hit_length is not None:
                length = hit_length
        xformable = UsdGeom.Xformable(self.cylinder.GetPrim())
        translate = _get_or_create_op(xformable, UsdGeom.XformOp.TypeTranslate)
        orient = _get_or_create_op(xformable, UsdGeom.XformOp.TypeOrient)
        world_pos = (
            nozzle_pos[0] + nozzle_dir[0] * length * 0.5,
            nozzle_pos[1] + nozzle_dir[1] * length * 0.5,
            nozzle_pos[2] + nozzle_dir[2] * length * 0.5,
        )
        translate.Set(Gf.Vec3d(*world_pos))
        orient.Set(_rotation_to_quat(nozzle_dir))
        self.cylinder.CreateHeightAttr(float(length))

        point_pos = (
            nozzle_pos[0] + nozzle_dir[0] * length,
            nozzle_pos[1] + nozzle_dir[1] * length,
            nozzle_pos[2] + nozzle_dir[2] * length + self.cfg.laser_point_surface_offset,
        )
        point_xformable = UsdGeom.Xformable(self.point.GetPrim())
        _get_or_create_op(point_xformable, UsdGeom.XformOp.TypeTranslate).Set(Gf.Vec3d(*point_pos))
