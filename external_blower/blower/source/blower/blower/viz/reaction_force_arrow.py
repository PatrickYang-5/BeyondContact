"""Reaction-force arrow visualization for blower source bodies."""

from __future__ import annotations

import math

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
    length = math.sqrt(dx * dx + dy * dy + dz * dz)
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


def _normalize(v: tuple[float, float, float]) -> tuple[float, float, float] | None:
    length = math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])
    if length < 1e-9:
        return None
    return v[0] / length, v[1] / length, v[2] / length


def _add(a, b):
    return a[0] + b[0], a[1] + b[1], a[2] + b[2]


def _mul(s: float, v):
    return s * v[0], s * v[1], s * v[2]


class ReactionForceArrowVisualizer:
    """Draw one recoil arrow per visualized environment.

    The arrow points opposite the applied reaction force, so it visually aligns
    with the blower thrust direction while length still encodes reaction force
    magnitude.
    """

    def __init__(self, cfg: VizCfg, prim_root: str = "/World/Debug/ReactionForceArrows"):
        self.cfg = cfg
        self.stage = omni.usd.get_context().get_stage()
        self.root_path = prim_root
        _ensure_xform(self.stage, prim_root)
        self._created_envs: set[int] = set()

    def update_many(
        self,
        env_ids: list[int],
        arrow_origins: list[tuple[float, float, float]],
        reaction_forces: list[tuple[float, float, float]],
    ) -> None:
        if not self.cfg.show_reaction_force_arrow:
            return

        active_env_ids = set()
        for env_id, origin, reaction_force in zip(env_ids, arrow_origins, reaction_forces):
            self._ensure_arrow(env_id)
            active_env_ids.add(env_id)
            self._update_arrow(env_id, origin, reaction_force)

        for env_id in self._created_envs - active_env_ids:
            self._hide_arrow(env_id)

    def clear(self, env_ids: list[int] | None = None) -> None:
        if env_ids is None:
            env_ids = list(self._created_envs)
        for env_id in env_ids:
            if env_id in self._created_envs:
                self._hide_arrow(env_id)

    def _ensure_arrow(self, env_id: int) -> None:
        if env_id in self._created_envs:
            return

        env_root = f"{self.root_path}/env_{env_id:04d}"
        _ensure_xform(self.stage, env_root)

        color = Gf.Vec3f(*[float(c) for c in self.cfg.reaction_force_arrow_color])

        shaft = UsdGeom.Cylinder.Define(self.stage, f"{env_root}/Shaft")
        shaft.CreateRadiusAttr(float(self.cfg.reaction_force_arrow_shaft_radius))
        shaft.CreateHeightAttr(float(self.cfg.reaction_force_arrow_min_length))
        shaft.CreateDisplayColorAttr([color])
        shaft_xform = UsdGeom.Xformable(shaft.GetPrim())
        _get_or_create_op(shaft_xform, UsdGeom.XformOp.TypeTranslate)
        _get_or_create_op(shaft_xform, UsdGeom.XformOp.TypeOrient)

        head = UsdGeom.Cone.Define(self.stage, f"{env_root}/Head")
        head.CreateRadiusAttr(float(self.cfg.reaction_force_arrow_head_radius))
        head.CreateHeightAttr(float(self.cfg.reaction_force_arrow_head_length))
        head.CreateDisplayColorAttr([color])
        head_xform = UsdGeom.Xformable(head.GetPrim())
        _get_or_create_op(head_xform, UsdGeom.XformOp.TypeTranslate)
        _get_or_create_op(head_xform, UsdGeom.XformOp.TypeOrient)

        self._created_envs.add(env_id)
        self._hide_arrow(env_id)

    def _update_arrow(
        self,
        env_id: int,
        origin: tuple[float, float, float],
        reaction_force: tuple[float, float, float],
    ) -> None:
        force_mag = math.sqrt(
            reaction_force[0] * reaction_force[0]
            + reaction_force[1] * reaction_force[1]
            + reaction_force[2] * reaction_force[2]
        )
        if force_mag < float(self.cfg.reaction_force_arrow_min_force):
            self._hide_arrow(env_id)
            return

        direction = _normalize((-reaction_force[0], -reaction_force[1], -reaction_force[2]))
        if direction is None:
            self._hide_arrow(env_id)
            return

        length = force_mag * float(self.cfg.reaction_force_arrow_length_per_newton)
        length = max(float(self.cfg.reaction_force_arrow_min_length), length)
        length = min(float(self.cfg.reaction_force_arrow_max_length), length)

        head_length = min(float(self.cfg.reaction_force_arrow_head_length), 0.6 * length)
        shaft_length = max(length - head_length, 1e-4)

        shaft_center = _add(origin, _mul(0.5 * shaft_length, direction))
        head_center = _add(origin, _mul(shaft_length + 0.5 * head_length, direction))
        orient = _rotation_to_quat(direction)

        env_root = f"{self.root_path}/env_{env_id:04d}"
        shaft = UsdGeom.Cylinder(self.stage.GetPrimAtPath(f"{env_root}/Shaft"))
        shaft.CreateHeightAttr(float(shaft_length))
        shaft_xform = UsdGeom.Xformable(shaft.GetPrim())
        _get_or_create_op(shaft_xform, UsdGeom.XformOp.TypeTranslate).Set(Gf.Vec3d(*shaft_center))
        _get_or_create_op(shaft_xform, UsdGeom.XformOp.TypeOrient).Set(orient)

        head = UsdGeom.Cone(self.stage.GetPrimAtPath(f"{env_root}/Head"))
        head.CreateHeightAttr(float(head_length))
        head_xform = UsdGeom.Xformable(head.GetPrim())
        _get_or_create_op(head_xform, UsdGeom.XformOp.TypeTranslate).Set(Gf.Vec3d(*head_center))
        _get_or_create_op(head_xform, UsdGeom.XformOp.TypeOrient).Set(orient)

    def _hide_arrow(self, env_id: int) -> None:
        env_root = f"{self.root_path}/env_{env_id:04d}"
        hidden = Gf.Vec3d(0.0, 0.0, -1000.0)
        for prim_name in ("Shaft", "Head"):
            prim = self.stage.GetPrimAtPath(f"{env_root}/{prim_name}")
            if prim.IsValid():
                xform = UsdGeom.Xformable(prim)
                _get_or_create_op(xform, UsdGeom.XformOp.TypeTranslate).Set(hidden)
