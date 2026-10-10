# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Visualization terms and spawners for the airflow box-push task."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING

from isaaclab.sim.spawners.spawner_cfg import SpawnerCfg
from isaaclab.sim.utils import clone
from isaaclab.utils import configclass

from .box_push_common import _ensure_box_push_state, _goal_center_xy_local, _goal_radius, _ids_to_tensor

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


@clone
def spawn_goal_region_wireframe(
    prim_path: str,
    cfg: "GoalRegionWireframeCfg",
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
):
    """Spawn a red square outline with USD curves, avoiding a filled surface and cast shadows."""
    from pxr import Gf, UsdGeom

    stage = cfg.stage if getattr(cfg, "stage", None) is not None else None
    if stage is None:
        import omni.usd

        stage = omni.usd.get_context().get_stage()

    if stage is None:
        raise RuntimeError("Cannot spawn goal-region wireframe without an active USD stage.")

    if stage.GetPrimAtPath(prim_path).IsValid():
        raise ValueError(f"A prim already exists at path: '{prim_path}'.")

    root = UsdGeom.Xform.Define(stage, prim_path)
    xformable = UsdGeom.Xformable(root.GetPrim())

    if translation is not None:
        xformable.AddTranslateOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(
            Gf.Vec3d(*translation)
        )

    if orientation is not None:
        xformable.AddOrientOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(
            Gf.Quatd(orientation[0], orientation[1], orientation[2], orientation[3])
        )

    xformable.AddScaleOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(
        Gf.Vec3d(1.0, 1.0, 1.0)
    )

    half_extent = float(cfg.half_extent)
    z = float(cfg.z_offset)

    points = [
        Gf.Vec3f(-half_extent, -half_extent, z),
        Gf.Vec3f(half_extent, -half_extent, z),
        Gf.Vec3f(half_extent, half_extent, z),
        Gf.Vec3f(-half_extent, half_extent, z),
        Gf.Vec3f(-half_extent, -half_extent, z),
    ]

    curve = UsdGeom.BasisCurves.Define(stage, f"{prim_path}/Outline")
    curve.CreateTypeAttr("linear")
    curve.CreateWrapAttr("nonperiodic")
    curve.CreateCurveVertexCountsAttr([len(points)])
    curve.CreatePointsAttr(points)
    curve.CreateWidthsAttr([float(cfg.line_width)] * len(points))
    curve.CreateDisplayColorPrimvar(UsdGeom.Tokens.constant).Set([Gf.Vec3f(*cfg.color)])
    curve.CreateDisplayOpacityPrimvar(UsdGeom.Tokens.constant).Set([float(cfg.opacity)])

    return root.GetPrim()


@configclass
class GoalRegionWireframeCfg(SpawnerCfg):
    """Spawner config for the square goal-region wireframe."""

    func: Callable = spawn_goal_region_wireframe
    half_extent: float = MISSING
    line_width: float = 0.008
    color: tuple[float, float, float] = (1.0, 0.0, 0.0)
    opacity: float = 1.0
    z_offset: float = 0.0
    stage: object | None = None


@clone
def spawn_heading_line(
    prim_path: str,
    cfg: "HeadingLineCfg",
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
):
    """Spawn a straight line through the local origin, along local +/-X.

    The line is symmetric on purpose: it shows an undirected axis command, so
    it must not grow an arrow head that would imply one of the two directions.
    """
    from pxr import Gf, UsdGeom

    stage = cfg.stage if getattr(cfg, "stage", None) is not None else None
    if stage is None:
        import omni.usd

        stage = omni.usd.get_context().get_stage()

    if stage is None:
        raise RuntimeError("Cannot spawn heading line without an active USD stage.")

    if stage.GetPrimAtPath(prim_path).IsValid():
        raise ValueError(f"A prim already exists at path: '{prim_path}'.")

    root = UsdGeom.Xform.Define(stage, prim_path)
    xformable = UsdGeom.Xformable(root.GetPrim())

    if translation is not None:
        xformable.AddTranslateOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(*translation))

    # Author the standard translate/orient/scale triple. InteractiveScene wraps
    # every AssetBaseCfg in an XformPrimView, which rejects a prim carrying any
    # other op set — so the yaw has to be an orient op, not a rotateZ one.
    orientation = orientation if orientation is not None else (1.0, 0.0, 0.0, 0.0)
    xformable.AddOrientOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(
        Gf.Quatd(orientation[0], Gf.Vec3d(orientation[1], orientation[2], orientation[3]))
    )
    xformable.AddScaleOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(1.0, 1.0, 1.0))

    half_length = float(cfg.half_length)
    z = float(cfg.z_offset)
    points = [Gf.Vec3f(-half_length, 0.0, z), Gf.Vec3f(half_length, 0.0, z)]

    curve = UsdGeom.BasisCurves.Define(stage, f"{prim_path}/Line")
    curve.CreateTypeAttr("linear")
    curve.CreateWrapAttr("nonperiodic")
    curve.CreateCurveVertexCountsAttr([len(points)])
    curve.CreatePointsAttr(points)
    curve.CreateWidthsAttr([float(cfg.line_width)] * len(points))
    curve.CreateDisplayColorPrimvar(UsdGeom.Tokens.constant).Set([Gf.Vec3f(*cfg.color)])
    curve.CreateDisplayOpacityPrimvar(UsdGeom.Tokens.constant).Set([float(cfg.opacity)])

    return root.GetPrim()


@configclass
class HeadingLineCfg(SpawnerCfg):
    """Spawner config for a straight line marking an undirected heading command."""

    func: Callable = spawn_heading_line
    half_length: float = MISSING
    line_width: float = 0.012
    color: tuple[float, float, float] = (0.1, 0.95, 0.35)
    opacity: float = 1.0
    z_offset: float = 0.0
    stage: object | None = None


def update_heading_line_visual(
    env: ManagerBasedRLEnv,
    centers_xy,
    angles,
    center_z: float = 0.009,
    visual_prim_name: str = "HeadingCommand",
    max_visualized_envs: int | None = None,
    env_ids: Sequence[int] | slice | None = None,
) -> None:
    """Point each env's heading line at its own commanded angle.

    ``centers_xy`` and ``angles`` are per-env tensors in the environment's
    local frame, ``angles`` in radians. Same per-prim authoring caveat as
    ``update_goal_region_visual``: see ``_author_local_xform``.
    """
    if not (env.sim.has_gui() or env.sim.has_rtx_sensors()):
        return

    try:
        import omni.usd
        from pxr import Gf
    except Exception:
        return

    stage = omni.usd.get_context().get_stage()
    if stage is None:
        return

    if max_visualized_envs is None:
        num_visualized = int(env.num_envs)
    else:
        num_visualized = min(int(env.num_envs), int(max_visualized_envs))

    visual_env_ids = _ids_to_tensor(env, env_ids)
    visual_env_ids = visual_env_ids[visual_env_ids < num_visualized]

    for env_id_tensor in visual_env_ids:
        env_id = int(env_id_tensor.item())
        prim = stage.GetPrimAtPath(f"{env.scene.env_prim_paths[env_id]}/{visual_prim_name}")

        if not prim.IsValid():
            continue

        translate_value = Gf.Vec3d(
            float(centers_xy[env_id, 0].item()),
            float(centers_xy[env_id, 1].item()),
            float(center_z),
        )
        _author_local_xform(prim, translate_value, Gf.Vec3d(1.0, 1.0, 1.0),
                            yaw_rad=float(angles[env_id].item()))


def update_goal_region_visual(
    env: ManagerBasedRLEnv,
    goal_center: Sequence[float],
    goal_radius: float,
    visual_prim_name: str = "GoalRegion",
    base_radius: float | None = None,
    max_visualized_envs: int | None = None,
    env_ids: Sequence[int] | slice | None = None,
) -> None:
    """Update each env's goal-region wireframe to its per-env goal center/size.

    Implementation notes
    --------------------
    1. ``max_visualized_envs`` defaults to ``None``, which is treated as
       "update all envs". The previous default of 32 silently dropped envs
       17..N when running PLAY/eval with ``--num_envs`` larger than 32, so
       those envs' wireframes never moved off their initial spawn pose and
       all looked identical to env_0. If you want to throttle visualization
       for performance during training, pass an explicit integer.

    2. When IsaacLab clones the source env via ``copy_from_source=False``
       (the default), each ``/World/envs/env_N`` subtree is a USD *Inherits*
       of ``/World/envs/env_0``. Mutating XformOps through the high-level
       ``Xformable`` API tends to write back to the inherited attribute spec
       on env_0, which then propagates to every clone. To get per-env
       independence we author the three xform attributes
       (``xformOp:translate``, ``xformOp:scale``, ``xformOpOrder``) directly
       through ``prim.CreateAttribute(...).Set(...)``, which writes on the
       current edit target's local spec for *this* prim, bypassing inherits.
    """
    if not (env.sim.has_gui() or env.sim.has_rtx_sensors()):
        return

    try:
        import omni.usd
        from pxr import Gf, Sdf, UsdGeom, Vt
    except Exception:
        return

    _ensure_box_push_state(env, goal_radius)

    stage = omni.usd.get_context().get_stage()
    if stage is None:
        return

    if base_radius is None:
        base_radius = float(goal_radius)

    center_z = float(goal_center[2]) if len(goal_center) > 2 else 0.005
    centers_xy = _goal_center_xy_local(env, goal_center)
    radii = _goal_radius(env, goal_radius)

    # ``None`` means "no per-call cap" — visualize as many envs as exist.
    if max_visualized_envs is None:
        num_visualized = int(env.num_envs)
    else:
        num_visualized = min(int(env.num_envs), int(max_visualized_envs))

    visual_env_ids = _ids_to_tensor(env, env_ids)
    visual_env_ids = visual_env_ids[visual_env_ids < num_visualized]

    for env_id_tensor in visual_env_ids:
        env_id = int(env_id_tensor.item())
        prim_path = f"{env.scene.env_prim_paths[env_id]}/{visual_prim_name}"
        prim = stage.GetPrimAtPath(prim_path)

        if not prim.IsValid():
            continue

        translate_value = Gf.Vec3d(
            float(centers_xy[env_id, 0].item()),
            float(centers_xy[env_id, 1].item()),
            center_z,
        )

        half_extent = float(radii[env_id].item())
        scale_xy = max(half_extent / max(float(base_radius), 1e-6), 1e-3)
        scale_value = Gf.Vec3d(scale_xy, scale_xy, 1.0)

        _author_local_xform(prim, translate_value, scale_value)


def _author_local_xform(prim, translate_value, scale_value, yaw_rad=None):
    """Force-author per-prim ``xformOp:translate``, ``xformOp:scale``, and ``xformOpOrder``.

    Uses ``prim.CreateAttribute(...).Set(...)`` for each of the three
    transform-related attributes. ``CreateAttribute`` returns the prim's own
    attribute (creating the spec on the current edit target if it doesn't
    yet exist locally), and ``Set`` then writes the value into that local
    spec — so the value applies to *this* prim and is not echoed out to
    sibling clones via USD inherits.

    We rewrite ``xformOpOrder`` on every call to replace whatever the cloner
    inherited from env_0 (e.g. ``["xformOp:translate", "xformOp:orient",
    "xformOp:scale"]``) with a per-env order that only references the ops we
    actually author here. Passing ``yaw_rad`` keeps the orient op in that
    order and rewrites it as a rotation about Z; leaving it ``None`` drops
    orient and authors only translate and scale.
    """
    from pxr import Gf, Sdf, Vt

    translate_attr = prim.CreateAttribute(
        "xformOp:translate",
        Sdf.ValueTypeNames.Double3,
        custom=False,
    )
    translate_attr.Set(translate_value)

    order = ["xformOp:translate"]

    if yaw_rad is not None:
        half = 0.5 * float(yaw_rad)
        orient_attr = prim.CreateAttribute(
            "xformOp:orient",
            Sdf.ValueTypeNames.Quatd,
            custom=False,
        )
        orient_attr.Set(Gf.Quatd(math.cos(half), Gf.Vec3d(0.0, 0.0, math.sin(half))))
        order.append("xformOp:orient")

    scale_attr = prim.CreateAttribute(
        "xformOp:scale",
        Sdf.ValueTypeNames.Double3,
        custom=False,
    )
    scale_attr.Set(scale_value)
    order.append("xformOp:scale")

    order_attr = prim.CreateAttribute(
        "xformOpOrder",
        Sdf.ValueTypeNames.TokenArray,
        custom=False,
    )
    order_attr.Set(Vt.TokenArray(order))


def _get_or_create_xform_op(xformable, op_type, precision=None):
    """Legacy helper kept for backward compatibility.

    Do **not** use for prims that are USD-Inherits clones of a source env —
    use ``_author_local_xform`` (called from ``update_goal_region_visual``)
    instead, which bypasses the Xformable API and writes per-prim attributes
    directly so each clone has independent transform values.
    """
    from pxr import UsdGeom

    for op in xformable.GetOrderedXformOps():
        if op.GetOpType() == op_type:
            return op

    if precision is None:
        precision = UsdGeom.XformOp.PrecisionDouble

    return xformable.AddXformOp(op_type, precision)