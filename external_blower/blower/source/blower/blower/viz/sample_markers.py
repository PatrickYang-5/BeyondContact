"""Sample point debug markers for blower targets."""

from __future__ import annotations

import omni.usd
from pxr import Gf, UsdGeom
from blower.configs.airflow_cfg import VizCfg


def _ensure_xform(stage, path: str):
    prim = stage.GetPrimAtPath(path)
    return prim if prim.IsValid() else UsdGeom.Xform.Define(stage, path).GetPrim()


def _get_or_create_translate_op(xformable):
    for op in xformable.GetOrderedXformOps():
        if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
            return op
    return xformable.AddTranslateOp()


class SampleMarkerVisualizer:
    def __init__(self, cfg: VizCfg, prim_root: str = '/World/Debug/SampleMarkers'):
        self.cfg = cfg
        self.stage = omni.usd.get_context().get_stage()
        self.root_path = prim_root
        _ensure_xform(self.stage, prim_root)

    def update(self, sample_positions_world: list[tuple[float, float, float]] | None):
        if not self.cfg.show_sample_markers or sample_positions_world is None:
            return
        for idx, p in enumerate(sample_positions_world):
            marker_path = f'{self.root_path}/marker_{idx:04d}'
            prim = self.stage.GetPrimAtPath(marker_path)
            if not prim.IsValid():
                sphere = UsdGeom.Sphere.Define(self.stage, marker_path)
                sphere.CreateRadiusAttr(self.cfg.sample_marker_radius)
                sphere.CreateDisplayColorAttr([Gf.Vec3f(*[float(c) for c in self.cfg.sample_marker_color])])
                _get_or_create_translate_op(UsdGeom.Xformable(sphere.GetPrim()))
            xformable = UsdGeom.Xformable(self.stage.GetPrimAtPath(marker_path))
            _get_or_create_translate_op(xformable).Set(Gf.Vec3d(float(p[0]), float(p[1]), float(p[2])))
        idx = len(sample_positions_world)
        while True:
            marker_path = f'{self.root_path}/marker_{idx:04d}'
            prim = self.stage.GetPrimAtPath(marker_path)
            if not prim.IsValid():
                break
            _get_or_create_translate_op(UsdGeom.Xformable(prim)).Set(Gf.Vec3d(0.0, 0.0, -1000.0))
            idx += 1
