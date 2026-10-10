"""Translucent cone shells ported from airflow_project/scripts/utils/airflow_viz.py.

Geometry/materials are cached; nozzle poses come from the simulation tensors,
not USD articulation transforms, so Fabric does not leave the nozzle stale.
"""
import math
import numpy as np
from pxr import UsdGeom, Gf, UsdShade, Sdf

from .jet_cone import (
    _centerline_velocity, _threshold_boundary_radius,
    _find_threshold_end_distance, _velocity_to_color, _normalize,
)

def _ensure_prim(stage, prim_path):
    prim = stage.GetPrimAtPath(prim_path)
    return prim if prim.IsValid() else None


def _get_or_create_translate_op(xformable):
    for op in xformable.GetOrderedXformOps():
        if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
            return op
    return xformable.AddTranslateOp()


def _get_or_create_orient_op(xformable):
    for op in xformable.GetOrderedXformOps():
        if op.GetOpType() == UsdGeom.XformOp.TypeOrient:
            return op
    return xformable.AddOrientOp()


def _rotation_to_quat(nozzle_dir):
    """
    Compute quaternion that rotates local +Z to nozzle_dir.
    Returns Gf.Quatf.
    """
    z_axis = np.array([0.0, 0.0, 1.0], dtype=np.float64)

    d = np.asarray(nozzle_dir, dtype=np.float64).reshape(3)
    n = np.linalg.norm(d)
    if n < 1e-12:
        return Gf.Quatf(1.0, 0.0, 0.0, 0.0)

    d = d / n
    dot = float(np.dot(z_axis, d))

    if dot > 0.9999:
        return Gf.Quatf(1.0, 0.0, 0.0, 0.0)

    if dot < -0.9999:
        # 180 deg around X.
        return Gf.Quatf(0.0, 1.0, 0.0, 0.0)

    cross = np.cross(z_axis, d)
    q = np.array([1.0 + dot, cross[0], cross[1], cross[2]], dtype=np.float64)
    q /= np.linalg.norm(q)

    return Gf.Quatf(float(q[0]), float(q[1]), float(q[2]), float(q[3]))


def _set_no_shadow_attrs(prim):
    prim.CreateAttribute(
        "primvars:doNotCastShadows",
        Sdf.ValueTypeNames.Bool,
    ).Set(True)

    prim.CreateAttribute(
        "primvars:omni:rtx:raytracing:visibility:shadow",
        Sdf.ValueTypeNames.Bool,
    ).Set(False)

    prim.CreateAttribute(
        "primvars:omni:rtx:raytracing:visibility:transmission",
        Sdf.ValueTypeNames.Bool,
    ).Set(False)

    prim.CreateAttribute(
        "primvars:omni:rtx:raytracing:visibility:refraction",
        Sdf.ValueTypeNames.Bool,
    ).Set(False)

    prim.CreateAttribute(
        "primvars:omni:rtx:raytracing:visibility:indirect",
        Sdf.ValueTypeNames.Bool,
    ).Set(False)


# ================================================================
# mesh / material helpers
# ================================================================

def _build_frustum_shell_mesh(
    z0,
    z1,
    r0,
    r1,
    num_segments,
    close_caps=False,
):
    """
    Build a local frustum shell aligned with +Z.

    Local geometry:
        ring 0 at z0 with radius r0
        ring 1 at z1 with radius r1

    Returns:
        points, face_vertex_counts, face_vertex_indices
    """
    points = []

    dtheta = 2.0 * math.pi / num_segments

    # lower ring
    for i in range(num_segments):
        theta = i * dtheta
        points.append(Gf.Vec3f(
            float(r0 * math.cos(theta)),
            float(r0 * math.sin(theta)),
            float(z0),
        ))

    # upper ring
    for i in range(num_segments):
        theta = i * dtheta
        points.append(Gf.Vec3f(
            float(r1 * math.cos(theta)),
            float(r1 * math.sin(theta)),
            float(z1),
        ))

    face_counts = []
    face_indices = []

    # side quads
    for i in range(num_segments):
        j = (i + 1) % num_segments

        lower_i = i
        lower_j = j
        upper_i = num_segments + i
        upper_j = num_segments + j

        face_counts.append(4)
        face_indices.extend([lower_i, lower_j, upper_j, upper_i])

    if close_caps:
        # Optional lower cap
        lower_center_idx = len(points)
        points.append(Gf.Vec3f(0.0, 0.0, float(z0)))

        for i in range(num_segments):
            j = (i + 1) % num_segments
            face_counts.append(3)
            face_indices.extend([lower_center_idx, j, i])

        # Optional upper cap
        upper_center_idx = len(points)
        points.append(Gf.Vec3f(0.0, 0.0, float(z1)))

        for i in range(num_segments):
            j = (i + 1) % num_segments
            face_counts.append(3)
            face_indices.extend([
                upper_center_idx,
                num_segments + i,
                num_segments + j,
            ])

    return points, face_counts, face_indices


def _create_or_update_mesh(
    stage,
    prim_path,
    points,
    face_counts,
    face_indices,
):
    prim = _ensure_prim(stage, prim_path)

    if prim is None:
        mesh = UsdGeom.Mesh.Define(stage, prim_path)
        prim = mesh.GetPrim()
    else:
        mesh = UsdGeom.Mesh(prim)

    mesh.CreatePointsAttr(points)
    mesh.CreateFaceVertexCountsAttr(face_counts)
    mesh.CreateFaceVertexIndicesAttr(face_indices)
    mesh.CreateDoubleSidedAttr(True)

    _set_no_shadow_attrs(prim)

    return mesh


def _ensure_transparent_material(stage, mat_path, color, alpha, emissive_strength=0.65):
    """
    Create/update a non-refractive translucent debug material.

    This is intentionally NOT a physically correct glass-like material:
    - no transmission
    - no ior/refraction
    - no clearcoat
    - no specular highlight
    - mostly emissive
    - alpha-blended viewport/debug visualization
    """
    mat_prim = stage.GetPrimAtPath(mat_path)

    if not mat_prim.IsValid():
        material = UsdShade.Material.Define(stage, mat_path)

        shader = UsdShade.Shader.Define(stage, f"{mat_path}/PreviewSurface")
        shader.CreateIdAttr("UsdPreviewSurface")

        material.CreateSurfaceOutput().ConnectToSource(
            shader.ConnectableAPI(),
            "surface",
        )
    else:
        material = UsdShade.Material(mat_prim)
        shader_prim = stage.GetPrimAtPath(f"{mat_path}/PreviewSurface")
        shader = UsdShade.Shader(shader_prim)

    color = np.asarray(color, dtype=np.float64).reshape(3)
    color = np.clip(color, 0.0, 1.0)

    diffuse = Gf.Vec3f(
        float(color[0]),
        float(color[1]),
        float(color[2]),
    )

    # 偏 debug overlay：主要靠 emissive 显色，而不是靠场景灯光。
    emissive = Gf.Vec3f(
        float(color[0] * emissive_strength),
        float(color[1] * emissive_strength),
        float(color[2] * emissive_strength),
    )

    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(diffuse)
    shader.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).Set(emissive)
    shader.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(float(alpha))

    # 关键：避免玻璃感、折射感、高光感。
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(1.0)
    shader.CreateInput("specularColor", Sdf.ValueTypeNames.Color3f).Set(
        Gf.Vec3f(0.0, 0.0, 0.0)
    )
    shader.CreateInput("clearcoat", Sdf.ValueTypeNames.Float).Set(0.0)
    shader.CreateInput("clearcoatRoughness", Sdf.ValueTypeNames.Float).Set(1.0)

    # 有些版本支持这些 PreviewSurface 输入；不支持时通常只是无效输入，不影响。
    shader.CreateInput("ior", Sdf.ValueTypeNames.Float).Set(1.0)
    shader.CreateInput("opacityThreshold", Sdf.ValueTypeNames.Float).Set(0.0)

    mat_prim = material.GetPrim()

    # Omniverse / Kit 常用透明材质 hint。
    mat_prim.CreateAttribute(
        "omni:kit:material:blendMode",
        Sdf.ValueTypeNames.Token,
    ).Set("transparent")

    mat_prim.CreateAttribute(
        "omni:kit:material:transparent",
        Sdf.ValueTypeNames.Bool,
    ).Set(True)

    # 禁止把它当成可折射/透射材质处理。
    mat_prim.CreateAttribute(
        "omni:kit:material:transmission",
        Sdf.ValueTypeNames.Float,
    ).Set(0.0)

    mat_prim.CreateAttribute(
        "omni:kit:material:ior",
        Sdf.ValueTypeNames.Float,
    ).Set(1.0)

    return material


class MeshJetConeVisualizer:
    """Project-style translucent shell bands, isolated per blower and environment."""

    def __init__(self, cfg, jet_cfg, prim_root):
        import omni.usd
        self.cfg = cfg
        self.jet_cfg = jet_cfg
        self._root = prim_root
        self._stage = omni.usd.get_context().get_stage()
        self._signatures = {}

    def update(self, nozzle_pos, nozzle_dir, intensity=1.0):
        self.update_many([nozzle_pos], [nozzle_dir], [intensity])

    def update_many(self, nozzle_positions, nozzle_dirs, intensities=None):
        cfg, jet = self.cfg, self.jet_cfg
        if intensities is None:
            intensities = [1.0] * len(nozzle_positions)
        active = set()
        for index, (pos, direction, intensity) in enumerate(zip(nozzle_positions, nozzle_dirs, intensities)):
            path = f"{self._root}/env_{index:04d}"
            axis = _normalize(tuple(float(v) for v in direction))
            if not cfg.show_jet_cone or axis is None:
                continue
            active.add(path)
            prim = self._stage.GetPrimAtPath(path)
            if not prim.IsValid():
                prim = UsdGeom.Xform.Define(self._stage, path).GetPrim()
                self._signatures.pop(path, None)
            xform = UsdGeom.Xformable(prim)
            _get_or_create_translate_op(xform).Set(Gf.Vec3d(*(float(v) for v in pos)))
            _get_or_create_orient_op(xform).Set(_rotation_to_quat(axis))
            visual_intensity = max(0.0, float(intensity)) if cfg.jet_cone_use_flow_intensity else 1.0
            signature = (
                visual_intensity,
                tuple(getattr(jet, name) for name in
                      ('U0', 'B', 'eps', 'nozzle_radius', 'spread_k', 'axial_model', 'xc', 'decay_k')),
                tuple(getattr(cfg, name) for name in
                      ('jet_cone_num_slices', 'jet_cone_num_segments', 'jet_cone_max_distance',
                       'jet_cone_velocity_threshold', 'jet_cone_sigma_scale', 'jet_cone_start_distance',
                       'jet_cone_min_alpha', 'jet_cone_max_alpha', 'jet_cone_alpha_gamma',
                       'jet_cone_emissive_strength', 'jet_cone_close_caps')),
            )
            if self._signatures.get(path) != signature:
                self._build_shell(path, visual_intensity)
                self._signatures[path] = signature
        for path in set(self._signatures) - active:
            self._stage.RemovePrim(path)
            del self._signatures[path]

    def _build_shell(self, path, intensity):
        cfg, jet = self.cfg, self.jet_cfg
        # Rebuild only when shape/material parameters change; poses update separately.
        for child in ('Bands', 'Materials'):
            child_path = f"{path}/{child}"
            if self._stage.GetPrimAtPath(child_path).IsValid():
                self._stage.RemovePrim(child_path)
            UsdGeom.Scope.Define(self._stage, child_path)
        bands = max(1, int(cfg.jet_cone_num_slices))
        segments = max(8, int(cfg.jet_cone_num_segments))
        start = cfg.jet_cone_start_distance
        end = _find_threshold_end_distance(
            jet, intensity, cfg.jet_cone_velocity_threshold, start,
            max(cfg.jet_cone_max_distance, start + 1e-4),
        )
        end = max(end, start + 1e-4)
        edges = [start + (end - start) * (i / bands) ** 2 for i in range(bands + 1)]
        for i, (z0, z1) in enumerate(zip(edges, edges[1:])):
            radii = [_threshold_boundary_radius(z, jet, intensity,
                     cfg.jet_cone_velocity_threshold, cfg.jet_cone_sigma_scale) for z in (z0, z1)]
            ratio = max(0.0, min(1.0, _centerline_velocity((z0 + z1) / 2, jet, intensity)
                                / max(float(jet.U0), 1e-12)))
            color = _velocity_to_color(ratio)
            alpha = cfg.jet_cone_min_alpha + (cfg.jet_cone_max_alpha - cfg.jet_cone_min_alpha) * ratio ** cfg.jet_cone_alpha_gamma
            alpha = max(0.0, min(1.0, alpha))
            points, counts, indices = _build_frustum_shell_mesh(
                z0, z1, *radii, segments, close_caps=cfg.jet_cone_close_caps,
            )
            mesh = _create_or_update_mesh(self._stage, f"{path}/Bands/band_{i:03d}", points, counts, indices)
            mesh.CreateSubdivisionSchemeAttr('none')
            material = _ensure_transparent_material(
                self._stage, f"{path}/Materials/mat_{i:03d}", color, alpha,
                cfg.jet_cone_emissive_strength,
            )
            UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material)
            mesh.CreateDisplayColorAttr([Gf.Vec3f(*color)])
            mesh.CreateDisplayOpacityAttr([alpha])
