"""Surface sample generation and body-to-world transforms for blower targets."""

from __future__ import annotations

import math
import warnings
from pathlib import Path

import numpy as np
import torch

from blower.configs.airflow_cfg import SamplingCfg


def _axis_frame_from_label(axis_label: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    frames = {
        'x':  ([0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]),
        '-x': ([0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [-1.0, 0.0, 0.0]),
        'y':  ([1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]),
        '-y': ([1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]),
        'z':  ([1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]),
        '-z': ([1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]),
    }
    key = axis_label.lower()
    if key not in frames:
        raise ValueError(f'Unsupported axis_label: {axis_label}')
    return tuple(np.array(v, dtype=np.float64) for v in frames[key])


def _cell_center_values(a: float, b: float, n: int) -> np.ndarray:
    n = max(1, int(n))
    width = float(b) - float(a)
    return np.asarray([float(a) + (i + 0.5) * width / n for i in range(n)], dtype=np.float64)


def cylinder_surface_area(radius: float, height: float) -> float:
    radius = float(radius)
    height = float(height)
    return 2.0 * math.pi * radius * height + 2.0 * math.pi * radius ** 2


def _tensor_from_numpy(array: np.ndarray, device: str = 'cpu') -> torch.Tensor:
    return torch.as_tensor(array, dtype=torch.float32, device=device)


def is_omnidirectional_normal(normals: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    return torch.norm(normals, dim=-1) < eps


def build_box_surface_samples(size: tuple[float, float, float], sampling_cfg: SamplingCfg | None = None):
    if sampling_cfg is None:
        sampling_cfg = SamplingCfg()

    sx, sy, sz = map(float, size)
    hx, hy, hz = sx * 0.5, sy * 0.5, sz * 0.5
    target_area = sampling_cfg.target_area
    min_n = sampling_cfg.min_n
    max_n = sampling_cfg.max_n

    positions = []
    normals = []
    areas = []

    def counts_for_face(a: float, b: float) -> tuple[int, int]:
        face_area = a * b
        total_n = max(1, int(math.ceil(face_area / max(target_area, 1e-12))))
        aspect = max(a, 1e-12) / max(b, 1e-12)
        nu = int(math.ceil(math.sqrt(total_n * aspect)))
        nv = int(math.ceil(total_n / max(nu, 1)))
        nu = max(min_n, min(max_n, nu))
        nv = max(min_n, min(max_n, nv))
        return nu, nv

    def add_face_pair(half: float, face_area: float, grid_a: np.ndarray, grid_b: np.ndarray, axis_idx: int):
        area_each = face_area / (len(grid_a) * len(grid_b))
        for a in grid_a:
            for b in grid_b:
                for sign in (+1.0, -1.0):
                    pos = np.zeros(3, dtype=np.float64)
                    nrm = np.zeros(3, dtype=np.float64)
                    pos[axis_idx] = sign * half
                    nrm[axis_idx] = sign
                    others = [i for i in range(3) if i != axis_idx]
                    pos[others[0]] = a
                    pos[others[1]] = b
                    positions.append(pos)
                    normals.append(nrm)
                    areas.append(area_each)

    ny, nz = counts_for_face(sy, sz)
    add_face_pair(hx, sy * sz, _cell_center_values(-hy, hy, ny), _cell_center_values(-hz, hz, nz), axis_idx=0)

    nx, nz = counts_for_face(sx, sz)
    add_face_pair(hy, sx * sz, _cell_center_values(-hx, hx, nx), _cell_center_values(-hz, hz, nz), axis_idx=1)

    nx, ny = counts_for_face(sx, sy)
    add_face_pair(hz, sx * sy, _cell_center_values(-hx, hx, nx), _cell_center_values(-hy, hy, ny), axis_idx=2)

    return {
        'positions_body': _tensor_from_numpy(np.vstack(positions)),
        'normals_body': _tensor_from_numpy(np.vstack(normals)),
        'areas': _tensor_from_numpy(np.array(areas, dtype=np.float32)),
    }


def build_sphere_surface_samples(radius: float, sampling_cfg: SamplingCfg | None = None):
    if sampling_cfg is None:
        sampling_cfg = SamplingCfg()

    radius = float(radius)
    total_area = 4.0 * math.pi * radius * radius
    n_points = max(12, int(math.ceil(total_area / max(sampling_cfg.target_area, 1e-12))))
    area_each = total_area / n_points
    points = []
    normals = []
    areas = []

    golden_angle = math.pi * (3.0 - math.sqrt(5.0))
    for i in range(n_points):
        y = 1.0 - 2.0 * (i + 0.5) / n_points
        r_xy = math.sqrt(max(0.0, 1.0 - y * y))
        phi = i * golden_angle
        n = np.array([r_xy * math.cos(phi), y, r_xy * math.sin(phi)], dtype=np.float64)
        points.append(radius * n)
        normals.append(n)
        areas.append(area_each)

    return {
        'positions_body': _tensor_from_numpy(np.vstack(points)),
        'normals_body': _tensor_from_numpy(np.vstack(normals)),
        'areas': _tensor_from_numpy(np.array(areas, dtype=np.float32)),
    }


def build_cylinder_surface_samples(radius: float, height: float, axis: str = 'z', sampling_cfg: SamplingCfg | None = None):
    if sampling_cfg is None:
        sampling_cfg = SamplingCfg()

    radius = float(radius)
    height = float(height)
    spacing = sampling_cfg.spacing

    u, v, w = _axis_frame_from_label(axis)
    positions = []
    normals = []
    areas = []

    circumference = 2.0 * math.pi * radius
    ntheta = max(
        sampling_cfg.min_ntheta,
        min(
            sampling_cfg.max_ntheta,
            max(1, int(math.ceil(circumference / max(spacing, 1e-12)))),
        ),
    )
    nz = max(
        sampling_cfg.min_nz,
        min(
            sampling_cfg.max_nz,
            max(1, int(math.ceil(height / max(spacing, 1e-12)))),
        ),
    )

    dz = height / nz
    z_values = np.array([-0.5 * height + (i + 0.5) * dz for i in range(nz)], dtype=np.float64)
    area_side = (circumference * dz) / ntheta
    dtheta = 2.0 * math.pi / ntheta

    for z in z_values:
        for k in range(ntheta):
            theta = (k + 0.5) * dtheta
            radial_dir = math.cos(theta) * u + math.sin(theta) * v
            positions.append(radius * radial_dir + z * w)
            normals.append(radial_dir)
            areas.append(area_side)

    nr = max(1, int(math.ceil(radius / max(spacing, 1e-12))))
    dr = radius / nr
    for cap_sign in (-1.0, +1.0):
        z_cap = cap_sign * 0.5 * height
        normal = cap_sign * w
        for ir in range(nr):
            r0 = ir * dr
            r1 = min(radius, (ir + 1) * dr)
            r_mid = 0.5 * (r0 + r1)
            ring_area = math.pi * (r1 * r1 - r0 * r0)
            if r_mid < 1e-10:
                positions.append(z_cap * w)
                normals.append(normal)
                areas.append(ring_area)
                continue
            ntheta_cap = max(
                sampling_cfg.min_ntheta,
                min(
                    sampling_cfg.max_ntheta,
                    max(1, int(math.ceil(2.0 * math.pi * r_mid / max(spacing, 1e-12)))),
                ),
            )
            dtheta_cap = 2.0 * math.pi / ntheta_cap
            for k in range(ntheta_cap):
                theta = (k + 0.5) * dtheta_cap
                radial_dir = math.cos(theta) * u + math.sin(theta) * v
                positions.append(r_mid * radial_dir + z_cap * w)
                normals.append(normal)
                areas.append(ring_area / ntheta_cap)

    return {
        'positions_body': _tensor_from_numpy(np.vstack(positions)),
        'normals_body': _tensor_from_numpy(np.vstack(normals)),
        'areas': _tensor_from_numpy(np.array(areas, dtype=np.float32)),
    }


def _load_trimesh_geometry(mesh_path: str | Path):
    try:
        import trimesh
    except ImportError as exc:
        raise ImportError(
            "Mesh surface sampling requires the optional dependency 'trimesh'. "
            "Install it or disable mesh sampling to use the primitive fallback."
        ) from exc

    mesh = trimesh.load(str(mesh_path), force='scene')
    if isinstance(mesh, trimesh.Scene):
        if hasattr(mesh, 'to_geometry'):
            mesh = mesh.to_geometry()
        else:
            mesh = mesh.dump(concatenate=True)

    if mesh is None or getattr(mesh, 'is_empty', False):
        raise ValueError(f'No triangle mesh geometry found in {mesh_path}')
    if len(getattr(mesh, 'faces', ())) == 0:
        raise ValueError(f'Mesh has no faces: {mesh_path}')
    return mesh


def _triangle_area_data(vertices: np.ndarray, faces: np.ndarray):
    triangles = vertices[faces]
    cross = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    double_area = np.linalg.norm(cross, axis=1)
    valid = double_area > 1e-14
    areas = 0.5 * double_area[valid]
    normals = np.zeros((int(valid.sum()), 3), dtype=np.float64)
    normals[:] = cross[valid] / double_area[valid, None]
    return faces[valid], areas, normals


def _sample_triangle_surface_even(
    vertices: np.ndarray,
    faces: np.ndarray,
    sampling_cfg: SamplingCfg,
    representative_area: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    faces, face_areas, face_normals = _triangle_area_data(vertices, faces)
    total_area = float(face_areas.sum())
    if not np.isfinite(total_area) or total_area <= 0.0:
        raise ValueError('Mesh has non-positive surface area.')

    sample_area = total_area if representative_area is None else float(representative_area)
    if not np.isfinite(sample_area) or sample_area <= 0.0:
        raise ValueError('Representative mesh area must be positive.')

    count = max(1, int(math.ceil(sample_area / max(sampling_cfg.target_area, 1e-12))))
    oversample_factor = max(1, int(getattr(sampling_cfg, 'mesh_oversample_factor', 8)))
    candidate_count = max(count, count * oversample_factor)
    rng = np.random.default_rng(int(getattr(sampling_cfg, 'mesh_seed', 0)))

    cdf = np.cumsum(face_areas)
    face_u = (np.arange(candidate_count, dtype=np.float64) + rng.random(candidate_count))
    face_u *= total_area / candidate_count
    rng.shuffle(face_u)
    candidate_face_ids = np.searchsorted(cdf, face_u, side='right')
    candidate_face_ids = np.clip(candidate_face_ids, 0, len(faces) - 1)

    r1 = rng.random(candidate_count)
    r2 = rng.random(candidate_count)
    sqrt_r1 = np.sqrt(r1)
    b0 = 1.0 - sqrt_r1
    b1 = sqrt_r1 * (1.0 - r2)
    b2 = sqrt_r1 * r2

    tri = vertices[faces[candidate_face_ids]]
    candidates = (
        b0[:, None] * tri[:, 0]
        + b1[:, None] * tri[:, 1]
        + b2[:, None] * tri[:, 2]
    )

    selected = _farthest_point_indices(candidates, count, rng)
    points = candidates[selected]
    normals = face_normals[candidate_face_ids[selected]]
    areas = np.full(len(points), sample_area / len(points), dtype=np.float32)
    return points, normals, areas, sample_area


def _farthest_point_indices(points: np.ndarray, count: int, rng: np.random.Generator) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    if len(points) <= count:
        return np.arange(len(points), dtype=np.int64)

    selected = np.empty(count, dtype=np.int64)
    first = int(rng.integers(0, len(points)))
    selected[0] = first
    min_dist2 = np.sum((points - points[first]) ** 2, axis=1)
    min_dist2[first] = -1.0

    for i in range(1, count):
        idx = int(np.argmax(min_dist2))
        selected[i] = idx
        dist2 = np.sum((points - points[idx]) ** 2, axis=1)
        min_dist2 = np.minimum(min_dist2, dist2)
        min_dist2[selected[: i + 1]] = -1.0

    return selected


def _prim_world_transform_matrix(prim) -> np.ndarray:
    """Return a USD prim's world transform as a column-vector 4x4 matrix."""
    from pxr import Usd, UsdGeom

    xformable = UsdGeom.Xformable(prim)
    world_xform = xformable.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    return np.array(world_xform, dtype=np.float64).T


def _iter_usd_mesh_prims(root_prim):
    """Yield mesh prims under root_prim, including root_prim itself."""
    from pxr import UsdGeom

    stack = [root_prim]
    while stack:
        prim = stack.pop()
        if prim.IsA(UsdGeom.Mesh):
            yield prim
        stack.extend(reversed(list(prim.GetChildren())))


def _triangulated_face_indices(face_counts: np.ndarray, face_indices: np.ndarray) -> np.ndarray:
    """Triangulate USD polygon faces with a fan from each face's first vertex."""
    triangles = []
    offset = 0
    for count in face_counts:
        count = int(count)
        face = face_indices[offset: offset + count]
        offset += count
        if count < 3:
            continue
        for i in range(1, count - 1):
            triangles.append((int(face[0]), int(face[i]), int(face[i + 1])))
    return np.asarray(triangles, dtype=np.int64)


def _usd_mesh_triangles_in_target_body(target_prim_path: str) -> tuple[np.ndarray, np.ndarray]:
    """Extract all mesh triangles below a spawned USD target prim in its body frame."""
    import omni.usd
    from pxr import UsdGeom

    stage = omni.usd.get_context().get_stage()
    if stage is None:
        raise RuntimeError('No active USD stage is available for mesh sampling.')

    target_prim = stage.GetPrimAtPath(str(target_prim_path))
    if not target_prim.IsValid():
        raise RuntimeError(f'Invalid target prim path: {target_prim_path}')

    target_world = _prim_world_transform_matrix(target_prim)
    target_world_inv = np.linalg.inv(target_world)

    vertices_all = []
    faces_all = []
    vertex_offset = 0

    for mesh_prim in _iter_usd_mesh_prims(target_prim):
        mesh = UsdGeom.Mesh(mesh_prim)
        points_attr = mesh.GetPointsAttr().Get()
        counts_attr = mesh.GetFaceVertexCountsAttr().Get()
        indices_attr = mesh.GetFaceVertexIndicesAttr().Get()
        if points_attr is None or counts_attr is None or indices_attr is None:
            continue

        points = np.asarray(points_attr, dtype=np.float64)
        face_counts = np.asarray(counts_attr, dtype=np.int64)
        face_indices = np.asarray(indices_attr, dtype=np.int64)
        faces = _triangulated_face_indices(face_counts, face_indices)
        if len(points) == 0 or len(faces) == 0:
            continue

        mesh_world = _prim_world_transform_matrix(mesh_prim)
        mesh_to_target = target_world_inv @ mesh_world
        points_h = np.concatenate(
            [points, np.ones((len(points), 1), dtype=np.float64)],
            axis=1,
        )
        points_body = (mesh_to_target @ points_h.T).T[:, :3]

        vertices_all.append(points_body)
        faces_all.append(faces + vertex_offset)
        vertex_offset += len(points_body)

    if not vertices_all:
        raise ValueError(f'No mesh triangles found under {target_prim_path}')

    return np.vstack(vertices_all), np.vstack(faces_all)


def build_mesh_surface_samples(
    mesh_path: str | Path,
    sampling_cfg: SamplingCfg | None = None,
    scale: tuple[float, float, float] | None = None,
    representative_area: float | None = None,
):
    """Sample a triangle mesh surface into body-frame airflow primitives.

    The mesh is sampled approximately evenly over surface area using trimesh's
    triangle geometry. Each point receives its containing triangle's face normal.
    Detailed meshes can pass a representative area so visual mesh detail does
    not change point density or total integrated force.
    """
    if sampling_cfg is None:
        sampling_cfg = SamplingCfg()

    mesh = _load_trimesh_geometry(mesh_path)
    if scale is not None:
        mesh = mesh.copy()
        mesh.apply_scale(np.asarray(scale, dtype=np.float64).reshape(3))

    points, normals, areas, _ = _sample_triangle_surface_even(
        np.asarray(mesh.vertices, dtype=np.float64),
        np.asarray(mesh.faces, dtype=np.int64),
        sampling_cfg,
        representative_area=representative_area,
    )

    return {
        'positions_body': _tensor_from_numpy(points),
        'normals_body': _tensor_from_numpy(normals),
        'areas': _tensor_from_numpy(areas),
    }


def build_mesh_surface_samples_for_prim(
    target_prim_path: str,
    sampling_cfg: SamplingCfg | None = None,
    representative_area: float | None = None,
):
    """Sample the actual USD mesh hierarchy under a spawned target prim."""
    if sampling_cfg is None:
        sampling_cfg = SamplingCfg()

    vertices, faces = _usd_mesh_triangles_in_target_body(target_prim_path)
    points, normals, areas, _ = _sample_triangle_surface_even(
        vertices,
        faces,
        sampling_cfg,
        representative_area=representative_area,
    )

    return {
        'positions_body': _tensor_from_numpy(points),
        'normals_body': _tensor_from_numpy(normals),
        'areas': _tensor_from_numpy(areas),
    }


def build_bottle_mesh_or_cylinder_samples(
    mesh_path: str | Path,
    sampling_cfg: SamplingCfg | None = None,
    scale: tuple[float, float, float] | None = None,
    fallback_radius: float = 0.034,
    fallback_height: float = 0.20,
    fallback_axis: str = 'z',
    fallback_offset_body: tuple[float, float, float] = (0.021, 0.0, 0.11),
):
    """Build bottle samples from the mesh, falling back to the old cylinder."""
    if sampling_cfg is None:
        sampling_cfg = SamplingCfg()

    representative_area = cylinder_surface_area(fallback_radius, fallback_height)
    try:
        return build_mesh_surface_samples(
            mesh_path,
            sampling_cfg=sampling_cfg,
            scale=scale,
            representative_area=representative_area,
        )
    except Exception as exc:
        warnings.warn(
            f'Bottle mesh sampling failed ({exc}); falling back to cylinder approximation.',
            RuntimeWarning,
            stacklevel=2,
        )
        sample_batch = build_cylinder_surface_samples(
            fallback_radius,
            fallback_height,
            axis=fallback_axis,
            sampling_cfg=sampling_cfg,
        )
        return offset_samples_in_body(sample_batch, fallback_offset_body)


def build_line_surface_samples(length: float, axis: str = 'y', sampling_cfg: SamplingCfg | None = None):
    if sampling_cfg is None:
        sampling_cfg = SamplingCfg()

    length = float(length)
    spacing = sampling_cfg.spacing
    n_points = max(2, int(math.ceil(length / max(spacing, 1e-12))) + 1)
    _, _, w = _axis_frame_from_label(axis)
    positions = []
    normals = []
    areas = []
    for i in range(n_points):
        t = -0.5 * length + (i / max(n_points - 1, 1)) * length
        positions.append(t * w)
        normals.append(np.zeros(3, dtype=np.float64))
        areas.append(sampling_cfg.target_area)
    return {
        'positions_body': _tensor_from_numpy(np.vstack(positions)),
        'normals_body': _tensor_from_numpy(np.vstack(normals)),
        'areas': _tensor_from_numpy(np.array(areas, dtype=np.float32)),
    }


def offset_samples_in_body(sample_batch: dict, offset_body: tuple[float, float, float]):
    offset = torch.as_tensor(offset_body, dtype=torch.float32)
    return {
        'positions_body': sample_batch['positions_body'] + offset.unsqueeze(0),
        'normals_body': sample_batch['normals_body'],
        'areas': sample_batch['areas'],
    }


def _quat_to_rotation_matrix(quat: torch.Tensor) -> torch.Tensor:
    w, x, y, z = quat.unbind(-1)
    xx = x * x
    yy = y * y
    zz = z * z
    xy = x * y
    xz = x * z
    yz = y * z
    wx = w * x
    wy = w * y
    wz = w * z

    return torch.stack([
        torch.stack([1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy)], dim=-1),
        torch.stack([2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx)], dim=-1),
        torch.stack([2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy)], dim=-1),
    ], dim=-2)


def transform_samples_to_world(root_pos: torch.Tensor, root_quat: torch.Tensor, sample_batch: dict):
    root_pos = root_pos.to(dtype=torch.float32)
    root_quat = root_quat.to(dtype=torch.float32)
    positions_body = sample_batch['positions_body']
    normals_body = sample_batch['normals_body']
    areas = sample_batch['areas']
    num_envs = root_pos.shape[0]
    num_samples = positions_body.shape[0]

    R = _quat_to_rotation_matrix(root_quat)
    positions_body_expanded = positions_body.unsqueeze(0).expand(num_envs, num_samples, 3)
    normals_body_expanded = normals_body.unsqueeze(0).expand(num_envs, num_samples, 3)
    R_t = R.transpose(-2, -1).unsqueeze(1)

    positions_world = torch.matmul(
        positions_body_expanded.unsqueeze(-2),
        R_t,
    ).squeeze(-2) + root_pos.unsqueeze(1)

    normals_rotated = torch.matmul(
        normals_body_expanded.unsqueeze(-2),
        R_t,
    ).squeeze(-2)

    omni_mask = is_omnidirectional_normal(normals_body)
    omni_mask = omni_mask.unsqueeze(0).expand(num_envs, num_samples)
    normals_world = torch.where(
        omni_mask.unsqueeze(-1),
        torch.zeros_like(normals_rotated),
        torch.nn.functional.normalize(normals_rotated, dim=-1, eps=1e-9),
    )

    return {
        'positions_world': positions_world,
        'normals_world': normals_world,
        'areas': areas,
        'is_omni': omni_mask,
    }


def build_default_samples_for_type(target_type: str, sampling_cfg: SamplingCfg | None = None):
    target_type = target_type.lower()
    if sampling_cfg is None:
        sampling_cfg = SamplingCfg()

    if target_type == 'box':
        return build_box_surface_samples((0.05, 0.05, 0.05), sampling_cfg)
    if target_type == 'cuboid':
        return build_box_surface_samples((0.07, 0.07, 0.14), sampling_cfg)
    if target_type == 'ball':
        return build_sphere_surface_samples(0.025, sampling_cfg)
    if target_type == 'bottle':
        mesh_path = Path(__file__).resolve().parents[4] / 'assets' / 'plastic_water_bottle.glb'
        return build_bottle_mesh_or_cylinder_samples(mesh_path, sampling_cfg=sampling_cfg)
    if target_type == 'cigarette':
        return build_line_surface_samples(0.085, axis='y', sampling_cfg=sampling_cfg)
    raise ValueError(f'Unsupported target_type: {target_type}')
