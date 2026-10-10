"""Target object factory configs for blower scenes."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import MISSING
from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
from isaaclab.sim.spawners.shapes import CuboidCfg as SimBoxCfg, CylinderCfg as SimCylinderCfg, SphereCfg as SimSphereCfg
from isaaclab.sim.utils import clone
from isaaclab.utils import configclass

from blower.configs.airflow_cfg import SamplingCfg
from blower.models.sampling import (
    build_box_surface_samples,
    build_bottle_mesh_or_cylinder_samples,
    build_cylinder_surface_samples,
    build_line_surface_samples,
    build_sphere_surface_samples,
    offset_samples_in_body,
)

_ASSETS_DIR = Path(__file__).resolve().parents[4] / 'assets'


def _quat_from_euler_xyz_deg(euler_xyz_deg: tuple[float, float, float]) -> tuple[float, float, float, float]:
    roll, pitch, yaw = (math.radians(float(v)) for v in euler_xyz_deg)
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    return (
        cy * cr * cp + sy * sr * sp,
        cy * sr * cp - sy * cr * sp,
        cy * cr * sp + sy * sr * cp,
        sy * cr * cp - cy * sr * sp,
    )


def _apply_collision_recursive(prim, approximation: str) -> None:
    from pxr import UsdGeom, UsdPhysics

    if prim.IsA(UsdGeom.Mesh):
        UsdPhysics.CollisionAPI.Apply(prim)
        UsdPhysics.MeshCollisionAPI.Apply(prim).GetApproximationAttr().Set(approximation)
    for child in prim.GetChildren():
        _apply_collision_recursive(child, approximation)


@clone
def spawn_referenced_rigid_asset(
    prim_path: str,
    cfg: 'ReferencedRigidAssetCfg',
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
):
    import isaacsim.core.utils.prims as prim_utils
    from isaaclab.sim import schemas

    if prim_utils.is_prim_path_valid(prim_path):
        raise ValueError(f"A prim already exists at path: '{prim_path}'.")

    prim = prim_utils.create_prim(
        prim_path,
        prim_type='Xform',
        usd_path=cfg.asset_path,
        translation=translation,
        orientation=orientation,
        scale=cfg.scale,
    )

    if cfg.collision_props is not None:
        _apply_collision_recursive(prim, cfg.collision_approximation)
        schemas.modify_collision_properties(prim_path, cfg.collision_props)
    if cfg.mass_props is not None:
        schemas.define_mass_properties(prim_path, cfg.mass_props)
    if cfg.rigid_props is not None:
        schemas.define_rigid_body_properties(prim_path, cfg.rigid_props)
    return prim


@configclass
class ReferencedRigidAssetCfg(RigidObjectSpawnerCfg):
    func: Callable = spawn_referenced_rigid_asset
    asset_path: str = MISSING
    scale: tuple[float, float, float] | None = None
    collision_approximation: str = 'convexDecomposition'


@configclass
class BallCfg:
    radius: float = 0.025
    mass: float = 0.05
    color: tuple[float, float, float] = (1.0, 0.2, 0.2)
    spawn_z: float = 0.03

    @property
    def target_type(self) -> str:
        return 'ball'

    def to_rigid_object_cfg(self, prim_path: str) -> RigidObjectCfg:
        return RigidObjectCfg(
            prim_path=prim_path,
            spawn=SimSphereCfg(
                radius=self.radius,
                mass_props=sim_utils.MassPropertiesCfg(mass=self.mass),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    rigid_body_enabled=True,
                    kinematic_enabled=False,
                ),
                collision_props=sim_utils.CollisionPropertiesCfg(
                    collision_enabled=True,
                ),
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=self.color,
                ),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=(0.0, 0.0, self.spawn_z),
                rot=(1.0, 0.0, 0.0, 0.0),
            ),
        )

    def build_body_samples(self, sampling_cfg: SamplingCfg | None = None):
        if sampling_cfg is None:
            sampling_cfg = SamplingCfg()
        return build_sphere_surface_samples(self.radius, sampling_cfg)

@configclass
class BoxCfg:
    size: tuple[float, float, float] = (0.05, 0.05, 0.05)
    mass: float = 0.05
    color: tuple[float, float, float] = (1.0, 0.2, 0.2)
    spawn_z: float = 0.03

    @property
    def target_type(self) -> str:
        return 'box'

    def to_rigid_object_cfg(self, prim_path: str) -> RigidObjectCfg:
        return RigidObjectCfg(
            prim_path=prim_path,
            spawn=SimBoxCfg(
                size=self.size,
                mass_props=sim_utils.MassPropertiesCfg(mass=self.mass),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    rigid_body_enabled=True,
                    kinematic_enabled=False,
                ),
                collision_props=sim_utils.CollisionPropertiesCfg(
                    collision_enabled=True,
                ),
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=self.color,
                ),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=(0.0, 0.0, self.spawn_z),
                rot=(1.0, 0.0, 0.0, 0.0),
            ),
        )

    def build_body_samples(self, sampling_cfg: SamplingCfg | None = None):
        if sampling_cfg is None:
            sampling_cfg = SamplingCfg()
        return build_box_surface_samples(self.size, sampling_cfg)

@configclass
class CuboidCfg(BoxCfg):
    """7 x 7 x 14 cm target using the same surface quadrature as BoxCfg.

    The long side is local Z. Mass and material defaults are inherited from
    BoxCfg; the initial center leaves 5 mm clearance above the ground.
    """

    size: tuple[float, float, float] = (0.07, 0.07, 0.14)
    spawn_z: float = 0.075

    @property
    def target_type(self) -> str:
        return 'cuboid'


@configclass
class BottleCfg:
    cylinder_radius: float = 0.034
    cylinder_height: float = 0.20
    mass: float = 0.05
    color: tuple[float, float, float] = (0.8, 0.8, 1.0)
    spawn_z: float = 0.10683
    offset_body: tuple[float, float, float] = (0.021, 0.0, 0.11)
    use_asset: bool = True
    asset_filename: str = 'plastic_water_bottle.glb'
    asset_path: str | None = None
    scale: tuple[float, float, float] = (1.0, 1.0, 1.0)
    rotate_xyz: tuple[float, float, float] = (0.0, 0.0, 90.0)
    collision_approximation: str = 'convexDecomposition'
    use_mesh_samples: bool = True

    @property
    def target_type(self) -> str:
        return 'bottle'

    @property
    def resolved_asset_path(self) -> str:
        if self.asset_path:
            return str(Path(self.asset_path).expanduser().resolve())
        return str((_ASSETS_DIR / self.asset_filename).resolve())

    def to_rigid_object_cfg(self, prim_path: str) -> RigidObjectCfg:
        if self.use_asset:
            return RigidObjectCfg(
                prim_path=prim_path,
                spawn=ReferencedRigidAssetCfg(
                    asset_path=self.resolved_asset_path,
                    scale=self.scale,
                    mass_props=sim_utils.MassPropertiesCfg(mass=self.mass),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        rigid_body_enabled=True,
                        kinematic_enabled=False,
                    ),
                    collision_props=sim_utils.CollisionPropertiesCfg(
                        collision_enabled=True,
                    ),
                    collision_approximation=self.collision_approximation,
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(0.0, 0.0, self.spawn_z),
                    rot=_quat_from_euler_xyz_deg(self.rotate_xyz),
                ),
            )

        return RigidObjectCfg(
            prim_path=prim_path,
            spawn=SimCylinderCfg(
                radius=self.cylinder_radius,
                height=self.cylinder_height,
                mass_props=sim_utils.MassPropertiesCfg(mass=self.mass),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    rigid_body_enabled=True,
                    kinematic_enabled=False,
                ),
                collision_props=sim_utils.CollisionPropertiesCfg(
                    collision_enabled=True,
                ),
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=self.color,
                ),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=(0.0, 0.0, self.spawn_z),
                rot=(1.0, 0.0, 0.0, 0.0),
            ),
        )

    def build_body_samples(self, sampling_cfg: SamplingCfg | None = None):
        if sampling_cfg is None:
            sampling_cfg = SamplingCfg()
        if self.use_asset and self.use_mesh_samples:
            return build_bottle_mesh_or_cylinder_samples(
                self.resolved_asset_path,
                sampling_cfg=sampling_cfg,
                scale=self.scale,
                fallback_radius=self.cylinder_radius,
                fallback_height=self.cylinder_height,
                fallback_axis='z',
                fallback_offset_body=self.offset_body,
            )
        sample_batch = build_cylinder_surface_samples(
            radius=self.cylinder_radius,
            height=self.cylinder_height,
            axis='z',
            sampling_cfg=sampling_cfg,
        )
        return offset_samples_in_body(sample_batch, self.offset_body)

@configclass
class CigaretteCfg:
    length: float = 0.085
    radius: float = 0.003
    mass: float = 0.01
    color: tuple[float, float, float] = (0.8, 0.8, 0.2)
    spawn_z: float = 0.05

    @property
    def target_type(self) -> str:
        return 'cigarette'

    def to_rigid_object_cfg(self, prim_path: str) -> RigidObjectCfg:
        return RigidObjectCfg(
            prim_path=prim_path,
            spawn=SimCylinderCfg(
                radius=self.radius,
                height=self.length,
                axis='Y',
                mass_props=sim_utils.MassPropertiesCfg(mass=self.mass),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    rigid_body_enabled=True,
                    kinematic_enabled=False,
                ),
                collision_props=sim_utils.CollisionPropertiesCfg(
                    collision_enabled=True,
                ),
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=self.color,
                ),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=(0.0, 0.0, self.spawn_z),
                rot=(1.0, 0.0, 0.0, 0.0),
            ),
        )

    def build_body_samples(self, sampling_cfg: SamplingCfg | None = None):
        if sampling_cfg is None:
            sampling_cfg = SamplingCfg()
        return build_line_surface_samples(self.length, axis='y', sampling_cfg=sampling_cfg)


def build_target_object_samples(target_type: str, sampling_cfg: SamplingCfg | None = None):
    target_type = target_type.lower()
    if sampling_cfg is None:
        sampling_cfg = SamplingCfg()

    if target_type == 'box':
        return build_box_surface_samples((0.05, 0.05, 0.05), sampling_cfg)
    if target_type == 'cuboid':
        return CuboidCfg().build_body_samples(sampling_cfg)
    if target_type == 'ball':
        return build_sphere_surface_samples(0.025, sampling_cfg)
    if target_type == 'bottle':
        return BottleCfg().build_body_samples(sampling_cfg)
    if target_type == 'cigarette':
        return build_line_surface_samples(0.085, axis='y', sampling_cfg=sampling_cfg)
    raise ValueError(f'Unsupported target_type: {target_type}')
