"""Re-usable blower asset definitions for Isaac Lab scenes."""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.sim.spawners.shapes import CylinderCfg

SIMPLE_BLOWER_CFG = RigidObjectCfg(
    prim_path='{ENV_REGEX_NS}/Blower',
    spawn=CylinderCfg(
        radius=0.05,
        height=0.2,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            kinematic_enabled=True,
        ),
        collision_props=sim_utils.CollisionPropertiesCfg(
            collision_enabled=False,
        ),
        visual_material=sim_utils.PreviewSurfaceCfg(
            diffuse_color=(0.2, 0.2, 0.8),
        ),
    ),
    init_state=RigidObjectCfg.InitialStateCfg(
        pos=(0.5, 0.0, 0.5),
        rot=(1.0, 0.0, 0.0, 0.0),
    ),
)
