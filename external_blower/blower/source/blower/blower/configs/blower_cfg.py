"""Blower asset configurations for Isaac Lab.

This module provides pre-configured asset definitions for blower devices.
Users can import these directly into their InteractiveSceneCfg or customize them.

Usage example::

    from blower.assets import SIMPLE_BLOWER_CFG

    @configclass
    class MySceneCfg(InteractiveSceneCfg):
        blower = SIMPLE_BLOWER_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Blower",
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0.5, 0.0, 0.5)),
        )
"""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.sim.spawners.shapes import CylinderCfg


SIMPLE_BLOWER_CFG = RigidObjectCfg(
    prim_path="{ENV_REGEX_NS}/Blower",
    spawn=sim_utils.CylinderCfg(
        radius=0.05,
        height=0.2,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            kinematic_enabled=True,  # Blower is typically kinematic (fixed or controlled externally)
        ),
        collision_props=sim_utils.CollisionPropertiesCfg(
            collision_enabled=False,  # Blower doesn't need collision by default
        ),
        visual_material=sim_utils.PreviewSurfaceCfg(
            diffuse_color=(0.2, 0.2, 0.8),  # Blue color for visibility
        ),
    ),
    init_state=RigidObjectCfg.InitialStateCfg(
        pos=(0.5, 0.0, 0.5),
        rot=(1.0, 0.0, 0.0, 0.0),  # Identity quaternion, blower pointing along +X
    ),
)
"""A simple cylinder-shaped blower asset.

The blower is kinematic by default (no physics simulation on itself),
and points along the +X direction. Users should adjust ``init_state``
to position and orient it in their scene.
"""
