# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Piper-L articulation and blower-mount configuration."""

import math
from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import DCMotorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.sim.schemas import activate_contact_sensors
from isaaclab.utils import configclass
from pxr import PhysxSchema, Usd, UsdPhysics


PIPER_L_ASSET_DIR = Path(__file__).resolve().parent / "assets" / "piper_l"
PIPER_L_INSTANCEABLE_USD_PATH = PIPER_L_ASSET_DIR / "piper_l_blower_fixed.usd"

PIPER_ARM_JOINTS = ["joint[1-6]"]
PIPER_GRIPPER_JOINTS = ["joint[78]"]
PIPER_CONTACT_BODY_PATTERN = "link[4-8]"
PIPER_BOX_PROXIMITY_BODIES = ("link6", "link7", "link8")

PIPER_END_EFFECTOR_BODY = "link6"
PIPER_NOZZLE_OFFSET_POS = (0.08, 0.0, 0.05)
PIPER_NOZZLE_OFFSET_QUAT = (1.0, 0.0, 0.0, 0.0)
PIPER_NOZZLE_LOCAL_AXIS = "z"
PIPER_BASE_HEIGHT_ABOVE_TABLE = 0.03
PIPER_CONTROL_DT = 1.0 / 30.0
PIPER_CONTROL_DECIMATION = 4
PIPER_SIM_DT = PIPER_CONTROL_DT / PIPER_CONTROL_DECIMATION

PIPER_DEFAULT_ARM_JOINT_POS = {
    "joint1": 0.0,
    "joint2": 1.5,
    "joint3": -1.3,
    "joint4": 0.0,
    "joint5": 0.5,
    "joint6": 0.0,
}
"""Single source of truth for zero raw arm action and observation baseline."""

PIPER_DEFAULT_JOINT_POS = {
    **PIPER_DEFAULT_ARM_JOINT_POS,
    "joint[78]": 0.0,
}
"""Default full robot joint pose used by IsaacLab default_joint_pos."""

PIPER_LOCKED_GRIPPER_JOINTS = ("joint7", "joint8")
PIPER_LOCKED_GRIPPER_POS = 0.0
PIPER_ARM_JOINT_LIMITS_RAD = {
    "joint1": (-2.618, 2.618),
    "joint2": (0.0, 3.14),
    "joint3": (-2.697, 0.0),
    "joint4": (-1.832, 1.832),
    "joint5": (-1.22, 1.22),
    "joint6": (-3.14, 3.14),
}

PIPER_ARM_ACTUATOR_CFG = DCMotorCfg(
    joint_names_expr=["joint[1-3]"],
    saturation_effort=32.0,
    effort_limit=32.0,
    velocity_limit=5.0,
    stiffness={
        "joint1": 150.0,
        "joint2": 150.0,
        "joint3": 150.0,
    },
    damping={
        "joint1": 10.0,
        "joint2": 10.0,
        "joint3": 10.0,
    },
    friction={".*": 0.0},
    dynamic_friction={".*": 0.0},
    viscous_friction={".*": 0.0},
)

PIPER_WRIST_ACTUATOR_CFG = DCMotorCfg(
    joint_names_expr=["joint[4-6]"],
    saturation_effort=8.0,
    effort_limit=8.0,
    velocity_limit=5.0,
    stiffness={
        "joint4": 10.0,
        "joint5": 10.0,
        "joint6": 10.0,
    },
    damping={
        "joint4": 0.8,
        "joint5": 0.8,
        "joint6": 0.8,
    },
    friction={".*": 0.0},
    dynamic_friction={".*": 0.0},
    viscous_friction={".*": 0.0},
)

PIPER_GRIPPER_ACTUATOR_CFG = DCMotorCfg(
    joint_names_expr=["joint[78]"],
    saturation_effort=2.0,
    effort_limit=2.0,
    velocity_limit=2.0,
    stiffness={"joint7": 10.0, "joint8": 10.0},
    damping={"joint7": 0.8, "joint8": 0.8},
    friction={".*": 0.0},
    dynamic_friction={".*": 0.0},
    viscous_friction={".*": 0.0},
)


@configclass
class PiperSim2RealParamsCfg:
    """Sim2real parameter-search bounds matching the Piper PACE setup."""

    joint_order: list[str] = [
        "joint1",
        "joint2",
        "joint3",
        "joint4",
        "joint5",
        "joint6",
        "joint7",
    ]
    bounds_params: list[list[float]] | None = None

    def __post_init__(self):
        num_joints = len(self.joint_order)
        bounds_params = [[0.0, 0.0] for _ in range(4 * num_joints + 1)]

        for joint_id in range(num_joints):
            # Armature [kg m^2].
            bounds_params[joint_id] = [1e-5, 1.0]

            # Viscous friction [N m s/rad] (or [N s/m] for the gripper).
            bounds_params[num_joints + joint_id] = [0.0, 1.0]

            # Static/dynamic friction [N m] (or [N] for the gripper).
            bounds_params[2 * num_joints + joint_id] = [0.0, 1.0]

            # Encoder bias: fixed to zero for all joints in this task.
            bounds_params[3 * num_joints + joint_id] = [0.0, 0.0]

        # Global actuation delay [simulation steps].
        bounds_params[4 * num_joints] = [0.0, 2.0]
        self.bounds_params = bounds_params


def spawn_piper_l_instanceable(
    prim_path: str,
    cfg: sim_utils.UsdFileCfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **kwargs,
):
    """Spawn Piper from instanceable USD and fix its real base link."""
    spawn_cfg = cfg.copy()
    articulation_props = spawn_cfg.articulation_props
    spawn_cfg.articulation_props = None
    prim = sim_utils.spawn_from_usd(prim_path, spawn_cfg, translation, orientation, **kwargs)

    stage = prim.GetStage()
    for robot_prim in sim_utils.find_matching_prims(prim_path, stage=stage):
        robot_path = robot_prim.GetPath().pathString
        if spawn_cfg.activate_contact_sensors:
            _make_piper_links_editable(stage, robot_path)
        rigid_root_prim = next(
            (
                candidate_prim
                for candidate_prim in Usd.PrimRange(robot_prim)
                if candidate_prim.HasAPI(UsdPhysics.RigidBodyAPI)
                and candidate_prim.HasAPI(UsdPhysics.ArticulationRootAPI)
            ),
            None,
        )
        if rigid_root_prim is None:
            raise RuntimeError(f"Could not find Piper rigid articulation root below: {robot_path}")

        # Recreate the imported world joint so its frames include calibrated
        # root transforms instead of snapping the root body back to identity.
        imported_root_joint_paths = [
            candidate_prim.GetPath()
            for candidate_prim in Usd.PrimRange(robot_prim)
            if candidate_prim.IsA(UsdPhysics.FixedJoint)
            and rigid_root_prim.GetPath() in UsdPhysics.FixedJoint(candidate_prim).GetBody1Rel().GetTargets()
            and not UsdPhysics.FixedJoint(candidate_prim).GetBody0Rel().GetTargets()
        ]
        for joint_path in imported_root_joint_paths:
            stage.GetPrimAtPath(joint_path).SetActive(False)

        # Converted USDs may keep articulation APIs on non-rigid wrapper prims.
        # Remove those so the fixed-base articulation can be created on the
        # first actual rigid body link in both old and new USD layouts.
        for candidate_prim in Usd.PrimRange(robot_prim):
            is_rigid_body = candidate_prim.HasAPI(UsdPhysics.RigidBodyAPI)
            if is_rigid_body:
                continue
            if candidate_prim.HasAPI(UsdPhysics.ArticulationRootAPI):
                candidate_prim.RemoveAPI(UsdPhysics.ArticulationRootAPI)
            if candidate_prim.HasAPI(PhysxSchema.PhysxArticulationAPI):
                candidate_prim.RemoveAPI(PhysxSchema.PhysxArticulationAPI)

        if articulation_props is not None:
            sim_utils.modify_articulation_root_properties(
                rigid_root_prim.GetPath().pathString,
                articulation_props,
                stage=stage,
            )
        if spawn_cfg.activate_contact_sensors:
            _activate_piper_contact_sensors(stage, robot_path)
        _set_piper_arm_joint_limits(stage, robot_path)
        _lock_piper_gripper_joints(stage, robot_path)

    return prim


def _make_piper_links_editable(stage, robot_path: str) -> None:
    """Expand component-asset instances so per-link physics APIs can be authored."""
    root = stage.GetPrimAtPath(robot_path)
    if not root:
        return

    # De-instancing can reveal another nested instance, so repeat until the
    # complete articulation hierarchy is editable.
    while True:
        instance_prims = [prim for prim in Usd.PrimRange(root) if prim.IsInstance()]
        if not instance_prims:
            return
        changed = False
        for instance_prim in instance_prims:
            changed = instance_prim.SetInstanceable(False) or changed
        if not changed:
            raise RuntimeError(f"Could not make Piper instance hierarchy editable below: {robot_path}")


def _activate_piper_contact_sensors(stage, robot_path: str) -> None:
    """Enable contact reports on every rigid link in nested imported USDs."""
    root = stage.GetPrimAtPath(robot_path)
    if not root:
        return

    rigid_body_paths = [
        prim.GetPath().pathString
        for prim in Usd.PrimRange(root)
        if prim.HasAPI(UsdPhysics.RigidBodyAPI)
    ]
    if not rigid_body_paths:
        raise RuntimeError(f"Could not find Piper rigid bodies below: {robot_path}")

    # IsaacLab's helper stops descending after the first rigid body. Converted
    # MJCF assets can nest link rigid bodies, so activate each one explicitly.
    for rigid_body_path in rigid_body_paths:
        activate_contact_sensors(rigid_body_path, stage=stage)


def _set_piper_arm_joint_limits(stage, robot_path: str) -> None:
    """Restore physical limits when a system-identification USD has unbounded joints."""
    root = stage.GetPrimAtPath(robot_path)
    if not root:
        return

    for prim in Usd.PrimRange(root):
        limits_rad = PIPER_ARM_JOINT_LIMITS_RAD.get(prim.GetName())
        if limits_rad is None:
            continue

        joint = UsdPhysics.RevoluteJoint(prim)
        if not joint:
            continue

        joint.CreateLowerLimitAttr().Set(math.degrees(limits_rad[0]))
        joint.CreateUpperLimitAttr().Set(math.degrees(limits_rad[1]))


def _lock_piper_gripper_joints(stage, robot_path: str) -> None:
    """Force the gripper slide joints to stay closed at zero."""
    root = stage.GetPrimAtPath(robot_path)
    if not root:
        return

    for prim in Usd.PrimRange(root):
        if prim.GetName() not in PIPER_LOCKED_GRIPPER_JOINTS:
            continue

        joint = UsdPhysics.PrismaticJoint(prim)
        if not joint:
            continue

        joint.CreateLowerLimitAttr().Set(PIPER_LOCKED_GRIPPER_POS)
        joint.CreateUpperLimitAttr().Set(PIPER_LOCKED_GRIPPER_POS)


PIPER_L_CFG = ArticulationCfg(
    # ``fix_root_link`` moves ArticulationRootAPI from the rigid base link to
    # its parent, producing the fixed-base structure expected by PhysX.
    articulation_root_prim_path="/base_link",
    spawn=sim_utils.UsdFileCfg(
        func=spawn_piper_l_instanceable,
        usd_path=str(PIPER_L_INSTANCEABLE_USD_PATH),
        activate_contact_sensors=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=1,
            fix_root_link=True,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, PIPER_BASE_HEIGHT_ABOVE_TABLE),
        joint_pos=PIPER_DEFAULT_JOINT_POS,
    ),
    actuators={
        "piper_shoulder": PIPER_ARM_ACTUATOR_CFG,
        "piper_forearm": PIPER_WRIST_ACTUATOR_CFG,
        "piper_gripper": PIPER_GRIPPER_ACTUATOR_CFG,
    },
    soft_joint_pos_limit_factor=0.95,
)
"""Piper-L with a fixed base, position-controlled arm, and locked gripper."""
