# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Gripper-assisted box push with an additional cuboid heading requirement."""

from __future__ import annotations

import copy
import os

from blower.calibration import spawn_calibrated_table
from blower.mdp import AirflowTargetCfg
from blower.objects import CuboidCfg
from isaaclab.assets import AssetBaseCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.sim import RigidBodyMaterialCfg
from isaaclab.utils import configclass

from blower_arm_rl.tasks.manager_based.blower_arm_rl import mdp
from blower_arm_rl.tasks.manager_based.blower_arm_rl.mdp import cuboid_orientation as orientation
from blower_arm_rl.tasks.manager_based.blower_arm_rl.mdp.box_push_visual import HeadingLineCfg
from .blower_box_push_cfg import BOX_PUSH_PLAY_AIRFLOW_VIZ, _configure_airflow_visualizations
from .blower_gripper_assist_box_push_cfg import (
    GRIPPER_ASSIST_LEFT_RAIL_COLLISION_BOX_CENTERS,
    GRIPPER_ASSIST_RAIL_COLLISION_BOX_HALF_EXTENTS,
    GRIPPER_ASSIST_RIGHT_RAIL_COLLISION_BOX_CENTERS,
    PiperBlowerGripperAssistBoxPushEnvCfg,
)


@configclass
class CuboidOrientationTaskCfg:
    size: tuple[float, float, float] = (0.07, 0.07, 0.14)
    mass: float = 0.12
    # The 60 x 60 cm goal spans X [0.70, 1.40] and Y +-0.35 across resets.
    # Keep the measured table safety bounds, with room to overshoot the goal.
    workspace_bounds: tuple[float, float, float, float] = (0.10, 1.62, -0.52, 0.52)
    goal_half_extent: float = 0.30  # 60 x 60 cm, fixed throughout curriculum.
    # Keep goal-position sampling at the original stage-0 range throughout
    # training and play. Equal endpoints disable spatial expansion while the
    # heading, hold-time, and spawn curricula continue to advance normally.
    goal_center_range_start: tuple[float, float, float, float] = (0.90, 1.00, -0.05, 0.05)
    goal_center_range_final: tuple[float, float, float, float] = (0.90, 1.00, -0.05, 0.05)
    # Shifted 10 cm away from the robot. The X upper bound sits at the hard
    # ceiling the goal's near edge imposes (1.00 - 0.30 goal half width -
    # footprint - clearance, so 0.59 to 0.64 depending on the sampled
    # rotation); anything higher would be clamped away for most rotations.
    spawn_bounds: tuple = (
        (0.40, 0.60, -0.15, 0.15),
        (0.35, 0.60, -0.20, 0.20),
        (0.35, 0.60, -0.20, 0.20),
        (0.35, 0.60, -0.20, 0.20),
    )
    # All stages use arbitrary planar heading; spawn is physically outside goal.
    tolerance_deg: tuple[float, ...] = (30.0, 30.0, 15.0, 10.0)
    hold_s: tuple[float, ...] = (0.5, 0.5, 0.75, 1.0)
    # Commanded long-axis heading, uniform in center +- half range. The command
    # is mod 180 deg, so a 90 deg half range would cover every heading; this
    # caps out at 45 instead, reaching the two diagonals either side of the
    # center and never asking for a heading along the base X axis. Stage 0
    # starts narrower still, but wide enough that tracking cannot be ignored.
    command_center_deg: float = 90.0
    command_half_range_deg: tuple[float, ...] = (20.0, 45.0, 45.0, 45.0)
    # Marker for the commanded heading. Half length slightly exceeds the
    # cuboid's own half length (0.07) so both ends stay visible past the
    # object. The marker rides on the cuboid, which lying flat occupies z up to
    # 0.07, so the line has to hover above that or it is buried inside the box
    # and invisible -- which defeats the point of drawing it.
    heading_line_half_length: float = 0.10
    heading_line_height: float = 0.085
    spawn_clearance: float = 0.03
    spawn_height_clearance: float = 0.002
    table_height: float = 0.0
    flat_tolerance_deg: float = 10.0
    # Rolling to the next long face crosses a 45 deg edge balance, so stop just
    # short of it. Raise this if airflow gusts end episodes on harmless wobble;
    # lower it to catch a roll earlier.
    tumble_tolerance_deg: float = 40.0
    tumble_ignore_initial_s: float = 0.10
    # Milder than the -200 out-of-workspace penalty on purpose: tumbling is the
    # failure mode airflow is most likely to cause, and an equally harsh
    # penalty teaches the policy to keep the fan off instead of aiming better.
    tumble_penalty_weight: float = -50.0
    support_height_tolerance: float = 0.01
    max_center_height: float = 0.15
    linear_speed_tolerance: float = 0.025
    angular_speed_tolerance: float = 0.15
    angle_near_goal_scale: float = 0.20
    angle_progress_std: float = 0.35
    initial_level: int = 0
    evaluation: bool = False
    advance_success_rate: float = 0.75
    retreat_success_rate: float = 0.30
    min_completed_episodes: int = 256
    curriculum_warmup_steps: int = 3000
    curriculum_interval_steps: int = 2000
    position_noise_range: tuple[float, float] = (-0.01, 0.01)
    # Offline identification out/fit_135055 (session 20260923_135055_111042,
    # 23 real 50%/1 s pulses, ideal-fan assumption). The object friction was
    # fit with the support fixed at 0.3/0.2 and PhysX "average" combining, so
    # the two only make sense together: the object side absorbs the low table
    # value (effective static/dynamic ~0.90/0.79). Measured on an 8 cm cube.
    box_static_friction: float = 0.80
    box_dynamic_friction: float = 0.6
    table_static_friction: float = 0.3
    table_dynamic_friction: float = 0.2
    drag_coefficient: float = 1.40
    # PhysX rigid-body angular damping of the object. 0.05 is the PhysX default
    # the identification replay also starts from; airflow_project/scripts/
    # identify_cuboid.py prints the value to use here.
    box_angular_damping: float = 0.05
    # Object friction is resampled per reset in nominal * (1 -+ scale); the
    # table stays at its identified value.
    friction_randomization_scale: float = 0.2   


@configclass
class PiperBlowerGripperAssistCuboidOrientationEnvCfg(PiperBlowerGripperAssistBoxPushEnvCfg):
    cuboid_orientation: CuboidOrientationTaskCfg = CuboidOrientationTaskCfg()

    def __post_init__(self):
        super().__post_init__()
        task = self.cuboid_orientation
        stages = len(task.tolerance_deg)
        for name in ("hold_s", "command_half_range_deg", "spawn_bounds"):
            if len(getattr(task, name)) != stages:
                raise ValueError(f"cuboid_orientation.{name} must have one entry per curriculum stage ({stages})")
        x0, x1, y0, y1 = task.goal_center_range_final
        geom = self.box_push.geometry
        geom.goal_center = (0.5 * (x0 + x1), 0.5 * (y0 + y1), 0.006)
        geom.goal_radius_start = geom.goal_radius_final = task.goal_half_extent
        # Metadata only: reset_cuboid_orientation samples the band itself, and
        # the inherited goal-center curriculum stays disabled below.
        geom.goal_center_xy_range_start = task.goal_center_range_start
        geom.goal_center_xy_range = task.goal_center_range_final
        geom.workspace_bounds = task.workspace_bounds
        geom.box_aim_height = 0.035
        geom.box_half_extent = 0.035  # Nominal side face for legacy pre-push aiming.
        half_size = tuple(s / 2 for s in task.size)

        cuboid = CuboidCfg(size=task.size, mass=task.mass, color=(0.9, 0.35, 0.1),
                           spawn_z=task.size[0] / 2 + task.spawn_height_clearance)
        self.scene.box = cuboid.to_rigid_object_cfg("{ENV_REGEX_NS}/Box")
        self._apply_identified_physics()
        self.scene.box.init_state.pos = (0.40, 0.0, cuboid.spawn_z)
        self.scene.box.init_state.rot = (0.5, -0.5, 0.5, 0.5)
        self.scene.goal_region.spawn.half_extent = task.goal_half_extent
        self.scene.goal_region.init_state.pos = geom.goal_center
        # Line through the goal center showing the commanded long-axis heading.
        # Symmetric, with no arrow head: the command is an undirected axis, so
        # either end of the cuboid may point along it.
        self.scene.heading_command = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/HeadingCommand",
            spawn=HeadingLineCfg(half_length=task.heading_line_half_length),
            init_state=AssetBaseCfg.InitialStateCfg(
                pos=(geom.goal_center[0], geom.goal_center[1], task.heading_line_height),
                rot=(1.0, 0.0, 0.0, 0.0),
            ),
        )
        self.actions.airflow.target_type = cuboid.target_type
        self.actions.airflow.target_object_cfg = cuboid
        self.actions.airflow.targets = (AirflowTargetCfg(
            asset_name="box", target_type=cuboid.target_type, target_object_cfg=cuboid,
            sampling_cfg=self.actions.airflow.sampling_cfg,
        ),)
        self.events.reset_box = EventTerm(func=orientation.reset_cuboid_orientation, mode="reset")

        # Keep inherited position/flow/setup rewards and regularizers. Update
        # their task parameters after the parent has finished wiring them.
        for term in vars(self.rewards).values():
            params = getattr(term, "params", None)
            if not isinstance(params, dict):
                continue
            if "goal_center" in params:
                params["goal_center"] = geom.goal_center
            if "goal_radius" in params:
                params["goal_radius"] = task.goal_half_extent
            if "aim_height" in params:
                params["aim_height"] = geom.box_aim_height
            if "box_half_extent" in params:
                params["box_half_extent"] = geom.box_half_extent
        for name in ("box_goal_dense", "box_goal_progress", "goal_inside", "goal_centering",
                     "airflow_directed_force", "arm_action_deviation"):
            term = getattr(self.rewards, name, None)
            if term is not None:
                term.params["box_half_extent"] = half_size
        self.rewards.box_goal_dense.params["std"] = 0.25  # Broad enough at far reset distances.
        self.rewards.goal_inside.weight = 2.0  # Position alone is useful but not the complete task.
        self.rewards.goal_stable = RewTerm(func=orientation.cuboid_orientation_stable,
                                          weight=self.box_push.rewards.goal_stable_weight)
        self.rewards.orientation_dense = RewTerm(func=orientation.cuboid_orientation_dense, weight=1.0)
        self.rewards.orientation_progress = RewTerm(func=orientation.cuboid_orientation_progress, weight=2.0)
        self.rewards.pose_inside = RewTerm(func=orientation.cuboid_pose_inside, weight=4.0)
        self.rewards.box_out_of_workspace.func = orientation.cuboid_out_of_workspace_penalty
        self.rewards.box_out_of_workspace.params = {}
        self.rewards.box_tumbled = RewTerm(func=orientation.cuboid_tumbled_penalty,
                                          weight=task.tumble_penalty_weight)
        # Failure penalties are added after the ordinary sum is clamped at zero.
        self.unclipped_reward_terms = tuple(self.unclipped_reward_terms) + ("box_tumbled",)
        # Home-posture regularization should wait for the complete position+angle goal.
        self.rewards.arm_action_deviation.func = orientation.cuboid_arm_action_deviation
        self.rewards.arm_action_deviation.params = {}

        self.observations.policy.goal_pos.params["goal_center"] = geom.goal_center
        self.observations.policy.goal_orientation = ObsTerm(func=orientation.cuboid_goal_observation)
        self.observations.policy.box_pose = ObsTerm(func=orientation.cuboid_policy_pose)
        self.observations.critic.box_state = ObsTerm(func=orientation.cuboid_privileged_state)
        self.curriculum.goal_region = None
        self.curriculum.orientation = CurrTerm(func=orientation.cuboid_orientation_curriculum)
        # Retain the original robot reset ranges; the new pose curriculum updates
        # the same reset-scale buffer rather than running two conflicting curricula.
        self.terminations.box_out_of_workspace = DoneTerm(func=orientation.cuboid_out_of_workspace)
        self.terminations.box_tumbled = DoneTerm(func=orientation.cuboid_tumbled)
        self.terminations.link6_collision_box_cube_contact.params["target_box_half_extents"] = half_size

        # Airflow has to do the reorienting, so no part of the end effector may
        # touch the cuboid. link6 already terminated on contact; the two gripper
        # rails were the remaining exception, and they end the episode too now.
        contact = dict(self.terminations.link6_collision_box_cube_contact.params)
        for name, body, centers in (
            ("left_rail_box_cube_contact", "link7", GRIPPER_ASSIST_LEFT_RAIL_COLLISION_BOX_CENTERS),
            ("right_rail_box_cube_contact", "link8", GRIPPER_ASSIST_RIGHT_RAIL_COLLISION_BOX_CENTERS),
        ):
            setattr(self.terminations, name, DoneTerm(
                func=mdp.body_local_boxes_box_contact_force,
                params={**contact,
                        "robot_cfg": SceneEntityCfg("robot", body_names=[body]),
                        "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[body]),
                        "box_centers": centers,
                        "box_half_extents": GRIPPER_ASSIST_RAIL_COLLISION_BOX_HALF_EXTENTS,
                        "box_quats": None},
            ))
        # Inherit 10 s and success=None; the gripper-assist contact allowance is gone.

    def _apply_identified_physics(self):
        """Identified contact/drag physics; replaces the inherited box-push friction range."""
        task = self.cuboid_orientation
        if os.environ.get("AIRFLOW_CALIBRATION_PROFILE"):
            # The parent would calibrate its 8 cm box, which the cuboid then replaces.
            raise ValueError("Cuboid task carries its identified physics in CuboidOrientationTaskCfg; "
                             "unset AIRFLOW_CALIBRATION_PROFILE")

        def material(static, dynamic):
            return RigidBodyMaterialCfg(static_friction=static, dynamic_friction=dynamic, restitution=0.0,
                                        friction_combine_mode="average", restitution_combine_mode="average")

        self.scene.box.spawn.physics_material = material(task.box_static_friction, task.box_dynamic_friction)
        self.scene.box.spawn.rigid_props.angular_damping = task.box_angular_damping
        # The table is a USD asset, so bind to its colliders after import.
        self.scene.table.spawn.calibration_support_material = material(
            task.table_static_friction, task.table_dynamic_friction)
        self.scene.table.spawn.func = spawn_calibrated_table
        self.actions.airflow.drag_cfg = copy.deepcopy(self.actions.airflow.drag_cfg)
        self.actions.airflow.drag_cfg.C_D = task.drag_coefficient
        scale = task.friction_randomization_scale
        self.events.randomize_box_material.params.update(
            static_friction_range=(task.box_static_friction * (1 - scale), task.box_static_friction * (1 + scale)),
            dynamic_friction_range=(task.box_dynamic_friction * (1 - scale), task.box_dynamic_friction * (1 + scale)),
        )


@configclass
class PiperBlowerGripperAssistCuboidOrientationEnvCfg_PLAY(PiperBlowerGripperAssistCuboidOrientationEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.cuboid_orientation.initial_level = len(self.cuboid_orientation.tolerance_deg) - 1
        self.cuboid_orientation.evaluation = True
        self.scene.num_envs = 16
        self.observations.policy.enable_corruption = False
        self.events.reset_robot_joints.params["initial_range_scale"] = 0.0
        self.events.reset_robot_joints.params["velocity_range"] = (0.0, 0.0)
        self.events.reset_robot_joints.params["max_resample_attempts"] = 1
        self.events.randomize_robot_actuator_gains = None
        self.events.randomize_box_material = None
        self.events.randomize_box_mass = None
        self.actions.airflow.airflow_u0_range = None
        self.actions.airflow.airflow_spread_k_range = None
        _configure_airflow_visualizations(self, **BOX_PUSH_PLAY_AIRFLOW_VIZ)
