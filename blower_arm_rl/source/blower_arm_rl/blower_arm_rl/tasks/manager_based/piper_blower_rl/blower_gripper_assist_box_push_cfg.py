# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Piper box-push variant that allows gripper-assisted setup pushes."""

from __future__ import annotations

import math
import os

from blower.configs import JetCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from blower_arm_rl.tasks.manager_based.blower_arm_rl import mdp

from .blower_box_push_cfg import (
    BOX_PUSH_PLAY_AIRFLOW_VIZ,
    BOX_PUSH_TRAIN_AIRFLOW_VIZ,
    _apply_play_final_curriculum_state,
    _configure_airflow_visualizations,
)
from .blower_box_push_env_cfg import PiperBlowerBoxPushInitEnvCfg
from .piper_l_cfg import PIPER_L_ASSET_DIR, PIPER_LOCKED_GRIPPER_POS


GOAL_CENTER_XY_RANGE_START = (0.55, 0.75, -0.08, 0.08)
GOAL_CENTER_XY_RANGE_FINAL = (0.40, 1.40, -0.15, 0.15)
BOX_SPAWN_BOUNDS = (0.15, 0.60, -0.20, 0.20)
WORKSPACE_BOUNDS = (0.05, 1.65, -0.30, 0.30)
PIPER_GRIPPER_ASSIST_BLOWER_USD_PATH = PIPER_L_ASSET_DIR / "piper_l_blower_fixed.usd"

GRIPPER_ASSIST_LOW_HEIGHT_RESET = 0.01
GRIPPER_ASSIST_COLLISION_CORNER_MIN_HEIGHT = 0.01
GRIPPER_ASSIST_LINK6_COLLISION_BOX_CENTERS = (
    (0.0, 0.0, 0.06),
    (0.068, 0.0, -0.0065),
    (-0.0760961, -0.03, 0.032),
)
GRIPPER_ASSIST_LINK6_COLLISION_BOX_HALF_EXTENTS = (
    (0.09, 0.01, 0.01),
    (0.0315, 0.0535, 0.0285),
    (0.017, 0.045, 0.0125),
)
GRIPPER_ASSIST_LINK6_COLLISION_BOX_QUATS = (
    (0.500398, -0.5, -0.499602, 0.5),
    (0.707, 0.707, 0.0, 0.0),
    (0.819648, 0.0, -0.572867, 0.0),
)
GRIPPER_ASSIST_LEFT_RAIL_COLLISION_BOX_CENTERS = ((0.0, -0.025, -0.003),)
GRIPPER_ASSIST_RIGHT_RAIL_COLLISION_BOX_CENTERS = ((0.0, 0.025, -0.003),)
GRIPPER_ASSIST_RAIL_COLLISION_BOX_HALF_EXTENTS = ((0.01, 0.03, 0.002),)

GRIPPER_ASSIST_GOAL_RADIUS_START = 0.12
GRIPPER_ASSIST_GOAL_RADIUS_FINAL = 0.10
GRIPPER_ASSIST_BOX_HALF_EXTENT = 0.04

GRIPPER_ASSIST_BOX_MASS_RANDOMIZATION_RANGE = (0.08, 0.15)
GRIPPER_ASSIST_BOX_STATIC_FRICTION_RANDOMIZATION_RANGE = (0.8, 1.50)
GRIPPER_ASSIST_BOX_DYNAMIC_FRICTION_RANDOMIZATION_RANGE = (0.7, 1.40)
GRIPPER_ASSIST_BOX_RESTITUTION_RANDOMIZATION_RANGE = (0.0, 0.0)
GRIPPER_ASSIST_BOX_MATERIAL_RANDOMIZATION_BUCKETS = 32
GRIPPER_ASSIST_AIRFLOW_JET_CFG = JetCfg(
    nozzle_radius=0.04,
    spread_k=0.0764,
    U0=27.0,
    axial_model="exponential",
    xc=0.2,
    decay_k=2.65,
)
GRIPPER_ASSIST_AIRFLOW_REACTION_FORCE_MAX = (
    0.5
    * 1.225
    * GRIPPER_ASSIST_AIRFLOW_JET_CFG.U0**2
    * math.pi
    * GRIPPER_ASSIST_AIRFLOW_JET_CFG.nozzle_radius**2
)
GRIPPER_ASSIST_AIRFLOW_U0_RANDOMIZATION_RANGE = (24.3, 29.7)
GRIPPER_ASSIST_AIRFLOW_SPREAD_K_RANDOMIZATION_RANGE = (0.0688, 0.0840)
GRIPPER_ASSIST_AIRFLOW_RANDOMIZATION_BUCKETS = 4


@configclass
class PiperBlowerGripperAssistBoxPushEnvCfg(PiperBlowerBoxPushInitEnvCfg):
    """Box-push task where the end effector may contact the box.

    The goal region stays in the current far workspace, while the box reset
    distribution expands toward the robot so the policy can learn to nudge bad
    initial poses with the gripper before switching to airflow.
    """

    # Sum ordinary rewards, clamp to zero, then add failure penalties.
    only_positive_rewards: bool = True
    unclipped_reward_terms: tuple[str, ...] = ("box_out_of_workspace",)

    def __post_init__(self):
        # Start from the current far target band and expand x with curriculum.
        self.box_push.geometry.goal_center_xy_range = GOAL_CENTER_XY_RANGE_FINAL
        self.box_push.geometry.goal_center_xy_range_start = GOAL_CENTER_XY_RANGE_START
        self.box_push.geometry.goal_radius_start = GRIPPER_ASSIST_GOAL_RADIUS_START
        self.box_push.geometry.goal_radius_final = GRIPPER_ASSIST_GOAL_RADIUS_FINAL
        self.box_push.geometry.workspace_bounds = WORKSPACE_BOUNDS
        self.box_push.geometry.box_half_extent = GRIPPER_ASSIST_BOX_HALF_EXTENT
        self.box_push.geometry.box_aim_height = 0.06

        # This task overrides reset_box after base wiring: box spawn is an
        # independent rectangle instead of an annulus around the sampled goal.
        self.box_push.reset.box_radial_distance_range = (0.06, 0.48)
        self.box_push.reset.box_angle_center = math.pi
        self.box_push.reset.box_angle_span_start = math.pi
        self.box_push.reset.box_angle_span_final = math.pi
        self.box_push.reset.box_clearance = 0.015

        self.box_push.reset.robot_joint_position_ranges.update(
            {
                "joint1": (-1.10, 1.10),
                "joint2": (0.00, 0.90),
                "joint3": (-0.90, 0.00),
                "joint4": (-1.20, 1.20),
                "joint5": (-1.10, 0.70),
                "joint6": (-1.10, 1.10),
                "joint[78]": (PIPER_LOCKED_GRIPPER_POS, PIPER_LOCKED_GRIPPER_POS),
            }
        )
        self.box_push.reset.robot_joint_velocity_range = (-0.02, 0.02)
        self.box_push.reset.robot_joint_limit_margin = 0.04
        self.box_push.reset.robot_max_resample_attempts = 24
        self.box_push.reset.robot_enforce_self_collision_safety = False

        self.box_push.prepush.nozzle_standoff = 0.12
        self.box_push.prepush.nozzle_height = 0.06
        # Keep a close nominal pose, but give exploratory poses useful credit.
        self.box_push.prepush.position_std = 0.08
        self.box_push.prepush.height_std = 0.05
        self.box_push.prepush.axis_std = 0.15
        # Ready = range suitability * goalward jet direction * box-center aim.
        # Distance is 3D nozzle-to-box-center distance, not surface clearance.
        self.box_push.prepush.ready_geometry = {
            "distance": 0.12,
            "distance_std": 0.08,
            "direction_std_deg": 30.0,
            "aim_std_deg": 15.0,
        }

        self.box_push.domain_randomization.box_mass_range = GRIPPER_ASSIST_BOX_MASS_RANDOMIZATION_RANGE
        self.box_push.domain_randomization.box_static_friction_range = (
            GRIPPER_ASSIST_BOX_STATIC_FRICTION_RANDOMIZATION_RANGE
        )
        self.box_push.domain_randomization.box_dynamic_friction_range = (
            GRIPPER_ASSIST_BOX_DYNAMIC_FRICTION_RANDOMIZATION_RANGE
        )
        self.box_push.domain_randomization.box_restitution_range = GRIPPER_ASSIST_BOX_RESTITUTION_RANDOMIZATION_RANGE
        self.box_push.domain_randomization.box_material_buckets = GRIPPER_ASSIST_BOX_MATERIAL_RANDOMIZATION_BUCKETS

        self.box_push.airflow.reaction_force_max = GRIPPER_ASSIST_AIRFLOW_REACTION_FORCE_MAX
        self.box_push.airflow.u0_range = GRIPPER_ASSIST_AIRFLOW_U0_RANDOMIZATION_RANGE
        self.box_push.airflow.spread_k_range = GRIPPER_ASSIST_AIRFLOW_SPREAD_K_RANDOMIZATION_RANGE
        self.box_push.airflow.randomization_buckets = GRIPPER_ASSIST_AIRFLOW_RANDOMIZATION_BUCKETS

        self.box_push.curriculum.success_threshold = 0.72
        self.box_push.curriculum.failure_threshold = 0.25
        self.box_push.curriculum.angle_range_step = 0.0
        self.box_push.curriculum.expand_angle_before_shrinking_goal = False
        self.box_push.curriculum.robot_reset_initial_scale = 1.0
        self.box_push.curriculum.robot_reset_shrink_enabled = True
        self.box_push.curriculum.robot_reset_success_threshold = 0.85
        self.box_push.curriculum.robot_reset_failure_threshold = 0.35
        self.box_push.curriculum.robot_reset_shrink_step = 0.08
        self.box_push.curriculum.robot_reset_grow_step = 0.04
        self.box_push.curriculum.robot_reset_final_scale = 0.0
        self.box_push.curriculum.robot_reset_warmup_steps = 8000
        self.box_push.curriculum.robot_reset_min_steps_between_updates = 2000
        self.box_push.curriculum.goal_center_range_curriculum_enabled = True
        self.box_push.curriculum.goal_center_range_success_threshold = 0.70
        self.box_push.curriculum.goal_center_range_failure_threshold = 0.18
        self.box_push.curriculum.goal_center_range_step = 0.10
        self.box_push.curriculum.goal_center_range_retreat_step = 0.05
        self.box_push.curriculum.goal_center_range_final_scale = 1.0
        self.box_push.curriculum.goal_center_range_warmup_steps = 3000
        self.box_push.curriculum.goal_center_range_min_steps_between_updates = 1500

        # Keep task progress dominant, with close upstream positioning and precise aiming.
        self.box_push.rewards.alive_weight = 0.0
        self.box_push.rewards.box_goal_dense_weight = 2.0
        self.box_push.rewards.box_goal_progress_weight = 3.0
        self.box_push.rewards.box_velocity_toward_goal_weight = 0.0  # Previous weight: 2.0; duplicates goal progress.
        self.box_push.rewards.goal_inside_weight = 8.0  # Previous weight: 4.0; covered by gated centering/stability.
        self.box_push.rewards.goal_centering_weight = 0.0   # Previous weight: 3.0;
        self.box_push.rewards.goal_centering_std = 0.055
        self.box_push.rewards.goal_stable_weight = 2.0
        self.box_push.rewards.goal_stable_std = 0.055
        self.box_push.rewards.goal_escape_min_hold_steps = 1
        self.box_push.rewards.success_weight = 0.0
        self.box_push.rewards.goal_escape_weight = 0.0

        self.box_push.rewards.blower_box_range_weight = 0.05
        self.box_push.rewards.blower_box_range_distance = 0.12
        self.box_push.rewards.blower_box_range_std = 0.09
        # Small setup reward; useful airflow motion remains the main blower reward.
        self.box_push.rewards.blower_push_alignment_weight = 0.00
        self.box_push.rewards.blower_push_nozzle_standoff = self.box_push.prepush.nozzle_standoff
        self.box_push.rewards.blower_push_height = self.box_push.prepush.nozzle_height
        self.box_push.rewards.blower_push_position_std = self.box_push.prepush.position_std
        self.box_push.rewards.blower_push_height_std = self.box_push.prepush.height_std
        self.box_push.rewards.blower_push_position_blend = 0.15
        self.box_push.rewards.blower_push_alignment_power = 2.0
        self.box_push.rewards.blower_nozzle_low_height_weight = -0.1
        self.box_push.rewards.blower_nozzle_min_height = GRIPPER_ASSIST_LOW_HEIGHT_RESET
        self.box_push.rewards.blower_nozzle_low_height_margin = 0.025
        self.box_push.rewards.blower_downward_pitch_weight = -0.02
        self.box_push.rewards.blower_downward_pitch_angle_scale = 0.35
        self.box_push.rewards.airflow_velocity_toward_goal_weight = 6.0
        self.box_push.rewards.airflow_min_action = 0.05
        self.box_push.rewards.airflow_ready_reward_floor = 0.50
        # Require a real goal-directed jet force, not merely a nonzero fan command.
        self.box_push.rewards.airflow_velocity_force_scale_n = 0.10
        # Credit real force before static friction is overcome. Per-episode
        # integral cap prevents indefinite stationary blowing for reward.
        self.box_push.rewards.airflow_directed_force_weight = 1.0
        self.box_push.rewards.airflow_directed_force_scale_n = 0.8
        self.box_push.rewards.airflow_directed_force_credit_budget = 1.0
        self.box_push.rewards.airflow_before_ready_weight = -0.4
        self.box_push.rewards.velocity_without_airflow_weight = 0.0  # Previous weight: -0.50; allow gripper assist.

        self.box_push.rewards.robot_box_proximity_weight = 0.0
        self.box_push.rewards.robot_box_proximity_body_names = ("link6", "link7", "link8")
        self.box_push.rewards.robot_contact_weight = -2.0
        self.box_push.rewards.robot_contact_force_threshold = 2.0
        self.box_push.rewards.robot_contact_body_names = "link[1-5]"
        self.box_push.rewards.robot_contact_termination_body_names = None
        self.box_push.rewards.robot_contact_termination_force_threshold = 0.01
        self.box_push.rewards.robot_contact_termination_ignore_initial_s = 0.10
        self.box_push.rewards.robot_box_contact_clearance = 0.0
        self.box_push.rewards.robot_box_contact_ignore_initial_s = 0.0

        self.box_push.rewards.airflow_energy_weight = -0.08
        self.box_push.rewards.airflow_rate_weight = -0.08
        self.box_push.rewards.airflow_penalty_warmup_steps = 5000
        self.box_push.rewards.airflow_penalty_ramp_steps = 10000
        self.box_push.rewards.airflow_penalty_start_scale = 0.0
        self.box_push.rewards.arm_action_deviation_weight = -0.005
        self.box_push.rewards.arm_action_deviation_warmup_steps = 10000
        self.box_push.rewards.arm_action_deviation_ramp_steps = 20000
        self.box_push.rewards.arm_action_deviation_start_scale = 0.0
        self.box_push.rewards.action_rate_weight = -0.08
        # Fan changes are regularized on their smoothed output by airflow_rate.
        self.box_push.rewards.action_rate_term_name = "arm_action"
        self.box_push.rewards.joint_vel_weight = -0.03
        # Unthresholded L2 throughout; ease exploration before applying full weight.
        self.box_push.rewards.ee_acceleration_weight = -0.002
        self.box_push.rewards.ee_acceleration_ignore_initial_s = 0.1
        self.box_push.rewards.ee_acceleration_warmup_steps = 10000
        self.box_push.rewards.ee_acceleration_ramp_steps = 30000
        self.box_push.rewards.ee_acceleration_start_scale = 0.10
        self.box_push.rewards.nozzle_forward_penalty_weight = -0.0
        self.box_push.rewards.nozzle_forward_min_x = 0.05
        self.box_push.rewards.nozzle_forward_max_x = 0.9

        self.box_push.arm_action.smoothing_tau_s = 0.18

        super().__post_init__()

        self.observations.policy.enable_corruption = True
        self.scene.robot.spawn.usd_path = str(PIPER_GRIPPER_ASSIST_BLOWER_USD_PATH)
        self.scene.robot.spawn.activate_contact_sensors = True
        # The non-identity link0 transform is preserved by the MJCF importer as
        # an extra Xform, so the rigid articulation root is the nested link0.
        self.scene.robot.articulation_root_prim_path = "/piper_l_blower_fixed/link0/link0"
        # Imported rigid links are siblings below the outer link0 Xform. The
        # contact sensor's leaf regex matches direct children of this path.
        self.scene.contact_forces.prim_path = (
            "{ENV_REGEX_NS}/Robot/piper_l_blower_fixed/link0/.*"
        )
        self.scene.contact_forces.filter_prim_paths_expr = []
        self.actions.airflow.jet_cfg = GRIPPER_ASSIST_AIRFLOW_JET_CFG
        self.actions.airflow.reaction_force_max = GRIPPER_ASSIST_AIRFLOW_REACTION_FORCE_MAX
        self.actions.airflow.airflow_u0_range = GRIPPER_ASSIST_AIRFLOW_U0_RANDOMIZATION_RANGE
        self.actions.airflow.airflow_spread_k_range = GRIPPER_ASSIST_AIRFLOW_SPREAD_K_RANDOMIZATION_RANGE
        self.actions.airflow.airflow_randomization_buckets = GRIPPER_ASSIST_AIRFLOW_RANDOMIZATION_BUCKETS
        self.actions.airflow.show_mount_pose_marker = False
        self.actions.airflow.mount_pose_marker_max_envs = 8
        self.scene.table.init_state.pos = (0.8, 0.0, 0.0)
        self.scene.table.spawn.scale = (1.2, 1.5, 1.0)
        self.observations.critic.box_state.func = mdp.box_goal_privileged_dynamics
        self.observations.critic.box_state.params = {
            "goal_center": self.box_push.geometry.goal_center,
            "goal_radius": self.box_push.geometry.goal_radius_start,
            "asset_cfg": SceneEntityCfg("box"),
        }
        self.observations.critic.prepush_setup = None
        self.terminations.link6_collision_corner_too_low = DoneTerm(
            func=mdp.body_local_boxes_too_low,
            params={
                "robot_cfg": SceneEntityCfg("robot", body_names=["link6"]),
                "box_centers": GRIPPER_ASSIST_LINK6_COLLISION_BOX_CENTERS,
                "box_half_extents": GRIPPER_ASSIST_LINK6_COLLISION_BOX_HALF_EXTENTS,
                "box_quats": GRIPPER_ASSIST_LINK6_COLLISION_BOX_QUATS,
                "min_height": GRIPPER_ASSIST_COLLISION_CORNER_MIN_HEIGHT,
                "ignore_initial_s": self.box_push.rewards.robot_contact_termination_ignore_initial_s,
            },
        )
        self.terminations.link6_collision_box_cube_contact = DoneTerm(
            func=mdp.body_local_boxes_box_contact_force,
            params={
                "robot_cfg": SceneEntityCfg("robot", body_names=["link6"]),
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["link6"]),
                "box_cfg": SceneEntityCfg("box"),
                "box_half_extent": self.box_push.geometry.box_half_extent,
                "box_centers": GRIPPER_ASSIST_LINK6_COLLISION_BOX_CENTERS,
                "box_half_extents": GRIPPER_ASSIST_LINK6_COLLISION_BOX_HALF_EXTENTS,
                "box_quats": GRIPPER_ASSIST_LINK6_COLLISION_BOX_QUATS,
                "force_threshold": self.box_push.rewards.robot_contact_termination_force_threshold,
                "clearance": self.box_push.rewards.robot_box_contact_clearance,
                "ignore_initial_s": self.box_push.rewards.robot_box_contact_ignore_initial_s,
            },
        )
        self.terminations.left_rail_collision_corner_too_low = DoneTerm(
            func=mdp.body_local_boxes_too_low,
            params={
                "robot_cfg": SceneEntityCfg("robot", body_names=["link7"]),
                "box_centers": GRIPPER_ASSIST_LEFT_RAIL_COLLISION_BOX_CENTERS,
                "box_half_extents": GRIPPER_ASSIST_RAIL_COLLISION_BOX_HALF_EXTENTS,
                "min_height": GRIPPER_ASSIST_COLLISION_CORNER_MIN_HEIGHT,
                "ignore_initial_s": self.box_push.rewards.robot_contact_termination_ignore_initial_s,
            },
        )
        self.terminations.right_rail_collision_corner_too_low = DoneTerm(
            func=mdp.body_local_boxes_too_low,
            params={
                "robot_cfg": SceneEntityCfg("robot", body_names=["link8"]),
                "box_centers": GRIPPER_ASSIST_RIGHT_RAIL_COLLISION_BOX_CENTERS,
                "box_half_extents": GRIPPER_ASSIST_RAIL_COLLISION_BOX_HALF_EXTENTS,
                "min_height": GRIPPER_ASSIST_COLLISION_CORNER_MIN_HEIGHT,
                "ignore_initial_s": self.box_push.rewards.robot_contact_termination_ignore_initial_s,
            },
        )

        self.events.reset_box.func = mdp.reset_box_state_uniform_with_goal_range
        self.events.reset_box.params.update(
            {
                "goal_center_range": GOAL_CENTER_XY_RANGE_FINAL,
                "goal_center_range_start": GOAL_CENTER_XY_RANGE_START,
                "box_spawn_bounds": BOX_SPAWN_BOUNDS,
                "workspace_bounds": WORKSPACE_BOUNDS,
                "box_clearance": self.box_push.reset.box_clearance,
                "enforce_outside_goal_square": True,
            }
        )
        for key in (
            "radial_distance_range",
            "angle_center",
            "angle_span_start",
            "angle_span_final",
        ):
            self.events.reset_box.params.pop(key, None)

        self.curriculum.goal_region.params.update(
            {
                name: getattr(self.box_push.curriculum, name)
                for name in (
                    "goal_center_range_curriculum_enabled",
                    "goal_center_range_success_threshold",
                    "goal_center_range_failure_threshold",
                    "goal_center_range_step",
                    "goal_center_range_retreat_step",
                    "goal_center_range_final_scale",
                    "goal_center_range_warmup_steps",
                    "goal_center_range_min_steps_between_updates",
                )
            }
        )
        _configure_airflow_visualizations(self, **BOX_PUSH_TRAIN_AIRFLOW_VIZ)

        # Preserve success as a metric through goal_stable, but keep training
        # episodes alive and allow link6/link7/link8 to touch the box.
        self.rewards.success = None
        self.rewards.goal_escape = None
        self.rewards.robot_box_proximity = None
        self.terminations.success = None
        self.terminations.robot_box_contact = None

        if calibration_path := os.environ.get("AIRFLOW_CALIBRATION_PROFILE"):
            from blower.calibration import apply_to_rl, load_profile
            apply_to_rl(self, load_profile(calibration_path))


@configclass
class PiperBlowerGripperAssistBoxPushEnvCfg_PLAY(PiperBlowerGripperAssistBoxPushEnvCfg):
    """Evaluation cfg for the gripper-assist box-push variant."""

    def __post_init__(self):
        super().__post_init__()

        _apply_play_final_curriculum_state(self)

        self.box_push.geometry.goal_center_xy_range_start = GOAL_CENTER_XY_RANGE_FINAL
        self.events.reset_box.params.update(
            {
                "goal_center_range": GOAL_CENTER_XY_RANGE_FINAL,
                "goal_center_range_start": GOAL_CENTER_XY_RANGE_FINAL,
                "box_spawn_bounds": BOX_SPAWN_BOUNDS,
                "workspace_bounds": WORKSPACE_BOUNDS,
            }
        )

        fixed_robot_joint_positions = {
            joint_name: float(joint_position)
            for joint_name, joint_position in self.box_push.arm_action.position_offset.items()
        }
        fixed_robot_joint_positions["joint[78]"] = PIPER_LOCKED_GRIPPER_POS
        fixed_robot_joint_ranges = {
            joint_name: (joint_position, joint_position)
            for joint_name, joint_position in fixed_robot_joint_positions.items()
        }

        self.box_push.reset.robot_joint_position_ranges = fixed_robot_joint_ranges
        self.box_push.reset.robot_joint_velocity_range = (0.0, 0.0)
        self.box_push.curriculum.robot_reset_initial_scale = 0.0
        self.box_push.curriculum.robot_reset_shrink_enabled = False

        self.events.reset_robot_joints.params.update(
            {
                "joint_position_ranges": fixed_robot_joint_ranges,
                "velocity_range": (0.0, 0.0),
                "max_resample_attempts": 1,
                "target_joint_positions": fixed_robot_joint_positions,
                "initial_range_scale": 0.0,
            }
        )
        self.curriculum.goal_region.params.update(
            {
                "robot_reset_shrink_enabled": False,
                "robot_reset_final_scale": 0.0,
                "goal_center_range_curriculum_enabled": False,
                "goal_center_range_final_scale": 1.0,
            }
        )

        self.scene.num_envs = 16
        self.observations.policy.enable_corruption = False
        self.observations.policy.box_pose.params["noise_range"] = (0.0, 0.0)
        self.events.randomize_robot_actuator_gains = None
        self.events.randomize_box_material = None
        self.events.randomize_box_mass = None
        self.actions.airflow.airflow_u0_range = None
        self.actions.airflow.airflow_spread_k_range = None
        _configure_airflow_visualizations(self, **BOX_PUSH_PLAY_AIRFLOW_VIZ)
