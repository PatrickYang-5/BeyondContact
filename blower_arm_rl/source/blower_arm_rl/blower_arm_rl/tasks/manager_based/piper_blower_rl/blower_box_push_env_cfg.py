# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Piper airflow task: push a box into a goal region and keep it there.

The task is intentionally still end-to-end with joint-position actions.  The
extra structure below only makes reset, reward, curriculum, and airflow tuning
easy to inspect and override.
"""

from __future__ import annotations

import math

from blower.configs import JetCfg, SamplingCfg, VizCfg
from blower.mdp import AirflowTargetCfg
from blower.objects import BoxCfg

from isaaclab.assets import AssetBaseCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass

from blower_arm_rl.tasks.manager_based.blower_arm_rl import mdp
from blower_arm_rl.tasks.manager_based.blower_arm_rl.mdp.airflow import SmoothedAirflowActionTermCfg

from . import joint_pos_env_cfg
from .piper_l_cfg import (
    PIPER_ARM_JOINT_LIMITS_RAD,
    PIPER_ARM_JOINTS,
    PIPER_BOX_PROXIMITY_BODIES,
    PIPER_CONTACT_BODY_PATTERN,
    PIPER_CONTROL_DECIMATION,
    PIPER_DEFAULT_ARM_JOINT_POS,
    PIPER_END_EFFECTOR_BODY,
    PIPER_LOCKED_GRIPPER_POS,
    PIPER_NOZZLE_LOCAL_AXIS,
    PIPER_NOZZLE_OFFSET_POS,
    PIPER_NOZZLE_OFFSET_QUAT,
    PIPER_SIM_DT,
    PiperSim2RealParamsCfg,
)


BOX_PUSH_BOX_CFG = BoxCfg(
    size=(0.08, 0.08, 0.08),
    mass=0.12,
    color=(0.9, 0.15, 0.1),
    spawn_z=0.05,
)

AIRFLOW_JET_CFG = JetCfg(
    nozzle_radius=0.04,
    spread_k=0.35,
    U0=40.0,
    B=5.9,
    eps=0.03,
)
AIRFLOW_REACTION_FORCE_MAX = (
    0.5 * 1.225 * AIRFLOW_JET_CFG.U0**2 * math.pi * AIRFLOW_JET_CFG.nozzle_radius**2
)
BOX_MASS_RANDOMIZATION_RANGE = (0.08, 0.15)
BOX_STATIC_FRICTION_RANDOMIZATION_RANGE = (0.45, 1.20)
BOX_DYNAMIC_FRICTION_RANDOMIZATION_RANGE = (0.35, 1.10)
BOX_RESTITUTION_RANDOMIZATION_RANGE = (0.0, 0.0)
BOX_MATERIAL_RANDOMIZATION_BUCKETS = 32
AIRFLOW_U0_RANDOMIZATION_RANGE = (30.0, 50.0)
AIRFLOW_SPREAD_K_RANDOMIZATION_RANGE = (0.25, 0.50)
AIRFLOW_RANDOMIZATION_BUCKETS = 16
JOINT_POS_OBS_NOISE_RANGE = (-0.01, 0.01)
JOINT_VEL_OBS_NOISE_RANGE = (-0.01, 0.01)
CUBE_POS_OBS_NOISE_RANGE = (-0.01, 0.01)
OBS_NOISE_WARMUP_STEPS = 30000
OBS_NOISE_RAMP_STEPS = 60000

# Backward-compatible aliases used by play cfgs and quick experiments.
GOAL_CENTER = (0.42, 0.0, 0.006)
GOAL_CENTER_XY_RANGE = (0.55, 0.75, -0.08, 0.08)
GOAL_RADIUS_START = 0.12
GOAL_RADIUS_FINAL = 0.09
GOAL_HOLD_STEPS = 50
WORKSPACE_BOUNDS = (0.18, 1.20, -0.20, 0.20)
BOX_GOAL_RADIAL_DISTANCE_RANGE = (0.08, 0.16)
BOX_HALF_EXTENT = 0.04
BOX_AIM_HEIGHT = BOX_PUSH_BOX_CFG.spawn_z + 0.015
GOAL_VISUAL_BASE_RADIUS = GOAL_RADIUS_START
ROBOT_BOX_PROXIMITY_BODIES = PIPER_BOX_PROXIMITY_BODIES

PREPUSH_NOZZLE_STANDOFF = 0.10
PREPUSH_NOZZLE_HEIGHT = 0.22
PREPUSH_POSITION_STD = 0.07
PREPUSH_HEIGHT_STD = 0.06
PREPUSH_AXIS_STD = 0.22

PIPER_BOX_PUSH_JOINT_RANGES = {
    "joint1": (-0.80, 0.80),
    "joint2": (0.80, 1.50),
    "joint3": (-1.30, -1.00),
    "joint4": (-0.80, 0.80),
    "joint5": (-0.90, 0.50),
    "joint6": (-3.10, -1.70),
    "joint[78]": (0.0, 0.0),
}

PIPER_BOX_PUSH_ACTION_OFFSET = dict(PIPER_DEFAULT_ARM_JOINT_POS)

PIPER_BOX_PUSH_ACTION_SCALE = {
    "joint1": 1.0,
    "joint2": 1.0,
    "joint3": 1.0,
    "joint4": 1.0,
    "joint5": 1.0,
    "joint6": 1.0,
}

PIPER_BOX_PUSH_ACTION_CLIP = dict(PIPER_ARM_JOINT_LIMITS_RAD)

PIPER_BOX_PUSH_ROBOT_RESET_FINAL_HALF_WIDTHS = {
    "joint1": 0.0,
    "joint2": 0.0,
    "joint3": 0.0,
    "joint4": 0.0,
    "joint5": 0.0,
    "joint6": 0.0,
    "joint[78]": 0.0,
}

PIPER_BOX_PUSH_ROBOT_RESET_TARGET = {
    **PIPER_BOX_PUSH_ACTION_OFFSET,
    "joint[78]": 0.0,
}

PIPER_BOX_PUSH_SHOULDER_STIFFNESS = 150.0
PIPER_BOX_PUSH_SHOULDER_DAMPING = 10.0
PIPER_BOX_PUSH_FOREARM_STIFFNESS = 10.0
PIPER_BOX_PUSH_FOREARM_DAMPING = 0.8
PIPER_BOX_PUSH_STIFFNESS_RANDOMIZATION_SCALE_RANGE = (0.8, 1.2)
PIPER_BOX_PUSH_DAMPING_RANDOMIZATION_SCALE_RANGE = (0.8, 1.2)
PIPER_BOX_PUSH_MAX_JOINT_STEP_RAD = 0.03


# ---------------------------------------------------------------------
# Tunable task parameter blocks
# ---------------------------------------------------------------------


@configclass
class BoxPushGeometryParamsCfg:
    goal_center: tuple[float, float, float] = GOAL_CENTER
    goal_center_xy_range: tuple[float, float, float, float] | None = GOAL_CENTER_XY_RANGE
    goal_center_xy_range_start: tuple[float, float, float, float] | None = None
    goal_radius_start: float = GOAL_RADIUS_START
    goal_radius_final: float = GOAL_RADIUS_FINAL
    goal_hold_steps: int = GOAL_HOLD_STEPS
    workspace_bounds: tuple[float, float, float, float] = WORKSPACE_BOUNDS
    box_half_extent: float = BOX_HALF_EXTENT
    box_aim_height: float = BOX_AIM_HEIGHT


@configclass
class BoxPushResetParamsCfg:
    robot_joint_position_ranges: dict[str, tuple[float, float]] = PIPER_BOX_PUSH_JOINT_RANGES
    robot_joint_velocity_range: tuple[float, float] = (-0.02, 0.02)
    robot_joint_limit_margin: float = 0.04
    robot_max_resample_attempts: int = 16
    robot_enforce_self_collision_safety: bool = False

    box_radial_distance_range: tuple[float, float] = BOX_GOAL_RADIAL_DISTANCE_RANGE
    box_radial_distance_range_start: tuple[float, float] | None = None
    box_angle_center: float = math.pi
    box_angle_span_start: float = 0.35
    box_angle_span_final: float = 1.20
    box_clearance: float = 0.015


@configclass
class BoxPushPrepushParamsCfg:
    nozzle_standoff: float = PREPUSH_NOZZLE_STANDOFF
    nozzle_height: float = PREPUSH_NOZZLE_HEIGHT
    position_std: float = PREPUSH_POSITION_STD
    height_std: float = PREPUSH_HEIGHT_STD
    axis_std: float = PREPUSH_AXIS_STD
    # None preserves the original fixed pre-push pose scoring for other tasks.
    ready_geometry: dict[str, float] | None = None


@configclass
class BoxPushArmActionParamsCfg:
    position_offset: dict[str, float] = PIPER_BOX_PUSH_ACTION_OFFSET
    position_scale: dict[str, float] = PIPER_BOX_PUSH_ACTION_SCALE
    position_clip: dict[str, tuple[float, float]] = PIPER_BOX_PUSH_ACTION_CLIP
    use_default_offset: bool = False
    smoothing_tau_s: float = 0.12
    max_joint_step_rad: float = PIPER_BOX_PUSH_MAX_JOINT_STEP_RAD
    shoulder_stiffness: float = PIPER_BOX_PUSH_SHOULDER_STIFFNESS
    shoulder_damping: float = PIPER_BOX_PUSH_SHOULDER_DAMPING
    forearm_stiffness: float = PIPER_BOX_PUSH_FOREARM_STIFFNESS
    forearm_damping: float = PIPER_BOX_PUSH_FOREARM_DAMPING


@configclass
class BoxPushRewardParamsCfg:
    alive_weight: float = 0.0
    box_goal_dense_weight: float = 2.0
    box_goal_dense_std: float = 0.12
    box_goal_progress_weight: float = 1.0
    box_goal_progress_std: float = 0.12
    box_velocity_toward_goal_weight: float = 0.0

    goal_inside_weight: float = 1.0
    goal_centering_weight: float = 2.0
    goal_centering_std: float = 0.055
    goal_stable_weight: float = 2.0
    goal_stable_std: float = 0.055
    goal_escape_weight: float = 0.0
    goal_escape_min_hold_steps: int = 1
    success_weight: float = 0.0

    blower_box_range_weight: float = 0.8
    blower_box_range_distance: float = 0.17
    blower_box_range_std: float = 0.08
    blower_push_alignment_weight: float = 2.0
    blower_push_nozzle_standoff: float = PREPUSH_NOZZLE_STANDOFF
    blower_push_position_std: float = 0.08
    blower_push_height: float = PREPUSH_NOZZLE_HEIGHT
    blower_push_height_std: float = PREPUSH_HEIGHT_STD
    blower_push_position_blend: float = 0.30
    blower_push_alignment_power: float = 1.0
    blower_push_min_horizontal_fraction: float = 0.75
    blower_push_horizontal_std: float = 0.20
    blower_nozzle_low_height_weight: float = 0.0
    blower_nozzle_min_height: float = BOX_AIM_HEIGHT
    blower_nozzle_low_height_margin: float = 0.03
    blower_downward_pitch_weight: float = 0.0
    blower_downward_pitch_angle_scale: float = 0.35

    airflow_velocity_toward_goal_weight: float = 0.0
    airflow_min_action: float = 0.15
    airflow_ready_reward_floor: float = 0.0
    airflow_velocity_force_scale_n: float = 0.0  # Zero preserves the ungated reward.
    airflow_directed_force_weight: float = 0.0
    airflow_directed_force_scale_n: float = 0.8
    airflow_directed_force_credit_budget: float = 1.0
    airflow_before_ready_weight: float = 0.0
    velocity_without_airflow_weight: float = 0.0
    velocity_without_airflow_deadband: float = 0.008
    robot_box_proximity_weight: float = -6.0
    robot_box_proximity_clearance: float = 0.06
    robot_box_contact_clearance: float = 0.025
    robot_box_contact_ignore_initial_s: float = 0.15
    robot_box_proximity_body_names: tuple[str, ...] = PIPER_BOX_PROXIMITY_BODIES
    robot_contact_weight: float = -12.0
    robot_contact_force_threshold: float = 0.5
    robot_contact_body_names: str = PIPER_CONTACT_BODY_PATTERN
    robot_contact_termination_body_names: str | tuple[str, ...] | None = ".*"
    robot_contact_termination_force_threshold: float = 0.01
    robot_contact_termination_ignore_initial_s: float = 0.05

    airflow_energy_weight: float = -0.02
    airflow_rate_weight: float = -0.005
    airflow_penalty_warmup_steps: int = 5000
    airflow_penalty_ramp_steps: int = 20000
    airflow_penalty_start_scale: float = 0.0
    arm_action_deviation_weight: float = 0.0
    arm_action_deviation_warmup_steps: int = 0
    arm_action_deviation_ramp_steps: int = 0
    arm_action_deviation_start_scale: float = 1.0
    action_rate_weight: float = -0.008
    action_rate_term_name: str | None = None  # None keeps the original all-action penalty.
    joint_vel_weight: float = -0.00025
    ee_acceleration_weight: float = 0.0
    ee_acceleration_ignore_initial_s: float = 0.1
    ee_acceleration_warmup_steps: int = 0
    ee_acceleration_ramp_steps: int = 0
    ee_acceleration_start_scale: float = 1.0
    box_out_of_workspace_weight: float = -200.0
    nozzle_forward_penalty_weight: float = 0.0
    nozzle_forward_min_x: float = 0.05
    nozzle_forward_max_x: float | None = None
    all_in_goal_joint_home_weight: float = 0.0
    all_in_goal_joint_home_std: float = 0.8


@configclass
class BoxPushCurriculumParamsCfg:
    success_threshold: float = 0.72
    failure_threshold: float = 0.25
    shrink_factor: float = 0.90
    grow_factor: float = 1.05
    success_ema_momentum: float = 0.92
    warmup_steps: int = 3000
    min_steps_between_updates: int = 1500
    angle_range_step: float = 0.0
    expand_angle_before_shrinking_goal: bool = False
    robot_reset_initial_scale: float = 1.0
    robot_reset_shrink_enabled: bool = True
    robot_reset_success_threshold: float = 0.85
    robot_reset_failure_threshold: float = 0.35
    robot_reset_shrink_step: float = 0.10
    robot_reset_grow_step: float = 0.05
    robot_reset_final_scale: float = 0.0
    robot_reset_warmup_steps: int = 6000
    robot_reset_min_steps_between_updates: int = 1500
    robot_reset_final_half_widths: dict[str, float] = PIPER_BOX_PUSH_ROBOT_RESET_FINAL_HALF_WIDTHS

    box_spawn_distance_curriculum_enabled: bool = False
    box_spawn_distance_success_threshold: float = 0.70
    box_spawn_distance_failure_threshold: float = 0.25
    box_spawn_distance_step: float = 0.10
    box_spawn_distance_retreat_step: float = 0.05
    box_spawn_distance_final_scale: float = 1.0
    box_spawn_distance_warmup_steps: int = 3000
    box_spawn_distance_min_steps_between_updates: int = 1500

    goal_center_range_curriculum_enabled: bool = False
    goal_center_range_success_threshold: float = 0.70
    goal_center_range_failure_threshold: float = 0.25
    goal_center_range_step: float = 0.10
    goal_center_range_retreat_step: float = 0.05
    goal_center_range_final_scale: float = 1.0
    goal_center_range_warmup_steps: int = 3000
    goal_center_range_min_steps_between_updates: int = 1500


@configclass
class BoxPushAirflowVizParamsCfg:
    show_jet_cone: bool | None = None
    show_laser_ray: bool | None = None
    show_sample_markers: bool | None = None
    show_reaction_force_arrow: bool | None = None
    max_visualized_envs: int | None = None


@configclass
class BoxPushAirflowParamsCfg:
    action_mapping: str = "clamp_01"
    smoothing_tau_s: float = 0.20
    reset_settle_s: float = 0.50
    target_area: float = 1.0e-4
    min_axial_distance: float = 0.0
    velocity_cutoff: float = 0.0
    apply_reaction_force: bool = True
    reaction_force_max: float = AIRFLOW_REACTION_FORCE_MAX
    u0_range: tuple[float, float] | None = AIRFLOW_U0_RANDOMIZATION_RANGE
    spread_k_range: tuple[float, float] | None = AIRFLOW_SPREAD_K_RANDOMIZATION_RANGE
    randomization_buckets: int = AIRFLOW_RANDOMIZATION_BUCKETS
    viz: BoxPushAirflowVizParamsCfg = BoxPushAirflowVizParamsCfg()


@configclass
class BoxPushDomainRandomizationParamsCfg:
    box_mass_range: tuple[float, float] | None = BOX_MASS_RANDOMIZATION_RANGE
    box_static_friction_range: tuple[float, float] | None = BOX_STATIC_FRICTION_RANDOMIZATION_RANGE
    box_dynamic_friction_range: tuple[float, float] | None = BOX_DYNAMIC_FRICTION_RANDOMIZATION_RANGE
    box_restitution_range: tuple[float, float] = BOX_RESTITUTION_RANDOMIZATION_RANGE
    box_material_buckets: int = BOX_MATERIAL_RANDOMIZATION_BUCKETS
    robot_stiffness_scale_range: tuple[float, float] | None = PIPER_BOX_PUSH_STIFFNESS_RANDOMIZATION_SCALE_RANGE
    robot_damping_scale_range: tuple[float, float] | None = PIPER_BOX_PUSH_DAMPING_RANDOMIZATION_SCALE_RANGE


@configclass
class BoxPushTaskParamsCfg:
    geometry: BoxPushGeometryParamsCfg = BoxPushGeometryParamsCfg()
    reset: BoxPushResetParamsCfg = BoxPushResetParamsCfg()
    prepush: BoxPushPrepushParamsCfg = BoxPushPrepushParamsCfg()
    arm_action: BoxPushArmActionParamsCfg = BoxPushArmActionParamsCfg()
    rewards: BoxPushRewardParamsCfg = BoxPushRewardParamsCfg()
    curriculum: BoxPushCurriculumParamsCfg = BoxPushCurriculumParamsCfg()
    airflow: BoxPushAirflowParamsCfg = BoxPushAirflowParamsCfg()
    domain_randomization: BoxPushDomainRandomizationParamsCfg = BoxPushDomainRandomizationParamsCfg()
    sim2real: PiperSim2RealParamsCfg = PiperSim2RealParamsCfg()

    num_envs: int = 1024
    env_spacing: float = 2.5
    episode_length_s: float = 10.0


def _prepush_term_params(task: BoxPushTaskParamsCfg) -> dict:
    return {
        "asset_cfg": SceneEntityCfg("box"),
        "airflow_term_name": "airflow",
        "goal_center": task.geometry.goal_center,
        "nozzle_standoff": task.prepush.nozzle_standoff,
        "nozzle_height": task.prepush.nozzle_height,
        "box_half_extent": task.geometry.box_half_extent,
        "desired_height": task.geometry.box_aim_height,
        "position_std": task.prepush.position_std,
        "height_std": task.prepush.height_std,
        "axis_std": task.prepush.axis_std,
        "ready_geometry": task.prepush.ready_geometry,
    }


def _make_viz_cfg() -> VizCfg:
    viz_cfg = VizCfg()
    viz_cfg.show_jet_cone = False
    viz_cfg.show_laser_ray = True
    viz_cfg.show_sample_markers = False
    viz_cfg.jet_cone_use_flow_intensity = False
    viz_cfg.jet_cone_num_slices = 16
    viz_cfg.jet_cone_num_segments = 32
    viz_cfg.jet_cone_update_interval = 4
    viz_cfg.jet_cone_line_thickness = 2.0
    viz_cfg.laser_ray_length = 1.5
    viz_cfg.laser_ray_radius = 0.006
    viz_cfg.laser_point_radius = 0.03
    viz_cfg.laser_point_color = (1.0, 0.0, 0.0)
    viz_cfg.laser_use_hit_plane = True
    viz_cfg.laser_hit_plane_z = 0.0
    viz_cfg.laser_hit_plane_max_length = 3.0
    viz_cfg.laser_point_surface_offset = 0.025
    viz_cfg.show_reaction_force_arrow = True
    viz_cfg.reaction_force_arrow_length_per_newton = 0.04
    viz_cfg.reaction_force_arrow_max_length = 0.5
    viz_cfg.reaction_force_arrow_shaft_radius = 0.008
    viz_cfg.reaction_force_arrow_head_radius = 0.026
    viz_cfg.reaction_force_arrow_head_length = 0.065
    viz_cfg.reaction_force_arrow_color = (1.0, 0.55, 0.05)
    return viz_cfg


def _make_airflow_viz_cfg(task: BoxPushTaskParamsCfg) -> VizCfg:
    viz_cfg = _make_viz_cfg()
    viz = task.airflow.viz
    if viz.show_jet_cone is not None:
        viz_cfg.show_jet_cone = bool(viz.show_jet_cone)
    if viz.show_laser_ray is not None:
        viz_cfg.show_laser_ray = bool(viz.show_laser_ray)
    if viz.show_sample_markers is not None:
        viz_cfg.show_sample_markers = bool(viz.show_sample_markers)
    if viz.show_reaction_force_arrow is not None:
        viz_cfg.show_reaction_force_arrow = bool(viz.show_reaction_force_arrow)
    if viz.max_visualized_envs is not None:
        max_visualized_envs = int(viz.max_visualized_envs)
        if max_visualized_envs < 0:
            max_visualized_envs = int(task.num_envs)
        viz_cfg.max_visualized_envs = max(0, max_visualized_envs)
    return viz_cfg


# ---------------------------------------------------------------------
# Manager groups: observation, reset, reward, termination, curriculum
# ---------------------------------------------------------------------


@configclass
class ObservationsCfg:
    """Observation groups for asymmetric actor-critic training."""

    @configclass
    class PolicyCfg(ObsGroup):
        joint_pos = ObsTerm(
            func=mdp.noisy_joint_pos,
            params={
                "noise_range": JOINT_POS_OBS_NOISE_RANGE,
                "warmup_steps": OBS_NOISE_WARMUP_STEPS,
                "ramp_steps": OBS_NOISE_RAMP_STEPS,
                "asset_cfg": SceneEntityCfg("robot", joint_names=PIPER_ARM_JOINTS),
            },
            history_length=10,
        )
        joint_vel = ObsTerm(
            func=mdp.noisy_joint_vel_rel,
            params={
                "noise_range": JOINT_VEL_OBS_NOISE_RANGE,
                "warmup_steps": OBS_NOISE_WARMUP_STEPS,
                "ramp_steps": OBS_NOISE_RAMP_STEPS,
                "asset_cfg": SceneEntityCfg("robot", joint_names=PIPER_ARM_JOINTS),
            },
            history_length=10,
        )

        goal_pos = ObsTerm(
            func=mdp.goal_xy,
            params={
                "goal_center": GOAL_CENTER,
            },
        )

        box_pose = ObsTerm(
            func=mdp.noisy_box_pose_local,
            params={
                "noise_range": CUBE_POS_OBS_NOISE_RANGE,
                "warmup_steps": OBS_NOISE_WARMUP_STEPS,
                "ramp_steps": OBS_NOISE_RAMP_STEPS,
                "asset_cfg": SceneEntityCfg("box"),
            },
        )

        goal_radius = ObsTerm(
            func=mdp.goal_radius_obs,
            params={"goal_radius": GOAL_RADIUS_START},
        )

        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.goal_radius = None
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObsGroup):
        box_state = ObsTerm(
            func=mdp.box_privileged_state,
            params={
                "goal_center": GOAL_CENTER,
                "goal_radius": GOAL_RADIUS_START,
                "asset_cfg": SceneEntityCfg("box"),
            },
        )

        prepush_setup = ObsTerm(
            func=mdp.prepush_setup_features,
            params=_prepush_term_params(BoxPushTaskParamsCfg()),
        )

        nozzle_axis_features = ObsTerm(
            func=mdp.nozzle_axis_features,
            params={
                "airflow_term_name": "airflow",
            },
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class EventCfg:
    """Reset events for the around-goal box-push task."""

    randomize_robot_actuator_gains = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=PIPER_ARM_JOINTS),
            "stiffness_distribution_params": PIPER_BOX_PUSH_STIFFNESS_RANDOMIZATION_SCALE_RANGE,
            "damping_distribution_params": PIPER_BOX_PUSH_DAMPING_RANDOMIZATION_SCALE_RANGE,
            "operation": "scale",
            "distribution": "uniform",
        },
    )

    randomize_box_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("box"),
            "static_friction_range": BOX_STATIC_FRICTION_RANDOMIZATION_RANGE,
            "dynamic_friction_range": BOX_DYNAMIC_FRICTION_RANDOMIZATION_RANGE,
            "restitution_range": BOX_RESTITUTION_RANDOMIZATION_RANGE,
            "num_buckets": BOX_MATERIAL_RANDOMIZATION_BUCKETS,
            "make_consistent": True,
        },
    )

    randomize_box_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("box"),
            "mass_distribution_params": BOX_MASS_RANDOMIZATION_RANGE,
            "operation": "abs",
            "distribution": "uniform",
            "recompute_inertia": True,
            "min_mass": 1.0e-4,
        },
    )

    reset_robot_joints = EventTerm(
        func=mdp.reset_robot_joints_by_safe_sample,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "joint_position_ranges": PIPER_BOX_PUSH_JOINT_RANGES,
            "velocity_range": (-0.02, 0.02),
            "joint_limit_margin": 0.04,
            "max_resample_attempts": 16,
            "enforce_self_collision_safety": False,
            "target_joint_positions": PIPER_BOX_PUSH_ROBOT_RESET_TARGET,
            "target_joint_half_widths": PIPER_BOX_PUSH_ROBOT_RESET_FINAL_HALF_WIDTHS,
            "range_scale_attr": "_box_push_robot_reset_range_scale",
            "initial_range_scale": 1.0,
        },
    )

    reset_box = EventTerm(
        func=mdp.reset_box_state_around_goal,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("box"),
            "goal_center": GOAL_CENTER,
            "goal_center_range": GOAL_CENTER_XY_RANGE,
            "goal_radius": GOAL_RADIUS_START,
            "spawn_z": BOX_PUSH_BOX_CFG.spawn_z,
            "radial_distance_range": BOX_GOAL_RADIAL_DISTANCE_RANGE,
            "workspace_bounds": WORKSPACE_BOUNDS,
            "angle_center": math.pi,
            "angle_span_start": 0.35,
            "angle_span_final": 1.20,
            "box_clearance": 0.015,
            "enforce_outside_goal_square": True,
        },
    )


@configclass
class RewardsCfg:
    """Compact reward terms for airflow box pushing."""

    alive = RewTerm(func=mdp.is_alive, weight=0.0)

    box_goal_dense = RewTerm(
        func=mdp.box_goal_distance_tanh,
        weight=2.0,
        params={
            "asset_cfg": SceneEntityCfg("box"),
            "goal_center": GOAL_CENTER,
            "goal_radius": GOAL_RADIUS_START,
            "std": 0.12,
        },
    )

    box_goal_progress = RewTerm(
        func=mdp.box_goal_progress,
        weight=1.0,
        params={
            "asset_cfg": SceneEntityCfg("box"),
            "goal_center": GOAL_CENTER,
            "goal_radius": GOAL_RADIUS_START,
            "std": 0.12,
        },
    )

    box_velocity_toward_goal = RewTerm(
        func=mdp.box_velocity_toward_goal_reward,
        weight=0.0,
        params={
            "asset_cfg": SceneEntityCfg("box"),
            "goal_center": GOAL_CENTER,
        },
    )

    goal_inside = RewTerm(
        func=mdp.box_goal_inside,
        weight=1.0,
        params={
            "asset_cfg": SceneEntityCfg("box"),
            "goal_center": GOAL_CENTER,
            "goal_radius": GOAL_RADIUS_START,
            "box_half_extent": BOX_HALF_EXTENT,
        },
    )

    goal_centering = RewTerm(
        func=mdp.box_goal_centering_reward,
        weight=2.0,
        params={
            "asset_cfg": SceneEntityCfg("box"),
            "goal_center": GOAL_CENTER,
            "goal_radius": GOAL_RADIUS_START,
            "std": 0.055,
            "box_half_extent": BOX_HALF_EXTENT,
        },
    )

    goal_stable = RewTerm(
        func=mdp.box_goal_low_velocity_reward,
        weight=2.0,
        params={
            "asset_cfg": SceneEntityCfg("box"),
            "goal_center": GOAL_CENTER,
            "goal_radius": GOAL_RADIUS_START,
            "std": 0.055,
            "box_half_extent": BOX_HALF_EXTENT,
            "required_hold_steps": GOAL_HOLD_STEPS,
        },
    )

    success = RewTerm(
        func=mdp.box_goal_held_success,
        weight=0.0,
        params={
            "asset_cfg": SceneEntityCfg("box"),
            "goal_center": GOAL_CENTER,
            "goal_radius": GOAL_RADIUS_START,
            "required_hold_steps": GOAL_HOLD_STEPS,
            "box_half_extent": BOX_HALF_EXTENT,
        },
    )

    goal_escape = RewTerm(
        func=mdp.box_goal_escape_penalty,
        weight=0.0,
        params={
            "asset_cfg": SceneEntityCfg("box"),
            "goal_center": GOAL_CENTER,
            "goal_radius": GOAL_RADIUS_START,
            "required_hold_steps": GOAL_HOLD_STEPS,
            "min_hold_steps": 1,
            "box_half_extent": BOX_HALF_EXTENT,
        },
    )

    blower_box_range = RewTerm(
        func=mdp.blower_box_distance_reward,
        weight=0.8,
        params={
            "asset_cfg": SceneEntityCfg("box"),
            "airflow_term_name": "airflow",
            "desired_distance": 0.20,
            "std": 0.055,
        },
    )

    blower_push_alignment = RewTerm(
        func=mdp.blower_horizontal_push_alignment_reward,
        weight=2.0,
        params={
            "asset_cfg": SceneEntityCfg("box"),
            "airflow_term_name": "airflow",
            "goal_center": GOAL_CENTER,
            "nozzle_standoff": 0.14,
            "position_std": 0.14,
            "desired_height": 0.13,
            "aim_height": BOX_AIM_HEIGHT,
            "height_std": 0.07,
            "position_blend": 0.50,
            "min_horizontal_fraction": 0.75,
            "horizontal_std": 0.20,
        },
    )

    blower_nozzle_low_height = RewTerm(
        func=mdp.blower_nozzle_low_height_with_airflow_penalty,
        weight=0.0,
        params={
            "airflow_term_name": "airflow",
            "min_height": BOX_AIM_HEIGHT,
            "height_margin": 0.03,
            "penalty_warmup_steps": 5000,
            "penalty_ramp_steps": 20000,
            "penalty_start_scale": 0.0,
        },
    )

    blower_downward_pitch = RewTerm(
        func=mdp.blower_downward_pitch_with_airflow_penalty,
        weight=0.0,
        params={
            "airflow_term_name": "airflow",
            "angle_scale": 0.35,
            "penalty_warmup_steps": 5000,
            "penalty_ramp_steps": 20000,
            "penalty_start_scale": 0.0,
        },
    )

    airflow_velocity_toward_goal = RewTerm(
        func=mdp.box_velocity_toward_goal_with_airflow_reward,
        weight=0.0,
        params={
            **_prepush_term_params(BoxPushTaskParamsCfg()),
            "min_airflow": 0.15,
        },
    )

    airflow_directed_force = RewTerm(
        func=mdp.AirflowDirectedForceReward,
        weight=0.0,
        params={"goal_center": GOAL_CENTER, "goal_radius": GOAL_RADIUS_START},
    )

    airflow_before_ready = RewTerm(
        func=mdp.airflow_before_ready_penalty,
        weight=0.0,
        params={
            **_prepush_term_params(BoxPushTaskParamsCfg()),
            "penalty_warmup_steps": 5000,
            "penalty_ramp_steps": 20000,
            "penalty_start_scale": 0.0,
        },
    )

    velocity_without_airflow = RewTerm(
        func=mdp.box_velocity_toward_goal_without_airflow_penalty,
        weight=0.0,
        params={
            "asset_cfg": SceneEntityCfg("box"),
            "goal_center": GOAL_CENTER,
            "airflow_term_name": "airflow",
            "min_airflow": 0.15,
            "velocity_deadband": 0.008,
        },
    )

    robot_box_proximity = RewTerm(
        func=mdp.robot_box_proximity_penalty,
        weight=0.0,
        params={
            "box_cfg": SceneEntityCfg("box"),
            "robot_cfg": SceneEntityCfg("robot", body_names=list(ROBOT_BOX_PROXIMITY_BODIES)),
            "box_half_extent": BOX_HALF_EXTENT,
            "clearance": 0.04,
        },
    )

    robot_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-8.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=PIPER_CONTACT_BODY_PATTERN),
            "threshold": 1.0,
        },
    )

    airflow_energy = RewTerm(
        func=mdp.airflow_energy,
        weight=-0.02,
        params={
            "airflow_term_name": "airflow",
            "penalty_warmup_steps": 5000,
            "penalty_ramp_steps": 20000,
            "penalty_start_scale": 0.0,
        },
    )

    airflow_rate = RewTerm(
        func=mdp.airflow_rate_l2,
        weight=-0.005,
        params={
            "airflow_term_name": "airflow",
            "penalty_warmup_steps": 5000,
            "penalty_ramp_steps": 20000,
            "penalty_start_scale": 0.0,
        },
    )

    arm_action_deviation = RewTerm(
        func=mdp.arm_action_deviation_l2,
        weight=0.0,
        params={
            "action_term_name": "arm_action",
            "goal_center": GOAL_CENTER,
            "goal_radius": GOAL_RADIUS_START,
            "box_half_extent": BOX_HALF_EXTENT,
            "penalty_warmup_steps": 0,
            "penalty_ramp_steps": 0,
            "penalty_start_scale": 1.0,
            "asset_cfg": SceneEntityCfg("box"),
        },
    )

    action_rate = RewTerm(
        func=mdp.action_rate_l2,
        weight=-0.001,
    )

    joint_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-0.00005,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )

    ee_acceleration = RewTerm(
        func=mdp.EndEffectorAccelerationPenalty,
        weight=0.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["link6"]),
            "point_offset": (0.08, 0.0, 0.05),
            "ignore_initial_s": 0.1,
        },
    )

    box_out_of_workspace = RewTerm(
        func=mdp.box_out_of_workspace_penalty,
        weight=-140.0,
        params={
            "asset_cfg": SceneEntityCfg("box"),
            "workspace_bounds": WORKSPACE_BOUNDS,
            "z_bounds": (-0.02, 0.25),
        },
    )

    nozzle_forward = RewTerm(
        func=mdp.nozzle_forward_position_penalty,
        weight=0.0,
        params={
            "airflow_term_name": "airflow",
            "min_forward_x": 0.05,
            "max_forward_x": None,
        },
    )


@configclass
class TerminationsCfg:
    """Termination terms for held success, failure, and timeout."""

    success = DoneTerm(
        func=mdp.box_in_goal_held,
        params={
            "asset_cfg": SceneEntityCfg("box"),
            "goal_center": GOAL_CENTER,
            "goal_radius": GOAL_RADIUS_START,
            "required_hold_steps": GOAL_HOLD_STEPS,
            "box_half_extent": BOX_HALF_EXTENT,
        },
    )

    box_out_of_workspace = DoneTerm(
        func=mdp.box_out_of_workspace,
        params={
            "asset_cfg": SceneEntityCfg("box"),
            "workspace_bounds": WORKSPACE_BOUNDS,
            "z_bounds": (-0.02, 0.25),
        },
    )

    robot_box_contact = DoneTerm(
        func=mdp.robot_box_too_close,
        params={
            "box_cfg": SceneEntityCfg("box"),
            "robot_cfg": SceneEntityCfg("robot", body_names=list(ROBOT_BOX_PROXIMITY_BODIES)),
            "box_half_extent": BOX_HALF_EXTENT,
            "clearance": 0.012,
            "ignore_initial_s": 0.25,
        },
    )

    robot_table_contact = None

    blower_nozzle_too_low = DoneTerm(
        func=mdp.nozzle_too_low,
        params={
            "airflow_term_name": "airflow",
            "min_height": BOX_AIM_HEIGHT,
            "ignore_initial_s": 0.10,
        },
    )

    robot_attachment_too_low = None

    time_out = DoneTerm(func=mdp.time_out, time_out=True)


@configclass
class CurriculumCfg:
    """Adaptive goal-size curriculum with full-circle box resets."""

    goal_region = CurrTerm(
        func=mdp.adaptive_goal_region_curriculum,
        params={
            "goal_center": GOAL_CENTER,
            "goal_radius_start": GOAL_RADIUS_START,
            "goal_radius_final": GOAL_RADIUS_FINAL,
            "success_threshold": 0.72,
            "failure_threshold": 0.25,
            "shrink_factor": 0.90,
            "grow_factor": 1.05,
            "success_ema_momentum": 0.92,
            "warmup_steps": 3000,
            "min_steps_between_updates": 1500,
            "initial_range_step": 0.0,
            "angle_span_start": 0.35,
            "angle_span_final": 1.20,
            "expand_angle_before_shrinking_goal": False,
            "success_term_name": "success",
            "required_hold_steps": GOAL_HOLD_STEPS,
            "robot_reset_shrink_enabled": True,
            "robot_reset_success_threshold": 0.85,
            "robot_reset_failure_threshold": 0.35,
            "robot_reset_shrink_step": 0.10,
            "robot_reset_grow_step": 0.05,
            "robot_reset_final_scale": 0.0,
            "robot_reset_warmup_steps": 6000,
            "robot_reset_min_steps_between_updates": 1500,
        },
    )


def _apply_task_params(env_cfg: "PiperBlowerBoxPushInitEnvCfg", task: BoxPushTaskParamsCfg) -> None:
    geom = task.geometry
    reset = task.reset
    rewards = task.rewards
    curriculum = task.curriculum
    domain_randomization = task.domain_randomization

    prepush_params = _prepush_term_params(task)

    env_cfg.scene.num_envs = int(task.num_envs)
    env_cfg.scene.env_spacing = float(task.env_spacing)
    env_cfg.episode_length_s = float(task.episode_length_s)

    env_cfg.actions.arm_action.use_default_offset = task.arm_action.use_default_offset
    env_cfg.actions.arm_action.offset = task.arm_action.position_offset
    env_cfg.actions.arm_action.scale = task.arm_action.position_scale
    env_cfg.actions.arm_action.clip = task.arm_action.position_clip
    env_cfg.actions.arm_action.smoothing_tau_s = task.arm_action.smoothing_tau_s
    env_cfg.actions.arm_action.max_joint_step_rad = task.arm_action.max_joint_step_rad
    _apply_arm_pd_params(env_cfg, task.arm_action)

    robot_reset_target = {
        joint_name: float(joint_pos)
        for joint_name, joint_pos in task.arm_action.position_offset.items()
    }
    robot_reset_target["joint[78]"] = PIPER_LOCKED_GRIPPER_POS

    env_cfg.events.reset_robot_joints.params.update(
        {
            "joint_position_ranges": reset.robot_joint_position_ranges,
            "velocity_range": reset.robot_joint_velocity_range,
            "joint_limit_margin": reset.robot_joint_limit_margin,
            "max_resample_attempts": reset.robot_max_resample_attempts,
            "enforce_self_collision_safety": reset.robot_enforce_self_collision_safety,
            "target_joint_positions": robot_reset_target,
            "target_joint_half_widths": curriculum.robot_reset_final_half_widths,
            "range_scale_attr": "_box_push_robot_reset_range_scale",
            "initial_range_scale": curriculum.robot_reset_initial_scale,
        }
    )
    env_cfg.events.reset_box.params.update(
        {
            "goal_center": geom.goal_center,
            "goal_center_range": geom.goal_center_xy_range,
            "goal_radius": geom.goal_radius_start,
            "spawn_z": BOX_PUSH_BOX_CFG.spawn_z,
            "radial_distance_range": reset.box_radial_distance_range,
            "workspace_bounds": geom.workspace_bounds,
            "angle_center": reset.box_angle_center,
            "angle_span_start": reset.box_angle_span_start,
            "angle_span_final": reset.box_angle_span_final,
            "box_clearance": reset.box_clearance,
        }
    )
    if domain_randomization.box_mass_range is None:
        env_cfg.events.randomize_box_mass = None
    else:
        env_cfg.events.randomize_box_mass.params.update(
            {
                "mass_distribution_params": domain_randomization.box_mass_range,
            }
        )

    if (
        domain_randomization.box_static_friction_range is None
        or domain_randomization.box_dynamic_friction_range is None
    ):
        env_cfg.events.randomize_box_material = None
    else:
        env_cfg.events.randomize_box_material.params.update(
            {
                "static_friction_range": domain_randomization.box_static_friction_range,
                "dynamic_friction_range": domain_randomization.box_dynamic_friction_range,
                "restitution_range": domain_randomization.box_restitution_range,
                "num_buckets": domain_randomization.box_material_buckets,
            }
        )

    if (
        domain_randomization.robot_stiffness_scale_range is None
        and domain_randomization.robot_damping_scale_range is None
    ):
        env_cfg.events.randomize_robot_actuator_gains = None
    else:
        env_cfg.events.randomize_robot_actuator_gains.params.update(
            {
                "stiffness_distribution_params": domain_randomization.robot_stiffness_scale_range,
                "damping_distribution_params": domain_randomization.robot_damping_scale_range,
            }
        )

    env_cfg.observations.policy.goal_pos.params["goal_center"] = geom.goal_center
    policy_prepush_setup = getattr(env_cfg.observations.policy, "prepush_setup", None)
    if policy_prepush_setup is not None:
        policy_prepush_setup.params.update(prepush_params)
    critic_prepush_setup = getattr(env_cfg.observations.critic, "prepush_setup", None)
    if critic_prepush_setup is not None:
        critic_prepush_setup.params.update(prepush_params)
    policy_goal_radius = getattr(env_cfg.observations.policy, "goal_radius", None)
    if policy_goal_radius is not None:
        policy_goal_radius.params["goal_radius"] = geom.goal_radius_start
    env_cfg.observations.critic.box_state.params.update(
        {
            "goal_center": geom.goal_center,
            "goal_radius": geom.goal_radius_start,
        }
    )

    env_cfg.rewards.alive.weight = rewards.alive_weight
    env_cfg.rewards.box_goal_dense.weight = rewards.box_goal_dense_weight
    env_cfg.rewards.box_goal_dense.params.update(
        {
            "goal_center": geom.goal_center,
            "goal_radius": geom.goal_radius_start,
            "std": rewards.box_goal_dense_std,
        }
    )
    env_cfg.rewards.box_goal_progress.weight = rewards.box_goal_progress_weight
    env_cfg.rewards.box_goal_progress.params.update(
        {
            "goal_center": geom.goal_center,
            "goal_radius": geom.goal_radius_start,
            "std": rewards.box_goal_progress_std,
        }
    )
    env_cfg.rewards.box_velocity_toward_goal.weight = rewards.box_velocity_toward_goal_weight
    env_cfg.rewards.box_velocity_toward_goal.params["goal_center"] = geom.goal_center
    env_cfg.rewards.goal_inside.weight = rewards.goal_inside_weight
    env_cfg.rewards.goal_inside.params.update(
        {
            "goal_center": geom.goal_center,
            "goal_radius": geom.goal_radius_start,
            "box_half_extent": geom.box_half_extent,
        }
    )
    env_cfg.rewards.goal_centering.weight = rewards.goal_centering_weight
    env_cfg.rewards.goal_centering.params.update(
        {
            "goal_center": geom.goal_center,
            "goal_radius": geom.goal_radius_start,
            "std": rewards.goal_centering_std,
            "box_half_extent": geom.box_half_extent,
        }
    )
    env_cfg.rewards.goal_stable.weight = rewards.goal_stable_weight
    env_cfg.rewards.goal_stable.params.update(
        {
            "goal_center": geom.goal_center,
            "goal_radius": geom.goal_radius_start,
            "std": rewards.goal_stable_std,
            "box_half_extent": geom.box_half_extent,
            "required_hold_steps": geom.goal_hold_steps,
        }
    )
    env_cfg.rewards.success.weight = rewards.success_weight
    env_cfg.rewards.success.params.update(
        {
            "goal_center": geom.goal_center,
            "goal_radius": geom.goal_radius_start,
            "required_hold_steps": geom.goal_hold_steps,
            "box_half_extent": geom.box_half_extent,
        }
    )
    env_cfg.rewards.goal_escape.weight = rewards.goal_escape_weight
    env_cfg.rewards.goal_escape.params.update(
        {
            "goal_center": geom.goal_center,
            "goal_radius": geom.goal_radius_start,
            "required_hold_steps": geom.goal_hold_steps,
            "min_hold_steps": rewards.goal_escape_min_hold_steps,
            "box_half_extent": geom.box_half_extent,
        }
    )
    env_cfg.rewards.blower_box_range.weight = rewards.blower_box_range_weight
    env_cfg.rewards.blower_box_range.params.update(
        {
            "desired_distance": rewards.blower_box_range_distance,
            "std": rewards.blower_box_range_std,
        }
    )
    env_cfg.rewards.blower_push_alignment.weight = rewards.blower_push_alignment_weight
    env_cfg.rewards.blower_push_alignment.params.update(
        {
            "goal_center": geom.goal_center,
            "nozzle_standoff": rewards.blower_push_nozzle_standoff,
            "position_std": rewards.blower_push_position_std,
            "desired_height": rewards.blower_push_height,
            "aim_height": geom.box_aim_height,
            "height_std": rewards.blower_push_height_std,
            "position_blend": rewards.blower_push_position_blend,
            "alignment_power": rewards.blower_push_alignment_power,
            "min_horizontal_fraction": rewards.blower_push_min_horizontal_fraction,
            "horizontal_std": rewards.blower_push_horizontal_std,
        }
    )
    env_cfg.rewards.blower_nozzle_low_height.weight = rewards.blower_nozzle_low_height_weight
    env_cfg.rewards.blower_nozzle_low_height.params.update(
        {
            "min_height": rewards.blower_nozzle_min_height,
            "height_margin": rewards.blower_nozzle_low_height_margin,
            "penalty_warmup_steps": rewards.airflow_penalty_warmup_steps,
            "penalty_ramp_steps": rewards.airflow_penalty_ramp_steps,
            "penalty_start_scale": rewards.airflow_penalty_start_scale,
        }
    )
    env_cfg.rewards.blower_downward_pitch.weight = rewards.blower_downward_pitch_weight
    env_cfg.rewards.blower_downward_pitch.params.update(
        {
            "angle_scale": rewards.blower_downward_pitch_angle_scale,
            "penalty_warmup_steps": rewards.airflow_penalty_warmup_steps,
            "penalty_ramp_steps": rewards.airflow_penalty_ramp_steps,
            "penalty_start_scale": rewards.airflow_penalty_start_scale,
        }
    )
    env_cfg.rewards.airflow_velocity_toward_goal.weight = rewards.airflow_velocity_toward_goal_weight
    env_cfg.rewards.airflow_velocity_toward_goal.params.update(
        {
            **prepush_params,
            "min_airflow": rewards.airflow_min_action,
            "ready_reward_floor": rewards.airflow_ready_reward_floor,
            "force_scale_n": rewards.airflow_velocity_force_scale_n,
        }
    )
    env_cfg.rewards.airflow_directed_force.weight = rewards.airflow_directed_force_weight
    env_cfg.rewards.airflow_directed_force.params.update(
        goal_center=geom.goal_center,
        goal_radius=geom.goal_radius_start,
        box_half_extent=geom.box_half_extent,
        force_scale_n=rewards.airflow_directed_force_scale_n,
        episode_credit_budget=rewards.airflow_directed_force_credit_budget,
    )
    env_cfg.rewards.airflow_before_ready.weight = rewards.airflow_before_ready_weight
    env_cfg.rewards.airflow_before_ready.params.update(
        {
            **prepush_params,
            "penalty_warmup_steps": rewards.airflow_penalty_warmup_steps,
            "penalty_ramp_steps": rewards.airflow_penalty_ramp_steps,
            "penalty_start_scale": rewards.airflow_penalty_start_scale,
        }
    )
    env_cfg.rewards.velocity_without_airflow.weight = rewards.velocity_without_airflow_weight
    env_cfg.rewards.velocity_without_airflow.params.update(
        {
            "goal_center": geom.goal_center,
            "min_airflow": rewards.airflow_min_action,
            "velocity_deadband": rewards.velocity_without_airflow_deadband,
        }
    )
    env_cfg.rewards.robot_box_proximity.weight = rewards.robot_box_proximity_weight
    env_cfg.rewards.robot_box_proximity.params.update(
        {
            "box_half_extent": geom.box_half_extent,
            "clearance": rewards.robot_box_proximity_clearance,
            "robot_cfg": SceneEntityCfg("robot", body_names=list(rewards.robot_box_proximity_body_names)),
        }
    )
    env_cfg.rewards.robot_contacts.weight = rewards.robot_contact_weight
    env_cfg.rewards.robot_contacts.params.update(
        {
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=rewards.robot_contact_body_names),
            "threshold": rewards.robot_contact_force_threshold,
        }
    )
    env_cfg.rewards.airflow_energy.weight = rewards.airflow_energy_weight
    env_cfg.rewards.airflow_energy.params.update(
        {
            "penalty_warmup_steps": rewards.airflow_penalty_warmup_steps,
            "penalty_ramp_steps": rewards.airflow_penalty_ramp_steps,
            "penalty_start_scale": rewards.airflow_penalty_start_scale,
        }
    )
    env_cfg.rewards.airflow_rate.weight = rewards.airflow_rate_weight
    env_cfg.rewards.airflow_rate.params.update(
        {
            "penalty_warmup_steps": rewards.airflow_penalty_warmup_steps,
            "penalty_ramp_steps": rewards.airflow_penalty_ramp_steps,
            "penalty_start_scale": rewards.airflow_penalty_start_scale,
        }
    )
    env_cfg.rewards.arm_action_deviation.weight = rewards.arm_action_deviation_weight
    env_cfg.rewards.arm_action_deviation.params.update(
        {
            "goal_center": geom.goal_center,
            "goal_radius": geom.goal_radius_start,
            "box_half_extent": geom.box_half_extent,
            "penalty_warmup_steps": rewards.arm_action_deviation_warmup_steps,
            "penalty_ramp_steps": rewards.arm_action_deviation_ramp_steps,
            "penalty_start_scale": rewards.arm_action_deviation_start_scale,
        }
    )
    env_cfg.rewards.action_rate.weight = rewards.action_rate_weight
    if rewards.action_rate_term_name is not None:
        env_cfg.rewards.action_rate.func = mdp.action_term_rate_l2
        env_cfg.rewards.action_rate.params = {"action_term_name": rewards.action_rate_term_name}
    env_cfg.rewards.joint_vel.weight = rewards.joint_vel_weight
    env_cfg.rewards.ee_acceleration.weight = rewards.ee_acceleration_weight
    env_cfg.rewards.ee_acceleration.params["ignore_initial_s"] = rewards.ee_acceleration_ignore_initial_s
    env_cfg.rewards.ee_acceleration.params.update(
        penalty_warmup_steps=rewards.ee_acceleration_warmup_steps,
        penalty_ramp_steps=rewards.ee_acceleration_ramp_steps,
        penalty_start_scale=rewards.ee_acceleration_start_scale,
    )
    env_cfg.rewards.box_out_of_workspace.weight = rewards.box_out_of_workspace_weight
    env_cfg.rewards.box_out_of_workspace.params["workspace_bounds"] = geom.workspace_bounds
    env_cfg.rewards.nozzle_forward.weight = rewards.nozzle_forward_penalty_weight
    env_cfg.rewards.nozzle_forward.params["min_forward_x"] = rewards.nozzle_forward_min_x
    env_cfg.rewards.nozzle_forward.params["max_forward_x"] = rewards.nozzle_forward_max_x

    if env_cfg.terminations.success is not None:
        env_cfg.terminations.success.params.update(
            {
                "goal_center": geom.goal_center,
                "goal_radius": geom.goal_radius_start,
                "required_hold_steps": geom.goal_hold_steps,
                "box_half_extent": geom.box_half_extent,
            }
        )
    env_cfg.terminations.box_out_of_workspace.params["workspace_bounds"] = geom.workspace_bounds
    env_cfg.terminations.robot_box_contact.params.update(
        {
            "box_half_extent": geom.box_half_extent,
            "clearance": rewards.robot_box_contact_clearance,
            "ignore_initial_s": rewards.robot_box_contact_ignore_initial_s,
            "robot_cfg": SceneEntityCfg("robot", body_names=list(rewards.robot_box_proximity_body_names)),
        }
    )
    env_cfg.terminations.blower_nozzle_too_low.params.update(
        {
            "min_height": rewards.blower_nozzle_min_height,
            "ignore_initial_s": rewards.robot_contact_termination_ignore_initial_s,
        }
    )
    if rewards.robot_contact_termination_body_names is None:
        env_cfg.terminations.robot_table_contact = None
    else:
        env_cfg.terminations.robot_table_contact = DoneTerm(
            func=mdp.robot_filtered_contact,
            params={
                "sensor_cfg": SceneEntityCfg(
                    "contact_forces",
                    body_names=rewards.robot_contact_termination_body_names,
                ),
                "threshold": rewards.robot_contact_termination_force_threshold,
                "ignore_initial_s": rewards.robot_contact_termination_ignore_initial_s,
            },
        )

    env_cfg.curriculum.goal_region.params.update(
        {
            "goal_center": geom.goal_center,
            "goal_radius_start": geom.goal_radius_start,
            "goal_radius_final": geom.goal_radius_final,
            "success_threshold": curriculum.success_threshold,
            "failure_threshold": curriculum.failure_threshold,
            "shrink_factor": curriculum.shrink_factor,
            "grow_factor": curriculum.grow_factor,
            "success_ema_momentum": curriculum.success_ema_momentum,
            "warmup_steps": curriculum.warmup_steps,
            "min_steps_between_updates": curriculum.min_steps_between_updates,
            "initial_range_step": curriculum.angle_range_step,
            "angle_span_start": reset.box_angle_span_start,
            "angle_span_final": reset.box_angle_span_final,
            "expand_angle_before_shrinking_goal": curriculum.expand_angle_before_shrinking_goal,
            "required_hold_steps": geom.goal_hold_steps,
            "robot_reset_shrink_enabled": curriculum.robot_reset_shrink_enabled,
            "robot_reset_success_threshold": curriculum.robot_reset_success_threshold,
            "robot_reset_failure_threshold": curriculum.robot_reset_failure_threshold,
            "robot_reset_shrink_step": curriculum.robot_reset_shrink_step,
            "robot_reset_grow_step": curriculum.robot_reset_grow_step,
            "robot_reset_final_scale": curriculum.robot_reset_final_scale,
            "robot_reset_warmup_steps": curriculum.robot_reset_warmup_steps,
            "robot_reset_min_steps_between_updates": curriculum.robot_reset_min_steps_between_updates,
        }
    )


def _apply_arm_pd_params(
    env_cfg: "PiperBlowerBoxPushInitEnvCfg",
    arm_action: BoxPushArmActionParamsCfg,
) -> None:
    """Apply task-local Piper joint PD parameters.

    These actuator gains affect both policy tracking and scripted success retreat.
    Increase damping if the arm rings around the target; reduce stiffness if it
    still overshoots after the retreat trajectory is slow enough.
    """
    actuators = env_cfg.scene.robot.actuators

    if "piper_shoulder" in actuators:
        actuators["piper_shoulder"].stiffness = float(arm_action.shoulder_stiffness)
        actuators["piper_shoulder"].damping = float(arm_action.shoulder_damping)
    if "piper_forearm" in actuators:
        actuators["piper_forearm"].stiffness = float(arm_action.forearm_stiffness)
        actuators["piper_forearm"].damping = float(arm_action.forearm_damping)


def _make_airflow_action_cfg(task: BoxPushTaskParamsCfg) -> SmoothedAirflowActionTermCfg:
    target_sampling_cfg = SamplingCfg(target_area=task.airflow.target_area)
    viz_cfg = _make_airflow_viz_cfg(task)

    return SmoothedAirflowActionTermCfg(
        asset_name="box",
        source_type="articulation_body",
        source_asset_name="robot",
        source_body_name=PIPER_END_EFFECTOR_BODY,
        nozzle_offset_pos=PIPER_NOZZLE_OFFSET_POS,
        nozzle_offset_quat=PIPER_NOZZLE_OFFSET_QUAT,
        nozzle_local_axis=PIPER_NOZZLE_LOCAL_AXIS,
        target_type=BOX_PUSH_BOX_CFG.target_type,
        target_object_cfg=BOX_PUSH_BOX_CFG,
        targets=(
            AirflowTargetCfg(
                asset_name="box",
                target_type=BOX_PUSH_BOX_CFG.target_type,
                target_object_cfg=BOX_PUSH_BOX_CFG,
                sampling_cfg=target_sampling_cfg,
            ),
        ),
        mode="surface",
        min_axial_distance=task.airflow.min_axial_distance,
        velocity_cutoff=task.airflow.velocity_cutoff,
        jet_cfg=AIRFLOW_JET_CFG,
        sampling_cfg=target_sampling_cfg,
        viz_cfg=viz_cfg,
        action_mapping=task.airflow.action_mapping,
        smoothing_tau_s=task.airflow.smoothing_tau_s,
        reset_settle_s=task.airflow.reset_settle_s,
        apply_reaction_force=task.airflow.apply_reaction_force,
        reaction_force_max=task.airflow.reaction_force_max,
        airflow_u0_range=task.airflow.u0_range,
        airflow_spread_k_range=task.airflow.spread_k_range,
        airflow_randomization_buckets=task.airflow.randomization_buckets,
    )


# ---------------------------------------------------------------------
# Base env cfg wiring
# ---------------------------------------------------------------------


@configclass
class PiperBlowerBoxPushInitEnvCfg(joint_pos_env_cfg.PiperReachEnvCfg):
    """Initialize the box-push task from BoxPushTaskParamsCfg."""

    box_push: BoxPushTaskParamsCfg = BoxPushTaskParamsCfg()

    def __post_init__(self):
        super().__post_init__()

        self.sim.dt = PIPER_SIM_DT
        self.decimation = PIPER_CONTROL_DECIMATION

        self.observations = ObservationsCfg()
        self.events = EventCfg()
        self.rewards = RewardsCfg()
        self.terminations = TerminationsCfg()
        self.curriculum = CurriculumCfg()
        self.actions.arm_action = mdp.SmoothedJointPositionActionCfg(
            asset_name="robot",
            joint_names=PIPER_ARM_JOINTS,
            scale=0.5,
            use_default_offset=True,
        )

        _apply_task_params(self, self.box_push)

        self.commands.ee_pose.debug_vis = False

        geom = self.box_push.geometry
        reset = self.box_push.reset

        self.scene.robot.spawn.activate_contact_sensors = True
        self.scene.contact_forces = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Robot/base_link/.*",
            history_length=3,
            filter_prim_paths_expr=["{ENV_REGEX_NS}/Table"],
        )

        self.scene.box = BOX_PUSH_BOX_CFG.to_rigid_object_cfg(prim_path="{ENV_REGEX_NS}/Box")
        self.scene.box.init_state.pos = (
            geom.goal_center[0] - reset.box_radial_distance_range[0],
            geom.goal_center[1],
            BOX_PUSH_BOX_CFG.spawn_z,
        )

        self.scene.goal_region = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/GoalRegion",
            spawn=mdp.GoalRegionWireframeCfg(
                half_extent=geom.goal_radius_start,
                line_width=0.008,
                color=(1.0, 0.0, 0.0),
                opacity=1.0,
            ),
            init_state=AssetBaseCfg.InitialStateCfg(
                pos=geom.goal_center,
                rot=(1.0, 0.0, 0.0, 0.0),
            ),
        )

        self.actions.airflow = _make_airflow_action_cfg(self.box_push)
