# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Parameter override cfgs for the airflow box-push task."""

from __future__ import annotations

import math

from isaaclab.utils import configclass

from .blower_box_push_env_cfg import PiperBlowerBoxPushInitEnvCfg
from .piper_l_cfg import PIPER_LOCKED_GRIPPER_POS


BOX_PUSH_TRAIN_AIRFLOW_VIZ = {
    "show_jet_cone": False,
    "show_laser_ray": False,
    "show_sample_markers": False,
    "show_reaction_force_arrow": False,
    "max_visualized_envs": 16,
}


BOX_PUSH_PLAY_AIRFLOW_VIZ = {
    "show_jet_cone": False,
    "show_laser_ray": True,
    "show_sample_markers": False,
    "show_reaction_force_arrow": True,
    "max_visualized_envs": 16,
}


def _set_goal_radius_param(term, goal_radius: float) -> None:
    params = getattr(term, "params", None)
    if isinstance(params, dict) and "goal_radius" in params:
        params["goal_radius"] = goal_radius


def _apply_play_final_curriculum_state(env_cfg: "PiperBlowerBoxPushEnvCfg_PLAY") -> None:
    """Freeze PLAY at the terminal goal-size curriculum state."""
    final_goal_radius = float(env_cfg.box_push.geometry.goal_radius_final)
    env_cfg.box_push.geometry.goal_radius_start = final_goal_radius

    env_cfg.events.reset_box.params["goal_radius"] = final_goal_radius
    policy_goal_radius = getattr(env_cfg.observations.policy, "goal_radius", None)
    if policy_goal_radius is not None:
        policy_goal_radius.params["goal_radius"] = final_goal_radius
    env_cfg.observations.critic.box_state.params["goal_radius"] = final_goal_radius

    for term_name in (
        "box_goal_dense",
        "box_goal_progress",
        "goal_inside",
        "goal_centering",
        "goal_stable",
        "success",
        "goal_escape",
        "airflow_directed_force",
    ):
        _set_goal_radius_param(getattr(env_cfg.rewards, term_name, None), final_goal_radius)

    _set_goal_radius_param(getattr(env_cfg.terminations, "success", None), final_goal_radius)

    env_cfg.curriculum.goal_region.params.update(
        {
            "goal_radius_start": final_goal_radius,
            "goal_radius_final": final_goal_radius,
            "initial_range_step": 0.0,
        }
    )

    if hasattr(env_cfg.scene, "goal_region"):
        env_cfg.scene.goal_region.spawn.half_extent = final_goal_radius


def _configure_airflow_visualizations(
    env_cfg,
    *,
    show_jet_cone: bool = False,
    show_laser_ray: bool = True,
    show_sample_markers: bool = False,
    show_reaction_force_arrow: bool = True,
    max_visualized_envs: int | None = None,
) -> None:
    if max_visualized_envs is None:
        resolved_max_visualized_envs = int(env_cfg.scene.num_envs)
    else:
        resolved_max_visualized_envs = max(0, int(max_visualized_envs))

    airflow_params = getattr(getattr(env_cfg, "box_push", None), "airflow", None)
    viz_params = getattr(airflow_params, "viz", None)
    if viz_params is not None:
        viz_params.show_jet_cone = bool(show_jet_cone)
        viz_params.show_laser_ray = bool(show_laser_ray)
        viz_params.show_sample_markers = bool(show_sample_markers)
        viz_params.show_reaction_force_arrow = bool(show_reaction_force_arrow)
        viz_params.max_visualized_envs = resolved_max_visualized_envs

    airflow_cfg = getattr(getattr(env_cfg, "actions", None), "airflow", None)
    viz_cfg = getattr(airflow_cfg, "viz_cfg", None)
    if viz_cfg is None:
        return

    viz_cfg.show_jet_cone = bool(show_jet_cone)
    viz_cfg.show_laser_ray = bool(show_laser_ray)
    viz_cfg.show_sample_markers = bool(show_sample_markers)
    viz_cfg.show_reaction_force_arrow = bool(show_reaction_force_arrow)
    viz_cfg.max_visualized_envs = resolved_max_visualized_envs


@configclass
class PiperBlowerBoxPushEnvCfg(PiperBlowerBoxPushInitEnvCfg):
    """Editable parameter override layer for training.

    Keep the action space end-to-end: this class only changes reset, reward, and
    curriculum parameters.
    """

    def __post_init__(self):
        # Reset/exploration overrides.
        self.box_push.reset.robot_joint_position_ranges.update(
            {
                "joint1": (-0.80, 0.80),
                "joint2": (0.00, 0.40),
                "joint3": (-0.50, 0.00),
                "joint4": (-0.80, 0.80),
                "joint5": (-0.90, 0.50),
                "joint6": (-3.10, -1.70),
                "joint[78]": (PIPER_LOCKED_GRIPPER_POS, PIPER_LOCKED_GRIPPER_POS),
            }
        )
        self.box_push.reset.robot_joint_velocity_range = (-0.02, 0.02)
        self.box_push.reset.robot_joint_limit_margin = 0.04
        self.box_push.reset.robot_max_resample_attempts = 16
        self.box_push.reset.robot_enforce_self_collision_safety = False
        self.box_push.reset.box_radial_distance_range = (0.08, 0.18)
        self.box_push.reset.box_angle_center = math.pi
        self.box_push.reset.box_angle_span_start = 0.35
        self.box_push.reset.box_angle_span_final = 1.20
        self.box_push.reset.box_clearance = 0.015
        self.box_push.geometry.goal_center_xy_range = (0.55, 0.75, -0.08, 0.08)
        self.box_push.arm_action.smoothing_tau_s = 0.12

        # Pre-push setup target.
        self.box_push.prepush.nozzle_standoff = 0.10
        self.box_push.prepush.nozzle_height = 0.22
        self.box_push.prepush.position_std = 0.07
        self.box_push.prepush.height_std = 0.06
        self.box_push.prepush.axis_std = 0.22

        # Piper starts with a narrow robot-side sector and expands conservatively.
        self.box_push.curriculum.success_threshold = 0.72
        self.box_push.curriculum.failure_threshold = 0.25
        self.box_push.curriculum.angle_range_step = 0.0
        self.box_push.curriculum.expand_angle_before_shrinking_goal = False
        self.box_push.curriculum.robot_reset_initial_scale = 1.0
        self.box_push.curriculum.robot_reset_shrink_enabled = True
        self.box_push.curriculum.robot_reset_success_threshold = 0.85
        self.box_push.curriculum.robot_reset_failure_threshold = 0.35
        self.box_push.curriculum.robot_reset_shrink_step = 0.10
        self.box_push.curriculum.robot_reset_grow_step = 0.05
        self.box_push.curriculum.robot_reset_final_scale = 0.0
        self.box_push.curriculum.robot_reset_warmup_steps = 6000
        self.box_push.curriculum.robot_reset_min_steps_between_updates = 1500

        # Reward balance.
        self.box_push.rewards.box_goal_dense_weight = 2.0
        self.box_push.rewards.box_goal_progress_weight = 1.0
        self.box_push.rewards.box_velocity_toward_goal_weight = 0.0
        self.box_push.rewards.goal_inside_weight = 2.0
        self.box_push.rewards.goal_centering_weight = 2.0
        self.box_push.rewards.goal_centering_std = 0.055
        self.box_push.rewards.goal_stable_weight = 2.0
        self.box_push.rewards.goal_stable_std = 0.055
        self.box_push.rewards.goal_escape_min_hold_steps = 1
        self.box_push.rewards.success_weight = 0.0
        self.box_push.rewards.goal_escape_weight = 0.0
        self.box_push.rewards.blower_box_range_weight = 0.8
        self.box_push.rewards.blower_box_range_distance = 0.11
        self.box_push.rewards.blower_box_range_std = 0.08
        self.box_push.rewards.blower_push_alignment_weight = 1.0
        self.box_push.rewards.blower_push_nozzle_standoff = 0.10
        self.box_push.rewards.blower_push_position_std = 0.14
        self.box_push.rewards.blower_push_height = 0.22
        self.box_push.rewards.blower_push_height_std = 0.08
        self.box_push.rewards.blower_push_position_blend = 0.50
        self.box_push.rewards.blower_push_min_horizontal_fraction = 0.75
        self.box_push.rewards.airflow_velocity_toward_goal_weight = 0.0
        self.box_push.rewards.velocity_without_airflow_weight = 0.0
        self.box_push.rewards.robot_box_proximity_weight = -6.0
        self.box_push.rewards.robot_box_proximity_clearance = 0.06
        self.box_push.rewards.robot_box_contact_clearance = 0.025
        self.box_push.rewards.robot_box_contact_ignore_initial_s = 0.15
        self.box_push.rewards.robot_box_proximity_body_names = ("link6", "link7", "link8")
        self.box_push.rewards.robot_contact_weight = -12.0
        self.box_push.rewards.robot_contact_force_threshold = 0.5
        self.box_push.rewards.robot_contact_body_names = "link[4-8]"
        self.box_push.rewards.airflow_energy_weight = -0.005
        self.box_push.rewards.airflow_rate_weight = -0.005
        self.box_push.rewards.airflow_penalty_warmup_steps = 5000
        self.box_push.rewards.airflow_penalty_ramp_steps = 20000
        self.box_push.rewards.airflow_penalty_start_scale = 0.0
        self.box_push.rewards.action_rate_weight = -0.005
        self.box_push.rewards.joint_vel_weight = -0.002

        super().__post_init__()
        _configure_airflow_visualizations(self, **BOX_PUSH_TRAIN_AIRFLOW_VIZ)

        # Keep held-success as an engineering signal for metrics, curriculum,
        # and PLAY retreat, but do not end training episodes on success.
        self.rewards.success = None
        self.rewards.goal_escape = None
        self.terminations.success = None


@configclass
class PiperBlowerBoxPushEnvCfg_PLAY(PiperBlowerBoxPushEnvCfg):
    """Evaluation/visualization cfg.

    Freeze curriculum-driven task parameters at their final evaluation values.
    For cleaner visual inspection, start the robot from the exact scripted
    success-retreat working posture.
    """

    def __post_init__(self):
        super().__post_init__()

        _apply_play_final_curriculum_state(self)

        # Match the robot reset posture to play.py's scripted success-retreat
        # target: zero raw arm action -> BoxPushArmActionParamsCfg.position_offset.
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
            }
        )

        # Visualization / evaluation tweaks that are independent of reset
        # randomization.
        self.scene.num_envs = 16
        self.observations.policy.enable_corruption = False
        self.observations.policy.box_pose.params["noise_range"] = (0.0, 0.0)
        self.events.randomize_robot_actuator_gains = None
        self.events.randomize_box_material = None
        self.events.randomize_box_mass = None
        self.actions.airflow.airflow_u0_range = None
        self.actions.airflow.airflow_spread_k_range = None
        _configure_airflow_visualizations(self, **BOX_PUSH_PLAY_AIRFLOW_VIZ)
