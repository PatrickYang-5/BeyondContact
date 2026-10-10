# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reward terms for the airflow box-push task."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import ManagerTermBase, SceneEntityCfg
from isaaclab.utils.math import quat_apply

from .box_push_common import (
    _airflow_nozzle_pose_and_axis_w,
    _box_fully_inside_goal,
    _box_goal_distance,
    _box_xy_w,
    _desired_nozzle_position_w,
    _ensure_box_push_state,
    _goal_center_xy_w,
    _goal_radius,
    _log_mean,
    _prepush_setup_terms,
    _update_goal_hold_state,
)
from .box_push_observation import box_to_goal_xy, nozzle_to_box
from .box_push_termination import box_out_of_workspace

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class EndEffectorAccelerationPenalty(ManagerTermBase):
    """L2 squared acceleration of a body-local point over policy steps.

    Uses actual point velocity v_link + omega cross r, not action differences.
    Returns ax^2 + ay^2 + az^2, with acceleration in m/s^2 and no threshold.
    Per-environment history is invalidated on reset to exclude reset impulses.
    """

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._previous_velocity = torch.zeros((env.num_envs, 3), device=env.device)
        self._valid = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    def reset(self, env_ids=None):
        self._valid[slice(None) if env_ids is None else env_ids] = False

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        asset_cfg: SceneEntityCfg,
        point_offset: tuple[float, float, float] = (0.08, 0.0, 0.05),
        ignore_initial_s: float = 0.1,
        penalty_warmup_steps: int = 0,
        penalty_ramp_steps: int = 0,
        penalty_start_scale: float = 1.0,
    ) -> torch.Tensor:
        robot: Articulation = env.scene[asset_cfg.name]
        velocity = robot.data.body_link_lin_vel_w[:, asset_cfg.body_ids]
        if velocity.shape[1] != 1:
            raise ValueError("EndEffectorAccelerationPenalty requires exactly one body")
        angular_velocity = robot.data.body_link_ang_vel_w[:, asset_cfg.body_ids][:, 0]
        orientation = robot.data.body_link_quat_w[:, asset_cfg.body_ids][:, 0]
        offset = velocity.new_tensor(point_offset).expand(env.num_envs, -1)
        radius_w = quat_apply(orientation, offset)
        point_velocity = velocity[:, 0] + torch.cross(angular_velocity, radius_w, dim=-1)
        acceleration = (point_velocity - self._previous_velocity) / env.step_dt
        ready = self._valid & (env.episode_length_buf * env.step_dt > ignore_initial_s)
        acceleration = torch.where(ready.unsqueeze(-1), acceleration, torch.zeros_like(acceleration))
        penalty = acceleration.square().sum(dim=-1)
        self._previous_velocity.copy_(point_velocity.detach())
        self._valid[:] = True
        _log_mean(env, "Metrics/ee_acceleration_mps2", torch.linalg.vector_norm(acceleration, dim=-1))
        _log_mean(env, "Metrics/ee_acceleration_l2", penalty)
        scale = _global_step_penalty_scale(
            env, penalty_warmup_steps, penalty_ramp_steps, penalty_start_scale,
            log_name="Metrics/ee_acceleration_penalty_scale",
        )
        return penalty * scale


class AirflowDirectedForceReward(ManagerTermBase):
    """Bounded per-episode credit for starting a useful jet, even before sliding.

    A stationary box can earn this credit only until its episode budget is used.
    Further reward must come from motion/task terms. Budget is measured in
    integral reward units before the manager's weight, not in wall-clock time.
    """

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._used_credit = torch.zeros(env.num_envs, device=env.device)

    def reset(self, env_ids=None):
        self._used_credit[slice(None) if env_ids is None else env_ids] = 0.0

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        goal_center: Sequence[float],
        goal_radius: float,
        box_half_extent: float = 0.04,
        force_scale_n: float = 0.8,
        episode_credit_budget: float = 1.0,
        airflow_term_name: str = "airflow",
        asset_cfg: SceneEntityCfg = SceneEntityCfg("box"),
    ) -> torch.Tensor:
        if force_scale_n <= 0.0 or episode_credit_budget < 0.0:
            raise ValueError("force_scale_n must be positive and episode_credit_budget nonnegative")
        airflow = env.action_manager.get_term(airflow_term_name)
        delta = box_to_goal_xy(env, goal_center, asset_cfg)
        direction = delta / torch.linalg.vector_norm(delta, dim=1, keepdim=True).clamp_min(1e-6)
        force = airflow.target_force_w(env.scene[asset_cfg.name])[:, :2]
        directed_force = (force * direction).sum(dim=1).clamp_min(0.0)
        inside = _box_fully_inside_goal(env, asset_cfg, goal_center, goal_radius, box_half_extent)
        outside = (~inside).float()
        outside_count = outside.sum().clamp_min(1.0)
        _log_mean(env, "Metrics/airflow_usage_outside_goal",
                  (airflow.processed_actions[:, 0] * outside).sum() / outside_count)
        _log_mean(env, "Metrics/airflow_force_outside_goal_n",
                  (directed_force * outside).sum() / outside_count)
        reward_rate = (directed_force / force_scale_n).clamp(max=1.0)
        reward_rate *= (~inside) & (airflow.processed_actions[:, 0] > 0.0)
        remaining = (episode_credit_budget - self._used_credit).clamp_min(0.0)
        credit = torch.minimum(reward_rate * env.step_dt, remaining)
        self._used_credit += credit.detach()
        _log_mean(env, "Metrics/airflow_startup_credit_used", self._used_credit)
        return credit / env.step_dt


def action_term_rate_l2(
    env: ManagerBasedRLEnv, action_term_name: str = "arm_action",
) -> torch.Tensor:
    """Raw action change for one term; other actuators have separate penalties."""
    manager = env.action_manager
    index = manager.active_terms.index(action_term_name)
    start = sum(manager.action_term_dim[:index])
    stop = start + manager.action_term_dim[index]
    return (manager.action[:, start:stop] - manager.prev_action[:, start:stop]).square().sum(dim=1)


def box_goal_distance_tanh(
    env: ManagerBasedRLEnv,
    std: float,
    goal_center: Sequence[float],
    goal_radius: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("box"),
    box_half_extent: float | Sequence[float] = 0.0,
) -> torch.Tensor:
    distance = _box_goal_distance(env, asset_cfg, goal_center, goal_radius, box_half_extent)

    _log_mean(env, "Metrics/box_to_goal_distance", distance)
    _log_mean(env, "Metrics/goal_radius", _goal_radius(env, goal_radius))

    return 1.0 - torch.tanh(distance / std)


def box_goal_progress(
    env: ManagerBasedRLEnv,
    std: float,
    goal_center: Sequence[float],
    goal_radius: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("box"),
    box_half_extent: float | Sequence[float] = 0.0,
) -> torch.Tensor:
    distance = _box_goal_distance(env, asset_cfg, goal_center, goal_radius, box_half_extent)

    if not hasattr(env, "_box_push_prev_goal_distance"):
        _ensure_box_push_state(env, goal_radius)
        env._box_push_prev_goal_distance[:] = distance

    reward = (env._box_push_prev_goal_distance - distance) / std
    env._box_push_prev_goal_distance[:] = distance.detach()

    return reward


def box_goal_inside(
    env: ManagerBasedRLEnv,
    goal_center: Sequence[float],
    goal_radius: float,
    box_half_extent: float = 0.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("box"),
) -> torch.Tensor:
    """Instantaneous inside-goal indicator.

    This is only a shaping signal. It is not terminal success.
    """
    _ensure_box_push_state(env, goal_radius)

    inside = _box_fully_inside_goal(
        env,
        asset_cfg=asset_cfg,
        goal_center=goal_center,
        goal_radius=goal_radius,
        box_half_extent=box_half_extent,
    )

    _log_mean(env, "Metrics/goal_inside_rate", inside.float())

    return inside.float()


def box_goal_centering_reward(
    env: ManagerBasedRLEnv,
    goal_center: Sequence[float],
    goal_radius: float,
    std: float,
    box_half_extent: float = 0.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("box"),
) -> torch.Tensor:
    """Reward being inside the goal and closer to its center."""
    _ensure_box_push_state(env, goal_radius)

    box_delta = _box_xy_w(env, asset_cfg) - _goal_center_xy_w(env, goal_center)
    inside = _box_fully_inside_goal(
        env,
        asset_cfg=asset_cfg,
        goal_center=goal_center,
        goal_radius=goal_radius,
        box_half_extent=box_half_extent,
    ).float()
    center_distance = torch.linalg.norm(box_delta, dim=1)
    reward = inside * torch.exp(-torch.square(center_distance / max(float(std), 1e-6)))

    _log_mean(env, "Metrics/box_to_goal_center_distance", center_distance)

    return reward


def box_goal_hold_fraction(
    env: ManagerBasedRLEnv,
    goal_center: Sequence[float],
    goal_radius: float,
    required_hold_steps: int,
    box_half_extent: float = 0.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("box"),
) -> torch.Tensor:
    """Reward how much of the required hold duration has been completed.

    Returns a value in [0, 1].
    """
    _update_goal_hold_state(
        env,
        goal_center=goal_center,
        goal_radius=goal_radius,
        required_hold_steps=required_hold_steps,
        box_half_extent=box_half_extent,
        asset_cfg=asset_cfg,
    )

    hold_fraction = env._box_push_goal_hold_steps.float() / max(int(required_hold_steps), 1)
    hold_fraction = torch.clamp(hold_fraction, 0.0, 1.0)

    _log_mean(env, "Metrics/goal_hold_fraction", hold_fraction)

    return hold_fraction


def box_goal_held_success(
    env: ManagerBasedRLEnv,
    goal_center: Sequence[float],
    goal_radius: float,
    required_hold_steps: int,
    box_half_extent: float = 0.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("box"),
) -> torch.Tensor:
    """Return 1.0 only after the box has stayed inside the goal long enough."""
    _update_goal_hold_state(
        env,
        goal_center=goal_center,
        goal_radius=goal_radius,
        required_hold_steps=required_hold_steps,
        box_half_extent=box_half_extent,
        asset_cfg=asset_cfg,
    )

    success = env._box_push_goal_held_success.float()

    _log_mean(env, "Metrics/held_success_rate_step", success)

    return success


def box_goal_escape_penalty(
    env: ManagerBasedRLEnv,
    goal_center: Sequence[float],
    goal_radius: float,
    required_hold_steps: int,
    min_hold_steps: int = 1,
    box_half_extent: float = 0.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("box"),
) -> torch.Tensor:
    """Penalize leaving the goal after entry, scaled by the best hold progress so far."""
    _update_goal_hold_state(
        env,
        goal_center=goal_center,
        goal_radius=goal_radius,
        required_hold_steps=required_hold_steps,
        box_half_extent=box_half_extent,
        asset_cfg=asset_cfg,
    )

    escaped = (
        env._box_push_goal_left_after_entry
        & (env._box_push_goal_max_hold_steps >= int(min_hold_steps))
        & (~env._box_push_goal_held_success)
    )
    hold_progress = env._box_push_goal_max_hold_steps.float() / max(int(required_hold_steps), 1)
    penalty = escaped.float() * torch.clamp(hold_progress, 0.0, 1.0)

    _log_mean(env, "Metrics/goal_escape_rate", escaped.float())
    _log_mean(env, "Metrics/goal_escape_penalty_scale", penalty)

    return penalty


def box_goal_success(
    env: ManagerBasedRLEnv,
    goal_center: Sequence[float],
    goal_radius: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("box"),
) -> torch.Tensor:
    """Backward-compatible instantaneous success.

    The new cfg does not use this for termination. It is kept only so older
    configs do not break.
    """
    return box_goal_inside(env, goal_center, goal_radius, asset_cfg=asset_cfg)


def airflow_energy(
    env: ManagerBasedRLEnv,
    airflow_term_name: str = "airflow",
    penalty_warmup_steps: int = 0,
    penalty_ramp_steps: int = 0,
    penalty_start_scale: float = 1.0,
) -> torch.Tensor:
    airflow_term = env.action_manager.get_term(airflow_term_name)
    usage = airflow_term.processed_actions[:, 0]
    penalty_scale = _global_step_penalty_scale(
        env,
        warmup_steps=penalty_warmup_steps,
        ramp_steps=penalty_ramp_steps,
        start_scale=penalty_start_scale,
        log_name="Metrics/airflow_penalty_scale",
    )

    _log_mean(env, "Metrics/airflow_usage", usage)
    _log_mean(env, "Metrics/airflow_high_usage_fraction", (usage >= 0.7).float())
    _log_mean(env, "Metrics/airflow_raw_negative_fraction", (airflow_term.raw_actions[:, 0] < 0.0).float())

    return usage.square() * penalty_scale


def airflow_rate_l2(
    env: ManagerBasedRLEnv,
    airflow_term_name: str = "airflow",
    penalty_warmup_steps: int = 0,
    penalty_ramp_steps: int = 0,
    penalty_start_scale: float = 1.0,
) -> torch.Tensor:
    airflow_term = env.action_manager.get_term(airflow_term_name)
    previous = getattr(airflow_term, "previous_processed_actions", airflow_term.raw_actions)
    penalty_scale = _global_step_penalty_scale(
        env,
        warmup_steps=penalty_warmup_steps,
        ramp_steps=penalty_ramp_steps,
        start_scale=penalty_start_scale,
    )

    return (airflow_term.processed_actions[:, 0] - previous[:, 0]).square() * penalty_scale


def arm_action_deviation_l2(
    env: ManagerBasedRLEnv,
    action_term_name: str = "arm_action",
    goal_center: Sequence[float] | None = None,
    goal_radius: float | None = None,
    box_half_extent: float = 0.0,
    penalty_warmup_steps: int = 0,
    penalty_ramp_steps: int = 0,
    penalty_start_scale: float = 1.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("box"),
) -> torch.Tensor:
    """Penalize joint-position targets deviating from the zero-action target.

    For the box-push joint policy, a zero raw arm action maps to the configured
    task working posture through the action offset. This term is therefore a
    task-space "small action" regularizer expressed in joint-target space. If a
    goal region is provided, the penalty is only active after the box is inside
    the goal so exploration and setup pushes are not suppressed.
    """
    action_term = env.action_manager.get_term(action_term_name)
    processed = action_term.processed_actions

    if hasattr(action_term, "zero_action_joint_targets"):
        reference = action_term.zero_action_joint_targets
    else:
        reference = torch.zeros_like(processed)

    deviation = processed - reference
    penalty = torch.sum(deviation.square(), dim=1)

    if goal_center is not None and goal_radius is not None:
        inside = _box_fully_inside_goal(
            env,
            asset_cfg=asset_cfg,
            goal_center=goal_center,
            goal_radius=float(goal_radius),
            box_half_extent=box_half_extent,
        ).float()
        penalty = penalty * inside
        _log_mean(env, "Metrics/arm_action_deviation_gate", inside)

    penalty_scale = _global_step_penalty_scale(
        env,
        warmup_steps=penalty_warmup_steps,
        ramp_steps=penalty_ramp_steps,
        start_scale=penalty_start_scale,
        log_name="Metrics/arm_action_deviation_penalty_scale",
    )
    penalty = penalty * penalty_scale

    _log_mean(env, "Metrics/arm_action_deviation_l2", penalty)

    return penalty


def _global_step_penalty_scale(
    env: ManagerBasedRLEnv,
    warmup_steps: int,
    ramp_steps: int,
    start_scale: float,
    log_name: str | None = None,
) -> torch.Tensor:
    start_scale = min(max(float(start_scale), 0.0), 1.0)
    warmup_steps = max(int(warmup_steps), 0)
    ramp_steps = max(int(ramp_steps), 0)
    step = float(getattr(env, "common_step_counter", 0))

    if step < warmup_steps:
        scale = start_scale
    elif ramp_steps <= 0:
        scale = 1.0
    else:
        progress = min(max((step - float(warmup_steps)) / float(ramp_steps), 0.0), 1.0)
        scale = start_scale + (1.0 - start_scale) * progress

    scale_tensor = torch.tensor(scale, dtype=torch.float32, device=env.device)
    if log_name is not None:
        _log_mean(env, log_name, scale_tensor)
    return scale_tensor


def _airflow_usage_gate(
    env: ManagerBasedRLEnv,
    airflow_term_name: str,
    min_airflow: float,
) -> torch.Tensor:
    airflow_term = env.action_manager.get_term(airflow_term_name)
    usage = airflow_term.processed_actions[:, 0]

    if min_airflow <= 0.0:
        return torch.clamp(usage, 0.0, 1.0)

    return torch.clamp((usage - float(min_airflow)) / (1.0 - float(min_airflow)), 0.0, 1.0)


def blower_box_distance_reward(
    env: ManagerBasedRLEnv,
    desired_distance: float,
    std: float,
    airflow_term_name: str = "airflow",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("box"),
) -> torch.Tensor:
    distance = torch.linalg.norm(nozzle_to_box(env, airflow_term_name, asset_cfg), dim=1)
    reward = torch.exp(-torch.square((distance - desired_distance) / std))

    return reward


def blower_horizontal_push_alignment_reward(
    env: ManagerBasedRLEnv,
    goal_center: Sequence[float],
    nozzle_standoff: float = 0.14,
    position_std: float = 0.14,
    desired_height: float = 0.13,
    aim_height: float = 0.065,
    height_std: float = 0.07,
    position_blend: float = 0.50,
    min_horizontal_fraction: float = 0.75,
    horizontal_std: float = 0.20,
    airflow_term_name: str = "airflow",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("box"),
    alignment_power: float = 1.0,
) -> torch.Tensor:
    """Reward standing upstream of the box and blowing horizontally toward the goal."""
    box: RigidObject = env.scene[asset_cfg.name]
    nozzle_pos_w, _, axis_w = _airflow_nozzle_pose_and_axis_w(env, airflow_term_name)

    desired_nozzle_pos, box_to_goal_dir = _desired_nozzle_position_w(
        env,
        goal_center=goal_center,
        nozzle_standoff=nozzle_standoff,
        desired_height=desired_height,
        asset_cfg=asset_cfg,
    )

    position_error_xy = torch.linalg.norm(nozzle_pos_w[:, :2] - desired_nozzle_pos[:, :2], dim=1)
    height_error = torch.abs(nozzle_pos_w[:, 2] - desired_nozzle_pos[:, 2])
    upstream_position = torch.exp(
        -torch.square(position_error_xy / max(float(position_std), 1e-6))
        -torch.square(height_error / max(float(height_std), 1e-6))
    )

    target_pos = box.data.root_pos_w.clone()
    target_pos[:, :2] = box.data.root_pos_w[:, :2] - 0.5 * float(nozzle_standoff) * box_to_goal_dir
    target_pos[:, 2] = env.scene.env_origins[:, 2] + float(aim_height)

    nozzle_to_target = target_pos - nozzle_pos_w
    nozzle_to_target_dir = nozzle_to_target / torch.linalg.norm(
        nozzle_to_target,
        dim=1,
        keepdim=True,
    ).clamp_min(1e-6)

    axis_to_box = torch.clamp(torch.sum(axis_w * nozzle_to_target_dir, dim=1), min=0.0)

    axis_xy = axis_w[:, :2]
    horizontal_fraction = torch.linalg.norm(axis_xy, dim=1).clamp(0.0, 1.0)
    axis_xy_dir = axis_xy / horizontal_fraction.unsqueeze(1).clamp_min(1e-6)

    axis_to_goal = torch.clamp(torch.sum(axis_xy_dir * box_to_goal_dir, dim=1), min=0.0)
    horizontal_gate = torch.exp(
        -torch.square(
            torch.clamp(float(min_horizontal_fraction) - horizontal_fraction, min=0.0)
            / max(float(horizontal_std), 1e-6)
        )
    )

    directional_push = (axis_to_box * axis_to_goal).pow(max(float(alignment_power), 1.0)) * horizontal_gate
    position_blend = min(max(float(position_blend), 0.0), 1.0)

    return upstream_position * (position_blend + (1.0 - position_blend) * directional_push)


def blower_nozzle_low_height_with_airflow_penalty(
    env: ManagerBasedRLEnv,
    min_height: float,
    height_margin: float,
    penalty_warmup_steps: int = 0,
    penalty_ramp_steps: int = 0,
    penalty_start_scale: float = 1.0,
    airflow_term_name: str = "airflow",
) -> torch.Tensor:
    """Penalize blowing while the nozzle is below the configured minimum height."""
    airflow_term = env.action_manager.get_term(airflow_term_name)
    usage = torch.clamp(airflow_term.processed_actions[:, 0], min=0.0)
    nozzle_pos_w, _, _ = _airflow_nozzle_pose_and_axis_w(env, airflow_term_name)

    # Keep the minimum nozzle height table-relative, independent of the robot
    # base mounting height.
    table_z = env.scene.env_origins[:, 2]
    min_z = table_z + float(min_height)
    low_violation = torch.clamp(
        (min_z - nozzle_pos_w[:, 2]) / max(float(height_margin), 1e-6),
        min=0.0,
    )
    penalty_scale = _global_step_penalty_scale(
        env,
        warmup_steps=penalty_warmup_steps,
        ramp_steps=penalty_ramp_steps,
        start_scale=penalty_start_scale,
    )
    penalty = usage * low_violation.square() * penalty_scale

    _log_mean(env, "Metrics/blower_nozzle_low_height_violation", torch.clamp(min_z - nozzle_pos_w[:, 2], min=0.0))

    return penalty


def blower_downward_pitch_with_airflow_penalty(
    env: ManagerBasedRLEnv,
    angle_scale: float,
    penalty_warmup_steps: int = 0,
    penalty_ramp_steps: int = 0,
    penalty_start_scale: float = 1.0,
    airflow_term_name: str = "airflow",
) -> torch.Tensor:
    """Penalize blowing with a downward nozzle pitch.

    The penalty is linear in both airflow action and downward pitch angle. A
    horizontal or upward-pointing airflow axis has zero penalty.
    """
    airflow_term = env.action_manager.get_term(airflow_term_name)
    usage = torch.clamp(airflow_term.processed_actions[:, 0], min=0.0)
    _, _, axis_w = _airflow_nozzle_pose_and_axis_w(env, airflow_term_name)

    horizontal = torch.linalg.norm(axis_w[:, :2], dim=1).clamp_min(1e-6)
    downward = torch.clamp(-axis_w[:, 2], min=0.0)
    downward_pitch = torch.atan2(downward, horizontal)
    penalty_scale = _global_step_penalty_scale(
        env,
        warmup_steps=penalty_warmup_steps,
        ramp_steps=penalty_ramp_steps,
        start_scale=penalty_start_scale,
    )
    penalty = usage * downward_pitch / max(float(angle_scale), 1e-6) * penalty_scale

    _log_mean(env, "Metrics/blower_downward_pitch_rad", downward_pitch)

    return penalty


def blower_box_goal_alignment_reward(
    env: ManagerBasedRLEnv,
    goal_center: Sequence[float],
    airflow_term_name: str = "airflow",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("box"),
) -> torch.Tensor:
    """Reward nozzle-box direction alignment with box-goal direction.

    High reward means nozzle->box is aligned with box->goal.
    This encourages the nozzle to stay behind the box and push it toward the goal.
    """
    box: RigidObject = env.scene[asset_cfg.name]
    airflow_term = env.action_manager.get_term(airflow_term_name)
    nozzle_pos, _ = airflow_term._nozzle_pose_w()

    box_xy = box.data.root_pos_w[:, :2]
    nozzle_xy = nozzle_pos[:, :2]
    goal_xy = _goal_center_xy_w(env, goal_center)

    nozzle_to_box_xy = box_xy - nozzle_xy
    box_to_goal_xy_vec = goal_xy - box_xy

    nozzle_to_box_dir = nozzle_to_box_xy / torch.linalg.norm(
        nozzle_to_box_xy,
        dim=1,
        keepdim=True,
    ).clamp_min(1e-6)

    box_to_goal_dir = box_to_goal_xy_vec / torch.linalg.norm(
        box_to_goal_xy_vec,
        dim=1,
        keepdim=True,
    ).clamp_min(1e-6)

    alignment = torch.sum(nozzle_to_box_dir * box_to_goal_dir, dim=1)
    reward = torch.clamp(alignment, min=0.0)

    return reward


def blower_axis_box_alignment_reward(
    env: ManagerBasedRLEnv,
    airflow_term_name: str = "airflow",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("box"),
) -> torch.Tensor:
    """Reward the actual airflow axis pointing toward the box center in 3D."""
    box: RigidObject = env.scene[asset_cfg.name]
    nozzle_pos_w, _, axis_w = _airflow_nozzle_pose_and_axis_w(env, airflow_term_name)

    nozzle_to_box_vec = box.data.root_pos_w - nozzle_pos_w
    nozzle_to_box_dir = nozzle_to_box_vec / torch.linalg.norm(
        nozzle_to_box_vec,
        dim=1,
        keepdim=True,
    ).clamp_min(1e-6)

    alignment = torch.sum(axis_w * nozzle_to_box_dir, dim=1)
    reward = torch.clamp(alignment, min=0.0)

    return reward


def blower_axis_horizontal_goal_alignment_reward(
    env: ManagerBasedRLEnv,
    goal_center: Sequence[float],
    min_horizontal_fraction: float = 0.55,
    horizontal_std: float = 0.20,
    airflow_term_name: str = "airflow",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("box"),
) -> torch.Tensor:
    """Reward airflow axis with useful horizontal component toward the goal.

    For a box on a table, a near-vertical jet mostly presses the object into the
    table. This term asks the XY projection of the jet to point along box->goal
    and asks the jet to have enough horizontal magnitude.
    """
    _, _, axis_w = _airflow_nozzle_pose_and_axis_w(env, airflow_term_name)

    axis_xy = axis_w[:, :2]
    horizontal_fraction = torch.linalg.norm(axis_xy, dim=1)
    axis_xy_dir = axis_xy / horizontal_fraction.unsqueeze(1).clamp_min(1e-6)

    box_to_goal = box_to_goal_xy(env, goal_center, asset_cfg)
    box_to_goal_dir = box_to_goal / torch.linalg.norm(
        box_to_goal,
        dim=1,
        keepdim=True,
    ).clamp_min(1e-6)

    xy_alignment = torch.clamp(torch.sum(axis_xy_dir * box_to_goal_dir, dim=1), min=0.0)
    horizontal_gate = torch.exp(
        -torch.square(
            torch.clamp(float(min_horizontal_fraction) - horizontal_fraction, min=0.0)
            / max(float(horizontal_std), 1e-6)
        )
    )

    reward = xy_alignment * horizontal_gate

    return reward


def blower_axis_attack_angle_reward(
    env: ManagerBasedRLEnv,
    desired_horizontal_fraction: float = 0.82,
    std: float = 0.16,
    require_downward: bool = True,
    downward_min_fraction: float = 0.05,
    downward_std: float = 0.15,
    airflow_term_name: str = "airflow",
) -> torch.Tensor:
    """Reward a non-vertical jet angle with a strong horizontal component.

    horizontal_fraction = ||axis_xy|| = sin(theta_from_vertical).
    A value near 0 is vertical. A value near 1 is horizontal.

    desired_horizontal_fraction=0.82 means roughly 55 deg from vertical,
    or about 35 deg from the table plane.
    """
    _, _, axis_w = _airflow_nozzle_pose_and_axis_w(env, airflow_term_name)

    horizontal_fraction = torch.linalg.norm(axis_w[:, :2], dim=1).clamp(0.0, 1.0)
    angle_reward = torch.exp(
        -torch.square(
            (horizontal_fraction - float(desired_horizontal_fraction))
            / max(float(std), 1e-6)
        )
    )

    if require_downward:
        downward_fraction = torch.clamp(-axis_w[:, 2], min=0.0, max=1.0)
        downward_gate = torch.exp(
            -torch.square(
                torch.clamp(float(downward_min_fraction) - downward_fraction, min=0.0)
                / max(float(downward_std), 1e-6)
            )
        )
        reward = angle_reward * downward_gate
    else:
        reward = angle_reward

    return reward


def blower_aim_side_contact_reward(
    env: ManagerBasedRLEnv,
    goal_center: Sequence[float],
    box_half_extent: float,
    desired_height: float,
    std: float,
    airflow_term_name: str = "airflow",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("box"),
) -> torch.Tensor:
    """Reward the airflow axis aiming at the rear side of the box.

    The target point is slightly behind the box center relative to box->goal and
    at a configurable height. For a cube on a table, this is a better surrogate
    than aiming at the geometric center because it encourages a horizontal push
    on the upstream side face.
    """
    box: RigidObject = env.scene[asset_cfg.name]
    nozzle_pos_w, _, axis_w = _airflow_nozzle_pose_and_axis_w(env, airflow_term_name)

    box_pos_w = box.data.root_pos_w
    box_to_goal = _goal_center_xy_w(env, goal_center) - box_pos_w[:, :2]
    push_dir_xy = box_to_goal / torch.linalg.norm(
        box_to_goal,
        dim=1,
        keepdim=True,
    ).clamp_min(1e-6)

    target_pos = box_pos_w.clone()
    target_pos[:, :2] = box_pos_w[:, :2] - float(box_half_extent) * push_dir_xy
    target_pos[:, 2] = float(desired_height)

    nozzle_to_target = target_pos - nozzle_pos_w
    nozzle_to_target_dir = nozzle_to_target / torch.linalg.norm(
        nozzle_to_target,
        dim=1,
        keepdim=True,
    ).clamp_min(1e-6)

    alignment = torch.clamp(torch.sum(axis_w * nozzle_to_target_dir, dim=1), min=0.0)
    reward = torch.exp(-torch.square((1.0 - alignment) / max(float(std), 1e-6)))

    return reward


def blower_prepush_position_reward(
    env: ManagerBasedRLEnv,
    goal_center: Sequence[float],
    nozzle_standoff: float,
    nozzle_height: float,
    box_half_extent: float,
    desired_height: float,
    position_std: float,
    height_std: float,
    axis_std: float,
    airflow_term_name: str = "airflow",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("box"),
    ready_geometry: dict[str, float] | None = None,
) -> torch.Tensor:
    """Reward reaching the anti-goal pre-push nozzle pose before applying airflow."""
    terms = _prepush_setup_terms(
        env,
        goal_center=goal_center,
        nozzle_standoff=nozzle_standoff,
        nozzle_height=nozzle_height,
        box_half_extent=box_half_extent,
        desired_height=desired_height,
        position_std=position_std,
        height_std=height_std,
        axis_std=axis_std,
        airflow_term_name=airflow_term_name,
        asset_cfg=asset_cfg,
        ready_geometry=ready_geometry,
    )

    return terms["position_reward"]



def blower_prepush_ready_reward(
    env: ManagerBasedRLEnv,
    goal_center: Sequence[float],
    nozzle_standoff: float,
    nozzle_height: float,
    box_half_extent: float,
    desired_height: float,
    position_std: float,
    height_std: float,
    axis_std: float,
    airflow_term_name: str = "airflow",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("box"),
    ready_geometry: dict[str, float] | None = None,
) -> torch.Tensor:
    """Reward the complete setup: stand behind the box and aim at its upstream side."""
    terms = _prepush_setup_terms(
        env,
        goal_center=goal_center,
        nozzle_standoff=nozzle_standoff,
        nozzle_height=nozzle_height,
        box_half_extent=box_half_extent,
        desired_height=desired_height,
        position_std=position_std,
        height_std=height_std,
        axis_std=axis_std,
        airflow_term_name=airflow_term_name,
        asset_cfg=asset_cfg,
        ready_geometry=ready_geometry,
    )

    return terms["ready"]



def airflow_before_ready_penalty(
    env: ManagerBasedRLEnv,
    goal_center: Sequence[float],
    nozzle_standoff: float,
    nozzle_height: float,
    box_half_extent: float,
    desired_height: float,
    position_std: float,
    height_std: float,
    axis_std: float,
    penalty_warmup_steps: int = 0,
    penalty_ramp_steps: int = 0,
    penalty_start_scale: float = 1.0,
    airflow_term_name: str = "airflow",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("box"),
    ready_geometry: dict[str, float] | None = None,
) -> torch.Tensor:
    """Penalize blowing before the nozzle is in a useful pre-push pose."""
    airflow_term = env.action_manager.get_term(airflow_term_name)
    usage = airflow_term.processed_actions[:, 0]
    terms = _prepush_setup_terms(
        env,
        goal_center=goal_center,
        nozzle_standoff=nozzle_standoff,
        nozzle_height=nozzle_height,
        box_half_extent=box_half_extent,
        desired_height=desired_height,
        position_std=position_std,
        height_std=height_std,
        axis_std=axis_std,
        airflow_term_name=airflow_term_name,
        asset_cfg=asset_cfg,
        ready_geometry=ready_geometry,
    )

    penalty_scale = _global_step_penalty_scale(
        env,
        warmup_steps=penalty_warmup_steps,
        ramp_steps=penalty_ramp_steps,
        start_scale=penalty_start_scale,
    )
    penalty = usage.square() * (1.0 - terms["ready"]) * penalty_scale

    return penalty



def nozzle_forward_position_penalty(
    env: ManagerBasedRLEnv,
    min_forward_x: float = 0.05,
    max_forward_x: float | None = None,
    airflow_term_name: str = "airflow",
) -> torch.Tensor:
    """Penalize the nozzle for being too far back or over-extended forward.

    In the real world the arm body and base block airflow when the nozzle is
    positioned behind the robot. This term penalises nozzle_x < robot_base_x +
    min_forward_x. If max_forward_x is provided, it also penalizes excessive
    forward extension, discouraging long, straight-arm reaching.

    Returns a non-negative penalty value; apply with a negative reward weight.
    """
    nozzle_pos_w, _, _ = _airflow_nozzle_pose_and_axis_w(env, airflow_term_name)
    nozzle_x_local = nozzle_pos_w[:, 0] - env.scene.env_origins[:, 0]
    violation = torch.clamp(float(min_forward_x) - nozzle_x_local, min=0.0)
    if max_forward_x is not None:
        violation = violation + torch.clamp(nozzle_x_local - float(max_forward_x), min=0.0)
    return violation.square()


def box_velocity_toward_goal_reward(
    env: ManagerBasedRLEnv,
    goal_center: Sequence[float],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("box"),
) -> torch.Tensor:
    """Reward positive box velocity along the current box->goal direction."""
    box: RigidObject = env.scene[asset_cfg.name]

    direction = box_to_goal_xy(env, goal_center, asset_cfg)
    direction = direction / torch.linalg.norm(
        direction,
        dim=1,
        keepdim=True,
    ).clamp_min(1e-6)

    vel_xy = box.data.root_lin_vel_w[:, :2]
    vel_toward_goal = torch.sum(vel_xy * direction, dim=1)
    reward = torch.clamp(vel_toward_goal, min=0.0)

    return reward


def box_velocity_toward_goal_with_airflow_reward(
    env: ManagerBasedRLEnv,
    goal_center: Sequence[float],
    nozzle_standoff: float,
    nozzle_height: float,
    box_half_extent: float,
    desired_height: float,
    position_std: float,
    height_std: float,
    axis_std: float,
    min_airflow: float = 0.15,
    ready_reward_floor: float = 0.0,
    force_scale_n: float = 0.0,
    airflow_term_name: str = "airflow",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("box"),
    ready_geometry: dict[str, float] | None = None,
) -> torch.Tensor:
    """Reward useful box velocity when the blower is on.

    `ready_reward_floor` gives partial credit before the full pre-push geometry
    is reached, so difficult curriculum stages do not learn to suppress airflow
    simply because the strict ready gate is hard to satisfy.
    """
    box: RigidObject = env.scene[asset_cfg.name]

    direction = box_to_goal_xy(env, goal_center, asset_cfg)
    direction = direction / torch.linalg.norm(
        direction,
        dim=1,
        keepdim=True,
    ).clamp_min(1e-6)

    vel_xy = box.data.root_lin_vel_w[:, :2]
    vel_toward_goal = torch.clamp(torch.sum(vel_xy * direction, dim=1), min=0.0)

    terms = _prepush_setup_terms(
        env,
        goal_center=goal_center,
        nozzle_standoff=nozzle_standoff,
        nozzle_height=nozzle_height,
        box_half_extent=box_half_extent,
        desired_height=desired_height,
        position_std=position_std,
        height_std=height_std,
        axis_std=axis_std,
        airflow_term_name=airflow_term_name,
        asset_cfg=asset_cfg,
        ready_geometry=ready_geometry,
    )
    usage_gate = _airflow_usage_gate(env, airflow_term_name, min_airflow)
    ready_floor = min(max(float(ready_reward_floor), 0.0), 1.0)
    ready_gate = ready_floor + (1.0 - ready_floor) * terms["ready"]
    reward = vel_toward_goal * ready_gate * usage_gate

    # Optional physical-effect gate: switching the fan on while pushing by hand
    # is insufficient if the jet misses the box or pushes away from the goal.
    # This is a contribution proxy, not a counterfactual separation of motion.
    if force_scale_n > 0.0:
        airflow = env.action_manager.get_term(airflow_term_name)
        force_xy = airflow.target_force_w(box)[:, :2]
        force_toward_goal = torch.sum(force_xy * direction, dim=1).clamp_min(0.0)
        force_gate = (force_toward_goal / force_scale_n).clamp(max=1.0)
        reward = reward * force_gate
        _log_mean(env, "Metrics/airflow_force_toward_goal_n", force_toward_goal)
        _log_mean(env, "Metrics/airflow_force_gate", force_gate)

    _log_mean(env, "Metrics/airflow_velocity_toward_goal", reward)

    return reward



def box_velocity_toward_goal_without_airflow_penalty(
    env: ManagerBasedRLEnv,
    goal_center: Sequence[float],
    min_airflow: float = 0.15,
    velocity_deadband: float = 0.01,
    airflow_term_name: str = "airflow",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("box"),
) -> torch.Tensor:
    """Penalize box progress that happens while airflow is effectively off.

    This makes gripper/body pushes expensive even if they briefly improve the
    task distance.
    """
    box: RigidObject = env.scene[asset_cfg.name]

    direction = box_to_goal_xy(env, goal_center, asset_cfg)
    direction = direction / torch.linalg.norm(
        direction,
        dim=1,
        keepdim=True,
    ).clamp_min(1e-6)

    vel_xy = box.data.root_lin_vel_w[:, :2]
    vel_toward_goal = torch.clamp(
        torch.sum(vel_xy * direction, dim=1) - float(velocity_deadband),
        min=0.0,
    )
    usage_gate = _airflow_usage_gate(env, airflow_term_name, min_airflow)
    penalty = vel_toward_goal * (1.0 - usage_gate)

    _log_mean(env, "Metrics/velocity_without_airflow", penalty)

    return penalty


def robot_box_proximity_penalty(
    env: ManagerBasedRLEnv,
    box_half_extent: float,
    clearance: float,
    robot_cfg: SceneEntityCfg = SceneEntityCfg(
        "robot",
        body_names=["panda_hand", "panda_leftfinger", "panda_rightfinger"],
    ),
    box_cfg: SceneEntityCfg = SceneEntityCfg("box"),
) -> torch.Tensor:
    """Penalize hand/finger bodies entering a small clearance shell around the box."""
    robot: Articulation = env.scene[robot_cfg.name]
    box: RigidObject = env.scene[box_cfg.name]

    body_ids = robot_cfg.body_ids if robot_cfg.body_ids is not None else slice(None)
    body_pos_w = robot.data.body_pos_w[:, body_ids]
    if body_pos_w.dim() == 2:
        body_pos_w = body_pos_w.unsqueeze(1)

    half_extent = torch.full((1, 1, 3), float(box_half_extent), dtype=torch.float32, device=env.device)
    box_pos_w = box.data.root_pos_w.unsqueeze(1)
    outside_box_delta = torch.clamp(torch.abs(body_pos_w - box_pos_w) - half_extent, min=0.0)
    surface_distance = torch.linalg.norm(outside_box_delta, dim=-1)
    closest_distance = torch.min(surface_distance, dim=1).values

    clearance = max(float(clearance), 1e-6)
    penalty = torch.square(torch.clamp((clearance - closest_distance) / clearance, min=0.0, max=1.0))

    _log_mean(env, "Metrics/robot_box_closest_distance", closest_distance)
    _log_mean(env, "Metrics/robot_box_proximity_penalty", penalty)

    return penalty


def box_velocity_toward_goal_when_ready_reward(
    env: ManagerBasedRLEnv,
    goal_center: Sequence[float],
    nozzle_standoff: float,
    nozzle_height: float,
    box_half_extent: float,
    desired_height: float,
    position_std: float,
    height_std: float,
    axis_std: float,
    airflow_term_name: str = "airflow",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("box"),
    ready_geometry: dict[str, float] | None = None,
) -> torch.Tensor:
    """Reward box velocity toward goal only after the nozzle is set up."""
    box: RigidObject = env.scene[asset_cfg.name]

    direction = box_to_goal_xy(env, goal_center, asset_cfg)
    direction = direction / torch.linalg.norm(
        direction,
        dim=1,
        keepdim=True,
    ).clamp_min(1e-6)

    vel_xy = box.data.root_lin_vel_w[:, :2]
    vel_toward_goal = torch.clamp(torch.sum(vel_xy * direction, dim=1), min=0.0)
    terms = _prepush_setup_terms(
        env,
        goal_center=goal_center,
        nozzle_standoff=nozzle_standoff,
        nozzle_height=nozzle_height,
        box_half_extent=box_half_extent,
        desired_height=desired_height,
        position_std=position_std,
        height_std=height_std,
        axis_std=axis_std,
        airflow_term_name=airflow_term_name,
        asset_cfg=asset_cfg,
        ready_geometry=ready_geometry,
    )

    reward = vel_toward_goal * terms["ready"]

    return reward



def box_goal_low_velocity_reward(
    env: ManagerBasedRLEnv,
    goal_center: Sequence[float],
    goal_radius: float,
    std: float = 0.08,
    box_half_extent: float = 0.0,
    required_hold_steps: int = 0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("box"),
) -> torch.Tensor:
    """Reward the box being slow after entering the goal region.

    When ``required_hold_steps`` is positive, this dense term also maintains the
    held-success state used by metrics, curriculum, and PLAY retreat. This keeps
    the engineering success signal alive even when sparse success reward and
    success termination are disabled.
    """
    box: RigidObject = env.scene[asset_cfg.name]

    if required_hold_steps > 0:
        _update_goal_hold_state(
            env,
            goal_center=goal_center,
            goal_radius=goal_radius,
            required_hold_steps=required_hold_steps,
            box_half_extent=box_half_extent,
            asset_cfg=asset_cfg,
        )
        inside = env._box_push_goal_inside.float()
        hold_fraction = env._box_push_goal_hold_steps.float() / max(int(required_hold_steps), 1)
        hold_fraction = torch.clamp(hold_fraction, 0.0, 1.0)
        success = env._box_push_goal_held_success.float()
        episode_success = (
            env._box_push_goal_max_hold_steps >= max(int(required_hold_steps), 1)
        ).float()
        _log_mean(env, "Metrics/goal_hold_fraction", hold_fraction)
        _log_mean(env, "Metrics/success", success)
        _log_mean(env, "Metrics/episode_success", episode_success)
        _log_mean(env, "Metrics/held_success_rate_step", success)
    else:
        inside = _box_fully_inside_goal(
            env,
            asset_cfg=asset_cfg,
            goal_center=goal_center,
            goal_radius=goal_radius,
            box_half_extent=box_half_extent,
        ).float()

    speed = torch.linalg.norm(box.data.root_lin_vel_w[:, :2], dim=1)
    reward = inside * torch.exp(-torch.square(speed / max(float(std), 1e-6)))

    return reward


def box_forward_velocity_reward(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("box"),
) -> torch.Tensor:
    """Reward positive x velocity of the box.

    This term is kept for compatibility. In the current cfg its weight is 0.0
    because the goal is to keep the box inside the target region, not to keep
    pushing it forward forever.
    """
    box: RigidObject = env.scene[asset_cfg.name]
    vx = box.data.root_lin_vel_w[:, 0]

    reward = torch.clamp(vx, min=0.0)

    return reward


def box_out_of_workspace_penalty(
    env: ManagerBasedRLEnv,
    workspace_bounds: Sequence[float],
    z_bounds: Sequence[float] = (-0.02, 0.25),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("box"),
) -> torch.Tensor:
    out_of_workspace = box_out_of_workspace(
        env,
        workspace_bounds,
        z_bounds,
        asset_cfg,
    ).float()

    _log_mean(env, "Metrics/out_of_workspace_rate", out_of_workspace)

    return out_of_workspace
