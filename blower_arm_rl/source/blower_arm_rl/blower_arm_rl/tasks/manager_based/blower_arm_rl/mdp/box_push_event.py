# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reset event terms for the airflow box-push task."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import math as math_utils

from .box_push_common import (
    _activate_pending_goal_radius,
    _as_float_tensor,
    _ensure_box_push_state,
    _goal_center_xy_local,
    _ids_to_tensor,
    _reset_goal_hold_state,
    _spawn_sector_from_goal_delta,
    _update_prev_goal_distance,
)
from .box_push_visual import update_goal_region_visual

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def reset_robot_joints_by_offset(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    position_range: Sequence[float],
    velocity_range: Sequence[float],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Reset robot joints with additive offsets around the default pose.

    The stock reach task uses multiplicative scaling, which leaves joints whose
    default position is close to zero almost unchanged. Additive offsets give the
    joint-policy version of this task more initial nozzle-pose diversity without
    changing the action space.
    """
    asset: Articulation = env.scene[asset_cfg.name]

    env_ids = _ids_to_tensor(env, env_ids)
    if env_ids.numel() == 0:
        return

    if asset_cfg.joint_ids != slice(None):
        iter_env_ids = env_ids[:, None]
    else:
        iter_env_ids = env_ids

    joint_pos = asset.data.default_joint_pos[iter_env_ids, asset_cfg.joint_ids].clone()
    joint_vel = asset.data.default_joint_vel[iter_env_ids, asset_cfg.joint_ids].clone()

    joint_pos += math_utils.sample_uniform(
        float(position_range[0]),
        float(position_range[1]),
        joint_pos.shape,
        joint_pos.device,
    )
    joint_vel += math_utils.sample_uniform(
        float(velocity_range[0]),
        float(velocity_range[1]),
        joint_vel.shape,
        joint_vel.device,
    )

    joint_pos_limits = asset.data.soft_joint_pos_limits[iter_env_ids, asset_cfg.joint_ids]
    joint_pos = joint_pos.clamp_(joint_pos_limits[..., 0], joint_pos_limits[..., 1])

    joint_vel_limits = asset.data.soft_joint_vel_limits[iter_env_ids, asset_cfg.joint_ids]
    joint_vel = joint_vel.clamp_(-joint_vel_limits, joint_vel_limits)

    asset.write_joint_state_to_sim(
        joint_pos,
        joint_vel,
        joint_ids=asset_cfg.joint_ids,
        env_ids=env_ids,
    )


def reset_robot_joints_by_safe_sample(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    joint_position_ranges: Mapping[str, Sequence[float]],
    velocity_range: Sequence[float],
    joint_limit_margin: float = 0.04,
    max_resample_attempts: int = 16,
    enforce_self_collision_safety: bool = True,
    target_joint_positions: Mapping[str, float] | None = None,
    target_joint_half_widths: Mapping[str, float] | float | None = None,
    range_scale_attr: str = "_box_push_robot_reset_range_scale",
    initial_range_scale: float = 1.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Reset robot joints from conservative absolute ranges.

    The ranges are joint-name keyed and are clipped to the articulation soft
    joint limits with a small margin.  A Franka-specific rejection filter keeps
    samples out of obvious folded wrist / tucked elbow self-collision regions.
    This is intentionally conservative and cheap enough to run during reset.
    """
    asset: Articulation = env.scene[asset_cfg.name]

    env_ids = _ids_to_tensor(env, env_ids)
    if env_ids.numel() == 0:
        return

    _initialize_robot_reset_range_scale(env, range_scale_attr, initial_range_scale)

    joint_pos = asset.data.default_joint_pos[env_ids].clone()
    joint_vel = asset.data.default_joint_vel[env_ids].clone()

    pending = torch.ones(env_ids.numel(), dtype=torch.bool, device=env.device)

    num_attempts = max(int(max_resample_attempts), 1)
    for _ in range(num_attempts):
        if not torch.any(pending):
            break

        pending_rows = torch.nonzero(pending, as_tuple=False).squeeze(-1)
        candidate = _sample_joint_positions_from_ranges(
            env,
            asset,
            env_ids[pending_rows],
            joint_position_ranges,
            joint_limit_margin,
            target_joint_positions,
            target_joint_half_widths,
            range_scale_attr,
        )

        if enforce_self_collision_safety:
            safe = _franka_self_collision_safe_mask(asset, candidate)
        else:
            safe = torch.ones(candidate.shape[0], dtype=torch.bool, device=env.device)

        accepted_rows = pending_rows[safe]
        if accepted_rows.numel() > 0:
            joint_pos[accepted_rows] = candidate[safe]

        pending[pending_rows] = ~safe

    if torch.any(pending):
        fallback_rows = torch.nonzero(pending, as_tuple=False).squeeze(-1)
        joint_pos[fallback_rows] = asset.data.default_joint_pos[env_ids[fallback_rows]]

    velocity_scale = _robot_reset_range_scale(env, range_scale_attr)
    joint_vel += math_utils.sample_uniform(
        float(velocity_range[0]) * velocity_scale,
        float(velocity_range[1]) * velocity_scale,
        joint_vel.shape,
        joint_vel.device,
    )

    joint_vel_limits = asset.data.soft_joint_vel_limits[env_ids]
    joint_vel = joint_vel.clamp_(-joint_vel_limits, joint_vel_limits)

    asset.write_joint_state_to_sim(
        joint_pos,
        joint_vel,
        env_ids=env_ids,
    )

    env.extras.setdefault("log", {})["Metrics/robot_reset_fallback_rate"] = pending.float().mean()
    env.extras.setdefault("log", {})["Metrics/robot_reset_range_scale"] = torch.tensor(
        _robot_reset_range_scale(env, range_scale_attr),
        dtype=torch.float32,
        device=env.device,
    )


def _sample_joint_positions_from_ranges(
    env: ManagerBasedRLEnv,
    asset: Articulation,
    env_ids: torch.Tensor,
    joint_position_ranges: Mapping[str, Sequence[float]],
    joint_limit_margin: float,
    target_joint_positions: Mapping[str, float] | None = None,
    target_joint_half_widths: Mapping[str, float] | float | None = None,
    range_scale_attr: str = "_box_push_robot_reset_range_scale",
) -> torch.Tensor:
    joint_pos = asset.data.default_joint_pos[env_ids].clone()
    limit_low, limit_high = _joint_position_limits_with_margin(
        asset.data.soft_joint_pos_limits[env_ids],
        joint_limit_margin,
    )

    for joint_name_expr, joint_range in joint_position_ranges.items():
        joint_ids, _ = asset.find_joints(joint_name_expr)
        if len(joint_ids) == 0:
            raise ValueError(f"No joints matched reset range expression: {joint_name_expr!r}")

        ids = torch.as_tensor(joint_ids, dtype=torch.long, device=env.device)
        user_low_value = float(joint_range[0])
        user_high_value = float(joint_range[1])

        if target_joint_positions is not None and joint_name_expr in target_joint_positions:
            scale_value = _robot_reset_range_scale(env, range_scale_attr)
            target_value = float(target_joint_positions[joint_name_expr])
            half_width = _target_joint_half_width(target_joint_half_widths, joint_name_expr)

            final_low_value = target_value - half_width
            final_high_value = target_value + half_width
            user_low_value = final_low_value + scale_value * (user_low_value - final_low_value)
            user_high_value = final_high_value + scale_value * (user_high_value - final_high_value)

        user_low = torch.full((env_ids.numel(), len(joint_ids)), user_low_value, device=env.device)
        user_high = torch.full((env_ids.numel(), len(joint_ids)), user_high_value, device=env.device)

        low = torch.maximum(user_low, limit_low[:, ids])
        high = torch.minimum(user_high, limit_high[:, ids])

        invalid = high < low
        if torch.any(invalid):
            midpoint = 0.5 * (limit_low[:, ids] + limit_high[:, ids])
            low = torch.where(invalid, midpoint, low)
            high = torch.where(invalid, midpoint, high)

        joint_pos[:, ids] = low + torch.rand_like(low) * (high - low)

    return joint_pos


def _robot_reset_range_scale(env: ManagerBasedRLEnv, range_scale_attr: str) -> float:
    scale = getattr(env, range_scale_attr, 1.0)
    if isinstance(scale, torch.Tensor):
        scale = float(torch.clamp(scale.detach(), 0.0, 1.0).item())
    else:
        scale = max(min(float(scale), 1.0), 0.0)
    return scale


def _initialize_robot_reset_range_scale(
    env: ManagerBasedRLEnv,
    range_scale_attr: str,
    initial_range_scale: float,
) -> None:
    if hasattr(env, range_scale_attr) and int(env.common_step_counter) != 0:
        return

    value = max(min(float(initial_range_scale), 1.0), 0.0)
    setattr(
        env,
        range_scale_attr,
        torch.tensor(value, dtype=torch.float32, device=env.device),
    )


def _target_joint_half_width(
    target_joint_half_widths: Mapping[str, float] | float | None,
    joint_name_expr: str,
) -> float:
    if target_joint_half_widths is None:
        return 0.0

    if isinstance(target_joint_half_widths, Mapping):
        return max(float(target_joint_half_widths.get(joint_name_expr, 0.0)), 0.0)

    return max(float(target_joint_half_widths), 0.0)


def _joint_position_limits_with_margin(
    limits: torch.Tensor,
    margin: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    limit_low = limits[..., 0]
    limit_high = limits[..., 1]

    margin = max(float(margin), 0.0)
    width = limit_high - limit_low
    use_margin = width > 2.0 * margin

    return (
        torch.where(use_margin, limit_low + margin, limit_low),
        torch.where(use_margin, limit_high - margin, limit_high),
    )


def _franka_self_collision_safe_mask(
    asset: Articulation,
    joint_pos: torch.Tensor,
) -> torch.Tensor:
    name_to_id = {name: index for index, name in enumerate(asset.joint_names)}
    required_names = [f"panda_joint{index}" for index in range(1, 8)]
    if not all(name in name_to_id for name in required_names):
        return torch.ones(joint_pos.shape[0], dtype=torch.bool, device=joint_pos.device)

    q1 = joint_pos[:, name_to_id["panda_joint1"]]
    q2 = joint_pos[:, name_to_id["panda_joint2"]]
    q3 = joint_pos[:, name_to_id["panda_joint3"]]
    q4 = joint_pos[:, name_to_id["panda_joint4"]]
    q5 = joint_pos[:, name_to_id["panda_joint5"]]
    q6 = joint_pos[:, name_to_id["panda_joint6"]]
    q7 = joint_pos[:, name_to_id["panda_joint7"]]

    safe = torch.ones(joint_pos.shape[0], dtype=torch.bool, device=joint_pos.device)

    # Keep the reset set in the front/table-side working posture family.
    safe &= torch.abs(q1) <= 1.80
    safe &= (-1.15 <= q2) & (q2 <= 0.35)
    safe &= torch.abs(q3) <= 1.70
    safe &= (-2.80 <= q4) & (q4 <= -0.85)
    safe &= torch.abs(q5) <= 1.70
    safe &= (1.10 <= q6) & (q6 <= 3.20)
    safe &= torch.abs(q7) <= 2.50

    # Conservative folded-arm rejection heuristics for common Franka self contacts.
    safe &= ~((q2 < -0.95) & (q4 < -2.45))
    safe &= ~((q4 < -2.45) & (q6 > 2.90) & (torch.abs(q5) > 1.15))
    safe &= ~((torch.abs(q3) > 1.25) & (torch.abs(q5) > 1.25) & (q6 > 2.80))

    return safe


def reset_box_state_between_robot_and_goal(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    goal_center: Sequence[float],
    goal_radius: float,
    spawn_z: float,
    x_distance_range: Sequence[float],
    y_half_range: float,
    workspace_bounds: Sequence[float],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("box"),
):
    """Reset the box only on the robot side of the goal.

    The simplified layout is:

        robot / nozzle  ->  box  ->  goal

    The intended push direction is +x.

    Sampling rule:

        box_x = goal_x - uniform(x_distance_range[0], x_distance_range[1])
        box_y = goal_y + uniform(-y_half_range, y_half_range)
    """
    _ensure_box_push_state(env, goal_radius)

    env_ids = _ids_to_tensor(env, env_ids)
    if env_ids.numel() == 0:
        return

    box: RigidObject = env.scene[asset_cfg.name]
    root_states = box.data.default_root_state[env_ids].clone()

    center = _goal_center_xy_local(env, goal_center)[env_ids]
    bounds = _as_float_tensor(env, workspace_bounds)

    active_goal_radius = _activate_pending_goal_radius(env, env_ids, goal_radius)
    current_half_extent = float(active_goal_radius[0].item())

    min_dist = float(x_distance_range[0])
    max_dist = float(x_distance_range[1])

    # Do not spawn inside or too close to the current goal square.
    min_dist = max(min_dist, current_half_extent + 0.025)

    if max_dist <= min_dist:
        max_dist = min_dist + 0.02

    num_resets = env_ids.numel()

    samples = torch.empty((num_resets, 2), dtype=torch.float32, device=env.device)

    distance_from_goal = math_utils.sample_uniform(
        min_dist,
        max_dist,
        (num_resets,),
        env.device,
    )

    samples[:, 0] = center[:, 0] - distance_from_goal
    samples[:, 1] = center[:, 1] + math_utils.sample_uniform(
        -float(y_half_range),
        float(y_half_range),
        (num_resets,),
        env.device,
    )

    samples[:, 0].clamp_(float(bounds[0].item()), float(bounds[1].item()))
    samples[:, 1].clamp_(float(bounds[2].item()), float(bounds[3].item()))

    positions = root_states[:, 0:3]
    positions[:, :2] = env.scene.env_origins[env_ids, :2] + samples
    positions[:, 2] = spawn_z

    yaw = math_utils.sample_uniform(
        -torch.pi,
        torch.pi,
        (num_resets,),
        env.device,
    )

    orientations = math_utils.quat_from_euler_xyz(
        torch.zeros_like(yaw),
        torch.zeros_like(yaw),
        yaw,
    )

    velocities = torch.zeros_like(root_states[:, 7:13])

    box.write_root_pose_to_sim(
        torch.cat([positions, orientations], dim=-1),
        env_ids=env_ids,
    )

    box.write_root_velocity_to_sim(
        velocities,
        env_ids=env_ids,
    )

    _update_prev_goal_distance(env, env_ids, asset_cfg, goal_center, goal_radius)
    _reset_goal_hold_state(env, env_ids)

    update_goal_region_visual(env, goal_center, goal_radius=goal_radius, env_ids=env_ids)


def reset_box_state_uniform_with_goal_range(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    goal_center: Sequence[float],
    goal_center_range: Sequence[float] | None,
    goal_radius: float,
    spawn_z: float,
    box_spawn_bounds: Sequence[float],
    workspace_bounds: Sequence[float],
    goal_center_range_start: Sequence[float] | None = None,
    goal_center_range_scale_attr: str = "_box_push_goal_center_range_scale",
    box_clearance: float = 0.015,
    enforce_outside_goal_square: bool = True,
    max_resample_attempts: int = 32,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("box"),
):
    """Reset the goal from a curriculum range and the box from an independent rectangle."""
    _ensure_box_push_state(env, goal_radius)

    env_ids = _ids_to_tensor(env, env_ids)
    if env_ids.numel() == 0:
        return

    box: RigidObject = env.scene[asset_cfg.name]
    root_states = box.data.default_root_state[env_ids].clone()

    bounds = _as_float_tensor(env, workspace_bounds)
    active_goal_radius = _activate_pending_goal_radius(env, env_ids, goal_radius)
    current_half_extent = float(active_goal_radius[0].item())

    if goal_center_range is not None:
        if not hasattr(env, "_box_push_goal_center_xy"):
            env._box_push_goal_center_xy = _goal_center_xy_local(env, goal_center).clone()

        requested = _as_float_tensor(env, goal_center_range)
        if goal_center_range_start is not None:
            start_range = _as_float_tensor(env, goal_center_range_start)
            scale = getattr(env, goal_center_range_scale_attr, torch.tensor(1.0, device=env.device))
            scale = torch.clamp(scale, 0.0, 1.0)
            requested = start_range + scale * (requested - start_range)

        margin = current_half_extent + float(box_clearance)
        x_min = max(float(requested[0].item()), float(bounds[0].item()) + margin)
        x_max = min(float(requested[1].item()), float(bounds[1].item()) - margin)
        y_min = max(float(requested[2].item()), float(bounds[2].item()) + margin)
        y_max = min(float(requested[3].item()), float(bounds[3].item()) - margin)

        fallback_center = _as_float_tensor(env, goal_center[:2])
        if x_max <= x_min:
            x_min = x_max = float(fallback_center[0].item())
        if y_max <= y_min:
            y_min = y_max = float(fallback_center[1].item())

        sampled_goal = torch.empty((env_ids.numel(), 2), dtype=torch.float32, device=env.device)
        sampled_goal[:, 0] = math_utils.sample_uniform(x_min, x_max, (env_ids.numel(),), env.device)
        sampled_goal[:, 1] = math_utils.sample_uniform(y_min, y_max, (env_ids.numel(),), env.device)
        env._box_push_goal_center_xy[env_ids] = sampled_goal

    center = _goal_center_xy_local(env, goal_center)[env_ids]
    spawn = _as_float_tensor(env, box_spawn_bounds)
    margin = float(box_clearance)

    x_min = max(float(spawn[0].item()), float(bounds[0].item()) + margin)
    x_max = min(float(spawn[1].item()), float(bounds[1].item()) - margin)
    y_min = max(float(spawn[2].item()), float(bounds[2].item()) + margin)
    y_max = min(float(spawn[3].item()), float(bounds[3].item()) - margin)

    if x_max <= x_min:
        x_max = x_min + 0.02
    if y_max <= y_min:
        y_max = y_min + 0.02

    num_resets = env_ids.numel()
    samples = torch.empty((num_resets, 2), dtype=torch.float32, device=env.device)
    active = torch.ones(num_resets, dtype=torch.bool, device=env.device)

    attempts = max(int(max_resample_attempts), 1)
    for _ in range(attempts):
        active_count = int(active.sum().item())
        if active_count == 0:
            break

        samples[active, 0] = math_utils.sample_uniform(x_min, x_max, (active_count,), env.device)
        samples[active, 1] = math_utils.sample_uniform(y_min, y_max, (active_count,), env.device)

        if enforce_outside_goal_square:
            delta = torch.abs(samples - center)
            active = torch.all(delta <= current_half_extent + float(box_clearance), dim=1)
        else:
            active = torch.zeros(num_resets, dtype=torch.bool, device=env.device)

    if enforce_outside_goal_square and torch.any(active):
        delta = samples[active] - center[active]
        signs = torch.where(delta >= 0.0, 1.0, -1.0)
        abs_delta = torch.abs(delta)
        axis = torch.argmax(abs_delta, dim=1)
        rows = torch.arange(delta.shape[0], device=env.device)
        projected = delta.clone()
        projected[rows, axis] = signs[rows, axis] * (current_half_extent + float(box_clearance))
        samples[active] = center[active] + projected
        samples[:, 0].clamp_(x_min, x_max)
        samples[:, 1].clamp_(y_min, y_max)

    positions = root_states[:, 0:3]
    positions[:, :2] = env.scene.env_origins[env_ids, :2] + samples
    positions[:, 2] = spawn_z

    yaw = math_utils.sample_uniform(-torch.pi, torch.pi, (num_resets,), env.device)
    orientations = math_utils.quat_from_euler_xyz(
        torch.zeros_like(yaw),
        torch.zeros_like(yaw),
        yaw,
    )
    velocities = torch.zeros_like(root_states[:, 7:13])

    box.write_root_pose_to_sim(
        torch.cat([positions, orientations], dim=-1),
        env_ids=env_ids,
    )
    box.write_root_velocity_to_sim(velocities, env_ids=env_ids)

    _update_prev_goal_distance(env, env_ids, asset_cfg, goal_center, goal_radius)
    _reset_goal_hold_state(env, env_ids)
    update_goal_region_visual(env, goal_center, goal_radius=goal_radius, env_ids=env_ids)


def reset_box_state_around_goal(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    goal_center: Sequence[float],
    goal_center_range: Sequence[float] | None,
    goal_radius: float,
    spawn_z: float,
    radial_distance_range: Sequence[float],
    workspace_bounds: Sequence[float],
    angle_center: float = 3.141592653589793,
    angle_span_start: float | None = None,
    angle_span_final: float | None = None,
    box_clearance: float = 0.015,
    enforce_outside_goal_square: bool = True,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("box"),
):
    """Reset the box around the goal region.

    The goal center is sampled first, then the box is sampled in an
    annulus-like region around that same goal center.

    When angle spans are provided, the reset distribution starts near
    angle_center and expands according to env._box_push_initial_range_scale.
    With angle_center=pi this starts on the robot side of the goal and can grow
    to the full circle while preserving the end-to-end joint action policy.

    The square goal exclusion uses rejection plus a final projection so reset
    states cannot begin as accidental successes near the square diagonals.
    """
    _ensure_box_push_state(env, goal_radius)

    env_ids = _ids_to_tensor(env, env_ids)
    if env_ids.numel() == 0:
        return

    box: RigidObject = env.scene[asset_cfg.name]
    root_states = box.data.default_root_state[env_ids].clone()

    bounds = _as_float_tensor(env, workspace_bounds)

    active_goal_radius = _activate_pending_goal_radius(env, env_ids, goal_radius)
    current_half_extent = float(active_goal_radius[0].item())

    min_dist = float(radial_distance_range[0])
    max_dist = float(radial_distance_range[1])

    if goal_center_range is not None:
        if not hasattr(env, "_box_push_goal_center_xy"):
            env._box_push_goal_center_xy = _goal_center_xy_local(env, goal_center).clone()

        requested = _as_float_tensor(env, goal_center_range)
        margin = current_half_extent + float(box_clearance)

        x_min = max(float(requested[0].item()), float(bounds[0].item()) + margin)
        x_max = min(float(requested[1].item()), float(bounds[1].item()) - margin)
        y_min = max(float(requested[2].item()), float(bounds[2].item()) + margin)
        y_max = min(float(requested[3].item()), float(bounds[3].item()) - margin)

        fallback_center = _as_float_tensor(env, goal_center[:2])
        if x_max <= x_min:
            x_min = x_max = float(fallback_center[0].item())
        if y_max <= y_min:
            y_min = y_max = float(fallback_center[1].item())

        sampled_goal = torch.empty((env_ids.numel(), 2), dtype=torch.float32, device=env.device)
        sampled_goal[:, 0] = math_utils.sample_uniform(x_min, x_max, (env_ids.numel(),), env.device)
        sampled_goal[:, 1] = math_utils.sample_uniform(y_min, y_max, (env_ids.numel(),), env.device)
        env._box_push_goal_center_xy[env_ids] = sampled_goal

    center = _goal_center_xy_local(env, goal_center)[env_ids]

    # Keep the box outside the goal square at reset.
    min_dist = max(min_dist, current_half_extent + float(box_clearance))

    if max_dist <= min_dist:
        max_dist = min_dist + 0.02

    num_resets = env_ids.numel()

    samples = torch.empty((num_resets, 2), dtype=torch.float32, device=env.device)

    if angle_span_start is None or angle_span_final is None:
        angle_span = float(torch.pi)
    else:
        scale = float(torch.clamp(env._box_push_initial_range_scale, 0.0, 1.0).item())
        angle_span = float(angle_span_start) + scale * (float(angle_span_final) - float(angle_span_start))
        angle_span = min(max(angle_span, 0.0), float(torch.pi))

    def _sample(active: torch.Tensor) -> None:
        active_count = int(active.sum().item())
        if active_count == 0:
            return

        if angle_span >= float(torch.pi) - 1e-5:
            angles = math_utils.sample_uniform(
                -torch.pi,
                torch.pi,
                (active_count,),
                env.device,
            )
        else:
            angles = float(angle_center) + math_utils.sample_uniform(
                -angle_span,
                angle_span,
                (active_count,),
                env.device,
            )

        distances = math_utils.sample_uniform(
            min_dist,
            max_dist,
            (active_count,),
            env.device,
        )

        samples[active, 0] = center[active, 0] + distances * torch.cos(angles)
        samples[active, 1] = center[active, 1] + distances * torch.sin(angles)

    active = torch.ones(num_resets, dtype=torch.bool, device=env.device)
    _sample(active)

    for _ in range(32):
        delta = samples - center
        inside_goal = torch.all(
            torch.abs(delta) <= current_half_extent + float(box_clearance),
            dim=1,
        )
        outside_workspace = (
            (samples[:, 0] < bounds[0])
            | (samples[:, 0] > bounds[1])
            | (samples[:, 1] < bounds[2])
            | (samples[:, 1] > bounds[3])
        )
        if enforce_outside_goal_square:
            active = outside_workspace | inside_goal
        else:
            active = outside_workspace
        if not torch.any(active):
            break
        _sample(active)

    samples[:, 0].clamp_(float(bounds[0].item()), float(bounds[1].item()))
    samples[:, 1].clamp_(float(bounds[2].item()), float(bounds[3].item()))

    if enforce_outside_goal_square:
        delta = samples - center
        inside_goal = torch.all(
            torch.abs(delta) <= current_half_extent + float(box_clearance),
            dim=1,
        )
        if torch.any(inside_goal):
            safe_delta = delta[inside_goal]
            near_zero = torch.linalg.norm(safe_delta, dim=1) < 1e-6
            if torch.any(near_zero):
                safe_delta[near_zero] = torch.tensor(
                    [-1.0, 0.0],
                    dtype=torch.float32,
                    device=env.device,
                )
            scale = (
                current_half_extent
                + float(box_clearance)
                + 0.005
            ) / torch.max(torch.abs(safe_delta), dim=1).values.clamp_min(1e-6)
            samples[inside_goal] = center[inside_goal] + safe_delta * scale.unsqueeze(1)

    spawn_delta = samples - center
    env._box_push_spawn_angle[env_ids] = torch.atan2(spawn_delta[:, 1], spawn_delta[:, 0])
    env._box_push_spawn_sector[env_ids] = _spawn_sector_from_goal_delta(spawn_delta)

    positions = root_states[:, 0:3]
    positions[:, :2] = env.scene.env_origins[env_ids, :2] + samples
    positions[:, 2] = spawn_z

    yaw = math_utils.sample_uniform(
        -torch.pi,
        torch.pi,
        (num_resets,),
        env.device,
    )

    orientations = math_utils.quat_from_euler_xyz(
        torch.zeros_like(yaw),
        torch.zeros_like(yaw),
        yaw,
    )

    velocities = torch.zeros_like(root_states[:, 7:13])

    box.write_root_pose_to_sim(
        torch.cat([positions, orientations], dim=-1),
        env_ids=env_ids,
    )

    box.write_root_velocity_to_sim(
        velocities,
        env_ids=env_ids,
    )

    _update_prev_goal_distance(env, env_ids, asset_cfg, goal_center, goal_radius)
    _reset_goal_hold_state(env, env_ids)

    update_goal_region_visual(env, goal_center, goal_radius=goal_radius, env_ids=env_ids)


def reset_box_state_curriculum(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    goal_center: Sequence[float],
    goal_radius: float,
    spawn_z: float,
    near_half_range: Sequence[float],
    final_half_range: Sequence[float],
    workspace_bounds: Sequence[float],
    min_spawn_goal_distance: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("box"),
):
    """Original curriculum reset.

    This is kept for compatibility, but the simplified cfg does not use it.
    """
    _ensure_box_push_state(env, goal_radius)

    env_ids = _ids_to_tensor(env, env_ids)
    if env_ids.numel() == 0:
        return

    box: RigidObject = env.scene[asset_cfg.name]
    root_states = box.data.default_root_state[env_ids].clone()

    curriculum_scale = float(torch.clamp(env._box_push_initial_range_scale, 0.0, 1.0).item())

    near = _as_float_tensor(env, near_half_range)
    final = _as_float_tensor(env, final_half_range)
    half_range = near + curriculum_scale * (final - near)

    center = _as_float_tensor(env, goal_center[:2])
    bounds = _as_float_tensor(env, workspace_bounds)

    samples = torch.empty((env_ids.numel(), 2), dtype=torch.float32, device=env.device)

    active_goal_radius = _activate_pending_goal_radius(env, env_ids, goal_radius)
    current_half_extent = float(active_goal_radius[0].item())
    min_distance = max(float(min_spawn_goal_distance), current_half_extent * 1.05)

    active = torch.ones(env_ids.numel(), dtype=torch.bool, device=env.device)

    for _ in range(32):
        if not torch.any(active):
            break

        active_count = int(active.sum().item())

        samples[active, 0] = center[0] + math_utils.sample_uniform(
            -half_range[0],
            half_range[0],
            (active_count,),
            env.device,
        )

        samples[active, 1] = center[1] + math_utils.sample_uniform(
            -half_range[1],
            half_range[1],
            (active_count,),
            env.device,
        )

        samples[:, 0].clamp_(float(bounds[0].item()), float(bounds[1].item()))
        samples[:, 1].clamp_(float(bounds[2].item()), float(bounds[3].item()))

        sample_delta = torch.abs(samples - center.unsqueeze(0))
        inside_square = torch.all(sample_delta <= current_half_extent + 0.025, dim=1)

        active = inside_square | (
            torch.linalg.norm(samples - center.unsqueeze(0), dim=1) < min_distance
        )

    positions = root_states[:, 0:3]
    positions[:, :2] = env.scene.env_origins[env_ids, :2] + samples
    positions[:, 2] = spawn_z

    yaw = math_utils.sample_uniform(
        -torch.pi,
        torch.pi,
        (env_ids.numel(),),
        env.device,
    )

    orientations = math_utils.quat_from_euler_xyz(
        torch.zeros_like(yaw),
        torch.zeros_like(yaw),
        yaw,
    )

    velocities = torch.zeros_like(root_states[:, 7:13])

    box.write_root_pose_to_sim(
        torch.cat([positions, orientations], dim=-1),
        env_ids=env_ids,
    )

    box.write_root_velocity_to_sim(
        velocities,
        env_ids=env_ids,
    )

    _update_prev_goal_distance(env, env_ids, asset_cfg, goal_center, goal_radius)
    _reset_goal_hold_state(env, env_ids)

    update_goal_region_visual(env, goal_center, goal_radius=goal_radius, env_ids=env_ids)
