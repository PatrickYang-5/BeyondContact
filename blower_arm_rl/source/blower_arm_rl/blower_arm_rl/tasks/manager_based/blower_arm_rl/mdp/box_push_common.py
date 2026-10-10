# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Shared helpers and state for the airflow box-push task."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import math as math_utils

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _ids_to_tensor(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int] | slice | torch.Tensor | None,
) -> torch.Tensor:
    if env_ids is None or isinstance(env_ids, slice):
        return torch.arange(env.num_envs, dtype=torch.long, device=env.device)

    return torch.as_tensor(env_ids, dtype=torch.long, device=env.device)


def _as_float_tensor(env: ManagerBasedRLEnv, values: Sequence[float]) -> torch.Tensor:
    return torch.tensor(values, dtype=torch.float32, device=env.device)


def _ensure_box_push_state(
    env: ManagerBasedRLEnv,
    goal_radius: float,
    initial_range_scale: float = 0.0,
) -> None:
    if not hasattr(env, "_box_push_goal_radius"):
        env._box_push_goal_radius = torch.full(
            (env.num_envs,),
            float(goal_radius),
            dtype=torch.float32,
            device=env.device,
        )

    if not hasattr(env, "_box_push_pending_goal_radius"):
        env._box_push_pending_goal_radius = torch.tensor(
            float(goal_radius),
            dtype=torch.float32,
            device=env.device,
        )

    if not hasattr(env, "_box_push_prev_goal_distance"):
        env._box_push_prev_goal_distance = torch.zeros(
            env.num_envs,
            dtype=torch.float32,
            device=env.device,
        )

    if not hasattr(env, "_box_push_success_ema"):
        env._box_push_success_ema = torch.tensor(
            0.0,
            dtype=torch.float32,
            device=env.device,
        )

    if not hasattr(env, "_box_push_initial_range_scale"):
        env._box_push_initial_range_scale = torch.tensor(
            float(initial_range_scale),
            dtype=torch.float32,
            device=env.device,
        )

    if not hasattr(env, "_box_push_last_curriculum_update_step"):
        env._box_push_last_curriculum_update_step = -1

    if not hasattr(env, "_box_push_robot_reset_range_scale"):
        env._box_push_robot_reset_range_scale = torch.tensor(
            1.0,
            dtype=torch.float32,
            device=env.device,
        )

    if not hasattr(env, "_box_push_last_robot_reset_curriculum_update_step"):
        env._box_push_last_robot_reset_curriculum_update_step = -1

    if not hasattr(env, "_box_push_box_spawn_distance_scale"):
        env._box_push_box_spawn_distance_scale = torch.tensor(
            0.0,
            dtype=torch.float32,
            device=env.device,
        )

    if not hasattr(env, "_box_push_last_box_spawn_distance_curriculum_update_step"):
        env._box_push_last_box_spawn_distance_curriculum_update_step = -1

    if not hasattr(env, "_box_push_goal_center_range_scale"):
        env._box_push_goal_center_range_scale = torch.tensor(
            0.0,
            dtype=torch.float32,
            device=env.device,
        )

    if not hasattr(env, "_box_push_last_goal_center_range_curriculum_update_step"):
        env._box_push_last_goal_center_range_curriculum_update_step = -1

    if not hasattr(env, "_box_push_goal_hold_steps"):
        env._box_push_goal_hold_steps = torch.zeros(
            env.num_envs,
            dtype=torch.long,
            device=env.device,
        )

    if not hasattr(env, "_box_push_goal_inside"):
        env._box_push_goal_inside = torch.zeros(
            env.num_envs,
            dtype=torch.bool,
            device=env.device,
        )

    if not hasattr(env, "_box_push_goal_spatial_inside"):
        env._box_push_goal_spatial_inside = torch.zeros(
            env.num_envs,
            dtype=torch.bool,
            device=env.device,
        )

    if not hasattr(env, "_box_push_goal_was_spatial_inside"):
        env._box_push_goal_was_spatial_inside = torch.zeros(
            env.num_envs,
            dtype=torch.bool,
            device=env.device,
        )

    if not hasattr(env, "_box_push_goal_left_after_entry"):
        env._box_push_goal_left_after_entry = torch.zeros(
            env.num_envs,
            dtype=torch.bool,
            device=env.device,
        )

    if not hasattr(env, "_box_push_goal_max_hold_steps"):
        env._box_push_goal_max_hold_steps = torch.zeros(
            env.num_envs,
            dtype=torch.long,
            device=env.device,
        )

    if not hasattr(env, "_box_push_goal_held_success"):
        env._box_push_goal_held_success = torch.zeros(
            env.num_envs,
            dtype=torch.bool,
            device=env.device,
        )

    if not hasattr(env, "_box_push_goal_hold_last_step"):
        env._box_push_goal_hold_last_step = -1

    if not hasattr(env, "_box_push_spawn_angle"):
        env._box_push_spawn_angle = torch.zeros(
            env.num_envs,
            dtype=torch.float32,
            device=env.device,
        )

    if not hasattr(env, "_box_push_spawn_sector"):
        env._box_push_spawn_sector = torch.zeros(
            env.num_envs,
            dtype=torch.long,
            device=env.device,
        )


def _goal_center_xy_local(env: ManagerBasedRLEnv, goal_center: Sequence[float]) -> torch.Tensor:
    if hasattr(env, "_box_push_goal_center_xy"):
        return env._box_push_goal_center_xy

    center = _as_float_tensor(env, goal_center[:2]).unsqueeze(0)
    return center.repeat(env.num_envs, 1)


def _goal_center_xy_w(env: ManagerBasedRLEnv, goal_center: Sequence[float]) -> torch.Tensor:
    return env.scene.env_origins[:, :2] + _goal_center_xy_local(env, goal_center)


def _goal_radius(env: ManagerBasedRLEnv, goal_radius: float) -> torch.Tensor:
    _ensure_box_push_state(env, goal_radius)

    radius = env._box_push_goal_radius

    if not isinstance(radius, torch.Tensor):
        radius = torch.full(
            (env.num_envs,),
            float(radius),
            dtype=torch.float32,
            device=env.device,
        )
        env._box_push_goal_radius = radius

    return radius


def _pending_goal_radius(env: ManagerBasedRLEnv, goal_radius: float) -> torch.Tensor:
    _ensure_box_push_state(env, goal_radius)

    radius = env._box_push_pending_goal_radius

    if not isinstance(radius, torch.Tensor):
        radius = torch.tensor(
            float(radius),
            dtype=torch.float32,
            device=env.device,
        )
        env._box_push_pending_goal_radius = radius

    return radius


def _activate_pending_goal_radius(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int] | slice | torch.Tensor | None,
    goal_radius: float,
) -> torch.Tensor:
    """Apply the curriculum goal size only to envs that are resetting now."""
    _ensure_box_push_state(env, goal_radius)

    env_ids = _ids_to_tensor(env, env_ids)
    pending_radius = _pending_goal_radius(env, goal_radius)
    env._box_push_goal_radius[env_ids] = pending_radius

    return env._box_push_goal_radius[env_ids]


def _log_mean(env: ManagerBasedRLEnv, name: str, value: torch.Tensor) -> None:
    env.extras.setdefault("log", {})[name] = value.detach().mean()


def _log_sector_metrics(env: ManagerBasedRLEnv, prefix: str, value: torch.Tensor) -> None:
    """Log a scalar metric by spawn sector.

    Sectors are measured from goal to box at reset:
        0: robot-side / negative x
        1: far-side / positive x
        2: positive y side
        3: negative y side
    """
    if not hasattr(env, "_box_push_spawn_sector"):
        return

    sector_names = ("robot_side", "far_side", "y_pos", "y_neg")
    value = value.detach()
    for sector_id, sector_name in enumerate(sector_names):
        mask = env._box_push_spawn_sector == sector_id
        if torch.any(mask):
            env.extras.setdefault("log", {})[f"{prefix}/{sector_name}"] = value[mask].mean()


def _spawn_sector_from_goal_delta(delta_xy: torch.Tensor) -> torch.Tensor:
    abs_delta = torch.abs(delta_xy)
    x_dominant = abs_delta[:, 0] >= abs_delta[:, 1]

    sector = torch.zeros(delta_xy.shape[0], dtype=torch.long, device=delta_xy.device)
    sector[x_dominant & (delta_xy[:, 0] >= 0.0)] = 1
    sector[(~x_dominant) & (delta_xy[:, 1] >= 0.0)] = 2
    sector[(~x_dominant) & (delta_xy[:, 1] < 0.0)] = 3
    return sector


def _box_xy_w(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    box: RigidObject = env.scene[asset_cfg.name]
    return box.data.root_pos_w[:, :2]


def _desired_nozzle_position_w(
    env: ManagerBasedRLEnv,
    goal_center: Sequence[float],
    nozzle_standoff: float,
    desired_height: float,
    asset_cfg: SceneEntityCfg,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the desired upstream nozzle position and box-to-goal direction."""
    box: RigidObject = env.scene[asset_cfg.name]
    box_to_goal = _goal_center_xy_w(env, goal_center) - box.data.root_pos_w[:, :2]
    push_dir_xy = box_to_goal / torch.linalg.norm(box_to_goal, dim=1, keepdim=True).clamp_min(1e-6)

    desired_nozzle_pos = box.data.root_pos_w.clone()
    desired_nozzle_pos[:, :2] -= float(nozzle_standoff) * push_dir_xy
    desired_nozzle_pos[:, 2] = env.scene.env_origins[:, 2] + float(desired_height)
    return desired_nozzle_pos, push_dir_xy


def _box_fully_inside_goal(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    goal_center: Sequence[float],
    goal_radius: float,
    box_half_extent: float | Sequence[float],
) -> torch.Tensor:
    """Return true when the box's projected corners are all inside the goal square."""
    _ensure_box_push_state(env, goal_radius)

    box: RigidObject = env.scene[asset_cfg.name]
    largest_half_extent = (box_half_extent if isinstance(box_half_extent, (int, float))
                           else max(box_half_extent))
    if largest_half_extent <= 0.0:
        goal_half_extent = _goal_radius(env, goal_radius).unsqueeze(1)
        delta = torch.abs(_box_xy_w(env, asset_cfg) - _goal_center_xy_w(env, goal_center))
        return torch.all(delta <= goal_half_extent, dim=1)

    half_extent = torch.as_tensor(box_half_extent, dtype=torch.float32, device=env.device).clamp_min(0.0)
    signs = torch.tensor(
        [
            [-1.0, -1.0, -1.0],
            [-1.0, -1.0, 1.0],
            [-1.0, 1.0, -1.0],
            [-1.0, 1.0, 1.0],
            [1.0, -1.0, -1.0],
            [1.0, -1.0, 1.0],
            [1.0, 1.0, -1.0],
            [1.0, 1.0, 1.0],
        ],
        dtype=torch.float32,
        device=env.device,
    )
    local_corners = signs * half_extent
    num_corners = local_corners.shape[0]

    quat_w = box.data.root_quat_w[:, None, :].expand(-1, num_corners, -1).reshape(-1, 4)
    corner_offsets = local_corners.unsqueeze(0).expand(env.num_envs, -1, -1).reshape(-1, 3)
    corner_offsets_w = math_utils.quat_apply(quat_w, corner_offsets).reshape(env.num_envs, num_corners, 3)
    corners_xy_w = box.data.root_pos_w[:, None, :2] + corner_offsets_w[:, :, :2]

    goal_half_extent = _goal_radius(env, goal_radius).view(env.num_envs, 1, 1)
    goal_xy_w = _goal_center_xy_w(env, goal_center).unsqueeze(1)
    corner_delta = torch.abs(corners_xy_w - goal_xy_w)

    return torch.all(torch.all(corner_delta <= goal_half_extent, dim=2), dim=1)


def _distance_direction_aim_terms(
    nozzle_pos_w: torch.Tensor,
    axis_w: torch.Tensor,
    box_pos_w: torch.Tensor,
    push_dir_xy: torch.Tensor,
    distance: float,
    distance_std: float,
    direction_std_deg: float,
    aim_std_deg: float,
) -> dict[str, torch.Tensor]:
    """Independent range, goalward jet direction and box-center aiming scores."""
    if distance <= 0 or min(distance_std, direction_std_deg, aim_std_deg) <= 0:
        raise ValueError("Ready distance and score scales must be positive")
    delta = box_pos_w - nozzle_pos_w
    actual_distance = torch.linalg.vector_norm(delta, dim=1)
    axis_norm = torch.linalg.vector_norm(axis_w, dim=1, keepdim=True)
    axis = axis_w / axis_norm.clamp_min(1e-6)
    box_dir = delta / actual_distance.unsqueeze(1).clamp_min(1e-6)
    goal_norm = torch.linalg.vector_norm(push_dir_xy, dim=1, keepdim=True)
    goal_dir = push_dir_xy / goal_norm.clamp_min(1e-6)

    # Dot with the horizontal goal vector in 3D: vertical jets must not receive
    # a perfect direction score merely because their tiny XY projection aligns.
    direction_cos = (axis[:, :2] * goal_dir).sum(dim=1).clamp(-1.0, 1.0)
    aim_cos = (axis * box_dir).sum(dim=1).clamp(-1.0, 1.0)
    direction_angle = torch.acos(direction_cos)
    aim_angle = torch.acos(aim_cos)
    radians_per_degree = torch.pi / 180.0
    position_score = torch.exp(-((actual_distance - distance) / distance_std).square())
    direction_score = torch.exp(-(direction_angle / (direction_std_deg * radians_per_degree)).square())
    aim_score = torch.exp(-(aim_angle / (aim_std_deg * radians_per_degree)).square())
    direction_score *= (direction_cos > 0) & (goal_norm[:, 0] > 1e-6) & (axis_norm[:, 0] > 1e-6)
    aim_score *= (aim_cos > 0) & (actual_distance > 1e-6) & (axis_norm[:, 0] > 1e-6)
    return {
        "position_score": position_score,
        "direction_score": direction_score,
        "aim_score": aim_score,
        "distance": actual_distance,
        "direction_error_deg": direction_angle / radians_per_degree,
        "aim_error_deg": aim_angle / radians_per_degree,
        "ready": position_score * direction_score * aim_score,
    }


def _prepush_setup_terms(
    env: ManagerBasedRLEnv,
    goal_center: Sequence[float],
    nozzle_standoff: float,
    nozzle_height: float,
    box_half_extent: float,
    desired_height: float,
    position_std: float,
    height_std: float,
    axis_std: float,
    airflow_term_name: str,
    asset_cfg: SceneEntityCfg,
    ready_geometry: dict[str, float] | None = None,
) -> dict[str, torch.Tensor]:
    """Compute reusable pre-push position and aiming features."""
    box: RigidObject = env.scene[asset_cfg.name]
    nozzle_pos_w, _, axis_w = _airflow_nozzle_pose_and_axis_w(env, airflow_term_name)

    box_pos_w = box.data.root_pos_w
    desired_nozzle_pos, push_dir_xy = _desired_nozzle_position_w(
        env,
        goal_center=goal_center,
        nozzle_standoff=nozzle_standoff,
        desired_height=nozzle_height,
        asset_cfg=asset_cfg,
    )

    target_pos = box_pos_w.clone()
    target_pos[:, :2] = box_pos_w[:, :2] - float(box_half_extent) * push_dir_xy
    target_pos[:, 2] = env.scene.env_origins[:, 2] + float(desired_height)

    nozzle_to_target = target_pos - nozzle_pos_w
    nozzle_to_target_dir = nozzle_to_target / torch.linalg.norm(
        nozzle_to_target,
        dim=1,
        keepdim=True,
    ).clamp_min(1e-6)

    nozzle_to_box_xy = box_pos_w[:, :2] - nozzle_pos_w[:, :2]
    nozzle_to_box_dir = nozzle_to_box_xy / torch.linalg.norm(
        nozzle_to_box_xy,
        dim=1,
        keepdim=True,
    ).clamp_min(1e-6)

    desired_delta = desired_nozzle_pos - nozzle_pos_w
    position_error_xy = torch.linalg.norm(desired_delta[:, :2], dim=1)
    height_error = torch.abs(desired_delta[:, 2])

    position_reward = torch.exp(
        -torch.square(position_error_xy / max(float(position_std), 1e-6))
        -torch.square(height_error / max(float(height_std), 1e-6))
    )

    stand_alignment = torch.clamp(torch.sum(nozzle_to_box_dir * push_dir_xy, dim=1), min=0.0)
    axis_alignment = torch.clamp(torch.sum(axis_w * nozzle_to_target_dir, dim=1), min=0.0)
    axis_reward = torch.exp(-torch.square((1.0 - axis_alignment) / max(float(axis_std), 1e-6)))

    ready = position_reward * stand_alignment * axis_reward
    direction_score = stand_alignment
    aim_score = axis_reward
    if ready_geometry is not None:
        geometry = _distance_direction_aim_terms(
            nozzle_pos_w, axis_w, box_pos_w, push_dir_xy, **ready_geometry,
        )
        position_reward = geometry["position_score"]
        direction_score = geometry["direction_score"]
        aim_score = geometry["aim_score"]
        ready = geometry["ready"]
        for key in ("position_score", "direction_score", "aim_score", "ready",
                    "distance", "direction_error_deg", "aim_error_deg"):
            log_key = "Metrics/airflow_ready" if key == "ready" else f"Metrics/airflow_ready_{key}"
            _log_mean(env, log_key, geometry[key])

    return {
        "desired_nozzle_pos": desired_nozzle_pos,
        "desired_delta": desired_delta,
        "push_dir_xy": push_dir_xy,
        "position_error_xy": position_error_xy,
        "height_error": height_error,
        "position_reward": position_reward,
        "stand_alignment": stand_alignment,
        "axis_alignment": axis_alignment,
        "axis_reward": axis_reward,
        "direction_score": direction_score,
        "aim_score": aim_score,
        "ready": ready,
    }


def _box_goal_distance(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    goal_center: Sequence[float],
    goal_radius: float,
    box_half_extent: float | Sequence[float] = 0.0,
) -> torch.Tensor:
    _ensure_box_push_state(env, goal_radius)

    half_extent = _goal_radius(env, goal_radius).unsqueeze(1)
    delta = torch.abs(_box_xy_w(env, asset_cfg) - _goal_center_xy_w(env, goal_center)) - half_extent
    # Optional physical footprint: old tasks keep their center-distance default.
    if not isinstance(box_half_extent, (int, float)) or box_half_extent > 0:
        body_half = torch.as_tensor(box_half_extent, dtype=torch.float32, device=env.device).expand(3).clamp_min(0)
        extent_w = math_utils.matrix_from_quat(env.scene[asset_cfg.name].data.root_quat_w).abs() @ body_half
        delta = delta + extent_w[:, :2]

    return torch.linalg.norm(torch.clamp(delta, min=0.0), dim=1)


def _update_prev_goal_distance(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg,
    goal_center: Sequence[float],
    goal_radius: float,
) -> None:
    distance = _box_goal_distance(env, asset_cfg, goal_center, goal_radius)
    env._box_push_prev_goal_distance[env_ids] = distance[env_ids]


def _reset_goal_hold_state(env: ManagerBasedRLEnv, env_ids: torch.Tensor) -> None:
    env._box_push_goal_hold_steps[env_ids] = 0
    env._box_push_goal_inside[env_ids] = False
    env._box_push_goal_spatial_inside[env_ids] = False
    env._box_push_goal_was_spatial_inside[env_ids] = False
    env._box_push_goal_left_after_entry[env_ids] = False
    env._box_push_goal_max_hold_steps[env_ids] = 0
    env._box_push_goal_held_success[env_ids] = False


def _update_goal_hold_state(
    env: ManagerBasedRLEnv,
    goal_center: Sequence[float],
    goal_radius: float,
    required_hold_steps: int,
    box_half_extent: float = 0.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("box"),
) -> None:
    """Update consecutive in-goal hold counters.

    The counter increases only when the box is inside the goal region.
    If the box leaves the goal region, the counter is reset to zero.

    The update is guarded so it happens at most once per env step even if
    multiple reward / termination terms query the same state.
    """
    _ensure_box_push_state(env, goal_radius)

    current_step = int(env.common_step_counter)
    if env._box_push_goal_hold_last_step == current_step:
        return

    spatial_inside = _box_fully_inside_goal(
        env,
        asset_cfg=asset_cfg,
        goal_center=goal_center,
        goal_radius=goal_radius,
        box_half_extent=box_half_extent,
    )

    env._box_push_goal_left_after_entry[:] = (
        env._box_push_goal_was_spatial_inside
        & (~spatial_inside)
        & (~env._box_push_goal_held_success)
    )
    env._box_push_goal_spatial_inside[:] = spatial_inside
    env._box_push_goal_inside[:] = spatial_inside
    env._box_push_goal_hold_steps[:] = torch.where(
        spatial_inside,
        env._box_push_goal_hold_steps + 1,
        torch.zeros_like(env._box_push_goal_hold_steps),
    )
    env._box_push_goal_max_hold_steps[:] = torch.maximum(
        env._box_push_goal_max_hold_steps,
        env._box_push_goal_hold_steps,
    )
    env._box_push_goal_held_success[:] = env._box_push_goal_hold_steps >= int(required_hold_steps)
    env._box_push_goal_was_spatial_inside[:] = (
        env._box_push_goal_was_spatial_inside | spatial_inside
    )
    env._box_push_goal_hold_last_step = current_step


def _airflow_nozzle_pose_and_axis_w(
    env: ManagerBasedRLEnv,
    airflow_term_name: str,
    local_axis: Sequence[float] | str | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return nozzle position, orientation, and world-frame airflow axis.

    By default, the local axis is read from the airflow action term cfg so
    reward/observation geometry follows the same nozzle setup that applies the
    physical airflow force.
    """
    airflow_term = env.action_manager.get_term(airflow_term_name)
    nozzle_pos_w, nozzle_quat_w = airflow_term._nozzle_pose_w()

    if local_axis is None:
        local_axis = getattr(airflow_term.cfg, "nozzle_local_axis", "z")

    if isinstance(local_axis, str):
        sign = -1.0 if local_axis.startswith("-") else 1.0
        axis_name = local_axis[1:] if local_axis.startswith("-") else local_axis
        axis_map = {
            "x": (1.0, 0.0, 0.0),
            "y": (0.0, 1.0, 0.0),
            "z": (0.0, 0.0, 1.0),
        }
        if axis_name not in axis_map:
            raise ValueError(f"Unsupported nozzle local axis: {local_axis!r}")
        local_axis = tuple(sign * value for value in axis_map[axis_name])

    axis_l = torch.tensor(local_axis, dtype=torch.float32, device=env.device).repeat(env.num_envs, 1)
    axis_w = math_utils.quat_apply(nozzle_quat_w, axis_l)
    axis_w = axis_w / torch.linalg.norm(axis_w, dim=1, keepdim=True).clamp_min(1e-6)

    return nozzle_pos_w, nozzle_quat_w, axis_w
