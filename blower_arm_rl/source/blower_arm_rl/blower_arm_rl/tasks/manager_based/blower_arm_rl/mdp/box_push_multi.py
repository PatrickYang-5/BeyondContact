# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Thin multi-box wrappers for the airflow box-push task."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import math as math_utils

from .box_push_common import (
    _activate_pending_goal_radius,
    _as_float_tensor,
    _box_fully_inside_goal,
    _box_goal_distance,
    _box_xy_w,
    _ensure_box_push_state,
    _goal_center_xy_local,
    _goal_center_xy_w,
    _goal_radius,
    _ids_to_tensor,
    _log_mean,
    _reset_goal_hold_state,
    _spawn_sector_from_goal_delta,
)
from .box_push_observation import box_privileged_state, nozzle_to_box
from .box_push_reward import (
    blower_box_distance_reward,
    blower_horizontal_push_alignment_reward,
    box_velocity_toward_goal_reward,
    robot_box_proximity_penalty,
)
from .box_push_termination import box_out_of_workspace, robot_box_too_close
from .box_push_visual import update_goal_region_visual

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _box_names(box_asset_names: Sequence[str]) -> tuple[str, ...]:
    names = tuple(str(name) for name in box_asset_names)
    if len(names) == 0:
        raise ValueError("multi-box terms require at least one box asset name.")
    return names


def _box_cfg(name: str) -> SceneEntityCfg:
    return SceneEntityCfg(name)


def _ensure_multi_box_push_state(
    env: ManagerBasedRLEnv,
    goal_radius: float,
    num_boxes: int,
    initial_range_scale: float = 0.0,
) -> None:
    _ensure_box_push_state(env, goal_radius, initial_range_scale=initial_range_scale)

    expected_shape = (env.num_envs, int(num_boxes))
    if (
        not hasattr(env, "_multi_box_push_prev_goal_distance")
        or tuple(env._multi_box_push_prev_goal_distance.shape) != expected_shape
    ):
        env._multi_box_push_prev_goal_distance = torch.zeros(
            expected_shape,
            dtype=torch.float32,
            device=env.device,
        )

    if (
        not hasattr(env, "_multi_box_push_goal_inside_each")
        or tuple(env._multi_box_push_goal_inside_each.shape) != expected_shape
    ):
        env._multi_box_push_goal_inside_each = torch.zeros(
            expected_shape,
            dtype=torch.bool,
            device=env.device,
        )


def _multi_box_goal_distances(
    env: ManagerBasedRLEnv,
    box_asset_names: Sequence[str],
    goal_center: Sequence[float],
    goal_radius: float,
) -> torch.Tensor:
    names = _box_names(box_asset_names)
    _ensure_multi_box_push_state(env, goal_radius, len(names))
    return torch.stack(
        [
            _box_goal_distance(env, _box_cfg(name), goal_center, goal_radius)
            for name in names
        ],
        dim=1,
    )


def _multi_box_goal_center_distances(
    env: ManagerBasedRLEnv,
    box_asset_names: Sequence[str],
    goal_center: Sequence[float],
    goal_radius: float,
) -> torch.Tensor:
    names = _box_names(box_asset_names)
    _ensure_multi_box_push_state(env, goal_radius, len(names))
    goal_xy = _goal_center_xy_w(env, goal_center)
    return torch.stack(
        [
            torch.linalg.norm(_box_xy_w(env, _box_cfg(name)) - goal_xy, dim=1)
            for name in names
        ],
        dim=1,
    )


def _multi_box_goal_reward_distances(
    env: ManagerBasedRLEnv,
    box_asset_names: Sequence[str],
    goal_center: Sequence[float],
    goal_radius: float,
    distance_mode: str,
) -> torch.Tensor:
    mode = str(distance_mode).lower()
    if mode in ("edge", "boundary", "area"):
        return _multi_box_goal_distances(env, box_asset_names, goal_center, goal_radius)
    if mode in ("center", "area_center", "goal_center"):
        return _multi_box_goal_center_distances(env, box_asset_names, goal_center, goal_radius)

    raise ValueError(
        f"Unsupported multi-box goal distance_mode={distance_mode!r}. "
        "Expected 'edge' or 'center'."
    )


def _multi_box_inside_goal(
    env: ManagerBasedRLEnv,
    box_asset_names: Sequence[str],
    goal_center: Sequence[float],
    goal_radius: float,
    box_half_extent: float,
) -> torch.Tensor:
    names = _box_names(box_asset_names)
    _ensure_multi_box_push_state(env, goal_radius, len(names))
    return torch.stack(
        [
            _box_fully_inside_goal(
                env,
                asset_cfg=_box_cfg(name),
                goal_center=goal_center,
                goal_radius=goal_radius,
                box_half_extent=box_half_extent,
            )
            for name in names
        ],
        dim=1,
    )


def _multi_box_reward_aggregate(
    env: ManagerBasedRLEnv,
    rewards: torch.Tensor,
    box_asset_names: Sequence[str],
    goal_center: Sequence[float],
    goal_radius: float,
    box_half_extent: float,
    aggregation: str,
) -> torch.Tensor:
    mode = str(aggregation).lower()
    names = _box_names(box_asset_names)

    if mode in ("mean", "average"):
        return rewards.mean(dim=1)
    if mode == "sum":
        return rewards.sum(dim=1)
    if mode in ("min", "worst"):
        return rewards.min(dim=1).values
    if mode in ("max", "best"):
        return rewards.max(dim=1).values

    raise ValueError(
        f"Unsupported multi-box reward aggregation={aggregation!r}. "
        "Expected 'mean', 'sum', 'min', or 'max'."
    )


def _update_multi_box_prev_goal_distance(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    box_asset_names: Sequence[str],
    goal_center: Sequence[float],
    goal_radius: float,
) -> None:
    distances = _multi_box_goal_distances(env, box_asset_names, goal_center, goal_radius)
    env._multi_box_push_prev_goal_distance[env_ids] = distances[env_ids]


def _update_multi_box_prev_goal_center_distance(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    box_asset_names: Sequence[str],
    goal_center: Sequence[float],
    goal_radius: float,
) -> None:
    distances = _multi_box_goal_center_distances(env, box_asset_names, goal_center, goal_radius)
    if (
        not hasattr(env, "_multi_box_push_prev_goal_center_distance")
        or tuple(env._multi_box_push_prev_goal_center_distance.shape) != tuple(distances.shape)
    ):
        env._multi_box_push_prev_goal_center_distance = distances.detach().clone()
    else:
        env._multi_box_push_prev_goal_center_distance[env_ids] = distances[env_ids]


def _multi_box_prev_reward_distance_attr(distance_mode: str) -> str:
    mode = str(distance_mode).lower()
    if mode in ("center", "area_center", "goal_center"):
        return "_multi_box_push_prev_goal_center_distance"
    return "_multi_box_push_prev_goal_distance"


def _ensure_multi_box_prev_reward_distance(
    env: ManagerBasedRLEnv,
    distances: torch.Tensor,
    distance_mode: str,
) -> torch.Tensor:
    attr_name = _multi_box_prev_reward_distance_attr(distance_mode)
    if not hasattr(env, attr_name) or tuple(getattr(env, attr_name).shape) != tuple(distances.shape):
        setattr(env, attr_name, distances.detach().clone())
    return getattr(env, attr_name)


def _update_multi_box_goal_hold_state(
    env: ManagerBasedRLEnv,
    goal_center: Sequence[float],
    goal_radius: float,
    required_hold_steps: int,
    box_half_extent: float,
    box_asset_names: Sequence[str],
) -> None:
    names = _box_names(box_asset_names)
    _ensure_multi_box_push_state(env, goal_radius, len(names))

    current_step = int(env.common_step_counter)
    if env._box_push_goal_hold_last_step == current_step:
        return

    inside_each = _multi_box_inside_goal(
        env,
        box_asset_names=names,
        goal_center=goal_center,
        goal_radius=goal_radius,
        box_half_extent=box_half_extent,
    )
    all_inside = torch.all(inside_each, dim=1)

    env._multi_box_push_goal_inside_each[:] = inside_each
    env._box_push_goal_left_after_entry[:] = (
        env._box_push_goal_was_spatial_inside
        & (~all_inside)
        & (~env._box_push_goal_held_success)
    )
    env._box_push_goal_spatial_inside[:] = all_inside
    env._box_push_goal_inside[:] = all_inside
    env._box_push_goal_hold_steps[:] = torch.where(
        all_inside,
        env._box_push_goal_hold_steps + 1,
        torch.zeros_like(env._box_push_goal_hold_steps),
    )
    env._box_push_goal_max_hold_steps[:] = torch.maximum(
        env._box_push_goal_max_hold_steps,
        env._box_push_goal_hold_steps,
    )
    env._box_push_goal_held_success[:] = env._box_push_goal_hold_steps >= int(required_hold_steps)
    env._box_push_goal_was_spatial_inside[:] = env._box_push_goal_was_spatial_inside | all_inside
    env._box_push_goal_hold_last_step = current_step

    _log_mean(env, "Metrics/multi_box_goal_inside_fraction", inside_each.float().mean(dim=1))
    _log_mean(env, "Metrics/multi_box_all_inside_rate", all_inside.float())


def reset_multi_box_state_around_goal(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    box_asset_names: Sequence[str],
    goal_center: Sequence[float],
    goal_center_range: Sequence[float] | None,
    goal_radius: float,
    spawn_z: float,
    radial_distance_range: Sequence[float],
    workspace_bounds: Sequence[float],
    spawn_workspace_bounds: Sequence[float] | None = None,
    goal_center_range_start: Sequence[float] | None = None,
    goal_center_range_scale_attr: str = "_box_push_goal_center_range_scale",
    radial_distance_range_start: Sequence[float] | None = None,
    radial_distance_scale_attr: str = "_box_push_box_spawn_distance_scale",
    angle_center: float = 3.141592653589793,
    angle_span_start: float | None = None,
    angle_span_final: float | None = None,
    box_half_extent: float = 0.025,
    box_clearance: float = 0.015,
    inter_box_clearance: float = 0.02,
    enforce_outside_goal_square: bool = True,
):
    """Reset several box assets around one shared, possibly randomized goal."""
    names = _box_names(box_asset_names)
    _ensure_multi_box_push_state(env, goal_radius, len(names))

    env_ids = _ids_to_tensor(env, env_ids)
    if env_ids.numel() == 0:
        return

    bounds = _as_float_tensor(env, workspace_bounds)
    spawn_bounds = bounds if spawn_workspace_bounds is None else _as_float_tensor(env, spawn_workspace_bounds)
    active_goal_radius = _activate_pending_goal_radius(env, env_ids, goal_radius)
    current_half_extent = float(torch.max(active_goal_radius).item())

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
    final_radial_range = _as_float_tensor(env, radial_distance_range)
    if radial_distance_range_start is None:
        effective_radial_range = final_radial_range
    else:
        start_radial_range = _as_float_tensor(env, radial_distance_range_start)
        scale = getattr(env, radial_distance_scale_attr, torch.tensor(1.0, device=env.device))
        scale = torch.clamp(scale, 0.0, 1.0)
        effective_radial_range = start_radial_range + scale * (final_radial_range - start_radial_range)

    min_dist = max(float(effective_radial_range[0].item()), current_half_extent + float(box_clearance))
    max_dist = float(effective_radial_range[1].item())
    if max_dist <= min_dist:
        max_dist = min_dist + 0.02

    if angle_span_start is None or angle_span_final is None:
        angle_span = float(torch.pi)
    else:
        scale = float(torch.clamp(env._box_push_initial_range_scale, 0.0, 1.0).item())
        angle_span = float(angle_span_start) + scale * (float(angle_span_final) - float(angle_span_start))
        angle_span = min(max(angle_span, 0.0), float(torch.pi))

    num_resets = env_ids.numel()
    previous_samples: list[torch.Tensor] = []
    min_center_separation = 2.0 * float(box_half_extent) + float(inter_box_clearance)

    for box_index, name in enumerate(names):
        samples = torch.empty((num_resets, 2), dtype=torch.float32, device=env.device)
        active = torch.ones(num_resets, dtype=torch.bool, device=env.device)

        for _ in range(64):
            active_count = int(active.sum().item())
            if active_count == 0:
                break

            if angle_span >= float(torch.pi) - 1e-5:
                angles = math_utils.sample_uniform(-torch.pi, torch.pi, (active_count,), env.device)
            else:
                angles = float(angle_center) + math_utils.sample_uniform(
                    -angle_span,
                    angle_span,
                    (active_count,),
                    env.device,
                )

            distances = math_utils.sample_uniform(min_dist, max_dist, (active_count,), env.device)
            samples[active, 0] = center[active, 0] + distances * torch.cos(angles)
            samples[active, 1] = center[active, 1] + distances * torch.sin(angles)

            delta = samples - center
            inside_goal = torch.all(
                torch.abs(delta) <= current_half_extent + float(box_clearance),
                dim=1,
            )
            outside_workspace = (
                (samples[:, 0] < spawn_bounds[0])
                | (samples[:, 0] > spawn_bounds[1])
                | (samples[:, 1] < spawn_bounds[2])
                | (samples[:, 1] > spawn_bounds[3])
            )

            too_close = torch.zeros(num_resets, dtype=torch.bool, device=env.device)
            for previous in previous_samples:
                too_close |= torch.linalg.norm(samples - previous, dim=1) < min_center_separation

            active = outside_workspace | too_close
            if enforce_outside_goal_square:
                active |= inside_goal

        if torch.any(active):
            fallback_angle = float(angle_center) + 2.0 * float(torch.pi) * box_index / max(len(names), 1)
            fallback_distance = min(max_dist, min_dist + 0.5 * min_center_separation * box_index)
            samples[active, 0] = center[active, 0] + fallback_distance * torch.cos(
                torch.tensor(fallback_angle, dtype=torch.float32, device=env.device)
            )
            samples[active, 1] = center[active, 1] + fallback_distance * torch.sin(
                torch.tensor(fallback_angle, dtype=torch.float32, device=env.device)
            )

        samples[:, 0].clamp_(float(spawn_bounds[0].item()), float(spawn_bounds[1].item()))
        samples[:, 1].clamp_(float(spawn_bounds[2].item()), float(spawn_bounds[3].item()))

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
                    current_half_extent + float(box_clearance) + 0.005
                ) / torch.max(torch.abs(safe_delta), dim=1).values.clamp_min(1e-6)
                samples[inside_goal] = center[inside_goal] + safe_delta * scale.unsqueeze(1)

        if box_index == 0:
            spawn_delta = samples - center
            env._box_push_spawn_angle[env_ids] = torch.atan2(spawn_delta[:, 1], spawn_delta[:, 0])
            env._box_push_spawn_sector[env_ids] = _spawn_sector_from_goal_delta(spawn_delta)

        box: RigidObject = env.scene[name]
        root_states = box.data.default_root_state[env_ids].clone()
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

        box.write_root_pose_to_sim(torch.cat([positions, orientations], dim=-1), env_ids=env_ids)
        box.write_root_velocity_to_sim(velocities, env_ids=env_ids)
        previous_samples.append(samples.clone())

    _update_multi_box_prev_goal_distance(env, env_ids, names, goal_center, goal_radius)
    _update_multi_box_prev_goal_center_distance(env, env_ids, names, goal_center, goal_radius)
    _reset_goal_hold_state(env, env_ids)
    env._multi_box_push_goal_inside_each[env_ids] = False

    update_goal_region_visual(env, goal_center, goal_radius=goal_radius, env_ids=env_ids)


def multi_box_to_goal_xy(
    env: ManagerBasedRLEnv,
    goal_center: Sequence[float],
    box_asset_names: Sequence[str],
) -> torch.Tensor:
    return torch.cat(
        [
            _goal_center_xy_w(env, goal_center) - _box_xy_w(env, _box_cfg(name))
            for name in _box_names(box_asset_names)
        ],
        dim=1,
    )


def multi_nozzle_to_boxes(
    env: ManagerBasedRLEnv,
    box_asset_names: Sequence[str],
    airflow_term_name: str = "airflow",
) -> torch.Tensor:
    return torch.cat(
        [
            nozzle_to_box(env, airflow_term_name=airflow_term_name, asset_cfg=_box_cfg(name))
            for name in _box_names(box_asset_names)
        ],
        dim=1,
    )


def multi_box_goal_radius_obs(env: ManagerBasedRLEnv, goal_radius: float) -> torch.Tensor:
    return _goal_radius(env, goal_radius).unsqueeze(1)


def multi_box_privileged_state(
    env: ManagerBasedRLEnv,
    goal_center: Sequence[float],
    goal_radius: float,
    box_asset_names: Sequence[str],
) -> torch.Tensor:
    return torch.cat(
        [
            box_privileged_state(
                env,
                goal_center=goal_center,
                goal_radius=goal_radius,
                asset_cfg=_box_cfg(name),
            )
            for name in _box_names(box_asset_names)
        ],
        dim=1,
    )


def multi_box_inside_goal_obs(
    env: ManagerBasedRLEnv,
    goal_center: Sequence[float],
    goal_radius: float,
    box_half_extent: float,
    box_asset_names: Sequence[str],
) -> torch.Tensor:
    """Per-box inside-goal flag as float, shape [N, num_boxes].

    Privileged: gives the critic explicit knowledge of which boxes are already
    collected. Without this, the critic must infer target status from goal_delta
    magnitudes, which is harder to learn and noisy near the goal boundary.
    """
    names = _box_names(box_asset_names)
    return _multi_box_inside_goal(env, names, goal_center, goal_radius, box_half_extent).float()


def multi_box_hold_progress_obs(
    env: ManagerBasedRLEnv,
    goal_radius: float,
    box_asset_names: Sequence[str],
    required_hold_steps: int,
) -> torch.Tensor:
    """All-boxes-in-goal hold progress as a scalar [N, 1], clamped to [0, 1].

    Privileged: tells the critic how close the episode is to the held-success
    terminal condition. Zero until all boxes are simultaneously inside the goal.
    """
    _ensure_multi_box_push_state(env, goal_radius, len(_box_names(box_asset_names)))
    hold_fraction = env._box_push_goal_hold_steps.float() / max(int(required_hold_steps), 1)
    return hold_fraction.unsqueeze(1).clamp(0.0, 1.0)


def multi_box_goal_center_l2_reward(
    env: ManagerBasedRLEnv,
    goal_center: Sequence[float],
    goal_radius: float,
    box_asset_names: Sequence[str],
    distance_threshold: float,
    box_half_extent: float = 0.0,
    aggregation: str = "mean",
) -> torch.Tensor:
    return multi_box_goal_distance_inverse_quadratic_reward(
        env,
        std=distance_threshold,
        goal_center=goal_center,
        goal_radius=goal_radius,
        box_asset_names=box_asset_names,
        box_half_extent=box_half_extent,
        aggregation=aggregation,
        distance_mode="center",
    )


def multi_box_goal_distance_inverse_quadratic_reward(
    env: ManagerBasedRLEnv,
    std: float,
    goal_center: Sequence[float],
    goal_radius: float,
    box_asset_names: Sequence[str],
    box_half_extent: float = 0.0,
    aggregation: str = "mean",
    distance_mode: str = "center",
) -> torch.Tensor:
    """Dense goal-distance reward with a non-zero tail for every positive distance.

    The reward is 1 / (1 + (d / std)^2).  With std=0.20, the raw slope at
    d=0.20 is -2.5 per meter, while still keeping a smooth positive tail for
    boxes that are far from the goal.
    """
    distances = _multi_box_goal_reward_distances(
        env,
        box_asset_names=box_asset_names,
        goal_center=goal_center,
        goal_radius=goal_radius,
        distance_mode=distance_mode,
    )
    distance_scale = max(float(std), 1e-6)
    rewards = 1.0 / (1.0 + torch.square(distances / distance_scale))

    center_distances = _multi_box_goal_center_distances(env, box_asset_names, goal_center, goal_radius)
    edge_distances = _multi_box_goal_distances(env, box_asset_names, goal_center, goal_radius)

    _log_mean(env, "Metrics/multi_box_mean_goal_distance", edge_distances.mean(dim=1))
    _log_mean(env, "Metrics/multi_box_min_goal_distance", edge_distances.min(dim=1).values)
    _log_mean(env, "Metrics/multi_box_max_goal_distance", edge_distances.max(dim=1).values)
    _log_mean(env, "Metrics/multi_box_mean_goal_center_distance", center_distances.mean(dim=1))
    _log_mean(env, "Metrics/multi_box_sum_goal_center_distance", center_distances.sum(dim=1))
    _log_mean(env, "Metrics/multi_box_goal_dense_reward", rewards.sum(dim=1))
    _log_mean(env, "Metrics/goal_radius", _goal_radius(env, goal_radius))

    return _multi_box_reward_aggregate(
        env,
        rewards,
        box_asset_names=box_asset_names,
        goal_center=goal_center,
        goal_radius=goal_radius,
        box_half_extent=box_half_extent,
        aggregation=aggregation,
    )


def multi_box_goal_progress(
    env: ManagerBasedRLEnv,
    std: float,
    goal_center: Sequence[float],
    goal_radius: float,
    box_asset_names: Sequence[str],
    box_half_extent: float = 0.0,
    aggregation: str = "mean",
    distance_mode: str = "edge",
) -> torch.Tensor:
    names = _box_names(box_asset_names)
    distances = _multi_box_goal_reward_distances(
        env,
        box_asset_names=names,
        goal_center=goal_center,
        goal_radius=goal_radius,
        distance_mode=distance_mode,
    )

    _ensure_multi_box_push_state(env, goal_radius, len(names))
    prev_distances = _ensure_multi_box_prev_reward_distance(env, distances, distance_mode)
    rewards = (prev_distances - distances) / max(float(std), 1e-6)
    reward = _multi_box_reward_aggregate(
        env,
        rewards,
        box_asset_names=names,
        goal_center=goal_center,
        goal_radius=goal_radius,
        box_half_extent=box_half_extent,
        aggregation=aggregation,
    )
    prev_distances[:] = distances.detach()

    return reward


def multi_box_goal_inside_fraction(
    env: ManagerBasedRLEnv,
    goal_center: Sequence[float],
    goal_radius: float,
    box_half_extent: float,
    box_asset_names: Sequence[str],
    aggregation: str = "mean",
) -> torch.Tensor:
    inside_each = _multi_box_inside_goal(
        env,
        box_asset_names=box_asset_names,
        goal_center=goal_center,
        goal_radius=goal_radius,
        box_half_extent=box_half_extent,
    )
    fraction = inside_each.float().mean(dim=1)

    _log_mean(env, "Metrics/multi_box_goal_inside_fraction", fraction)
    _log_mean(env, "Metrics/multi_box_all_inside_rate", torch.all(inside_each, dim=1).float())

    mode = str(aggregation).lower()
    if mode in ("mean", "average", "fraction"):
        return fraction
    if mode in ("sum", "count"):
        return inside_each.float().sum(dim=1)
    if mode in ("all", "all_inside"):
        return torch.all(inside_each, dim=1).float()

    raise ValueError(
        f"Unsupported multi-box goal-inside aggregation={aggregation!r}. "
        "Expected 'mean', 'sum', or 'all'."
    )


def multi_box_goal_centering_reward(
    env: ManagerBasedRLEnv,
    goal_center: Sequence[float],
    goal_radius: float,
    std: float,
    box_half_extent: float,
    box_asset_names: Sequence[str],
    aggregation: str = "mean",
) -> torch.Tensor:
    rewards = []
    goal_xy = _goal_center_xy_w(env, goal_center)
    for name in _box_names(box_asset_names):
        inside = _box_fully_inside_goal(
            env,
            asset_cfg=_box_cfg(name),
            goal_center=goal_center,
            goal_radius=goal_radius,
            box_half_extent=box_half_extent,
        ).float()
        center_distance = torch.linalg.norm(_box_xy_w(env, _box_cfg(name)) - goal_xy, dim=1)
        rewards.append(inside * torch.exp(-torch.square(center_distance / max(float(std), 1e-6))))

    rewards = torch.stack(rewards, dim=1)
    return _multi_box_reward_aggregate(
        env,
        rewards,
        box_asset_names=box_asset_names,
        goal_center=goal_center,
        goal_radius=goal_radius,
        box_half_extent=box_half_extent,
        aggregation=aggregation,
    )


def multi_box_goal_low_velocity_reward(
    env: ManagerBasedRLEnv,
    goal_center: Sequence[float],
    goal_radius: float,
    std: float,
    box_half_extent: float,
    required_hold_steps: int,
    box_asset_names: Sequence[str],
    aggregation: str = "mean",
) -> torch.Tensor:
    names = _box_names(box_asset_names)
    _update_multi_box_goal_hold_state(
        env,
        goal_center=goal_center,
        goal_radius=goal_radius,
        required_hold_steps=required_hold_steps,
        box_half_extent=box_half_extent,
        box_asset_names=names,
    )

    rewards = []
    for box_index, name in enumerate(names):
        box: RigidObject = env.scene[name]
        speed = torch.linalg.norm(box.data.root_lin_vel_w[:, :2], dim=1)
        inside = env._multi_box_push_goal_inside_each[:, box_index].float()
        rewards.append(inside * torch.exp(-torch.square(speed / max(float(std), 1e-6))))

    hold_fraction = env._box_push_goal_hold_steps.float() / max(int(required_hold_steps), 1)
    hold_fraction = torch.clamp(hold_fraction, 0.0, 1.0)
    success = env._box_push_goal_held_success.float()
    episode_success = (env._box_push_goal_max_hold_steps >= max(int(required_hold_steps), 1)).float()
    _log_mean(env, "Metrics/goal_hold_fraction", hold_fraction)
    _log_mean(env, "Metrics/success", success)
    _log_mean(env, "Metrics/episode_success", episode_success)

    rewards = torch.stack(rewards, dim=1)
    return _multi_box_reward_aggregate(
        env,
        rewards,
        box_asset_names=names,
        goal_center=goal_center,
        goal_radius=goal_radius,
        box_half_extent=box_half_extent,
        aggregation=aggregation,
    )


def multi_box_goal_held_success(
    env: ManagerBasedRLEnv,
    goal_center: Sequence[float],
    goal_radius: float,
    required_hold_steps: int,
    box_half_extent: float,
    box_asset_names: Sequence[str],
) -> torch.Tensor:
    _update_multi_box_goal_hold_state(
        env,
        goal_center=goal_center,
        goal_radius=goal_radius,
        required_hold_steps=required_hold_steps,
        box_half_extent=box_half_extent,
        box_asset_names=box_asset_names,
    )
    return env._box_push_goal_held_success.float()


def multi_box_goal_escape_penalty(
    env: ManagerBasedRLEnv,
    goal_center: Sequence[float],
    goal_radius: float,
    required_hold_steps: int,
    min_hold_steps: int,
    box_half_extent: float,
    box_asset_names: Sequence[str],
) -> torch.Tensor:
    _update_multi_box_goal_hold_state(
        env,
        goal_center=goal_center,
        goal_radius=goal_radius,
        required_hold_steps=required_hold_steps,
        box_half_extent=box_half_extent,
        box_asset_names=box_asset_names,
    )
    escaped = (
        env._box_push_goal_left_after_entry
        & (env._box_push_goal_max_hold_steps >= int(min_hold_steps))
        & (~env._box_push_goal_held_success)
    )
    hold_progress = env._box_push_goal_max_hold_steps.float() / max(int(required_hold_steps), 1)
    return escaped.float() * torch.clamp(hold_progress, 0.0, 1.0)


def multi_box_velocity_toward_goal_reward(
    env: ManagerBasedRLEnv,
    goal_center: Sequence[float],
    box_asset_names: Sequence[str],
    goal_radius: float = 0.0,
    box_half_extent: float = 0.0,
    aggregation: str = "mean",
) -> torch.Tensor:
    names = _box_names(box_asset_names)
    rewards = [
        box_velocity_toward_goal_reward(env, goal_center=goal_center, asset_cfg=_box_cfg(name))
        for name in names
    ]
    reward = _multi_box_reward_aggregate(
        env,
        torch.stack(rewards, dim=1),
        box_asset_names=names,
        goal_center=goal_center,
        goal_radius=goal_radius,
        box_half_extent=box_half_extent,
        aggregation=aggregation,
    )
    _log_mean(env, "Metrics/multi_box_velocity_toward_goal", reward)
    return reward


def multi_blower_box_distance_reward(
    env: ManagerBasedRLEnv,
    desired_distance: float,
    std: float,
    goal_center: Sequence[float],
    goal_radius: float,
    box_half_extent: float,
    box_asset_names: Sequence[str],
    aggregation: str = "mean",
    airflow_term_name: str = "airflow",
) -> torch.Tensor:
    names = _box_names(box_asset_names)
    rewards = torch.stack(
        [
            blower_box_distance_reward(
                env,
                desired_distance=desired_distance,
                std=std,
                airflow_term_name=airflow_term_name,
                asset_cfg=_box_cfg(name),
            )
            for name in names
        ],
        dim=1,
    )
    return _multi_box_reward_aggregate(
        env,
        rewards,
        box_asset_names=names,
        goal_center=goal_center,
        goal_radius=goal_radius,
        box_half_extent=box_half_extent,
        aggregation=aggregation,
    )


def multi_blower_horizontal_push_alignment_reward(
    env: ManagerBasedRLEnv,
    goal_center: Sequence[float],
    goal_radius: float,
    box_half_extent: float,
    box_asset_names: Sequence[str],
    nozzle_standoff: float = 0.14,
    position_std: float = 0.14,
    desired_height: float = 0.13,
    height_std: float = 0.07,
    position_blend: float = 0.50,
    min_horizontal_fraction: float = 0.75,
    horizontal_std: float = 0.20,
    aggregation: str = "mean",
    airflow_term_name: str = "airflow",
) -> torch.Tensor:
    names = _box_names(box_asset_names)
    rewards = torch.stack(
        [
            blower_horizontal_push_alignment_reward(
                env,
                goal_center=goal_center,
                nozzle_standoff=nozzle_standoff,
                position_std=position_std,
                desired_height=desired_height,
                height_std=height_std,
                position_blend=position_blend,
                min_horizontal_fraction=min_horizontal_fraction,
                horizontal_std=horizontal_std,
                airflow_term_name=airflow_term_name,
                asset_cfg=_box_cfg(name),
            )
            for name in names
        ],
        dim=1,
    )
    return _multi_box_reward_aggregate(
        env,
        rewards,
        box_asset_names=names,
        goal_center=goal_center,
        goal_radius=goal_radius,
        box_half_extent=box_half_extent,
        aggregation=aggregation,
    )


def multi_blower_target_alignment_reward(
    env: ManagerBasedRLEnv,
    goal_center: Sequence[float],
    goal_radius: float,
    box_half_extent: float,
    box_asset_names: Sequence[str],
    nozzle_standoff: float = 0.14,
    position_std: float = 0.14,
    desired_height: float = 0.13,
    height_std: float = 0.07,
    position_blend: float = 0.50,
    min_horizontal_fraction: float = 0.75,
    horizontal_std: float = 0.20,
    airflow_term_name: str = "airflow",
) -> torch.Tensor:
    """Alignment reward for the best-positioned out-of-goal box.

    Unlike mean/sum aggregation (which creates a diffuse pull toward the average
    upstream position of all boxes), this uses max over boxes not yet inside the
    goal. The arm is rewarded for being well-aligned to any ONE out-of-goal box,
    eliminating the 'chase all boxes simultaneously' oscillation. When all boxes
    are inside the goal the reward is zero, and the arm is held in place by the
    motion penalties alone.
    """
    names = _box_names(box_asset_names)

    per_box_alignment = torch.stack(
        [
            blower_horizontal_push_alignment_reward(
                env,
                goal_center=goal_center,
                nozzle_standoff=nozzle_standoff,
                position_std=position_std,
                desired_height=desired_height,
                height_std=height_std,
                position_blend=position_blend,
                min_horizontal_fraction=min_horizontal_fraction,
                horizontal_std=horizontal_std,
                airflow_term_name=airflow_term_name,
                asset_cfg=_box_cfg(name),
            )
            for name in names
        ],
        dim=1,
    )  # [N, num_boxes]

    # Exclude boxes already inside goal so they no longer attract the arm.
    in_goal = _multi_box_inside_goal(env, names, goal_center, goal_radius, box_half_extent)
    masked = per_box_alignment.masked_fill(in_goal, 0.0)

    # Max over out-of-goal boxes: arm only needs to be well-aligned to any one.
    reward = masked.max(dim=1).values

    _log_mean(env, "Metrics/multi_blower_target_alignment", reward)
    return reward


def multi_blower_target_box_distance_reward(
    env: ManagerBasedRLEnv,
    desired_distance: float,
    std: float,
    goal_center: Sequence[float],
    goal_radius: float,
    box_half_extent: float,
    box_asset_names: Sequence[str],
    airflow_term_name: str = "airflow",
) -> torch.Tensor:
    """Distance reward to the nearest out-of-goal box.

    Replaces mean aggregation with max over out-of-goal boxes to match the
    targeting logic of multi_blower_target_alignment_reward.
    """
    names = _box_names(box_asset_names)

    per_box_dist = torch.stack(
        [
            blower_box_distance_reward(
                env,
                desired_distance=desired_distance,
                std=std,
                airflow_term_name=airflow_term_name,
                asset_cfg=_box_cfg(name),
            )
            for name in names
        ],
        dim=1,
    )

    in_goal = _multi_box_inside_goal(env, names, goal_center, goal_radius, box_half_extent)
    masked = per_box_dist.masked_fill(in_goal, 0.0)
    return masked.max(dim=1).values


def multi_all_in_goal_joint_home_reward(
    env: ManagerBasedRLEnv,
    goal_center: Sequence[float],
    goal_radius: float,
    box_half_extent: float,
    box_asset_names: Sequence[str],
    target_joint_pos: Sequence[float],
    std: float = 0.4,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward joints returning to target positions when all boxes are inside the goal.

    Only active when every box is fully inside the goal region. Returns a Gaussian
    reward based on the L2 distance of joint positions from target_joint_pos.
    When any box is still outside the goal this returns zero, letting the alignment
    and distance rewards drive the arm instead.
    """
    names = _box_names(box_asset_names)
    in_goal = _multi_box_inside_goal(env, names, goal_center, goal_radius, box_half_extent)
    all_in_goal = in_goal.all(dim=1).float()  # [N]

    asset = env.scene[asset_cfg.name]
    joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]  # [N, num_joints]
    target = torch.tensor(list(target_joint_pos), dtype=joint_pos.dtype, device=joint_pos.device)
    deviation_sq = torch.sum((joint_pos - target) ** 2, dim=-1)  # [N]
    reward = torch.exp(-deviation_sq / (2.0 * std ** 2))

    _log_mean(env, "Metrics/all_in_goal_joint_home", reward * all_in_goal)
    return reward * all_in_goal


def multi_robot_box_proximity_penalty(
    env: ManagerBasedRLEnv,
    box_half_extent: float,
    clearance: float,
    box_asset_names: Sequence[str],
    robot_cfg: SceneEntityCfg = SceneEntityCfg(
        "robot",
        body_names=["panda_hand", "panda_leftfinger", "panda_rightfinger"],
    ),
) -> torch.Tensor:
    penalties = [
        robot_box_proximity_penalty(
            env,
            box_half_extent=box_half_extent,
            clearance=clearance,
            robot_cfg=robot_cfg,
            box_cfg=_box_cfg(name),
        )
        for name in _box_names(box_asset_names)
    ]
    return torch.stack(penalties, dim=1).max(dim=1).values


def multi_box_out_of_workspace(
    env: ManagerBasedRLEnv,
    workspace_bounds: Sequence[float],
    box_asset_names: Sequence[str],
    z_bounds: Sequence[float] = (-0.02, 0.25),
) -> torch.Tensor:
    out = [
        box_out_of_workspace(
            env,
            workspace_bounds=workspace_bounds,
            z_bounds=z_bounds,
            asset_cfg=_box_cfg(name),
        )
        for name in _box_names(box_asset_names)
    ]
    return torch.stack(out, dim=1).any(dim=1)


def multi_box_out_of_workspace_penalty(
    env: ManagerBasedRLEnv,
    workspace_bounds: Sequence[float],
    box_asset_names: Sequence[str],
    z_bounds: Sequence[float] = (-0.02, 0.25),
) -> torch.Tensor:
    out_of_workspace = multi_box_out_of_workspace(
        env,
        workspace_bounds=workspace_bounds,
        box_asset_names=box_asset_names,
        z_bounds=z_bounds,
    ).float()
    _log_mean(env, "Metrics/out_of_workspace_rate", out_of_workspace)
    return out_of_workspace


def multi_box_in_goal_held(
    env: ManagerBasedRLEnv,
    goal_center: Sequence[float],
    goal_radius: float,
    required_hold_steps: int,
    box_half_extent: float,
    box_asset_names: Sequence[str],
) -> torch.Tensor:
    _update_multi_box_goal_hold_state(
        env,
        goal_center=goal_center,
        goal_radius=goal_radius,
        required_hold_steps=required_hold_steps,
        box_half_extent=box_half_extent,
        box_asset_names=box_asset_names,
    )
    return env._box_push_goal_held_success.bool()


def multi_robot_box_too_close(
    env: ManagerBasedRLEnv,
    box_half_extent: float,
    clearance: float,
    box_asset_names: Sequence[str],
    ignore_initial_s: float = 0.50,
    robot_cfg: SceneEntityCfg = SceneEntityCfg(
        "robot",
        body_names=["panda_hand", "panda_leftfinger", "panda_rightfinger"],
    ),
) -> torch.Tensor:
    contacts = [
        robot_box_too_close(
            env,
            box_half_extent=box_half_extent,
            clearance=clearance,
            ignore_initial_s=ignore_initial_s,
            robot_cfg=robot_cfg,
            box_cfg=_box_cfg(name),
        )
        for name in _box_names(box_asset_names)
    ]
    return torch.stack(contacts, dim=1).any(dim=1)
