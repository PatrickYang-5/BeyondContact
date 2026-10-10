# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Curriculum terms for the airflow box-push task."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from .box_push_common import _ensure_box_push_state, _ids_to_tensor, _pending_goal_radius

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def adaptive_goal_region_curriculum(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int] | slice | torch.Tensor,
    goal_center: Sequence[float],
    goal_radius_start: float,
    goal_radius_final: float,
    success_threshold: float,
    failure_threshold: float,
    shrink_factor: float,
    grow_factor: float,
    success_ema_momentum: float,
    warmup_steps: int,
    min_steps_between_updates: int,
    initial_range_step: float,
    angle_span_start: float | None = None,
    angle_span_final: float | None = None,
    expand_angle_before_shrinking_goal: bool = True,
    success_term_name: str = "success",
    required_hold_steps: int = 1,
    robot_reset_shrink_enabled: bool = False,
    robot_reset_success_threshold: float = 0.85,
    robot_reset_failure_threshold: float = 0.35,
    robot_reset_shrink_step: float = 0.10,
    robot_reset_grow_step: float = 0.05,
    robot_reset_final_scale: float = 0.0,
    robot_reset_warmup_steps: int = 6000,
    robot_reset_min_steps_between_updates: int = 1500,
    box_spawn_distance_curriculum_enabled: bool = False,
    box_spawn_distance_success_threshold: float = 0.70,
    box_spawn_distance_failure_threshold: float = 0.25,
    box_spawn_distance_step: float = 0.10,
    box_spawn_distance_retreat_step: float = 0.05,
    box_spawn_distance_final_scale: float = 1.0,
    box_spawn_distance_warmup_steps: int = 3000,
    box_spawn_distance_min_steps_between_updates: int = 1500,
    goal_center_range_curriculum_enabled: bool = False,
    goal_center_range_success_threshold: float = 0.70,
    goal_center_range_failure_threshold: float = 0.25,
    goal_center_range_step: float = 0.10,
    goal_center_range_retreat_step: float = 0.05,
    goal_center_range_final_scale: float = 1.0,
    goal_center_range_warmup_steps: int = 3000,
    goal_center_range_min_steps_between_updates: int = 1500,
):
    _ensure_box_push_state(env, goal_radius_start)

    env_ids = _ids_to_tensor(env, env_ids)

    success_rate = torch.tensor(0.0, dtype=torch.float32, device=env.device)

    if env_ids.numel() > 0:
        if hasattr(env, "_box_push_goal_max_hold_steps"):
            success_rate = (
                env._box_push_goal_max_hold_steps[env_ids] >= max(int(required_hold_steps), 1)
            ).float().mean()
        elif (
            hasattr(env, "termination_manager")
            and success_term_name in env.termination_manager.active_terms
        ):
            success_rate = env.termination_manager.get_term(success_term_name)[env_ids].float().mean()

    if env_ids.numel() > 0:
        env._box_push_success_ema = (
            success_ema_momentum * env._box_push_success_ema
            + (1.0 - success_ema_momentum) * success_rate
        )

    enough_steps = env.common_step_counter >= warmup_steps

    enough_since_update = (
        env._box_push_last_curriculum_update_step < 0
        or env.common_step_counter - env._box_push_last_curriculum_update_step >= min_steps_between_updates
    )

    if enough_steps and enough_since_update:
        pending_radius = _pending_goal_radius(env, goal_radius_start)
        radius = float(pending_radius.item())
        success_ema = float(env._box_push_success_ema.item())

        range_scale = float(torch.clamp(env._box_push_initial_range_scale, 0.0, 1.0).item())

        if (
            success_ema >= success_threshold
            and expand_angle_before_shrinking_goal
            and initial_range_step > 0.0
            and range_scale < 1.0
        ):
            env._box_push_initial_range_scale = torch.clamp(
                env._box_push_initial_range_scale + initial_range_step,
                0.0,
                1.0,
            )
            env._box_push_last_curriculum_update_step = env.common_step_counter

        elif success_ema >= success_threshold and radius > goal_radius_final:
            new_radius = max(goal_radius_final, radius * shrink_factor)
            env._box_push_pending_goal_radius.fill_(new_radius)

            env._box_push_initial_range_scale = torch.clamp(
                env._box_push_initial_range_scale + initial_range_step,
                0.0,
                1.0,
            )

            env._box_push_last_curriculum_update_step = env.common_step_counter

        elif success_ema <= failure_threshold and radius < goal_radius_start:
            env._box_push_pending_goal_radius.fill_(min(goal_radius_start, radius * grow_factor))
            env._box_push_last_curriculum_update_step = env.common_step_counter

        elif success_ema <= failure_threshold and initial_range_step > 0.0 and range_scale > 0.0:
            env._box_push_initial_range_scale = torch.clamp(
                env._box_push_initial_range_scale - 0.5 * initial_range_step,
                0.0,
                1.0,
            )
            env._box_push_last_curriculum_update_step = env.common_step_counter

    robot_reset_scale = torch.clamp(env._box_push_robot_reset_range_scale, 0.0, 1.0)
    enough_robot_steps = env.common_step_counter >= int(robot_reset_warmup_steps)
    enough_since_robot_update = (
        env._box_push_last_robot_reset_curriculum_update_step < 0
        or env.common_step_counter - env._box_push_last_robot_reset_curriculum_update_step
        >= int(robot_reset_min_steps_between_updates)
    )

    if robot_reset_shrink_enabled and enough_robot_steps and enough_since_robot_update:
        success_ema = float(env._box_push_success_ema.item())
        scale = float(robot_reset_scale.item())
        final_scale = max(min(float(robot_reset_final_scale), 1.0), 0.0)

        if success_ema >= robot_reset_success_threshold and scale > final_scale:
            new_scale = max(final_scale, scale - max(float(robot_reset_shrink_step), 0.0))
            env._box_push_robot_reset_range_scale.fill_(new_scale)
            env._box_push_last_robot_reset_curriculum_update_step = env.common_step_counter

        elif success_ema <= robot_reset_failure_threshold and scale < 1.0:
            new_scale = min(1.0, scale + max(float(robot_reset_grow_step), 0.0))
            env._box_push_robot_reset_range_scale.fill_(new_scale)
            env._box_push_last_robot_reset_curriculum_update_step = env.common_step_counter

    spawn_distance_scale = torch.clamp(env._box_push_box_spawn_distance_scale, 0.0, 1.0)
    enough_spawn_distance_steps = env.common_step_counter >= int(box_spawn_distance_warmup_steps)
    enough_since_spawn_distance_update = (
        env._box_push_last_box_spawn_distance_curriculum_update_step < 0
        or env.common_step_counter - env._box_push_last_box_spawn_distance_curriculum_update_step
        >= int(box_spawn_distance_min_steps_between_updates)
    )

    if (
        box_spawn_distance_curriculum_enabled
        and enough_spawn_distance_steps
        and enough_since_spawn_distance_update
    ):
        success_ema = float(env._box_push_success_ema.item())
        scale = float(spawn_distance_scale.item())
        final_scale = max(min(float(box_spawn_distance_final_scale), 1.0), 0.0)

        if success_ema >= box_spawn_distance_success_threshold and scale < final_scale:
            new_scale = min(final_scale, scale + max(float(box_spawn_distance_step), 0.0))
            env._box_push_box_spawn_distance_scale.fill_(new_scale)
            env._box_push_last_box_spawn_distance_curriculum_update_step = env.common_step_counter

        elif success_ema <= box_spawn_distance_failure_threshold and scale > 0.0:
            new_scale = max(0.0, scale - max(float(box_spawn_distance_retreat_step), 0.0))
            env._box_push_box_spawn_distance_scale.fill_(new_scale)
            env._box_push_last_box_spawn_distance_curriculum_update_step = env.common_step_counter

    goal_center_range_scale = torch.clamp(env._box_push_goal_center_range_scale, 0.0, 1.0)
    enough_goal_center_steps = env.common_step_counter >= int(goal_center_range_warmup_steps)
    enough_since_goal_center_update = (
        env._box_push_last_goal_center_range_curriculum_update_step < 0
        or env.common_step_counter - env._box_push_last_goal_center_range_curriculum_update_step
        >= int(goal_center_range_min_steps_between_updates)
    )

    if (
        goal_center_range_curriculum_enabled
        and enough_goal_center_steps
        and enough_since_goal_center_update
    ):
        success_ema = float(env._box_push_success_ema.item())
        scale = float(goal_center_range_scale.item())
        final_scale = max(min(float(goal_center_range_final_scale), 1.0), 0.0)

        if success_ema >= goal_center_range_success_threshold and scale < final_scale:
            new_scale = min(final_scale, scale + max(float(goal_center_range_step), 0.0))
            env._box_push_goal_center_range_scale.fill_(new_scale)
            env._box_push_last_goal_center_range_curriculum_update_step = env.common_step_counter

        elif success_ema <= goal_center_range_failure_threshold and scale > 0.0:
            new_scale = max(0.0, scale - max(float(goal_center_range_retreat_step), 0.0))
            env._box_push_goal_center_range_scale.fill_(new_scale)
            env._box_push_last_goal_center_range_curriculum_update_step = env.common_step_counter

    return {
        "goal_radius": env._box_push_goal_radius.mean(),
        "next_goal_radius": _pending_goal_radius(env, goal_radius_start),
        "success_rate": success_rate,
        "success_ema": env._box_push_success_ema,
        "robot_reset_range_scale": env._box_push_robot_reset_range_scale,
        "box_spawn_distance_scale": env._box_push_box_spawn_distance_scale,
        "goal_center_range_scale": env._box_push_goal_center_range_scale,
    }
