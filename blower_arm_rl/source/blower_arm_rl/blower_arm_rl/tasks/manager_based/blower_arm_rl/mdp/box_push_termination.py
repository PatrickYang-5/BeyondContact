# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Termination terms for the airflow box-push task."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils import math as math_utils

from .box_push_common import (
    _airflow_nozzle_pose_and_axis_w,
    _box_fully_inside_goal,
    _update_goal_hold_state,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def box_out_of_workspace(
    env: ManagerBasedRLEnv,
    workspace_bounds: Sequence[float],
    z_bounds: Sequence[float] = (-0.02, 0.25),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("box"),
) -> torch.Tensor:
    box: RigidObject = env.scene[asset_cfg.name]

    pos_local_xy = box.data.root_pos_w[:, :2] - env.scene.env_origins[:, :2]
    z = box.data.root_pos_w[:, 2]

    xmin, xmax, ymin, ymax = workspace_bounds
    zmin, zmax = z_bounds

    return (
        (pos_local_xy[:, 0] < xmin)
        | (pos_local_xy[:, 0] > xmax)
        | (pos_local_xy[:, 1] < ymin)
        | (pos_local_xy[:, 1] > ymax)
        | (z < zmin)
        | (z > zmax)
    )


def box_in_goal(
    env: ManagerBasedRLEnv,
    goal_center: Sequence[float],
    goal_radius: float,
    box_half_extent: float = 0.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("box"),
) -> torch.Tensor:
    """Backward-compatible instantaneous goal termination.

    The new held-success cfg does not use this.
    """
    return _box_fully_inside_goal(
        env,
        asset_cfg=asset_cfg,
        goal_center=goal_center,
        goal_radius=goal_radius,
        box_half_extent=box_half_extent,
    ).bool()


def box_in_goal_held(
    env: ManagerBasedRLEnv,
    goal_center: Sequence[float],
    goal_radius: float,
    required_hold_steps: int,
    box_half_extent: float = 0.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("box"),
) -> torch.Tensor:
    """Terminate only after the box has stayed inside the goal long enough."""
    _update_goal_hold_state(
        env,
        goal_center=goal_center,
        goal_radius=goal_radius,
        required_hold_steps=required_hold_steps,
        box_half_extent=box_half_extent,
        asset_cfg=asset_cfg,
    )

    return env._box_push_goal_held_success.bool()


def robot_box_too_close(
    env: ManagerBasedRLEnv,
    box_half_extent: float,
    clearance: float,
    ignore_initial_s: float = 0.50,
    robot_cfg: SceneEntityCfg = SceneEntityCfg(
        "robot",
        body_names=["panda_hand", "panda_leftfinger", "panda_rightfinger"],
    ),
    box_cfg: SceneEntityCfg = SceneEntityCfg("box"),
) -> torch.Tensor:
    """Terminate when hand/finger bodies enter a tiny contact shell around the box."""
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

    elapsed_s = env.episode_length_buf.to(env.device) * env.step_dt
    ready = elapsed_s >= float(ignore_initial_s)

    return ready & (closest_distance <= float(clearance))


def robot_contact(
    env: ManagerBasedRLEnv,
    threshold: float,
    ignore_initial_s: float = 0.0,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces"),
) -> torch.Tensor:
    """Terminate when selected robot bodies contact anything above a force threshold."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    body_ids = sensor_cfg.body_ids if sensor_cfg.body_ids is not None else slice(None)
    net_contact_forces = contact_sensor.data.net_forces_w_history[:, :, body_ids, :]
    force_norm = torch.norm(net_contact_forces, dim=-1)
    has_contact = torch.any(torch.max(force_norm, dim=1).values > float(threshold), dim=1)

    elapsed_s = env.episode_length_buf.to(env.device) * env.step_dt
    ready = elapsed_s >= float(ignore_initial_s)

    return ready & has_contact


def robot_filtered_contact(
    env: ManagerBasedRLEnv,
    threshold: float,
    ignore_initial_s: float = 0.0,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces"),
) -> torch.Tensor:
    """Terminate when selected robot bodies contact the sensor's filtered prims."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    body_ids = sensor_cfg.body_ids if sensor_cfg.body_ids is not None else slice(None)

    force_matrix = contact_sensor.data.force_matrix_w_history
    if force_matrix is None:
        return robot_contact(
            env,
            threshold=threshold,
            ignore_initial_s=ignore_initial_s,
            sensor_cfg=sensor_cfg,
        )

    force_matrix = force_matrix[:, :, body_ids, :, :]
    force_norm = torch.norm(force_matrix, dim=-1)
    has_contact = torch.any(force_norm > float(threshold), dim=(1, 2, 3))

    elapsed_s = env.episode_length_buf.to(env.device) * env.step_dt
    ready = elapsed_s >= float(ignore_initial_s)

    return ready & has_contact


def nozzle_too_far_back(
    env: ManagerBasedRLEnv,
    min_forward_x: float = 0.05,
    ignore_initial_s: float = 0.10,
    airflow_term_name: str = "airflow",
) -> torch.Tensor:
    """Terminate when the nozzle x position (local frame) falls below min_forward_x.

    Mirrors nozzle_forward_position_penalty but resets the episode immediately,
    preventing the policy from learning to exploit behind-the-base configurations
    that the penalty alone was too weak to fully suppress.
    """
    nozzle_pos_w, _, _ = _airflow_nozzle_pose_and_axis_w(env, airflow_term_name)
    nozzle_x_local = nozzle_pos_w[:, 0] - env.scene.env_origins[:, 0]

    elapsed_s = env.episode_length_buf.to(env.device) * env.step_dt
    ready = elapsed_s >= float(ignore_initial_s)

    return ready & (nozzle_x_local < float(min_forward_x))


def nozzle_too_low(
    env: ManagerBasedRLEnv,
    min_height: float,
    ignore_initial_s: float = 0.10,
    airflow_term_name: str = "airflow",
) -> torch.Tensor:
    """Terminate when the airflow nozzle falls below a table-relative height."""
    nozzle_pos_w, _, _ = _airflow_nozzle_pose_and_axis_w(env, airflow_term_name)
    min_z = env.scene.env_origins[:, 2] + float(min_height)

    elapsed_s = env.episode_length_buf.to(env.device) * env.step_dt
    ready = elapsed_s >= float(ignore_initial_s)

    return ready & (nozzle_pos_w[:, 2] < min_z)


def body_local_boxes_too_low(
    env: ManagerBasedRLEnv,
    min_height: float,
    box_centers: Sequence[Sequence[float]],
    box_half_extents: Sequence[Sequence[float]],
    box_quats: Sequence[Sequence[float]] | None = None,
    ignore_initial_s: float = 0.10,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Terminate when any configured body-local collision box falls below table height."""
    robot: Articulation = env.scene[robot_cfg.name]
    body_ids = robot_cfg.body_ids if robot_cfg.body_ids is not None else slice(None)

    body_pos_w = robot.data.body_pos_w[:, body_ids]
    body_quat_w = robot.data.body_quat_w[:, body_ids]
    if body_pos_w.dim() == 2:
        body_pos_w = body_pos_w.unsqueeze(1)
        body_quat_w = body_quat_w.unsqueeze(1)

    centers = torch.tensor(box_centers, dtype=torch.float32, device=env.device)
    half_extents = torch.tensor(box_half_extents, dtype=torch.float32, device=env.device)
    if box_quats is None:
        quats = torch.zeros((centers.shape[0], 4), dtype=torch.float32, device=env.device)
        quats[:, 0] = 1.0
    else:
        quats = torch.tensor(box_quats, dtype=torch.float32, device=env.device)

    corner_signs = torch.tensor(
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

    min_z = torch.full((env.num_envs,), float("inf"), dtype=torch.float32, device=env.device)
    num_envs, num_bodies = body_pos_w.shape[:2]
    body_quat_flat = body_quat_w.reshape(num_envs * num_bodies, 4)

    for center, half_extent, quat in zip(centers, half_extents, quats):
        local_corners = center.unsqueeze(0) + math_utils.quat_apply(
            quat.unsqueeze(0).repeat(corner_signs.shape[0], 1),
            corner_signs * half_extent.unsqueeze(0),
        )
        local_corners = local_corners.view(1, 1, corner_signs.shape[0], 3).expand(num_envs, num_bodies, -1, -1)
        world_offsets = math_utils.quat_apply(
            body_quat_flat.unsqueeze(1).expand(-1, corner_signs.shape[0], -1).reshape(-1, 4),
            local_corners.reshape(-1, 3),
        ).view(num_envs, num_bodies, corner_signs.shape[0], 3)
        corners_w = body_pos_w.unsqueeze(2) + world_offsets
        min_z = torch.minimum(min_z, torch.amin(corners_w[..., 2], dim=(1, 2)))

    table_z = env.scene.env_origins[:, 2] + float(min_height)
    elapsed_s = env.episode_length_buf.to(env.device) * env.step_dt
    ready = elapsed_s >= float(ignore_initial_s)

    return ready & (min_z < table_z)


def body_local_boxes_box_contact_force(
    env: ManagerBasedRLEnv,
    force_threshold: float,
    box_half_extent: float,
    box_centers: Sequence[Sequence[float]],
    box_half_extents: Sequence[Sequence[float]],
    box_quats: Sequence[Sequence[float]] | None = None,
    clearance: float = 0.0,
    ignore_initial_s: float = 0.10,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    box_cfg: SceneEntityCfg = SceneEntityCfg("box"),
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces"),
    target_box_half_extents: Sequence[float] | None = None,
) -> torch.Tensor:
    """Terminate when body-local boxes touch the task box with measured contact force."""
    robot: Articulation = env.scene[robot_cfg.name]
    box: RigidObject = env.scene[box_cfg.name]
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]

    robot_body_ids = robot_cfg.body_ids if robot_cfg.body_ids is not None else slice(None)
    sensor_body_ids = sensor_cfg.body_ids if sensor_cfg.body_ids is not None else robot_body_ids

    body_pos_w = robot.data.body_pos_w[:, robot_body_ids]
    body_quat_w = robot.data.body_quat_w[:, robot_body_ids]
    if body_pos_w.dim() == 2:
        body_pos_w = body_pos_w.unsqueeze(1)
        body_quat_w = body_quat_w.unsqueeze(1)

    net_contact_forces = contact_sensor.data.net_forces_w_history[:, :, sensor_body_ids, :]
    force_norm = torch.norm(net_contact_forces, dim=-1)
    has_force = torch.any(torch.max(force_norm, dim=1).values > float(force_threshold), dim=1)

    centers = torch.tensor(box_centers, dtype=torch.float32, device=env.device)
    half_extents = torch.tensor(box_half_extents, dtype=torch.float32, device=env.device)
    if box_quats is None:
        quats = torch.zeros((centers.shape[0], 4), dtype=torch.float32, device=env.device)
        quats[:, 0] = 1.0
    else:
        quats = torch.tensor(box_quats, dtype=torch.float32, device=env.device)

    corner_signs = torch.tensor(
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

    num_envs, num_bodies = body_pos_w.shape[:2]
    body_quat_flat = body_quat_w.reshape(num_envs * num_bodies, 4)
    box_half = torch.full((num_envs, 1, 3), float(box_half_extent) + float(clearance), device=env.device)
    box_min = box.data.root_pos_w.unsqueeze(1) - box_half
    box_max = box.data.root_pos_w.unsqueeze(1) + box_half
    if target_box_half_extents is not None:
        # Compare in the target frame for non-cubic targets. Preserve the
        # original world-AABB behavior for all existing scalar-size callers.
        box_half = torch.tensor(target_box_half_extents, device=env.device).view(1, 1, 3) + clearance
        box_min, box_max = -box_half, box_half
    any_overlap = torch.zeros((env.num_envs,), dtype=torch.bool, device=env.device)

    for center, half_extent, quat in zip(centers, half_extents, quats):
        local_corners = center.unsqueeze(0) + math_utils.quat_apply(
            quat.unsqueeze(0).repeat(corner_signs.shape[0], 1),
            corner_signs * half_extent.unsqueeze(0),
        )
        local_corners = local_corners.view(1, 1, corner_signs.shape[0], 3).expand(num_envs, num_bodies, -1, -1)
        world_offsets = math_utils.quat_apply(
            body_quat_flat.unsqueeze(1).expand(-1, corner_signs.shape[0], -1).reshape(-1, 4),
            local_corners.reshape(-1, 3),
        ).view(num_envs, num_bodies, corner_signs.shape[0], 3)
        corners_w = body_pos_w.unsqueeze(2) + world_offsets
        if target_box_half_extents is not None:
            target_quat = box.data.root_quat_w[:, None, None, :].expand(-1, num_bodies, 8, -1)
            corners_w = math_utils.quat_apply_inverse(
                target_quat.reshape(-1, 4),
                (corners_w - box.data.root_pos_w[:, None, None, :]).reshape(-1, 3),
            ).view(num_envs, num_bodies, 8, 3)
        local_box_min = torch.amin(corners_w, dim=2)
        local_box_max = torch.amax(corners_w, dim=2)
        overlaps = torch.all((local_box_min <= box_max) & (local_box_max >= box_min), dim=-1)
        any_overlap |= torch.any(overlaps, dim=1)

    elapsed_s = env.episode_length_buf.to(env.device) * env.step_dt
    ready = elapsed_s >= float(ignore_initial_s)

    return ready & has_force & any_overlap
