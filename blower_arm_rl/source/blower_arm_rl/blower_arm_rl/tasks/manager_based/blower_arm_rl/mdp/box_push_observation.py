# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Observation terms for the airflow box-push task."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg

from .box_push_common import (
    _airflow_nozzle_pose_and_axis_w,
    _box_xy_w,
    _goal_center_xy_w,
    _goal_radius,
    _prepush_setup_terms,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def goal_radius_obs(env: ManagerBasedRLEnv, goal_radius: float) -> torch.Tensor:
    return _goal_radius(env, goal_radius).unsqueeze(1)


def _policy_corruption_enabled(env: ManagerBasedRLEnv) -> bool:
    policy_obs_cfg = getattr(getattr(env.cfg, "observations", None), "policy", None)
    return policy_obs_cfg is None or bool(getattr(policy_obs_cfg, "enable_corruption", True))


def _scheduled_noise_scale(env: ManagerBasedRLEnv, warmup_steps: int, ramp_steps: int) -> float:
    if not _policy_corruption_enabled(env):
        return 0.0
    step = int(getattr(env, "common_step_counter", 0))
    warmup_steps = max(int(warmup_steps), 0)
    ramp_steps = max(int(ramp_steps), 0)
    if step < warmup_steps:
        return 0.0
    if ramp_steps == 0:
        return 1.0
    return min(float(step - warmup_steps) / float(ramp_steps), 1.0)


def _log_obs_noise_scale(env: ManagerBasedRLEnv, scale: float) -> None:
    extras = getattr(env, "extras", None)
    if not isinstance(extras, dict):
        return
    log = extras.setdefault("log", {})
    log["Metrics/obs_noise_scale"] = float(scale)


def _add_scheduled_uniform_noise(
    env: ManagerBasedRLEnv,
    value: torch.Tensor,
    noise_range: Sequence[float],
    warmup_steps: int,
    ramp_steps: int,
) -> torch.Tensor:
    scale = _scheduled_noise_scale(env, warmup_steps, ramp_steps)
    _log_obs_noise_scale(env, scale)
    if scale <= 0.0:
        return value
    low, high = float(noise_range[0]) * scale, float(noise_range[1]) * scale
    if low == 0.0 and high == 0.0:
        return value
    return value + torch.empty_like(value).uniform_(low, high)


def noisy_joint_pos_rel(
    env: ManagerBasedRLEnv,
    noise_range: Sequence[float],
    warmup_steps: int,
    ramp_steps: int,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    value = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    return _add_scheduled_uniform_noise(env, value, noise_range, warmup_steps, ramp_steps)


def noisy_joint_pos(
    env: ManagerBasedRLEnv,
    noise_range: Sequence[float],
    warmup_steps: int,
    ramp_steps: int,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    value = asset.data.joint_pos[:, asset_cfg.joint_ids]
    return _add_scheduled_uniform_noise(env, value, noise_range, warmup_steps, ramp_steps)


def noisy_joint_vel_rel(
    env: ManagerBasedRLEnv,
    noise_range: Sequence[float],
    warmup_steps: int,
    ramp_steps: int,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    value = asset.data.joint_vel[:, asset_cfg.joint_ids] - asset.data.default_joint_vel[:, asset_cfg.joint_ids]
    return _add_scheduled_uniform_noise(env, value, noise_range, warmup_steps, ramp_steps)


def box_pos_xy_local(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("box"),
) -> torch.Tensor:
    return _box_xy_w(env, asset_cfg) - env.scene.env_origins[:, :2]


def box_vel_xy(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("box"),
) -> torch.Tensor:
    box: RigidObject = env.scene[asset_cfg.name]
    return box.data.root_lin_vel_w[:, :2]


def box_to_goal_xy(
    env: ManagerBasedRLEnv,
    goal_center: Sequence[float],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("box"),
) -> torch.Tensor:
    return _goal_center_xy_w(env, goal_center) - _box_xy_w(env, asset_cfg)


def _noisy_box_pos_w(
    env: ManagerBasedRLEnv,
    noise_range: Sequence[float],
    warmup_steps: int,
    ramp_steps: int,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("box"),
) -> torch.Tensor:
    """Return a per-step cached noisy box position shared by cube observation terms."""
    box: RigidObject = env.scene[asset_cfg.name]
    scale = _scheduled_noise_scale(env, warmup_steps, ramp_steps)
    _log_obs_noise_scale(env, scale)
    if scale <= 0.0:
        return box.data.root_pos_w

    low, high = float(noise_range[0]) * scale, float(noise_range[1]) * scale
    if low == 0.0 and high == 0.0:
        return box.data.root_pos_w

    step = int(getattr(env, "common_step_counter", 0))
    cache = getattr(env, "_box_push_noisy_box_pos_obs_cache", None)
    cache_key = (asset_cfg.name, step, low, high, warmup_steps, ramp_steps)
    if (
        cache is None
        or cache.get("key") != cache_key
        or cache.get("pos") is None
        or cache["pos"].shape != box.data.root_pos_w.shape
        or cache["pos"].device != box.data.root_pos_w.device
    ):
        noise = torch.empty_like(box.data.root_pos_w).uniform_(low, high)
        cache = {"key": cache_key, "pos": box.data.root_pos_w + noise}
        setattr(env, "_box_push_noisy_box_pos_obs_cache", cache)
    return cache["pos"]


def noisy_box_to_goal_xy(
    env: ManagerBasedRLEnv,
    goal_center: Sequence[float],
    noise_range: Sequence[float],
    warmup_steps: int,
    ramp_steps: int,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("box"),
) -> torch.Tensor:
    noisy_box_pos = _noisy_box_pos_w(env, noise_range, warmup_steps, ramp_steps, asset_cfg)
    return _goal_center_xy_w(env, goal_center) - noisy_box_pos[:, :2]


def goal_xy(
    env: ManagerBasedRLEnv,
    goal_center: Sequence[float],
) -> torch.Tensor:
    """Goal center XY in each environment's local frame."""
    return _goal_center_xy_w(env, goal_center) - env.scene.env_origins[:, :2]


def noisy_box_pose_local(
    env: ManagerBasedRLEnv,
    noise_range: Sequence[float],
    warmup_steps: int,
    ramp_steps: int,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("box"),
) -> torch.Tensor:
    """Noisy box position in each environment's local frame plus world orientation."""
    box: RigidObject = env.scene[asset_cfg.name]
    box_pos_local = _noisy_box_pos_w(env, noise_range, warmup_steps, ramp_steps, asset_cfg) - env.scene.env_origins
    return torch.cat([box_pos_local, box.data.root_quat_w], dim=1)


def nozzle_to_box(
    env: ManagerBasedRLEnv,
    airflow_term_name: str = "airflow",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("box"),
) -> torch.Tensor:
    box: RigidObject = env.scene[asset_cfg.name]
    airflow_term = env.action_manager.get_term(airflow_term_name)
    nozzle_pos, _ = airflow_term._nozzle_pose_w()

    return box.data.root_pos_w - nozzle_pos


def noisy_nozzle_to_box(
    env: ManagerBasedRLEnv,
    noise_range: Sequence[float],
    warmup_steps: int,
    ramp_steps: int,
    airflow_term_name: str = "airflow",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("box"),
) -> torch.Tensor:
    airflow_term = env.action_manager.get_term(airflow_term_name)
    nozzle_pos, _ = airflow_term._nozzle_pose_w()
    return _noisy_box_pos_w(env, noise_range, warmup_steps, ramp_steps, asset_cfg) - nozzle_pos


def nozzle_to_prepush_target(
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
    """Vector from the current nozzle pose to the desired pre-push nozzle pose."""
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
    return terms["desired_delta"]



def prepush_setup_features(
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
    """Compact setup features for learning when to move, aim, and blow.

    Returns:
        [desired_delta_xyz, push_dir_xy, position_reward, stand_alignment,
         axis_alignment, ready_gate]
        With ready_geometry, the three scalar scores are distance, jet-to-goal
        direction and box-center aim. The vector fields remain pose diagnostics.
    """
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
    return torch.cat(
        [
            terms["desired_delta"],
            terms["push_dir_xy"],
            terms["position_reward"].unsqueeze(1),
            (terms["direction_score"] if ready_geometry is not None else terms["stand_alignment"]).unsqueeze(1),
            (terms["aim_score"] if ready_geometry is not None else terms["axis_alignment"]).unsqueeze(1),
            terms["ready"].unsqueeze(1),
        ],
        dim=1,
    )



def nozzle_axis_w(
    env: ManagerBasedRLEnv,
    airflow_term_name: str = "airflow",
) -> torch.Tensor:
    """World-frame airflow axis observation, shape [num_envs, 3]."""
    _, _, axis_w = _airflow_nozzle_pose_and_axis_w(env, airflow_term_name)
    return axis_w


def nozzle_axis_xy(
    env: ManagerBasedRLEnv,
    airflow_term_name: str = "airflow",
) -> torch.Tensor:
    """Normalized XY projection of the airflow axis, shape [num_envs, 2]."""
    _, _, axis_w = _airflow_nozzle_pose_and_axis_w(env, airflow_term_name)
    axis_xy = axis_w[:, :2]
    return axis_xy / torch.linalg.norm(axis_xy, dim=1, keepdim=True).clamp_min(1e-6)


def nozzle_axis_features(
    env: ManagerBasedRLEnv,
    airflow_term_name: str = "airflow",
) -> torch.Tensor:
    """Compact direction features for the policy.

    Returns [axis_x, axis_y, axis_z, horizontal_fraction, downward_fraction].
    """
    _, _, axis_w = _airflow_nozzle_pose_and_axis_w(env, airflow_term_name)
    horizontal_fraction = torch.linalg.norm(axis_w[:, :2], dim=1, keepdim=True).clamp(0.0, 1.0)
    downward_fraction = torch.clamp(-axis_w[:, 2:3], min=0.0, max=1.0)
    return torch.cat([axis_w, horizontal_fraction, downward_fraction], dim=1)


def box_privileged_state(
    env: ManagerBasedRLEnv,
    goal_center: Sequence[float],
    goal_radius: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("box"),
) -> torch.Tensor:
    box: RigidObject = env.scene[asset_cfg.name]

    box_pos_local = box.data.root_pos_w - torch.cat(
        [
            env.scene.env_origins[:, :2],
            torch.zeros(env.num_envs, 1, device=env.device),
        ],
        dim=1,
    )

    goal_delta = box_to_goal_xy(env, goal_center, asset_cfg)
    radius = _goal_radius(env, goal_radius).unsqueeze(1)

    return torch.cat(
        [
            box_pos_local,
            box.data.root_quat_w,
            box.data.root_lin_vel_w,
            box.data.root_ang_vel_w,
            goal_delta,
            radius,
        ],
        dim=1,
    )


def box_goal_privileged_dynamics(
    env: ManagerBasedRLEnv,
    goal_center: Sequence[float],
    goal_radius: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("box"),
) -> torch.Tensor:
    """Exact box dynamics and goal-relative state for critic value estimation."""
    box: RigidObject = env.scene[asset_cfg.name]
    goal_delta = box_to_goal_xy(env, goal_center, asset_cfg)
    radius = _goal_radius(env, goal_radius).unsqueeze(1)
    return torch.cat([box.data.root_lin_vel_w, box.data.root_ang_vel_w, goal_delta, radius], dim=1)
