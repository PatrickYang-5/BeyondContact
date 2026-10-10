# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Arm action terms for the airflow box-push task."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from isaaclab.envs.mdp.actions.actions_cfg import JointPositionActionCfg
from isaaclab.envs.mdp.actions.joint_actions import JointPositionAction
from isaaclab.utils import configclass


class SmoothedJointPositionAction(JointPositionAction):
    """Joint-position action with a first-order low-pass filter on position targets."""

    cfg: "SmoothedJointPositionActionCfg"

    def __init__(self, cfg: "SmoothedJointPositionActionCfg", env):
        super().__init__(cfg, env)
        self._target_processed_actions = torch.zeros_like(self._processed_actions)
        self._previous_processed_actions = torch.zeros_like(self._processed_actions)
        self._processed_velocity_targets = torch.zeros_like(self._processed_actions)
        self._scripted_joint_targets = torch.zeros_like(self._processed_actions)
        self._scripted_joint_target_mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    @property
    def zero_action_joint_targets(self) -> torch.Tensor:
        """Joint-position targets produced by a zero raw action."""
        targets = torch.zeros_like(self._raw_actions) * self._scale + self._offset
        if self.cfg.clip is not None:
            targets = torch.clamp(
                targets,
                min=self._clip[:, :, 0],
                max=self._clip[:, :, 1],
            )
        return targets

    def set_scripted_joint_targets(
        self,
        env_ids: Sequence[int] | torch.Tensor,
        joint_targets: torch.Tensor,
    ) -> None:
        """Override processed joint-position targets for selected environments."""
        self._scripted_joint_targets[env_ids] = joint_targets.to(self.device)
        self._scripted_joint_target_mask[env_ids] = True

    def clear_scripted_joint_targets(self, env_ids: Sequence[int] | torch.Tensor | None = None) -> None:
        if env_ids is None:
            self._scripted_joint_target_mask[:] = False
        else:
            self._scripted_joint_target_mask[env_ids] = False

    def seed_filter_from_joint_pos(self, env_ids: Sequence[int] | torch.Tensor) -> None:
        """Seed the low-pass filter from actual joint positions for a smooth mode switch.

        Call this just before activating scripted mode so that the filter history is
        aligned with where the arm physically is, preventing PD-controller jolt from
        a mismatch between the filtered target and the actual position.
        """
        current_pos = self._asset.data.joint_pos
        if not isinstance(self._joint_ids, slice):
            current_pos = current_pos[:, self._joint_ids]
        pos = current_pos[env_ids].clone()
        self._processed_actions[env_ids] = pos
        self._previous_processed_actions[env_ids] = pos
        self._target_processed_actions[env_ids] = pos
        self._processed_velocity_targets[env_ids] = 0.0

    def process_actions(self, actions: torch.Tensor):
        self._raw_actions[:] = actions.to(self.device)

        target_actions = self._raw_actions * self._scale + self._offset
        if self.cfg.clip is not None:
            target_actions = torch.clamp(
                target_actions,
                min=self._clip[:, :, 0],
                max=self._clip[:, :, 1],
            )
        self._target_processed_actions[:] = target_actions
        self._previous_processed_actions[:] = self._processed_actions

        if self.cfg.smoothing_tau_s > 0.0:
            alpha = self._env.step_dt / (self.cfg.smoothing_tau_s + self._env.step_dt)
            next_actions = (
                alpha * self._target_processed_actions
                + (1.0 - alpha) * self._previous_processed_actions
            )
        else:
            next_actions = self._target_processed_actions

        if self.cfg.max_joint_step_rad > 0.0:
            delta = torch.clamp(
                next_actions - self._previous_processed_actions,
                min=-self.cfg.max_joint_step_rad,
                max=self.cfg.max_joint_step_rad,
            )
            next_actions = self._previous_processed_actions + delta
        self._processed_actions[:] = next_actions

        if torch.any(self._scripted_joint_target_mask):
            mask = self._scripted_joint_target_mask
            if self.cfg.smoothing_tau_s > 0.0:
                alpha = self._env.step_dt / (self.cfg.smoothing_tau_s + self._env.step_dt)
                scripted_next_actions = (
                    alpha * self._scripted_joint_targets[mask]
                    + (1.0 - alpha) * self._previous_processed_actions[mask]
                )
            else:
                scripted_next_actions = self._scripted_joint_targets[mask]
            if self.cfg.max_joint_step_rad > 0.0:
                scripted_delta = torch.clamp(
                    scripted_next_actions - self._previous_processed_actions[mask],
                    min=-self.cfg.max_joint_step_rad,
                    max=self.cfg.max_joint_step_rad,
                )
                scripted_next_actions = self._previous_processed_actions[mask] + scripted_delta
            self._processed_actions[mask] = scripted_next_actions
            self._target_processed_actions[mask] = self._scripted_joint_targets[mask]

        dt = max(float(self._env.step_dt), 1e-6)
        self._processed_velocity_targets[:] = (self._processed_actions - self._previous_processed_actions) / dt

    def apply_actions(self):
        self._asset.set_joint_position_target(self._processed_actions, joint_ids=self._joint_ids)
        self._asset.set_joint_velocity_target(self._processed_velocity_targets, joint_ids=self._joint_ids)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        super().reset(env_ids)

        current_joint_pos = self._asset.data.joint_pos[env_ids]
        if not isinstance(self._joint_ids, slice):
            current_joint_pos = current_joint_pos[:, self._joint_ids]
        self._target_processed_actions[env_ids] = current_joint_pos
        self._previous_processed_actions[env_ids] = current_joint_pos
        self._processed_actions[env_ids] = current_joint_pos
        self._processed_velocity_targets[env_ids] = 0.0
        self.clear_scripted_joint_targets(env_ids)


@configclass
class SmoothedJointPositionActionCfg(JointPositionActionCfg):
    class_type: type = SmoothedJointPositionAction

    smoothing_tau_s: float = 0.12
    """First-order low-pass filter time constant in seconds. Set to 0 to disable smoothing."""

    max_joint_step_rad: float = 0.0
    """Maximum per-step joint target change in radians. Set to 0 to disable."""
