# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Piper-L reach environment used as the base scene for blower tasks."""

import math

from isaaclab.utils import configclass

import isaaclab_tasks.manager_based.manipulation.reach.mdp as mdp
from isaaclab_tasks.manager_based.manipulation.reach.reach_env_cfg import ReachEnvCfg

from .piper_l_cfg import PIPER_ARM_JOINTS, PIPER_END_EFFECTOR_BODY, PIPER_L_CFG


@configclass
class PiperReachEnvCfg(ReachEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.robot = PIPER_L_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        # Position every environment first, then create each Piper's world
        # fixed joint at that environment's own table origin.
        self.scene.replicate_physics = True

        self.rewards.end_effector_position_tracking.params["asset_cfg"].body_names = [PIPER_END_EFFECTOR_BODY]
        self.rewards.end_effector_position_tracking_fine_grained.params["asset_cfg"].body_names = [
            PIPER_END_EFFECTOR_BODY
        ]
        self.rewards.end_effector_orientation_tracking.params["asset_cfg"].body_names = [PIPER_END_EFFECTOR_BODY]

        self.actions.arm_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=PIPER_ARM_JOINTS,
            scale=0.35,
            use_default_offset=True,
        )

        self.commands.ee_pose.body_name = PIPER_END_EFFECTOR_BODY
        self.commands.ee_pose.ranges.pos_x = (0.18, 0.55)
        self.commands.ee_pose.ranges.pos_y = (-0.20, 0.20)
        self.commands.ee_pose.ranges.pos_z = (0.15, 0.55)
        self.commands.ee_pose.ranges.pitch = (-math.pi, math.pi)


@configclass
class PiperReachEnvCfg_PLAY(PiperReachEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
