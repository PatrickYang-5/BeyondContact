# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Piper-L gripper-assisted translation and rectangular reorientation."""
import gymnasium as gym
from . import agents

for task, module, cfg in (
    ("GripperAssistBoxPush", "blower_gripper_assist_box_push_cfg", "PiperBlowerGripperAssistBoxPush"),
    ("GripperAssistCuboidOrientation", "blower_gripper_assist_cuboid_orientation_cfg", "PiperBlowerGripperAssistCuboidOrientation"),
):
    for suffix, cfg_suffix in (("", ""), ("-Play", "_PLAY")):
        gym.register(
            id=f"Isaac-Blower-Piper-{task}{suffix}-v0",
            entry_point=f"{__name__}.positive_reward_env:PiperPositiveRewardEnv",
            disable_env_checker=True,
            kwargs={
                "env_cfg_entry_point": f"{__name__}.{module}:{cfg}EnvCfg{cfg_suffix}",
                "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:{cfg}PPORunnerCfg",
            },
        )
