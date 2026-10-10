# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg




@configclass
class PiperBlowerBoxPushPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 64
    max_iterations = 5000
    save_interval = 100
    experiment_name = "piper_blower_box_push"
    run_name = ""
    obs_groups = {
        "policy": ["policy"],
        "critic": ["policy", "critic"],
    }
    clip_actions = 1.0
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.002,
        num_learning_epochs=6,
        num_mini_batches=8,
        learning_rate=3.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )




@configclass
class PiperBlowerGripperAssistBoxPushPPORunnerCfg(PiperBlowerBoxPushPPORunnerCfg):
    experiment_name = "piper_blower_gripper_assist_box_push"
    clip_actions = None
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        noise_std_type="log",
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=6,
        num_mini_batches=8,
        learning_rate=3.0e-4,
        schedule="adaptive",
        # At 30 Hz, extend the discounted horizon from ~3.3 s to ~6.7 s
        # for the sequence: align, build airflow, transport, settle.
        gamma=0.995,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class PiperBlowerGripperAssistCuboidOrientationPPORunnerCfg(PiperBlowerGripperAssistBoxPushPPORunnerCfg):
    experiment_name = "piper_blower_gripper_assist_cuboid_orientation"
    # Initial policy mean of the raw airflow action (train.py, fresh runs only). Force scales
    # with intensity^2 and the command is low-pass filtered, so a zero-mean start averages
    # ~0.3 intensity and never moves the cuboid. 1.5 averages ~0.83 (>=0.8 two thirds of the
    # time); the airflow penalty warmup then teaches turning it off when not aimed.
    airflow_action_init_bias: float = 1.5
