# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
# Underscore spellings are accepted too. Unmatched options fall through
# parse_known_args() into the hydra list instead of erroring, so a mistyped
# --disable_success_retreat used to be swallowed in silence and the retreat
# stayed on -- which looks exactly like the policy failing.
parser.add_argument(
    "--disable-success-retreat",
    "--disable_success_retreat",
    dest="disable_success_retreat",
    action="store_true",
    default=False,
    help="Disable scripted retreat to the box-push working posture after held success.",
)
parser.add_argument(
    "--retreat-duration-s",
    "--retreat_duration_s",
    dest="retreat_duration_s",
    type=float,
    default=2.0,
    help="Seconds to spend returning the arm to the box-push working posture after held success.",
)
parser.add_argument(
    "--retreat-trigger-hold-steps",
    "--retreat_trigger_hold_steps",
    dest="retreat_trigger_hold_steps",
    type=int,
    default=10,
    help="Consecutive successful environment steps before scripted retreat starts in play.",
)
parser.add_argument(
    "--retreat-trigger",
    "--retreat_trigger",
    dest="retreat_trigger",
    choices=("held", "inside"),
    default="held",
    help="What counts as success before the retreat starts. 'held' (default) waits for the"
    " task's own held-success signal: the pose goal kept AND the object at rest for the full"
    " required hold time. 'inside' is the old behaviour and fires on the instantaneous pose"
    " goal alone, which starts the retreat before the hold completes -- the arm then sweeps"
    " away while still blowing, and the policy is locked out before the success is banked.",
)
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import os
import time

import gymnasium as gym
import torch
from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper, export_policy_as_jit, export_policy_as_onnx
from isaaclab_rl.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import blower_arm_rl.tasks  # noqa: F401


def _action_term_slices(action_manager) -> dict[str, slice]:
    term_slices = {}
    start = 0
    for term_name, term_dim in zip(action_manager.active_terms, action_manager.action_term_dim):
        stop = start + term_dim
        term_slices[term_name] = slice(start, stop)
        start = stop
    return term_slices


def _smoothstep(progress: torch.Tensor) -> torch.Tensor:
    progress = torch.clamp(progress, 0.0, 1.0)
    return 2.0 * progress - progress * progress


def _cap_airflow_viz_env_count(env_cfg) -> None:
    airflow_cfg = getattr(getattr(env_cfg, "actions", None), "airflow", None)
    viz_cfg = getattr(airflow_cfg, "viz_cfg", None)
    scene_cfg = getattr(env_cfg, "scene", None)
    if viz_cfg is not None and scene_cfg is not None and hasattr(viz_cfg, "max_visualized_envs"):
        viz_cfg.max_visualized_envs = min(
            max(0, int(viz_cfg.max_visualized_envs)),
            max(0, int(scene_cfg.num_envs)),
        )


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Play with RSL-RL agent."""
    # grab task name for checkpoint path
    task_name = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Play", "")

    # override configurations with non-hydra CLI arguments
    agent_cfg: RslRlBaseRunnerCfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    _cap_airflow_viz_env_count(env_cfg)

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    if args_cli.disable_fabric:
        print("[WARN] --disable_fabric is ignored for this task; Fabric remains enabled for articulation control.")
    if (
        not args_cli.disable_success_retreat
        and hasattr(env_cfg, "box_push")
        and hasattr(env_cfg, "terminations")
        and hasattr(env_cfg.terminations, "success")
    ):
        env_cfg.terminations.success = None

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rsl_rl", train_task_name)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)

    # set the log directory for the environment (works for all environment types)
    env_cfg.log_dir = log_dir

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # load previously trained model
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    runner.load(resume_path)

    # obtain the trained policy for inference
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    # extract the neural network module
    # we do this in a try-except to maintain backwards compatibility.
    try:
        # version 2.3 onwards
        policy_nn = runner.alg.policy
    except AttributeError:
        # version 2.2 and below
        policy_nn = runner.alg.actor_critic

    # extract the normalizer
    if hasattr(policy_nn, "actor_obs_normalizer"):
        normalizer = policy_nn.actor_obs_normalizer
    elif hasattr(policy_nn, "student_obs_normalizer"):
        normalizer = policy_nn.student_obs_normalizer
    else:
        normalizer = None

    # export policy to onnx/jit
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    export_policy_as_jit(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.pt")
    export_policy_as_onnx(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.onnx")

    dt = env.unwrapped.step_dt
    term_slices = (
        _action_term_slices(env.unwrapped.action_manager)
        if hasattr(env.unwrapped, "action_manager")
        else {}
    )
    retreat_enabled = (
        not args_cli.disable_success_retreat
        and "arm_action" in term_slices
        and "airflow" in term_slices
    )
    arm_term = env.unwrapped.action_manager.get_term("arm_action") if retreat_enabled else None
    retreat_steps = max(int(args_cli.retreat_duration_s / dt), 1)
    retreat_trigger_hold_steps = max(int(args_cli.retreat_trigger_hold_steps), 1)
    retreat_state = torch.zeros(env.num_envs, dtype=torch.long, device=env.unwrapped.device)
    retreat_countdown = torch.zeros(env.num_envs, dtype=torch.long, device=env.unwrapped.device)
    retreat_trigger_hold_count = torch.zeros(env.num_envs, dtype=torch.long, device=env.unwrapped.device)
    airflow_action_dim = (
        term_slices["airflow"].stop - term_slices["airflow"].start
        if "airflow" in term_slices
        else 0
    )
    arm_target_dim = arm_term.action_dim if arm_term is not None else 0
    retreat_start_arm_targets = torch.zeros(env.num_envs, arm_target_dim, device=env.unwrapped.device)
    retreat_goal_arm_targets = (
        arm_term.zero_action_joint_targets.detach().clone()
        if arm_term is not None and hasattr(arm_term, "zero_action_joint_targets")
        else torch.zeros(env.num_envs, arm_target_dim, device=env.unwrapped.device)
    )
    retreat_start_airflow_actions = torch.zeros(env.num_envs, airflow_action_dim, device=env.unwrapped.device)

    # Waiting for held success means the object is already at rest and the fan
    # is already off by the time the arm moves, so the retreat cannot spoil the
    # very success it is reacting to.
    retreat_trigger_buffer = (
        "_box_push_goal_held_success" if args_cli.retreat_trigger == "held" else "_box_push_goal_inside"
    )
    if retreat_enabled:
        criterion = (
            "held the full pose goal at rest for the task's required hold time"
            if args_cli.retreat_trigger == "held"
            else "inside the goal (instantaneous pose only)"
        )
        print(
            f"[INFO] Success retreat is enabled: envs that have {criterion} for "
            f"{retreat_trigger_hold_steps} consecutive steps will ramp scripted arm joint targets "
            f"to the working posture over {args_cli.retreat_duration_s:.3f}s and airflow to zero."
        )

    # reset environment
    obs = env.get_observations()
    timestep = 0
    # simulate environment
    while simulation_app.is_running():
        start_time = time.time()
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            actions = policy(obs)
            if retreat_enabled:
                previous_actions = env.unwrapped.action_manager.action.detach()
                succeeded = getattr(env.unwrapped, retreat_trigger_buffer, None)
                if succeeded is not None:
                    succeeded = succeeded.to(device=env.unwrapped.device, dtype=torch.bool)
                    waiting = retreat_state == 0
                    retreat_trigger_hold_count[waiting & succeeded] += 1
                    retreat_trigger_hold_count[waiting & (~succeeded)] = 0
                    newly_successful = (
                        retreat_trigger_hold_count >= retreat_trigger_hold_steps
                    ) & waiting
                    if torch.any(newly_successful):
                        retreat_state[newly_successful] = 1
                        retreat_countdown[newly_successful] = retreat_steps
                        retreat_trigger_hold_count[newly_successful] = 0
                        if hasattr(arm_term, "seed_filter_from_joint_pos"):
                            arm_term.seed_filter_from_joint_pos(newly_successful)
                        retreat_start_arm_targets[newly_successful] = arm_term.processed_actions[
                            newly_successful
                        ].detach()
                        retreat_start_airflow_actions[newly_successful] = previous_actions[
                            newly_successful,
                            term_slices["airflow"],
                        ]

                    retreating = retreat_state == 1
                    if torch.any(retreating):
                        actions = actions.clone()

                        progress = (
                            retreat_steps - retreat_countdown[retreating]
                        ).float() / max(float(retreat_steps), 1.0)
                        blend = _smoothstep(progress).unsqueeze(1)
                        arm_targets = (
                            (1.0 - blend) * retreat_start_arm_targets[retreating]
                            + blend * retreat_goal_arm_targets[retreating]
                        )
                        # Apply a decaying upward bias on joint2 in the early steps
                        # to counteract gravity-induced droop before PD can hold position.
                        early_bias = 0.3 * torch.clamp(1.0 - progress * 1.0, 0.0, 1.0)
                        arm_targets[:, 1] = arm_targets[:, 1] - early_bias
                        arm_term.set_scripted_joint_targets(retreating, arm_targets)
                        actions[retreating, term_slices["arm_action"]] = 0.0
                        actions[retreating, term_slices["airflow"]] = (
                            (1.0 - blend) * retreat_start_airflow_actions[retreating]
                        )

                        retreat_countdown[retreating] -= 1
                        finished = retreating & (retreat_countdown <= 0)
                        retreat_state[finished] = 2

                    holding = retreat_state == 2
                    if torch.any(holding):
                        actions = actions.clone()
                        arm_term.set_scripted_joint_targets(holding, retreat_goal_arm_targets[holding])
                        actions[holding, term_slices["arm_action"]] = 0.0
                        actions[holding, term_slices["airflow"]] = 0.0

            # env stepping
            obs, _, dones, _ = env.step(actions)
            # reset recurrent states for episodes that have terminated
            policy_nn.reset(dones)
            if retreat_enabled:
                done_mask = dones.to(device=env.unwrapped.device, dtype=torch.bool)
                if torch.any(done_mask):
                    retreat_state[done_mask] = 0
                    retreat_countdown[done_mask] = 0
                    retreat_trigger_hold_count[done_mask] = 0
                    if arm_term is not None and hasattr(arm_term, "clear_scripted_joint_targets"):
                        arm_term.clear_scripted_joint_targets(done_mask)
        if args_cli.video:
            timestep += 1
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
