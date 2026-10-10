"""CPU tests of the reward's tensor logic, without launching Isaac Sim."""
import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


@pytest.fixture
def reward_env():
    # Load the real class without importing the simulator-dependent mdp package.
    source = Path(__file__).resolve().parents[1] / "source/blower_arm_rl/blower_arm_rl/tasks/manager_based/blower_arm_rl/mdp/box_push_reward.py"
    tree = ast.parse(source.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "EndEffectorAccelerationPenalty")
    scale_node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_global_step_penalty_scale")
    class Base:
        def __init__(self, cfg, env):
            pass
    def rotate(q, v):
        t = 2 * torch.cross(q[:, 1:], v, dim=-1)
        return v + q[:, :1] * t + torch.cross(q[:, 1:], t, dim=-1)
    namespace = dict(torch=torch, ManagerTermBase=Base, ManagerBasedRLEnv=object,
                     SceneEntityCfg=object, quat_apply=rotate, _log_mean=lambda *args: None)
    exec(compile(ast.Module(body=[scale_node, node], type_ignores=[]), str(source), "exec"), namespace)
    data = SimpleNamespace(
        body_link_lin_vel_w=torch.zeros(2, 1, 3),
        body_link_ang_vel_w=torch.zeros(2, 1, 3),
        body_link_quat_w=torch.tensor([[[1., 0, 0, 0]]] * 2),
    )
    env = SimpleNamespace(num_envs=2, device="cpu", step_dt=0.1,
                          episode_length_buf=torch.ones(2, dtype=torch.long) * 10,
                          scene={"robot": SimpleNamespace(data=data)})
    reward = namespace["EndEffectorAccelerationPenalty"](None, env)
    cfg = SimpleNamespace(name="robot", body_ids=[0])
    return reward, env, data, cfg


def test_acceleration_l2_sum_across_axes(reward_env):
    reward, env, data, cfg = reward_env
    reward(env, cfg)
    data.body_link_lin_vel_w[:, 0] = torch.tensor([[0.3, 0.4, 0.0], [0.1, 0.2, 0.2]])
    # Accelerations [3,4,0] and [1,2,2] give squared norms 25 and 9.
    torch.testing.assert_close(reward(env, cfg), torch.tensor([25., 9.]))
    torch.testing.assert_close(reward(env, cfg), torch.zeros(2))  # Constant velocity.


def test_partial_reset_ignores_teleport_only_for_reset_env(reward_env):
    reward, env, data, cfg = reward_env
    reward(env, cfg)
    reward.reset(torch.tensor([0]))
    data.body_link_lin_vel_w[:, 0, 0] = torch.tensor([100., 1.])
    torch.testing.assert_close(reward(env, cfg), torch.tensor([0., 100.]))
    reward.reset()
    data.body_link_lin_vel_w += 100
    torch.testing.assert_close(reward(env, cfg), torch.zeros(2))


def test_reset_settling_updates_history_without_penalty(reward_env):
    reward, env, data, cfg = reward_env
    env.episode_length_buf[:] = 0
    reward(env, cfg)
    data.body_link_lin_vel_w[:] = 10
    env.episode_length_buf[:] = 1
    torch.testing.assert_close(reward(env, cfg), torch.zeros(2))
    env.episode_length_buf[:] = 2
    torch.testing.assert_close(reward(env, cfg), torch.zeros(2))


def test_rotating_offset_captures_centripetal_acceleration(reward_env):
    reward, env, data, cfg = reward_env
    data.body_link_ang_vel_w[:, 0, 2] = 1
    reward(env, cfg, point_offset=(1., 0, 0))
    # A 90-degree rotation changes point velocity while link-origin velocity stays zero.
    data.body_link_quat_w[:] = torch.tensor([2**-0.5, 0, 0, 2**-0.5])
    expected = torch.full((2,), 2 / 0.1**2)
    torch.testing.assert_close(reward(env, cfg, point_offset=(1., 0, 0)), expected)


def test_small_acceleration_is_also_penalized(reward_env):
    reward, env, data, cfg = reward_env
    reward(env, cfg)
    data.body_link_lin_vel_w[:, 0, 0] = 0.01
    torch.testing.assert_close(reward(env, cfg), torch.full((2,), 0.01))


def test_global_ramp_preserves_velocity_history_and_l2(reward_env):
    reward, env, data, cfg = reward_env
    params = dict(penalty_warmup_steps=100, penalty_ramp_steps=200, penalty_start_scale=0.1)
    reward(env, cfg, **params)
    for step, scale in [(0, 0.1), (100, 0.1), (200, 0.55), (300, 1.0), (500, 1.0)]:
        env.common_step_counter = step
        data.body_link_lin_vel_w[:, 0, 0] += 0.2
        # Each velocity increment is 2 m/s^2, even across the warmup boundary.
        torch.testing.assert_close(reward(env, cfg, **params), torch.full((2,), 4 * scale))
