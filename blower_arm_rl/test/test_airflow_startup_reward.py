"""Behavioral checks for starting airflow and separating fan/arm regularization."""
import ast
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
import torch


@pytest.fixture
def setup():
    source = Path(__file__).resolve().parents[1] / 'source/blower_arm_rl/blower_arm_rl/tasks/manager_based/blower_arm_rl/mdp/box_push_reward.py'
    names = {'AirflowDirectedForceReward', 'action_term_rate_l2'}
    nodes = [n for n in ast.parse(source.read_text()).body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names]
    class Base:
        def __init__(self, cfg, env):
            pass
    ns = dict(torch=torch, ManagerBasedRLEnv=object, ManagerTermBase=Base, Sequence=list,
              SceneEntityCfg=lambda name: NS(name=name), _log_mean=lambda *args: None,
              box_to_goal_xy=lambda env, *args: env.goal_delta,
              _box_fully_inside_goal=lambda env, *args: env.inside)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), 'exec'), ns)
    force = torch.tensor([[0.8, 0., 0.], [0.4, 0., 0.]])
    airflow = NS(processed_actions=torch.ones(2, 1), target_force_w=lambda _: force)
    env = NS(num_envs=2, device='cpu', step_dt=0.3, scene={'box': object()},
             goal_delta=torch.tensor([[1., 0.], [1., 0.]]), inside=torch.zeros(2, dtype=torch.bool),
             action_manager=NS(get_term=lambda _: airflow))
    return ns, env, airflow, force


def test_credit_without_box_motion_but_not_without_directed_force(setup):
    ns, env, airflow, force = setup
    f = ns['AirflowDirectedForceReward'](None, env)
    params = dict(goal_center=(1.,0.,0.), goal_radius=0.1)
    torch.testing.assert_close(f(env, **params), torch.tensor([1., .5]))
    force[:] = torch.tensor([[-1., 0., 0.], [0., 1., 0.]])
    torch.testing.assert_close(f(env, **params), torch.zeros(2))
    force[:, 0] = 1.
    airflow.processed_actions.zero_()
    torch.testing.assert_close(f(env, **params), torch.zeros(2))


@pytest.mark.parametrize('dt', [1/30, 0.3])
def test_integral_cap_goal_mask_and_partial_reset(setup, dt):
    ns, env, _, _ = setup
    env.step_dt = dt
    f = ns['AirflowDirectedForceReward'](None, env)
    params = dict(goal_center=(1.,0.,0.), goal_radius=0.1)
    total = torch.zeros(2)
    for _ in range(100):
        total += f(env, **params) * dt
    torch.testing.assert_close(total, torch.ones(2))
    torch.testing.assert_close(f(env, **params), torch.zeros(2))
    f.reset(torch.tensor([0]))
    env.inside[0] = True
    torch.testing.assert_close(f(env, **params), torch.zeros(2))
    assert f._used_credit[0] == 0  # No budget consumed inside the goal.
    env.inside[0] = False
    torch.testing.assert_close(f(env, **params), torch.tensor([1., 0.]))
    f.reset()
    torch.testing.assert_close(f._used_credit, torch.zeros(2))


def test_arm_rate_uses_named_slice_not_action_order(setup):
    ns, env, _, _ = setup
    # Fan is deliberately placed first. A large fan change must not affect arm cost.
    env.action_manager = NS(active_terms=['airflow', 'arm_action'], action_term_dim=[1, 2],
                            action=torch.tensor([[99., 3., 4.], [-99., 0., 0.]]),
                            prev_action=torch.zeros(2, 3))
    torch.testing.assert_close(ns['action_term_rate_l2'](env), torch.tensor([25., 0.]))
    with pytest.raises(ValueError):
        ns['action_term_rate_l2'](env, 'missing')
