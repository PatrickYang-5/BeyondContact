"""Check useful-flow credit independently of the simulator, using actual reward code."""
import ast
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
import torch


@pytest.fixture
def setup():
    mdp = Path(__file__).resolve().parents[1] / 'source/blower_arm_rl/blower_arm_rl/tasks/manager_based/blower_arm_rl/mdp'
    source = mdp / 'box_push_reward.py'
    names = {'box_velocity_toward_goal_with_airflow_reward', '_airflow_usage_gate'}
    nodes = [n for n in ast.parse(source.read_text()).body if isinstance(n, ast.FunctionDef) and n.name in names]
    airflow_tree = ast.parse((mdp / 'airflow.py').read_text())
    cls = next(n for n in airflow_tree.body if isinstance(n, ast.ClassDef) and n.name == 'SmoothedAirflowActionTerm')
    accessor = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'target_force_w')
    ns = dict(torch=torch, ManagerBasedRLEnv=object, Sequence=list,
              SceneEntityCfg=lambda name: NS(name=name), _log_mean=lambda *args: None,
              box_to_goal_xy=lambda env, *args: torch.tensor([[1., 0.]]),
              _prepush_setup_terms=lambda env, **kwargs: {'ready': env.ready})
    exec(compile(ast.Module(body=nodes + [accessor], type_ignores=[]), str(source), 'exec'), ns)
    box = NS(data=NS(root_lin_vel_w=torch.tensor([[0.1, 0., 0.]])))
    force = torch.tensor([[[0.1, 0., 0.]]])
    airflow = NS(processed_actions=torch.ones(1, 1), _target_states=[{'asset': box, 'forces': force}])
    airflow.target_force_w = lambda asset: ns['target_force_w'](airflow, asset)
    env = NS(scene={'box': box}, ready=torch.ones(1), action_manager=NS(get_term=lambda _: airflow))
    params = dict(goal_center=(1., 0., 0.), nozzle_standoff=0.12, nozzle_height=0.06,
                  box_half_extent=0.04, desired_height=0.06, position_std=0.08,
                  height_std=0.05, axis_std=0.15, min_airflow=0.05,
                  ready_reward_floor=0.5, force_scale_n=0.1)
    return ns['box_velocity_toward_goal_with_airflow_reward'], env, airflow, force, params


@pytest.mark.parametrize('force_x, expected', [(0., 0.), (-0.1, 0.), (0.05, 0.05), (0.1, 0.1), (10., 0.1)])
def test_force_projection_and_saturation(setup, force_x, expected):
    f, env, airflow, force, params = setup
    force[:, :, 0] = force_x
    torch.testing.assert_close(f(env, **params), torch.tensor([expected]))


def test_no_credit_for_fan_off_stationary_or_wrong_way(setup):
    f, env, airflow, force, params = setup
    airflow.processed_actions.zero_()
    assert f(env, **params).item() == 0
    airflow.processed_actions.fill_(1)
    for speed in [0., -0.1]:
        env.scene['box'].data.root_lin_vel_w[:, 0] = speed
        assert f(env, **params).item() == 0


def test_exploration_floor_and_legacy_opt_out(setup):
    f, env, airflow, force, params = setup
    env.ready.zero_()
    torch.testing.assert_close(f(env, **params), torch.tensor([0.05]))
    force.zero_()
    params['force_scale_n'] = 0.
    torch.testing.assert_close(f(env, **params), torch.tensor([0.05]))


def test_accessor_rejects_unconfigured_target(setup):
    _, _, airflow, _, _ = setup
    with pytest.raises(ValueError, match='not an airflow target'):
        airflow.target_force_w(object())
