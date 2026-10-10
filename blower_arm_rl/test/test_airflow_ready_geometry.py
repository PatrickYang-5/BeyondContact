"""Independent geometry and shared readiness integration, without Isaac Sim."""
import ast
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1] / 'source/blower_arm_rl/blower_arm_rl/tasks/manager_based'
GEOMETRY = dict(distance=0.12, distance_std=0.08, direction_std_deg=30., aim_std_deg=15.)


@pytest.fixture
def namespace():
    source = ROOT / 'blower_arm_rl/mdp/box_push_common.py'
    names = {'_distance_direction_aim_terms', '_prepush_setup_terms'}
    nodes = [n for n in ast.parse(source.read_text()).body if isinstance(n, ast.FunctionDef) and n.name in names]
    def desired(env, **kwargs):
        p = env.scene['box'].data.root_pos_w.clone()
        p[:, :2] -= kwargs['nozzle_standoff'] * env.goal_dir
        p[:, 2] = kwargs['desired_height']
        return p, env.goal_dir
    ns = dict(torch=torch, ManagerBasedRLEnv=object, Sequence=list, SceneEntityCfg=object,
              _log_mean=lambda *args: None, _desired_nozzle_position_w=desired,
              _airflow_nozzle_pose_and_axis_w=lambda env, _: (env.nozzle, None, env.axis))
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), 'exec'), ns)
    return ns


def evaluate(ns, box=(.12, 0., 0.), axis=(1., 0., 0.), goal=(1., 0.)):
    return ns['_distance_direction_aim_terms'](
        torch.zeros(1, 3), torch.tensor([axis]), torch.tensor([box]), torch.tensor([goal]), **GEOMETRY)


def test_perfect_pose_and_distance_are_independent(namespace):
    perfect = evaluate(namespace)
    assert perfect['ready'].item() == pytest.approx(1.)
    far = evaluate(namespace, box=(.28, 0., 0.))
    assert far['position_score'].item() == pytest.approx(float(torch.exp(torch.tensor(-4.))))
    assert far['aim_score'].item() == far['direction_score'].item() == 1.


def test_aimed_at_box_but_wrong_push_direction(namespace):
    r = evaluate(namespace, box=(0., .12, 0.), axis=(0., 1., 0.))
    assert r['position_score'].item() == r['aim_score'].item() == 1.
    assert r['direction_score'].item() == r['ready'].item() == 0.


def test_goalward_jet_that_misses_box(namespace):
    r = evaluate(namespace, box=(0., .12, 0.))
    assert r['position_score'].item() == r['direction_score'].item() == 1.
    assert r['aim_score'].item() == r['ready'].item() == 0.


def test_angles_are_in_degrees_not_cosine_error(namespace):
    theta = torch.tensor(torch.pi / 12)
    r = evaluate(namespace, axis=(float(theta.cos()), float(theta.sin()), 0.))
    assert r['aim_score'].item() == pytest.approx(float(torch.exp(torch.tensor(-1.))), rel=1e-5)
    assert r['direction_score'].item() == pytest.approx(float(torch.exp(torch.tensor(-.25))), rel=1e-5)


@pytest.mark.parametrize('axis,box,goal', [
    ((-1.,0.,0.), (.12,0.,0.), (1.,0.)),
    ((0.,0.,1.), (.12,0.,0.), (1.,0.)),
    ((0.,0.,0.), (.12,0.,0.), (1.,0.)),
    ((1.,0.,0.), (0.,0.,0.), (1.,0.)),
    ((1.,0.,0.), (.12,0.,0.), (0.,0.)),
])
def test_reverse_vertical_and_degenerate_geometry_cannot_be_ready(namespace, axis, box, goal):
    result = evaluate(namespace, box=box, axis=axis, goal=goal)
    assert all(torch.isfinite(v).all() for v in result.values())
    assert result['ready'].item() == 0.


def test_new_mode_has_no_fixed_world_height_or_goal_axis(namespace):
    f = namespace['_distance_direction_aim_terms']
    nozzle = torch.tensor([[3., 2., 5.], [0., 0., 0.]])
    box = nozzle + torch.tensor([[0., .12, 0.], [.12, 0., 0.]])
    result = f(nozzle, torch.tensor([[0.,1.,0.], [1.,0.,0.]]), box,
               torch.tensor([[0.,1.], [1.,0.]]), **GEOMETRY)
    torch.testing.assert_close(result['ready'], torch.ones(2))


def test_shared_helper_opt_in_and_legacy_compatibility(namespace):
    class Scene(dict):
        env_origins = torch.zeros(1,3)
    env = NS(scene=Scene(box=NS(data=NS(root_pos_w=torch.tensor([[.12,0.,.06]])))),
             nozzle=torch.tensor([[0.,0.,.06]]), axis=torch.tensor([[1.,0.,0.]]),
             goal_dir=torch.tensor([[1.,0.]]))
    params = dict(goal_center=(1.,0.,0.), nozzle_standoff=.12, nozzle_height=.06,
                  box_half_extent=.04, desired_height=.06, position_std=.08,
                  height_std=.05, axis_std=.15, airflow_term_name='airflow', asset_cfg=NS(name='box'))
    f = namespace['_prepush_setup_terms']
    original = f(env, **params)
    assert original['ready'].item() == pytest.approx(1.)
    # Raise nozzle and box together; new readiness is unchanged, old pose loses height credit.
    env.nozzle[:,2] += .20
    env.scene['box'].data.root_pos_w[:,2] += .20
    new = f(env, **params, ready_geometry=GEOMETRY)
    assert new['ready'].item() == pytest.approx(1.)
    assert f(env, **params)['ready'].item() < 1e-6


def test_config_geometry_reaches_both_rewards_and_observation(namespace):
    ns = namespace
    ns['SceneEntityCfg'] = lambda name: NS(name=name)
    ns['box_to_goal_xy'] = lambda env, *args: env.goal_dir
    for filename, names in [
        ('blower_arm_rl/mdp/box_push_reward.py', {'airflow_before_ready_penalty',
         'box_velocity_toward_goal_with_airflow_reward', '_global_step_penalty_scale', '_airflow_usage_gate'}),
        ('blower_arm_rl/mdp/box_push_observation.py', {'prepush_setup_features'}),
        ('piper_blower_rl/blower_box_push_env_cfg.py', {'_prepush_term_params'}),
    ]:
        source = ROOT / filename
        ns['BoxPushTaskParamsCfg'] = object
        nodes = [n for n in ast.parse(source.read_text()).body if isinstance(n, ast.FunctionDef) and n.name in names]
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), 'exec'), ns)
    task = NS(geometry=NS(goal_center=(1.,0.,0.), box_half_extent=.04, box_aim_height=.06),
              prepush=NS(nozzle_standoff=.12, nozzle_height=.06, position_std=.08,
                         height_std=.05, axis_std=.15, ready_geometry=GEOMETRY))
    params = ns['_prepush_term_params'](task)
    class Scene(dict):
        env_origins = torch.zeros(1,3)
    # Elevated perfect geometry separates the new metric from the legacy height target.
    box = NS(data=NS(root_pos_w=torch.tensor([[.12,0.,.5]]), root_lin_vel_w=torch.tensor([[.1,0.,0.]])))
    env = NS(scene=Scene(box=box), nozzle=torch.tensor([[0.,0.,.5]]), axis=torch.tensor([[1.,0.,0.]]),
             goal_dir=torch.tensor([[1.,0.]]), device='cpu', common_step_counter=100,
             action_manager=NS(get_term=lambda _: NS(processed_actions=torch.ones(1,1))))
    assert ns['airflow_before_ready_penalty'](env, **params).item() == pytest.approx(0.)
    assert ns['box_velocity_toward_goal_with_airflow_reward'](env, **params).item() == pytest.approx(.1)
    features = ns['prepush_setup_features'](env, **params)
    assert features.shape == (1,9)
    torch.testing.assert_close(features[:,5:], torch.ones(1,4))
    env.axis *= -1
    assert ns['airflow_before_ready_penalty'](env, **params).item() == pytest.approx(1.)
    assert ns['box_velocity_toward_goal_with_airflow_reward'](env, **params).item() == pytest.approx(0.)
