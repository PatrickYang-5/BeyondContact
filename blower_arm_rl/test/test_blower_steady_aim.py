"""Exercise actual tensor rewards on controlled nozzle/box states."""
import ast
from pathlib import Path
from types import SimpleNamespace as NS

import torch


def load_rewards(env):
    source = Path(__file__).resolve().parents[1] / 'source/blower_arm_rl/blower_arm_rl/tasks/manager_based/blower_arm_rl/mdp/box_push_reward.py'
    names = {'blower_horizontal_push_alignment_reward', 'blower_box_distance_reward'}
    nodes = [n for n in ast.parse(source.read_text()).body if isinstance(n, ast.FunctionDef) and n.name in names]
    def desired(env, **kwargs):
        p = env.scene['box'].data.root_pos_w.clone()
        p[:, 0] -= kwargs['nozzle_standoff']
        p[:, 2] = kwargs['desired_height']
        return p, torch.tensor([[1., 0.]]).expand(len(p), -1)
    ns = dict(torch=torch, ManagerBasedRLEnv=object, Sequence=list,
              SceneEntityCfg=lambda name: NS(name=name),
              _airflow_nozzle_pose_and_axis_w=lambda env, _: (env.nozzle, None, env.axis),
              _desired_nozzle_position_w=desired, _log_mean=lambda *args: None,
              nozzle_to_box=lambda env, *args: env.scene['box'].data.root_pos_w - env.nozzle)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), 'exec'), ns)
    return ns


def make_env():
    scene = {'robot': NS(data=NS(body_link_lin_vel_w=torch.zeros(1,1,3),
                                 body_link_ang_vel_w=torch.zeros(1,1,3),
                                 body_link_pos_w=torch.tensor([[[0.38,0,0.06]]]))),
             'box': NS(data=NS(root_pos_w=torch.tensor([[0.5,0,0.04]]), root_lin_vel_w=torch.zeros(1,3)))}
    class Scene(dict):
        env_origins = torch.zeros(1,3)
    return NS(scene=Scene(scene), nozzle=torch.tensor([[0.38,0,0.06]]), axis=torch.tensor([[1.,0,0]]),
              step_dt=0.1, episode_length_buf=torch.tensor([10]),
              action_manager=NS(get_term=lambda _: NS(processed_actions=torch.ones(1,1))))


def test_close_aligned_pose_beats_far_or_oblique_pose():
    env = make_env()
    f = load_rewards(env)['blower_horizontal_push_alignment_reward']
    kwargs = dict(goal_center=(1.,0,0), nozzle_standoff=0.12, position_std=0.05,
                  desired_height=0.06, aim_height=0.06, height_std=0.035,
                  position_blend=0.15, alignment_power=8)
    aligned = f(env, **kwargs).item()
    env.axis[:] = torch.tensor([[2**-0.5,2**-0.5,0]])
    oblique = f(env, **kwargs).item()
    env.axis[:] = torch.tensor([[1.,0,0]])
    env.nozzle[:,0] = 0.20
    far = f(env, **kwargs).item()
    assert aligned > 0.99 and oblique < 0.2 and far < 0.01

