"""CPU tests for total clipping order, scaling, stateful evaluation and reset."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1] / 'source/blower_arm_rl/blower_arm_rl/tasks/manager_based/piper_blower_rl'
spec = importlib.util.spec_from_file_location('reward_clipping', ROOT / 'reward_clipping.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
PositiveRewardManager = module.PositiveRewardManager


class Manager:
    active_terms = ['tracking', 'acceleration', 'box_out_of_workspace']

    def __init__(self):
        self._step_reward = torch.tensor([[2., -3., 0.], [2., -0.5, 0.], [2., -3., -4.], [2., -0.5, -4.]])
        self.calls = 0
        self.reset_ids = None

    def compute(self, dt):
        self.calls += 1
        return self._step_reward.sum(dim=1) * dt

    def reset(self, ids):
        self.reset_ids = ids
        return {'original_term_log': torch.tensor(1.)}


@pytest.fixture
def setup():
    base = Manager()
    env = SimpleNamespace(num_envs=4, device='cpu', extras={}, max_episode_length_s=10.)
    return base, env, PositiveRewardManager(base, env, ('box_out_of_workspace',))


@pytest.mark.parametrize('dt', [1., 1/30])
def test_clip_sum_then_add_failure_with_dt_once(setup, dt):
    base, env, reward = setup
    actual = reward.compute(dt)
    torch.testing.assert_close(actual, torch.tensor([0., 1.5, -4., -2.5]) * dt)
    assert base.calls == 1  # Do not recompute excluded/stateful rewards.
    assert env.extras['log']['Metrics/reward_clip_fraction'] == 0.5
    torch.testing.assert_close(base._step_reward[:, 1], torch.tensor([-3., -0.5, -3., -0.5]))


def test_no_exclusions_clips_entire_sum(setup):
    base, env, _ = setup
    reward = PositiveRewardManager(base, env, ())
    torch.testing.assert_close(reward.compute(1.), torch.tensor([0., 1.5, 0., 0.]))


def test_partial_and_full_reset_log_real_reward(setup):
    base, _, reward = setup
    reward.compute(1.)
    ids = torch.tensor([0, 2])
    logs = reward.reset(ids)
    assert base.reset_ids is ids
    assert logs['original_term_log'] == 1
    torch.testing.assert_close(logs['Episode_Reward/positive_clip_correction'], torch.tensor(0.1))
    torch.testing.assert_close(logs['Episode_Reward/total_after_clipping'], torch.tensor(-0.2))
    torch.testing.assert_close(reward._total_sum, torch.tensor([0., 1.5, 0., -2.5]))
    reward.reset()
    assert base.reset_ids is None
    assert reward._total_sum.count_nonzero() == 0


def test_invalid_term_rejected(setup):
    base, env, _ = setup
    with pytest.raises(ValueError, match='Unknown unclipped'):
        PositiveRewardManager(base, env, ('typo',))


def test_duplicate_exclusion_is_not_double_counted(setup):
    base, env, _ = setup
    reward = PositiveRewardManager(base, env, ('box_out_of_workspace',) * 2)
    torch.testing.assert_close(reward.compute(1.), torch.tensor([0., 1.5, -4., -2.5]))


def test_environment_switch_preserves_base_manager_when_disabled(setup):
    import ast
    base, env, _ = setup
    class BaseEnv:
        def load_managers(self):
            self.reward_manager = base
    tree = ast.parse((ROOT / 'positive_reward_env.py').read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef))
    ns = {'ManagerBasedRLEnv': BaseEnv, 'PositiveRewardManager': PositiveRewardManager}
    exec(compile(ast.Module(body=[cls], type_ignores=[]), 'positive_reward_env.py', 'exec'), ns)
    wrapper = ns['PiperPositiveRewardEnv']()
    wrapper.__dict__.update(env.__dict__)
    wrapper.cfg = SimpleNamespace(only_positive_rewards=False, unclipped_reward_terms=('box_out_of_workspace',))
    wrapper.load_managers()
    assert wrapper.reward_manager is base
    wrapper.cfg.only_positive_rewards = True
    wrapper.load_managers()
    assert isinstance(wrapper.reward_manager, PositiveRewardManager)
