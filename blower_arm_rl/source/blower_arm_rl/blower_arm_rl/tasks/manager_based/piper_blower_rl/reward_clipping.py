"""Reward-manager adapter for legged-gym-style positive ordinary rewards."""

import torch


class PositiveRewardManager:
    """Keep the base manager's term evaluation/reset and clip only ordinary totals.

    Excluded terms are read from the already weighted per-step buffer, so stateful
    reward terms execute exactly once. Individual term logs remain unclipped.
    """

    def __init__(self, manager, env, unclipped_terms):
        self._manager = manager
        self._env = env
        names = manager.active_terms
        missing = set(unclipped_terms) - set(names)
        if missing:
            raise ValueError(f"Unknown unclipped reward terms: {sorted(missing)}")
        self._unclipped_indices = [names.index(name) for name in dict.fromkeys(unclipped_terms)]
        self._correction_sum = torch.zeros(env.num_envs, device=env.device)
        self._total_sum = torch.zeros_like(self._correction_sum)

    def __getattr__(self, name):
        return getattr(self._manager, name)

    def __str__(self):
        return str(self._manager)

    def compute(self, dt):
        raw_total = self._manager.compute(dt)
        # IsaacLab stores weighted reward rates (without dt) in _step_reward.
        excluded = self._manager._step_reward[:, self._unclipped_indices].sum(dim=1) * dt
        ordinary = raw_total - excluded
        correction = (-ordinary).clamp_min(0)
        result = ordinary.clamp_min(0) + excluded
        self._correction_sum += correction
        self._total_sum += result
        log = self._env.extras.setdefault("log", {})
        log["Metrics/reward_clip_fraction"] = (ordinary < 0).float().mean()
        log["Metrics/reward_before_clipping"] = raw_total.mean()
        log["Metrics/reward_after_clipping"] = result.mean()
        return result

    def reset(self, env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        log = self._manager.reset(env_ids)
        # Match IsaacLab's per-term episode logs: episode sums / maximum duration.
        duration = self._env.max_episode_length_s
        log["Episode_Reward/positive_clip_correction"] = self._correction_sum[ids].mean() / duration
        log["Episode_Reward/total_after_clipping"] = self._total_sum[ids].mean() / duration
        self._correction_sum[ids] = 0
        self._total_sum[ids] = 0
        return log
