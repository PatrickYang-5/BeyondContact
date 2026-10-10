"""Piper environment with optional non-negative ordinary reward totals."""

from isaaclab.envs import ManagerBasedRLEnv

from .reward_clipping import PositiveRewardManager


class PiperPositiveRewardEnv(ManagerBasedRLEnv):
    def load_managers(self):
        super().load_managers()
        if self.cfg.only_positive_rewards:
            self.reward_manager = PositiveRewardManager(
                self.reward_manager, self, self.cfg.unclipped_reward_terms
            )
            print(f"[INFO] Positive reward clipping enabled; unclipped terms: {self.cfg.unclipped_reward_terms}")
