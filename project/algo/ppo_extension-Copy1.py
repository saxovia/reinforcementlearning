from .agent_base import BaseAgent
from .ppo_utils import Policy
from .ppo_agent import PPOAgent

import utils.common_utils as cu
import torch
import numpy as np
import torch.nn.functional as F
import time

class PPOExtension(PPOAgent):
    def __init__(self, config=None):
        super(PPOExtension, self).__init__(config)
        self.dual_clip_value = 5.0
        self.current_episode = 0

    def ppo_update(self, states, actions, rewards, next_states, dones, old_log_probs, targets):
        action_dists, values = self.policy(states)
        values = values.squeeze()
        
        new_action_probs = action_dists.log_prob(actions)
        ratio = torch.exp(new_action_probs - old_log_probs)
        clipped_ratio = torch.clamp(ratio, 1 - self.clip, 1 + self.clip)

        adaptive_clip = max(4.0, self.dual_clip_value - (self.current_episode / self.cfg.train_episodes) * 4.0)


        advantages = (targets - values).detach()
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        high_advantages = advantages > (advantages.mean() + 2 * advantages.std())
        clipped_advantages = torch.where(high_advantages, torch.tensor(adaptive_clip, device=self.device), advantages)

        advantages = advantages.unsqueeze(1).expand_as(ratio)
        clipped_advantages = clipped_advantages.unsqueeze(1).expand_as(ratio)

        policy_objective = -torch.min(ratio * advantages, clipped_ratio * clipped_advantages)
        value_loss = F.smooth_l1_loss(values, targets, reduction="mean")
        entropy = action_dists.entropy().mean()

        policy_objective = policy_objective.mean()
        loss = policy_objective + 0.5 * value_loss - 0.01 * entropy

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        #gradient clipping #changed
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)

    def train_iteration(self, ratio_of_episodes):
        self.current_episode += 1 #for the adaptive clipping
        return super().train_iteration(ratio_of_episodes)