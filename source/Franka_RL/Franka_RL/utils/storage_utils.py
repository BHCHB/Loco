import torch
from copy import deepcopy

from rsl_rl.storage import RolloutStorage

class RolloutStorageWithDictObs(RolloutStorage):
    def __init__(self, training_type, num_envs, num_transitions_per_env, obs_shape, privileged_obs_shape, actions_shape, rnd_state_shape=None, device="cpu"):
        super().__init__(training_type, num_envs, num_transitions_per_env, obs_shape, privileged_obs_shape, actions_shape, rnd_state_shape, device)

        self.observations = [None for _ in range(self.num_transitions_per_env)]
        if privileged_obs_shape is not None:
            self.privileged_observations = [None for _ in range(self.num_transitions_per_env)]

    # def from_parent(parent: RolloutStorage):
    #     return RolloutStorageWithDictObs(parent.training_type, parent.num_envs, parent.num_transitions_per_env, parent.obs_shape, parent.privileged_obs_shape, parent.actions_shape, parent.rnd_state_shape, parent.device)
    
    def add_transitions(self, transition):
        # check if the transition is valid
        if self.step >= self.num_transitions_per_env:
            raise OverflowError("Rollout buffer overflow! You should call clear() before adding new transitions.")

        # Core
        self.observations[self.step] = deepcopy(transition.observations)
        if self.privileged_observations is not None:
            self.privileged_observations[self.step] = deepcopy(transition.privileged_observations)
        self.actions[self.step].copy_(transition.actions)
        self.rewards[self.step].copy_(transition.rewards.view(-1, 1))
        self.dones[self.step].copy_(transition.dones.view(-1, 1))

        # for distillation
        if self.training_type == "distillation":
            self.privileged_actions[self.step].copy_(transition.privileged_actions)

        # for reinforcement learning
        if self.training_type == "rl":
            self.values[self.step].copy_(transition.values)
            self.actions_log_prob[self.step].copy_(transition.actions_log_prob.view(-1, 1))
            self.mu[self.step].copy_(transition.action_mean)
            self.sigma[self.step].copy_(transition.action_sigma)

        # For RND
        if self.rnd_state_shape is not None:
            self.rnd_state[self.step].copy_(transition.rnd_state)

        # For RNN networks
        self._save_hidden_states(transition.hidden_states)

        # increment the counter
        self.step += 1