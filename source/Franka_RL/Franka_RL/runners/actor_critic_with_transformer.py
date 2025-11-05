from __future__ import annotations

import os
import statistics
import time
import torch
from collections import deque
import torch.nn as nn
from torch.distributions import Normal
from hydra.utils import instantiate

import rsl_rl
from rsl_rl.algorithms import PPO, Distillation
from rsl_rl.env import VecEnv
from rsl_rl.modules import (
    ActorCritic,
    ActorCriticRecurrent,
    EmpiricalNormalization,
    StudentTeacher,
    StudentTeacherRecurrent,
)
from rsl_rl.utils import store_code_state
from rsl_rl.utils import resolve_nn_activation


class ActorCriticWithTransformer(nn.Module):
    is_recurrent = False
    
    def __init__(
        self,
        num_actor_obs,
        num_critic_obs,
        num_actions,
        actor_config,
        critic_config,
        init_noise_std=1.0,
        noise_std_type: str = "scalar",
        **kwargs,
    ):
        if kwargs:
            print(
                "ActorCritic.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs.keys()])
            )
        super().__init__()

        # Policy
        self.actor = instantiate(actor_config)

        # Value Function
        self.critic = instantiate(critic_config)

        print(f"Actor: {self.actor}")
        print(f"Critic: {self.critic}")

        # Action noise
        self.noise_std_type = noise_std_type
        if self.noise_std_type == "scalar":
            self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        elif self.noise_std_type == "log":
            self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(num_actions)))
        else:
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")

        # Action distribution (populated in update_distribution)
        self.distribution = None
        # disable args validation for speedup
        Normal.set_default_validate_args(False)

    @staticmethod
    def init_weights(sequential, scales):
        [
            torch.nn.init.orthogonal_(module.weight, gain=scales[idx])
            for idx, module in enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))
        ]

    def reset(self, dones=None):
        """
        Reset hidden states / sequence cache for done environments.
        
        For Transformer with sequence caching, this clears the history buffer.
        """
        if hasattr(self.actor, 'clear_cache'):
            if dones is None:
                # Reset all environments
                self.actor.clear_cache()
            else:
                # Reset only done environments
                done_indices = torch.nonzero(dones, as_tuple=False).squeeze(-1)
                if len(done_indices) > 0:
                    self.actor.clear_cache(done_indices)
    
    def update_normalization(self, obs):
        """Update observation normalization.
        
        This method is called by PPO.process_env_step() to update the normalizers.
        Since we handle normalization at the runner level (not inside the policy),
        this is a no-op.
        """
        pass

    def forward(self):
        raise NotImplementedError

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def update_distribution(self, observations):
        # compute mean
        mean = self.actor(observations)
        # compute standard deviation with protection against negative values
        if self.noise_std_type == "scalar":
            # ✅ Protect against negative std values during training
            std = torch.abs(self.std).expand_as(mean)
        elif self.noise_std_type == "log":
            # log_std is naturally protected (exp always positive)
            std = torch.exp(self.log_std).expand_as(mean)
        else:
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")
        # create distribution with clamped std for numerical stability
        safe_std = torch.clamp(std, min=1e-4, max=10.0)
        self.distribution = Normal(mean, safe_std)

    def act(self, observations, **kwargs):
        """Sample action from current distribution with clipping."""
        self.update_distribution(observations)
        actions = self.distribution.sample()
        # ✅ Clip actions to [-1, 1]
        clipped_actions = torch.clamp(actions, -1.0, 1.0)
        return clipped_actions

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, observations):
        """Deterministic action for inference with clipping."""
        actions_mean = self.actor(observations)
        # ✅ Clip actions to [-1, 1] for safe deployment
        clipped_actions = torch.clamp(actions_mean, -1.0, 1.0)
        return clipped_actions

    def evaluate(self, critic_observations, masks=None, hidden_states=None):
        """
        Evaluate the value function.
        
        Args:
            critic_observations: Critic observations (can be dict or tensor)
            masks: Episode masks (for recurrent policies)
            hidden_states: Hidden states (for recurrent policies)
        
        Returns:
            value: Value estimates [batch_size, 1]
        """
        # Handle dict/TensorDict input
        if isinstance(critic_observations, dict) or hasattr(critic_observations, 'keys'):
            # Extract tensor from dict - prefer 'critic' key for privileged observations
            if 'critic' in critic_observations.keys():
                critic_tensor = critic_observations['critic']
            elif 'policy' in critic_observations.keys():
                critic_tensor = critic_observations['policy']
            else:
                critic_tensor = next(iter(critic_observations.values()))
        else:
            critic_tensor = critic_observations
        
        value = self.critic(critic_tensor)
        return value

    def load_state_dict(self, state_dict, strict=True):
        """Load the parameters of the actor-critic model.

        Args:
            state_dict (dict): State dictionary of the model.
            strict (bool): Whether to strictly enforce that the keys in state_dict match the keys returned by this
                           module's state_dict() function.

        Returns:
            bool: Whether this training resumes a previous training. This flag is used by the `load()` function of
                  `OnPolicyRunner` to determine how to load further parameters (relevant for, e.g., distillation).
        """

        super().load_state_dict(state_dict, strict=strict)
        return True

    


