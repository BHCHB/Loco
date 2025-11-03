"""
Actor-Critic with Dual Embedding Architecture.

Architecture:
    Input: observations["critic"] (238 dims) = [policy_obs(45) + privileged_obs(193)]
           ↓
    Manual slice separation
    ↙                          ↘
policy_obs[0:45]           privileged_obs[45:238]
    ↓                              ↓
Policy Embedding MLP          Privileged Embedding MLP
(45 → policy_feature_dim)     (193 → priv_feature_dim)
    ↓                              ↓
policy_features               priv_features
    ↓                              ↘
Transformer                         ↘
    ↓                                ↘
Actor Output                  concat(policy + priv features)
                                     ↓
                                 Critic MLP
                                     ↓
                                  Value
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Normal
from hydra.utils import instantiate


class ActorCriticWithDualEmbedding(nn.Module):
    """Actor-Critic with separate embedding MLPs for policy and privileged observations."""
    
    is_recurrent = False
    
    def __init__(
        self,
        num_obs,                # 45 (policy obs) - matches runner's first arg
        num_privileged_obs,     # 238 (policy + privileged obs) - matches runner's second arg
        num_actions,            # 12
        policy_embed_config,    # Policy embedding MLP config (45 → feature_dim)
        privileged_embed_config,  # Privileged embedding MLP config (193 → feature_dim)
        actor_config,           # Actor (Transformer) config
        critic_config,          # Critic MLP config
        init_noise_std=1.0,
        noise_std_type: str = "scalar",
        **kwargs,
    ):
        if kwargs:
            print(
                "ActorCriticWithDualEmbedding.__init__ got unexpected arguments: "
                + str(list(kwargs.keys()))
            )
        super().__init__()
        
        # Store dimensions (use standard naming internally)
        self.num_actor_obs = num_obs  # 45
        self.num_critic_obs = num_privileged_obs  # 238
        self.num_privileged_obs = num_privileged_obs - num_obs  # 193
        
        # Policy embedding: 45 → feature_dim
        self.policy_embed = instantiate(policy_embed_config)
        print(f"Policy Embedding: {self.policy_embed}")
        
        # Privileged embedding: 193 → feature_dim  
        self.privileged_embed = instantiate(privileged_embed_config)
        print(f"Privileged Embedding: {self.privileged_embed}")
        
        # Actor (Transformer): takes policy features
        self.actor = instantiate(actor_config)
        print(f"Actor: {self.actor}")
        
        # Critic: takes concatenated features (policy + privileged)
        self.critic = instantiate(critic_config)
        print(f"Critic: {self.critic}")
        
        # Action noise
        self.noise_std_type = noise_std_type
        if self.noise_std_type == "scalar":
            self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        elif self.noise_std_type == "log":
            self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(num_actions)))
        else:
            raise ValueError(f"Unknown std type: {self.noise_std_type}")
        
        # Action distribution
        self.distribution = None
        Normal.set_default_validate_args(False)
    
    def reset(self, dones=None):
        """Reset sequence cache for Transformer."""
        if hasattr(self.actor, 'reset'):
            self.actor.reset(dones)
    
    def update_normalization(self, obs):
        """Update observation normalization.
        
        This method is called by PPO.process_env_step() to update the normalizers.
        Since we handle normalization inside the embedding MLPs (RunningMeanStd),
        this is a no-op at the policy level.
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
    
    def _extract_observations(self, observations):
        """Extract policy and privileged observations from input.
        
        Args:
            observations: Can be dict or tensor
                - If dict: expects {"policy": [N,45], "critic": [N,238]}
                - If tensor: expects [N,238] or [N,45]
        
        Returns:
            policy_obs: [N, 45]
            privileged_obs: [N, 193]
        """
        if isinstance(observations, dict) or hasattr(observations, 'keys'):
            # Dict input: extract from keys
            if "policy" in observations.keys():
                policy_obs = observations["policy"]
            elif "critic" in observations.keys():
                # Extract policy part from critic observations
                critic_obs = observations["critic"]
                policy_obs = critic_obs[:, :self.num_actor_obs]
            else:
                # Fallback: use first value
                obs_tensor = next(iter(observations.values()))
                policy_obs = obs_tensor[:, :self.num_actor_obs] if obs_tensor.shape[1] >= self.num_actor_obs else obs_tensor
            
            # Extract privileged observations
            if "critic" in observations.keys():
                critic_obs = observations["critic"]
                privileged_obs = critic_obs[:, self.num_actor_obs:]
            else:
                # No privileged obs available, create zeros
                privileged_obs = torch.zeros(
                    policy_obs.shape[0], self.num_privileged_obs,
                    device=policy_obs.device, dtype=policy_obs.dtype
                )
        else:
            # Tensor input
            if observations.shape[1] == self.num_critic_obs:
                # Full critic observations
                policy_obs = observations[:, :self.num_actor_obs]
                privileged_obs = observations[:, self.num_actor_obs:]
            elif observations.shape[1] == self.num_actor_obs:
                # Only policy observations
                policy_obs = observations
                privileged_obs = torch.zeros(
                    observations.shape[0], self.num_privileged_obs,
                    device=observations.device, dtype=observations.dtype
                )
            else:
                raise ValueError(
                    f"Unexpected observation shape: {observations.shape}. "
                    f"Expected {self.num_actor_obs} or {self.num_critic_obs}"
                )
        
        return policy_obs, privileged_obs
    
    def update_distribution(self, observations):
        """Update action distribution using policy observations.
        
        Args:
            observations: Policy observations (dict or tensor)
        """
        # Extract policy observations
        policy_obs, _ = self._extract_observations(observations)
        
        # Embed policy observations
        policy_features = self.policy_embed(policy_obs)
        
        # Wrap features in dict for Transformer
        features_dict = {"policy_features": policy_features}
        
        # Get action mean from actor (Transformer)
        mean = self.actor(features_dict)
        
        # Compute standard deviation
        if self.noise_std_type == "scalar":
            std = self.std.expand_as(mean)
        elif self.noise_std_type == "log":
            std = torch.exp(self.log_std).expand_as(mean)
        else:
            raise ValueError(f"Unknown std type: {self.noise_std_type}")
        
        # Create distribution
        safe_std = torch.clamp(std, min=1e-4, max=10.0)
        self.distribution = Normal(mean, safe_std)
    
    def act(self, observations, **kwargs):
        """Sample action from current distribution."""
        self.update_distribution(observations)
        return self.distribution.sample()
    
    def get_actions_log_prob(self, actions):
        """Get log probability of actions."""
        return self.distribution.log_prob(actions).sum(dim=-1)
    
    def act_inference(self, observations):
        """Deterministic action for inference."""
        policy_obs, _ = self._extract_observations(observations)
        policy_features = self.policy_embed(policy_obs)
        # Wrap features in dict for Transformer
        features_dict = {"policy_features": policy_features}
        actions_mean = self.actor(features_dict)
        return actions_mean
    
    def evaluate(self, critic_observations, masks=None, hidden_states=None):
        """Evaluate value function using full privileged observations.
        
        Args:
            critic_observations: Critic observations (dict or tensor)
            masks: Episode masks (unused, for compatibility)
            hidden_states: Hidden states (unused, for compatibility)
        
        Returns:
            value: Value estimates [batch_size, 1]
        """
        # Extract both policy and privileged observations
        policy_obs, privileged_obs = self._extract_observations(critic_observations)
        
        # Embed both parts
        policy_features = self.policy_embed(policy_obs)
        privileged_features = self.privileged_embed(privileged_obs)
        
        # Concatenate features
        full_features = torch.cat([policy_features, privileged_features], dim=-1)
        
        # Evaluate value
        value = self.critic(full_features)
        
        return value
    
    def load_state_dict(self, state_dict, strict=True):
        """Load model parameters."""
        super().load_state_dict(state_dict, strict=strict)
        return False  # Not resuming distillation
    
    def get_std(self):
        """Get current action standard deviation."""
        if self.noise_std_type == "scalar":
            return self.std
        elif self.noise_std_type == "log":
            return torch.exp(self.log_std)
        else:
            raise ValueError(f"Unknown std type: {self.noise_std_type}")
    
    def set_std(self, std):
        """Set action standard deviation."""
        if self.noise_std_type == "scalar":
            self.std.data = std
        elif self.noise_std_type == "log":
            self.log_std.data = torch.log(std)
        else:
            raise ValueError(f"Unknown std type: {self.noise_std_type}")
