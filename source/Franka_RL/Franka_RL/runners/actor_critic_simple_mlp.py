"""
Simple Actor-Critic with Direct MLP (No Feature Extraction).

Architecture:
    Policy Obs (45 dims) → Actor MLP → Actions (12 dims)
    
    Critic Obs (238 dims = 45 policy + 193 privileged) → Critic MLP → Value (1 dim)

Key Features:
- Actor directly processes raw policy observations (no embedding layer)
- Critic directly processes concatenated observations (no embedding layer)
- Simplest possible architecture for faster training
- Suitable for environments where observations are already well-scaled
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Normal
from hydra.utils import instantiate


class ActorCriticSimpleMLP(nn.Module):
    """Simple Actor-Critic with direct MLP processing of raw observations."""
    
    is_recurrent = False
    
    def __init__(
        self,
        num_obs,                # 45 (policy obs)
        num_privileged_obs,     # 238 (policy + privileged obs)
        num_actions,            # 12
        actor_config,           # Actor MLP config: 45 → 12
        critic_config,          # Critic MLP config: 238 → 1
        init_noise_std=1.0,
        noise_std_type: str = "scalar",
        **kwargs,
    ):
        if kwargs:
            print(
                "ActorCriticSimpleMLP.__init__ got unexpected arguments: "
                + str(list(kwargs.keys()))
            )
        super().__init__()
        
        # Store dimensions
        self.num_actor_obs = num_obs  # 45
        self.num_critic_obs = num_privileged_obs  # 238
        self.num_privileged_obs = num_privileged_obs - num_obs  # 193
        
        # Actor MLP: 45 → 12 (direct processing)
        self.actor = instantiate(actor_config)
        print(f"[ActorCriticSimpleMLP] Actor MLP: {self.actor}")
        
        # Critic MLP: 238 → 1 (direct processing)
        self.critic = instantiate(critic_config)
        print(f"[ActorCriticSimpleMLP] Critic MLP: {self.critic}")
        
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
        
        print(f"[ActorCriticSimpleMLP] Initialized with:")
        print(f"  Policy obs dim: {num_obs}")
        print(f"  Critic obs dim: {num_privileged_obs}")
        print(f"  Action dim: {num_actions}")
        print(f"  Init noise std: {init_noise_std}")
    
    def reset(self, dones=None):
        """Reset (no-op for MLP, kept for compatibility)."""
        pass
    
    def reset_noise_std(self, noise_std: float):
        """Manually update action noise std for exploration schedule.
        
        Args:
            noise_std: New standard deviation value (typically decaying from 1.0 to 0.1)
        """
        self.std.data.fill_(noise_std)
    
    def update_normalization(self, obs):
        """Update observation normalization.
        
        No normalization in this simple architecture (assumes pre-scaled observations).
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
        """Extract policy and critic observations from input.
        
        Args:
            observations: Can be dict or tensor
                - If dict: expects {"policy": [N,45], "critic": [N,238]}
                - If tensor: expects [N,238] or [N,45]
        
        Returns:
            policy_obs: [N, 45] - for actor
            critic_obs: [N, 238] - for critic (policy + privileged)
        """
        if isinstance(observations, dict) or hasattr(observations, 'keys'):
            # Dict input: extract from keys
            if "policy" in observations.keys():
                policy_obs = observations["policy"]
            elif "critic" in observations.keys():
                # Extract policy part from critic observations
                critic_obs_full = observations["critic"]
                policy_obs = critic_obs_full[:, :self.num_actor_obs]
            else:
                # Fallback: use first value
                obs_tensor = next(iter(observations.values()))
                policy_obs = obs_tensor[:, :self.num_actor_obs] if obs_tensor.shape[1] >= self.num_actor_obs else obs_tensor
            
            # Extract full critic observations
            if "critic" in observations.keys():
                critic_obs = observations["critic"]
            else:
                # Construct critic obs by concatenating policy + privileged
                # If no privileged obs available, pad with zeros
                privileged_obs = torch.zeros(
                    policy_obs.shape[0], self.num_privileged_obs,
                    device=policy_obs.device, dtype=policy_obs.dtype
                )
                critic_obs = torch.cat([policy_obs, privileged_obs], dim=-1)
        else:
            # Tensor input
            if observations.shape[1] == self.num_critic_obs:
                # Full critic observations: [N, 238]
                policy_obs = observations[:, :self.num_actor_obs]
                critic_obs = observations
            elif observations.shape[1] == self.num_actor_obs:
                # Only policy observations: [N, 45]
                policy_obs = observations
                # Pad with zeros for privileged part
                privileged_obs = torch.zeros(
                    observations.shape[0], self.num_privileged_obs,
                    device=observations.device, dtype=observations.dtype
                )
                critic_obs = torch.cat([policy_obs, privileged_obs], dim=-1)
            else:
                raise ValueError(
                    f"Unexpected observation shape: {observations.shape}. "
                    f"Expected {self.num_actor_obs} or {self.num_critic_obs}"
                )
        
        return policy_obs, critic_obs
    
    def update_distribution(self, observations):
        """Update action distribution using policy observations.
        
        Args:
            observations: Policy observations (dict or tensor)
        """
        # Extract policy observations
        policy_obs, _ = self._extract_observations(observations)
        
        # Directly feed into actor MLP (no embedding)
        mean = self.actor(policy_obs)
        
        # Compute standard deviation with protection against negative values
        if self.noise_std_type == "scalar":
            # ✅ Protect against negative std values during training
            # The std parameter can become negative during gradient descent
            std = torch.abs(self.std).expand_as(mean)
        elif self.noise_std_type == "log":
            # log_std is naturally protected (exp always positive)
            std = torch.exp(self.log_std).expand_as(mean)
        else:
            raise ValueError(f"Unknown std type: {self.noise_std_type}")
        
        # Create distribution with clamped std for numerical stability
        # Clamp to [1e-4, 10.0] to prevent both numerical issues and overly large exploration
        safe_std = torch.clamp(std, min=1e-4, max=10.0)
        self.distribution = Normal(mean, safe_std)
    
    def act(self, observations, **kwargs):
        """Sample action from current distribution with clipping."""
        self.update_distribution(observations)
        actions = self.distribution.sample()
        clipped_actions = torch.clamp(actions, -1.0, 1.0)
        return clipped_actions
    
    def get_actions_log_prob(self, actions):
        """Get log probability of actions."""
        return self.distribution.log_prob(actions).sum(dim=-1)
    
    def act_inference(self, observations):
        """Deterministic action for inference with clipping."""
        policy_obs, _ = self._extract_observations(observations)
        actions_mean = self.actor(policy_obs)
        # ✅ Clip actions to [-1, 1] for safe deployment
        clipped_actions = torch.clamp(actions_mean, -1.0, 1.0)
        return clipped_actions
    
    def evaluate(self, critic_observations, masks=None, hidden_states=None):
        """Evaluate value function using full observations.
        
        Args:
            critic_observations: Critic observations (dict or tensor)
            masks: Episode masks (unused, for compatibility)
            hidden_states: Hidden states (unused, for compatibility)
        
        Returns:
            value: Value estimates [batch_size, 1]
        """
        # Extract critic observations (238 dims)
        _, critic_obs = self._extract_observations(critic_observations)
        
        # Directly feed into critic MLP (no embedding)
        value = self.critic(critic_obs)
        
        return value
    
    def load_state_dict(self, state_dict, strict=True):
        """Load model parameters with architecture compatibility checking."""
        
        # Check architecture compatibility
        checkpoint_has_transformer = any('seqTransEncoder' in k for k in state_dict.keys())
        checkpoint_has_embedding = any('policy_embed' in k or 'privileged_embed' in k for k in state_dict.keys())
        
        # Detect architecture mismatch
        if checkpoint_has_transformer:
            print("\n" + "="*80)
            print("[WARNING] Architecture Mismatch Detected!")
            print("="*80)
            print("Checkpoint architecture: Transformer-based ActorCritic")
            print("Current model architecture: Simple MLP ActorCritic")
            print("\nThese architectures are incompatible.")
            print("[INFO] Skipping checkpoint loading - model will use random initialization")
            print("="*80 + "\n")
            return False
        
        if checkpoint_has_embedding:
            print("\n" + "="*80)
            print("[WARNING] Architecture Mismatch Detected!")
            print("="*80)
            print("Checkpoint architecture: Dual MLP with Embedding layers")
            print("Current model architecture: Simple MLP without Embedding")
            print("\nThese architectures are incompatible.")
            print("[INFO] Skipping checkpoint loading - model will use random initialization")
            print("="*80 + "\n")
            return False
        
        # Load model parameters normally
        missing_keys, unexpected_keys = super().load_state_dict(state_dict, strict=False)
        
        # Report loading status
        if unexpected_keys:
            print(f"[WARNING] Unexpected keys in checkpoint: {len(unexpected_keys)} items")
            if strict:
                error_msg = f"Unexpected key(s) in state_dict: {', '.join(unexpected_keys[:10])}"
                if len(unexpected_keys) > 10:
                    error_msg += f" ... and {len(unexpected_keys) - 10} more"
                raise RuntimeError(error_msg)
        
        if missing_keys:
            print(f"[INFO] Missing keys in checkpoint: {len(missing_keys)} items (will use random init)")
        
        if not missing_keys and not unexpected_keys:
            print("[INFO] Successfully loaded all parameters from checkpoint")
        
        return True
    
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
