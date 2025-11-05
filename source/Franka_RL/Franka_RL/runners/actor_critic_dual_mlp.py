"""
Simple Asymmetric Actor-Critic with Dual MLP (No Transformer).

Architecture:
    Policy Obs (45 dims) → Policy MLP → Actor MLP → Actions (12 dims)
                             ↓
                         policy_features
                             ↓
                          concat(512)
                             ↓
    Privileged Obs (193 dims) → Privileged MLP → priv_features
                                                       ↓
                                                   Critic MLP → Value (1 dim)

Key Features:
- Actor only uses policy observations (45 dims)
- Critic uses both policy and privileged observations
- No Transformer, pure MLP architecture
- Suitable for deployment (actor is independent)
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Normal
from hydra.utils import instantiate


class ActorCriticDualMLP(nn.Module):
    """Simple Asymmetric Actor-Critic with separate MLPs for policy and privileged observations."""
    
    is_recurrent = False
    
    def __init__(
        self,
        num_obs,                # 45 (policy obs)
        num_privileged_obs,     # 238 (policy + privileged obs)
        num_actions,            # 12
        policy_embed_config,    # Policy embedding MLP config (45 → feature_dim)
        privileged_embed_config,  # Privileged embedding MLP config (193 → feature_dim)
        actor_config,           # Actor MLP config
        critic_config,          # Critic MLP config
        init_noise_std=1.0,
        noise_std_type: str = "scalar",
        **kwargs,
    ):
        if kwargs:
            print(
                "ActorCriticDualMLP.__init__ got unexpected arguments: "
                + str(list(kwargs.keys()))
            )
        super().__init__()
        
        # Store dimensions
        self.num_actor_obs = num_obs  # 45
        self.num_critic_obs = num_privileged_obs  # 238
        self.num_privileged_obs = num_privileged_obs - num_obs  # 193
        
        # Policy embedding: 45 → feature_dim (e.g., 256)
        self.policy_embed = instantiate(policy_embed_config)
        print(f"Policy Embedding MLP: {self.policy_embed}")
        
        # Privileged embedding: 193 → feature_dim (e.g., 256)
        self.privileged_embed = instantiate(privileged_embed_config)
        print(f"Privileged Embedding MLP: {self.privileged_embed}")
        
        # Actor MLP: takes policy features
        self.actor = instantiate(actor_config)
        print(f"Actor MLP: {self.actor}")
        
        # Critic MLP: takes concatenated features (policy + privileged)
        self.critic = instantiate(critic_config)
        print(f"Critic MLP: {self.critic}")
        
        # Action noise (use log_std for better numerical stability)
        self.noise_std_type = noise_std_type
        if self.noise_std_type == "scalar":
            # Legacy scalar parameterization (convert to log internally for stability)
            self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(num_actions)))
            print(f"[INFO] Converting scalar std to log_std parameterization for numerical stability")
        elif self.noise_std_type == "log":
            self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(num_actions)))
        else:
            raise ValueError(f"Unknown std type: {self.noise_std_type}")
        
        # Action distribution
        self.distribution = None
        Normal.set_default_validate_args(False)
    
    def reset(self, dones=None):
        """Reset (no-op for MLP, kept for compatibility)."""
        pass
    
    def reset_noise_std(self, noise_std: float):
        """Manually update action noise std for exploration schedule."""
        self.log_std.data.fill_(torch.log(torch.tensor(noise_std)))
    
    def update_normalization(self, obs):
        """Update observation normalization.
        
        Normalization is handled inside the embedding MLPs (RunningMeanStd),
        so this is a no-op at the policy level.
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
        
        # Get action mean from actor MLP
        mean = self.actor(policy_features)
    
        # log_std ∈ [-20, 2] → std ∈ [2e-9, 7.4]
        log_std_clamped = torch.clamp(self.log_std, min=-20.0, max=2.0)
        std = torch.exp(log_std_clamped).expand_as(mean)
        
        # Additional safety clamp for distribution creation
        safe_std = torch.clamp(std, min=1e-6, max=10.0)
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
        policy_features = self.policy_embed(policy_obs)
        actions_mean = self.actor(policy_features)
        clipped_actions = torch.clamp(actions_mean, -1.0, 1.0)
        return clipped_actions
    
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
        """Load model parameters with architecture compatibility checking and std conversion."""
        

        if 'std' in state_dict and 'log_std' not in state_dict:
            print("[INFO] Converting legacy 'std' parameter to 'log_std'")
            std_value = state_dict.pop('std')  # Remove old 'std'
            # Clamp std to valid range before taking log
            std_value = torch.clamp(std_value, min=1e-6, max=10.0)
            state_dict['log_std'] = torch.log(std_value)  # Add new 'log_std'
            print(f"  std range: [{std_value.min().item():.6f}, {std_value.max().item():.6f}]")
            print(f"  log_std range: [{state_dict['log_std'].min().item():.6f}, {state_dict['log_std'].max().item():.6f}]")
        
        # Check architecture compatibility
        checkpoint_has_old_actor = any('actor.0.weight' in k or 'actor.mlp.' in k for k in state_dict.keys())
        checkpoint_has_transformer = any('actor.seqTransEncoder' in k for k in state_dict.keys())
        model_has_mlp_actor = hasattr(self.actor, 'mlp') or isinstance(self.actor, nn.Sequential)
        
        
        # Filter out RunningMeanStd buffer keys
        running_mean_std_patterns = [
            'running_obs_norm.mean', 
            'running_obs_norm.var', 
            'running_obs_norm.count'
        ]
        
        rms_state = {}
        model_state = {}
        unexpected_filtered = []
        
        for key, value in state_dict.items():
            if any(pattern in key for pattern in running_mean_std_patterns):
                rms_state[key] = value
            # Filter out transformer keys if loading into MLP
            elif checkpoint_has_transformer and ('seqTransEncoder' in key or 'sequence_pos_encoder' in key):
                unexpected_filtered.append(key)
                continue
            else:
                model_state[key] = value
        
        if unexpected_filtered:
            print(f"[INFO] Filtered out {len(unexpected_filtered)} keys from Transformer architecture")
        
        # Load model parameters
        missing_keys, unexpected_keys = super().load_state_dict(model_state, strict=False)
        
        # Try to load RMS buffers if they exist
        rms_loaded = []
        rms_skipped = []
        for key, value in rms_state.items():
            try:
                parts = key.split('.')
                obj = self
                for part in parts[:-1]:
                    obj = getattr(obj, part)
                
                if hasattr(obj, parts[-1]):
                    buffer = getattr(obj, parts[-1])
                    buffer.data.copy_(value)
                    rms_loaded.append(key)
                else:
                    rms_skipped.append(key)
            except (AttributeError, RuntimeError):
                rms_skipped.append(key)
        
        # Report loading status
        if rms_loaded:
            print(f"[INFO] Loaded RunningMeanStd buffers: {len(rms_loaded)} items")
        if rms_skipped:
            print(f"[INFO] Skipped uninitialized RunningMeanStd buffers: {len(rms_skipped)} items")
        
        # Check for truly unexpected keys
        if strict and len(unexpected_keys) > 0:
            print(f"[WARNING] Unexpected keys in checkpoint: {len(unexpected_keys)} items")
            if not checkpoint_has_transformer:
                error_msg = f"Unexpected key(s) in state_dict: {', '.join(unexpected_keys[:10])}"
                if len(unexpected_keys) > 10:
                    error_msg += f" ... and {len(unexpected_keys) - 10} more"
                raise RuntimeError(error_msg)
        
        if missing_keys:
            print(f"[INFO] Missing keys in checkpoint: {len(missing_keys)} items (will use random init)")
        
        return False
    
    def get_std(self):
        """Get current action standard deviation."""
        log_std_clamped = torch.clamp(self.log_std, min=-20.0, max=2.0)
        return torch.exp(log_std_clamped)
    
    def set_std(self, std):
        """Set action standard deviation."""
        self.log_std.data = torch.log(torch.clamp(std, min=1e-6, max=10.0))
