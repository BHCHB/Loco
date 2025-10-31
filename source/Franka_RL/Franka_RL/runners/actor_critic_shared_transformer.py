"""
Actor-Critic with Shared Transformer Encoder

Architecture:
    Input → Shared Transformer Encoder (with RoPE) → Actor Head / Critic Head
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Normal
from hydra.utils import instantiate


class ActorCriticSharedTransformer(nn.Module):
    """
    Actor-Critic with shared Transformer encoder.
    
    This architecture uses a single Transformer encoder (with RoPE) that is shared
    between the actor and critic networks. The encoder learns a common representation
    of the observation history, which is then processed by separate actor and critic heads.
    
    Architecture:
        observations → [Shared Transformer Encoder] → features
                              ↓                ↓
                        [Actor Head]    [Critic Head]
                              ↓                ↓
                          actions           values
    
    Args:
        num_actor_obs (int): Number of actor observations
        num_critic_obs (int): Number of critic observations (typically same as actor)
        num_actions (int): Number of actions
        shared_encoder_config: Hydra config for shared Transformer encoder
        actor_head_config: Hydra config for actor head (MLP)
        critic_head_config: Hydra config for critic head (MLP)
        init_noise_std (float): Initial standard deviation for action noise
        noise_std_type (str): Type of noise std ("scalar" or "log")
    """
    
    is_recurrent = False  # For compatibility with rsl_rl
    
    def __init__(
        self,
        num_actor_obs: int,
        num_critic_obs: int,
        num_actions: int,
        shared_encoder_config,
        actor_head_config,
        critic_head_config,
        init_noise_std: float = 1.0,
        noise_std_type: str = "scalar",
        **kwargs,
    ):
        if kwargs:
            print(f"[ActorCriticSharedTransformer] Unused kwargs: {list(kwargs.keys())}")
        
        super().__init__()
        
        print("\n" + "="*80)
        print("Initializing ActorCriticSharedTransformer")
        print("="*80)
        
        # ========== Shared Transformer Encoder ==========
        print("\n[1/3] Creating Shared Transformer Encoder...")
        self.shared_encoder = instantiate(shared_encoder_config)
        
        # Get encoder output dimension
        encoder_output_dim = shared_encoder_config.config.latent_dim
        print(f"  ✓ Encoder output dim: {encoder_output_dim}")
        
        # ========== Actor Head ==========
        print("\n[2/3] Creating Actor Head...")
        # Ensure actor head input matches encoder output
        if hasattr(actor_head_config, 'num_in'):
            actor_head_config.num_in = encoder_output_dim
        actor_head_config.num_out = num_actions
        
        self.actor_head = instantiate(actor_head_config)
        print(f"  ✓ Actor Head: {encoder_output_dim} → {num_actions}")
        
        # ========== Critic Head ==========
        print("\n[3/3] Creating Critic Head...")
        # Ensure critic head input matches encoder output
        if hasattr(critic_head_config, 'num_in'):
            critic_head_config.num_in = encoder_output_dim
        critic_head_config.num_out = 1
        
        self.critic_head = instantiate(critic_head_config)
        print(f"  ✓ Critic Head: {encoder_output_dim} → 1")
        
        # ========== Action Noise ==========
        self.noise_std_type = noise_std_type
        if self.noise_std_type == "scalar":
            self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
            print(f"\n[Action Noise] Scalar std initialized to {init_noise_std}")
        elif self.noise_std_type == "log":
            self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(num_actions)))
            print(f"\n[Action Noise] Log std initialized to log({init_noise_std})")
        else:
            raise ValueError(f"Unknown std type: {self.noise_std_type}")
        
        # Action distribution
        self.distribution = None
        Normal.set_default_validate_args(False)
        
        # Print parameter statistics
        self._print_parameter_count()
        
        print("="*80)
        print("✅ ActorCriticSharedTransformer initialized successfully!")
        print("="*80 + "\n")
    
    def _print_parameter_count(self):
        """Print parameter count for each component."""
        encoder_params = sum(p.numel() for p in self.shared_encoder.parameters())
        actor_params = sum(p.numel() for p in self.actor_head.parameters())
        critic_params = sum(p.numel() for p in self.critic_head.parameters())
        
        # Count std/log_std parameters
        if hasattr(self, 'std'):
            noise_params = self.std.numel()
        elif hasattr(self, 'log_std'):
            noise_params = self.log_std.numel()
        else:
            noise_params = 0
        
        total_params = encoder_params + actor_params + critic_params + noise_params
        
        print("\n" + "-"*80)
        print("Parameter Count Summary:")
        print("-"*80)
        print(f"  Shared Encoder:  {encoder_params:>10,} ({encoder_params/total_params*100:>5.1f}%)")
        print(f"  Actor Head:      {actor_params:>10,} ({actor_params/total_params*100:>5.1f}%)")
        print(f"  Critic Head:     {critic_params:>10,} ({critic_params/total_params*100:>5.1f}%)")
        print(f"  Noise Params:    {noise_params:>10,} ({noise_params/total_params*100:>5.1f}%)")
        print(f"  {'─'*80}")
        print(f"  Total:           {total_params:>10,} (100.0%)")
        print("-"*80)
        
        # Compare with separate network estimate
        separate_network_estimate = encoder_params * 2 + actor_params + critic_params
        savings = separate_network_estimate - total_params
        savings_pct = savings / separate_network_estimate * 100
        
        print(f"\n💡 Estimated savings vs. separate networks:")
        print(f"  Separate networks: ~{separate_network_estimate:,} params")
        print(f"  Shared encoder:     {total_params:,} params")
        print(f"  Savings:           ~{savings:,} params ({savings_pct:.1f}%)")
        print("-"*80)
    
    def forward_shared_encoder(self, observations):
        """
        Forward pass through the shared Transformer encoder.
        
        Args:
            observations: Observations (can be dict, TensorDict, or tensor)
            
        Returns:
            encoded_features: [batch_size, latent_dim]
        """
        return self.shared_encoder(observations)
    
    def reset(self, dones=None):
        """
        Reset hidden states (for recurrent policies).
        
        For Transformer with cache, clear the cache for done environments.
        """
        if hasattr(self.shared_encoder, 'clear_cache'):
            if dones is None:
                # Reset all environments
                self.shared_encoder.clear_cache()
            else:
                # Reset only done environments
                done_indices = torch.nonzero(dones, as_tuple=False).squeeze(-1)
                if len(done_indices) > 0:
                    self.shared_encoder.clear_cache(done_indices)
    
    def update_normalization(self, obs):
        """
        Update observation normalization statistics.
        
        This is called by PPO.process_env_step() to update normalizers.
        Since normalization is handled at runner level, this is a no-op.
        """
        pass
    
    def forward(self):
        """Not used - kept for compatibility."""
        raise NotImplementedError
    
    @property
    def action_mean(self):
        """Get the mean of the action distribution."""
        return self.distribution.mean
    
    @property
    def action_std(self):
        """Get the standard deviation of the action distribution."""
        return self.distribution.stddev
    
    @property
    def entropy(self):
        """Get the entropy of the action distribution."""
        return self.distribution.entropy().sum(dim=-1)
    
    def update_distribution(self, observations):
        """
        Update the action distribution based on observations.
        
        Args:
            observations: Actor observations
        """
        # 1. Encode observations through shared encoder
        encoded = self.forward_shared_encoder(observations)
        
        # 2. Pass through actor head to get action mean
        mean = self.actor_head(encoded)
        
        # 3. Compute standard deviation
        if self.noise_std_type == "scalar":
            std = self.std.expand_as(mean)
        elif self.noise_std_type == "log":
            std = torch.exp(self.log_std).expand_as(mean)
        else:
            raise ValueError(f"Unknown std type: {self.noise_std_type}")
        # 3.1 Clamp std for numerical stability (remove diagnostic attribute storage)
        safe_std = torch.clamp(std, min=1e-4, max=10.0)
        # 4. Create distribution
        self.distribution = Normal(mean, safe_std)
    
    def act(self, observations, **kwargs):
        """
        Sample actions from the current policy.
        
        Args:
            observations: Actor observations
            
        Returns:
            actions: Sampled actions
        """
        self.update_distribution(observations)
        return self.distribution.sample()
    
    def get_actions_log_prob(self, actions):
        """
        Compute log probabilities of given actions under current distribution.
        
        Args:
            actions: Actions to evaluate
            
        Returns:
            log_probs: Log probabilities
        """
        return self.distribution.log_prob(actions).sum(dim=-1)
    
    def act_inference(self, observations):
        """
        Get deterministic actions for inference (mean of distribution).
        
        Args:
            observations: Actor observations
            
        Returns:
            actions_mean: Deterministic actions
        """
        encoded = self.forward_shared_encoder(observations)
        actions_mean = self.actor_head(encoded)
        return actions_mean
    
    def evaluate(self, critic_observations, masks=None, hidden_states=None):
        """
        Evaluate state values.
        
        Args:
            critic_observations: Critic observations (can be dict, TensorDict, or tensor)
            masks: Not used (for compatibility with recurrent policies)
            hidden_states: Not used (for compatibility with recurrent policies)
            
        Returns:
            value: State values [batch_size, 1]
        """
        # Handle dict/TensorDict input
        if isinstance(critic_observations, dict) or hasattr(critic_observations, 'keys'):
            # Try to get 'policy' key first (common convention)
            if 'policy' in critic_observations.keys():
                critic_tensor = critic_observations['policy']
            else:
                # Otherwise use the first value
                critic_tensor = next(iter(critic_observations.values()))
        else:
            critic_tensor = critic_observations
        
        # 1. Encode observations through shared encoder
        encoded = self.forward_shared_encoder(critic_tensor)
        
        # 2. Pass through critic head to get value
        value = self.critic_head(encoded)
        
        return value
    
    def load_state_dict(self, state_dict, strict=True):
        """
        Load state dict with compatibility for different architectures.
        
        This allows loading checkpoints from separate actor-critic models
        by mapping parameters appropriately.
        """
        try:
            # Try direct loading first
            super().load_state_dict(state_dict, strict=strict)
            print("✅ Loaded state dict successfully")
        except Exception as e:
            print(f"⚠️  Direct loading failed: {e}")
            print("Attempting to load with parameter mapping...")
            
            # Create new state dict with mapped keys
            new_state_dict = {}
            
            for key, value in state_dict.items():
                # Map old actor keys to new structure
                if key.startswith('actor.'):
                    # Try to map to shared_encoder or actor_head
                    new_key = key.replace('actor.', 'shared_encoder.')
                    if new_key in self.state_dict():
                        new_state_dict[new_key] = value
                    else:
                        new_key = key.replace('actor.', 'actor_head.')
                        if new_key in self.state_dict():
                            new_state_dict[new_key] = value
                
                # Map old critic keys to new structure
                elif key.startswith('critic.'):
                    # Try to map to critic_head
                    new_key = key.replace('critic.', 'critic_head.')
                    if new_key in self.state_dict():
                        new_state_dict[new_key] = value
                
                # Keep std/log_std as is
                elif key in ['std', 'log_std']:
                    new_state_dict[key] = value
                
                else:
                    new_state_dict[key] = value
            
            # Load mapped state dict
            super().load_state_dict(new_state_dict, strict=False)
            print("✅ Loaded state dict with parameter mapping")
    
    def get_shared_encoder_output_dim(self):
        """Get the output dimension of the shared encoder."""
        return self.shared_encoder.config.latent_dim
    
    def freeze_shared_encoder(self):
        """Freeze the shared encoder parameters (for fine-tuning)."""
        for param in self.shared_encoder.parameters():
            param.requires_grad = False
        print("🔒 Shared encoder frozen")
    
    def unfreeze_shared_encoder(self):
        """Unfreeze the shared encoder parameters."""
        for param in self.shared_encoder.parameters():
            param.requires_grad = True
        print("🔓 Shared encoder unfrozen")
