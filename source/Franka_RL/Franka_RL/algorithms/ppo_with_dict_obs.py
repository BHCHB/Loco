import torch
from rsl_rl.algorithms import PPO
from rsl_rl.storage import RolloutStorage


class PPOWithDictObs(PPO):
    """PPO algorithm extended to support dict observations and separate actor/critic observations."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def init_storage(self, training_type, num_envs, num_transitions_per_env, obs_dict, actions_shape):
        """Initialize rollout storage with dict-format observations.
        
        Args:
            training_type: "rl" or "distillation"
            num_envs: Number of parallel environments
            num_transitions_per_env: Number of transitions per environment
            obs_dict: Observation dictionary {"policy": tensor, "critic": tensor, ...}
            actions_shape: Action shape (tensor or tuple)
        """
        if isinstance(actions_shape, torch.Tensor):
            actions_shape_tuple = tuple(actions_shape.shape)
        elif isinstance(actions_shape, int):
            actions_shape_tuple = (actions_shape,)
        elif isinstance(actions_shape, (tuple, list)):
            actions_shape_tuple = tuple(actions_shape)
        else:
            actions_shape_tuple = (actions_shape,)
        
        obs_with_batch = {}
        for key, value in obs_dict.items():
            if isinstance(value, torch.Tensor):
                if value.dim() == 1:
                    obs_with_batch[key] = value.unsqueeze(0).expand(num_envs, -1)
                else:
                    obs_with_batch[key] = value
            elif isinstance(value, int):
                obs_with_batch[key] = torch.zeros(num_envs, value, device=self.device)
            elif isinstance(value, (tuple, list)):
                obs_with_batch[key] = torch.zeros(num_envs, *value, device=self.device)
            else:
                raise ValueError(f"Unsupported obs type for key '{key}': {type(value)}")
        
        # Create storage with dict observations
        self.storage = RolloutStorage(
            training_type,
            num_envs,
            num_transitions_per_env,
            obs_with_batch,         
            actions_shape_tuple,
            self.device,
        )
    
    def act(self, obs, privileged_obs=None):
        """Sample actions from the policy with separate actor and critic observations.
    
        Args:
            obs: Policy observations (for actor), tensor or dict
            privileged_obs: Privileged observations (for critic), tensor or dict
    
        Returns:
            actions: Sampled actions
        """
        if not isinstance(obs, dict):
            obs_dict = {"policy": obs}
        else:
            obs_dict = obs
        
        if privileged_obs is not None and not isinstance(privileged_obs, dict):
            privileged_obs_dict = {"critic": privileged_obs}
        else:
            privileged_obs_dict = privileged_obs if privileged_obs is not None else obs_dict
        
        self.transition.observations = obs_dict
        self.transition.privileged_observations = privileged_obs_dict
        
        obs_tensor = obs if isinstance(obs, torch.Tensor) else obs_dict["policy"]
        privileged_obs_tensor = privileged_obs if isinstance(privileged_obs, torch.Tensor) else privileged_obs_dict.get("critic", obs_tensor)
        
        actions = self.policy.act(obs_tensor)
        action_log_prob = self.policy.get_actions_log_prob(actions).detach()
        action_mean = self.policy.action_mean.detach()
        action_std = self.policy.action_std.detach()

        self.transition.actions = actions.detach()
        self.transition.actions_log_prob = action_log_prob
        self.transition.action_mean = action_mean
        self.transition.action_sigma = action_std

        values = self.policy.evaluate(privileged_obs_tensor).detach()
        self.transition.values = values

        if hasattr(self.policy, 'distribution') and self.policy.distribution is not None:
            try:
                if hasattr(self.policy, 'action_std'):
                    std = self.policy.action_std
                    clamped_std = torch.clamp(std, min=1e-6, max=10.0)
                    if not torch.allclose(std, clamped_std):
                        mean = self.policy.action_mean
                        from torch.distributions import Normal
                        self.policy.distribution = Normal(mean, clamped_std)
            except Exception:
                pass
        return actions
    
    def compute_returns(self, last_privileged_obs):
        """Compute returns using privileged observations for value estimation.
        
        Args:
            last_privileged_obs: Last step's privileged observations
        """
        last_values = self.policy.evaluate(last_privileged_obs).detach()
        self.storage.compute_returns(
            last_values, self.gamma, self.lam, normalize_advantage=not self.normalize_advantage_per_mini_batch
        )
