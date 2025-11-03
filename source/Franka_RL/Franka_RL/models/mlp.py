import torch
from torch import nn, Tensor
from hydra.utils import instantiate
from Franka_RL.models.common import NormObsBase
from Franka_RL.utils import model_utils


def build_mlp(config, num_in: int, num_out: int):
    indim = num_in
    layers = []
    for i, layer in enumerate(config.layers):
        layers.append(nn.Linear(indim, layer.units))
        if layer.use_layer_norm and i == 0:
            layers.append(nn.LayerNorm(layer.units))
        layers.append(model_utils.get_activation_func(layer.activation))
        indim = layer.units

    layers.append(nn.Linear(indim, num_out))
    return nn.Sequential(*layers)


class MLP(nn.Module):
    def __init__(self, config, num_in: int, num_out: int):
        super().__init__()
        self.config = config
        self.mlp = build_mlp(self.config, num_in, num_out)

    def forward(self, input, *args, **kwargs):
        return self.mlp(input)


class MLP_WithNorm(NormObsBase):
    def __init__(self, config, num_in: int, num_out: int):
        super().__init__(config, num_in, num_out)
        self.mlp = build_mlp(self.config, num_in, num_out)

    def forward(self, input, return_norm_obs=False):
        obs = super().forward(input)
        outs: Tensor = self.mlp(obs)

        if return_norm_obs:
            return {"outs": outs, f"norm_{self.config.obs_key}": obs}
        else:
            return outs


class MultiHeadedMLP(nn.Module):
    def __init__(self, config, num_out: int):
        super().__init__()
        self.config = config
        self.num_out = num_out
        self.obs_slice = {}
        self.obs_key_map = {}  # Map head name to obs_key

        input_models = {}
        self.feature_size = 0
        for key, input_cfg in self.config.input_models.items():
            model = instantiate(input_cfg)
            input_models[key] = model
            # Use head name (key) as slice index to avoid conflicts when multiple heads share same obs_key
            self.obs_slice[key] = slice(input_cfg.config.slice_start_idx, input_cfg.config.slice_end_idx, input_cfg.config.get("slice_step", 1))
            self.obs_key_map[key] = input_cfg.config.obs_key
            self.feature_size += model.num_out
        self.input_models = nn.ModuleDict(input_models)

        self.trunk: MLP = instantiate(self.config.trunk, num_in=self.feature_size)

    def forward(self, input, return_norm_obs=False):
        # Handle dict/TensorDict input
        if isinstance(input, dict) or hasattr(input, 'keys'):
            # For multi-head critic, we need to extract based on obs_key from each head
            # Since all heads might use the same obs_key (e.g., "critic"), we extract once
            pass  # Will extract per-head below
        else:
            input_tensor = input
        
        if return_norm_obs:
            norm_obs = {}
        outs = []

        for key, model in self.input_models.items():
            # Extract tensor based on this head's obs_key
            if isinstance(input, dict) or hasattr(input, 'keys'):
                obs_key = self.obs_key_map[key]
                if obs_key in input.keys():
                    input_tensor = input[obs_key]
                elif 'policy' in input.keys():
                    input_tensor = input['policy']
                else:
                    input_tensor = next(iter(input.values()))
            else:
                input_tensor = input
            
            # Handle different tensor dimensions and slice using head name (key)
            if input_tensor.ndim == 1:
                sliced = input_tensor[self.obs_slice[key]]
            elif input_tensor.ndim >= 2:
                sliced = input_tensor[:, self.obs_slice[key]]
            else:
                raise ValueError(f"Unexpected tensor dimension: {input_tensor.ndim}")
            
            out = model(sliced, return_norm_obs=return_norm_obs)
            if return_norm_obs:
                out, norm_obs[f"norm_{model.config.obs_key}_{key}"] = (
                    out["outs"],
                    out[f"norm_{model.config.obs_key}"],
                )
            outs.append(out)

        outs = torch.cat(outs, dim=-1)
        outs: Tensor = self.trunk(outs)

        if return_norm_obs:
            ret_dict = {**{"outs": outs}, **norm_obs}
            return ret_dict
        else:
            return outs
