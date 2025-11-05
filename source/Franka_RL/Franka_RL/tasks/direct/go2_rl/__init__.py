# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Go2 locomotion environment registration - Direct RL workflow
"""

import gymnasium as gym
from . import agents

##
# Register Gym environments.
##

gym.register(
    id="Go2-Train",
    entry_point=f"{__name__}.go2_env:Go2Env",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.go2_env_cfg:Go2EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}:rsl_rl_ppo_transformer_cfg.yaml", 
    },
)

gym.register(
    id="Go2-MLP",
    entry_point=f"{__name__}.go2_env:Go2Env",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.go2_env_cfg:Go2EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}:rsl_rl_ppo_dual_mlp_cfg.yaml",
    },
)

gym.register(
    id="Go2-Simple",
    entry_point=f"{__name__}.go2_env:Go2Env",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.go2_env_cfg:Go2EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}:rsl_rl_ppo_simple_mlp_cfg.yaml",
    },
)
