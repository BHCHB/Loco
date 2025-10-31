import gymnasium as gym

from . import agents



from .go2_low_base_cfg import Go2BaseRoughEnvCfg, Go2BaseRoughEnvCfg_PLAY, Go2RoughPPORunnerCfg

##
# Register Gym environments.
##

gym.register(
    id="go2_base",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.go2_low_base_cfg:Go2BaseRoughEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}:rsl_rl_ppo_cfg.yaml",
    },  
)

""""
gym.register(
    id="go2_base_play",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": Go2BaseRoughEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}:rsl_rl_ppo_cfg.yaml",
    },
)
"""