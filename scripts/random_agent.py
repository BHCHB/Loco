# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to an environment with random action agent."""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Random agent for Isaac Lab environments.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

import Franka_RL.tasks  # noqa: F401


def main():
    """Random actions agent with Isaac Lab environment."""
    # create environment configuration
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    # create environment
    env = gym.make(args_cli.task, cfg=env_cfg)

    # print info (this is vectorized environment)
    print(f"[INFO]: Gym observation space: {env.observation_space}")
    print(f"[INFO]: Gym action space: {env.action_space}")
    # 打印reset前的obs
    print("[DEBUG] obs before reset:")
    try:
        obs_before = env.get_obs() if hasattr(env, 'get_obs') else None
        print(obs_before)
    except Exception as e:
        print(f"[DEBUG] Cannot get obs before reset: {e}")
    # reset environment
    obs_after = env.reset()
    print("[DEBUG] obs after reset:")
    print(obs_after)
    # simulate environment
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            # sample actions from -1 to 1
            actions = 1 * (2 * torch.rand(env.action_space.shape, device=env.unwrapped.device) - 1)
            # apply actions
            obs, reward, terminated, truncated, info = env.step(actions)
            
            
            """
            # Print debug info if any episode ends
            if (hasattr(terminated, '__iter__') and any(terminated)) or (hasattr(truncated, '__iter__') and any(truncated)):
                print(f"[DEBUG] terminated: {terminated}, truncated: {truncated}")
                print(f"[DEBUG] info: {info}")
            # Print episode end reasons if available
            if hasattr(info, 'keys') and 'env' in info:
                env_infos = info['env']
                if 'episode_end_reason' in env_infos:
                    reasons = env_infos['episode_end_reason']
                    for i, reason in enumerate(reasons):
                        if reason != 0:
                            print(f"[Episode End] Env {i}: reason={reason}")
            elif isinstance(info, dict) and 'episode_end_reason' in info:
                reasons = info['episode_end_reason']
                for i, reason in enumerate(reasons):
                    if reason != 0:
                        print(f"[Episode End] Env {i}: reason={reason}")
            """

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
