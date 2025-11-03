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

    print("\n" + "="*80)
    print("ENVIRONMENT DEBUG INFORMATION")
    print("="*80)
    
    # ========== 1. Observation Space ==========
    print("\n[1] OBSERVATION SPACE:")
    print(f"  Type: {type(env.observation_space)}")
    print(f"  Shape: {env.observation_space.shape}")
    print(f"  Low: {env.observation_space.low}")
    print(f"  High: {env.observation_space.high}")
    
    # ========== 2. Action Space ==========
    print("\n[2] ACTION SPACE:")
    print(f"  Type: {type(env.action_space)}")
    print(f"  Shape: {env.action_space.shape}")
    print(f"  Num Actions: {env.action_space.shape[-1]}")
    print(f"  Low (min): {env.action_space.low}")
    print(f"  High (max): {env.action_space.high}")
    
    # ========== 3. Robot Joint Information ==========
    print("\n[3] ROBOT JOINT INFORMATION:")
    if hasattr(env.unwrapped, 'robot'):
        robot = env.unwrapped.robot
        print(f"  Robot Type: {type(robot).__name__}")
        print(f"  Joint Names ({len(robot.joint_names)}):")
        for i, joint_name in enumerate(robot.joint_names):
            print(f"    [{i:2d}] {joint_name}")
        
        print(f"\n  Body Names ({len(robot.body_names)}):")
        for i, body_name in enumerate(robot.body_names):
            print(f"    [{i:2d}] {body_name}")
        
        print(f"\n  DOF (Degrees of Freedom): {robot.num_joints}")
    
    # ========== 4. Go2 Robot Configuration ==========
    if 'Go2' in args_cli.task or 'go2' in args_cli.task.lower():
        print("\n[4] GO2 ROBOT CONFIGURATION:")
        try:
            from Franka_RL.robots import QuadrupedRobotFactory
            go2_robot = QuadrupedRobotFactory.create_robot("unitree_go2")
            
            print(f"  DOF Names ({len(go2_robot.dof_names)}):")
            for i, dof_name in enumerate(go2_robot.dof_names):
                print(f"    [{i:2d}] {dof_name}")
            
            print(f"\n  Foot Names:")
            for i, foot_name in enumerate(go2_robot.foot_names):
                print(f"    [{i}] {foot_name}")
            
            print(f"\n  Initial Joint Positions:")
            for joint_name, pos in go2_robot.init_state.joint_pos.items():
                print(f"    {joint_name:20s}: {pos:7.3f} rad")
                
        except Exception as e:
            print(f"  Could not load Go2 config: {e}")
    
    # ========== 5. Reset and Get Initial Observation ==========
    print("\n[5] RESET ENVIRONMENT:")
    obs_dict, _ = env.reset()
    
    print(f"  Observation Type: {type(obs_dict)}")
    if isinstance(obs_dict, dict):
        print(f"  Observation Keys: {list(obs_dict.keys())}")
        for key, value in obs_dict.items():
            if isinstance(value, torch.Tensor):
                print(f"    '{key}': shape={value.shape}, dtype={value.dtype}, device={value.device}")
                print(f"      Range: [{value.min().item():.3f}, {value.max().item():.3f}]")
                print(f"      Mean: {value.mean().item():.3f}, Std: {value.std().item():.3f}")
    else:
        print(f"  Observation Shape: {obs_dict.shape}")
        print(f"  Observation Range: [{obs_dict.min().item():.3f}, {obs_dict.max().item():.3f}]")
    
    print("\n" + "="*80)
    print("STARTING RANDOM ACTION SIMULATION")
    print("="*80 + "\n")
    
    step_count = 0
    print_interval = 50  # Print detailed info every 50 steps
    
    # simulate environment
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            # sample actions from -1 to 1
            actions = 1 * (2 * torch.rand(env.action_space.shape, device=env.unwrapped.device) - 1)
            
            # ========== Print Action Details (every N steps) ==========
            if step_count % print_interval == 0:
                print(f"\n[STEP {step_count}] ACTION DETAILS:")
                print(f"  Actions Shape: {actions.shape}")
                print(f"  Actions Range: [{actions.min().item():.3f}, {actions.max().item():.3f}]")
                print(f"  Actions (Env 0): {actions[0].cpu().numpy()}")
                
                # Print joint-wise actions
                if hasattr(env.unwrapped, 'robot'):
                    print(f"\n  Joint-wise Actions (Env 0):")
                    for i, joint_name in enumerate(env.unwrapped.robot.joint_names[:actions.shape[-1]]):
                        print(f"    [{i:2d}] {joint_name:20s}: {actions[0, i].item():7.3f}")
            
            # apply actions
            obs, reward, terminated, truncated, info = env.step(actions)
            
            # ========== Print Observation Details (every N steps) ==========
            if step_count % print_interval == 0:
                print(f"\n[STEP {step_count}] OBSERVATION DETAILS:")
                if isinstance(obs, dict):
                    for key, value in obs.items():
                        if isinstance(value, torch.Tensor):
                            print(f"  '{key}':")
                            print(f"    Shape: {value.shape}")
                            print(f"    Range: [{value.min().item():.3f}, {value.max().item():.3f}]")
                            print(f"    Mean: {value.mean().item():.3f}, Std: {value.std().item():.3f}")
                            if value.shape[-1] <= 20:  # Only print if small dimension
                                print(f"    Env 0 data: {value[0].cpu().numpy()}")
                else:
                    print(f"  Shape: {obs.shape}")
                    print(f"  Range: [{obs.min().item():.3f}, {obs.max().item():.3f}]")
                
                print(f"\n  Reward (Env 0): {reward[0].item():.3f}")
                print(f"  Terminated (Env 0): {terminated[0].item() if hasattr(terminated[0], 'item') else terminated[0]}")
                print(f"  Truncated (Env 0): {truncated[0].item() if hasattr(truncated[0], 'item') else truncated[0]}")
                
                # Print robot state
                if hasattr(env.unwrapped, 'robot'):
                    robot = env.unwrapped.robot
                    joint_pos = robot.data.joint_pos[0]  # First environment
                    joint_vel = robot.data.joint_vel[0]
                    
                    print(f"\n  Robot State (Env 0):")
                    print(f"    Joint Positions: {joint_pos.cpu().numpy()}")
                    print(f"    Joint Velocities: {joint_vel.cpu().numpy()}")
                    
                    if hasattr(robot.data, 'root_lin_vel_w'):
                        print(f"    Base Linear Vel: {robot.data.root_lin_vel_w[0].cpu().numpy()}")
                    if hasattr(robot.data, 'root_ang_vel_w'):
                        print(f"    Base Angular Vel: {robot.data.root_ang_vel_w[0].cpu().numpy()}")
            
            step_count += 1
            
            
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
