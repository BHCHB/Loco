# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Privileged observation functions for Direct RL workflow."""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from Franka_RL.tasks.direct.go2_rl.go2_env import Go2Env


def get_base_lin_vel(env: Go2Env, add_noise: bool = True) -> torch.Tensor:
    """Get base linear velocity in base frame (privileged observation).
    
    Args:
        env: The Go2 environment instance
        add_noise: Whether to add noise to the observation
        
    Returns:
        Base linear velocity [N, 3]
    """
    # Use pre-computed base_lin_vel_local from _compute_intermediate_values
    base_lin_vel = env.base_lin_vel_local.clone()
    
    if add_noise and "base_lin_vel" in env.cfg.noise_scales_critic:
        noise_scale = env.cfg.noise_scales_critic["base_lin_vel"]
        noise = torch.randn_like(base_lin_vel) * noise_scale
        base_lin_vel += noise
    
    return base_lin_vel


def get_base_ang_vel(env: Go2Env, add_noise: bool = True) -> torch.Tensor:
    """Get base angular velocity in base frame (privileged observation).
    
    Args:
        env: The Go2 environment instance
        add_noise: Whether to add noise to the observation
        
    Returns:
        Base angular velocity [N, 3]
    """
    # Use pre-computed base_ang_vel_local from _compute_intermediate_values
    base_ang_vel = env.base_ang_vel_local.clone()
    
    if add_noise and "base_ang_vel" in env.cfg.noise_scales_critic:
        noise_scale = env.cfg.noise_scales_critic["base_ang_vel"]
        noise = torch.randn_like(base_ang_vel) * noise_scale
        base_ang_vel += noise
    
    return base_ang_vel


def get_projected_gravity(env: Go2Env, add_noise: bool = True) -> torch.Tensor:
    """Get projected gravity vector in base frame (privileged observation).
    
    Args:
        env: The Go2 environment instance
        add_noise: Whether to add noise to the observation
        
    Returns:
        Projected gravity vector [N, 3]
    """
    # Use pre-computed projected_gravity from _compute_intermediate_values
    projected_gravity = env.projected_gravity.clone()
    
    if add_noise and "projected_gravity" in env.cfg.noise_scales_critic:
        noise_scale = env.cfg.noise_scales_critic["projected_gravity"]
        noise = torch.randn_like(projected_gravity) * noise_scale
        projected_gravity += noise
    
    return projected_gravity


def get_height_scan(env: Go2Env, offset: float = 0.5) -> torch.Tensor:
    """Get height scan from RayCaster sensor (privileged observation).
    
    This follows Isaac Lab's mdp.height_scan implementation:
    height = sensor_height - hit_point_z - offset
    
    Args:
        env: The Go2 environment instance
        offset: Height offset to subtract (default: 0.5)
        
    Returns:
        Height scan data [N, num_rays], clipped to [-1.0, 1.0]
    """
    sensor = env.scene.sensors["height_scanner"]
    # height scan: height = sensor_height - hit_point_z - offset
    height_scan = sensor.data.pos_w[:, 2].unsqueeze(1) - sensor.data.ray_hits_w[..., 2] - offset
    # Clip to [-1.0, 1.0] as per standard practice
    height_scan = torch.clamp(height_scan, -1.0, 1.0)
    return height_scan
