# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

# Import robot factory
from Franka_RL.robots import QuadrupedRobotFactory

##
# Pre-defined configs
##
from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG, FLAT_TERRAINS_CFG  # isort: skip



def get_robot_cfg() -> ArticulationCfg:
    """Get Go2 robot configuration from factory."""
    go2_robot = QuadrupedRobotFactory.create_robot("unitree_go2")
    
    robot_cfg = ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=go2_robot._usd_path,
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                retain_accelerations=False,
                linear_damping=0.0,
                angular_damping=0.0,
                max_linear_velocity=1000.0,
                max_angular_velocity=1000.0,
                max_depenetration_velocity=1.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=4,
                solver_velocity_iteration_count=0,
            ),
        ),
        init_state=go2_robot.init_state,
        actuators=go2_robot.actuators,
    )
    return robot_cfg


@configclass
class Go2EnvCfg(DirectRLEnvCfg):
    """Configuration for the Go2 locomotion environment."""

    
    observation_space = 45
    action_space = 12
    state_space = 238  # 45 (policy) + 3 (base_lin_vel) + 3 (base_ang_vel) + 3 (projected_gravity) + 3 (velocity_commands) + 12 (joint_pos) + 12 (joint_vel) + 12 (actions) + 187 (height_scan: 17x11 grid) - 42 (overlap with policy) = 45 + 193
    future_frame = 5

    # env
    episode_length_s = 20.0
    decimation = 4
    action_scale = 0.25  
    num_envs = 4096
    env_spacing = 5.0

    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 200,
        render_interval=decimation,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )
    

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=FLAT_TERRAINS_CFG,  
        max_init_terrain_level=0,  
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path="{NVIDIA_NUCLEUS_DIR}/Materials/Base/Architecture/Shingles_01.mdl",
            project_uvw=True,
        ),
        debug_vis=False,
    )

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=num_envs, 
        env_spacing=env_spacing, 
        replicate_physics=True
    )

  
    robot: ArticulationCfg = "Franka_RL.tasks.direct.go2_rl.go2_env_cfg:get_robot_cfg"
    
    # Contact sensor for detecting illegal contacts
    contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/.*",
        history_length=3,
        track_air_time=True,
        debug_vis=False,
    )
    
    # Height scanner for terrain perception (privileged observation)
    height_scanner: RayCasterCfg = RayCasterCfg(
        prim_path="/World/envs/env_.*/Robot/base",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment="yaw",  # Updated from deprecated attach_yaw_only=True
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )
    

    # normalization
    obs_scales = {
        "lin_vel": 2.0,
        "ang_vel": 0.25,
        "dof_pos": 0.25,
        "dof_vel": 0.05,
       # "gravity": 1.0,
        "commands": [2.0, 2.0, 0.25],  #  [vx_scale, vy_scale, wz_scale]
    }

    # noise
    noise_scales = {
        "lin_vel": 0.1,
        "ang_vel": 0.2,
        "dof_pos": 0.01,
        "dof_vel": 0.5,
       # "gravity": 0.05,
    }
    
    # noise for critic observations (privileged)
    noise_scales_critic = {
        "base_lin_vel": 0.1,
        "base_ang_vel": 0.2,
        "projected_gravity": 0.05,
    }

   
    # Command configuration (compatible with DirectCommandHelper)
    commands_cfg = {
        "lin_vel_x_range": [-2.0, 2.0],
        "lin_vel_y_range": [-2.0, 2.0],
        "ang_vel_z_range": [-1.0, 1.0],
        "heading_range": [-3.14, 3.14],
        
        # Advanced command features (for DirectCommandHelper)
        "resampling_time_range": [8.0, 12.0],   # Resample commands every 8-12 seconds
        "enable_heading_control": False,         # Disable heading control for basic locomotion
        "enable_standing_envs": True,            # Enable standing environments
        "rel_standing_envs": 0.1,                # 10% of environments will stand still
        "enable_metrics": False,                 # Disable metrics tracking for training
    }
    
    
    init_state_cfg = {
        "pos_range": {
            "x": [-2.5, 2.5],  
            "y": [-2.5, 2.5],  
            "z": [0.32, 0.32],
        },
        "rot_range": {
            "roll": [0.0, 0.0],
            "pitch": [0.0, 0.0],
            "yaw": [-3.14, 3.14],
        },
        "vel_range": {
            "x": [-0.5, 0.5],
            "y": [-0.5, 0.5],
            "z": [0.0, 0.0],
        },
    }
    
    termination_cfg = {
        "base_height_min": 0.1,
        "base_height_max": 3.0,
        "roll_pitch_max": 1.0,  
    }
    
    # termination 
    termination_height = 0.15
    
    # LocoFormer-style Reward Weights
    reward_weights = {
        # Main tracking rewards (positive)
        "track_lin_vel_xy_exp": 1.5,       # Linear velocity tracking (exponential)
        "track_ang_vel_z_exp": 0.8,        # Angular velocity tracking (exponential)

        # Orientation and motion stability (negative)
        "flat_orientation": -0.5,          # Penalize body tilt (gravity vector error)
        "lin_vel_z_l2": -0.05,              # Penalize vertical velocity (suppress jumping/sinking)
        "ang_vel_xy_l2": -0.05,            # Penalize roll/pitch rates (suppress body rotation)

        # Energy and smoothness (negative)
        "torque_l2": -1.0e-4,              # Penalize torque squared (energy efficiency)
        "action_rate_l2": -0.5e-3,         # Penalize action changes (encourage smooth control)

        # Survival and constraints (positive/negative)
        "alive_bonus": 0.5,                # Reward for staying alive each step
        "joint_limit": -0.3,               # Penalize joint limit violations
        "illegal_contact": -2.0,           # Penalize non-foot body contacts
        "termination_penalty": -5.0,       # Penalize episode termination (falling)
    }
    
    # LocoFormer-style Reward Parameters
    reward_params = {
        # Exponential reward temperature (variance for Gaussian kernel)
        "velocity_tracking_std_lin": 0.25,  # Linear velocity tracking sensitivity
        "velocity_tracking_std_ang": 0.25,  # Angular velocity tracking sensitivity
        
        # Constraint thresholds
        "joint_limit_margin": 0.1,          # Joint deviation margin (rad) before penalty
        "illegal_contact_threshold": 1.0,   # Contact force threshold (N) for illegal contacts
    }
    
    def __post_init__(self):
        """Post initialization."""
    
        if isinstance(self.robot, str):
            
            self.robot = get_robot_cfg()
        
       
        self.scene.robot = self.robot

