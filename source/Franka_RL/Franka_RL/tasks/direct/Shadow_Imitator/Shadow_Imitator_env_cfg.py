# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.sensors import CameraCfg

from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR


import torch

@configclass
class ShandImitatorEnvCfg(DirectRLEnvCfg):
    # env
    episode_length_s = 2.5
    # decimation: Number of control action updates @ sim DT per policy DT
    decimation = 4
    action_joint_scale = 0.1
    action_pos_scale = 500
    action_rot_scale = 20
    action_moving_scale = 1.0
    action_space = 7
    observation_space = 63
    state_space = 63

    future_frame = 5

    # for h1 shadow
    num_dof = 30
    human_delay = 0.0  # delay in seconds
    human_freq = 5
    human_resample_on_env_reset = True
    human_filename = "reorderd_ACCAD_walk_10fps.npy"
    
    # log train info cfg
    log_train_info = False
    # log train info cfg
    log_train_info = False
    info_buffer_size = 500000
    log_dir = './log_info/'

    # simulation
    sim: SimulationCfg = SimulationCfg(dt=0.005, render_interval=decimation)
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="average",
            restitution_combine_mode="average",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        debug_vis=False,
    )

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=2.0, replicate_physics=False)

    # robot
    robot = 'shadow'
    side = 'right'
        
    # dataset
    dataset_type = 'DexYcb'
    dataset_path = ''

    gym2lab_order: list = [0, 5, 10, 1, 6, 11, 15, 2, 7, 12, 16, 3, 8, 13, 17, 4, 9, 14, 18]
    heading_weight: float = 0.5
    up_weight: float = 0.1

    tightenMethod = "exp_decay"
    tightenFactor = 0.7 # 1.0 means no tightening restriction
    tightenSteps = 128000

    energy_cost_scale: float = 0.05
    actions_cost_scale: float = 0.01
    alive_reward_scale: float = 2.0
    dof_vel_scale: float = 0.1

    death_cost: float = -1.0
    termination_height: float = 5.8

    angular_velocity_scale: float = 0.25
    contact_force_scale: float = 0.01