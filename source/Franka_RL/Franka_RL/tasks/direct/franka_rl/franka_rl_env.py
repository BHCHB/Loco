# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import os
import random
import pendulum
import json
import zarr
from collections.abc import Sequence

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import sample_uniform, quat_error_magnitude
from isaaclab.markers import VisualizationMarkersCfg, VisualizationMarkers
from isaaclab.markers.config import FRAME_MARKER_CFG

from isaaclab.sensors import Camera

from .franka_rl_env_cfg import FrankaRlEnvCfg

def normalize_angle(x):
    return torch.atan2(torch.sin(x), torch.cos(x))

def sample_int_from_float(x):
    if int(x) == x:
        return int(x)
    return int(x) if np.random.rand() < (x - int(x)) else int(x) + 1

def to_euler_angles(quat):
    assert quat.shape[-1] == 7
    from scipy.spatial.transform import Rotation as R

    r = R.from_quat(quat[3:].cpu().numpy())
    v = quat[:3].cpu().numpy()
    r = r.as_euler("xyz", degrees=False)
    # concat v and r
    return torch.tensor(np.concatenate((v, r), axis=-1), dtype=torch.float32)

class FrankaRlEnv(DirectRLEnv):
    cfg: FrankaRlEnvCfg

    def __init__(self, cfg: FrankaRlEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self.action_scale = self.cfg.action_scale  # type: ignore
        self.joint_gears = torch.tensor(self.cfg.joint_gears, dtype=torch.float32, device=self.sim.device)  # type: ignore
        self.joint_limits = torch.tensor(self.cfg.joint_limitation, dtype=torch.float32, device=self.sim.device)
        self.motor_effort_ratio = torch.ones_like(self.joint_gears, device=self.sim.device)
        self._joint_dof_idx, _ = self.robot.find_joints(".*")

        # human retargeted poses
        self.dt = self.cfg.decimation * self.cfg.sim.dt

        self.future_frame = 5
        self.move_on = 1

        self._init_traj()

        # import omni.replicator.core as rep

        # # create render product
        # self._render_product = rep.create.render_product(
        #     self.cfg.viewer.cam_prim_path, self.cfg.viewer.resolution
        # )
        # # create rgb annotator -- used to read data from the render product
        # self._rgb_annotator = rep.AnnotatorRegistry.get_annotator("rgb", device="cpu")
        # self._rgb_annotator.attach([self._render_product])

    def load_traj_data(self):
        print("[INFO] Process Dataset...")
        self.dataset_path = self.cfg.dataset
        self.traj_split = torch.tensor(np.array(zarr.open(os.path.join(self.dataset_path, 'meta', 'episode_ends'), mode = 'r')), dtype=torch.int, device=self.device)
        target_jt = torch.tensor(np.array(zarr.open(os.path.join(self.dataset_path, 'data', 'state'), mode = 'r')), dtype=torch.float32, device=self.device)[:, :7]
        target_eepose = torch.tensor(np.array(zarr.open(os.path.join(self.dataset_path, 'data', 'eepose'), mode = 'r')), dtype=torch.float32, device=self.device)
        self.traj_num = self.traj_split.shape[-1]
        self.traj_len = torch.zeros_like(self.traj_split, dtype=torch.int, device=self.device)
        self.traj_len[0] = self.traj_split[0]
        self.traj_len[1:] = self.traj_split[1:] - self.traj_split[:-1]
        self.traj_len_max = torch.max(self.traj_len)

        self.target_eepose_seq = torch.zeros(self.traj_num, self.traj_len_max + 30, 7).to(dtype=torch.float32, device=self.device)
        self.target_jt_seq = torch.zeros(self.traj_num, self.traj_len_max + 30, 7).to(dtype=torch.float32, device=self.device)
        count = 0
        for traj_id in range(0, self.traj_num):
            self.target_eepose_seq[traj_id, :self.traj_len[traj_id]] = target_eepose[count : self.traj_split[traj_id]]
            self.target_eepose_seq[traj_id, self.traj_len[traj_id]:] = target_eepose[self.traj_split[traj_id] - 1]
            self.target_jt_seq[traj_id, :self.traj_len[traj_id]] = target_jt[count : self.traj_split[traj_id]]
            self.target_jt_seq[traj_id, self.traj_len[traj_id]:] = target_jt[self.traj_split[traj_id] - 1]
            count = self.traj_split[traj_id]
        print(f"[INFO] Load {self.traj_num} Trajectories from {self.dataset_path}.")

    def _init_traj(self):
        self.load_traj_data()
        if self.cfg.human_resample_on_env_reset:
            self.target_jt_i = torch.randint(0, self.traj_num, (self.num_envs, )).to(dtype=torch.int, device=self.device)
        else:
            self.target_jt_i = torch.zeros(self.num_envs, dtype=torch.int, device=self.device)
        self.target_jt_j = (torch.rand(self.num_envs).to(dtype=torch.float32, device=self.device) * self.traj_len[self.target_jt_i]).to(torch.int)

        self.target_eepose = self.target_eepose_seq[self.target_jt_i, self.target_jt_j]
        self.target_jt = self.target_jt_seq[self.target_jt_i, self.target_jt_j]

        self.target_jt_dt = 1 / self.cfg.human_freq  # type: ignore
        self.target_jt_update_steps = self.target_jt_dt / self.dt  # not necessary integer
        assert self.dt <= self.target_jt_dt
        self.target_jt_update_steps_int = sample_int_from_float(self.target_jt_update_steps)

    def update_target(self, reset_env_ids):
        resample_i = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        if self.cfg.human_resample_on_env_reset:
            resample_i[reset_env_ids] = True
            self.target_jt_i[reset_env_ids] = torch.randint_like(reset_env_ids, low=0, high=self.traj_num).to(dtype=torch.int, device=self.device)
            self.target_jt_j[reset_env_ids] = (torch.rand(reset_env_ids.shape[0]).to(dtype=torch.float32, device=self.device) * self.traj_len[self.target_jt_i[reset_env_ids]]).to(torch.int)
        else:
            self.target_jt_j[reset_env_ids] = (torch.rand(reset_env_ids.shape[0]).to(dtype=torch.float32, device=self.device) * self.traj_len[self.target_jt_i[reset_env_ids]]).to(torch.int)

            self.target_jt = self.target_jt_seq[self.target_jt_i, self.target_jt_j]
            self.target_eepose = self.target_eepose_seq[self.target_jt_i, self.target_jt_j]

        if self.common_step_counter % self.target_jt_update_steps_int == 0:
            if self.common_step_counter == 0:
                self.target_jt_j += 1
            else:
                self.target_jt_j += self.move_on


    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot)
        # add ground plane
        self.cfg.terrain.num_envs = self.scene.cfg.num_envs  # type: ignore
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing  # type: ignore
        self.terrain = self.cfg.terrain.class_type(self.cfg.terrain)  # type: ignore
        # clone, filter, and replicate
        # self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])  # type: ignore
        # add articultion to scene
        self.scene.articulations["robot"] = self.robot
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

        # Create camera
        # self.camera = Camera(self.cfg.camera)
        # self.camera.set_world_poses_from_view(eyes=[2.0, 2.0, 2.0], targets=[0.0, 0.0, 0.0])

        # Markers
        frame_marker_cfg = FRAME_MARKER_CFG.copy()
        frame_marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
        self.ee_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/ee_current"))
        self.goal_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/ee_goal"))


    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        # limit the action in joint limitations
        self.last_actions = self.actions
        # self.actions = (torch.tanh(actions) + 1) / 2 * (self.joint_limits[:, 1] - self.joint_limits[:, 0]) + self.joint_limits[:, 0]
        self.actions = torch.clamp(torch.tanh(actions * self.action_scale) * torch.pi / 6 + self.dof_pos, min = self.joint_limits[:, 0], max=self.joint_limits[:, 1])
        # self.actions = torch.tanh(actions * self.action_scale) * self.joint_gears
        pass

    def _apply_action(self) -> None:
        self.robot.set_joint_position_target(self.actions, joint_ids=range(0, 7))
        # self.robot.set_jointset_joint_effort_target(self.actions, joint_ids=range(0, 7))

    def _compute_intermediate_values(self):
        self.dof_pos, self.dof_vel = self.robot.data.joint_pos, self.robot.data.joint_vel
        self.dof_pos = self.dof_pos[:, :7]
        self.dof_vel = self.dof_vel[:, :7]
        self.eepose = self.robot.data.body_state_w[:, self.robot.body_names.index("panda_hand"), :7] - F.pad(
            self.scene.env_origins, (0, 4)
        )
        self.eepose_diff = self.target_eepose - self.eepose

        self.eepose_error = self.reward_pose(self.eepose, self.target_eepose, tanh_weight=0.0, pos_norm=2)[0]
        

    def _get_observations(self) -> dict:
        joint = self.dof_pos.unsqueeze(1)
        joint_vel = self.dof_vel.unsqueeze(1)
        eepose = self.eepose.unsqueeze(1)
        eepose_diff = self.eepose_diff.unsqueeze(1)
        target_eepose = self.target_eepose_seq[self.target_jt_i.unsqueeze(1), self.target_jt_j.unsqueeze(1) + torch.arange(self.future_frame).to(device=self.device)].reshape(self.num_envs, 1, self.future_frame * 7)
        obs = torch.cat(
            (
                joint,
                joint_vel, 
                eepose,
                eepose_diff, 
                target_eepose,
            ),
            dim=-1,
        )
        # observations = {"policy": {"self_obs": obs.squeeze()}}
        observations = {"policy": obs.squeeze()}
        return observations

    def _get_rewards(self) -> torch.Tensor:
        eepose_reward = torch.exp( 10 * self.reward_pose(self.eepose, self.target_eepose, tanh_weight=0.0, pos_norm=2)[0])

        joint_reward = torch.exp(- torch.norm(self.target_jt - self.dof_pos, p=2, dim=-1))

        died, _ = self._get_dones()
        alive_reward = torch.where(died, -100, 0)

        action_diff = self.actions - self.last_actions
        smooth_reward =  torch.exp(- torch.norm(action_diff, p=2, dim=-1))
        
        # Log Reward
        self._log_reward("Episode_Reward/eepose_reward", eepose_reward)
        self._log_reward("Episode_Reward/joint_reward", joint_reward)
        self._log_reward("Episode_Reward/alive_reward", alive_reward)
        self._log_reward("Episode_Reward/smooth_reward", smooth_reward)

        total_reward = (
            100 * 
            eepose_reward *
            joint_reward +
            alive_reward
            # smooth_reward
        )
        self._log_reward("Episode_Reward/total_reward", total_reward)

        return total_reward

    
    def position_command_error(self, des_pos, curr_pos, std=0.1, p=1):
        """Penalize tracking of the position error using L1-norm.

        The function computes the position error between the desired position (from the command) and the
        current position of the asset's body (in world frame). The position error is computed as the L1-norm
        of the difference between the desired and current positions.
        """
        # obtain the desired and current positions
        distance = torch.norm(curr_pos - des_pos, p=p, dim=-1)
        if len(distance.size()) == 2:
            distance = distance.mean(dim=-1)
        error_tanh = 1 - torch.tanh(distance / std)
        return distance, error_tanh

    def orientation_command_error(self, des_quat, curr_quat) -> torch.Tensor:
        """Penalize tracking orientation error using shortest path.

        The function computes the orientation error between the desired orientation (from the command) and the
        current orientation of the asset's body (in world frame). The orientation error is computed as the shortest
        path between the desired and current orientations.
        """
        magnitude = quat_error_magnitude(curr_quat, des_quat)
        if len(magnitude.size()) == 2:
            magnitude = magnitude.mean(dim=-1)
        return magnitude

    def reward_pose(self, des_pose, curr_pose, tanh_weight=0.1, pos_norm=1):
        pos_error, pos_tanh = self.position_command_error(des_pose[..., :3], curr_pose[..., :3], p=pos_norm)
        ori_error = self.orientation_command_error(des_pose[..., 3:7], curr_pose[..., 3:7])
        # self.pos_error = pos_error
        # self.ori_error = ori_error
        return (pos_error * -0.2 + pos_tanh * tanh_weight + ori_error * -0.1), pos_error, pos_tanh, ori_error

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self._compute_intermediate_values()
        time_out = self.target_jt_j >= self.traj_len[self.target_jt_i]

        died = - self.eepose_error > 0.15
        return died, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        self.robot.reset(env_ids)
        super()._reset_idx(env_ids)

        joint_pos = self.robot.data.default_joint_pos[env_ids]
        joint_pos[:, :7] = self.target_jt[env_ids]
        joint_vel = self.robot.data.default_joint_vel[env_ids]
        
        default_root_state = self.robot.data.default_root_state[env_ids]
        default_root_state[:, :3] += self.scene.env_origins[env_ids]
        
        self.robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self.robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        # this can only be turned on if robot joint is corrected
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        self._compute_intermediate_values()
    
    def _log_reward(self, reward_name, reward_data):

        # add "log" key to extras
        if "log" not in self.extras:
            self.extras["log"] = {}
        self.extras["log"][reward_name] = reward_data
    
    def step(self, action: torch.Tensor):
        action = action.to(self.device)
        # add action noise
        if self.cfg.action_noise_model:
            action = self._action_noise_model.apply(action)

        # process actions
        self._pre_physics_step(action)

        # check if we need to do rendering within the physics loop
        # note: checked here once to avoid multiple checks within the loop
        is_rendering = self.sim.has_gui() or self.sim.has_rtx_sensors()
        # is_rendering = True

        for _ in range(self.cfg.decimation):
            self._sim_step_counter += 1
            # set actions into buffers
            self._apply_action()
            # set actions into simulator
            self.scene.write_data_to_sim()
            # simulate
            self.sim.step(render=False)
            # render between steps only if the GUI or an RTX sensor needs it
            # note: we assume the render interval to be the shortest accepted rendering interval.
            #    If a camera needs rendering at a faster frequency, this will lead to unexpected behavior.
            if self._sim_step_counter % self.cfg.sim.render_interval == 0 and is_rendering:
                self.sim.render()
            # update buffers at sim dt
            self.scene.update(dt=self.physics_dt)

        # Visualize Markers
        self.ee_marker.visualize(self.eepose[:, 0:3] + self.scene.env_origins, self.eepose[:, 3:7])
        self.goal_marker.visualize(self.target_eepose[:, 0:3] + self.scene.env_origins, self.target_eepose[:, 3:7])

        # post-step:
        # -- update env counters (used for curriculum generation)
        self.episode_length_buf += 1  # step in current episode (per env)
        self.common_step_counter += 1  # total step (common for all envs)

        self.reset_terminated[:], self.reset_time_outs[:] = self._get_dones()
        self.reset_buf = self.reset_terminated | self.reset_time_outs
        self.reward_buf = self._get_rewards()

        # -- reset envs that terminated/timed-out and log the episode information
        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        # -- update target joint with reset
        self.update_target(reset_env_ids)
        if len(reset_env_ids) > 0:
            self._reset_idx(reset_env_ids)

        # post-step: step interval event
        if self.cfg.events:
            if "interval" in self.event_manager.available_modes:
                self.event_manager.apply(mode="interval", dt=self.step_dt)

        # update observations
        self.obs_buf = self._get_observations()

        # add observation noise
        # note: we apply no noise to the state space (since it is used for critic networks)
        if self.cfg.observation_noise_model:
            self.obs_buf["policy"] = self._observation_noise_model.apply(self.obs_buf["policy"])

        # return observations, rewards, resets and extras
        return self.obs_buf, self.reward_buf, self.reset_terminated, self.reset_time_outs, self.extras

@torch.jit.script
def compute_rewards(
    rew_scale_alive: float,
    rew_scale_terminated: float,
    rew_scale_pole_pos: float,
    rew_scale_cart_vel: float,
    rew_scale_pole_vel: float,
    pole_pos: torch.Tensor,
    pole_vel: torch.Tensor,
    cart_pos: torch.Tensor,
    cart_vel: torch.Tensor,
    reset_terminated: torch.Tensor,
):
    rew_alive = rew_scale_alive * (1.0 - reset_terminated.float())
    rew_termination = rew_scale_terminated * reset_terminated.float()
    rew_pole_pos = rew_scale_pole_pos * torch.sum(torch.square(pole_pos).unsqueeze(dim=1), dim=-1)
    rew_cart_vel = rew_scale_cart_vel * torch.sum(torch.abs(cart_vel).unsqueeze(dim=1), dim=-1)
    rew_pole_vel = rew_scale_pole_vel * torch.sum(torch.abs(pole_vel).unsqueeze(dim=1), dim=-1)
    total_reward = rew_alive + rew_termination + rew_pole_pos + rew_cart_vel + rew_pole_vel
    return total_reward