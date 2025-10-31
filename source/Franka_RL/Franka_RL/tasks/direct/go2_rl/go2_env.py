# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Go2 quadruped locomotion environment using Direct RL workflow."""

from __future__ import annotations

import torch
import math
from collections.abc import Sequence

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import quat_apply_inverse, quat_apply, sample_uniform

from .go2_env_cfg import Go2EnvCfg
from Franka_RL.utils.command_helper import DirectCommandHelper, CommandConfig


class Go2Env(DirectRLEnv):
    """Go2 quadruped locomotion environment."""
    
    cfg: Go2EnvCfg
    
    def __init__(self, cfg: Go2EnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        
        self.action_scale = self.cfg.action_scale
        self.dt = self.cfg.decimation * self.cfg.sim.dt
        self.default_joint_pos = None
        self.current_learning_iteration = 0  

        # Initialize command system (use DirectCommandHelper for automatic management)
        self._init_commands()
        
        # Load reward weights and parameters from config
        self.reward_weights = self.cfg.reward_weights
        self.reward_params = self.cfg.reward_params
        
        self.obs_scales = self.cfg.obs_scales
        
        self._joint_dof_idx = None
        self._base_idx = None
        self._feet_indices = None
        
        self.last_actions = torch.zeros(self.num_envs, 12, dtype=torch.float32, device=self.device)
        self.last_dof_vel = torch.zeros(self.num_envs, 12, dtype=torch.float32, device=self.device)
        self._latest_policy_obs_seq: torch.Tensor | None = None
        
        # Store raw actions for action rate penalty
        self.raw_actions = torch.zeros(self.num_envs, 12, dtype=torch.float32, device=self.device)
        self.last_raw_actions = torch.zeros(self.num_envs, 12, dtype=torch.float32, device=self.device)
        
    def _init_commands(self):
        """Initialize the velocity command system using DirectCommandHelper."""
        # Create CommandConfig from environment configuration
        command_cfg = CommandConfig(
            lin_vel_x_range=tuple(self.cfg.commands_cfg["lin_vel_x_range"]),
            lin_vel_y_range=tuple(self.cfg.commands_cfg["lin_vel_y_range"]),
            ang_vel_z_range=tuple(self.cfg.commands_cfg["ang_vel_z_range"]),
            heading_range=tuple(self.cfg.commands_cfg["heading_range"]),
            resampling_time_range=tuple(self.cfg.commands_cfg["resampling_time_range"]),
            enable_heading_control=self.cfg.commands_cfg["enable_heading_control"],
            enable_standing_envs=self.cfg.commands_cfg["enable_standing_envs"],
            rel_standing_envs=self.cfg.commands_cfg["rel_standing_envs"],
            enable_metrics=self.cfg.commands_cfg["enable_metrics"],
        )
        
        # Create DirectCommandHelper instance
        self.command_helper = DirectCommandHelper(
            num_envs=self.num_envs,
            device=self.device,
            cfg=command_cfg,
        )
        
        # Compatibility: maintain commands buffer for existing code
        self.commands = torch.zeros(
            self.num_envs, 4, dtype=torch.float32, device=self.device
        )
    def set_learning_iteration(self, iteration: int):
        """Set current learning iteration (called by runner)."""
        self.current_learning_iteration = iteration
        
    def _setup_scene(self):
        """Set up the simulation scene."""
        self.robot = Articulation(self.cfg.robot)
        self._contact_sensor = ContactSensor(self.cfg.contact_sensor)
        
        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self.terrain = self.cfg.terrain.class_type(self.cfg.terrain)
        
        self.scene.articulations["robot"] = self.robot
        self.scene.sensors["contact_sensor"] = self._contact_sensor
        self.scene.clone_environments(copy_from_source=False)
        
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])
        
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

        self._joint_dof_idx = None
        self._base_idx = None
        self._feet_indices = None
    
    def _initialize_indices(self):
        """Initialize robot joint/body indices and default joint positions."""
        if self._joint_dof_idx is not None:
            return

        self._joint_dof_idx, _ = self.robot.find_joints(".*")
        self._base_idx, _ = self.robot.find_bodies("base")
        self._base_idx = self._base_idx[0]

        fl_idx, _ = self.robot.find_bodies("FL_foot")
        fr_idx, _ = self.robot.find_bodies("FR_foot")
        rl_idx, _ = self.robot.find_bodies("RL_foot")
        rr_idx, _ = self.robot.find_bodies("RR_foot")

        self._feet_indices = [fl_idx[0], fr_idx[0], rl_idx[0], rr_idx[0]]

        if self.default_joint_pos is None:
            from Franka_RL.robots import QuadrupedRobotFactory
            go2_robot = QuadrupedRobotFactory.create_robot("unitree_go2")
            joint_pos_dict = go2_robot.init_state.joint_pos
            joint_pos_list = [joint_pos_dict[name] for name in go2_robot.dof_names]
            self.default_joint_pos = torch.tensor(joint_pos_list, dtype=torch.float32, device=self.device)
    
    def step(self, action: torch.Tensor):
        """Execute one time-step of the environment's dynamics with automatic command management."""
        # Update command system (automatic resampling and standing environment handling)
       
        if self.common_step_counter % 500 == 0:
            print(f"[Go2Env.step] Step {self.common_step_counter}, Env0 commands: {self.commands[0, :3]}")

        if self.cfg.commands_cfg["enable_heading_control"]:
            self.command_helper.update(self.dt, self.robot)
        else:
            self.command_helper.update(self.dt)
        # Update commands buffer for compatibility

        self.commands = self.command_helper.get_commands_with_heading()
        
        # Continue with normal step logic   不确定新command是否成功进入obs空间
        return super().step(action)
            
    def _pre_physics_step(self, actions: torch.Tensor):
        """Pre-process actions before the physics step."""
        if self._joint_dof_idx is None:
            self._initialize_indices()

        # Store raw actions for action rate calculation
        self.last_raw_actions = self.raw_actions.clone()
        self.raw_actions = actions.clone()
        
        # Store previous joint position targets
        self.last_actions = self.actions.clone() if hasattr(self, 'actions') else self.dof_pos
        
        # Compute joint position targets
        joint_pos_target = self.dof_pos + actions * self.action_scale
        self.actions = joint_pos_target

    def _apply_action(self):
        """Apply actions to the robot."""
        self.robot.set_joint_position_target(self.actions, joint_ids=self._joint_dof_idx)
        
    def _compute_intermediate_values(self):
        """Compute intermediate values for rewards and observations."""
        if self._joint_dof_idx is None:
            self._initialize_indices()

        self.dof_pos = self.robot.data.joint_pos[:, :12]
        self.dof_vel = self.robot.data.joint_vel[:, :12]

        base_state = self.robot.data.root_state_w
        self.base_pos = base_state[:, :3]
        self.base_quat = base_state[:, 3:7]
        self.base_lin_vel = base_state[:, 7:10]
        self.base_ang_vel = base_state[:, 10:13]

        gravity_vec = torch.tensor([[0.0, 0.0, -1.0]], device=self.device).repeat(self.num_envs, 1)
        self.projected_gravity = quat_apply_inverse(self.base_quat, gravity_vec)
        self.base_lin_vel_local = quat_apply_inverse(self.base_quat, self.base_lin_vel)
        self.base_ang_vel_local = quat_apply_inverse(self.base_quat, self.base_ang_vel)

        forward_vec = quat_apply(self.base_quat, torch.tensor([1.0, 0.0, 0.0], device=self.device).repeat(self.num_envs, 1))
        self.heading = torch.atan2(forward_vec[:, 1], forward_vec[:, 0])
        self.heading_error = torch.abs(self.heading - self.commands[:, 3])
        
        # Get velocity commands from helper (updated in step())
        self.lin_vel_error = self.base_lin_vel_local[:, :2] - self.commands[:, :2]
        self.ang_vel_error = self.base_ang_vel_local[:, 2] - self.commands[:, 2]

        body_states = self.robot.data.body_state_w
        feet_pos_list = []
        for foot_idx in self._feet_indices:
            feet_pos_list.append(body_states[:, foot_idx, :3])
        self.feet_pos = torch.stack(feet_pos_list, dim=1)
        self.feet_contact = (self.feet_pos[:, :, 2] < 0.02).float()

        self.base_height = self.base_pos[:, 2]
        self.dof_acc = (self.dof_vel - self.last_dof_vel) / self.dt
        self.last_dof_vel = self.dof_vel.clone()
        
    def _get_observations(self) -> dict:
        """Compute observations."""
        self._compute_intermediate_values()

        obs_list = []
        base_lin_vel_obs = self.base_lin_vel_local * self.obs_scales["lin_vel"]
        obs_list.append(base_lin_vel_obs)

        base_ang_vel_obs = self.base_ang_vel_local * self.obs_scales["ang_vel"]
        obs_list.append(base_ang_vel_obs)

        commands_obs = self.commands[:, :3] * torch.tensor(self.obs_scales["commands"], device=self.device)
        obs_list.append(commands_obs)

        dof_pos_normalized = (self.dof_pos - self.default_joint_pos.unsqueeze(0)) * self.obs_scales["dof_pos"]
        obs_list.append(dof_pos_normalized)

        dof_vel_scaled = self.dof_vel * self.obs_scales["dof_vel"]
        obs_list.append(dof_vel_scaled)

        obs_list.append(self.last_actions)
        
        obs = torch.cat(obs_list, dim=-1)
        self._latest_policy_obs_seq = obs.unsqueeze(1)

        observations = {"policy": obs}
        self.extras["observations"] = observations
        return observations
        
    def _get_rewards(self) -> torch.Tensor:
        """LocoFormer-style rewards: exponential tracking + regularization penalties."""
        self._compute_intermediate_values()

        # Velocity tracking (exponential rewards)
        lin_err_sq = torch.sum(self.lin_vel_error ** 2, dim=1)
        lin_track = torch.exp(-lin_err_sq / self.reward_params["velocity_tracking_std_lin"]) \
                    * self.reward_weights["track_lin_vel_xy_exp"]
        
        ang_err_sq = self.ang_vel_error ** 2
        ang_track = torch.exp(-ang_err_sq / self.reward_params["velocity_tracking_std_ang"]) \
                    * self.reward_weights["track_ang_vel_z_exp"]

        # Orientation stability (gravity vector error)
        gravity_target = torch.tensor([0.0, 0.0, -1.0], device=self.device).repeat(self.num_envs, 1)
        gravity_err_sq = torch.sum((self.projected_gravity - gravity_target) ** 2, dim=1)
        r_flat = gravity_err_sq * self.reward_weights["flat_orientation"]

        # Vertical velocity penalty (suppress jumping/sinking)
        lin_vel_z = self.base_lin_vel_local[:, 2]
        r_lin_vel_z = (lin_vel_z ** 2) * self.reward_weights["lin_vel_z_l2"]

        # Roll/pitch rate penalty (suppress body rotation)
        ang_vel_xy = self.base_ang_vel_local[:, :2]
        ang_vel_xy_sq = torch.sum(ang_vel_xy ** 2, dim=1)
        r_ang_vel_xy = ang_vel_xy_sq * self.reward_weights["ang_vel_xy_l2"]

        # Energy penalty (torque squared)
        tau = self.robot.data.applied_torque[:, :12]
        torque_l2 = torch.sum(tau * tau, dim=1)
        r_energy = torque_l2 * self.reward_weights["torque_l2"]

        # Action smoothness penalty (L2 norm of action changes)
        da = self.raw_actions - self.last_raw_actions
        action_rate_l2 = torch.sum(da * da, dim=1)
        r_action_rate = action_rate_l2 * self.reward_weights["action_rate_l2"]
        """
        # ---------- 关节越界（|q - q_default| 超 margin 的超额部分） ----------
        margin = self.reward_params["joint_limit_margin"]
        q_err = torch.abs(self.dof_pos - self.default_joint_pos.unsqueeze(0)) - margin
        q_violate = torch.clamp(q_err, min=0.0)
        r_joint_limit = torch.sum(q_violate, dim=1) * self.reward_weights["joint_limit"]

        # ---------- 非法碰撞（base/hip/calf 等非足端接触力超阈） ----------
        illegal = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)

        base_hit = self._check_illegal_contact_by_name(["base"], threshold=self.reward_params["illegal_contact_threshold"])
        illegal += base_hit.float()

        hip_hit = self._check_illegal_contact_by_name(
            ["FL_hip", "FR_hip", "RL_hip", "RR_hip"],
            threshold=self.reward_params["illegal_contact_threshold"]
        )
        calf_hit = self._check_illegal_contact_by_name(
            ["FL_calf", "FR_calf", "RL_calf", "RR_calf"],
            threshold=self.reward_params["illegal_contact_threshold"]
        )
        illegal += hip_hit.float() + calf_hit.float()
        r_illegal = illegal * self.reward_weights["illegal_contact"]
        """
        # Survival bonus (fixed reward per timestep for alive agents)
        died, _ = self._get_dones()
        alive_mask = ~died  # Boolean mask: True for alive envs
        r_alive = alive_mask.float() * self.reward_weights["alive_bonus"]

        # Termination penalty
        r_terminal = died.float() * self.reward_weights["termination_penalty"]

        # Total reward
        total_reward = (
            lin_track + 
            ang_track + 
            r_flat + 
            r_lin_vel_z + 
            r_ang_vel_xy + 
            r_energy + 
            r_action_rate + 
            r_alive + 
            #r_joint_limit + 
            #r_illegal + 
            r_terminal
        )

        # Logging
        self.extras["log"] = {
            "rewards/lin_track": lin_track.mean(),
            "rewards/ang_track": ang_track.mean(),
            "rewards/flat": r_flat.mean(),
            "rewards/lin_vel_z_l2": r_lin_vel_z.mean(),
            "rewards/ang_vel_xy_l2": r_ang_vel_xy.mean(),
            "rewards/energy_tau_l2": r_energy.mean(),
            "rewards/action_rate_l2": r_action_rate.mean(),
            "rewards/alive_bonus": r_alive.mean(),
            #"rewards/joint_limit": r_joint_limit.mean(),
            #"rewards/illegal": r_illegal.mean(),
            "rewards/termination": r_terminal.mean(),
            "rewards/total": total_reward.mean(),
            "debug/num_died": died.sum(),
            "debug/survival_rate": alive_mask.float().mean(),  # Ratio of alive envs
            #"debug/illegal_contacts": illegal.sum(),
        }

        # Optional: command tracking metrics
        if self.cfg.commands_cfg["enable_metrics"]:
            metrics = self.command_helper.get_metrics()
            if metrics:
                self.extras["log"]["commands/error_vel_xy"] = metrics["error_vel_xy"].mean()
                self.extras["log"]["commands/error_vel_yaw"] = metrics["error_vel_yaw"].mean()
        
        # Log standing environments ratio
        if self.cfg.commands_cfg["enable_standing_envs"]:
            standing_ratio = self.command_helper.is_standing_env.float().mean()
            self.extras["log"]["commands/standing_ratio"] = standing_ratio

        # OLD reward function (commented out for reference)
        """
        # OLD: Base height reward (relative to environment origin)
        target_height = self.reward_params["target_height"]
        relative_height = self.base_pos[:, 2] - self.scene.env_origins[:, 2]
        height_error_squared = torch.square(relative_height - target_height)
        height_reward = height_error_squared * self.reward_weights["base_height_reward"]

        # OLD: Feet air time reward (always positive)
        cmd_norm = torch.norm(self.commands[:, :2], dim=1)
        is_moving = (cmd_norm > self.reward_params["moving_threshold"]).float()
        
        feet_air_time = self._contact_sensor.data.last_air_time[:, self._feet_indices]
        first_contact = self._contact_sensor.compute_first_contact(self.dt)[:, self._feet_indices]
        
        threshold = self.reward_params["feet_air_time_threshold"]
        feet_air_time_reward = torch.sum(
            torch.clamp(feet_air_time - threshold, min=0.0) * first_contact, 
            dim=1
        )
        feet_air_time_reward *= is_moving * self.reward_weights["feet_air_time"]

        # OLD: Joint deviation penalty (separate hip and other joints)
        hip_indices = [0, 1, 2, 3]
        hip_deviation = torch.sum(torch.abs(
            self.dof_pos[:, hip_indices] - self.default_joint_pos[hip_indices].unsqueeze(0)
        ), dim=1) * self.reward_weights["hip_deviation"]
        
        thigh_calf_indices = [4, 5, 6, 7, 8, 9, 10, 11]
        joint_deviation = torch.sum(torch.abs(
            self.dof_pos[:, thigh_calf_indices] - self.default_joint_pos[thigh_calf_indices].unsqueeze(0)
        ), dim=1) * self.reward_weights["joint_deviation"]

        # OLD: Power penalty (|P| = |τ * qdot|)
        applied_torques = self.robot.data.applied_torque[:, :12]
        power = applied_torques * self.dof_vel
        power_reward = torch.sum(torch.abs(power), dim=1) * self.reward_weights["joint_power"]

        # OLD: DOF acceleration penalty
        # dof_acc_reward = torch.sum(torch.square(self.dof_acc), dim=1) * self.reward_weights["dof_acc_l2"]

        # OLD: Action smoothness penalties (L1 norm)
        # action_diff = self.raw_actions - self.last_raw_actions
        # action_smoothness_reward = torch.sum(torch.abs(action_diff), dim=1) * self.reward_weights["action_smoothness"]
        """

        return total_reward
        
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute termination conditions with curriculum."""
        # Calculate training progress (0.0 at start, 1.0 at 10000 iterations)
        training_progress = min(self.current_learning_iteration / 10000, 1.0)
        

        # Base checks 
        base_contact = self._check_illegal_contact_by_name(["base"], threshold=5.0)
        
        # Height bounds
        relative_height = self.base_pos[:, 2] - self.scene.env_origins[:, 2]
        out_of_bounds = (relative_height < 0.01) | (relative_height > 5.0)
        
        terminated = base_contact | out_of_bounds
        
        # Gradually enable additional checks (only after 50% training)
        if training_progress > 0.5:
            # Enable tilt check
            tilted = torch.abs(self.projected_gravity[:, 2]) < torch.cos(
                torch.tensor(self.cfg.termination_cfg["roll_pitch_max"], device=self.device)
            )
            terminated = terminated | tilted
    
        # Enable calf/hip checks only after 75% training
        if training_progress > 0.75:
            hip_contact = self._check_illegal_contact_by_name(
                ["FL_hip", "FR_hip", "RL_hip", "RR_hip"], threshold=1.0
            )
            calf_contact = self._check_illegal_contact_by_name(
                ["FL_calf", "FR_calf", "RL_calf", "RR_calf"], threshold=1.0
            )
            terminated = terminated | hip_contact | calf_contact
    
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        return terminated, time_out
    
    def _check_illegal_contact(self, body_indices: list, threshold: float = 1.0) -> torch.Tensor:
        """Check if specified body parts have illegal contact (contact force exceeds threshold)."""
        net_contact_forces = self._contact_sensor.data.net_forces_w_history
        selected_forces = net_contact_forces[:, :, body_indices, :]
        contact_force_magnitude = torch.norm(selected_forces, dim=-1)
        max_contact_force = torch.max(contact_force_magnitude, dim=1)[0]
        illegal_contact = torch.any(max_contact_force > threshold, dim=1)
        
        return illegal_contact
    
    def _check_illegal_contact_by_name(self, body_names: list[str], threshold: float = None) -> torch.Tensor:
        """Check illegal contact by body names."""
        if threshold is None:
            threshold = self.reward_params["illegal_contact_threshold"]
            
        body_indices = []
        for name in body_names:
            try:
                idx = self._contact_sensor.body_names.index(name)
                body_indices.append(idx)
            except ValueError:
                pass
        
        if not body_indices:
            return torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        
        return self._check_illegal_contact(body_indices, threshold)
    
    def _reset_idx(self, env_ids: Sequence[int] | None):
        """Reset specified environments."""
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self.robot._ALL_INDICES

        self.robot.reset(env_ids)
        
        pos_range = self.cfg.init_state_cfg["pos_range"]
        # Start with environment origins (terrain height)
        init_pos = self.scene.env_origins[env_ids].clone()
        # Add random x, y offsets
        init_pos[:, 0] += sample_uniform(pos_range["x"][0], pos_range["x"][1], (len(env_ids),), self.device)
        init_pos[:, 1] += sample_uniform(pos_range["y"][0], pos_range["y"][1], (len(env_ids),), self.device)
        # Set absolute z height above terrain
        init_pos[:, 2] += pos_range["z"][0]  # 0.32m above terrain

        rot_range = self.cfg.init_state_cfg["rot_range"]
        yaw = sample_uniform(rot_range["yaw"][0], rot_range["yaw"][1], (len(env_ids),), self.device)
        init_quat = torch.zeros(len(env_ids), 4, device=self.device)
        init_quat[:, 0] = torch.cos(yaw / 2)
        init_quat[:, 3] = torch.sin(yaw / 2)

        vel_range = self.cfg.init_state_cfg["vel_range"]
        init_vel = torch.zeros(len(env_ids), 6, device=self.device)
        init_vel[:, 0] = sample_uniform(vel_range["x"][0], vel_range["x"][1], (len(env_ids),), self.device)
        init_vel[:, 1] = sample_uniform(vel_range["y"][0], vel_range["y"][1], (len(env_ids),), self.device)

        if self.default_joint_pos is None:
            from Franka_RL.robots import QuadrupedRobotFactory
            go2_robot = QuadrupedRobotFactory.create_robot("unitree_go2")
            joint_pos_dict = go2_robot.init_state.joint_pos
            joint_pos_list = [joint_pos_dict[name] for name in go2_robot.dof_names]
            self.default_joint_pos = torch.tensor(joint_pos_list, dtype=torch.float32, device=self.device)

        joint_pos = self.default_joint_pos.unsqueeze(0).repeat(len(env_ids), 1)
        joint_pos += sample_uniform(-0.05, 0.05, joint_pos.shape, self.device)  # Reduced from ±0.1 to ±0.05
        joint_vel = torch.zeros(len(env_ids), 12, device=self.device)

        root_state = torch.cat([init_pos, init_quat, init_vel], dim=1)
        self.robot.write_root_state_to_sim(root_state, env_ids)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        # Reset commands using DirectCommandHelper
        self.command_helper.reset(env_ids)
        self.commands = self.command_helper.get_commands_with_heading()

        self.last_actions[env_ids] = 0.0
        self.last_dof_vel[env_ids] = 0.0
        self.raw_actions[env_ids] = 0.0
        self.last_raw_actions[env_ids] = 0.0
        
        # Reset episode length buffer (critical fix for DirectRLEnv)
        self.episode_length_buf[env_ids] = 0
