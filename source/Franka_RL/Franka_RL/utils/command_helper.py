# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Command Helper for Direct RL Environments.

这个模块提供了类似Manager-Based的Command系统功能，但可以在Direct环境中使用。
支持自动时间管理、命令重采样、静止环境和heading控制等功能。
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING
from dataclasses import dataclass

from isaaclab.utils.math import sample_uniform, wrap_to_pi

if TYPE_CHECKING:
    from isaaclab.assets import Articulation

__all__ = ["DirectCommandHelper", "CommandConfig"]


@dataclass
class CommandConfig:
    """命令系统配置类"""
    
    # 命令范围
    lin_vel_x_range: tuple[float, float] = (-2.0, 2.0)
    lin_vel_y_range: tuple[float, float] = (-2.0, 2.0)
    ang_vel_z_range: tuple[float, float] = (-1.5, 1.5)
    heading_range: tuple[float, float] = (-3.14, 3.14)
    
    # 时间管理
    resampling_time_range: tuple[float, float] = (8.0, 12.0)  # 8-12秒随机重采样
    
    # 特殊模式
    enable_heading_control: bool = False      # 是否启用heading控制
    heading_control_stiffness: float = 0.5    # heading控制的P增益
    rel_heading_envs: float = 1.0             # 使用heading控制的环境比例（0-1）
    
    enable_standing_envs: bool = False        # 是否启用静止环境
    rel_standing_envs: float = 0.1            # 静止环境的比例（0-1）
    
    # 误差追踪
    enable_metrics: bool = False              # 是否启用误差追踪


class DirectCommandHelper:
    """
    Direct环境的命令辅助类，提供类似Manager-Based的功能。
    
    功能特性：
    - ✅ 自动时间管理和重采样
    - ✅ 静止环境支持（可选）
    - ✅ Heading控制支持（可选）
    - ✅ 误差追踪（可选）
    - ✅ 完全独立，不依赖Manager框架
    
    使用示例：
        >>> cfg = CommandConfig(
        ...     lin_vel_x_range=(-2.0, 2.0),
        ...     enable_heading_control=True,
        ...     enable_standing_envs=True,
        ... )
        >>> helper = DirectCommandHelper(num_envs=4096, device="cuda", cfg=cfg)
        >>> 
        >>> # 在环境step中调用
        >>> def step(self, action):
        ...     self.command_helper.update(self.step_dt, self.robot)
        ...     commands = self.command_helper.get_commands()
    """
    
    def __init__(
        self,
        num_envs: int,
        device: str | torch.device,
        cfg: CommandConfig | None = None,
    ):
        """
        初始化命令辅助类。
        
        Args:
            num_envs: 环境数量
            device: 计算设备
            cfg: 命令配置，如果为None则使用默认配置
        """
        self.num_envs = num_envs
        self.device = device
        self.cfg = cfg if cfg is not None else CommandConfig()
        
        # 命令缓冲区 [num_envs, 3] = [vx, vy, ωz]
        self.vel_command_b = torch.zeros(num_envs, 3, dtype=torch.float32, device=device)
        
        # Heading相关（如果启用）
        if self.cfg.enable_heading_control:
            self.heading_target = torch.zeros(num_envs, dtype=torch.float32, device=device)
            self.is_heading_env = torch.zeros(num_envs, dtype=torch.bool, device=device)
        
        # 静止环境标志（如果启用）
        if self.cfg.enable_standing_envs:
            self.is_standing_env = torch.zeros(num_envs, dtype=torch.bool, device=device)
        
        # 时间管理
        self.command_time_left = torch.zeros(num_envs, dtype=torch.float32, device=device)
        
        # 误差追踪（如果启用）
        if self.cfg.enable_metrics:
            self.metrics = {
                "error_vel_xy": torch.zeros(num_envs, dtype=torch.float32, device=device),
                "error_vel_yaw": torch.zeros(num_envs, dtype=torch.float32, device=device),
            }
    
    def get_commands(self) -> torch.Tensor:
        """
        获取当前的速度命令。
        
        Returns:
            命令张量 [num_envs, 3]，包含 [vx, vy, ωz]
        """
        return self.vel_command_b
    
    def get_commands_with_heading(self) -> torch.Tensor:
        """
        获取命令（包含heading）。
        
        Returns:
            命令张量 [num_envs, 4]，包含 [vx, vy, ωz, heading]
        """
        if self.cfg.enable_heading_control:
            return torch.cat([self.vel_command_b, self.heading_target.unsqueeze(-1)], dim=-1)
        else:
            # 如果没有heading，返回全0的heading列
            return torch.cat([
                self.vel_command_b,
                torch.zeros(self.num_envs, 1, dtype=torch.float32, device=self.device)
            ], dim=-1)
    
    def update(
        self,
        dt: float,
        robot: Articulation | None = None,
    ):
        """
        更新命令系统（模仿Manager-Based的compute方法）。
        
        Args:
            dt: 时间步长（秒）
            robot: 机器人对象，用于heading控制（如果启用）
        """
        # 1. 更新误差指标（如果启用）
        if self.cfg.enable_metrics and robot is not None:
            self._update_metrics(robot)
        
        # 2. 时间管理：减少剩余时间
        self.command_time_left -= dt
        
        # 3. 检测需要重采样的环境
        resample_ids = (self.command_time_left <= 0.0).nonzero(as_tuple=False).flatten()
        
        if len(resample_ids) > 0:
            # 重采样命令
            self._resample_commands(resample_ids)
            
            # 重置时间（随机8-12秒）
            self.command_time_left[resample_ids] = sample_uniform(
                self.cfg.resampling_time_range[0],
                self.cfg.resampling_time_range[1],
                (len(resample_ids),),
                self.device
            )
        
        # 4. Heading控制（如果启用）
        if self.cfg.enable_heading_control and robot is not None:
            self._update_heading_control(robot)
        
        # 5. 处理静止环境（如果启用）
        if self.cfg.enable_standing_envs:
            standing_ids = self.is_standing_env.nonzero(as_tuple=False).flatten()
            self.vel_command_b[standing_ids, :] = 0.0
    
    def reset(self, env_ids: torch.Tensor | None = None):
        """
        重置指定环境的命令。
        
        Args:
            env_ids: 要重置的环境索引，如果为None则重置所有环境
        """
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        
        # 立即重采样
        self._resample_commands(env_ids)
        
        # 重置时间
        self.command_time_left[env_ids] = sample_uniform(
            self.cfg.resampling_time_range[0],
            self.cfg.resampling_time_range[1],
            (len(env_ids),),
            self.device
        )
        
        # 重置误差指标（如果启用）
        if self.cfg.enable_metrics:
            self.metrics["error_vel_xy"][env_ids] = 0.0
            self.metrics["error_vel_yaw"][env_ids] = 0.0
    
    def get_metrics(self) -> dict[str, torch.Tensor]:
        """
        获取命令跟踪误差指标。
        
        Returns:
            包含误差指标的字典
        """
        if self.cfg.enable_metrics:
            return self.metrics.copy()
        else:
            return {}
    
    def _resample_commands(self, env_ids: torch.Tensor):
        """
        重采样命令（私有方法）。
        
        Args:
            env_ids: 要重采样的环境索引
        """
        n = len(env_ids)
        r = torch.empty(n, device=self.device)
        
        # 采样线性速度 (x, y)
        self.vel_command_b[env_ids, 0] = r.uniform_(
            self.cfg.lin_vel_x_range[0], 
            self.cfg.lin_vel_x_range[1]
        )
        self.vel_command_b[env_ids, 1] = r.uniform_(
            self.cfg.lin_vel_y_range[0], 
            self.cfg.lin_vel_y_range[1]
        )
        
        # 采样角速度 (z)
        self.vel_command_b[env_ids, 2] = r.uniform_(
            self.cfg.ang_vel_z_range[0], 
            self.cfg.ang_vel_z_range[1]
        )
        
        # Heading控制（如果启用）
        if self.cfg.enable_heading_control:
            # 采样目标朝向
            self.heading_target[env_ids] = r.uniform_(
                self.cfg.heading_range[0], 
                self.cfg.heading_range[1]
            )
            # 随机决定哪些环境使用heading控制
            self.is_heading_env[env_ids] = r.uniform_(0.0, 1.0) <= self.cfg.rel_heading_envs
        
        # 静止环境（如果启用）
        if self.cfg.enable_standing_envs:
            # 随机决定哪些环境保持静止
            self.is_standing_env[env_ids] = r.uniform_(0.0, 1.0) <= self.cfg.rel_standing_envs
    
    def _update_heading_control(self, robot: Articulation):
        """
        更新heading控制（私有方法）。
        
        Args:
            robot: 机器人对象
        """
        # 找到需要heading控制的环境
        env_ids = self.is_heading_env.nonzero(as_tuple=False).flatten()
        
        if len(env_ids) > 0:
            # 计算heading误差
            current_heading = robot.data.heading_w[env_ids]
            heading_error = wrap_to_pi(self.heading_target[env_ids] - current_heading)
            
            # P控制：ω = K * Δθ
            computed_ang_vel = self.cfg.heading_control_stiffness * heading_error
            
            # 限幅
            computed_ang_vel = torch.clip(
                computed_ang_vel,
                min=self.cfg.ang_vel_z_range[0],
                max=self.cfg.ang_vel_z_range[1],
            )
            
            # 更新角速度命令
            self.vel_command_b[env_ids, 2] = computed_ang_vel
    
    def _update_metrics(self, robot: Articulation):
        """
        更新误差指标（私有方法）。
        
        Args:
            robot: 机器人对象
        """
        # 线性速度误差 (XY平面)
        vel_error = torch.norm(
            self.vel_command_b[:, :2] - robot.data.root_lin_vel_b[:, :2],
            dim=-1
        )
        self.metrics["error_vel_xy"] += vel_error
        
        # 角速度误差 (Z轴)
        yaw_error = torch.abs(
            self.vel_command_b[:, 2] - robot.data.root_ang_vel_b[:, 2]
        )
        self.metrics["error_vel_yaw"] += yaw_error


# 预定义配置模板
class CommandPresets:
    """预定义的命令配置模板"""
    
    @staticmethod
    def basic() -> CommandConfig:
        """基础配置：只有速度采样，无高级功能"""
        return CommandConfig(
            lin_vel_x_range=(-1.0, 1.0),
            lin_vel_y_range=(-1.0, 1.0),
            ang_vel_z_range=(-1.0, 1.0),
            resampling_time_range=(10.0, 10.0),
            enable_heading_control=False,
            enable_standing_envs=False,
            enable_metrics=False,
        )
    
    @staticmethod
    def standard() -> CommandConfig:
        """标准配置：对标Manager-Based的默认设置"""
        return CommandConfig(
            lin_vel_x_range=(-2.0, 2.0),
            lin_vel_y_range=(-2.0, 2.0),
            ang_vel_z_range=(-1.5, 1.5),
            resampling_time_range=(8.0, 12.0),
            enable_heading_control=False,
            enable_standing_envs=True,
            rel_standing_envs=0.1,
            enable_metrics=False,
        )
    
    @staticmethod
    def advanced() -> CommandConfig:
        """高级配置：包含所有功能"""
        return CommandConfig(
            lin_vel_x_range=(-2.0, 2.0),
            lin_vel_y_range=(-2.0, 2.0),
            ang_vel_z_range=(-1.5, 1.5),
            heading_range=(-3.14, 3.14),
            resampling_time_range=(8.0, 12.0),
            enable_heading_control=True,
            heading_control_stiffness=0.5,
            rel_heading_envs=1.0,
            enable_standing_envs=True,
            rel_standing_envs=0.1,
            enable_metrics=True,
        )
    
    @staticmethod
    def training() -> CommandConfig:
        """训练配置：快速重采样，提高样本多样性"""
        return CommandConfig(
            lin_vel_x_range=(-2.0, 2.0),
            lin_vel_y_range=(-2.0, 2.0),
            ang_vel_z_range=(-1.5, 1.5),
            resampling_time_range=(5.0, 8.0),  # 更频繁的重采样
            enable_heading_control=False,
            enable_standing_envs=True,
            rel_standing_envs=0.15,  # 更多静止环境
            enable_metrics=True,
        )
    
    @staticmethod
    def evaluation() -> CommandConfig:
        """评估配置：较慢重采样，稳定测试"""
        return CommandConfig(
            lin_vel_x_range=(-1.5, 1.5),
            lin_vel_y_range=(-1.0, 1.0),
            ang_vel_z_range=(-1.0, 1.0),
            resampling_time_range=(15.0, 20.0),  # 更慢的重采样
            enable_heading_control=False,
            enable_standing_envs=False,  # 评估时不要静止
            enable_metrics=True,
        )
