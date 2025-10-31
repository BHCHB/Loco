# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
This script demonstrates different single-arm manipulators.

.. code-block:: bash

    # Usage
    ./isaaclab.sh -p scripts/demos/arms.py

"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="This script demonstrates different single-arm manipulators.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import numpy as np
import torch
import zarr
import os

import isaacsim.core.utils.prims as prim_utils

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

##
# Pre-defined configs
##
# isort: off
from isaaclab_assets import (
    FRANKA_PANDA_CFG,
    UR10_CFG,
    KINOVA_JACO2_N7S300_CFG,
    KINOVA_JACO2_N6S300_CFG,
    KINOVA_GEN3_N7_CFG,
    SAWYER_CFG,
)

from Franka_RL.robots import DexHandFactory

# isort: on


def define_origins(num_origins: int, spacing: float) -> list[list[float]]:
    """Defines the origins of the the scene."""
    # create tensor based on number of environments
    env_origins = torch.zeros(num_origins, 3)
    # create a grid of origins
    num_rows = np.floor(np.sqrt(num_origins))
    num_cols = np.ceil(num_origins / num_rows)
    xx, yy = torch.meshgrid(torch.arange(num_rows), torch.arange(num_cols), indexing="xy")
    env_origins[:, 0] = spacing * xx.flatten()[:num_origins] - spacing * (num_rows - 1) / 2
    env_origins[:, 1] = spacing * yy.flatten()[:num_origins] - spacing * (num_cols - 1) / 2
    env_origins[:, 2] = 0.0
    # return the origins
    return env_origins.tolist()


def design_scene() -> tuple[dict, list[list[float]]]:
    """Designs the scene."""
    # Ground-planeShand_Retargeting
    cfg = sim_utils.GroundPlaneCfg()
    cfg.func("/World/defaultGroundPlane", cfg)
    # Lights
    cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
    cfg.func("/World/Light", cfg)

    # Create separate groups called "Origin1", "Origin2", "Origin3"
    # Each group will have a mount and a robot on top of it
    origins = define_origins(num_origins=1, spacing=2.0)

    # Origin 1 with Franka Panda
    prim_utils.create_prim("/World/Origin1", "Xform", translation=origins[0])
    # -- Table
    # cfg = sim_utils.UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd")
    # cfg.func("/World/Origin1/Table", cfg, translation=(0.55, 0.0, 1.05))
    # -- Robot
    # franka_arm_cfg = FRANKA_PANDA_CFG.replace(prim_path="/World/Origin1/Robot")
    # franka_arm_cfg.init_state.pos = (0.0, 0.0, 0.0)
    # franka_panda = Articulation(cfg=franka_arm_cfg)

    # shand_cfg = SHAND_CFG.replace(prim_path="/World/Origin1/Robot")
    # shand = Articulation(cfg=shand_cfg)

    dexhand = DexHandFactory.create_hand('shadow')
    robot_cfg = ArticulationCfg(
        prim_path="/World/Origin1/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=dexhand._usd_path,
            activate_contact_sensors=True,
            # visible=False,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
                retain_accelerations=True,
                max_depenetration_velocity=1000.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=True,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=0,
                sleep_threshold=0.005,
                stabilization_threshold=0.0005,
            ),
            # collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
            joint_drive_props=sim_utils.JointDrivePropertiesCfg(drive_type="force"),
            fixed_tendons_props=sim_utils.FixedTendonPropertiesCfg(limit_stiffness=30.0, damping=0.1),
        ),
        init_state=dexhand.init_state,
        actuators=dexhand.actuators,
        soft_joint_pos_limit_factor=1.0,
    )
    robot = Articulation(robot_cfg)

    # Origin 2 with UR10
    # prim_utils.create_prim("/World/Origin2", "Xform", translation=origins[1])
    # # -- Table
    # cfg = sim_utils.UsdFileCfg(
    #     usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/Stand/stand_instanceable.usd", scale=(2.0, 2.0, 2.0)
    # )
    # cfg.func("/World/Origin2/Table", cfg, translation=(0.0, 0.0, 1.03))
    # # -- Robot
    # ur10_cfg = UR10_CFG.replace(prim_path="/World/Origin2/Robot")
    # ur10_cfg.init_state.pos = (0.0, 0.0, 1.03)
    # ur10 = Articulation(cfg=ur10_cfg)

    # # Origin 3 with Kinova JACO2 (7-Dof) arm
    # prim_utils.create_prim("/World/Origin3", "Xform", translation=origins[2])
    # # -- Table
    # cfg = sim_utils.UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/ThorlabsTable/table_instanceable.usd")
    # cfg.func("/World/Origin3/Table", cfg, translation=(0.0, 0.0, 0.8))
    # # -- Robot
    # kinova_arm_cfg = KINOVA_JACO2_N7S300_CFG.replace(prim_path="/World/Origin3/Robot")
    # kinova_arm_cfg.init_state.pos = (0.0, 0.0, 0.8)
    # kinova_j2n7s300 = Articulation(cfg=kinova_arm_cfg)

    # # Origin 4 with Kinova JACO2 (6-Dof) arm
    # prim_utils.create_prim("/World/Origin4", "Xform", translation=origins[3])
    # # -- Table
    # cfg = sim_utils.UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/ThorlabsTable/table_instanceable.usd")
    # cfg.func("/World/Origin4/Table", cfg, translation=(0.0, 0.0, 0.8))
    # # -- Robot
    # kinova_arm_cfg = KINOVA_JACO2_N6S300_CFG.replace(prim_path="/World/Origin4/Robot")
    # kinova_arm_cfg.init_state.pos = (0.0, 0.0, 0.8)
    # kinova_j2n6s300 = Articulation(cfg=kinova_arm_cfg)

    # # Origin 5 with Sawyer
    # prim_utils.create_prim("/World/Origin5", "Xform", translation=origins[4])
    # # -- Table
    # cfg = sim_utils.UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd")
    # cfg.func("/World/Origin5/Table", cfg, translation=(0.55, 0.0, 1.05))
    # # -- Robot
    # kinova_arm_cfg = KINOVA_GEN3_N7_CFG.replace(prim_path="/World/Origin5/Robot")
    # kinova_arm_cfg.init_state.pos = (0.0, 0.0, 1.05)
    # kinova_gen3n7 = Articulation(cfg=kinova_arm_cfg)

    # # Origin 6 with Kinova Gen3 (7-Dof) arm
    # prim_utils.create_prim("/World/Origin6", "Xform", translation=origins[5])
    # # -- Table
    # cfg = sim_utils.UsdFileCfg(
    #     usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/Stand/stand_instanceable.usd", scale=(2.0, 2.0, 2.0)
    # )
    # cfg.func("/World/Origin6/Table", cfg, translation=(0.0, 0.0, 1.03))
    # # -- Robot
    # sawyer_arm_cfg = SAWYER_CFG.replace(prim_path="/World/Origin6/Robot")
    # sawyer_arm_cfg.init_state.pos = (0.0, 0.0, 1.03)
    # sawyer = Articulation(cfg=sawyer_arm_cfg)

    # return the scene information
    scene_entities = {
        "shand": robot,
        # "franka_panda": franka_panda,
        # "ur10": ur10,
        # "kinova_j2n7s300": kinova_j2n7s300,
        # "kinova_j2n6s300": kinova_j2n6s300,
        # "kinova_gen3n7": kinova_gen3n7,
        # "sawyer": sawyer,
    }
    return scene_entities, origins


def load_traj():
    dataset = './dataset/maniskill'
    delta_eepose = torch.tensor(np.array(zarr.open(os.path.join(dataset, 'data', 'action'), mode='r')), dtype=torch.float32, device=args_cli.device)
    joint_pose = torch.tensor(np.array(zarr.open(os.path.join(dataset, 'data', 'state'), mode='r')), dtype=torch.float32, device=args_cli.device)[:, :7]
    traj_split = torch.tensor(np.array(zarr.open(os.path.join(dataset, 'meta', 'episode_ends'), mode = 'r')), dtype=torch.int, device=args_cli.device)
    return delta_eepose, joint_pose, traj_split

def run_simulator(sim: sim_utils.SimulationContext, entities: dict[str, Articulation], origins: torch.Tensor):
    """Runs the simulation loop."""
    # Define simulation stepping
    sim_dt = sim.get_physics_dt()
    sim_time = 0.0
    count = 0
    traj_idx = 0
    robot = entities['shand']
    # root state
    root_state = robot.data.default_root_state.clone()
    root_state[:, :3] += origins[0]
    robot.write_root_pose_to_sim(root_state[:, :7])
    robot.write_root_velocity_to_sim(root_state[:, 7:])
    robot.reset()
    joint_vel = robot.data.default_joint_vel.clone()
    while(simulation_app.is_running()):
        pass
    # eepose = torch.zeros(joint_pose.shape[0], 7, dtype=torch.float32, device=args_cli.device)

    # for target_joint_pose in joint_pose:
    #     target_pose = torch.zeros(9, dtype=torch.float32, device=args_cli.device)
    #     target_pose[:7] = target_joint_pose
    #     robot.write_joint_state_to_sim(target_pose, joint_vel)
    #     robot.reset()
    #     eepose[count] = robot.data.body_state_w[0, robot.body_names.index("panda_hand"), :7]
    #     count += 1
    #     sim.step()
    #     if(count == split[traj_idx]):
    #         print(f"[INFO]: Successfully Process {traj_idx} Trajectory.")
    #         traj_idx += 1
    #         # if(traj_idx == 10):
    #         #     break

    # pass
    # z = zarr.create_array(store="./dataset/data/eepose",
    # shape=eepose.shape,
    # chunks=(100, 7), dtype="<f4",
    # compressors=zarr.codecs.BloscCodec(cname="zstd", clevel=3, shuffle=zarr.codecs.BloscShuffle.shuffle),
    # )
    # z[:, :] = eepose.cpu().numpy()

    # pass
    # # Simulate physics
    # while simulation_app.is_running():
    # #     resetr
        
    #     if count % 200 == 0:
    #         # reset counters
    #         sim_time = 0.0
    #         count = 0
    #         # reset the scene entities
    #         for index, robot in enumerate(entities.values()):
    #             # root state
    #             root_state = robot.data.default_root_state.clone()
    #             root_state[:, :3] += origins[index]
    #             robot.write_root_pose_to_sim(root_state[:, :7])
    #             robot.write_root_velocity_to_sim(root_state[:, 7:])
    #             # set joint positions
    #             joint_pos, joint_vel = robot.data.default_joint_pos.clone(), robot.data.default_joint_vel.clone()
    #             robot.write_joint_state_to_sim(joint_pos, joint_vel)
    #             # clear internal buffers
    #             robot.reset()
    #         print("[INFO]: Resetting robots state...")
    #     # apply random actions to the robots
    #     for robot in entities.values():
    #         # generate random joint positions
    #         joint_pos_target = robot.data.default_joint_pos + torch.randn_like(robot.data.joint_pos) * 0.1
    #         joint_pos_target = joint_pos_target.clamp_(
    #             robot.data.soft_joint_pos_limits[..., 0], robot.data.soft_joint_pos_limits[..., 1]
    #         )
    #         # apply action to the robot
    #         robot.set_joint_position_target(joint_pos_target)
    #         # write data to sim
    #         robot.write_data_to_sim()
    #     # perform step
    #     sim.step()
    #     # update sim-time
    #     sim_time += sim_dt
    #     count += 1
    #     # update buffers
    #     for robot in entities.values():
    #         robot.update(sim_dt)


def main():
    """Main function."""
    # Initialize the simulation context
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    # Set main camera
    sim.set_camera_view([3.5, 0.0, 3.2], [0.0, 0.0, 0.5])
    # design scene
    scene_entities, scene_origins = design_scene()
    # delta_eepose, joint_pose , split = load_traj()
    scene_origins = torch.tensor(scene_origins, device=sim.device)
    # Play the simulator
    sim.reset()
    # Now we are ready!
    print("[INFO]: Setup complete...")
    # Run the simulator
    run_simulator(sim, scene_entities, scene_origins)


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
