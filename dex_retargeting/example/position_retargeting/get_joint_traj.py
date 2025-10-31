import tempfile
from pathlib import Path
from typing import Optional, Tuple, List
import csv

import numpy as np
import tyro
import sapien
import torch
from pytransform3d import transformations as pt

from dex_retargeting import yourdfpy as urdf
from dataset import DexYCBVideoDataset
from dex_retargeting.constants import RobotName, HandType, get_default_config_path, RetargetingType
from mano_layer import MANOLayer
from dex_retargeting.retargeting_config import RetargetingConfig

# For numpy version compatibility
np.bool = bool
np.int = int
np.float = float
np.str = str
np.complex = complex
np.object = object
np.unicode = np.unicode_

ROBOT_JOINT_TRAJ_DIR = "./traj/"

def compute_hand_geometry(hand_pose_frame, extrinsic_mat, mano_layer, use_camera_frame=False):
    # pose parameters all zero, no hand is detected
        if np.abs(hand_pose_frame).sum() < 1e-5:
            return None, None
        p = torch.from_numpy(hand_pose_frame[:, :48].astype(np.float32))
        t = torch.from_numpy(hand_pose_frame[:, 48:51].astype(np.float32))
        vertex, joint = mano_layer(p, t)
        vertex = vertex.cpu().numpy()[0]
        joint = joint.cpu().numpy()[0]
        if not use_camera_frame:
            pose_vec = pt.pq_from_transform(extrinsic_mat)
            camera_pose = sapien.Pose(pose_vec[0:3], pose_vec[3:7]).inv()
            camera_mat = camera_pose.to_transformation_matrix()
            vertex = vertex @ camera_mat[:3, :3].T + camera_mat[:3, 3]
            vertex = np.ascontiguousarray(vertex)
            joint = joint @ camera_mat[:3, :3].T + camera_mat[:3, 3]
            joint = np.ascontiguousarray(joint)

        return vertex, joint

def get_joint_traj(robots: Optional[List[RobotName]], data_root: Path, data_indices: List[int], fps: int):
    dataset = DexYCBVideoDataset(data_root, hand_type="right")
    
    # Data ID, feel free to change it to visualize different trajectory
    print(len(dataset))
    data_id = 3
    
    data = dataset[data_id]
    hand_pose = data["hand_pose"]
    hand_shape = data["hand_shape"]
    extrinsic_mat = data["extrinsics"]
    mano_layer = MANOLayer("right", hand_shape.astype(np.float32))
    num_frame = hand_pose.shape[0]
    
    # Skip frames where human hand is not detected in DexYCB dataset
    start_frame = 0
    for i in range(0, num_frame):
        init_hand_pose_frame = hand_pose[i]
        vertex, joint = compute_hand_geometry(init_hand_pose_frame, extrinsic_mat, mano_layer)
        if vertex is not None:
            start_frame = i
            break
    
    for robot_name in robots:
        config_path = get_default_config_path(robot_name, RetargetingType.position, HandType.right)
        
        # Add 6-DoF dummy joint at the root of each robot to make them move freely in the space
        override = dict(add_dummy_free_joint=True)
        config = RetargetingConfig.load_from_file(config_path, override=override)
        retargeting = config.build()
        robot_file_name = Path(config.urdf_path).stem

        # Build robot
        urdf_path = Path(config.urdf_path)
        if "glb" not in urdf_path.stem:
            urdf_path = urdf_path.with_stem(urdf_path.stem + "_glb")
        robot_urdf = urdf.URDF.load(str(urdf_path), add_dummy_free_joints=True, build_scene_graph=False)
        urdf_name = urdf_path.name
        temp_dir = tempfile.mkdtemp(prefix="dex_retargeting-")
        temp_path = f"{temp_dir}/{urdf_name}"
        robot_urdf.write_xml_file(temp_path)
        
        # Retargeting
        traj = []
        traj.append(retargeting.optimizer.target_joint_names)
        for i in range(start_frame, num_frame):
            hand_pose_frame = hand_pose[i]
            vertex, joint = compute_hand_geometry(hand_pose_frame, extrinsic_mat, mano_layer)
            indices = retargeting.optimizer.target_link_human_indices
            ref_value = joint[indices, :]
            retargeting.retarget(ref_value)
            qpos = retargeting.last_qpos
            traj.append(qpos)
            
        # Save the traj as csv file
        file_name = robot_name.name + "_" + data["capture_name"] + ".csv"
        with open(ROBOT_JOINT_TRAJ_DIR + file_name, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(traj)
        
        
        
def main(dexycb_dir: str, robots: Optional[List[RobotName]] = None, data_indices: List[int] = None, fps: int = 10):
    """
    Retarget the human trajectories for grasping object inside DexYCB dataset to the robot.
    The robot trajectories were saved as csv file in 'traj' directory

    Args:
        dexycb_dir: Data root path to the dexycb dataset
        robots: The names of robots to retarget
        fps: frequency to render hand-object trajectory

    """
    
    data_root = Path(dexycb_dir).absolute()
    robot_dir = Path(__file__).absolute().parent.parent.parent / "assets" / "robots" / "hands"
    RetargetingConfig.set_default_urdf_dir(robot_dir)
    if not data_root.exists():
        raise ValueError(f"Path to DexYCB dir: {data_root} does not exist.")
    else:
        print(f"Using DexYCB dir: {data_root}")
        
    get_joint_traj(robots, data_root, data_indices, fps)
        
    
        
if __name__ == "__main__":
    tyro.cli(main)