import isaaclab.sim as sim_utils
from isaaclab.actuators.actuator_cfg import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from .base import DexHand
from .decorators import register_dexhand
from abc import ABC, abstractmethod

class Shadow(DexHand, ABC):
    def __init__(self):
        super().__init__()
        self._urdf_path = None
        self.side = None
        self.name = "shadow"
        self.body_names = [
            'palm', 
            'ffknuckle', 
            'lfmetacarpal', 
            'mfknuckle', 
            'rfknuckle', 
            'thbase', 
            'ffproximal', 
            'lfknuckle', 
            'mfproximal', 
            'rfproximal', 
            'thproximal', 
            'ffmiddle', 
            'lfproximal', 
            'mfmiddle', 
            'rfmiddle', 
            'thhub', 
            'ffdistal', 
            'lfmiddle', 
            'mfdistal', 
            'rfdistal', 
            'thmiddle', 
            'fftip', 
            'lfdistal', 
            'mftip', 
            'rftip', 
            'thdistal', 
            'lftip', 
            'thtip',
        ]
        self.dof_names = [
            'FFJ4', 
            'LFJ5', 
            'MFJ4', 
            'RFJ4', 
            'THJ5', 
            'FFJ3', 
            'LFJ4', 
            'MFJ3', 
            'RFJ3', 
            'THJ4', 
            'FFJ2', 
            'LFJ3', 
            'MFJ2', 
            'RFJ2', 
            'THJ3', 
            'FFJ1', 
            'LFJ2', 
            'MFJ1', 
            'RFJ1', 
            'THJ2', 
            'LFJ1', 
            'THJ1',
        ]
        self.wrist_name = 'palm'
        self.contact_body_names = [
            "thdistal",
            "ffdistal",
            "mfdistal",
            "rfdistal",
            "lfdistal",
        ]
        self.init_state = ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.5),
            rot=(-0.707, 0.0, 0.0, 0.707),
            joint_pos={
                ".*J.*": 0.0,
            },
        )
        self.actuators = {
            "fingers": ImplicitActuatorCfg(
                joint_names_expr=[".*J.*"],
                effort_limit={
                    ".*": 5,
                },
                stiffness={
                    ".*": 5,
                },
                damping={
                    ".*": 0.5,
                },
            ),
        }
        self.dof_limit = [
            [- 0.349, 0.349],
            [0, 0.785],
            [- 0.349, 0.349],
            [- 0.349, 0.349],
            [- 1.047, 1.047],
            [- 0.262, 1.571],
            [- 0.349, 0.349],
            [- 0.262, 1.571],
            [- 0.262, 1.571],
            [0, 1.222],
            [0, 1.571],
            [- 0.262, 1.571],
            [0, 1.571],
            [0, 1.571],
            [- 0.209, 0.209],
            [0, 1.571],
            [0, 1.571],
            [0, 1.571],
            [0, 1.571],
            [- 0.698, 0.698],
            [0, 1.571],
            [-0.262, 1.571],
        ]
        self.weight_idx = {
            "thumb_tip": [27],
            "index_tip": [21],
            "middle_tip": [23],
            "ring_tip": [24],
            "pinky_tip": [26],
            "level_1_joints": [1, 3, 4, 5, 6, 7, 8, 9, 10, 12],
            "level_2_joints": [2, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22],
        }

    def __str__(self):
        return self.name

@register_dexhand("shadow_rh")
class ShadowRH(Shadow):
    def __init__(self):
        super().__init__()
        self._usd_path = "./assets/Shadow/shadow_hand_right.usd"
        self.side = "rh"

    def __str__(self):
        return super().__str__() + "_rh"

# SHAND_CFG = ArticulationCfg(
#     spawn=sim_utils.UsdFileCfg(
#         usd_path=f"./assets/Shadow/shadow_hand_right_woarm.usd",
#         activate_contact_sensors=True,
#         # visible=False,
#         rigid_props=sim_utils.RigidBodyPropertiesCfg(
#             disable_gravity=True,
#             retain_accelerations=True,
#             max_depenetration_velocity=1000.0,
#         ),
#         articulation_props=sim_utils.ArticulationRootPropertiesCfg(
#             enabled_self_collisions=True,
#             solver_position_iteration_count=8,
#             solver_velocity_iteration_count=0,
#             sleep_threshold=0.005,
#             stabilization_threshold=0.0005,
#         ),
#         # collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
#         joint_drive_props=sim_utils.JointDrivePropertiesCfg(drive_type="force"),
#         fixed_tendons_props=sim_utils.FixedTendonPropertiesCfg(limit_stiffness=30.0, damping=0.1),
#     ),
#     init_state=ArticulationCfg.InitialStateCfg(
#         pos=(0.0, 0.0, 0.5),
#         # rot=(0.0, 0.0, 0.0, 1.0),
#         rot=(-0.707, 0.0, 0.0, 0.707),
#         # joint_pos={
#         #     "shoulder_pan_joint": 1.7,
#         #     "shoulder_lift_joint": -1.712,
#         #     "elbow_joint": 1.712,
#         #     "wrist_1_joint": 1.7,
#         #     "wrist_2_joint": 1.7,
#         #     "wrist_3_joint": 1.7,
#         #     ".*J.*": 0.0,
#         # },
#         joint_pos={
#             ".*J.*": 0.0,
#         },
#     ),
#     actuators={
#         "fingers": ImplicitActuatorCfg(
#             # joint_names_expr=["WR.*", "(FF|MF|RF|LF|TH)J(4|3|2|1)", "(LF|TH)J5"],
#             joint_names_expr=[".*J.*"],
#             effort_limit={
#                 ".*": 5,
#                 # "WRJ2": 4.785,
#                 # "WRJ1": 2.175,
#                 # "(FF|MF|RF|LF)J(2|1)": 0.7245,
#                 # "FFJ(4|3)": 0.9,
#                 # "MFJ(4|3)": 0.9,
#                 # "RFJ(4|3)": 0.9,
#                 # "LFJ(5|4|3)": 0.9,
#                 # "THJ5": 2.3722,
#                 # "THJ4": 1.45,
#                 # "THJ(3|2)": 0.99,
#                 # "THJ1": 0.81,
#             },
#             stiffness={
#                 ".*": 5,
#                 # "WRJ.*": 5.0,
#                 # "(FF|MF|RF|LF|TH)J(4|3|2|1)": 1.0,
#                 # "(LF|TH)J5": 1.0,
#             },
#             damping={
#                 ".*": 0.5,
#                 # "WRJ.*": 0.5,
#                 # "(FF|MF|RF|LF|TH)J(4|3|2|1)": 0.1,
#                 # "(LF|TH)J5": 0.1,
#             },
#         ),
#     },
#     soft_joint_pos_limit_factor=1.0,
# )
