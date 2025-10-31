import isaaclab.sim as sim_utils
from isaaclab.actuators.actuator_cfg import ImplicitActuatorCfg,DCMotorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR
from .base import QuadrupedRobot
from .decorators import register_quadruped
from abc import ABC, abstractmethod


class UnitreeGo2(QuadrupedRobot, ABC):
    """Unitree Go2 quadruped robot implementation."""
    
    def __init__(self):
        super().__init__()
        
        # Basic robot information
        self._usd_path = f"{ISAACLAB_NUCLEUS_DIR}/Robots/Unitree/Go2/go2.usd"
        
        self.name = "unitree_go2"
        
        # Leg configuration
        self.leg_names = ["FL", "FR", "RL", "RR"]  # Front Left, Front Right, Rear Left, Rear Right
        
        # Body part names (19 bodies total - matching exact USD structure)
        # USD order: base, hips (FL,FR), Head_upper, hips (RL,RR), thighs (FL,FR), Head_lower, thighs (RL,RR), calves, feet
        self.body_names = [
            "base",                                          # 0: Base body
            "FL_hip", "FR_hip",                             # 1-2: Front hip bodies
            "Head_upper",                                    # 3: Head upper part
            "RL_hip", "RR_hip",                             # 4-5: Rear hip bodies
            "FL_thigh", "FR_thigh",                         # 6-7: Front thigh bodies
            "Head_lower",                                    # 8: Head lower part (between front and rear thighs)
            "RL_thigh", "RR_thigh",                         # 9-10: Rear thigh bodies
            "FL_calf", "FR_calf", "RL_calf", "RR_calf",     # 11-14: Calf bodies
            "FL_foot", "FR_foot", "RL_foot", "RR_foot",     # 15-18: Foot bodies
        ]
     
        # DOF names (12 DOF total - 3 per leg)
        self.dof_names = [
            # Front Left leg
            "FL_hip_joint", "FR_hip_joint", "RL_hip_joint", "RR_hip_joint",
            "FL_thigh_joint","FR_thigh_joint", "RL_thigh_joint","RR_thigh_joint", 
            "FL_calf_joint","FR_calf_joint", "RL_calf_joint","RR_calf_joint",
        ]
        
        # Specific body part names (with correct indices based on exact USD order)
        self.base_name = "base"  # Index: 0
        self.hip_names = ["FL_hip", "FR_hip", "RL_hip", "RR_hip"]  # Indices: 1, 2, 4, 5
        self.head_names = ["Head_upper", "Head_lower"]  # Indices: 3, 8
        self.thigh_names = ["FL_thigh", "FR_thigh", "RL_thigh", "RR_thigh"]  # Indices: 6, 7, 9, 10
        self.calf_names = ["FL_calf", "FR_calf", "RL_calf", "RR_calf"]  # Indices: 11-14
        self.foot_names = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]  # Indices: 15-18
        
        # Contact bodies (feet with contact sensors)
        self.contact_body_names = self.foot_names.copy()
        
        # Initial state configuration
        self.init_state = ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.4),  # Initial position: 0.4m height
            rot=(0.0, 0.0, 0.0, 1.0),  # Initial orientation: no rotation (quaternion w,x,y,z)
            joint_pos={
                # Hip joints (abduction/adduction)
                "FL_hip_joint": 0.1,
                "FR_hip_joint": -0.1,
                "RL_hip_joint": 0.1,
                "RR_hip_joint": -0.1,
                # Thigh joints (hip flexion/extension)
                "FL_thigh_joint": 0.8,  
                "FR_thigh_joint": 0.8,  
                "RL_thigh_joint": 1.0,
                "RR_thigh_joint": 1.0,
                # Calf joints (knee flexion/extension)
                "FL_calf_joint": -1.5,
                "FR_calf_joint": -1.5,
                "RL_calf_joint": -1.5,
                "RR_calf_joint": -1.5,
            },
            joint_vel={
                # All joints start with zero velocity
                ".*": 0.0,
            },
        )

        self.soft_joint_pos_limit_factor=0.9

        # Actuator configuration (based on Go2 specifications)
        self.actuators = {
            "base_legs": DCMotorCfg(
            joint_names_expr=[".*_hip_joint", ".*_thigh_joint", ".*_calf_joint"],
            effort_limit=23.5,
            saturation_effort=23.5,
            velocity_limit=30.0,
            stiffness=40.0,
            damping=1.0,
            friction=0.0,
            ),
        }
        """""
        # Joint angle limits in radians [min, max]
        self.dof_limit = [
            # FL leg (Front Left)
            [-0.802, 0.802],   # FL_hip_joint: ±46° (abduction/adduction)
            [-1.047, 4.188],   # FL_thigh_joint: -60° to 240° (hip flexion/extension)
            [-2.697, -0.916],  # FL_calf_joint: -154.5° to -52.5° (knee flexion/extension)
            
            # FR leg (Front Right)
            [-0.802, 0.802],   # FR_hip_joint: ±46°
            [-1.047, 4.188],   # FR_thigh_joint: -60° to 240°
            [-2.697, -0.916],  # FR_calf_joint: -154.5° to -52.5°
            
            # RL leg (Rear Left)
            [-0.802, 0.802],   # RL_hip_joint: ±46°
            [-1.047, 4.188],   # RL_thigh_joint: -60° to 240°
            [-2.697, -0.916],  # RL_calf_joint: -154.5° to -52.5°
            
            # RR leg (Rear Right)
            [-0.802, 0.802],   # RR_hip_joint: ±46°
            [-1.047, 4.188],   # RR_thigh_joint: -60° to 240°
            [-2.697, -0.916],  # RR_calf_joint: -154.5° to -52.5°
        ]
        
        # Weight indices for reward computation and analysis
        self.weight_idx = {
            # Body indices
            "base": [0],                              # Base body
            "feet": [4, 8, 12, 15],                  # Foot bodies (FL, FR, RL, RR)
            "legs": [1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15],  # All leg bodies
            "hip_bodies": [1, 5, 9, 13],             # Hip bodies
            "thigh_bodies": [2, 6, 10, 14],          # Thigh bodies  
            "calf_bodies": [3, 7, 11, 15],           # Calf bodies
            
            # Joint indices
            "hip_joints": [0, 1, 2, 3],              # Hip joint indices
            "thigh_joints": [4, 5, 6, 7],           # Thigh joint indices  
            "calf_joints": [8, 9, 10, 11],            # Calf joint indices
            
            # Leg-wise indices
            "FL_leg": [1, 2, 3, 4],                  # Front Left leg bodies
            "FR_leg": [5, 6, 7, 8],                  # Front Right leg bodies
            "RL_leg": [9, 10, 11, 12],               # Rear Left leg bodies
            "RR_leg": [13, 14, 15, 16],              # Rear Right leg bodies
        }
        """

    def get_leg_joints(self, leg_name: str):
        """Get joint names for a specific leg."""
        if leg_name not in self.leg_names:
            raise ValueError(f"Invalid leg name '{leg_name}'. Available: {self.leg_names}")
        
        return [
            f"{leg_name}_hip_joint",
            f"{leg_name}_thigh_joint", 
            f"{leg_name}_calf_joint"
        ]

    def get_foot_name(self, leg_name: str):
        """Get foot body name for a specific leg."""
        if leg_name not in self.leg_names:
            raise ValueError(f"Invalid leg name '{leg_name}'. Available: {self.leg_names}")
        return f"{leg_name}_foot"

    def get_leg_joint_indices(self, leg_name: str):
        """Get joint indices for a specific leg."""
        if leg_name not in self.leg_names:
            raise ValueError(f"Invalid leg name '{leg_name}'. Available: {self.leg_names}")
        
        leg_index = self.leg_names.index(leg_name)
        start_idx = leg_index * 3  # 3 joints per leg
        return [start_idx, start_idx + 1, start_idx + 2]

    def validate_configuration(self):
        """Validate robot configuration."""
        errors = []
        
        # Check basic properties
        if not self.name:
            errors.append("Robot name not set")
            
        if not self._usd_path:
            errors.append("USD file path not set")
        
        # Check leg configuration
        if not self.leg_names or len(self.leg_names) != self.n_legs:
            errors.append(f"Expected {self.n_legs} legs, got {len(self.leg_names) if self.leg_names else 0}")
        
        # Check DOF configuration
        if not self.dof_names or len(self.dof_names) != self.expected_n_dofs:
            errors.append(f"Expected {self.expected_n_dofs} DOFs, got {len(self.dof_names) if self.dof_names else 0}")
        
        # Check body configuration (base + 4*legs + 2 head parts for Go2)
        expected_bodies = 1 + (self.n_legs * 4) + 2  # base + 4 parts per leg + head_upper + head_lower
        if not self.body_names or len(self.body_names) != expected_bodies:
            errors.append(f"Expected {expected_bodies} bodies, got {len(self.body_names) if self.body_names else 0}")
        
        # Check foot configuration
        if not self.foot_names or len(self.foot_names) != self.n_legs:
            errors.append(f"Expected {self.n_legs} feet, got {len(self.foot_names) if self.foot_names else 0}")
        
        # Check DOF limits
        #if not self.dof_limit or len(self.dof_limit) != self.expected_n_dofs:
        #   errors.append(f"Expected {self.expected_n_dofs} DOF limits, got {len(self.dof_limit) if self.dof_limit else 0}")
        
        if errors:
            raise ValueError(f"Go2 robot configuration errors: {'; '.join(errors)}")
        
        return True

    def __str__(self):
        return f"{self.name} ({self.n_dofs} DOFs, {self.n_bodies} bodies)"


@register_quadruped("unitree_go2")
class UnitreeGo2Standard(UnitreeGo2):
    """Standard Unitree Go2 robot configuration."""
    
    def __init__(self):
        super().__init__()
        # Validate configuration on initialization
        self.validate_configuration()

    def __str__(self):
        return super().__str__() + " [Standard]"


# Alternative configurations for different scenarios
@register_quadruped("unitree_go2_rough")
class UnitreeGo2Rough(UnitreeGo2):
    """Unitree Go2 configured for rough terrain."""
    
    def __init__(self):
        super().__init__()
        
        # Modify actuator settings for rough terrain
        self.actuators["base_legs"].stiffness = {
            ".*_hip_joint": 100.0,     # Increased stiffness for stability
            ".*_thigh_joint": 100.0,   
            ".*_calf_joint": 100.0,    
        }
        self.actuators["base_legs"].damping = {
            ".*_hip_joint": 5.0,       # Increased damping for stability
            ".*_thigh_joint": 5.0,    
            ".*_calf_joint": 5.0,     
        }
        
        self.validate_configuration()

    def __str__(self):
        return super().__str__() + " [Rough Terrain]"


@register_quadruped("unitree_go2_speed")
class UnitreeGo2Speed(UnitreeGo2):
    """Unitree Go2 configured for high-speed locomotion."""
    
    def __init__(self):
        super().__init__()
        
        # Modify actuator settings for speed
        self.actuators["base_legs"].stiffness = {
            ".*_hip_joint": 60.0,      # Reduced stiffness for agility
            ".*_thigh_joint": 60.0,   
            ".*_calf_joint": 60.0,    
        }
        self.actuators["base_legs"].damping = {
            ".*_hip_joint": 3.0,       # Reduced damping for responsiveness
            ".*_thigh_joint": 3.0,    
            ".*_calf_joint": 3.0,     
        }
        
        self.validate_configuration()

    def __str__(self):
        return super().__str__() + " [High Speed]"


# Create Isaac Lab compatible ArticulationCfg (commented for reference)
# GO2_CFG = ArticulationCfg(
#     spawn=sim_utils.UsdFileCfg(
#         usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Robots/Unitree/Go2/go2.usd",
#         activate_contact_sensors=True,
#         rigid_props=sim_utils.RigidBodyPropertiesCfg(
#             disable_gravity=False,
#             retain_accelerations=False,
#             max_depenetration_velocity=1.0,
#         ),
#         articulation_props=sim_utils.ArticulationRootPropertiesCfg(
#             enabled_self_collisions=False,
#             solver_position_iteration_count=4,
#             solver_velocity_iteration_count=0,
#         ),
#     ),
#     init_state=UnitreeGo2().init_state,
#     actuators=UnitreeGo2().actuators,
# )
