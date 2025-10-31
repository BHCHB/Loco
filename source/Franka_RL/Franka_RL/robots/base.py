from abc import ABC, abstractmethod
import os
import numpy as np


class DexHand(ABC):
    def __init__(self):
        self._usd_path = None
        self.side = None
        self.name = None
        self.body_names = None
        self.dof_names = None
        self.wrist_name = None
        self.contact_body_names = None
        self.init_state = None
        self.actuators = None
        self.dof_limit = None
        self.weight_idx = None

    
    @property
    def n_dofs(self):
        return len(self.dof_names)

    @property
    def n_bodies(self):
        return len(self.body_names)
    
class QuadrupedRobot(ABC):
    """Base class for quadruped robots like Go2, A1, etc."""
    
    def __init__(self):
        self._usd_path = None
        self.name = None
        self.body_names = None
        self.dof_names = None
        self.contact_body_names = None
        self.init_state = None
        self.actuators = None
        self.dof_limit = None
        
        # Quadruped specific properties
        self.base_name = None
        self.leg_names = None  # e.g., ["FL", "FR", "RL", "RR"]
        self.foot_names = None  # e.g., ["FL_foot", "FR_foot", ...]
        self.hip_names = None
        self.thigh_names = None
        self.calf_names = None
        
        # Robot specifications
        self.n_legs = 4
        self.dof_per_leg = 3  # hip, thigh, calf
        
        # Weight indices for reward computation
        self.weight_idx = None

    @property
    def n_dofs(self) -> int:
        """Number of degrees of freedom."""
        return len(self.dof_names) if self.dof_names else 0

    @property
    def n_bodies(self) -> int:
        """Number of body parts."""
        return len(self.body_names) if self.body_names else 0

    @property
    def expected_n_dofs(self) -> int:
        """Expected number of DOFs for a quadruped (typically 12)."""
        return self.n_legs * self.dof_per_leg
