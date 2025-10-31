from .base import DexHand,QuadrupedRobot

import os
from .factory import DexHandFactory,QuadrupedRobotFactory

current_dir = os.path.dirname(os.path.abspath(__file__))
base_package = __name__

DexHandFactory.auto_register_hands(current_dir, base_package)
QuadrupedRobotFactory.auto_register_robots(current_dir, base_package)