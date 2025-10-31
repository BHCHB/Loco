import os
import importlib
from .base import DexHand,QuadrupedRobot   


class DexHandFactory:
    _registry = {}

    @classmethod
    def register(cls, dexhand_type: str, hand_class):
        """Register a new hand type."""
        cls._registry[dexhand_type] = hand_class

    @classmethod
    def create_hand(cls, dexhand_type: str, side: str = 'right', *args, **kwargs) -> DexHand:
        assert side in ["left", "right"], f"Invalid side '{side}', must be 'left' or 'right'."
        """Create a hand instance by type."""
        dexhand_type += "_rh" if side == "right" else "_lh"
        if dexhand_type not in cls._registry:
            raise ValueError(f"Hand type '{dexhand_type}' not registered.")
        return cls._registry[dexhand_type](*args, **kwargs)

    @classmethod
    def auto_register_hands(cls, directory: str, base_package: str):
        """Automatically import all hand modules in the directory."""
        for filename in os.listdir(directory):
            if filename.endswith(".py") and filename != "__init__.py":
                module_name = f"{base_package}.{filename[:-3]}"
                try:
                    importlib.import_module(module_name)
                except ImportError as e:
                    print(f"Warning: Could not import hand module {module_name}: {e}")
                

class QuadrupedRobotFactory:
    """Factory for creating quadruped robots."""
    _registry = {}

    @classmethod
    def register(cls, robot_type: str, robot_class):
        """Register a new quadruped robot type."""
        cls._registry[robot_type] = robot_class

    @classmethod
    def create_robot(cls, robot_type: str, *args, **kwargs) -> QuadrupedRobot:
        """Create a quadruped robot instance by type."""
        if robot_type not in cls._registry:
            available_types = list(cls._registry.keys())
            raise ValueError(f"Robot type '{robot_type}' not registered. Available types: {available_types}")
        return cls._registry[robot_type](*args, **kwargs)

    @classmethod
    def get_available_robots(cls):
        """Get list of available robot types."""
        return list(cls._registry.keys())

    @classmethod
    def auto_register_robots(cls, directory: str, base_package: str):
        """Automatically import all robot modules in the directory."""
        for filename in os.listdir(directory):
            if filename.endswith(".py") and filename not in ["__init__.py", "base.py", "factory.py", "decorators.py"]:
                module_name = f"{base_package}.{filename[:-3]}"
                try:
                    importlib.import_module(module_name)
                except ImportError as e:
                    print(f"Warning: Could not import robot module {module_name}: {e}")