from .factory import DexHandFactory,QuadrupedRobotFactory

def register_dexhand(dexhand_type):
    def decorator(cls):
        DexHandFactory.register(dexhand_type, cls)
        return cls

    return decorator

def register_quadruped(robot_type):
    def decorator(cls):
        QuadrupedRobotFactory.register(robot_type, cls)
        return cls

    return decorator