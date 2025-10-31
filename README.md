# RL for Dexhand & Arm Manipulation

## Overview

This repository provides RL framework for Dexhand and robotics arm manipulation tasks. It allows you to train your agent to complete diverse Human-Object Interection(HOI) tasks by imitating demo trajectories.

## Installation

- Clone or copy this repository:
    ```bash
    git clone https://github.com/BHCHB/Loco.git
    ```

- It's recommended to create a new python environment for the repository(e.g. Conda environment)
:
    ```bash
    conda create -n frankarl python=3.11
    conda activate loco
    ```

- Install Isaac Lab by following the [installation guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html).
  We recommend using the conda installation as it simplifies calling Python scripts from the terminal.

- Using a python interpreter that has Isaac Lab installed, install the dependences and the library in editable mode using:

    ```bash
    pip install --no-index --no-cache-dir pytorch3d==0.7.3 -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py38_cu117_pyt1131/download.html
    pip install -r requirements.txt
    # use 'PATH_TO_isaaclab.sh|bat -p' instead of 'python' if Isaac Lab is not installed in Python venv or conda
    python -m pip install -e source/Franka_RL
    ```

- Verify that the extension is correctly installed by listing the available tasks (Note: If the task name changes, it may be necessary to update the search pattern `"Template-"` in the `scripts/list_envs.py` file so that it can be listed.):
    ```bash
    # use 'FULL_PATH_TO_isaaclab.sh|bat -p' instead of 'python' if Isaac Lab is not installed in Python venv or conda
    python scripts/list_envs.py
    ```
## Robots

 - This repo now support robot models in following table:

    | Category | Robot Model | Config File Path | Branch |
    |------|------|------|------|
    | Dexhand | Shadow Hand | source/Franka_RL/Franka_RL/robots/Shadow.py | origin/master|

    For instance, the dexhand models can be loaded into environments through `DexHandFactory` class:
    ```python
    from Franka_RL.robots import DexHandFactory
    robot = DexHandFactory.create_hand(
        dexhand_type=your_dexhand_name, 
        side=your_dexhand_side
    )
    ```
 - To customize your own robot models, please refer to the examples in `source/Franka_RL/Franka_RL/robots`. Follow these steps: 
    1. (Optional) Create a new Factory class e.g. `RobotFactory` in `source/Franka_RL/Franka_RL/robots/factory.py`.
    2. (Optional) Create a new Base class e.g `BaseRobot` in `source/Franka_RL/Franka_RL/robots/base.py`.
    3. Create your robot class in `source/Franka_RL/Franka_RL/robots`. Ensure that the robot class inherits from one of the Base class (e.g. `Dexhand` or `BaseRobot`), and it's properly registered in `__init__.py` 

## Datasets

 <!-- TODO: complete the Dataset Factory-->

## RL Libraries
 - The repo provides three RL libraries at present: `rsl_rl`, `rl_games` and `skrl`. It's worth noting that this repo has developed novel `OnPolicyPPORunner` and `ActorCritic` class for `rsl_rl` lib. It is possible to define more complex actor and critic networks, not limited to MLP now. For more details, please check `source/Franka_RL/Franka_RL/runners` for the modified classes, and check `source/Franka_RL/Franka_RL/tasks/direct/franka_rl/agents/rsl_rl_ppo_cfg.yaml` for a template config file. 
 
    Run `scripts/rsl_rl/train_new.py` to run the modified `rsl_rl` lib. This repo also reserves the original `rsl_rl` lib by running `scripts/rsl_rl/train.py`.

## Tasks

 - The existed tasks are listed in the following table:

    | Task Name | Task Path | Task Description|
    |-------|--------|----|
    | Franka-Train | source/Franka_RL/Franka_RL/tasks/direct/franka_rl | Train a Franka Panda Arm following end effector trajectories|
## Troubleshooting

### Pylance Missing Indexing of Extensions

In some VsCode versions, the indexing of part of the extensions is missing.
In this case, add the path to your extension in `.vscode/settings.json` under the key `"python.analysis.extraPaths"`.

```json
{
    "python.analysis.extraPaths": [
        "<path-to-ext-repo>/source/Franka_RL"
    ]
}
```

### Pylance Crash

If you encounter a crash in `pylance`, it is probable that too many files are indexed and you run out of memory.
A possible solution is to exclude some of omniverse packages that are not used in your project.
To do so, modify `.vscode/settings.json` and comment out packages under the key `"python.analysis.extraPaths"`
Some examples of packages that can likely be excluded are:

```json
"<path-to-isaac-sim>/extscache/omni.anim.*"         // Animation packages
"<path-to-isaac-sim>/extscache/omni.kit.*"          // Kit UI tools
"<path-to-isaac-sim>/extscache/omni.graph.*"        // Graph UI tools
"<path-to-isaac-sim>/extscache/omni.services.*"     // Services tools
...
```