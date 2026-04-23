# Project Setup and Environment Registration

This README guides you through adding custom Spider environments to Gymnasium and Gymnasium-Robotics, as well as the overall project structure and file placement.

## 1. Add the `spider.xml` Model

1. Locate your Gymnasium MuJoCo assets directory:
   ```
   <your-python-env>/Lib/site-packages/gymnasium/envs/mujoco/assets
   ```
2. Copy `spider.xml` from the project root into this directory:
   ```bash
   cp path/to/project/spider.xml <your-python-env>/Lib/site-packages/gymnasium/envs/mujoco/assets/
   ```

## 2. Register the Spider Environment

1. In the MuJoCo environments folder, add the Python file:
   ```
   <your-python-env>/Lib/site-packages/gymnasium/envs/mujoco/spider_v0.py
   ```
   with your `SpiderEnv` implementation.

2. Open the `__init__.py` file in:
   ```
   <your-python-env>/Lib/site-packages/gymnasium/envs/
   ```
3. Add the following registration code:
   ```python
   from gymnasium.envs.registration import register

   register(
       id="Spider-v0",
       entry_point="gymnasium.envs.mujoco.spider_v0:SpiderEnv",
       max_episode_steps=1000,
       reward_threshold=6000.0,
   )
   ```

## 3. Add the Custom Maze Environment

1. Copy `spider_maze_v0.py` into the Gymnasium-Robotics maze folder:
   ```
   <your-python-env>/Lib/site-packages/gymnasium_robotics/envs/maze/spider_maze_v0.py
   ```
2. Update the package `__init__.py` in:
   ```
   <your-python-env>/Lib/site-packages/gymnasium_robotics/
   ```
   to import or register your custom maze environment, for example:
   ```python
   from .envs.maze.spider_maze_v0 import SpiderMaze_UMaze

   __all__ = [
       # ... other environments ...
       "SpiderMaze_UMaze",
   ]
   ```

## 4. Project Directory Structure

Ensure your project root has the following layout:

```
Project/
├── Videos/
│   ├── ddpg-spider-step-0-to-step-2000.mp4
│   ├── ppo-ant-step-0-to-step-1000.mp4
│   ├── ppo-spid-step-0-to-step-1000.mp4
│   ├── ppo-spid-step-0-to-step-2000.mp4
│   ├── ppo-spider-step-0-to-step-1000.mp4
│   └── td3-spider-step-0-to-step-2000.mp4
├── main.py
├── maze_v4.py
├── spider_maze_v0.py
├── spider_v0.py
├── spiderm.zip
├── train.py
├── __init__.py
└── spider.xml
```

- **`Videos/`**: Recorded rollouts for each algorithm and environment.
- **`main.py`**: Entry point for experiments.
- **`maze_v4.py`**, **`spider_maze_v0.py`**, **`spider_v0.py`**: Custom environment definitions.
- **`spiderm.zip`**: Pretrained model archive.
- **`train.py`**: Training script.
- **`spider.xml`**: MuJoCo model file.

## 5. Verification and Testing

1. Restart your Python session or IDE to reload Gymnasium.
2. Run a quick test:
   ```python
   import gymnasium as gym
   env = gym.make("Spider-v0")
   obs, _ = env.reset()
   print("Spider-v0 environment loaded successfully")
   ```
3. For the Maze variant:
   ```python
   import gymnasium as gym
   env = gym.make("SpiderMaze_UMaze-v0")
   obs, _ = env.reset()
   print("SpiderMaze_UMaze-v0 environment loaded successfully")
   ```

## 🎥 Output Videos
![Spider Demo](videos/ddpg-spider-step-0-to-step-2000.mp4)
