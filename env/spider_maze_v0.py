"""A maze environment with the Gymnasium Ant agent (https://github.com/Farama-Foundation/Gymnasium/blob/main/gymnasium/envs/mujoco/ant_v4.py).

The code is inspired by the D4RL repository hosted on GitHub (https://github.com/Farama-Foundation/D4RL), published in the paper
'D4RL: Datasets for Deep Data-Driven Reinforcement Learning' by Justin Fu, Aviral Kumar, Ofir Nachum, George Tucker, Sergey Levine.

Original Author of the code: Justin Fu

The modifications made involve reusing the code in Gymnasium for the Ant environment and in `point_maze/maze_env.py`.
The new code also follows the Gymnasium API and Multi-goal API

This project is covered by the Apache 2.0 License.
"""

import sys
from os import path
from typing import Dict, List, Optional, Union

import numpy as np
from gymnasium import spaces
from gymnasium.envs.mujoco.spider_v0 import SpiderEnv
from gymnasium.utils.ezpickle import EzPickle

from gymnasium_robotics.envs.maze.maps import U_MAZE
from gymnasium_robotics.envs.maze.maze_v4 import MazeEnv
from gymnasium_robotics.utils.mujoco_utils import MujocoModelNames


class SpiderMazeEnv(MazeEnv, EzPickle):
    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
        "render_fps": 50,
    }

    def __init__(
        self,
        render_mode: Optional[str] = None,
        maze_map: List[List[Union[str, int]]] = U_MAZE,
        reward_type: str = "sparse",
        continuing_task: bool = True,
        reset_target: bool = False,
        xml_file: Union[str, None] = None,
        **kwargs,
    ):
        if xml_file is None:
            spider_xml_file_path = path.join(
                path.dirname(sys.modules[SpiderEnv.__module__].__file__), "assets/spider.xml"
            )
        else:
            spider_xml_file_path = xml_file
        super().__init__(
            agent_xml_path=spider_xml_file_path,
            maze_map=maze_map,
            maze_size_scaling=4,
            maze_height=0.5,
            reward_type="both",# reward_type,
            continuing_task=continuing_task,
            reset_target=reset_target,
            **kwargs,
        )
        # Create the MuJoCo environment, include position observation of the Ant for GoalEnv
        self.spider_env = SpiderEnv(
            xml_file=self.tmp_xml_file_path,
            exclude_current_positions_from_observation=False,
            # exclude_current_positions_from_observation=True,
            render_mode=render_mode,
            reset_noise_scale=0.0,
            **kwargs,
        )
        self._model_names = MujocoModelNames(self.spider_env.model)
        self.target_site_id = self._model_names.site_name2id["target"]

        self.action_space = self.spider_env.action_space
        obs_shape: tuple = self.spider_env.observation_space.shape
        self.observation_space = spaces.Dict(
            dict(
                observation=spaces.Box(
                    -np.inf, np.inf, shape=(obs_shape[0] - 2,), dtype="float64"
                ),
                achieved_goal=spaces.Box(-np.inf, np.inf, shape=(2,), dtype="float64"),
                desired_goal=spaces.Box(-np.inf, np.inf, shape=(2,), dtype="float64"),
            )
        )

        self.render_mode = render_mode
        EzPickle.__init__(
            self,
            render_mode,
            maze_map,
            reward_type,
            continuing_task,
            reset_target,
            **kwargs,
        )

    def reset(self, *, seed: Optional[int] = None, **kwargs):
        super().reset(seed=seed, **kwargs)

        self.spider_env.init_qpos[:2] = self.reset_pos

        obs, info = self.spider_env.reset(seed=seed)
        obs_dict = self._get_obs(obs)
        info["success"] = bool(
            np.linalg.norm(obs_dict["achieved_goal"] - self.goal) <= 0.45
        )

        return obs_dict, info

    def step(self, action):
        """
        group = action[-1]
        action = action[:6]
        if group == 0:    ## front, back
            front = action[0:3]
            back = action[3:]
            action = np.array([front, -front, back, -back]).flatten()
        elif group == 1:    ## left, right
            left1 = action[0:3]
            left2 = left1.copy()
            left2[0] = -left2[0]
            right1 = action[3:]
            right2 = right1.copy()
            right2[0] = -right2[0]
            action = np.array([right1, left1,left2, right2]).flatten()
        elif group == 2:   ## diagonal
            D1 = action[0:3]  ## r1, l2
            D1A = D1.copy()
            D1A[1:] = -D1A[1:]
            D2 = action[3:]   ## l1, r2
            D2A = D2.copy()
            D2A[1:] = -D2A[1:]
            action = np.array([D1, D2, D1A, D2A]).flatten()
        else:
            pass
        """
        spider_obs, spider_reward, spider_terminated, _, info = self.spider_env.step(action)
        obs = self._get_obs(spider_obs)
        maze_reward = self.compute_reward(obs["achieved_goal"], self.goal, info)
        #print(f"Before Maze reward: {maze_reward}")
        # maze_reward = [maze_reward*2 if maze_reward<0 else maze_reward/2][0]
        #print(f"After Maze reward: {maze_reward}")
        reward = maze_reward #+ (spider_reward)
        #print(f"Spider reward: {spider_reward}")
        terminated = self.compute_terminated(obs["achieved_goal"], self.goal, info) or spider_terminated
        truncated = self.compute_truncated(obs["achieved_goal"], self.goal, info)
        info["success"] = bool(np.linalg.norm(obs["achieved_goal"] - self.goal) <= 0.45)

        if self.render_mode == "human":
            self.render()

        # Update the goal position if necessary
        self.update_goal(obs["achieved_goal"])
        return obs, reward, terminated, truncated, info

    def _get_obs(self, spider_obs: np.ndarray) -> Dict[str, np.ndarray]:
        achieved_goal = spider_obs[:2]
        observation = spider_obs[2:]
        #return np.concatenate((observation, achieved_goal, self.goal.copy()))
        return {
            "observation": observation.copy(),
            "achieved_goal": achieved_goal.copy(),
            "desired_goal": self.goal.copy(),
        }

    def update_target_site_pos(self):
        self.spider_env.model.site_pos[self.target_site_id] = np.append(
            self.goal, self.maze.maze_height / 2 * self.maze.maze_size_scaling
        )

    def render(self):
        return self.spider_env.render()

    def close(self):
        super().close()
        self.spider_env.close()

    @property
    def model(self):
        return self.spider_env.model

    @property
    def data(self):
        return self.spider_env.data
