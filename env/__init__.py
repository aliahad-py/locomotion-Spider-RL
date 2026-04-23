# noqa: D104
from gymnasium.envs.registration import register

from gymnasium_robotics.core import GoalEnv
from gymnasium_robotics.envs.maze import maps
from gymnasium_robotics.envs.multiagent_mujoco import mamujoco_v1


def register_envs():
    """Register all environment ID's to Gymnasium."""

    def _merge(a, b):
        a.update(b)
        return a

    for reward_type in ["sparse", "dense"]:
        suffix = "Dense" if reward_type == "dense" else ""
        kwargs = {
            "reward_type": reward_type,
        }
        # ----- SpiderMaze -----
        version = "v0"
        register(
            id="Spider",
            entry_point=f"gymnasium.envs.mujoco.spider_{version}:SpiderEnv",
            max_episode_steps=1000,
            reward_threshold=6000.0,
        )

        register(
            id=f"SpiderMaze_UMaze{suffix}-{version}",
            entry_point=f"gymnasium_robotics.envs.maze.spider_maze_{version}:SpiderMazeEnv",
            kwargs=_merge(
                {
                    "maze_map": maps.U_MAZE,
                },
                kwargs,
            ),
            max_episode_steps=700,
        )

        register(
            id=f"SpiderMaze_Open{suffix}-{version}",
            entry_point=f"gymnasium_robotics.envs.maze.spider_maze_{version}:SpiderMazeEnv",
            kwargs=_merge(
                {
                    "maze_map": maps.OPEN,
                },
                kwargs,
            ),
            max_episode_steps=700,
        )

        register(
            id=f"SpiderMaze_Open_Diverse_G{suffix}-{version}",
            entry_point=f"gymnasium_robotics.envs.maze.spider_maze_{version}:SpiderMazeEnv",
            kwargs=_merge(
                {
                    "maze_map": maps.OPEN_DIVERSE_G,
                },
                kwargs,
            ),
            max_episode_steps=700,
        )

        register(
            id=f"SpiderMaze_Open_Diverse_GR{suffix}-{version}",
            entry_point=f"gymnasium_robotics.envs.maze.spider_maze_{version}:SpiderMazeEnv",
            kwargs=_merge(
                {
                    "maze_map": maps.OPEN_DIVERSE_GR,
                },
                kwargs,
            ),
            max_episode_steps=700,
        )

        register(
            id=f"SpiderMaze_Medium{suffix}-{version}",
            entry_point=f"gymnasium_robotics.envs.maze.spider_maze_{version}:SpiderMazeEnv",
            kwargs=_merge(
                {
                    "maze_map": maps.MEDIUM_MAZE,
                },
                kwargs,
            ),
            max_episode_steps=1000,
        )

        register(
            id=f"SpiderMaze_Medium_Diverse_G{suffix}-{version}",
            entry_point=f"gymnasium_robotics.envs.maze.spider_maze_{version}:SpiderMazeEnv",
            kwargs=_merge(
                {
                    "maze_map": maps.MEDIUM_MAZE_DIVERSE_G,
                },
                kwargs,
            ),
            max_episode_steps=1000,
        )

        register(
            id=f"SpiderMaze_Medium_Diverse_GR{suffix}-{version}",
            entry_point=f"gymnasium_robotics.envs.maze.spider_maze_{version}:SpiderMazeEnv",
            kwargs=_merge(
                {
                    "maze_map": maps.MEDIUM_MAZE_DIVERSE_GR,
                },
                kwargs,
            ),
            max_episode_steps=1000,
        )

        register(
            id=f"SpiderMaze_Large{suffix}-{version}",
            entry_point=f"gymnasium_robotics.envs.maze.spider_maze_{version}:SpiderMazeEnv",
            kwargs=_merge(
                {
                    "maze_map": maps.LARGE_MAZE,
                },
                kwargs,
            ),
            max_episode_steps=1000,
        )

        register(
            id=f"SpiderMaze_Large_Diverse_G{suffix}-{version}",
            entry_point=f"gymnasium_robotics.envs.maze.spider_maze_{version}:SpiderMazeEnv",
            kwargs=_merge(
                {
                    "maze_map": maps.LARGE_MAZE_DIVERSE_G,
                },
                kwargs,
            ),
            max_episode_steps=1000,
        )

        register(
            id=f"SpiderMaze_Large_Diverse_GR{suffix}-{version}",
            entry_point=f"gymnasium_robotics.envs.maze.Spider_maze_{version}:SpiderMazeEnv",
            kwargs=_merge(
                {
                    "maze_map": maps.LARGE_MAZE_DIVERSE_GR,
                },
                kwargs,
            ),
            max_episode_steps=1000,
        )

register_robotics_envs()