from stable_baselines3 import PPO
import gymnasium as gym
import gymnasium_robotics
import numpy as np
from gymnasium.spaces import Box

gym.register_envs(gymnasium_robotics)

ENV_ID = "SpiderMaze_UMaze-v0"
MODEL_PATH = "spidermaze"


# ================================
# Wrappers (MUST match training)
# ================================
class ControlFrequencyWrapper(gym.Wrapper):
    def __init__(self, env, repeat=2):
        super().__init__(env)
        self.repeat = repeat
    def step(self, action):
        total_reward = 0.0
        done = False
        trunc = False
        for _ in range(self.repeat):
            obs, reward, done, trunc, info = self.env.step(action)
            total_reward += reward
            if done or trunc:
                break
        return obs, total_reward, done, trunc, info


class FlattenObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        obs_space = env.observation_space.spaces
        total_size = sum(np.prod(space.shape) for space in obs_space.values())+1
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(total_size,),
            dtype=np.float32
        )
    def observation(self, obs):
        achieved_goal = obs["achieved_goal"]
        desired_goal = obs["desired_goal"]
        distance = np.linalg.norm(achieved_goal - desired_goal, axis=-1)
        flat_obs = np.concatenate([
            obs[key].ravel() for key in sorted(obs.keys())
        ])
        # Append distance
        return np.concatenate([flat_obs, np.array([distance], dtype=np.float32)])


class ActionWrapperSmooth(gym.ActionWrapper):
    def __init__(self, env, scale=0.8, alpha=1):
        super().__init__(env)
        self.scale = scale
        self.alpha = alpha
        self.prev_action = np.zeros(env.action_space.shape, dtype=np.float32)
        
    def action(self, action):
        smoothed = self.alpha * action + (1 - self.alpha) * self.prev_action
        self.prev_action = smoothed
        return smoothed * self.scale
class ActionScaleWrapper(gym.ActionWrapper):
    def __init__(self, env, scale=0.8):
        super().__init__(env)
        self.scale = scale
        # NEW action space exposed to agent
        self.action_space = Box(
            low=-scale,
            high=scale,
            shape=env.action_space.shape,
            dtype=np.float32
        )
    def action(self, action):
        return action

# ================================
# Create Environment
# ================================
def make_env(render=True):
    env = gym.make(
        ENV_ID,
        render_mode="human" if render else None,
        max_episode_steps=2000
    )
    #env = FlattenObservationWrapper(env)
    #env = ActionWrapperSmooth(env, scale=1)
    env = ControlFrequencyWrapper(env, repeat=5)
    return env


# ================================
# TEST FUNCTION
# ================================
def test(n_episodes=5, render=True):
    env = make_env(render=render)
    model = PPO.load(MODEL_PATH)

    rewards = []

    for ep in range(n_episodes):
        obs, _ = env.reset()
        done = False
        trunc = False
        total_reward = 0

        while not (done or trunc):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, trunc, info = env.step(action)
            total_reward += reward
            print(f"Actions: {action}, Reward: {reward}")
        rewards.append(total_reward)
        print(f"Episode {ep+1}: Reward = {total_reward:.2f}")

    print("\n📊 Test Summary")
    print(f"Average Reward: {np.mean(rewards):.2f}")
    print(f"Min Reward: {np.min(rewards):.2f}")
    print(f"Max Reward: {np.max(rewards):.2f}")

    env.close()


# ================================
# ENTRY POINT
# ================================
if __name__ == "__main__":
    test(n_episodes=100, render=True)