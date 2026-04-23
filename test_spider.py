from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import gymnasium as gym
import numpy as np
from gymnasium.spaces import Box
from env import register
import time

register.register_envs()
# ================================
# CONFIG
# ================================
ENV_ID = "Spider"
MODEL_PATH = "spider"
EPISODES = 100

# ================================
# ✅ Control Frequency Wrapper (MATCH TRAINING)
# ================================
class ControlFrequencyWrapper(gym.Wrapper):
    def __init__(self, env, repeat=10):
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


# ================================
# ✅ Action Wrapper (MATCH TRAINING)
# ================================
class ActionWrapperSmooth(gym.ActionWrapper):
    def __init__(self, env, scale=0.8, alpha=0.3):
        super().__init__(env)
        self.scale = scale
        self.alpha = alpha
        self.prev_action = np.zeros(env.action_space.shape, dtype=np.float32)

        self.action_space = Box(
            low=-scale,
            high=scale,
            shape=env.action_space.shape,
            dtype=np.float32
        )

    def action(self, action):
        smoothed = self.alpha * action + (1 - self.alpha) * self.prev_action
        self.prev_action = smoothed
        return smoothed * self.scale


# ================================
# CREATE ENV (MATCH TRAINING)
# ================================
def make_env():
    env = gym.make(
        ENV_ID,
        max_episode_steps=1000,
        render_mode="human"
    )
    env = ControlFrequencyWrapper(env, repeat=10)   # 🔥 MUST match training
    env = ActionWrapperSmooth(env)                  # 🔥 MUST match training
    return env


env = DummyVecEnv([make_env])


# ================================
# LOAD MODEL
# ================================
model = PPO.load(MODEL_PATH)

# ================================
# TEST LOOP
# ================================
for episode in range(1, EPISODES + 1):
    obs = env.reset()
    done = False
    total_reward = 0

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        total_reward += reward[0]
        # Slow down visualization (optional)
        time.sleep(0.02)

        print(f"Action: {action}, Reward: {reward}")

    print(f"Episode {episode} | Total Reward: {total_reward}")

env.close()