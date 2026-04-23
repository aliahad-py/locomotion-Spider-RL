from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecVideoRecorder
from stable_baselines3 import PPO
from gymnasium.spaces import Box
import gymnasium_robotics
import gymnasium as gym
import numpy as np
import os

gym.register_envs(gymnasium_robotics)

ENV_ID = "SpiderMaze_UMaze-v0"
MODEL_PATH = "spidermaze"
TIMESTEPS = 10_000_000
N_ENVS = 10
VIDEO_SIZE = 2000

# ================================
# ✅ Control Frequency Wrapper (PRO WAY)
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
# ✅ Flatten Observation Wrapper
# ================================
class FlattenObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        obs_space = env.observation_space.spaces
        total_size = sum(np.prod(space.shape) for space in obs_space.values())+1
        self.observation_space = Box(
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

# ================================
# ✅ Action Scaling + Smoothing
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
        # Smooth actions
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
# ENV FACTORY
# ================================
def make_env(render=False):
    env = gym.make(
        ENV_ID,
        render_mode="human" if render else "rgb_array",
        max_episode_steps=2000
    )
    #env = FlattenObservationWrapper(env)
    env = ActionWrapperSmooth(env, scale=0.4)
    env = ControlFrequencyWrapper(env, repeat=5)
    return env



# ================================
# TRAINING
# ================================
def train():
    env = SubprocVecEnv([make_env for _ in range(N_ENVS)])
    if os.path.exists(MODEL_PATH + ".zip"):
        print(f"Loading existing model from {MODEL_PATH}...")
        model = PPO.load(MODEL_PATH, env=env)
    else:
        print("Creating new model...")
        model = PPO(
            "MultiInputPolicy",
            env,
            learning_rate=5e-5,
            n_steps=2048,
            batch_size=256,
            n_epochs=30,          # ✅ reduced (was too high)
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.1,
            ent_coef=0.01,       # ✅ less randomness → stable walking
            vf_coef=0.5,
            max_grad_norm=0.5,
            verbose=1
        )

    try:
        model.learn(total_timesteps=TIMESTEPS)
        model.save(MODEL_PATH)
    except KeyboardInterrupt:
        print("Training interrupted. Saving...")
        model.save(MODEL_PATH)

    env.close()
    print("✅ Training complete!")


# ================================
# VIDEO RECORDING
# ================================
def record_video():
    video_folder = "videos/"
    os.makedirs(video_folder, exist_ok=True)
    eval_env = DummyVecEnv([
        lambda: make_env()
    ])
    eval_env = VecVideoRecorder(
        eval_env,
        video_folder,
        record_video_trigger=lambda step: step == 0,
        video_length=VIDEO_SIZE,
        name_prefix="ppo-spiderMaze1"
    )
    model = PPO.load(MODEL_PATH)
    obs = eval_env.reset()
    for _ in range(VIDEO_SIZE):
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, _ = eval_env.step(action)
        if done:
            obs = eval_env.reset()
    eval_env.close()
    print("✅ Video recorded!")


# ================================
# MAIN
# ================================
if __name__ == "__main__":
    train()
    record_video()
