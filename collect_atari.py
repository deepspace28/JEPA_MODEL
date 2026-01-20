import ale_py  # <-- add this line first
import gymnasium as gym
import numpy as np
import os

os.makedirs("data/atari", exist_ok=True)

env = gym.make("ALE/Breakout-v5", render_mode="rgb_array")

for ep in range(100):
    obs, _ = env.reset()
    for t in range(200):
        action = env.action_space.sample()
        next_obs, reward, terminated, truncated, _ = env.step(action)

        np.save(f"data/atari/obs_{ep}_{t}.npy", obs)
        np.save(f"data/atari/act_{ep}_{t}.npy", action)
        np.save(f"data/atari/nxt_{ep}_{t}.npy", next_obs)

        obs = next_obs
        if terminated or truncated:
            break

env.close()
