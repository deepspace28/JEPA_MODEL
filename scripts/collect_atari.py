import ale_py
import gymnasium as gym
import numpy as np
from pathlib import Path


def collect(root: str = "data/clips/train", episodes: int = 10, steps: int = 200, clip_len: int = 16):
    Path(root).mkdir(parents=True, exist_ok=True)
    env = gym.make("ALE/Breakout-v5", render_mode="rgb_array")
    clip_id = 0

    for ep in range(episodes):
        obs, _ = env.reset()
        frames = []
        for _ in range(steps):
            frames.append(obs)
            action = env.action_space.sample()
            obs, _reward, terminated, truncated, _ = env.step(action)
            if len(frames) == clip_len:
                np.save(Path(root) / f"clip_{clip_id:06d}.npy", np.array(frames, dtype=np.uint8))
                clip_id += 1
                frames = []
            if terminated or truncated:
                break
    env.close()


if __name__ == "__main__":
    collect()
