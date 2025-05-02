from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import CheckpointCallback
from rl_try import CatheterEnv
import matplotlib.pyplot as plt
from compute_desired_trajectory import compute_desired_trajectory
import numpy as np
import os

# === Create Save Directory ===
save_path = "ppo_checkpoints"
os.makedirs(save_path, exist_ok=True)

# === Create Environment ===
env = CatheterEnv(horizon=100)
check_env(env)

# === Define Checkpoint Callback ===
checkpoint_callback = CheckpointCallback(
    save_freq=1e4,              # Save every 100 steps
    save_path=save_path,        # Directory to save to
    name_prefix="ppo_catheter", # Model file prefix
    save_replay_buffer=False,
    save_vecnormalize=False
)

# === Train Model with Callback ===
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=int(1e15), callback=checkpoint_callback)  # Adjusted total_timesteps for practicality

# === Test Trained Model ===
obs, _ = env.reset()
done = False
trajectory = [env.state[:2]]

while not done:
    action, _ = model.predict(obs)
    obs, reward, terminated, truncated, _ = env.step(action)
    done = terminated or truncated
    trajectory.append(env.state[:2])

# === Plot ===
trajectory = np.array(trajectory)
plt.plot(trajectory[:, 0], trajectory[:, 1], label="RL Tip Path")
Xd, _ = compute_desired_trajectory(k=0, N=120, step=env.step_size)
plt.plot(Xd[:len(trajectory), 0], Xd[:len(trajectory), 1], 'k--', label="Desired")
plt.xlabel("x (m)")
plt.ylabel("y (m)")
plt.legend()
plt.grid(True)
plt.title("RL vs Desired Trajectory")
plt.axis("equal")
plt.show()
