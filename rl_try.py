import gymnasium as gym
from gymnasium import spaces
import numpy as np
from catheter_system import catheter_update, catheter_params
from compute_desired_trajectory import compute_desired_trajectory

class CatheterEnv(gym.Env):
    def __init__(self, horizon=100):
        super(CatheterEnv, self).__init__()
        self.Ts = catheter_params['Ts']
        self.horizon = horizon
        self.magnet_distance = 0.25
        self.step_size = catheter_params['v'] * self.Ts
        self.action_space = spaces.Box(low=np.deg2rad(60), high=np.deg2rad(120), shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(7,), dtype=np.float32)

        self.reset()

    def compute_control_input(self, x, theta):
        tip = x[:2]
        offset = np.array([0.0, self.magnet_distance])
        magnet_xy = tip + offset
        return np.hstack((magnet_xy, theta))

    def step(self, action):
        theta = action[0]
        u = self.compute_control_input(self.state, theta)
        next_state = catheter_update(0, self.state, u, {**catheter_params, "return_full": False})

        self.step_count += 1
        tip = next_state[:2]
        desired_tip = self.Xd[self.step_count][:2]
        
        # Compute distance to trajectory
        dist_to_traj = np.linalg.norm(tip - desired_tip)

        # === Reward ===
        reward = -dist_to_traj
        if tip[1] > desired_tip[1]:
            reward -= 5.0 * (tip[1] - desired_tip[1])**2

        # === Early termination if too far from path ===
        MAX_DIST = 0.02  # 5 cm
        terminated = bool(dist_to_traj > MAX_DIST)
        if terminated:
            reward -= 10.0  # Strong penalty for failing

        # === Episode end due to horizon ===
        truncated = bool(self.step_count >= self.horizon)


        self.state = next_state
        obs = self._get_obs()
        return obs, reward, terminated, truncated, {}

    def _get_obs(self):
        tip = self.state[:2]
        theta = np.array([self.prev_theta])
        desired_tip = self.Xd[self.step_count][:2]
        obs = np.concatenate([tip, desired_tip, theta, self.state[2:]])
        return obs.astype(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)  # for seeding the RNG properly
        self.state = np.array([0.03, 0.0, 0.0, 0.0])
        self.step_count = 0
        self.Xd, _ = compute_desired_trajectory(k=0, N=self.horizon+20, step=self.step_size)
        self.prev_theta = np.deg2rad(90)
        return self._get_obs(), {}  # Return (obs, info)

