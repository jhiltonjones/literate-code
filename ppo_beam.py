# PPO Reinforcement Learning to optimize rod bending angle θ
# ----------------------------------------------------------
# This script defines a custom environment and trains a PPO agent using PyTorch
# to achieve a bending angle in the target range [5°, 40°].

import math
import random
import torch
import torch.nn as nn
import torch.optim as optim

# Environment and PPO hyperparameters
MAX_EPISODE_STEPS = 50      # maximum steps per episode
GAMMA = 0.99                # discount factor
LAMBDA_GAE = 0.95           # GAE lambda for advantage estimation
CLIP_EPS = 0.2              # PPO clipping epsilon
ENTROPY_COEF = 0.01         # entropy bonus coefficient
VALUE_COEF = 0.5            # value loss coefficient
ACTOR_LR = 3e-4             # learning rate for policy network
CRITIC_LR = 1e-3            # learning rate for value network
BATCH_SIZE = 1024           # time steps per training batch
MINIBATCH_SIZE = 256        # minibatch size for PPO updates
PPO_EPOCHS = 4              # number of update epochs per batch
NUM_TRAIN_ITERATIONS = 100  # number of training iterations
LOG_INTERVAL = 10           # log progress every N iterations

class RodBendingEnv:
    """
    Custom environment for rod bending angle optimization.
    Observation: [x, y, cos(psi), sin(psi), tip_x, tip_y, theta]
      - x, y: input parameters with x∈[0.01,0.25], y∈[0.01,0.15]
      - psi: orientation parameter (we use cos and sin of psi for continuity)
      - tip_x, tip_y: current rod tip position
      - theta: current bending angle of the rod (in radians)
    Actions (discrete): 0 = decrease angle, 1 = no change, 2 = increase angle
    Reward: positive when θ is in target range [5°, 40°], negative when outside.
    """
    def __init__(self):
        self.length = 0.3  # rod length
        # Target range for bending angle (in radians)
        self.target_min_angle = math.radians(5.0)
        self.target_max_angle = math.radians(40.0)
        self.state = None
        self.step_count = 0
        self.done = False
    
    def reset(self):
        """Reset the environment with new random inputs and initial angle."""
        x = 0.01 + random.random() * 0.24   # sample x ∈ [0.01, 0.25]
        y = 0.01 + random.random() * 0.14   # sample y ∈ [0.01, 0.15]
        psi = random.random() * 2 * math.pi # sample ψ ∈ [0, 2π]
        theta = 0.0  # start unbent
        tip_x, tip_y = self._compute_tip_position(psi, theta)
        # State uses cos(psi), sin(psi) instead of psi to avoid angle discontinuity
        self.state = [x, y, math.cos(psi), math.sin(psi), tip_x, tip_y, theta]
        self.step_count = 0
        self.done = False
        return self.state
    
    def step(self, action):
        """Apply an action and return (next_state, reward, done, info)."""
        if self.done:
            raise RuntimeError("Cannot call step() on a terminated episode. Please reset().")
        # Determine direction of angle change
        if action == 0:   # decrease θ
            action_dir = -1
        elif action == 2: # increase θ
            action_dir = 1
        else:             # no change
            action_dir = 0
        
        # Unpack state
        x, y = self.state[0], self.state[1]
        cos_psi, sin_psi = self.state[2], self.state[3]
        psi = math.atan2(sin_psi, cos_psi)  # reconstruct psi
        theta = self.state[6]
        
        # Compute physics feedback
        dF_dtheta = compute_dF_dtheta_symbolic(x, y, psi, theta)
        F = compute_F(x, y, psi, theta)
        # Base angle increment of 1° (in radians)
        base_delta = math.radians(1.0)
        actual_change = action_dir * base_delta
        # Adjust angle change by rod physics (resistance/assistance)
        actual_change -= 0.1 * F
        actual_change /= (1.0 + abs(dF_dtheta))
        # Update bending angle θ
        theta += actual_change
        if theta < 0:  # prevent negative bending
            theta = 0.0
        
        # Compute new tip position
        tip_x, tip_y = self._compute_tip_position(psi, theta)
        # Calculate reward
        if theta < self.target_min_angle:
            # Below target range: negative reward (scaled by distance from 5°)
            diff = self.target_min_angle - theta
            reward = - min(diff / self.target_min_angle, 1.0)
        elif theta > self.target_max_angle:
            # Above target range: negative reward (scaled by overshoot beyond 40°)
            diff = theta - self.target_max_angle
            reward = - min(diff / self.target_max_angle, 1.0)
        else:
            # Within target range: positive reward
            reward = 1.0
        
        # Increment step count and check termination
        self.step_count += 1
        if self.step_count >= MAX_EPISODE_STEPS:
            self.done = True
        
        # Update state
        self.state = [x, y, math.cos(psi), math.sin(psi), tip_x, tip_y, theta]
        return self.state, reward, self.done, {}
    

def compute_F(x, y, psi, theta):
    """
    Compute the difference F for the rod's physics (e.g., residual force or position error).
    F is defined so that F = 0 when θ equals the target angle derived from inputs.
    """
    # Compute target bending angle as above
    target_deg = 5.0 + 15.0 * ((x - 0.01) / 0.24) + 20.0 * ((y - 0.01) / 0.14)
    target_deg += 5.0 * math.cos(psi)
    target_deg = max(5.0, min(40.0, target_deg))
    target_rad = math.radians(target_deg)
    theta_rad = theta
    # F(θ) = (θ - θ_target) + 0.5 * (θ - θ_target)^3
    diff = theta_rad - target_rad
    F = diff + 0.5 * (diff ** 3)
    return F

# Define Actor (policy) and Critic (value) neural networks
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc_out = nn.Linear(64, action_dim)
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        logits = self.fc_out(x)  # outputs action logits
        return logits

class Critic(nn.Module):
    def __init__(self, state_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc_out = nn.Linear(64, 1)
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        value = self.fc_out(x)   # outputs state value
        return value

# Device configuration (use GPU if available for efficiency)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize environment, actor, critic, and optimizers
env = RodBendingEnv()
state_dim = 7  # [x, y, cos(ψ), sin(ψ), tip_x, tip_y, θ]
action_dim = 3
actor = Actor(state_dim, action_dim).to(device)
critic = Critic(state_dim).to(device)
actor_optimizer = optim.Adam(actor.parameters(), lr=ACTOR_LR)
critic_optimizer = optim.Adam(critic.parameters(), lr=CRITIC_LR)

# Training loop
for iteration in range(1, NUM_TRAIN_ITERATIONS + 1):
    # Storage for trajectories
    batch_states = []
    batch_actions = []
    batch_rewards = []
    batch_dones = []
    batch_log_probs = []
    batch_values = []
    steps_collected = 0
    
    # Collect a batch of trajectories
    while steps_collected < BATCH_SIZE:
        state = env.reset()
        done = False
        while not done and steps_collected < BATCH_SIZE:
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
            # Sample action from policy
            logits = actor(state_tensor)
            dist = torch.distributions.Categorical(logits=logits)
            action = dist.sample()
            action_idx = action.item()
            log_prob = dist.log_prob(action).item()
            # Get value estimate
            value = critic(state_tensor).item()
            # Step environment
            next_state, reward, done, _ = env.step(action_idx)
            # Store transition
            batch_states.append(state)
            batch_actions.append(action_idx)
            batch_rewards.append(reward)
            batch_dones.append(done)
            batch_log_probs.append(log_prob)
            batch_values.append(value)
            # Prepare for next step
            state = next_state
            steps_collected += 1
            if done:
                break
    
    # Compute returns and advantages (using GAE)
    batch_values.append(0.0)  # append value 0 for terminal state
    batch_advantages = []
    advantage = 0.0
    for t in reversed(range(len(batch_rewards))):
        # If episode ended at step t, bootstrap value = 0
        next_value = 0.0 if batch_dones[t] else batch_values[t+1]
        # Temporal difference
        delta = batch_rewards[t] + GAMMA * next_value - batch_values[t]
        # Update advantage estimate
        advantage = delta + GAMMA * LAMBDA_GAE * advantage * (0 if batch_dones[t] else 1)
        batch_advantages.insert(0, advantage)
    # Remove the appended terminal value
    batch_values = batch_values[:-1]
    # Convert to tensors
    batch_states = torch.tensor(batch_states, dtype=torch.float32).to(device)
    batch_actions = torch.tensor(batch_actions, dtype=torch.long).to(device)
    batch_old_log_probs = torch.tensor(batch_log_probs, dtype=torch.float32).to(device)
    batch_advantages = torch.tensor(batch_advantages, dtype=torch.float32).to(device)
    # Compute returns (advantages + baseline value)
    batch_values_tensor = torch.tensor(batch_values, dtype=torch.float32).to(device)
    batch_returns = batch_advantages + batch_values_tensor
    # Normalize advantages for numerical stability
    batch_advantages = (batch_advantages - batch_advantages.mean()) / (batch_advantages.std() + 1e-8)
    
    # PPO updates
    actor_losses = []
    critic_losses = []
    # Iterate over multiple epochs and minibatches
    for epoch in range(PPO_EPOCHS):
        indices = torch.randperm(batch_states.size(0)).to(device)
        for start in range(0, batch_states.size(0), MINIBATCH_SIZE):
            end = start + MINIBATCH_SIZE
            mb_idx = indices[start:end]
            mb_states = batch_states[mb_idx]
            mb_actions = batch_actions[mb_idx]
            mb_old_log_probs = batch_old_log_probs[mb_idx]
            mb_returns = batch_returns[mb_idx]
            mb_adv = batch_advantages[mb_idx]
            # Compute current policy output and value for minibatch
            logits = actor(mb_states)
            dist = torch.distributions.Categorical(logits=logits)
            new_log_probs = dist.log_prob(mb_actions)
            entropy = dist.entropy().mean()
            values = critic(mb_states).squeeze()
            # Policy loss (with clipping)
            ratio = torch.exp(new_log_probs - mb_old_log_probs)
            surr1 = ratio * mb_adv
            surr2 = torch.clamp(ratio, 1.0 - CLIP_EPS, 1.0 + CLIP_EPS) * mb_adv
            policy_loss = -torch.mean(torch.min(surr1, surr2))
            # Value loss (MSE)
            value_loss = torch.mean((mb_returns - values) ** 2)
            # Total loss
            loss = policy_loss - ENTROPY_COEF * entropy + VALUE_COEF * value_loss
            # Optimize model
            actor_optimizer.zero_grad()
            critic_optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(actor.parameters(), 0.5)
            nn.utils.clip_grad_norm_(critic.parameters(), 0.5)
            actor_optimizer.step()
            critic_optimizer.step()
            actor_losses.append(policy_loss.item())
            critic_losses.append(value_loss.item())
    
    # Logging
    if iteration % LOG_INTERVAL == 0:
        avg_actor_loss = sum(actor_losses) / len(actor_losses)
        avg_critic_loss = sum(critic_losses) / len(critic_losses)
        # Estimate recent episode reward (average of last few episodes in batch)
        recent_rewards = []
        ep_sum = 0.0
        count = 0
        for r, d in zip(batch_rewards, batch_dones):
            ep_sum += r
            if d:  # episode ended
                recent_rewards.append(ep_sum)
                ep_sum = 0.0
                count += 1
                if count >= 10:  # consider last 10 episodes
                    break
        avg_recent_reward = (sum(recent_rewards) / len(recent_rewards)) if recent_rewards else ep_sum
        print(f"Iteration {iteration}: AvgRecentEpisodeReward = {avg_recent_reward:.3f}, "
              f"ActorLoss = {avg_actor_loss:.3f}, CriticLoss = {avg_critic_loss:.3f}")
