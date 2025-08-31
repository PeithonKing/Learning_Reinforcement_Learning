import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import matplotlib.pyplot as plt
from tqdm import tqdm


ENV_NAME = "MountainCar-v0"
EPISODES = 5000
GAMMA = 0.99
LR = 1e-3
BATCH_SIZE = 64
MEMORY_SIZE = 50000
EPS_START = 1.0
EPS_END = 0.05
EPS_DECAY = 0.995
TARGET_UPDATE = 10   # update target network every N episodes
MODEL_PATH = "dqn_mountaincar.pth"
PLOT_PATH = "mountaincar_rewards.png"
continue_training = True

# Use GPU
device = torch.device("cpu")

# Q-Network
class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )
    def forward(self, x):
        return self.net(x)

# Experience Replay Buffer
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.array(next_states),
            np.array(dones, dtype=np.float32)
        )
    def __len__(self):
        return len(self.buffer)

# Environment setup
env = gym.make(ENV_NAME)
state_dim = env.observation_space.shape[0]   # (position, velocity)
action_dim = env.action_space.n              # 3 actions

policy_net = DQN(state_dim, action_dim).to(device)
target_net = DQN(state_dim, action_dim).to(device)

if continue_training:
    state_dict_to_copy = torch.load(MODEL_PATH, map_location=torch.device('cpu'))
else:
    state_dict_to_copy = policy_net.state_dict()

policy_net.load_state_dict(state_dict_to_copy)
target_net.load_state_dict(state_dict_to_copy)
target_net.eval()

optimizer = optim.Adam(policy_net.parameters(), lr=LR)
memory = ReplayBuffer(MEMORY_SIZE)

epsilon = EPS_START
rewards_per_episode = []

from tqdm import tqdm

# Training Loop
progress_bar = tqdm(range(EPISODES), desc="Avg Reward (last 100): -200.00, Epsilon: 1.00")
for episode in progress_bar:
    state, _ = env.reset()
    total_reward = 0

    done = False
    while not done:
        # Epsilon-greedy action
        if random.random() < epsilon:
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                q_values = policy_net(state_tensor)
                action = q_values.argmax().item()

        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        memory.push(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward

        # Train step if enough samples
        if len(memory) >= BATCH_SIZE:
            states, actions, rewards, next_states, dones = memory.sample(BATCH_SIZE)

            states = torch.FloatTensor(states).to(device)
            actions = torch.LongTensor(actions).unsqueeze(1).to(device)
            rewards = torch.FloatTensor(rewards).unsqueeze(1).to(device)
            next_states = torch.FloatTensor(next_states).to(device)
            dones = torch.FloatTensor(dones).unsqueeze(1).to(device)

            q_values = policy_net(states).gather(1, actions)
            next_q_values = target_net(next_states).max(1, keepdim=True)[0]
            expected_q_values = rewards + GAMMA * next_q_values * (1 - dones)

            loss = nn.MSELoss()(q_values, expected_q_values.detach())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # Decay epsilon
    epsilon = max(EPS_END, epsilon * EPS_DECAY)

    rewards_per_episode.append(total_reward)

    # Update target network
    if episode % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())

    # Update tqdm description every 100 episodes
    if (episode+1) % 100 == 0:
        avg_reward = np.mean(rewards_per_episode[-100:])
        progress_bar.set_description(f"Avg Reward (last 100): {avg_reward:.2f}, Epsilon: {epsilon:.2f}")

env.close()

# Save model
torch.save(policy_net.state_dict(), MODEL_PATH)

# Plot rewards
mean_rewards = np.zeros(EPISODES)
for t in range(EPISODES):
    mean_rewards[t] = np.mean(rewards_per_episode[max(0, t-100):(t+1)])
plt.plot(mean_rewards)
plt.xlabel("Episodes")
plt.ylabel("Average Reward (100 eps)")
plt.title("DQN on MountainCar-v0")
plt.savefig(PLOT_PATH)
print("Training finished, model saved.")
