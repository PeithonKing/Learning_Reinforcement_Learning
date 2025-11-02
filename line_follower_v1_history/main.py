import gymnasium as gym
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import random, os
from collections import deque
import matplotlib.pyplot as plt
from tqdm import tqdm

from evaluate import evaluate_model
from models import Actor, Critic
import line_follower_v1

# --- Hyperparameters ---
ENV_NAME = "line_follower_v1"
EPISODES = 1000
GAMMA = 0.99
LR_ACTOR = 1e-4
LR_CRITIC = 1e-3
BATCH_SIZE = 64
MEMORY_SIZE = 5000
TAU = 1e-3
NOISE_STDDEV = 0.1
TARGET_UPDATE = 10
MODEL_PATH = "ddpg_linefollower_v1.pth"
SEED = 23

# set the seed for both torch and numpy and everything
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

continue_training = True

# --- Environment & Model Parameters ---
sensor_grid = (4, 3)
# track = "oval"
# track = "hexagon"
track = "rounded_square_orig"
# track = "square_orig"
# track = "rounded_square"
# track = "square"
max_steps = 200
hitbox = 40
history_length = 5

# Allow experimenting with different action/state encoder sizes
hidden_dim = 32
hidden_layers = 1

# Optionally allow spacing parameters for parity with v0 (env can ignore if unsupported)
x_spacing = 40
y_spacing = 20

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --- Experience Replay ---
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    def push(self, states, action, reward, next_state, done):
        self.buffer.append((states, action, reward, next_state, done))
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
    def __len__(self):
        return len(self.buffer)

# --- Environment and Model Setup ---
env = gym.make(
    f'my_gym_envs/{ENV_NAME}',
    render_mode=None,
    sensor_grid=sensor_grid,
    track=track,
    max_steps=max_steps,
    hitbox=hitbox,
    x_spacing=x_spacing,
    y_spacing=y_spacing,
)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
max_action = float(env.action_space.high[0])

actor = Actor(state_dim*history_length, action_dim, max_action, hidden_dim, hidden_layers).to(device)
actor_target = Actor(state_dim*history_length, action_dim, max_action, hidden_dim, hidden_layers).to(device)
actor_optimizer = optim.Adam(actor.parameters(), lr=LR_ACTOR)

critic = Critic(state_dim*history_length, action_dim, hidden_dim, hidden_layers).to(device)
critic_target = Critic(state_dim*history_length, action_dim, hidden_dim, hidden_layers).to(device)
critic_optimizer = optim.Adam(critic.parameters(), lr=LR_CRITIC)

start_episode = 0
rewards_per_episode = []
test_rewards = []
test_episodes = []
max_test_reward = 0

# --- Load Checkpoint to Continue Training ---
if continue_training and os.path.exists(MODEL_PATH):
    print("Continuing training from saved model.")
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    actor.load_state_dict(checkpoint["actor_state_dict"])
    actor_target.load_state_dict(checkpoint["actor_state_dict"])
    critic.load_state_dict(checkpoint["critic_state_dict"])
    critic_target.load_state_dict(checkpoint["critic_state_dict"])
    actor_optimizer.load_state_dict(checkpoint["actor_optimizer_state_dict"])
    critic_optimizer.load_state_dict(checkpoint["critic_optimizer_state_dict"])
    
    start_episode = checkpoint["episode"] + 1
    rewards_per_episode = checkpoint["rewards_per_episode"]
    test_rewards = checkpoint["test_rewards"]
    test_episodes = checkpoint["test_episodes"]
    
    assert checkpoint["history_length"] == history_length, "History length mismatch"
    assert checkpoint["sensor_grid"] == sensor_grid, "Sensor grid mismatch"
    assert checkpoint["hidden_dim"] == hidden_dim, "Hidden dim mismatch"
    assert checkpoint["hidden_layers"] == hidden_layers, "Hidden layers mismatch"

memory = ReplayBuffer(MEMORY_SIZE)


# --- Training Loop ---
progress_bar = tqdm(total=EPISODES, initial=start_episode, dynamic_ncols=True)
for episode in range(start_episode, EPISODES):
    state, _ = env.reset()
    total_reward = 0
    done = truncated = False

    history = deque(maxlen=history_length)
    for _ in range(history_length - 1):
        history.append(np.zeros(state_dim))
    history.append(state)
    while not done and not truncated:
        with torch.no_grad():
            mem_state = np.array(history).flatten()
            action = actor(torch.FloatTensor(mem_state).to(device)).cpu().numpy()
            noise = np.random.normal(0, max_action * NOISE_STDDEV, size=action_dim)
            action = (action + noise).clip(env.action_space.low, env.action_space.high)

        next_state, reward, done, truncated, _ = env.step(action)
        history.append(next_state)
        next_mem_state = np.array(history).flatten()
        memory.push(mem_state, action, reward, next_mem_state, done or truncated)
        # state = next_state
        total_reward += reward

        if len(memory) > BATCH_SIZE:
            batch = memory.sample(BATCH_SIZE)
            states, actions, rewards, next_states, dones = zip(*batch)
            states = torch.FloatTensor(np.array(states)).to(device)
            actions = torch.FloatTensor(np.array(actions)).to(device)
            rewards = torch.FloatTensor(rewards).unsqueeze(1).to(device)
            next_states = torch.FloatTensor(np.array(next_states)).to(device)
            dones = torch.FloatTensor(dones).unsqueeze(1).to(device)

            # Critic Update
            next_actions = actor_target(next_states)
            target_q = critic_target(next_states, next_actions)
            expected_q = rewards + (1 - dones) * GAMMA * target_q
            current_q = critic(states, actions)
            critic_loss = nn.MSELoss()(current_q, expected_q.detach())
            critic_optimizer.zero_grad()
            critic_loss.backward()
            critic_optimizer.step()

            # Actor Update
            actor_loss = -critic(states, actor(states)).mean()
            actor_optimizer.zero_grad()
            actor_loss.backward()
            actor_optimizer.step()

            # Soft Target Updates
            for param, target_param in zip(critic.parameters(), critic_target.parameters()):
                target_param.data.copy_(TAU * param.data + (1.0 - TAU) * target_param.data)
            for param, target_param in zip(actor.parameters(), actor_target.parameters()):
                target_param.data.copy_(TAU * param.data + (1.0 - TAU) * target_param.data)

    rewards_per_episode.append(total_reward)
    progress_bar.update(1)

    # --- Periodic Checkpointing, Evaluation, and Plotting ---
    if (episode + 1) % TARGET_UPDATE == 0:
        avg_reward = np.mean(rewards_per_episode[-TARGET_UPDATE:])
        progress_bar.set_description(f"Avg Reward: {avg_reward:.2f}")

        # Evaluate current policy
        test_reward = evaluate_model(
            actor,
            ENV_NAME,
            None,
            sensor_grid,
            history_length,
            track,
            max_steps,
            hitbox,
            episodes=10,
            x_spacing=x_spacing,
            y_spacing=y_spacing,
        )
        test_rewards.append(test_reward)
        test_episodes.append(episode)

        # Smoothed training rewards (window=100 padded) like v0 script
        window = 10
        if len(rewards_per_episode) >= window:
            left_pad = window // 2
            right_pad = window - 1 - left_pad
            padded = np.concatenate([
                np.full(left_pad, rewards_per_episode[0]),
                rewards_per_episode,
                np.full(right_pad, rewards_per_episode[-1])
            ])
            smoothed = np.convolve(padded, np.ones(window)/window, mode='valid')
        else:
            smoothed = rewards_per_episode

        # Plot
        x_range = range(len(rewards_per_episode))
        plt.plot(x_range, rewards_per_episode, color='tab:blue', alpha=0.3)
        plt.plot(x_range, smoothed, color='tab:blue', label=f"Train Reward (smoothed {window})")
        plt.plot(test_episodes, test_rewards, 'tab:orange', label="Test Reward")
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.legend()
        plt.grid()
        plt.title(f"({ENV_NAME}) Episode {episode+1}")
        plt.savefig("rewards_plot_v1.png")
        plt.close()

        
        if test_reward > max_test_reward:
            max_test_reward = test_reward
            torch.save({
                "episode": episode,
                "actor_state_dict": actor.state_dict(),
                "critic_state_dict": critic.state_dict(),
                "actor_optimizer_state_dict": actor_optimizer.state_dict(),
                "critic_optimizer_state_dict": critic_optimizer.state_dict(),
                "rewards_per_episode": rewards_per_episode,
                "test_rewards": test_rewards,
                "test_episodes": test_episodes,
                "sensor_grid": sensor_grid,
                "hitbox": hitbox,
                "track": track,
                "max_steps": max_steps,
                "state_dim": state_dim,
                "action_dim": action_dim,
                "max_action": max_action,
                "hidden_dim": hidden_dim,
                "hidden_layers": hidden_layers,
                "x_spacing": x_spacing,
                "y_spacing": y_spacing,
                "history_length": history_length,
            }, MODEL_PATH)
            print(f"Saved new best model with test reward {test_reward:.2f} at episode {episode+1}")

env.close()
progress_bar.close()
