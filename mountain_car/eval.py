# BLOCK_VARIABLES
ENV_NAME = "MountainCar-v0"
MODEL_PATH = "dqn_mountaincar.pth"
EPISODES = 10

# ---------------------------------------------------
import gymnasium as gym
import torch
import torch.nn as nn

# Define the same network architecture
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

# Load environment and model
env = gym.make(ENV_NAME, render_mode="human")
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

policy_net = DQN(state_dim, action_dim)
policy_net.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
policy_net.eval()

# Play episodes
for ep in range(EPISODES):
    state, _ = env.reset()
    done = False
    total_reward = 0

    while not done:
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            q_values = policy_net(state_tensor)
            action = q_values.argmax().item()

        next_state, reward, terminated, truncated, _ = env.step(action)
        state = next_state
        total_reward += reward
        done = terminated or truncated

    print(f"Episode {ep+1}: Total Reward = {total_reward}")

env.close()
