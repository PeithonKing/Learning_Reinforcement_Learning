import gymnasium as gym
import torch
import torch.nn as nn
# import numpy as np
import os
import line_follower_v1

# --- Actor Network Definition ---
# Redefined here to keep the script self-contained.
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        # Using a slightly larger network to match potential complexity
        self.net = nn.Sequential(
            nn.Linear(state_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, action_dim),
            nn.Tanh()
        )
        self.max_action = max_action

    def forward(self, state):
        return self.net(state) * self.max_action

def evaluate_model(model, env_name, render_mode, sensor_grid, track, max_steps, hitbox, episodes, verbose=False):
    total_rewards = []
    for ep in range(episodes):
        env = gym.make(
            f'my_gym_envs/{env_name}',
            render_mode=render_mode,
            sensor_grid=sensor_grid,
            track=track,
            max_steps=max_steps,
            hitbox=hitbox,
        )
        
        env.metadata["render_fps"] = 24
        
        state, _ = env.reset()
        done = False
        truncated = False
        total_reward = 0

        while not done and not truncated:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state)
                action = model(state_tensor).numpy()

            next_state, reward, done, truncated, _ = env.step(action)
            state = next_state
            total_reward += reward
            
        total_rewards.append(total_reward)
        if verbose: print(f"Episode {ep+1}: Total Reward = {total_reward}")

    env.close()
    return sum(total_rewards) / len(total_rewards)

if __name__ == "__main__":
    ENV_NAME = "line_follower_v1"
    MODEL_PATH = "ddpg_linefollower_v1.pth" # Corrected path to match training script

    assert os.path.exists(MODEL_PATH), f"Model file not found: {MODEL_PATH}"

    # Load the checkpoint dictionary
    checkpoint = torch.load(MODEL_PATH, map_location=torch.device('cpu'))
    
    sensor_grid = checkpoint["sensor_grid"]
    max_steps = checkpoint["max_steps"]
    state_dim = checkpoint["state_dim"]
    action_dim = checkpoint["action_dim"]
    max_action = checkpoint["max_action"]
    track = checkpoint["track"]
    track = "square"
    hitbox = checkpoint["hitbox"]
    hitbox = 30

    # Initialize the actor network
    actor_net = Actor(state_dim, action_dim, max_action)
    # Correctly load the actor's state dictionary from the checkpoint
    actor_net.load_state_dict(checkpoint["actor_state_dict"])
    actor_net.eval()

    # Run evaluation episodes
    avg_reward = evaluate_model(
        actor_net,
        ENV_NAME,
        "human",
        sensor_grid,
        track,
        max_steps,
        hitbox,
        episodes=10,
        verbose=True
    )
