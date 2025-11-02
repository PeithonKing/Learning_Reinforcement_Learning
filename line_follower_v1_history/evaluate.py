import gymnasium as gym
import torch, os
import numpy as np
from collections import deque

import line_follower_v1
from models import Actor

def evaluate_model(model, env_name, render_mode, sensor_grid, history_length, track, max_steps, hitbox, episodes, x_spacing=None, y_spacing=None, verbose=False):
    total_rewards = []
    for ep in range(episodes):
        env = gym.make(
            f'my_gym_envs/{env_name}',
            render_mode=render_mode,
            sensor_grid=sensor_grid,
            track=track,
            max_steps=max_steps,
            hitbox=hitbox,
            x_spacing=x_spacing,
            y_spacing=y_spacing,
            verbose=verbose
        )
        
        # env.metadata["render_fps"] = 24
        
        state, _ = env.reset()
        done = False
        truncated = False
        total_reward = 0
        
        history = deque(maxlen=history_length)
        for _ in range(history_length - 1):
            history.append(np.zeros(len(state)))
        history.append(state)

        while not done and not truncated:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(np.array(history).flatten())
                action = model(state_tensor).numpy()

            next_state, reward, done, truncated, _ = env.step(action)
            history.append(next_state)
            # state = next_state
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
    hitbox = checkpoint["hitbox"]
    hidden_dim = checkpoint.get("hidden_dim", 64)
    hidden_layers = checkpoint.get("hidden_layers", 2)
    x_spacing = checkpoint.get("x_spacing", None)
    y_spacing = checkpoint.get("y_spacing", None)
    history_length = checkpoint.get("history_length", 5)

    # Initialize the actor network
    actor_net = Actor(state_dim*history_length, action_dim, max_action, hidden_dim=hidden_dim, hidden_layers=hidden_layers)
    # Correctly load the actor's state dictionary from the checkpoint
    actor_net.load_state_dict(checkpoint["actor_state_dict"])
    actor_net.eval()

    # Run evaluation episodes
    avg_reward = evaluate_model(
        actor_net,
        ENV_NAME,
        "human",
        sensor_grid,
        history_length,
        track,
        max_steps,
        hitbox,
        episodes=10,
        x_spacing=x_spacing,
        y_spacing=y_spacing,
        verbose=True
    )
    print(f"Average reward over 10 evaluation episodes: {avg_reward:.2f}")
