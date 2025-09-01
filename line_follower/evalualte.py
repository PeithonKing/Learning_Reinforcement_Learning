import gymnasium as gym
import torch
import torch.nn as nn

class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 32),
            nn.ReLU(),
            nn.Linear(32, action_dim)
        )

    def forward(self, x):
        return self.net(x)

def evaluate_model(model, env_name, render_mode, sensor_grid, track, max_steps, hitbox, episodes, verbose=False):
    env = gym.make(
        f'gymnasium_env/{env_name}', render_mode=render_mode,
        sensor_grid=sensor_grid,
        track=track,
        max_steps=max_steps,
        hitbox=hitbox,
    )
    
    total_rewards = []
    for ep in range(episodes):
        state, _ = env.reset()
        done = False
        total_reward = 0

        while not done:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                q_values = model(state_tensor)
                action = q_values.argmax().item()

            next_state, reward, terminated, truncated, _ = env.step(action)
            state = next_state
            total_reward += reward
            done = terminated or truncated

        total_rewards.append(total_reward)
        if verbose: print(f"Episode {ep+1}: Total Reward = {total_reward}")

    env.close()
    return sum(total_rewards) / len(total_rewards)

if __name__ == "__main__":
    ENV_NAME = "line_follower_v0"
    MODEL_PATH = "dqn_linefollower.pth"
    
    import os
    assert os.path.exists(MODEL_PATH), f"Model file not found: {MODEL_PATH}"

    loaded_model = torch.load(MODEL_PATH, map_location=torch.device('cpu'), weights_only=False)
    sensor_grid = loaded_model["sensor_grid"]
    track = loaded_model["track"]
    max_steps = loaded_model["max_steps"]
    hitbox = loaded_model["hitbox"]

    policy_net = DQN(sensor_grid[0]*sensor_grid[1], loaded_model["action_dim"])
    policy_net.load_state_dict(loaded_model["state_dict"])
    policy_net.eval()

    for run in range(10):
        avg_reward = evaluate_model(
            policy_net,
            ENV_NAME,
            "human",
            sensor_grid,
            track,
            max_steps,
            hitbox,
            episodes=1,
            verbose=True
        )
