import numpy as np
import gymnasium as gym

class QTable:
    def __init__(self, shape, init_type='random'):
        if init_type == 'zeros':
            self.table = np.zeros(shape)
        elif init_type == 'random':
            self.table = np.random.normal(loc=1.0, scale=0.1, size=shape)
        elif isinstance(init_type, int) or isinstance(init_type, float):
            self.table = np.full(shape, init_type)
        else:
            raise ValueError("Invalid init_type. Use 'zeros' or 'random'.")

    def __getitem__(self, key):
        return self.table[key]

    def __setitem__(self, key, value):
        self.table[key] = value
    
    def predict(self, state, temp=1):
        q_values = self.table[state, :]
        if temp==0: return np.argmax(q_values)
        scaled_q = q_values / temp
        probabilities = np.exp(scaled_q - np.max(scaled_q)) / np.sum(np.exp(scaled_q - np.max(scaled_q)))
        return np.random.choice(range(len(q_values)), p=probabilities)

def evaluate(q_table, n=100):
    env = gym.make("my_gym_envs/snake_ladder_v0", max_steps=10)
    rewards = []
    for _ in range(n):
        state, _ = env.reset()
        state -= 1
        done = False
        total_reward = 0
        while not done:
            action = q_table.predict(state, temp=0.1)  # Greedy evaluation
            new_state, reward, terminated, truncated, _ = env.step(action + 1)
            new_state -= 1
            done = terminated or truncated
            total_reward += reward
            state = new_state
        rewards.append(total_reward)
    env.close()
    return np.mean(rewards)
