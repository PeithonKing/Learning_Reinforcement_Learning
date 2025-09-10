import numpy as np
import random
from tqdm import trange
import matplotlib.pyplot as plt
import gymnasium as gym
import snake_ladder
from utils import evaluate, QTable
import matplotlib.ticker as ticker
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.animation import FuncAnimation, PillowWriter

# --- BLOCK_VARIABLES ---
EPISODES = int(5e4)
GAMMA = 0.99
LR = 0.01
TEMP_START = 1.0
TEMP_END = 0.01
MODEL_PATH = "q_table.npy"
PLOT_PATH = "reward_graph.png"
TEMP_DECAY = (TEMP_END / TEMP_START) ** (1 / (1.0 * EPISODES))
# TEMP_DECAY = 0.9999
SEED = 24
print(f"{TEMP_DECAY=}")

# --- SETUP ---
np.random.seed(SEED)
random.seed(SEED)

env = gym.make("my_gym_envs/snake_ladder_v0", max_steps=30)
state_space = env.observation_space.n
action_space = env.action_space.n

# q_table = QTable((state_space, action_space), init_type='zeros')
q_table = QTable((state_space, action_space), init_type='random')
rewards_all_episodes = []
temperatures = []
# evaluation tracking
test_rewards = []
test_episodes = []
temperature = TEMP_START
saved_models = []

# --- TRAINING LOOP ---
pbar = trange(EPISODES)
for episode in pbar:
    state, _ = env.reset()
    state -= 1
    done = False
    
    while not done:
        action = q_table.predict(state, temp=temperature)

        # Environment step
        new_state, reward, terminated, truncated, _ = env.step(action + 1)
        new_state -= 1
        done = terminated or truncated

        # Q-table update (Bellman equation)
        target = reward + GAMMA * np.max(q_table[new_state, :])
        loss = q_table[state, action] - target
        q_table[state, action] -= LR * loss
        state = new_state

    # Decay temperature and record reward
    temperature = max(TEMP_END, temperature * TEMP_DECAY)
    rewards_all_episodes.append(reward)
    temperatures.append(temperature)

    pbar.set_description(f"Episode {episode+1}/{EPISODES}, Reward: {np.mean(rewards_all_episodes[-1000:]):.2f}, Temp: {temperature:.4f}")

    if (episode+1) % 1000 == 0:
        # np.save(MODEL_PATH, q_table)
        saved_models.append((q_table.table.copy(), float(temperature), int(episode + 1)))

        # Run evaluation (no temperature / greedy)
        eval_reward = evaluate(q_table, n=1000)
        test_rewards.append(eval_reward)
        test_episodes.append(episode-500)

        # Plot Q-table
        fig, ax = plt.subplots(figsize=(20, 8))
        im = ax.imshow(q_table.table.T, cmap='hot')
        ax.set_xticks(np.arange(0, state_space, 2))
        ax.set_xticklabels(np.arange(1, state_space+1, 2))
        ax.set_yticks(np.arange(0, action_space, 1))
        ax.set_yticklabels(np.arange(1, action_space+1, 1))
        # Thinner horizontal colorbar below the image while preserving image width
        divider = make_axes_locatable(ax)
        # make the colorbar thicker (20% of axes height) and move it further down (pad=0.25)
        cax = divider.append_axes("bottom", size="20%", pad=0.3)
        cbar = fig.colorbar(im, cax=cax, orientation='horizontal')
        cbar.set_label('Q-value')
        plt.savefig(f"q_table.png", bbox_inches='tight')
        plt.close()

        # Create and save the plot
        window = 2000
        if len(rewards_all_episodes) > window:
            rewards_smoothed = np.convolve(rewards_all_episodes, np.ones(window)/window, mode='valid')
            fig, ax1 = plt.subplots(figsize=(10, 6))

            # Plot smoothed rewards
            ax1.plot(5 - np.log10(rewards_smoothed), color='tab:blue', label='Smoothed Train Reward')
            ax1.set_xlabel("Episode")
            ax1.set_ylabel("Steps to complete (smooth)", color='tab:blue')
            ax1.tick_params(axis='y', labelcolor='tab:blue')

            # Apply grid only to ax1 (steps axis)
            ax1.yaxis.set_major_locator(ticker.MultipleLocator(1))
            ax1.grid(True, axis='y')

            # Plot evaluation points (unsmoothed)
            ax1.plot(test_episodes, 5 - np.log10(test_rewards), color='tab:orange', marker=".", linestyle='--', label='Eval Reward')

            # Plot temperature on twin axis
            ax2 = ax1.twinx()
            ax2.plot(temperatures[window-1:], color='tab:red', label='Temperature')
            ax2.set_ylabel("Temperature", color='tab:red')
            ax2.tick_params(axis='y', labelcolor='tab:red')
            ax2.yaxis.set_major_locator(ticker.MultipleLocator(0.1))  # no grid since ax1 owns the grid

            # X axis ticks
            ax1.xaxis.set_major_locator(ticker.MultipleLocator(10000))

            # Combine legends from both axes
            handles1, labels1 = ax1.get_legend_handles_labels()
            handles2, labels2 = ax2.get_legend_handles_labels()
            # ax1.legend(handles1 + handles2, labels1 + labels2, loc='upper right')
            ax1.legend(handles1 + handles2, labels1 + labels2)

            plt.title(f"Completion Steps, Eval & Temperature (t={temperature:.2f})")
            plt.savefig(PLOT_PATH, bbox_inches='tight')
            plt.close()


# --- CREATE ANIMATED GIF OF SAVED Q-TABLES ---
if saved_models:
    # Compute consistent color scale across all checkpoints (extract arrays)
    models = [m[0] for m in saved_models]
    vmin = min([m.min() for m in models])
    vmax = max([m.max() for m in models])

    # Create an animation using FuncAnimation so frames are rendered by matplotlib
    fig, ax = plt.subplots(figsize=(20, 3))
    init_model = models[0]
    im = ax.imshow(init_model.T, cmap='hot', vmin=vmin, vmax=vmax, animated=True)

    # Match ticks/labels to the static q_table plot
    ax.set_xticks(np.arange(0, state_space, 2))
    ax.set_xticklabels(np.arange(1, state_space + 1, 2))
    ax.set_yticks(np.arange(0, action_space, 1))
    ax.set_yticklabels(np.arange(1, action_space + 1, 1))

    # Add a horizontal colorbar below the image while preserving width
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("bottom", size="20%", pad=0.35)
    cbar = fig.colorbar(im, cax=cax, orientation='horizontal')
    cbar.set_label('Q-value')

    # Tighten the figure layout so there is minimal whitespace around the axes and colorbar.
    # We'll set conservative margins that work with the appended colorbar.
    fig.subplots_adjust(left=0.02, right=0.98, top=0.95, bottom=0.12)

    title = ax.set_title("Q-table checkpoint 0")

    def update(frame_idx):
        model, temp, ep = saved_models[frame_idx]
        im.set_array(model.T)
        title.set_text(f"Q-table Episode {ep} — Temperature {temp:.4f}")
        return [im, title]

    anim = FuncAnimation(fig, update, frames=len(saved_models), interval=100, blit=True)

    gif_path = "q_table_evolution.gif"
    writer = PillowWriter(fps=1000/100)
    anim.save(gif_path, writer=writer, dpi=300)
    print(f"Saved {gif_path} with {len(saved_models)} frames")

    plt.close(fig)

