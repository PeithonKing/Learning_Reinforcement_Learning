# Snake & Ladder v0 — Magical Dice Environment

This custom Gymnasium environment implements the classic Snakes & Ladders board (cells 1–100) with a **magical dice**: the agent chooses the exact outcome (1–6) each turn. It is the packaged version of the earlier experiment in [`../../snake_ladder`](../../snake_ladder) and is used by the revisit training to study shortest‑path behavior under exponential reward shaping.

<p align="center">
	<img src="../../snake_ladder/layout.png" alt="Board layout" width="360" />
</p>

Env ID: `my_gym_envs/snake_ladder_v0`

## Game

- Board: $10\times10$ grid of cells 1 to 100 (row/column visualization not required for learning; state is just the cell number).
- Snakes & Ladders mapping: supplied internally via an `npcs` dictionary (keys: start cell, values: destination cell). A move landing on a key teleports the agent to the mapped cell.
- Magical dice: on every turn the agent selects an intended die outcome in {1, …, 6}.
- Movement rule: if `state + action > 100`, the token stays in place (no overshoot bounce‑back).
- **Objective: reach cell 100 in as few turns as possible.**
- Episode end conditions:
	- `terminated` when state == 100.
	- `truncated` when turns ≥ `max_steps` (default 15).

We know for this standard snake-ladder configuration, the shortest path is **5 moves**.

## Action Space

- `spaces.Discrete(6)` — valid actions are the integers 1–6 (selected die face). Passing a value outside 1 to 6 raises an assertion error.
- You may add 1 to the 0 based action before input to the environment to make it 1 based.

## Observation Space

- `spaces.Discrete(100)` — the current cell number (1..100).  
- When integrating with Q‑tables you may subtract 1 to map cells to 0..99 indices (the revisit training does `state -= 1`).

## Reward

Exponential terminal reward only (sparse): the environment issues reward **only when the episode terminates or truncates**: $\text{reward} = 10^{(5 - \text{turns})}$

This yields heavy exponential preference for fewer turns. Examples:

| Turns | Reward |
| ----- | ------ |
| 5     | 1.0    |
| 6     | 0.1    |
| 7     | 0.01   |
| 8     | 0.001  |
| ...     | ...  |

Implications:

- The agent receives no intermediate shaping signal—credit assignment is entirely deferred to the end of each episode.
- Large gaps (×10 per saved step) make near‑optimal policies unstable unless exploration anneals.
- Truncated (non‑goal) episodes also receive a (tiny) reward determined by total turns, which still differentiates shorter failed attempts.

## Usage

```python
import gymnasium as gym
import snake_ladder  # registers env id

env = gym.make(
    "my_gym_envs/snake_ladder_v0",
    max_steps=15
)
```

Parameters:

- `max_steps` (int): episode truncation cap (default 15).
- `npcs` (dict): custom snakes/ladders mapping (override default).
