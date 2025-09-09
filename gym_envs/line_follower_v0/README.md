# Line Follower v0 — Discrete Sensors, Discrete Actions

This custom Gymnasium environment simulates a small car with an underside sensor grid driving on a 2D track image. The task is sense the track with binary sensors and steer to collect coins along the path.

## Game

- The world is a 2D track loaded from an image (black and white, the image should represent the waypoints in some way), plus an ordered list of waypoints (“coins”).
- The agent controls a differential drive car (two wheels) with a fixed-speed mapping per action.
- Objective: collect as many coins as possible within the allowed time steps (`max_steps`). Coins must be collected in order.
- A coin is collected when the car’s position falls inside a circular hitbox of a given radius around the next coin.

### Action Space

- Discrete(3)
  - 0 → slow down left wheel (turn right)
  - 1 → both wheels normal speed (go straight)
  - 2 → slow down right wheel (turn left)

### Observation Space

- MultiBinary of size `sensor_grid[0] * sensor_grid[1]` (flattened grid).
- Each bit corresponds to one downward-facing sensor under the car body:
  - 1 if the pixel beneath that sensor is black
  - 0 if it is white
- Default `sensor_grid`: `(4, 6)` = 24 bits.

### Reward

- Reward equals the number of coins captured in the current step (0 or more), based on a circular hitbox around the next coin in sequence.
- No explicit per-step penalty; the episode is limited by `max_steps`.
- Episode ends by truncation when the step budget is exhausted (terminated is always False coz the car is never killed).

## Usage

```python
import gymnasium as gym
import line_follower_v0

env = gym.make(
    "my_gym_envs/line_follower_v0",
    render_mode=None,   # or "human" / "rgb_array"
    sensor_grid=(4, 6),
    track="oval",
    max_steps=200,
    hitbox=20,
)
```

You can also register external track folders at runtime (optional):

```python
from line_follower_v0.envs.main import LineFollowerEnv
LineFollowerEnv.add_track_folder("path/to/my_tracks")
```

## Files

- `envs/main.py`: The Gymnasium environment implementation (`LineFollowerEnv`).
- `envs/car.py`: Car kinematics, sensor layout, coin logic (hitbox + reward), and rendering helpers.
- `tracks/`: Built-in track PNGs and waypoint `.npy` files.
