# Line Follower v1 — Discrete Sensors, Continuous Actions

This custom Gymnasium environment is a variant of [Line Follower v0](../line_follower_v0/README.md), adapted for continuous actions. It simulates a small car with an underside sensor grid driving on a 2D track image. The task is to sense the track with binary sensors and steer to collect coins along the path, but with direct control over wheel speeds for algorithms like DDPG.

## Game

- Same as [v0](../line_follower_v0/README.md#game): 2D track (black and white), ordered waypoints (“coins”), collect as many as possible within `max_steps` via circular hitbox.
- Agent controls a differential drive car with continuous wheel speeds.
- Objective: collect as many coins as possible within the allowed time steps (`max_steps`). Coins must be collected in order.

### Action Space

- Box(low=-5.0, high=5.0, shape=(2,), dtype=np.float32)
  - [0]: speed of left wheel (range: -5 to 5)
  - [1]: speed of right wheel (range: -5 to 5)

### Observation Space

- Same as [v0](../line_follower_v0/README.md#observation-space): MultiBinary of size `sensor_grid[0] * sensor_grid[1]` (flattened grid).
- Each bit: 1 if the pixel beneath the sensor is black, 0 if white.
- Default `sensor_grid`: `(4, 6)` = 24 bits.

### Reward

- Same as [v0](../line_follower_v0/README.md#reward): Reward equals the number of coins captured in the current step (0 or more), based on a circular hitbox around the next coin.
- No explicit per-step penalty; episode limited by `max_steps`.
- Episode ends by truncation when the step budget is exhausted (terminated is always False).

## Usage

```python
import gymnasium as gym
import line_follower_v1

env = gym.make(
    "my_gym_envs/line_follower_v1",
    render_mode=None,   # or "human" / "rgb_array"
    sensor_grid=(4, 6),
    track="oval",
    max_steps=200,
    hitbox=20,
)
```

You can also register external track folders at runtime (optional):

```python
from line_follower_v1.envs.main import LineFollowerEnv
LineFollowerEnv.add_track_folder("path/to/my_tracks")
```
