# import gymnasium as gym
from gymnasium import spaces
# import pygame, os
import numpy as np
# from matplotlib import image
# from importlib import resources

# from line_follower_v0.envs.car import Car, Coins, to_pygame
from line_follower_v0.envs import LineFollowerEnv as LineFollower_v0


class LineFollowerEnv(LineFollower_v0):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.action_space = spaces.Box(low=-5.0, high=5.0, shape=(2,), dtype=np.float32)

    def step(self, action):
        dt = 0.05  # time step
        left_speed, right_speed = np.clip(
            action,
            self.action_space.low,
            self.action_space.high
        )
        self.car.move(left_speed, right_speed, dt)
        observation =  self._get_obs()
        reward =  self.car_coins.get_reward()

        self.curr_step += 1

        if self.render_mode == "human":
            self._render_frame(observation)

        return (
            observation,
            reward,
            False,
            self.curr_step > self.max_steps,
            {}
        )
