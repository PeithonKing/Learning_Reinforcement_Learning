import gymnasium as gym
from gymnasium import spaces
import pygame, os, random
import numpy as np
from matplotlib import image
from importlib import resources

WIDTH, HEIGHT = 600, 600

BLUE   = (  0,   0, 255)  # #0000FF
GREEN  = (  0, 255,   0)  # #00FF00
RED    = (255,   0,   0)  # #FF0000
WHITE  = (255, 255, 255)  # #FFFFFF
BLACK  = (  0,   0,   0)  # #000000
YELLOW = (255, 255,   0)  # #FFFF00

NPCs = {
    # Snakes
    21: 3,
    31: 8,
    47: 30,
    52: 23,
    76: 41,
    81: 62,
    88: 67,
    98: 12,

    # Ladders
    4: 75,
    5: 15,
    19: 41,
    28: 50,
    35: 96,
    44: 82,
    58: 94,
    59: 95,
    70: 91,
}

class SnakeLadderEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(
        self, render_mode=None,
        npcs = NPCs,
        max_steps = 15,
    ):
        self.npcs = npcs
        self.max_steps = max_steps
        self.observation_space = spaces.Discrete(100)
        self.action_space = spaces.Discrete(6)

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        
        self.state = None
        self.turns = None

        self.window = None
        self.clock = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.state = 1
        self.turns = 0

        if self.render_mode == "human":
            self._render_frame()

        return self.state, {}

    def get_reward(self):
        x = self.turns
        return 10**(5-x)

    def step(self, action):
        assert action in range(1, 7), f"Invalid action (1 <= action <= 6) but {action = }"

        self.turns += 1
        
        # goes to next position; doesn't move if exceeds 100
        went_to = self.state + action if self.state + action <= 100 else self.state
        
        # check snake or ladder
        if self.npcs.get(went_to):
            went_to = self.npcs[went_to]
        
        self.state = went_to
        terminated = self.state == 100
        truncated = self.turns >= self.max_steps

        if self.render_mode == "human":
            self._render_frame()

        return (
            self.state,
            self.get_reward() if terminated or truncated else 0,
            terminated,
            truncated,
            {}
        )

    def _render_frame(self):  # never gets called if render_mode is None
        raise NotImplementedError()

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
