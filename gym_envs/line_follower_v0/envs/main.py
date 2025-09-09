import gymnasium as gym
from gymnasium import spaces
import pygame, os, random
import numpy as np
from matplotlib import image
from importlib import resources

from .car import Car, Coins, to_pygame

WIDTH, HEIGHT = 800, 500

BLUE   = (  0,   0, 255)  # #0000FF
GREEN  = (  0, 255,   0)  # #00FF00
RED    = (255,   0,   0)  # #FF0000
WHITE  = (255, 255, 255)  # #FFFFFF
BLACK  = (  0,   0,   0)  # #000000
YELLOW = (255, 255,   0)  # #FFFF00

def rgb2gray(rgb):
    return np.dot(rgb[..., :4], [0.25, 0.25, 0.25, 0.25])


action_to_inputs = (
    (0.1, 1.0),  # slow down left wheel
    (1.0, 1.0),  # both wheels normal speed
    (1.0, 0.1),  # slow down right wheel
)


class LineFollowerEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}
    USER_TRACK_PATHS = []

    @classmethod
    def add_track_folder(cls, folder):
        """Add a folder to search for user tracks (prepend for priority)."""
        cls.USER_TRACK_PATHS.insert(0, folder)

    def __init__(
        self, render_mode=None,
        sensor_grid = (4, 6),
        track="path",  # options = ["path", "path2"]
        max_steps=200,
        hitbox=20,
    ):
        self.sensor_grid = sensor_grid
        self.track = track
        self.max_steps = max_steps
        self.hitbox = hitbox

        self.observation_space = spaces.MultiBinary(
            (sensor_grid[0] * sensor_grid[1],)
        )

        self.action_space = spaces.Discrete(len(action_to_inputs))

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        self.window = None
        self.clock = None
        self.curr_step = None
        
    def load_track(self, track: str):
        png_path = npy_path = None

        # First look in user folders
        for folder in self.USER_TRACK_PATHS:
            candidate_png = os.path.join(folder, f"{track}.png")
            candidate_npy = os.path.join(folder, f"{track}_waypoints.npy")
            if os.path.exists(candidate_png) and os.path.exists(candidate_npy):
                png_path = candidate_png
                npy_path = candidate_npy
                break

        # Then look in package tracks
        if png_path is None or npy_path is None:
            try:
                with resources.path("line_follower_v0.tracks", f"{track}.png") as p:
                    png_path = p
                with resources.path("line_follower_v0.tracks", f"{track}_waypoints.npy") as p:
                    npy_path = p
            except FileNotFoundError:
                raise FileNotFoundError(
                    f"Track '{track}' not found in user folders or default package tracks."
                )

        self.track_image = (1 - rgb2gray(image.imread(png_path))).astype(bool)
        self.waypoints = np.load(npy_path)[::10]
        # reverse the waypoints with 50% probability
        if random.random() < 0.5:
            self.waypoints = self.waypoints[::-1]
        self.pygame_track = pygame.image.load(png_path)


    def _get_obs(self):
    #     return {"agent": self._agent_location, "target": self._target_location}
        return self.car.get_state(self.track_image).flatten()  # TODO: no need to flatten I guess

    # def _get_info(self):
    #     return {
    #         "distance": np.linalg.norm(
    #             self._agent_location - self._target_location, ord=1
    #         )
    #     }

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)
        
        self.load_track(self.track)
        self.curr_step = 0

        random_waypoint_index = self.np_random.integers(0, len(self.waypoints)-1)
        loc_idx = random_waypoint_index
        this_pos = self.waypoints[random_waypoint_index]
        next_pos = self.waypoints[(random_waypoint_index + 1) % len(self.waypoints)]
        vec = next_pos - this_pos
        angle = np.arctan2(-vec[1], vec[0])
        # print(this_pos, angle*180/np.pi)
        
        self.car = Car(
            sensor_grid=self.sensor_grid,
            position=to_pygame(this_pos),
            angle=angle
        )

        self.car_coins = Coins(
            coins=np.roll(self.waypoints, -((loc_idx + 2) % len(self.waypoints)), axis=0),
            car=self.car,
            radius=self.hitbox,
        )
        
        observation = self._get_obs()

        if self.render_mode == "human":
            self._render_frame(observation)

        # return observation, None
        return observation, {}

    # def step(self, action):
    #     # Map the action (element of {0,1,2,3}) to the direction we walk in
    #     direction = self._action_to_direction[action]
    #     # We use `np.clip` to make sure we don't leave the grid
    #     self._agent_location = np.clip(
    #         self._agent_location + direction, 0, self.size - 1
    #     )
    #     # An episode is done iff the agent has reached the target
    #     terminated = np.array_equal(self._agent_location, self._target_location)
    #     reward = 1 if terminated else 0  # Binary sparse rewards
    #     observation = self._get_obs()
    #     info = self._get_info()

    #     if self.render_mode == "human":
    #         self._render_frame()

    #     return observation, reward, terminated, False, info

    def step(self, action):
        dt = 0.05  # time step
        left_speed, right_speed = action_to_inputs[action]
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

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()
        # else return None

    def _render_frame(self, sensor_vals=None):  # never gets called if render_mode is None
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((WIDTH, HEIGHT))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((WIDTH, HEIGHT))
        canvas.fill(WHITE)
        canvas.blit(self.pygame_track, (0, 0))
        
        vals = sensor_vals if sensor_vals is not None else self._get_obs()
        
        self.car_coins.display(canvas)
        self.car.display(canvas, vals=vals)

        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to
            # keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
