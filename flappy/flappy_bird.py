import numpy as np
import random


class FlappyBirds:

    def __init__(
        self,
        nx: int = 10,
        ny: int = 50,
        g:float = 1,
        # bird_x:int = 0,  # don't change this
        bird_u:float = 0,
        obs_v:int = 1,
        obs_w:int = 1,  # width of opening = 1 + 2*obs_width
        up_dv:int = 5,  # change in velocity when button is pressed
        reward: list = [-1, 0, 1] # reward for dying, doing nothing, and passing an obstacle
    ):
        self.nx = nx
        self.ny = ny
        self.g = g  # gravitational acceleration
        self.action_space = np.array([0, 1])
        self.observation_space = nx*ny
        self.up_dv = up_dv
        self.reward = reward

        # Bird Properties
        self.bird_x = 0  # bird_x  # fixed, never changes
        self.bird_y = ny//2-1  # changes, simulate gravity
        self.bird_u = bird_u  # const, used only for resetting
        self.bird_v = bird_u  # changes, simulate gravity

        # Obstacle Properties
        self.obs_x = nx-1  # changes, fixed speed
        self.obs_y = random.randint(obs_w, ny-1-obs_w)  # obstacle opening
        self.obs_v = obs_v # fixed, never changes
        self.obs_w = obs_w # fixed, never changes

    def reset(self):
        # Bird Properties
        self.bird_y = self.ny//2  # changes, simulate gravity
        self.bird_v = self.bird_u  # changes, simulate gravity

        # Obstacle Properties
        self.obs_x = self.nx-1  # changes, fixed speed
        self.obs_y = random.randint(0, self.ny-1)  # obstacle opening
        return self.obs_x - self.bird_x, self.obs_y - self.bird_y, self.bird_y

    def step(self, action):
        if action == 1: self.bird_v -= self.up_dv
        done = False
        reward = 0

        self.bird_y = self.bird_y + self.bird_v + 0.5*self.g
        ret_bird_y = int(round(self.bird_y, 0))
        self.bird_v += self.g
        self.obs_x = self.obs_x - self.obs_v

        dist_x = self.obs_x - self.bird_x
        dist_y = self.obs_y - ret_bird_y

        if ret_bird_y < 0:
            return dist_x, self.obs_y, 0, True, self.reward[0]
        elif ret_bird_y >= self.ny:
            return dist_x, self.obs_y - self.ny, self.ny-1, True, self.reward[0]

        if dist_x <= 0:
            if abs(dist_y) <= self.obs_w:
                return dist_x, dist_y, ret_bird_y, True, self.reward[2]
            else:
                return dist_x, dist_y, ret_bird_y, True, self.reward[0]

        return dist_x, dist_y, ret_bird_y, False, self.reward[1]

    def render(self):
        # print(f"{self.bird_y = }, {self.bird_v = }")
        a = [["."]*self.nx for _ in range(self.ny)]

        a[int(round(self.bird_y, 0))%self.ny][self.bird_x] = "B"

        for i in range(self.ny):
            if not (self.obs_y-self.obs_w <= i <= self.obs_y+self.obs_w):
                a[i][self.obs_x] = "O"

        for row in a:
            for element in row:
                print(element, end="  ")
            print()