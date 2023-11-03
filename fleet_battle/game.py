import numpy as np
import random

# each ship has multiple orientations, each orientation has a matrix and a center point

ships = [
    [
        np.array([
            [1, 1, 1, 1, 1]
        ]),
        np.array([
            [1],
            [1],
            [1],
            [1],
            [1]
        ])
    ],
    [
        np.array([
            [1, 1, 1, 1]
        ]),
        np.array([
            [1],
            [1],
            [1],
            [1]
        ])
    ],
    [
        np.array([
            [1, 1, 1],
            [0, 1, 0]
        ]),
        np.array([
            [0, 1, 0],
            [1, 1, 1]
        ]),
        np.array([
            [1, 0],
            [1, 1],
            [1, 0]
        ]),
        np.array([
            [0, 1],
            [1, 1],
            [0, 1]
        ])
    ],
    [
        np.array([
            [1, 1, 1, 0],
            [0, 1, 1, 1]
        ]),
        np.array([
            [0, 1],
            [1, 1],
            [1, 1],
            [1, 0]
        ]),
    ],
    [
        np.array([
            [1, 1, 1]
        ]),
        np.array([
            [1],
            [1],
            [1]
        ])
    ],
    [
        np.array([
            [1, 1]
        ]),
        np.array([
            [1],
            [1]
        ])
    ]
]


class FleetBattleEnv:

    def __init__(self, action_space = (10, 10), ships = ships, max_steps = None):
        self.ships = ships
        self.action_space = action_space
        self.max_steps = max_steps
        # self.reset()  # don't reset here, reset in the loop

    def reset(self):
        self.state = np.zeros((2, *self.action_space))
        # 0th (10, 10) matrix represents bombed cells
        # 1st (10, 10) matrix represents hit cells
        # ofcourse 1st matrix is a subset of 0th matrix
        self.turns = 0
        self.score = 0

        # build map, which will be hidden from the agent
        self.map = np.zeros(self.action_space)
        # for i, ship in enumerate(self.ships):  # remove the enumerate, it's just for debugging
        for ship in self.ships:
            ship_orientation = random.choice(ship)
            placed = False
            while not placed:
                x = random.randint(0, self.action_space[0] - ship_orientation.shape[0])
                y = random.randint(0, self.action_space[1] - ship_orientation.shape[1])
                
                if self.map[x:x+ship_orientation.shape[0], y:y+ship_orientation.shape[1]].sum() == 0:
                    placed = True

            # self.map[x:x+ship_orientation.shape[0], y:y+ship_orientation.shape[1]] = ship_orientation * 30 * (i + 1)
            self.map[x:x+ship_orientation.shape[0], y:y+ship_orientation.shape[1]] = ship_orientation

        return self.state

    def step(self, action):
        self.turns += 1
        hit = False
        self.state[0, action[0], action[1]] = 1
        if self.map[action[0], action[1]] > 0:
            self.state[1, action[0], action[1]] = 1
            hit = True
            self.score += 1

        return (
            self.state,                   # state
            self.turns >= self.max_steps, # game over or not
            1 if hit else 0,              # reward
            # could have just been int(hit), but this is more readable
            self.score,                   # score
        )


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    env = FleetBattleEnv()
    print(env.map.shape)
    plt.imshow(env.map, cmap="gray")
    plt.show()