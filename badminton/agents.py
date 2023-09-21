import numpy as np


class Ball:
    ...

class Player:
    def __init__(self, team:int):  # team will be either 0 or 1
        self.team = team
        self.ID = ...  # random string
        self.brain = ...  # neural network
        self.optimizer = ...  # optimizer
        self.loss = ... # loss function
        self.reset()  # here self.position and self.velocity are initialized

    def __str__(self):
        return f"Player {self.number} of team {self.team}"

    def reset(self):
        self.position = np.array([np.random.rand(), np.random.rand(), 0])*600
        self.velocity = np.array([0.0, 0.0, 0.0])
        
    def action(self, players:list, ball:Ball, dt=0.1):
        # players is a list of 2n players (n for each team)
        # the players are sorted in a particular order:
        #     1. the first player should be the player itself
        #     2. the next n-1 players should be the players of the same team
        #     3. the next n players should be the players of the other team
        # then an (10 * 2n + 6) dimentional vector is made where the first
        # 10 * 2n elements are the positions, velocities, accelerations
        # (each an xyz vector) and team of the players and the last 6
        # elements are the position and velocity of the ball
        # then this vector is fed to the neural network and the output is
        # the acceleration vector of this player which is returned
        acceleration_vector = ... # implement
        return acceleration_vector

    def step(self, action, dt=0.1):  # action is the acceleration vector in xyz
        action[2] = 0  # because the player cannot jump
        # print(f"{action = }")
        self.velocity += (action * 1000 - self.velocity * 0.5) * dt
        self.position += self.velocity * dt
        return self.position
    
    def update(self, reward: int):
        # use the reward to do the back propagation and update the weights of the neural network
        # remember, the reward has to be maximised in this case


if __name__ == "__main__":
    import pygame
    pygame.init()    

    screen = pygame.display.set_mode((1000, 600))
    pygame.display.set_caption("Player Simulation")

    WHITE  = (255, 255, 255)  # #FFFFFF
    BLACK  = (  0,   0,   0)  # #000000

    clock = pygame.time.Clock()

    player = Player(1, 1)
    screen.fill(WHITE)

    pygame.display.update()

    dt = 0.05
    done = False
    FPS = 5/dt
    while not done:
        clock.tick(FPS)
        action = np.array([0.0, 0.0, 0.0])
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT: action[0] = -1
                if event.key == pygame.K_RIGHT: action[0] = 1
                if event.key == pygame.K_UP: action[1] = -1
                if event.key == pygame.K_DOWN: action[1] = 1
                
        player.step(action, dt)
        
        print(f"{player.position = }")
        print(f"{player.velocity = }")
        
        screen.fill(WHITE)
        pygame.draw.circle(screen, BLACK, (player.position[:2]), 5)
        
        pygame.display.update()
    pygame.quit()