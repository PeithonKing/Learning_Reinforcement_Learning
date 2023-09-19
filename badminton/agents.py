import numpy as np

class Player:
    def __init__(self, team, number):
        self.team = team
        self.number = number
        self.reset()
        
    def __str__(self):
        return f"Player {self.number} of team {self.team}"
    
    def reset(self):
        self.position = np.array([np.random.rand(), np.random.rand(), 0])*600
        self.velocity = np.array([0.0, 0.0, 0.0])
    
    def step(self, action, dt=0.1):  # action is the acceleration
        action[2] = 0
        print(f"{action = }")
        self.velocity += (action * 1000 - self.velocity * 0.5) * dt
        self.position += self.velocity * dt
        return self.position



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