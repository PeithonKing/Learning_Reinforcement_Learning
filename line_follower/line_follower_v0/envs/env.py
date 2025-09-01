import pygame
import numpy as np
from car import Car
from matplotlib import image
# from matplotlib import pyplot

pygame.init()

WIDTH, HEIGHT = 800, 500
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Line Follower")

BLUE   = (  0,   0, 255)  # #0000FF
GREEN  = (  0, 255,   0)  # #00FF00
RED    = (255,   0,   0)  # #FF0000
WHITE  = (255, 255, 255)  # #FFFFFF
BLACK  = (  0,   0,   0)  # #000000

clock = pygame.time.Clock()
pygame.display.update()

cars = [
    Car(
        sensor_grid=(4, 6),
        angle=np.pi/8
    )
]
track = pygame.image.load("2.png")

def rgb2gray(rgb):
    return np.dot(rgb[..., :4], [0.25, 0.25, 0.25, 0.25])

im = (1-rgb2gray(image.imread("2.png"))).astype(bool)


dt = 0.05
done = False
FPS = 5/dt
while not done:
    screen.fill(WHITE)
    # screen.blit(track, (0, 0))
    clock.tick(FPS)
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            done = True

    for car in cars:
        vals = car.get_state(im)
        car.move(0.5, 0, dt)  # left speed, right speed, dt
        car.display(screen, vals = vals)
    
    pygame.display.update()
pygame.quit()


# class LineFollower:
#     def __init__(self, car, track = 1, dt=0.05):
#         pygame.init()
#         self.WIDTH, self.HEIGHT = 800, 500
#         self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
#         pygame.display.set_caption("Line Follower")
#         self.car = car
#         self.track = pygame.image.load(f"{track}.png")
#         self.clock = pygame.time.Clock()
#         self.dt = dt
#         self.reset()
    
#     def reset(self):
#         self.car.reset()
#         self.out_of_track_count = 0
#         return self.car.get_state(self.track)
    
#     def step(self, action):
#         pass
        
#     def get_reward(self, action, a=1, b=10, c = 0.5):
#         # action = [speed of left wheel, speed of right wheel]
#         # more positive translational speed, more reward, coeff = a
#         # out of track for consecutive 10 steps, kill, penalty, coeff = b
#         # if car.get_state(self.track).sum() == 0: car is out of track, any other value means it is in track
#         # The more the value of car.get_state(self.track).sum(), give more reward, coeff = c

#         sensors_on_track = self.car.get_state(self.track).sum()
#         translational_speed = (action[0] + action[1]) / 2

#         if not sensors_on_track:
#             self.out_of_track_count += 1
#             if self.out_of_track_count >= 10:
#                 return -b, True  # penalty for being out of track for consecutive 10 steps and kill the car
#             else:
#                 return -b, False  # penalty for going off track
        
#         self.out_of_track_count = 0  # reset out_of_track_count if car is back in the track

#         reward = translational_speed * a + sensors_on_track * c

#         return reward, False