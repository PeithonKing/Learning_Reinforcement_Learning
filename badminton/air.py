import pygame
import numpy as np
from utils import WHITE, BLACK, RED

g = 25
size = 1000

# origin will be at the bottom left corner , such that the court comes in the first quadrant

pygame.init()

# font = pygame.font.SysFont("arial", 30)

def d2p(distance, unit=1):  # distance to pixels
    """Converts distance to pixels

    Args:
        distance (float): distance in real world
        unit (int, optional): 0: meters, 1: feet. Defaults to feet (1).

    Returns:
        float: distance in pixels
    """
    ret = distance*size/44
    if unit == 0:
        return ret*100/(2.54*12)  # 1 inch = 2.54 cm, 1 foot = 12 inches
    return ret


X_C = d2p(44)  # 44 feet
Y_C = d2p(20)  # 20 feet
Z_C = Y_C

space = d2p(1.1)  # 1.1 feet
WIDTH = X_C + space*2
HEIGHT = Y_C + space*2 + Z_C
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Air Drag Simulation")


def next_state(s, v, dt, c = 1, g = g):
    a = np.array([-c*v[0], -c*v[1], -c*v[2]-g])
    # print(f"{a.dtype = }")
    # print(f"{v.dtype = }")
    v_ = v + a*dt
    s_ = s + v_*dt
    return s_, v_

def transform(s: np.ndarray, view: int = 0):
    """Converts regular coordinates to pygame coordinates and the
    views go to the two different parts of the screen.

    Args:
        s (np.ndarray): position vector iterable of length 3
        view (int): 0: top view, 1: side view
        
    Returns:
        tuple: (x, y) coordinates
    """
    if len(s) == 2:
        x, y = s
        z = 0
    else: x, y, z = s
    # x needs no change
    # y should not be sent if view == 1,
    # y = HEIGHT - y if y<HEIGHT/2 else -1
    # z should not be sent if view == 0
    # z = HEIGHT/2 - z
    # if view == 0: x and y are sent, if view == 1: x and z are sent
    x = x + space
    if view == 0:  # top view
        return x, HEIGHT-(y+space) if (y+space)<=HEIGHT/2 else -1
    if view == 1:  # side view
        return x, Z_C - z

def draw_court():
    net_height = d2p(5+1/12)  # 5 feet 1 inch
    screen.fill(WHITE)
    
    # middle separation line between the two views
    pygame.draw.line(screen, BLACK, (0, Z_C), (WIDTH, Z_C), 1)
    
    # net in view == 0 (top view)
    pygame.draw.line(screen, BLACK, transform((X_C/2, 0, 0)), transform((X_C/2, Y_C, 0)), 3)
    # net in view == 1 (side view)
    pygame.draw.line(screen, BLACK, (WIDTH/2, Z_C), (WIDTH/2, Z_C-net_height), 3)

    # Court
    points = [
        transform((  0,   0)),
        transform((X_C,   0)),
        transform((X_C, Y_C)),
        transform((  0, Y_C)),
    ]
    pygame.draw.lines(screen, BLACK, True, points, 1)  # outer rectangle
    pygame.draw.line(screen, BLACK, transform((     d2p(2.5),            0)), transform((     d2p(2.5),          Y_C)), 1)  # left back service line
    pygame.draw.line(screen, BLACK, transform(( X_C-d2p(2.5),            0)), transform(( X_C-d2p(2.5),          Y_C)), 1)  # right back service line
    pygame.draw.line(screen, BLACK, transform((    d2p(15.5),            0)), transform((    d2p(15.5),          Y_C)), 1)  # left front service line
    pygame.draw.line(screen, BLACK, transform((X_C-d2p(15.5),            0)), transform((X_C-d2p(15.5),          Y_C)), 1)  # right front service line
    pygame.draw.line(screen, BLACK, transform((            0,     d2p(1.5))), transform((          X_C,     d2p(1.5))), 1)  # side line 1
    pygame.draw.line(screen, BLACK, transform((            0, Y_C-d2p(1.5))), transform((          X_C, Y_C-d2p(1.5))), 1)  # side line 2
    pygame.draw.line(screen, BLACK, transform((            0,      d2p(10))), transform((    d2p(15.5),      d2p(10))), 1)  # middle line left
    pygame.draw.line(screen, BLACK, transform((X_C-d2p(15.5),   d2p(10, 1))), transform((          X_C,      d2p(10))), 1)  # middle line right


clock = pygame.time.Clock()

pygame.display.update()


# s = np.array([0.0, 0.0, 0.0])
# v = np.array([100.0, 0.0, 100.0])

dt = 0.05
done = False
FPS = 5/dt
while not done:

    # regular pygame stuff
    clock.tick(FPS)
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            done = True

    # reset the game when ball hits the ground
    if s[2]<=0:  # (s[2] = z) <= 0 means it hit the ground
        s = np.array([0.0, 0.0, 0.0])
        v = np.array([100.0, 100.0, 100.0])*1.3
        draw_court()
        print("again")

    # update the position and velocity and draw the ball
    s, v = next_state(s, v, dt, c=0.2)
    # print(f"{s = }, {v = }")
    pygame.draw.circle(screen, RED, transform(s, 0), radius=1)  # top view
    pygame.draw.circle(screen, RED, transform(s, 1), radius=1)  # side view
    
    pygame.display.update()
pygame.quit()