import pygame
import numpy as np

level = "level1"
FPS = 24

# # game properties
prop = {
    "level1": {
        "dim": (6, 12),
        "mines": 10,
        "square_side": 50,
    },
    "level2": {
        "dim": (10, 20),
        "mines": 35,
        "square_side": 35,
    },
    "level3": {
        "dim": (13, 27),
        "mines": 75,
        "square_side": 25,
    },
}
game_dim = prop[level]["dim"]
mines = prop[level]["mines"]
square_side = prop[level]["square_side"]




pygame.init()

font = pygame.font.SysFont("arial", 30)

WIDTH, HEIGHT = square_side * game_dim[0], square_side * game_dim[1]
print(game_dim)
print(WIDTH, HEIGHT)
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Minesweeper")

RED    = (255,   0,   0)  # #FF0000
BLUE   = (  0,   0, 255)  # #0000FF
WHITE  = (255, 255, 255)  # #FFFFFF
BROWN1 = (229, 194, 159)  # #E5C29F
BROWN2 = (215, 184, 153)  # #D7B899
GREEN1 = (170, 215,  81)  # #AAD751
GREEN2 = (162, 209,  73)  # #A2D149


TEXT = {i: font.render(str(i), False, BLUE) for i in range(1, 9)}


grid = [[None for i in range(game_dim[0])] for j in range(game_dim[1])]
flattened_square = []
flagged = []



def reveal():  # Not to be used in production
    a = [["." for i in range(game_dim[0])] for j in range(game_dim[1])]
    for square in flattened_square:
        if square.mine:
            a[square.x][square.y] = "-"
        else:
            a[square.x][square.y] = square.neighbouring_mines if square.neighbouring_mines else "."

    for i in range(len(a)):
        for j in range(len(a[i])):
            print(a[i][j], end="  ")
        print()





class Square:
    def __init__(self, x, y, xp, yp, shade, side = square_side):
        self.x = x
        self.y = y
        self.xp = xp
        self.yp = yp
        self.colour = (GREEN1, BROWN1) if shade else (GREEN2, BROWN2)  # True for dark, False for light
        self.side = side
        self.mine:bool = False
        self.flagged = False
        self.dug = False
        self.neighbouring_mines = 0
        self.repaint(0)

    def repaint(self, shade=0):
        pygame.draw.rect(
            screen,
            self.colour[shade],
            (self.xp, self.yp, self.side, self.side)
        )
        
    def dig(self):
        if self.mine:
            pass
        self.dug = True
        self.repaint(1)
        
        if self.neighbouring_mines:
            screen.blit(TEXT[self.neighbouring_mines], (self.xp+0.35*square_side, self.yp+0.17*square_side))
            for neighbour in self.get_neighbours([1, 3, 4, 6]):
                if neighbour and not neighbour.dug and not neighbour.mine and neighbour.neighbouring_mines == 0:
                    neighbour.dig()
        else:
            for neighbour in self.get_neighbours([1, 3, 4, 6]):
                if neighbour and not neighbour.dug and not neighbour.mine:
                    neighbour.dig()

    def tl(self):
        X = self.x-1
        Y = self.y-1
        if X not in range(game_dim[1]) or Y not in range(game_dim[0]): return None
        try: return grid[X][Y]
        except IndexError: return None
    def tm(self):
        X = self.x
        Y = self.y-1
        if X not in range(game_dim[1]) or Y not in range(game_dim[0]): return None
        try: return grid[X][Y]
        except IndexError: return None
    def tr(self):
        X = self.x+1
        Y = self.y-1
        if X not in range(game_dim[1]) or Y not in range(game_dim[0]): return None
        try: return grid[X][Y]
        except IndexError: return None
    def lm(self):
        X = self.x-1
        Y = self.y
        if X not in range(game_dim[1]) or Y not in range(game_dim[0]): return None
        try: return grid[X][Y]
        except IndexError: return None
    def rm(self):
        X = self.x+1
        Y = self.y
        if X not in range(game_dim[1]) or Y not in range(game_dim[0]): return None
        try: return grid[X][Y]
        except IndexError: return None
    def bl(self):
        X = self.x-1
        Y = self.y+1
        if X not in range(game_dim[1]) or Y not in range(game_dim[0]): return None
        try: return grid[X][Y]
        except IndexError: return None
    def bm(self):
        X = self.x
        Y = self.y+1
        if X not in range(game_dim[1]) or Y not in range(game_dim[0]): return None
        try: return grid[X][Y]
        except IndexError: return None
    def br(self):
        X = self.x+1
        Y = self.y+1
        if X not in range(game_dim[1]) or Y not in range(game_dim[0]): return None
        try: return grid[X][Y]
        except IndexError: return None

    def get_neighbours(self, filter:list = None):
        a = [self.tl(), self.tm(), self.tr(), self.lm(), self.rm(), self.bl(), self.bm(), self.br()]
        if not filter: return a
        return [a[i] for i in filter]

    def __str__(self):
        return f"Square({self.x}, {self.y})"

def set_stage():
    # pygame.draw.rect(screen, colour, (x_top, y_top, x_len, y_len))
    global grid
    dark = False
    for i in range(game_dim[0]):
        for j in range(game_dim[1]):
            grid[j][i] = Square(j, i, i * square_side, j * square_side, dark)
            flattened_square.append(grid[j][i])
            dark = not dark
        if j%2: dark = not dark

    # plan mines randomly
    mined_square = np.random.choice(flattened_square, mines, replace=False)
    for square in mined_square:
        square.mine = True
    # number the neighbouring squares
    for mined_square in mined_square:
        for square in mined_square.get_neighbours():
            if square is not None:
                square.neighbouring_mines += 1

    reveal()


def get_square(click_x, click_y): # returns square index
    return grid[click_y//square_side][click_x//square_side]

def toggle_flag(square):
    global flagged

    if square.flagged:
        square.flagged = False
        flagged.remove(square)
        square.repaint()
        return

    square.flagged = True
    flagged.append(square)
    x, y = square.xp+0.5*square_side, square.yp+0.5*square_side
    pygame.draw.circle(screen, RED, (x, y), 0.12*square_side)


clock = pygame.time.Clock()

set_stage()
# for i in range(len(grid)):
#     for j in range(len(grid[i])):
#         print(grid[i][j], end=" ")
#     print()
pygame.display.update()

done = False
while not done:
    clock.tick(FPS)
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            done = True
        if event.type == pygame.MOUSEBUTTONUP:
            # event.button
            # 1 - left click
            # 2 - middle click
            # 3 - right click
            # 4 - scroll up
            # 5 - scroll down
            pos = pygame.mouse.get_pos()
            square = get_square(*pos)
            # print(square.x, square.y, "click" if event.button == 1 else "right click")

            if event.button == 1:  # left click digs the location
                if square.flagged:
                    pass
                elif square.mine:
                    print("Game Over")
                    done = True
                else:
                    square.dig()
                        
                
                
                
            # elif event.button == 2:
            #     for f in flagged: print(f)
            elif event.button == 3:  # right click flags the location
                toggle_flag(square)

    pygame.display.update()

pygame.quit()