import numpy as np
import pygame

def to_pygame(points, height=500):
    """
    Flip y-axis for Pygame screen coordinates.

    Args:
        points (array-like): Single point [x, y] or array of points [[x1, y1], ...].
        height (int, optional): Screen or canvas height. Defaults to 500.

    Returns:
        np.ndarray: Points with y-axis flipped, same shape as input.
    """
    points = np.asarray(points)
    points_flipped = points.copy()
    points_flipped[..., 1] = height - points_flipped[..., 1]
    return points_flipped

def rotate_points(points, theta):
    theta -= np.pi/2
    # Create the rotation matrix (taking into account the y-axis pointing down)
    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)],
                                [np.sin(theta), np.cos(theta)]])
    # Apply the rotation matrix to the points
    rotated_points = np.dot(rotation_matrix, points.T).T
    return rotated_points

class Car:
    def __init__(
        self,
        sensor_grid = (4, 6),
        position=np.array([100, 100]),
        angle = np.pi/8
    ):
        self.sensor_grid = sensor_grid  # (rows, cols)
        self.width  = sensor_grid[0]*20
        self.height = sensor_grid[1]*20  # (width, height) in pixels
        self.pos0     = position  # (x, y) in pixels
        self.ang0     = angle     # angle in radians
        self.reset()
        self.sensors  = np.zeros(sensor_grid)
        self.corners  = np.array(
            [
                [-self.width/2, -self.height/2],  # bottom left
                [ self.width/2, -self.height/2],  # bottom right
                [ self.width/2,  self.height/2],  # top right
                [-self.width/2,  self.height/2]   # top left
            ]
        )
        self.sensor_points = self._get_sensor_points_(self.width, self.height, *sensor_grid)
        
    def reset(self):
        self.angle = self.ang0
        self.position = self.pos0

    def get_car(self):
        """Get the corners and sensor points of the car.

        Returns:
            corners (np.array): Array of shape (4, 2) containing the x and y coordinates of the corners of the car.
            sensors (np.array): Array of shape (n, 2) containing the x and y coordinates of the sensors of the car.
        """
        corners = rotate_points(self.corners,       self.angle) + self.position
        sensors = rotate_points(self.sensor_points, self.angle) + self.position
        return corners, sensors

    def move(self, speed_left_wheel, speed_right_wheel, dt):
        """Move the car forward in time. Update the position and angle of the car.

        Args:
            speed_right_wheel (float): Speed of the right wheel in some units.
            speed_left_wheel (float): Speed of the left wheel in the same units.
            dt (float): Change in time in some units.
        """
        distance_between_wheels = self.width
        current_location = self.position
        current_angle = self.angle
        
        speed_left_wheel *= 100
        speed_right_wheel *= 100

        if speed_right_wheel == speed_left_wheel:
            distance_moved = speed_right_wheel * dt
            new_x = current_location[0] + distance_moved * np.cos(current_angle)
            new_y = current_location[1] + distance_moved * np.sin(current_angle)
            new_location = np.array((new_x, new_y))
            
            self.position = new_location
            # self.angle no change
        else:
            angular_velocity = (speed_right_wheel - speed_left_wheel) / distance_between_wheels

            # Calculate the new angle of the vehicle
            change_in_angle = angular_velocity * dt
            new_angle = current_angle + change_in_angle
            movement_angle = current_angle + change_in_angle / 2
            
            distance_moved = (speed_right_wheel + speed_left_wheel) * dt / 2
            
            direction = np.array([np.cos(movement_angle), np.sin(movement_angle)])
            delta = distance_moved * direction
            
            # this method is not very accurate, but it works for now provided distance_moved is small
            new_location = current_location + delta
            
            self.position = new_location
            self.angle = new_angle

    def display(self, screen, color = (0, 0, 255), vals=None, sensor_color = (255, 0, 0)):
        """Display the car on the screen. Both the body and the sensors are displayed.

        Args:
            screen (pygame.Surface): The screen on which the car is to be displayed.
            color (tuple, optional): The color of the car. Defaults to a dark blue.
            vals (iterable): values read by the sensors
            sensor_color (tuple, optional): The colour if the sensor is activated. Defaults to red.
        """
        corners, sensors = self.get_car()
        
        # draw the car
        # print(to_pygame(corners), "\n")
        pygame.draw.polygon(screen, color, to_pygame(corners), 2)
        

        # draw the sensors
        for sensor in sensors:
            pygame.draw.circle(screen, color, to_pygame(sensor), 4, 1)
            

        # put a red line to mark the front of the car
        pygame.draw.line(screen, (255, 0, 0), to_pygame(np.mean(corners, axis=0)), to_pygame(np.mean(corners[2:], axis=0)), 2)
        
        # pygame.draw.circle(screen, (0, 255, 0), to_pygame(corners[0]), 4, 2)  # colour = yellow

        # show the values read by the sensor if values are provided
        if vals is not None:
            if len(sensors) != len(vals):
                raise ValueError(f"Number of sensors and number of values must be the same. Got {len(sensors)} sensors and {len(vals)} values.")
            for sensor, val in zip(sensors, vals):
                if val: pygame.draw.circle(screen, sensor_color, to_pygame(sensor), 4)
    
    def get_state(self, image):
        """Get the values read by the sensors.

        Args:
            image (np.array): The image on which the sensors are to be used.

        Returns:
            np.array: Array of shape (n,) containing the values read by the sensors.
        """
        sensors = to_pygame(self.get_car()[1])
        # print(sensors)
        vals = []
        for sensor in sensors:
            # print(sensor)
            # print(image[int(sensor[1]), int(sensor[0])])
            try:
                got = image[int(sensor[1]), int(sensor[0])]
            except IndexError:
                got = False
            vals.append(got)
        return np.array(vals)


    def _get_sensor_points_(self, height, width, rows, columns):
        """Get the positions of the sensors in the car's frame of reference."""
        # in goes number of sensors and the height and width of the car
        # out comes the position of sensors in the car's frame of reference
        # Calculate the distance between adjacent rows and columns
        row_distance = height / (rows)
        col_distance = width / (columns)
        
        # Calculate the x and y coordinates of each point using list comprehensions
        points = np.array([[[j-(rows-1)/2, (columns-1)/2-i] for j in range(rows)] for i in range(columns)])

        return points.reshape(-1, 2)*np.array([row_distance, col_distance])


class Coins:
    def __init__(self, coins, car, radius=30):
        self.coins = coins
        self.radius = radius
        self.car = car
        # print(*coins, sep="\n")
        # print(coins[:, 0].min(), coins[:, 0].max())
        # print(coins[:, 1].min(), coins[:, 1].max())
        # quit()

    def get_reward(self):
        position = to_pygame(self.car.position)
        # position = self.car.position
        reward = 0
        while reward<len(self.coins):
            # the coins have to be collected in order
            coin = self.coins[0]
            if not np.linalg.norm(position - coin) < self.radius:
                break
            reward += 1
            self.coins = np.roll(self.coins, -1, axis=0)
        # if reward: print(reward)
        return reward

    def display(self, screen):
        YELLOW = (255, 255, 0)
        RED = (255, 0, 0)
        GREEN = (0, 255, 0)
        rad = 4
        for i, coin in enumerate(self.coins, start=1):
            t = i / len(self.coins)   # 0 → 1 across coins
            # interpolate between YELLOW and BLACK
            r = int(YELLOW[0] * max(1 - 2*t, 0))
            g = int(YELLOW[1] * max(1 - 2*t, 0))
            b = int(YELLOW[2] * max(1 - 2*t, 0))
            color = (r, g, b)
            pygame.draw.circle(screen, color, coin, rad)
        # pygame.draw.circle(screen, GREEN, self.coins[0], rad)

        pygame.draw.circle(screen, GREEN, to_pygame(self.car.position), self.radius, 1)
