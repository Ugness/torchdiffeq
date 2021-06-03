"""This example spawns (bouncing) balls randomly on a L-shape constructed of
two segment shapes. Not interactive.
"""

__version__ = "$Id:$"
__docformat__ = "reStructuredText"

# Python imports
import random
random.seed(1234, version=2)
from typing import List
import numpy as np

# Library imports
import pygame

# pymunk imports
import pymunk
import pymunk.pygame_util

class BouncyBalls(object):
    """
    This class implements a simple scene in which there is a static platform (made up of a couple of lines)
    that don't move. Balls appear occasionally and drop onto the platform. They bounce around.
    """

    def __init__(self) -> None:
        # Space
        self._space = pymunk.Space()
        self._space.gravity = (0.0, 9.8)

        # Physics
        # Time step
        self._dt = 1.0 / 30.0
        # Number of physics steps per screen frame
        self._physics_steps_per_frame = 1

        # pygame
        pygame.init()
        # self._screen = pygame.display.set_mode((600, 600))
        self._clock = pygame.time.Clock()

        # self._draw_options = pymunk.pygame_util.DrawOptions(self._screen)

        # Static barrier walls (lines) that the balls bounce off of
        self._add_static_scenery()

        # Balls that exist in the world
        self._balls: List[pymunk.Circle] = []

        # Execution control and time until the next ball spawns
        self._running = True
        self._ticks_to_next_ball = 10

    def run(self):
        """
        The main loop of the game.
        :return: None
        """
        max_t = 100
        t = 0
        # Main loop
        data = []
        for i in range(2):
            self._create_ball()
        while t < max_t:
            # Progress time forward
            for x in range(self._physics_steps_per_frame):
                self._space.step(self._dt)

            # self._process_events()
            self._update_balls()
            # self._clear_screen()
            # self._draw_objects()
            # pygame.display.flip()
            # Delay fixed time between frames
            # self._clock.tick(50)
            # pygame.display.set_caption("fps: " +
            x1, y1 = self._space.bodies[0].position
            x2, y2 = self._space.bodies[1].position
            data.append([x1, y1, x2, y2])
            t += 1
        return data

    def _add_static_scenery(self) -> None:
        """
        Create the static bodies.
        :return: None
        """
        static_body = self._space.static_body
        static_lines = [
            pymunk.Segment(static_body, (0, 0), (0, 5), 0.0),
            pymunk.Segment(static_body, (0, 5), (5, 5), 0.0),
            pymunk.Segment(static_body, (5, 5), (5, 0), 0.0),
            pymunk.Segment(static_body, (5, 0), (0, 0), 0.0),
        ]
        for line in static_lines:
            line.elasticity = 0.95
            line.friction = 0.9
        self._space.add(*static_lines)

    def _process_events(self) -> None:
        """
        Handle game and events like keyboard input. Call once per frame only.
        :return: None
        """
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self._running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                self._running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_p:
                pygame.image.save(self._screen, "bouncing_balls.png")

    def _update_balls(self) -> None:
        """
        Create/remove balls as necessary. Call once per frame only.
        :return: None
        """
        # for i in range(2):
        #     self._create_ball()
        # Remove balls that fall below 100 vertically
        # balls_to_remove = [ball for ball in self._balls if ball.body.position.y > 500]
        # for ball in balls_to_remove:
        #     self._space.remove(ball, ball.body)
        #     self._balls.remove(ball)

    def _create_ball(self) -> None:
        """
        Create a ball.
        :return:
        """
        mass = 10
        radius = 0.5
        inertia = pymunk.moment_for_circle(mass, 0, radius, (0, 0))
        body = pymunk.Body(mass, inertia)
        x = random.uniform(radius, 5-radius)
        y = random.uniform(radius, 5-radius)
        body.position = x, y
        # x = random.randint(1, 50)
        # y = random.randint(1, 50)
        # x_pm = random.choice([-1, 1])
        # y_pm = random.choice([-1, 1])
        # body.force = x_pm * x * 10000, y_pm * y * 10000
        shape = pymunk.Circle(body, radius, (0, 0))
        shape.elasticity = 0.95
        shape.friction = 0.9
        self._space.add(body, shape)
        self._balls.append(shape)

    def _clear_screen(self) -> None:
        """
        Clears the screen.
        :return: None
        """
        # self._screen.fill(pygame.Color("white"))
        return

    def _draw_objects(self) -> None:
        """
        Draw the objects.
        :return: None
        """
        # self._space.debug_draw(self._draw_options)
        return


if __name__ == "__main__":
    train = 1000
    test = 25
    val = 25

    trainset = []
    testset = []
    valset = []
    for i in range(train):
        game = BouncyBalls()
        data = game.run()
        trainset.append(data)
    for i in range(val):
        game = BouncyBalls()
        data = game.run()
        valset.append(data)
    for i in range(test):
        game = BouncyBalls()
        data = game.run()
        testset.append(data)

    trainset = np.array(trainset)
    testset = np.array(testset)
    valset = np.array(valset)

    print(trainset.shape)
    print(testset.shape)
    print(valset.shape)
    np.save('train.npy', trainset)
    np.save('test.npy', testset)
    np.save('val.npy', valset)
