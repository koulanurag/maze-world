import gymnasium as gym
import numpy as np
import pygame
from gymnasium import spaces


class MazeWorldEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(
        self,
        render_mode=None,
        maze_width=11,
        maze_height=11,
        maze_complexity=0.75,
        maze_density=0.75,
    ):
        if maze_width % 2 == 0 or maze_height % 2 == 0:
            raise ValueError("width/height of maze should be odd")

        self.maze_width = maze_width
        self.maze_height = maze_height
        self.maze_complexity = maze_complexity
        self.maze_density = maze_density

        # The size of the PyGame window
        self.window_pixel_size = 25
        self.window_size = (
            maze_width * self.window_pixel_size,
            maze_height * self.window_pixel_size,
        )

        # Observations are dictionaries with the agent's and the target's location.
        # Each location is encoded as an element of {0, ..., `size`}^2, i.e. MultiDiscrete([size, size]).
        self.observation_space = spaces.Dict(
            {
                "agent": spaces.Box(
                    low=np.array([1, 1]),
                    high=np.array([maze_width - 2, maze_height - 2]),
                    shape=(2,),
                    dtype=int,
                ),
                "target": spaces.Box(
                    low=np.array([1, 1]),
                    high=np.array([maze_width - 2, maze_height - 2]),
                    shape=(2,),
                    dtype=int,
                ),
            }
        )

        # We have 4 actions, corresponding to "right", "up", "left", "down"
        self.action_space = spaces.Discrete(4)

        """
        The following dictionary maps abstract actions from `self.action_space` to
        the direction we will walk in if that action is taken.
        I.e. 0 corresponds to "right", 1 to "up" etc.
        """
        self._action_to_direction = {
            0: np.array([1, 0]),
            1: np.array([0, 1]),
            2: np.array([-1, 0]),
            3: np.array([0, -1]),
        }

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        self.canvas = None

        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        self.window = None
        self.clock = None

    def _get_obs(self):
        return {"agent": self._agent_location, "target": self._target_location}

    def _get_info(self):
        return {
            "distance": np.linalg.norm(
                self._agent_location - self._target_location, ord=1
            )
        }

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        # generate a maze:
        self.maze_map = self.generate_maze(
            width=self.maze_width,
            height=self.maze_height,
            complexity=self.maze_complexity,
            density=self.maze_density,
        )
        self._prev_agent_location = None
        self._agent_location = np.array([1, 1])
        self._target_location = np.array([self.maze_width - 2, self.maze_height - 2])

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()
        self.canvas = None

        return observation, info

    def _no_obstacle(self, location):
        return not (self.maze_map[location[0], location[1]] == 1)  # check for walls

    def step(self, action):
        terminated = False

        # Map the action (element of {0,1,2,3}) to the direction we walk in
        direction = self._action_to_direction[action]
        new_agent_location = self._agent_location + direction

        if self._no_obstacle(new_agent_location):
            self._prev_agent_location, self._agent_location = (
                self._agent_location,
                new_agent_location,
            )
            if np.array_equal(
                self._agent_location, self._target_location
            ):  # goal location check
                reward = +1  # reward for reaching goal location
                terminated = True
            else:
                reward = -0.01  # step cost
        else:
            reward = -1  # high penalty for colliding with walls

        observation = self._get_obs()
        info = self._get_info()
        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, False, info

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            pygame.display.set_caption(f'Maze - {self.maze_width} x {self.maze_height}')
            self.window = pygame.display.set_mode(self.window_size)
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        if self.canvas is None:
            self.canvas = pygame.Surface(self.window_size)
            self.canvas.fill((255, 255, 255))

            # draw walls
            for x in range(self.maze_width):
                for y in range(self.maze_height):
                    if self.maze_map[x, y] == 1:
                        pygame.draw.rect(
                            self.canvas,
                            (169, 169, 169),
                            pygame.Rect(
                                np.array([x, y]) * self.window_pixel_size,
                                (self.window_pixel_size, self.window_pixel_size),
                            ),
                        )

            # draw the target
            pygame.draw.rect(
                self.canvas,
                (255, 0, 0),
                pygame.Rect(
                    self._target_location * self.window_pixel_size,
                    (self.window_pixel_size, self.window_pixel_size),
                ),
            )

        # Clean previous agent location
        if self._prev_agent_location is not None:
            pygame.draw.circle(
                self.canvas,
                (255, 255, 255),
                (self._prev_agent_location + 0.5) * self.window_pixel_size,
                self.window_pixel_size / 4,
            )
        # Draw new agent location
        pygame.draw.circle(
            self.canvas,
            (0, 0, 255),
            (self._agent_location + 0.5) * self.window_pixel_size,
            self.window_pixel_size / 4,
        )

        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(self.canvas, self.canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.canvas)), axes=(1, 0, 2)
            )

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()

    def generate_maze(self, width=81, height=51, complexity=0.75, density=0.75):
        r"""Generate a random maze array.

        It only contains two kind of objects, obstacle and free space. The numerical value for obstacle
        is ``1`` and for free space is ``0``.

        Code from https://en.wikipedia.org/wiki/Maze_generation_algorithm
        """

        if width % 2 == 0 or height % 2 == 0:
            raise ValueError("width/height must be an odd integer")

        # Only odd shapes
        shape = ((height // 2) * 2 + 1, (width // 2) * 2 + 1)
        # Adjust complexity and density relative to maze size
        complexity = int(complexity * (5 * (shape[0] + shape[1])))
        density = int(density * ((shape[0] // 2) * (shape[1] // 2)))

        # Build
        maze = np.zeros(shape, dtype=bool)
        # Fill borders
        maze[0, :] = maze[-1, :] = 1
        maze[:, 0] = maze[:, -1] = 1
        # Make aisles
        for i in range(density):
            x, y = (
                self.np_random.integers(0, shape[1] // 2 + 1) * 2,
                self.np_random.integers(0, shape[0] // 2 + 1) * 2,
            )
            maze[y, x] = 1
            for j in range(complexity):
                neighbours = []
                if x > 1:
                    neighbours.append((y, x - 2))
                if x < shape[1] - 2:
                    neighbours.append((y, x + 2))
                if y > 1:
                    neighbours.append((y - 2, x))
                if y < shape[0] - 2:
                    neighbours.append((y + 2, x))
                if len(neighbours):
                    y_, x_ = neighbours[self.np_random.integers(0, len(neighbours))]
                    if maze[y_, x_] == 0:
                        maze[y_, x_] = 1
                        maze[y_ + (y - y_) // 2, x_ + (x - x_) // 2] = 1
                        x, y = x_, y_

        return maze.astype(int)
