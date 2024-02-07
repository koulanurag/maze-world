import gymnasium as gym
import numpy as np
import pygame
from gymnasium import spaces


class MazeEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(
        self,
        generate_maze_fn,
        render_mode: str = None,
        maze_width: int = None,
        maze_height: int = None,
    ):
        self.generate_maze_fn = generate_maze_fn
        self.maze_width = maze_width
        self.maze_height = maze_height

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
                    low=np.array([0, 0]),
                    high=np.array([maze_width - 1, maze_height - 1]),
                    shape=(2,),
                    dtype=int,
                ),
                "target": spaces.Box(
                    low=np.array([0, 0]),
                    high=np.array([maze_width - 1, maze_height - 1]),
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
        self._prev_agent_location = None
        self.maze_map, self._agent_location, self._target_location = (
            self.generate_maze_fn()
        )
        if not np.array_equal(self.maze_map.shape, [self.maze_width, self.maze_height]):
            raise ValueError(
                "Shape of Generated Maze doesn't match with specified maze width and height"
            )

        # return initial parameters
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
            pygame.display.set_caption(f"Maze - {self.maze_width} x {self.maze_height}")
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
