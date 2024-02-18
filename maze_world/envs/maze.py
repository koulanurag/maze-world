from copy import copy

import gymnasium as gym
import numpy as np
import pygame
from gymnasium import spaces


class MazeEnv(gym.Env):
    r"""The main Maze Environment class for implementing different maze environments

    The class encapsulates maze environments with arbitrary behind-the-scenes dynamics
    through the :meth:`step` and :meth:`reset` functions.

    The main API methods that users of this class need to know are:

    - :meth:`step` - Updates an environment with actions returning the next agent observation, the reward for taking that actions, if the environment has terminated or truncated due to the latest action and information from the environment about the step, i.e. metrics, debug info.
    - :meth:`reset` - Resets the environment to an initial state, required before calling step.
      Returns the first agent observation for an episode and information, i.e. metrics, debug info.
    - :meth:`render` - Renders the environments to help visualise what the agent see, examples modes are "human", "rgb_array", "ansi" for text.
    - :meth:`close` - Closes the environment, important when external software is used, i.e. pygame for rendering, databases

    Environments have additional attributes for users to understand the implementation

    - :attr:`action_space` - The Space object corresponding to valid actions, all valid actions should be contained within the space.
    - :attr:`observation_space` - The Space object corresponding to valid observations, all valid observations should be contained within the space.
    - :attr:`reward_range` - A tuple corresponding to the minimum and maximum possible rewards for an agent over an episode. The default reward range is set to :math:`(-\infty,+\infty)`.
    - :attr:`spec` - An environment spec that contains the information used to initialize the environment from :meth:`gymnasium.make`
    - :attr:`metadata` - The metadata of the environment, i.e. render modes, render fps
    - :attr:`np_random` - The random number generator for the environment. This is automatically assigned during ``super().reset(seed=seed)`` and when assessing ``self.np_random``.

    Note:
        To get reproducible sampling of actions, a seed can be set with ``env.action_space.seed(123)``.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(
        self,
        generate_maze_fn: callable,
        render_mode: str = None,
        maze_width: int = None,  # columns
        maze_height: int = None,  # rows
    ):
        """

        :param generate_maze_fn: This function is called during every reset of the environment and is expected to three items in following order:

            - maze-map:  numpy array of  map where "1" represents wall and "0" represents floor.
            - agent location: tuple (x,y) where x and y represent location  of agent
            - target location: tuple (x,y) where x and y represent target location  of the agent

        :param render_mode: specifies one of the following:

            - None (default): no render is computed.
            - “human”: The environment is continuously rendered in the current display or terminal, usually for human consumption. This rendering should occur during step() and render() doesn’t need to be called. Returns None.
            - “rgb_array”: Return a single frame representing the current state of the environment. A frame is a np.ndarray with shape (x, y, 3) representing RGB values for an x-by-y pixel image.
            - “ansi”: Return a strings (str) or StringIO.StringIO containing a terminal-style text representation for each time step. The text can include newlines and ANSI escape sequences (e.g. for colors).
            - “rgb_array_list” and “ansi_list”: List based version of render modes are possible (except Human) through the wrapper, gymnasium.wrappers.RenderCollection that is automatically applied during gymnasium.make(...,render_mode="rgb_array_list"). The frames collected are popped after render() is called or reset().

        :param maze_width: The width of the maze
        :param maze_height:  The height of the maze
        """

        self.generate_maze_fn = generate_maze_fn
        self.maze_width = maze_width
        self.maze_height = maze_height

        # The size of the PyGame window
        self._window_pixel_size = 25
        self._window_size = (
            maze_width * self._window_pixel_size,
            maze_height * self._window_pixel_size,
        )

        # Id of the elements in 2d maze:
        # 0 => floor
        # 1 => wall
        # 2 => agent
        self.observation_space = spaces.Dict(
            {
                "agent": spaces.Box(
                    low=np.zeros((self.maze_height, self.maze_width)),
                    high=np.ones((self.maze_height, self.maze_width)) * 2,
                    shape=(
                        self.maze_height,
                        self.maze_width,
                    ),
                    dtype=int,
                ),
                "target": spaces.Box(
                    low=np.zeros((self.maze_height, self.maze_width)),
                    high=np.ones((self.maze_height, self.maze_width)) * 2,
                    shape=(
                        self.maze_height,
                        self.maze_width,
                    ),
                    dtype=int,
                ),
            }
        )

        """
            We have 4 actions, corresponding to "right", "up", "left", "down"
        """

        self.action_space = spaces.Discrete(4)

        """
        The following dictionary maps abstract actions from `self.action_space` to
        the direction we will walk in if that action is taken.
        I.e. 0 corresponds to "right", 1 to "up" etc.
        """
        self._action_to_direction = {
            0: np.array([0, 1]),  # right
            1: np.array([-1, 0]),  # up
            2: np.array([0, -1]),  # left
            3: np.array([1, 0]),  # down
        }

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        self._canvas = None

        """
        If human-rendering is used, `self._window` will be a reference
        to the window that we draw to. `self._clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        self._window = None
        self._clock = None

    def _get_obs(self):
        agent_maze = copy(self.maze_map)
        agent_maze[self._agent_location[0], self._agent_location[1]] = 2

        target_maze = copy(self.maze_map)
        target_maze[self._target_location[0], self._target_location[1]] = 2

        return {"agent": agent_maze, "target": target_maze}

    def _get_info(self):
        return {
            "distance": np.linalg.norm(
                self._agent_location - self._target_location, ord=1
            ),
            "agent": self._agent_location,
            "target": self._target_location,
        }

    def reset(self, seed: int = None, options=None):
        r"""
        Resets the environment to its initial state and generates a new random maze configuration.

        Args:
            seed (int, optional): Seed for the random number generator. Defaults to None.
            options (None): Unused parameter.

        Returns:
            observation: Agent's observation of the initial environment state.
            info (dict): Additional information about the environment.

        Raises:
            ValueError: If the shape of the maze generated by generate_maze_fn() doesn't match the specified maze width and height.

        """
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        # generate a maze:
        self._prev_agent_location = None
        self.maze_map, self._agent_location, self._target_location = (
            self.generate_maze_fn()
        )
        if not np.array_equal(self.maze_map.shape, [self.maze_height, self.maze_width]):
            raise ValueError(
                f"Shape of Generated Maze doesn't match with"
                f" specified maze width and height."
                f" Generate maze shape is {self.maze_map.shape}, "
                f"whereas specified maze width is {self.maze_width}"
                f" and height is {self.maze_height}"
            )

        # return initial parameters
        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()
        self._canvas = None

        return observation, info

    def _no_obstacle(self, location):
        return not (self.maze_map[location[0], location[1]] == 1)  # check for walls

    def step(self, action: int):
        """
        Take a step in the environment.

        Args:
            action (int): The action to take.

        Returns:
            observation: Agent's observation of the current environment.
            reward (float): Reward received after taking the step.
            terminated (bool): Whether the episode has terminated or not.
            truncated (bool): Whether the episode has been truncated due to max episode steps.
            info (dict): Additional information about the step.
        """
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
        if self._window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            pygame.display.set_caption(f"Maze - {self.maze_width} x {self.maze_height}")
            self._window = pygame.display.set_mode(self._window_size)
        if self._clock is None and self.render_mode == "human":
            self._clock = pygame.time.Clock()

        if self._canvas is None:
            self._canvas = pygame.Surface(self._window_size)
            self._canvas.fill((255, 255, 255))

            # draw walls
            for x in range(self.maze_height):
                for y in range(self.maze_width):
                    if self.maze_map[x, y] == 1:
                        pygame.draw.rect(
                            self._canvas,
                            (169, 169, 169),
                            pygame.Rect(
                                np.array([y, x]) * self._window_pixel_size,
                                (self._window_pixel_size, self._window_pixel_size),
                            ),
                        )

            # draw the target
            pygame.draw.rect(
                self._canvas,
                (255, 0, 0),
                pygame.Rect(
                    self._target_location[::-1] * self._window_pixel_size,
                    (self._window_pixel_size, self._window_pixel_size),
                ),
            )

        # Clean previous agent location
        if self._prev_agent_location is not None:
            pygame.draw.circle(
                self._canvas,
                (255, 255, 255),
                (self._prev_agent_location[::-1] + 0.5) * self._window_pixel_size,
                self._window_pixel_size / 4,
            )
        # Draw new agent location
        pygame.draw.circle(
            self._canvas,
            (0, 0, 255),
            (self._agent_location[::-1] + 0.5) * self._window_pixel_size,
            self._window_pixel_size / 4,
        )

        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self._window.blit(self._canvas, self._canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self._clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self._canvas)), axes=(1, 0, 2)
            )

    def close(self):
        """
        Closes the environment.

        This method shuts down the Pygame display if it was initialized.

        """
        if self._window is not None:
            pygame.display.quit()
            pygame.quit()


ACTION_MEANING = {
    0: "Left",
    1: "Up",
    2: "Right",
    3: "Down",
}
OBSERVATION_MEANING = {
    0: "empty",
    1: "wall",
    2: "agent",
}
OBJECT_ID = {v: k for k, v in OBSERVATION_MEANING.items()}
