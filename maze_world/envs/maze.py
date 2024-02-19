from copy import copy

import gymnasium as gym
import numpy as np
import pygame
from gymnasium import spaces


class MazeEnv(gym.Env):
    r"""The main Maze Environment class for implementing different maze environments

    The class encapsulates maze environments with arbitrary behind-the-scenes dynamics
    through the :meth:`step` and :meth:`reset` functions.

    :example:
        >>> import gymnasium as gym
        >>> def generate_maze_fn():
        ...     maze_map = np.array(
        ...         [
        ...             [1, 1, 1, 1, 1, 1, 1],
        ...             [1, 0, 0, 0, 0, 0, 1],
        ...             [1, 0, 0, 0, 0, 0, 1],
        ...             [1, 0, 0, 0, 0, 0, 1],
        ...             [1, 1, 1, 1, 1, 1, 1],
        ...         ]
        ...     )
        ...     agent_loc = np.array([1, 1])
        ...     target_loc = np.array([3, 5])
        ...     return maze_map, agent_loc, target_loc
        >>> env = MazeEnv(generate_maze_fn, None, 5, 7)

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

        :param generate_maze_fn: This function is called during every reset of the environment and is expected to return three items in following order:

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

    @property
    def action_space(self):
        """
        Actions available corresponding to each index: ["right", "up", "left", "down"].

        :return: Discrete action space object representing the possible actions.
        :rtype: gym.spaces.Discrete
        """
        return spaces.Discrete(4)

    @property
    def observation_space(self):
        """
        Defines the observation space of the 2D maze environment.

        The observation space consists of two elements:
        - 'agent': Represents the position of the agent in the maze.
        - 'target': Represents the position of the target in the maze.

        In the 2D maze:
            - 0 corresponds to an empty floor.
            - 1 corresponds to a wall.
            - 2 corresponds to the agent or the target.

        :return: Dictionary containing the observation space for the agent and the target.
            - 'agent': gym.spaces.Box object representing the agent's position.
            - 'target': gym.spaces.Box object representing the target's position.

        :rtype: gym.spaces.Dict
        """
        return spaces.Dict(
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
        """
        Resets the environment to its initial state and generates a new random maze configuration.

        :param seed: Seed for the random number generator. Defaults to None.
        :param options: Unused parameter.

        :return:
            - observation: Agent's observation of the initial environment state.
            - info (dict): Additional information about the environment.
        :rtype: tuple

        :raises ValueError: If the shape of the maze generated by generate_maze_fn() doesn't match the specified maze width and height.
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

        :param action: The action to take.
        :type action: int

        :return: A tuple containing:
            - observation: Agent's current observation of the environment.
            - reward (float): Reward received after taking the step.
            - terminated (bool): Whether the episode has terminated or not.
            - truncated (bool): Whether the episode has been truncated due to max episode steps.
            - info (dict): Additional information about the step.
        :rtype: tuple
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
