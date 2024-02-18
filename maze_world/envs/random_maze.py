import numpy as np

from .maze import MazeEnv
from ..utils import WilsonMazeGenerator


class RandomMazeEnv(MazeEnv):
    r"""Extends the MazeEnv class to create random mazes of specified sizes at each reset."""

    def __init__(
        self,
        render_mode: str = None,
        maze_width: int = 11,
        maze_height: int = 11,
    ):
        r"""
        :param render_mode: specify one of the following:

            - None (default): no render is computed.
            - “human”: The environment is continuously rendered in the current display or terminal, usually for human consumption. This rendering should occur during step() and render() doesn’t need to be called. Returns None.
            - “rgb_array”: Return a single frame representing the current state of the environment. A frame is a np.ndarray with shape (x, y, 3) representing RGB values for an x-by-y pixel image.
            - “ansi”: Return a strings (str) or StringIO.StringIO containing a terminal-style text representation for each time step. The text can include newlines and ANSI escape sequences (e.g. for colors).
            - “rgb_array_list” and “ansi_list”: List based version of render modes are possible (except Human) through the wrapper, gymnasium.wrappers.RenderCollection that is automatically applied during gymnasium.make(...,render_mode="rgb_array_list"). The frames collected are popped after render() is called or reset().

        :param maze_width: The width of the maze
        :param maze_height:  The height of the maze

        Raises:
            ValueError: If the width or height of the maze is not odd.
        """
        if maze_width % 2 == 0 or maze_height % 2 == 0:
            raise ValueError("width/height of maze should be odd")

        self.maze_width = maze_width
        self.maze_height = maze_height

        def _generate_maze():
            """
            Generates a random internal maze configuration.

            Returns:
                tuple: A tuple containing the maze configuration, agent location, and target location.
            """

            # generate internal maze( other than outside wall area)
            generator = WilsonMazeGenerator(maze_height - 2, maze_width - 2)
            generator.generate_maze()

            maze_config = np.ones((maze_height, maze_width), dtype=int)

            # fill maze by skipping outside walls
            maze_config[1:-1, 1:-1] = 1 - np.array(generator.grid)

            agent_location, target_location = self.np_random.choice(
                [
                    ([1, 1], [maze_config.shape[0] - 2, maze_config.shape[1] - 2]),
                    ([maze_config.shape[0] - 2, maze_config.shape[1] - 2], [1, 1]),
                    ([maze_config.shape[0] - 2, 1], [1, maze_config.shape[1] - 2]),
                    ([1, maze_config.shape[1] - 2], [maze_config.shape[0] - 2, 1]),
                ]
            )
            return maze_config, agent_location, target_location

        MazeEnv.__init__(
            self,
            render_mode=render_mode,
            generate_maze_fn=_generate_maze,
            maze_width=maze_width,
            maze_height=maze_height,
        )
