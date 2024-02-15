import numpy as np

from .maze import MazeEnv
from ..utils import WilsonMazeGenerator


class RandomMazeEnv(MazeEnv):
    def __init__(
        self,
        render_mode: str = None,
        maze_width: int = 11,
        maze_height: int = 11,
    ):
        if maze_width % 2 == 0 or maze_height % 2 == 0:
            raise ValueError("width/height of maze should be odd")

        self.maze_width = maze_width
        self.maze_height = maze_height

        def _generate_maze():
            # generate internal maze( other than outside wall area)
            generator = WilsonMazeGenerator(maze_height - 2, maze_width - 2)
            generator.generate_maze()

            maze_config = np.ones((maze_height, maze_width), dtype=int)

            # fill maze by skipping outside walls
            maze_config[1:-1, 1:-1] = 1 - np.array(generator.grid)

            agent_location = np.array([1, 1])
            target_location = np.array(
                [maze_config.shape[0] - 2, maze_config.shape[1] - 2]
            )
            return maze_config, agent_location, target_location

        MazeEnv.__init__(
            self,
            render_mode=render_mode,
            generate_maze_fn=_generate_maze,
            maze_width=maze_width,
            maze_height=maze_height,
        )
