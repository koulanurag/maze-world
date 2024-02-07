import numpy as np

from .maze import MazeEnv


class RandomMazeEnv(MazeEnv):
    def __init__(
        self,
        render_mode: str = None,
        maze_width: int = 11,
        maze_height: int = 11,
        maze_complexity: float = 0.75,
        maze_density: float = 0.75,
    ):

        if maze_width % 2 == 0 or maze_height % 2 == 0:
            raise ValueError("width/height of maze should be odd")

        self.maze_width = maze_width
        self.maze_height = maze_height
        self.maze_complexity = maze_complexity
        self.maze_density = maze_density

        def _generate_maze():
            maze_config = self.generate_maze(
                width=maze_width,
                height=maze_height,
                complexity=maze_complexity,
                density=maze_density,
            )
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
