import numpy as np


def generate_maze(width=81, height=51, complexity=0.75, density=0.75):
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
            np.random.randint(0, shape[1] // 2 + 1) * 2,
            np.random.randint(0, shape[0] // 2 + 1) * 2,
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
                y_, x_ = neighbours[np.random.randint(0, len(neighbours))]
                if maze[y_, x_] == 0:
                    maze[y_, x_] = 1
                    maze[y_ + (y - y_) // 2, x_ + (x - x_) // 2] = 1
                    x, y = x_, y_

    return maze.astype(int)


if __name__ == "__main__":
    maze = generate_maze(11, 11, density=0.75)
    print(maze)
    print(maze.dtype)
    print(maze.shape)
