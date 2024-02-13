import logging

from gymnasium.envs.registration import register

logger = logging.getLogger(__name__)

register(
    id="RandomMaze-11x11-v0",
    entry_point="maze_world.envs:RandomMazeEnv",
    max_episode_steps=200,
    kwargs={
        "maze_width": 11,
        "maze_height": 11,
        "maze_complexity": 1,
        "maze_density": 1,
    },
)

register(
    id="RandomMaze-15x15-v0",
    entry_point="maze_world.envs:RandomMazeEnv",
    max_episode_steps=200,
    kwargs={
        "maze_width": 15,
        "maze_height": 15,
        "maze_complexity": 1,
        "maze_density": 1,
    },
)

register(
    id="RandomMaze-21x21-v0",
    entry_point="maze_world.envs:RandomMazeEnv",
    max_episode_steps=400,
    kwargs={
        "maze_width": 21,
        "maze_height": 21,
        "maze_complexity": 1,
        "maze_density": 1,
    },
)

register(
    id="RandomMaze-31x31-v0",
    entry_point="maze_world.envs:RandomMazeEnv",
    max_episode_steps=400,
    kwargs={
        "maze_width": 31,
        "maze_height": 31,
        "maze_complexity": 1,
        "maze_density": 1,
    },
)

register(
    id="RandomMaze-51x51-v0",
    entry_point="maze_world.envs:RandomMazeEnv",
    max_episode_steps=500,
    kwargs={
        "maze_width": 51,
        "maze_height": 51,
        "maze_complexity": 1,
        "maze_density": 1,
    },
)

register(
    id="RandomMaze-101x101-v0",
    entry_point="maze_world.envs:RandomMazeEnv",
    max_episode_steps=1000,
    kwargs={
        "maze_width": 101,
        "maze_height": 101,
        "maze_complexity": 1,
        "maze_density": 1,
    },
)
