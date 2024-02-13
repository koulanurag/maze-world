import gymnasium as gym
import pytest

from maze_world.utils import maze_dijkstra_solver


@pytest.mark.parametrize(
    "env_name",
    [
        "maze_world:RandomMaze-11x11-v0",
        "maze_world:RandomMaze-15x15-v0",
        "maze_world:RandomMaze-21x21-v0",
    ],
)
def test_dijkstra_solver(env_name):
    env = gym.make(env_name)
    observation, info = env.reset(seed=0, options={})
    episode_score = 0.0

    optimal_actions = maze_dijkstra_solver(
        env.unwrapped.maze_map.astype(bool),
        env.unwrapped._action_to_direction.values(),
        info["agent"],
        info["target"],
    )
    for action in optimal_actions:
        observation, reward, terminated, truncated, info = env.step(action)
        episode_score += reward
        if terminated or truncated:
            break
    env.close()
    assert (observation["agent"] == observation["target"]).all()
