import gymnasium as gym
import numpy as np
import pytest


@pytest.fixture(scope="module")
def env():
    def _generate_maze_fn():
        # This function would be called on every reset

        maze_map = np.array(
            [
                [1, 1, 1, 1, 1, 1, 1],
                [1, 0, 0, 0, 0, 0, 1],
                [1, 0, 0, 0, 0, 0, 1],
                [1, 0, 0, 0, 0, 0, 1],
                [1, 1, 1, 1, 1, 1, 1],
            ]
        )
        agent_loc = np.array([1, 1])
        target_loc = np.array([3, 5])
        return maze_map, agent_loc, target_loc

    gym.envs.register(
        id="EmptyMaze-v0",
        entry_point="maze_world.envs:MazeEnv",
        max_episode_steps=200,
        kwargs={
            "generate_maze_fn": _generate_maze_fn,
            "maze_height": 5,
            "maze_width": 7,
        },
    )
    env = gym.make("maze_world:EmptyMaze-v0", render_mode="human")
    yield env
    env.close()


@pytest.mark.parametrize(
    "step_actions,agent_positions,step_rewards, terminations, truncations",
    [
        (
                [0, 1, 2, 3, 3, 2, 0, 0, 0, 0],
                [(1, 1), (1, 2), (1, 2), (1, 1), (2, 1), (3, 1), (3, 1), (3, 2), (3, 3), (3, 4), (3, 5)],
                [-0.01, -1.0, -0.01, -0.01, -0.01, -1.0, -0.01, -0.01, -0.01, 1],
                [False, False, False, False, False, False, False, False, False, True],
                [False, False, False, False, False, False, False, False, False, False],
        )
    ],
)
def test_empty_maze_steps(env, step_actions, agent_positions, step_rewards, terminations, truncations):
    observation, info = env.reset()

    assert all(agent_positions[0] == observation["agent"])
    for step_i, action in enumerate(step_actions):
        next_observation, reward, terminated, truncated, info = env.step(action)

        assert all(agent_positions[step_i + 1] == next_observation["agent"])
        assert step_rewards[step_i] == reward
        assert terminations[step_i] == terminated
        assert truncations[step_i] == truncated
