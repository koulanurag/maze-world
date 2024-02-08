import gymnasium as gym
import pytest


@pytest.mark.parametrize(
    "env_name",
    [
        "RandomMaze-11x11-v0",
        "RandomMaze-21x21-v0",
        "RandomMaze-31x31-v0",
        "RandomMaze-101x101-v0",
    ],
)
def test_init(env_name):
    env = gym.make(f"maze_world:{env_name}")
    observation, info = env.reset()

    assert all(info["agent"] == [1, 1]), "initial agent position is not correct"
    assert all(
        info["target"]
        == [
            env.unwrapped.maze_width - 2,
            env.unwrapped.maze_height - 2,
        ]
    ), "target position is not correct"
    assert observation[info["agent"][0], info["agent"][1]] == 2
    assert observation[info["target"][0], info["target"][1]] == 3

    # rollout
    for step_i in range(5):
        action = env.action_space.sample()
        next_observation, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break
    observation, info = env.reset()

    assert all(
        info["agent"] == [1, 1]
    ), "initial agent position is not correct, after episode run and reset"
    assert all(
        info["target"]
        == [
            env.unwrapped.maze_width - 2,
            env.unwrapped.maze_height - 2,
        ]
    ), "target position is not correct, after episode run and reset"
    assert observation[info["agent"][0], info["agent"][1]] == 2
    assert observation[info["target"][0], info["target"][1]] == 3
