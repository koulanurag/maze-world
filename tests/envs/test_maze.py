import gymnasium as gym
import pytest


@pytest.mark.parametrize('env_name',
                         ["RandomMaze-11x11-v0"])
def test_init(env_name):
    env = gym.make(f'maze-world:{env_name}')
    observation, info = env.reset()
    assert env._prev_agent_location is None

    # rollout
    for step_i in range(5):
        action = env.action_space.sample()
        next_observation, reward, terminated, truncated, info = env.step(action)

    observation, info = env.reset()
    assert env._prev_agent_location is None
