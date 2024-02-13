# maze-world

Random maze environments with different size and complexity for reinforcement learning research.

![Python package](https://github.com/koulanurag/maze-world/workflows/Python%20package/badge.svg)
![Python Version](https://img.shields.io/pypi/pyversions/maze-world)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/koulanurag/maze-world/blob/main/examples/colab_example.ipynb)


## Installation

- Using PyPI:
   ```bash
   pip install maze-world
   ```

- Directly from source (recommended):
   ```bash
   git clone https://github.com/koulanurag/maze-world.git
   cd maze-world
   pip install -e .
   ```
## Environments Zoo!

<div style="text-align:center;">
  <table>
    <tr>
      <td><b>RandomMaze-11x11-v0</b></td>
      <td><b>RandomMaze-21x21-v0</b></td>
      <td><b>RandomMaze-31x31-v0</b></td>
      <td><b>RandomMaze-101x101-v0</b></td>
    </tr>
    <tr>
      <td><img src="/static/RandomMaze-11x11-v0.gif" alt="RandomMAze-11x11-v0.gif" width="200"/></td>
      <td><img src="/static/RandomMaze-21x21-v0.gif" alt="RandomMAze-21x21-v0.gif" width="200"/></td>
      <td><img src="/static/RandomMaze-31x31-v0.gif" alt="RandomMAze-11x11-v0.gif" width="200"/></td>
      <td><img src="/static/RandomMaze-101x101-v0.gif" alt="RandomMAze-21x21-v0.gif" width="200"/></td>
    </tr>
  </table>
</div>

## Usage:

1. Basics:

    ```python
    import gymnasium as gym
    
    env = gym.make("maze_world:RandomMaze-11x11-v0", render_mode="human")
    terminated, truncated = False, False
    observation, info = env.reset(seed=0, options={})
    episode_score = 0.
   
    while not (terminated or truncated):
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)
        episode_score += reward
   
    env.close()
    ```
2. Creating custom size random maze:

  ```python
    import gymnasium as gym
    import maze_world
    
    gym.envs.register(
        id='RandomMaze-7x7-v0',
        entry_point='maze_world.envs:RandomMazeEnv',
        max_episode_steps=200,
        kwargs={
            "maze_width": 7,
            "maze_height": 7,
            "maze_complexity": 1,
            "maze_density": 1,
        },
    )
    env = gym.make("maze_world:RandomMaze-7x7-v0")
  ```
3. Creating maze with pre-specified map:

  ```python
    import gymnasium as gym

    def _generate_maze_fn():
        # This function would be called on every reset
    
        maze_map = [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
                    [1, 0, 0, 0, 0, 0, 0, 0, 0, 1]
                    [1, 0, 0, 0, 0, 0, 0, 0, 0, 1]
                    [1, 1, 1, 1, 1, 1, 1, 0, 0, 1]
                    [1, 1, 1, 1, 1, 1, 1, 0, 0, 1]
                    [1, 1, 1, 1, 1, 1, 1, 0, 0, 1]
                    [1, 0, 0, 0, 0, 0, 0, 0, 0, 1]
                    [1, 0, 0, 0, 0, 0, 0, 0, 0, 1]
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]
        agent_loc = [1, 1]
        target_loc = [1, 7]
        return maze_map, agent_loc, target_loc


    gym.envs.register(
        id='UMaze-v0',
        entry_point='maze_world.envs:MazeEnv',
        max_episode_steps=200,
        kwargs={
            "generate_maze_fn": _generate_maze_fn,
            "maze_height": 9,
            "maze_wwidth": 9,
        },
    )
    env = gym.make("maze_world:UMaze-v0")
  ```
## Testing:

- Install: ```pip install -e ".[test]" ```
- Run: ```pytest```

## Development:

If you would like to develop it further; begin by installing following:

```pip install -e ".[develop]" ```