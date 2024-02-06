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

## Usage:

```python
import gymnasium as gym

env = gym.make("maze-world:RandomMaze-11x11-v0", render_mode="human")
terminated, truncated = False, False
observation, info = env.reset(seed=0, options={})
episode_score = 0.
while not (terminated or truncated):
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)
    episode_score += reward
env.close()

```

## Testing:

- Install: ```pip install -e ".[test]" ```
- Run: ```pytest```

## Environments Zoo!

<div style="text-align:center;">
  <table>
    <tr>
      <td><b>RandomMaze-11x11-v0</b></td>
      <td><b>RandomMaze-21x21-v0</b></td>
    </tr>
    <tr>
      <td><img src="/static/RandomMaze-11x11-v0.gif" alt="RandomMAze-11x11-v0.gif" width="200"/></td>
      <td><img src="/static/RandomMaze-21x21-v0.gif" alt="RandomMAze-21x21-v0.gif" width="200"/></td>
    </tr>
    <tr>
      <td><b>RandomMaze-31x31-v0</b></td>
      <td><b>RandomMaze-101x101-v0</b></td>
    </tr>
    <tr>
      <td><img src="/static/RandomMaze-31x31-v0.gif" alt="RandomMAze-11x11-v0.gif" width="200"/></td>
      <td><img src="/static/RandomMaze-101x101-v0.gif" alt="RandomMAze-21x21-v0.gif" width="200"/></td>
    </tr>
  </table>
</div>
