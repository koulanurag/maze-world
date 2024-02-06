# maze-world
It's a collection of multi agent environments based on OpenAI gym. Also, you can use [**minimal-marl**](https://github.com/koulanurag/minimal-marl) to warm-start training of agents.

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

env = gym.make("maze-world:RandomMaze-10x10-v0", render_mode="human")
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


