import gymnasium as gym

# Note: Know more about rendering from https://younis.dev/blog/render-api/

env = gym.make("maze_world:RandomMaze-11x11-v0", render_mode="human")
terminated, truncated = False, False
observation, info = env.reset(seed=1, options={})
episode_score = 0.0
while not (terminated or truncated):
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)
    episode_score += reward
env.close()
