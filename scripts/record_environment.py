"""
Usage: python scripts/record_environment.py --env RandomMaze-11x11-v0
"""

import argparse
import os

import gymnasium as gym
import imageio
from maze_world.utils import maze_dijkstra_solver

def parse_arguments():
    parser = argparse.ArgumentParser(description="Record the environment.")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="docs/source/_static/gifs/",
        help="Output directory with GIF record.",
    )
    parser.add_argument(
        "--env", type=str, help="Name of recorded environment.", required=True
    )
    parser.add_argument(
        "--frames", type=int, default=500, help="Number of frames in GIF record."
    )
    parser.add_argument("--fps", type=int, default=21, help="Frame per second.")
    return parser.parse_args()


def main(args):
    env = gym.make("maze_world:" + args.env, render_mode="rgb_array")
    pics = []

    observation, info = env.reset()
    step_count = 0

    step_actions = maze_dijkstra_solver(
        env.unwrapped.maze_map.astype(bool),
        env.unwrapped._action_to_direction.values(),
        info["agent"],
        info["target"],
    )

    for action in step_actions:
        step_count += 1
        pics.append(env.render())

        _, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

    print("Environment finished.")
    imageio.mimwrite(
        os.path.join(args.output_dir, args.env + ".gif"),
        pics[: args.frames],
        fps=args.fps,
        loop=0,
    )


if __name__ == "__main__":
    args = parse_arguments()
    main(args)
