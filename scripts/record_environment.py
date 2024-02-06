"""
Usage: python scripts/record_environment.py --env RandomMaze-11x11-v0
"""

import argparse
import os

import gymnasium as gym
import imageio


def parse_arguments():
    parser = argparse.ArgumentParser(description="Record the environment.")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="static/",
        help="Output directory with GIF record.",
    )
    parser.add_argument(
        "--env", type=str, help="Name of recorded environment.", required=True
    )
    parser.add_argument(
        "--frames", type=int, default=100, help="Number of frames in GIF record."
    )
    parser.add_argument("--fps", type=int, default=21, help="Frame per second.")
    return parser.parse_args()


def main(args):
    env = gym.make("maze_world:" + args.env, render_mode="rgb_array")
    pics = []

    env.reset()
    step_count = 0
    done = False
    while not done and step_count < 50:
        step_count += 1
        pics.append(env.render())

        _, _, terminated, truncated, _ = env.step(env.action_space.sample())
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
