"""Esper Training Script - PPO entry point.

Usage:
    python -m esper.scripts.train --episodes 100 --device cuda:0
    python -m esper.scripts.train --episodes 100 --vectorized --n-envs 6
"""

from __future__ import annotations

import argparse
import sys


def main():
    parser = argparse.ArgumentParser(description="Train Esper PPO agent")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--max-epochs", type=int, default=25)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--vectorized", action="store_true")
    parser.add_argument("--n-envs", type=int, default=4)
    parser.add_argument("--save", type=str, default=None)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--entropy-coef", type=float, default=0.1)

    args = parser.parse_args()

    # Import here to avoid slow startup for --help
    from esper.simic.ppo import train_ppo_vectorized

    if args.vectorized:
        train_ppo_vectorized(
            n_episodes=args.episodes,
            max_epochs=args.max_epochs,
            n_envs=args.n_envs,
            device=args.device,
            save_path=args.save,
            lr=args.lr,
            entropy_coef=args.entropy_coef,
        )
    else:
        # For sequential, use vectorized with 1 env
        train_ppo_vectorized(
            n_episodes=args.episodes,
            max_epochs=args.max_epochs,
            n_envs=1,
            device=args.device,
            save_path=args.save,
            lr=args.lr,
            entropy_coef=args.entropy_coef,
        )


if __name__ == "__main__":
    main()
