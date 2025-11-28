"""Esper Evaluate Script - Head-to-head comparison.

Usage:
    python -m esper.scripts.evaluate --policy models/ppo.pt --episodes 10
"""

from __future__ import annotations

import argparse


def main():
    parser = argparse.ArgumentParser(description="Evaluate policy vs heuristic")
    parser.add_argument("--policy", type=str, required=True)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--device", type=str, default="cuda:0")

    args = parser.parse_args()

    print(f"Evaluating {args.policy} for {args.episodes} episodes")
    # TODO: Implement evaluation loop using simic.ppo head-to-head functions


if __name__ == "__main__":
    main()
