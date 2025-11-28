"""Esper Generate Script - Heuristic data generation.

Usage:
    python -m esper.scripts.generate --episodes 1000 --output data/episodes/
"""

from __future__ import annotations

import argparse


def main():
    parser = argparse.ArgumentParser(description="Generate training data with heuristic")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--max-epochs", type=int, default=25)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda:0")

    args = parser.parse_args()

    from esper.tamiyo import HeuristicTamiyo
    from esper.simic.episodes import EpisodeCollector

    print(f"Generating {args.episodes} episodes to {args.output}")
    # TODO: Implement generation loop using existing datagen logic


if __name__ == "__main__":
    main()
