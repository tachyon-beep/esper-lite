"""Main script for diverse data generation."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from esper.datagen.orchestrator import GenerationOrchestrator, GenerationPlan
from esper.datagen.architectures import create_model
from esper.datagen.policies import BehaviorPolicy
from esper.datagen.configs import RewardComponents, StepMetadata, ActionProbabilities
from esper.datagen.health import DatasetHealthCheck


def generate_episode(
    plan: GenerationPlan,
    device: str = "cuda",
    verbose: bool = False,
) -> dict:
    """Generate a single episode according to plan.

    This is a simplified version - the full implementation will integrate
    with the existing simic_overnight.py infrastructure.

    Returns:
        Episode dict with full metadata
    """
    env_config = plan.get_env_config()
    policy_config = plan.get_policy_config()

    if verbose:
        print(f"  Generating {plan.episode_id}")
        print(f"    Env: {env_config.architecture}, LR={env_config.learning_rate}")
        print(f"    Policy: {policy_config.policy_id}, ε={policy_config.epsilon}")

    # Create model
    model = create_model(env_config.architecture, num_classes=10)
    model = model.to(device)

    # Create policy
    policy = BehaviorPolicy(policy_config)

    # TODO: Full episode generation with training loop
    # For now, return a skeleton episode
    episode = {
        "episode_id": plan.episode_id,
        "schema_version": "2.0.0",
        "behavior_policy": policy_config.to_dict(),
        "environment": env_config.to_dict(),
        "random_seed": plan.random_seed,
        "decisions": [],
        "final_accuracy": 0.0,
        "best_accuracy": 0.0,
        "total_return": 0.0,
        "episode_length": 0,
        "termination_reason": "not_implemented",
    }

    return episode


def main():
    parser = argparse.ArgumentParser(description="Generate diverse offline RL data")
    parser.add_argument("--output-dir", default="data/datagen_v3", help="Output directory")
    parser.add_argument("--episodes-per-combo", type=int, default=10, help="Episodes per combination")
    parser.add_argument("--batch-size", type=int, default=10, help="Batch size for generation")
    parser.add_argument("--device", default="cuda:0", help="Device to use")
    parser.add_argument("--dry-run", action="store_true", help="Print plan without generating")
    parser.add_argument("--health-check", action="store_true", help="Run health checks on existing data")
    args = parser.parse_args()

    print("=" * 60)
    print("Diverse Data Generation System")
    print("=" * 60)

    # Create orchestrator
    orch = GenerationOrchestrator(
        output_dir=args.output_dir,
        episodes_per_combo=args.episodes_per_combo,
    )

    summary = orch.get_progress_summary()
    print(f"Total planned: {summary['total_planned']}")
    print(f"Completed: {summary['completed']}")
    print(f"Remaining: {summary['remaining']}")
    print()

    if args.dry_run:
        print("DRY RUN - showing first 20 planned episodes:")
        for plan in orch.plans[:20]:
            print(f"  {plan.episode_id}")
        print(f"  ... and {len(orch.plans) - 20} more")
        return

    if args.health_check:
        print("Running health checks on existing data...")
        episodes = _load_existing_episodes(args.output_dir)
        if episodes:
            checker = DatasetHealthCheck()
            results = checker.run_all(episodes)
            checker.print_report(results)
        else:
            print("No episodes found to check")
        return

    # Generation loop
    print(f"Starting generation (batch_size={args.batch_size})...")
    start_time = time.time()

    while True:
        batch = orch.get_next_batch(batch_size=args.batch_size)
        if not batch:
            break

        for plan in batch:
            episode = generate_episode(plan, device=args.device, verbose=True)

            # Save episode
            episode_path = Path(args.output_dir) / f"{plan.episode_id}.json"
            with open(episode_path, "w") as f:
                json.dump(episode, f, indent=2)

            orch.mark_complete(plan.episode_id)

        # Progress update
        summary = orch.get_progress_summary()
        elapsed = time.time() - start_time
        print(f"\nProgress: {summary['completed']}/{summary['total_planned']} "
              f"({summary['progress_pct']:.1f}%) - {elapsed:.0f}s elapsed")

    print("\nGeneration complete!")
    print(f"Total time: {time.time() - start_time:.1f}s")


def _load_existing_episodes(output_dir: str) -> list[dict]:
    """Load existing episodes from output directory."""
    episodes = []
    for path in Path(output_dir).glob("*.json"):
        if path.name.startswith("."):
            continue
        with open(path) as f:
            episodes.append(json.load(f))
    return episodes


if __name__ == "__main__":
    main()
