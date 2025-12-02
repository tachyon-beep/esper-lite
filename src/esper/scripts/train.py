#!/usr/bin/env python3
"""Training CLI for Simic RL algorithms.

Usage:
    # Train PPO
    PYTHONPATH=src python -m esper.scripts.train ppo --episodes 100 --device cuda:0

    # Vectorized PPO
    PYTHONPATH=src python -m esper.scripts.train ppo --vectorized --n-envs 4 --devices cuda:0 cuda:1

    # Heuristic (h-esper)
    PYTHONPATH=src python -m esper.scripts.train heuristic --max-epochs 75 --max-batches 50
"""

import argparse

from esper.nissa import get_hub, ConsoleOutput, FileOutput


def main():
    parser = argparse.ArgumentParser(description="Train Simic RL agents")

    # Global options (apply to all subcommands)
    parser.add_argument("--telemetry-file", type=str, default=None,
                        help="Save Nissa telemetry to JSONL file")

    subparsers = parser.add_subparsers(dest="algorithm", required=True)

    # Heuristic subcommand
    heur_parser = subparsers.add_parser("heuristic", help="Train with heuristic policy (h-esper)")
    heur_parser.add_argument("--episodes", type=int, default=1)
    heur_parser.add_argument("--max-epochs", type=int, default=75)
    heur_parser.add_argument("--max-batches", type=int, default=50, help="Batches per epoch (None=all)")
    heur_parser.add_argument("--task", default="cifar10")
    heur_parser.add_argument("--device", default="cuda:0")
    heur_parser.add_argument("--seed", type=int, default=42)

    # PPO subcommand
    ppo_parser = subparsers.add_parser("ppo", help="Train PPO agent")
    ppo_parser.add_argument("--episodes", type=int, default=100)
    ppo_parser.add_argument("--max-epochs", type=int, default=75)  # Increased from 25 to allow seed fossilization
    ppo_parser.add_argument("--update-every", type=int, default=5)
    ppo_parser.add_argument("--lr", type=float, default=3e-4)
    ppo_parser.add_argument("--clip-ratio", type=float, default=0.2)
    ppo_parser.add_argument("--entropy-coef", type=float, default=0.05)  # Increased from 0.01 to prevent premature convergence
    ppo_parser.add_argument("--entropy-coef-start", type=float, default=None,
        help="Initial entropy coefficient (default: use --entropy-coef)")
    ppo_parser.add_argument("--entropy-coef-end", type=float, default=None,
        help="Final entropy coefficient (default: use --entropy-coef)")
    ppo_parser.add_argument("--entropy-coef-min", type=float, default=0.1,
        help="Minimum entropy coefficient floor to prevent exploration collapse (default: 0.1)")
    ppo_parser.add_argument("--entropy-anneal-episodes", type=int, default=0,
        help="Episodes over which to anneal entropy (0=fixed, no annealing)")
    ppo_parser.add_argument("--gamma", type=float, default=0.99)
    ppo_parser.add_argument("--task", default="cifar10", help="Task preset (cifar10 or tinystories)")
    ppo_parser.add_argument("--save", help="Path to save model")
    ppo_parser.add_argument("--resume", help="Path to checkpoint to resume from")
    ppo_parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    ppo_parser.add_argument("--device", default="cuda:0")
    ppo_parser.add_argument("--vectorized", action="store_true")
    ppo_parser.add_argument("--n-envs", type=int, default=4)
    ppo_parser.add_argument("--devices", nargs="+")
    ppo_parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="DataLoader workers per environment (overrides task default)",
    )
    ppo_parser.add_argument("--no-telemetry", action="store_true", help="Disable telemetry features (27-dim instead of 37-dim)")

    args = parser.parse_args()

    # Wire Nissa console telemetry to the global hub so all
    # lifecycle events (including fossilization) are visible
    # alongside training logs.
    hub = get_hub()
    hub.add_backend(ConsoleOutput())

    # Add file output if requested
    file_backend = None
    if args.telemetry_file:
        file_backend = FileOutput(args.telemetry_file)
        hub.add_backend(file_backend)
        print(f"Telemetry will be saved to: {args.telemetry_file}")

    if args.algorithm == "heuristic":
        from esper.simic.training import train_heuristic
        train_heuristic(
            n_episodes=args.episodes,
            max_epochs=args.max_epochs,
            max_batches=args.max_batches if args.max_batches > 0 else None,
            device=args.device,
            task=args.task,
            seed=args.seed,
        )

    elif args.algorithm == "ppo":
        use_telemetry = not args.no_telemetry
        if args.vectorized:
            from esper.simic.vectorized import train_ppo_vectorized
            train_ppo_vectorized(
                n_episodes=args.episodes,
                n_envs=args.n_envs,
                max_epochs=args.max_epochs,
                device=args.device,
                devices=args.devices,
                task=args.task,
                use_telemetry=use_telemetry,
                lr=args.lr,
                clip_ratio=args.clip_ratio,
                entropy_coef=args.entropy_coef,
                entropy_coef_start=args.entropy_coef_start,
                entropy_coef_end=args.entropy_coef_end,
                entropy_coef_min=args.entropy_coef_min,
                entropy_anneal_episodes=args.entropy_anneal_episodes,
                gamma=args.gamma,
                save_path=args.save,
                resume_path=args.resume,
                seed=args.seed,
                num_workers=args.num_workers,
            )
        else:
            from esper.simic.training import train_ppo
            train_ppo(
                n_episodes=args.episodes,
                max_epochs=args.max_epochs,
                update_every=args.update_every,
                device=args.device,
                task=args.task,
                use_telemetry=use_telemetry,
                lr=args.lr,
                clip_ratio=args.clip_ratio,
                entropy_coef=args.entropy_coef,
                entropy_coef_start=args.entropy_coef_start,
                entropy_coef_end=args.entropy_coef_end,
                entropy_coef_min=args.entropy_coef_min,
                entropy_anneal_episodes=args.entropy_anneal_episodes,
                gamma=args.gamma,
                save_path=args.save,
                resume_path=args.resume,
                seed=args.seed,
            )

if __name__ == "__main__":
    main()
