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

from esper.nissa import get_hub, ConsoleOutput, FileOutput, DirectoryOutput


def main():
    parser = argparse.ArgumentParser(description="Train Simic RL agents")

    # Parent parser for shared telemetry options
    telemetry_parent = argparse.ArgumentParser(add_help=False)
    telemetry_parent.add_argument("--telemetry-file", type=str, default=None,
                                  help="Save Nissa telemetry to JSONL file")
    telemetry_parent.add_argument("--telemetry-dir", type=str, default=None,
                                  help="Save Nissa telemetry to timestamped folder in this directory")
    telemetry_parent.add_argument(
        "--telemetry-level",
        type=str,
        choices=["off", "minimal", "normal", "debug"],
        default="normal",
        help="Telemetry verbosity level (default: normal)",
    )

    subparsers = parser.add_subparsers(dest="algorithm", required=True)

    # Heuristic subcommand
    heur_parser = subparsers.add_parser("heuristic", help="Train with heuristic policy (h-esper)",
                                        parents=[telemetry_parent])
    heur_parser.add_argument("--episodes", type=int, default=1)
    heur_parser.add_argument("--max-epochs", type=int, default=75)
    heur_parser.add_argument("--max-batches", type=int, default=50, help="Batches per epoch (None=all)")
    heur_parser.add_argument("--task", default="cifar10",
                              choices=["cifar10", "cifar10_deep", "tinystories"])
    heur_parser.add_argument("--device", default="cuda:0")
    heur_parser.add_argument("--seed", type=int, default=42)
    heur_parser.add_argument("--slots", nargs="+", default=["mid"],
        choices=["early", "mid", "late"],
        help="Seed slots to enable (default: mid)")
    heur_parser.add_argument("--max-seeds", type=int, default=None,
        help="Maximum total seeds across all slots (default: unlimited)")
    heur_parser.add_argument("--max-seeds-per-slot", type=int, default=None,
        help="Maximum seeds per slot (default: unlimited)")

    # PPO subcommand
    ppo_parser = subparsers.add_parser("ppo", help="Train PPO agent",
                                       parents=[telemetry_parent])
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
    ppo_parser.add_argument("--task", default="cifar10",
                             choices=["cifar10", "cifar10_deep", "tinystories"],
                             help="Task preset")
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
    ppo_parser.add_argument("--gpu-preload", action="store_true",
        help="Preload dataset to GPU for 8x faster data loading (CIFAR-10 only, uses ~0.75GB VRAM)")
    ppo_parser.add_argument("--slots", nargs="+", default=["mid"],
        choices=["early", "mid", "late"],
        help="Seed slots to enable (default: mid)")
    ppo_parser.add_argument("--max-seeds", type=int, default=None,
        help="Maximum total seeds across all slots (default: unlimited)")
    ppo_parser.add_argument("--max-seeds-per-slot", type=int, default=None,
        help="Maximum seeds per slot (default: unlimited)")

    args = parser.parse_args()

    # Create TelemetryConfig from CLI argument
    from esper.simic.telemetry_config import TelemetryConfig, TelemetryLevel

    level_map = {
        "off": TelemetryLevel.OFF,
        "minimal": TelemetryLevel.MINIMAL,
        "normal": TelemetryLevel.NORMAL,
        "debug": TelemetryLevel.DEBUG,
    }
    telemetry_config = TelemetryConfig(level=level_map[args.telemetry_level])

    # Map telemetry level to console severity filter
    # debug level -> show debug events, normal/minimal -> show info+ only
    console_min_severity = "debug" if args.telemetry_level == "debug" else "info"

    # Wire Nissa console telemetry to the global hub so all
    # lifecycle events (including fossilization) are visible
    # alongside training logs.
    hub = get_hub()
    hub.add_backend(ConsoleOutput(min_severity=console_min_severity))

    # Add file output if requested
    file_backend = None
    if args.telemetry_file:
        file_backend = FileOutput(args.telemetry_file)
        hub.add_backend(file_backend)
        print(f"Telemetry will be saved to: {args.telemetry_file}")

    # Add directory output if requested
    dir_backend = None
    if args.telemetry_dir:
        dir_backend = DirectoryOutput(args.telemetry_dir)
        hub.add_backend(dir_backend)
        print(f"Telemetry will be saved to: {dir_backend.output_dir}")

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
                gpu_preload=args.gpu_preload,
                telemetry_config=telemetry_config,
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
                telemetry_config=telemetry_config,
            )

if __name__ == "__main__":
    main()
