#!/usr/bin/env python3
"""Training CLI for Simic RL algorithms.

Usage:
    # Train PPO
    PYTHONPATH=src python -m esper.scripts.train ppo --episodes 100 --device cuda:0

    # Train IQL
    PYTHONPATH=src python -m esper.scripts.train iql --pack data/pack.json --epochs 100

    # Vectorized PPO
    PYTHONPATH=src python -m esper.scripts.train ppo --vectorized --n-envs 4 --devices cuda:0 cuda:1
"""

import argparse


def main():
    parser = argparse.ArgumentParser(description="Train Simic RL agents")
    subparsers = parser.add_subparsers(dest="algorithm", required=True)

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
    ppo_parser.add_argument("--entropy-anneal-episodes", type=int, default=0,
        help="Episodes over which to anneal entropy (0=fixed, no annealing)")
    ppo_parser.add_argument("--gamma", type=float, default=0.99)
    ppo_parser.add_argument("--save", help="Path to save model")
    ppo_parser.add_argument("--device", default="cuda:0")
    ppo_parser.add_argument("--vectorized", action="store_true")
    ppo_parser.add_argument("--n-envs", type=int, default=4)
    ppo_parser.add_argument("--devices", nargs="+")
    ppo_parser.add_argument("--no-telemetry", action="store_true", help="Disable telemetry features (27-dim instead of 37-dim)")

    # IQL subcommand
    iql_parser = subparsers.add_parser("iql", help="Train IQL agent")
    iql_parser.add_argument("--pack", required=True, help="Path to data pack")
    iql_parser.add_argument("--epochs", type=int, default=100)
    iql_parser.add_argument("--steps-per-epoch", type=int, default=1000)
    iql_parser.add_argument("--batch-size", type=int, default=256)
    iql_parser.add_argument("--gamma", type=float, default=0.99)
    iql_parser.add_argument("--tau", type=float, default=0.7)
    iql_parser.add_argument("--beta", type=float, default=3.0)
    iql_parser.add_argument("--lr", type=float, default=3e-4)
    iql_parser.add_argument("--cql-alpha", type=float, default=0.0)
    iql_parser.add_argument("--reward-shaping", action="store_true")
    iql_parser.add_argument("--save", help="Path to save model")
    iql_parser.add_argument("--device", default="cpu")

    # Comparison subcommand
    cmp_parser = subparsers.add_parser("compare", help="Compare IQL vs heuristic")
    cmp_parser.add_argument("--model", required=True, help="Path to IQL model")
    cmp_parser.add_argument("--mode", choices=["live", "head-to-head"], default="head-to-head")
    cmp_parser.add_argument("--episodes", type=int, default=5)
    cmp_parser.add_argument("--max-epochs", type=int, default=25)
    cmp_parser.add_argument("--device", default="cpu")

    args = parser.parse_args()

    if args.algorithm == "ppo":
        use_telemetry = not args.no_telemetry
        if args.vectorized:
            from esper.simic.vectorized import train_ppo_vectorized
            train_ppo_vectorized(
                n_episodes=args.episodes,
                n_envs=args.n_envs,
                max_epochs=args.max_epochs,
                device=args.device,
                devices=args.devices,
                use_telemetry=use_telemetry,
                lr=args.lr,
                clip_ratio=args.clip_ratio,
                entropy_coef=args.entropy_coef,
                entropy_coef_start=args.entropy_coef_start,
                entropy_coef_end=args.entropy_coef_end,
                entropy_anneal_episodes=args.entropy_anneal_episodes,
                gamma=args.gamma,
                save_path=args.save,
            )
        else:
            from esper.simic.training import train_ppo
            train_ppo(
                n_episodes=args.episodes,
                max_epochs=args.max_epochs,
                update_every=args.update_every,
                device=args.device,
                use_telemetry=use_telemetry,
                lr=args.lr,
                clip_ratio=args.clip_ratio,
                entropy_coef=args.entropy_coef,
                gamma=args.gamma,
                save_path=args.save,
            )

    elif args.algorithm == "iql":
        from esper.simic.training import train_iql
        train_iql(
            pack_path=args.pack,
            epochs=args.epochs,
            steps_per_epoch=args.steps_per_epoch,
            batch_size=args.batch_size,
            gamma=args.gamma,
            tau=args.tau,
            beta=args.beta,
            lr=args.lr,
            cql_alpha=args.cql_alpha,
            use_reward_shaping=args.reward_shaping,
            device=args.device,
            save_path=args.save,
        )

    elif args.algorithm == "compare":
        from esper.simic.comparison import live_comparison, head_to_head_comparison
        if args.mode == "live":
            live_comparison(
                model_path=args.model,
                n_episodes=args.episodes,
                max_epochs=args.max_epochs,
                device=args.device,
            )
        else:
            head_to_head_comparison(
                model_path=args.model,
                n_episodes=args.episodes,
                max_epochs=args.max_epochs,
                device=args.device,
            )


if __name__ == "__main__":
    main()
