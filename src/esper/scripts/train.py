#!/usr/bin/env python3
"""Training CLI for Simic RL algorithms.

Usage:
    # Train PPO (vectorized by default)
    PYTHONPATH=src python -m esper.scripts.train ppo --episodes 100 --n-envs 4

    # Multi-GPU PPO
    PYTHONPATH=src python -m esper.scripts.train ppo --n-envs 4 --devices cuda:0 cuda:1

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
    telemetry_parent.add_argument(
        "--dashboard",
        action="store_true",
        help="Enable real-time WebSocket dashboard (requires: pip install esper-lite[dashboard])",
    )
    telemetry_parent.add_argument(
        "--dashboard-port",
        type=int,
        default=8000,
        help="Dashboard server port (default: 8000)",
    )
    telemetry_parent.add_argument(
        "--tui",
        action="store_true",
        help="Enable Rich terminal UI for live training monitoring (replaces console output)",
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
    ppo_parser.add_argument("--n-envs", type=int, default=4)
    ppo_parser.add_argument("--devices", nargs="+")
    ppo_parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="DataLoader workers per environment (overrides task default)",
    )
    ppo_parser.add_argument(
        "--no-telemetry",
        action="store_true",
        help="Disable telemetry features (50-dim instead of 60-dim)",
    )
    ppo_parser.add_argument("--gpu-preload", action="store_true",
        help="Preload dataset to GPU for 8x faster data loading (CIFAR-10 only, uses ~0.75GB VRAM)")
    ppo_parser.add_argument("--slots", nargs="+", default=["mid"],
        choices=["early", "mid", "late"],
        help="Seed slots to enable (default: mid)")
    ppo_parser.add_argument("--max-seeds", type=int, default=None,
        help="Maximum total seeds across all slots (default: unlimited)")
    ppo_parser.add_argument("--max-seeds-per-slot", type=int, default=None,
        help="Maximum seeds per slot (default: unlimited)")
    ppo_parser.add_argument(
        "--reward-mode",
        type=str,
        choices=["shaped", "sparse", "minimal"],
        default="shaped",
        help="Reward mode: shaped (dense, default), sparse (terminal-only), minimal (sparse + early-cull)"
    )
    ppo_parser.add_argument(
        "--param-budget",
        type=int,
        default=500_000,
        help="Parameter budget for sparse reward efficiency calculation (default: 500000)"
    )
    ppo_parser.add_argument(
        "--param-penalty",
        type=float,
        default=0.1,
        help="Parameter penalty weight in sparse reward (default: 0.1)"
    )
    ppo_parser.add_argument(
        "--sparse-scale",
        type=float,
        default=1.0,
        help="Reward scale for sparse mode (DRL Expert: try 2.0-3.0 if learning fails)"
    )

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

    # Wire Nissa telemetry to the global hub so all
    # lifecycle events (including fossilization) are visible
    # alongside training logs.
    hub = get_hub()

    # Use TUI or Console output based on --tui flag
    tui_backend = None
    if args.tui:
        from esper.karn import TUIOutput
        tui_backend = TUIOutput()
        hub.add_backend(tui_backend)
        # TUI auto-starts on first event, no startup message needed
    else:
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

    # Add WebSocket dashboard if requested
    dashboard_backend = None
    if args.dashboard:
        try:
            from esper.karn import DashboardServer

            # DashboardServer provides integrated HTTP + WebSocket:
            # - Serves dashboard HTML at http://localhost:PORT/
            # - WebSocket telemetry at ws://localhost:PORT/ws
            # - Queues events from sync training loop
            dashboard_backend = DashboardServer(port=args.dashboard_port)
            dashboard_backend.start()
            hub.add_backend(dashboard_backend)

            # Get all network interfaces for dashboard URLs
            def get_network_interfaces():
                """Get all network interface IPs."""
                interfaces = ["localhost", "127.0.0.1"]
                try:
                    import socket
                    hostname = socket.gethostname()
                    # Get all IPs for this host
                    for info in socket.getaddrinfo(hostname, None, socket.AF_INET):
                        ip = info[4][0]
                        if ip not in interfaces and not ip.startswith("127."):
                            interfaces.append(ip)
                    # Also try to get the primary LAN IP
                    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                    try:
                        s.connect(("8.8.8.8", 80))
                        lan_ip = s.getsockname()[0]
                        if lan_ip not in interfaces:
                            interfaces.insert(2, lan_ip)  # Put after localhost
                    except Exception:
                        pass
                    finally:
                        s.close()
                except Exception:
                    pass
                return interfaces

            # Print clickable dashboard links (OSC 8 hyperlinks for modern terminals)
            interfaces = get_network_interfaces()
            print()
            print(f"  \033[1;36m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\033[0m")
            print(f"  \033[1m🔬 Live Dashboard\033[0m (listening on all interfaces)")
            for iface in interfaces:
                url = f"http://{iface}:{args.dashboard_port}"
                # OSC 8 format: \033]8;;URL\033\\TEXT\033]8;;\033\\
                hyperlink = f"\033]8;;{url}\033\\{url}\033]8;;\033\\"
                label = " (local)" if iface in ("localhost", "127.0.0.1") else " (LAN)" if not iface.startswith("127.") else ""
                print(f"     → {hyperlink}\033[90m{label}\033[0m")
            print(f"  \033[1;36m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\033[0m")
            print()
        except ImportError:
            print("Warning: Dashboard dependencies not installed.")
            print("  Install with: pip install esper-lite[dashboard]")

    # Add Karn research telemetry collector
    # KarnCollector captures events into typed store for research analytics
    from esper.karn import get_collector
    karn_collector = get_collector()
    hub.add_backend(karn_collector)

    try:
        if args.algorithm == "heuristic":
            from esper.simic.training import train_heuristic
            train_heuristic(
                n_episodes=args.episodes,
                max_epochs=args.max_epochs,
                max_batches=args.max_batches if args.max_batches > 0 else None,
                device=args.device,
                task=args.task,
                seed=args.seed,
                slots=args.slots,
            )

        elif args.algorithm == "ppo":
            use_telemetry = not args.no_telemetry
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
                slots=args.slots,
                max_seeds=args.max_seeds,
                max_seeds_per_slot=args.max_seeds_per_slot,
                reward_mode=args.reward_mode,
                param_budget=args.param_budget,
                param_penalty_weight=args.param_penalty,
                sparse_reward_scale=args.sparse_scale,
            )
    finally:
        # Clean up TUI backend if used
        if tui_backend is not None:
            tui_backend.close()
        # Close all hub backends
        hub.close()

if __name__ == "__main__":
    main()
