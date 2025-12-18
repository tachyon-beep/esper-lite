#!/usr/bin/env python3
"""Training CLI for Simic RL algorithms."""

import argparse

from esper.nissa import ConsoleOutput, DirectoryOutput, FileOutput, get_hub
from esper.simic.training import TrainingConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train Simic RL agents")

    telemetry_parent = argparse.ArgumentParser(add_help=False)
    telemetry_parent.add_argument(
        "--telemetry-file",
        type=str,
        default=None,
        help="Save Nissa telemetry to JSONL file",
    )
    telemetry_parent.add_argument(
        "--telemetry-dir",
        type=str,
        default=None,
        help="Save Nissa telemetry to timestamped folder in this directory",
    )
    telemetry_parent.add_argument(
        "--telemetry-level",
        type=str,
        choices=["off", "minimal", "normal", "debug"],
        default="normal",
        help="Telemetry verbosity level (default: normal)",
    )
    telemetry_parent.add_argument(
        "--telemetry-lifecycle-only",
        action="store_true",
        help="Keep lightweight seed lifecycle telemetry even when ops telemetry is disabled",
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
        "--no-tui",
        action="store_true",
        help="Disable Rich terminal UI (uses console output instead)",
    )
    telemetry_parent.add_argument(
        "--tui-layout",
        type=str,
        choices=["compact", "standard", "wide", "auto"],
        default="auto",
        help="TUI layout mode: compact (< 100 cols), standard (100-150), wide (150+), auto (detect)",
    )
    telemetry_parent.add_argument(
        "--export-karn",
        type=str,
        default=None,
        metavar="PATH",
        help="Export Karn telemetry store to JSONL file after training",
    )
    telemetry_parent.add_argument(
        "--overwatch",
        action="store_true",
        help="Launch Overwatch TUI for real-time monitoring (replaces Rich TUI)",
    )

    subparsers = parser.add_subparsers(dest="algorithm", required=True)

    heur_parser = subparsers.add_parser(
        "heuristic",
        help="Train with heuristic policy (h-esper)",
        parents=[telemetry_parent],
    )
    heur_parser.add_argument("--episodes", type=int, default=1)
    heur_parser.add_argument("--max-epochs", type=int, default=75)
    heur_parser.add_argument("--max-batches", type=int, default=50, help="Batches per epoch (None=all)")
    heur_parser.add_argument("--task", default="cifar10",
                              choices=["cifar10", "cifar10_deep", "cifar10_blind", "tinystories"])
    heur_parser.add_argument("--device", default="cuda:0")
    heur_parser.add_argument("--seed", type=int, default=42)
    heur_parser.add_argument(
        "--slots",
        nargs="+",
        default=["r0c0", "r0c1", "r0c2"],
        help="Canonical slot IDs to use (e.g., r0c0 r0c1 r0c2). Default: r0c0 r0c1 r0c2",
    )
    heur_parser.add_argument("--max-seeds", type=int, default=None,
        help="Maximum total seeds across all slots (default: unlimited)")
    heur_parser.add_argument(
        "--min-fossilize-improvement",
        type=float,
        default=None,
        metavar="PCT",
        help="Min improvement (%%) required to fossilize a seed (default: 0.5%%). "
             "Lower values risk reward hacking; higher values require stronger contribution.",
    )

    ppo_parser = subparsers.add_parser(
        "ppo",
        help="Train PPO agent",
        parents=[telemetry_parent],
    )
    ppo_parser.add_argument(
        "--preset",
        choices=["cifar10", "cifar10_deep", "cifar10_blind", "tinystories"],
        default="cifar10",
        help="TrainingConfig preset to use (hyperparameters + slots)",
    )
    ppo_parser.add_argument(
        "--config-json",
        type=str,
        default=None,
        help="Path to JSON config (overrides preset; fails on unknown keys)",
    )
    ppo_parser.add_argument(
        "--task",
        default="cifar10",
        choices=["cifar10", "cifar10_deep", "cifar10_blind", "tinystories"],
        help="Task preset",
    )
    ppo_parser.add_argument("--save", help="Path to save model")
    ppo_parser.add_argument("--resume", help="Path to checkpoint to resume from")
    ppo_parser.add_argument("--device", default="cuda:0")
    ppo_parser.add_argument("--devices", nargs="+")
    ppo_parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="DataLoader workers per environment (overrides task default)",
    )
    ppo_parser.add_argument(
        "--gpu-preload",
        action="store_true",
        help="Preload dataset to GPU for 8x faster data loading (CIFAR-10 only, uses ~0.75GB VRAM)",
    )
    ppo_parser.add_argument(
        "--amp",
        action="store_true",
        help="Enable automatic mixed precision (CUDA only)",
    )
    ppo_parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override seed (otherwise use config value)",
    )

    return parser


def validate_slots(slot_ids: list[str]) -> list[str]:
    """Validate CLI slot arguments use canonical format.

    Args:
        slot_ids: List of slot ID strings from CLI

    Returns:
        Validated slot IDs (unchanged if valid)

    Raises:
        ValueError: If any slot ID is invalid or uses legacy format
    """
    from esper.leyline.slot_id import validate_slot_id, parse_slot_id, SlotIdError

    for slot_id in slot_ids:
        if not validate_slot_id(slot_id):
            # Try to parse to get detailed error message
            try:
                parse_slot_id(slot_id)
            except SlotIdError as e:
                # Re-raise with CLI-specific guidance
                raise ValueError(str(e)) from e
            # If validate returns False but parse doesn't raise, it's an unknown issue
            raise ValueError(
                f"Invalid slot ID '{slot_id}'. Use canonical format: r0c0, r0c1, r0c2, etc."
            )
    return slot_ids


def main():
    parser = build_parser()
    args = parser.parse_args()

    # Create TelemetryConfig from CLI argument
    from esper.simic.telemetry import TelemetryConfig, TelemetryLevel

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

    # Determine UI mode
    import sys
    tui_backend = None
    is_tty = sys.stdout.isatty()
    use_overwatch = args.overwatch
    # Overwatch replaces Rich TUI (mutually exclusive)
    use_tui = not args.no_tui and is_tty and not use_overwatch

    if not is_tty and not args.no_tui:
        print("Non-TTY detected, using console output instead of TUI")

    if use_tui:
        from esper.karn import TUIOutput
        # Pass layout mode (None for auto-detect)
        layout = None if args.tui_layout == "auto" else args.tui_layout
        tui_backend = TUIOutput(force_layout=layout)
        hub.add_backend(tui_backend)
        # TUI auto-starts on first event
    elif not use_overwatch:
        # Only add console if NOT using either TUI or Overwatch
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

    # Setup Karn collector for stateful telemetry (P1-04)
    karn_collector = None
    if args.export_karn:
        from esper.karn import KarnCollector
        karn_collector = KarnCollector()
        hub.add_backend(karn_collector)

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
            print("  \033[1;36m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\033[0m")
            print("  \033[1m🔬 Live Dashboard\033[0m (listening on all interfaces)")
            for iface in interfaces:
                url = f"http://{iface}:{args.dashboard_port}"
                # OSC 8 format: \033]8;;URL\033\\TEXT\033]8;;\033\\
                hyperlink = f"\033]8;;{url}\033\\{url}\033]8;;\033\\"
                label = " (local)" if iface in ("localhost", "127.0.0.1") else " (LAN)" if not iface.startswith("127.") else ""
                print(f"     → {hyperlink}\033[90m{label}\033[0m")
            print("  \033[1;36m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\033[0m")
            print()
        except ImportError:
            print("Warning: Dashboard dependencies not installed.")
            print("  Install with: pip install esper-lite[dashboard]")

    # Setup Overwatch backend if requested
    overwatch_backend = None
    if use_overwatch:
        from esper.karn import OverwatchBackend
        from esper.leyline import DEFAULT_N_ENVS

        # Determine num_envs for Overwatch display
        if args.algorithm == "ppo":
            # For PPO, get from config
            if args.config_json:
                temp_config = TrainingConfig.from_json_path(args.config_json)
            else:
                if args.preset == "cifar10":
                    temp_config = TrainingConfig.for_cifar10()
                elif args.preset == "cifar10_deep":
                    temp_config = TrainingConfig.for_cifar10_deep()
                elif args.preset == "cifar10_blind":
                    temp_config = TrainingConfig.for_cifar10_blind()
                else:
                    temp_config = TrainingConfig.for_tinystories()
            num_envs = temp_config.n_envs
        elif args.algorithm == "heuristic":
            # For heuristic, use number of slots
            num_envs = len(args.slots)
        else:
            # Fallback for unknown algorithms
            num_envs = DEFAULT_N_ENVS

        overwatch_backend = OverwatchBackend(num_envs=num_envs)
        hub.add_backend(overwatch_backend)

    # Add Karn research telemetry collector
    # KarnCollector captures events into typed store for research analytics
    from esper.karn import get_collector
    karn_collector = get_collector()
    hub.add_backend(karn_collector)

    # Define training function to enable background execution for Overwatch
    def run_training():
        """Execute the training algorithm."""
        try:
            if args.algorithm == "heuristic":
                # Validate slot IDs use canonical format
                validated_slots = validate_slots(args.slots)

                from esper.simic.training import train_heuristic
                train_heuristic(
                    n_episodes=args.episodes,
                    max_epochs=args.max_epochs,
                    max_batches=args.max_batches if args.max_batches > 0 else None,
                    device=args.device,
                    task=args.task,
                    seed=args.seed,
                    slots=validated_slots,
                    telemetry_config=telemetry_config,
                    telemetry_lifecycle_only=args.telemetry_lifecycle_only,
                    min_fossilize_improvement=args.min_fossilize_improvement,
                )

            elif args.algorithm == "ppo":
                if args.config_json:
                    config = TrainingConfig.from_json_path(args.config_json)
                else:
                    if args.preset == "cifar10":
                        config = TrainingConfig.for_cifar10()
                    elif args.preset == "cifar10_deep":
                        config = TrainingConfig.for_cifar10_deep()
                    elif args.preset == "cifar10_blind":
                        config = TrainingConfig.for_cifar10_blind()
                    else:
                        config = TrainingConfig.for_tinystories()

                if args.seed is not None:
                    config.seed = args.seed
                if args.amp:
                    config.amp = True
                if telemetry_config.level.name == "OFF":
                    config.use_telemetry = False

                print(config.summary())

                from esper.simic.training import train_ppo_vectorized
                train_ppo_vectorized(
                    device=args.device,
                    devices=args.devices,
                    task=args.task,
                    save_path=args.save,
                    resume_path=args.resume,
                    num_workers=args.num_workers,
                    gpu_preload=args.gpu_preload,
                    telemetry_config=telemetry_config,
                    telemetry_lifecycle_only=args.telemetry_lifecycle_only,
                    quiet_analytics=use_tui or use_overwatch,
                    **config.to_train_kwargs(),
                )
        except Exception:
            import traceback
            traceback.print_exc()
            raise

    try:
        if use_overwatch:
            # Overwatch mode: run training in background thread, Overwatch controls terminal
            import threading
            from esper.karn.overwatch import OverwatchApp

            training_thread = threading.Thread(target=run_training, daemon=True)
            training_thread.start()

            # Run Overwatch TUI in main thread (blocks until user quits)
            app = OverwatchApp(backend=overwatch_backend)
            app.run()
        else:
            # Normal mode: run training directly in main thread
            run_training()
    finally:
        # Export Karn telemetry if requested (P1-04)
        if karn_collector and args.export_karn:
            from pathlib import Path
            export_path = Path(args.export_karn)
            count = karn_collector.store.export_jsonl(export_path)
            print(f"Exported {count} Karn records to {export_path}")

        # Clean up TUI backend if used
        if tui_backend is not None:
            tui_backend.close()
        # Close all hub backends
        hub.close()

if __name__ == "__main__":
    main()
