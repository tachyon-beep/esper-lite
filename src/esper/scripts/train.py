#!/usr/bin/env python3
"""Training CLI for Simic RL algorithms."""

import argparse
import logging

import torch

_logger = logging.getLogger(__name__)

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
        "--gradient-telemetry-stride",
        type=int,
        default=None,
        help="Stride for gradient telemetry collection (default: 10, or 1 if level is debug)",
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
        help="DEPRECATED: Use --sanctum instead. This flag is ignored.",
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
    telemetry_parent.add_argument(
        "--sanctum",
        action="store_true",
        help="Launch Sanctum TUI for developer debugging (replaces Rich TUI, mutually exclusive with --overwatch)",
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
        choices=["cifar10", "cifar10_stable", "cifar10_deep", "cifar10_blind", "tinystories"],
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
        "--amp-dtype",
        type=str,
        choices=["auto", "float16", "bfloat16", "off"],
        default="auto",
        help="AMP dtype: auto (detect BF16 support), float16, bfloat16, or off. "
             "BF16 eliminates GradScaler overhead on Ampere+ GPUs. (default: auto)",
    )
    ppo_parser.add_argument(
        "--compile-mode",
        type=str,
        choices=["default", "max-autotune", "reduce-overhead", "off"],
        default="default",
        help="torch.compile mode: default (fast compile), max-autotune (slow compile, faster runtime), "
             "reduce-overhead (minimal overhead), or off. (default: default)",
    )
    ppo_parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override seed (otherwise use config value)",
    )
    ppo_parser.add_argument(
        "--ab-test",
        type=str,
        choices=["shaped-vs-simplified", "shaped-vs-sparse"],
        default=None,
        help="Run A/B test: split envs between two reward modes (requires even n_envs)",
    )
    ppo_parser.add_argument(
        "--dual-ab",
        type=str,
        choices=["shaped-vs-simplified", "shaped-vs-sparse", "simplified-vs-sparse"],
        default=None,
        help="True A/B test: train separate policies on separate GPUs",
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

    # Check mutual exclusion
    if args.overwatch and args.sanctum:
        parser.error("--overwatch and --sanctum are mutually exclusive. Choose one.")

    # Determine UI mode
    import sys
    is_tty = sys.stdout.isatty()
    use_overwatch = args.overwatch
    use_sanctum = args.sanctum

    if not is_tty and not args.no_tui:
        print("Non-TTY detected, using console output instead of TUI")

    # Warn about deprecated --tui-layout flag
    if args.tui_layout != "auto":
        print(
            f"WARNING: --tui-layout is deprecated and ignored. "
            f"Use --sanctum for the developer TUI or --overwatch for operator monitoring."
        )

    # Add console output if not using a TUI backend
    if not use_overwatch and not use_sanctum:
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
                    except OSError as e:
                        _logger.debug("Network interface discovery failed: %s", e)
                    finally:
                        s.close()
                except OSError as e:
                    _logger.debug("Network interface discovery failed: %s", e)
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
                elif args.preset == "cifar10_stable":
                    temp_config = TrainingConfig.for_cifar10_stable()
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

    # Setup Sanctum backend if requested
    sanctum_backend = None
    if use_sanctum:
        from esper.karn.sanctum import SanctumBackend
        from esper.leyline import DEFAULT_N_ENVS

        # Determine num_envs for Sanctum display (same logic as Overwatch)
        if args.algorithm == "ppo":
            # For PPO, get from config
            if args.config_json:
                temp_config = TrainingConfig.from_json_path(args.config_json)
            else:
                if args.preset == "cifar10":
                    temp_config = TrainingConfig.for_cifar10()
                elif args.preset == "cifar10_stable":
                    temp_config = TrainingConfig.for_cifar10_stable()
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

        sanctum_backend = SanctumBackend(num_envs=num_envs)
        hub.add_backend(sanctum_backend)

    # Add Karn research telemetry collector
    # KarnCollector captures events into typed store for research analytics
    from esper.karn import get_collector
    karn_collector = get_collector()
    hub.add_backend(karn_collector)

    # Event to signal when DataLoader workers are spawned (for TUI synchronization)
    # When using TUI backends (Sanctum/Overwatch), main thread waits for this event
    # before starting Textual, ensuring workers spawn while terminal FDs are valid.
    import threading
    dataloader_ready_event: threading.Event | None = None
    if use_overwatch or use_sanctum:
        dataloader_ready_event = threading.Event()

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
                    gradient_telemetry_stride=args.gradient_telemetry_stride if args.gradient_telemetry_stride is not None else (1 if args.telemetry_level == "debug" else 10),
                )

            elif args.algorithm == "ppo":
                if args.config_json:
                    config = TrainingConfig.from_json_path(args.config_json)
                else:
                    if args.preset == "cifar10":
                        config = TrainingConfig.for_cifar10()
                    elif args.preset == "cifar10_stable":
                        config = TrainingConfig.for_cifar10_stable()
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
                # CLI amp_dtype overrides config only if explicitly set (not default 'auto')
                # args.amp_dtype is always defined by argparse (lines 160-166 of this file)
                if args.amp_dtype:
                    config.amp_dtype = args.amp_dtype
                # CLI compile_mode overrides config
                # args.compile_mode is always defined by argparse (lines 167-174 of this file)
                if args.compile_mode:
                    config.compile_mode = args.compile_mode
                if telemetry_config.level.name == "OFF":
                    config.use_telemetry = False

                # Handle gradient telemetry stride: CLI > debug default (1) > config default (10)
                if args.gradient_telemetry_stride is not None:
                    config.gradient_telemetry_stride = args.gradient_telemetry_stride
                elif args.telemetry_level == "debug":
                    config.gradient_telemetry_stride = 1

                # Handle A/B testing - set on config for validation
                if args.ab_test:
                    if config.n_envs % 2 != 0:
                        raise ValueError("--ab-test requires even number of envs")
                    half = config.n_envs // 2
                    if args.ab_test == "shaped-vs-simplified":
                        config.ab_reward_modes = ["shaped"] * half + ["simplified"] * half
                    elif args.ab_test == "shaped-vs-sparse":
                        config.ab_reward_modes = ["shaped"] * half + ["sparse"] * half
                    print(f"[A/B Test] {half} envs SHAPED vs {half} envs {args.ab_test.split('-vs-')[1].upper()}")

                # Handle dual-policy A/B testing
                if args.dual_ab:
                    # Check for GPU availability
                    if not torch.cuda.is_available():
                        raise ValueError("--dual-ab requires CUDA")
                    if torch.cuda.device_count() < 2:
                        raise ValueError("--dual-ab requires at least 2 GPUs")

                    # Parse dual-ab choice to reward modes
                    from esper.simic.rewards import RewardMode
                    mode_map = {
                        "shaped": RewardMode.SHAPED,
                        "simplified": RewardMode.SIMPLIFIED,
                        "sparse": RewardMode.SPARSE,
                    }
                    parts = args.dual_ab.split("-vs-")
                    mode_a = mode_map[parts[0]]
                    mode_b = mode_map[parts[1]]

                    group_configs = [
                        ("A", mode_a),
                        ("B", mode_b),
                    ]
                    devices = ["cuda:0", "cuda:1"]

                    print(f"[Dual-Policy A/B Test] Training Policy A ({mode_a.value}) on {devices[0]} vs Policy B ({mode_b.value}) on {devices[1]}")
                    print(config.summary())

                    # Use task from config if specified, otherwise CLI arg
                    effective_task = config.task if config.task else args.task

                    from esper.simic.training import train_dual_policy_ab
                    train_dual_policy_ab(
                        n_envs_per_group=config.n_envs,
                        group_configs=group_configs,
                        devices=devices,
                        n_episodes=config.n_episodes,
                        max_epochs=config.episode_length,
                        task=effective_task,
                        lr=config.learning_rate,
                        clip_ratio=config.clip_ratio,
                        entropy_coef=config.entropy_coef,
                        entropy_coef_min=config.entropy_coef_min,
                        gamma=config.gamma,
                        gae_lambda=config.gae_lambda,
                        lstm_hidden_dim=config.lstm_hidden_dim,
                        seed=config.seed,
                        use_telemetry=config.use_telemetry,
                        slots=config.slots,
                    )
                else:
                    print(config.summary())

                    # Use task from config if specified, otherwise CLI arg
                    effective_task = config.task if config.task else args.task

                    from esper.simic.training import train_ppo_vectorized
                    train_ppo_vectorized(
                        device=args.device,
                        devices=args.devices,
                        task=effective_task,
                        save_path=args.save,
                        resume_path=args.resume,
                        num_workers=args.num_workers,
                        gpu_preload=args.gpu_preload,
                        telemetry_config=telemetry_config,
                        telemetry_lifecycle_only=args.telemetry_lifecycle_only,
                        quiet_analytics=use_overwatch or use_sanctum,
                        ready_event=dataloader_ready_event,
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

            # CRITICAL: Wait for DataLoader workers to spawn BEFORE starting Textual.
            # Textual modifies terminal file descriptors, which breaks multiprocessing spawn.
            # By waiting here, workers spawn while FDs are still valid.
            if dataloader_ready_event is not None:
                print("Waiting for DataLoader workers to initialize...")
                dataloader_ready_event.wait(timeout=60.0)  # 60s timeout for slow datasets
                if not dataloader_ready_event.is_set():
                    print("WARNING: DataLoader initialization timed out, starting TUI anyway")

            # Run Overwatch TUI in main thread (blocks until user quits)
            app = OverwatchApp(backend=overwatch_backend)
            app.run()
        elif use_sanctum:
            # Sanctum mode: run training in background thread, Sanctum controls terminal
            import threading
            import traceback
            from esper.karn.sanctum import SanctumApp

            # Track training errors for debugging
            training_error = [None]  # Use list to allow mutation from thread

            def training_wrapper():
                """Wrap training to capture exceptions."""
                try:
                    run_training()
                except Exception as e:
                    training_error[0] = traceback.format_exc()
                    # Log to stderr (visible in Textual console)
                    import sys
                    print(f"\n[TRAINING ERROR]\n{training_error[0]}", file=sys.stderr)

            training_thread = threading.Thread(target=training_wrapper, daemon=True)
            training_thread.start()

            # CRITICAL: Wait for DataLoader workers to spawn BEFORE starting Textual.
            # Textual modifies terminal file descriptors, which breaks multiprocessing spawn.
            # By waiting here, workers spawn while FDs are still valid.
            if dataloader_ready_event is not None:
                print("Waiting for DataLoader workers to initialize...")
                dataloader_ready_event.wait(timeout=60.0)  # 60s timeout for slow datasets
                if not dataloader_ready_event.is_set():
                    print("WARNING: DataLoader initialization timed out, starting TUI anyway")

            # Run Sanctum TUI in main thread (blocks until user quits)
            # Pass training_thread so TUI can monitor if it's alive
            app = SanctumApp(
                backend=sanctum_backend,
                num_envs=num_envs,
                training_thread=training_thread,
            )
            app.run()

            # After TUI exits, show any training error
            if training_error[0]:
                print(f"\n[Training crashed with error:]\n{training_error[0]}")
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

        # Close all hub backends (includes Overwatch/Sanctum if used)
        hub.close()

if __name__ == "__main__":
    # CRITICAL: Set spawn method before main() to avoid fork issues with
    # Textual TUI + PyTorch DataLoader workers. The 'fork' method fails
    # when the main process runs a TUI (Textual/Overwatch/Sanctum) because
    # forked workers inherit state that can't be safely duplicated.
    import multiprocessing
    multiprocessing.set_start_method("spawn")
    main()
