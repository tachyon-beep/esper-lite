"""IQL Policy Comparison and Evaluation Utilities.

This module provides functions for comparing trained IQL policies against
heuristic baselines in two modes:

1. Live Comparison: Both policies observe the same training trajectory
   (heuristic controls, IQL just predicts what it would do)

2. Head-to-Head: Each policy actually controls separate training runs
   to compare final outcomes

Extracted from original iql.py (~350 LOC).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from esper.leyline import SimicAction, SeedStage, SeedTelemetry
from esper.simic.episodes import TrainingSnapshot
from esper.simic.features import obs_to_base_features, telemetry_to_features
from esper.simic.gradient_collector import collect_seed_gradients
from esper.simic.iql import IQL
from esper.tamiyo import HeuristicTamiyo, HeuristicPolicyConfig, SignalTracker


# =============================================================================
# Model Loading
# =============================================================================

def load_iql_model(model_path: str, device: str = "cpu") -> tuple[IQL, str]:
    """Load a trained IQL model with automatic dimension detection.

    Detects the state dimension from the saved model and determines the
    telemetry mode based on the input dimension:
    - 27-dim: No telemetry ('none')
    - 37-dim: Seed telemetry ('seed') - 27 base + 10 seed
    - 54-dim: Legacy full-model telemetry ('legacy') - 27 base + 27 legacy

    Args:
        model_path: Path to saved model checkpoint
        device: Device to load model onto

    Returns:
        Tuple of (IQL agent, telemetry_mode string)
        telemetry_mode is one of: 'none', 'seed', 'legacy'

    Raises:
        ValueError: If state dimension is not one of the supported values
    """
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)

    state_dim = checkpoint['state_dim']

    # Determine telemetry mode from state dimension
    if state_dim == 27:
        telemetry_mode = 'none'
    elif state_dim == 37:
        telemetry_mode = 'seed'
    elif state_dim == 54:
        telemetry_mode = 'legacy'
    else:
        raise ValueError(
            f"Unknown state dimension: {state_dim}. "
            f"Expected 27 (no telemetry), 37 (seed telemetry), or 54 (legacy telemetry)."
        )

    # Log which mode is being used
    print(f"Loading IQL model: state_dim={state_dim}, telemetry_mode='{telemetry_mode}'")

    iql = IQL(
        state_dim=state_dim,
        action_dim=checkpoint['action_dim'],
        gamma=checkpoint.get('gamma', 0.99),
        tau=checkpoint.get('tau', 0.7),
        beta=checkpoint.get('beta', 3.0),
        device=device,
    )
    iql.q_network.load_state_dict(checkpoint['q_network'])
    iql.v_network.load_state_dict(checkpoint['v_network'])

    return iql, telemetry_mode


# =============================================================================
# Feature Conversion
# =============================================================================

def snapshot_to_features(
    snapshot: TrainingSnapshot,
    use_telemetry: bool = False,
    seed_telemetry: SeedTelemetry | None = None,
) -> list[float]:
    """Convert TrainingSnapshot to feature vector for IQL.

    Args:
        snapshot: Training state snapshot
        use_telemetry: Whether to include telemetry features
        seed_telemetry: Per-seed telemetry (10 dims). If None and use_telemetry=True,
                       falls back to zeros with a warning.

    Returns:
        Feature vector:
        - 27 dims if use_telemetry=False
        - 37 dims if use_telemetry=True (27 base + 10 seed)
    """
    # Convert snapshot to observation dict format expected by obs_to_base_features
    obs = {
        'epoch': snapshot.epoch,
        'global_step': snapshot.global_step,
        'train_loss': snapshot.train_loss,
        'val_loss': snapshot.val_loss,
        'loss_delta': snapshot.loss_delta,
        'train_accuracy': snapshot.train_accuracy,
        'val_accuracy': snapshot.val_accuracy,
        'accuracy_delta': snapshot.accuracy_delta,
        'plateau_epochs': snapshot.plateau_epochs,
        'best_val_accuracy': snapshot.best_val_accuracy,
        'best_val_loss': snapshot.best_val_loss,
        'loss_history_5': list(snapshot.loss_history_5),
        'accuracy_history_5': list(snapshot.accuracy_history_5),
        'has_active_seed': snapshot.has_active_seed,
        'seed_stage': snapshot.seed_stage,
        'seed_epochs_in_stage': snapshot.seed_epochs_in_stage,
        'seed_alpha': snapshot.seed_alpha,
        'seed_improvement': snapshot.seed_improvement,
        'available_slots': snapshot.available_slots,
    }

    features = obs_to_base_features(obs)

    if use_telemetry:
        if seed_telemetry is not None:
            features.extend(seed_telemetry.to_features())
        else:
            # Check if we have an active seed that needs telemetry
            if snapshot.has_active_seed:
                # CRITICAL: Zero-padding causes distribution shift (DRL review)
                # Refuse to proceed rather than corrupt the model's inputs
                raise ValueError(
                    "seed_telemetry is required when use_telemetry=True and seed is active. "
                    "Zero-padding telemetry features causes distribution shift and "
                    "degrades policy quality. Either provide real telemetry or set "
                    "use_telemetry=False."
                )
            else:
                # No active seed - zeros are semantically correct
                from esper.leyline import SeedTelemetry
                features.extend([0.0] * SeedTelemetry.feature_dim())

    return features


# =============================================================================
# Live Comparison Mode
# =============================================================================

def live_comparison(
    model_path: str,
    n_episodes: int = 5,
    max_epochs: int = 25,
    device: str = "cpu",
) -> dict:
    """Compare IQL policy decisions against heuristic Tamiyo (observation only).

    This mode runs a single training trajectory and asks both policies what
    they would do at each step. Actions are NOT executed - both policies
    observe the same unmodified training run.

    WARNING: This mode does not support telemetry features. If the loaded
    model was trained with telemetry (37-dim or 54-dim), it will be evaluated
    using only base features (27-dim) since no seeds are created. For accurate
    telemetry-aware evaluation, use head_to_head_comparison.

    Args:
        model_path: Path to saved IQL model checkpoint
        n_episodes: Number of episodes to run
        max_epochs: Maximum epochs per episode
        device: Device to run on

    Returns:
        Dictionary with comparison results including agreement rates
    """
    import torchvision
    import torchvision.transforms as transforms
    from torch.utils.data import DataLoader

    print("=" * 60)
    print("Live Comparison: Heuristic Tamiyo vs IQL Policy")
    print("=" * 60)

    # Load IQL model (use_telemetry inferred from checkpoint)
    print(f"Loading IQL model from {model_path}...")
    iql, telemetry_mode = load_iql_model(model_path, device=device)
    use_telemetry_model = (telemetry_mode in ['seed', 'legacy'])
    print(f"  State dim: {iql.q_network.net[0].in_features} (telemetry mode: {telemetry_mode})")

    # Warn about telemetry limitation in live_comparison mode
    if use_telemetry_model:
        import warnings
        warnings.warn(
            f"Model was trained with telemetry_mode='{telemetry_mode}', but live_comparison "
            "cannot provide telemetry (no seeds created). Evaluation will use base features "
            "only (27-dim). Results may not be accurate. Use head_to_head_comparison for "
            "telemetry-aware evaluation.",
            UserWarning,
        )

    # Force disable telemetry since we can't provide it
    use_telemetry = False

    # Load CIFAR-10
    print("Loading CIFAR-10...")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    trainloader = DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)
    testloader = DataLoader(testset, batch_size=128, shuffle=False, num_workers=2)

    results = {
        'heuristic_accuracies': [],
        'iql_would_have': [],  # What IQL would have chosen
        'agreements': [],
        'action_counts': {'heuristic': {a.name: 0 for a in SimicAction},
                         'iql': {a.name: 0 for a in SimicAction}},
    }

    for ep_idx in range(n_episodes):
        print(f"\n--- Episode {ep_idx + 1}/{n_episodes} ---")
        torch.manual_seed(42 + ep_idx)

        # Setup
        from esper.tolaria import create_model
        model = create_model(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.get_host_parameters(), lr=0.01, momentum=0.9)

        tamiyo = HeuristicTamiyo(HeuristicPolicyConfig())
        tracker = SignalTracker()

        ep_agreements = 0
        ep_total = 0

        for epoch in range(1, max_epochs + 1):
            # Train
            model.train()
            train_loss, train_correct, train_total = 0.0, 0, 0
            for inputs, labels in trainloader:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                _, pred = outputs.max(1)
                train_total += labels.size(0)
                train_correct += pred.eq(labels).sum().item()
            train_loss /= len(trainloader)
            train_acc = 100.0 * train_correct / train_total

            # Validate
            model.eval()
            val_loss, val_correct, val_total = 0.0, 0, 0
            with torch.no_grad():
                for inputs, labels in testloader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()
                    _, pred = outputs.max(1)
                    val_total += labels.size(0)
                    val_correct += pred.eq(labels).sum().item()
            val_loss /= len(testloader)
            val_acc = 100.0 * val_correct / val_total

            # Update tracker
            active_seeds = []
            if model.has_active_seed:
                active_seeds = [model.seed_state]
            available_slots = 0 if model.has_active_seed else 1
            signals = tracker.update(
                epoch=epoch,
                global_step=epoch * len(trainloader),
                train_loss=train_loss,
                train_accuracy=train_acc,
                val_loss=val_loss,
                val_accuracy=val_acc,
                active_seeds=active_seeds,
                available_slots=available_slots,
            )

            # Get heuristic decision (requires signals and active_seeds)
            h_action = tamiyo.decide(signals, active_seeds)

            # Get IQL decision
            # Pad history to 5 elements
            loss_hist = list(signals.loss_history[-5:]) if signals.loss_history else []
            while len(loss_hist) < 5:
                loss_hist.insert(0, 0.0)
            acc_hist = list(signals.accuracy_history[-5:]) if signals.accuracy_history else []
            while len(acc_hist) < 5:
                acc_hist.insert(0, 0.0)

            snapshot = TrainingSnapshot(
                epoch=signals.epoch,
                global_step=signals.global_step,
                train_loss=signals.train_loss,
                val_loss=signals.val_loss,
                loss_delta=signals.loss_delta,
                train_accuracy=signals.train_accuracy,
                val_accuracy=signals.val_accuracy,
                accuracy_delta=signals.accuracy_delta,
                plateau_epochs=signals.plateau_epochs,
                best_val_accuracy=signals.best_val_accuracy,
                best_val_loss=min(signals.loss_history) if signals.loss_history else float('inf'),
                loss_history_5=tuple(loss_hist),
                accuracy_history_5=tuple(acc_hist),
                has_active_seed=model.has_active_seed,
                available_slots=available_slots,
            )

            features = snapshot_to_features(snapshot, use_telemetry=False)  # No telemetry in live mode
            state_tensor = torch.tensor([features], dtype=torch.float32, device=device)
            iql_action_idx = iql.get_action(state_tensor, deterministic=True)
            iql_action = SimicAction(iql_action_idx).name

            # Track actions - both now use Action enum
            results['action_counts']['heuristic'][h_action.action.name] += 1
            results['action_counts']['iql'][iql_action] += 1

            if h_action.action.name == iql_action:
                ep_agreements += 1
            ep_total += 1

            # Note: We don't execute actions in this comparison - both policies
            # just observe the same training trajectory to compare decisions

        final_acc = val_acc
        agreement_rate = ep_agreements / ep_total if ep_total > 0 else 0

        results['heuristic_accuracies'].append(final_acc)
        results['agreements'].append(agreement_rate)

        print(f"  Final accuracy: {final_acc:.2f}%")
        print(f"  Agreement rate: {agreement_rate*100:.1f}%")

    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)

    avg_acc = sum(results['heuristic_accuracies']) / len(results['heuristic_accuracies'])
    avg_agreement = sum(results['agreements']) / len(results['agreements'])

    print(f"Average accuracy (heuristic): {avg_acc:.2f}%")
    print(f"Average agreement rate: {avg_agreement*100:.1f}%")
    print()
    print("Action distribution:")
    print(f"  {'Action':<12} {'Heuristic':>10} {'IQL':>10}")
    print(f"  {'-'*12} {'-'*10} {'-'*10}")

    h_total = sum(results['action_counts']['heuristic'].values())
    i_total = sum(results['action_counts']['iql'].values())

    for action in SimicAction:
        h_pct = results['action_counts']['heuristic'][action.name] / h_total * 100 if h_total > 0 else 0
        i_pct = results['action_counts']['iql'][action.name] / i_total * 100 if i_total > 0 else 0
        print(f"  {action.name:<12} {h_pct:>9.1f}% {i_pct:>9.1f}%")

    return results


# =============================================================================
# Head-to-Head Comparison Mode
# =============================================================================

def head_to_head_comparison(
    model_path: str,
    n_episodes: int = 5,
    max_epochs: int = 25,
    device: str = "cpu",
) -> dict:
    """Run head-to-head comparison where each policy ACTUALLY controls training.

    Unlike live_comparison (where both observe the same trajectory), this function
    runs TWO SEPARATE training runs per episode:
    1. Heuristic Tamiyo controls one run
    2. IQL policy controls another run

    We compare final accuracies to determine which policy is better.

    Note: use_telemetry is inferred from the model checkpoint's state_dim.
    """
    import torchvision
    import torchvision.transforms as transforms
    from torch.utils.data import DataLoader
    import torch.optim as optim

    print("=" * 60)
    print("Head-to-Head: Heuristic Tamiyo vs IQL Policy")
    print("=" * 60)
    print("Each policy EXECUTES its decisions in separate training runs")
    print()

    # Load IQL model (use_telemetry inferred from checkpoint)
    print(f"Loading IQL model from {model_path}...")
    iql, telemetry_mode = load_iql_model(model_path, device=device)
    use_telemetry = (telemetry_mode in ['seed', 'legacy'])
    state_dim = iql.q_network.net[0].in_features
    print(f"  State dim: {state_dim} (telemetry mode: {telemetry_mode})")

    # Load CIFAR-10 once
    print("Loading CIFAR-10...")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    trainloader = DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)
    testloader = DataLoader(testset, batch_size=128, shuffle=False, num_workers=2)

    results = {
        'heuristic_accuracies': [],
        'iql_accuracies': [],
        'heuristic_wins': 0,
        'iql_wins': 0,
        'ties': 0,
        'action_counts': {
            'heuristic': {a.name: 0 for a in SimicAction},
            'iql': {a.name: 0 for a in SimicAction},
        },
    }

    # Create heuristic Tamiyo ONCE - maintains state across episodes
    tamiyo = HeuristicTamiyo(HeuristicPolicyConfig())

    def run_training_episode(
        policy_name: str,
        action_fn,  # function(signals, model, tracker) -> SimicAction
        seed: int,
    ) -> tuple[float, dict]:
        """Run a single training episode controlled by the given policy."""
        torch.manual_seed(seed)

        from esper.tolaria import create_model
        from esper.simic.gradient_collector import SeedGradientCollector

        model = create_model(device)
        criterion = nn.CrossEntropyLoss()
        host_optimizer = optim.SGD(model.get_host_parameters(), lr=0.01, momentum=0.9)
        seed_optimizer = None

        # Instantiate gradient collector for telemetry
        gradient_collector = SeedGradientCollector()

        tracker = SignalTracker()
        action_counts = {a.name: 0 for a in SimicAction}
        seeds_created = 0

        for epoch in range(1, max_epochs + 1):
            seed_state = model.seed_state

            # Training phase - mode depends on seed state
            # We'll collect gradients after backward pass for seed parameters
            gradient_stats = None

            if seed_state is None:
                # No seed - normal training
                model.train()
                for inputs, labels in trainloader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    host_optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    # No seed to collect gradients from
                    host_optimizer.step()
            elif seed_state.stage == SeedStage.GERMINATED:
                # Auto-advance to TRAINING
                seed_state.transition(SeedStage.TRAINING)
                seed_optimizer = optim.SGD(model.get_seed_parameters(), lr=0.01, momentum=0.9)
                model.train()
                for inputs, labels in trainloader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    seed_optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    # Collect gradients AFTER backward, BEFORE optimizer.step
                    gradient_stats = gradient_collector.collect(model.get_seed_parameters())
                    seed_optimizer.step()
            elif seed_state.stage == SeedStage.TRAINING:
                # Seed isolated training
                if seed_optimizer is None:
                    seed_optimizer = optim.SGD(model.get_seed_parameters(), lr=0.01, momentum=0.9)
                model.train()
                for inputs, labels in trainloader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    seed_optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    # Collect gradients AFTER backward, BEFORE optimizer.step
                    gradient_stats = gradient_collector.collect(model.get_seed_parameters())
                    seed_optimizer.step()
            elif seed_state.stage in (SeedStage.BLENDING, SeedStage.FOSSILIZED):
                # Blending or fossilized - joint training
                if seed_state.stage == SeedStage.BLENDING:
                    step = seed_state.metrics.epochs_in_current_stage
                    model.seed_slot.update_alpha_for_step(step)
                model.train()
                for inputs, labels in trainloader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    host_optimizer.zero_grad()
                    if seed_optimizer:
                        seed_optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    # Collect gradients from seed parameters
                    gradient_stats = gradient_collector.collect(model.get_seed_parameters())
                    host_optimizer.step()
                    if seed_optimizer:
                        seed_optimizer.step()
            else:
                # Fallback - normal training
                model.train()
                for inputs, labels in trainloader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    host_optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    # No seed to collect gradients from
                    host_optimizer.step()

            # Validation phase
            model.eval()
            train_loss, train_correct, train_total = 0.0, 0, 0
            with torch.no_grad():
                for inputs, labels in trainloader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    train_loss += loss.item()
                    _, pred = outputs.max(1)
                    train_total += labels.size(0)
                    train_correct += pred.eq(labels).sum().item()
            train_loss /= len(trainloader)
            train_acc = 100.0 * train_correct / train_total

            val_loss, val_correct, val_total = 0.0, 0, 0
            with torch.no_grad():
                for inputs, labels in testloader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()
                    _, pred = outputs.max(1)
                    val_total += labels.size(0)
                    val_correct += pred.eq(labels).sum().item()
            val_loss /= len(testloader)
            val_acc = 100.0 * val_correct / val_total

            # Update seed metrics if active
            if model.has_active_seed:
                model.seed_state.metrics.record_accuracy(val_acc)

            # Sync telemetry after validation (if we have gradient stats and active seed)
            if model.has_active_seed and gradient_stats is not None:
                model.seed_state.sync_telemetry(
                    gradient_norm=gradient_stats['gradient_norm'],
                    gradient_health=gradient_stats['gradient_health'],
                    has_vanishing=gradient_stats['has_vanishing'],
                    has_exploding=gradient_stats['has_exploding'],
                    epoch=epoch,
                    max_epochs=max_epochs,
                )

            # Update signal tracker
            active_seeds = [model.seed_state] if model.has_active_seed else []
            available_slots = 0 if model.has_active_seed else 1
            signals = tracker.update(
                epoch=epoch,
                global_step=epoch * len(trainloader),
                train_loss=train_loss,
                train_accuracy=train_acc,
                val_loss=val_loss,
                val_accuracy=val_acc,
                active_seeds=active_seeds,
                available_slots=available_slots,
            )

            # Get action from policy
            action = action_fn(signals, model, tracker, use_telemetry)
            action_counts[action.name] += 1

            # Execute action
            if SimicAction.is_germinate(action):
                if not model.has_active_seed:
                    blueprint_id = SimicAction.get_blueprint_id(action)
                    seed_id = f"seed_{seeds_created}"
                    model.germinate_seed(blueprint_id, seed_id)
                    seeds_created += 1
                    seed_optimizer = None

            elif action == SimicAction.ADVANCE:
                if model.has_active_seed:
                    if model.seed_state.stage == SeedStage.TRAINING:
                        model.seed_state.transition(SeedStage.BLENDING)
                        model.seed_state.metrics.reset_stage_baseline()
                        model.seed_slot.start_blending(total_steps=5, temperature=1.0)
                    elif model.seed_state.stage == SeedStage.BLENDING:
                        model.seed_state.transition(SeedStage.FOSSILIZED)
                        model.seed_state.metrics.reset_stage_baseline()
                        model.seed_slot.set_alpha(1.0)

            elif action == SimicAction.CULL:
                if model.has_active_seed:
                    model.cull_seed()
                    seed_optimizer = None

        return val_acc, action_counts

    # Define policy action functions
    def heuristic_action_fn(signals, model, tracker, use_telemetry):
        """Get action from heuristic Tamiyo."""
        # Get active seeds from model
        active_seeds = [model.seed_state] if model.has_active_seed and model.seed_state else []
        decision = tamiyo.decide(signals, active_seeds)
        # Decision.action is already an Action enum from leyline
        return decision.action

    def iql_action_fn(signals, model, tracker, use_telemetry):
        """Get action from IQL policy."""
        # Build features
        loss_hist = list(signals.loss_history[-5:]) if signals.loss_history else []
        while len(loss_hist) < 5:
            loss_hist.insert(0, 0.0)
        acc_hist = list(signals.accuracy_history[-5:]) if signals.accuracy_history else []
        while len(acc_hist) < 5:
            acc_hist.insert(0, 0.0)

        available_slots = 0 if model.has_active_seed else 1
        snapshot = TrainingSnapshot(
            epoch=signals.epoch,
            global_step=signals.global_step,
            train_loss=signals.train_loss,
            val_loss=signals.val_loss,
            loss_delta=signals.loss_delta,
            train_accuracy=signals.train_accuracy,
            val_accuracy=signals.val_accuracy,
            accuracy_delta=signals.accuracy_delta,
            plateau_epochs=signals.plateau_epochs,
            best_val_accuracy=signals.best_val_accuracy,
            best_val_loss=min(signals.loss_history) if signals.loss_history else float('inf'),
            loss_history_5=tuple(loss_hist),
            accuracy_history_5=tuple(acc_hist),
            has_active_seed=model.has_active_seed,
            available_slots=available_slots,
        )

        # Pass real telemetry to snapshot_to_features when use_telemetry=True
        seed_telemetry = model.seed_state.telemetry if model.has_active_seed else None
        features = snapshot_to_features(
            snapshot,
            use_telemetry=use_telemetry,
            seed_telemetry=seed_telemetry
        )
        state_tensor = torch.tensor([features], dtype=torch.float32, device=device)
        action_idx = iql.get_action(state_tensor, deterministic=True)
        return SimicAction(action_idx)

    # Run episodes
    for ep_idx in range(n_episodes):
        print(f"\n{'='*60}")
        print(f"Episode {ep_idx + 1}/{n_episodes}")
        print(f"{'='*60}")

        base_seed = 42 + ep_idx * 1000

        # Run heuristic policy
        print(f"\nRunning HEURISTIC policy (seed={base_seed})...")
        h_acc, h_actions = run_training_episode("heuristic", heuristic_action_fn, base_seed)
        print(f"  Final accuracy: {h_acc:.2f}%")
        for action, count in h_actions.items():
            results['action_counts']['heuristic'][action] += count

        # Run IQL policy (SAME seed for fair comparison)
        print(f"\nRunning IQL policy (seed={base_seed})...")
        iql_acc, iql_actions = run_training_episode("iql", iql_action_fn, base_seed)
        print(f"  Final accuracy: {iql_acc:.2f}%")
        for action, count in iql_actions.items():
            results['action_counts']['iql'][action] += count

        # Record results
        results['heuristic_accuracies'].append(h_acc)
        results['iql_accuracies'].append(iql_acc)

        # Determine winner
        if iql_acc > h_acc + 0.5:  # 0.5% threshold to avoid noise
            results['iql_wins'] += 1
            winner = "IQL"
        elif h_acc > iql_acc + 0.5:
            results['heuristic_wins'] += 1
            winner = "Heuristic"
        else:
            results['ties'] += 1
            winner = "TIE"

        print(f"\n  WINNER: {winner} (H={h_acc:.2f}% vs IQL={iql_acc:.2f}%)")

        # Reset tamiyo state for next episode (blueprint rotation, etc.)
        if hasattr(tamiyo, 'reset'):
            tamiyo.reset()

    # Summary
    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)

    avg_h = sum(results['heuristic_accuracies']) / len(results['heuristic_accuracies'])
    avg_iql = sum(results['iql_accuracies']) / len(results['iql_accuracies'])

    print("\nAverage Accuracy:")
    print(f"  Heuristic: {avg_h:.2f}%")
    print(f"  IQL:       {avg_iql:.2f}%")
    print(f"  Δ:         {avg_iql - avg_h:+.2f}%")

    print("\nWin/Loss Record:")
    print(f"  IQL wins:       {results['iql_wins']}")
    print(f"  Heuristic wins: {results['heuristic_wins']}")
    print(f"  Ties:           {results['ties']}")

    print("\nAction Distributions:")
    print(f"  {'Action':<12} {'Heuristic':>10} {'IQL':>10}")
    print(f"  {'-'*12} {'-'*10} {'-'*10}")

    h_total = sum(results['action_counts']['heuristic'].values())
    i_total = sum(results['action_counts']['iql'].values())

    for action in SimicAction:
        h_pct = results['action_counts']['heuristic'][action.name] / h_total * 100 if h_total > 0 else 0
        i_pct = results['action_counts']['iql'][action.name] / i_total * 100 if i_total > 0 else 0
        print(f"  {action.name:<12} {h_pct:>9.1f}% {i_pct:>9.1f}%")

    # Final verdict
    print(f"\n{'='*60}")
    if results['iql_wins'] > results['heuristic_wins']:
        print("VERDICT: IQL is BETTER than heuristic Tamiyo!")
    elif results['heuristic_wins'] > results['iql_wins']:
        print("VERDICT: Heuristic Tamiyo is better than IQL.")
    else:
        print("VERDICT: It's a TIE - no clear winner.")
    print(f"{'='*60}")

    return results


__all__ = [
    "load_iql_model",
    "snapshot_to_features",
    "live_comparison",
    "head_to_head_comparison",
]
