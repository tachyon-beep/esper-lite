"""Esper Evaluate Script - PPO Tamiyo Diagnostic Tool.

Runs a trained PPO model through evaluation episodes and generates
diagnostic reports to understand what Tamiyo has learned.

Usage:
    python -m esper.scripts.evaluate --model models/ppo.pt --task cifar10 --episodes 20
    python -m esper.scripts.evaluate --model models/ppo.pt --task tinystories --episodes 50 --verbose

Diagnostics provided:
    1. Action distribution + entropy (detect collapsed/random policies)
    2. Temporal action patterns (phase-dependent behavior)
    3. Value calibration (V(s) vs actual returns)
    4. State-action contingency (does policy respond to observations?)
    5. Seed success analysis (are seeds helping?)
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass, field
from typing import NamedTuple

import torch
import torch.nn as nn

from esper.runtime import TaskSpec, get_task_spec
from esper.utils.loss import compute_task_loss_with_metrics
from esper.leyline import SeedStage
from esper.leyline.actions import get_blueprint_from_action, is_germinate_action


class StepRecord(NamedTuple):
    """Record of a single decision step."""

    episode: int
    epoch: int
    state: list[float]
    action: int
    action_name: str
    log_prob: float
    value: float
    reward: float
    entropy: float
    val_accuracy: float
    has_seed: bool
    seed_stage: int
    loss_trend: str  # 'improving', 'stagnating', 'worsening'


@dataclass
class EpisodeRecord:
    """Record of a complete episode."""

    episode_id: int
    final_accuracy: float
    total_return: float
    steps: list[StepRecord] = field(default_factory=list)
    seeds_created: int = 0
    seeds_culled: int = 0
    seeds_fossilized: int = 0
    accuracy_curve: list[float] = field(default_factory=list)


@dataclass
class DiagnosticReport:
    """Complete diagnostic report for Tamiyo."""

    # Episode summary
    n_episodes: int = 0
    mean_final_accuracy: float = 0.0
    std_final_accuracy: float = 0.0
    mean_return: float = 0.0

    # 1. Action distribution + entropy
    action_counts: dict[str, int] = field(default_factory=dict)
    action_frequencies: dict[str, float] = field(default_factory=dict)
    mean_entropy: float = 0.0
    min_entropy: float = 0.0
    max_entropy: float = 0.0
    entropy_by_phase: dict[str, float] = field(default_factory=dict)

    # 2. Temporal patterns
    action_by_phase: dict[str, dict[str, float]] = field(default_factory=dict)
    germination_timing: list[int] = field(default_factory=list)
    mean_seed_lifetime: float = 0.0

    # 3. Value calibration
    value_return_correlation: float = 0.0
    value_mae: float = 0.0
    value_bias: float = 0.0

    # 4. State-action contingency
    action_by_loss_trend: dict[str, dict[str, float]] = field(default_factory=dict)
    action_by_seed_state: dict[str, dict[str, float]] = field(default_factory=dict)

    # 5. Seed success
    total_seeds_created: int = 0
    total_seeds_fossilized: int = 0
    total_seeds_culled: int = 0
    seed_survival_rate: float = 0.0
    mean_accuracy_with_seeds: float = 0.0
    mean_accuracy_without_seeds: float = 0.0

    # Red flags
    red_flags: list[str] = field(default_factory=list)


def classify_loss_trend(loss_history: list[float]) -> str:
    """Classify loss trend from history."""
    if len(loss_history) < 2:
        return "unknown"
    recent = loss_history[-3:] if len(loss_history) >= 3 else loss_history
    delta = recent[-1] - recent[0]
    if delta < -0.05:
        return "improving"
    elif delta > 0.05:
        return "worsening"
    else:
        return "stagnating"


def classify_phase(epoch: int, max_epochs: int) -> str:
    """Classify training phase."""
    progress = epoch / max_epochs
    if progress < 0.33:
        return "early"
    elif progress < 0.66:
        return "mid"
    else:
        return "late"


def run_diagnostic_episode(
    agent,
    trainloader,
    testloader,
    episode_id: int,
    task_spec: TaskSpec,
    max_epochs: int = 25,
    device: str = "cuda:0",
) -> EpisodeRecord:
    """Run a single episode collecting diagnostic data."""
    from esper.tolaria import create_model
    from esper.tamiyo import SignalTracker
    from esper.simic.ppo import signals_to_features
    from esper.simic.rewards import compute_shaped_reward, SeedInfo

    torch.manual_seed(episode_id * 1000)

    model = create_model(task=task_spec, device=device)
    criterion = nn.CrossEntropyLoss()
    host_optimizer = torch.optim.SGD(
        model.get_host_parameters(), lr=task_spec.host_lr, momentum=0.9, weight_decay=5e-4
    )
    seed_optimizer = None
    signal_tracker = SignalTracker()
    ActionEnum = task_spec.action_enum
    task_type = task_spec.task_type

    record = EpisodeRecord(episode_id=episode_id, final_accuracy=0.0, total_return=0.0)
    seed_birth_epoch = None

    # Track host params and added params for compute rent
    host_params = sum(p.numel() for p in model.get_host_parameters() if p.requires_grad)
    params_added = 0  # Accumulates when seeds are fossilized

    for epoch in range(1, max_epochs + 1):
        seed_state = model.seed_state

        # Training phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        if seed_state is None:
            for inputs, targets in trainloader:
                inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
                host_optimizer.zero_grad(set_to_none=True)
                outputs = model(inputs)
                loss, correct_batch, batch_total = compute_task_loss_with_metrics(
                    outputs, targets, criterion, task_type
                )
                loss.backward()
                host_optimizer.step()
                running_loss += loss.item()
                total += batch_total
                correct += correct_batch

        elif seed_state.stage == SeedStage.GERMINATED:
            seed_state.transition(SeedStage.TRAINING)
            seed_optimizer = torch.optim.SGD(
                model.get_seed_parameters(), lr=task_spec.seed_lr, momentum=0.9
            )
            for inputs, targets in trainloader:
                inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
                seed_optimizer.zero_grad(set_to_none=True)
                outputs = model(inputs)
                loss, correct_batch, batch_total = compute_task_loss_with_metrics(
                    outputs, targets, criterion, task_type
                )
                loss.backward()
                seed_optimizer.step()
                running_loss += loss.item()
                total += batch_total
                correct += correct_batch

        elif seed_state.stage == SeedStage.TRAINING:
            if seed_optimizer is None:
                seed_optimizer = torch.optim.SGD(
                    model.get_seed_parameters(), lr=task_spec.seed_lr, momentum=0.9
                )
            for inputs, targets in trainloader:
                inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
                seed_optimizer.zero_grad(set_to_none=True)
                outputs = model(inputs)
                loss, correct_batch, batch_total = compute_task_loss_with_metrics(
                    outputs, targets, criterion, task_type
                )
                loss.backward()
                seed_optimizer.step()
                running_loss += loss.item()
                total += batch_total
                correct += correct_batch

        elif seed_state.stage == SeedStage.BLENDING:
            for inputs, targets in trainloader:
                inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
                host_optimizer.zero_grad(set_to_none=True)
                if seed_optimizer:
                    seed_optimizer.zero_grad(set_to_none=True)
                outputs = model(inputs)
                loss, correct_batch, batch_total = compute_task_loss_with_metrics(
                    outputs, targets, criterion, task_type
                )
                loss.backward()
                host_optimizer.step()
                if seed_optimizer:
                    seed_optimizer.step()
                running_loss += loss.item()
                total += batch_total
                correct += correct_batch

        elif seed_state.stage == SeedStage.FOSSILIZED:
            for inputs, targets in trainloader:
                inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
                host_optimizer.zero_grad(set_to_none=True)
                outputs = model(inputs)
                loss, correct_batch, batch_total = compute_task_loss_with_metrics(
                    outputs, targets, criterion, task_type
                )
                loss.backward()
                host_optimizer.step()
                running_loss += loss.item()
                total += batch_total
                correct += correct_batch
        else:
            # Fallback
            for inputs, targets in trainloader:
                inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
                host_optimizer.zero_grad(set_to_none=True)
                outputs = model(inputs)
                loss, correct_batch, batch_total = compute_task_loss_with_metrics(
                    outputs, targets, criterion, task_type
                )
                loss.backward()
                host_optimizer.step()
                running_loss += loss.item()
                total += batch_total
                correct += correct_batch

        train_loss = running_loss / len(trainloader)
        train_acc = 100.0 * correct / total if total > 0 else 0.0

        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.inference_mode():
            for inputs, targets in testloader:
                inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
                outputs = model(inputs)
                loss, correct_batch, batch_total = compute_task_loss_with_metrics(
                    outputs, targets, criterion, task_type
                )
                val_loss += loss.item()
                total += batch_total
                correct += correct_batch

        val_loss /= len(testloader)
        val_acc = 100.0 * correct / total if total > 0 else 0.0
        record.accuracy_curve.append(val_acc)

        # Update signals
        active_seeds = [seed_state] if seed_state else []
        available_slots = 0 if model.has_active_seed else 1
        signals = signal_tracker.update(
            epoch=epoch,
            global_step=epoch * len(trainloader),
            train_loss=train_loss,
            train_accuracy=train_acc,
            val_loss=val_loss,
            val_accuracy=val_acc,
            active_seeds=active_seeds,
            available_slots=available_slots,
        )

        # Record seed metrics for lifecycle gating
        if seed_state and seed_state.metrics:
            seed_state.metrics.record_accuracy(val_acc)

        # Mechanical lifecycle advance (blending/shadowing dwell)
        model.seed_slot.step_epoch()
        seed_state = model.seed_state

        # Get features and query policy
        features = signals_to_features(
            signals, model, use_telemetry=False
        )
        state = torch.tensor([features], dtype=torch.float32, device=device)

        # Get action with full distribution info
        with torch.inference_mode():
            dist, value = agent.network(state)
            entropy = dist.entropy().item()

            # Deterministic for evaluation
            action_idx = dist.probs.argmax().item()
            log_prob = dist.log_prob(torch.tensor(action_idx, device=device)).item()

        action = ActionEnum(action_idx)

        # Compute reward with cost params
        acc_delta = signals.metrics.accuracy_delta
        total_params = params_added + model.active_seed_params
        reward = compute_shaped_reward(
            action=action,  # Pass Enum Member
            acc_delta=acc_delta,
            val_acc=val_acc,
            seed_info=SeedInfo.from_seed_state(seed_state, model.active_seed_params),
            epoch=epoch,
            max_epochs=max_epochs,
            total_params=total_params,
            host_params=host_params,
        )
        record.total_return += reward

        # Classify state for contingency analysis
        loss_trend = classify_loss_trend(list(signals.loss_history))

        # Record step
        step = StepRecord(
            episode=episode_id,
            epoch=epoch,
            state=features,
            action=action_idx,
            action_name=action.name,
            log_prob=log_prob,
            value=value.item(),
            reward=reward,
            entropy=entropy,
            val_accuracy=val_acc,
            has_seed=model.has_active_seed,
            seed_stage=seed_state.stage.value if seed_state else 0,
            loss_trend=loss_trend,
        )
        record.steps.append(step)

        # Execute action
        if is_germinate_action(action):
            if not model.has_active_seed:
                blueprint_id = get_blueprint_from_action(action)
                seed_id = f"seed_{record.seeds_created}"
                model.germinate_seed(blueprint_id, seed_id)
                record.seeds_created += 1
                seed_birth_epoch = epoch
                seed_optimizer = None

        elif action == ActionEnum.FOSSILIZE:
            if model.has_active_seed and model.seed_state.stage in (
                SeedStage.PROBATIONARY,
                SeedStage.SHADOWING,
            ):
                # Use SeedSlot.advance_stage so fossilization respects gates
                # and emits telemetry via Nissa.
                gate_result = model.seed_slot.advance_stage(SeedStage.FOSSILIZED)
                if gate_result.passed:
                    params_added += model.active_seed_params
                    model.seed_slot.set_alpha(1.0)
                    record.seeds_fossilized += 1

        elif action == ActionEnum.CULL:
            if model.has_active_seed:
                model.cull_seed()
                record.seeds_culled += 1
                seed_optimizer = None

    record.final_accuracy = val_acc
    return record


def compute_diagnostics(
    records: list[EpisodeRecord], max_epochs: int
) -> DiagnosticReport:
    """Compute all diagnostics from episode records."""
    report = DiagnosticReport()
    report.n_episodes = len(records)

    if not records:
        return report

    # Episode summaries
    accuracies = [r.final_accuracy for r in records]
    returns = [r.total_return for r in records]
    report.mean_final_accuracy = sum(accuracies) / len(accuracies)
    report.std_final_accuracy = (
        sum((a - report.mean_final_accuracy) ** 2 for a in accuracies) / len(accuracies)
    ) ** 0.5
    report.mean_return = sum(returns) / len(returns)

    # Collect all steps
    all_steps = [step for r in records for step in r.steps]

    # 1. Action distribution + entropy
    action_counts = defaultdict(int)
    entropies = []
    entropy_by_phase = defaultdict(list)

    for step in all_steps:
        action_counts[step.action_name] += 1
        entropies.append(step.entropy)
        phase = classify_phase(step.epoch, max_epochs)
        entropy_by_phase[phase].append(step.entropy)

    total_actions = sum(action_counts.values())
    report.action_counts = dict(action_counts)
    report.action_frequencies = (
        {k: v / total_actions for k, v in action_counts.items()}
        if total_actions > 0
        else {}
    )
    report.mean_entropy = sum(entropies) / len(entropies) if entropies else 0
    report.min_entropy = min(entropies) if entropies else 0
    report.max_entropy = max(entropies) if entropies else 0
    report.entropy_by_phase = {k: sum(v) / len(v) for k, v in entropy_by_phase.items()}

    # 2. Temporal patterns
    action_by_phase = defaultdict(lambda: defaultdict(int))
    germination_epochs = []
    seed_lifetimes = []

    for record in records:
        seed_birth = None
        for step in record.steps:
            phase = classify_phase(step.epoch, max_epochs)
            action_by_phase[phase][step.action_name] += 1

            if "GERMINATE" in step.action_name and not step.has_seed:
                germination_epochs.append(step.epoch)
                seed_birth = step.epoch
            elif step.action_name == "CULL" and seed_birth:
                seed_lifetimes.append(step.epoch - seed_birth)
                seed_birth = None

    # Normalize action_by_phase
    for phase, counts in action_by_phase.items():
        total = sum(counts.values())
        report.action_by_phase[phase] = {k: v / total for k, v in counts.items()}

    report.germination_timing = germination_epochs
    report.mean_seed_lifetime = (
        sum(seed_lifetimes) / len(seed_lifetimes) if seed_lifetimes else 0
    )

    # 3. Value calibration
    values = [step.value for step in all_steps]

    # Compute actual returns (discounted sum of future rewards)
    # Use gamma=0.99 to match PPO's discount factor
    gamma = 0.99
    actual_returns = []
    for record in records:
        for i, step in enumerate(record.steps):
            future_return = sum(
                gamma**j * s.reward for j, s in enumerate(record.steps[i:])
            )
            actual_returns.append(future_return)

    if len(values) > 1:
        # Correlation
        mean_v = sum(values) / len(values)
        mean_r = sum(actual_returns) / len(actual_returns)
        cov = sum(
            (v - mean_v) * (r - mean_r) for v, r in zip(values, actual_returns)
        ) / len(values)
        std_v = (sum((v - mean_v) ** 2 for v in values) / len(values)) ** 0.5
        std_r = (
            sum((r - mean_r) ** 2 for r in actual_returns) / len(actual_returns)
        ) ** 0.5
        report.value_return_correlation = (
            cov / (std_v * std_r) if std_v * std_r > 0 else 0
        )

        # MAE and bias
        errors = [v - r for v, r in zip(values, actual_returns)]
        report.value_mae = sum(abs(e) for e in errors) / len(errors)
        report.value_bias = sum(errors) / len(errors)

    # 4. State-action contingency
    action_by_loss_trend = defaultdict(lambda: defaultdict(int))
    action_by_seed_state = defaultdict(lambda: defaultdict(int))

    for step in all_steps:
        action_by_loss_trend[step.loss_trend][step.action_name] += 1
        seed_state_key = "has_seed" if step.has_seed else "no_seed"
        action_by_seed_state[seed_state_key][step.action_name] += 1

    for trend, counts in action_by_loss_trend.items():
        total = sum(counts.values())
        report.action_by_loss_trend[trend] = {k: v / total for k, v in counts.items()}

    for state, counts in action_by_seed_state.items():
        total = sum(counts.values())
        report.action_by_seed_state[state] = {k: v / total for k, v in counts.items()}

    # 5. Seed success
    report.total_seeds_created = sum(r.seeds_created for r in records)
    report.total_seeds_fossilized = sum(r.seeds_fossilized for r in records)
    report.total_seeds_culled = sum(r.seeds_culled for r in records)

    if report.total_seeds_created > 0:
        report.seed_survival_rate = (
            report.total_seeds_fossilized / report.total_seeds_created
        )

    # Accuracy with/without seeds (per-step analysis)
    acc_with_seeds = [s.val_accuracy for s in all_steps if s.has_seed]
    acc_without_seeds = [s.val_accuracy for s in all_steps if not s.has_seed]
    report.mean_accuracy_with_seeds = (
        sum(acc_with_seeds) / len(acc_with_seeds) if acc_with_seeds else 0
    )
    report.mean_accuracy_without_seeds = (
        sum(acc_without_seeds) / len(acc_without_seeds) if acc_without_seeds else 0
    )

    # Red flags
    max_freq = (
        max(report.action_frequencies.values()) if report.action_frequencies else 0
    )
    if max_freq > 0.9:
        dominant = max(report.action_frequencies, key=report.action_frequencies.get)
        report.red_flags.append(
            f"COLLAPSED POLICY: {dominant} chosen {max_freq * 100:.1f}% of time"
        )

    if report.mean_entropy < 0.1:
        report.red_flags.append(
            f"LOW ENTROPY: {report.mean_entropy:.3f} - policy may be too deterministic"
        )
    elif report.mean_entropy > 1.8:
        # Max entropy for 7 actions is ln(7) ~ 1.95
        report.red_flags.append(
            f"HIGH ENTROPY: {report.mean_entropy:.3f} - policy may be near-random"
        )

    if report.value_return_correlation < 0.5:
        report.red_flags.append(
            f"POOR VALUE CALIBRATION: R={report.value_return_correlation:.2f} - value function is not predictive"
        )

    if report.total_seeds_created == 0:
        report.red_flags.append("NO SEEDS CREATED: Tamiyo never germinated")
    elif report.seed_survival_rate < 0.1:
        report.red_flags.append(
            f"LOW SEED SURVIVAL: {report.seed_survival_rate * 100:.1f}% - seeds rarely reach fossilization"
        )

    return report


def format_report(report: DiagnosticReport, verbose: bool = False) -> str:
    """Format diagnostic report as human-readable text."""
    lines = []
    lines.append("=" * 70)
    lines.append("TAMIYO DIAGNOSTIC REPORT")
    lines.append("=" * 70)

    # Summary
    lines.append(f"\nEpisodes evaluated: {report.n_episodes}")
    lines.append(
        f"Final accuracy: {report.mean_final_accuracy:.2f}% ± {report.std_final_accuracy:.2f}%"
    )
    lines.append(f"Mean return: {report.mean_return:.2f}")

    # Red flags
    if report.red_flags:
        lines.append("\n" + "!" * 70)
        lines.append("RED FLAGS:")
        for flag in report.red_flags:
            lines.append(f"  ⚠ {flag}")
        lines.append("!" * 70)

    # 1. Action Distribution
    lines.append("\n" + "-" * 70)
    lines.append("1. ACTION DISTRIBUTION")
    lines.append("-" * 70)
    for action, freq in sorted(report.action_frequencies.items(), key=lambda x: -x[1]):
        count = report.action_counts[action]
        bar = "█" * int(freq * 40)
        lines.append(f"  {action:20s} {freq * 100:5.1f}% ({count:4d}) {bar}")

    lines.append(
        f"\n  Entropy: mean={report.mean_entropy:.3f}, min={report.min_entropy:.3f}, max={report.max_entropy:.3f}"
    )
    if report.entropy_by_phase:
        lines.append("  Entropy by phase:")
        for phase in ["early", "mid", "late"]:
            if phase in report.entropy_by_phase:
                lines.append(f"    {phase}: {report.entropy_by_phase[phase]:.3f}")

    # 2. Temporal Patterns
    lines.append("\n" + "-" * 70)
    lines.append("2. TEMPORAL PATTERNS")
    lines.append("-" * 70)

    if report.action_by_phase:
        lines.append("  Action distribution by training phase:")
        for phase in ["early", "mid", "late"]:
            if phase in report.action_by_phase:
                top_actions = sorted(
                    report.action_by_phase[phase].items(), key=lambda x: -x[1]
                )[:3]
                top_str = ", ".join(f"{a}={p * 100:.0f}%" for a, p in top_actions)
                lines.append(f"    {phase:6s}: {top_str}")

    if report.germination_timing:
        mean_germ = sum(report.germination_timing) / len(report.germination_timing)
        lines.append(
            f"\n  Germination timing: mean epoch {mean_germ:.1f} (n={len(report.germination_timing)})"
        )

    lines.append(f"  Mean seed lifetime: {report.mean_seed_lifetime:.1f} epochs")

    # 3. Value Calibration
    lines.append("\n" + "-" * 70)
    lines.append("3. VALUE CALIBRATION")
    lines.append("-" * 70)
    lines.append(
        f"  V(s) vs actual return correlation: {report.value_return_correlation:.3f}"
    )
    lines.append(f"  Mean absolute error: {report.value_mae:.3f}")
    lines.append(f"  Bias (mean error): {report.value_bias:+.3f}")

    quality = (
        "GOOD"
        if report.value_return_correlation > 0.7
        else "FAIR"
        if report.value_return_correlation > 0.5
        else "POOR"
    )
    lines.append(f"  Value function quality: {quality}")

    # 4. State-Action Contingency
    lines.append("\n" + "-" * 70)
    lines.append("4. STATE-ACTION CONTINGENCY")
    lines.append("-" * 70)

    if report.action_by_loss_trend:
        lines.append("  Action distribution by loss trend:")
        for trend in ["improving", "stagnating", "worsening"]:
            if trend in report.action_by_loss_trend:
                top_actions = sorted(
                    report.action_by_loss_trend[trend].items(), key=lambda x: -x[1]
                )[:3]
                top_str = ", ".join(f"{a}={p * 100:.0f}%" for a, p in top_actions)
                lines.append(f"    {trend:11s}: {top_str}")

    if report.action_by_seed_state:
        lines.append("\n  Action distribution by seed state:")
        for state in ["no_seed", "has_seed"]:
            if state in report.action_by_seed_state:
                top_actions = sorted(
                    report.action_by_seed_state[state].items(), key=lambda x: -x[1]
                )[:3]
                top_str = ", ".join(f"{a}={p * 100:.0f}%" for a, p in top_actions)
                lines.append(f"    {state:8s}: {top_str}")

    # 5. Seed Success
    lines.append("\n" + "-" * 70)
    lines.append("5. SEED SUCCESS ANALYSIS")
    lines.append("-" * 70)
    lines.append(f"  Seeds created: {report.total_seeds_created}")
    lines.append(f"  Seeds fossilized: {report.total_seeds_fossilized}")
    lines.append(f"  Seeds culled: {report.total_seeds_culled}")
    lines.append(f"  Survival rate: {report.seed_survival_rate * 100:.1f}%")

    if report.mean_accuracy_with_seeds > 0 or report.mean_accuracy_without_seeds > 0:
        delta = report.mean_accuracy_with_seeds - report.mean_accuracy_without_seeds
        lines.append(
            f"\n  Mean accuracy with seeds: {report.mean_accuracy_with_seeds:.2f}%"
        )
        lines.append(
            f"  Mean accuracy without seeds: {report.mean_accuracy_without_seeds:.2f}%"
        )
        lines.append(f"  Difference: {delta:+.2f}%")

    lines.append("\n" + "=" * 70)

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Diagnostic evaluation of trained PPO Tamiyo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m esper.scripts.evaluate --model models/ppo.pt --task cifar10 --episodes 20
  python -m esper.scripts.evaluate --model models/ppo.pt --task tinystories --episodes 50 --verbose
        """,
    )
    parser.add_argument(
        "--model", type=str, required=True, help="Path to trained PPO model (.pt file)"
    )
    parser.add_argument(
        "--task",
        type=str,
        default="cifar10",
        help="Task preset to evaluate (e.g., cifar10, tinystories)",
    )
    parser.add_argument(
        "--episodes", type=int, default=20, help="Number of evaluation episodes"
    )
    parser.add_argument(
        "--max-epochs", type=int, default=25, help="Max epochs per episode"
    )
    parser.add_argument(
        "--device", type=str, default="cuda:0", help="Device for evaluation"
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Print detailed per-episode info"
    )
    parser.add_argument("--output", type=str, default=None, help="Save report to file")

    args = parser.parse_args()

    # Lazy imports
    from esper.simic.ppo import PPOAgent

    task_spec = get_task_spec(args.task)
    ActionEnum = task_spec.action_enum

    print(f"Loading model from {args.model}...")
    agent = PPOAgent.load(args.model, device=args.device)
    agent.network.eval()

    print(
        f"Task: {task_spec.name} (topology={task_spec.topology}, type={task_spec.task_type})"
    )
    print(f"Loading data for task '{task_spec.name}'...")
    trainloader, testloader = task_spec.create_dataloaders()

    print(f"Running {args.episodes} evaluation episodes...")
    records = []

    for ep in range(args.episodes):
        record = run_diagnostic_episode(
            agent=agent,
            trainloader=trainloader,
            testloader=testloader,
            episode_id=ep,
            task_spec=task_spec,
            max_epochs=args.max_epochs,
            device=args.device,
        )
        records.append(record)

        if args.verbose:
            print(
                f"  Episode {ep + 1}: acc={record.final_accuracy:.2f}%, return={record.total_return:.2f}, seeds={record.seeds_created}"
            )
        else:
            print(f"  Episode {ep + 1}/{args.episodes} complete", end="\r")

    print()

    # Compute and display diagnostics
    report = compute_diagnostics(records, args.max_epochs)
    report_text = format_report(report, verbose=args.verbose)
    print(report_text)

    if args.output:
        with open(args.output, "w") as f:
            f.write(report_text)
        print(f"\nReport saved to {args.output}")


if __name__ == "__main__":
    main()
