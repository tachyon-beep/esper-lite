#!/usr/bin/env python3
"""Phase 5 reward calibration helper.

Summarizes reward component scales and alpha deltas from telemetry JSONL files.
Outputs suggested magnitudes for BaseSlotRent and alpha-shock coefficients.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median


def _percentile(sorted_values: list[float], q: float) -> float | None:
    if not sorted_values:
        return None
    if q <= 0:
        return sorted_values[0]
    if q >= 1:
        return sorted_values[-1]
    idx = int(math.ceil(q * len(sorted_values))) - 1
    idx = max(0, min(idx, len(sorted_values) - 1))
    return sorted_values[idx]


def _summarize(values: list[float]) -> dict[str, float]:
    if not values:
        return {}
    values_sorted = sorted(values)
    return {
        "count": float(len(values_sorted)),
        "mean": mean(values_sorted),
        "median": median(values_sorted),
        "p90": _percentile(values_sorted, 0.9) or 0.0,
        "p95": _percentile(values_sorted, 0.95) or 0.0,
        "min": values_sorted[0],
        "max": values_sorted[-1],
    }


def _find_event_paths(base: Path, limit: int) -> list[Path]:
    if base.is_file():
        return [base]
    if not base.exists():
        return []
    candidates = sorted(base.glob("telemetry_*/events.jsonl"))
    if limit > 0:
        candidates = candidates[-limit:]
    return candidates


@dataclass
class RandomTamiyo:
    topology: str = "cnn"
    seed: int = 0
    germinate_bias: float = 0.0

    def __post_init__(self) -> None:
        from esper.kasmina.blueprints import BlueprintRegistry
        from esper.leyline.actions import (
            build_action_enum,
            get_blueprint_from_action_name,
            is_germinate_action_name,
        )

        if self.germinate_bias < 0:
            raise ValueError("germinate_bias must be >= 0")

        self._action_enum = build_action_enum(self.topology)
        self._germinate_actions = []
        for action in self._action_enum:
            if not is_germinate_action_name(action.name):
                continue
            blueprint_id = get_blueprint_from_action_name(action.name)
            try:
                spec = BlueprintRegistry.get(self.topology, blueprint_id)
            except ValueError:
                continue
            if spec.param_estimate > 0:
                self._germinate_actions.append(action)
        self._rng = random.Random(self.seed)

    def reset(self) -> None:
        self._rng.seed(self.seed)

    def decide(self, signals, active_seeds):
        from esper.leyline import SeedStage
        from esper.tamiyo.decisions import TamiyoDecision

        Action = self._action_enum
        germinate_actions = self._germinate_actions
        candidates = [Action.WAIT]
        weights = [1.0]
        holding_seeds = []
        available_slots = getattr(signals, "available_slots", 0)

        if available_slots > 0:
            candidates.extend(germinate_actions)
            weights.extend([1.0 + self.germinate_bias] * len(germinate_actions))

        if active_seeds:
            candidates.append(Action.PRUNE)
            weights.append(1.0)
            holding_seeds = [seed for seed in active_seeds if seed.stage == SeedStage.HOLDING]
            if holding_seeds:
                candidates.append(Action.FOSSILIZE)
                weights.append(1.0)

        action = self._rng.choices(candidates, weights=weights, k=1)[0]
        target_seed_id = None
        if action == Action.PRUNE and active_seeds:
            target_seed_id = self._rng.choice(active_seeds).seed_id
        elif action == Action.FOSSILIZE and holding_seeds:
            target_seed_id = self._rng.choice(holding_seeds).seed_id

        return TamiyoDecision(action=action, target_seed_id=target_seed_id, reason="random")


def _run_mock_episode(
    *,
    policy,
    max_epochs: int,
    max_batches: int | None,
    base_seed: int,
    task: str,
) -> list[float]:
    from esper.runtime import get_task_spec
    from esper.simic.training.helpers import run_heuristic_episode

    task_spec = get_task_spec(task)
    trainloader, testloader = task_spec.create_dataloaders(
        batch_size=32,
        num_workers=0,
        mock=True,
    )
    policy.reset()
    _, _, episode_rewards = run_heuristic_episode(
        policy=policy,
        trainloader=trainloader,
        testloader=testloader,
        max_epochs=max_epochs,
        max_batches=max_batches,
        base_seed=base_seed,
        device="cpu",
        task_spec=task_spec,
        slots=["r0c1"],
        telemetry_config=None,
        telemetry_lifecycle_only=True,
    )
    return episode_rewards


def _run_mock_episodes(
    *,
    policy_factory,
    episodes: int,
    max_epochs: int,
    max_batches: int | None,
    base_seed: int,
    task: str,
) -> list[float]:
    rewards: list[float] = []
    for idx in range(episodes):
        policy = policy_factory(idx)
        episode_rewards = _run_mock_episode(
            policy=policy,
            max_epochs=max_epochs,
            max_batches=max_batches,
            base_seed=base_seed + idx * 1000,
            task=task,
        )
        rewards.extend(episode_rewards)
    return rewards


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 5 reward calibration helper")
    parser.add_argument(
        "--telemetry-base",
        type=Path,
        default=Path("telemetry"),
        help="Base directory containing telemetry_*/events.jsonl or a JSONL file",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Limit to last N telemetry directories (0 = no limit)",
    )
    parser.add_argument(
        "--max-events",
        type=int,
        default=0,
        help="Stop after N events (0 = no limit)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output path for JSON summary",
    )
    parser.add_argument(
        "--mock-heuristic",
        action="store_true",
        help="Run mock episodes with heuristic policy for reward scale",
    )
    parser.add_argument(
        "--mock-random",
        action="store_true",
        help="Run mock episodes with random policy for reward scale",
    )
    parser.add_argument(
        "--random-germinate-bias",
        type=float,
        default=0.0,
        help="Extra weight for germinate actions in random policy (0 = uniform)",
    )
    parser.add_argument(
        "--mock-episodes",
        type=int,
        default=1,
        help="Number of mock episodes to run per policy",
    )
    parser.add_argument(
        "--mock-epochs",
        type=int,
        default=3,
        help="Max epochs per mock episode",
    )
    parser.add_argument(
        "--mock-batches",
        type=int,
        default=4,
        help="Max batches per epoch for mock runs",
    )
    parser.add_argument(
        "--mock-task",
        type=str,
        default="cifar_baseline",
        help="Task name for mock runs",
    )
    parser.add_argument(
        "--mock-seed",
        type=int,
        default=123,
        help="Base seed for mock runs",
    )
    args = parser.parse_args()

    event_paths = _find_event_paths(args.telemetry_base, args.limit)
    run_mock = args.mock_heuristic or args.mock_random
    if not event_paths and not run_mock:
        print("No telemetry events.jsonl files found.")
        return 1

    reward_fields = [
        "total_reward",
        "compute_rent",
        "action_shaping",
        "pbrs_bonus",
        "bounded_attribution",
        "stage_bonus",
        "terminal_bonus",
    ]

    reward_values: dict[str, list[float]] = {field: [] for field in reward_fields}
    reward_abs: dict[str, list[float]] = {field: [] for field in reward_fields}
    alpha_by_seed: dict[str, float] = {}
    alpha_deltas: list[float] = []

    if event_paths:
        processed = 0
        for path in event_paths:
            with path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    if args.max_events and processed >= args.max_events:
                        break
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        event = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    processed += 1

                    data = event.get("data", {})
                    event_type = event.get("event_type")
                    if event_type == "REWARD_COMPUTED":
                        for field in reward_fields:
                            if field in data and isinstance(data[field], (int, float)):
                                reward_values[field].append(float(data[field]))
                                reward_abs[field].append(abs(float(data[field])))

                    seed_id = event.get("seed_id")
                    alpha = data.get("alpha")
                    if seed_id and isinstance(alpha, (int, float)):
                        alpha_val = float(alpha)
                        if seed_id in alpha_by_seed:
                            alpha_deltas.append(alpha_val - alpha_by_seed[seed_id])
                        alpha_by_seed[seed_id] = alpha_val
            if args.max_events and processed >= args.max_events:
                break

    summary = {
        "events": [str(path) for path in event_paths],
        "reward": {field: _summarize(values) for field, values in reward_values.items()},
        "reward_abs": {field: _summarize(values) for field, values in reward_abs.items()},
        "alpha_delta": _summarize(alpha_deltas),
        "alpha_delta_abs": _summarize([abs(v) for v in alpha_deltas]),
        "alpha_delta_sq": _summarize([v * v for v in alpha_deltas]),
    }

    mock_batches = None if args.mock_batches <= 0 else args.mock_batches
    mock_topology = None

    if run_mock:
        from esper.runtime import get_task_spec
        from esper.tamiyo.heuristic import HeuristicTamiyo

        mock_topology = get_task_spec(args.mock_task).topology

        summary["mock"] = {
            "config": {
                "episodes": args.mock_episodes,
                "max_epochs": args.mock_epochs,
                "max_batches": mock_batches,
                "task": args.mock_task,
                "topology": mock_topology,
                "base_seed": args.mock_seed,
                "random_germinate_bias": args.random_germinate_bias,
            },
            "results": {},
        }

    if args.mock_heuristic:
        if mock_topology is None:
            raise RuntimeError("mock_topology missing; set --mock-task for mock runs.")
        heuristic_rewards = _run_mock_episodes(
            policy_factory=lambda _: HeuristicTamiyo(topology=mock_topology),
            episodes=args.mock_episodes,
            max_epochs=args.mock_epochs,
            max_batches=mock_batches,
            base_seed=args.mock_seed,
            task=args.mock_task,
        )
        summary["mock"]["results"]["heuristic"] = {
            "reward": _summarize(heuristic_rewards),
            "reward_abs": _summarize([abs(v) for v in heuristic_rewards]),
        }

    if args.mock_random:
        if mock_topology is None:
            raise RuntimeError("mock_topology missing; set --mock-task for mock runs.")
        random_rewards = _run_mock_episodes(
            policy_factory=lambda idx: RandomTamiyo(
                topology=mock_topology,
                seed=args.mock_seed + idx,
                germinate_bias=args.random_germinate_bias,
            ),
            episodes=args.mock_episodes,
            max_epochs=args.mock_epochs,
            max_batches=mock_batches,
            base_seed=args.mock_seed,
            task=args.mock_task,
        )
        summary["mock"]["results"]["random"] = {
            "reward": _summarize(random_rewards),
            "reward_abs": _summarize([abs(v) for v in random_rewards]),
        }

    median_reward = summary["reward_abs"]["total_reward"].get("median", 0.0)
    if not median_reward and "mock" in summary:
        for mode in ("heuristic", "random"):
            median_reward = summary["mock"]["results"].get(mode, {}).get("reward_abs", {}).get("median", 0.0)
            if median_reward:
                break
    median_rent = summary["reward_abs"]["compute_rent"].get("median", 0.0)
    p90_delta_sq = summary["alpha_delta_sq"].get("p90", 0.0)

    target_shock = 0.1 * median_reward if median_reward > 0 else 0.01
    if p90_delta_sq > 0:
        shock_coef = target_shock / max(p90_delta_sq, 1e-6)
        shock_note = "Shock targets ~10% of median abs total_reward at p90(alpha_delta^2)."
    else:
        shock_coef = 0.0
        shock_note = "alpha_delta stats missing; shock_coef is a placeholder until telemetry is available."
    base_slot_rent = max(median_rent, 0.05 * median_reward)

    summary["suggested"] = {
        "base_slot_rent": base_slot_rent,
        "shock_coef": shock_coef,
        "shock_target_magnitude": target_shock,
        "notes": shock_note,
    }

    output = json.dumps(summary, indent=2, sort_keys=True)
    print(output)

    if args.output is not None:
        args.output.write_text(output, encoding="utf-8")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
