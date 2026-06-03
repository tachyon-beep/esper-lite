# Development Guide

This guide covers common commands, testing procedures, and configuration details for working with Esper.

## Quick Start

Requires Python 3.11+ and PyTorch.

```bash
uv sync
```

## Running Esper

Esper primarily runs via the `esper.scripts.train` entry point.

### Heuristic Baseline
To run with a non-learned (heuristic) policy:

```bash
PYTHONPATH=src uv run python -m esper.scripts.train heuristic \
  --task cifar_baseline --episodes 1
```

### PPO Training (Learned Policy)
To run the Simic RL training loop using PPO:

```bash
PYTHONPATH=src uv run python -m esper.scripts.train ppo \
  --task cifar_baseline \
  --rounds 100 \
  --envs 4 \
  --episode-length 150 \
  --device cuda:0
```

## CLI Configuration Flags

### Core Scaling
*   `--rounds N` (100): PPO update rounds.
*   `--envs K` (4): Parallel environments per round.
*   `--episode-length L` (150): Steps per env per round (also the LSTM horizon).
*   `--ppo-epochs E` (1): PPO update passes over rollout data.
*   `--memory-size H` (512): LSTM hidden size.

### Hardware & Performance
*   `--device` (`cuda:0`): Policy device.
*   `--devices` (none): Multi-GPU env devices (e.g., `cuda:0 cuda:1`).
*   `--num-workers`: DataLoader workers.
*   `--compile-mode` (`default`): `torch.compile` mode.

### Monitoring & UI
*   `--sanctum`: Textual TUI for debugging.
*   `--overwatch`: Web dashboard.
*   `--telemetry-dir PATH`: Write telemetry artefacts.
*   `--wandb`: Enable Weights & Biases tracking.

## Reward Configuration

Rewards provide selection pressure in `simic`.

| Parameter | Default | Description |
| :--- | :--- | :--- |
| `reward_mode` | `"shaped"` | Strategy (e.g., `shaped`, `simplified`, `basic`, `basic_plus`, `sparse`). |
| `reward_family` | `"contribution"` | Counterfactual contribution vs direct loss delta. |
| `param_budget` | `500000` | Budget for seeds. Penalty applies if exceeded. |
| `rent_host_params_floor` | `200` | Normalization floor to prevent tiny hosts from being crushed by rent shock. |

*   **Shaped**: Default. Rich feedback but potential for Goodhart's law.
*   **Basic Plus**: Includes a **Drip Mechanism** (post-fossilization accountability) to prevent premature fossilization gaming.
*   **Sparse**: Terminal-only reward.

## A/B Testing
You can use `--dual-ab` to train separate policies on separate GPUs (e.g., comparing `shaped-vs-simplified`).

## Testing

```bash
uv run pytest -q
```
