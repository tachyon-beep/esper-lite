# Reward-Efficiency Proof Packet

- Telemetry dir: `telemetry/proof-rehearsal`
- Verdict: `BLOCKED`

## Cohorts

- `telemetry_2026-06-13_081513` group `A`: task=cifar_impaired, reward_mode=shaped, envs=2, env_episodes=4, episode_length=25
- `telemetry_2026-06-13_081513` group `B`: task=cifar_impaired, reward_mode=simplified, envs=2, env_episodes=4, episode_length=25

## Reproduction Commands

- Equivalent training command: `PYTHONPATH=src uv run python -m esper.scripts.train ppo --task cifar_impaired --dual-ab shaped-vs-simplified --rounds 2 --envs 2 --episode-length 25 --telemetry-dir telemetry/proof-rehearsal`
- Packet command: `PYTHONPATH=src uv run python scripts/proof_packet.py --telemetry-dir telemetry/proof-rehearsal --output <packet.md>`

## Confounder Ledger

- BLOCKING `VALUE_COLLAPSE_DETECTED` run `telemetry_2026-06-13_081513` group `A` env=None episode=4 batch=2: explained_variance=-0.000 < 0.1 (at 100% training)
- BLOCKING `VALUE_COLLAPSE_DETECTED` run `telemetry_2026-06-13_081513` group `B` env=None episode=2 batch=1: explained_variance=-0.000 < 0.0 (at 50% training)
- BLOCKING `VALUE_COLLAPSE_DETECTED` run `telemetry_2026-06-13_081513` group `B` env=None episode=4 batch=2: explained_variance=-0.007 < 0.1 (at 100% training)
- BLOCKING `GRADIENT_ANOMALY` run `telemetry_2026-06-13_081513` group `B` env=None episode=4 batch=2: norm_drift=0.656 > 0.5

## Learnability Gate

- PPO updates include per-head learnability telemetry.

## Lifecycle Efficiency

- `telemetry_2026-06-13_081513` group `A`: germinated=10, fossilized=0, pruned=6, stage_changes=33, fossilize_rate=0.0, prune_rate=0.6
- `telemetry_2026-06-13_081513` group `B`: germinated=9, fossilized=0, pruned=8, stage_changes=34, fossilize_rate=0.0, prune_rate=0.8888888888888888

## Accuracy ROI

- `telemetry_2026-06-13_081513` group `A` reward=shaped: episodes=4, mean_final_accuracy=20.90386284722222, mean_param_ratio=0.7337662337662337, mean_accuracy_roi=35.70813301282051
- `telemetry_2026-06-13_081513` group `B` reward=simplified: episodes=4, mean_final_accuracy=20.665147569444446, mean_param_ratio=0.025974025974025976, mean_accuracy_roi=209.29361979166666

## Decision

The proof run cannot support a continue/revise/stop product verdict until blocking confounders are cleared.
