# Reward Compare + Reward Mix Design

Date: 2025-12-20
Status: approved for implementation

## Summary

This design replaces the old A/B reward split with a renamed **reward mix** mode and
adds a true **reward compare** mode that trains multiple isolated policies in a single
run. Reward compare divides envs evenly across variants and drops any remainder envs
(e.g., 14 envs with a,b,c => 12 total, 4 per variant). All training state is isolated
per variant; only telemetry is shared.

## Goals

- Keep a single base TrainingConfig for normal (non-experimental) runs.
- Rename reward A/B split to **reward mix** and remove old names/flags.
- Implement **reward compare** with multiple policy variants trained concurrently.
- Preserve reward-system integrity: no shared learning state across variants.
- Use telemetry tagging to separate cohorts in TUI/dashboard.

## Non-goals

- Weighted reward-mix ratios (parked for later).
- Cross-policy parameter sharing or distillation.
- Backwards compatibility shims for old flags or fields.

## Configuration shape

Top-level TrainingConfig remains the canonical non-experimental policy. Experimental
modes live under `experimental` and are optional.

```json
{
  "reward_mode": "shaped",
  "reward_family": "contribution",
  "experimental": {
    "reward_mix": {
      "modes": ["shaped", "sparse", "simplified", "shaped"],
      "name": "mix-1"
    },
    "reward_compare": {
      "variants": [
        {"name": "shaped", "override": {"reward_mode": "shaped"}},
        {"name": "sparse", "override": {"reward_mode": "sparse"}},
        {"name": "minimal", "override": {"reward_mode": "minimal"}}
      ]
    }
  }
}
```

Rules:
- `reward_mix` and `reward_compare` are mutually exclusive.
- `reward_compare` must define at least two variants.
- `reward_compare.override` is reward-only (allow-list):
  `reward_mode`, `reward_family`, `sparse_reward_scale`, `param_budget`,
  `param_penalty_weight`.

## CLI surface

- Replace `--ab-test` with `--reward-mix`.
- `--reward-mix shaped,sparse` is the common case; this expands to a per-env list
  only when the split is even. For uneven cases, users must specify config JSON.
- No `--ab-test`, `ab_reward_modes`, or `ab_group` compatibility remains.

## Training architecture

### Reward mix (single policy)

- Single PPO agent, optimizer, rollout buffer, normalizers.
- Per-env reward configs derived from `reward_mix.modes` (or CLI expansion).
- Behavior is a strict rename of the current A/B reward split.

### Reward compare (multi-policy)

- One PPO agent per variant, each fully isolated:
  - Policy/value network
  - Optimizer
  - Rollout buffer
  - Observation normalizer
  - Reward normalizer
  - Env state and signal tracker
- Env count is floored: `envs_per_variant = n_envs // variants`.
- Remainder envs are not created; emit info telemetry and console note.
- Each variant uses its own batch iterator (no shared iterator state). If
  comparability is desired, iterators use the same seed to align batch order.

## Telemetry and UI

Telemetry tagging:
- `policy_variant`: name from reward_compare.variant
- `reward_mix_group`: reward mode name for reward_mix

UI behavior:
- TUI/dashboard display a cohort marker per env using `policy_variant` in compare
  mode and `reward_mix_group` in mix mode. If both exist, compare takes precedence.

## Validation and error handling

- Error if both `reward_mix` and `reward_compare` are present.
- Error if `reward_compare` has < 2 variants or `envs_per_variant == 0`.
- Error on any override key outside the allow-list.
- Error if reward rules are violated (e.g., reward_family=loss and reward_mode!=shaped).
- Warn/info when remainder envs are dropped and report effective env count.

## Migration and removals

- Remove `--ab-test` CLI flag.
- Remove `ab_reward_modes` from TrainingConfig and config JSON.
- Remove `ab_group` telemetry field, replace with `reward_mix_group`.
- Update docs and dashboards to new field names. No compatibility shims.

## Testing plan

- Unit tests:
  - Config parsing and validation (mutual exclusivity, override allow-list).
  - Env split math and remainder handling for reward_compare.
  - Telemetry tagging (`policy_variant`, `reward_mix_group`).
- Integration tests:
  - Reward mix: per-env reward modes applied without changing policy count.
  - Reward compare: separate policies trained concurrently and isolated.
- UI tests:
  - Sanctum/Overwatch render cohort markers for both modes.

