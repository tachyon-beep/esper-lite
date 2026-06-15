# BF16 PPO Artifact Quarantine

Date: 2026-06-15

## Cutoff

Commit `81a3a3a7db8e7e79cd76279934c721b84f61c999` fixed the BF16 autocast
cache poisoning bug at `2026-06-14 10:41:03 +1000`:

```text
fix(simic): FP32 Q-value telemetry forward to stop autocast cache poisoning
```

Any pre-cutoff PPO artifact is proof-grade only when its own metadata proves it
was not a BF16 AMP policy run.

## Local Telemetry Audit

I scanned every local `telemetry/**/events.jsonl` `TRAINING_STARTED` row with a
bounded extractor that preserves explicit `false` values. No local telemetry row
is confirmed BF16. Every local row with AMP metadata reports:

```text
amp_enabled=false
amp_dtype=null
```

This covers the local reward-efficiency, proof-rehearsal, health-audit,
codex-smoke, and January 2026 PPO telemetry directories currently present under
`telemetry/`.

## Local Checkpoint Audit

The two local checkpoints are:

```text
models/3slot-3seed-imparied-basic_plus-130126-1435.pt
models/3slot-3seed-imparied-basic_plus-130126-1800.pt
```

Both are pre-cutoff artifacts. They contain `config` and `metadata` dictionaries,
but no `amp_enabled`, `amp_dtype`, or equivalent precision provenance field.

## Disposition

- Confirmed BF16-corrupted local telemetry: none found.
- Confirmed non-BF16 local telemetry: all local `TRAINING_STARTED` rows with
  precision metadata show AMP disabled.
- Pre-cutoff checkpoints without precision provenance: quarantined for
  policy-learning proof claims. They may be useful for exploratory debugging, but
  they must not be used as evidence that PPO policy/value heads trained correctly.

## Proof Rule

Reward-efficiency or PPO-learning verdicts must cite only runs whose
`TRAINING_STARTED` metadata proves the precision mode. A pre-cutoff artifact with
missing precision metadata is not proof-grade even when its telemetry appears
healthy.
