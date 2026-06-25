# Run Sheet — Controller-Contribution Gate (off-switch / fixed-schedule vs morphogenetic)

```yaml
id: controller-contribution-gate-runsheet
created: 2026-06-23
owner: <you>
supports_plan: docs/plans/ready/2026-06-22-ppo-learning-gate-reward-efficiency-statistics.md
filigree_package: esper-lite-a2abff5ec5
status: ready-to-run
```

## 0. Why this gate runs before any algorithm decision

The open question is "is recurrent PPO the right algorithm?". That question is **undecidable** until we know the learned controller is contributing *at all*. If a disabled controller (off-switch) reaches the same host accuracy as the learned controller, then no algorithm change — PPO, SAC, GRPO, Dreamer — can help, because there is no controllable signal to optimize. This gate is the cheapest experiment that can *kill* the algorithm-pivot conversation, so it goes first.

**The gate measures terminal host validation accuracy per arm, across ≥10 seeds.** It does *not* rely on explained-variance (EV) — EV is a within-training thermometer and is denominator-unstable on low-return-variance substrates (see §8). Final accuracy is the outcome that matters and is comparable across arms.

## 1. Decision rule (read this first — it defines success)

Primary metric: **terminal host validation accuracy** (best stable val-acc per run), aggregated across seeds.

Let `M` = morphogenetic (learned controller), `OFF` = off-switch (controller forced to WAIT, no growth), `FIX` = fixed-schedule (hand-coded growth to a comparable final shape).

| Outcome (across ≥10 seeds, with CI separation) | Interpretation | Consequence for the algorithm question | proof_packet verdict (≈) |
|---|---|---|---|
| `M` does **not** beat `OFF` | Controller contributes nothing; growth ≈ no-growth | **Algorithm pivot is moot.** Problem is signal/reward/observability, not the optimizer. Stop debating PPO-vs-X. | `STOP_THEORY` |
| `M` beats `OFF` but **not** `FIX` | Controller adds capacity, but a dumb static schedule does as well or better → the *learned policy* adds no value over a heuristic | Not an algorithm-family problem; reward/credit-assignment problem. Pursue the in-PPO reforms (de-shape value target, per-head adv-norm). | `REVISE_ALGORITHM` |
| `M` beats **both** `OFF` and `FIX` | Controller demonstrably learns useful, non-trivial morphogenesis | Algorithm questions (sample efficiency, value-target noise) are now **legitimately on the table** | `CONTINUE` |

Pre-registered threshold (commit before running, do not move after seeing data):
- **Effect**: median(`M` final-acc) − median(`OFF` final-acc) ≥ **2.0 pp** (one preset's `improvement_threshold`), and the 95% CI of the difference excludes 0.
- **Same** test for `M` vs `FIX`.
- Substrate gate is judged **per substrate** (impaired and baseline are reported separately; see §5).

## 2. Arms (map to existing proof-baseline cohorts)

These already exist as first-class cohorts in `build_blueprint_health_proof_plan` (`src/esper/simic/training/proof_baselines.py`) and are enforced inside the PPO runtime before action sampling:

| Arm | Cohort `mode` | `lifecycle_policy` | Learns? | Role |
|---|---|---|---|---|
| **OFF** (off-switch) | `off_switch` | `force_wait_only` | no | host-alone floor |
| **FIX** (fixed-schedule) | `fixed_schedule` | `apply_declared_lifecycle_schedule` (`FIXED_SCHEDULE_GERMINATE_R0C0_V1`) | no | heuristic-growth control |
| static-initial | `static_initial` | `freeze_initial_topology` | no | second no-growth control |
| static-final | `static_final` | `freeze_replayed_final_topology` | no | "same final shape, no learning" control |
| **M-shaped** (morphogenetic) | `lockstep_reward_ab_A` | `paired_lockstep_reward_comparison` | **yes** | treatment (shaped reward) |
| M-simplified | `lockstep_reward_ab_B` | `paired_lockstep_reward_comparison` | **yes** | treatment (simplified reward) |

> The fixed schedule germinates in **`r0c0`**, so **every arm must use `--slots r0c0`** for a fair comparison (note: the default config slot is `r0c1` — override it).

## 3. Held-constant manifest (matched-cohort contract — child #4 of the plan)

Every arm, every seed, must share: `task`, `slots=[r0c0]`, `param_budget`, `max_epochs` (host horizon), `n_envs`, `lstm_hidden_dim`, precision (`amp`, `amp_dtype`), `compile_mode`, telemetry provenance, and the seed matrix. The **only** things allowed to vary are the arm's `proof_baseline_mode`/`lifecycle_policy` and (for M-shaped vs M-simplified) the reward mode. Record the exact values in §11.

## 4. Seed matrix (≥10) and the pairing decision

Use **≥10 base seeds**: `42 43 44 45 46 47 48 49 50 51` (extend to 15 if a pilot shows high variance).

**SELECTED: Path B (paired).** Path A is retained as a zero-code fallback only.

- **Path B — paired (SELECTED), needs a thin driver (prerequisite — not yet implemented).** Run `OFF`, `FIX`, `static_initial`, and `M` at the **same host-init seed** per replicate, ≥10 replicates → compare with **Wilcoxon signed-rank** (paired). At this substrate's contribution CoV (~2.66), pairing removes host-init variance — the dominant nuisance — and is worth far more than extra seeds. It also lets you run the non-learning arms cheaply (see §9). Driver spec in §7B. **Launch is blocked until the driver exists.**
- **Path A — built-in, runnable today, zero new code (UNPAIRED across arms) — FALLBACK.** Each `--proof-baseline` invocation runs all 6 cohorts at `base_seed = --seed`, but each cohort is offset (off_switch=+10k, fixed=+40k, lockstep=+50k), so within a base seed the arms are **not** host-init matched. Across 10 base seeds you get 10 independent samples per arm → compare with **Mann-Whitney U** (unpaired). Use only if you need a result before the paired driver is ready.

## 5. Substrates

Run the gate on **both**, reported separately:
- **`cifar_impaired`** — the contested substrate (where EV stalled). This is the substrate the verdict is about.
- **`cifar_baseline`** — positive control. The controller is *known* to lift here, so `M` must beat `OFF` on baseline; if it doesn't, the harness/metric is broken, not the controller.

## 6. Pre-flight checklist

- [x] CIFAR-10 present (`./data/cifar-10-batches-py`) — verified 2026-06-23. (P0-2: missing CIFAR **aborts**, no synthetic fallback.)
- [x] 2× RTX 4060 Ti (16 GB). cuda:1 idle, cuda:0 lightly used — verify free at launch with `nvidia-smi`.
- [ ] Working tree clean / on the intended commit; record `git rev-parse HEAD` in §11.
- [ ] **Smoke run** (1 seed, tiny budget) end-to-end through `proof_packet` BEFORE committing GPU-days — see §7C.
- [ ] Decide Path A vs B and substrate scope (§11).

## 7A. Path A — built-in `--proof-baseline` loop (run today)

One invocation per (substrate, base seed). Write all seeds of a substrate into **one parent telemetry dir** so Karn aggregates them.

```bash
# impaired substrate, 10 seeds, all into one telemetry dir
TELDIR=telemetry/gate-impaired-2026-06-23
for S in 42 43 44 45 46 47 48 49 50 51; do
  PYTHONPATH=src uv run python -m esper.scripts.train ppo \
    --preset cifar_impaired \
    --task cifar_impaired \
    --slots r0c0 \
    --proof-baseline blueprint-health \
    --seed "$S" \
    --telemetry-dir "$TELDIR" \
    --device cuda:0 \
    --gpu-preload --amp --amp-dtype bfloat16 --compile-mode default
done
```

Run the `cifar_baseline` substrate in parallel on `cuda:1` (separate `TELDIR=telemetry/gate-baseline-2026-06-23`, `--preset cifar_baseline --task cifar_baseline --device cuda:1`).

All standard ppo flags (`--gpu-preload`, `--amp`, `--compile-mode`) propagate to every cohort. `--proof-baseline` and `--dual-ab` are mutually exclusive.

## 7B. Path B — paired driver (recommended; spec)

A thin driver (mirror `scripts/ev_liftoff_experiment.py`) that, for each `seed` in the matrix and each arm, calls `train_ppo_vectorized(**kwargs)` with a **matched `seed`**, a distinct `group_id`, and a shared `proof_baseline_pair_id=f"gate-{substrate}-seed{seed}"`. Arm → kwargs:

```text
OFF:            proof_baseline_mode="off_switch",  proof_baseline_lifecycle_policy="force_wait_only"
static_initial: proof_baseline_mode="static_initial", proof_baseline_lifecycle_policy="freeze_initial_topology"
FIX:            proof_baseline_mode="fixed_schedule", proof_baseline_lifecycle_policy="apply_declared_lifecycle_schedule",
                proof_baseline_schedule_id/_hash/_version/_action_count = FIXED_SCHEDULE_GERMINATE_R0C0_* (leyline.proof_baselines)
M (treatment):  no proof_baseline_mode  (free learned controller), reward_mode="shaped"
```

Shared across arms per replicate: `task`, `slots=["r0c0"]`, `max_epochs`, `n_envs`, precision, `compile_mode`, `telemetry_dir`, and **`seed`**. Non-learning arms (OFF/FIX/static) can use a **short** `n_episodes` (see §9); M uses the full learning budget.

## 7C. Smoke run (do before either path)

```bash
PYTHONPATH=src uv run python -m esper.scripts.train ppo \
  --preset cifar_impaired --task cifar_impaired --slots r0c0 \
  --proof-baseline blueprint-health --seed 42 \
  --telemetry-dir /tmp/gate-smoke --device cuda:1 \
  --gpu-preload --amp --amp-dtype bfloat16
# then confirm the packet ingests and does NOT spuriously block:
PYTHONPATH=src uv run python scripts/proof_packet.py \
  --telemetry-dir /tmp/gate-smoke --proof-profile reward-efficiency \
  --output /tmp/gate-smoke-packet.md
```
A short host budget (lower `max_epochs` via a `--config-json`) keeps the smoke under ~15 min; its only job is to prove the pipeline + packet ingestion before the long run.

## 8. Analysis & verdict

**Automated:** point the proof packet at each substrate's parent dir.
```bash
PYTHONPATH=src uv run python scripts/proof_packet.py \
  --telemetry-dir telemetry/gate-impaired-2026-06-23 \
  --proof-profile reward-efficiency \
  --output docs/analysis/gate-impaired-2026-06-23-packet.md
```
The packet emits one of `CONTINUE` / `REVISE_ALGORITHM` / `STOP_THEORY` / `BLOCKED_{INSTRUMENTATION,PRECISION,MATH,MECHANICS}`. A `BLOCKED_*` is a pipeline failure, **not** evidence — fix and re-run (§10). The packet already blocks if any required baseline mode is missing (`missing_required_baseline_modes`), which is why the full 6-cohort plan must be run, not a subset.

**Manual cross-check (don't trust a single number):** read terminal accuracy per (arm, seed) from Karn and compute the §1 statistics yourself.
```
# confirm columns first: mcp__esper-karn__describe_view episode_outcomes ; ... runs
# sketch: terminal val-acc per cohort, grouped by proof_baseline_mode, across seeds
SELECT r.proof_baseline_mode, eo.seed, MAX(eo.final_val_accuracy) AS final_acc
FROM episode_outcomes eo JOIN runs r USING (run_dir)
WHERE r.proof_baseline_pair_id LIKE 'gate-%'   -- or = 'blueprint-health-proof' for Path A
GROUP BY r.proof_baseline_mode, eo.seed
ORDER BY r.proof_baseline_mode, eo.seed;
```
Then per arm report: per-seed values, mean ± std, median + IQR, and the M−OFF / M−FIX difference with a bootstrap 95% CI (paired Wilcoxon for Path B, Mann-Whitney for Path A). Apply the §1 decision rule.

## 9. Measurement caveats (banked from prior runs — do not re-learn these the hard way)

- **`metrics["entropy"]` is a coefficient-weighted bonus, not raw entropy.** Do not read it as exploration health and do not "force annealing". Use per-head entropies / `clip_fraction` if you need an exploration read. (This gate keys on final accuracy, so it's mostly immune — but reviewers will ask.)
- **EV is denominator-unstable** when batch return-variance is tiny (`EV = 1 − Var(R−V)/Var(R)`). Expected on `OFF`/impaired (flat returns). Treat low-return-variance arms' EV as **inconclusive**, never as "critic broken". This is exactly why the gate uses accuracy, not EV.
- **Impaired floor:** if `OFF` and `M` both sit at the host-alone band (~21%), the impaired substrate may simply have no headroom — that is itself a `STOP_THEORY` signal *for impaired*, but check the baseline positive control before generalizing.

## 10. Abort / blocker conditions

- Any arm crashes mid-run (watch for the LSTM hidden/feature-batch mismatch class on desynced truncation) → capture the traceback, do not silently drop the seed.
- `proof_packet` returns `BLOCKED_INSTRUMENTATION` (zero `PPO_UPDATE_COMPLETED` rows), `BLOCKED_PRECISION` (missing `amp_enabled`/`amp_dtype`), `BLOCKED_MATH`, or `BLOCKED_MECHANICS` → fix the pipeline and re-run; report as a blocker, not a result.
- GPU contention with another job on cuda:0 → pin the gate to a free device.

## 11. Run record (fill in at launch)

```
commit:            <git rev-parse HEAD>
path:              B (paired)        [SELECTED 2026-06-23]
substrates:        cifar_impaired  +  cifar_baseline (positive control)   [SELECTED]
seeds:             42..51   (n = 10; extend to 15 if pilot variance is high)
slots:             r0c0
preset/task:       impaired: cifar_impaired/cifar_impaired ; control: cifar_baseline/cifar_baseline
param_budget:      ______   max_epochs: ____   n_envs: ____
precision:         amp=on dtype=bfloat16   compile=default
telemetry dirs:    impaired=____________  baseline=____________
launched:          <date>   by: <you>
```

## 12. Cost model & schedule

- `cifar_impaired` preset → `n_episodes=100`. Path A runs ~6 full cohorts + 1 short static-final source per base seed.
- Rough per-cohort ~1.5–3 h with `--gpu-preload --amp` (PPO/LSTM/TBPTT compute dominates even on the tiny impaired host). ⇒ Path A ≈ **8–17 GPU-h per base seed**, ×10 ≈ **80–170 GPU-h**; on 2 GPUs ≈ **2–4 days wall-clock**.
- **Path B is materially cheaper**: the non-learning arms (`OFF`/`FIX`/`static`) reach steady accuracy almost immediately, so run them at a short `n_episodes` (≈ a handful) — only `M` needs the full 100-episode learning budget. This roughly halves the bill and is a second reason to prefer Path B.
- Pilot first: 2–3 seeds to get a variance estimate, then decide whether 10 or 15 seeds are needed for CI separation at the 2.0 pp threshold.

## Decisions (resolved 2026-06-23)
1. **Path: B (paired).** Wilcoxon signed-rank across matched host-init seeds.
2. **Substrates: both** — `cifar_impaired` (contested) + `cifar_baseline` (positive control), one per GPU.
3. **Seeds: 10** (42–51), extend to 15 if the pilot shows high variance.

## Prerequisite before launch
- [ ] **Paired-gate driver not yet implemented** (§7B spec). This is the only blocker between this run sheet and launch. Until it exists, the gate can only be run via the Path A fallback.
- [ ] After the driver lands: pilot (2–3 seeds) → confirm `proof_packet` ingests + variance estimate → full 10-seed run.
