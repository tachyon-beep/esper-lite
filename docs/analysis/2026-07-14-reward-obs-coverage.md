# Reward–Observation Coverage Report

> **Main-session review (2026-07-14): ACCEPTED.** The #1 finding (counterfactual_total_improvement aliases
> obs dim +12 ONLY in single-seed episodes; multi-seed → the policy is blind to the signal every
> anti-gaming/fossilize/prune gate keys on) verifies against first-hand structures: the CF-matrix mask
> validation from the IQR read (n=1 solo ≡ all-off) and the personally-read reward gates
> (contribution.py:484-509). The velocity cross-correction against the pytorch audit doc reflects post-fix
> reality correctly (PDR-0100 disposition). Register updated: the counterfactual-gate-signal gap and
> acc_at_germination join the Obs V5 candidate list; num_contributing_fossilized noted low-priority.

Date: 2026-07-14. Defect register A4 (PDR-0100). Tracker: esper-lite-c739c3ab97 (partial).
Analysis only — no code changes. Author: dispatched agent (main-session-reviewable; every load-bearing
number re-derived from source first-hand, per the "personally verify" standing rule).

**Question answered.** For every input consumed by (a) the SHAPED reward path
(`compute_contribution_reward`) and (b) the heuristic controller (`HeuristicTamiyo`), does the signal reach
the RL policy's observation? The payload class is **ACCIDENTALLY-UNPLUMBED**: *the policy optimizes signals it
cannot see.*

## Method & classification rule

The four classes, with a **crisp, non-overlapping boundary** (the INFERABLE/UNPLUMBED line is what makes or
breaks this report — "the LSTM could theoretically latch it in hidden state" does **not** earn INFERABLE, or
the payload class collapses to empty):

- **OBSERVED-DIRECTLY** — reaches the obs as a named feature dim in `batch_obs_to_features`
  (`features.py:837–997`). Cite the dim.
- **INFERABLE-FROM-OBSERVED** — a *stateless* (or single-step-diff) function of features present in the
  **current obs or the 5-deep history window** (`loss_history[5]` dims 3–7, `acc_history[5]` dims 8–12).
- **INTENTIONALLY-HIDDEN** — design says the policy must not see it. Cite the design decision.
- **ACCIDENTALLY-UNPLUMBED** — exists, no design reason to hide, does not reach obs, and reconstruction needs a
  **self-maintained running counter or data older than the 5-window.** This is the payload.

Anchor for the rule: the task pre-classifies `plateau_epochs` — a deterministic function of full history an
LSTM *could* latch — as ACCIDENTALLY-UNPLUMBED. That is the calibration tell; "recoverable if the event is
recent" goes in the note column, not the class.

The obs layout is taken as verified in `docs/analysis/2026-07-14-obs-audit-pytorch.md §0` and re-checked
against `features.py:837–997`. V3 (default) dims cited; per-slot offsets are `+N` off
`slot_offset = 24 + slot_idx*32`.

---

## Table 1 — SHAPED reward path (`compute_contribution_reward`, contribution.py:412)

Every argument in the signature + every `seed_info` field the body reads. Trainer source is
`action_execution.py` (the reward call site, :940–1090) and `vectorized_trainer.py` (the counterfactual
compute, :1355–1460). Confidence: **HIGH** for this group unless noted (call site + feature encoder traced end
to end).

| input | computed-where (file:line) | classification | obs dim / reason | note |
|-------|----------------------------|----------------|------------------|------|
| `seed_contribution` (per-slot LOO counterfactual) | `action_execution.py:944` = `val_acc − baseline_accs[slot]`; same qty set on metrics at `vectorized_trainer.py:1379,1393` | **OBSERVED-DIRECTLY** | **dim +12** (`counterfactual_contribution`) | The reward's PRIMARY attribution signal — observed. V3 coerces `None→0` (`features.py:906–909`) and clamps `/_IMPROVEMENT_CLAMP_PCT_PTS`; value identical to reward's otherwise. |
| `counterfactual_total_improvement` (all-seeds-OFF counterfactual) | `action_execution.py:970–972` = `val_acc − all_disabled_accs[i]`; `all_disabled_accs` set `vectorized_trainer.py:1407` | **SPLIT: OBSERVED (1 seed) / ACCIDENTALLY-UNPLUMBED (≥2 seeds)** | single active seed → all-off baseline == the LOO baseline → **aliases dim +12**; multiple seeds → all-off baseline diverges, reaches telemetry only (emitter `vt:1446`), **never obs** | **#1 finding.** Gates *every* anti-gaming branch: discount (`:495`), ratio penalty (`:499`), fossilize suppress (`:641`), blending warning (`:682`), fossilize legitimacy (`:838`), prune/fossilize shaping (`:1092,:1292`). In multi-seed episodes the policy is blind to the exact signal that decides fossilize vs prune. |
| `val_acc` (all-seeds-on acc) | `action_execution.py:1047` (`env_state.val_acc`; set `vt:1366`) | **OBSERVED-DIRECTLY** | **dim 2** (`val_accuracy/100`) | — |
| `seed_info.stage` | `action_execution.py:973` (`SeedInfo.from_seed_state`) | **OBSERVED-DIRECTLY** | **dims +1..+10** (per-slot stage one-hot) | Drives every stage-gated branch. |
| `seed_info.epochs_in_stage` | `action_execution.py:973` | **OBSERVED-DIRECTLY** | **dim +28** (`epochs_in_stage_norm`) | Legitimacy discount, holding-indecision ramp, PBRS progress. |
| `seed_info.seed_age_epochs` (`metrics.epochs_total`) | `types.py:65` | **OBSERVED-DIRECTLY** | **dim +30** (`seed_age_norm`) | Timing discount (`germination_epoch = epoch − seed_age_epochs`, `:587`) + prune age gate. |
| `seed_info.interaction_sum` | `types.py:66` | **OBSERVED-DIRECTLY** | **dim +22** (`interaction_sum/10`, clamped) | Synergy bonus (`:721`). Only the summed scalar reaches obs; pairwise `I_ij` does not (not a reward input either). |
| `epoch` | `action_execution.py:1049` | **OBSERVED-DIRECTLY** | **dim 0** (`epoch_norm`) | — |
| `max_epochs` | `action_execution.py:1050` | **OBSERVED-DIRECTLY** | **dim 0** (constant; dim0 = epoch/max_epochs) | Terminal-bonus gate `epoch==max_epochs`. |
| `stable_val_acc` (ESCROW only) | `action_execution.py:983–996` = `min(acc_history[-k:])` | **OBSERVED-DIRECTLY** | **dim 23** (`stable_val_acc`) | SHAPED passes `None`; obs carries it regardless. |
| `escrow_credit_prev` (ESCROW only) | `action_execution.py:997,1066` (`env_state.escrow_credit[slot]`) | **OBSERVED-DIRECTLY** | **dim +31** (`escrow_credit_prev`, symlog/7) | ESCROW-only in reward math. |
| `acc_delta` (`signals.metrics.accuracy_delta`) | `action_execution.py:1054`; `tracker.py:110` = 1-epoch diff | **INFERABLE-FROM-OBSERVED** | `acc_history[-1] − acc_history[-2]` (dims 8–12) | In SHAPED **telemetry-only** — sets `components.base_acc_delta` (`:453`), never enters reward math. |
| `alpha_delta_sq_sum` | `action_execution.py:904,1064` (`compute_rent_and_shock_inputs`) | **INFERABLE-FROM-OBSERVED** (MED) | stateless fn of per-slot Δalpha; per-slot **dim +20** (`alpha_velocity`) is the per-step Δalpha | Clamp `[-1,1]` on dim +20 loses precision on large alpha steps → sq-sum only approximately recoverable. Feeds `alpha_shock`. |
| `num_fossilized_seeds` (`env_state.seeds_fossilized`) | `action_execution.py:1057` | **INFERABLE-FROM-OBSERVED** (MED) | count of FOSSILIZED per-slot one-hot bits this step | Valid **iff** fossils retain a slot/report — confirmed: `action_execution.py:1010` iterates slots counting `stage==FOSSILIZED`, so the fossil block is present. Base **dim 15 aliases** it (sums HOLDING+FOSSILIZED, `features.py:825`). Feeds occupancy + fossil rent. |
| `n_active_seeds` | `action_execution.py:1004–1023,1070` | **INFERABLE-FROM-OBSERVED** (MED) | count of non-fossil/non-pruned per-slot stage one-hots; base **dims 13/14/15** also aggregate | Feeds occupancy rent. |
| `host_params` (`scoreboard.host_params`) | `action_execution.py:1052` | **ACCIDENTALLY-UNPLUMBED** | no param-count dim in obs | Rent denominator `growth_ratio = overhead/host_params` (`:743`). **The headline asymmetry:** reward optimizes gain-**per-param**, policy sees no capacity signal. |
| `total_params` (`model.total_params`) | `action_execution.py:1051` | **ACCIDENTALLY-UNPLUMBED** | no param-count dim | Rent `effective_overhead = total−host` fallback (`:739`). Param-count family. |
| `effective_seed_params` (alpha-weighted BaseSlotRent) | `action_execution.py:904,1063` (`compute_rent_and_shock_inputs`) | **ACCIDENTALLY-UNPLUMBED** | no param-count dim | The *actual* rent input (`:737`). Sharpens obs-audit item #7 — a **derived** param count, not raw. |
| `acc_at_germination` | `action_execution.py:1024–1028` (`env_state.acc_at_germination[slot]`) | **ACCIDENTALLY-UNPLUMBED** | per-seed latched baseline; germination usually predates the 5-epoch acc window | Sets `progress = progress_acc − acc_at_germination` (`:523`), the attribution **scale** for all positive contributions (harmonic/min of progress×contribution). obs-audit item #11. |
| `seed_info.improvement_since_stage_start` (`current_val_acc − accuracy_at_stage_start`) | `types.py:63` | **ACCIDENTALLY-UNPLUMBED** | `accuracy_at_stage_start` latched per-seed; pre-window for any seed older than 5 epochs in stage | Used **only** on the proxy path when the counterfactual is absent (`:628–634`). Note: partially recoverable if the stage began inside the acc window. |
| `action` (LifecycleOp being rewarded) | `action_execution.py:1045` | **N/A** (policy's own output) | previous op echoed at **dims 16–22** (one step, op-only) | Not a state signal to observe. |
| `slot_id` / `seed_id` | `action_execution.py:1067–1068` | **N/A** (telemetry keys) | — | Only used to key hub emits in reward-hacking/ransomware checks. |
| `seed_info.previous_stage` / `previous_epochs_in_stage` | `types.py:74–75` | **INFERABLE (weak) / design-internal** | not a dim; recoverable from stage-history only if the transition is in-window | PBRS telescoping internal; PBRS is policy-invariant *by design* so the policy needn't see these. |
| `seed_info.seed_params` | `types.py:73` | **not read by SHAPED** | — | Carried on `SeedInfo` but `compute_contribution_reward` uses `effective_seed_params` instead. |
| `seed_info.boost_received` | `types.py:67` | **not read by SHAPED** | — | Only `compute_scaffold_hindsight_credit` reads it (dead path, per project memory). Also obs-removed (`features.py:408`). |

### Table 1b — Adjacent reward-family inputs (NOT read by SHAPED's `compute_contribution_reward`)

The defect register headlines some of these, so they are included — but the consuming function is stated
explicitly so no reader mistakes them for SHAPED inputs. Confidence **HIGH** unless noted.

| input | computed-where | consumed by | classification | obs dim / reason |
|-------|----------------|-------------|----------------|------------------|
| `committed_val_acc` (FOSSILIZED-only acc) | `vectorized_trainer.py:1422` (`committed` config kind); default `:1367` | `compute_sparse_reward:914`, `compute_minimal_reward` | **ACCIDENTALLY-UNPLUMBED** | obs carries only all-seeds `val_acc` (**dim 2**); the fossilized-only "committed" accuracy — the ground truth a fossilize/prune policy should optimize — is on `env_state`, never obs. **Register #1.** |
| `fossilized_seed_params` | `action_execution.py:1011–1017` | `compute_sparse_reward:915` (param_cost) | **ACCIDENTALLY-UNPLUMBED** | param-count family; no obs dim. |
| `num_contributing_fossilized` (`env_state.contributing_fossilized`) | `action_execution.py:1058` | `compute_simplified_reward:1210` (`n*2.0`); SHAPED **telemetry-only** (`:893`) | **ACCIDENTALLY-UNPLUMBED** | no per-fossil "still contributing" signal in obs. Genuinely-new candidate (below). LOW–MED priority (telemetry-only in SHAPED). |
| `fossilized_contributions` (per-fossil live LOO, drip) | `action_execution.py:1078–1085` | `compute_basic_reward` (BASIC_PLUS drip) | **INTENTIONALLY-UNMEASURED-BY-DESIGN** | Fossils are excluded from the ablation post-permanence (the permanence arc's subject); dim +12 of a fossil slot is structurally unmeasured. Honor register classification. |

---

## Table 2 — Heuristic controller (`HeuristicTamiyo`, heuristic.py)

Inputs from `signals` (`tracker.py`) and `SeedState`. Confidence **HIGH** throughout (read `heuristic.py` +
`tracker.py` end to end); the internal-state group is MED (RL-analog reasoning).

| input | computed-where (file:line) | classification | obs dim / reason | note |
|-------|----------------------------|----------------|------------------|------|
| `signals.metrics.epoch` | `tracker.py:187` | **OBSERVED-DIRECTLY** | **dim 0** | Embargo (`heuristic.py:169`) + min-epochs gate (`:187`). |
| `seed.stage` | seed state; `heuristic.py:228` | **OBSERVED-DIRECTLY** | **dims +1..+10** | Stage dispatch. |
| `seed.epochs_in_stage` | seed state; `heuristic.py:230` | **OBSERVED-DIRECTLY** | **dim +28** | Prune "epochs without improvement" gate (`_should_prune:320`). |
| `seed.alpha_controller.alpha_target` | `heuristic.py:256` | **OBSERVED-DIRECTLY** | **dim +15** (`alpha_target`) | BLENDING→HOLDING advance test. |
| `seed.alpha_controller.alpha_mode` | `heuristic.py:257` | **OBSERVED-DIRECTLY** | **dim +16** (`alpha_mode_norm`) | Same test (`== HOLD`). |
| `seed.alpha` (current) | `heuristic.py:258,268` | **OBSERVED-DIRECTLY** | **dim +11** (`current_alpha`) | Same test + BLENDING wait reason. |
| `seed.metrics.counterfactual_contribution` | `heuristic.py:274`; set `vt:1393` | **OBSERVED-DIRECTLY** | **dim +12** | HOLDING fossilize/prune decision + ransomware check. Same qty as reward `seed_contribution`. |
| `seed.blueprint_id` | `heuristic.py:333` (`_prune_seed` penalty) | **OBSERVED-DIRECTLY** (categorical) | `blueprint_indices` → 4-dim learned embedding concatenated into LSTM input (`factored_lstm.py:821`) | Not an obs *dim* but reaches the recurrent input. |
| `signals.available_slots` | `tracker.py` update arg; `heuristic.py:193` | **INFERABLE-FROM-OBSERVED** | `num_slots − Σ per-slot is_active` (**dim +0** per slot) | — |
| `signals.metrics.host_stabilized` | `tracker.py:196` | **INTENTIONALLY-HIDDEN** | design | `features.py:305,677`: "Removed host_stabilized — Tamiyo learns stability from raw telemetry." Germination gate (`heuristic.py:180`). |
| `signals.metrics.plateau_epochs` | `tracker.py:127–130,195` | **ACCIDENTALLY-UNPLUMBED** | running counter of consecutive sub-threshold acc epochs; exceeds the 5-epoch acc window | The **exact germination trigger** (`heuristic.py:200`); the RL policy is blind to it. **Register / obs-audit #2.** |
| `seed.metrics.improvement_since_stage_start` | seed metrics; `heuristic.py:229` | **ACCIDENTALLY-UNPLUMBED** | `accuracy_at_stage_start` latched, pre-window | TRAINING/BLENDING prune trigger (`_should_prune`). Same field as reward Table-1 row. |
| `seed.metrics.total_improvement` (host-drift-confounded, since germination) | seed metrics; `heuristic.py:275,279–283` | **ACCIDENTALLY-UNPLUMBED** | not an obs dim; needs `acc_at_germination` (also unobserved) | Ransomware detection (`:280`). The **reward deliberately avoids** this field (host-drift-confounded, `contribution.py:477–489`) — confirmed by grep: no reward function reads plain `seed_info.total_improvement`. The heuristic still keys on it. |
| Heuristic internal state: `_last_prune_epoch` (embargo), `_blueprint_penalties`, `_blueprint_index`, `_last_decay_epoch` | `heuristic.py:136–138,332,343,379` | **CONTROLLER-INTERNAL** | not an external signal | RL analog is the LSTM hidden state; the only reinforcing input is the one-step, op-only `last_action_op` echo (dims 16–22). Not a fair "unplumbed" charge — it is the controller's own memory. |

---

## Summary

### Counts per class

**SHAPED reward (`compute_contribution_reward`), decision-relevant inputs:**

| class | count | members |
|-------|-------|---------|
| OBSERVED-DIRECTLY | 10 | seed_contribution(+12), val_acc(2), stage(+1..10), epochs_in_stage(+28), seed_age(+30), interaction_sum(+22), epoch(0), max_epochs(0), stable_val_acc(23·escrow), escrow_credit_prev(+31·escrow) |
| INFERABLE-FROM-OBSERVED | 4 | acc_delta, alpha_delta_sq_sum, num_fossilized_seeds, n_active_seeds |
| ACCIDENTALLY-UNPLUMBED | 6 | **counterfactual_total_improvement (≥2 seeds)**, host_params, total_params, effective_seed_params, acc_at_germination, improvement_since_stage_start |
| INTENTIONALLY-HIDDEN | 0 | — (none inside the SHAPED reward; `contribution_velocity` is obs-only, see corrections) |
| carried-but-unused / N/A | 5 | seed_params, boost_received, previous_stage/epochs, action, slot_id/seed_id |

*(counterfactual_total_improvement is counted once, under UNPLUMBED, since the ≥2-seed case is the one that
matters; it is OBSERVED in the single-seed case.)*

**Adjacent reward families (sparse/minimal/simplified/basic):** 3 ACCIDENTALLY-UNPLUMBED
(committed_val_acc, fossilized_seed_params, num_contributing_fossilized) + 1 INTENTIONALLY-UNMEASURED
(fossilized_contributions).

**Heuristic controller:** OBSERVED-DIRECTLY 8 · INFERABLE 1 · INTENTIONALLY-HIDDEN 1 (host_stabilized) ·
ACCIDENTALLY-UNPLUMBED 3 (plateau_epochs, improvement_since_stage_start, total_improvement) ·
CONTROLLER-INTERNAL 1 group.

### The accidentally-unplumbed list, ranked by how directly the reward keys on it

1. **`counterfactual_total_improvement` / the all-seeds-off baseline** *(SHAPED reward; multi-seed episodes)* —
   gates **every** anti-gaming, discount, ratio-penalty, fossilize-suppress, fossilize-legitimacy, blending-warning
   and prune branch. Aliases dim +12 only when exactly one seed is active. **Highest leverage.**
2. **`host_params` + `total_params` + `effective_seed_params`** *(SHAPED reward)* — the rent penalty fires on
   **every step**; the policy optimizes committed-gain-per-param with **zero** param/capacity signal. The
   single sharpest "optimize what you can't see" case (obs-audit's headline asymmetry, confirmed).
3. **`acc_at_germination`** *(SHAPED reward)* — the attribution **scale** for all positive contributions
   (`progress`). Unobservable germination reference.
4. **`plateau_epochs`** *(heuristic germination trigger)* — a pure running counter; the RL policy cannot see the
   exact signal the heuristic germinates on.
5. **`committed_val_acc`** *(sparse/minimal reward family)* — the fossilized-only ground-truth accuracy; obs
   carries only all-seeds val_acc (dim 2). Register #1.
6. **`improvement_since_stage_start`** *(SHAPED proxy path + heuristic prune)* — stage-relative causal proxy;
   partially in-window-recoverable but not for seeds parked >5 epochs in a stage.
7. **`total_improvement`** *(heuristic ransomware detection)* — the confounded since-germination improvement;
   the reward deliberately routes around it, the heuristic does not.
8. **`fossilized_seed_params`** *(sparse family param cost)* — param-count family.
9. **`num_contributing_fossilized`** *(simplified family; SHAPED telemetry-only)* — no per-fossil contribution
   signal in obs.

### New unplumbed signals the prior audits missed / sharpened

Honest finding: **the two prior audits were largely complete for the reward/heuristic *input* set.** The
value-add here is consumer-labeling and three precision points, not a pile of new gaps:

- **`num_contributing_fossilized` — genuinely new (low priority).** Neither prior audit lists it. It is a real
  input to `compute_simplified_reward` and telemetry-only in SHAPED. A distinct signal from `committed_val_acc`
  (a *count of contributing fossils*, not an accuracy).
- **`effective_seed_params` — a sharpening, not new.** The obs-audit's param-count item (#7) named raw
  seed/host counts; the reward's *actual* rent input is the alpha-weighted BaseSlotRent derived count.
- **`alpha_delta_sq_sum` — enumerated, covered (NOT a gap).** It is INFERABLE from per-slot `alpha_velocity`
  (dim +20), so it is not accidentally-unplumbed; flagged only because neither audit enumerated it as a reward
  input. Note the clamp precision caveat.

### Corrections to prior audits (cited)

- **`contribution_velocity` (obs dim +13) — classification changed accidental → intentional; it is *not* a
  reward or heuristic input (so: a cross-check note, not a table row).** `obs-audit-pytorch.md §P0` frames it as
  a *live* `to_leyline` plumbing bug (a fossilize-lookahead signal the pipeline computes but drops). Current
  source disagrees: `features.py:912–920` documents the leyline transport as **fixed** (telemetry now carries
  the true value) while dim +13 is **deliberately** hardwired to `0.0` — a matched-observations constant so
  every r9-calibrated prior stays comparable during the permanence experiment (**PDR-0100**), with the live
  value exposed only at Obs V5. So its state moved from "accidental plumbing bug" to "intentional schema
  constant." It remains blind to the policy, but by design now, not by accident. (It is neither a reward nor a
  heuristic input, so it does not appear in the tables.)
- **Scope clarification on obs-audit #3 (train–val gap) and #4 (full CF structure: Shapley `std`, `regime`,
  pairwise `I_ij` matrix).** These are real obs gaps but **neither `compute_contribution_reward` nor the
  heuristic consumes them** — the reward keys only on the counterfactual *scalars* (`counterfactual_total_improvement`
  + LOO `seed_contribution` + summed `interaction_sum`). So for *this* report's question ("what does the reward
  optimize that it can't see") they rank below the scalars the reward actually reads. They matter for a
  richer-observation argument, not for reward-input coverage.

### Confidence per row group

| group | confidence | basis |
|-------|-----------|-------|
| Reward OBSERVED-DIRECTLY | **HIGH** | traced call site (`action_execution.py:940–1090`) → encoder (`features.py:837–997`) end to end; seed_contribution↔dim+12 identity re-derived from `vt:1379,1393` |
| Reward ACCIDENTALLY-UNPLUMBED (param family, counterfactual_total_improvement, acc_at_germination) | **HIGH** | sources read first-hand; all-off vs LOO divergence re-derived from `vt:1355–1460` |
| Reward `improvement_since_stage_start` | **MED-HIGH** | proxy-path-only; in-window-recoverable nuance is a judgment call |
| Reward INFERABLE (two counts) | **MED** | assumes fossils retain slot blocks — confirmed at `action_execution.py:1010`, but base-dim aliasing (dim 15) means the clean count needs the per-slot one-hots |
| Heuristic (all rows) | **HIGH** | `heuristic.py` + `tracker.py` read end to end |
| Heuristic internal-state group | **MED** | RL-analog (hidden-state) reasoning, not a source fact |
