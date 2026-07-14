# Observation-space sufficiency audit for Tamiyo

**Question (owner, verbatim):** "do we give tamiyo enough data about the hosts to make these decisions?"
**Tracker:** esper-lite-f7b46a5f52. **Date:** 2026-07-14. **Scope:** analysis / future-work scoping only — no code changed.
**Author:** drl-expert (teammate). **SME protocol:** confidence + information gaps stated per finding.

---

## 0. One-paragraph answer

**Partly — and the gap is not uniform.** The observation is *rich about the seed*, *moderate about the host globally*, and *structurally blind about the host per-site*. For decisions that only need seed quality (PRUNE, and SET_ALPHA **target**) the obs is adequate and the corresponding heads behave. For the decisions that need to match a module to a *place in the host* (GERMINATE-where, blueprint-what) the state contains **nothing** about per-site host tissue — and, decisively, **even the information that is present cannot reach the head that needs it**, because every action head is an independent projection of one slot-averaged trunk (`factored_lstm.py:837–845`). So the near-uniform blueprint head has **two** compounding causes: no site signal in the obs *and* no architectural path for a site signal to condition the head. "Give Tamiyo more data" is necessary but **not sufficient** for the blueprint pathology; the head must also be made slot-conditional.

---

## 1. What the observation actually contains (verified inventory)

Source of truth: `src/esper/tamiyo/policy/features.py::batch_obs_to_features` (base :828–869, per-slot :871–992). Network input is exactly `cat([obs, blueprint_embeddings])` (`factored_lstm.py:820–823`); masks are applied *after* the head projection (`:849–864`), never as input.

### Base features — 24 dims (host-GLOBAL)
| dims | feature | source |
|---|---|---|
| 0 | epoch_norm | `metrics.epoch / max_epochs` |
| 1 | val_loss (symlog/7) | `metrics.val_loss` |
| 2 | val_accuracy/100 | `metrics.val_accuracy` |
| 3–7 | loss history (5, symlog, sentinel-padded) | `signal.loss_history` |
| 8–12 | accuracy history (5) | `signal.accuracy_history` |
| 13–15 | stage distribution (training / blending / holding+fossilized) | counts over slots |
| 16 | last_action_success | `env_state.last_action_success` |
| 17–22 | last_action **op** one-hot (6) | `env_state.last_action_op` |
| 23 | escrow stable val-acc | `env_state.stable_val_acc_for_observation` |

### Per-slot features — 32 dims × num_slots (V3); +3 under V4 (all describe the SEED)
`is_active`(0); stage one-hot(1–10); `current_alpha`(11); **counterfactual contribution**(12); contribution velocity(13); blend tempo(14); alpha scaffolding × 8 (15–22: target, mode, steps_total, steps_done, time_to_target, velocity, algorithm, **interaction_sum**); telemetry × 4 (23–26: gradient_norm, gradient_health, has_vanishing, has_exploding); gradient_health_prev(27); epochs_in_stage(28); counterfactual_fresh(29); seed_age(30); escrow_credit(31). **V4 adds** cf_observed(32), cf_frozen(33), cf_age_norm(34).

Blueprint identity enters only as a 4-dim learned **embedding at the input** keyed by *which blueprint currently occupies the slot* (`factored_lstm.py:821–823`); a dormant slot is the null embedding and an all-zero 32-dim block (`features.py:876–878`).

### The spine
- **Seed-rich:** contribution, velocity, interaction, gradient health (+prev), full alpha-schedule state, age, escrow — a detailed local picture of each *occupied* slot.
- **Host-global-moderate:** loss/accuracy + a 5-step history, stage counts, epoch, escrow accuracy.
- **Host-per-site-blind:** *nothing* describes the host tissue at a slot site. Confirmed: a grep for `.position` / `.surface` / `channels_for_slot` / `.channels` in `tamiyo/policy` returns nothing — the static site descriptors that exist in `InjectionSpec` never enter the obs. Dormant slots are observationally symmetric (all zeros).

---

## 2. The decisive architectural finding (reframes the whole audit)

The owner asked about *data*. But "a decision can be better than chance" needs **two** conjuncts: (1) the state *contains* the discriminating information, **and** (2) the head that makes the decision can *access* it at decision time. I verified conjunct (2) and it fails for the per-slot-action heads.

**Every head is an independent projection of one shared LSTM representation** (`factored_lstm.py:837–845`):
```
slot_logits      = self.slot_head(lstm_out)
blueprint_logits = self.blueprint_head(lstm_out)   # lstm_out is slot-AVERAGED; no gather of the sampled slot
style/tempo/alpha_* = <head>(lstm_out)
op_logits        = self.op_head(lstm_out)
```
There is **no autoregressive routing** — the slot the `slot_head` picks is never fed to the blueprint/style/tempo/alpha heads. The blueprint embeddings enter only at the *input* (so the trunk *sees* which blueprints occupy slots), but the blueprint head *output* is a function of a representation that mixes all slots and cannot be conditioned on "which slot am I germinating into."

**Consequence:** the blueprint head is structurally limited to expressing the **site-independent marginal** — "which blueprints are best *on average across the host*" — and is architecturally *incapable* of expressing "attention **here**, conv **there**," regardless of what obs features you add. This exactly reconciles every r9 number (§5): above-uniform (gradient is flowing), 7/13 realized, but weak — the signature of a head that learned a marginal and could not learn a conditional.

**This flips the remedy for the top gap:** per-site obs features are necessary but not sufficient; the blueprint/style/tempo/alpha heads must *also* be made slot-conditional (gather the sampled slot's per-site block + a slot-identity embedding into those heads — an autoregressive `P(blueprint | slot, s)` factorization).

*Confidence: High* (docstring `:4–12` + head defs `:410–461` + forward `:837–845` are unambiguous). *Information gap:* I did not trace `get_action`/`evaluate_actions` sampling order to confirm there is no *implicit* slot conditioning via masking; masks are logit-only (`:849–864`), which cannot substitute for representational conditioning, so I assess this as low-risk.

---

## 3. Decision → information map

For each decision: what the state must contain (from first principles) → what is visible → grade.

### GERMINATE — where (slot), what (13 blueprints), when
- **Needs:** (WHERE) per-site evidence of a bottleneck — which layer is under-capacity / has vanishing or saturated activations / high loss-gradient; (WHAT) which inductive bias fits that site — conv (local spatial), attention (long-range), norm (distribution shift), lora/mlp (capacity); (WHEN) host has plateaued and stabilized enough to benefit.
- **Visible:** global loss/acc + 5-step history (WHEN, weak); stage counts; states of *occupied* slots. Site identity is **positional only** (fixed tensor order) — no channels/position/surface. Per-site *need*: **none**. And per §2 the blueprint head cannot condition on the slot anyway.
- **Grade:** WHERE = **weak** (positional prior only). WHAT/blueprint = **structurally blind + architecturally unconditioned** — the worst-served decision. WHEN = **partial** (plateau capped at 5-step window; `plateau_epochs` withheld — §4).
- **Predicts:** blueprint learns a site-independent marginal → above-uniform-but-weak. ✅ matches r9 A3.

### SET_ALPHA_TARGET — target, speed, curve
- **Needs (target):** how much of the seed to blend in ← how much it helps (contribution) and whether the host tolerates it (stability). **Needs (speed/curve):** how fast/smooth to ramp ← host sensitivity to perturbation (does val-loss spike as α rises?).
- **Visible (target):** contribution(12), velocity(13), interaction(22), gradient health(24), full alpha state(15–21). **Adequate.** → alpha_target shows a real 3-way preference (0.7:43 / 0.5:33 / 1.0:21). ✅ healthiest head.
- **Visible (speed/curve):** the *consequence* of a ramp choice — host perturbation attributable to *this* ramp — is **not cleanly visible**; and these heads bite only on the SET_ALPHA-with-non-instant-speed subset (small causal fraction), then are causal-masked in advantage (`telemetry.py:210–212`). So they are **credit-starved first**, weak-signal second, and (per §2) architecture-limited third.
- **Grade:** target = **served**; speed/curve = **collapsed** (credit-starvation dominant). ✅ matches r9 A1.

### PRUNE — remove a bad/dead seed
- **Needs:** is it net-negative or dead weight — marginal contribution, trend (recovering vs declining), interaction (freeloader/enabler), gradient death.
- **Visible:** contribution(12), velocity/EMA-trend(13), interaction_sum(22), gradient_health(24/27), has_vanishing/exploding(25/26), age(30), freshness(29). **Well-served.**
- **Missing:** counterfactual *trajectory* (only a 1-step EMA slope, not a multi-step curve); `boost_received` (strongest single-partner interaction — computed in `SeedMetrics` but **removed from V3 obs**, `features.py:408`); slot-reuse opportunity cost.
- **Grade:** **good.** The observed PRUNE pathology (PDR-0079 forced-prune destruction; the policy-wide floor dead-zone, MEMORY floor-gradient-dead-zone) is a **reward/optimizer** problem, **not** obs-starvation. Do not attribute it to the obs.

### FOSSILIZE — irreversible commit
- **Needs:** confidence the contribution is real, **stable**, and will persist — contribution *stability/variance*, remaining-horizon value (commit now vs later), and pending/committed status.
- **Visible:** contribution + velocity + freshness + gradient health + age + escrow; remaining horizon = `max_epochs − epoch` derivable from dim 0. V4 adds cf staleness/frozen (32–34).
- **Missing:** (a) **pending_settlement bit (F2)** — *aliased today* (source-proof 2026-07-14: two identical-telemetry HOLDING seeds, one pending, are byte-identical except a 1-step op echo); *ruled in* post-F2. (b) **contribution stability** — only an EMA slope, no variance/trend-at-commit. (c) explicit remaining-horizon value (derivable but not surfaced).
- **Grade:** **moderate**; F2 bit (already scoped) + stability are the gaps.

### WAIT — the do-nothing baseline
- **Needs:** opportunity cost of *not* acting — is there a plateau to break (germinate), a ripe seed to advance/fossilize, a bad seed to prune?
- **Visible:** seed readiness IS visible (per-slot states) → WAIT-vs-act-on-*existing*-seed is estimable. But "should I *start* something" hinges on plateau detection, and the 5-step history **cannot distinguish a 5-epoch pause from a 20-epoch entrenched plateau** (both are flat windows); `plateau_epochs` (the unbounded count) is withheld.
- **Grade:** **partial** — opportunity cost of continued waiting during a long plateau is only *weakly/indirectly* visible.

---

## 4. Gap taxonomy (three tiers)

- **(a) MISSING-AND-MEASURABLE** — signal exists in trainer/telemetry, never enters obs:
  - Static per-site descriptors: `InjectionSpec.channels / position / surface / layer_range / row / col` (`injection_spec.py:37–71`). Computed at host build (`kasmina/host.py:116,533`); `channels` already sits in `SlotConfig._channel_map`. **Cheap.**
  - `plateau_epochs`, `host_stabilized` — computed in `tamiyo/tracker.py:195,196` and **consumed by the heuristic controller** (`heuristic.py:180,200,211`) but withheld from the RL obs.
  - Global host learning signals: `train_loss`, `train_accuracy` (→ train/val generalization gap), `host_grad_norm` (`EpochCompletedPayload`, `telemetry.py:686–687,713–714`); `best_val_accuracy/loss`, `loss_delta/accuracy_delta` (`TrainingMetrics`).
  - Slot-local gradient health: `collect_per_layer_gradients` (`debug_telemetry.py:43`) produces per-layer `grad_norm/std/zero_fraction/…`; `InjectionSpec.layer_range` maps each slot to its layer span → slot-local aggregation is *an aggregation away*. Currently gated off (`telemetry_config.py:37 per_layer_gradients=False`), not slot-mapped, not in obs. **Tier-a with modest glue.**
  - `boost_received`, `upstream/downstream_alpha_sum`, `seed_param_count/host_param_count` (`SeedMetrics`) — computed, removed from / never in V3 obs.
- **(b) MISSING-AND-UNMEASURED** — needs new instrumentation:
  - Per-site **activation / feature-map** statistics: mean/var, dead-channel fraction, saturation, activation-gradient flow at each injection surface. Not collected anywhere. (This is TIP Phase D1's proposed emit.)
- **(c) PRESENT-BUT-DEGRADED** — aliased / stale / window-limited:
  - Pending-committed membership: **aliased** (F2, source-proof).
  - Plateau duration: **window-capped** at 5 steps (loss/acc history) — long plateau ≡ brief pause.
  - Counterfactual value at commit: **staleness-aliased** on V3 (None→0 coercion, `features.py:906–910`); V4 fixes via shrink-to-sentinel + cf_age_norm/cf_frozen (`:895–902, 989–992`).

---

## 5. Top-5 gaps, ranked by decision impact

Ranking is by *which observed head pathology the gap explains* and breadth. Each carries a falsifiable "the policy could use this because…" (anti-"add everything" discipline), tier, confidence, and information gaps.

### Gap 1 — Per-site host perception **+ slot-conditional heads** (obs *and* architecture)
- **Explains:** blueprint near-uniform (r9 A1/A3); weak slot differentiation; the site-*inability* of style/tempo/alpha.
- **Two-part remedy (both required):** (i) **obs** — static site descriptors (tier a, cheap) → slot-local gradient health aggregated over `layer_range` (tier a + glue) → activation/dead-channel/saturation stats (tier b); (ii) **architecture** — route the sampled slot's per-site block + a slot-identity embedding into blueprint/style/tempo/alpha via an autoregressive `P(head | slot, s)` factorization.
- **Falsifiable "could use this because…":** a conv bias helps a high-resolution PRE_POOL early site and attention helps a late low-resolution POST_POOL site — so cf-contribution should differ measurably across (blueprint × surface); the D2 site-feature predictor and the `P(blueprint | slot)` table (§7) can confirm the signal exists **before** any change. If cf-contribution is invariant to (blueprint × surface), this gap is *not* real and should be dropped.
- **Tier:** a (static + local-grad) / b (activation stats) + architecture. **Confidence: High** that this is the dominant *observability* cause of the blueprint pathology; **Medium** on the exact minimal feature set; **efficacy is gated** on the reward-horizon fix (§6 third rival) — obs + architecture are necessary but not independently sufficient if germinated seeds are destroyed before blueprint choice reaches return (PDR-0079). **Sequence Gap 1 after / alongside the floor-prune workstream**, not ahead of it. **Info gaps:** need the D2 ladder to rank static-vs-dynamic site features by marginal predictive value; the survivor-signal read (§7 test 4) to confirm the reward path can carry blueprint signal; have not measured whether 3 fixed slots make the positional prior "good enough" that only the *architecture* half matters.

### Gap 2 — `plateau_epochs` / learning-curve phase (WAIT opportunity cost, GERMINATE timing)
- **Explains:** WAIT's under-estimable opportunity cost; weak GERMINATE timing.
- **Remedy:** add `plateau_epochs` (unbounded count) and a plateau/still-climbing/converged phase indicator to the base obs.
- **Falsifiable "could use this because…":** the 5-step history structurally cannot separate a 5-epoch pause from a 20-epoch entrenched plateau (identical flat windows), yet the correct action differs (keep waiting vs germinate); the unbounded count carries exactly that marginal bit — and the heuristic controller already conditions germination on it (`heuristic.py:200`), evidence it is decision-relevant.
- **Tier:** a. **Confidence: High** it adds real information; **Medium** it moves the RL policy (the op head is *also* floor-bound — a separate root cause the obs cannot fix). **Info gap:** cannot separate "WAIT is obs-limited" from "WAIT is floor-limited" without the learnable-fraction/floor audit (§7).

### Gap 3 — Global host generalization + gradient phase: train/val gap, `host_grad_norm`, `host_stabilized`
- **Explains:** GERMINATE *necessity* (add capacity) vs STOP (overfitting); the host-phase context all ops share.
- **Remedy:** add train/val accuracy gap, `host_grad_norm` (symlog), and the `host_stabilized` latch to the base obs.
- **Falsifiable "could use this because…":** overfitting (train↑, val flat) and under-capacity (both flat) demand opposite actions but are **indistinguishable from val-only history**; the train/val gap is the discriminator. `host_stabilized` was deliberately removed ("learn stability from raw telemetry", `features.py:305`) — a latch is a cheaper, non-reconstructable signal than a memory integral.
- **Tier:** a. **Confidence: Medium-High** on the train/val gap; **Medium** on host_grad_norm (may be redundant given loss history). **Info gap:** no measurement of whether the LSTM already reconstructs stability adequately.

### Gap 4 — FOSSILIZE commit-quality: contribution **stability** + explicit remaining-horizon value
- **Explains:** the residual FOSSILIZE gap *after* the F2 pending bit lands.
- **Remedy:** add contribution variance/rolling-stability (computable from existing cf history) and an explicit remaining-horizon dim (`(max_epochs − epoch)/max_epochs`).
- **Falsifiable "could use this because…":** an irreversible commit needs contribution *variance*, not just its EMA slope — a high-mean/high-variance seed and a stable one have identical (dim12, dim13) but very different commit value; the current obs cannot tell them apart.
- **Tier:** a (stability) / c (horizon derivable-but-implicit). **Confidence: Medium.** **Info gap:** unquantified how often commits are made on unstable-but-positive contributions (needs a fossilize-outcome vs contribution-variance read).

### Gap 5 — Alpha-ramp consequence signal for the collapsed speed/curve heads
- **Explains:** alpha_speed / alpha_curve full collapse (r9 A1).
- **Remedy (contingent):** expose host-perturbation-attributable-to-the-ramp (Δval-loss over the ramp window) — *but only after* the credit-starvation is ruled out.
- **Falsifiable "could use this because…":** curve/speed only differ in transition smoothness; if the host never perturbs measurably during a ramp, there is nothing to learn from and no obs feature helps.
- **Tier:** b + architecture. **Confidence: Low** that obs is the binding constraint here — I assess these heads as **credit-starved first** (tiny causal/learnable fraction + causal masking + floor). **Info gap:** need `head_alpha_curve_learnable_fraction` + floor-binding fraction to confirm before spending any obs/instrumentation budget. **Ranked last deliberately** — this is the gap most likely to be a non-obs problem.

**Explicitly NOT proposed** (information-ceiling discipline): re-adding the removed V2 features wholesale (`boost_received`, upstream/downstream alpha) without a predictive story; GPU/util/timing signals (no decision story); duplicating an explicit countdown when epoch(0)+pending-bit already determine it (source-proof §2).

---

## 6. Rival-hypothesis verdict: starved vs floor-pathology (head-specific)

The banked interpretation is floor pathology + weak preference. The rival is information starvation. **The correct answer is per-head, and it is a synthesis, not a coin-flip:**

| Head | r9 signature | Verdict | Reasoning |
|---|---|---|---|
| **blueprint** | above-uniform (18–20% vs 8%), 7/13 realized, weak (A3) | **INFORMATION-STARVED + ARCHITECTURE-LIMITED** (primary) | Above-uniform ⇒ gradient *is* flowing (a dead head sits at its init prior); it is causally relevant on *every* GERMINATE, so it is **not** gradient-step-starved. It learned the site-**independent marginal** but not site-**conditional** preference — exactly what a site-blind head fed to a slot-averaged trunk (§2) produces. Floor is a *secondary* contributor (GERMINATE samples are not guaranteed floor-free). |
| **alpha_speed, alpha_curve** | max-prob 1.00, single action, 100% collapse (A1) | **CREDIT/GRADIENT-STARVED (primary), floor + weak-signal (secondary)** | Collapse to *one* action (not uniform) is the signature of a head that got a small consistent early gradient and then had no exploration pressure. They bite only on a small causal subset and are causal-masked otherwise → tiny learnable fraction. Obs starvation is at most a tertiary factor. |
| **alpha_target** | real 3-way spread 0.7/0.5/1.0 (A3) | **NEITHER — healthiest** | Adequate seed-quality signal reaches it; it expresses genuine preference. Evidence the pipeline *can* learn a preference when the signal is present and the head is causally exercised. |
| **op** | ~43% floored, pinned-rate rising over training (A1/A2) | **FLOOR-PATHOLOGY (primary)** — *not* obs | The confidence-trap + policy-wide floor dead-zone (MEMORY floor-gradient-dead-zone). Obs changes will **not** fix this; the differentiable-floor fix will. |

**A third live rival for blueprint — reward-path / credit-horizon starvation.** My two blueprint causes are obs-absence and architecture. There is a third that sits under *none* of them: if germinated seeds are **destroyed before blueprint choice propagates to return**, the blueprint advantage is near-zero-mean noise → near-uniform, *independent of obs or architecture*. The evidence this is live is in this very report: PDR-0079 (55% of good held seeds force-pruned within ~13 epochs) and the policy-wide floor dead-zone (MEMORY floor-gradient-dead-zone). Gradient can flow to the head (consistent with above-uniform) yet point at *nothing informative* because the target is noise. I hold info-starved + architecture as the **primary** reading, but this reward-horizon rival must be acknowledged and discriminated (§7), not assumed away — and it has a sharp consequence for prioritization: **a positive D2 result proves information-PRESENCE, not fix-EFFICACY.** Even if site-features lift the offline predictor, the online blueprint head will not recover if the reward path is broken. **Gap 1's efficacy is therefore likely gated on the floor/prune fix** (a separate, already-identified workstream) — sequence it that way; do not present obs + architecture as independently sufficient.

**Net:** the rival hypothesis is *correct for blueprint* (with the crucial additions that it is also an architecture gap *and* is efficacy-gated on the reward-horizon fix), a *minor* factor for speed/curve, *irrelevant* for alpha_target, and *wrong* for op. The banked "floor + weak preference" reading is right for op and a secondary truth for blueprint — the interpretations are compatible once separated by head.

**Evidence that would distinguish (cheap → decisive):**
1. **`P(blueprint | slot)` contingency table** from logged r9 GERMINATE decisions. **Flat rows ⇒ the policy is site-blind *today*** (consistent with §2, whichever cause). Cheapest, most specific, and directly exercises the routing question. *(Data in hand; the direct read of "does behaviour vary by site at all.")*
2. **`head_blueprint_learnable_fraction`** (`PPOUpdatePayload`, `telemetry.py:956`) — high ⇒ rules out gradient-step starvation ⇒ isolates info-starvation + architecture. **Ground the "gradient is flowing" claim on this, don't assert it.**
3. **Floor-binding fraction on GERMINATE samples** — separates the floor's secondary contribution from the info/architecture primary.

---

## 7. What to test OFFLINE first (no obs change, no retrain)

All three run on **already-logged rollouts** — the ceiling discipline the owner's TIP Phase D2 codifies. Sequence:

1. **`P(blueprint | slot)` and `P(blueprint | surface)` contingency tables** (motivating read). Flat ⇒ site-blind today. Cheapest; frames everything below. *Cost: one telemetry query.*
2. **D2 information-ceiling ladder** (the discriminator). Train predictors {host-only | slot-only | full ObsV3 | ObsV3 + candidate site-features} → {cf-contribution, fossilize-success, future return}. This is what separates **"info absent from obs"** (predictor at chance from ObsV3, lifts with site-features) from **"critic can't extract present info"** (predictor already good from ObsV3). It *pre-validates the specific fix before any obs change* — if adding site-features doesn't lift the predictor, Gap 1's obs half is dead on arrival. *Lead confirmatory test.*
3. **Per-head learnable-fraction + floor-binding audit** across heads (all fields already logged: `head_*_learnable_fraction`, `head_*_gradient_state`). Confirms the §6 head-by-head split — high blueprint LF + flat `P(blueprint|slot)` = info/architecture starvation; low speed/curve LF = credit starvation. *Cost: aggregate existing columns.*
4. **Reward-horizon discriminator** (for the §6 third rival, same tooling class). Among seeds that *survive to blend/hold*, does blueprint identity predict cf-contribution? And what fraction of germinated seeds survive past the ~13-epoch force-prune horizon? If blueprint is uninformative *even among survivors*, the signal genuinely isn't there; if survival is rare, the online blueprint advantage is dominated by destruction, not blueprint choice — and Gap 1 will not move the head until the floor/prune fix lands. *Cost: two aggregates over logged lifecycle joins.*

**Efficacy vs presence — do not conflate.** (2) proves whether the *information is present/extractable*; it does **not** prove Gap 1 will fix the online head. A positive D2 plus a *broken* reward path (4) means "build Gap 1" would still not recover blueprint. Read the sequence as: (1) motivates → (2) confirms information value → (4) confirms the reward path can carry it → *then* build. Only if (2) shows site-features lift the ceiling **and** (4) shows survivors carry blueprint signal should any obs-schema change (Obs V5) or the slot-conditional-head rework be built — and never silently mid-experiment (TIP D3/D4 discipline). The `P(blueprint|slot)` table (1), learnable-fraction audit (3), and survival read (4) are the "cheap confirmatory reads"; (2) is the load-bearing discriminator.

---

## 8. Temporal / Markov sufficiency (the F2 lesson generalized)

What the LSTM is currently forced to **reconstruct from memory** that should arguably be **Markov-visible**:
- **Pending-commit membership** — a single-step, op-only, non-slot-attributed echo over ~W intervening WAITs (source-proof §2). *Fragile — the F2 case.*
- **Plateau duration beyond 5 steps** — the history window truncates it (§3 WAIT).
- **Host-stability latch** — `host_stabilized` deliberately removed; must be integrated from raw loss history.
- **Commit epoch** — needed for any commit-relative schedule; only the fixed-cadence countdown is derivable from epoch(0).

Compounding this: **masks are logit-only** (`factored_lstm.py:849–864`). The trunk representation — and therefore *every* head and the value function — is computed **blind to the constraint structure** (e.g. "all slots full ⇒ only WAIT is valid" is invisible to the representation; it is only stamped onto the logits afterward). This weakens slot/op differentiation and interacts with the forced-WAIT / decision-density issue (D5 telemetry, `telemetry.py:1108–1116`). A per-slot `occupancy`/`pending` bit and a `plateau_epochs` count are the Markov-visible fixes that reduce the memory burden the F2 case showed to be fragile.

---

## 9. How this slots into TIP Phase D

This audit **confirms and sharpens** TIP Phase D's thesis ("per-slot features describe the seed, not the host tissue; dormant slots are observationally symmetric"). Two revisions:
- **D adds an architecture workstream.** D as written is obs-only (D1 emit → D2 probe → D3/D4 obs V4). The §2 finding means the blueprint fix is **obs + slot-conditional heads**; D2's ladder should therefore also test whether a *slot-conditional readout* (not just richer features) lifts the ceiling.
- **D1's cost is lower than assumed.** Static site descriptors are free (`InjectionSpec`), and slot-local gradient health is an *aggregation over `layer_range`* of the existing `collect_per_layer_gradients` — not greenfield. Only activation-map stats are truly new instrumentation.

---

## 10. SME protocol summary

- **Overall confidence: High** on the structural spine (seed-rich / host-global-moderate / host-per-site-blind) and on the head-routing finding (§2) — both read directly from source with cites. **Medium** on the head-by-head rival verdict (§6) — reasoned from r9's sealed aggregates + architecture, grounded pending the three offline reads. **Low-Medium** on the minimal obs feature set — that is precisely what the D2 ladder is for.
- **Key information gaps:** (1) `head_blueprint_learnable_fraction` and the `P(blueprint|slot)` table were not pulled — the Karn store timed out; I specified them as the cheap confirmatory reads rather than mine 6.8 GB of r9 events. (2) No measurement of the 3-fixed-slot positional prior's strength. (3) `get_action`/`evaluate_actions` sampling order not traced for implicit conditioning (assessed low-risk given logit-only masking). (4) Fossilize-outcome vs contribution-variance not measured (Gap 4).
- **Caveat:** r9 is K=1 / obs-v3 / shaped (SEALED). The heads' behaviour under K>1 / obs-v4 may differ; the *structural* findings (missing per-site features, shared-trunk head independence, logit-only masks) are regime-independent and hold today on the current code.
