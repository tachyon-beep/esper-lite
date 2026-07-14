# Round-24 brief — Holistic observation-space audit: does Tamiyo see enough about the hosts?

Date: 2026-07-14. Prepared for prime review (self-contained; assumes the reader knows the arc through
round 23 / PDR-0098 but nothing session-local). Provenance: the F2 pending-visibility finding (the committed
flag was structurally aliased) prompted the owner to ask whether that hole was an instance of a class:
*"do we give Tamiyo enough data about the hosts to make these decisions?"* Two specialist audits ran in
parallel — drl-expert (Markov-sufficiency: what should the policy see, per decision) and pytorch-expert
(pipeline engineering: what exists, what reaches the obs, what it costs to change) — followed by a
main-session verification pass in which every load-bearing claim below was re-checked against source
first-hand. Claims are tagged [VERIFIED] (main session read the code), [AGENT] (specialist claim, spot-checked
plausible, not independently re-read), or [CORRECTED] (specialist claim amended on verification). Full
reports: `docs/analysis/2026-07-14-obs-audit-{drl,pytorch}.md`. This audit is FUTURE-WORK SCOPING — nothing
here alters the frozen-track permanence experiment except where explicitly flagged (Q2/Q3).

---

## 1. The two headline findings

### 1.1 [VERIFIED] The action heads are architecturally slot-blind — no obs addition alone can fix blueprint conditioning

`factored_lstm.py:838-845`: all eight action heads (slot, blueprint, style, tempo, alpha_target,
alpha_speed, alpha_curve, op) are independent projections of the same `lstm_out`; masks are applied to
logits post-forward; there is NO autoregressive routing (verified — the sampling path was checked for
slot-first conditioning; none exists). The joint policy factorizes as Π P(head | trunk), so the blueprint
head cannot condition on the *sampled* slot.

**[CORRECTED] — the severity nuance.** The drl audit phrased this as "slot-averaged trunk … cannot express
'attention here, conv there' regardless of what obs you add." Two amendments from verification: (a) the
trunk input is NOT slot-averaged — it carries all three slots' feature blocks position-wise, so the
blueprint distribution CAN be state-conditional, including on per-site features, through the trunk; what is
impossible is conditioning on the sampled slot *draw* within a step; (b) r9's slot head ran at 0.68–0.75
confidence, so in the common near-deterministic case the trunk largely "knows" the winning slot and can
shade the blueprint marginal toward it. **The correct statement: site-conditioned blueprint choice is
strongly degraded (marginal-over-draws with a partial mitigation channel), not impossible.** The remedy is
unchanged — per-site obs features (necessary) + autoregressive slot→downstream routing (to make them fully
usable) — but the D2 ladder read (below) should be interpreted against "degraded," not "impossible."

### 1.2 [VERIFIED — a live defect, not a gap] Obs dim +13 (`contribution_velocity`) is silently dead: constant 0.0, every slot, every run ever, BOTH obs versions

Four-link chain, each link read first-hand: the trainer computes velocity on kasmina-internal metrics
(`vectorized_trainer.py:1383-1388`, EMA 0.7/0.3) → `SeedMetrics.to_leyline()` (`slot.py:281-301`)
enumerates 20 fields and OMITS it → reports carry `metrics=self.metrics.to_leyline()` (`slot.py:570`) →
`features.py:911-914` reads the leyline dataclass default 0.0 (`reports.py:54`). One-line fix; filed as bug
esper-lite-f0a82adccb; timing is Q3 below. Cheap runtime confirmation: dump `obs_normalizer.var` for the
+13 dims (should be ~0).

**Cross-report consequence (main-session catch):** the drl audit graded PRUNE "well-served (contribution +
velocity + interaction + gradient health)" — but velocity is dead, so **PRUNE's information grade drops a
notch**, and the drl report's per-head table should be read with that amendment.

---

## 2. The decision→information map (drl audit, with main-session amendments)

| Decision | Grade | Basis |
|---|---|---|
| GERMINATE (where/what/when) | **Worst-served** | Zero per-site host features (dormant slots read as all-zeros); blueprint head additionally architecture-degraded (§1.1); WHEN weak (plateau signal withheld, §3) |
| SET_ALPHA target | Served | The healthiest head (real 3-way preference, r9 A3) — proves the pipeline learns preference when signal is present and the head is exercised |
| SET_ALPHA speed/curve | Collapsed — **credit-starved (primary)**, obs secondary | Collapse to ONE action ≠ uniform; tiny causal/learnable fraction; [AGENT] |
| PRUNE | Moderately served [CORRECTED from "well-served"] | Contribution + interaction + gradient health live; velocity DEAD (§1.2); its known pathology (force-prune) is reward/floor, not obs |
| FOSSILIZE | Moderate | F2's pending bit now ruled in; remaining gaps: contribution *stability* (variance — only an EMA slope exists, and it's the dead dim) and explicit remaining-horizon |
| WAIT | Partial | Opportunity cost weakly visible; the 5-step history window cannot distinguish a 5-epoch pause from a 20-epoch plateau |

**Per-head rival-hypothesis verdicts** (starved vs floor vs credit vs horizon; [AGENT], consistent with
main-session first-hand r9/phase-1 knowledge): blueprint = information-starved + architecture-degraded
(above-uniform ⇒ gradient flows; it learned the marginal it can express); alpha_speed/curve =
credit-starved primary; alpha_target = healthy; op = floor pathology (no obs change fixes it). Plus a
third live rival for blueprint: **reward-horizon starvation** — with 55% of good held seeds force-pruned
inside ~13 epochs (PDR-0079), blueprint advantage is near-zero-mean noise regardless of obs or
architecture. Discriminator specified (below). Note the arc interaction: **the permanence settlement +
floor repair IS the fix for the horizon channel — so this future work is partially gated on the current
experiment's outcome.**

---

## 3. The exists-but-unplumbed inventory (pytorch audit; top items, tiers = engineering cost)

Framing [AGENT, spot-verified on the starred items]: **six of the top signals feed the reward or the
heuristic controller but NOT the RL obs — the policy is asked to optimize a target built from signals it
cannot see.**

1. ★[VERIFIED] `committed_val_acc` — host accuracy with ONLY fossilized seeds (set at
   `vectorized_trainer.py:1367/:1422`; zero references in features.py). Arguably THE signal for a
   permanence-deciding policy; the obs carries only all-seeds val_acc. Tier 0.
2. ★[VERIFIED] `plateau_epochs` — the exact germination trigger the heuristic controller keys on
   (`tamiyo/tracker.py:195,287`); absent from the obs. Tier 0.
3. [AGENT] train_accuracy/train_loss → the train−val generalization gap (overfit-vs-undercapacity is
   indistinguishable from val-only history). Tier 0.
4. [AGENT] The full counterfactual structure — pairwise interaction matrix, Shapley std (uncertainty!),
   regime classification — reduced to one clamped scalar + one summed scalar before the policy sees it.
   Tier B/1.
5. ★[VERIFIED] `boost_received`/`upstream_alpha_sum`/`downstream_alpha_sum` — computed, copied to leyline,
   and explicitly REMOVED in V3 (`features.py:408` docstring). Tier 0 (restoration).
6.–8. [AGENT] gradient-norm ratios, seed/host param counts (the reward optimizes per-param gain; the
   policy sees no param counts), health/previous-stage flags. Tier 0.

**Pipeline pathologies (ranked):** P0 = the dead velocity dim (§1.2). P1 = the EMA obs-normalizer runs over
sentinels/one-hots — but [AGENT, carefully argued] sentinel distinguishability SURVIVES normalization; the
real edge effects are bounded (clip spikes only for long-dead dims). P2 = ★[VERIFIED] `num_holding` counts
HOLDING **or FOSSILIZED** (`features.py:825`) — the policy's own occupancy summary cannot distinguish
parked from permanent, which is this project's entire subject. P3 = the collapsed alpha heads are LEARNED
collapse, not mask-forced (masks all-ones) — consistent with the credit-starvation verdict. No
train/act normalization skew (verified by the agent; both paths consume the same tensor).

---

## 4. Direct consequences for the CURRENT arc (not future work)

- **F2 implementation landmines** [VERIFIED layout math]: the per-slot layout is
  `slot_offset = 24 + slot_idx * slot_feature_size` with VERSIONED size constants — adding the pending bit
  at +35 without bumping `OBS_V4_SLOT_FEATURE_SIZE` 35→36 in lockstep makes slot 1's base collide with
  slot 0's new dim at flat index 59: silent corruption, no shape check fires. Plus: explicit schema-version
  bump (forces the desired normalizer re-warm via the checkpoint contract), both arms run the SAME larger
  schema (arm A hardwires 0 — never arm-gate the dim's existence), and any kasmina-side per-slot signal
  needs its `to_leyline` copy line (the exact omission class that produced §1.2).
- **num_holding (P2)** conflates HOLDING+FOSSILIZED in the base dims — worth an interpretation note in the
  experiment's telemetry reads (occupancy-conditioned analyses see parked+permanent as one number).

---

## 5. The offline read ladder (all on logged data; no obs change; no retrain)

1. `P(blueprint | slot)` contingency table — flat ⇒ site-blind today (cheapest, most specific).
2. **D2 information-ceiling ladder (load-bearing):** predictors {host-only | slot-only | ObsV3 |
   ObsV3+candidate-site-features} → {cf-contribution, fossilize-success}. Separates information-absent from
   critic-can't-extract; pre-validates the obs half of gap #1 BEFORE any schema change. Pre-registered
   caveat: **a positive D2 proves info-PRESENCE, not fix-EFFICACY** — the online head won't move if the
   reward path (floor/horizon) is broken.
3. Per-head learnable-fraction + floor-binding audit (fields already logged).
4. Reward-horizon discriminator: among survivors past the ~13-epoch destruction horizon, does blueprint
   choice predict cf-contribution?
5. `obs_normalizer.var` dump — runtime confirmation of §1.2 and a general dead-dim sweep.

---

## 6. Questions for the primes

**Q1 — Sequencing.** Proposed: this work is a POST-permanence-experiment bet (the horizon-starvation rival
means blueprint-head efficacy is confounded until the settlement/floor work lands), with ONLY the offline
read ladder (§5) running before/alongside. Agree, or is any piece safely parallel?

**Q2 — Architecture scope.** The autoregressive slot→downstream-head routing is a policy-network change
(retrain, new PPO dynamics, interacts with per-head credit assignment). Should it enter the same future bet
as the obs additions (the drl audit's view: obs-necessary-but-not-sufficient argues for bundling), or is
staged obs-first/architecture-second cleaner given the D2 ladder can only test the obs half?

**Q3 — The dead velocity dim: fix timing.** It is constant 0.0 in ALL historical data — so NOT fixing it is
the maximally matched-control choice for the permanence experiment (an inert dim, identical across arms and
across history), while fixing it bundled with the F2 re-warm is one fewer future schema bump but changes the
obs distribution mid-arc. Main-session lean: DEFER to the post-experiment Obs V5 batch; the pytorch agent
recommended fix-with-F2. Adjudicate.

**Q4 — Obs V5 batching.** The re-warm is the dominant recurring cost, arguing for ONE batched schema bump
(velocity fix + committed_val_acc + plateau + gap + selected Tier-0s) gated on D2 results — vs the
minimal-delta instinct of adding nothing until each candidate individually clears an information-ceiling
probe. Where's the right point on that spectrum, and does the answer change the F2 implementation now
(which currently adds exactly one bit)?

**Q5 — The num_holding conflation (P2)** — worth fixing in the same future bump, or is the parked/permanent
distinction better carried by the F2 pending bit + a separate fossil count than by re-defining an existing
dim (which silently changes an input's meaning under matched-control)?

## 7. Anti-overclaim ledger (what this audit does NOT establish)

- No efficacy claim: every gap is information-PRESENCE reasoning; whether the policy would USE any added
  signal is untested (D2 tests extractability; only a run tests use).
- The per-head rival verdicts rest on sealed-r9 aggregates + architecture reading; the four §5 reads are
  specified but NOT executed (Karn store timeouts prevented two confirmatory pulls).
- The r9 evidence is K=1/obs-v3/shaped regime; structural findings (architecture, dead dim, unplumbed
  inventory) are regime-independent and hold on current code; behavioral magnitudes are not.
- Items 6–8 of the inventory and the P1 normalizer magnitudes are [AGENT] tier — plausibility-checked, not
  independently re-derived.
