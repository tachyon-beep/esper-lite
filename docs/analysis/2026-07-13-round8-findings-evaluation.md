# Round-8 — Independent evaluation of the round-7 findings

Date: 2026-07-13 · Owner-requested: "evaluate the findings made in the last session (they're quite numerous)"
Evaluator: product owner (Claude), with an independent adversarial **drl-expert** (RL theory) and **pytorch-expert**
(autograd/numerics), both of which *numerically executed* the shipping symbols rather than trusting the doc; plus
external primes (GPT/Claude) relayed by the owner, whose new claims are re-verified against source below.
Sources audited: `src/esper/tamiyo/policy/action_masks.py`, `src/esper/tamiyo/networks/factored_lstm.py`,
`src/esper/simic/agent/ppo_update.py`, `.../rewards/contribution.py`, `.../rollout_buffer.py`, `leyline` floor
constants; `docs/analysis/2026-07-13-decision-point-diagnosis.md` (rounds 1-7); PDR-0069/0070/0071.

**Empirical (telemetry) recomputes LANDED** (direct DuckDB over the 6.4 GB `events.jsonl`, ~1.08M decisions/seed):
cluster-B LOO overlap CONFIRMED & strengthened (exact medians revised — banked pair was a pool); |H|=1 corroborated at
63%/72%; hindsight inert 0.004-0.005% confirmed (functional-but-rare-trigger); **N1 blueprint clamp REVISED DOWN**
(realized num_valid=7, ceiling ~2× uniform, realized 1.3-1.4× — a mild ceiling, NOT a near-uniform sampler). Anti-artifact
gate verified (91% PRUNE null-rate = entirely TRAINING-stage, correctly excluded).

---

## Bottom line

The round-7 finding set is **among the more solid diagnostic work in this epic** — the core mechanism is *forced by
the autograd*, not argued, and both independent experts reproduced it numerically. The session was also unusually
self-disciplined (its own "DO NOT BANK" ledger retracted ~7 over-reads, including a *comforting* one). The evaluation
does not overturn the reframe; it does three things:

1. **Confirms the mechanism cluster (claims 1-4) as SOUND** — the strongest is the |H|=1 whole-head case, which is
   *bit-exactly* zero gradient (not fp-noise).
2. **Localizes the one real overclaim** to a single inferential step (claim 5, "the gate passes → the signal exists"),
   which is not merely unmeasured but **directionally ambiguous**, and to PDR-0069's *headline verbs* ("root cause,"
   "gate passes"). **PDR-0071 already self-corrected most of this**; the fix is to hold the epic and the workspace to
   0071's calibration, not 0069's.
3. **Surfaces five things the 7-round session under-counted** — four of them one transform wearing four hats. The
   anti-WAIT floor is simultaneously a **gradient censor**, an **expressiveness ceiling** (wide heads), a **KL-dilution
   source** (optimizer step size), and a **monitoring blind-spot** (entropy/collapse health is *guaranteed* to look
   green exactly where the head is deadest). Each round saw one facet.

**Reframe verdict:** justified as a **reprioritization** (you cannot move a zero-gradient logit with a critic or reward
arm, so run the primitive experiment before buying more critic/reward runs); overstated as a **supersession** ("root
cause found") because causal sufficiency is unproven (fix unrun), advantage is unlogged (gate unverified), and the
censoring is *partial* (a real |H|≥2 minority + shared-trunk + boundary channels still train). Honest status = PDR-0071
§5's own words: *the leading proximal explanation for flat HOLDING preferences, not a universal cause.*

---

## Triage of the finding set

| # | Finding | Cluster | Verdict |
|---|---------|---------|---------|
| 1 | Per-action dead-zone (floor-bound ⇒ zero logit gradient, ratio 1.0) | A floor | **SOUND** (autograd-forced; both experts ran it) |
| 2 | Whole-head dead-zone at \|H\|=1 (entire output constant) | A floor | **SOUND** (bit-exact 0.0); rate reconstructed, now **corroborated by telemetry at 63%/72%** (banked 67%/80% slightly high) |
| 3 | Policy-wide (all 8 heads, differentiable update leg) | A floor | **SOUND as a code fact**; blast radius *measured* only for op |
| 4 | SET_ALPHA 0.55 = analytic cap `1−(n−1)f`, not learned | A floor | **SOUND** |
| 5 | "The gate passes; advantage rises; signal exists" | A floor | **OVERSTATED** — unknown in *both* directions; single most important overclaim |
| 6 | Fix = differentiable mixture floor, λ∈{0,0.3,0.6}, STE rejected | A floor | **SOUND as stated**; the *experimental equivalence* it implies is **confounded** (λ fixed vs num_valid) |
| 7 | Co-requisite sequencing (`hindsight_credit` gate) | A floor | **REASONABLE but SPECULATIVE**; and the gate metric is mis-specified (below) |
| — | Calibrated-diagnosis empirics (LOO medians n=2, negative-LOO, re-blend 75/88%, churn) | B empirics | **Recomputed (telemetry): conclusion CONFIRMED & strengthened; exact medians REVISED** (banked pair was a pool) |
| — | "DO NOT BANK" over-read ledger | C retractions | **Complete & correct** on spot-check (see below) |
| N1 | **Blueprint / wide-head EXPRESSIVENESS clamp** (survives any gradient fix) | NEW | **Mechanism real; severity REVISED DOWN by telemetry** — num_valid=7 (not ≈13), ceiling ~2× uniform, realized 1.27-1.41×: a mild ceiling, NOT a near-uniform sampler |
| N2 | **`approx_kl` dilution → inflated step size for all heads** | NEW | **CONFIRMED, live** (`target_kl=0.015`) |
| N3 | **Entropy/collapse monitoring structurally blind to the dead-zone** | NEW | **CONFIRMED** (sharpest instrument-validity finding) |
| N4 | **Floor-bound samples enter global advantage standardization** | NEW | **CONFIRMED mechanism**; magnitude needs re-run |
| N5 | **λ-sweep confound** (fixed λ changes floor magnitude by num_valid) | NEW | **CONFIRMED**; fix = per-num_valid λ (`f+(1−nf)p_i`) |

---

## Cluster A — the floor dead-zone (claims 1-7)

### SOUND: the mechanism (claims 1-4)

Verified three independent ways (my algebra, drl-expert numeric run, pytorch-expert numeric run on the *shipping*
symbols):

- **Per-action (claim 1).** For an underweight action, `_apply_floor_to_logits` returns `q_i = effective_floor`
  (`action_masks.py:569-573`), and `effective_floor = min(min_prob, 0.99/num_valid)` depends only on the boolean mask.
  The trailing `log_softmax` does *not* leak gradient back: the transform preserves total valid mass = 1 identically in
  `z` (`|U|·f + remaining_mass ≡ 1`), so `logsumexp ≡ log 1 = 0` and `logp_i = log f` is constant. Measured grad-norm
  **1.8e-9** vs **0.816** for an overweight control in the same batch. Ratio **exactly 1.0**.
- **Whole-head (claim 2).** At |H|=1 the sole overweight op's `q_k = p_k·(remaining_mass/p_k) = remaining_mass = 1−(n−1)f`
  — the softmax *cancels*, the entire vector is constant. Measured: the winner itself (0.55) has grad-norm **0.0
  bit-exactly** (not fp-noise). This is the strongest instance and the broadest (kills the whole head, dominant op
  included). **Reinforcing wrinkle (pytorch-expert):** an action that becomes *so* dominant it is the sole overweight
  op re-enters |H|=1 — dominance itself re-freezes the head (measured ratio 16.0, grad 3.7e-8).
- **Policy-wide (claim 3).** `PROBABILITY_FLOOR_PER_HEAD` (leyline) is nonzero on all 8 heads (op 0.15, slot 0.05,
  blueprint 0.12, style 0.08, tempo 0.12, alpha_target 0.08, alpha_speed/curve 0.06), and the floor is applied in the
  **differentiable** `evaluate_actions` loop (`factored_lstm.py:1533`, over every head) as well as rollout. Mechanism is
  genuinely policy-wide; *evidenced* blast radius is op-only.
- **Cap (claim 4).** `1−3·0.15 = 0.55` for 4 legal ops, reproduced exactly. The round-7 retraction of round-6's
  "SET_ALPHA learned" is correct.

**Free confirmation worth banking (Claude-prime, checks out).** The round-2 "policy invariance" table (op_conf ≈0.35,
op_entropy ≈0.87, flat across every LOO bin) is the *analytic fingerprint of the pinned simplex*: for `[0.55,.15,.15,.15]`
E[P(chosen)] = Σq² = 0.37 and H/ln4 = 0.853; blended with a ~25% |H|≥2 minority you land on the reported 0.35/0.87. That
table — collected three rounds before the mechanism was found — was never "LOO isn't the axis"; it was "there is no
output freedom to *have* an axis." Strong retrospective CORROBORATION (per gpt-prime: NOT an *independent proof* — those
aggregate statistics are not unique to the pinned simplex; the direct autograd result + the measured |H|=1 rate are the evidence).

Precision caveats the "exactly zero / exactly 1.0" phrasing glosses (adopt these verbs):
- **Two-tier exactness:** output-logit exactly constant; the log-prob the loss sees is zero *up to ~1e-9 fp noise*
  (from the logsumexp-over-overweight renorm); only |H|=1 is bit-exact.
- **Interior-only, not boundary:** ratio=1.0 holds only while the *chosen* action stays strictly inside the underweight
  region across the PPO epoch. It breaks at the floor crossing (autograd flows nothing through the `<`/`>=` condition —
  a genuine discontinuity with a defined-but-misleading interior subgradient).
- **Per-sample, not per-parameter (both experts insist):** a floor-bound *(state, action)* pair delivers no credit to
  *its own action*; the shared op-head weights still move via other samples. **"The op-head can't learn" overstates the
  autograd result.** Defensible form: *the head learns commit preference only from a heavily down-sampled,
  selection-biased minority of realized outcomes.*

### OVERSTATED: claim 5 ("the gate passes → the signal exists") — the one real overclaim

The *immediate* FOSSILIZE reward is genuinely LOO-graded (−0.7 → +8.2; follows from the reward formula). The inference
to **advantage** does not follow, and it is stronger than "unmeasured" — it is **directionally ambiguous**:

- Advantage `A ≈ r + γV(s′) − V(s) + continuation`. `V(s′)` is **not** op-independent (fossilizing consumes the seed)
  and is **not logged**.
- PDR-0069's supporting argument — "V(s) is op-independent, so advantage rises too" — is a **non-sequitur**. V(s) exists
  precisely to *absorb* the predictable, LOO-driven part of the return; the LOO feature feeds the shared trunk that
  produces V(s) (`features.py:827-832` → trunk, per the session's own code trace). A well-fit LOO-conditioned baseline
  can **flatten** the fossilize advantage even under a perfectly graded reward. So advantage-grading is unknown in *both*
  directions.
- "The policy cannot act on it" overshoots (the |H|≥2 minority + trunk + boundary crossings carry gradient).

**This is the load-bearing residual overclaim.** It is narrow: PDR-0071 §4 already walked the head wording back to
"a strong LOO-graded commit reward the op head usually cannot DIRECTLY learn from" and added the reversal trigger
"if per-decision advantage-vs-LOO is FLAT, the gate does NOT pass." Delete the assertion "signal survives the 9-pt
spread" (PDR-0071 §4) — it is unmeasured and contradicts that same reversal trigger.

**The decisive consequence for sequencing:** the *only* bridge from "mechanism exists" to "fixing it helps" is
per-decision advantage-vs-LOO. That makes the **harness-enablers task (esper-lite-7fe21bd091) the validity gate for the
entire reframe, not a convenience.** AND — a zero-GPU shortcut may exist: **`value_estimate` (per-decision V(s)) is
already in the decision telemetry** (`emitters.py:261,351`). If per-decision reward + episode structure are also present,
GAE advantages are reconstructable *offline from the existing seed41/42 logs*, letting us read the true gate **before
buying any GPU** rather than waiting on the re-run. This is the single highest-value unread number in the set.

### SOUND-but-confounded: claim 6 (the fix and the λ-sweep)

Differentiability is real (`∂q/∂z = (1−λ)J_softmax`, nonzero incl. at |H|=1; measured 5.4e-4 … 7.8e-3 on the exact head
the current fn zeroes). STE-rejection is correct. But the *experimental* claim "λ=0.6 = current behavior + gradient" is
confounded on two axes:

- **Distributional, not just min/max.** At n=4, current maps raw `[.70,.10,.10,.10]→[.55,.15,.15,.15]`; mixture λ=0.6 →
  `[.43,.19,.19,.19]`. Same guarantee, materially different interior redistribution.
- **num_valid-dependent (N5).** Mixture floor `= λ/num_valid`: n=2→0.30, n=3→0.20, **n=4→0.15**, n=5→0.12; the current
  floor `= min(0.15, 0.99/n)` = 0.15 across n=2..4. Real HOLDING/PRUNE states span num_valid (age-gating→n=3;
  GERMINATE-legal→n=5). So a **fixed** λ=0.6 is a heavier, state-varying floor everywhere except the n=4 slice, and the
  sweep confounds "restore gradient" with "change exploration magnitude by state."
- **Illegal-action pitfall (pytorch-expert):** the uniform term must be `mask/num_valid` and the softmax over masked
  logits, with `q` re-masked before `log`, or you `log(≈0)` on illegal actions → spurious large gradients. (The mixture
  is otherwise lower-bounded `q ≥ λ/num_valid`, needs no clamp, cannot NaN, and is *compile-friendlier* than the current
  nested `torch.where`.)

**Fix for the experiment:** use a **per-num_valid** floor that pins magnitude at the current 0.15 for all n
(gpt-prime's `q_i = f + (1−nf)p_i` form), so λ isolates the *gradient* property; and keep the **current hard transform
behind a config flag as a matched control arm** (else new instrumented runs are compared to uninstrumented seed41/42).
"One primitive, one build" is right about the *code*; it does not mean "no control arm."

### REASONABLE-but-SPECULATIVE + mis-specified metric: claim 7 (co-requisite)

Promoting `hindsight_credit`/transfer to a shipping-gate co-requisite is defensible **risk management** — unfreezing a
gradient onto a known-misaligned proxy (current LOO) turns a *random explorer* into a *proxy-maximiser*, which is
strictly worse; "the dead-zone may be an accidental safety property" is a real possibility. But:

- The "LOO-greedy commit collapse" failure mode is *predicted, not observed*. Present it as a hypothesis the guardrails
  test, not established causation (PDR-0071's reversal triggers already do this correctly).
- **The gate as written is satisfiable on a technicality.** "Until the transfer settlement *carries signal*" — a
  `hindsight_credit` that fires but arrives 200 steps downstream, γ-discounted and GAE-smeared, "carries signal" and
  does nothing. **Restate the gate in advantage terms:** *per-decision advantage at FOSSILIZE must be graded by
  transferred value, not by current LOO* — measured on the very per-decision advantage log claim 5 already demands.
  Same instrument, three jobs (gate read, collapse-brake read, ship read).
- **The trigger is keyed on a non-causal field.** `compute_scaffold_hindsight_credit` (`contribution.py:1230`) returns
  0 unless `boost_given>0 AND beneficiary_improvement>0`, and `beneficiary_improvement` is passed as `total_improvement`
  (`fossilize.py:215`) — the **non-causal host-drift** field the session itself retracted as over-read #7. So the 0.005%
  rate most plausibly reflects a **rare trigger scenario** (a co-resident scaffold relationship + positive host drift +
  the beneficiary fossilizing), *not* a broken channel with a tunable weight. **Root-cause the 0.005% before promoting
  it to a shipping gate** — otherwise a possible design change (add a causal transferred-value field) is mis-labelled as
  "turn the existing channel back on." **Telemetry confirms:** hindsight fires **0.0054% (s41) / 0.0040% (s42)** of
  decisions — de facto inert, but the channel is **functional, not broken** (43-58 real positive fires/seed); the low
  rate is a rare *trigger scenario* — credit lands on the beneficiary's *next* action (mostly WAIT/SET_ALPHA/PRUNE,
  almost never FOSSILIZE). So "make it carry signal" is a **design** question (add a causal transferred-value field), not
  "turn a broken channel back on." (Top-level `hindsight_credit` is always null; only `reward_components.hindsight_credit`
  populates.)

### The single most under-weighted thing (drl-expert)

**The mechanism cannot distinguish "the floor froze a real commit preference" from "there is no commit preference and the
floor is just where a weak/absent logit lands."** The dead-zone is **state-dependent, not a frozen logit** — a FOSSILIZE
preference that genuinely grew (via the |H|≥2 minority, the shared trunk, or boundary crossings) *would* cross the floor
and train normally. So the whole mechanism is equally consistent with the live alternative hypothesis: *SET_ALPHA is the
new WAIT; the policy doesn't want to commit and the floor merely masks it.* **Only the λ=0 necessity smoke
disambiguates** — and the docs correctly pre-register that λ=0 likely collapses to SET_ALPHA, not WAIT (so "didn't
collapse to WAIT" ≠ "floor unnecessary"). Do not let the reframe bury this alternative.

---

## The NEW findings the 7-round session under-counted

### N1 — Blueprint / wide-head EXPRESSIVENESS clamp (a *different* pathology; severity REVISED DOWN by telemetry)

`effective_floor = min(min_prob, 0.99/num_valid)`. On a **wide** head the `0.99/n` branch binds and the cap
`1−(n−1)f` collapses toward uniform. **BlueprintAction has 13 members** (confirmed). If a GERMINATE decision has ~13
legal blueprints: floor ≈ 0.99/13 = 0.076, cap = 1−12·0.076 ≈ **0.086**, uniform = 1/13 ≈ 0.077 → **the blueprint head
can express at most ~1.1× preference over uniform, for *any* blueprint, regardless of its logits.** This is a
**representational** clamp, not a gradient clamp — **it survives any gradient fix.** PDR-0069/0071 diagnosed only the
*learning* pathology (op head); this *expressiveness* pathology on wide heads is separate and un-noticed.

- Contrast: the op head (4 ops, floor 0.15) has cap/uniform = **2.2×** — ample room; its problem is purely the |H|=1
  gradient dead-zone.
- **Severity RESOLVED by telemetry (REVISED DOWN).** The realized legal-blueprint set is **7, not ≈13** (CIFAR
  hard-masks 6 of the 13 blueprints), so the `0.99/n` branch does NOT bind (0.99/7 = 0.141 > 0.12) and the floor stays at
  min_prob 0.12: ceiling `1−6·0.12 = 0.28 = 1.96× uniform`, and the head *realizes* 1.27-1.41× uniform with a
  **seed-specific top blueprint** (s41 bottleneck 20.2%, s42 conv_heavy 18.1%; blueprint_confidence median ~0.13,
  max ~0.33). So the blueprint head expresses genuine, logit-driven preference — it is **NOT** a near-uniform sampler.
  The specific hypothesis (n≈13 → ~1.1× → "decorative logit layer") is **REFUTED on the numbers.** What survives, weaker:
  the 0.12 floor still imposes a real ceiling — no blueprint can exceed ~28% — capping how peaked GERMINATE can get. A
  real-but-mild constraint, not a separate crisis, and **not a new epic item.**
- **N5 (the λ confound) still stands, magnitude revised.** A fixed λ=0.6 matches the op head (n=4 → floor 0.15) cleanly,
  but for the blueprint head (n=7) it loosens the floor 0.12→0.086 (cap 0.28→0.49, a ~1.7× expressiveness jump — not the
  ~3× originally argued, but still a confound). Per-num_valid λ (`f+(1−nf)p_i`) remains the correct control so the op-head
  effect is isolable.

### N2 — `approx_kl` dilution inflates step size for every head (blast radius → optimizer)

`approx_kl` is a **weighted sum across heads** (`ppo_update.py:99`) and it drives KL early-stopping (`ppo_update.py:134`
and `vectorized.py:546-548`: `early_stop = approx_kl > 1.5·target_kl`). **`target_kl` defaults to 0.015 and the run
config does not override it** — so early-stop is *live*. Floored heads contribute structural-zero KL, diluting the
aggregate → early-stop fires *later* than the true (op-inclusive) KL warrants → **inflated effective step size / extra
epochs for the learnable heads.** A confirmed second-order effect PDR-0069/0071 never mention. (Also makes PDR-0071
amendment 6 — log per-head KL — load-bearing, not a nice-to-have: the aggregate is contaminated.)

### N3 — Entropy/collapse monitoring is STRUCTURALLY BLIND to the dead-zone (sharpest finding)

Both entropy terms consume the **post-floor** logits (`ppo_update.py:406-418` bonus; `429-450` floor penalty; and
`factored_lstm.py:1535`). Two consequences, both measured by the drl-expert:
- At |H|=1 the entropy **bonus** gradient is ~0 (8.5e-10) — the last "indirect escape channel" PDR-0069 hoped for is
  *also* dead exactly where the surrogate is.
- The entropy **floor penalty** never fires there: at |H|=1 the floor lower-bounds normalized entropy at **0.61 (n=2) /
  0.745 (n=3) / 0.853 (n=4)** — all far above the op entropy floor (0.30) and the collapse threshold (0.08).

**"Entropy healthy" and "op-head gradient dead" are guaranteed to co-occur.** The same transform that zeroes the
gradient guarantees the head *looks green* to every entropy/collapse instrument. This retro-explains why the entropy
thermometer work never flagged this, and it is a first-order input to the Telemetry Improvement Program (a historical
scar the scar-suite should encode).

### N4 — Floor-bound zero-gradient samples enter global advantage standardization

`rollout_buffer.normalize_advantages()` (`:730-761`) pools all valid timesteps and subtracts a global mean / divides by
a global std (floored) with **no floor-aware exclusion**. So the ~94% gradient-censored commit samples shift the mean
and inflate the std that rescales the ~6% of commit samples that *can* learn — attenuating the real signal. Per-head
advantage norm exists but is default-OFF (`ppo_agent.py:146`). Mechanism confirmed; magnitude needs the re-run (this is
an independent argument for enabling per-head adv-norm, which was already flagged in a prior memory).

### N5 — the λ-sweep confound

Covered under claim 6. Fix: per-num_valid floor (`f+(1−nf)p_i`) + hard-floor control arm behind a flag.

---

## Cluster B — the calibrated-diagnosis empirics (RECOMPUTED)

Independently recomputed from the raw `events.jsonl` (direct DuckDB, ~1.08M decisions/seed), applying the exact
anti-artifact discipline: gated on **stage-at-decision = HOLDING**, fresh non-null `seed_contribution` on the decision
row, **no carry-forward**. The extraction reproduced the known ~91% PRUNE null-rate as *entirely* TRAINING-stage (α=0)
prunes — confirming the gate is correct and the banked discipline was applied consistently (not just to the *retracted*
numbers).

- **0% missingness on the HOLDING-gated cohort** — banked 0% **CONFIRMED**, both seeds, both ops.
- **Per-seed medians (last-measured LOO):** s41 FOSSILIZE **10.79** / PRUNE-from-HOLDING **10.14**; s42 FOSSILIZE
  **9.96** / PRUNE-from-HOLDING **9.00**. The banked pair (10.4 / 9.6) reads as a pool/average and does not reproduce
  per-seed → **REVISE the exact numbers**; the **conclusion is CONFIRMED and strengthened** — FOSSILIZE and
  PRUNE-from-HOLDING are **not separable by current LOO** (median gap only 0.65 [s41] / 0.96 [s42]; ~10% negative and
  ~65% ≥5 in *both* cohorts; IQRs nearly coincide). The sign of current LOO does not distinguish the realized terminal
  op — the load-bearing input to "current LOO is not the commit-worthiness axis."
- **|H|=1 corroboration (claim 2):** combined 0.55∪0.15 fingerprint = **63.3% (s41) / 72.4% (s42)** of HOLDING op
  decisions (stricter max-op=0.55 gives 47%/59% as a lower bracket). Same cross-seed direction as the reconstructed
  67%/80%, magnitude a few points lower — **cite 63%/72%.**
- **Not recomputed:** the re-blend 75%/88% and germinate/prune churn figures were out of this pass's scope — they remain
  at the session's original (appropriately-hedged) status. **Provenance:** the telemetry agent crashed mid-output and was
  resumed from transcript; figures cross-check against the known ~91% null-rate anchor (reasonable to trust), but
  spot-check any single number that becomes load-bearing for the GPU go/no-go.

## Cluster C — the "DO NOT BANK" ledger: spot-check

Reviewed for **completeness**, not just correctness. The retractions are sound and the lessons general (check a field's
null-rate AND its computation-condition before any statistic; a tidy benign story is as much a red flag as a tidy
alarming one; `total_improvement` is non-causal). No additional same-class over-read spotted in the BANK section by
inspection — and the cluster-B recompute (landed) reproduced the 91% null-rate as entirely TRAINING-stage, confirming the
discipline held for the banked numbers too. The ledger is a model of the discipline the
Telemetry Improvement Program is meant to institutionalize.

---

## Recommended sequence (evaluation → action)

Zero-GPU, in leverage order, *before* designing the arm:
1. **`num_valid` per head + blueprint/α-target realized-choice histograms** (settles N1 severity; falsifiable today).
2. **Is per-decision `value` + reward + episode structure in telemetry? If yes, reconstruct GAE offline and read the
   true advantage-vs-LOO gate** (settles claim 5 — the reframe's only unproven bridge — without GPU). `value_estimate`
   is confirmed present; check reward + dones.
3. **Root-cause `hindsight_credit` = 0.005%** (trigger-scenario rarity vs broken channel) before it stays on the
   shipping gate.
4. **Entropy pre/post-floor + advantage-standardization** — largely closed here (N3/N4 confirmed by code); quantify N4's
   magnitude from the re-run.
5. **Land the harness enablers (esper-lite-7fe21bd091)** — now framed as the *validity gate*, incl. per-head KL (N2).

Then design the arm with **per-num_valid λ** and a **hard-floor control**, and score it on *mechanism* metrics (per-head
KL activity where the head was frozen; raw FOSSILIZE/PRUNE sensitivity to LOO; a live advantage-vs-LOO relation; no
multi-choice collapse) + the product guardrails — **not** on "floor-bound rate → ~0," which is a tautology under any
mixture.

## Workspace drift this evaluation obliges the owner to fix

`current-state.md` ("the op head **can't learn** … because the anti-WAIT floor censors its gradient") and `metrics.md`
("MECHANISM IDENTIFIED … the commit logits **can't learn**") state **causal sufficiency as established.** The evaluation
shows: mechanism *established* (zero gradient on floored samples), causal sufficiency *hypothesized, pending the λ-sweep*
(the |H|≥2 minority still trains; the "SET_ALPHA is the new WAIT" alternative is live). Corrected verb throughout:
*established that floor-bound samples deliver no credit to their own action; whether removing the floor restores
commit-selectivity is what the λ-sweep tests.* This is the exact over-claim class the `instrument-validity-before-
interpretation` memory warns about; fix it in the resume brief so the next session doesn't inherit it as settled.

---

## Round-8 reviewer adjudication (2026-07-14 — gpt-prime + claude-prime, relayed by owner)

Both primes endorse the reframe-as-reprioritization, the drift fix, and the read-before-build ordering. Net: **nothing
overturns the mechanism or the evaluation.** Adjudication of the substantive new points:

**R0 — the floor is SINGLE-PASS (verified on the shipping fn, not by inspection).** gpt-prime's counterexample
reproduces exactly: raw `(0.700, 0.151, 0.0745, 0.0745)` → `(0.5758, 0.1242, 0.15, 0.15)`. An overweight op just above
the floor is scaled to **0.1242 < the 0.15 floor**; the max **0.5758 > the 0.55 cap**; normalized entropy **0.8267 <
claude-prime's H_min(4)=0.853**. Consequences: (a) **do NOT use "entropy ≥ H_min(num_valid) or STOP" as a validation
gate — it FALSE-FIRES** (claude-prime's proposed R2-as-STOP is unsafe); (b) the wide-head cap/entropy *bounds* are the
idealized (iterative) result, not the shipping behaviour — but **N1 severity is settled EMPIRICALLY by the realized
histogram** (telemetry: n=7, 1.3-1.4×), sidestepping the algebra dispute, and **claude-prime's n=13/near-uniform read is
refuted** by that measurement (it reasoned from the enum size, not the realized legal set — the same "good reasoning on
an unread input" it flagged); (c) the dead-zone is **untouched** (|H|=1 grad = 0.00e+00 exactly); (d) NEW minor
correctness note: **P(FOSSILIZE) ≥ 0.15 is NOT guaranteed** at the margin.

**§3 — the KL sequencing bomb (claude-prime; genuinely new, HYPOTHESIS).** The aggregate `approx_kl` that drives
early-stop (`target_kl=0.015`) is diluted by structural-zero heads — **forced slot (num_valid=1, ~97% of HOLDING),
floored op, pinned blueprint.** If early-stop rarely fires (test = **P8**), PPO has run with a **vacuous trust region** —
a **competing root cause for the Dec-2025 99.9%-WAIT collapse that predates the floor and is reward-independent.** The
floor was installed as the remedy and diluted KL *further* while clamping the symptom out of view. **Therefore the λ=0
necessity smoke will collapse and be mis-read as "the floor is necessary."** If P8 holds, **the KL-aggregation fix
(per-head or gradient-bearing-sample-only KL for early-stop) lands BEFORE any GPU arm** — it is a harness fix, not an
experiment. (N2 re-scoped: bank that structural zeros enter a live aggregate-KL mechanism; do NOT yet bank "inflates
every head's step size" — that needs the P8/per-head-KL read.)
**Read B result (2026-07-14, P8):** early-stop fired in **0/600 updates on BOTH seeds** — `approx_kl` is identically
0.000e+00. The trust region IS vacuous (claude-prime's §3 conclusion holds), **BUT the cause is K=1
(`recurrent_n_epochs=1`), not head-dilution:** at a single epoch the diagnostic KL is `KL(θ‖θ)=0` by construction (anchor
recomputed from the same θ at update start; ratio ≡ 1.0, so ratio-clip is *also* inert). Consequences: (1) the
head-dilution/N2 mechanism is **untestable on K=1** — per-head KL is 0 for every head at ratio≡1.0, so it becomes real
only at K≥2; (2) the aggregation fix is **misdirected** for these runs — the levers are **K≥2 epochs** (drift accumulates
on epochs ≥1) or **post-step KL gating** (rollout vs post-update policy), both owner-gated core-loop changes; (3) the λ=0
necessity smoke remains confounded (a no-trust-region run collapses easily regardless of the floor), so the §3 sequencing
caution stands on its conclusion even though its mechanism was wrong. A competing, reward-independent structural fact for
the owner DECIDE, alongside Read A.

**§5 — the floor already ran the RCT (claude-prime; highest-leverage zero-GPU read).** The op mix is flat across every
LOO bin (55/17/15/13, both seeds) ⇒ FOSSILIZE was forced at ~15-17% *independent of LOO* ⇒ marginal ignorability. So the
existing telemetry already estimates the counterfactual: read **`E[A | chosen=FOSSILIZE] vs SET_ALPHA vs PRUNE` per LOO
bin** (stratify by age/occupancy/host-trend). Three pre-registered outcomes: **graded** → PDR-0069 confirmed;
**flat** → reframe collapses, reward/critic line reopens (N2 prime suspect); **inverted** → committing high-LOO seeds is
*bad for return*, and the epic's founding premise ("under-fossilization is a defect") is itself the over-read — **never
tested, an aesthetic judgment.** Pre-register the inverted branch NOW. Caveat (claude-prime's own): the outcome is the
critic's return, not ground truth — it tells you what the fix would *teach*, not what's true; still exactly PDR-0071's
reversal trigger, at zero GPU.

**Experiment redesign — gpt-prime sharpens N5 (adopt).** The first causal arm changes **ONLY the op head** (leave the
other 7 on the shipping hard floor), else a commitment change could come from newly-expressive blueprint/α/slot heads.
Parameterize by **ρ = fraction of the current exploration envelope retained** (`q = (1−ρλ*)p + ρλ*·U`, λ* = n·f per head
per state), ρ∈{0, 0.5, 1} — this endpoint-matches the current floor at ρ=1 and removes the num_valid confound *within* the
op head. Arms: fresh hard-floor control (instrumented) · ρ=1 endpoint-matched · ρ=0.5 · ρ=0 abortable necessity smoke.
Global λ=0.6 is confirmed confounded (moves both axes on 7 of 8 heads). Note: gpt's minimal-isolation primitive on the
*blueprint* head would leave the learned policy ~1% of the mass — a perfect control but a useless *fix*, so the wide-head
expressiveness question is a SEPARATE arm with its own λ, addressed after its severity is measured.

**hindsight_credit — remove from the shipping gate entirely (both primes).** Fire-rate 0.005% is confirmed, but the
*eligible-event* denominator, per-FOSSILIZE rate, magnitude-when-fired, and advantage contribution are still unread, and
it is keyed on the non-causal `total_improvement` (over-read #12). Restate the gate in ADVANTAGE terms: *per-decision
advantage at FOSSILIZE must be graded by transferred value, not current LOO.* The co-requisite *reasoning* (don't unfreeze
a gradient onto a misaligned proxy) survives regardless of that specific channel's fate.

**Hold as HYPOTHESES, not banked:** "permanently frozen" (too strong — the gradient is gated on the head's own
uncertainty, a ~25% duty cycle; test P6/P7 = |H|=1 rate rising over training); "SET_ALPHA is stale lock-in" (plausible;
test with a fixed probe corpus across checkpoints, not an aggregate rate); N4 "attenuation" *direction* (recompute the
scale on learnable samples — inclusion can shift mean/std either way); N1 full severity (empirically mild at n=7).

**Consolidated zero-GPU read queue (unanimous ordering):**
1. **R0 single-pass — DONE** (above).
2. **Entropy-loss pre/post-floor + does `approx_kl` feed only early-stop or also adaptive LR/ent-coef** — partial:
   entropy is post-floor (dead at |H|=1) ✓, early-stop confirmed ✓; adaptive-LR/ent-coef coupling UNCHECKED.
3. **P8 + per-head KL** — does early-stop actually fire? (the §3 gate).
4. **Offline advantage reconstruction with a parity check** (must reproduce training-time mean≈0/std≈1, else it is a
   proxy that may not "kill" the experiment) → the §5 three-way differential + inverted branch.
5. **hindsight numerator/denominator/eligibility/magnitude.**
6. **Per-head legal-cardinality + realized histograms** — blueprint done (n=7). α heads: enum sizes settle it —
   AlphaTargetAction=3, AlphaSpeedAction=4, AlphaCurveAction=5, **none ≥9**, so none are in the clamp regime
   (claude-prime's "SET_ALPHA_TARGET executed with a uniformly random target" fear is refuted by the enum size).
7. Then the op-isolated ρ-sweep with a fresh hard-floor control — KL fix landed first if P8 fails.
