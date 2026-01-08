# Proposal for Investigation: Counterfactual Probe in Esper

**Subtitle:** *From Counterfactual Oracle to Counterfactual Probe (Telemetry‑only belief at scale)*
**Date:** 8 January 2026
**Status:** Draft / Investigation proposal + experiment plan
**Working title:** *Counterfactual Probe (Learned Contribution Inference) for 50–100+ Seeds*

---

## 0. Thesis

Esper’s “counterfactual oracle” (true ablations / Shapley-ish measurement) is *gold*, but it’s also *expensive*. At 50–100+ seeds, always-on oracle truth becomes the bottleneck: either compute explodes or we stop measuring and fly blind.

The **Counterfactual Probe** is a learned, telemetry-only estimator that predicts “what the oracle would say” well enough to guide decisions most of the time.

Key idea:

> **Oracle = expensive truth** (sparse, auditable, used as labels).
> **Probe = cheap belief** (dense, always available, used for control).

This preserves Esper’s core conceit: **Tamiyo learns to behave like a counterfactual analyst from Kasmina telemetry**, not by being handed the exam paper.

**WEP:** *Highly likely* to be required if you want Emrakul-scale complexity without turning your training loop into an ablation benchmark harness.

---

## 1. Goals and non-goals

### Goals

1. **Scale**: enable 50–100+ seeds with *sublinear* counterfactual compute growth.
2. **Preserve the philosophy**: policy input remains telemetry-derived; probe outputs are allowed because they’re learned from telemetry.
3. **Improve stability**: reduce “superstition” and boundary-timed behaviour by giving Tamiyo a consistent belief state (with uncertainty).
4. **Remain debuggable**: make probe-vs-oracle disagreements visible in Sanctum/Overwatch.

### Non-goals (initial investigation)

* Replacing the oracle entirely on day one.
* Using probe outputs as the primary reward signal immediately (high reward-hacking risk).
* Perfect Shapley fidelity; the goal is *decision-adequate* counterfactual judgement, not axiomatic attribution purity.

---

## 2. Problem statement: the Truth Bandwidth Wall

With many seeds, the oracle is expensive because it requires:

* extra forward passes (full vs ablated),
* sometimes multiple seeds per step,
* and interaction effects make it tempting to measure everything (which is infeasible).

When you stop measuring, the agent either:

* becomes overly conservative (“do nothing attractor”), or
* starts gaming whatever dense proxies remain.

The probe aims to restore **dense decision support** without dense expensive truth.

---

## 3. Proposed mechanism: a learned “Probe” that predicts counterfactual outcomes

### 3.1 What is the probe?

A neural module that consumes **policy-legal features** (Kasmina + existing obs) and outputs predicted counterfactual metrics per slot/seed, plus uncertainty.

Think of it as:
**“What would the oracle likely conclude if we audited this slot right now?”**

### 3.2 What the probe predicts (initial target set)

Start with a small, stable set and expand later.

**Primary outputs (per slot):**

1. **ĉ_contrib**: predicted counterfactual contribution

   * recommended target: `Δloss` style signal (smoother than acc)
2. **p_harm**: probability the seed will be net-negative if left in place (or if fossilised)

   * helps avoid late liquidation / panic pruning
3. **uncertainty / staleness proxy**: how much to trust the probe’s estimate

   * can be explicit predictive variance or a calibrated confidence score

**Optional later outputs:**

* **time_to_payoff**: expected steps to “proven stable improvement”
* **Δcredit_expected** (escrow mode): predicted escrow delta trend if we WAIT
* **α-sensitivity**: expected marginal effect of alpha plan changes (more ambitious; postpone)

**WEP:** *Very likely* that “ĉ_contrib + p_harm + uncertainty” is the sweet spot for first investigations.

---

## 4. Training signal: oracle truth becomes labels (not observations)

### 4.1 Teacher labels (oracle)

Oracle measurements can come from:

* existing always-on counterfactual pipeline (in small-seed regimes), and/or
* Active Perception `AUDIT` events (sparse labels at scale), and/or
* periodic scheduled audits (e.g., audit K random slots per episode as exploration).

These produce training pairs:
`(telemetry_features_t, oracle_counterfactual_t)`.

### 4.2 Student training (probe)

The probe is trained via supervised / auxiliary losses. It is never given oracle values as inputs at action time.

Recommended losses:

* `L_contrib = Huber(ĉ_contrib, contrib_true)`
  (Huber handles noise and occasional spikes)
* `L_harm = BCEWithLogits(p_harm, harm_label)`
* `L_uncert = NLL` style loss if modelling variance (optional)

**Important design guardrail:**
The probe must not be trained on “future information” unless you deliberately label it post-hoc and accept the semantics. For the first pass, keep targets computable at time *t* (or at audit time).

---

## 5. Architecture options (ordered by increasing complexity)

### Option A: Per-slot independent probe (fastest to ship)

* Each slot gets a small MLP
* Input: slot telemetry + global context summary
* Output: ĉ_contrib, p_harm, confidence

Pros: simplest, compile-friendly, stable.
Cons: misses cross-slot interaction effects.

### Option B: Set-based probe with cross-slot attention (likely needed for 50+)

* Treat slots as a set; run a small attention block across slot embeddings
* Output per slot

Pros: models synergy/competition; still far cheaper than oracle.
Cons: more complexity; needs care with normalisation and stability.

### Option C: Interaction-aware probe (explicit pairwise features)

* Predict per-slot marginal plus a low-rank interaction term
* Useful if synergy matters a lot (you already have `interaction_sum` hints)

Pros: interpretable; captures scaffold effects.
Cons: engineering overhead.

**Recommendation for investigation:**
Start with **Option A**, validate value, then move to **Option B** once you see clear interaction-driven probe errors.

**WEP:** *Likely* Option A will be “good enough” at 3–10 seeds; *likely* you’ll want Option B for 50+.

---

## 6. How the probe is used (and what it must never do)

### 6.1 Allowed uses (safe)

1. **Observation feature for the actor**: append probe outputs to Tamiyo obs.

   * Because outputs are telemetry-derived, they fit Esper’s philosophy.
2. **Decision gating**: use confidence to discourage irreversible actions under high uncertainty.
3. **Audit targeting**: choose which slot to `AUDIT` based on probe uncertainty (“active learning”).
4. **UI / interpretability**: compare predicted vs true oracle values.

### 6.2 Not allowed in Phase 1 (high-risk)

* Using probe outputs directly as the primary reward or shaping term.

Reason: if the agent can manipulate the probe (directly or indirectly), it will. Reward hacking becomes “model hacking”. Even if you freeze probe weights during rollouts, the agent can learn states that fool it.

**WEP:** *Very likely* that probe-as-reward creates a new class of adversarial behaviours unless heavily constrained.

### 6.3 A cautious later extension (Phase 3+)

A hybrid reward that uses oracle when available and probe as a *very weak* proxy when not, with strong caps and confidence discount. Treat this as a separate research question.

---

## 7. Relationship to Active Perception (AUDIT)

The probe and `AUDIT` are complementary:

* **AUDIT**: “pay compute to obtain ground truth for one slot now”
* **Probe**: “use telemetry to maintain beliefs for all slots all the time”

Together they form a tidy loop:

1. Probe makes predictions + uncertainty for all slots
2. Policy chooses actions using predictions
3. Policy sometimes calls `AUDIT` for high-uncertainty/high-stakes slots
4. Audit produces labels; probe trains online; uncertainty improves over time

This is basically *active learning inside RL*.

---

## 8. Implementation plan (phased)

### Phase 1: Plumbing + offline probe prototype (minimum viable science)

**Deliverables**

1. **Data logging**: record telemetry features + oracle labels when oracle is computed

   * store per slot, per step, with seed stage metadata
2. **Standalone probe model**: train offline, validate metrics (correlation, calibration)
3. **UI telemetry**: show probe vs oracle for inspected env/slot

**Success bar**: probe is non-trivial and stable offline.

---

### Phase 2: Online auxiliary training (probe learns during PPO)

**Deliverables**

1. Add probe head to the model as an auxiliary module
2. Add `L_probe` into PPO update (weighted by λ_probe)
3. Train probe only on steps where oracle labels exist (audited or scheduled)
4. Emit calibration and drift telemetry

**Success bar**: PPO stays stable; probe improves over time; behaviour starts shifting (less superstition).

---

### Phase 3: Policy consumption (probe becomes part of Tamiyo’s “mind”)

**Deliverables**

1. Append probe outputs to Tamiyo observation vector
2. Add conservative decision logic:

   * discourage FOSSILISE/PRUNE when probe uncertainty high (soft penalties or thresholds)
3. Use probe uncertainty to drive audit selection

**Success bar**: behaviour improves without reintroducing boundary-timed scripts.

---

### Phase 4: Scale-up protocol (reduce oracle frequency)

**Deliverables**

1. Replace “oracle every step” with:

   * sparse audits + periodic scheduled audits
2. Track probe error as a function of audit rate
3. Find minimal audit budget that maintains decision quality

**Success bar**: 50+ seeds feasible; compute cost manageable.

---

## 9. Experiment plan

### 9.1 Baselines

* **Oracle-every-step** (small seed counts only): upper bound for decision support
* **No oracle/no probe**: telemetry-only
* **Probe trained offline, not used by policy** (diagnostic only)
* **Probe used by policy** (full intended usage)

### 9.2 Ablations

1. Probe architecture: per-slot MLP vs set-attention
2. Targets: `Δloss` vs `Δacc`
3. Labels: audit-only vs scheduled random labels vs mixed
4. Use: probe-only vs probe+uncertainty gating vs probe+audit targeting
5. Probe output noise/dropout to prevent over-reliance

### 9.3 Key metrics

**Probe quality**

* Pearson/Spearman correlation: ĉ_contrib vs contrib_true
* calibration: Brier score / ECE for harm probabilities
* selective prediction: error vs confidence (does uncertainty actually mean anything?)
* stage-conditioned error: TRAINING vs BLENDING vs HOLDING (important)

**Policy behaviour**

* rate of irreversible actions taken under high uncertainty
* late-episode liquidation spikes (prune/fossilise vs time)
* alpha plan reset/thrash frequency
* seed utilisation patterns (slam/hold/backfill dynamics)
* final task outcome (acc/loss) + stability (grad norm distribution)

**Compute**

* oracle calls per episode
* wall time per episode
* throughput vs number of seeds

---

## 10. Telemetry + UI plan (make it fun and diagnosable)

Add a “Probe vs Oracle” section per env/slot:

Per slot:

* `ĉ_contrib`, `contrib_true` (when available)
* error (abs/relative), EMA error
* confidence/uncertainty
* `p_harm` and realised harm label (episode-end)
* audit history: last audit step, number of audits

Global:

* coverage curve: “top 20% confidence contains X% of accurate predictions”
* audit allocation: are audits going to high-uncertainty / high-stakes states?

This directly supports the new debugging class:

* “Why did it fossilise without auditing?” → probe confidence high + low predicted harm
* “Why did it keep one seed in holding?” → probe uncertainty high, option value retained
* “Why did it prune at 147?” → predicted harm spike + low payoff time remaining

---

## 11. Risks and mitigations

### Risk A: Probe becomes a new reward-hack surface

Even if you never use it as reward, the policy could learn to reach states where probe is systematically optimistic.

**Mitigations**

* keep probe as observation + auxiliary training only (Phase 1–3)
* include confidence estimates; penalise high-confidence wrong predictions (supervised)
* periodic random audits to prevent “dark corners” where probe is never checked

### Risk B: Probe overfits to audit batch

If audit uses a fixed batch, the probe may learn to predict that batch rather than general contribution.

**Mitigations**

* rotating audit cache or small batch pool
* log audit batch IDs for analysis
* measure generalisation with a separate validation set of oracle labels

### Risk C: Interaction effects dominate

If contributions are highly non-local, per-slot MLP probe may be misled.

**Mitigations**

* promote to set-attention probe
* add global context and stage embeddings
* explicitly model synergy/competition later if needed

### Risk D: Training instability due to auxiliary loss

Probe loss can fight PPO if weighted badly.

**Mitigations**

* small λ_probe with ramp-up schedule
* separate optimiser or LR for probe head
* gradient norm clipping per-loss component

---

## 12. Success criteria

### Scaling success

* oracle compute becomes sparse and controllable (audit budget), while decision quality remains acceptable
* 50–100+ seeds training throughput remains practical

### Behavioural success

* fewer superstitious irreversible actions
* more “investigate → decide” loops (audit or probe-confidence gating)
* reduced boundary minmaxing and late liquidation spikes

### Conceptual success (Esper thesis)

* Tamiyo behaves like a counterfactual analyst using telemetry-derived beliefs
* oracle is a measurement instrument, not a sensory firehose
* probe outputs are interpretable and improve over time with sparse truth

**WEP:** *Likely* achievable if you keep probe out of the reward path early and treat uncertainty seriously.

---

## 13. Open questions to resolve early

1. What is the primary oracle label for the probe: `Δloss`, bounded attribution, escrow delta, or something else?
2. Do we label harm as “eventual negative total improvement” or “negative audited contribution at time t”?
3. How much interaction modelling is needed at 50 seeds (MLP vs attention)?
4. What audit schedule/budget gives the best label efficiency?
5. Do we allow probe outputs into the actor immediately, or do we run a diagnostic-only period first to verify calibration?

---

### Closing note

Conceptually, this turns Esper from “a system that *has* counterfactuals” into a system that can **survive under uncertainty**, allocate attention, and learn an internal model of “what helps” — which is exactly the kind of organism you need once the architecture stops being toy-sized.
