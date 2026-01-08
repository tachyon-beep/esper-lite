# Exploratory tasking statement: Counterfactual Analyst Head for Esper (Telemetry-only student, counterfactual teacher)

## Objective

Prototype and evaluate a **telemetry-only “Counterfactual Analyst Head”** that learns to predict counterfactual signals (teacher) from Kasmina telemetry (student), and then feeds its *own predictions* into Tamiyo’s decision policy. The goal is to preserve Esper’s foundational conceit: **Tamiyo should act like a counterfactual analyst without access to true counterfactual data at inference time**, while still using counterfactual validation as a training signal.

### Working hypothesis

A learned estimator trained on true counterfactual targets will:

1. improve policy stability and decision quality (especially around boundary effects and thrash), and
2. produce interpretable, calibratable “analyst-style” signals that can be surfaced in Sanctum/Overwatch for debugging,
   without violating the principle that Tamiyo never sees the real counterfactuals at action time.

**WEP:** likely to highly likely, depending on target choice and loss weighting.

---

## Scope and constraints

### Hard constraints (non-negotiable)

* **Policy inference must never consume true counterfactual values** (seed_contribution, escrow deltas, etc.).
* The estimator head must use **only Kasmina telemetry + existing policy-visible observations**.
* True counterfactual values may be used only as:

  * reward computation inputs (existing), and/or
  * **supervised/auxiliary learning targets** for the analyst head.

### Allowed

* Adding new model outputs and auxiliary losses.
* Adding new telemetry to log predicted vs true targets.
* Conditioning policy inputs on the **predicted** analyst outputs (student outputs only).

### Out of scope for this exploratory pass

* Changing the fundamental reward mode design (keep current modes; just add auxiliary head).
* Introducing privileged critic that directly sees oracle counterfactuals (treat as separate experiment later).

---

## Deliverables

### D0: Define “teacher targets” and data availability contract

Confirm which ground-truth quantities are available per timestep and per slot in the current pipeline and at what latency.

Candidate targets (start small; expand after baseline works):

1. **ĉ_slot:** predicted per-slot seed contribution (regression)

   * target: `seed_contribution` (or bounded/escrow form, whichever is most stable per mode)
2. **p(harm | telemetry):** probability a seed will go net-negative within the remaining horizon (binary classification)

   * target: `1{future_total_improvement < 0}` or `1{seed_contribution < threshold}` (define threshold)
3. **Δcredit_expected:** expected escrow delta for the next N steps if we WAIT (regression)

   * target: `escrow_delta` or delta in escrow_credit_target over a short lookahead window
4. **time_to_payoff:** rough payoff timescale estimate (regression or ordinal bins)

   * target: steps-to-peak contribution, steps-to-stable improvement, or first time improvement crosses threshold

**Notes:**

* Prefer targets that are **well-defined at time t** (computed from information available up to t, or clearly delayed but associated with t via logging).
* If a target is only computable later, define a clean way to attach it back to earlier steps (e.g. at episode end, label each step with eventual outcome).

---

### D1: Add Analyst Head module (telemetry-only)

Implement a new network head that takes **policy-visible observation vectors + Kasmina telemetry features** and outputs the chosen targets.

Design guidance:

* Keep it lightweight initially (2-layer MLP) to reduce confounds.
* Output shapes:

  * contribution prediction: scalar per slot (or scalar per active seed)
  * harm probability: sigmoid/logit per slot
  * optional: time-to-payoff (scalar or categorical bins)

**Interface contract:**

* `analyst_pred = AnalystHead(obs_policy_visible, telemetry_features)`
* Analyst outputs are optionally appended to actor inputs as **derived features**.

---

### D2: Add auxiliary supervised loss during PPO update

Train the analyst head using teacher targets while training PPO.

Loss terms (illustrative):

* `L_contrib = Huber(ĉ_slot, c_true)` (robust regression)
* `L_harm = BCEWithLogits(p̂_harm, harm_label)`
* `L_credit = Huber(Δĉ_credit, Δcredit_true)`
* `L_time = CE(time_bucket_pred, time_bucket_true)` or Huber for scalar

Total loss:

* `L_total = L_PPO + λ_cf * L_cf`

Where `L_cf` is a weighted sum of the selected analyst losses and λ_cf is tuned to avoid dominating PPO.

**Critical requirement:**
Analyst head training must not leak teacher targets into policy inputs (only predictions are fed forward).

---

### D3: Policy consumption experiment (ablation matrix)

Run controlled experiments to test whether feeding predicted signals improves behaviour and reduces boundary/pathology patterns.

Ablations:

* A0: baseline (no analyst head)
* A1: analyst head trained, predictions NOT fed to policy (pure diagnostics)
* A2: analyst head trained, predictions fed to policy as additional obs features
* A3: predictions fed, but with dropout/noise to prevent over-reliance and encourage robustness

Primary behavioural outcomes to watch:

* end-of-episode liquidation (spikes near epoch ~147)
* early slam / mid hold / late backfill scheduling
* alpha thrash frequency and plan reset rate
* fossilisation cadence and quality

---

### D4: Calibration and interpretability telemetry (UI-ready)

Log predicted vs true targets with enough structure to plot and debug.

Metrics to emit per update (at least):

* correlation / R² between ĉ_slot and c_true (per stage, per slot)
* AUROC / AUPRC for harm predictor (per stage, per slot)
* calibration error (ECE) for harm probabilities (optional)
* “prediction confidence” (if modelling uncertainty)
* error histograms conditioned on stage (TRAINING, BLENDING, HOLDING, etc.)

UI suggestion:

* In Sanctum/Overwatch, show side-by-side:

  * predicted contribution vs true counterfactual contribution
  * predicted harm probability vs realised harm
  * annotate decisions: “policy acted on predicted harm spike”

---

## Success criteria

### Technical success

* Analyst loss converges to non-trivial performance:

  * contribution predictor achieves meaningful correlation with true signal (target TBD by baseline noise)
  * harm classifier beats simple heuristics (e.g., stage-only baseline)
* PPO remains stable (no increased grad norm explosions, no reward hacking regressions).

### Behavioural success

Evidence that policy decisions become less “boundary-timed” and more state-timed:

* reduced pruning/fossilisation spikes near the episode end
* reduced alpha plan resets per epoch (more “set and observe”)
* improved seed selection quality (higher contribution of kept/fossilised seeds)
* improved final score or at least improved stability at equal compute

### Conceptual success (Esper philosophy)

* At inference time, Tamiyo can make decisions using **only telemetry-derived predictions**, not oracle counterfactuals.
* The predicted signals are interpretable enough to reason about and debug (visible in UI).

---

## Risks and mitigations

**Risk:** analyst head becomes a crutch and policy overfits to its noise
Mitigation:

* dropout/noise on analyst outputs to policy
* limit bandwidth (coarse bins rather than high-res scalars)
* stage-conditioned losses and calibration metrics

**Risk:** training instability due to aux loss dominating PPO
Mitigation:

* small λ_cf, ramp-up schedule, gradient norm clipping per loss component

**Risk:** target definition leakage / future information
Mitigation:

* strictly define target timestamping
* avoid targets that require future information unless explicitly labelled post-hoc and treated carefully

---

## Questions to answer

1. Which teacher targets are most stable in escrow vs shaped modes: raw seed_contribution, bounded attribution, or escrow deltas?
2. Should analyst head share any trunk with actor/critic, or be fully separate to avoid representational leakage?
3. Best practice for joint optimisation: single optimiser vs separate optimisers (and LR ratios)?
4. How to structure per-slot outputs (fixed K slots vs only active slots) while keeping tensor shapes compile-friendly?
5. Evaluation plan: what’s the best “dumb baseline” heuristic to compare harm prediction against?

============================

## Why aleatoric uncertainty is the “missing organ”

Without uncertainty, the Probe is forced into a dumb binary:

* either it confidently guesses (even when it shouldn’t), or
* you bolt on heuristics (“audit every N steps” / “audit before fossilise”)

Neither is “active perception”. Active perception needs a self-report like:

> “I can predict this contribution reliably” vs “this region of state-space is noisy / unfamiliar.”

Aleatoric uncertainty (data noise / irreducible variability) is especially relevant in Esper because the thing you’re predicting is inherently noisy:

* counterfactual deltas depend on batch composition
* host drift changes the meaning of the same seed
* alpha plans can make short-term deltas misleading
* RL stochasticity makes “same state” not actually the same

So σ isn’t just a convenience. It’s the mechanism that turns `AUDIT` into an *agentic sensor allocation policy*.

**WEP:** *Highly likely* that adding σ is necessary to make audit targeting efficient and non-calendar-like.

---

## Two kinds of uncertainty (and which one Gemini proposed)

There are two uncertainty types people mix up:

1. **Aleatoric uncertainty**: “the world is noisy here”

   * Even with infinite data, you can’t pin it down perfectly.
   * Fits your case: counterfactuals fluctuate due to stochastic training and sampling.

2. **Epistemic uncertainty**: “the model is ignorant here”

   * Reducible with more data.
   * Important for novelty detection.

Gemini suggested aleatoric (μ, σ with Gaussian NLL). That’s great as a first pass, *and* it often functions as a practical “I don’t know” signal in deep nets even though it’s not a pure epistemic measure.

If you later want “true novelty” detection, you can add epistemic approximations (ensembles, MC dropout). But σ-first is the right minimal step.

---

## Recommended implementation detail: predict μ and log σ²

Numerically, you do not want the network outputting σ directly without constraints.

Use:

* output `mu` (float)
* output `log_var` (float, unconstrained)
* define `var = exp(log_var)` or `softplus(log_var)` for stability

Then train with Gaussian NLL.

In PyTorch terms:

* either use `torch.nn.GaussianNLLLoss` with `var`
* or implement the NLL yourself (it’s tiny and gives you exact control)

Key stability trick: clamp the variance.

Example ranges that usually behave:

* `log_var` clamped to `[-10, 4]` (var ≈ [4.5e-5, 54.6])
* adjust once you see telemetry

---

## What σ should mean in Esper

You want σ to reflect “expected oracle measurement error”, not just “model is scared”.

So define your training target carefully:

### Target choice matters

For contribution prediction, prefer a target that has repeatable semantics:

* `Δloss` on a fixed audit batch (or small cached batch pool)
* not raw validation accuracy delta (too quantised)

Even better: define “oracle contribution” for labels as an *average across a few micro-batches* during audit:

* Audit runs K mini-evals and returns mean + sample variance
* That gives you:

  * `y = mean(Δloss)`
  * `s² = var(Δloss)` (empirical noise)
* Now σ has a “real world” reference

This makes σ meaningful, and you can test calibration:

* do predicted σ match empirical variance?

---

## How σ drives `AUDIT` without becoming another hack

Once you have σ, you can use it in three clean ways:

### 1) Audit targeting (Active Perception proper)

Select audit target by expected value of information:

A simple heuristic proxy is:

* audit priority ∝ σ * stake

Where “stake” can be:

* probability of taking an irreversible action soon
* predicted harm probability
* or “occupied slots are expensive”

Example intuition:

* High σ and high p(harm): audit now
* Low σ and low p(harm): don’t waste compute

### 2) Conservative gating for irreversible actions

Soft rule:

* discourage `FOSSILISE`/`PRUNE` when σ is high
* but don’t hard forbid it unless in training-wheels mode

This reduces superstition without making the system brittle.

### 3) UI / debugging

Surfacing σ alongside μ and oracle measurements is *chef’s kiss*:

* “it made a wrong call but σ was huge” → the agent had reason to be uncertain
* “it made a wrong call with tiny σ” → probe is miscalibrated (real bug)

---

## The key risk: σ-inflation (“I don’t know” spam)

If the probe can get away with always outputting huge σ, it can dodge loss.

The NLL naturally penalises that because of the `log σ²` term, but it can still drift if your label noise is large.

Mitigations:

* clamp σ (hard ceiling)
* add mild regulariser pulling σ toward a baseline
* optionally supervise σ toward empirical audit variance if you compute it

If you do K-sample audits, you can train σ to match observed s² (or at least correlate).

---

## Concrete patch to your proposal/spec (drop-in)

Here’s a clean “addition block” you can paste into the Counterfactual Probe proposal.

### Add to “What the probe predicts”

**Add output: aleatoric uncertainty**

* The probe outputs `(μ, σ)` for predicted contribution (and optionally for harm logits too).
* σ represents predicted observation noise / expected oracle error.
* σ is used as the agent’s “I don’t know” signal to drive `OP_AUDIT` selection and conservative irreversible decisions.

### Add to “Training signal”

**Gaussian NLL loss**

* Train contribution predictions using Gaussian negative log likelihood:

  * `L = 0.5 * ( (y-μ)² / σ² + log σ² )`
* Implementation uses `log_var` output and clamped variance for numerical stability.
* This encourages the probe to:

  * predict low σ when it is reliably correct
  * predict high σ in noisy/novel regimes (“AUDIT ME!”)

### Add to “Audit synergy”

**Audit as label + calibration**

* When an audit runs multiple micro-evals (K batches), record:

  * oracle mean contribution `y`
  * empirical variance `s²`
* Use `s²` to evaluate calibration and optionally supervise σ.
