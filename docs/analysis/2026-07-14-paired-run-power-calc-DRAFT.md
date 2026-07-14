# Paired-run power calculation — P2 + GATE-DOM + GATE-INFL (DRAFT)

**Status: DRAFT — NOT frozen. Main-session review + owner ratification at freeze required before any number here enters the pre-registration.** Freeze blocker #2 for the "make permanence visible" pre-registration (PDR-0081 §Group A/B1, PDR-0082 §2). Tracker: `esper-lite-2852cf851c`. Design-only; no training code touched; no GPU authorized. drl-expert deliverable, 2026-07-14.

**Scope.** One parametric power framework covering the three metrics that share the **per-run paired unit** (PDR-0081 B1 pseudoreplication fix): **P2** (quote-sensitive commitment slope), **GATE-DOM** (high-quality suppression), **GATE-INFL** (low-quality inflation). Because there is **no obs-v4/K=4 pilot**, the between-run SD of the per-run estimates (σ_d) is **unknown**; this document does **not** invent one. It delivers (a) a parametric power surface over *assumed* σ_d; (b) a blinded post-screen variance-reassessment rule; (c) a frozen max-n proposal; (d) an explicit statement of what the calc structurally cannot tell us.

---

## 0. Headline numbers (one-sided α = 0.05, paired one-sample t)

- Power depends **only** on `(n, d_z)` where `d_z = (Δ − τ)/σ_d` is the standardized paired effect. The missing pilot costs us the **scale** of σ_d, not the **shape** of the curve.
- **n = 10 reaches ≥ 80 % power iff `d_z ≥ 0.853`, i.e. iff `σ_d ≤ 1.17·Δ`.** For 90 %: `d_z ≥ 1.005`, i.e. `σ_d ≤ 0.995·Δ`.
- **n = 5 reaches ≥ 80 % power iff `d_z ≥ 1.359`, i.e. `σ_d ≤ 0.736·Δ`** — the screen tier is coarse (as the pre-reg already says for P1).
- **Bounding resolution (the expected-null role):** a null point estimate yields a one-sided 95 % upper bound on the effect of **0.953·σ_d at n = 5** and **0.580·σ_d at n = 10**. This — not a detection-power number — is what an Exp-1 arm-B P2 null buys.
- **A raw-units σ_d range is unavailable pre-pilot; the ratio `σ_d ≤ 1.17·Δ` IS the answer.** §5 gives an r9-anchored procedure to freeze an *absolute* material effect Δ so the owner can read the ceiling in absolute terms.

---

## 1. The estimand and the test

### 1.1 The per-run paired unit (why this is the whole design)
Treatment (arm) is assigned at the **training-run/seed** level, not per decision. The old E5 ("per-decision N is thousands") was pseudoreplication — decisions within a run are *not* treatment replicates; they only **sharpen** that run's single estimate. So for each metric `m` and each seed `s`:

- run `(s, arm)` → **one** per-run estimate `θ̂_{m,arm,s}` (a regression **slope** for P2 and the GATE-DOM slope; a **tercile rate/contrast** for GATE-INFL and the GATE-DOM top-tercile margin), computed over that run's eligible-HOLDING decisions with the frozen stratification (num_valid, age, occupancy, round, horizon, mask, quote-trend);
- paired difference on matched seeds: `d_{m,s} = θ̂_{m,B,s} − θ̂_{m,A,s}` (or B→C, A→C for Exp-2 contrasts);
- population model: `d_{m,s} ~ iid(mean = Δ_m, SD = σ_{d,m})`, `s = 1..n`. **σ_{d,m} is the unknown**; it folds in between-seed heterogeneity, each run's finite-decision estimation error, and — crucially — whatever the seed-pairing removes (see §7).

The sample size for **every** test in this framework is `n = number of seeds` (5 screen, 10 claim), **not** the decision count.

### 1.2 The test statistic (matches the pre-reg's one-sided-95%-LCB style)
One-sample (paired) t on `{d_{m,s}}`:

```
t = d̄_m / (ŝ_{d,m} / √n),      df = n − 1
```

Reject H0 (`Δ_m ≤ τ_m`) one-sided at α = 0.05 when `t > t_{0.95, n−1}`, **equivalently** when the one-sided 95 % **lower** confidence bound of `Δ_m` exceeds `τ_m`. This is exactly the E4 convention already frozen for P1 ("test = one-sided 95 % LCB of the paired diff > −0.3 pp"). Verified critical values used throughout: `t_{0.95,4} = 2.13185`, `t_{0.95,9} = 1.83311` (appendix self-check).

### 1.3 The three metrics: one framework, three σ_d, three directions
| Metric | Per-run estimate `θ̂` | H1 direction (one-sided) | τ | Fired ⇒ | Data used |
|---|---|---|---|---|---|
| **P2** | slope of `P(request)` on `q_decision`, full eligible-HOLDING | Exp-2 arm C: `Δ = slope_C − slope_B > 0` (coupled bet) | 0 | COUPLED-MECHANISM signal | all deciles (lowest σ_d) |
| **GATE-DOM** | slope of pre-floor `p_FOSS` on `q_decision` (B vs A) **+** top-tercile `z_FOSS − z_SET_ALPHA` margin | suppression: `slope_B < slope_A` | 0 | NOT SETTLEMENT-PROTOCOL-SAFE | slope: all; margin: top tercile |
| **GATE-INFL** | bottom-tercile pre-floor FOSS preference / request rate (B vs A) | inflation: `rate_B > rate_A` | 0 | pricing pathology (commit-everything) | bottom tercile (highest σ_d) |

**Gate-specific differences to carry through the calc:**
1. **Direction.** DOM is one-sided toward *suppression*; INFL one-sided toward *inflation*; P2-in-C one-sided toward *positive* slope. The power *math* is identical (sign of `Δ` flips); the *interpretation* differs.
2. **Estimator variance (the load-bearing gate difference).** P2 and the GATE-DOM slope use **all** eligible decisions → lowest per-run estimation error. GATE-INFL (bottom tercile) and the GATE-DOM top-tercile margin use **~⅓** the decisions and contrast extremes → inherently larger σ_d. If per-run estimation error dominates σ_d, a single-tercile estimate inflates SD by ≈√3 ≈ 1.73× and a top−bottom contrast by ≈√6 ≈ 2.45× versus a full-range slope; if between-run heterogeneity dominates, inflation is small. Because **n scales with σ_d²**, a 1.7–2.5× σ_d inflation demands **≈3–6× the runs** at the same `(Δ, power)`. **Consequence: each metric's σ_d must be reassessed separately (§5), and max-n must be sized off the noisiest load-bearing metric — almost certainly a tercile gate, not P2.**
3. **One-sided-in-B interpretation asymmetry (PDR-0082 §2).** In arm B the op head updates only on the |H|≥2 minority, so realized rates are floor-clamped and the gates score **pre-floor** preferences. A **fired** gate is a strong alarm; a **null** gate is weak comfort, **not** safety — it is only comfort if the design had power to catch the harm. This is why the gates' power (their MDE) is the load-bearing number for Exp-1, and why we recommend a *stricter* target power for them than for P2 (§4, §6).

---

## 2. The power math

Under H1 with true effect `Δ_m` and SD `σ_{d,m}`, the statistic follows a **noncentral t** with `df = n−1` and noncentrality

```
δ = √n · (Δ_m − τ_m) / σ_{d,m} = √n · d_z,        d_z = (Δ_m − τ_m)/σ_{d,m}   (Cohen's d_z)
Power(n, d_z) = P(T' > t_{0.95, n−1}) = 1 − F_nct(t_{0.95,n−1}; df = n−1, ncp = √n·d_z)
```

Computed exactly by numerical integration of the noncentral-t; cross-checked against 3×10⁵-trial Monte-Carlo (agreement to ≤ 0.001, appendix). **Power is a function of `(n, d_z)` alone** — the scale-free core.

The **minimum detectable effect** at power `1−β`:

```
MDE = d_z*(n, α, β) · σ_d,      d_z* = the standardized MDE (Table 2)
```

At exactly 50 % power the MDE collapses to the one-sided 95 % confidence half-width, `d_z*(50%) = t_{0.95,n−1}/√n` — this is the **bounding resolution** a null point estimate certifies (used for the expected-null P2, §4).

---

## 3. (a) The parametric power surface

### Table 1 — standardized power `Power(n, d_z)`, one-sided α = 0.05
| `d_z = (Δ−τ)/σ_d` | n=5 | n=8 | n=10 | n=12 | n=16 |
|---:|---:|---:|---:|---:|---:|
| 0.25 | 0.120 | 0.158 | 0.181 | 0.203 | 0.245 |
| 0.50 | 0.239 | 0.357 | 0.427 | 0.492 | 0.604 |
| 0.75 | 0.401 | 0.605 | 0.707 | 0.785 | 0.888 |
| **1.00** | **0.580** | 0.815 | **0.898** | 0.945 | 0.985 |
| 1.25 | 0.741 | 0.936 | 0.977 | 0.992 | 0.999 |
| 1.50 | 0.862 | 0.984 | 0.997 | 0.999 | 1.000 |
| 1.75 | 0.936 | 0.997 | 1.000 | 1.000 | 1.000 |
| 2.00 | 0.975 | 1.000 | 1.000 | 1.000 | 1.000 |

### Table 2 — standardized MDE multiplier `d_z* = MDE/σ_d` at target power
| n | df | t_{0.95} | @50 % (=95 % UCB coeff = null bound) | @80 % | @90 % |
|---:|---:|---:|---:|---:|---:|
| 5 | 4 | 2.1318 | **0.888** (bound = 0.953·σ_d)† | **1.359** | 1.610 |
| 8 | 7 | 1.8946 | 0.645 | 0.978 | 1.153 |
| **10** | 9 | 1.8331 | **0.563** (bound = 0.580·σ_d)† | **0.853** | **1.005** |
| 12 | 11 | 1.7959 | 0.506 | 0.766 | 0.903 |
| 16 | 15 | 1.7531 | 0.431 | 0.652 | 0.767 |

† The "@50 %" column is the standardized effect at the α-boundary; the **bounding resolution** (one-sided 95 % UCB half-width at a null point estimate) is `t_{0.95,n−1}/√n · σ_d` = **0.953·σ_d (n=5)**, **0.580·σ_d (n=10)**. (These differ slightly from the @50 % `d_z*` because the latter is defined against `Δ−τ` at the critical point; both are reported for auditability.)

### Table 3 — the σ_d-parameterized surface (what "power vs assumed SD" looks like)
Cells = `Power(n=5) / Power(n=10)`, in **common notional slope units u** (arbitrary — see the caveat). Read along constant `Δ/σ_d` diagonals; the numbers are meaningful as **ratios**, not as absolute slopes.

| σ_d ↓ \ Δ → | 0.05 | 0.10 | 0.15 | 0.20 | 0.30 |
|---:|---:|---:|---:|---:|---:|
| 0.05 | 0.58/0.90 | 0.97/1.00 | 1.00/1.00 | 1.00/1.00 | 1.00/1.00 |
| 0.10 | 0.24/0.43 | 0.58/0.90 | 0.86/1.00 | 0.97/1.00 | 1.00/1.00 |
| 0.15 | 0.15/0.25 | 0.34/0.62 | 0.58/0.90 | 0.79/0.99 | 0.97/1.00 |
| 0.20 | 0.12/0.18 | 0.24/0.43 | 0.40/0.71 | 0.58/0.90 | 0.86/1.00 |
| 0.30 | 0.09/0.12 | 0.15/0.25 | 0.24/0.43 | 0.34/0.62 | 0.58/0.90 |
| 0.40 | 0.08/0.10 | 0.12/0.18 | 0.17/0.29 | 0.24/0.43 | 0.40/0.71 |

**Caveat (instrument validity):** the slope units `u` are **ungrounded** — no pilot fixes the scale of either `Δ` or the `q_decision` axis. This grid is a *ratio device*; the only defensible reading is along the diagonal `Δ/σ_d = const`. §5 supplies the r9 procedure that would attach a real number to the `Δ` column.

### Table 4 — the σ_d ceiling for ≥ 80 % power at a fixed target effect
(80 %-MDE multipliers: n=5 needs `Δ ≥ 1.359·σ_d`; n=10 needs `Δ ≥ 0.853·σ_d`.)

| target `Δ` (u) | σ_d ≤ for 80 % @ n=5 | σ_d ≤ for 80 % @ n=10 |
|---:|---:|---:|
| 0.05 | 0.037 | 0.059 |
| 0.10 | 0.074 | 0.117 |
| 0.15 | 0.110 | 0.176 |
| 0.20 | 0.147 | 0.235 |
| 0.30 | 0.221 | 0.352 |

**Direct answer to "the assumed-SD range over which n=10 achieves 80 % power at plausible effects":** n=10 is ≥ 80 %-powered **whenever `σ_d ≤ 1.17·Δ`** (and ≥ 90 % whenever `σ_d ≤ 0.995·Δ`). Equivalently, n=10 detects (80 %) any true effect that is **at least 0.85 between-run SDs**; n=5 only reaches down to **1.36 SDs**.

### Table 5 — READ-3's "2–5× MDE" framing
If the design is built so the plausible/target effect sits at a multiple of the 50 %-power threshold (the null-bound resolution):

| n | 1× (bound) | 2× | 3× | 5× |
|---:|---|---|---|---|
| 5 | 0.953·σ_d (p≈0.55) | 1.907·σ_d (p≈0.96) | 2.860·σ_d (p≈1.00) | 4.767·σ_d (p≈1.00) |
| 10 | 0.580·σ_d (p≈0.52) | 1.159·σ_d (p≈0.96) | 1.739·σ_d (p≈1.00) | 2.898·σ_d (p≈1.00) |

READ-3's convention (target effect ≥ 2–5× the MDE) maps to: **a "comfortable" n=10 design wants the true effect ≥ 1.16·σ_d (2×) for ~96 % power.** At n=5 the same comfort requires ≥ 1.91·σ_d — a large effect relative to between-run spread.

### 3.1 Calibrating intuition for plausible between-run spread (labeled analogies — NOT σ_d estimates)
These are handed to give a *feel* for how large between-run dispersion runs in this system. **None is the P2/gate slope σ_d; each is a different metric.** Using any of them as a variance estimate would violate the pre-registration.

- **Stage-2 OFF-wave `ev` levels** across 5 runs: `0.132 / 0.114 / 0.194 / 0.045 / 0.282` → between-run SD ≈ **0.089** on mean ≈ 0.153 (CV ≈ **58 %**). *Analogy only* (a value-metric level, not a slope). Takeaway: between-run heterogeneity here is **not small** — do not assume σ_d will be a tiny fraction of Δ. The within-run `ev` IQRs (0.257–0.468) show within-run dispersion is itself large, which is the raw material of per-run estimation error.
- **P1 terminal-accuracy paired-diff noise floor ≈ 0.77 pp at n=5** (pre-reg §1.7). *Analogy only* (a different primary, accuracy not slope) — but it is a genuine *per-run paired-difference* SD scale and it is why §1.7 already calls the n=5 tier coarse and defers P1 superiority to n=10. Expect the P2/gate screen to be similarly coarse.

---

## 4. Role per metric — detection vs bounding, and the target power

**This is where the "expected-null primary" discipline lives.**

- **P2, Exp-2 arm C (genuine detection).** "Did the coupled bet fire": test `slope_C − slope_B > 0`, one-sided. Target power **80 %**. Table 1/2 apply directly.
- **P2, Exp-1 arm B (EXPECTED NULL — bounding, not detection).** Under the hard floor, `P(request) ≈ 0.15` regardless of quote, so the slope difference is *predicted* to be ≈ 0 (the degeneracy signature; pre-reg §1.6: "a non-null B slope is not a failure"). A classical "power to reject H0" number is **the wrong object here** — the null is the design's prediction, not a hypothesis to be "passed." The informative quantity is the **bounding resolution**: a null screen bounds the settlement's effect on quote-sensitivity to `< 0.953·σ_d (n=5)` / `< 0.580·σ_d (n=10)` at one-sided 95 %. A null with adequate resolution ⇒ "we can rule out a slope shift larger than X"; a null with poor resolution (σ_d large) ⇒ **uninformative** — indistinguishable from "underpowered." **Do not report Exp-1 P2 as a detection-power success.**
- **GATE-DOM / GATE-INFL (harm detection — the Exp-1 safety verdict).** Because a **null gate is only weak comfort**, the load-bearing question is: *what magnitude of harm could a null be hiding?* — i.e. the gate MDE. A false "safe" here **ships a bad protocol** (SETTLEMENT-PROTOCOL-SAFE flag-land). Recommend **target power 90 %** for the gates (stricter than P2's 80 %): the asymmetric cost of a missed alarm justifies the tighter MDE. Combined with the tercile-variance inflation (§1.3.2), **the gates — not P2 — will set the required n.**

Multiplicity note: three one-sided gates each at α = 0.05 inflate the family-wise *false-alarm* rate, but gate firing is **conservative-toward-owner** (a false alarm costs a redesign, not a bad ship), so we deliberately **do not** α-correct the gates (correction would *reduce* harm sensitivity). Symmetrically, multiplicity does not inflate the *false-safety* rate. State this in the frozen doc rather than importing a Bonferroni that would blunt the safety instrument.

---

## 5. (b) The blinded post-screen variance-reassessment rule

**The problem this solves (PDR-0081 B1):** there is no obs-v4/K=4 pilot, so n=10 **cannot be pre-certified**. After the n=5 screen we measure σ_d *blind* (arm labels concealed) and convert the frozen relative MDE into an absolute one to decide whether n=10 is adequately powered — without ever inspecting the treatment contrast.

### 5.1 Closing the circularity: freeze the ABSOLUTE material effect from r9 (scale knowable now; only variance needs the screen)
A reassessment that compares a *relative* MDE (`0.853·σ_d`) to a *relative* target (`k·σ_d`) is **vacuous** — the ratio is fixed regardless of the measured σ_d. The target must be frozen in **absolute slope units**. The key asymmetry: **the `q_decision` scale is analyst-computable from r9 *now*; only the variance needs the K=4 screen.**

**Pre-freeze r9 calibration read (r9-only, no-peek-safe — does NOT touch the new arms; add to the freeze-order read queue).** Over r9 eligible-HOLDING decisions (FOSSILIZE-legal, valid quote, frozen mask/cardinality), compute `q_decision = EWMA(c_{≤t−1})` analyst-side and take its interquartile range `IQR_q` (report the 10–90 interdecile range too). Freeze:

```
Δ_material := ΔP_req / IQR_q,      with ΔP_req = 0.10 (owner-ratified materiality)
```

i.e. "the slope corresponding to a 10-percentage-point swing in request probability across the interquartile quote range." `ΔP_req` is a **pre-registered materiality choice**, ratified with the freeze, never retuned from results.

> **NOT executed in this draft** (respects the sequenced no-peek freeze order and keeps the draft design-only). The absolute `Δ_material` number lands when this read runs. **Caveat:** r9 is K=1/obs-v3; its `q_decision` spread is an **analogy** for the K=4/obs-v4 range, not a guarantee. If the in-regime range differs materially, `Δ_material` is mis-scaled — but note σ_d itself **is** measured in-regime from the screen; only the `Δ_material` scale borrows r9. Flag as a known limitation (§7).

### 5.2 The blinded variance estimator (what is computed)
From the n=5 screen's per-run estimates, form the **unsigned within-pair spread** for each seed s (arm labels concealed — you never learn which member of the pair is A vs B):

```
δ_s = | θ̂_{(1),s} − θ̂_{(2),s} |               (unsigned; sign = the effect direction, withheld)
σ̃_d²  = ( Σ_{s=1}^{n} δ_s² ) / (n − 1)          (blinded, mean-inclusive variance)
```

Because `δ_s² = d_s²`, this equals `s²_d + [n/(n−1)]·d̄²` — it **overestimates** σ_d² by the (unknown, withheld) squared effect. That upward bias is a **feature**: it is conservative (biases required n **upward**, never down) and it requires only the unsigned spread, so it is genuinely blind to the treatment contrast. Computed **separately per metric** (P2 slope, GATE-DOM slope, GATE-DOM top-tercile margin, GATE-INFL bottom-tercile rate) — they have different σ_d (§1.3.2).

### 5.3 What may NOT be inspected (the blinding boundary)
Frozen prohibition, in force from screen-completion until the claim tier is decided:
- **the sign of any `d_s`**, the mean paired difference `d̄` (= the treatment effect / any B−A, C−B, C−A contrast), any per-arm mean, or any arm-labeled quantity;
- any P1/P2/gate **point estimate or LCB** at the screen tier (the screen is SCREEN_PASS-only and never ACCEPT — pre-reg §1.7);
- any re-choice of `ΔP_req`, τ, the stratification, the estimator, or the metric set (all frozen pre-launch).
Only the **unsigned within-pair spreads `{δ_s}`** and the pre-frozen `Δ_material` enter the reassessment. A second analyst (or a masked pipeline that emits only `{δ_s}`) is the clean way to enforce this.

### 5.4 The decision rule (reassessed variance → required n)
For each load-bearing metric, with frozen target power `π` (P2 = 0.80; gates = 0.90) and α = 0.05 one-sided:

```
required d_z*(π)  from Table 2   (n=10: 0.853 @80%, 1.005 @90%)
n=10 is CERTIFIED for metric m  ⇔  Δ_material / σ̃_{d,m}  ≥  d_z*(π)
                                ⇔  σ̃_{d,m}  ≤  Δ_material / d_z*(π)
```

Equivalently, invert Table 1: find the smallest `n` with `Power(n, Δ_material/σ̃_{d,m}) ≥ π` → `n_req(m)`.

**Variance-of-variance posture (df = 4 — the advisor's trap).** A 5-pair variance estimate is *itself* very noisy. A one-sided UCB on σ_d inflates required n by `df/χ²_{α,df}`:

| UCB confidence | df=4 (screen) n-inflation (SD-inflation) |
|---|---|
| 95 % | **5.63×** (2.37×) |
| 80 % | 2.43× (1.56×) |
| 70 % | 1.82× (1.35×) |
| 50 % (median) | 1.19× (1.09×) |

A **95 % UCB near-always triggers escalate** and is *double* conservative on top of §5.2's mean-inflation — stacking them over-escalates. **Recommended posture: the blinded point estimate `σ̃_d` is PRIMARY** (it is already conservatively mean-inflated); **report the 80 % UCB as a sensitivity read** but do not gate on it. *(Open question §8: owner may prefer a 70–80 % UCB as primary for the safety gates — a deliberate, statable choice, not a silent 95 %.)*

### 5.5 The abort / escalate branch
- `n_req(m) ≤ n_max` for **all** load-bearing metrics → proceed: run seeds 46–50 to complete the claim tier at n=10 (or `n_req` if the owner authorized a stretch).
- `n_req(m) > n_max` for **any** load-bearing metric → **STOP the claim tier for that metric and escalate to the owner** before spending the second-wave GPU. Owner options (pre-registered per PDR-0082 reversal trigger): (i) authorize a seed-set stretch beyond 41–50 (itself a pre-reg amendment); (ii) accept a **descriptive-only** claim tier for that metric with the achieved MDE explicitly stated and a pre-registered statement that "the fatal case is live at screen"; (iii) redesign the estimator to cut σ_d (more decisions/run via longer runs, tighter stratification, or the Option-B shadow-ensemble quote §2.5.7) before re-screening. **No silent truncation, no silent n-cap.**

---

## 6. (c) Frozen max-n proposal

**Proposal: frozen max-n = 10** (= the pre-registered seed ceiling 41–50), with an **interim blinded reassessment after the n=5 screen** that gates whether the second seed-wave (46–50) is spent.

**Rationale.**
- The pre-registration already fixes seeds at 41–45 (screen) / 41–50 (claim); n=10 is the maximum reachable **without amending the frozen seed set**. Going beyond is itself a pre-reg change and must be decided at freeze, not mid-run.
- The reassessment's job with max-n = 10 is **certification, not enlargement**: it converts "we hope n=10 is enough" into "we measured σ_d blind and n=10 is / is not adequately powered," with the escalate branch (§5.5) as the honest response to "infeasible as scoped." Placing it **after the screen** means the claim-tier GPU is committed only if it will be adequately powered.

**Budget (each run ≈ 6 h GPU, Stage-2 200-round precedent; paired arms multiply run count).**

| Tier | Exp-1 (A,B) | Exp-2 (A,B,C) |
|---|---|---|
| n=5 screen | 10 runs ≈ 60 GPU-h | 15 runs ≈ 90 GPU-h |
| **n=10 claim (max-n)** | **20 runs ≈ 120 GPU-h** | **30 runs ≈ 180 GPU-h** |
| stretch n=12 (seeds 41–52)* | 24 runs ≈ 144 GPU-h | 36 runs ≈ 216 GPU-h |
| stretch n=16 (seeds 41–56)* | 32 runs ≈ 192 GPU-h | 48 runs ≈ 288 GPU-h |

\* **Optional owner-pre-authorized stretch** — only if the owner wants to avoid a mid-experiment escalation round-trip. Requires freezing the extended seed set **now** (a pre-registration amendment). If not pre-authorized, a reassessed `n_req > 10` follows the §5.5 escalate branch. **Recommendation:** freeze max-n = 10 as primary; let the owner *optionally* pre-authorize a stretch to n=12 or n=16 at the freeze signature. Given the tercile-gate variance inflation (§1.3.2), a stretch is a reasonable hedge for the **gates specifically** — but it is the owner's budget call, surfaced here, not assumed.

---

## 7. (d) What this power calc structurally CANNOT tell us

1. **It cannot certify the claim tier before the blinded reassessment.** Every number here is **conditional on an assumed σ_d**; it is not predictive of the actual power until σ_d is measured (§5). The parametric surface tells you *the shape of the trade*, never *where on it you sit*.
2. **Assumed-variance power is conditional, not predictive.** The honest deliverable is the *relative* ceiling `σ_d ≤ 1.17·Δ` (n=10, 80 %). A raw-units range is unavailable pre-pilot; the r9 anchor (§5.1) is the only route to an absolute reading, and it borrows a K=1/obs-v3 **scale** as an analogy (its own limitation).
3. **For the Exp-1 arm-B P2 (expected null), "power" is bounding, not detection.** A null is the *design prediction* (degeneracy under the floor). The calc quantifies how tightly we can bound the effect (0.95·σ_d @ n=5), **not** a probability of "detecting" something the design predicts is absent. The **gates carry the Exp-1 safety verdict; P2 does not.**
4. **It cannot verify that the pairing actually buys variance reduction.** "Paired fresh inits" shares only the **seed stream**, not the initialization; if the between-arm correlation ρ is low, the pairing's df penalty (n−1 vs an unpaired 2n−2) is **not** compensated and the paired test can be *less* powerful than unpaired. The calc *assumes* pairing helped (σ_d is written as if it already folds in the reduction). **Only the blinded σ_d reassessment reveals whether pairing worked** — if σ̃_d is large, low ρ is one candidate cause, and the estimand/pairing may need revisiting before the claim tier.
5. **At n=5 the test is granular and assumption-laden.** An exact one-sided paired sign-flip / signed-rank permutation test (the assumption-lean robustness sibling to the t-based LCB) has a **discreteness floor: the minimum achievable one-sided p = 1/2⁵ = 0.031** — so α is effectively 0.031, and only the maximally-extreme configuration reaches significance. The t-interval's 95 % coverage rests on approximate normality of **5** per-run differences, which 5 points cannot verify. Both are real limits of the screen tier (consistent with the pre-reg already treating n=5 as SCREEN_PASS-only, never ACCEPT).
6. **It says nothing about estimand validity.** If the per-run slope is confounded (late-request masking censors late commits → estimand drift, pre-reg D5; cross-slot quote leakage → G-CROSS-SLOT), the power to detect it is beside the point — a well-powered test of a biased estimand is precisely worse than an underpowered one. Power is necessary, not sufficient; the stratification/masking guardrails are separate load-bearing work.

---

## 8. Open questions for the main session

1. **`ΔP_req` materiality (owner-ratified).** Is `ΔP_req = 0.10` (a 10-pp request-probability swing across the interquartile quote range) the right frozen materiality for `Δ_material`? This single choice sets the absolute target the whole reassessment turns on. (§5.1)
2. **Variance-of-variance posture for the safety gates.** Point-estimate-primary (recommended) vs a deliberate 70–80 % UCB primary for GATE-DOM/GATE-INFL specifically, given the asymmetric cost of a missed alarm at df=4. (§5.4)
3. **Target power split.** Confirm P2 = 80 %, gates = 90 %. If the gates must be 90 % and their σ_d is tercile-inflated, n=10 is at real risk of failing certification → the stretch pre-authorization (Q4) becomes load-bearing. (§4, §6)
4. **Max-n stretch pre-authorization.** Freeze max-n = 10 only (escalate on `n_req>10`), or pre-authorize a stretch to n=12/16 (seeds 41–52/41–56, a seed-set amendment) at the freeze signature? (§6)
5. **Add the r9 `q_decision`-IQR read to the freeze-order queue.** It is a new r9-only calibration read (distinct from the already-queued r9 measurement-rate read); it must run pre-freeze for `Δ_material` to be a real number. Confirm it is legitimate under the no-peek order (r9 is archival; touches no new arm). (§5.1)
6. **Estimator-variance reduction as a contingency.** If reassessment escalates, is longer-run (more decisions/run) or the Option-B shadow-ensemble quote the preferred σ_d-reduction lever before re-screening? (§5.5)

---

## Appendix — computation method & self-checks

- **Noncentral-t power** by Simpson integration of `P(T'>t) = E_s[Φ(δ − t·s)]`, `s = √(χ²_{n−1}/(n−1))`, `δ = √n·d_z`. Central-t quantiles by bisection on the same integrator.
- **Self-checks (all passed):** central-t CDF `F(0) = 0.50000` (df 4, 9); `t_{0.95,4} = 2.13185`, `t_{0.95,9} = 1.83311`, `t_{0.95,19} = 1.72913`, `t_{0.95,11} = 1.79588` — all match textbook to 5 d.p.; exact power vs 3×10⁵-trial Monte-Carlo: `(n=5,d_z=1.0)` 0.5797 vs 0.5795, `(n=10,d_z=1.0)` 0.8975 vs 0.8981, `(n=5,d_z=0.5)` 0.2390 vs 0.2396, `(n=10,d_z=0.75)` 0.7066 vs 0.7072.
- **χ² UCB multipliers** (`df/χ²_{α,df}`) by Simpson integration of the χ² pdf + bisection: df=4 → 5.63× (95 %), 2.43× (80 %); df=9 → 2.71× (95 %), 1.67× (80 %).
- All figures reproducible from the stand-alone scripts used to author this draft (numerically self-contained; no scipy).
