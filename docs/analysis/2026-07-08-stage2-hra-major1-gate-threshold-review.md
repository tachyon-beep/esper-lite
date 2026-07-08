# Stage-2 HRA — MAJOR-1 Acceptance Gate: owner-set threshold review

- **Reviewer:** drl-expert (SME), 2026-07-08
- **Subject:** proposed values for the `<owner-set before ON>` slots in §11 of
  `2026-07-06-stage2-hra-major1-acceptance-gate.md`, reviewed BEFORE freeze.
- **Freeze-order note:** every proposal below resolves from OFF-arm data or fixed
  constants only, so all are freeze-order legal (no ON peeking). The critique is
  statistical soundness, not freeze legality.

Verdict shorthand: **APPROVE** = freeze as proposed; **APPROVE+** = freeze the
value but add the flagged rider before ON; **ADJUST** = change before freeze.

---

## 1. `Δparam_max` = max(0.10·median_s(added_params_off), IQR_s(added_params_off)) — **APPROVE+**

**Sound.** Freeze-order clean; the 0.10·median term doubles as the IQR→0 floor
(same role the `0.05` plays for `δ`), so a homogeneous OFF corpus cannot drive
the ceiling to zero and spuriously fail G2. Because the true param effect is ≈0
(critic-only, head-only detached V_cf), the dominant risk here is a *false G2
fail* (spurious INCONCLUSIVE → owner), not a false pass — so a permissive
ceiling is the correct error bias.

**Failure mode it lets through (documented, backstopped):** the ceiling uses a
*cross-seed* dispersion (IQR over OFF seeds) as the bound on a *paired* quantity
(median of ON−OFF per-seed differences). Pairing cancels the shared seed-level
baseline, so the paired-difference noise scale is generally *smaller* than the
cross-seed IQR. A systematic, consistent-sign param inflation (e.g. capacity-
farming the contribution reward) that is smaller than the between-seed spread
would pass G2. This is inherent to freezing from OFF-only data (the paired
ON−OFF distribution is unobservable pre-freeze), so it is accepted, not fixed.
**Backstop:** capacity-farming co-elevates germinate/fossilize churn → caught by
G3; and G2 is a *soft* gate (fail → INCONCLUSIVE → owner), not a silent pass.
Recommend the report state which term bound the ceiling (relative vs IQR) so the
owner can see when 0.10·median is doing all the work at large param scales.

---

## 2. `W` = max(10, plateau(value_target_scale on OFF)), cf_value_loss plateau as post-hoc CHECK — **ADJUST**

**Structure is correct and is the *only* freeze-legal option.** The spec's own §8
`W` definition embeds an ON-side condition (`cf_value_loss` plateau); you cannot
evaluate it before freeze without peeking. Demoting it to a post-hoc validity
flag and freezing `W` from OFF-observable `value_target_scale` is the right move,
and re-freezing `W` post-hoc would itself be peeking (choosing W to flatter ON) —
correctly avoided.

**Why it still needs adjustment — the bias is not symmetric.** §8 states the
slowest re-warm is `cf_value_normalizer` (high-variance R_cf), which
`value_target_scale` *does not track* and which is ON-leg-only. So a `W` frozen
from OFF `value_target_scale` can *under*-cover the ON cf re-warm. An undersized
W retains re-warm-contaminated ON updates, which depress/roughen ON `ev_sum`,
`ev_main`, and residual volatility — biasing **against** ON on LEG-A, LEG-B, AND
MECH simultaneously (a Type-II / miss-a-real-win bias). The post-hoc flag is
precisely the guard for this, so it must have teeth:

1. **Flag consequence must be explicit, not passive.** If `cf_value_loss` has not
   plateaued by `W` on an ON run, that run's scored window is re-warm-contaminated
   → downgrade to **INVALID (§1)** / owner-adjudication and re-run with a longer
   *pre-registered* W. "Flag it and score anyway" silently banks contaminated,
   anti-ON data. (INVALID here is safe: it never manufactures a false ACCEPT.)
2. **Pin the plateau detector algorithmically before freeze.** "`plateau(...)`"
   is underdetermined; an unspecified plateau rule is a peeking surface (analyst
   latitude picks W). Freeze an exact rule, e.g. *first update u where the
   rolling relative change of value_target_scale over the trailing K updates
   < p* (state K and p). Apply the identical rule to the ON `cf_value_loss`
   post-hoc check.
3. **Aggregate across OFF seeds conservatively.** A single global integer W must
   cover the *slowest*-warming OFF seed, so use `max` (or a high quantile, ≥P90)
   over per-seed OFF plateau updates — not the median. "plateau observed on OFF
   arms" (plural) currently reads ambiguous; median would systematically
   undersize W and re-introduce the anti-ON bias.

With 400 ep → ~96 updates and a plateau expected in the low tens, W in the
10–25 range leaves ~70–86 scored updates — ample. No concern on the discard cost.

---

## 3. `floored_asymmetry_max` = 0.10 — **APPROVE+**

**Well-calibrated to the update-count noise floor.** With ~86 scored updates the
binomial SE of a floored fraction near p≈0.1 is ≈0.03; the SE of the ON−OFF
difference ≈0.045, so 0.10 sits ~2.2 SE above noise — tight enough to catch a
selection-biasing asymmetry, loose enough not to fire on sampling noise (and
update-to-update autocorrelation shrinks the effective n, making a tighter 0.05
threshold produce spurious INVALIDs). This is the core MAJOR-1 defense — an
asymmetric exclusion set means ON and OFF level/vol are computed over different
update subsamples, confounding the paired EV comparison. 0.10 is a defensible
freeze.

**Gap it leaves (recommend closing before ON):** §8B registers only an
*asymmetry* bound, no *absolute* floored-fraction ceiling. A **symmetric-but-high**
floored fraction (both legs, say, 45% floored) passes the asymmetry gate yet
means the EV statistic rests on a minority of updates and is fragile. This bites
hardest on the **`ev_main` / returns_main series feeding MECH**: by the
intervention's premise `returns_main = returns_total − returns_cf` is the
low-variance remainder *most* likely to fall under the 1.0 floor. A gutted-but-
symmetric MECH series would let a floor artifact satisfy the structural MECH
gate. Recommend adding an absolute rider: **if either leg's floored fraction on a
given metric series > 0.5, that metric's leg → INCONCLUSIVE/INVALID regardless of
asymmetry**, scoped especially to the ev_main/MECH series. (Code already reads
`ev_return_variance` and per-leg fractions, so this is a threshold, not new
plumbing.)

---

## 4. `g3_ratio_max` = 1.5 (churn material-elevation) — **APPROVE+**

**The value is defensible.** G3 uses *per-episode rates* averaged over 400
episodes, so the ratio-of-means is well-estimated (low SE) — a 1.5× sustained
elevation is a real regime difference, not noise, and won't spuriously fire the
way a small-sample ratio would. Against baselines germinate 12.4 / prune 11.6 per
ep, 1.5× triggers at ~18.6 / ~17.4 — a materiality line consistent with an
"is this a reward-farm signature" question. G3 is soft (fail → INCONCLUSIVE →
owner), so 1.5 erring slightly permissive is acceptable.

**Two riders before freeze:**
1. **Per-channel, not aggregate.** §6 names germinate / prune / fossilize as three
   channels with different implications (fossilize ties to G2 params; germinate/
   prune are the churn-farm signature). Specify G3 holds ⟺ *each* channel ratio
   < 1.5 (i.e. max over channels < 1.5), not a pooled/aggregate ratio that can
   dilute a single-channel spike.
2. **Abs-floor the fossilize channel.** Fossilization is a rare, near-terminal
   event; its per-episode count can be low enough that the ratio is Poisson-noisy.
   Mirror G4's `g4_abs_floor` for the fossilize channel (germinate/prune at
   ~12/ep need no floor). Also prefer a *paired per-seed* ratio median over pooled
   means for consistency with G1/G2 and to remove between-seed variance.

---

## 5. `g4_ratio_max` = 2.0, `g4_abs_floor` = 5 events/run — **ADJUST**

**Values are reasonably chosen** — 2.0 sits deliberately above the known-benign
1.68× governor-rollback asymmetry (PDR-0026), and a 5-event/run floor sensibly
suppresses Poisson ratio blowups (e.g. 4-vs-1) on rare channels. But the
**single-ratio framing contradicts the spec's own carve-out and mis-instruments
the near-zero-base channels:**

1. **Governor rollbacks must not hard-gate.** §6 G4 explicitly makes governor-
   rollback asymmetry "an RCA trigger, not a hard gate" (PDR-0026). Yet G4==hold
   is a *required* input to `clean` (§7). Applying `g4_ratio_max` to rollbacks
   would let a rollback ratio flip G4→not-hold and block ACCEPT — the exact hard-
   gating PDR-0026 forbids. **Separate governor-rollback (report + RCA trigger,
   does NOT flip G4 hold) from the pathology detectors (which determine hold).**
2. **A pure ratio is the wrong instrument for detectors whose healthy base rate is
   ~0** (reward-hacking, value-collapse, gradient-anomaly). With OFF≈0 the ratio
   is undefined/floored, so abs_floor=5 makes them gate *only* by the floor — an
   ON cluster of, e.g., 4 reward-hacking flags vs 0 on OFF is suppressed and never
   trips. For these, add a **separate absolute ON-count trigger** (ON ≥ k with
   OFF≈0 → G4 not-hold + RCA), independent of the ratio. The ratio+floor is the
   right tool only for the moderate-base rollback channel — which per (1) should
   not hard-gate anyway.
3. **Below-floor channels must still be *reported* raw**, not silently dropped.
   "Floor suppresses the ratio *gate*" is fine; "floor suppresses *visibility*"
   would hide a genuine 4-vs-0 reward-hacking cluster from the owner.

---

## 6. `advantage_std_floored` = report fractions + OBSERVED flag, no hard cutoff — **APPROVE+**

**Report-only is defensible and the owner's refusal to freeze an arbitrary cutoff
is sound.** Verified in code: `advantage_std_floored` is a per-update bool
aggregated to a fraction; the **LEG-B gated statistic is `IQR_u(sqrt(1−ev_u))`,
computed from `ev` — independent of the clamped advantage std** (raw
`pre_norm_advantage_std` IQR is only a *covariate*, §4). So the floor's path to
the *gated* statistic is indirect (via training dynamics), not a direct
measurement contaminant, and there is no principled benign firing-rate to
pre-register. `per_head_advantage_norm=False` and identical `ADVANTAGE_STD_FLOOR`
are already frozen both legs (§8F/§10), removing the config confound.

**Rider — add the mirage linkage.** The residual risk is a floor-artifact LEG-B
"win": if the ON advantage std is being *clamped more often* than OFF, its
volatility can look reduced for a non-mechanistic reason. This is the same shape
as the existing §4 `gradient_cv`-rising mirage rule. Add: **a LEG-B PASS that
co-occurs with ON-elevated `advantage_std_floored` fraction → downgrade to
INCONCLUSIVE + RCA.** The report must state the *direction* (ON floored more/less
than OFF), not just the magnitudes — direction is what distinguishes a real win
from a clamp artifact. Given the indirect path, this is a safeguard, not a
correction; report-only + this linkage is sufficient.

---

## 7. Episode budget = 400/run (~96 updates, ~86 scored) — **APPROVE+**

**Strongly justified.** LEG-B ("is the win real") and every `vol(·)` statistic
rest on **IQR-over-updates**, whose stability scales ~1/√(n_updates). 200 ep →
~48 updates → ~38 scored is marginal for a quartile spread; 400 ep → ~86 scored
is comfortable. §10 already says "longer strictly better for the volatility
estimates" — 400 is the right call, and doubling the 200-ep prior is proportionate
to the LEG-B dependence on within-run dispersion.

**Two riders:**
1. **Pin the exact *scored-update* floor, not just the episode count.** §1
   invalidates a run with "fewer scored updates than the pre-registered budget,"
   but 0.24 updates/ep is an *estimate* and realized updates/ep will vary run to
   run. If you pre-register 400 *episodes* but validity is checked on *updates*, a
   run can be spuriously INVALID or spuriously pass. Freeze **both**: 400 episodes
   AND a minimum scored-update count (e.g. a floor set below the expected ~86 with
   margin). This is a genuine §11 under-specification, not just this proposal's.
2. **Add a within-run non-stationarity check.** A 400-ep run may enter a late
   training regime (lifecycle saturation, policy drift) qualitatively unlike the
   200-ep prior, so pooling all ~86 updates into one median/IQR could mix regimes.
   Report a split-half IQR-stability check per leg so the added updates are shown
   comparable, not silently regime-mixed. The existing `Var(returns_total)`
   covariate (§8E) partially surfaces this; split-half makes it explicit.

Compute note: n=5 screen = 10 runs × 400 = 4,000 ep; n=10 claim = 20 runs =
8,000 ep. Confirm the budget supports it — statistically it is the right choice.

---

## 8. Unfrozen / under-specified items the proposals miss (§11 completeness)

1. **Plateau-detection algorithm for `W`** — must be a pinned exact rule (K, p),
   not "plateau(...)"; otherwise it is a peeking surface. (Proposal 2.)
2. **Cross-seed aggregation for `W`** — specify max/high-quantile over OFF seeds,
   not median. (Proposal 2.)
3. **Exact scored-update floor for §1 validity** — episode budget ≠ update budget;
   pin both. (Proposal 7.)
4. **Absolute floored-fraction ceiling** — §8B registers asymmetry only; add an
   absolute ceiling for the floor-prone ev_main/MECH series. (Proposal 3.)
5. **G3 channel granularity + fossilize abs-floor** — per-channel application and
   a rare-event floor are unspecified. (Proposal 4.)
6. **G4 governor-rollback carve-out + absolute pathology-detector trigger** — the
   single-ratio framing contradicts PDR-0026 and mis-instruments zero-base
   detectors. (Proposal 5.)
7. **`δ` at the n=5 screen** is a coarse 5-point bootstrap SE (the fallback's
   `n<5` trigger does not fire at exactly 5). Not owner-set, so out of scope, but
   the `max(0.05, …)` floor is load-bearing at the screen — worth an eyes-open
   note that the screen δ is noisy by construction.

None of these change the freeze-order legality; they close gaps that would
otherwise surface as silent contamination or as analyst latitude at freeze time.

---

## Confidence Assessment

- **High** that the freeze-order structure is sound and that Proposals 3, 6, 7 are
  statistically well-founded at the proposed values (calibrated to the ~86-update
  noise floor and the ev-independence of the gated statistic, both verified
  against spec + code).
- **High** that Proposal 5 (G4) as framed conflicts with the spec's own PDR-0026
  governor-rollback carve-out and mis-instruments zero-base detectors — this is a
  logical conflict, not a judgment call.
- **Medium-high** on the Proposal 2 (W) anti-ON undersizing risk — the mechanism
  (cf_value_normalizer re-warm untracked by value_target_scale) is stated in the
  spec; the *magnitude* of undersizing is unobservable pre-freeze, which is
  exactly why the post-hoc check needs teeth.
- **Medium** on the specific numeric lines for Proposals 1 and 4 (Δparam_max
  formula, g3=1.5) — defensible but calibrated against means/anchors whose
  cross-seed/paired dispersion I have not measured directly.

## Risk Assessment

- **Highest residual risk:** Proposal 2 undersized W → systematic anti-ON bias
  across all three EV legs → a real Stage-2 win reads as null/reject. Mitigated
  only if the post-hoc cf_value_loss flag forces INVALID/re-run, not a note.
- **Second:** Proposal 5 rollback ratio hard-gating → either (a) blocks a clean
  ACCEPT on a benign rollback asymmetry, or (b) if rollbacks are excluded, a real
  ON-specific reward-hacking cluster with OFF≈0 slips the floor. Both fixed by the
  carve-out + absolute trigger.
- **Third (low, backstopped):** Proposal 1 permissive paired ceiling and Proposal
  3 symmetric-high flooring — each has a named backstop (G3; the recommended
  absolute floored-fraction ceiling).
- Type-I (false ACCEPT) risk is structurally low: the composite is a conjunction
  of all gates and ACCEPT banks only at n=10. The dominant error mode across these
  slots is Type-II (false reject / INCONCLUSIVE), which is the safer direction for
  an acceptance gate but wastes real positive results — hence the emphasis on not
  under-sizing W and not over-tightening the asymmetry/ratio gates.

## Information Gaps

- Cross-seed IQR of OFF `fossilized_params` and of per-episode churn — needed to
  know whether Δparam_max and g3=1.5 sit well above the paired noise floor. These
  are OFF-arm quantities computable at freeze step (2); recommend the freeze report
  print them so the owner freezes with the dispersion visible.
- Realized `updates/episode` for a 400-ep run (0.24 is a Stage-0 estimate) — sets
  the scored-update floor for §1.
- The plateau trajectory of `value_target_scale` vs `cf_value_loss` on OFF/ON
  pilots — would quantify the W undersizing gap; unobservable for ON pre-freeze by
  design.
- Base rates of the value-collapse / gradient-anomaly / reward-hacking detectors
  in prior runs — sets the absolute pathology-detector trigger k for G4.

## Caveats

- This review judges *statistical soundness of the threshold values*, taking the
  spec's algebra (§3 identity, §4 scale-invariant statistic, §8B floor mechanics)
  and the cited code as the contract; I verified the EV-floor clamp and the
  advantage-std-floored aggregation in code, not the full scorer/wrapper.
- All "add a rider" recommendations are freeze-time threshold/rule additions, not
  training-code changes — consistent with the pure-scorer/wrapper split (§11).
- I did not run the A/B or inspect live telemetry; anchors are as supplied
  (metrics.md n=5 churn, PDR-0026 rollback ratio, Stage-0 update cadence).
- Approvals are conditional on the freeze report printing the OFF-arm dispersions
  it resolves from (δ, Δparam_max terms, floored fractions, churn IQRs), so the
  owner freezes with the calibration visible rather than on formula trust.
