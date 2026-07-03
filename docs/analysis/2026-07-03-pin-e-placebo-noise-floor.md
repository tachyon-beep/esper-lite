# PIN-E Placebo Noise Floor — tau Calibration (D1 + D2)

Date: 2026-07-03 · Task: esper-lite-94869250f1 · Gate: esper-lite-f22a1d48a7 (first leg)
Plan: `docs/plans/ready/2026-07-03-pin-e-placebo-harness.md` (dual-reviewed) · PDRs: 0014, 0015
Analyzer: `scripts/pin_e_placebo_analyze.py` (commit 9be132c1) · Report JSON: `telemetry/pin_e/pin_e_noise_floor_report.json`

## Headline

**Recommended tau (`shapley_synergy_noise_floor`) = +0.28 pp — CONSERVATIVE-PROVISIONAL
LOWER BOUND** (the 1e-3 arm's P99 bootstrap upper CI +0.272, rounded up to the next
grid-aligned margin). **The epsilon plateau test FAILED** (a PDR-0015 trigger condition):
the floor is magnitude-dependent, so any placebo-derived tau under-estimates the floor for
real-magnitude committed seeds. tau MUST be recalibrated at the first ON run (F2 leg)
before any reading of the live term is trusted. Non-degeneracy PASSED on both arms —
the PDR-0014 reversal trigger (degenerate/vacuous floor → method option (a)) did NOT fire.

## Setup (provenance-gated)

Six runs, 2 arms × 3 seeds (41/42/43), 30 episode-rounds × 12 envs × 150 epochs each:
three near-inert placebo seeds (3×3 depthwise-conv residual, `seed_lr_override=0`,
frozen delta) forced through the declared serial schedule to co-resident HOLDING α=1
(epochs 51–150), never fossilized. `shapley_synergy_scale=0.0` throughout (hard
preflight). The fused val pass logs the full 2³ counterfactual factorial every epoch;
φ, c_paid = v({s})−v(∅), and signed excess e = φ−c_paid are assembled offline per
terminal episode. All values quantized on the val grid (100/834 ≈ 0.1199 pp).
N = 1080 episodes = 3240 per-seed terminal samples per arm.

## D2 — tau (signed terminal excess), per arm

| | std=1e-4 | std=1e-3 |
|---|---|---|
| N (samples / episodes) | 3240 / 1080 | 3240 / 1080 |
| mean | +0.0020 | +0.0014 |
| P50 | 0.0000 | 0.0000 |
| P90 | +0.0600 | +0.1199 |
| P95 | +0.0799 | +0.1599 |
| **P99 (= tau)** | **+0.1201** | **+0.2401** |
| block-bootstrap 95% CI of P99 (B=10000, block=episode) | [+0.1000, +0.1401] | [+0.2398, +0.2721] |
| max / min | +0.2001 / −0.1801 | +0.4402 / −0.3797 |
| nonzero fraction | 38.1% | 87.5% |

Mean excess ≈ 0 on both arms — the Shapley assembly is unbiased for a null player; the
efficiency identity Σφ = v(C)−v(∅) held exactly on all 2160 terminals (hard-checked).
Full-run tau (1e-3) is consistent with the smoke preview (+0.240 vs +0.252) with the CI
tightened ~9× ([+0.240, +0.272] vs [+0.106, +0.380]).

## Epsilon plateau — FAILED (flagged trigger)

tau(1e-3) = +0.240 vs tau(1e-4) = +0.120; CIs disjoint. A 10× smaller placebo halves the
floor (sub-linear scaling), and the 1e-4 arm sits at ~one quantization grid step — the
basement any nonzero effect can reach. **The floor is therefore NOT ε-independent
measurement noise; it scales with the null player's own perturbation magnitude.**

This is the PDR-0015 sharpened trigger condition, and it is hereby reported as FIRED.
Interpretation (for the owner's gate decision):

- The trigger's drafted consequence was "revert to methodology §5 option (a) — K
  resampled val minibatches." Option (a) measures val-resampling noise at fixed weights;
  it is *also* magnitude-blind and would not repair this. The data shows the floor has a
  magnitude-independent basement (~1 grid step) plus a magnitude-tracking component.
- The direction is conservative-compatible: real committed seeds perturb the network
  far more than ε=1e-3, so their zero-synergy excess noise plausibly EXCEEDS +0.24. A
  placebo tau is a **lower bound** on the correct deadband — precisely the
  CONSERVATIVE-PROVISIONAL framing pre-ratified in the plan (R5). What the plateau
  failure changes is the *strength* of the provisionality: recalibration at the first ON
  run is now mandatory, not advisory, and the heteroscedastic caveat in
  esper-lite-94869250f1 ("noise at zero contribution only") is confirmed empirically.
- Options at the gate: (i) accept tau=+0.28 as the provisional lower bound and proceed
  to F2 with mandatory ON-run recalibration (recommended); (ii) supplement with option
  (a) to decompose the basement vs magnitude components (informational, ~zero GPU-cost
  offline variant unavailable — needs K extra val passes per epoch, a run-time change);
  (iii) re-scope per the drafted trigger. The recommendation is (i): the lower bound is
  usable *because* the term's other guards (cap, normalized_cap, efficiency clamp G)
  bound the damage of an under-sized tau, and F2 recalibration closes the gap.

## D1 — GATE-0 per-stage credited-LOO line (source: credited)

Grid step 0.1199 pp. GERMINATED/TRAINING: structural zero (α=0, STE), no credited
samples, as designed. Pre-germination NONE rows carry no credit.

| stage | arm 1e-4: mean / std (N) | arm 1e-3: mean / std (N) |
|---|---|---|
| BLENDING | +0.0002 / 0.0457 (7560) | −0.0025 / 0.1238 (7560) |
| HOLDING (tau-relevant) | +0.0001 / 0.0612 (135000) | +0.0017 / 0.1832 (135000) |

**GATE-0 line:** at the freeloader threshold the credit system is *unbiased* (per-stage
|mean| ≤ 2% of one grid step, CoV undefined below grid) with a per-step noise floor of
std ≈ 0.06–0.18 pp, itself magnitude-dependent (3× across the 10× ε range — consistent
with the D2 plateau failure). Non-degeneracy: the placebo's spread is driven by real
val-sample boundary flips (the same mechanism that moves a real seed's c_t), satisfying
the PDR-0014 requirement; the structural same-noise-sources argument is in plan §2.6.
Data-quality note: the 1e-4 arm logged 161,998 of 162,000 expected snapshots (2 missing
TRAINING rows, 1.2e-5 of data; immaterial).

## False-positive budget at tau=+0.28

By construction P(e > P99) ≈ 1% per null seed-episode at ε=1e-3; at tau=+0.28 (above the
upper CI) the empirical exceedance in this dataset is 0.7% (23/3240; max observed +0.44,
i.e. worst single leak before scale/caps ≈ 0.16 pp). With k=3 committed seeds, expect
≲2%/episode of episodes to pay any spurious top-up, each bounded by
min(cap, scale·(e−tau)) and the efficiency clamp G. CAVEAT: these exceedance rates are
placebo-magnitude rates; real-magnitude nulls will exceed more often (the plateau
finding) — the budget is illustrative until F2 recalibration.

## Masking equivalence (drl Rec3) — PASSED

`tests/kasmina/test_placebo_masking_equivalence.py` (commit 1c05ba4f): HOLDING
alpha-override path ≡ live scalar path ≡ FOSSILIZED live forward, bit-identical; the
disabled-config leg (override=0 ≡ host-only ≡ never-germinated) was already pinned by
`test_committed_shapley_kernel_masking.py`. tau measured on HOLDING placebos transfers
to the live fossil-masking regime.

## Verdicts

| Check | Verdict |
|---|---|
| Non-degeneracy (PDR-0014 trigger) | **PASSED** both arms — trigger NOT fired |
| Epsilon plateau (PDR-0015 condition) | **FAILED — trigger FIRED**; floor is magnitude-dependent; owner decision at the gate |
| Shapley efficiency identity | exact on all 2160 terminals |
| Masking equivalence (Rec3) | PASSED, bit-identical |
| D1 bias at freeloader threshold | unbiased (sub-grid) |
| tau recommendation | **+0.28 pp, CONSERVATIVE-PROVISIONAL LOWER BOUND; mandatory recalibration at first ON run** |
