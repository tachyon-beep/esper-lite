# F3 gate-mechanism simulation: candidate qualification gates vs measured near-cliff noise

Date: 2026-07-14. Owner directive: "can't yolo that one" — data-backed selection (PDR-0098). Machinery:
the phase-1 threshold-variance Monte-Carlo (seeded, 20k draws/cell), evaluated at the MEASURED near-cliff
noise (PDR-0092: σ P50 = 0.82 pp, P90 = 1.87 pp). Windfall = max P(pay) over μ ∈ {0.85, 0.90, 0.95}
(all below the 1.0 cliff; bound ΔP < 0.10 owner-ratified). Denial = mean P(deny) over μ ∈ {1.5, 2, 3}
(deserved premiums).

## Results (P(pay) per cell)

| candidate | σ | μ=0.85 | 0.90 | 0.95 | 1.05 | 1.50 | 2.00 | 3.00 | windfall | denial |
|---|---|---|---|---|---|---|---|---|---|---|
| S1 raw span5 n=5 (pre-reg primary) | 0.82 | .351 | .399 | .449 | .551 | .900 | .994 | 1.0 | **.449 FAIL** | .035 |
| S2 LCB z=1 n=5 (pre-reg fallback) | 0.82 | .116 | .143 | .172 | .241 | .647 | .940 | 1.0 | **.172 FAIL** | .138 |
| S3 LCB z=1.5 n=5 | 0.82 | .059 | .078 | .097 | .143 | .495 | .866 | .999 | .097 pass | .213 |
| S3 | 1.87 | | | | | | | | **.109 FAIL** | .504 |
| S4 LCB z=2 n=5 | 0.82/1.87 | | | | | | | | .052/.059 pass | **.300/.626** |
| S5 raw span10 n=10 | 0.82 | | | | | | | | .439 FAIL | .013 |
| S6 LCB z=1 span10 n=10 | 0.82 | | | | | | | | .154 FAIL | .067 |
| S7 margin 0.5·sd n=5 | 0.82 | | | | | | | | .150 FAIL | .154 |
| S8 margin 1.0·sd n=5 | 0.82/1.87 | | | | | | | | .040/.045 pass | **.342/.676** |
| **S9 LCB z=1.5 span10 n=10** | **0.82** | .040 | .055 | .077 | .138 | .668 | .979 | 1.0 | **.077 PASS** | **.118** |
| **S9** | **1.87** | .070 | .079 | .091 | .118 | .308 | .591 | .951 | **.091 PASS** | .384 |

(Full grid reproducible from the inline script in the session record; 20k draws, seed 41; i.i.d. N(μ,σ)
window model — same machinery as the phase-1 formula test.)

## Verdict — PROPOSED F3 resolution (owner ratifies at freeze)

**S9: extend the confirmation window to ≥10 valid measurements (`min_window=10`), smooth with a span-10
EWMA, qualify at `LCB(z=1.5) ≥ 1.0`.** The only candidate on the efficient frontier that meets ΔP<0.10 at
BOTH measured noise levels; its denial cost (0.118 at σ=0.82) is 2.5–3× lower than the other passers
(S4: 0.300, S8: 0.342). Both pre-registered forms (S1, S2) fail as PDR-0092 predicted.

**Costs, stated:** (a) the window roughly doubles → settlement delay d grows (~10–19 epochs at W=10);
discount-neutralization handles the prior's PV; the annuity horizon shortens slightly; (b) lock duration
grows ~1.5–2× → burden estimate ~2.7–3.5% locked epochs (still ≪ the 25% trigger; PDR-0095's overlay is
rate-linear); (c) the late-request mask cuts in ~5–10 epochs earlier (more late-masked requests near
horizon — replay quantifies); (d) feasibility is guaranteed by the 100% measurement cadence (PDR-0087).

**Structural honesty:** at μ=0.95 (0.05 below the cliff, noise 0.82) NO feasible windowed gate can
discriminate — S9 meets the bound there by being CONSERVATIVE, which costs ~33% denial at μ=1.5. That
residual is a real, designed anti-commit force on marginal-but-genuine seeds and must be named in
GATE-DOM's interpretation notes (marginal seeds under-commit partly by gate design, not policy choice).
The i.i.d.-normal window model and the σ anchor (r9 tail, K=1/obs-v3) are the standing caveats
(PDR-0092); the K=4 screen re-measures σ in-regime and this table re-evaluates in minutes if σ moves.
