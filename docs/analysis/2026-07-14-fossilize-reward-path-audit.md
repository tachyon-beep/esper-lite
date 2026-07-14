# FOSSILIZE reward-path audit (freeze blocker #2, rounds 15–17)

Date: 2026-07-14. Required by the round-15 owner adjudication ("trace the complete FOSSILIZE reward path") before any freeze. Source of truth: `src/esper/simic/rewards/contribution.py` + `rewards.py` + `shaping.py` + `leyline/__init__.py`, read at `5ca2a04f`. Scope: **SHAPED mode** (the experiment regime; A/B/C all run SHAPED per §1.3). Invariant under test (gpt-prime): *"No contribution measurement available at, or selected by, the request instant may enter an immediate FOSSILIZE payment."*

## 1. The complete ledger

### At the commit instant (FOSSILIZE action, valid contributing HOLDING seed)
| # | Term | Value | c-dependence at request instant | Where |
|---|---|---|---|---|
| 1 | Per-step provisional attribution | `contribution_weight(1.0)·attributed(c_spot)·discounts` | **spot-c-graded** but action-independent (a WAIT step pays the same) → not a FOSSILIZE-differential incentive. Zeroed if lifetime counterfactual < 0. | `contribution.py:569-601,644-654` |
| 2 | Flat base bonus | `fossilize_base_bonus 0.5 × legitimacy` | **spot-c-GATED**: paid only if `c_spot ≥ DEFAULT_MIN_FOSSILIZE_CONTRIBUTION (1.0)` | `:207, shaping helper` |
| 3 | Graded bonus | `0.1 × c_spot × legitimacy` | **spot-c-graded** (the known sell-at-spike term) | `:208` |
| 4 | "Immediate terminal" bonus | `3.0 × tanh(1/3) × legitimacy ≈ 0.964 × legitimacy` | **CONSTANT in magnitude** — `tanh(1.0/quality_ceiling)`, NOT `tanh(c)`. But **spot-c-GATED** (same `c_spot ≥ 1.0` + counterfactual ≥ 0 gate) | `:849-863` |
| 5 | Transaction cost | `fossilize_cost −0.01` | none | `:202` |
| 6 | **PBRS progress-forfeiture** (realized next step, HOLDING→FOSSILIZED) | `0.3·(0.995·6.0 − (5.5 + min(dwell·0.3, 2.0)))` = **−0.309 at dwell 5, −0.459 at dwell ≥7**; positive only for dwell ≤1 | dwell-graded, not c-graded | `_contribution_pbrs_bonus`, `STAGE_POTENTIALS` |

`legitimacy = min(1, epochs_in_HOLDING / MIN_HOLDING_EPOCHS(5))`. Penalty branches: invalid (no seed / not HOLDING / counterfactual None) → **−1.0**; counterfactual < 0 → **−0.5 − min(|cf|·0.2, 1.0) − 0.3·ransomware** (counterfactual-graded at the instant); `c_spot < 1.0` → **−0.2** (`fossilize_noncontributing_penalty`).

### The stay-in-HOLDING branch (the alternative being priced against)
Per-epoch: provisional attribution ~`c` (the ~3/epoch for good seeds) **minus** `holding_warning` **−0.1→−0.3/epoch** (non-terminal actions in HOLDING, `epochs_in_stage ≥ 2` ∧ `bounded_attribution > 0` — an anti-park penalty that fires exactly on GOOD seeds) **plus** PBRS progress accrual ≈ +0.08/epoch until the 2.0 cap (dwell ~7), then ≈ −0.01/epoch; occupancy rent above `free_slots=1`.

### Post-commit, per epoch until horizon
Attribution → **0** (frozen; the H4 fact); `fossilized_rent` **−0.002/epoch**; fossils still count in `n_occupied` (no occupancy escape); global terminal `val_acc × 0.05` at `max_epochs` (fossils contribute only causally via val_acc). **Drip** (70% pool, per-epoch keyed on *current* counterfactual): **BASIC_PLUS ONLY — inactive in SHAPED**; must stay mode-fenced from the settlement annuity (no coexistence path).

## 2. Findings

- **F1 — The "tanh contribution bonus" was mischaracterized in rounds 15–17.** It is constant-magnitude (`3.0·tanh(1/3) ≈ 0.964`), not contribution-scaled. It is spot-c-*gated*, not c-*graded*.
- **F2 — The shipping flat commit prior is ≈ 1.464 × legitimacy, not 0.5.** Terms 2+4 are BOTH flat and both live in arm A. The owner's matched-control rule ("retain exactly the existing flat premium; remove every contribution-dependent payment"), applied to the audited ledger, keys on **0.5 + 0.964 ≈ 1.464** — deleting the tanh term as "contribution-dependent" would itself be an intervention (halving the existing flat prior), the exact mistake the matched-control frame exists to prevent. **Recommendation: consolidate both flat terms into a single flat premium = 0.5 + 3.0·tanh(1/3) ≈ 1.464 (× legitimacy), remove only the 0.1·c grading.** One-line owner confirm needed (the adjudication said "+0.5" on the belief the tanh term was c-scaled).
- **F3 — The eligibility gate violates the invariant even after `0.1·c` dies.** `c_spot ≥ 1.0` at the request instant makes the *whole* ≈1.464 payment a binary function of the spot LOO — a threshold-crossing sell-at-spike channel (spike above 1.0 → collect; dip below → −0.2). **Fix: re-key the gate (and the −0.2 branch) from `c_spot` to `q_decision`** (the lagged EWMA) — preserves the gate's function, removes the spike instrument.
- **F4 — PR28 CONFIRMED.** Additional c/counterfactual-keyed commit-time terms beyond the three known: (a) the threshold gate on both flat bonuses (F3); (b) the `−0.2` noncontributing branch (spot-c-keyed); (c) the counterfactual-graded invalid branch (`−0.5 − |cf|·0.2 − ransomware`); (d) the bounded-attribution zeroing keyed on lifetime counterfactual at the instant.
- **F5 — NEW, unpredicted: PBRS progress-forfeiture is a dwell-graded commit-time penalty (−0.31…−0.46 at legitimate dwells).** Committing forfeits HOLDING's accrued progress potential. Settlement interaction: the request→boundary window lengthens dwell → deeper forfeiture at the boundary transition → **G-LEDGER must include the PBRS potential path**, and the boundary transition's PBRS must be priced into the ledger-continuity check.
- **F6 — Round-15's dominance ledger missed the stay-branch penalty.** Stay is not pure `c·Σγ^k`: good seeds pay −0.1→−0.3/epoch indecision. Small vs ~3/epoch attribution, but it belongs in any B−A ledger statement.
- **F7 — Corrected fixed ledger (fully-legitimate commit, dwell 5, ~75 epochs remaining):** `+1.464 − 0.01 − 0.309 − PV(0.002/epoch ≈ 0.125)` ≈ **+1.02**. Not gpt's ≈+0.37 (0.5-only arithmetic) and not round-15's net-negative (zeroed-everything arithmetic). The −113-scale forfeited attribution stream remains the dominant term; that is the annuity's job.
- **F8 — The negative-carry escape hatch is LIVE in the shipping reward (round-17 §3(i) is not hypothetical).** A declining seed with lifetime counterfactual ≥ 0 but spot `c < 1.0` (including c < 0) can FOSSILIZE for a one-shot −0.2, then pay 0 attribution forever — vs staying and bleeding `c<0` per epoch. "Commit the decliner to stop the bleeding" is priced IN today. The settlement must apply the identical sign/clip transform through the annuity (negative `q_settle` → negative annuity) or mask requests at negative `q_decision`; do NOT rely on the behavioral gate to catch it (gates are one-sided in B).

## 3. Invariant verdict
**FAILS as shipped, four ways** (0.1·c grading; spot-c threshold gate on ≈1.464; spot-c −0.2 branch; instant-counterfactual-graded penalties). **Compliant form:** single flat premium ≈1.464×legitimacy, eligibility + penalty branches re-keyed to `q_decision`, validity re-keyed to the settlement's FRESH-staleness requirement, all c-graded *payment* moved to the policy-unselectable `q_settle` annuity, premium + maintenance-start paid at the **boundary** (not the request), PBRS boundary transition priced in G-LEDGER.

PR29 note: negative-LOO share ~10–12% (round-1 banked) makes F8 material, not a corner case.
