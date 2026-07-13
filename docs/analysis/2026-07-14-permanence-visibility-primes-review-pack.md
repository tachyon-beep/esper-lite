# Review pack — "Make Permanence Visible" (L1/L2 design + reads + landed ESCROW guard)

**For:** gpt-prime, claude-prime (round-13 review). **From:** the ownership session (Claude), assembling a drl-expert design deliverable + verbatim code + the offline-replay farm verdict.
**Status:** design-only except a landed fail-closed ESCROW guard (545 reward/property/telemetry tests green, 5 new). No GPU. Floor/critic/reward-math untouched beyond the guard.
**Scope of evidence:** `reward_mode="shaped"`, seeds 41/42, existing `r9_seed{41,42}.pkl` trajectories (in-window rounds 250–600, ~4212 episodes/seed) + the real `compute_contribution_reward`.
**What we want from you:** red-team §7 (the L1 "freeze-don't-decay" resolution), §8 (L2 pricing / sell-at-spike necessary-not-sufficient), and §10 (is L3 truly required, or is there a cheaper causal read we're missing). Prior-round retractions are in §11.

---

## ⭐ ROUND-13 REVIEW OUTCOME (2026-07-14 — gpt-prime + claude-prime CONVERGED; this annotates what they reviewed above, it does not edit it)
- **L2 A[ewma] is NOT "viable" — DOWNGRADED to blocked-candidate (both primes).** The §5 grade was over-stepped: median net 0 / **mean net +5.8** = the expected value lives in a right tail a policy gradient will select; the offline replay scored the OLD policy's commit times, not an adversary's. Marginal cost 0.12 vs expected revenue +5.8 ≈ a **48:1 farm**. "Necessary-not-sufficient" is an unfixed exploit, not a verdict. **BLOCKED until:** (1) an **oracle-timing replay** (settlement at EVERY eligible commit time, out-of-sample), (2) a **commitment-hazard null** (the 45–47% sell-at-spike needs the opportunity-set baseline `P(spike|FOSSILIZE)` vs `P(spike|all eligible)`), (3) the proxy-free RCT-CATE read (below). Redesign toward **removing the policy's control of the settlement instant** (fixed-time / lagged-quote / non-cancellable confirmation-window settlement) — EWMA shrinks the windfall but cannot remove a max-operator the policy times.
- **L1 lands, with a representation correction (both primes).** Land the canonical `ContributionState` + an EXPLICIT observation status/mask (`counterfactual_observed`/`frozen`/`age_norm`) reaching the POLICY — a bare `−1` sentinel is NOT self-describing (a true negative contribution also clips to −1). **HOLD invariant 2 (freeze-forever):** a value frozen at full magnitude with freshness→0 over 70 epochs is the mirror lie (H7's rising pre-commit slope says the frozen anchor is biased HIGH); **shrink dim-12 toward UNKNOWN as staleness rises** (not toward 0 = original bug, not frozen-forever = mirror bug).
- **"De-shape not survivable" SOFTENED (both):** reward-mass ≠ policy-gradient signal (PPO standardizes advantages). Correct: de-shape removes the dominant dense channel → not a drop-in remedy; and H8's real headline is that the **host-task signal was never strong enough to train on** — a finding about the OBJECTIVE (it needs an anchor), retained as a diagnostic control.
- **A cheaper causal read exists before full L3 (both):** the existing pinned-simplex |H|=1 RCT can test whether the pre-commit EWMA quote predicts **proxy-free product outcomes** (val-acc@1/5/10/25, terminal acc, AUC, compute-to-target, destructive events) via a CATE regression (`β3 = FOSSILIZE × q_t`), episode-clustered — zero GPU. claude-prime adds: build a **terminal host-level objective** (final acc vs no-morphogenesis control, one scalar/episode) as the anchor to *validate* the 95% proxy. Full L3 branching likely still needed for shipping, but not first.
- **Diagnosis nuance (gpt-prime):** it is NOT one `None→0` producing all three — the OBSERVATION is a `None→0` coercion; the REWARD has a SEPARATE explicit provisional-only gate (`seed_contribution is not None and not seed_is_fossilized`). Two mechanisms, one root gap. The critic is the LEADING explanation, not proven ("perfectly fit" not banked; re-test after L1/L2).
- **ESCROW guard: kept (both).** Hardening (future): `"configured"` is a caller assertion — before enabling ESCROW in a real fossilize-capable config, require a concrete validated settlement-strategy, not a string.
- **K>1: still open, P0, five rounds — do it (both). Independent; every run is contaminated until fixed.**

---

## 1. The verdict in one paragraph

One measurement gap — a fossilized seed's counterfactual (LOO) is **structurally undefined**, because fossils are deliberately excluded from the ablation (disabling a baked-in seed measures *host damage*, not contribution) — produces **THREE coupled symptoms** via `None→0` coercions: the reward stops paying (`bounded_attribution` → the realized −113/−102 FOSS−SET_ALPHA return forfeit), the observation reads a fossil's contribution as 0, and the critic "overstates" the commit penalty because it is fit to that lying observation. Separately, the `hindsight_credit` scaffold channel is inert for a **different** reason (corrects our prior "designed settlement" framing). The fix is **L1 freeze-don't-zero** (observation carries the last-valid value + rising uncertainty; no guessed decay) **+ L2 settle-then-annuitize** (reward: an EWMA-priced annuity replacing the forfeited stream). It is **training-legible but not product-valid without L3** (the last-valid LOO = retained value is an *assumption*; drift is unmeasurable offline). **De-shape is not survivable** (the cf stream is ~95% of all positive reward). The premise *"under-fossilization is a defect"* remains **UNMEASURED, not wrong** — L1/L2 make it testable for the first time.

---

## 2. The defect, in verbatim code

### 2a. The root: fossils are excluded from the counterfactual ablation (`vectorized_trainer.py:956-976`)
```python
# CRITICAL: Exclude FOSSILIZED seeds from ablation in most modes.
# Fossilized seeds are permanently integrated - disabling them
# measures damage to the host, not the seed's contribution.
# The host was trained WITH the fossilized seed's output;
# suddenly removing it causes catastrophic accuracy drops.
#
# EXCEPTION: In BASIC_PLUS mode (drip_fraction > 0), we need
# contribution data for fossilized seeds to compute drip payouts.
active_slot_list = []
for sid in slots:
    if not model.has_active_seed_in_slot(sid):
        continue
    slot = cast(SeedSlotProtocol, model.seed_slots[sid])
    if slot.state is None or slot.alpha <= 0:
        continue
    if slot.state.stage == SeedStage.FOSSILIZED:
        if self.reward_config.drip_fraction == 0.0:   # SHAPED run: drip=0 → fossils SKIPPED
            continue
    active_slot_list.append(sid)
```
The rationale is **correct** (the naive ablation is biased — it measures host damage). The defect is the *downstream response to the resulting `None`*: substituting **zero** instead of representing "unmeasured."

### 2b. Symptom 1 — reward: `bounded_attribution` gates out fossils (`contribution.py`, SHAPED path)
`seed_is_fossilized = seed_info.stage == STAGE_FOSSILIZED` (line 463). In SHAPED mode the per-epoch cf credit is computed **only** under `if seed_contribution is not None and not seed_is_fossilized:` (line 569) — a fossil takes the `bounded_attribution = 0.0` branch. Realized effect (ground-truth MC on r9): FOSS−SET_ALPHA return contrast **−113/−102**, ~99% this stream.

### 2c. Symptom 2 — observation: `None → 0.0` in the contribution feature (`features.py:827-832`)
```python
# Improvement (1 dim) - causal counterfactual contribution only
contribution = report.metrics.counterfactual_contribution
if contribution is None:
    contribution = 0.0                       # ← the silent lie: "unmeasured" becomes "zero"
contribution_norm = max(-1.0, min(contribution / _IMPROVEMENT_CLAMP_PCT_PTS, 1.0))
obs[env_idx, slot_offset + 12] = contribution_norm
```
The **freshness** channel that could say "stale, not zero" already exists (`features.py:895-899`, dim 29 = `γ^epochs_since_counterfactual`, with an `OBS_V3_UNKNOWN_SENTINEL(−1.0)` for never-measured). The value dim (12) collapses never/stale/fresh all to 0.0; the freshness dim (29) already distinguishes them. **That asymmetry is the entire fixable gap.**

### 2d. Symptom 3 — critic: fit to the lie (no separate code; a consequence of 2c)
The main critic sees a fossil's contribution feature drop from "+c" to 0 and correctly predicts accuracy damage that never materializes → its measured 4–6× "overstatement" of the FOSS main-stream penalty (V_main −6.6/−5.4 vs realized −1.0/−1.2) is a **symptom of 2c**, not an independent defect. **Retire the critic as an independent line** (PDR-0066 "critic is the wrong lever" was right, for this reason).

### 2e. NOT a symptom — `hindsight_credit` is a scaffold channel (H3, corrects our prior framing)
`compute_scaffold_hindsight_credit(boost_given, beneficiary_improvement) = tanh(boost·benef·0.1)·0.1`, ×`γ^delay`, summed over `scaffold_boost_ledger` entries for this beneficiary, capped at `MAX_HINDSIGHT_CREDIT=0.2`. It keys on `beneficiary_improvement = total_improvement` (**host-drift-confounded**, not the counterfactual) and pays *other* (scaffold) seeds. It **cannot** settle a fossilizing seed's own −113 forfeit (~500× too small, wrong input), and is inert for a *different* reason (empty ledger / total_improvement). **So `hindsight_credit`-inert is a SEPARATE defect and L2 is a NEW channel, not a reactivation.**

---

## 3. Read results (all zero-GPU, from r9 + the real reward fn)

| Read | Result |
|---|---|
| **H3** hindsight | scaffold channel, cap 0.2, keyed on `total_improvement` — cannot settle the −113; a separate defect (see §2e) |
| **H4** escrow | current `[3,3,3,None]/[H,H,H,FOSS]` path **FREEZES** (no clawback; terminal forfeit exempts FOSSILIZED). Clawback is **LATENT** — fires only when a fossil is *measured* negative (drip>0 / future L3) with no settlement → guard stays a construction-raise, **not** a hard error |
| **H5** provenance | ONE source (fossil ablation exclusion) → coupled sinks: reward gate, obs dim-12 `None→0`, freshness dim-29, fossilize/anti-gaming gates, critic |
| **H7** drift | pre-commit anchor **usable** (last-valid LOO median ~12; slope +0.24/+0.28, rising; **sign stable 86–87%**; mild sell-at-spike bias; age@FOSS ~74–79/150). Post-commit **true drift UNMEASURABLE offline** (it is the excluded quantity) → **freeze the value, raise uncertainty, do NOT bake a decay-γ; L3 for true drift** |
| **H8** scale | `bounded_attribution` = **~95% of positive reward, >100% of net return** → **de-shape not survivable**; the fix must preserve+correct the cf channel, never remove it |

---

## 4. The specs of the change

### 4a. Canonical contribution-state (one source; replaces three independent `None→0` coercions)
```python
class CounterfactualStatus(Enum):          # → leyline
    NEVER_MEASURED = "never_measured"      # birth / TRAINING pre-ablation → UNKNOWN sentinel, NOT 0
    FRESH          = "fresh"               # measured this epoch (incl. a true measured 0.0)
    STALE          = "stale"               # last_valid held, uncertainty rising
    FROZEN_AT_FOSSILIZE = "frozen"         # captured at commit; excluded from ablation thereafter

@dataclass(slots=True)
class ContributionState:
    seed_generation_id: int                              # lifecycle identity (clears on germinate/prune/reuse)
    counterfactual_status: CounterfactualStatus
    last_valid_counterfactual_contribution: float | None # None IFF NEVER_MEASURED
    epochs_since_counterfactual: int                     # 0 when FRESH; rises when STALE/FROZEN
    measured_this_epoch: bool
```
Consumed identically by observation (value dim + freshness dim), reward (`bounded_attribution`/settlement), and settlement (L2).

### 4b. L1 — freeze-don't-zero (observation), invariants
1. FOSSILIZE preserves `last_valid` (captured at the commit instant, staleness=0), status→FROZEN — **not 0**.
2. Freshness signals **rising uncertainty**; the **value is held constant — no decay-γ baked in** (H7: true drift unmeasurable). Obs distinguishes "uncertain but last-known +c" from "measured 0."
3. Birth/TRAINING-no-measurement = NEVER_MEASURED = **UNKNOWN sentinel in the value dim** (extend the sentinel already used for freshness dim-29 to value dim-12) — not measured-0.
4. A true measured 0 stays distinguishable from absence (FRESH+0.0 ≠ NEVER_MEASURED sentinel).
5. germinate/prune/slot-reuse **CLEAR** the cached `last_valid` via `seed_generation_id` — else the carry-forward artifact reappears *inside the live policy*.
6. **Obs schema bump v3→v4**: dim-12 semantics change ⟹ the LSTM was trained on the old semantics ⟹ **re-warm/retrain required**, not a silent swap.
7. Telemetry: log raw-optional + emitted value + status + epochs_since + seed_generation_id.

### 4c. L2 — settle-then-annuitize (reward), invariant `G_stay_provisional ≈ G_fossilize_now`
At FOSSILIZE (precondition `staleness==0`): `S = c_settle · Σ_{k=1..H} γ^k`, with **`c_settle = EWMA/min-over-window` of the seed's per-epoch cf rate (NOT spot)**, `H = max_epochs − fossilize_epoch`, γ=0.995. **Disburse S as a FIXED per-epoch annuity**, schedule **frozen at fossilize, never re-measured** (so it never needs the invalid fossil ablation). The recurring `bounded_attribution` stays 0 for the fossil and is *replaced* by the annuity. Double-pay prevention: the annuity *replaces* the forfeited stream; the existing one-shot fossilize bonus is netted **down** (or kept as a small explicit commitment premium). Negative last_valid → settle 0; never_measured → S=0; too-stale → force a fresh ablation before commit or settle 0 (fail-closed).
- **A1 (one-shot lump) REJECTED:** S≈50–100 is a ~30–60× reward spike → PPO variance bomb.
- **B (freeze-and-continue: re-pay `last_valid·freshness` each epoch) REJECTED offline:** with freshness=1 (no measurable decay) it is an **uncapped perpetuity** → the 800:1 farm; erases the seat-cost signal. Viable only after L3 measures decay-γ.

### 4d. The ESCROW guard (LANDED — the one shippable item)
New fail-closed field + `__post_init__` raise on `contribution.py` (verbatim):
```python
escrow_fossil_settlement: Literal["unconfigured", "configured"] = "unconfigured"
# ... in __post_init__:
if self.reward_mode == RewardMode.ESCROW and self.escrow_fossil_settlement == "unconfigured":
    raise ValueError(
        "RewardMode.ESCROW is active but no permanent-contribution/settlement strategy is "
        "configured ... missing observability must NOT be interpreted as a measured loss of "
        "value ... Scoped to the current missing-input implementation; not a universal escrow ban."
    )
```
New test `tests/simic/rewards/escrow/test_escrow_fossil_guard.py` (guard raise/opt-in/scoping + the H4 freeze sequence + the latent measured-negative clawback). Isolated escrow-math harnesses set the opt-in (FOSSILIZE unreachable). **5 new tests + 545 reward/property/telemetry pass.**

---

## 5. The offline-replay FARM VERDICT (the load-bearing check)

Per fossilize event on r9: the SHAPED forfeited discounted cf stream (the per-event −113 mechanism) vs three settlement prices.

| quantity | seed 41 | seed 42 |
|---|---|---|
| forfeited discounted cf stream / FOSS (median; mean) | 50.1; 216.2 | 79.1; 241.3 |
| **A[spot]** net(lump−forfeit) median / **mean** | 0.0 / **+11.6** | 0.0 / **+9.5** |
| **A[ewma]** net median / **mean** | 0.0 / **+5.8** | 0.0 / **+4.9** |
| **A[min-window]** net median / **mean** | 0.0 / **−54.9** | 0.0 / **−60.5** |
| windfall exposure (spot−minwin) mean; frac>0 | +66.5; 0.49 | +69.9; 0.48 |
| sell-at-spike (spot rate > window mean) | 0.45 | 0.47 |

**Verdict:** every A pricing removes the −113 discontinuity (median net → 0). **A[spot] FARMS on the tail** (positive-mean overpayment; reject spot). **A[ewma] is VIABLE** (mean residual +5, no systematic peak-timing windfall) **with `staleness==0` + one-shot-bonus netting** — recommended. **A[min-window]** over-corrects (conservative floor). **B FARMS** (reject unless L3). **Necessary-not-sufficient confirmed:** EWMA/min-window + `staleness==0` collapse the +12→+5 mean windfall, but the policy still selects commit *timing* (sell-at-spike persists 45–47%) — so they are necessary, not sufficient.

---

## 6. What L1/L2 WOULD and WOULD NOT establish

**WOULD (training legibility):** remove the −113 reward discontinuity; stop the obs reading a fossil at 0; retire the critic "overstatement" as a symptom; make commitment **scoreable** so the |H|=1 RCT can finally be re-run with a working instrument; close the ESCROW latent-clawback class (guard landed).
**WOULD NOT (product validity — needs L3):** prove the imputed permanent value is causally correct (last_valid LOO = retained value is an *assumption*; drift unmeasurable offline); resolve whether **"under-fossilization is a defect"** (still UNMEASURED); prove fossilizing is optimal (A2 makes it *neutral* to staying, by construction; the floor still forces 93–94% of commits — coupled, downstream); measure the true decay-γ.

---

## 7. Open questions for you (red-team targets)

1. **The freeze-don't-decay resolution (§4b invariant 2).** We rejected baking a decay-γ into the value channel because H7 shows true drift is unmeasurable offline; instead the value is frozen and *uncertainty* rises. Is "frozen value + rising freshness" the right representation, or does a frozen value the policy can't discount become its own lie (an *over*-statement of permanence) that the critic will fit — the mirror of the bug we just fixed? Is there a better uncertainty encoding?
2. **L2 pricing / sell-at-spike (§5).** EWMA + `staleness==0` are necessary-not-sufficient (the timing selection persists at 45–47%). Is there a settlement rule that removes the *timing* incentive, not just the magnitude windfall — e.g. settle on a pre-commitment-time estimator the policy can't select, or a min-over-full-lifetime? What breaks?
3. **Is L3 truly required (§6, §10)?** L1/L2 buy legibility, not causality. Is there a cheaper causal read for "does pre-commit LOO predict retained value" than a matched-decision permanent-value instrument — e.g. a terminal-only host-level objective, or a shadow-host? Or is L3 unavoidable?
4. **De-shape (§3, H8).** The cf stream is ~95% of positive reward, so de-shape removes almost all trainable signal. Does that change your prior view that de-shaping was a candidate remedy — and does it imply the *whole* reward is a shaped proxy with almost no underlying host-task signal to fall back on?

---

## 8. Retractions carried in from prior rounds (so you're not reviewing stale claims)
RETRACTED: PDR-0069 "the gate passes"/"reward already correct"; PDR-0071 §4 "signal survives the 9-pt spread"; the round-9 "reward-specification / commitment-avoidance is optimal / vindicates the EV-stab epic"; "ESCROW as the fix"; and (this round) "`hindsight_credit`/escrow settlement is the designed L2" (H3: it is a scaffold channel). RETAINED: the floor dead-zone is a real, autograd-confirmed op-head gradient censor, but COUPLED (FOSSILIZE is floor-forced 93–94% → fixing the gradient alone drives fossilization ~0) and DEFERRED behind the instrument; K=1 leaves PPO with no operative trust-region guard (independent P0). Premise: *"under-fossilization is a defect"* is UNMEASURED, not wrong.
```
```
Product-workspace refs: PDR-0074 (diagnosis), PDR-0076 (approved refocus + conditions), `docs/analysis/2026-07-14-advantage-loo-read-preregistration.md` (the full Read-A→round-11 arc).
