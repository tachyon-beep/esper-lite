# Phase-2: non-selectable settlement implementation (flag-on-HOLDING, default-OFF) + executable B-path replay — REVISION 2

```yaml
status: ready-pending-re-review
created: 2026-07-14
revised: 2026-07-14 (rev 2 — closes drl plan-review B1–B5/N1–N5 + the W1 window-targeting finding + round-23 safety semantics)
owner_authorization: PDR-0090 (convergence endorsement; scope = default-off impl + tests + replay + drl review + docs)
NOT_in_scope: training-config activation; GPU use; obs changes (pending-visibility = OWNER FORK, see §F2)
spec: docs/analysis/2026-07-14-permanence-visible-preregistration.md §2 (rounds 15–21 + the §2.5.9 W1 amendment)
acceptance: §2.6 seven gates + the TEN replay areas (§A below, transcribed) + 8 construction invariants (§B) + round-22 pins as re-worded in N5
reviewed_by: [drl-expert plan review r1: REQUEST-CHANGES (B1–B5, N1–N5) — all addressed below; re-review pending]
tracker: esper-lite-57a56da421
owner_forks_open: [F1 abort-payment semantics, F2 pending-visibility/G-OBSERVABILITY]
```

## §0. Epoch-phase placement (B1 — the load-bearing pin; everything below assumes it)

The live pipeline per epoch (phase-1 finding, verified): **metrics phase** (`record_accuracy` ticks stage clocks;
counterfactual matrix computed; val_acc recorded) → **reward phase** (`compute_reward` on pre-action state,
action_execution.py:1089) → **dispatch phase** (mutations, :1292–1348; `seeds_fossilized` bumps here) →
**step_epoch** (:1643).

Pinned placements:
- **Request (epoch R):** the FOSSILIZE action routes to request-issuance at DISPATCH phase of R (after R's reward).
  The request-row reward is computed with the pre-action HOLDING state and `action==FOSSILIZE` (branch table §2).
  c_R was recorded at R's metrics phase, BEFORE the request existed → c_R belongs to `q_decision`, excluded from
  `q_settle`.
- **Window rows (R+1 … B−1):** seed HOLDING+committed; masking prevents targeting; the LEDGER pays the window
  provisional (§1/W1). Measurements c_{R+1}…c_{B−1} accumulate to the settlement window.
- **Boundary (epoch B):** `q_settle` locks at REWARD phase of B and INCLUDES c_B (recorded at B's metrics phase,
  before the lock — §2.3's "collected after the request, before the boundary [settlement act]"). Valid window =
  **[R+1, B]**, giving B−R measurements; the ≥5 rule requires **B ≥ R+5**, else extend to the next boundary
  (zero-slack §2.3). The boundary package (prior|−0.2 branch ÷γ^d with d = B−R, + annuity epoch 1) is paid at B's
  REWARD phase via the ledger adapter (summed into the env step reward per §2.5.7 — NOT keyed on the policy action,
  which is WAIT-elsewhere). **Provisional is explicitly suppressed at B** (the annuity replaces it from B inclusive
  — G-EVENT-ORDER: no epoch pays both, none pays neither). The HOLDING→FOSSILIZED transition executes at DISPATCH
  phase of B (after B's reward) — reusing the certified A-path mechanics.
- **Maintenance starts B+1** (`num_fossilized_seeds` bumps at B's dispatch → first charged B+1). This EQUALS arm
  A's semantics (commit at t_c → first `fossilized_rent` at t_c+1): G-MATCHED-CONTROL's maintenance leg asserts
  B+1 ≡ t_c+1 parity, resolving the "starts at the boundary" wording (B1d).
- ÷γ^d index convention: d = B − R (request row to boundary row, both reward-phase); pinned by unit test.

## §1. W1 rev 3 — THE SINGLE-OWNER PRINCIPLE: the ledger owns the committed seed's ENTIRE per-seed reward row

The reward is computed exactly once per env step for the single `target_slot`'s seed (loops at
`action_execution.py:817`/`:1005` are env/count-only — no per-seed reward loop; `:944-947` keys every per-seed
component on the target). Masking the pending slot therefore silences not just the provisional but ALL of its
per-seed components (pbrs_bonus `:721-726`, holding_warning `:706-719`, blending_warning, synergy) — and,
conversely, any targeting leak (see the WAIT gap below) double-pays them all. **Fix (re-review W1-A):**
- **For window epochs [R+1, B−1] the settlement ledger is the SOLE OWNER of the committed seed's entire
  per-seed reward row** — it reproduces, each epoch, exactly what an ALWAYS-TARGETED HOLDING seed would earn
  under the production law at that epoch's actual inputs: provisional (c_t fresh, or the contribution=None
  branches — proxy/zero — when unmeasured; NO carry, NO imputation — N-r2b), the within-HOLDING PBRS
  climb/drag, warning semantics per the §2 table (suppressed for committed), synergy at the seed's inputs.
  Summed into the env step reward.
- **Defense-in-depth reward-path guard:** the policy path suppresses ALL per-seed components when
  `seed_info.committed` — so even the masking edge cases cannot double-pay. Named leak vectors the guard
  closes (re-review): `slot_by_op[LifecycleOp.WAIT]` is a SEPARATE tensor from the morphology masks
  (action_masks.py:229-238) and would otherwise still admit the committed slot as a WAIT target; and the
  single-active-seed fallback (`_resolve_target_slot` → index 0, vectorized.py:151).
- **Cadence reconciliation (re-review W1-B):** Δ_designed(x) for G-PBRS/G-DESIGNED-DELTA is computed by the
  phase-1 harness under ALWAYS-TARGETED cadence — the ledger enforces exactly that cadence for the committed
  seed, so the real-run B−A delta matches the harness Δ by construction. The replay's B path models the
  LEDGER-driven per-seed row, never a policy-targeted one. (Without this, the window PBRS alone misses by
  ~0.078/epoch × d ≈ 0.4–0.8 against the |Δ| ≤ ~0.14 harness scale.)
- Estimand + telemetry: the deterministic every-epoch cadence (provisional AND pbrs) vs S's when-targeted
  cadence is a real protocol-package component — labeled alongside the Option-A lock and included in the
  lock-burden/aliasing telemetry (N-r2c). Negative c flows every window epoch — the hatch stays closed.

## §2. Request-row branch table (B2 — which `action==FOSSILIZE` branches apply)

| Code site | At the request row | Rationale |
|---|---|---|
| fossilize action_shaping (contribution.py:824–863: base+0.1·c+tanh, penalties −1.0/−0.5/ransomware/−0.2) | **SUPPRESSED**; only `fossilize_cost=−0.01` charges | sterile instant, §2.5.4 |
| cf<0 attribution zeroing (:644–654) | **DISABLED** — the row pays provisional as a continuing-HOLDING row (negative carry flows) | otherwise the request forgives negative carry for free = audit-F8 at the request row |
| holding_warning terminal exemption (:709) | **KEPT** — no warning at the request row | the request ends indecision (§2.5.9; the +27.7 relief leg of G-MATCHED-CONTROL) |

N1 made explicit: the request row PAYS provisional normally (bounded_attribution stream) + `fossilize_cost` in
action_shaping — two different code regions; G-EVENT-ORDER's "request epoch pays provisional once + −0.01".

## §3. The boundary package + T_annuity enumeration (B3, N2)

At B (reward phase), from the locked `q_settle` and boundary-frozen inputs:
- **Branch:** `q_settle ≥ 1.0` (or LCB form per config) → prior = `(0.5 + 3.0*math.tanh(1.0/3.0)) ×
  legitimacy_request ÷ γ^d`; else → `−0.2 ÷ γ^d`. **The −0.5-damage/−0.3-ransomware request-instant branches are
  intentionally DROPPED at the boundary** (N2): the harmful-seed case is carried by the SIGNED annuity — a negative
  `q_settle` pays negative every epoch to horizon, strictly stronger than a one-shot −0.5. The replay asserts the
  F8 case (bleeding seed: B's total ≤ S's; no free escape).
- **T_annuity ≡ T_provisional at boundary-frozen inputs (enumerated):** annuity/epoch = `contribution_weight ×
  attributed(c:=q_settle; progress:=progress_B) × attribution_discount(cf_B) × timing_discount(germination_epoch)`
  where `progress_B = val_acc_B − acc_at_germination`, `cf_B` = counterfactual at B, and the attributed() branches
  are the shipping ones (progress-gating; sqrt/formula cap; negative passthrough for q_settle<0). **ratio_penalty is
  EXCLUDED** (it is the spot-spike anti-gaming term; the spot channel is structurally closed by the two-quote
  design; including it would double-penalize a smoothed, policy-unselectable statistic). Computed ONCE at B; paid
  per epoch [B, horizon]; truncated by construction.
- G-CONTINUITY target defined (B3): constant c AND constant val_acc synthetic → provisional/epoch ==
  annuity/epoch exactly; asserted at 1e-6.
- **Transform-law consistency at B (N-r2a, documented resolution):** the WINDOW keeps the FULL live
  provisional law INCLUDING ratio_penalty — identical to S's law, the strongest B−S continuity property (the
  window is provisional in kind; a live spike is paid for B exactly as it would be for S). The ANNUITY excludes
  ratio_penalty (§3 rationale: its input is a smoothed, policy-unselectable statistic — the spot-spike detector
  has no referent). The transform change at B tracks the input regime change (live c_t → frozen q_settle) and
  is priced in Δ_designed at B, not hidden.
- **Δ_designed cadence statement (W1-B):** all Δ_designed(x) values entering G-PBRS and G-DESIGNED-DELTA are
  computed under the always-targeted cadence the §1 ledger enforces; the replay's B path models the ledger row.

## §4. Safety and governor semantics (B4 + round-23; two classes — converged content adopted)

- **Ordinary governor / resource-management morphology** (scheduled prunes, auto-advance, non-emergency paths):
  **SUSPENDED by the Option-A lock** during any open window — the coherent reading of the ratified lock (any op
  that moves the ensemble-relative quote breaks the cross-slot construction, whoever issues it).
- **Hard safety interventions** (governor panic/rollback, invalid-state): **OVERRIDE commitment — safety always
  wins.** Executes immediately; emits `FOSSILIZE_ABORTED_BY_SAFETY`; releases the global lock; preserves
  generation-ID integrity; lock-burden + abort telemetry mandatory; a material abort rate is reported as a
  protocol failure, never silently censored.
- **OWNER FORK F1 — abort PAYMENT semantics (primes diverge; both implemented behind config, replay tests both):**
  (a) claude-prime: settle at the scheduled boundary on q_settle-as-measured-so-far (liability preserved — an
  abort must not erase a signed liability; SETTLEMENT-INTERRUPTED outcome class, censored from P2/gate cohorts by
  pre-registration); (b) gpt-prime: pay NO prior and NO annuity; the request stays in intention-to-treat P2 and
  P1; abort reported as a guardrail event. Owner picks before freeze.
- The uncancellability assert is SCOPED to `initiator=="policy"` (N3): freeze-by-not-scheduling (mask SET_ALPHA;
  skip controller stepping for committed seeds); no hard assert on non-policy `set_alpha` paths.

## §5. Fail-closed + OFF-identity (B5)

- **Flag OFF ≡ shipping, replay-asserted:** the phase-1 goldens ARE the pre-flag record — the entire phase-1 suite
  passing on the flag-off path is the byte-identity proof for the reward stream; plus a behavioral sweep: flag ON
  with `committed` forced False everywhere ⇒ reward stream identical to OFF (stronger than the grep sweep, which
  is kept as a fast guard).
- **SeedState serialization:** the new fields (`committed`, `settlement_boundary_epoch`, `legitimacy_at_request`)
  enter `to_dict`/`from_dict` unconditionally (no-legacy: required keys, all call sites updated; pre-change
  checkpoints are incompatible — documented breakage per house policy). Mid-window save/load round-trip test
  (ledger + SeedState) required.
- Construction-raises: settlement config with `drip_fraction > 0`; with `RewardMode.ESCROW`; activation via a
  validated `FossilSettlementConfig` OBJECT only (escrow-precedent fail-closed, contribution.py:167).

## §6. Masking + validity (N4 — both sides of BUG-020, enumerated)

| New rule | Mask side (action_masks.py) | Validity side (must agree) |
|---|---|---|
| Pending slot off the slot head | slot-head legal-set excludes committed slots (builder learns `committed`) — **including the SEPARATE `slot_by_op[WAIT]` tensor (:229-238) and the single-active-seed fallback (`_resolve_target_slot` → index 0)** | `_parse_sampled_action` validity + handler `can_fossilize` (fossilize.py:157) reject committed targets; **plus the §1 reward-path guard (suppress per-seed components when `seed_info.committed`) as the backstop that makes any residual leak double-pay impossible** |
| Option-A global suspension | morphology ops masked for ALL slots while any window is open (global signal threaded into the builder) | `_preflight_lifecycle_mutation` rejects morphology ops while locked; ordinary-governor paths check the lock (§4) |
| Late-request mask | FOSSILIZE masked when no boundary ≥ R+min_window with ≥5 achievable measurements before horizon | request-issuance validity re-checks the same predicate |
| FRESH-required | FOSSILIZE masked for STALE/NEVER_MEASURED seeds (§2.5.3) | request-issuance validity re-checks |

## §A. The TEN replay acceptance areas (transcribed — was un-enumerated; reviewer gap)

1. **Pending-observation/value-head legibility (G-OBSERVABILITY)** — **OWNER FORK F2**: gpt-prime requires a
   source-level observability proof (is `committed` in the obs? are masks network inputs? is epoch/cadence visible
   enough to derive time-to-boundary? is the previous action in the recurrent input?) and, failing it, two
   common-schema fields for ALL arms (`fossilization_pending` 0/1 + normalized countdown; arm A emits zeros;
   explicit obs-schema version). claude-prime endorses invisibility + instrumented aliasing telemetry + a
   pre-registered SYMMETRY CHECK (critic error concentrated in open-window epochs in arm B only; if it appears in
   A it is not aliasing). **The source-level observability proof runs REGARDLESS (queued task) — both resolutions
   need its facts.** Until the owner rules, the plan implements NO obs change and ships the telemetry + symmetry
   check.
2. Active-slot/ablation continuity (committed seed stays in `active_slot_list`, measured every epoch).
3. Exactly one HOLDING→FOSSILIZED transition, at B — and G-PBRS asserts the **harness per-scenario Δ** (N5: the
   re-accrual is a spread climb, not a discrete mint event; PDR-0088 #2's form; round-22 pin 3a's intent honored —
   the boundary transition is the ONLY stage entry). **Δ_designed is computed under the §1 ledger's
   always-targeted cadence; the B-path replay models the ledger-driven per-seed row (W1-B).**
4. No unenumerated `committed=True` effects (grep sweep + the §5 behavioral sweep).
5. Signed contribution continuity (W1 ledger provisional + §3 annuity; negative carry never escapes: request row,
   window, boundary, post-boundary all asserted).
6. Event ordering (§0 placements; half-open intervals; no epoch pays both provisional and annuity, none neither).
7. Cross-slot lock (adversarial neighbor-op sequences; ordinary-governor suspension included).
8. Threshold variance (configured gate form raw|lcb from the noise read).
9. Late-request masking (+ estimand stratification note per D5).
10. Generation-ID isolation (ledger keyed by generation; reuse cannot inherit pending state or annuity).

## §B. The 8 construction invariants (gpt round-23, transcribed)

No `epochs_in_stage` reset at request · seed remains active+ablatable · α and manipulable controls freeze at
request · request irreversible BY POLICY (safety per §4) · pending slot leaves the policy's morphology target set ·
holding_warning suppressed · exactly one HOLDING→FOSSILIZED transition at B · fossil re-accrual begins only
post-boundary (asserted via harness Δ per §A.3).

## Touchpoints (build order — unchanged from rev 1 except as amended above)
1. Leyline contracts (FossilSettlementConfig + gate_form + abort_payment_mode enum; FOSSILIZE_REQUESTED /
   FOSSILIZE_SETTLED / FOSSILIZE_ABORTED_BY_SAFETY events; settlement components in RewardComponentsTelemetry;
   positive settlement/obs-schema stamps in TRAINING_STARTED).
2. Kasmina: committed fields + policy-scoped uncancellability + serialization (§5).
3. `settlement.py` pure ledger (§0 placements; W1 window provisional; §3 boundary package; §4 abort branches).
4. Action layer: request routing; §6 mask/validity table; lock-burden counters.
5. Reward path: §2 branch table; ledger adapter summation; aliasing telemetry + symmetry check.
6. Replay B path: §A areas + §B invariants + both F1 branches.
7. Docs + pre-reg annotations.

## Review + landing discipline
Re-review of THIS revision before build (same reviewer). Build flag-off; suites green; drl code review before
commit; lock-burden + aliasing telemetry verified in replay. No training activation anywhere in scope.
```
