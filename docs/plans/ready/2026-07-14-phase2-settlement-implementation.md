# Phase-2: non-selectable settlement implementation (flag-on-HOLDING, default-OFF) + executable B-path replay

```yaml
status: ready-pending-review
created: 2026-07-14
owner_authorization: PDR-0090 (convergence endorsement; scope = gpt-prime item 10)
scope: default-off implementation + tests + executable replay + drl review + docs
NOT_in_scope: training-config activation; GPU use; any obs/observation change (pending-visibility fork is OWNER-OPEN, PDR-0090 #6)
spec: docs/analysis/2026-07-14-permanence-visible-preregistration.md §2 (rounds 15–21 integrated)
acceptance: the seven replay gates (§2.6) + gpt-prime's ten Phase-2 invariants + round-22 pins (PDR-0089 #3)
reviewed_by: []   # drl-expert review REQUIRED before build (CLAUDE.md)
tracker: esper-lite-<phase2-task>
```

## Design constraints (non-negotiable, from the frozen-track spec)

1. **PENDING is a FLAG on HOLDING** (PDR-0088/0090): `committed=True` on `SeedState`; NO stage change at
   request; NO `epochs_in_stage` reset; the seed stays in `active_slot_list` (measured every epoch) mechanically.
2. **Request instant is STERILE** (§2.5.4): only `fossilize_cost=−0.01` charges; `legitimacy_request` frozen;
   NO contribution-conditioned payment of any sign (the `0.1·c` grading, spot-c gate, −0.2 branch, cf-graded
   penalties all move to the boundary keyed on `q_settle`).
3. **Boundary is schedule-determined** (§2.2): global cadence every `W_settle=10` epochs; settle at the first
   boundary ≥ request + `min_window=5` WITH ≥5 valid post-request measurements, else extend to the next boundary
   (§2.3 zero-slack rule). Requests too late for window+horizon are MASKED (§2.5.6), never terminal-settled.
4. **Two quotes** (§2.3): `q_decision` (lagged EWMA, analyst-side, NO payment keys on it) vs `q_settle`
   (span-5 EWMA over post-request measurements, ≥5 valid, locked at the boundary — the payout basis).
5. **Boundary package** (§2.4): fixed prior = the EXPRESSION `0.5 + 3.0*math.tanh(1.0/3.0)` × `legitimacy_request`,
   **discount-neutral ÷γ^d** (d = request→boundary delay; index convention unit-tested), gated on
   `q_settle ≥ 1.0` (gate form raw|lcb per config — the noise read decides; LCB = q̃ − z·s/√n, z=1); the
   sign-symmetric −0.2 branch also boundary-paid and ÷γ^d; signed annuity begins (rate `q_settle`/epoch to
   horizon, `T_annuity ≡ T_provisional` — identical sign/clip/eligibility, §2.5.2: negative q_settle → negative
   annuity; `None` never becomes 0); maintenance starts at the boundary.
6. **Option-A global audit lock** (§2.5.7, ratified conditionally): ONE pending fossilization at a time;
   policy-controlled morphology ops (GERMINATE/PRUNE/SET_ALPHA/second FOSSILIZE) suspended request→boundary via
   masking; host training + validation continue. Lock-burden telemetry REQUIRED (locked-epoch fraction,
   suppressed-op counts by type, quote-stratum overlap) — quantified before freeze.
7. **Fail-closed flag discipline** (the `escrow_fossil_settlement` precedent, contribution.py:167): the feature
   activates only via a validated `FossilSettlementConfig` OBJECT; construction-raise if combined with
   `drip_fraction > 0` (F-drip mode fence) or with `RewardMode.ESCROW`.
8. **No observation changes.** The pending-visibility fork is owner-open; Phase 2 instruments the aliasing cost
   (value-error during open windows) as TELEMETRY only.

## Touchpoints (build order)

### 1. Leyline (contracts first)
- `FossilSettlementConfig` dataclass: `w_settle=10`, `min_window=5`, `ewma_span=5`, `min_measurements=5`,
  `q_threshold=1.0`, `gate_form: Literal["raw","lcb"]`, `lcb_z=1.0`; the premium as a computed property
  evaluating the expression in-code (PDR-0084 #2 — never a hand-rounded literal).
- Telemetry: `FOSSILIZE_REQUESTED` / `FOSSILIZE_SETTLED` event types + payloads (§2.7/B10): request epoch,
  legitimacy_request, boundary epoch, d, q_settle, branch (prior|noncontributing), measurements count,
  extension count; lock-burden counters. New `RewardComponentsTelemetry` fields: `settlement_prior`,
  `settlement_annuity` (leyline/telemetry_contracts.py:41).
- Protocol version stamp for the settlement (§2.7) + a positive `settlement_enabled` config stamp in
  TRAINING_STARTED (rides the TIP observation about positive stamps).

### 2. Kasmina (`src/esper/kasmina/slot.py`)
- `SeedState.committed: bool = False`, `settlement_boundary_epoch: int | None`, `legitimacy_at_request:
  float | None`. At request: freeze the alpha controller (α, target, speed, curve — no mutations while
  committed; assert in `set_alpha`/controller entry points), set the fields. NO transition call.
- At settlement: the existing fossilize path executes (`advance_stage`/`transition(FOSSILIZED)`) — reuses the
  certified A-path mechanics (exactly one stage-entry mint, at the boundary — round-22 pin 3a).
- Guard: `prune()`/`advance_stage()` on a committed seed raises (uncancellable, §2.1) — masking is primary,
  this is the defensive assert.

### 3. Settlement ledger (`src/esper/simic/rewards/settlement.py`, NEW, pure functions)
- Per-slot ledger keyed by `seed_generation_id` (§2.5.5): request record (epoch, legitimacy, generation),
  post-request measurement accumulator, boundary resolution (`next_boundary(request_epoch)` +
  extension logic), `q_settle` EWMA (span 5, adjust=False — same convention as the phase-1 formula test,
  frozen per PDR-0089), gate evaluation (raw|lcb), boundary package computation (prior ÷γ^d both branches),
  per-epoch annuity stream to horizon.
- Pure-function core so the replay drives it directly; trainer integration is a thin adapter.

### 4. Action layer (`src/esper/simic/training/action_execution.py`, `handlers/fossilize.py`,
`src/esper/tamiyo/policy/action_masks.py`)
- Flag ON + op==FOSSILIZE on eligible HOLDING seed → request issuance (re-interpret FOSSILIZE, no schema bump —
  §2.1/S1): validate FRESH staleness (STALE/NEVER → masked/invalid, §2.5.3 — never zero-priced), record request,
  reward sees the sterile instant.
- Masking: pending slot leaves the slot head's legal set (PRIMARY suppression, §2.5.9); Option-A lock masks
  morphology ops globally while a window is open; late requests masked (§2.5.6: insufficient time for
  min_window + ≥5 measurements before horizon). Mask math must mirror the legality predicates exactly
  (BUG-020 precedent: mask and validity must agree).
- Lock-burden counters emitted per epoch while locked.

### 5. Reward path (`src/esper/simic/rewards/contribution.py`, flag-gated branch)
- Request row: suppress `_contribution_fossilize_shaping` + tanh bonus + `0.1·c`; charge only `fossilize_cost`.
- Window rows: provisional stream runs mechanically (seed is HOLDING + measured); `holding_warning` suppressed
  for `committed=True` (defensive assert — masking should make the state unreachable; the replay asserts it).
- Boundary row: `settlement_prior` (or the −0.2 branch) ÷γ^d + annuity begins + maintenance starts.
- Post-boundary rows: `settlement_annuity` = `T_provisional(q_settle)` per epoch (fossils' `bounded_attribution`
  stays 0 — the annuity is a NEW component, never additive with provisional for the same epoch: G-EVENT-ORDER).
- PBRS: NO new potential, NO stage change at request → nothing to do (the flag construction's whole point).

### 6. Executable B-path replay (extend `tests/simic/rewards/replay/`)
- `PathScenario` kind "B": request at dwell r, window d (with extension cases), driving the REAL flagged
  production path (compute_reward + settlement module), plus S and A comparators.
- The seven gates as tests (§2.6): G-CONTINUITY (B−S contribution-stream ≈0 at 1e-6, +/0/− constant c);
  G-MATCHED-CONTROL (B−A: prior PV, both branches, cost, warning-cessation parity, PBRS semantics, maintenance
  start); G-DESIGNED-DELTA (residual = Δ_obs − Δ_designed(x) ≈ 0; Δ_designed's PBRS term from the phase-1
  harness values); G-PBRS (**exactly one STAGE-ENTRY MINT, at the boundary** — round-22 pin 3a wording; the
  legitimate pre-cap HOLDING climb during the window must NOT fire it); G-EVENT-ORDER (half-open; no epoch pays
  both provisional and annuity, none pays neither); G-CROSS-SLOT (adversarial neighbor-op sequences under the
  lock); G-THRESHOLD-VARIANCE (already built; wire the configured gate form).
- Layer fence (round-22 pin 3b): the B≡A-at-boundary identity is asserted for the PBRS layer ONLY; the
  contribution and warning layers get their own explicit B−A/B−S expectations.
- Case matrix: A6 dimensions incl. missed-measurement extension, late-request mask, terminal-edge,
  generation-turnover, multi-slot adversarial.
- **No-unenumerated-effects sweep:** a test greps/inspects that `committed` is read ONLY at the enumerated
  sites (masking, warning suppression, controller freeze, settlement module, telemetry) — the PDR-0088
  reversal-trigger guard.

### 7. Docs
- Pre-reg §2.1/§2.6 implementation-status annotations (design → implemented-behind-flag, default-OFF).
- README flag documentation. This plan moves to completed/ when landed.

## Review + landing discipline
drl-expert plan review BEFORE build (this document). Build strictly flag-off; full suite + replay green;
drl-expert code review before commit (the arc's reviewed-commit pattern); lock-burden + aliasing telemetry
verified in the replay. NO training-config activation anywhere in this plan's scope.
```
