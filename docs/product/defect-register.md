# Defect & Pathology Register — Esper        Updated: 2026-07-14 evening (execution pass, checkpoint #75)

**A-gate status:** A2 ✅ DONE (`1df249ab` — transport fixed, dim pinned, bug esper-lite-f0a82adccb CLOSED with
fix_verification; class-closers landed: transport-completeness + dim-liveness tests). A3 ✅ static half done
(the liveness sweep with its documented not-yet-varied ledger) / runtime normalizer proof-packet still open on
esper-lite-c739c3ab97. A4 🔄 coverage-report agent in flight. A1+A5 🔄 settlement build: increments 1–2 of 6
done (`3b36aa56` pure core, `89831e8a` committed-flag state machine); next = action layer → reward adapter →
F2 obs bit → replay B-path → drl pre-commit review.

> The standing "what must we fix" ledger, owner-requested 2026-07-14. Split by WHEN each item must be
> addressed. Tracker IDs are authoritative for status; this file is the map. Nothing here is fixed
> silently mid-arc — every fix routes through its stated gate (matched-control discipline).

## A. GATES — must land BEFORE the next experiment runs

| # | Item | Tracker | Disposition |
|---|---|---|---|
| A1 | **F2 pending-bit obs increment** with the full landmine checklist: lockstep `OBS_V4_SLOT_FEATURE_SIZE` 35→36 (else slot-1 collides with slot-0's new dim at flat index 59, silently), explicit obs-schema version bump (forces the desired normalizer re-warm), SAME schema both arms (arm A hardwires 0), `to_leyline` copy line for any new kasmina-side field | esper-lite-57a56da421 (phase-2 build) | Owner-ruled (PDR-0098); part of build increment 4/5 |
| A2 | **Velocity transport fix** — add `contribution_velocity` to `SeedMetrics.to_leyline()` (slot.py:301) + a transport-completeness unit test (non-default value survives trainer→kasmina→leyline→encoder) + telemetry carries the TRUE value; **the OBS keeps dim +13 = literal 0.0 in V3/V4** (byte-identity replay-asserted) — live only at Obs V5 | esper-lite-f0a82adccb | Round-24 adjudicated (PDR-0100): gpt's transport-now + claude's obs-defer, no-legacy form (no version branch — a literal constant in the V3/V4 encoder, documented as schema semantics) |
| A3 | **Dead-dim sweep as a standing pre-run proof-packet item**: per-dim normalizer mean/var, exact-zero fraction, sentinel rate, clip rate, first/last nonzero round → closes the silent-default CLASS (three instances found to date: reward None→0, obs None→0, leyline omitted-field→0) | esper-lite-c739c3ab97 | Both primes converged; run once now (may find more velocity-class dims), then per-run |
| A4 | **Reward–obs coverage report** (one-time doc): classify every reward + heuristic input as observed / inferable-from-history / intentionally-hidden / accidentally-unplumbed | esper-lite-c739c3ab97 | SHOULD before freeze (not a hard gate); makes the "policy optimizes signals it cannot see" set explicit |
| A5 | The settlement build + B-path replay + blinded certification + adversarial pass | esper-lite-57a56da421 | The existing frozen-track sequence (PDR-0097/0098) |

## B. FIX AT OBS V5 — one bounded post-experiment schema batch (admission-gated: D2-cleared OR verified plumbing defect OR semantic correction; every low-cadence signal ships as value/observed/age triple — never "unmeasured means zero")

| Item | Why |
|---|---|
| Activate `contribution_velocity` (dim +13 goes live) | The A2 transport fix makes it real; V5 exposes it |
| `committed_val_acc` as an obs feature | The committed-capability signal (ALSO: pre-registered as an Exp-1 SECONDARY OUTCOME analysis-side NOW — PDR-0100) |
| `plateau_epochs` (+ learning-curve phase) | The heuristic's exact germination trigger; policy blind to it |
| Train−val generalization gap | Overfit-vs-undercapacity indistinguishable today |
| **Occupancy decomposition** — `num_provisional_holding` / `num_fossilized` / `num_pending` / `num_occupied`; `num_holding` keeps its (misnamed) frozen semantics, then deprecates | Bug esper-lite-e0f03d7800; NEVER redefine a live dim (both primes converged) |
| D2-cleared per-site features (InjectionSpec statics free; slot-local grad aggregates) | GERMINATE is the worst-served decision; gated on the D2 information-ceiling read |
| Generated feature layout (named blocks, no hand offsets, layout hash) + exhaustive transport test | gpt §8; kills the A1/A2 defect classes permanently |

## C. DEFERRED-PRICED defects (decision anchored; do NOT hot-fix)

| Item | Tracker | Anchor |
|---|---|---|
| `eis==0` PBRS branch unreachable (cross-stage deltas never pay live) | esper-lite-5aec86160a | Priced for the experiment (PDR-0085/0090); post-experiment = fix vs formally adopt skip-semantics |
| F8 negative-carry escape hatch (live in shipping/arm A) | — | Closed BY DESIGN in arm B (signed annuity); arm A keeps shipping semantics deliberately (matched control) |
| `num_holding` conflation | esper-lite-e0f03d7800 | Frozen until V5 (B above) |

## D. TRAINING PATHOLOGIES (not code bugs — addressed by experiments/programme, in stage order)

| Pathology | Primary cause (verdict) | Addressed by |
|---|---|---|
| op head: 94% floor-bound commits, dead-zone grows over training | Floor pathology (r9-confirmed) | Arm C's differentiable per-num_valid floor (the current experiment) |
| blueprint head: near-uniform (max-share 18–20%) | Information-starved + architecture-degraded (heads can't condition on the sampled slot) + horizon-starved (55% force-pruned <13 epochs); **floor clamp = ceiling ~2.0× uniform (CORRECTED from claude-prime's 1.12× — 7-legal CNN set, floor 0.12, realized range [0.12, 0.28]; NOT currently binding at observed max-share)** | Post-permanence obs/architecture programme, staged: (i) wide-head floor λ (the ceiling, so future confident heads can express), (ii) D2-gated per-site obs, (iii) autoregressive routing (gpt's op→slot→params hierarchy is the candidate shape, needs DRL design review) |
| alpha_speed / alpha_curve: collapsed to one action | Credit-starved (primary), not information | NOT worth V5 budget (both primes); revisit after credit/floor work |
| Horizon starvation (blueprint credit drowned by early destruction) | Force-prune hazard ~0.11–0.13/decision persists in ALL arms (arm C keeps the PRUNE minimum deliberately) | The experiment tests the ESCAPE ROUTE (priced early commitment), not hazard reduction — if C fires on P2 but blueprint credit still drowns, the hazard itself re-enters scope (claude-prime r24 §3) |

## E. LOGGED-UNREAD (don't lose)

- `test_ev_liftoff_k4` threshold drift (2.9× vs 3.0×, proven not-caused-by-diff; cause unread).
- Filigree observations nearing expiry: pgrep liveness scar (2026-07-26), missing positive obs-version stamp, scratchpad filename collisions.
- Read #1 (`P(blueprint|slot)`) is pre-registered ONE-SIDED: flat = uninformative (floor compresses realized probs into [0.12, 0.28]); peaked = informative. Read #4 must avoid the survival collider: blueprint→EARLY-contribution + blueprint→survival as separate outcomes (claude-prime r24 §1).
