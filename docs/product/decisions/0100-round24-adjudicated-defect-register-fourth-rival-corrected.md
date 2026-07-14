# PDR-0100 — Round-24 adjudicated: fourth rival CORRECTED by instrument (blueprint ceiling ≈2.0× uniform, not 1.12×, and NOT currently binding); velocity disposition = transport-now/obs-at-V5; committed_val_acc promoted to Exp-1 secondary; DEFECT REGISTER created (owner-requested)

Date: 2026-07-14   Status: accepted (adjudication + register + pre-reg PROPOSED line within standing authority; committed_val_acc secondary and all V5 content ratify at freeze/post-experiment respectively)
Follows PDR-0099. Inputs: claude-prime r24 + gpt-prime r24 (owner-relayed), adjudicated under the convergence rule with main-session verification. Artifacts: `docs/product/defect-register.md` (NEW standing workspace file, owner-requested: "what do we need to fix before we run our next experiment"), pre-reg §1.6 addition, task esper-lite-c739c3ab97.

## What does this buy?  (REQUIRED — PDR-0068)
The owner's fix-list now lives in the workspace with WHEN-gates attached (before-next-run / at-V5 / deferred-priced / programme), so nothing depends on PDR memory. And the verification rule caught its ninth instrument-corrected claim — this one internally inconsistent with data its own author had banked: the "1.12× cap" would make r9's measured 18–20% blueprint max-share impossible.

## The fourth-rival correction (verified in source this session)
claude-prime's blueprint floor clamp: computed at 13 legal with f=0.99/13 → cap/uniform ≈ 1.12×. ACTUAL (verified:
`PROBABILITY_FLOOR_PER_HEAD["blueprint"]=0.12` at leyline/__init__.py:303; `_apply_floor_to_logits` caps the floor
at 0.99/num_valid, action_masks.py:548-554; CNN topology masks to 8 blueprints MINUS NOOP → **7 legal**,
action_masks.py:240-249 + factored_actions.py:457): effective floor 0.12, single-action cap = 1−6×0.12 = **0.28**,
uniform = 0.143 → **ceiling ≈ 2.0× uniform; realized probs live in [0.12, 0.28]**. r9's 18–20% max-share sits
BELOW the cap → **the clamp is NOT currently binding** and does not explain the near-uniformity (the
information/architecture/horizon rivals stay primary). It IS a real ceiling that would bind a future confident
head — so claude-prime's ceiling-first staging survives in WEAKENED form: stage-1 of the future programme, not the
explanation of today's behavior. His read #1 one-sidedness point stands (the floor compresses the observable range
→ flat tables are partly construction; pre-register flat=uninformative / peaked=informative).

## Convergence audit (the standing rule applied)
- **CONVERGED → ADOPTED:** keep the experiment narrow (F2's one bit only; NO other feature activation — "scientific
  isolation"); velocity OBS exposure deferred to V5; run the offline read ladder NOW with the corrected designs
  (read #1 one-sided; read #4 splits blueprint→EARLY-contribution and blueprint→survival to kill the survival
  collider); one bounded V5 batch with admission gates (D2-cleared / verified-defect / semantic-correction),
  GERMINATE-led, missingness-triple discipline; NEVER redefine num_holding (additive decomposition at V5 +
  deprecation); obs and architecture staged as separately attributable treatments (B−A = information, C−B =
  factorization); pipeline proof obligations (transport-completeness test, dead-dim sweep as standing pre-run
  packet, reward-obs coverage report, generated layout at V5); alpha heads not worth V5 budget; op head = floor.
- **DIVERGED → ADJUDICATED (main session):** velocity transport-fix timing — gpt's fix-now adopted in NO-LEGACY
  form: to_leyline fix + transport test + telemetry live NOW; obs dim +13 stays a LITERAL 0.0 in the V3/V4 encoder
  (documented schema semantics, no runtime version branch — gpt's `if version<=V4` snippet is the exact pattern
  CLAUDE.md prohibits); byte-identity guarded by the phase-1 replay goldens. Rationale: same experiment effect as
  claude's defer, plus true-velocity telemetry for the aliasing/commitment analyses and an un-xfailed completeness
  test.
- **ONE-SIDED → ADOPTED with attribution:** committed_val_acc as a NAMED Exp-1 SECONDARY (claude §2; gpt silent,
  not opposed; analysis-side only) — pre-reg §1.6 PROPOSED line added, owner ratifies at freeze. gpt's routing
  hierarchy (op→slot→params) recorded as the candidate shape for programme stage-3, needs its own DRL design
  review. claude's §0 pins recorded: the 0.68–0.75 slot-confidence figure is the germination-time multi-slot
  cohort (READ 5's ~1.0 was mask-forced HOLDING — different cohort, weaker mitigation than it suggests); the F2
  ruling SUPERSEDED round-23's instrument-first resolution BY THE OWNER (PDR-0098), and the critic-error telemetry
  is now fix-validation, not forgone-cost measurement.
- **claude §3 precision adopted:** the experiment tests an ESCAPE ROUTE from horizon starvation (priced early
  commitment), not hazard reduction — the ~0.11–0.13 destruction hazard persists in all arms by design; if C fires
  on P2 but blueprint credit still drowns, the hazard re-enters scope.

## Decisions
1. **Defect register created** (`docs/product/defect-register.md`): A-gates before the next run (F2 landmines;
   velocity transport + test; dead-dim sweep; coverage report; the settlement track), B = the bounded V5 batch,
   C = deferred-priced (eis==0; F8-in-arm-A; num_holding), D = training pathologies with stage order, E =
   logged-unread. Linked from current-state.
2. **Pipeline-hardening task created** (esper-lite-c739c3ab97, P1, gates the next run).
3. Instrument-corrected-claim tally: **nine**.

## Reversal trigger
- If the dead-dim sweep (run-once, now) finds additional velocity-class dims → they join register A2's fix +
  the transport test's field list; if it finds NONE beyond +13, the class-closure claim gains its evidence.
- If the owner or primes reject the committed_val_acc secondary at freeze → the line drops; it is analysis-side
  and severable.
- If the D2 ladder shows site features add no held-out predictive power → register B's site-feature row drops and
  the programme re-centers on architecture/credit (unchanged from PDR-0099's trigger).
