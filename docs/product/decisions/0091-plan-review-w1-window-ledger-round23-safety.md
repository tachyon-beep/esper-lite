# PDR-0091 — Phase-2 plan review adjudicated (REQUEST-CHANGES → rev 2); NEW W1 finding: window provisional must be LEDGER-DRIVEN; round-23 safety semantics adopted (converged parts); two owner forks live (abort payment; pending-visibility); 80%-UCB certification adopted

Date: 2026-07-14   Status: accepted (adjudication + plan revision + spec amendment within standing authority; forks F1/F2 are OWNER-OPEN, implemented-behind-config/no-obs-change respectively)
Follows PDR-0090. Inputs: drl-expert plan review r1 (REQUEST-CHANGES, B1–B5/N1–N5, all code-cited and verified), claude-prime round-23, gpt-prime round-23 (owner-relayed). Artifacts: plan rev 2 (`docs/plans/ready/2026-07-14-phase2-settlement-implementation.md`), pre-reg §2.5.9 W1 amendment.

## What does this buy?  (REQUIRED — PDR-0068)
The review-then-build gate did its job before a line of code existed: five blocking under-specifications are now pinned (epoch-phase placement resolving four sub-defects at once; the F8-reopening request-row branch; the un-assertable annuity transform; safety semantics; OFF-identity). And verifying the review's B1a against code surfaced a finding the review itself missed — the masked pending slot can never be targeted, so the spec's window provisional was structurally unpayable as written. The seventh instrument-corrected claim of the arc, this time correcting the spec's own §2.5.9.

## Decisions
1. **W1 (new, this session): the window provisional is LEDGER-DRIVEN.** Reward pays only the targeted slot
   (`action_execution.py:944-947`); masking the pending slot makes the §2.5.9 window provisional unreachable via
   targeting and would let negative carry stop at request (F8 through the mask). The ledger pays measured c_t per
   window epoch through the full live transform; the cadence delta (every-epoch vs S's when-targeted) is an explicit
   protocol-package estimand component. Pre-reg §2.5.9 amended with a PDR marker.
2. **Review findings adopted, all verified against code:** B1 → §0 epoch-phase pin (request at dispatch of R;
   q_settle locks at reward-phase of B INCLUDING c_B, window [R+1,B], B ≥ R+5; provisional suppressed at B; transition
   at dispatch of B; maintenance B+1 ≡ arm A's t_c+1 parity — B1d resolved BY parity, not preference). B2 → request-row
   branch table (cf<0 zeroing DISABLED at request; warning exemption kept; shaping sterilized). B3 → T_annuity
   enumerated (frozen inputs progress_B/cf_B/timing; ratio_penalty EXCLUDED — the spot channel it guards is
   structurally closed; G-CONTINUITY target now defined). B5 → OFF-identity via the phase-1 suite + the
   committed-forced-False behavioral sweep; serialization unconditional per no-legacy, mid-window round-trip test.
   N1–N5 adopted (incl. N5: G-PBRS asserts the harness per-scenario Δ — round-22 pin 3a's *wording* corrected, intent
   kept). The ten replay areas + eight construction invariants TRANSCRIBED into the plan (were un-enumerated).
3. **Round-23 convergence applied:** ADOPTED (converged): hard safety OVERRIDES commitment — always — with
   `FOSSILIZE_ABORTED_BY_SAFETY`, lock release, generation-ID integrity, mandatory telemetry, never-silent reporting;
   ordinary governor/resource morphology SUSPENDED by the Option-A lock (the coherent reading of the ratified lock's
   purpose — any op moving the ensemble-relative quote, whoever issues it).
4. **OWNER FORK F1 — abort payment semantics (primes diverge):** claude-prime: settle at the scheduled boundary on
   q_settle-so-far (liability preserved; SETTLEMENT-INTERRUPTED censoring class) vs gpt-prime: no prior/no annuity +
   intention-to-treat P2 + guardrail reporting. Both implemented behind `abort_payment_mode`; replay tests both;
   owner picks before freeze.
5. **OWNER FORK F2 — pending-visibility (primes still diverge, round 23):** claude-prime banks the invisibility +
   instrumented-aliasing resolution (adding a pre-registered SYMMETRY CHECK: critic error concentrated in open-window
   epochs in arm B only); gpt-prime rejects it — requires a source-level observability proof and, failing it, two
   common-schema fields for ALL arms (pending bit + normalized countdown; arm A zeros; explicit obs-schema version)
   plus a G-OBSERVABILITY replay area. **The source-level observability proof is queued REGARDLESS** (both
   resolutions need its facts: is committed in the obs; are masks network inputs; is epoch/cadence visible; is the
   previous action in the recurrent input). No obs change until the owner rules.
6. **Blinded certification posture: gpt's override ADOPTED — one-sided 80% UCB on σ_d is PRIMARY** for the n=10
   certification (point estimate secondary, 90% UCB sensitivity). Claude-prime is explicitly indifferent
   ("reversible until freeze"), the power-calc draft itself offered this as the statable alternative, and the
   asymmetry is real: an optimistically-low df=4 point estimate certifies an underpowered design that cannot be
   repaired without returning to the owner. Gate outcome bounds stay UCB95 < Δ_material (PDR-0088).
7. **PDR-0090 authority annotation (claude-prime, recorded here — 0090 is committed and append-only):** the Phase-2
   authorization row is **owner-delegated-via-convergence-rule** — claude-prime's round-22 words were a
   recommendation on an owner-tier item, not convergence-as-ratification; the authority came from the owner's
   standing rule, not from two models agreeing.
8. **New pre-freeze tasks:** Option-A lock-burden ESTIMATE from r9 (counterfactual overlay: given observed commit
   rates and W=10 windows, what fraction of epochs would be locked and which morphology ops suppressed) — the one
   remaining freeze input with no in-flight task (claude-prime); the observability source-proof read (#5).

## Reversal trigger
- If the observability proof shows the critic ALREADY sees enough to derive pending+countdown → F2 collapses to
  claude-prime's resolution with a citation (no obs change, telemetry stands down to guardrail).
- If the re-review finds the §0 placement contradicts any trainer mechanics on a second look → placement re-derives
  before build; no code lands on an unpinned pipeline.
- If the r9 burden estimate shows Option-A locks ≫ expected (e.g., >25% of epochs) → PDR-0084's escalate-to-Option-B
  trigger fires BEFORE freeze.
- W1's estimand label: if the every-epoch ledger cadence is later judged to distort B−A materially (adversarial
  pass), the alternative is paying the window provisional only on epochs S-would-have-been-targeted — rejected for
  now as policy-conditioned (unimplementable counterfactual); revisit only with a concrete construction.
