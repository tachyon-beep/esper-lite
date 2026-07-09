# PDR-0049 — Telemetry Improvement Program captured as queued bet (intent)

Date: 2026-07-09
Status: decided (capture-as-intent; commitment deferred to post-Stage-2 DECIDE)
Decider: owner (john) directed the capture; owner-agent applied scope judgment per
explicit instruction ("use your judgment, don't assume it will be exactly this")

## Context

The owner drafted (with external-assistant help) a nominal "Telemetry Improvement
Program": a post-Stage-2 body of work whose purpose is not more logging but making
every experimental read self-defending — traceable from typed event → view → metric
definition → validity gate → PDR decision, with population and denominator explicit.
It generalizes the failure classes the last three months actually hit (PDR-0006
all-step-entropy false collapse; PDR-0026 informative-vs-bandwidth-bound nullity;
glob/partial-run contamination incl. today's `killed-by-session-reap-1/`; metric
naming drift; half-wired telemetry chains).

## The call

CAPTURE as a shaped Next bet (roadmap intent), NOT commit. Artifacts:
- Concept plan `docs/plans/concepts/2026-07-09-telemetry-improvement-program.md`
- Tracker epic `esper-lite-c62d4891b0` (P2; ticket breakdown deferred to commitment)
- Roadmap Next entry (intent only)

## Judgment applied to the nominal plan (deltas from the owner draft)

1. **Dedup against landed machinery.** Nominal §6 (critic/value telemetry) and §11
   (GPU/runtime) are largely already emitted (`PPO_UPDATE_COMPLETED` carries EV
   streams, return-var shares, pre/post-norm advantage std, choice-conditional
   entropies, allocator/cadence/dataloader stats). Phase B is scoped as
   audit-then-gap-fill; the Stage-2 wrapper's §1 validity gates and byte-identity
   discipline are the seeds of A3/rule-6, generalized rather than rebuilt.
2. **Added rule 11 (validity envelope).** New scar found 2026-07-09: with
   `recurrent_n_epochs=1` the PPO ratio is identically 1.0, so clip/ratio metrics
   read as constants. Registry metrics must declare the config conditions under
   which they are informative; consuming one outside its envelope is a validity
   failure.
3. **Rehomed items.** Dashboards/denominator-labelling → Sanctum tri-domain line
   (esper-lite-5fab041f3a); Shapley/tau calibration telemetry → stays with the
   parked Shapley line's reversal triggers (PDR-0026/0027); GPU co-tenancy
   descriptors → manifest fields (A1).
4. **Phase D merged with the research-verdict obs-feed line** — one observation
   programme. Offline information-ceiling probes gate Obs V4 (they discriminate
   "information absent" from "critic can't extract it", the open question from the
   2026-07-09 signal-quality discussion).
5. **Freeze safety made explicit.** Phase A is scorer/read-path only and startable
   during the Stage-2 freeze; Phases B–D wait for the ON wave (PDR-0048
   same-commit discipline). No tickets spawned now — tracker stays lean until the
   bet is committed.

## Owner ruling (same day, post-capture)

"Slow and steady is the rule for now — finish Stage-2 and then pick our next
direction." TIP does NOT start early despite Phase A's freeze-safety; no new work
lines open until the Stage-2 read is in hand. The post-Stage-2 DECIDE chooses among
TIP, reward-efficiency statistics, and whatever the packet read motivates.

## Reversal trigger

Re-open this capture if: (a) the Stage-2 packet read lands with zero scoring/validity
friction (weakens Phase-A urgency), or (b) the post-Stage-2 DECIDE prefers direct
Obs-V4 work over infrastructure (then Phase D's probes still run first as its gate),
or (c) the owner amends the program's thesis.

## Provenance note

Same-session context: Stage-2 OFF wave was killed 16:04 2026-07-08 by
launcher-session close and relaunched 23:16 from `fe177844` via setsid/nohup
(session-proof); trend read of the killed partials (149/145 updates) informed rules
10–11 and Phase D framing. Run-provenance commit for the live wave is `fe177844`
(two training-inert commits atop frozen `04b58ed4`; discipline intact).
