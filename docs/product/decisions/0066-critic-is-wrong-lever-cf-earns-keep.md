# PDR-0066 — The critic is the wrong lever; reframe to "does the cf shaping term earn its keep?" + adopt a pre-launch read-review gate

Date: 2026-07-13
Status: accepted (findings + process fix within grant; the direction reframe it proposes
is owner-gated — flagged)
Supersedes: PDR-0065's "next step = heavy-A′ pilot" recommendation (on the strength of the
exact-decomposition + matched-OFF data). Read: `docs/analysis/2026-07-13-on-leg-gram-read.md`
(advisor + 2 external peers reviewed).

## The finding (exact decomposition, matched pooled, rounds 400–600)

- **Main head near-perfect** (`Var(e_main)` ≈ 0.4–0.6). The ENTIRE total residual (~228) is
  the cf stream; component errors mildly anti-correlate.
- **The residual is reward-intrinsic, not critic-fixable at this capacity.** The direct-total
  OFF critic — which is exactly heavy-A′'s primary head — already exists and reaches EV ~0.64
  with the same ~228 absolute residual. So **heavy-A′'s upside over the rejected HRA is at
  most ~0.02–0.03 EV; its ceiling is already known from data on disk.** The A′/B/C
  critic-architecture tree has a low, largely-known ceiling.
- Corrections banked: the "+0.04 recalibration lift" was a pooled-vs-median statistic
  artifact (matched lift ≈ 0); "B favoured", "light-A′ refuted", "cf ceiling 0.57", and
  "plateau-1 split" all retracted as over-reads. Objective A stays rejected (screen stands),
  but the "large total-fit deficit" was substantially a pre-anneal/statistic artifact.

## The reframe (owner-gated DECIDE)

The epic has litigated A/A′/B/C — all critic-architecture moves — without ever establishing
that the thing they fix costs anything downstream. The data says the critic is near the
reward-intrinsic ceiling. So the live questions are NOT about the critic:

1. **Does the cf shaping term earn its keep? (recommended next test)** `G_cf` carries ~all
   return variance AND ~39% is unpredictable from state → it injects advantage noise (the
   opposite of shaping's purpose). Direct test: **ablate the cf term, measure the DOWNSTREAM
   outcome** (host accuracy / controller decision quality / committed-J). Collapse ⇒ cf is
   load-bearing, now quantified (the baseline the epic never had). No collapse ⇒ the
   critic-redesign tree dissolves and a term is deleted. Cheaper than the read just run.
2. **The missing downstream baseline:** connect `ev_sum` to a metric that matters before any
   further critic work. EV is instrumental; ~0.6 is unremarkable on its own.
3. **Path-C ceiling probe (cheap):** offline predictor on Obs V3 history for `Var(G_cf|s)` —
   information ceiling vs critic-capacity. Complements (1).

Recommendation: (1) first (it can dissolve the whole tree for less than a critic pilot),
with (3) as a cheap parallel if the rollout data is already logged. NOT a heavy-A′ pilot —
its ceiling is already ~known and ~0.02 above the rejected design.

## Process fix (adopted; amends the run-authorization discipline)

Every over-read this session (schedule-caused; B-favoured; light-A′-refuted; the pooled-vs-
median lift) was caught AFTER the compute yet catchable BEFORE. Memory notes are not controls.
**A pre-committed reading now gets an advisor pass BEFORE the run launches** — a gate, not a
recalled note. The standing grant's "PDR + pre-committed reading each time" gains: "...and the
reading is adversarially reviewed for what the metric can structurally NOT show, pre-launch."

## Reversal trigger

- If the cf-ablation downstream outcome shows cf is load-bearing (host/controller metric
  degrades materially when cf is off), the shaping term is justified and the question returns
  to reducing its advantage-noise cost (which MAY re-open a narrow critic/de-shaping angle) —
  but now with a downstream baseline, not blind.
- If a Path-C offline predictor substantially beats the ~228 residual, the ceiling is
  critic-capacity not information — heavy-A′/bigger critic regains plausibility.
