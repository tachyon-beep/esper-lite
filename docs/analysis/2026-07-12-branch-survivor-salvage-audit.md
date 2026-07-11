# Branch-survivor salvage audit — KEEP/DROP/CONFLICT manifest

Date: 2026-07-12 · Author: product-ownership session (checkpoint #39 arc)
Tracker: esper-lite-1f1e55f58f · Discipline: informed jettison (PDR-0038)
Status: **PRE-WORK** — direction decided; execution GATED on the 600-round diagnostic read.

## Purpose

The B-deferred winner-takes-all disposition (PDR-0038) reaches its trigger: the
Stage-2 screen returned REJECT, so the branch-vs-main survivor call is unblocked.
PDR-0038 forbids discarding either line blind — this manifest is the required
salvage audit (KEEP/DROP per work unit + consolidation plan) for owner sign-off.

## Direction decided (owner + advisor, 2026-07-12)

**Survivor = `main` as the single trunk.** Consolidate all durable work onto main;
retire `feat/ev-stab-stage2-hra` after salvage. Rationale (advisor-confirmed):
`main` is the default ref PRs target and the remote tracks — making the feature
branch the new trunk would need an escalation-gated default-branch swap / history
rewrite, so branch-as-trunk has *more* git friction, not less. No effort asymmetry
exists: this is a consolidation, nothing durable is discarded either way; the real
cost is reconciling the shared-file zone, which is identical regardless of trunk.

**Mechanism = MERGE, not cherry-pick.** Merge `feat/ev-stab-stage2-hra` into `main`
(preserving history), then excise the rejected HRA and confirm main's unique
reward-redesign track rides on top. Hand-cherry-picking ~100 commits would replay
every diff as a fresh conflict and lose attribution; a merge reconciles the two
trees once. The product-workspace superset (below) merges with **zero** conflict.

## Framing correction: this is a SHARED-FOUNDATION fork, not two rival lines

Initial read (from `git cherry` showing only 1/125 patch-equivalent) suggested two
independent reimplementations. **That is wrong.** Main's HRA commit `575d2560`
(2026-06-25) is a *squash* of the branch's EV-stab Stage-2 work onto the 0.3.0 line
— its own message says so: *"Collapses the completed EV-stabilization Stage 2 work…
Granular per-increment TDD history preserved on branch feat/ev-stab-stage2-hra."*
`git cherry` saw no matches because ~20 granular branch commits were squashed into
one — a packaging difference, not divergent code.

The true topology:
- **Shared foundation:** the EV-stab Stage-2 HRA + Stage-0 gate (main = squash,
  branch = granular + modest later refinement). `advantages.py` (per-head adv-norm,
  main's "one hard dependency") is **byte-identical** on both lines. `partition.py`
  differs by only ±60 lines (Stage-0 additend refinement, not a rival design).
- **Then they diverged:** main → reward-redesign / Committed-Shapley / PIN-E;
  branch → EV-stab continuation + Sanctum v2 + entropy fix + oracle sandbox +
  schedule-fix/vg-telemetry.

Consequence: the 24-file "conflict zone" is mostly **additive to a shared base**
(each side added to `ppo_agent.py`/`telemetry.py`/`contribution.py`), not
contradictory rewrites — so merge reconciliation is modest and localized.

## DROP — the rejected HRA treatment (both lines)

The n=5 screen REJECT (PDR-0059) is final for this HRA *as implemented*. It exists
on BOTH lines (main squashed, branch granular) and is the same design, so it is
excised from the unified trunk after merge — NOT carried forward.

| Component | Where | Excision note |
|-----------|-------|---------------|
| `cf_value_head` / `_compute_cf_value` (head-only V_cf critic) | tamiyo policy, `ppo_agent.py` | Gated behind `hra_value_decomposition` (default OFF) — excise flag + head |
| `split_reward_streams` subtractive partition | `rewards/partition.py` | Stage-0 diagnostic reuses part of this; keep the value-free decomposition, drop the V_cf routing |
| per-stream GAE (`returns_cf`/`returns_main`) | `rollout_buffer.py` | ON-leg only; excise |
| `RewardNormalizer.divide_by_std`, V_cf normalizer, VALUE_HEAD_SCHEMA v3 | `control/normalization.py`, checkpoint | Roll back schema bump with the head |

Note: the Stage-0 *diagnosis* instrument (value-free `Cov(R_cf,R)/Var(R)` gate) is
KEPT — it is diagnostic-only and already on both lines; only the HRA *treatment* drops.

## KEEP — branch-unique work to salvage onto main (merge brings these)

| Work unit | Commits | Salvage value if branch lost | HRA entanglement |
|-----------|---------|------------------------------|------------------|
| **Sanctum layout v2 + telemetry surfacing** | `9b91b60b`..`8e3f2d86` (~22) | Large reusable TUI framework (tab screen, A/B ExperimentPanel, AnomalyStrip severity model, Governor&Growth tab, per-head learnability strip, rollout-LSTM health, prune attribution) — would need full rebuild | CLEAN (surfaces EV metrics but framework is independent) |
| **Schedule-fix + vg Gram telemetry** | `7ea84dc6` | drl-approved; schedule fix is HRA-independent; vg telemetry is the PDR-0060 sufficient-stat contract for any future critic work | CLEAN |
| **Entropy-thermometer fix (Phase 1)** | `a9766498`, `8a037e14` | Adds `choice_conditional_head_entropies` (6 src files) — main has **0** references (main's `a84f9a78` is a lighter alarm re-point only). Branch's is the fuller, live-validated fix | CLEAN — but overlaps main's lighter fix in `emitters.py` + `ppo_coordinator.py` → small reconcile at merge |
| **Escrow-Markovian fix** | `89b50a16`, `0279fbe3` | PDR-0047 correctness fix (removed reward-path escrow delta clipping; Obs V3 schema v2) — independent bug fix | PARTIAL (Obs V3 touched by HRA; the Markovian fix itself is independent) |
| **Oracle sandbox proof telemetry** | `69e6d55b`..`ee5dfcc8`, `7ebaff72`, `0838cc40` | Independent proof-trace feature (tracker esper-lite-a36f3311a3) | CLEAN |
| **Per-head advantage normalization** | `2cb58ada` | Already **byte-identical** on main (`advantages.py`) — NO action needed | n/a (shared) |
| **Stage-2 MAJOR-1 acceptance harness** | `2fad1fc4`, `da2c5f00`, `9666fe0c` + hardening | Reusable scorer for a future re-pre-registered gate (manifest-driven scoring, validity envelope) — value depends on whether a redesigned HRA is pursued | PARTIAL (scores the dead design but the scoring engine is reusable) |
| **Product workspace 0038–0061 + newer vision.md** | 19 checkpoints | Continuity spine: PDRs 0038–0061 additive over the shared 0001–0037; branch vision.md carries the 2026-07-10 grant broadening main lacks | CLEAN — merges with ZERO conflict (append-only superset) |

## KEEP — main-unique work (already on trunk, must not be lost in merge)

| Work unit | Commits | Note |
|-----------|---------|------|
| Reward-redesign Phase −1/0 (J pin, scale falsifiers, Stage-0 telemetry) | `f1f2551f`, `406aeb25`, `58b977f9` | Distinct research track; branch never had it |
| Committed-Shapley top-up build | `a479d776`, `e0d98d5f`, `3448d38c`, `173150c9`, `5910c5e6` | Banked-validated instrument (PARKED per PDR-0027, scale 0.0) |
| PIN-E placebo noise-floor harness | `cbfd7693`, `a2a513ef`, `ce3ad31e`, `9be132c1` | tau lower-bound instrument |
| Adversarial SHAPED-reward audit | `f3d8db95` | 9 findings, PBRS condition table |
| EV variance-reduction research corpus | `127f25cb` | Plans/analysis |

These stay on the trunk by virtue of `main` being the merge base — the merge must
be verified not to revert them (they touch `contribution.py`/`reward_variance.py`
in the shared zone).

## Reconciliation zone (verify at merge)

24 source files touched on both sides; the genuinely-divergent (non-additive) ones:

| File | Branch Δ | Main Δ | Nature |
|------|----------|--------|--------|
| `ppo_agent.py` | 422 | 291 | Branch: schedule/vg/HRA; main: HRA-squash/reward — reconcile |
| `contribution.py` | 9 | 119 | Mostly main (reward-redesign) — keep main's |
| `reward_variance.py` | 152 | 223 | Both extended the Stage-0 decomposition — reconcile |
| `telemetry.py` | 233 | 289 | Both added payload fields — additive, reconcile field lists |
| `emitters.py`, `ppo_coordinator.py` | — | — | Entropy-fix overlap (above) |
| `advantages.py`, `ppo_update.py`, `rollout_buffer.py` | — | — | Identical or shared — no reconcile |

## Consolidation plan (execute POST-diagnostic)

Ordering note (advisor, 2026-07-12): the merge executes AFTER both the diagnostic read
and the epic-direction DECIDE, so every direction-contingent disposition below is
decided with full information — there is no interim in which dead code sits on the trunk.

1. Read out the 600-round diagnostic (gates execution; do not start before).
2. **Epic-direction DECIDE happens first** (also post-diagnostic) — its outcome drives
   step 5's acceptance-harness disposition.
3. **Tag the branch tip before any retirement** (owner, 2026-07-12): a durable,
   recoverable ref (e.g. `archive/ev-stab-stage2-hra-<sha>`) so nothing vanishes
   irrecoverably; the branch is then let go naturally as unreferenced — NO hard delete.
4. Branch off `main`; `git merge feat/ev-stab-stage2-hra` (history-preserving).
5. Excise the rejected HRA (DROP table) as one deliberate change on top; roll back
   VALUE_HEAD_SCHEMA v3 with it.
6. **EXPLICIT GATED LINE ITEM — acceptance-harness excise/keep (do NOT let it ride in
   by default).** A merge carries everything across, so "defer" becomes "keep by
   accident" unless this is an explicit step. Apply the pre-agreed discriminator on the
   epic-direction outcome:
   - epic = **redesigned HRA (A′/B)** or **TIP** → KEEP-and-adapt the reusable core
     (manifest-driven scorer, §11.3 validity envelope, scar-suite, mechanism-vs-outcome
     discipline); it is live seed material.
   - epic = **reward-efficiency** or **Stage-1 flag** → EXCISE the harness at merge; the
     tag (step 3) preserves it for later resurrection if TIP is picked up under serial
     focus.
7. Verify main's reward-redesign/Shapley/PIN-E track survived the merge intact.
8. Reconcile the small overlap zone (entropy fix, telemetry field lists, Stage-0).
9. Full suite green; then retire `feat/ev-stab-stage2-hra` per step 3 (tag-anchored,
   natural vanish — owner-approved 2026-07-12; hard deletion remains escalation).

## Anti-recurrence (owner's second ask — own PDR)

Root cause: two workstreams edited the same core files and both maintained the
product workspace — this lineage on the feature branch, other sessions/Codex
landing on `main` via PRs (e.g. the `575d2560` squash, the reward-redesign track).
Divergence was organizational, not technical. Durable fix: **one trunk, short-lived
branches merged back promptly, product workspace owned in a single place.** Record
as its own PDR; secondary to getting the reconciliation right.

## Owner sign-off (PDR-0038) — resolved 2026-07-12

- [x] **DROP list confirmed** — rejected HRA excised from trunk, both lineages.
- [~] **Acceptance harness = DEFERRED to the epic-direction DECIDE** (advisor-recommended;
      the merge post-dates that DECIDE, so it is decided with full information). Pre-agreed
      discriminator: A′/B or TIP → keep-and-adapt; reward-efficiency or Stage-1 → excise.
      Explicit gated line item in the consolidation plan (step 6); tag-preserved either way.
- [x] **Branch retirement approved WITH a safety anchor** — tag the tip before retiring;
      let it vanish naturally as an unreferenced branch; no hard delete.
- [~] **Anti-recurrence (PDR-0063) = DEFERRED** — stays `proposed`; owner will revisit.
