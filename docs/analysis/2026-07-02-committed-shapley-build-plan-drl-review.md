# Committed-Shapley top-up — BUILD PLAN review (drl-expert / RL-correctness)

**Date:** 2026-07-02 · **Reviewer:** drl-expert (SME protocol) · **Artifact under review:**
`docs/plans/ready/2026-07-02-committed-shapley-topup-build.md` (build plan), against the ratified design,
the reward-function-reviewer F1–F8, and the GATE 2 retro-write mandate. **Verified at working tree on
`feat/phase-minus1-scale-falsifier`** (code seams read directly, not taken from the plan's evidence map).

## VERDICT: approved-with-changes

The mechanism (§2) is RL-correct as specified, the delivery path (§4) checks out against the live code, and
the scale=0 no-op story holds. The changes below are (a) one real F2 hole, (b) test-sufficiency gaps, and
(c) resolutions to the six §6 open questions the plan explicitly asks reviewers to settle. None threaten the
shipped scale=0 baseline; all are safe to fold into implementation.

---

## Code-verified correctness confirmations (for the gate)

1. **Rollback/forfeit alignment (the sharpest hazard) — SAFE.** `handle_rollbacks` builds
   `rollback_env_indices` from `env_rollback_occurred` (ppo_coordinator.py:124-126) and forfeits the reward
   prefix *only for those indices* (`mark_terminal_with_penalty` → `rewards[env, episode_start:last_idx] =
   ROLLBACK_FORFEIT_REWARD`, sets `dones[env,last_idx]=True`; rollout_buffer.py:930-933). The plan excludes
   exactly `rollback_env_indices`, so the exclusion set == the forfeit set. Ordering is right:
   `handle_rollbacks` (vectorized_trainer.py:2895) runs before `run_update` (:2905). Non-rollback envs'
   `buffer.rewards` are untouched by the forfeit → the retro-write lands on a clean channel. **Skipping
   rollback envs is sufficient.**
2. **No cross-buffer fossil persistence — orphan-credit hazard is impossible.** `reset_episode_state`
   (parallel_env_state.py:217-233) asserts all slots DORMANT at episode start and raises otherwise ("Create a
   fresh MorphogeneticModel per episode"). So the terminal coalition C = seeds fossilized *this* episode only;
   every member has a t_f record in this buffer. WI-3 resetting records in `reset_episode_state` is aligned.
   **Load-bearing invariant (what I actually verified):** credit-assignment safety rests on (a) t_f == the
   fossilize transition's real `buffer.add` step_idx (F-B), and (b) credited envs being **non-rollback**
   (excluded → a clean single terminal at T-1). I verified `reset_episode_state`'s *contents* and the
   `_run_batch` epoch loop, but not its call-timing vs the per-batch `buffer.reset`, and did not exhaustively
   rule out a non-rollback env emitting an early non-truncation `done`. That gap is why F-B's "done between t
   and t_f" case matters — but with rollback-env exclusion it is belt-and-suspenders, not the thing standing
   between the build and corruption.
3. **GAE linearity — verified against the actual recursion** (rollout_buffer.py:601-624): `rewards[t]` enters
   `delta` additively; `last_gae = delta + γλ·nnt·last_gae` propagates a t_f injection backward as
   (γλ)^(t_f−t), truncated at any true terminal. Δadv(t_f)=credit_buf exactly (pre-advantage-normalization).
4. **Retro-write seam/ordering — correct.** `run_update` write goes after the empty-buffer guard (:302,
   which correctly no-ops when every rollback was a first-step panic) and before `run_ppo_updates_fn` (:341),
   which is where `compute_advantages_and_returns` runs. Coordinator already holds `reward_normalizer` and
   `env_reward_configs` (:357) — flags/normalizer reach delivery with no new constructor params.
5. **`divide_by_std` — as described** (normalization.py:241-257): no clip, no stat update, `<2 samples ⇒ raw
   passthrough`, floor = `max(self.epsilon, std)`.
6. **HRA hard-exclusion is justified by the code.** When `cf_value_normalizer` is set, GAE splits into
   total/cf/main streams (rollout_buffer.py:585-663); a retro-write into `buffer.rewards` lands only in the
   total→main stream with undefined cf routing. There is no correct answer for "which stream owns a
   retro-written synergy credit" without a design decision → hard ValueError + TODO is right.
7. **alpha_override=0 removes a slot exactly** via the lerp path (slot.py); the STE branch fires only when
   `alpha_override is None`, so WI-9's arbitrary-coalition test is genuinely load-bearing, not ceremony.
8. **k=1 ⇒ top_up ≡ 0 by construction.** For a singleton coalition φ(s) = v({s})−v(∅) = c_paid(s), so
   gap = max(0, −τ) = 0. The term only fires for k≥2 (correct: synergy requires ≥2 members). WI-1 tests this.
9. **Ordering (cap → G-clamp) is semantically clean** — both are monotone non-negative reductions; τ deadband
   is applied inside gap before scale/cap. No ordering bug. The term is strictly additive (≥0), so freeloader
   discrimination stays with null-player→0 + the untouched −0.2 gate, as the design intends.

## Required changes (severity-ranked)

**F-A (MEDIUM — real F2 hole).** WI-4 says `credit_buf = top_up / max(std, std_floor)` "via divide_by_std
with floor," but `divide_by_std` floors at `self.epsilon` (1e-8), **not** `shapley_synergy_std_floor` — calling
it does not apply the floor. Worse, WI-5's `__post_init__` guards only `cap==0`, not `std_floor==0`, so F2's
amplification bound can be silently OFF at enablement (scale>0, std_floor=0 ⇒ credit = top_up/std, unbounded as
std→0; the exact ±10-blowout F2 exists to prevent). Fix: (i) compute the divisor explicitly as
`max(sample_std, std_floor)` from the coordinator's `reward_normalizer`; (ii) make a **normalized-space hard
clamp the PRIMARY bound** (unconditional, calibration-free — bounds the worst case even if std_floor is
mis-set), std_floor secondary; (iii) `__post_init__`: `scale>0` ⇒ `std_floor>0` (or clamp active). Add a test
that at small std the delivered credit is bounded.

**F-B (MEDIUM — test sufficiency).** WI-10's identity set is necessary but not sufficient. Add:
(1) truncated terminal at T-1 (bootstrap, `next_non_terminal=1` — the NORMAL non-rollback episode end) vs a
true `done`; (2) a true `done` strictly between t and t_f — credit propagation must STOP at the done (the
boundary case flagged in the review brief); (3) t_f == last step; (4) t_f == 0; (5) an env with
`num_steps < buffer width` to catch off-by-one on t_f vs `step_counts`. WI-12 must assert the credited step is
**exactly the FOSSILIZE-action transition** (check the op at t_f), not merely "a reward changed" — t_f capture
== the fossilize transition's `buffer.add` step_idx is the single highest-risk indexing detail.

**F-C (LOW-MEDIUM — a build-time decision, not deferrable).** WI-6 mutual exclusion couples "siblings off" to
"scale>0," so the planned enablement A/B (scale=0 vs scale>0) confounds {remove interaction_bonus + hindsight}
with {add top-up}. Bounded by dormancy (~12 events/n=5) but non-zero and regime-dependent (F1's own caveat).
This must be **decided in this build**, not deferred: either couple (accept the dormancy-bounded confound and
say so) OR expose sibling-disable as an independent flag now so both A/B arms can run siblings-off. Both are
defensible; leaving it implicit is not — a later "expose the flag" change is a second touch of live reward
code. Does not affect shipped scale=0 behavior either way.

**F-F (LOW — WI-8 rename shifts the scale=0 telemetry-key stream).** The existing channel still fires at
scale=0 (WI-6 gates it only at scale>0), so scale=0 runs will emit `interaction_bonus` where they previously
emitted `synergy_bonus`. Intentional and correct under No-Legacy, but it means F5 "byte-identical at scale=0"
holds for **training behavior** (rewards/RNG/model state), NOT the telemetry-key stream. State this explicitly,
and give WI-8 its own serde round-trip + Karn-view test — WI-11b's golden *reward* tests do not exercise the
telemetry key and will not catch a rename regression in the emit/serde/view path.

**F-D (LOW — the one no-op-at-0 deviation).** WI-3 fossilize-step records are always-on (not scale-gated).
Benign because WI-4 is driven by the *top_up* records (empty at scale=0 since WI-2 is scale-gated), not the raw
fossilize list. Confirm the record list feeds no scale=0 reward/telemetry/RNG/serialization-compare path; keep
WI-11a/b guarding. Consider gating WI-3 on scale>0 too if strict F5 byte-identity is wanted (the drip-gap it
also closes is independent and can be a separate always-on fix).

**F-E (LOW — eval state).** WI-11d should assert not just RNG-state identity but that model buffers (batchnorm
running stats, if any) are unchanged across the 2^k terminal eval, and that coalition evals run under
`model.eval()` + `inference_mode` (dropout off, no autograd/BN-stat side effects). WI-2 must use
`fused_forward`'s `alpha_overrides` (stateless — no `slot.alpha` mutation) so it does not extend the
alpha_schedule/eval-restore hazards the evidence map flags at vectorized_trainer.py:1350-1373.

## Recommendations on the six §6 open questions

1. **Sibling reconciliation:** mutual exclusion (proposed) — **endorse** for safety (kills feedback
   double-pay outright). Reject net-into-c_paid (mixes fossilize-time and terminal quantities — unit/timing
   mismatch) and assert-dormant (fragile, regime-dependent). Refinement per F-C.
2. **F2 shape:** **both**, normalized-space clamp PRIMARY + std_floor secondary + `__post_init__` guard
   (see F-A). A clamp is strictly safer than std_floor alone: it needs no advance knowledge of minimum std.
3. **HRA:** hard ValueError (proposed) — **endorse**; verified undefined cf-stream routing (confirmation #6).
4. **v(S) ON-member alpha:** natural alpha (proposed) — **endorse**; matches `committed` config and the actual
   inference-time contribution. Forced-1.0 measures a counterfactual that mis-scales both φ and G.
5. **c_paid drift:** **confirm** the terminal standalone `v({s})−v(∅)` replaces the fossilize-time snapshot;
   it is the correct quantity for isolating the synergy component (φ − standalone = interaction), preserves
   null-player→0, and avoids mis-crediting post-fossilize standalone growth. FLAG the semantic shift in the
   design amendment: c_paid no longer equals "what the dense LOO channel paid," so the term is now "the
   terminal synergy component," and the double-pay guard is null-player + τ + cap + G-clamp, not exact c_paid
   matching.
6. **Rename blast radius:** rename now (proposed) — **endorse** (No-Legacy mandates it; deferring leaves a
   live "synergy" collision). One commit, golden-test guarded (WI-11b). Karn old-key caveat is unavoidable
   either way and is correctly surfaced.

## Caveats held honestly
- The synergy measured is **internal to the fossilized set** (non-fossilized active slots masked to 0). Synergy
  an enabling stem realizes *through a still-training/blending neighbour* is not captured. This is baked into
  the ratified terminal-φ design, not a build defect — note it so enablement reads φ correctly.
- G is measured with the same entrenchment-biased terminal ablation as φ (design already acknowledges: G bounds
  aggregate POT, not per-seed misattribution). Unchanged by this build.
- k is usually ≤1 in the control regime, where the term is provably inert (confirmation #8); the term's
  enablement value depends on episodes reaching k≥2 fossilized slots. Magnitude question, not build. **Point the
  enablement gate at WI-7's `k` field** — the k≥2 frequency determines whether the A/B has any signal to
  measure at all; a low k≥2 rate would make an OFF-vs-ON contrast under-powered before magnitude is even asked.

## SME confidence assessment

**Confidence: HIGH** that the plan is safe to build default-OFF and correct as an implementation of the
ratified design + F1–F8. The load-bearing seams (rollback/forfeit alignment, GAE linearity, `divide_by_std`,
retro-write ordering, HRA split, alpha-mask kernel, fossil non-persistence) were read directly from the working
tree, and the Shapley/`c_paid`/G-clamp math was re-derived. **MEDIUM** confidence that F-A/F-B, if unaddressed,
would bite: F-A can silently disable F2's amplification bound at enablement; F-B's t_f-indexing test is the
thin edge where a silent credit-assignment corruption could hide.

**Risk if the changes are NOT made:** F-A → an enablement run could inject an unbounded credit past the ±10 the
rest of the reward respects (training-destabilizing, but only at scale>0). F-B → an off-by-one on t_f credits
the wrong transition silently. Both are enablement-time, not shipped-default, risks. Everything shipped at
scale=0 is verified no-op for training behavior.

**Information gaps (honest):** (1) `reset_episode_state` call-*timing* vs the per-batch `buffer.reset` was not
traced — I relied on its DORMANT-slot assertion + the epoch-loop structure (see confirmation #2 reframing).
(2) The WI-2↔WI-3↔WI-4 handoff mechanics (how the per-env top_up records travel from the terminal val pass to
`run_update`) are an unbuilt implementation detail, reviewed as specified, not as code. (3) Fossilize
frequency — whether "0.207 fossilize/ep" is per-epoch or per-episode, and the resulting k≥2 rate — was not
resolved; it bounds the term's enablement signal, not its build correctness. (4) This is a **plan** review: the
GAE identities are verified analytically and corroborated by GATE 2's empirical self-check ("retro Δadv == δ_buf
exactly"), not by executing unbuilt tests.

**Caveat on scope:** I reviewed RL correctness + delivery-path safety. torch.compile/graph-break interaction of
the new WI-2 forward configs, and the pytorch-side memory/perf of 2^k extra forward slices, are the
pytorch-expert's lane and are out of this verdict.
