# PDR-0028 — EV-stab Stage-0: reconciliation + methodology correction; delivery adopts feat/ev-stab-stage2-hra

Date: 2026-07-05   Status: accepted (owner-ratified in-session: "Do it right on
ev-stab" + branch unification "flag only, don't act")
Author: Claude (agent)   Related: PDR-0027 (EV promoted to Now), the EV epic
esper-lite-f25b71c165, tasks esper-lite-3d67b09687 (Stage-0) / esper-lite-cfbdfdf040
(review gate → PDR-0029). Supersedes nothing.

## Context

Picking up the Now bet (EV-stabilization), the "first leg = Stage-0 instrumentation"
task turned out NOT to be greenfield work. Reconciliation found:

1. **Two diverging implementations of the EV epic.** The dedicated branch
   `feat/ev-stab-stage2-hra` (~24 commits) carries the real staged implementation
   (per-stream GAE, cf buffer threading, the HRA Stage-2 head, per-head adv-norm,
   and "Stage-0 gate metrics + per-stream EV on the ppo_updates view", commit
   `35374a1d`). The current Shapley branch `feat/phase-minus1-scale-falsifier`
   independently grew its OWN `partition.py` + `reward_variance.py` (per-STEP
   variance shares, from the Shapley Phase-0 evidence work) and picked up the Stage-2
   head via squash `575d2560`. The SAME file paths hold DIFFERENT code on the two
   branches — the single-source discipline the plan mandated is already violated.

2. **Stage-0's control-run gate is implemented NOWHERE (verified, not inferred).**
   On both branches `cov_rcf_return_share` / `r_main_cov` / `ev_main` / `ev_cf` are
   gated behind `if self.hra_value_decomposition:` (the Stage-2 ON leg). `35374a1d`
   says so verbatim ("computed ON-leg only", "OFF leg byte-identical") and explicitly
   DEFERRED the per-component `return_variance_shares`. `compute_variance_shares`
   exists but has zero production callers.

3. **The ON-leg reading is not a valid substitute** (two independent DRL reviews
   converged). The gate decomposes GAE **λ-returns** whose `returns_cf = A_cf + V_cf`
   bootstraps the cf head's own predictions, so it (a) can only be read under the
   treatment it is meant to justify, (b) has leg-(b) `r_main_cov` directionally biased
   TOWARD passing (λ-shrinkage deflates the measured std), and (c) is a post-treatment
   measurement of an on-policy state-visitation functional. A near-threshold reading
   (0.3–0.5 — the regime a 0.40 gate exists to adjudicate) is uninterpretable.

## Options

- **(a) Do it right on ev-stab** — adopt `feat/ev-stab-stage2-hra` as the EV-bet
  delivery branch; build the per-component buffer SoA + a value-free, raw-scale,
  per-RETURN covariance decomposition computed identically on both legs.
- **(b) Bank a provisional pass** by reading the ON-leg gate now (only valid if it
  lands decisively far from 0.40; near-threshold stays uninterpretable).
- **(c) Port Stage-0 onto the Shapley branch** — forks the impl further; rejected.
- **(d) Stop, write the reconciliation up** and defer implementation.

## Call

**(a). Owner-ratified.** Delivery of the EV bet happens on `feat/ev-stab-stage2-hra`
(the branch the workspace already named as the resume point). The canonical Stage-0
gate metric is redefined as a **value-free, raw-scale, per-return covariance
decomposition** (identical on both legs); the existing ON-leg λ-return metric is
retained but DEMOTED to a Stage-2 mechanism diagnostic. Landed this session on
ev-stab (commit `f721a5b3`, TDD 4/4 green): `compute_return_variance_shares`
(`src/esper/simic/telemetry/reward_variance.py`) — the pure core (share_i =
Cov(R_i,R)/Var(R) on discounted return-to-go; `share_attribution` = the >0.40 read;
`r_main_var_share` = Var(R_main)/Var(R), the smoothness leg; residual reconciliation;
degenerate guard). Remaining plumbing (per-component SoA, both-legs ppo_agent wiring,
6-file telemetry contract) tracked on esper-lite-3d67b09687.

**Branch unification is OWNER-GATED (flag only).** Merging/rebasing ev-stab into the
mainline or the "collapse to 0.3.0" is release-adjacent — flagged, NOT executed. The
two diverging `partition.py`/`reward_variance.py` copies are reconciled AT that
unification (ideally BOTH coexist: per-step secondary diagnostic + per-return gate).

## Rationale

The value-free metric reads identically on both legs — the control diagnostic Stage-0
requires — and emitting it ALONGSIDE the ON-leg λ-return version makes the gap between
them a direct measurement of V_cf contamination (a strictly-more-informative outcome
than a bugfix). Doing this on ev-stab (where the buffer/HRA machinery already lives)
avoids growing a third diverging copy on the Shapley branch.

## Reversal triggers

- Once the gate is wired and read on a capable-host **control (Stage-2 OFF)** run:
  `Cov(R_cf, R)/Var(R) ≤ 0.40` → the epic RE-SCOPES to the Stage-1 estimator fix only
  (the plan's own gate). A dense R_main term (`interaction_bonus`/`alpha_shock`/
  `pbrs_bonus`) shown as the variance culprit → pre-committed one-line move into R_cf.
- If the owner authorizes branch unification → reconcile the two copies then.
