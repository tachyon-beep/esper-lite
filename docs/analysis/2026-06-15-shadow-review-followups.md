# Shadow Review — Follow-Ups

**What this is:** A running list maintained by the read-only shadow reviewer while
another agent clears correctness bugs on branch `0.1.1`. Working assumption: more
good than bad — when you're back, triage this list or dump it.

**Last updated:** 2026-06-15 07:33 UTC (pass 12)
**Baseline at last pass:** HEAD `82155c14`, diff sha256 `a011992f…`, untracked sha256 `75ea11ea…`
**Overall so far:** No correctness regressions to the live/normal path across 7 passes. Tests strengthened, not gamed. **Pass 7 resolved the pass-6 🔴 scare: `static_final` became runner-supported, but the agent enabled it HONESTLY via `topology_only` — so F12 downgrades 🔴→🟡 (the `state_dict_exact` policy is now just a dead, never-emitted footgun + a test-fixture-drift).** All four proof-baseline cohorts are now supported and verified correct; normal training re-confirmed gated/unchanged. 🔴 section is back to just F1 (run the validation). The agent keeps building the F1 regime rather than declaring victory.

---

## Pass log

- **Pass 1 (02:17):** reviewed the initial 24-file batch — rollback tail-dominance, op-head FP32+floor, telemetry enrichments, Karn dedup, dead-code removal. All clear; F1 raised.
- **Pass 2 (02:31):** delta = +8 files (SeedInfo Optional widening, new `normalizer_checkpoint.py`, fossilize handler, record_stream test, vectorized save/restore refactor, leyline Obs-V3 constants). All clear (drl-expert + pytorch-expert verified). No new 🔴. Two trivial nits added to 🟡.
- **Pass 3 (02:53):** delta = +15 files — simic/PPO core UNCHANGED. New work: `proof_packet.py` (+290, typed verdicts + precision gate), Karn/Nissa telemetry (`collector`/`health`/`store`/`triggers`/`views`/`wandb` — `val_accuracy` → `float|None`), 3 planning docs (correctness-proof-strategy, ppo-stability-oracle-sandbox, proof-baseline-controls), ROADMAP/PLAN_TRACKER edits, `gpu_sync_whitelist.yaml`. python-code-reviewer verdict: correct, honest, fails closed, no bug-hiding. Docs honest (no over-claiming; reward-efficiency still deferred). No new 🔴. Two minor items (F6, F7).
- **Pass 4 (03:15):** narrow delta — simic/PPO core + telemetry UNCHANGED. `proof_packet.py` 432→664 lines (+232: lockstep A/B baseline checks, more verdict branches), `test_proof_packet.py` +170 (18 new tests), `test_vectorized_correctness.py` +61 (pins TRAINING_STARTED proof-baseline provenance via new `leyline.proof_baselines` constants). python-code-reviewer: still fails closed, no false-green path, tests uniformly strengthened. No new 🔴. Two minor coverage/docs items (F8, F9).
- **Pass 5 (03:38):** big delta — runtime proof-baseline mode added. `vectorized_trainer.py` +9→+216 (fixed-schedule lifecycle execution), `proof_packet.py` 664→968, new `leyline/proof_baselines.py` (+100) + `simic/training/proof_baselines.py` (+47), telemetry payload fields, new `test_proof_baselines.py` (+182). drl-expert: normal training path SAFE/unchanged (gated behind `proof_baseline_lifecycle_policy=None`), baseline design non-contaminating (mask-forced sampling + `forced_step` zeroes actor grad), deterministic + hash-pinned, fail-closed on invalid action. python-code-reviewer: proof_packet still fails closed. Tests strengthened. No new 🔴. New: F10 (hash not pinned), F11 (checkpoint format break); F8 refined.
- **Pass 6 (04:00):** `proof_packet.py` 968→1345, telemetry.py +71 (TopologyManifest payloads), new `static_final_replay.py` module + `vectorized_trainer.py` +216→+324 (static-final replay WIRED IN at line 60/783). Normal training re-verified SAFE — static-final replay gated behind `proof_baseline_lifecycle_policy == "freeze_replayed_final_topology"` (default None); runner REFUSES to fake a frozen control (vectorized_trainer.py:222-233). proof_packet still fails closed. Tests strengthened. **First 🔴: F12 (latent `state_dict_exact` impostor replay).** New 🟡: F13 (source_group_id false-close). NOTE: a delegated reviewer wrongly claimed static_final had "zero importers" — caught by cross-check; ALWAYS targeted-diff core training files (the aggregate diffstat number column misled 216 vs actual 324).
- **Pass 12 (07:33):** trivial — core source UNCHANGED (all training-loop files at pass-11 values). Delta = test-only: new `test_ppo_ratio_metrics.py` (+28, pins per-head clip-fraction causal masking — masked timesteps don't count as clipped for sparse heads) + test_proof_packet +45. Weakening scan clean. All watches stable. No findings. (Untracked-hash shift this pass = my own new spec doc `docs/superpowers/specs/2026-06-15-weft-ci-migration-design.md`, not agent code.)
- **Pass 11 (07:18):** core mostly flat; `ppo_coordinator.py` +5 (new hunk) = (a) removed a defensive `.get("explained_variance", 0.0)` → direct fail-loud access, (b) new `value_collapse_applicable = usable_actor_timesteps > 0` so the anomaly detector skips false value-collapse alarms on fully-forced control cohorts (needed for the proof regime). New tests pin both sides + a new test_anomaly_detector. Reviewed inline (small, clearly correct). No weakening (the `-9.0` removals are the pass-1 forfeit-prefix corrections). Watches stable. No new findings.
- **Pass 10 (06:52):** quiet — core training loop essentially flat (vectorized_trainer +6, all gated behind `proof_baseline_lifecycle_policy`; contribution/action_execution/ppo_coordinator/proof_baselines/vectorized unchanged). Delta = test hardening (test_proof_packet +75, test_ppo +69 [KL early-stop, value-clip, head grad-norm finiteness, BPTT invariant], test_vectorized_correctness +16). Weakening scan clean (only the known taxonomy migration). All watches stable: F12 constant still present-but-unemitted, F10 unpinned, F17 flush still 10s, HEAD still uncommitted. No new findings.
- **Pass 9 (06:25):** `vectorized_trainer.py` +57, `vectorized.py` +43, `proof_baselines.py` +8, `proof_packet.py` big (+1550/-41), heavy test growth (test_proof_packet +162, test_views/proof_baselines +53 each). drl-expert: normal path safe/gated; **NEW FIX — `_finalize_run_scoped_nissa_backends` removes per-run telemetry backends from the process-global hub (was a real double-emit/thread-leak across sequential in-process runs — matters for multi-cohort proof runs).** python-code-reviewer: proof_packet still fails closed; **F12 test-drift RESOLVED (topology_only CONTINUE happy path now tested)**. F10 still open. New: F16 (dead empty-tuple guard, trivial), F17 (new telemetry-flush-timeout-fails-run behavior, watch). Tests strengthened.
- **Pass 8 (05:58):** quiet pass — mostly TEST hardening (test_vectorized_correctness +74, test_proof_baselines +69, test_views +50, test_output +25); small source delta (vectorized_trainer +7, proof_baselines +18); proof_packet + its tests UNCHANGED. drl-expert: delta safe, normal path still gated, cohort isolation/determinism intact. Watches clean: F12 not re-escalated (only emitter still TOPOLOGY_ONLY), F10 still unpinned. No new 🔴. New trivial F15 (source forced steps unflagged). Tests strengthened (removed `support[...] is False` lines are legit — those cohorts are now supported).
- **Pass 7 (04:23):** all four proof-baseline cohorts flipped to `current_runner_supported=True`; new source-cohort orchestration (`_train_static_final_source_cohort`, `_extract_static_final_source_manifest`) in proof_baselines.py (+47→+144); vectorized_trainer.py +324→+399; proof_packet.py grew again (still fails closed, verdict classifier re-traced). **F12 escalation watch FIRED (static_final supported) but resolved benignly — runner uses honest `topology_only`, not `state_dict_exact`** → F12 downgraded 🔴→🟡. drl-expert: normal path safe (gates at 202/842/2278/2018), source cohort is a real learned run capturing FINAL topology, replay genuinely frozen, lockstep A/B correctly seed-paired (reward = sole variable), cohorts isolated. python-code-reviewer: fail-closed, no false-green; F10 still open; new minors. 🔴 back to just F1. New 🟡: F14 (dead field). Untracked churn = warpline skill + weft.toml/loomweave.yaml (tooling/env, not agent code).

---

## 🔴 Needs your decision

### F1 — Validate the two deferred P2 fixes with a retraining/ablation run
The agent implemented two fixes the *prior* author had deliberately deferred,
each flagged in-code as needing empirical validation:

- **Rollback tail-dominance** (Filigree `0a2fad920d`): a governor rollback now
  zeroes the entire episode's positive-reward prefix and keeps only the terminal
  penalty. Code is correct & internally consistent (drl-expert verified); but it
  *materially reshapes the catastrophe reward signal*. Prior comment: "needs a
  retraining run to tune."
- **Op-head FP32+floor sampling** (Filigree `05b4113bc1`): `forward()` now samples
  the op from the same FP32+floored distribution it's scored against (off-policy-ness
  removed; wiring + parity verified by pytorch-expert). Prior TODO: "changes the
  value-conditioned forward graph and needs its own parity gate/review."

**Action:** Both are *correct code* but *unvalidated training dynamics*. Per the
agent's own `bf16-artifact-quarantine.md` proof rule, don't cite them as "policy
trains well" until an ablation/retraining run on a precision-proven (post-cutoff)
config confirms it. Decide: schedule the run, or accept on faith.

**Pass-3 update:** the agent is actively building the *machinery* to run this
validation honestly — `proof_packet.py` now emits typed verdicts and forces the
precision-provenance gate under the reward-efficiency profile; new planning docs
(`correctness-proof-strategy`, `ppo-stability-oracle-sandbox`, `proof-baseline-controls`)
lay out the evidence ladder. PLAN_TRACKER still marks the reward-efficiency
experiment **deferred** (not done). So F1's *tooling* is being addressed; the
*empirical retraining run itself is still pending*. This is the rigorous path.

---

## 🟡 Minor / optional (no action required)

### F12 — `state_dict_exact` is a dead/unused replay policy + test-fixture drift (pass 6 🔴 → pass 7 🟡 DOWNGRADED)
**Pass-7 resolution:** `static_final` is now `current_runner_supported=True`, BUT the runner
emits `replay_weight_policy=TOPOLOGY_ONLY_REPLAY_WEIGHT_POLICY` (vectorized_trainer.py:643) —
the honest policy that accurately describes germinate-fresh-weights topology replay. So the
proof is NOT faked: the supported path is honest. What remains is two latent traps:
(a) `STATE_DICT_EXACT_REPLAY_WEIGHT_POLICY` (static_final_replay.py:17) is declared,
validated against, and SQL-accepted, but **no caller ever emits it** — a dead constant that
would falsely validate (manifest hash is topology/param-count only, no weight tensors) if
anyone ever used it. Delete it, or `# TODO: [FUTURE FUNCTIONALITY]` + implement real
state-dict replay with a weight-tensor hash. **(b) RESOLVED pass 9** — the `topology_only`
CONTINUE happy path is now exercised (`test_proof_packet_accepts_static_final_with_joined_topology_manifests`),
so F12 reduces to just the dead-constant cleanup (a). Minor cosmetic: proof_packet emits a
redundant double-violation for a `state_dict_exact` row (still BLOCKED_MATH; harmless).

### F14 — dead `VectorizedPPOTrainer.proof_baseline_mode` field (pass 7, trivial)
`proof_baseline_mode` is declared (vectorized_trainer.py:516) and passed in, but never read
inside the trainer — the mode already reaches telemetry via `TrainingStartedPayload`. Dead
field; drop it or consume it (no-dead-code policy). Trivial.

### F16 — dead `unsupported_policies` empty-tuple guard (pass 9, trivial)
With all proof-baseline policies now real branches, the residual `unsupported_policies: tuple = ()`
guard + its `if ... in` / raise in `apply_proof_baseline_action_controls` (vectorized_trainer.py
~292) is unreachable dead code. Delete per no-legacy policy. Trivial.

### F17 — telemetry-flush-timeout now FAILS an otherwise-good run (pass 9, watch)
The new `_finalize_run_scoped_nissa_backends` (vectorized.py:607) fixes a real bug (per-run
`BlueprintAnalytics`/`DirectoryOutput` were added to the process-global hub and never removed
→ double-emit + thread leak across sequential in-process runs; matters for multi-cohort proof
runs). GOOD fix. Side effect: on a successful run WITH `telemetry_dir`, `require_telemetry_flush=True`
raises `RuntimeError` if `hub.flush(timeout=10s)` stalls — a new failure mode for ALL
telemetry-writing runs, not just proof baselines. Defensible (proof-evidence integrity), but
worth knowing: a 10s telemetry drain stall now fails the run. Runs without `telemetry_dir`
never flush/raise. Watch whether the 10s timeout proves too tight under real load.

### F15 — source-cohort forced steps not flagged `proof_controlled_step` (pass 8, trivial)
The source-preflight policy `evolve_source_final_topology` forces actions every step (via
`_force_scheduled_action_masks`) but is NOT in the `proof_controlled_step` tuple
(vectorized_trainer.py:2022), so those forced steps contribute to the *source* agent's actor
loss. Harmless today (the source policy is discarded — only its topology manifest is used),
but a latent inconsistency: add `STATIC_FINAL_SOURCE_LIFECYCLE_POLICY` to the tuple, or a
comment noting the source policy is intentionally discarded.

### F2 — bf16 rounding asymmetry at masked op-logit positions
In `factored_lstm.py` op sampler, masked entries are a bf16-rounded `-1e4`
(e.g. `-9984.0`) rather than exact `-1e4`, because the sampler `.float()`s the
already-bf16-masked logits instead of re-masking in FP32 like the scorer does.
Negligible (masked positions are overwritten to `MASKED_LOGIT_VALUE` on output;
softmax mass ~0 either way). Optional tidy for perfect sampler/scorer symmetry.

### F3 — Stale comment in `contribution.py` (~line 598–600)
References the old `is not None` distinction for fields that are now non-optional
floats. Pre-existing (not introduced by this diff), spotted during review.
(Note: pass 2 made `SeedInfo.total_improvement` properly Optional, so this comment
is now closer to accurate — re-check whether it still needs touching.)

### F4 — Duplicated `None`-guard helpers (pass 2, trivial)
`_require_total_improvement` (`contribution.py:357`) and `_fossilized_total_improvement`
(`fossilize.py:50`) duplicate the same raise-on-None logic across modules. Both are
correct; could share one helper. Pure tidiness — dump-able.

### F5 — `RunningMeanStd` import path inconsistency (pass 2, trivial)
`normalizer_checkpoint.py` imports `from esper.simic.control import RunningMeanStd`
while its test imports `esper.simic.control.normalization.RunningMeanStd`. Both
resolve to the same class. Cosmetic.

### F6 — `VitalSigns.accuracy_improving` defaults `True` when val_accuracy is always absent (pass 3, minor)
In `karn/health.py`, if a run never emits `val_accuracy`, the accuracy branch is
skipped and `accuracy_improving` keeps its dataclass default `True` forever (and
stagnation never fires). Correct for the stagnation gate (can't declare stagnation
on absent data), but `accuracy_improving=True` is a *false positive* for anything
reading the field directly (Overwatch/Tamiyo?). Suggest an `accuracy_available: bool`
field or `bool | None`. Not a proof-integrity hole. (Worth a `entity_callers_list`
on `check_vitals` to size downstream impact.)

### F7 — proof_packet verdict ordering can mask precision blocker (pass 3, minor)
`_classify_verdict` checks instrumentation/runs/outcomes/learnability before
`precision_blockers`, so a run missing BOTH learnability telemetry AND precision
provenance reports top-line `BLOCKED_INSTRUMENTATION`, not `BLOCKED_PRECISION`.
Conservative (no false green; the Precision section still lists the blocker), but
the top-line string could send someone fixing instrumentation while missing the
precision gap. Consider compound/ordered reporting.

### F8 — proof_packet profile vs explicit-arg handling (pass 4, refined pass 5, minor)
Two facets of the same root issue (the profile silently controls gate params):
(a) `build_proof_packet()` defaults `proof_profile=generic` while the CLI defaults
to `reward-efficiency` — a programmatic caller omitting `proof_profile` silently
gets the weaker gate. (b) Conversely, `proof_profile="reward-efficiency"` silently
OVERRIDES a caller's explicit `require_blueprint_health_baselines=False` (and
precision) — only makes the gate *stricter* (not a false-green), but it's a silent
ignore of a passed argument; the project's anti-defensive-programming stance would
prefer a `ValueError`. Fix: raise on the conflict, or document the precedence.
Also cosmetic: the precision query's non-blocking `CASE` ELSE branch
(`'precision provenance present'`) is dead under its WHERE clause — a label trap.

### F9 — missing happy-path test for reward-efficiency CONTINUE (pass 4, minor)
No test exercises `proof_profile="reward-efficiency"` + complete baselines + valid
outcomes → `CONTINUE`. All the profile's *negative* cases are covered; the sunny
day for the primary CLI profile is not. Coverage gap, not a safety gap — a
regression against the happy path wouldn't be caught.

### F10 — fixed-schedule hash not pinned to an expected value (pass 5, MAJOR-latent)
`FIXED_SCHEDULE_GERMINATE_R0C0_HASH` (`leyline/proof_baselines.py`) is computed at
import time from `FactoredAction.to_indices()` (enum ordinal positions), but there
is no `assert HASH == "<known value>"`. If any factored-action enum (`LifecycleOp`,
`BlueprintAction`, `GerminationStyle`, …) is ever reordered, the hash silently
shifts — and all previously-recorded telemetry tagged with the old hash would then
read as hash-mismatched, producing spurious proof blockers. Not a current bug
(nothing reordered yet), but it's an integrity pin that isn't itself pinned. Fix:
add a module-level expected-hash assertion. **Most substantive new item this pass.**

### F11 — obs-normalizer checkpoint format break (pass 5, minor/operational)
The normalizer-checkpoint refactor changes the on-disk metadata format for ALL runs
(adds `obs_normalizer_contract`) and makes restore fail-loud on mismatch — so
pre-refactor checkpoints will no longer resume. This is an intended improvement
(fail-closed, consistent with the bf16-quarantine: old checkpoints already
quarantined), but it's an operational gotcha and arguably scope creep bundled into
the proof-baseline diff. Heads-up, not a defect.

### F13 — `source_group_id IS NULL` false-closes valid static-final proofs (pass 6, minor/MAJOR-latent)
In `_static_final_replay_blockers_query` (proof_packet.py ~688), a replay whose source
run legitimately has no `group_id` (single-env/ungrouped) is flagged
`'missing source_group_id'` and blocks (`BLOCKED_MATH`) — even though the join already
matches the source via `COALESCE(group_id,'')`. This is FALSE-CLOSED (blocks a valid
proof; never produces a false green), so it's conservative-but-wrong. Latent (static_final
unsupported). Fix: gate the NULL check on `source_event_id IS NOT NULL`, or rely on the
`matched_source_event_id IS NULL` join check. Only matters once real runs may omit group_id.

---

## ✅ Cleared — no action
- "Correctness push" commit `82155c14`: doc-only (legis-workflow SKILL.md). Vague message, harmless.
- `train.py` Karn dedup: correct (removed phantom 2nd collector; singleton still bound).
- `ISOLATION_VIOLATION` removal: clean dead-code deletion.
- Telemetry enrichments (rollback attribution, `blending_delta`, EPISODE_OUTCOME counts): additive; new Karn view fields surface already-emitted data.
- `blending_warning` → counterfactual gate, `holding_warning` guard removal: correct (removed guard was dead code).
- `bf16-artifact-quarantine.md`: good provenance discipline.
- **(pass 2) `normalizer_checkpoint.py` — latent-bug FIX, not just a refactor.** The old inline restore blindly loaded obs-normalizer stats whenever the key existed, so resuming a checkpoint with a *different* slot count / Obs-V3 schema would silently corrupt normalization. New code is contract-gated: mismatch or missing keys **raise** (fail-closed), round-trip is lossless (Welford + EMA modes, correct device/dtype/shape). Pinned by new fail-closed tests. Net safety improvement.
- **(pass 2) `SeedInfo.total_improvement`/`improvement_since_stage_start` → `float | None`.** "Not measured yet" is now a real state vs masquerading as `0.0`. `None` never escapes `from_seed_state` in production (`SeedMetrics` always truthy, property returns float), every consumer guards it, raises are fail-loud contract assertions (not bug-hiding). Behavior-preserving for production; tests strengthened.
- **(pass 2) `record_stream` test:** pins a real CUDA stream-ordering invariant (record_stream protects lifetime, not ordering; production correctly pairs it with `wait_stream`). Genuine regression guard, not a tautology.
- **(pass 2) `OBS_V3_UNKNOWN_SENTINEL` moved into leyline + `OBS_V3_FEATURE_SCHEMA_VERSION` added:** correct per "shared constants live in leyline."
- **(pass 3) `proof_packet.py` typed verdicts + precision gate:** fails closed — reward-efficiency profile *forces* `require_precision_provenance=True` (overrides caller `False`); missing provenance → `BLOCKED_PRECISION` before any ROI eval. No false-green path. Tests strengthened (`"BLOCKED"` → exact verdict constants).
- **(pass 3) Karn telemetry `val_accuracy: float=0.0 → float|None`:** same "None = not measured" doctrine as the SeedInfo widening; `is not None` guards are contract updates, not bug-hiding. A known limitation comment was promoted to a tested property. Tests strengthened.
- **(pass 3) Docs honest:** `cifar_blind → cifar_impaired` removes an unsubstantiated boast; reward-efficiency experiment still marked **deferred**; `proof-baseline-controls` openly declares static-final/fixed-schedule unsupported until replay/schedule machinery exists.
- **(pass 4) `proof_packet.py` 432→664 lines still fails closed:** full verdict classifier re-traced — no false-green path; lockstep A/B baseline checks added; SQL validity predicates stay symmetric with diagnostics; no bug-hiding. 18 new tests, all strengthening (acceptance now requires *outcome-bearing* evidence).
- **(pass 4) `test_vectorized_correctness.py` +61:** pins `TRAINING_STARTED` emitting full proof-baseline lifecycle/schedule provenance (new `leyline.proof_baselines` constants). Real provenance regression guard.
- **(pass 5) Runtime proof-baseline mode is safely gated:** `vectorized_trainer.py` +216 — default (no-baseline) training path byte-for-byte unchanged; fixed-schedule cohort forces the action mask to one valid candidate per head (policy still samples → consistent log_prob) and marks the step `forced_step=True` so it's excluded from the PPO actor gradient → scripted lifecycle ops don't contaminate policy learning. Deterministic, hash-pinned, fail-closed (atomic mask restore on invalid action). Rollback telemetry change is read-only (no reward-math change). drl-expert verified.
- **(pass 5) `proof_packet.py` at 968 lines still fails closed:** new lockstep/baseline-provenance/fixed-schedule-trace gates, predicate symmetry intact, no false-green path. Tests strengthened (typed verdict constants throughout).
- **(pass 6) Static-final replay wired into trainer but normal training still SAFE:** `_materialize_static_final_replay` is gated at vectorized_trainer.py:782 behind `proof_baseline_lifecycle_policy == "freeze_replayed_final_topology"` (default None). The runner REFUSES (RuntimeError, lines 222-233) to mask op→WAIT without validated replay — explicitly declining to fabricate a fake INITIAL-topology control. Strong anti-proof-faking discipline (the F12 `state_dict_exact` gap is the one inconsistency with it).
- **(pass 6) `proof_packet.py` at 1345 lines still fails closed:** new static-final-replay + topology-manifest gates, `_classify_verdict` ordering re-traced (instrumentation→precision→math→mechanics→ROI), predicate symmetry intact, no false-green. TopologyManifest payload round-trip pinned. Tests strengthened.
- **(pass 11) Anomaly detector no longer false-alarms on control cohorts:** `value_collapse_applicable` gates value-collapse diagnosis on `usable_actor_timesteps > 0`, so fully-forced proof-control rollouts (which have no actor agency) don't trip a spurious value-collapse anomaly. Also removed a defensive `.get("explained_variance", 0.0)` (now fail-loud). Tests added both sides.
- **(pass 9) Real telemetry-leak fix:** `_finalize_run_scoped_nissa_backends` removes per-run backends from the process-global hub — eliminates cross-run double-emit + worker-thread leak across sequential in-process `train_ppo_vectorized` calls (the multi-cohort proof regime). proof_packet still fails closed; `topology_only` happy path now tested. (New flush-timeout failure mode tracked as F17.)
- **(pass 7) All four proof-baseline cohorts now runner-supported & verified correct:** static_final enabled HONESTLY (topology_only replay of a real source cohort's FINAL topology; control genuinely frozen via op→WAIT + actor-grad-zeroed forced steps); lockstep A/B correctly seed-paired (reward is the only variable); each cohort an isolated `train_ppo_vectorized` run (own seed/model/optimizer) — no cross-cohort or main-policy contamination. Normal training path re-confirmed gated/unchanged. proof_packet still fails closed. Tests strengthened (refusal paths preserved by flipping support flags in tests).
</content>
</invoke>
