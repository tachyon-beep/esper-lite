# Post-P0-1 Hardening Sprint

```yaml
# Plan Metadata
id: post-p01-hardening-sprint
title: Post-P0-1 Hardening & Integration Sprint
type: ready
created: 2026-06-18
updated: 2026-06-18
owner: john

# Prioritization
urgency: high
value: >
  Lands the op-independent V(s) baseline (P0-1) cleanly onto main with honest
  value-fit telemetry under the new estimator and a refreshed, vuln-free
  dependency surface — turning a strong-but-noisy A/B result into a trustworthy,
  shippable baseline that no longer false-blocks the proof/experiment verdict.

# Constraints
complexity: L
risk: high
risk_notes: >
  Carries a checkpoint-breaking change (VALUE_HEAD_SCHEMA_VERSION=2) across the
  largest divergence main has seen (46 commits / 206 files as of `0.1.1`@05f7706c;
  re-run `git rev-list --count main..0.1.1` immediately before execution). EV telemetry change
  touches a real proof-blocking gate; a wrong threshold either re-introduces the
  false-alarm or hides a genuine value-collapse. Dependency bump pass spans 80
  alerts; a transitive regression could destabilise CI during the merge window.

# Dependencies
depends_on: []

soft_depends:
  - recurrent-ppo-multiepoch          # P0-1 lives on top of multi-epoch PR1/PR2

blocks: []

# Status
status_notes: >
  Umbrella authored 2026-06-18. Child design specs for items (1) and (2) drafted;
  child TDD plans pending. Dependency triage complete (R3, inlined below).
percent_complete: 5

# Expert Review (REQUIRED before promotion to ready)
# Child plans carry their own sign-offs (see Child Artifacts). The entries below
# cover the cross-domain decisions that live ONLY in this umbrella — the
# item-1-before-item-2-before-item-3 sequencing and the inlined dependency
# triage — which the child sign-offs do not reach (per CLAUDE.md specialist-review
# mandate and §Operating Rules promotion rule). Child sign-offs do not cover the
# umbrella's unique decisions, so they are recorded here explicitly rather than
# promoted.
reviewed_by:
  - reviewer: drl-expert
    date: 2026-06-18
    verdict: approved-with-changes
    scope: umbrella EV-gate ordering decision
    notes: >
      Approves §Sequencing item (1): EV-telemetry robustness MUST land on 0.1.1
      before the merge — merging P0-1 onto main while the EV artifact still
      hard-codes proof_blocking=true (views.py:649) would import a known
      false-blocker into the release branch. Required (folded into this umbrella):
      (1) the EV plan's Step 0 empirical floor calibration is a HARD PRECONDITION
      for merging the variance-floor default, not optional pre-flight; (2) the
      sprint exit criteria must include a genuine-collapse falsification/latency
      test (frozen value weights → VALUE_COLLAPSE_DETECTED fires within N updates
      under K=4) so the stricter AND-gate does not swallow a real collapse; (3) the
      downstream EV-consumer thresholds (karn/constants.py:120-121,
      ExperimentVerdictPanel.vue:129) were calibrated against the OLD Q(s,op)
      estimator and need a SEPARATE recalibration task under the op-marginal V(s) —
      stopping the false alarm is not the same as recalibrating every consumer.
  - reviewer: axiom-python-engineering
    date: 2026-06-18
    verdict: approved-with-changes
    scope: umbrella integration-sequencing & dependency-triage classification
    notes: >
      Approves §Sequencing items (2)/(3): schema/merge work precedes the
      dependency manifest/lockfile bump (the npm step can touch package.json,
      not lockfiles only); the bump is the LAST, independently revertable commit of
      the window. Required (folded in): the §3 dependency triage must NOT conflate
      dev-group with optional-extras vs core-runtime. datasets>=4.4.1 is a
      top-level [project].dependencies entry (pyproject.toml:11), so pyarrow (#77,
      high, UAF) is a REQUIRED-runtime transitive — reclassified and escalated
      below. Added an explicit flaky-vs-regression time-box so a single flaky
      transitive cannot block all three coordinated items.
```

## Strategic Intent

P0-1 replaced the op-conditioned `Q(s, op)` critic with an op-independent `V(s)`
baseline (commits `6a27b8e3`, `177a53aa`, `6d97391b`, `31bf8cb7`). The A/B was a
clear win on the things that matter: it fixed op-head starvation
(`head_op_grad_norm` late-alive fraction K=4 control 0.05 → P0-1 0.89), widened
the K=4−K1 explained-variance gap to +0.201 (control +0.137), and produced both
the best median EV (+0.239) and the lowest `value_loss` (0.099) of all arms.

But the win arrived **noisy and unmerged**. The headline `explained_variance`
std blew out (P0-1 K=4 EV mean −0.337, std 2.264 vs control 0.572) — a
*denominator artifact* of the EV estimator under an op-marginal critic, not a
value-head instability. That artifact currently feeds a real proof-blocking
telemetry gate. Meanwhile the win still lives only on `0.1.1`, which is 46
commits ahead of `main` (as of `0.1.1`@05f7706c; re-confirm before execution) and
carries a checkpoint break, and the default branch
carries ~80 open Dependabot alerts.

This sprint is **post-P0-1 hardening and integration**: make the new estimator's
telemetry honest, land the whole `0.1.1` line on `main` with a clean checkpoint
story, and clear the dependency surface — in one coordinated window.

## Child Artifacts

| Item | Spec | Plan |
|------|------|------|
| (1) EV-telemetry robustness | [docs/superpowers/specs/2026-06-18-ev-telemetry-robustness-design.md](../../superpowers/specs/2026-06-18-ev-telemetry-robustness-design.md) | [docs/plans/ready/2026-06-18-ev-telemetry-robustness-plan.md](./2026-06-18-ev-telemetry-robustness-plan.md) |
| (2) 0.1.1 → main merge | [docs/superpowers/specs/2026-06-18-main-merge-integration-design.md](../../superpowers/specs/2026-06-18-main-merge-integration-design.md) | [docs/plans/ready/2026-06-18-main-merge-integration-plan.md](./2026-06-18-main-merge-integration-plan.md) |
| (3) Dependency vuln triage | *(inline below — no separate spec/plan)* | *(inline below)* |

**Child sign-offs (for reference; do not cover this umbrella's unique decisions):**

- EV-robustness plan: `drl-expert` (approved-with-changes), `axiom-python-engineering` (approved).
- Merge-integration plan: `axiom-python-engineering` (approved), `plan-review-systems` (approved-with-changes).

The umbrella's cross-domain sequencing and the inlined dependency triage are
reviewed separately above (`reviewed_by`), because they exist only here.

### Item (1): EV-Telemetry Robustness

**Problem.** `explained_variance` is computed once per PPO update, pre-update, in
`src/esper/simic/agent/ppo_agent.py:619-631` as
`EV = 1 − Var(returns − denorm(values)) / Var(returns)` over the valid-mask
flattened batch, with a `Var(returns) > 1e-8` guard that returns `0.0` on
degenerate batches. The denominator is the *batch return variance* (`var_returns`
at line 622). On low-return-variance batches the tiny denominator makes EV crater
to −8 even though the absolute residual — and `value_loss` — is **small** on those
very updates. P0-1's op-marginal `V(s)` is *more* exposed than the old `Q(s, op)`,
which used to absorb cross-op spread that inflated the denominator.

**Goal.** Make EV an honest diagnostic under the new estimator **without hiding
any real value-fit problem**. A return-variance guard is legitimate numerical
handling, *not* defensive bug-masking (the spec argues this distinction
explicitly per CLAUDE.md). Candidate directions: (a) guard/clamp/winsorize EV
when `return_std` is below a principled floor and emit a flag; (b) emit a
return-variance-robust value-fit metric alongside raw EV (normalized RMSE, or EV
with a variance floor). Robust signals already emitted and immune to the
artifact: `v_return_correlation`, `bellman_error`, `td_error_mean/std`
(`leyline/value_metrics.py`), plus `value_loss`, `value_mean/std`,
`return_std/return_mean`.

**Step 0 (empirical floor calibration) is a HARD PRECONDITION, not pre-flight.**
The EV plan's Step 0 — calibrate the variance-floor default against the real
return-variance distribution via Karn MCP, and confirm the low-variance tail's
value fit is genuinely healthy — MUST complete and its result MUST be recorded
before the variance-floor default is merged. Setting the floor without the
empirical pass is the *quick fix* (Shifting-the-Burden): it would silence the
false alarm on an ungrounded number that can re-trigger or, worse, mask a real
low-variance collapse. Treat "floor merged without Step 0 evidence" as a hard
stop.

**The fundamental solution is wider than this sprint — name it so it is not
treated as done.** The variance-floor patch stops the *artifact-driven* false
alarm. It does **not** recalibrate every EV consumer to the op-marginal `V(s)`
estimator's expected EV range. The downstream display/verdict thresholds were
calibrated against the OLD `Q(s,op)` estimator and are now miscalibrated:

- `src/esper/karn/constants.py:120-121` —
  `EXPLAINED_VAR_WARNING=0.3` / `EXPLAINED_VAR_CRITICAL=0.0` (consumed by the
  Sanctum panels at `ppo_losses_panel.py`, `critic_calibration_panel.py`,
  `narrative_panel.py`).
- `src/esper/karn/overwatch/web/.../ExperimentVerdictPanel.vue:129` — hardcoded
  `0.3` verdict threshold.
- `anomaly_detector.py:66-69` phase-dependent thresholds and the
  `HealthGauges.vue:86` clamp (the clamp removal is in the EV plan's scope; the
  threshold recalibration is **not**).

Recalibrating these against the new estimator's empirical EV distribution is a
**separate, named follow-up task** (`P-EV-RECAL`, to be filed once Step 0 yields
the new EV distribution). It is *out of scope for this sprint* but is the
fundamental solution; stopping the false-block in the gate must not be silently
recorded as "EV consumers recalibrated."

**The part with teeth — consumer audit.** There is exactly one real **gate** on
the EV value: the `AnomalyDetector` value-collapse check
(`ppo_coordinator.py:512-524` → `anomaly_detector.check_value_function`,
`anomaly_detector.py:140-175`, phase-dependent thresholds −0.5 / −0.2 / 0.0 / 0.1).
It is gated by `value_collapse_applicable = usable_actor_timesteps > 0`, does
**not** halt training, and emits a `VALUE_COLLAPSE_DETECTED` event plus a
temporary telemetry-verbosity escalation (`vectorized.py:480-493`). That event
then feeds two Karn DuckDB views **with teeth**: `anomalies` (`views.py:583-605`)
and `run_confounders` (`views.py:606-662`), where every `VALUE_COLLAPSE_DETECTED`
row is hard-coded `proof_blocking=true` (line 649). So a K>1 EV-std blowout will
**spuriously block the proof/experiment verdict**. Remaining consumers are
display-only (Sanctum TUI panels, Overwatch web gauges/verdict panel) but several
use absolute EV thresholds that will now false-watch/false-fail. The spec
enumerates each consumer with file:line.

**Stricter gate, not blinder gate — and lock the detection latency.** The EV
plan's Step 5 migrates the gate to fire only on **both** a bad robust signal
**AND** a non-flagged EV. This removes the false alarm but introduces a
*Fixes-that-Fail* failure mode: a genuinely collapsed value head is now harder to
detect, and if the robust signals (`v_return_correlation`, `bellman_error`) are
noisy or slow during real collapse, the stricter AND-gate can *delay or swallow*
the firing, letting training run further into a degenerate state. The §Risks
table acknowledges the outcome ("EV guard hides a real value collapse"); the
*feedback structure* is the reinforcing loop B1 (anomaly → intervention) being
delayed by the new conjunction. **Mitigation is a falsification/latency test in
the sprint exit criteria** (see Success Measure 6): a synthetic genuine collapse
(value weights frozen, low `v_return_correlation`, `ev_low_return_variance`
False) must fire `VALUE_COLLAPSE_DETECTED` within N updates under K=4 — locking
the detection latency, not merely asserting that the false alarm stops.

**Observability — run_confounders requires a JOIN for full context.** The gate's
anomaly check is *synchronous* in the training loop (`ppo_coordinator.py:512-524`)
and emits the `VALUE_COLLAPSE_DETECTED` event, which drives the event-driven
`anomalies` / `run_confounders` views. The companion robust-signal columns
(`value_nrmse`, `ev_low_return_variance`, `v_return_correlation`) persist via the
*buffered* `PPOUpdatePayload` (PPO_UPDATE_COMPLETED, `leyline/telemetry.py:701`)
→ `ppo_updates` path. There is a timing window
where a `proof_blocking` row is visible before the same-batch robust signals land.
Post-hoc analysis must JOIN `run_confounders` to `ppo_updates` on batch/epoch to
recover the robust-signal context that justified (or suppressed) a firing. This
is documented in the EV plan (Step 5 or Step 6) and the analysis query pattern is
provided there; it is a second-order observability note, not a false-alarm issue.

### Item (2): 0.1.1 → main Merge Integration

**Shape of the divergence.** The merge is a **linear fast-forward, not a real
merge**: `git merge-base main 0.1.1` == `main` HEAD == `f7f1aece`, so `main` is a
strict ancestor of `0.1.1`, `git rev-list --count 0.1.1..main` is `0`, and zero
commits exist on `main` that are absent from `0.1.1`. The divergence is 46
commits / 206 files / +21869 −3078 (as of `0.1.1`@05f7706c; re-run the
merge-base / `rev-list` checks immediately before execution), dominated by simic/training (the 2614/1550
`vectorized_trainer.py` transaction-phase refactor + multi-epoch recurrent PPO +
P0-1) plus leyline contract moves, karn/telemetry, and ~36 simic test files.

**Checkpoint break.** The *only* checkpoint-break surface is the new
`VALUE_HEAD_SCHEMA_VERSION=2` constant (absent on `main` entirely) gating PPO
value-head loads in `ppo_agent.py`. `CHECKPOINT_VERSION` was already `2` on
`main`, and the kasmina slot `_SCHEMA_VERSION=2` is unchanged. Migration story is
**retrain-only** (no remap, by design, per No-Legacy).

**CI / test posture.** Single `.github/workflows/test-suite.yml`
(lint / typecheck / property / unit+integration / overwatch-web / nightly) over
~384 test files / ~5092 test functions. `uv.lock` changed only the version string
(no dep bumps), so item (3) is fully orthogonal and folds into the same window
without conflict.

### Item (3): Dependency Vulnerability Triage — R3

*(Self-contained section; no separate spec/plan. Fold the bump pass into the
main-merge window — see Sequencing.)*

**Data source:** `gh api /repos/tachyon-beep/esper-lite/dependabot/alerts
--paginate` (80 open alerts). `pyproject.toml`, `uv.lock`, and
`src/esper/karn/overwatch/web/package.json` inspected for direct-vs-transitive
provenance.

#### 3.1 Open alerts by severity

| Severity | Count |
|----------|-------|
| Critical | 1 |
| High | 34 |
| Medium | 32 |
| Low | 13 |
| **Total** | **80** |

**Manifest split:** `uv.lock` = 71 alerts (Python).
`src/esper/karn/overwatch/web/package-lock.json` = 9 alerts (npm, Overwatch
dashboard build tooling).

#### 3.2 Critical + High triage

`direct?` = relationship to **pyproject.toml** (Python) or **web/package.json**
(npm). "transitive" = pulled by a declared dep, not declared itself.

| # | Sev | Package | Direct/Trans | Vulnerable range | Patched | GHSA | Upstream source / notes |
|---|-----|---------|--------------|------------------|---------|------|-------------------------|
| 78 | **CRIT** | vitest | **direct (npm devDep)** | >=4.0.0,<4.1.0 | 4.1.0 | GHSA-5xrq-8626-4rwp | Overwatch web test tooling; arbitrary file read when UI server listening. Dev-only, not shipped. |
| 36 | high | vite | **direct (npm devDep)** | >=7.0.0,<=7.3.1 | 7.3.2 | GHSA-p9ff-h696-f583 | Dev-server arbitrary file read. Dev-only. |
| 35 | high | vite | direct (npm devDep) | >=7.1.0,<=7.3.1 | 7.3.2 | GHSA-v2wj-q39q-566r | server.fs.deny bypass. |
| 106 | high | vite | direct (npm devDep) | >=7.0.0,<=7.3.4 | 7.3.5 | GHSA-fx2h-pf6j-xcff | fs.deny bypass (Windows). |
| 71 | high | js-cookie | transitive (npm) | <=3.0.5 | 3.0.7 | GHSA-qjx8-664m-686j | prototype hijack; web. |
| 12 | high | minimatch | transitive (npm) | >=9.0.0,<9.0.7 | 9.0.7 | GHSA-7r86-cg39-jmmj | ReDoS; web build. |
| 67 | high | GitPython | transitive (uv.lock) | <=3.1.49 | 3.1.50 | GHSA-mv93-w799-cj2w | via **wandb** (optional extra). |
| 64 | high | GitPython | transitive | <=3.1.48 | 3.1.49 | GHSA-v87r-6q3f-2j67 | via wandb. |
| 60 | high | GitPython | transitive | <=3.1.47 | 3.1.48 | GHSA-7545-fcxq-7j24 | via wandb. |
| 47 | high | GitPython | transitive | <3.1.47 | 3.1.47 | GHSA-x2qx-6953-8485 | via wandb. |
| 46 | high | GitPython | transitive | >=3.1.30,<3.1.47 | 3.1.47 | GHSA-rpm5-65cw-6hj4 | Command injection; via wandb. |
| 117 | high | cryptography | transitive | >=0.5.0,<48.0.1 | 48.0.1 | GHSA-537c-gmf6-5ccf | bundled OpenSSL; via pyjwt[crypto]/jupyter/wandb. |
| 57 | high | jupyter-server | transitive | <=2.17.0 | 2.18.0 | GHSA-5mrq-x3x5-8v8f | via **jupyter** (dev group). |
| 56 | high | jupyter-server | transitive | <=2.17.0 | 2.18.0 | GHSA-24qx-w28j-9m6p | via jupyter. |
| 55 | high | jupyter-server | transitive | <=2.17.0 | 2.18.0 | GHSA-5789-5fc7-67v3 | via jupyter. |
| 63 | high | jupyterlab | transitive | <=4.5.6 | 4.5.7 | GHSA-mqcg-5x36-vfcg | via jupyter. |
| 58 | high | jupyterlab | transitive | >=4.0.0,<=4.5.6 | 4.5.7 | GHSA-37w4-hwhx-4rc4 | via jupyter. |
| 49 | high | jupyterlab | transitive | <=4.5.6 | 4.5.7 | GHSA-rch3-82jr-f9w9 | via jupyter. |
| 61 | high | notebook | transitive | >=7.0.0,<=7.5.5 | 7.5.6 | GHSA-mqcg-5x36-vfcg | via jupyter. |
| 48 | high | notebook | transitive | >=7.0.0,<=7.5.5 | 7.5.6 | GHSA-rch3-82jr-f9w9 | via jupyter. |
| 59 | high | mistune | transitive | >=3.0.0a1,<=3.2.0 | 3.2.1 | GHSA-8mp2-v27r-99xp | ReDoS; via nbconvert/jupyter. |
| 51 | high | pillow | transitive | >=10.3.0,<12.2.0 | 12.2.0 | GHSA-pwv6-vv43-88gr | OOB write; via torchvision/jupyter. |
| 40 | high | pillow | transitive | >=10.3.0,<12.2.0 | 12.2.0 | GHSA-whj4-6x5x-4v2j | decompression bomb. |
| **77** | **high** | **pyarrow** | **transitive — REQUIRED RUNTIME (via datasets)** | >=15.0.0,<23.0.1 | 23.0.1 | GHSA-rgxp-2hwp-jwgg | **UAF. `datasets>=4.4.1` is a top-level `[project].dependencies` entry (pyproject.toml:11), NOT dev/optional. Reachable in standard (non-wandb, non-jupyter) installs that use `datasets` for CIFAR-10 loading. ESCALATED — clear first in the bump.** |
| 100 | high | pyjwt | transitive | <2.13.0 | 2.13.0 | GHSA-xgmm-8j9v-c9wx | JWK-as-HMAC forge; via jupyter/wandb. |
| 104 | high | python-multipart | transitive — runtime-OPTIONAL (via fastapi[dashboard]) | <0.0.30 | 0.0.30 | GHSA-5rvq-cxj2-64vf | quadratic parsing; via `fastapi` under `[project.optional-dependencies].dashboard` (pyproject.toml:25). Reachable iff the `dashboard` extra is installed — not dev-only. |
| 62 | high | python-multipart | transitive — runtime-OPTIONAL (via fastapi[dashboard]) | <0.0.27 | 0.0.27 | GHSA-pp6c-gr5w-3c5g | DoS; via fastapi[dashboard]. |
| 124 | high | starlette | transitive — runtime-OPTIONAL (via fastapi[dashboard]) | >=0.4.1,<1.3.1 | 1.3.1 | GHSA-82w8-qh3p-5jfq | via fastapi[dashboard]. |
| 119 | high | starlette | transitive — runtime-OPTIONAL (via fastapi[dashboard]) | <1.1.0 | 1.1.0 | GHSA-wqp7-x3pw-xc5r | SSRF; via fastapi[dashboard]. |
| 121 | high | tornado | transitive | <6.5.6 | 6.5.6 | GHSA-mgf9-4vpg-hj56 | via jupyter/notebook. |
| 120 | high | tornado | transitive | <6.5.6 | 6.5.6 | GHSA-3x9g-8vmp-wqvf | via jupyter/notebook. |
| 93 | high | urllib3 | transitive | >=1.22,<2.6.3 | 2.6.3 | GHSA-38jv-5279-wg99 | via requests/wandb/datasets. |
| 83 | high | urllib3 | transitive | >=1.0,<2.6.0 | 2.6.0 | GHSA-2xpw-w6gg-jr37 | decompression bomb. |
| 82 | high | urllib3 | transitive | >=1.24,<2.6.0 | 2.6.0 | GHSA-gm62-xv2j-4w53 | unbounded links. |
| 69 | high | urllib3 | transitive | >=1.23,<2.7.0 | 2.7.0 | GHSA-qccp-gfcp-xxvc | header leak across origin. |

#### 3.3 Direct-dependency mapping (pyproject.toml)

Of the 71 Python alerts, only **4** touch packages declared directly in
`pyproject.toml` *as the named package*, and **none of those four are HIGH**:

| Package | pyproject location | Alert(s) | Sev | Fix | Action |
|---------|--------------------|----------|-----|-----|--------|
| torch | `dependencies` `torch>=2.8.0` | #79, #80 | low, low | #79→2.10.0; #80 **no fix yet** | #79 already satisfiable (constraint allows >=2.10); #80 watch. No pyproject change. |
| pytest | `[dependency-groups] dev` `pytest>=7.0.0` | #41 | medium | 9.0.3 | Dev-only; bump in lock. |
| transformers | `dependencies` `transformers>=4.57.3` | #38 | medium | **5.0.0rc3 (pre-release)** | DO NOT auto-bump to an rc; leave pinned, revisit at 5.0 GA. |

**Important — the "only 4 direct" framing is about the *named* package only.** The
*transitive* highs are NOT all dev/optional. The provenance tiers below
(§3.5) govern runtime exposure. In particular `pyarrow #77` (high, UAF) is a
**required-runtime transitive** because its parent `datasets>=4.4.1` is a
top-level `[project].dependencies` entry, not a dev or optional extra.

Everything else (all 34 HIGH, the critical, the bulk of medium/low) is
**transitive**. The declared constraints (`>=`) are already broad enough that the
patched versions are reachable via a plain lockfile refresh — no pin edits
required.

#### 3.4 Recommended bump plan

**Python (uv.lock — no pyproject.toml edits required):**

1. Run `uv lock --upgrade` then `uv sync` to pull patched transitives. **Verify
   pyarrow advances to >=23.0.1 first** (the only required-runtime high, alert
   #77 via the top-level `datasets` dep). Then confirm the GitPython cluster
   (→3.1.50 via wandb), urllib3 (→2.7.0), pillow (→12.2.0),
   starlette/python-multipart (fastapi[dashboard]), tornado/jupyter chain,
   cryptography (→48.0.1), pyjwt (→2.13.0), aiohttp (→3.14.1), idna,
   python-dotenv, mistune, bleach.
2. If a transitive refuses to advance (held by an upstream constraint), pin it
   explicitly via a `[tool.uv] constraint-dependencies` / override entry rather
   than touching `dependencies`.
3. **Do not** bump transformers to 5.0.0rc3 (pre-release). Keep
   `transformers>=4.57.3`; the medium alert stays open until 5.0 GA.
4. torch #80 has no fix — accept/snooze with a note; #79 clears once the lock
   picks up >=2.10.

**npm (Overwatch web — the current high/critical alerts are all dev/build tooling):**

5. In `src/esper/karn/overwatch/web`: `npm update` (or targeted
   `npm install vite@^7.3.5 vitest@^4.1.0`) plus `npm audit fix` to clear vitest
   (critical), vite (3 highs), js-cookie, minimatch, postcss. These flagged
   packages are dev/build tooling and `node_modules` is not shipped; risk is to
   local dev/CI only. (Note: Overwatch *does* have runtime npm dependencies —
   Vue / ECharts / vue-echarts, `package.json:14-18` — and the built `web/dist`
   bundle IS shipped as package data via `pyproject.toml:43-45`; the current
   alerts simply do not land on those runtime deps.)

#### 3.5 Risk framing — three provenance tiers (do not conflate)

The headline "1 critical, 34 high" overstates **but does not fully discount**
runtime exposure. Classify by tier, not by "shipped vs not":

- **(a) Dev-group only** — `jupyter` / `notebook` / `wandb` / `pytest` chains
  (jupyter-server, jupyterlab, tornado, mistune, pyjwt-via-jupyter, GitPython,
  most urllib3 paths) plus the entire npm critical+highs (vitest, vite,
  js-cookie, minimatch). Reachable only in a dev/CI environment. Lowest priority.
- **(b) Runtime-OPTIONAL extras** — `fastapi[dashboard]` (starlette #119/#124,
  python-multipart #62/#104), `wandb[optional]`. Declared under
  `[project.optional-dependencies]` (e.g. `dashboard`, pyproject.toml:25).
  Reachable **iff the operator installs the extra** — not dev-only, not core.
  Medium priority.
- **(c) Core-runtime REQUIRED** — `datasets>=4.4.1` is a top-level
  `[project].dependencies` entry (pyproject.toml:11), so **`pyarrow` (#77, high,
  UAF) is a required-runtime transitive**, reachable in a standard install that
  uses `datasets` for CIFAR-10 loading. This is the **highest-priority high** in
  the set; clear it first in the lock refresh and confirm explicitly.

Net effort is still low (two lock refreshes), so there is no reason to defer;
bundle into the merge window. But the bump plan (§3.4) must verify the **(c)**
tier (pyarrow) clears, not just assume "all highs are dev/optional."

## Sequencing & Dependencies

The three items share one delivery window. Ordering within it is what controls
risk.

```
(1) EV-telemetry robustness  ──┐
   Step 0 calibration (HARD)   │
   land on 0.1.1 FIRST         │
                               ▼
(2) 0.1.1 → main merge  ───────────────────────┐
   schema/migration work first                 │
                                                ▼
(3) Dependency bump pass  ──────────── LAST step of the merge window
```

1. **Item (1) lands on `0.1.1` before the merge — Step 0 calibration gates it.**
   EV-telemetry robustness is a prerequisite for a *trustworthy* merged baseline:
   merging P0-1 onto `main` while the EV artifact still hard-codes
   `proof_blocking=true` (`views.py:649`) would import a known false-blocker into
   the release branch. The EV plan's **Step 0 empirical floor calibration is a
   HARD PRECONDITION** for merging the variance-floor default (do not merge the
   floor on an ungrounded number — that is the Shifting-the-Burden quick fix).
   Land (1) on `0.1.1` first so it travels into `main` with the merge — a
   fast-forward (item 2), so no extra integration cost. **Note the wider
   fundamental solution `P-EV-RECAL`** (recalibrate karn/constants.py:120-121 and
   ExperimentVerdictPanel.vue:129 to the new estimator) is *not* discharged by
   item (1); it is filed separately and must not be marked done when the gate
   stops false-blocking.
2. **Item (2) schema/migration work precedes item (3).** Sequence the
   `VALUE_HEAD_SCHEMA_VERSION=2` checkpoint-break and the functional merge
   first, get CI green, *then* do the dependency bump. This keeps a flaky
   transitive from blocking the functional merge and isolates a dependency
   regression from the schema change.
3. **Item (3) is the LAST step of the merge window.** After `0.1.1` code lands
   and tests are green, run `uv lock --upgrade` / `uv sync` + `npm audit fix` as
   a single revertable dependency manifest/lockfile commit (the npm step can
   mutate `package.json` alongside `package-lock.json`), then re-run the full suite. The
   merge already re-establishes CI posture, so validating the bumps is "free" in
   the same CI pass, and a regression reverts independently.
4. **Item (3) is otherwise orthogonal.** `uv.lock` on `0.1.1` changed only the
   version string vs `main` (no dep bumps), so the bump diff does not conflict
   with the merge content regardless of exact ordering.

### Flaky-vs-regression time-box (Tragedy-of-the-Commons guard)

The single coordinated window is a shared CI resource: a failure in any one leg
(full local suite + PR CI + post-push CI + dependency-bump re-run + K=4 smoke)
can appear to block all three items, and "fix on 0.1.1 before PR" *adds commits*
which re-trigger the merge plan's Step-1 FF-topology check. To stop a single
flaky test from holding the whole window, apply this rule when Step 3 (full local
suite) is red:

- **Red due to a real regression in the 46-commit body → HARD STOP.** Fix on
  `0.1.1` before the PR; re-run the merge plan's Step-1 merge-base / FF-topology
  check after the fix (new commits invalidate the prior FF verification).
- **Red due to pre-existing flakiness (not introduced by the 0.1.1 body) → an
  `xfail` is permitted ONLY under the full quarantine gate below; otherwise the
  merge STOPS.** A pre-existing flaky test must not silently wave the coordinated
  window through. Because this is a release window, a flaky-quarantine `xfail`
  requires ALL of the following before it is applied, recorded in the merge
  plan's step log AND in the tracker issue:
  1. **Reproduction note** — the bisect evidence proving the failure also fails
     on `main` HEAD (`f7f1aece`), with the exact command and observed
     flake rate (e.g. N/M runs).
  2. **Owner + tracker issue** — a filed issue (filigree) with a named owner,
     linked from the `xfail` reason string, not an anonymous "known-flaky" note.
  3. **Expiry / SLA** — an explicit deadline on the `xfail` (date or "before
     next release"); the quarantine is time-boxed, not permanent, and is
     re-asserted (not auto-renewed) if it lapses.
  4. **Explicit approval** — sign-off from the merge owner recorded in the step
     log before the `xfail` lands. No self-service quarantine during the window.

  If any of the four is missing, do NOT `xfail` and proceed — STOP the merge and
  resolve (fix the test, or obtain the missing approval/issue/SLA) first. A
  pre-existing flaky transitive blocking CI is preferable to shipping an
  unaccountable quarantine into the release.

Classify by `git`-bisecting the failure against `main` HEAD (`f7f1aece`): if the
test also fails on the base, it is pre-existing flakiness, not a 0.1.1
regression.

## Scope Boundaries

**In scope.** EV-telemetry honesty under the op-marginal critic + full consumer
audit (item 1); the fast-forward integration of `0.1.1` onto `main` with the
checkpoint-break migration story and CI posture (item 2); the Dependabot bump
pass for the 80 open alerts (item 3).

**Out of scope.** Any further change to the value-head architecture or the P0-1
estimator itself (P0-1 is *landed*, not under revision here). New RL features,
reward changes, or curriculum work. Re-running the P0-1 A/B (the empirical result
is taken as given). A real (non-fast-forward) merge strategy — confirmed
unnecessary. Bumping `transformers` to a pre-release or any pin edit to runtime
`dependencies` not required by the lockfile refresh. **Recalibrating the EV
display/verdict thresholds (`karn/constants.py:120-121`,
`ExperimentVerdictPanel.vue:129`, `anomaly_detector.py:66-69`) to the new
op-marginal estimator (`P-EV-RECAL`) — this is the *fundamental* solution and is
explicitly deferred to its own task; stopping the gate false-block is NOT the
same as recalibrating every consumer.**

## Operating Rules

- **No-Legacy / no-bug-hiding (CLAUDE.md).** The EV variance guard must be
  argued as legitimate numerical handling, not defensive masking; the spec makes
  that distinction explicit. Checkpoint migration is retrain-only — no remap
  shim, no dual-path loader.
- **Leyline contracts.** Any new value-fit metric or flag is defined in
  `leyline` (`value_metrics.py` is the home for the existing robust signals).
- **Specialist review before promotion.** Child plans require `reviewed_by`
  sign-off: drl-expert (+ yzmir-deep-rl) for estimator/gate semantics,
  axiom-python-engineering for contract surface and consumer audit. **This
  umbrella's own cross-domain decisions (sequencing, dependency triage) are
  reviewed separately in the `reviewed_by` metadata above — child sign-offs do
  not reach them.** The umbrella is promoted to active only once both child plans
  carry sign-off *and* the umbrella `reviewed_by` is populated.
- **Telemetry is not deferrable.** The EV consumer audit and any new metric ship
  end-to-end (emit → leyline → karn view → panel), per the standing telemetry
  rule.
- **Diagnostic-erosion monitoring (do not let the floor silently drift).** After
  Step 0 sets the floor, the `ev_low_return_variance_count` column (EV plan
  Step 3.3, `ppo_updates` view) MUST be monitored: an anomaly rule fires if the
  flagged fraction exceeds ~20% of updates in a run, and the floor default is
  re-reviewed on any future reward-distribution or architecture change.
  Otherwise "stable-looking EV" silently masks the floor's contribution
  (reinforcing loop R1: stable EV → less scrutiny → floor artifact invisible).

## Success Measures

1. **EV is honest, gate no longer false-blocks.** On a K>1 run with low
   return-variance batches, `VALUE_COLLAPSE_DETECTED` no longer fires from the
   denominator artifact, and `run_confounders` (`views.py`) no longer marks those
   updates `proof_blocking=true`, while a genuinely collapsed value head *still*
   trips the gate (verified by test — see Measure 6).
2. **Every EV consumer enumerated and reconciled.** The spec lists each EV
   consumer (gate, anomalies/run_confounders views, Sanctum panels, Overwatch
   gauges/verdict) with file:line, and each is either confirmed safe, updated, or
   explicitly handed to `P-EV-RECAL` (the threshold recalibration follow-up). No
   consumer is silently left miscalibrated.
3. **`main` carries the full `0.1.1` line.** Fast-forward complete; CI
   (`test-suite.yml`) green across lint/typecheck/property/unit+integration/
   overwatch-web; checkpoint-break documented as retrain-only.
4. **Dependency surface clean, with the required-runtime high verified.**
   Critical + all HIGH alerts resolved via lock refresh + `npm audit fix`;
   **pyarrow #77 (required-runtime via `datasets`) confirmed advanced to
   >=23.0.1**; remaining open alerts limited to documented accept/watch items
   (torch #80 no-fix, transformers pre-release).
5. **One coordinated window, independently revertable.** Schema/merge and the
   dependency bump land as separable commits so either can revert without the
   other.
6. **Genuine-collapse falsification/latency lock.** A synthetic scenario — value
   weights frozen (simulating collapse), `v_return_correlation` low (< 0.1),
   `ev_low_return_variance` False — fires `VALUE_COLLAPSE_DETECTED` within N
   updates under K=4. This is a mandatory sprint-exit regression test that locks
   the detection latency of the stricter AND-gate, ensuring the false-alarm fix
   did not blind or delay real-collapse detection (EV plan Acceptance #6 / #6b).
7. **Monitoring loop committed.** `ev_low_return_variance_count` is exposed in the
   `ppo_updates` view and an anomaly rule fires when its flagged fraction exceeds
   ~20% of updates, closing the diagnostic-erosion loop so the floor default
   cannot silently drift on future reward-distribution/architecture changes.

## Risks

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| EV guard hides / delays a real value collapse (Fixes-that-Fail: stricter AND-gate swallows the firing if robust signals are noisy/slow) | medium | high | Keep raw EV + robust metrics (`v_return_correlation`, `bellman_error`, `td_error_*`) emitted unfloored; Success Measure 6 falsification test locks detection latency (frozen-weights collapse fires within N updates under K=4); §Operating Rules monitoring loop catches floor drift. |
| Variance floor merged on an ungrounded number (Shifting-the-Burden quick fix re-triggers the false alarm) | medium | high | EV plan Step 0 empirical calibration is a HARD PRECONDITION for merging the floor; validate the low-var tail's value-fit is genuinely healthy before declaring it denominator-only. |
| "False alarm stopped" mistaken for "EV consumers recalibrated" (fundamental solution silently treated as done) | medium | medium | `P-EV-RECAL` named as a separate out-of-scope task; Success Measure 2 requires every consumer be confirmed-safe / updated / handed to P-EV-RECAL — none silently left miscalibrated. |
| EV threshold change re-introduces false-alarm at other K | low | medium | Validate against both K=1 and K>1 batches; phase-dependent thresholds reviewed by drl-expert. |
| Checkpoint break surprises a consumer of pre-v2 checkpoints | low | high | Retrain-only is explicit and documented; `VALUE_HEAD_SCHEMA_VERSION=2` rejects pre-v2 loads loudly rather than silently mis-loading. |
| Single flaky test blocks all three coordinated items (Tragedy-of-the-Commons on the shared CI window) | medium | medium | Flaky-vs-regression time-box: bisect against `main` HEAD; pre-existing flakiness may be `xfail`-quarantined ONLY under the four-part release gate (reproduction note + owner/tracker issue + expiry/SLA + explicit merge-owner approval), else the merge STOPS; a true 0.1.1 regression is always a hard stop (and re-triggers the FF-topology check). |
| Large merge destabilises `main` CI | medium | high | Fast-forward (not a real merge) minimises integration surprises; full `test-suite.yml` run is the gate; dependency bump sequenced last and revertable. |
| Required-runtime transitive (pyarrow UAF via `datasets`) under-prioritised as "dev/optional" | medium | high | §3.5 tier (c) reclassifies pyarrow #77 as required-runtime; §3.4 clears it first; Success Measure 4 verifies it advances to >=23.0.1. |
| Transitive bump regresses a runtime path | low | medium | Bump is a dependency manifest/lockfile change (npm step may touch package.json), the LAST step, single revertable commit; core runtime deps carry only low/medium alerts so the change is concentrated in dev/optional tooling. |
| run_confounders read without robust-signal context (event-driven view vs buffered ppo_updates path) | low | low | Documented JOIN pattern (run_confounders ⋈ ppo_updates on batch/epoch) in the EV plan; second-order observability note, not a false-alarm. |
| transformers pressure to take the pre-release | low | low | Explicitly out of scope; alert stays open until 5.0 GA. |
```
