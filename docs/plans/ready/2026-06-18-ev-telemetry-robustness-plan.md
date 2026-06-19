# EV-Telemetry Robustness Under the Op-Marginal V(s) Critic — Executable TDD Plan

```yaml
# Plan Metadata
id: ev-telemetry-robustness
title: EV-Telemetry Robustness Under the Op-Marginal V(s) Critic
type: ready
created: 2026-06-18
updated: 2026-06-19
owner: Claude

urgency: high
value: >
  Make explained_variance an honest value-fit diagnostic under the post-P0-1 op-marginal V(s)
  critic, and stop the K>1 EV-std blowout (a denominator artifact) from false-alarming the
  value-collapse gates that mark proof confounders and invalidate the reward-efficiency verdict.

complexity: L
risk: medium
risk_notes: >
  Cross-domain telemetry-contract change (leyline -> simic types/metrics/emitters -> karn views ->
  overwatch web -> wandb). Reversible: additive payload fields only, NaN/False-tolerant defaults,
  NO VALUE_HEAD_SCHEMA_VERSION bump (telemetry payload, not checkpoint format). Two with-teeth risks:
  (a) a mis-set variance floor could attenuate a legitimately-negative EV — mitigated by unguarded
  companion signals (value_loss / v_return_correlation / bellman_error) that no floor touches, plus a
  config-exposed floor; (b) gate migration could mask a real value collapse — mitigated by making the
  gates STRICTER (require a bad robust signal AND a non-flagged EV), regression-locked by Acceptance #6
  AND the disagreement/co-occurrence regression in Acceptance #6b.

depends_on:
  - recurrent-ppo-multiepoch     # P0-1 / multi-epoch landed on 0.1.1; this is the telemetry follow-up

soft_depends: []

blocks: []

status_notes: >
  Spec approved (docs/superpowers/specs/2026-06-18-ev-telemetry-robustness-design.md). All code
  anchors re-verified against source 2026-06-18 and CORRECTED after a path/symbol audit:
  - Emitter is src/esper/simic/telemetry/emitters.py (NOT simic/emitters.py); mislabel :472, snapshot wiring :986.
  - Coordinator is src/esper/simic/training/ppo_coordinator.py (NOT simic/ppo_coordinator.py); gate call :512-524.
  - The payload class is PPOUpdatePayload (leyline/telemetry.py:701; explained_variance field :719).
    PolicySnapshotPayload does NOT exist anywhere — every prior reference was hallucinated.
  - return_variance ALREADY exists in PPOUpdateMetrics (types.py:112, float) and PPOUpdatePayload
    (telemetry.py:748, float = 0.0), populated from value_func_metrics at ppo_metrics.py:221. Do NOT
    re-add it; the EV-compute-path variance is a new distinct field ev_return_variance.
  - Cross-update EV mean/std is produced in vectorized.py _aggregate_ppo_metrics (:352), NOT
    ppo_metrics.py:63 (that path is the per-update / all-flagged nan default). _aggregate_ppo_metrics
    RAISES KeyError (:397) for any unregistered metric key — new keys MUST be registered or PPO crashes.
  - check_value_function (anomaly_detector.py:140) TODAY receives only explained_variance/current_episode/
    total_episodes; check_all (:447) forwards no robust signal; coordinator (:512-524) passes only
    explained_variance. B1: thread the PRIMARY robust signals bellman_error + value_loss (already on the
    payload: value_loss decl :706, bellman_error decl :744) plus secondary value_nrmse/v_return_correlation + ev_low_return_variance
    through ALL THREE layers. value_nrmse/v_return_correlation are floor-/sentinel-stabilized -> SECONDARY only.
  - P-EV-RECAL (2026-06-19) resolved the Step 0 query ambiguity by adding bellman_error
    and v_return_correlation to Karn ppo_updates; Step 0 uses ppo_updates, not raw_events.
  EV compute ppo_agent.py:619-628; aux EV :1268-1271; ppo_agent has NO `import math`.
percent_complete: 0

# Expert Review (REQUIRED before promotion to ready)
reviewed_by:
  - reviewer: drl-expert
    date: 2026-06-18
    verdict: approved-with-changes
    notes: >
      Estimator semantics and the no-bug-hiding argument are sound (floor changes only the
      ill-conditioned ratio denominator; value_loss / v_return_correlation / bellman_error stay
      unfloored as ground truth). Required changes folded in: (1) gate migration must be STRICTER not
      blinder — require both a bad robust signal AND a non-flagged EV before firing value_collapse
      (Step 5); (2) Acceptance #6 (genuine-collapse-still-fires) is a mandatory regression lock, plus
      #6b (disagreement / low-var-collapse via the robust arm); (3) lock the absolute floor default 1.0
      only after the empirical pass in Step 0, and validate the low-var tail's value-fit is genuinely
      healthy before declaring it a denominator-only artifact.
  - reviewer: axiom-python-engineering
    date: 2026-06-18
    verdict: approved
    notes: >
      Contract growth is additive and NaN/False-tolerant; leyline-first placement of new fields is
      correct. The variance floor and value_nrmse denominator floor are legitimate numerical handling,
      not defensive bug-hiding (the spec's four-ground argument holds; the flag makes the substitution
      observable). emitters.py:472 mislabel fix and HealthGauges.vue:86 clamp removal are correctness
      fixes consistent with the no-legacy / no-bug-hiding rules. No dual code paths introduced. New
      mandatory per-update fields are accessed directly (metrics["value_nrmse"]) so a plumbing gap fails
      loudly; .get() reserved for genuinely-optional payload fields.
```

## Source spec

`docs/superpowers/specs/2026-06-18-ev-telemetry-robustness-design.md` (Approved design — Option C: variance-floored EV + robust companion `value_nrmse` + explicit `ev_low_return_variance` flag).

## Branch

`0.1.1` (active). This is a telemetry-payload change only; **no `VALUE_HEAD_SCHEMA_VERSION` bump** (additive payload schema; checkpoint format unchanged).

## Discipline

Strict TDD, **RED → GREEN**, no-legacy single path. Every behavior-changing step states the failing test, the expected RED, the change, and the GREEN command (`uv run pytest -q ...`). The old `var_returns > 1e-8 → 0.0` special case is **deleted**, not kept alongside the floor (no dual path). Each step carries a rollback note.

**Access discipline (two distinct boundaries — do not conflate):**
- At the **live emitter/reducer/coordinator plumbing boundary** (Step 4 emitter, Step 5 coordinator gate call, the metrics dict), the new mandatory per-update metrics are accessed by **direct key** (`metrics["value_nrmse"]`), never `metrics.get("value_nrmse", default)` — a missing key there means current code is broken and must fail loudly (CLAUDE.md no-bug-hiding).
- At the **`PPOUpdatePayload.from_dict` deserialization boundary** (Step 3.2), the new fields are read with **`data.get(..., default)`** — a missing field there means an OLD persisted event predating this plan (schema evolution), which must default, not crash (Locked decision B4). This is NOT defensive bug-hiding; it is the documented additive-contract requirement, matching the existing `telemetry.py:1129-1140` pattern.

## Naming decisions (resolve the duplicate-field and hallucinated-class blockers up front)

- **Payload class is `PPOUpdatePayload`** (`leyline/telemetry.py:701`). The string `PolicySnapshotPayload` appears nowhere in the codebase; every step, test name, and acceptance row below uses `PPOUpdatePayload`. The emitter test asserting wiring is named `test_ppo_update_payload_carries_ev_robustness_fields`.
- **Do NOT add `return_variance`.** It already exists in `PPOUpdateMetrics` (`types.py:112`, `float`) and `PPOUpdatePayload` (`telemetry.py:748`, `float = 0.0`), populated from `value_func_metrics["return_variance"]` (`ppo_metrics.py:221`, the TELE-227 value). Re-declaring it is a class-definition error and a type conflict (non-optional `float` vs `float | None`). The new field for the EV-compute-path variance is **`ev_return_variance: float | None`** (distinct name, distinct source: `valid_returns.var()` at the EV compute site).
- New per-update fields introduced by this plan: `value_nrmse: float`, `ev_low_return_variance: bool`, `ev_return_variance: float | None`. New aggregated field: `ev_low_return_variance_count: int`. Aux mirror: `aux_ev_low_return_variance: bool`.

## Verified code anchors (re-verified + path/symbol-corrected against source 2026-06-18; authoritative over spec where drifted)

| Anchor | Location | Role |
|--------|----------|------|
| EV compute (source fix) | `src/esper/simic/agent/ppo_agent.py:619-628` (`var_returns = valid_returns.var()`; `if var_returns > 1e-8: ... else: torch.tensor(0.0)`) | **Primary fix site (Step 1)** |
| EV metric emit | `ppo_agent.py:631` `metrics["explained_variance"] = [explained_variance]` | Emit new fields alongside |
| **No `import math`** | `ppo_agent.py:1-57` (verified absent) | **Add `import math` (Step 1.2)** — `math.sqrt` would NameError otherwise |
| return_mean/return_std | `ppo_agent.py:634-642` | `ev_return_variance` (= `var_returns`) derives here |
| Robust value metrics compute | `ppo_agent.py:502-537` → `src/esper/leyline/value_metrics.py` (`v_return_correlation` :17, `bellman_error` :22) | Companion signals to route to gate-readable payload (Step 4) |
| Aux EV | `ppo_agent.py:1268-1271` (`target_var = ...clamp(min=0.01)`; `explained_var = 1.0 - residual_var/target_var`) | **Same artifact family (Step 7)** — already has a `clamp(min=0.01)` partial floor on a DIFFERENT (contribution-target) scale |
| Aux EV emit | `ppo_agent.py:1285,1314,1320,1328` | Emit aux flag |
| Floor config target | `ppo_agent.py:162` (`aux_warmup_steps: int = 1000` — sibling ctor kwarg site) | Add `ev_return_variance_floor: float = 1.0` ctor kwarg |
| `PPOUpdateMetrics.explained_variance` | `src/esper/simic/agent/types.py:81` (class at :38) | Add `value_nrmse`, `ev_low_return_variance`, `ev_return_variance`, `ev_low_return_variance_count` |
| `PPOUpdateMetrics.return_variance` (PRE-EXISTING) | `types.py:112` (`float`) | **Do NOT re-add — already present** |
| Aux types | `types.py:189` (`aux_explained_variance`) | Add `aux_ev_low_return_variance` |
| Per-update / all-flagged EV default | `src/esper/simic/agent/ppo_metrics.py:63` (`aggregated_result["explained_variance"] = nan` in the zero-epoch / degenerate branch) | NOT the cross-update reducer — this only sets the nan default for a degenerate single update |
| ppo_metrics return_variance (PRE-EXISTING) | `ppo_metrics.py:221` (from `value_func_metrics["return_variance"]`) | Existing TELE-227 wiring — untouched |
| **Cross-update EV aggregation (real site)** | `src/esper/simic/training/vectorized.py:352` `_aggregate_ppo_metrics`; reducer frozensets :184-304; `_PPO_MEAN_REDUCED_METRICS` contains `explained_variance` :193, `return_variance` :218, `aux_explained_variance` :241; KeyError fallthrough :397 | **Flagged-update exclusion rule + reducer registration (Step 3)** |
| `PPOUpdatePayload.explained_variance` | `src/esper/leyline/telemetry.py:719` (`float \| None = None`, NaN-tolerant); class declared :701; `v_return_correlation` :741; `return_variance` :748 | Add new fields (Step 2) |
| Payload-from-dict builder | `telemetry.py:962-976` (`explained_variance=data.get("explained_variance")`), optional-field block `:1129-1140` (every optional metric uses `data.get(..., default)`) | Carry new fields **via `.get(..., default)`** — schema evolution for persisted events, NOT direct indexing (Step 3.2, B4) |
| Emit EV → PPO update payload | `src/esper/simic/telemetry/emitters.py:986` (`explained_variance=metrics.get("explained_variance")`) | Carry new fields (Step 4) — **direct access for new mandatory fields** |
| **Mislabel: EV → value_variance** | `emitters.py:472` (`value_variance=metrics.get("explained_variance", 0.0)` → `AnalyticsSnapshotPayload.value_variance` → `batch_stats.value_variance`) | **Mislabel fix (Step 8)** |
| Gate call site | `src/esper/simic/training/ppo_coordinator.py:512-524` (`anomaly_detector.check_all(... explained_variance=metrics["explained_variance"] :516 ... value_collapse_applicable=metrics["usable_actor_timesteps"] > 0 :523)`) — **today passes ONLY `explained_variance`** | Pass PRIMARY robust signals (`bellman_error`, `value_loss`) + secondary (`value_nrmse`, `v_return_correlation`) + flag into gate (Step 5, B1) |
| Gate aggregator method | `src/esper/simic/telemetry/anomaly_detector.py:447` `check_all` (calls `check_value_function` at :485, guarded by `value_collapse_applicable` :483) — **today forwards no robust signal** | **Thread `bellman_error`/`value_loss`/`value_nrmse`/`v_return_correlation`/`ev_low_return_variance` through here too (Step 5, B1)** |
| Gate logic | `anomaly_detector.py:140-175` `check_value_function` (**today signature receives only `explained_variance`, `current_episode`, `total_episodes`** — verified :140-145); thresholds :66-69 (−0.5/−0.2/0.0/+0.1) | **Gate migration: add primary-robust-signal arm keyed on `bellman_error`/`value_loss` (Step 5, B1)** |
| Event mapping | `vectorized.py:480-493` (verbosity escalation), `:517` (`VALUE_COLLAPSE_DETECTED`), `:193` (EV metric registered), `:241` (aux EV) | Register new metrics; event unchanged but driven by stricter gate |
| Karn `ppo_updates` view | `src/esper/karn/mcp/views.py:123-131` (`value_loss`, `bellman_error`, `v_return_correlation`, `explained_variance`, `value_nrmse`, `ev_low_return_variance`, `ev_return_variance`) | Step 0 calibration evidence is queryable from `ppo_updates`; Step 6 keeps the robustness columns first-class |
| Karn `anomalies` view | `karn/mcp/views.py:596` (`VALUE_COLLAPSE_DETECTED`) | Inherits stricter gate (Step 5/6) |
| **Karn `run_confounders`** | `karn/mcp/views.py:606-662` (`true as proof_blocking` at :649; event at :652) | **Highest teeth — proof-blocking confounder.** Stays event-type based (B2): artefactual collapse is suppressed UPSTREAM by the Step 5 gate not emitting the event, NOT by a view-level filter. Optional flagged-exclusion on the cross-update EV std/mean is display-only and must not drop a proof-blocking row. |
| TUI health bool | `src/esper/karn/sanctum/widgets/reward_health.py:39` (`is_ev_healthy = ev_explained > 0.5`) | Base on `value_nrmse` (Step 6) |
| Persist/history | `aggregator.py`; `schema.py`; `snapshot_copy.py` | Carry new fields (Step 6) |
| wandb backend | `src/esper/nissa/wandb_backend.py:389-390` (`if p.explained_variance is not None: metrics["ppo/explained_variance"] = ...`) | Add `ppo/value_nrmse`, `ppo/ev_low_return_variance` (Step 6) |
| Overwatch verdict gate | `ExperimentVerdictPanel.vue:127-138` (`policyGate → 'watch'` when `explained_variance < 0.3`) | Use `value_nrmse`; ignore flagged EV (Step 6) |
| **Overwatch gauge clamp** | `HealthGauges.vue:86` (clamps EV to `[0,1]` — masks blowout); `:21-22` GOOD/WARN; `:91-92` status | **Display-masking defect fix (Step 6)** |
| Sanctum web types | `sanctum types.ts:316,405` | Carry new fields (Step 6) |
| Display thresholds | `constants.py:120-121` (`EXPLAINED_VAR_WARNING=0.3` / `EXPLAINED_VAR_CRITICAL=0.0`); consumers `ppo_losses_panel.py`, `critic_calibration_panel.py`, `narrative_panel.py` | Review; likely unchanged (display only) |

### Explicitly OUT OF SCOPE — do NOT touch (different signal, same `value_collapse` name)
| Location | Why excluded |
|----------|--------------|
| `karn/triggers.py:247-259` `PolicyAnomalyDetector.check_value_collapse(value_std)` (`PolicyThresholds.VALUE_STD_COLLAPSE`) | Keys on **value_std**, not EV. Unaffected by the EV artifact. |
| `store.py:354` `value_collapse: bool` | Also **value_std-derived**, not EV. |

---

## Step 0 — Empirical floor calibration (pre-flight, no code change)

> Locked decision 3 sets the default `ev_return_variance_floor = 1.0`, but drl-expert review requires confirming it sits well below the normal operating band before locking AND that the low-variance tail is genuinely a denominator-only artifact (not a real fit failure). This step produces the numeric the unit tests in Step 1 assert against and the evidence for the Step 5 robust-arm test.

- Using the Karn MCP (`mcp__esper-karn__query_sql` over the most recent K=4 P0-1 run via `run_dir`), pull the per-update `return_std` distribution from the `ppo_updates` view and compute `var_returns = return_std**2` percentiles. P-EV-RECAL locks `ppo_updates` as the executable path; do not fall back to `raw_events` unless these columns regress and the preflight blocks.
- Confirm: healthy-run `var_returns` is in the ~49–169 band (return std 7–13, per the multi-epoch design doc), and that the EV-blowout updates correspond to the low-`var_returns` tail.
- **Validate the tail's value-fit (Phase-2 question).** For the low-`var_returns` tail updates identified, ALSO pull `bellman_error`, `v_return_correlation`, and `value_loss` from the same updates. Confirm they are healthy (low `bellman_error`, high `v_return_correlation`, low `value_loss`). If ANY tail update has a bad robust signal, that is evidence the artifact and a real regression can CO-OCCUR; record it and ensure Acceptance #6b's robust-arm test covers that regime — the floor must not be relied on to classify those.
- **Lock the floor** at the value that excludes only the (confirmed-healthy) pathological tail (default `1.0` unless the data argues otherwise). Record the chosen value and the percentile rationale inline in the Step 1 ctor docstring.
- **No RED/GREEN** — this is calibration. Output is the locked constant + a one-line justification carried into the code comment + the tail value-fit evidence.

### Step 0 executable preflight (P-EV-RECAL, 2026-06-19)

`src/esper/karn/mcp/views.py` now projects the full calibration evidence from
`ppo_updates`: `return_std`, `value_loss`, `bellman_error`, and
`v_return_correlation`. The preflight query below fails loudly when the selected
run has no PPO updates or when any required evidence is absent; it uses no
defaults, `COALESCE`, or fallback payload path. Replace `<RUN_DIR>` with the
selected run, defaulting to the newest local PPO run observed during
P-EV-RECAL orientation: `telemetry_2026-06-16_160350`.

```sql
WITH updates AS (
    SELECT
        episodes_completed,
        group_id,
        return_std,
        power(return_std, 2) AS var_returns,
        explained_variance,
        value_loss,
        bellman_error,
        v_return_correlation
    FROM ppo_updates
    WHERE run_dir = '<RUN_DIR>'
),
summary AS (
    SELECT
        count(*) AS updates,
        sum(
            CASE
                WHEN return_std IS NULL
                  OR value_loss IS NULL
                  OR bellman_error IS NULL
                  OR v_return_correlation IS NULL
                THEN 1 ELSE 0
            END
        ) AS missing_required_rows,
        quantile_cont(var_returns, [0.0, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0]) AS var_return_quantiles,
        min(value_loss) AS min_value_loss,
        max(value_loss) AS max_value_loss,
        min(bellman_error) AS min_bellman_error,
        max(bellman_error) AS max_bellman_error,
        min(v_return_correlation) AS min_v_return_correlation,
        max(v_return_correlation) AS max_v_return_correlation
    FROM updates
)
SELECT
    CASE
        WHEN updates = 0 THEN error('EV calibration preflight found no PPO updates for run_dir')
        WHEN missing_required_rows > 0 THEN error('EV calibration preflight missing required value evidence')
        ELSE 'ok'
    END AS preflight_status,
    *
FROM summary;
```

Live evidence captured 2026-06-19 with `KarnMCPServer("telemetry")` for
`run_dir = 'telemetry_2026-06-16_160350'`: `preflight_status = 'ok'`,
`updates = 10`, `missing_required_rows = 0`, `var_return_quantiles =
[5.693101418319429, 6.490904748950143, 17.48170978181183,
40.15395539813676, 46.41391027192975, 54.826956276346074,
57.97769175911617]`, `value_loss = 0.10646478831768036..0.8902218341827393`,
`bellman_error = 0.25563251972198486..0.45969972014427185`, and
`v_return_correlation = -0.08301542699337006..0.12338030338287354`.

### Step 0 LOCKED values (Slice 3 — robust-arm gate thresholds, drl-expert sign-off 2026-06-18)

> The live Karn telemetry store was unavailable for an interactive percentile pull during Slice 3
> (the `ppo_updates` scan exceeded the 30s query budget). The robust-arm thresholds were therefore
> derived analytically from the design-doc healthy-band anchors (P0-1 K=4 A/B table) and the actual
> loss formula, then locked with drl-expert reasoning. They are exposed as `AnomalyDetector` dataclass
> fields so a future empirical pass can refine them without code surgery.

- **`value_loss_threshold = 0.5`.** `value_loss = 0.5 * mean((value - normalized_return)**2)` on the
  value-normalizer (unit-variance) target scale (`ppo_update.py:354-360`). Healthy median ≈ **0.099**
  (lowest of all arms, design-doc table). On the normalized scale the value head explains ZERO variance
  of its own trained target precisely when `mean((v - r_norm)**2) ≥ Var(r_norm) ≈ 1.0`, i.e.
  `value_loss ≥ 0.5`. So `0.5` is the principled "value head stopped explaining its target" boundary —
  ~5× the healthy median, firing on a genuine fit regression while leaving generous margin above
  healthy noise. This is the PRIMARY, scale-invariant signal.
- **`bellman_error_threshold = 10.0`.** `bellman_error = mean |TD error|` on the RAW return scale
  (returns std ~7–13 → var ~49–169). A healthy critic keeps mean |TD| well below the return spread;
  a genuine collapse drives the absolute residual to rival the signal magnitude. `10.0` sits at the
  lower edge of the healthy return-std band, so the residual must rival the return spread itself
  before it fires. Conservative against false-firing on a critic doing any useful work. SECONDARY
  primary (corroborates `value_loss`).
- Both are `OR`'d in the robust arm (`primary_robust_bad = bellman_error > 10.0 or value_loss > 0.5`).
  The EV arm additionally requires `primary_robust_bad AND not ev_low_return_variance` (B1: EV needs
  primary-robust agreement; it never fires alone). `value_nrmse` / `v_return_correlation` are SECONDARY
  corroboration in the detail string only.
- **Follow-up:** when the Karn store is queryable, run the Step-0 percentile pull to confirm the
  healthy `value_loss` / `bellman_error` distributions sit below these bounds with margin; refine the
  two dataclass fields if the empirical tail argues otherwise (no code change beyond the two numbers).

**Rollback note:** none (no code change for the calibration itself; the two thresholds are additive
`AnomalyDetector` dataclass fields landed in Slice 3 Step 5.0).

---

## Step 1 — RED→GREEN: variance-floored EV at the source (Acceptance #1)

**Files:** `src/esper/simic/agent/ppo_agent.py`; `src/esper/leyline/value_metrics.py`; `tests/simic/test_ppo_value_metrics.py`.

> **Helper seam required (W7).** The EV computation is currently inlined inside `PPOAgent.update()` after `buffer.get_batched_sequences()` (`ppo_agent.py:609-642`); there is NO function that takes `(valid_values, valid_returns, floor)` and returns `(explained_variance, value_nrmse, ev_low_return_variance, ev_return_variance)`. The unit tests below CANNOT call such a seam until it exists. **Step 1.2 must FIRST extract an explicit testable helper** — preferred home `leyline/value_metrics.py` (sibling to the existing `v_return_correlation`/`bellman_error` helpers) or a private `PPOAgent` method — and have `update()` call it (single path, no duplicated formula). The unit tests target that helper directly; **additionally add at least one end-to-end `PPOAgent.update()` regression** (`test_ppo_update_emits_ev_robustness_fields`) proving the helper is wired into the live update path (a helper that is green but unwired would otherwise pass silently).

### 1.1 RED — floored EV unit tests (against the new helper seam)
- **New test** `test_ppo_value_metrics.py::test_ev_floored_on_low_return_variance`: call the **new EV helper** with a synthetic `(valid_values, valid_returns)` pair with tiny `var_returns` (≪ floor) and a small absolute residual; assert the **pre-floor** formula would give EV ≪ −1 (compute it inline as the witness), and the **new** path gives a bounded EV (e.g. `> -2`) AND `ev_low_return_variance is True`.
- **New test** `test_ev_unchanged_on_healthy_variance`: a `var_returns` in the 49–169 band yields EV numerically within `1e-6` of the pre-floor value and `ev_low_return_variance is False`.
- **New test** `test_ev_no_special_case_zero`: a near-zero `var_returns` (the old `1e-8` branch) now produces a floored continuous value, **not** exactly `0.0`, and the flag is True (proves the sentinel-`0.0` path is gone).
- **New test** `test_ev_floor_default_matches_calibration`: assert `PPOAgent(...).ev_return_variance_floor == <locked Step 0 value>` (e.g. `1.0`). This is a meta-assertion so any drift from Step 0 is caught (the `> -2` bound tests alone would let a wrong default pass silently).
- **New test** `test_ev_degenerate_batch_raises_loudly` (optional but recommended): a `valid_returns` mask selecting ≤1 element. **A degenerate valid-return batch is a HARD BUG, not a telemetry-safe NaN case** (Locked decision B3). `valid_returns.var()` is NaN for `numel==1` (PyTorch default Bessel correction `correction=1`) and undefined for `numel==0`; the existing non-finite return-stat guard at `ppo_agent.py:636` (`raise RuntimeError("Non-finite returns detected ...")`) is the correct loud-failure path and **stays**. Assert that an `update()` over a degenerate mask **raises** rather than emitting any EV/`value_nrmse` metric. There is **no** `ev_low_return_variance=True`-on-degenerate convention and **no** NaN-by-convention EV: a degenerate mask means upstream masking/rollout plumbing is broken and must fail loudly, never reach the gate.
- **Expected RED:** the floor, the flag, and the config field do not exist yet; `AttributeError`/wrong value. (The degenerate-batch loud-raise is already enforced by `ppo_agent.py:636`; this plan introduces NO NaN-acceptance path that would swallow it.)

### 1.2 GREEN — replace the compute block (ppo_agent.py:619-628)
- **Extract the EV helper first (W7).** Add a pure function (preferred: `leyline/value_metrics.py`, e.g. `compute_floored_explained_variance(valid_values, raw_values, valid_returns, floor) -> tuple[Tensor, Tensor, bool, Tensor]`) returning `(explained_variance, value_nrmse, ev_low_return_variance, ev_return_variance)`. `PPOAgent.update()` calls this helper at `:619-628` instead of inlining the formula — single path, no duplication. The Step 1.1 / Step 2.1 unit tests target this helper; the end-to-end `test_ppo_update_emits_ev_robustness_fields` proves it is wired.
- **Add `import math`** where `math.sqrt` is used (in the helper's module if extracted to `value_metrics.py`, else `ppo_agent.py`; verified absent in `ppo_agent.py:1-57`). (Alternatively compute the floor std via `torch.tensor(floor).sqrt().item()` and add no import — pick one explicitly; this plan uses `import math`.)
- Add ctor kwarg `ev_return_variance_floor: float = 1.0` beside `aux_warmup_steps` (`ppo_agent.py:162`); store `self.ev_return_variance_floor` and `self.ev_var_floor_std = math.sqrt(self.ev_return_variance_floor)` in `__init__`; docstring records the Step 0 percentile rationale + locked value.
- The helper body (called from `:619-628`) is the single floored path. **Degenerate batches are a HARD BUG and must fail loudly** (Locked decision B3): the floored path computes EV only on a non-degenerate (`numel>=2`) batch, and the existing non-finite return-stat raise at `ppo_agent.py:636` remains the loud-failure gate for `numel==0`/`numel==1` (where `var()`/`std()` are NaN/undefined). Do NOT introduce a NaN-by-convention branch or an `ev_low_return_variance=True`-on-degenerate path — that would swallow the existing raise and let a broken mask reach the gate. The floor applies ONLY to the legitimate low-variance regime, never as a NaN sanitizer:
  ```python
  residual_var = (valid_returns - raw_values).var()
  var_returns = valid_returns.var()
  # numel<2 / non-finite var is a HARD BUG: the return-stat guard at :636
  # raises on it. The floor below handles ONLY genuine low (but finite,
  # numel>=2) variance — it is never a clamp(NaN) sanitizer.
  denom = torch.clamp(var_returns, min=self.ev_return_variance_floor)
  ev_low_return_variance = bool(var_returns < self.ev_return_variance_floor)
  explained_variance = 1.0 - residual_var / denom
  ev_return_variance = var_returns
  ```
  (`bool(var_returns < floor)` is well-defined for the finite, `numel>=2` batches that reach this code; the degenerate case never gets here because the `:636` guard raises first.)
- **Delete** the `if var_returns > 1e-8 ... else torch.tensor(0.0)` block entirely (Locked decision 4; no-legacy).
- Emit `ev_return_variance` and the flag into `metrics` next to `metrics["explained_variance"]` (`:631`): `metrics["ev_return_variance"] = [ev_return_variance]`, `metrics["ev_low_return_variance"] = [ev_low_return_variance]`. (`return_variance` is already emitted via the value-func path — do NOT touch it.)
- **GREEN command:** `uv run pytest tests/simic/test_ppo_value_metrics.py -k "ev_floored or ev_unchanged or ev_no_special_case or ev_floor_default or ev_degenerate or ppo_update_emits_ev" -q`

**Rollback note:** self-contained to the extracted `value_metrics.py` helper + the `ppo_agent.py` call site + one ctor kwarg + `import math` + new tests; `git revert` restores the inlined `1e-8`/`0.0` branch. No contract/schema surface touched yet, so revert strands nothing.

---

## Step 2 — RED→GREEN: `value_nrmse` companion metric (Acceptance #2)

**Files:** `src/esper/simic/agent/ppo_agent.py`; `src/esper/leyline/value_metrics.py`; `tests/simic/test_ppo_value_metrics.py`.

### 2.1 RED (against the Step 1 helper seam, W7)
- **New test** `test_value_nrmse_finite_and_monotone`: call the EV helper with a batch and a family of injected residual magnitudes; assert `value_nrmse` is finite, ≥ 0, and strictly increasing in injected residual magnitude.
- **New test** `test_value_nrmse_bounded_on_low_variance`: on the same low-`var_returns` batch from Step 1 (where EV would blow out), assert `value_nrmse` is finite and small (it reflects the low `value_loss`, not the crater EV).
- **New test** `test_value_nrmse_degenerate_batch_raises_loudly` (optional but recommended): numel==1 and numel==0 masks; assert `value_nrmse` is NEVER emitted because the degenerate batch raises loudly at the existing return-stat guard (`ppo_agent.py:636`) before any companion metric is computed (Locked decision B3 — degenerate masks are a hard bug, not a NaN-by-convention case).
- **Expected RED:** `value_nrmse` not emitted.

### 2.2 GREEN
- In the Step 1 helper body (which only executes on a non-degenerate, `numel>=2`, finite batch — degenerate masks raise loudly at `:636`), after the floored EV, compute (denominator floored on the **std** scale so it cannot blow up either):
  ```python
  value_nrmse = torch.sqrt(residual_var) / (valid_returns.std() + self.ev_var_floor_std)
  ```
  There is no degenerate-batch branch to set `value_nrmse = NaN`: a degenerate mask is a hard bug that has already raised before this line (Locked decision B3).
- Emit `metrics["value_nrmse"] = [value_nrmse]`.
- **GREEN command:** `uv run pytest tests/simic/test_ppo_value_metrics.py -k value_nrmse -q`

**Rollback note:** additive metric only; revert removes the compute + emit lines and the tests.

---

## Step 3 — RED→GREEN: contract plumbing + flagged-update aggregation (Acceptance #3, #4)

**Files:** `src/esper/simic/agent/types.py`; `src/esper/simic/agent/ppo_metrics.py`; `src/esper/simic/training/vectorized.py`; `src/esper/leyline/telemetry.py`; `tests/simic/test_ppo_value_metrics.py`; `tests/simic/test_telemetry_fields.py`; `tests/simic/test_vectorized.py`.

### 3.1 RED — contract field presence
- **New test** `test_telemetry_fields.py::test_ppo_update_payload_carries_ev_robustness_fields`: assert `PPOUpdatePayload` has `value_nrmse: float | None`, `ev_low_return_variance: bool`, `ev_return_variance: float | None`, `ev_low_return_variance_count: int`, and that the **primary robust gate signals** `bellman_error` / `value_loss` plus secondary `v_return_correlation` are reachable on the same payload a gate reads. **All three already exist** as payload fields (`value_loss` declared `telemetry.py:706`, required in `from_dict:968`; `bellman_error` declared `:744`, `from_dict:1135`; `v_return_correlation` declared `:741`, `from_dict:1132`) — no new payload field is needed for them; the B1 work is the gate-call plumbing (Step 5), not the contract.
- **New test** `test_ppo_value_metrics.py::test_ppo_update_metrics_carries_ev_robustness_fields`: assert `PPOUpdateMetrics` exposes `value_nrmse`, `ev_low_return_variance`, `ev_return_variance`, `ev_low_return_variance_count`. (Note `return_variance` is PRE-EXISTING — do not assert it as new.)
- **New test (B4 schema-evolution regression)** `test_telemetry_fields.py::test_ppo_update_payload_from_dict_old_event_without_ev_fields`: build a `data` dict containing ONLY the pre-plan required + optional fields (NO `value_nrmse` / `ev_low_return_variance` / `ev_return_variance` / `ev_low_return_variance_count`) and assert `PPOUpdatePayload.from_dict(data)` succeeds, yielding `value_nrmse is None`, `ev_low_return_variance is False`, `ev_return_variance is None`, `ev_low_return_variance_count == 0`. This locks the additive/default-tolerant contract claim so a future direct-index regression in `from_dict` fails loudly. (GREEN after 3.2 lands the `.get(..., default)` plumbing.)
- **Expected RED:** new fields absent (the metrics/payload tests); `from_dict` would `KeyError` if it direct-indexed the new fields (the B4 regression test).

### 3.2 GREEN — add fields (leyline first)
- `leyline/telemetry.py` `PPOUpdatePayload` (after `:719`): add `value_nrmse: float | None = None`, `ev_low_return_variance: bool = False`, `ev_return_variance: float | None = None`, `ev_low_return_variance_count: int = 0`. **`bellman_error` and `value_loss` already exist as payload fields** (`bellman_error` declared `:744`, `value_loss` declared `:706`) — do NOT re-add them; the gate just needs them threaded into the call (Step 5). **Do NOT add `return_variance` — it exists at `:748`.**
- `agent/types.py` `PPOUpdateMetrics` (after `:81`): add `value_nrmse: float`, `ev_low_return_variance: bool`, `ev_return_variance: float`, `ev_low_return_variance_count: int`. **Do NOT add `return_variance` — it exists at `:112`.**
- `telemetry.py` `PPOUpdatePayload.from_dict` (`:962-976` neighborhood): carry the new fields **using documented defaults, NOT direct indexing** (Locked decision B4). `from_dict` deserializes persisted/historical events; a missing new field means an OLD event predating this plan, which is legitimate **schema evolution**, not a current-code bug. This matches the existing pattern at `telemetry.py:1129-1140` where every optional metric uses `data.get(..., default)`. Add:
  - `value_nrmse=data.get("value_nrmse", None)`
  - `ev_low_return_variance=data.get("ev_low_return_variance", False)`
  - `ev_return_variance=data.get("ev_return_variance", None)`
  - `ev_low_return_variance_count=data.get("ev_low_return_variance_count", 0)`

  > Direct indexing (`data["value_nrmse"]`) is the correct discipline ONLY at the **live emitter/reducer plumbing boundary** (Step 4 / Step 5 coordinator call), where a missing key means current code is broken and must fail loudly. The `from_dict` deserialization boundary is the opposite case: missing == old event == default. This is NOT bug-hiding (the no-defensive-programming rule does not apply to schema evolution of persisted data).

### 3.3 RED — flagged-update aggregation rule at the REAL cross-update site (Acceptance #4)
> The cross-update EV mean/std is produced in `vectorized.py:352` `_aggregate_ppo_metrics` (driven by `_PPO_MEAN_REDUCED_METRICS`), NOT `ppo_metrics.py:63` (which aggregates a single update's per-epoch values; for EV that is a 1-element no-op). `_aggregate_ppo_metrics` RAISES `KeyError` (`:397`) for any metric key without a declared reducer — so the new keys MUST be registered or the PPO update crashes. This is a correctness requirement, not display registration.
- **New test** `tests/simic/test_vectorized.py::test_aggregate_ppo_metrics_no_keyerror_on_ev_fields`: pass an `update_metrics` list whose entries carry `value_nrmse`, `ev_low_return_variance`, `ev_return_variance`; assert `_aggregate_ppo_metrics` returns without raising `KeyError`.
- **New test** `tests/simic/test_vectorized.py::test_ev_aggregation_excludes_flagged_updates`: build a list mixing `ev_low_return_variance=True` and `=False` updates; assert run-level EV **mean is computed only over the unflagged updates** (equal to the unflagged-only computation), `value_nrmse` / `v_return_correlation` / `ev_return_variance` aggregate over **all** updates, and `ev_low_return_variance_count` equals the number of flagged updates.
- **New test** `tests/simic/test_vectorized.py::test_ev_aggregation_all_flagged`: every update has `ev_low_return_variance=True`; assert aggregated `explained_variance` is `nan`/`None` (no unflagged sample), `value_nrmse` still aggregates normally, `ev_low_return_variance_count` equals the count, and nothing raises.
- **Expected RED:** `_aggregate_ppo_metrics` raises `KeyError` for the unregistered keys; the mean reducer averages ALL updates' EV (cannot see the sibling flag).

### 3.4 GREEN — reducers + flagged exclusion in `_aggregate_ppo_metrics`
- In `vectorized.py`:
  - Register `value_nrmse` and `ev_return_variance` in `_PPO_MEAN_REDUCED_METRICS` (`:184`).
  - Add a **count reducer** for `ev_low_return_variance_count` (register in `_PPO_SUM_REDUCED_METRICS` — semantics: sum of per-update counts; each per-update `ev_low_return_variance_count` is `int(ev_low_return_variance)`). Populate the per-update `ev_low_return_variance_count` in `ppo_metrics`/emit as `1 if ev_low_return_variance else 0` so the cross-update sum is the flagged-update count.
  - Add `ev_low_return_variance` to an appropriate bool reducer (`_PPO_ANY_REDUCED_METRICS` :269 — semantics: "any update in this batch was flagged"; documented inline). This is the per-batch rollup, distinct from the integer count.
  - **Special-case `explained_variance`** in `_aggregate_ppo_metrics`: because the generic mean reducer cannot see a sibling key's flag, add an explicit branch BEFORE the frozenset dispatch that computes the EV mean over only the updates whose `ev_low_return_variance` is `False`; if no unflagged updates remain, set `nan`. (The same applies to EV std at the cross-update rollup layer — see below.)
- The cross-update EV **std/mean rollup** that the spec's `-0.337 mean / 2.264 std` blowout comes from is produced at the Karn/DuckDB layer (`run_confounders` / analysis views), not in the hot path (which only computes a per-batch mean). The flagged-exclusion `ev_low_return_variance == False` filter MAY be applied to that **display-statistic** std/mean (Step 6.1) — but it must NOT change `run_confounders` proof-blocking semantics (B2): the artefactual collapse is prevented upstream by the Step 5 gate, not by a view filter dropping an emitted event row.
- **GREEN command:** `uv run pytest tests/simic/test_vectorized.py tests/simic/test_ppo_value_metrics.py tests/simic/test_telemetry_fields.py -q`

**Rollback note:** additive dataclass fields + reducer registrations + one special-cased EV branch; revert removes fields/reducers and restores the all-updates mean. Old telemetry consumers tolerate missing fields via the defaults (no migration).

---

## Step 4 — RED→GREEN: emitter wiring (carry new fields to consumers) (Acceptance #3)

**Files:** `src/esper/simic/telemetry/emitters.py`; `tests/karn/test_analytics_snapshot_policy.py`.

### 4.1 RED
- **New test** `tests/karn/test_analytics_snapshot_policy.py::test_emitter_carries_ev_robustness_fields`: assert the PPO-update-payload emitter (`emitters.py:986`) writes the NEW fields `value_nrmse`, `ev_low_return_variance`, `ev_return_variance`, `ev_low_return_variance_count` through to `PPOUpdatePayload`, and that the gate's primary/secondary robust signals `bellman_error` / `value_loss` / `v_return_correlation` (already emitted) remain present.
- **Expected RED:** emitter only carries `explained_variance`.

### 4.2 GREEN
- Extend the `emitters.py:986` mapping to carry the new fields from the aggregated metrics into `PPOUpdatePayload`. Per the no-bug-hiding rule, access the new **mandatory** fields directly (`metrics["value_nrmse"]`, `metrics["ev_low_return_variance"]`, `metrics["ev_low_return_variance_count"]`) — a missing key is a plumbing bug that must fail loudly, not be masked by a `.get(..., default)`. `ev_return_variance` may legitimately be `None` (degenerate batch); `.get("ev_return_variance", None)` is acceptable there.
- **GREEN command:** `uv run pytest tests/karn/test_analytics_snapshot_policy.py -k test_emitter_carries_ev_robustness_fields -q`

**Rollback note:** wiring only; revert restores the EV-only mapping.

---

## Step 5 — RED→GREEN: gate migration (the part with teeth) (Acceptance #5, #6, #6b)

**Files:** `src/esper/simic/telemetry/anomaly_detector.py`; `src/esper/simic/training/ppo_coordinator.py`; `tests/simic/telemetry/` (anomaly-detector tests).

> drl-expert: the migration must make the gate **stricter about evidence**, not blinder. The **PRIMARY** robust gate signals are the **scale-anchored** `bellman_error` and `value_loss` — `value_nrmse` is floor-stabilized and `v_return_correlation` returns a low-variance sentinel, so **neither may be the primary deciding signal** (Locked decision B1). `check_value_function` fires `value_collapse` only when a **primary robust signal confirms a bad fit** (high `bellman_error` and/or high `value_loss`), with `value_nrmse` / `v_return_correlation` as **SECONDARY** display/support, **AND** the EV reading is not flag-attenuated (`ev_low_return_variance == False`). An update whose *only* symptom is artefactual low EV must NOT fire. Conversely, the robust arm MUST still fire on a genuine collapse even when `var_returns` is low (the co-occurrence case from Step 0), keyed on **high `bellman_error` or high `value_loss`**.
>
> **Threading note (corrected, B1):** `check_value_function` (`:140`) is invoked from `check_all` (`:485`), guarded by `value_collapse_applicable` (`:483`). The coordinator (`:512-524`) passes args INTO `check_all`, not into `check_value_function` directly. **`check_value_function` today only receives `explained_variance`** (verified `anomaly_detector.py:140-145`); `check_all` does not forward any robust signal (verified `:447-486`); the coordinator passes only `explained_variance=metrics["explained_variance"]` (verified `ppo_coordinator.py:516`). The new signals — **`value_loss` and `bellman_error` (primary)** plus `value_nrmse` and `v_return_correlation` (secondary) and `ev_low_return_variance` — MUST be threaded through **all three** layers: the coordinator call site, the `check_all` signature (`:447`), AND the `check_value_function` signature (`:140`). `bellman_error` and `value_loss` already exist on `PPOUpdatePayload` (`value_loss` declared `telemetry.py:706`; `bellman_error` declared `:744`) AND are registered aggregated metric keys (`vectorized.py:186` `value_loss`, `:214` `bellman_error`), so they are present in the coordinator's aggregated `metrics` dict — the gap is purely the gate-call plumbing, not new payload/reducer fields.

### 5.0 Signature safety (no-TypeError window)
- Add the new `check_all` parameters **keyword-only with conservative defaults**: `bellman_error: float = 0.0`, `value_loss: float = 0.0`, `value_nrmse: float = 0.0`, `v_return_correlation: float = 1.0`, `ev_low_return_variance: bool = False`. The `0.0` defaults for `bellman_error`/`value_loss` keep an un-updated caller's robust arm quiescent (conservative — does not false-fire), while the `False` default for `ev_low_return_variance` keeps the EV arm able to FIRE (conservative — never silently suppresses a real collapse). Add the same params to `check_value_function`'s signature. **`bellman_error` and `value_loss` are the PRIMARY robust signals (B1); `value_nrmse` / `v_return_correlation` are SECONDARY.**
- Update the coordinator call site (`ppo_coordinator.py:516-523` region) in the **same commit** as the signature change, passing `bellman_error=metrics["bellman_error"]`, `value_loss=metrics["value_loss"]`, `value_nrmse=metrics["value_nrmse"]`, `v_return_correlation=metrics["v_return_correlation"]`, `ev_low_return_variance=metrics["ev_low_return_variance"]` (direct access — these are mandatory aggregated fields already present on the payload).

### 5.1 RED — artifact no longer false-alarms (Acceptance #5)
- **New test** `test_check_value_function_ignores_artifactual_low_ev`: feed `check_value_function` (and `check_all`) an update with `ev_low_return_variance=True`, EV ≪ threshold, but **healthy primary robust signals** (low `bellman_error`, low `value_loss`) and healthy secondary signals (`value_nrmse` / `v_return_correlation`); assert **no** `value_collapse` anomaly is produced.
- **Expected RED:** current gate fires on the EV threshold alone (`:66-69`, `:140-175`).

### 5.2 RED — genuine collapse still fires (Acceptance #6, MANDATORY regression lock)
- **New test** `test_check_value_function_fires_on_genuine_collapse`: feed an update with healthy `var_returns` (`ev_low_return_variance=False`), genuinely negative EV, and **bad PRIMARY robust signals** (high `bellman_error`, high `value_loss`); secondary `value_nrmse`/`v_return_correlation` may corroborate but are NOT the deciding signal; assert the `value_collapse` anomaly **IS** produced. (Proves the change is not bug-masking.)

### 5.2b RED — disagreement + low-var co-occurrence (Acceptance #6b, MANDATORY)
- **New test** `test_check_value_function_fires_on_low_var_collapse_via_robust_arm`: low `var_returns` (`ev_low_return_variance=True`) BUT a genuinely bad **PRIMARY** robust signal (**high `bellman_error` and/or high `value_loss`**); assert the gate **STILL FIRES** via the robust arm. This is the load-bearing co-occurrence lock (B1): the floor-stabilized `value_nrmse` and the sentinel-returning `v_return_correlation` must NOT be relied on here — the scale-anchored `bellman_error`/`value_loss` are what catch a real regression hiding behind low variance.
- **New test** (coordinator-level, B1 required) `test_coordinator_emits_value_collapse_on_low_var_real_collapse`: drive the full `PPOCoordinator → check_all → check_value_function` path with an aggregated metrics dict representing a real low-variance collapse (high `bellman_error`/`value_loss`, `ev_low_return_variance=True`); assert a `VALUE_COLLAPSE_DETECTED` event is emitted. This proves the primary signals are actually threaded end-to-end, not just present in `check_value_function`'s signature.
- **New test** `test_check_value_function_disagreement_ev_bad_robust_ok`: `ev_low_return_variance=False`, EV ≪ threshold, but the **primary** robust signals sit just BELOW their bad thresholds (look OK). Assert the **intentional** behaviour explicitly: the gate requires primary-robust-signal agreement, so it does NOT fire on EV alone; assert and document this so a loose robust threshold cannot silently swallow a real collapse undetected (the disagreement is surfaced by EV remaining visible in telemetry even when the gate holds fire).
- **Expected RED before 5.3, GREEN after.**

### 5.3 GREEN — rewrite the gate condition
- In `anomaly_detector.py:140-175` rewrite `check_value_function`: keep the phase-dependent EV thresholds (`:66-69`) as **one** condition, and require **(PRIMARY-robust-signal-bad) AND (not ev_low_return_variance)** before emitting `value_collapse` via the EV arm; add a separate **robust arm** that fires when a **PRIMARY** robust signal is bad (**high `bellman_error` or high `value_loss`**) regardless of `ev_low_return_variance` (covers 5.2b). `value_nrmse` / `v_return_correlation` are SECONDARY corroboration only and must NOT be the sole trigger (B1: they are floor-/sentinel-stabilized). No dual path — replace the EV-only condition. Add `bellman_error_threshold` / `value_loss_threshold` config alongside the existing EV thresholds (`:66-69`); calibrate from the Step 0 healthy-band evidence.
- In `check_all` (`:447-485`): thread `bellman_error`, `value_loss`, `value_nrmse`, `v_return_correlation`, `ev_low_return_variance` into the `check_value_function(...)` call at `:485`.
- The `VALUE_COLLAPSE_DETECTED` event mapping (`vectorized.py:517`) and verbosity escalation (`:480-493`) are unchanged; they are now driven by the stricter gate.
- **Cross-dependency (load-bearing):** the gate consumes the **aggregated** `explained_variance` (`ppo_coordinator.py:516` passes `metrics["explained_variance"]`, the cross-update value from Step 3.4). So the Step 3.4 flagged-exclusion rule is exactly what the Step 5 gate keys on. Add **`test_gate_sees_unflagged_only_ev_mean`**: drive `check_all` with an aggregated metrics dict produced by `_aggregate_ppo_metrics` over a flagged/unflagged mix and assert the EV the gate evaluates equals the unflagged-only mean.
- **GREEN command:** `uv run pytest tests/simic/telemetry -k value_function -q` and `uv run pytest tests/simic/test_vectorized.py -k unflagged_only_ev -q`

**Rollback note:** gate condition is rewritten predicates + the coordinator/`check_all`/`check_value_function` passing extra args (keyword-only, defaulted: `bellman_error`/`value_loss` primary, `value_nrmse`/`v_return_correlation` secondary, `ev_low_return_variance`) + the unflagged-mean coupling test; revert restores the EV-only threshold. Because the change is *stricter* on the EV arm and additive on the primary-robust arm, a revert can only make the gate noisier, never strand a run.

---

## Step 6 — RED→GREEN: downstream consumer remediation (Acceptance #5, #7)

> Each sub-step is independently RED-testable where a test surface exists (karn views, persistence). Pure web/wandb display changes are verified by the karn-view / contract tests plus a targeted assertion where a test harness exists.

**Files:** `karn/mcp/views.py`; `aggregator.py`; `schema.py`; `snapshot_copy.py`; `karn/sanctum/widgets/reward_health.py`; `nissa/wandb_backend.py`; `vectorized.py`; Overwatch web (`ExperimentVerdictPanel.vue`, `HealthGauges.vue`, `sanctum types.ts`); `tests/karn/mcp/test_views.py`; `tests/karn/sanctum/test_reward_health.py`; `tests/karn/sanctum/test_aggregator.py`; `tests/nissa/test_wandb_backend.py`; web **Vitest** component specs under `src/esper/karn/overwatch/web/src/components/__tests__/` (B5).

### 6.1 RED — karn views carry new columns + stricter confounder + round-trip persistence
- **Extend** `tests/karn/mcp/test_views.py::test_ppo_updates_exposes_ev_robustness_columns`: assert `ppo_updates` exposes `value_nrmse`, `ev_low_return_variance`, `ev_return_variance` (`views.py:127`).
- **The fix is UPSTREAM, not in the view (Locked decision B2).** `run_confounders` is event-type based and correctly marks an *emitted* `VALUE_COLLAPSE_DETECTED` as `proof_blocking=true` (`views.py:596`, `:649-652`). That is the right semantics for a genuinely emitted collapse — asking the view to suppress an *existing* collapse event would move the fix to the wrong layer and risk hiding real defects. So the regression is asserted at the **detector/coordinator**, not the view:
  - **New test** `tests/simic/telemetry/...::test_coordinator_does_not_emit_value_collapse_on_artifact`: drive `PPOCoordinator → check_all` with an artefactual low-EV update (`ev_low_return_variance=True`, healthy primary `bellman_error`/`value_loss`); assert **no** `VALUE_COLLAPSE_DETECTED` event is emitted. The stricter Step 5 gate is what prevents the artefactual event from ever existing; with no event emitted, `run_confounders` has nothing to proof-block.
  - **Keep `run_confounders` proof-blocking for emitted collapse events** unchanged — do NOT add a view-level filter that suppresses an emitted `VALUE_COLLAPSE_DETECTED` row, unless the event schema is explicitly expanded with an artifact/non-collapse discriminator AND the view design is updated accordingly (out of scope here). The view's cross-update EV std/mean rollup may still filter `ev_low_return_variance == False` rows for the *display* statistic, but this is distinct from confounder proof-blocking.
- **New test** `tests/karn/sanctum/test_aggregator.py::test_ev_robustness_fields_round_trip`: emit a `PPOUpdatePayload` with all new fields populated; push it through `aggregator → schema → snapshot_copy` and assert the values survive intact (the production data path; view tests use raw-events fixtures and would miss a dropped aggregator/schema field).
- **GREEN:** add the three columns to `ppo_updates` (`:127`); persist new fields through `aggregator.py`, `schema.py`, `snapshot_copy.py`. **Do NOT change `run_confounders` proof-blocking semantics** (B2) — it stays event-type based; the artefactual-collapse suppression is achieved upstream by the Step 5 gate not emitting the event. The optional `ev_low_return_variance == False` filter on the `run_confounders` cross-update EV std/mean is a **display-statistic** refinement only and must not suppress an emitted `VALUE_COLLAPSE_DETECTED` proof-blocking row.
- **Command:** `uv run pytest tests/karn/mcp/test_views.py tests/karn/sanctum/test_aggregator.py -q`

### 6.2 RED — TUI health bool on robust signal
- **Test** `tests/karn/sanctum/test_reward_health.py::test_is_ev_healthy_uses_robust_signal`: `is_ev_healthy` (`reward_health.py:39`) is computed from a `value_nrmse` threshold (or EV only when `ev_low_return_variance == False`); assert an artefactual-low-EV / good-`value_nrmse` snapshot reports `is_ev_healthy == True`.
- **No-legacy:** in the SAME change, **DELETE** the old `ev_explained > 0.5` assertion/predicate (do not leave it alongside). If existing tests in `test_reward_health.py` assert the old predicate, update/remove them in this step.
- **GREEN:** rewrite `is_ev_healthy` accordingly.
- **Command:** `uv run pytest tests/karn/sanctum/test_reward_health.py -k is_ev_healthy -q`

### 6.3 RED — wandb + vectorized metric registration
- **Test** `tests/nissa/test_wandb_backend.py::test_wandb_emits_ev_robustness_metrics`: assert `ppo/value_nrmse` and `ppo/ev_low_return_variance` appear in the emitted metric dict when the payload carries them.
- **Test** `tests/simic/test_vectorized.py::test_ev_robustness_metrics_registered`: assert `value_nrmse`, `ev_return_variance`, `ev_low_return_variance`, `ev_low_return_variance_count` are in the appropriate reducer frozensets (the Step 3.4 registration; locks against a dropped reducer that would KeyError).
- **GREEN:** add `ppo/value_nrmse`, `ppo/ev_low_return_variance` at `wandb_backend.py:389-390`; confirm reducer registration from Step 3.4 (and aux at `:241`).
- **Command:** `uv run pytest tests/nissa/test_wandb_backend.py -k ev_robustness tests/simic/test_vectorized.py -k metrics_registered -q`

### 6.4 Overwatch web (Acceptance #7 — display-masking fix)
- `HealthGauges.vue:86`: **remove the `[0,1]` clamp** that masks negative EV; render the real value, and when `ev_low_return_variance` is set render a "low-return-variance" badge instead of a misleading red gauge (`:21-22` thresholds, `:91-92` status updated accordingly). This is the no-bug-hiding fix (Locked decision 8).
- `ExperimentVerdictPanel.vue:127-138`: drive `policyGate` off `value_nrmse` / `v_return_correlation`; ignore EV on flagged updates.
- `sanctum types.ts:316,405`: add the new fields. **First confirm the `sanctum.ts` generation path** — it IS generated: `package.json` exposes `npm run generate:types` → `scripts/generate_overwatch_types.py` → `src/types/sanctum.ts` (recent commit `6d97391b` "regenerate stale sanctum.ts"). Regenerate via that script rather than hand-editing (Information Gap 4 RESOLVED).
- **Command (B5 — Vitest, not Playwright):** the `HealthGauges` / `ExperimentVerdictPanel` assertions live in **Vitest component specs** (`src/components/__tests__/HealthGauges.spec.ts`, `ExperimentVerdictPanel.spec.ts`), and the `test` script is `vitest`. Playwright (`testDir: './e2e'`, verified `playwright.config.ts:14`) does NOT collect these specs, so a Playwright grep would pass while running zero of the relevant assertions. Use:
  `npm --prefix src/esper/karn/overwatch/web test -- --run HealthGauges ExperimentVerdictPanel`
  (or `npm --prefix src/esper/karn/overwatch/web test -- --run -t "HealthGauges|ExperimentVerdictPanel"`). Keep a Playwright smoke ONLY if browser-level wiring is added to `./e2e` for this change.

**Rollback note:** each consumer change is independent and additive (columns/fields) or a predicate swap; revert per-file restores prior behavior. The web clamp removal is a display fix with no data-contract impact.

---

## Step 7 — RED→GREEN: aux EV consistency (Acceptance #9)

**Files:** `src/esper/simic/agent/ppo_agent.py`; `src/esper/simic/agent/types.py`; `src/esper/simic/training/vectorized.py`; `tests/simic/test_ppo_value_metrics.py`.

> **Aux floor pre-flight (REQUIRED, parallel to Step 0 — Info Gap 2 upgraded to a blocking constraint).** The aux EV target is `contribution_targets` (contribution-prediction scale), structurally different from raw returns. `ppo_agent.py:1269` already does `target_var.clamp(min=0.01)` on THAT scale. **Do NOT reuse `ev_return_variance_floor` (raw-return scale, ~1.0)** — it would be a scale error. Before committing the aux floor, **lock a separate `aux_ev_return_variance_floor` constant** (which may remain `0.01` if the data supports it) from `contribution_targets.var()` percentiles. **Karn SQL CANNOT supply this (W6):** `contribution_targets` is an in-buffer update tensor (`ppo_agent.py:1239`, `:1263`), and the persisted `PPOUpdatePayload` / emitters (`telemetry.py:701`, `emitters.py:974`) expose only aggregate fields — there is no raw contribution-target column to query. So the pre-flight is an **in-process harness/test**, not a Karn SQL preflight: either (a) instrument a short K=4 `update()` run to print `contribution_targets.var()` percentiles, OR (b) first persist a new `aux_ev_target_variance` metric and then query that. Record the percentile rationale inline.

### 7.1 RED
- **New test** `test_aux_ev_floored_and_flagged`: on a low-`target_var` aux batch, assert `aux_explained_variance` is bounded and `aux_ev_low_return_variance is True`.
- **New test** `test_aux_ev_degenerate_batch`: numel<2 aux target; assert the degenerate aux batch is handled consistently with the main EV path (Locked decision B3 — a degenerate aux target is a hard bug; the aux floor applies only to finite `numel>=2` targets and must NOT NaN-sanitize a degenerate batch). If the aux path lacks an equivalent loud guard upstream, add one rather than emitting NaN-by-convention.
- **Expected RED:** no aux flag emitted.

### 7.2 GREEN
- At `ppo_agent.py:1268-1271`, replace the bare `clamp(min=0.01)` with the same floor-and-flag mechanism using the **separate, Step-7-pre-flight-locked `aux_ev_return_variance_floor`** (config-exposed ctor kwarg; do NOT reuse `ev_return_variance_floor`). Apply the same explicit small-N / non-finite guard as the main EV. Emit `metrics["aux_ev_low_return_variance"]` at the existing aux emit sites (`:1285,1314,1320,1328`). Add `aux_ev_low_return_variance` to `agent/types.py:189` neighborhood and register it in the appropriate reducer frozenset (`vectorized.py`, alongside `aux_explained_variance` at `:241`) so `_aggregate_ppo_metrics` does not KeyError.
- **GREEN command:** `uv run pytest tests/simic/test_ppo_value_metrics.py -k aux_ev -q`

**Rollback note:** aux EV change is isolated to the `:1268-1271` block + aux floor kwarg + aux emit/type/reducer registration; revert restores the bare clamp.

---

## Step 8 — RED→GREEN: emitter mislabel fix (Acceptance #8)

**Files:** `src/esper/simic/telemetry/emitters.py`; `tests/karn/test_analytics_snapshot_policy.py`; `tests/karn/mcp/test_views.py`.

> `emitters.py:472` writes EV into `AnalyticsSnapshotPayload.value_variance` → surfaces as `batch_stats.value_variance` in karn (`value_variance=metrics.get("explained_variance", 0.0)`). The field never held value variance. Fix the mislabel (no-legacy: do not keep the wrong name).

### 8.1 RED — audit readers NOW (required pre-merge checklist, not deferred), then assert correct labeling
- **Run the reader audit immediately** and enumerate the results in the test docstring + this step: grep `batch_stats.value_variance` across `karn/mcp/views.py`, all sanctum/overwatch panels, and any external consumer. Classify each reader as either:
  - (a) **was reading EV under the wrong name** → repoint to the EV-named field, OR
  - (b) **was reading `value_variance` expecting actual variance** → repoint to `return_variance`/`ev_return_variance` or the genuine value variance.
  Record the enumerated reader list as a required pre-merge checklist item.
- **New test** asserting EV is emitted under an EV-named field and the value-variance field carries actual value/return variance (per the audited semantics).
- **Expected RED:** EV currently flows into `value_variance`.

### 8.2 GREEN
- At `emitters.py:472`, emit EV under the EV-named field and put the real variance under the value-variance field. Per the no-bug-hiding rule, access the corrected field with **direct key access** (`metrics["explained_variance"]` / `metrics["ev_return_variance"]`) rather than `.get(..., 0.0)` — a missing mandatory metric is a bug that must surface. Update the karn `batch_stats` view + every audited reader to the corrected source.
- **GREEN command:** `uv run pytest tests/karn/test_analytics_snapshot_policy.py tests/karn/mcp/test_views.py -q`

**Rollback note:** the mislabel fix is a correctness change touching `emitters.py:472` + the `batch_stats` view + audited readers; revert restores the (wrong) prior wiring. Because the field never held the right data, the only "risk" is a chart that was already misleading.

---

## Acceptance-criterion → test traceability

| # | Criterion (spec headline) | Test(s) | Step |
|---|---------------------------|---------|------|
| 1 | EV floored at the source; old `1e-8 → 0.0` deleted; healthy batch unchanged; degenerate batch fails loudly (hard bug, not NaN-by-convention) | `test_ev_floored_on_low_return_variance`, `test_ev_unchanged_on_healthy_variance`, `test_ev_no_special_case_zero`, `test_ev_floor_default_matches_calibration`, `test_ev_degenerate_batch_raises_loudly` | 1 |
| 2 | Robust companion `value_nrmse` finite/small on artifact, monotone in residual, degenerate-guarded | `test_value_nrmse_finite_and_monotone`, `test_value_nrmse_bounded_on_low_variance`, `test_value_nrmse_degenerate_batch` | 2 |
| 3 | Flag + `value_nrmse` + `ev_return_variance` + count plumbed end-to-end (leyline → types → metrics → reducer → emitter → karn → wandb); `from_dict` defaults on old events (additive contract) | `test_ppo_update_payload_carries_ev_robustness_fields`, `test_ppo_update_payload_from_dict_old_event_without_ev_fields`, `test_ppo_update_metrics_carries_ev_robustness_fields`, `test_aggregate_ppo_metrics_no_keyerror_on_ev_fields`, `test_emitter_carries_ev_robustness_fields`, `test_ppo_updates_exposes_ev_robustness_columns`, `test_ev_robustness_fields_round_trip` | 2,3,4,6 |
| 4 | Cross-update aggregation excludes flagged updates from EV mean/std; flagged count reported; all-flagged → nan | `test_ev_aggregation_excludes_flagged_updates`, `test_ev_aggregation_all_flagged` | 3 |
| 5 | Gates no longer false-alarm on the artifact (fix is UPSTREAM at the detector, not the confounder view) | `test_check_value_function_ignores_artifactual_low_ev`, `test_coordinator_does_not_emit_value_collapse_on_artifact` | 5,6 |
| 6 | Genuine value collapse still fires on primary robust signals (not bug-masking) | `test_check_value_function_fires_on_genuine_collapse` | 5 |
| 6b | Primary-robust arm (`bellman_error`/`value_loss`) fires on low-var co-occurrence end-to-end; disagreement behaviour intentional + gate sees unflagged-only EV | `test_check_value_function_fires_on_low_var_collapse_via_robust_arm`, `test_coordinator_emits_value_collapse_on_low_var_real_collapse`, `test_check_value_function_disagreement_ev_bad_robust_ok`, `test_gate_sees_unflagged_only_ev_mean` | 5 |
| 7 | Display masking fixed (HealthGauges no `[0,1]` clamp; low-variance badge) | Vitest `npm --prefix src/esper/karn/overwatch/web test -- --run -t "HealthGauges\|ExperimentVerdictPanel"` (B5 — component specs, not Playwright/e2e) | 6.4 |
| 8 | Mislabel corrected (`emitters.py:472`); readers audited | Step 8 emitter + view tests | 8 |
| 9 | Aux EV floored + flagged consistently on its own scale | `test_aux_ev_floored_and_flagged`, `test_aux_ev_degenerate_batch` | 7 |

## Sequencing & dependencies

- **Step 0 (calibration) first** — produces the floor constant the Step 1 tests assert against AND the tail value-fit evidence for Acceptance #6b. **Step 7 has its own aux-floor pre-flight** (separate scale; do not reuse the main floor).
- **Steps 1 → 2 → 3 → 4** are the source→contract spine and must land in order (each depends on the prior field existing). Step 3 includes the reducer registration that prevents the `_aggregate_ppo_metrics` KeyError — without it the PPO update crashes.
- **Step 5 (gate migration) depends on Steps 3–4** (the gate reads the flag + robust signals from the aggregated/emitted payload). **HARD ORDERING:** the gate consumes the AGGREGATED `explained_variance` (`ppo_coordinator.py:516`), so the Step 3.4 flagged-exclusion rule is the exact coupling Step 5 keys on — `test_gate_sees_unflagged_only_ev_mean` locks it. Step 5 carries the with-teeth acceptance criteria (#5, #6, #6b) and **must not** merge before #6 and #6b are GREEN. The `check_all` signature change and the coordinator call-site update land in the SAME commit (no TypeError window).
- **Step 6 (consumers) depends on Step 5** (views inherit the stricter gate) and on Steps 2–3 (fields to display).
- **Steps 7 (aux) and 8 (mislabel)** are independent of 5/6 and can land in parallel after Step 1 (7) / Step 4 (8). Step 8's reader audit is a required pre-merge checklist item, not deferred.
- Suggested PR split: **PR-A** Steps 0–4 (source fix + contract + reducers, zero gate behavior change); **PR-B** Steps 5–6 (gate migration + consumers, the with-teeth change, requires drl-expert sign-off on thresholds); **PR-C** Steps 7–8 (aux + mislabel). PR-A is safe to ship alone (additive telemetry, no gate change). PR-B must not precede PR-A.

## Full verification gate

- **Golden preflight (do this first):** `grep -n 'var_returns\|return_variance\|low_variance\|return_std' tests/simic/test_ppo_update_golden.py` to find any low-variance fixture before running goldens; document the result inline. If a fixture is in the low-variance regime, regenerate that golden and document it as a deliberate floor change (Acceptance #1 guarantees healthy-variance goldens are numerically unchanged).
- `uv run pytest tests/simic/test_ppo_value_metrics.py tests/simic/test_telemetry_fields.py tests/simic/test_vectorized.py tests/simic/test_ppo_update_golden.py -q`
- `uv run pytest tests/simic/telemetry -q`
- `uv run pytest tests/karn/mcp/test_views.py tests/karn/test_analytics_snapshot_policy.py tests/karn/sanctum/test_reward_health.py tests/karn/sanctum/test_aggregator.py -q`
- `uv run pytest tests/nissa/test_wandb_backend.py -q`
- Regression set: `uv run pytest tests/simic/test_ppo.py tests/simic/test_ppo_normalization.py -q` (confirm EV-on-healthy-batch unchanged; no golden drift on healthy-variance fixtures).
- **Web component assertions (B5 — Vitest, not Playwright):** `npm --prefix src/esper/karn/overwatch/web test -- --run -t "HealthGauges|ExperimentVerdictPanel"` (the component specs live under `src/components/__tests__/`; Playwright's `./e2e` does not collect them). Add a Playwright e2e smoke only if browser-level wiring is in scope.
- **CI parity gates (W8 — REQUIRED before closeout; these are enforced in CI and the focused pytest/Vitest runs do not cover them):**
  - `uv run python scripts/lint_leyline_types.py` (leyline boundary — relevant: new fields are added in leyline first)
  - `uv run python scripts/lint_defensive_patterns.py` (no-bug-hiding — relevant: direct-key vs `from_dict` `.get` discipline)
  - `uv run python scripts/lint_gpu_sync.py` (GPU-sync — relevant: new tensor `.item()`/`bool()` conversions in the EV helper)
  - `uv run ruff check src/ tests/`
  - `MYPYPATH=src uv run mypy -p esper`
- Confirm each acceptance criterion maps to a GREEN test (table above).

## Residual risks (carried into the plan)

1. **Floor mis-set** (medium): too high attenuates legitimate negative EV on moderate-variance batches. Mitigation: Step 0 empirical calibration + tail value-fit validation; config-exposed; unguarded `value_loss` / `v_return_correlation` / `bellman_error` never floored; the robust arm (Acceptance #6b) fires independently of the flag. drl-expert review on the locked value.
2. **Gate migration masks a real collapse** (medium, with teeth): the EV arm could be set so loose it never fires. Mitigation: Acceptance #6 + #6b mandatory regression locks; the EV arm is **stricter** (bad PRIMARY robust signal AND non-flagged EV), and the **primary-robust arm keyed on scale-anchored `bellman_error`/`value_loss`** (NOT the floor-stabilized `value_nrmse` or sentinel `v_return_correlation`) fires on real collapse regardless of variance (B1); the coordinator-level test proves the primary signals are threaded end-to-end; drl-expert review on thresholds.
3. **Degenerate-batch handling** (Locked decision B3 — hard bug, fails loudly): degenerate `numel<2` valid-return masks are NOT a telemetry-safe NaN case. The existing non-finite return-stat raise at `ppo_agent.py:636` is the correct loud-failure path and is preserved; the floored EV path runs only on finite `numel>=2` batches and never NaN-sanitizes via `clamp`. No NaN reaches `_aggregate_ppo_metrics` (which skips only `None`, not NaN), so no reducer poisoning. Locked by `test_ev_degenerate_batch_raises_loudly`.
4. **`_aggregate_ppo_metrics` KeyError** (was a blocker): new keys MUST be registered in reducer frozensets or the PPO update crashes. Locked by `test_aggregate_ppo_metrics_no_keyerror_on_ev_fields` and `test_ev_robustness_metrics_registered`.
5. **Mislabel fix ripples** into `batch_stats.value_variance` readers (low): Step 8.1 audits readers before renaming (required pre-merge checklist).
6. **Contract growth across many surfaces** (low): additive fields, NaN/False defaults, no `VALUE_HEAD_SCHEMA_VERSION` bump.
7. **Golden drift** (low): only low-variance golden fixtures (if any) change; healthy-variance goldens numerically unchanged. Golden preflight grep guards this.

## Critical files

- `/home/john/esper-lite/src/esper/simic/agent/ppo_agent.py`
- `/home/john/esper-lite/src/esper/simic/agent/types.py`
- `/home/john/esper-lite/src/esper/simic/agent/ppo_metrics.py`
- `/home/john/esper-lite/src/esper/simic/training/vectorized.py`
- `/home/john/esper-lite/src/esper/leyline/telemetry.py`
- `/home/john/esper-lite/src/esper/leyline/value_metrics.py`
- `/home/john/esper-lite/src/esper/simic/telemetry/emitters.py`
- `/home/john/esper-lite/src/esper/simic/training/ppo_coordinator.py`
- `/home/john/esper-lite/src/esper/simic/telemetry/anomaly_detector.py`
- `/home/john/esper-lite/src/esper/karn/mcp/views.py`
- `/home/john/esper-lite/src/esper/karn/sanctum/widgets/reward_health.py`
- `/home/john/esper-lite/src/esper/nissa/wandb_backend.py`
- `/home/john/esper-lite/tests/simic/test_ppo_value_metrics.py`
- `/home/john/esper-lite/tests/simic/test_telemetry_fields.py`
- `/home/john/esper-lite/tests/simic/test_vectorized.py`
- `/home/john/esper-lite/tests/karn/mcp/test_views.py`
- `/home/john/esper-lite/tests/karn/test_analytics_snapshot_policy.py`
- `/home/john/esper-lite/tests/karn/sanctum/test_reward_health.py`
- `/home/john/esper-lite/tests/karn/sanctum/test_aggregator.py`
- `/home/john/esper-lite/tests/nissa/test_wandb_backend.py`

---

## Confidence Assessment

**Overall Confidence: High.** All code anchors re-verified against source on 2026-06-18 and path/symbol-corrected after a dedicated audit: emitter `src/esper/simic/telemetry/emitters.py` (:472 mislabel, :986 wiring); coordinator `src/esper/simic/training/ppo_coordinator.py` (:512-524); payload class `PPOUpdatePayload` (`leyline/telemetry.py:701`, field :719) — `PolicySnapshotPayload` does not exist; `return_variance` pre-exists (`types.py:112`, `telemetry.py:748`) and is NOT re-added; cross-update aggregation is `vectorized.py:352` `_aggregate_ppo_metrics` (KeyError at :397), NOT `ppo_metrics.py:63`; `check_value_function` (`anomaly_detector.py:140`) is invoked from `check_all` (:485); `ppo_agent.py` has no `import math`. The no-bug-hiding argument is load-bearing: the floor and `value_nrmse` denominator floor are legitimate numerical handling because the unguarded companions remain ground truth and the flag makes the substitution observable; new mandatory fields are accessed by direct key.

| Finding | Confidence | Basis |
|---------|------------|-------|
| Emitter / coordinator paths corrected to `telemetry/` and `training/` | High | `ls` confirmed the non-`telemetry`/non-`training` paths do not exist |
| `PolicySnapshotPayload` does not exist; class is `PPOUpdatePayload` | High | `grep -rn` zero hits for `PolicySnapshotPayload`; class at `telemetry.py:701` |
| `return_variance` pre-exists; new field is `ev_return_variance` | High | Read `types.py:112`, `telemetry.py:748`, `ppo_metrics.py:221` |
| Cross-update EV reducer is `_aggregate_ppo_metrics` with KeyError fallthrough | High | Read `vectorized.py:352-398`, frozensets :184-304 |
| `ppo_agent.py` lacks `import math`; degenerate var is NaN under default Bessel | High | Read `ppo_agent.py:1-57, 619-628` |
| `check_value_function` called from `check_all`, fed by coordinator :516 | High | Read `anomaly_detector.py:447-485`, `ppo_coordinator.py:512-524` |
| Aux EV target is a different scale (contribution prediction) | High | Read `ppo_agent.py:1268-1271` (`clamp(min=0.01)`) |
| Empirical floor default 1.0 sits below operating band | Moderate | Multi-epoch design doc (std 7–13); to be confirmed in Step 0 |

## Risk Assessment

**Implementation Risk: Medium.** The source fix (Steps 1–2) and contract plumbing (Steps 3–4) are mechanical with explicit unit tests, but two previously-latent blockers (degenerate-batch NaN leak; `_aggregate_ppo_metrics` KeyError) are now regression-locked. The with-teeth risk is concentrated in Step 5 (gate migration), gated by Acceptance #6 / #6b and a mandated stricter-not-blinder condition with an independent robust arm. **Reversibility: Easy** — additive payload fields, NaN/False defaults, no schema-version bump; old telemetry consumers tolerate missing fields via defaults; no checkpoint format change.

| Risk | Severity | Likelihood | Mitigation |
|------|----------|------------|------------|
| Gate migration masks a real value collapse | High | Possible if thresholds loose | Acceptance #6 + #6b regression locks; stricter EV arm + independent robust arm; drl-expert threshold review (Step 5) |
| Degenerate batch (numel<2) silently swallowed into telemetry | High | Certain if a NaN-acceptance path were added | Locked decision B3: degenerate masks are a hard bug; existing `:636` return-stat raise preserved; NO NaN-by-convention path; `test_ev_degenerate_batch_raises_loudly` |
| `_aggregate_ppo_metrics` KeyError on unregistered new keys | High | Certain without registration | Reducer registration (Step 3.4); `test_aggregate_ppo_metrics_no_keyerror_on_ev_fields` |
| Floor mis-set attenuates legitimate negative EV | Medium | Possible | Step 0 calibration + tail value-fit validation; config-exposed; robust arm fires independently |
| Aux floor reuses raw-return scale | Medium | Possible | Step 7 aux-floor pre-flight; separate `aux_ev_return_variance_floor`; never reuse `ev_return_variance_floor` |
| Mislabel rename breaks a `batch_stats.value_variance` chart | Medium | Possible | Step 8.1 reader audit (required pre-merge checklist) |
| Web clamp removal regresses gauge rendering | Low | Possible | Vitest component specs `HealthGauges` / `ExperimentVerdictPanel` (B5 — `npm test -- --run`, NOT Playwright/e2e) (Step 6.4) |

## Information Gaps

1. [ ] **Floor numeric** (Step 0): the locked `ev_return_variance_floor` value must be confirmed from the recent K=4 run's `return_std` distribution before the Step 1 tests are finalized, AND `test_ev_floor_default_matches_calibration` must assert the ctor default equals it.
2. [x] **Aux floor basis** (Step 7): RESOLVED to a required pre-flight (parallel to Step 0). Compute `contribution_targets.var()` percentiles via an **in-process harness/test** (W6: `contribution_targets` is an in-buffer tensor, NOT persisted/queryable via Karn SQL — `ppo_agent.py:1239`/`:1263`; emitters expose only aggregates), OR first persist an `aux_ev_target_variance` metric and query that. Lock a SEPARATE `aux_ev_return_variance_floor`; do NOT reuse the raw-return floor.
3. [x] **`batch_stats.value_variance` reader set** (Step 8): RESOLVED to a required pre-merge audit in Step 8.1 (not deferred) — grep + classify every reader before renaming.
4. [x] **Web type-gen** (Step 6.4): RESOLVED — `sanctum.ts` is generated via `npm run generate:types` → `scripts/generate_overwatch_types.py` (confirmed in `package.json`). Regenerate; do NOT hand-edit.

## Caveats & Required Follow-ups

**Before relying on this plan:**
- [ ] Run Step 0 calibration (floor + tail value-fit) and lock the floor constant; finalize Step 1 test bounds + `test_ev_floor_default_matches_calibration` against it.
- [ ] Run the Step 7 aux-floor pre-flight and lock `aux_ev_return_variance_floor` on the contribution-target scale.
- [ ] Run the Step 8.1 `batch_stats.value_variance` reader audit and enumerate readers before renaming.
- [ ] Obtain drl-expert sign-off on the Step 5 gate thresholds (PRIMARY `bellman_error` / `value_loss`; secondary `value_nrmse` / `v_return_correlation`) before PR-B merges.
- [ ] Extract the EV helper seam (Step 1.2, W7) into `leyline/value_metrics.py` (or a private `PPOAgent` method) BEFORE writing the Step 1/2 unit tests, and add the end-to-end `test_ppo_update_emits_ev_robustness_fields` wiring regression.
- [ ] Thread the PRIMARY robust signals `bellman_error` + `value_loss` (Step 5, B1) through coordinator → `check_all` → `check_value_function` and add `test_coordinator_emits_value_collapse_on_low_var_real_collapse`.
- [ ] Generate web types via `npm run generate:types` (Step 6.4) rather than hand-editing `sanctum.ts` (Information Gap 4 RESOLVED).

**Assumptions:** P0-1 (op-marginal V(s)) is landed and stable on 0.1.1; the EV-std blowout is the denominator artifact characterized in the spec (corroborated by lowest `value_loss` + best median EV co-occurring with the crater); `VALUE_HEAD_SCHEMA_VERSION` is **not** touched (telemetry payload only).

**Limitations:** This plan does not re-run training or execute the floor against live telemetry beyond the Step 0 / Step 7 calibrations; it verifies line/field/signature facts the steps depend on. It does not modify the value-std-derived `value_collapse` path (`karn/triggers.py`, `store.py`) — explicitly out of scope (different signal, same name).
