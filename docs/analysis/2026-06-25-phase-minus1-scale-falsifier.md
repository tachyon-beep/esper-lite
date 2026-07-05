# Phase −1 Scale-Fix Falsifier — Evidence Packet (GATE −1, scale leg)

**Status:** IMPLEMENTED + DESIGN-SIGNED-OFF + SMOKED — **A/B runs PENDING** (verdict PROVISIONAL).
**Date:** 2026-06-25
**Branch:** `feat/phase-minus1-scale-falsifier` (based on `0.3.0`, which carries EV-stab Stage-2 HRA — the reward-scale change therefore sits *on top of* the critic reparam, never bundled).
**Code commit:** `f1f2551f`
**Methodology:** `docs/plans/concepts/2026-06-24-reward-redesign-methodology.md` §5 (Phase −1), GATE −1. *(That doc currently lives on branch `codex/sanctum-pre-ready-crash`, pending merge into 0.3.0; this packet is self-contained.)*
**Sibling falsifier (OUT OF SCOPE here):** the controller-contribution gate, `docs/analysis/2026-06-23-controller-contribution-gate-runsheet.md` — feeds the same GATE −1/GATE 1 but is a different lever.

---

## 0. What this is (and is NOT)

This is a **falsifier**, not a feature. Its purpose is to potentially **KILL or down-scope** the
L-complexity reward redesign. The verified root cause of the reward pathology is a per-step **SCALE**
bug: SHAPED `bounded_attribution` is **un-clipped** (`contribution.py`, the `bounded_attribution =
config.contribution_weight * attributed` assignment) over a counterfactual measured in **0–100 accuracy
points** → O(6–16)/step, ~100× over the only final-tied term (`terminal_bonus = val_acc*0.05` ≈ 2.5
once). ESCROW already clips at `escrow_delta_clip=2.0`.

**Success = a clean GATE −1 verdict either way, NOT shipping a new reward.** If a cheap lever
(clip / unit-normalize / the existing ESCROW mode) resolves the farmable-tail + churn pathology, the
redesign is the WRONG lever and we **STOP**. If the pathology survives all three cheap levers on
`max_seeds ≥ 2`, the **scale leg of GATE −1 is satisfied** and admission is **PENDING the sibling
contribution-gate leg** — *ADMIT is not producible from this experiment alone.*

### The STOP/ADMIT asymmetry (load-bearing)
- **STOP is producible here alone.** A lever that resolves the behavioural pathology kills/down-scopes
  the redesign → recommend **STOP_THEORY / REVISE_ALGORITHM** (fix at config level).
- **ADMIT is NOT producible here.** SURVIVE-on-all-levers satisfies *only* the scale leg; GATE −1 is a
  conjunction also requiring the contribution-gate leg (CONTINUE-or-ambiguous). Report
  *"scale leg satisfied; admission PENDING the contribution-gate leg."* Do **not** declare GATE −1 passed
  and do **not** build the reward.

---

## 1. The verified root cause (cited-run data, re-measured 2026-06-25 via Karn)

Run `telemetry_2026-06-24_033049` (config-3slot-1seed-baseline-shaped, SHAPED, 12 vec-envs), `rewards`
view, **45,696 reward rows**:

| metric | value |
|---|---|
| positive `bounded_attribution` steps | 1,278 |
| negative steps | 413 |
| steps `> +2.0` | **681** (matches the methodology's cited 681/1278) |
| steps `< −2.0` | 69 |
| min / max | −15.346 / +16.479 |
| mean | +0.0774 |

**Two consequences for the design (both confirmed by the design-review panel):**
1. **Negatives are common and large** (413 steps; 69 below −2.0; min −15.3). A *symmetric* clamp would
   cap those harmful-seed penalties up to −2.0, cheapening the cost of holding harmful seeds — a confound
   orthogonal to the positive-tail hypothesis. ⇒ **Arm A clip is POSITIVE-ONLY** (decision D2).
2. **CAVEAT (panel finding):** these counts come from `components.bounded_attribution`, which is recorded
   **POST** the PRUNE sign-flip + `ratio_penalty`, whereas the clip acts **PRE**-flip. So for the A/B the
   "fraction > 2.0" *sanity* metric must be computed on **non-PRUNE steps** (where recorded == pre-flip
   value), or it will misread (a prune-of-harmful-seed has a negative pre-flip value that the
   positive-only clip leaves alone, then the flip makes it a large positive that survives). See §5.

---

## 2. The four arms (identical seeds / host / env / γ / action-costs)

All arms use `max_seeds = 3` (≥ 2 mandatory: under `max_seeds=1`, committing *ends* morphogenesis, so
"never commit" is partly structurally rational and confounds the result).

| Arm | Config | Change | sha256 (12 hex) |
|---|---|---|---|
| **Control** | `config-3slot-3seed-baseline-shaped.json` | none (status-quo SHAPED) | `7f199a3ac1b3` |
| **A — clip 2.0** | `config-3slot-3seed-baseline-shaped-clip2.json` | `shaped_attribution_clip=2.0` | `6958cb6d5dda` |
| **A — clip 5.0** | `config-3slot-3seed-baseline-shaped-clip5.json` | `shaped_attribution_clip=5.0` | `b6cbe605aca7` |
| **B — unit-norm** | `config-3slot-3seed-baseline-shaped-unitnorm.json` | `attribution_unit_normalize=true` | `7d3ed446898a` |
| **C — ESCROW** | `config-3slot-3seed-baseline-escrow.json` | `reward_mode=escrow` (clips at 2.0) | `1ed5f1c3ad3b` |

**Arm A (clip)** and **Arm B (/100)** are NOT two dials on one mechanism:
- **Arm B is a linear rescale, optimum-preserving for the attribution term IN ISOLATION** — but it
  down-weights attribution *relative to* the unscaled rent / terminal / action-shaping terms, so it
  re-weights the composite objective (a milder optimum shift than Arm A's nonlinearity, **not** a pure
  null). A B-only fix means the *mean-scale of the dense channel* was the dominant driver.
- **Arm A is the optimum-changing NONLINEARITY** — an A-only fix means the *farmable tail / nonlinearity*
  matters, not mean scale.
- **Arm C (ESCROW) is a DIRECTIONAL sanity arm only.** ESCROW uses `sqrt`/geometric mean vs SHAPED's
  `harmonic`, so an ESCROW-vs-SHAPED delta mixes the clip with the algebra. **Do not** read it as a clean
  isolation of the clip.

---

## 3. The code (committed `f1f2551f`) and its three design decisions

Two CONFIG-FLAGGED, default-OFF knobs on `ContributionRewardConfig`:
```python
shaped_attribution_clip: float = 0.0      # 0.0 = OFF
attribution_unit_normalize: bool = False  # False = OFF
```
Single guarded block, placed **inside the non-escrow branch, after both attribution sub-branches set
`bounded_attribution`, and before the FOSSILIZE-suppression / PRUNE sign-flip / `ratio_penalty`
post-processing**:
```python
if config.attribution_unit_normalize:
    bounded_attribution /= 100.0
if config.shaped_attribution_clip > 0.0:
    bounded_attribution = min(config.shaped_attribution_clip, bounded_attribution)  # POSITIVE-ONLY
```

| # | Decision | Rationale | Panel verdict |
|---|---|---|---|
| **D1** | Both levers cover the **proxy** channel (the block acts on the *unified* `bounded_attribution`, covering clean-counterfactual AND `proxy_contribution_weight·improvement_since_stage_start`). | The proxy path is "churn fuel" (§8); leaving it uncapped would confound a SURVIVE with an open farming channel (biases toward MORE work). | **CONFIRMED correct** (code-verified, both lenses) |
| **D2** | Arm A clip is **positive-only** (`min(clip,x)`), not symmetric. | Caps the farmable positive tail; leaves the negative-contribution penalty intact (symmetric would cheapen holding harmful seeds — see §1). | **CONFIRMED correct** (code-verified, both lenses) |
| **D3** | Arm B `/100` applies to **both signs**. | Dimensional consistency; avoids a 100:1 penalty:credit asymmetry. Linear rescale — optimum-preserving for the attribution term *in isolation* (re-weights it vs the unscaled rent/terminal terms; see §2). | **CONFIRMED correct** (code-verified, both lenses) |

**HARD CONSTRAINTS verified respected** (by both reviewing lenses): no new `RewardMode`; no critic/
value-path edit (reward-scalar only); flags default-OFF ⇒ byte-identical status quo; the clip is an
experimental arm, never the default.

**No-legacy end-state:** if the verdict is STOP→adopt one lever, the follow-up is to make that behaviour
the single code path and **delete** the un-clipped branch + the flag. The flag is provisional; do not flip
the default in this task.

### Tests (committed)
- `tests/simic/rewards/test_reward_golden.py`: OFF byte-identical golden (clean=**20.0** / negative=
  **−10.0** / proxy=**9.0**); clip caps positive tail at the bound (2.0 and 5.0); clip covers the proxy
  channel; clip is positive-only (negative untouched); normalize divides both signs by exactly 100;
  flags do not touch ESCROW.
- `tests/simic/test_ablation_flags.py`: config plumbing (`from_dict` → `to_train_kwargs` → `to_dict`).
- **Byte-identical-OFF proof:** the full reward + properties suites (`tests/simic/rewards/`,
  `tests/simic/properties/`, `test_rewards.py`, `test_reward_modes.py`) — **227 passed, unchanged**.
- `wardline scan src/esper/simic --fail-on ERROR` → exit 0 (no ERROR-level findings; changes are pure
  numeric/config, no new trust boundary).

### Wiring smoke (2026-06-25)
`train ppo --config-json <clip2, n_episodes=1 n_envs=2 max_epochs=8 max_seeds=2>` → **exit 0**; host
trained, a seed germinated→trained→PRUNED, `EPISODE_OUTCOME` + PPO update emitted. Confirms the reward
path executes end-to-end with the flag ON. (8 epochs is too short to drive a >2.0 contribution, so the
clamp itself was not exercised at runtime — the unit tests are the clamp-math proof.)

---

## 4. Design-review sign-off (PRE-runs, mandated)

Workflow `wf_8fd217c5-c07` (`phase-minus1-design-review`), 3 lenses + synthesis. **Verdict:
`GO_WITH_CHANGES`.**

- **drl-expert** — `ENDORSE_WITH_CHANGES`. D1/D2/D3 correct; arms isolate the lever; STOP/ADMIT asymmetry
  correct; stats sound.
- **yzmir-deep-rl:reward-function-reviewer** — `ENDORSE_WITH_CHANGES`. D1/D2/D3 correct; HARD CONSTRAINTS
  verified; one blocking finding (bootstrap unit — see §5).
- **yzmir-morphogenetic-rl:morphogenesis-reviewer** — lens errored mid-stream in the first panel; **re-run
  in the closing review (below) → `ENDORSE_WITH_CHANGES`.**

The prompt-mandated reviewers (drl-expert + reward-function-reviewer) both endorsed.

**Closing review** (workflow `wf_7c34f4d3-f4a`, morphogenesis re-run + packet audit + synthesis): **final
verdict `SIGN_OFF_WITH_CHANGES`, blocking = none.** D1/D2/D3, single-knob isolation, `max_seeds=3`, and —
newly verified — **governor independence** (the clip/normalize touches no governor/safety-gate path; the
governor keys on `current_loss` only, and the sole governor↔reward edge runs the safe direction) all
code-confirmed. All six prior required changes are incorporated and code-accurate. The two non-blocking
packet edits it owed (PRUNE-flip per-arm logging in §5.3; the Arm-B optimum-preserving qualifier in §2/D3)
are **applied**.

**Load-bearing follow-up (enforcement, not a design defect):** the seed-level block-bootstrap scorer does
**not yet exist** in-repo. GATE −1 validity is enforceable ONLY if the eventual scorer resamples the 5–10
RNG **seeds** (aggregating the 12 vec-envs *within* a seed), never the envs. This is pre-registered in §5.4
and must be honored when the A/B is scored.

---

## 5. Scoring — the GATE −1 metrics (pre-registered BEFORE launch)

> **GATE −1 question:** does the farmable-tail PATHOLOGY (the *policy behaviour* — churn + non-commitment)
> survive the cheap lever? — not the raw reward magnitude.

### 5.1 Sanity check — NOT the discriminator (do not score STOP on it; it is circular)
Fraction of positive per-step `bounded_attribution` exceeding 2.0; max episode reward; episode-reward
skew. Arm A clips at 2.0 and Arm B divides by 100, so this is driven to ~0 **mechanically** — it ONLY
confirms the lever bit.
**REQUIRED (panel):** compute this on **non-PRUNE steps** (`action_name <> 'PRUNE'`), because
`bounded_attribution` is recorded post the PRUNE sign-flip; PRUNE steps mix pre-flip sign populations and
the "driven to ~0" claim is false for them.

### 5.2 Discriminator — BEHAVIORAL metrics (these decide the verdict)
From existing telemetry (`seed_lifecycle`, `decisions`, `episode_outcomes`, `rewards`):
- **Churn:** per-episode germinate count, prune count (germinate-blend-PRUNE churn); `reward↔germinate`
  / `reward↔prune` correlations (control ≈ −0.44 / −0.43, scale-invariant — neither lever can move them
  mechanically).
- **Commitment — operationalized as accuracy, NOT count (panel-required):**
  **`commitment RISES` ≡ committed final accuracy IMPROVES OVER CONTROL** on paired seeds. Fossilization
  rate (Poisson, or `P(fossilize>0)`) is **reported** but does not by itself satisfy "commitment rises"
  (a do-nothing / stop-germinating policy trivially drops churn AND sits at host baseline — the accuracy
  guard defeats that false STOP).
- **Commitment-quality co-metrics (panel-recommended, report per-arm):** median epochs-in-HOLDING before
  fossilize; blend-duration-to-fossilize; **prune-of-good-seed counts** (the positive-only clip cheapens
  the prune-of-good-seed deterrent — a SURVIVE-biased, conservative side-effect that must be reported so a
  churn-drop in Arm A is attributed to genuine commitment, not a clamp-cheapened deterrent).

### 5.3 Uncapped positive dense channels — LOG per-arm (panel-required)
The positive-only clip does NOT cap: `synergy_bonus` (~`tanh·0.1`/step), `pbrs_bonus` / epoch-progress
dwelling ramp (germinate-side PBRS has a PRUNE clawback, so germinate→prune is not a gross free farm, but
the per-step dwelling additive survives all arms). All are O(0.1)/step — tiny vs the O(6–16) tail, and all
**SURVIVE-biased**. Report `synergy_bonus` and `pbrs_bonus` per-arm so a SURVIVE cannot be silently
attributed to synergy/PBRS farming rather than residual attribution.

**Also log per-arm PRUNE-reward magnitude (post-flip positive on PRUNE steps) — a live Arm-A-vs-Arm-B
discriminator (closing-review finding).** The positive-only clip acts PRE-flip on a prune-of-harmful-seed's
*negative* pre-flip `bounded_attribution` (`min(clip, x)` leaves a negative untouched); the PRUNE
sign-inversion then flips e.g. −15 → **+15, UNCAPPED under Arm A**. Arm B's sign-agnostic `/100` (pre-flip)
scales the same value to +0.15. So Arm A leaves a residual prune-farming channel that Arm B caps. This is
**SURVIVE-biased** for Arm A (prune-farming keeps churn UP, working *against* Arm A's own STOP), so it
cannot manufacture a false STOP — but it is a genuine discriminator: a **B-STOP-without-A-STOP** is
partially explained by "residual prune-farming that B caps and A does not." Report per-arm post-flip PRUNE
reward (`rewards` view, `action_name='PRUNE'`, positive `bounded_attribution`) so any Arm-A churn-drop is
attributed to genuine commitment, not an uncapped prune channel.

### 5.4 Statistics — the SOLE GATE −1 invalidator (blocking finding, now pinned)
- **Unit of analysis = per-seed paired delta** (arm − control on the *same* seed).
- **The block bootstrap MUST resample the ≥5 (target 10) independent RNG SEEDS as the resampling unit.**
  It must **NOT** resample the 12 vec-envs as independent draws: they share one drifting policy
  (non-i.i.d.), so env-level resampling narrows the CI artificially and can **manufacture a FALSE STOP**
  that wrongly kills the redesign. Envs are aggregated *within* a seed (cluster), then seeds are the
  bootstrap unit. **This is the only identified path to an invalidated GATE −1.**
- Report median paired delta + **90% seed-level block-bootstrap CI** per metric per arm.

### 5.5 Pre-registered STOP rule (fix BEFORE looking at results)
> **STOP (lever resolved the pathology)** iff, on paired seeds with seed-level bootstrap CIs, an arm shows
> **BOTH**: (a) churn DOWN vs control (germinate+prune counts; 90% CI excludes 0 in the down direction),
> **AND** (b) committed final accuracy UP vs control (90% CI excludes 0 in the up direction).
> **SURVIVE** = the conjunction fails on all of {clip 2.0, clip 5.0, unit-norm, ESCROW}.

Pre-registering the **direction + effect-size** matters because n=5–10 is a falsifier floor, not a powered
estimate: an underpowered null must not be re-read post-hoc as STOP. (Low power is conservative here — it
favours SURVIVE, which cannot kill the redesign.)

### 5.6 Coupling caveat (#5) — Stage-2 covariance share
`bounded_attribution` is simultaneously Stage-2's `R_cf` variance source, so any scale change re-opens the
Stage-2 `Cov(R_cf,R)/Var(R) > 0.40` gate. **Cheaply available** from the per-arm `rewards` telemetry:
report `Cov(bounded_attribution, total_reward)/Var(total_reward)` per arm as a proxy and flag if a lever
materially shrinks it (Stage 2 may be partly mooted). Full per-stream `ev_main/ev_cf` instrumentation is
the separate Stage-0 task (`esper-lite-3d67b09687`) — **not built here**.

---

## 6. Exact seed-matched launch commands (replay)

Same seed set across **every** arm (paired). Floor 5, target 10. Seeds: `41 42 43 44 45` (extend to
`46 47 48 49 50` for target 10). Run on the capable host; `--gpu-preload` removes the data-pipeline tax.

```bash
SEEDS="41 42 43 44 45"        # add 46..50 for target 10
DEV=cuda:0
declare -A ARMS=(
  [control]=configs/config-3slot-3seed-baseline-shaped.json
  [clip2]=configs/config-3slot-3seed-baseline-shaped-clip2.json
  [clip5]=configs/config-3slot-3seed-baseline-shaped-clip5.json
  [unitnorm]=configs/config-3slot-3seed-baseline-shaped-unitnorm.json
  [escrow]=configs/config-3slot-3seed-baseline-escrow.json
)
for arm in control clip2 clip5 unitnorm escrow; do
  for s in $SEEDS; do
    uv run python -m esper.scripts.train ppo \
      --config-json "${ARMS[$arm]}" --seed "$s" --max-seeds 3 \
      --device "$DEV" --gpu-preload
  done
done
```
- `--seed` overrides `config.seed` and drives `base_seed`; identical seeds across arms ⇒ paired.
- ESCROW automatically runs with `return_components=True` (forced by
  `_reward_components_required_for_state_transport`), so Arm C needs **no** code/flag change.
- **Do NOT use `--dual-ab`** (smoke-only, 2-GPU; not the ≥5-paired-seed evidence the gate requires).
- Total: 5 arms × 5 seeds = **25 runs** (×2 for target 10 = 50). Each ~200 episodes × 12 envs.

---

## 7. Verdict (PROVISIONAL — runs pending)

**Code + tests + design sign-off + wiring smoke are COMPLETE.** The paired A/B (≥25 GPU runs) has NOT been
executed in-session; the GATE −1 verdict is therefore **PENDING the runs**, to be scored per §5 and
filled in below.

| Arm | churn Δ (90% CI) | committed-acc Δ (90% CI) | STOP? | sanity (lever bit?) |
|---|---|---|---|---|
| clip 2.0 | _pending_ | _pending_ | _pending_ | _pending_ |
| clip 5.0 | _pending_ | _pending_ | _pending_ | _pending_ |
| unit-norm | _pending_ | _pending_ | _pending_ | _pending_ |
| ESCROW (directional) | _pending_ | _pending_ | _pending_ | _pending_ |

**Decision rule (pre-registered, §5.5):** if ANY non-ESCROW lever shows churn↓ AND committed-acc↑ over
control (seed-level CIs) → **STOP_THEORY / REVISE_ALGORITHM** — the pathology was the per-step scale bug;
fix at config level; the L-complexity redesign is **mooted/down-scoped**. Else → **"scale leg of GATE −1
satisfied; admission PENDING the sibling contribution-gate leg"** — do not declare GATE −1 passed; do not
build the reward.

---

## 8. Replay metadata

- Branch `feat/phase-minus1-scale-falsifier` off `0.3.0`; code commit `f1f2551f`.
- Arm config sha256: see §2.
- Cited diagnostic run: `telemetry_2026-06-24_033049`.
- Design review: workflow `wf_8fd217c5-c07` (drl-expert + reward-function-reviewer = GO_WITH_CHANGES) +
  closing review `wf_7c34f4d3-f4a` (morphogenesis + packet audit = SIGN_OFF_WITH_CHANGES, blocking none).
- Tracker: filigree `esper-lite-a221da47ea`.
- Seeds: `41–45` (target `41–50`), `max_seeds=3`, 12 vec-envs, 200 episodes, capable host (cifar_baseline).
