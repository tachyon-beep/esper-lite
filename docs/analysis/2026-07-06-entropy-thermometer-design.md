# Entropy Thermometer Fix — Design (Filigree esper-lite-425dcc4ca2)

Branch: `feat/entropy-thermometer-fix` (off `feat/ev-stab-stage2-hra`). drl-expert
design (2026-07-06), code-grounded. Design only; every claim is file:line.

## Verdict — MEASUREMENT ARTIFACT (with a separately-dead aggregate alarm)

The tracker's "entropy<0.1 / 594 anomalies from update 3" is an **artifact** of a
**per-head** detector reading raw per-head means over inactive steps:

- `ppo_coordinator.py:619-628` feeds `check_per_head_entropy_collapse` from
  `metrics["head_entropies"]`.
- `metrics["head_entropies"]` = raw `entropy[key].mean()` over ALL `valid_mask`
  steps (`ppo_agent.py:1322`) — no conditioning on causal relevance / >1 valid
  action / forced steps.
- For sparse heads (~18–20% active), ~80% of steps are a single placeholder
  action → normalized entropy exactly 0 (`action_masks.py:611`), so the raw mean
  ≈ 0 even when the head explores fully whenever it has a choice. That is the
  false-fire, per sparse head per update.

The single AGGREGATE alarm is separately miscalibrated into a **dead
false-negative**: emitted `entropy` = `mean_epochs(Σ_k coef_k · H̄_k)` where coefs
are 1.0–1.3 (`ppo_agent.py:75-84`, applied `:263-265`; no global `entropy_coef`
at `:1688`), H̄_k normalized [0,1] — structural range [0, 9.4]. Compared against
`DEFAULT_ENTROPY_COLLAPSE_THRESHOLD=0.1` (a normalized-[0,1] value,
`leyline/__init__.py:217-221`) at `emitters.py:1082` and Karn `collector.py:572`
→ `triggers.py:261-270`. With the probability floor active (op 0.15,
`leyline:275-284`, applied `factored_lstm.py:1531-1533`), op post-floor entropy
≈0.99 → the sum is structurally ≥~1 → these two alarms essentially never fire.

**Whether op has genuinely collapsed is currently unobservable** — the honest
signal (`conditional_head_entropies`, `ppo_agent.py:1361-1366`, aggregated
`ppo_metrics.py:148`) is computed but never emitted or alarmed.

## Fix — phased

### Phase 1 (instrument-honest, NO behavior change; also DIAGNOSTIC)
Choice-conditioned normalized entropy per head (mean of `entropy[key]` over steps
with ≥2 valid actions) + choice fraction. The `>1 valid action` mask already
exists (`action_choice_mask`, `ppo_agent.py:1367`).
- `ppo_agent.py:1361-1367`: condition `conditional_head_entropies` on
  `action_choice_mask & unforced`; accumulate `choice_fraction`.
- `ppo_metrics.py:143-151`: aggregate choice_fraction alongside conditional
  entropies.
- `leyline/telemetry.py` (~800): add `head_{name}_conditional_entropy` and
  `head_{name}_choice_fraction` (`float | None`), 8 heads.
- `emitters.py:1085-1102`: emit them; keep raw `head_{name}_entropy` as the
  artifact witness.
- `ppo_coordinator.py:619-628`: feed the per-head detector the CHOICE-CONDITIONED
  entropies, gated by `choice_fraction>0` (no choice steps → None → skip, never a
  0-fire). Thresholds `ENTROPY_COLLAPSE_PER_HEAD` (`leyline:241-250`) now apply to
  a genuine [0,1] quantity.
- Aggregate headline: emit **op choice-conditioned entropy** as `entropy`;
  re-point `entropy_collapsed` (`emitters.py:1082`) + Karn feed (`collector.py:572`)
  to it (0.1 collapse / 0.3 warn). The old weighted sum, if wanted, emits under a
  distinct name (`entropy_bonus_magnitude`) — never `entropy`. **CONTRACT CHANGE**:
  redefining `entropy` breaks Karn + the Sanctum TUI consumer; update all consumers
  in one commit (No-Legacy). Cross-branch: the Sanctum consumer lives on
  feat/sanctum-layout — coordinate.

### Phase 2 (behavioral — only if Phase 1 shows real collapse)
Entropy-floor penalty is ACTIVE but a structural no-op: it reads post-floor
entropy (≈0.99) so `max(0, 0.30−0.99)=0` never fires (`ppo_update.py:420-450`
reading post-floor `entropy` from `factored_lstm.py:1531-1535`). Fix: compute a
second PRE-floor normalized entropy at `factored_lstm.py:1531` (before
`_apply_floor_to_logits` at `:1533`), thread `raw_entropy` through
`evaluate_actions` → `ppo_agent.py:1440-1466` → `compute_losses`
(`ppo_update.py:264-276`) and feed it to `compute_entropy_floor_penalty`
(`:429-450`); bonus + telemetry keep post-floor entropy.

### Phase 3 (hardening — independent, closes R1 NaN)
Entropy/log-prob math is already fp32 (`.float()` at `factored_lstm.py:1531`;
`action_masks.py:591,603,545`) and amp defaults off (`config.py:132`). Residual:
under amp fp16, head Linears (`factored_lstm.py:1381-1388`) emit fp16 logits that
can overflow→inf→NaN softmax before the `:1531` upcast. Force policy logit heads
to fp32 output.

### NOT a lever
`entropy_anneal_episodes=3000` vs 200-ep run is real but NOT the collapse cause
(at update ~3, progress≈0 → coef≈start regardless; fast preset anneals DOWN
0.06→0.03, so the long schedule pins coef HIGH = protective). Do not "fix" it;
config-hygiene note only. No distinct "dormant std-floor" found in the entropy
path (the only std knob is `advantage_std_floored`, unrelated).

## Sparse-head trap
Denominator = the choice set (`masks[key].sum(-1)>1` ∩ unforced), never full step
count; gate alarms by choice_fraction>0 (None ≠ collapse); anchor on
availability/causal masks (as the floor penalty already does). Keep raw
`head_{name}_entropy` so `conditional > raw` for sparse heads is on-dashboard proof
the old reads were the artifact.

## Owner smoke (validates Phase 1; determines if Phase 2 is needed)
Short recurrent PPO, amp off, fast preset, e.g.
`uv run python -m esper.scripts.train ppo --episodes 60 --envs 4` (do NOT set
`--entropy-anneal-episodes`). Watch per-head `*_conditional_entropy` +
`*_choice_fraction`, esp. op; `entropy_floor_penalty` after Phase 2.
Pass: (1) op conditional entropy holds >~0.3, no crash to <0.1 by update ~3;
(2) sparse heads with choice_fraction>0 show conditional entropy ~op-level (not
~0) — direct proof the old reads were the artifact; (3) recalibrated per-head
alarm silent while healthy; (4) if op DOES crash, after Phase 2 the raw-entropy
`entropy_floor_penalty` becomes >0.

## Files
factored_lstm.py, ppo_agent.py, ppo_update.py, ppo_metrics.py, emitters.py,
ppo_coordinator.py, leyline/telemetry.py, karn/collector.py (+ tests). No change
to leyline/__init__.py thresholds (already correct [0,1] units once the feeding
metric is honest).

## Review gate
Per CLAUDE.md RL-domain rule: drl-expert / yzmir-deep-rl review the raw-entropy-
penalty vs probability-floor interaction (Phase 2) and recalibrated thresholds
before implementing Phase 2. Phase 1 is telemetry-honest and lower risk.

## Phase 1 validation (2026-07-06, cifar_baseline live smoke — PASS)
Landed as the ADDITIVE variant (owner call): new `head_{name}_choice_conditional_entropy`
alongside the raw witness; aggregate `entropy` headline redefinition deferred (it is the
cross-branch CONTRACT CHANGE above; the Sanctum consumer lives on feat/sanctum-layout).

Live smoke: `--task cifar_baseline --rounds 15 --envs 4 --episode-length 20 --gpu-preload
--no-tui` (amp off, seed 42). Ran all 15 PPO updates clean.
- **Crash bug found + fixed (commit 8a037e14):** the new metric key had no reducer in the
  cross-update whitelist `_aggregate_ppo_metrics` (`vectorized.py`), so every real run
  KeyError'd at the FIRST PPO update. The per-epoch builder/emitter unit tests never touch
  that seam — the suite was green while the code crashed on every live run. 2nd occurrence
  of this class after `deb6b11575`. Added the APPEND reducer + a targeted regression test.
- **Artifact eliminated:** sparse heads read 0.036–0.10 raw (at/below the ~0.08 per-head
  collapse threshold — the false-fire source) but 0.97–1.00 choice-conditional across all
  15 updates. `entropy_collapsed=False` on 15/15; **0 entropy-collapse anomalies (vs the
  historical ~594).** slot (fraction 0, single-slot host) correctly stays `None` — no
  fabricated 0. The 10 anomalies present are unrelated (Gradient/NaN/Value-Collapse).
- **Caveat confirmed:** op choice-conditional ≈0.85–1.00 (post-floor, structural) — silent
  to real op collapse by construction; that detection is Phase 2, not this change.

### Known fidelity limitation (deferred, out of Phase 1 scope)
The emitter means `choice_conditional` over the per-epoch series and gates per head only on
batch-level `mean_fraction<=0`. The upstream `clamp(min=1)` injects a 0.0 for any epoch a
head had no choice steps, so at `ppo_updates_per_batch>1` (or configs where a head's
per-epoch choice presence varies) those 0.0 entries dilute the mean toward the raw read we
are removing. Harmless at `updates_per_batch=1` (validated: kept heads have choices every
epoch). Honest fix = per-epoch fraction weighting / drop zero-choice epochs. TODO tagged at
the emitter mean site.
