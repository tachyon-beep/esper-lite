# Independent Go/No-Go Review — Reward-Redesign Methodology

```yaml
id: reward-methodology-gonogo-review
created: 2026-06-24
reviews: docs/plans/concepts/2026-06-24-reward-redesign-methodology.md
method: 4th independent multi-agent workflow (wf_90a109d8-63b) — 6 adversarial lenses,
        per-finding skeptic refutation, synthesis; against git HEAD 2cb58ada
verdict: GO_WITH_CONDITIONS
```

## Verdict

**GO_WITH_CONDITIONS** — promote the methodology from `concept` to `ready` and adopt it
as the governing process for the reward-alignment question, subject to the ordered
conditions below. This is a 4th independent review (the doc had 3 prior workflow passes);
it ran 6 adversarial lenses — citation/reality, telemetry re-measurement, DRL substance,
reward-hacking, morphogenetic-RL discipline, production-executability — each blind to the
others, with every "blocking" finding put past an independent skeptic.

**Result:** all 6 lenses returned GO_WITH_CONDITIONS; **all 11 blocking findings raised
were refuted** by the independent skeptic (`survived_skeptic=false` on every one). The
recurring refutation logic: each alleged blocker is either a code-hygiene defect the
methodology already captures inside a fail-closed gate (Phase-0 "enumerate every LIVE
term"; GATE-2 algebra normalization; GATE-3 "provably a shaping of J"), or forward
empirical/execution work the methodology correctly sequences behind its own gates.
Neither class invalidates the methodology's soundness.

### Why GO (confirmed strengths, all code-grounded against HEAD)

- **The load-bearing B1 crux is genuinely resolved and independently reproduced twice.**
  `reward↔fossilize = +0.214` (POSITIVE) on the cited 180-ep window — once via Karn SQL
  (`+0.2139`), once by parsing the raw 252 MB `events.jsonl` directly (`+0.214`). The
  alarming `−0.43` is verifiably the `reward↔prune` value mislabeled (`−0.432`). The old
  "flip −0.43→≥0" commitment target was therefore vacuous; the doc correctly redefines it
  as a committed-J-per-fossilize **rate/hazard**.
- **J = committed counterfactual gain per param is the correct anti-Goodhart anchor and is
  code-grounded.** `contribution.py:440-448` keys anti-gaming gates on the clean
  counterfactual (`val_acc − all_disabled_acc`) and *raises* if it is absent; host-drift is
  excluded (`types.py:33-37`). A zero-counterfactual parked seed yields `attributed≈0`, so
  the drift-farmer / park-the-freeloader hacks cannot land.
- **The composite fail-closed GATE-4 is a real ALIGNMENT addition** over the EV-only
  EV-stabilization epic — `reward↔J` can fire independently of the epic's
  `Cov(R_cf,R)/Var(R)>0.40` variance gate, and the attribution-share ceiling (variance
  *removed*, not diluted) + an absolute inter-decile-range clause genuinely close the
  terminal-loading/variance-dilution hack.
- **Sequencing is RL-correct: cheapest-falsifier-first.** Phase −1 (contribution gate that
  can `STOP_THEORY`; clip/ESCROW/scale A/B; `max_seeds≥2` re-measure) runs *before* the
  L-complexity redesign and can MOOT it entirely.
- **~20 code citations verified exact**, governor independence confirmed
  (`preflight_lifecycle_mutation` reads only structural/health facts; one-directional
  governor→reward), lifecycle-circularity grounding is code-true (FOSSILIZE never frees the
  slot; only PRUNE→DORMANT nulls state; fresh model per episode).

## The integrity of this GO

Adopting this methodology **commits us to run cheap falsifiers that may conclude no new
reward is warranted.** GATE −1 can return `STOP_THEORY` (signal/observability problem, not
a reward problem → halt) or route to `REVISE_ALGORITHM` (already-owned in-PPO reforms — no
new reward). This is the methodology working as designed, not a defect. "GO" means *adopt
the discipline*, not *build the reward*.

## Conditions to clear before promotion (`concept` → `ready`)

1. **Specialist sign-off (cleanest hard blocker).** `§10` + `reviewed_by` list all PENDING.
   This 4th review's lenses substantively cover **drl-expert**,
   **yzmir-deep-rl:reward-function-reviewer**, **yzmir-morphogenetic-rl:morphogenesis-reviewer**,
   and **axiom-planning:plan-review-reality** (all returned GO_WITH_CONDITIONS) — record them.
   **pytorch-expert** (only if the redesign touches the HRA/normalizer path) and
   **axiom-python-engineering** remain pending for implementation time — do **not** claim
   "all specialists cleared."
2. **Owner ratify-or-revert the two self-flagged closure edits** (the doc itself marks these
   as owner design decisions, not verified facts):
   - (a) commitment **unit** = alpha-weighted on-output-path **residency** (the review's
     finding: residency must be *counterfactual-weighted*, i.e. `Σ counterfactual × alpha-
     residency / params`; raw residency alone is hackable; fossilize-only is partly circular).
   - (b) the **§9 normative stance** — whether an irreversible committed graft is valued in
     its own right — which decides whether **GATE-4.3 is a real gate or a demoted diagnostic**.
   Make `§9` ratification an explicit hard checkpoint resolved **before GATE-4 pre-registration**.
3. **Free correctness edits during revise-first** (zero methodology impact):
   - `§5` figure `526/278` → `278/526` on the first-180-ep window (as written it reads
     "526 of 278", numerically impossible; Karn confirms 278 of 526 positive steps exceed 2.0).
   - stale `shaping.py` comments at lines **31, 47, and 60** (all three call BLENDING the
     largest increment; the largest delta is BLENDING→HOLDING `+2.0`).
4. **Attach the correctness-fix work as a formal `depends_on`.** The ESCROW
   `sqrt` vs SHAPED `harmonic` divergence (`contribution.py:523` vs `:548`) and the
   `pbrs_weight` telescoping test gap (`test_pbrs_properties` computes expected telescoping on
   raw potentials, no `pbrs_weight` scaling) are live bugs; filigree
   **esper-lite-3defe42928** ("Fix escrow telescoping break") already exists to bind.

## New findings to fold into the doc (raised this review, not in prior passes)

- **Window-sensitivity of +0.21 (disclose).** Reproduces `+0.214` on the 180-ep window but
  collapses to `+0.052` on the full 300-ep run (Spearman steadier: `+0.328` / `+0.214`). The
  narrative survives (≥0 on every cut; the mislabel finding reproduces exactly), and this
  *vindicates* the doc's pivot to a rate/hazard metric re-measured on ≥5 paired seeds — but
  the doc should state the +0.21 is window-/tail-specific, not a stable run-wide figure.
- **`pbrs_weight` is NOT a PBRS-invariance breaker (precision fix).** A *uniform* scalar `w`
  applied to `γΦ(s′)−Φ(s)` just rescales the potential to `w·Φ` and still telescopes
  (`Σ = w(γ^TΦ_T − Φ_0)`); the optimum is preserved. The real breakers are the **asymmetric**
  `germination_discount` (`contribution.py:789`), the PRUNE-forfeiture redefinition (`:1022`),
  and the escrow clip. Drop `pbrs_weight` from the breaker list — it makes Phase 2 do
  unnecessary work.
- **Two un-enumerated live dense terms** the §8/Phase-0 inventory must cover: the
  **proxy-path host-relative credit** (`contribution.py:606-612`, pays
  `0.3 × improvement_since_stage_start` — a host-wide, non-counterfactual delta — when the
  clean counterfactual is absent at BLENDING entry), `compute_scaffold_hindsight_credit`
  (LIVE at `training/handlers/fossilize.py:99`), and `_compute_synergy_bonus`
  (`contribution.py:699-703`). Add the proxy path to the §8 hack list (it is churn fuel).
- **Phase-2 enumeration is PBRS-centric** — extend it to name live non-PBRS dense terms
  (`alpha_shock`, `blending_warning`, `holding_warning`) so de-shape doesn't silently leave a
  non-telescoping term in the main stream.
- **Governor-independence CI is a cheap win that passes TODAY** — an import-graph test +
  signature-introspection on `preflight_lifecycle_mutation` would pass now; name it concretely
  rather than the generic "build-failing CI: governor-independence."
- **Phase-5 determinism is under-scoped** (per-subsystem RNG migration off global
  `torch.manual_seed` is real work) **but NOT a gate on lockstep-A/B validity** — `grep` over
  `src/esper/simic/rewards/` finds zero RNG usage; reward computation is pure deterministic
  arithmetic, so same-seed already yields "only the reward differs."
- **Path note:** live handlers are at `src/esper/simic/training/handlers/`, not
  `rewards/handlers/` — the doc's `fossilize.py:99` citation is correct relative to that dir.

## Path to a SHIPPED production reward (gated, may end early)

1. Promote `concept`→`ready` after conditions 1–4.
2. **Phase −1 cheap falsifiers FIRST** — these are config/flag tweaks to **existing** modes
   (per-step attribution clip / unit-normalization on SHAPED; ESCROW pay-at-commitment +
   clawback), **not** new reward classes. Infra gap: the preferred paired-gate driver
   (runsheet §7B) is **unimplemented** — Path A (`--proof-baseline`, zero new code) is
   runnable now as a lower-power provisional. Re-measure on `max_seeds≥2`.
   → **GATE −1 can STOP_THEORY or route to REVISE_ALGORITHM (no new reward).**
3. **Phase 0** lands Stage-0 instrumentation (filigree **esper-lite-3d67b09687**, open, 0%) —
   `ev_main/ev_cf/return_variance_shares` are absent from all of `src/` today — **plus**
   residency/on-output-path telemetry (the alignment-J unit; `grep` finds zero hits), built
   next to `RewardComponentsTelemetry`. Until both land, GATE-1's variance branch and
   GATE-4.2 are uncomputable. Pin J's λ + gain-baseline with a worked numeric example.
4. **GATE 1** decides scope; only its alignment-redesign branch (b/c) authorizes a new reward.
5. **Phase 2** normalizes ESCROW `sqrt` vs SHAPED `harmonic`; extends `test_pbrs_properties`
   to assert telescoping incl. `pbrs_weight` against *realized* shaping.
6. **Phases 3–6**: candidate lattice → pre-registered composite GATE-4 → both-substrate paired
   A/B (per-seed delta + env-clustered block bootstrap) → determinism/load-bearing/DPBA →
   promotion packet. Compute ≈480–800 GPU-h; pre-commit a candidate-count cap (≤5) once the
   §9 power calc runs.

## On "a new reward class" (the original ask)

The architecture supports it cleanly — a `RewardMode` enum member (`contribution.py:105-111`)
+ a dispatch branch in `compute_reward` (`rewards.py:66+`) + a `compute_<mode>_reward` handler
+ config fields + `RewardComponentsTelemetry` wiring — the exact pattern used **7 times**
already (SHAPED/ESCROW/BASIC/BASIC_PLUS/SPARSE/MINIMAL/SIMPLIFIED). **But the earliest
*sanctioned* new RewardMode is a Phase-3 candidate, strictly downstream of GATE 1** — and
GATE −1 may prevent one from *ever* being sanctioned. The Phase −1 cheap levers are
config/flag changes to existing modes, **not** new enum members. Caveat: Phase 3 may re-enter
the counterfactual as a Harutyunyan-2015 DPBA potential / HRA head / baseline INPUT rather
than a raw dense mode, so a new `RewardMode` may not be the integration surface at all — the
methodology explicitly prefers shaping-of-J / DPBA re-entry over a raw dense reward.
