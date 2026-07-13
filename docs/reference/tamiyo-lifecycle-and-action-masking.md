# Tamiyo — Seed Lifecycle & Action Masking (two-pager)

Source of truth: `leyline/stages.py` (states + transitions), `leyline/factored_actions.py`
(ops + heads), `leyline/causal_masks.py` (head relevance), `tamiyo/policy/action_masks.py`
(op availability per state). This doc summarizes them; the code is authoritative.

## 1. Seed lifecycle states (`SeedStage`)

A seed occupies one slot and moves through a botanical lifecycle. Stages (enum value):

| Stage | Val | Alpha / role | Contributes to forward pass? |
|-------|-----|--------------|------------------------------|
| `DORMANT` | 1 | empty slot, waiting | no |
| `GERMINATED` | 2 | created, α=0 | no (learning only) |
| `TRAINING` | 3 | α=0, learning from host error | no |
| `BLENDING` | 4 | α ramps 0→target over `tempo` epochs | **yes** (partial) |
| `HOLDING` | 6 | ramp complete **at the chosen `alpha_target`** (α = 0.5 / 0.7 / 1.0 — NOT necessarily 1.0; see §5 correction) — the **decision point** | **yes** |
| `FOSSILIZED` | 7 | permanently fused — **terminal success** | **yes** |
| `PRUNED` | 8 | removed — failure | no |
| `EMBARGOED` | 9 | post-removal cooldown | no |
| `RESETTING` | 10 | cleanup before slot reuse | no |

(Value 5 = old `SHADOWING`, removed. `UNKNOWN`=0 is a sentinel.)

## 2. Lifecycle transitions (`VALID_TRANSITIONS`)

```
DORMANT → GERMINATED → TRAINING → BLENDING → HOLDING ─┬─→ FOSSILIZED (terminal)
                                                       ├─→ BLENDING  (scheduled re-blend)
   (any of GERMINATED/TRAINING/BLENDING/HOLDING) ──────┴─→ PRUNED → EMBARGOED → RESETTING → DORMANT
```

- Every developing stage (GERMINATED, TRAINING, BLENDING, HOLDING) can go to **PRUNED**.
- **FOSSILIZE is reachable only from HOLDING.** Commitment requires reaching full amplitude
  first.
- After PRUNED, the slot cools down (EMBARGOED → RESETTING) and returns to DORMANT for reuse.

## 3. Tamiyo's factored action space

Each decision is a tuple across heads. The **op** head chooses the lifecycle operation; the
other heads parameterize it. `LifecycleOp`: `WAIT`(0) `GERMINATE`(1) `SET_ALPHA_TARGET`(2)
`PRUNE`(3) `FOSSILIZE`(4) `ADVANCE`(5).

| Head | Choices | Used by op(s) |
|------|---------|---------------|
| `op` | the 6 LifecycleOps | always relevant |
| `slot` | which slot/seed | all non-WAIT ops |
| `blueprint` | NOOP, CONV_LIGHT/SMALL/HEAVY, ATTENTION, FLEX_ATTENTION, NORM, DEPTHWISE, BOTTLENECK, LORA/LORA_LARGE, MLP_SMALL/MLP | GERMINATE |
| `style` | LINEAR_ADD, LINEAR_MULTIPLY, SIGMOID_ADD, GATED_GATE | GERMINATE, SET_ALPHA_TARGET |
| `tempo` | FAST(3 ep), STANDARD(5 ep), SLOW(8 ep) | GERMINATE (sets blend ramp length) |
| `alpha_target` | HALF, SEVENTY, FULL | GERMINATE, SET_ALPHA_TARGET |
| `alpha_speed` | INSTANT, FAST, MEDIUM | SET_ALPHA_TARGET, PRUNE |
| `alpha_curve` | (ramp shape) | SET_ALPHA_TARGET, PRUNE |

Two mask families exist (`leyline/causal_masks.py`): **causal masks** (a head mattered for the
op actually taken → used for policy-gradient credit) and **availability masks** (a head could
have mattered → used for entropy regularization so exploration doesn't collapse). The table
above is the head-relevance map both derive from.

## 4. WHEN each op is available (op masking — `action_masks.py`)

The op mask is computed over ALL enabled slots; an op is offered if **any** seed/slot
qualifies, and the `slot` head then picks which. Conditions:

| Op | Available when… |
|----|-----------------|
| **WAIT** | **always valid.** |
| **GERMINATE** | an enabled slot is DORMANT (empty) **AND** `total_seeds < max_seeds` **AND** no seed is in `{GERMINATED, TRAINING}` (early dev) **or** `{PRUNED, EMBARGOED, RESETTING}` (cleanup). |
| **ADVANCE** | a seed is in `{GERMINATED, TRAINING, BLENDING}` (and auto-advance not disabling it). |
| **FOSSILIZE** | a seed is in **HOLDING** only. |
| **PRUNE** | a seed is in `{GERMINATED, TRAINING, BLENDING, HOLDING}` **AND** `seed_age ≥ MIN_PRUNE_AGE` **AND** (alpha is in HOLD mode **or** governor override). |
| **SET_ALPHA_TARGET** | a seed is in `{BLENDING, HOLDING}` **AND** alpha is in HOLD mode. |

### Same rules, viewed per stage (what Tamiyo can do to a seed here)

| Seed stage | Ops available on it (WAIT always) | Blocks new GERMINATE? |
|------------|-----------------------------------|------------------------|
| DORMANT (empty) | GERMINATE (subject to the global gates above) | — |
| GERMINATED | ADVANCE→TRAINING; PRUNE (if age ok) | **yes** |
| TRAINING | ADVANCE→BLENDING; PRUNE (if age ok) | **yes** |
| BLENDING | ADVANCE→HOLDING (once α ramp done); PRUNE (HOLD-mode, age ok); SET_ALPHA_TARGET | **no** (scaffolding) |
| HOLDING | **FOSSILIZE**; PRUNE (HOLD-mode, age ok); SET_ALPHA_TARGET | **no** |
| FOSSILIZED | — (terminal, committed) | — |
| PRUNED / EMBARGOED / RESETTING | — (cleanup) | **yes** (until DORMANT) |

## 5. Why the masks are shaped this way (anti-farming + scaffolding)

- **Sequential development:** GERMINATED/TRAINING seeds **block** new GERMINATE (`D3`), so the
  policy can't spam germinations to farm the germination/attribution bonus.
- **Cleanup blocks germination:** PRUNED/EMBARGOED/RESETTING also block GERMINATE, closing the
  `GERMINATE → WAIT → PRUNE → repeat` farm; a slot must fully reset to DORMANT first.
- **`MIN_PRUNE_AGE`** stops instant germinate-then-prune churn.
- **BLENDING and HOLDING deliberately DO NOT block germination** — this is the intended
  **scaffolding** pattern: an older seed can be blending/holding (teaching the host) while a new
  seed germinates and trains. Culling a scaffold once its knowledge is locked in is normal and
  expected, not a defect.
- **FOSSILIZE only from HOLDING** requires a seed to complete its blend ramp — but **to its
  chosen `alpha_target`, NOT necessarily α=1.0** (CORRECTION, verified in `slot.py`: BLENDING→
  HOLDING fires on `reached_target`; the only special case is `alpha_target ≤ 0` → prune). So a
  seed germinated at `alpha_target=HALF` reaches HOLDING at α=0.5 and **can be committed at half
  amplitude.** `alpha_target` ∈ {HALF, SEVENTY, FULL} is therefore a commitment-shaping choice
  made at GERMINATE: it sets *where on its own contribution curve* a seed locks in, not whether
  it becomes eligible.

> **Commitment economics (verified in `contribution.py`, SHAPED mode):** while a seed is
> pre-fossilize it earns `bounded_attribution` (∝ contribution, per step) with no penalty unless
> it's *hurting*. On **FOSSILIZE**, attribution stops (`not seed_is_fossilized` guard → 0) AND a
> permanent `fossilized_maintenance_cost` rent begins. The compensating fossilize bonus is a
> one-shot proportional to the *instantaneous* contribution rate. So committing converts a paying
> asset into a rent-only liability, and the dominant alternative — **cull + re-germinate** —
> renews the income stream at the top of a fresh contribution hump. This is action-trace-identical
> to healthy scaffolding; only a **host-retention** measurement separates them.

> Diagnostic note: because BLENDING/HOLDING don't block germination and PRUNE is always
> available on developing seeds, high germinate/prune churn with a stable fossilize rate is
> consistent with *either* healthy scaffolding *or* attribution-farming. The masks prevent the
> crudest farms; whether a given cull is scaffolding (retained host benefit) or farming
> (contribution claimed then reversed) is a reward/telemetry question, not a masking one.

---

# 3. The reward functions

Source: `simic/rewards/contribution.py` (terms), `partition.py` (the R_main/R_cf split),
`loss_primary.py` (PBRS). Reward is computed **per-step, per-seed** and summed.

## Reward modes (`RewardMode`) and family

- **`SHAPED`** — the current mode (dense, all terms below). **`reward_family = contribution`.**
- `ESCROW` — attribution held in escrow, paid on outcome. `BASIC` / `BASIC_PLUS` — minimal +
  post-fossilization drip accountability. `SPARSE` — param-penalty + sparse terminal only.
  `MINIMAL` — sparse + early-prune penalty. `SIMPLIFIED` — PBRS + intervention cost + terminal
  (diagnostic; omits structural rent).

## SHAPED reward terms (what sums into `reward_raw`)

| Term | Sign | What it rewards / penalizes |
|------|------|------------------------------|
| **`base_acc_delta`** | ± | host validation-accuracy change this step |
| **`bounded_attribution`** (= **R_cf**) | ± | **the dense counterfactual credit** — `contribution_weight × seed_contribution`, with escrow / early-stage proxy variants, a D3 **timing_discount** (anti early-germination gaming), a **ratio_penalty** (anti-gaming, folded in), and sign-flip for negative contribution. **This is the variance-dominant "cf stream"** (~85% of return variance) and the term flagged deviant in the Dec-2025 diagnosis. |
| **`pbrs_bonus`** (stage_bonus) | + | potential-based shaping toward lifecycle progress (policy-invariant by PBRS construction) |
| **`blending_warning`** | − | seed hurting during BLENDING → nudge toward CULL (`−0.1 − escalation`) |
| **`holding_warning`** (indecision) | − | WAITing in HOLDING when counterfactual data exists → forces the fossilize/prune decision |
| **`synergy_bonus`** (interaction_bonus) | + | k≥2 co-fossilization synergy (gated: `attribution_discount ≥ 0.5 ∧ bounded_attribution > 0`) |
| **`compute_rent`** | − | parameter/compute cost of added structure (efficiency pressure) |
| **`alpha_shock`** | − | large alpha changes (blend-stability) |
| **capacity economics (D2)** | − | slot-saturation pressure |
| **`action_shaping`** incl. **fossilize shaping** | + | rewards **quality** fossilization (high `seed_contribution`), not quantity |
| **`terminal_bonus`** | + | `val_acc × terminal_acc_weight` at episode end |
| **fossilize bonus** | + | paid **immediately on FOSSILIZE** (P0 fix 2026-01-11; the `fossilize_terminal_bonus` field is now always 0, kept for telemetry compat) |
| **auto-prune penalty** | − | degenerate-policy prevention |

## Anti-gaming guards (why the reward is this baroque)

`ratio_penalty`, `attribution_discount` (D3 timing), **ransomware detection** (prune seeds with
high counterfactual but negative total improvement), `MIN_PRUNE_AGE`, and the germination-blocking
masks (page 2) all exist to stop specific farms discovered in review (WAIT-farming attribution,
germinate-prune churn, dependency-creating "ransomware" seeds).

## The R_main / R_cf partition (EV-stab / HRA)

`partition.py` splits every step exhaustively: **`R_cf = bounded_attribution`** (the sole
counterfactual addend) and **`R_main = reward_raw − R_cf`** (everything else). `r_main + r_cf ==
reward_raw` by construction. This is the split the EV-stabilization epic's value decomposition
(V_main + V_cf) targets.

---

# 4. Tamiyo's policy structure (right now)

`FactoredRecurrentActorCritic` — `tamiyo/networks/factored_lstm.py`, built via
`policy/factory.py` (`policy_type="lstm"`). One shared recurrent trunk feeding factored
actor heads + value heads.

```
obs features (Obs V3, symlog-compressed) + blueprint embedding
        │
   feature_net:  Linear → LayerNorm → ReLU            (→ feature_dim)
        │
   ResidualLSTM  (hidden=512, per-layer LayerNorm + residual + depth-scaled forget bias;
        │         replaced stacked LSTM to fix vanishing gradients; TBPTT over chunk_length)
        │
   shared_repr (lstm_out)  ──┬── ACTOR heads (each a small MLP, head_hidden=256):
                             │      op_head        → 6 LifecycleOps
                             │      slot_head       → num_slots
                             │      blueprint_head   → 13 blueprints (512→512→256→13)
                             │      style/tempo/alpha_target/alpha_speed/alpha_curve heads
                             │
                             └── CRITIC / value heads:
                                    state_value_head → V(s)      op-INDEPENDENT PPO baseline
                                                                 (512→256→128→64→1, LayerNorm)
                                    q_head           → Q(s, op)  op-conditioned; TELEMETRY-ONLY,
                                                                 trained detached toward the
                                                                 state-return target (NOT the baseline)
                                    cf_value_head    → V_cf(s)   built ONLY under
                                                                 hra_value_decomposition (rejected
                                                                 HRA); None on the default leg
```

Key points:
- **Recurrent by design:** decisions depend on lifecycle *history/trends* (is this seed improving?),
  so the LSTM carries hidden state across steps within an episode; features are **symlog-compressed**
  to stop LSTM saturation (h/c were blowing up on high-magnitude telemetry).
- **Factored action:** each head samples from a `MaskedCategorical` under the availability masks of
  page 2 (op mask gates valid ops; per-head masks gate the parameters). Forced/masked steps are
  excluded from the actor loss.
- **The PPO baseline is `V(s)` only** (P0-1): op-independent, or advantages are biased. `Q(s,op)` is
  a detached auxiliary kept alive purely so the op-value telemetry isn't init noise.
- **`V_cf` is the EV-stab addition** and is OFF by default (the HRA it belongs to was rejected,
  PDR-0059); on the default leg the topology is the two heads `V(s)` + `q_head`.
- Default dims: `lstm_hidden_dim = 512`, `head_hidden = 256`, `blueprint = 13`, `op = 6`.

