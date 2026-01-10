# Counterfactual Oracle: Learned Contribution Inference for Scalable Morphogenesis

**Subtitle:** *From Expensive Truth to Cheap Belief — Telemetry-Only Counterfactual Estimation*
**Date:** 9 January 2026
**Status:** Draft / Investigation Proposal
**Phase Gate:** Blocked on Phase 2.5 Reward Efficiency Protocol completion

---

## 0. Thesis

Esper's counterfactual oracle (true ablations / Shapley-ish measurement) is *gold*, but it's also *expensive*. At 50–100+ seeds, always-on oracle truth becomes the bottleneck: either compute explodes or we stop measuring and fly blind.

The **Counterfactual Oracle** is a learned, telemetry-only estimator that predicts "what the oracle would say" well enough to guide decisions most of the time.

Key framing:

> **Oracle = expensive truth** (sparse, auditable, used as labels).
> **Probe = cheap belief** (dense, always available, used for control).

This preserves Esper's core conceit: **Tamiyo learns to behave like a counterfactual analyst from Kasmina telemetry**, not by being handed the exam paper.

**WEP:** *Highly likely* to be required if you want Emrakul-scale complexity without turning your training loop into an ablation benchmark harness.

---

## 1. Goals and Non-Goals

### Goals

1. **Scale**: Enable 50–100+ seeds with *sublinear* counterfactual compute growth
2. **Preserve the philosophy**: Policy input remains telemetry-derived; probe outputs are allowed because they're learned from telemetry
3. **Improve stability**: Reduce "superstition" and boundary-timed behaviour by giving Tamiyo a consistent belief state (with uncertainty)
4. **Remain debuggable**: Make probe-vs-oracle disagreements visible in Sanctum/Overwatch

### Non-Goals (Initial Investigation)

- Replacing the oracle entirely on day one
- Using probe outputs as the primary reward signal (high reward-hacking risk)
- Perfect Shapley fidelity; the goal is *decision-adequate* counterfactual judgement, not axiomatic attribution purity

---

## 2. Problem Statement: The Truth Bandwidth Wall

With many seeds, the oracle is expensive because it requires:

- Extra forward passes (full vs ablated)
- Sometimes multiple seeds per step
- Interaction effects make it tempting to measure everything (which is infeasible)

When you stop measuring, the agent either:

- Becomes overly conservative ("do nothing attractor"), or
- Starts gaming whatever dense proxies remain

The probe aims to restore **dense decision support** without dense expensive truth.

---

## 3. Hard Constraints (Non-Negotiable)

- **Policy inference must never consume true counterfactual values** (seed_contribution, escrow deltas, etc.)
- The probe must use **only Kasmina telemetry + existing policy-visible observations**
- True counterfactual values may be used only as:
  - Reward computation inputs (existing), and/or
  - **Supervised/auxiliary learning targets** for the probe

### Allowed

- Adding new model outputs and auxiliary losses
- Adding new telemetry to log predicted vs true targets
- Conditioning policy inputs on **predicted** probe outputs (student outputs only)

### Out of Scope

- Changing the fundamental reward mode design (keep current modes; just add auxiliary head)
- Introducing privileged critic that directly sees oracle counterfactuals (separate experiment)

---

## 4. What the Probe Predicts

### 4.1 Primary Outputs (Per Slot)

Start with a small, stable set and expand later.

| Output | Type | Description | Target |
|--------|------|-------------|--------|
| **ĉ_contrib** | Regression (μ, σ) | Predicted counterfactual contribution | `seed_contribution` or `Δloss` |
| **p_harm** | Classification | Probability seed will be net-negative | `1{future_improvement < 0}` |
| **confidence** | Scalar | How much to trust the estimate | Derived from σ or calibrated score |

### 4.2 Aleatoric Uncertainty (Critical)

The probe outputs `(μ, σ)` for contribution prediction, not just a point estimate.

**Why uncertainty is the "missing organ":**

Without uncertainty, the probe is forced into a dumb binary:
- Either it confidently guesses (even when it shouldn't), or
- You bolt on heuristics ("audit every N steps" / "audit before fossilise")

Neither is "active perception". Active perception needs a self-report:

> "I can predict this contribution reliably" vs "this region of state-space is noisy / unfamiliar."

Aleatoric uncertainty (data noise / irreducible variability) is especially relevant because the thing you're predicting is inherently noisy:

- Counterfactual deltas depend on batch composition
- Host drift changes the meaning of the same seed
- Alpha plans can make short-term deltas misleading
- RL stochasticity makes "same state" not actually the same

**WEP:** *Highly likely* that adding σ is necessary to make audit targeting efficient and non-calendar-like.

### 4.3 Optional Later Outputs

- **time_to_payoff**: Expected steps to "proven stable improvement"
- **Δcredit_expected** (escrow mode): Predicted escrow delta trend if we WAIT
- **α-sensitivity**: Expected marginal effect of alpha plan changes (ambitious; postpone)

---

## 5. Architecture Options

### Option A: Per-Slot Independent MLP (Start Here)

Each slot gets a small MLP head.

```python
class ProbeHead(nn.Module):
    def __init__(self, input_dim: int, num_slots: int, hidden_dim: int = 128):
        super().__init__()
        self.num_slots = num_slots
        # Output: [mu, log_var, harm_logit] per slot = 3 * num_slots
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_slots * 3),
        )

    def forward(self, lstm_out: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        # lstm_out: [batch, seq, input_dim]
        # CRITICAL: Detach to prevent gradient flow to LSTM hidden states
        out = self.mlp(lstm_out.detach())
        out = out.reshape(*lstm_out.shape[:-1], self.num_slots, 3)

        mu = out[..., 0]                           # [batch, seq, num_slots]
        log_var = out[..., 1].clamp(-5, 2)         # Tighter than [-10, 4]
        harm_logit = out[..., 2]                   # [batch, seq, num_slots]

        return mu, log_var, harm_logit
```

**Pros:** Simplest, compile-friendly, stable.
**Cons:** Misses cross-slot interaction effects.

### Option B: Set-Based Probe with Cross-Slot Attention (50+ Seeds)

Treat slots as a set; run a small attention block across slot embeddings.

```python
class AttentionProbeHead(nn.Module):
    def __init__(self, input_dim: int, num_slots: int, embed_dim: int = 64):
        super().__init__()
        self.slot_embed = nn.Linear(input_dim // num_slots, embed_dim)
        self.attention = nn.MultiheadAttention(embed_dim, num_heads=4, batch_first=True)
        self.output_proj = nn.Linear(embed_dim, 3)  # mu, log_var, harm_logit

    def forward(self, lstm_out: Tensor, slot_mask: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        # CRITICAL: Use fixed slot count with masking for torch.compile
        # slot_mask: [batch, num_slots] - True for active slots
        # ...
```

**Pros:** Models synergy/competition; still far cheaper than oracle.
**Cons:** More complexity; needs care with normalisation and stability; torch.compile requires fixed shapes with masking.

### Option C: Interaction-Aware (Explicit Pairwise Features)

Predict per-slot marginal plus a low-rank interaction term.

**Pros:** Interpretable; captures scaffold effects.
**Cons:** Engineering overhead.

**Recommendation:** Start with **Option A**, validate value, then move to **Option B** once you see clear interaction-driven probe errors. Per-slot MLP will likely work for 3–10 seeds but will fail at 20+.

---

## 6. Training Signal: Oracle Truth Becomes Labels

### 6.1 Teacher Labels (Oracle)

Oracle measurements can come from:

- Existing always-on counterfactual pipeline (in small-seed regimes), and/or
- Active Perception `AUDIT` events (sparse labels at scale), and/or
- Periodic scheduled audits (e.g., audit K random slots per episode as exploration)

These produce training pairs: `(telemetry_features_t, oracle_counterfactual_t)`

### 6.2 Target Definition (Critical Design Decision)

| Target | Properties | Recommendation |
|--------|------------|----------------|
| **Δloss (raw)** | Noisy, unbounded, sensitive to batch composition | Avoid |
| **Δloss (bounded)** | Clipped to [-1, 1] or z-scored, stable | **Use this** |
| **Escrow delta** | Temporal credit, but escrow mechanics add complexity | Later |
| **Binary harm** | Easy to predict, loses nuance | **Use for p_harm** |

**Start with:** Bounded Δloss for contribution regression, binary harm for classification.

### 6.3 Gaussian NLL Loss (Uncertainty-Aware)

Train contribution predictions using Gaussian negative log likelihood:

```python
def gaussian_nll_loss(mu: Tensor, log_var: Tensor, target: Tensor) -> Tensor:
    """
    L = 0.5 * ( (y - mu)^2 / var + log(var) )

    Properties:
    - High sigma reduces loss for large errors (probe admits ignorance)
    - The log(var) term penalises inflated sigma (prevents "always uncertain")
    """
    var = F.softplus(log_var).clamp(min=1e-4, max=50.0)  # Numerically stable
    return 0.5 * ((target - mu).pow(2) / var + var.log()).mean()
```

**Key stability trick:** Use `softplus(log_var)` instead of `exp(log_var)` for numerical safety.

### 6.4 Complete Loss Function

```python
def compute_probe_loss(
    mu: Tensor,           # [batch, seq, num_slots]
    log_var: Tensor,      # [batch, seq, num_slots]
    harm_logit: Tensor,   # [batch, seq, num_slots]
    contrib_true: Tensor, # [batch, seq, num_slots] - NaN where no label
    harm_true: Tensor,    # [batch, seq, num_slots] - NaN where no label
) -> Tensor:
    # Mask-based loss for sparse labels
    contrib_mask = ~torch.isnan(contrib_true)
    harm_mask = ~torch.isnan(harm_true)

    L_contrib = torch.tensor(0.0, device=mu.device)
    L_harm = torch.tensor(0.0, device=mu.device)

    if contrib_mask.any():
        L_contrib = gaussian_nll_loss(
            mu[contrib_mask],
            log_var[contrib_mask],
            contrib_true[contrib_mask]
        )

    if harm_mask.any():
        L_harm = F.binary_cross_entropy_with_logits(
            harm_logit[harm_mask],
            harm_true[harm_mask]
        )

    return L_contrib + L_harm
```

### 6.5 Integration with PPO

```python
# In PPO update loop
L_ppo = policy_loss + value_coef * value_loss - entropy_coef * entropy

# Add probe loss with separate weighting
# CRITICAL: Keep lambda_probe small (0.001-0.01) to avoid dominating PPO
lambda_probe = 0.01
L_total = L_ppo + lambda_probe * L_probe

# Log separately for debugging
metrics["probe_contrib_loss"] = L_contrib.item()
metrics["probe_harm_loss"] = L_harm.item()
```

**Critical requirements:**
- Probe head in **separate optimizer group** with own learning rate
- Probe targets **normalized** to match PPO value loss scale
- Gradient norm clipping **per-loss-component**

---

## 7. How the Probe is Used (And What It Must Never Do)

### 7.1 Allowed Uses (Safe)

1. **Observation feature for the actor**: Append probe outputs to Tamiyo obs
   - Because outputs are telemetry-derived, they fit Esper's philosophy
2. **Decision gating**: Use σ to discourage irreversible actions under high uncertainty
3. **Audit targeting**: Choose which slot to `AUDIT` based on probe uncertainty
4. **UI / interpretability**: Compare predicted vs true oracle values

### 7.2 Not Allowed (High-Risk)

**Using probe outputs directly as reward or reward shaping.**

> "If the agent can manipulate the probe (directly or indirectly), it will. Reward hacking becomes 'model hacking'."

Even if you freeze probe weights during rollouts, the agent can learn states that fool it. This is a classic Goodhart failure mode.

### 7.3 How σ Drives AUDIT (Active Perception)

Once you have σ, use it in three clean ways:

**1. Audit Targeting**

Select audit target by expected value of information:

```python
audit_priority = sigma * stake
# Where "stake" = probability of taking irreversible action soon
#              or predicted harm probability
#              or "this slot is expensive to keep"
```

**2. Conservative Gating for Irreversible Actions**

Soft rule: Discourage `FOSSILISE`/`PRUNE` when σ is high, but don't hard forbid.

**3. UI / Debugging**

Surface σ alongside μ and oracle measurements:
- "It made a wrong call but σ was huge" → agent had reason to be uncertain
- "It made a wrong call with tiny σ" → probe is miscalibrated (real bug)

---

## 8. Relationship to Active Perception (AUDIT)

The probe and `AUDIT` are complementary:

- **AUDIT**: "Pay compute to obtain ground truth for one slot now"
- **Probe**: "Use telemetry to maintain beliefs for all slots all the time"

Together they form a tidy loop:

```
1. Probe makes predictions + uncertainty for all slots
2. Policy chooses actions using predictions
3. Policy sometimes calls AUDIT for high-uncertainty/high-stakes slots
4. Audit produces labels; probe trains online; uncertainty improves over time
```

This is **active learning inside RL**.

---

## 9. Implementation Plan (Phased)

### Phase 1: Offline Probe Prototype (Minimum Viable Science)

**Deliverables:**
1. Data logging: Record telemetry features + oracle labels when oracle is computed
2. Standalone probe model: Train offline, validate metrics (correlation, calibration)
3. UI telemetry: Show probe vs oracle for inspected env/slot

**Success bar:** Probe achieves non-trivial correlation with oracle offline.

---

### Phase 2: Online Auxiliary Training (Probe Learns During PPO)

**Deliverables:**
1. Add probe head to model as auxiliary module (**separate from actor/critic trunk**)
2. Add `L_probe` into PPO update (weighted by `lambda_probe = 0.01`)
3. Train probe only on steps where oracle labels exist (mask-based loss)
4. Emit calibration and drift telemetry

**Implementation details:**
```python
# In ppo_agent.py optimizer setup
self.optimizer = torch.optim.AdamW([
    {'params': actor_params, 'weight_decay': 0.0, 'name': 'actor'},
    {'params': shared_params, 'weight_decay': 0.0, 'name': 'shared'},
    {'params': critic_params, 'weight_decay': weight_decay, 'name': 'critic'},
    {'params': probe_params, 'lr': probe_lr, 'weight_decay': 0.0, 'name': 'probe'},  # NEW
])
```

**Success bar:** PPO stays stable; probe improves over time; no gradient explosion.

---

### Phase 3: Policy Consumption (Probe Becomes Part of Tamiyo's "Mind")

**Deliverables:**
1. Append probe outputs to Tamiyo observation vector (using `.detach()`)
2. Add conservative decision logic: discourage FOSSILISE/PRUNE when σ high
3. Use probe uncertainty to drive audit selection
4. **Add probe output dropout (10-20%)** to prevent over-reliance

**Success bar:** Behaviour improves without reintroducing boundary-timed scripts.

---

### Phase 4: Scale-Up Protocol (Reduce Oracle Frequency)

**Deliverables:**
1. Replace "oracle every step" with sparse audits + periodic scheduled audits
2. Track probe error as a function of audit rate
3. Find minimal audit budget that maintains decision quality
4. **Include random audit fraction (10-25%)** to prevent selection bias

**Success bar:** 50+ seeds feasible; compute cost manageable.

---

### Phase 5: Set-Attention Probe (Option B)

**Deliverables:**
1. Implement cross-slot attention architecture
2. Profile memory (expect ~270MB with gradients for 100 slots)
3. Validate torch.compile compatibility with fixed shapes + masking

**Success bar:** Interaction effects captured; per-slot MLP errors reduced.

---

## 10. PyTorch Implementation Gotchas

### 10.1 torch.compile Compatibility

```python
# GOOD: Fixed slot count with masking (compile-friendly)
probe_out = probe_head(lstm_out)  # [batch, seq, K*3] where K=max_slots
active_mask = slot_states != DORMANT

# BAD: Dynamic slot slicing (graph breaks)
active_slots = get_active_slots()  # Variable length!
attn_out = attention(slots[:, :active_slots, :])  # Breaks compile
```

### 10.2 Gradient Flow Isolation

```python
# CORRECT: Detach LSTM output before probe head
probe_out = self.probe_head(lstm_out.detach())

# WRONG: Probe gradients flow through LSTM hidden states
probe_out = self.probe_head(lstm_out)  # Unbounded BPTT memory growth
```

### 10.3 Numerical Stability

```python
# Better stability than raw exp()
var = F.softplus(log_var).clamp(min=1e-4, max=50.0)

# Tune clamp range to your contribution scale
# If seed_contribution ∈ [-0.1, 0.1], use tighter range
log_var = log_var.clamp(-5, 2)  # var ∈ [0.0067, 7.4]
```

### 10.4 Scale Normalization

```python
# Normalize probe targets to match PPO value loss scale
self.probe_normalizer = ValueNormalizer(device=device)

# In update:
self.probe_normalizer.update(contrib_true[has_label])
normalized_contrib = self.probe_normalizer.normalize(contrib_true)
```

---

## 11. Experiment Plan

### 11.1 Baselines

- **Oracle-every-step** (small seed counts only): Upper bound for decision support
- **No oracle/no probe**: Telemetry-only
- **Probe trained offline, not used by policy** (diagnostic only)
- **Probe used by policy** (full intended usage)

### 11.2 Critical Ablations

| Ablation | Options | Hypothesis |
|----------|---------|------------|
| Trunk sharing | Separate vs shared | Separate safer, shared more efficient |
| Probe dropout | 0% vs 10% vs 20% | 10-20% prevents over-reliance |
| Lambda_probe | 0.001 vs 0.01 vs 0.1 | Start small, increase if probe underfits |
| Random audit fraction | 10% vs 25% | Prevents selection bias |
| Targets | Δloss vs bounded Δloss | Bounded more stable |

### 11.3 Key Metrics

**Probe Quality:**
- Pearson/Spearman correlation: ĉ_contrib vs contrib_true (per stage)
- Calibration: Brier score / ECE for harm probabilities
- Selective prediction: Error vs confidence (does σ mean anything?)
- Stage-conditioned error: TRAINING vs BLENDING vs HOLDING

**Policy Behaviour:**
- Rate of irreversible actions taken under high σ
- Late-episode liquidation spikes (prune/fossilise vs time)
- Alpha plan reset/thrash frequency
- Seed utilisation patterns (slam/hold/backfill dynamics)

**Compute:**
- Oracle calls per episode
- Wall time per episode
- Throughput vs number of seeds

---

## 12. Telemetry + UI Plan

### 12.1 Per-Slot Metrics

Add to Karn `probe_calibration` view:

| Field | Type | Description |
|-------|------|-------------|
| `slot_id` | string | Slot identifier |
| `stage` | enum | Current seed stage |
| `mu_pred` | float | Predicted contribution |
| `sigma_pred` | float | Predicted uncertainty |
| `contrib_true` | float | Oracle contribution (when available) |
| `p_harm_pred` | float | Predicted harm probability |
| `harm_true` | bool | Realized harm (episode-end) |
| `error` | float | |mu_pred - contrib_true| |
| `last_audit_step` | int | When last audited |

### 12.2 Global Metrics

- Coverage curve: "Top 20% confidence contains X% of accurate predictions"
- Audit allocation: Are audits going to high-uncertainty / high-stakes states?
- Probe error vs episode timestep (detect staleness)

### 12.3 Sanctum/Overwatch Integration

Side-by-side display:
- Predicted contribution vs true counterfactual contribution
- Predicted harm probability vs realised harm
- σ confidence band visualization
- Annotate decisions: "policy acted on predicted harm spike"

---

## 13. Risks and Mitigations

### Risk A: Probe Becomes Reward-Hack Surface

Even if you never use it as reward, the policy could learn to reach states where probe is systematically optimistic.

**Mitigations:**
- Keep probe as observation + auxiliary training only
- Include confidence estimates; penalise high-confidence wrong predictions
- Periodic random audits to prevent "dark corners"
- Monitor probe error over time (if error increases, agent may be gaming it)

### Risk B: Auxiliary Loss Destabilizes PPO

Probe loss can fight PPO if weighted badly.

**Mitigations:**
- Small `lambda_probe` with ramp-up schedule
- Separate optimizer group with own LR
- Gradient norm clipping per-loss component
- Log probe loss separately to detect explosions

### Risk C: Probe Staleness During Rollouts

During 150-step rollouts, probe weights are fixed but host model evolves.

**Mitigations:**
- Track probe error vs episode timestep
- Add timestep embedding to probe inputs
- Log calibration per lifecycle stage

### Risk D: Selection Bias from Probe-Driven Audits

If probe drives audit selection, it influences its own training distribution.

**Mitigations:**
- Scheduled random audits (10-25% of total)
- Use random audits as held-out validation set
- Importance weighting for audit-derived labels

### Risk E: Interaction Effects Dominate

If contributions are highly non-local, per-slot MLP probe may be misled.

**Mitigations:**
- Promote to set-attention probe (Option B) at 20+ seeds
- Add global context and stage embeddings
- Explicitly model synergy/competition

### Risk F: σ-Inflation ("I Don't Know" Spam)

If probe can get away with always outputting huge σ, it dodges loss.

**Mitigations:**
- NLL naturally penalizes via `log σ²` term
- Clamp σ (hard ceiling)
- Add mild regulariser pulling σ toward baseline
- Supervise σ toward empirical audit variance

---

## 14. Success Criteria

### Technical Success

- Probe loss converges to non-trivial performance:
  - Contribution predictor achieves meaningful correlation (r > 0.5)
  - Harm classifier beats stage-only baseline (AUROC > 0.7)
- PPO remains stable (no increased grad norm explosions)

### Behavioural Success

Evidence that policy decisions become less "boundary-timed":
- Reduced pruning/fossilisation spikes near episode end
- Reduced alpha plan resets per epoch
- Improved seed selection quality
- Improved final score or stability at equal compute

### Scaling Success

- Oracle compute becomes sparse and controllable
- 50–100+ seeds training throughput remains practical
- Decision quality maintained with 10% audit budget

### Conceptual Success (Esper Philosophy)

- At inference time, Tamiyo makes decisions using **only telemetry-derived predictions**
- The predicted signals are interpretable and visible in UI
- Probe outputs improve over time with sparse truth

---

## 15. Open Questions

1. **Target definition:** Bounded Δloss vs z-scored vs escrow delta — which is most stable?
2. **Trunk sharing:** Separate probe trunk vs shared LSTM features — which performs better?
3. **Audit action reward:** Should AUDIT have explicit cost penalty, or rely on downstream effects?
4. **Temporal prediction:** Should probe predict "contribution now" or "contribution if fossilised in N steps"?
5. **Interaction threshold:** At what seed count does Option B (attention) become necessary?

---

## 16. Phase Gate

**This proposal is blocked on Phase 2.5 Reward Efficiency Protocol completion.**

Rationale: If the agent can't learn with dense oracle, adding a probe won't help. The probe is a scaling enabler, not a correctness fix. Once the agent demonstrates it can learn optimal growth strategies with full oracle access, the probe becomes essential for scaling beyond 10-20 seeds.

---

## Appendix A: Alternative Approaches Considered

### Privileged Critic (Asymmetric Actor-Critic)

Give the **critic** access to oracle counterfactuals during training (not the actor).

**Pros:**
- No auxiliary loss needed
- Critic naturally handles credit assignment

**Cons:**
- Requires oracle at every training step (doesn't reduce compute)
- Actor may learn to rely on critic signals unavailable at test time

**Verdict:** Orthogonal to probe approach; could be combined as parallel experiment.

### Model-Based RL with World Model

Learn a world model that predicts next-state given action, then simulate ablation rollouts.

**Pros:**
- More general (predicts full dynamics)
- Can be used for planning (MuZero-style)

**Cons:**
- Much more complex
- World model errors compound over multi-step rollouts

**Verdict:** Overkill for current needs; revisit for Phase 6 (Recursion).

---

## Appendix B: Expert Review Summary

This proposal was reviewed by DRL and PyTorch specialists. Key findings:

| Aspect | DRL Expert | PyTorch Expert |
|--------|------------|----------------|
| Utility | 8/10 | 8/10 |
| Priority | 7/10 | 7/10 |
| Core concern | Auxiliary loss stability | Gradient isolation |
| Key recommendation | Start with separate trunk | Use softplus for log_var |

Full reviews available in project discussion history.
