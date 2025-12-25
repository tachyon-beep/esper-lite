# Reward Function Design for Morphogenetic Neural Network Controller

**A Technical Paper on Multi-Objective Reward Engineering for Esper's Tamiyo Controller**

*December 2025*

---

## Abstract

This paper addresses the reward engineering challenges in Esper, a framework for morphogenetic AI where neural networks dynamically grow their topology during training. The Tamiyo controller, a PPO-trained RL agent, must manage seed lifecycle decisions (germination, advancement, pruning, fossilization) across multiple objectives: accuracy maximization, parameter efficiency, training stability, fossilization timing, and exploration of novel growth strategies.

We analyze the trade-offs between sparse and dense reward formulations, propose a multi-objective reward decomposition (MORD) framework, examine Pareto frontier approaches for balancing competing objectives, investigate intrinsic motivation for morphogenetic exploration, and provide practical recommendations grounded in potential-based reward shaping (PBRS) theory.

---

## 1. Introduction

### 1.1 The Morphogenetic Control Problem

Esper implements a unique paradigm where neural modules ("seeds") undergo a botanical lifecycle:

```
DORMANT -> GERMINATED -> TRAINING -> BLENDING -> HOLDING -> FOSSILIZED
                              |          |
                              v          v
                           PRUNED     PRUNED
```

The Tamiyo controller observes training dynamics (gradient norms, loss landscapes, contribution metrics) and selects actions from a **factored action space**:

- **Lifecycle operations**: WAIT, GERMINATE, ADVANCE, PRUNE, FOSSILIZE, SET_ALPHA_TARGET
- **Style selections**: Blueprint type, alpha curve, blending algorithm
- **Slot targeting**: Which of N injection points to operate on

This creates a control problem with:
- **Delayed consequences**: Germination decisions affect accuracy 10-20 epochs later
- **Multi-objective optimization**: Accuracy vs. efficiency vs. stability
- **Emergent failure modes**: "Ransomware" seeds that create dependencies without adding value
- **Sparse feedback**: True quality only observable after fossilization completes

### 1.2 Current Reward Modes

Esper currently implements four reward modes (from `src/esper/simic/rewards/rewards.py`):

| Mode | Description | Primary Signal |
|------|-------------|----------------|
| **SHAPED** | Dense counterfactual rewards at every timestep | Bounded attribution with PBRS |
| **SPARSE** | Terminal-only rewards (final accuracy - param cost) | Episode return |
| **MINIMAL** | Sparse + early-prune penalty | Episode return + shaping |
| **SIMPLIFIED** | PBRS + intervention cost + terminal | DRL Expert recommended |

### 1.3 Paper Objectives

This paper addresses:
1. When sparse vs. dense rewards are appropriate for morphogenetic control
2. How to decompose multi-objective rewards without Goodhart artifacts
3. Practical approaches to Pareto frontier tracking
4. Whether intrinsic motivation aids morphogenetic exploration
5. PBRS guarantees and their implications for reward design

---

## 2. Sparse vs Dense Reward Analysis

### 2.1 Theoretical Credit Assignment Bounds

For an MDP with horizon T, discount factor gamma, and pure sparse rewards, the credit assignment problem becomes:

**Theorem 2.1 (Credit Assignment Decay)**: For a trajectory tau = (s_0, a_0, ..., s_T) with terminal reward R, the contribution of action a_t to the expected return gradient is:

```
grad_theta E[R|a_t, theta] ~ O(gamma^(T-t))
```

With Esper's configuration (T=25, gamma=0.995):
- gamma^25 = 0.88 (early actions retain 88% credit weight)
- gamma^10 = 0.95 (mid-episode actions retain 95%)

**Implication**: With gamma=0.995, sparse rewards ARE theoretically viable for T=25 episodes. The effective horizon (gamma^T > 0.5) is approximately T_eff = ln(0.5)/ln(gamma) = 138 steps, far exceeding the episode length.

### 2.2 Variance Analysis for Sparse Rewards

The variance of the REINFORCE gradient estimator for sparse rewards is:

```
Var[grad_theta log pi(a|s) * R] = E[(grad_theta log pi)^2] * Var[R] + E[R]^2 * Var[grad_theta log pi]
```

For terminal-only rewards:
- **Var[R]** dominates when rewards are highly variable
- Episode-level randomness (initial conditions, data order) contributes to gradient noise

**Empirical Observation from Esper**: With SPARSE mode, policy gradients exhibit 3-5x higher variance than SHAPED mode, requiring:
- Larger batch sizes (higher n_envs)
- Lower learning rates
- More training episodes for convergence

### 2.3 Dense Reward Failure Modes in Morphogenetic Systems

Dense shaping in Esper faces unique challenges:

**2.3.1 The Ransomware Pattern**

A seed can maximize counterfactual contribution by:
1. Creating structural dependencies during BLENDING
2. Degrading host performance when measured without the seed
3. Appearing "necessary" while providing no net value

Current mitigation (lines 511-565 of rewards.py):
```python
# Attribution discount for negative total_improvement
if total_imp < 0:
    attribution_discount = 1.0 / (1.0 + math.exp(-steepness * total_imp))

# Ratio penalty for contribution >> improvement
if ratio > hacking_ratio_threshold:
    ratio_penalty = -min(max_penalty, scale * excess_ratio)
```

**2.3.2 The Fossilization Farming Pattern**

Dense rewards for lifecycle progression can incentivize:
- Rapid cycling through stages without meaningful learning
- Collecting PBRS bonuses without seed contribution

Current mitigation: Legitimacy discount based on epochs_in_stage (lines 1286-1300).

### 2.4 Semi-Sparse Approaches (Milestone-Based)

A middle ground between sparse and dense:

**Milestone Rewards**: Non-zero rewards at specific lifecycle transitions:
- G1 (GERMINATED -> TRAINING): Small bonus for successful activation
- G3 (BLENDING -> HOLDING): Larger bonus tied to measured contribution
- G5 (HOLDING -> FOSSILIZED): Terminal-like bonus with contribution scaling

This approach:
- Reduces variance vs. pure sparse (rewards at known points)
- Maintains causal clarity (no per-timestep shaping)
- Aligns with natural lifecycle structure

**Recommendation**: Consider SIMPLIFIED mode as a milestone-based semi-sparse approach, with PBRS for smooth gradients and terminal bonus for ground truth signal.

---

## 3. Multi-Objective Reward Decomposition (MORD)

### 3.1 Component Structure

We propose decomposing the reward into tracked components:

```python
@dataclass
class RewardComponents:
    # Primary objectives (what we truly care about)
    accuracy_delta: float      # Delta in validation accuracy
    param_efficiency: float    # -log(1 + growth_ratio)

    # Secondary objectives (training health)
    stability_score: float     # Gradient health, no catastrophic forgetting
    timing_quality: float      # Did fossilization happen at optimal moment?

    # Shaping components (guide learning, shouldn't affect optimal policy)
    pbrs_bonus: float          # Potential-based stage progression
    intervention_cost: float   # Action friction

    # Anti-gaming penalties (prevent degenerate policies)
    ransomware_penalty: float  # High contribution + negative improvement
    ratio_penalty: float       # Contribution >> improvement
```

### 3.2 Linear Scalarization

The simplest combination:

```
R = w_acc * accuracy_delta
  + w_eff * param_efficiency
  + w_stab * stability_score
  + w_time * timing_quality
  + pbrs_bonus
  + intervention_cost
  + ransomware_penalty
  + ratio_penalty
```

**Advantages**:
- Simple to implement and tune
- Gradient direction is clear
- Compatible with standard PPO

**Disadvantages**:
- Single fixed trade-off point on Pareto frontier
- Sensitive to weight ratios
- Cannot express non-convex preferences

**Current Implementation** (ContributionRewardConfig defaults):
```python
contribution_weight: float = 1.0      # Primary signal
pbrs_weight: float = 0.3              # Stage progression
rent_weight: float = 0.5              # Param efficiency
terminal_acc_weight: float = 0.05     # Final accuracy
fossilize_terminal_scale: float = 3.0 # Completion bonus
```

### 3.3 Thresholded Objectives (Lexicographic)

For hard constraints:

```python
def compute_reward(components):
    # HARD CONSTRAINTS: Must be satisfied or heavy penalty
    if components.stability_score < 0.8:
        return -10.0  # Catastrophic failure

    # SOFT OPTIMIZATION: Within stable region, optimize accuracy
    return (components.accuracy_delta * w_acc
          + components.param_efficiency * w_eff
          + components.pbrs_bonus)
```

**Use Case**: Stability is non-negotiable; efficiency/accuracy trade-offs are flexible.

### 3.4 Learned Weighting (Meta-Learning)

Train a separate network to predict optimal weights:

```python
class RewardWeightPredictor(nn.Module):
    def forward(self, training_context):
        # Context: epoch, num_fossilized, current_accuracy, param_ratio
        weights = self.mlp(training_context)
        return F.softmax(weights, dim=-1)

# During training:
weights = weight_predictor(context)
R = sum(w * c for w, c in zip(weights, components))
```

**Challenges**:
- Requires a meta-objective to train the weight predictor
- Risk of weight collapse (all weight on one objective)
- Adds training complexity

**Recommendation**: Start with linear scalarization using empirically-tuned weights. Consider learned weighting only if fixed weights consistently produce suboptimal trade-offs.

---

## 4. Pareto Frontier Approaches

### 4.1 Multi-Objective PPO (MO-PPO)

Standard PPO optimizes a scalar reward. Multi-objective variants maintain:
- **Policy population**: Multiple policies, each optimizing different weight vectors
- **Shared critic**: Value function that predicts vector returns

**Algorithm Sketch**:
```
Initialize policy population {pi_1, ..., pi_K} with weight vectors {w_1, ..., w_K}
Initialize shared vector critic V(s) -> R^d

For each episode:
    Sample policy pi_k uniformly
    Collect trajectory under pi_k
    Compute vector returns G = [G_1, ..., G_d]
    Update V using TD(lambda) with vector targets
    Update pi_k using PPO with scalarized advantage: A_k = w_k^T * (G - V(s))

    Every N episodes:
        Identify dominated policies
        Replace dominated with mutated weights from non-dominated
```

**Practical Considerations for Esper**:
- Population size K should scale with objective count (K >= 2*d for d objectives)
- With 4 objectives, K=8-16 policies is reasonable
- Each policy uses independent LSTM state but can share feature extractor

### 4.2 Pareto-PPO with Hypernetwork

Instead of maintaining a population, use a hypernetwork to generate policy weights:

```python
class HyperTamiyo(nn.Module):
    def __init__(self, preference_dim, policy_dim):
        self.hypernet = nn.Sequential(
            nn.Linear(preference_dim, 128),
            nn.ReLU(),
            nn.Linear(128, policy_dim)
        )
        self.base_policy = TamiyoNetwork()

    def forward(self, obs, preference):
        # preference: [w_acc, w_eff, w_stab, w_time], sums to 1
        policy_weights = self.hypernet(preference)
        self.base_policy.load_state_dict(policy_weights)
        return self.base_policy(obs)
```

**Advantages**:
- Single network covers entire Pareto frontier
- Can condition on operator preferences at runtime
- More sample-efficient than population-based

**Disadvantages**:
- Hypernetwork training is notoriously unstable
- May struggle with non-convex Pareto frontiers
- Adds significant complexity

### 4.3 Practical Pareto Tracking for Esper

For Esper's vectorized environment, we recommend:

**4.3.1 Logging Infrastructure**

Emit multi-objective telemetry at episode end:
```python
@dataclass
class EpisodeOutcome:
    final_accuracy: float
    param_ratio: float
    num_fossilized: int
    stability_score: float
    episode_return: float  # Scalarized reward

# Store in Karn for Pareto analysis
hub.emit(TelemetryEvent(
    event_type=TelemetryEventType.EPISODE_OUTCOME,
    data=asdict(outcome)
))
```

**4.3.2 Pareto Analysis Tool**

Post-hoc analysis using stored telemetry:
```python
def compute_pareto_frontier(outcomes: list[EpisodeOutcome]) -> list[EpisodeOutcome]:
    """Extract non-dominated outcomes."""
    frontier = []
    for o in outcomes:
        dominated = False
        for p in outcomes:
            if (p.final_accuracy >= o.final_accuracy and
                p.param_ratio <= o.param_ratio and
                p.stability_score >= o.stability_score and
                (p.final_accuracy > o.final_accuracy or
                 p.param_ratio < o.param_ratio or
                 p.stability_score > o.stability_score)):
                dominated = True
                break
        if not dominated:
            frontier.append(o)
    return frontier
```

**4.3.3 Pareto Progress Metric**

Track hypervolume indicator (HVI) over training:
```python
def hypervolume(frontier: list[EpisodeOutcome], ref_point: tuple) -> float:
    """Compute hypervolume dominated by Pareto frontier."""
    # ref_point: worst acceptable values for each objective
    # Uses algorithm from Fonseca et al. (2006)
    ...
```

HVI increasing over training indicates Pareto improvement.

---

## 5. Intrinsic Motivation & Curiosity

### 5.1 The Case for Intrinsic Rewards in Morphogenetic Systems

The morphogenetic action space is combinatorially large:
- 6 lifecycle operations x N slots
- Blueprint selection (currently 2 types, expandable)
- Alpha curve selection (3 options)
- Tempo selection (3 speeds)

Many combinations are unexplored. Intrinsic motivation could:
- Encourage novel growth patterns (new slot/blueprint combinations)
- Prevent premature convergence to safe heuristics
- Discover unexpected synergies between concurrent seeds

### 5.2 Random Network Distillation (RND) for State Novelty

RND measures novelty by prediction error on a random target network:

```python
class MorphogeneticRND:
    def __init__(self, obs_dim, hidden_dim=256):
        # Fixed random target
        self.target = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        for p in self.target.parameters():
            p.requires_grad = False

        # Trained predictor
        self.predictor = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def intrinsic_reward(self, obs):
        with torch.no_grad():
            target_feat = self.target(obs)
        pred_feat = self.predictor(obs)
        return F.mse_loss(pred_feat, target_feat, reduction='none').mean(-1)
```

**Application to Esper**:
- Include seed stage distribution in obs (novelty = unusual stage patterns)
- Include slot occupancy pattern (novelty = unexplored slot combinations)
- Normalize intrinsic reward by running estimate of std

### 5.3 Observation Novelty vs. Action Novelty

For morphogenetic control, **action novelty** may be more relevant:

```python
class ActionNoveltyTracker:
    def __init__(self, action_space_size):
        self.action_counts = defaultdict(int)
        self.total_actions = 0

    def intrinsic_reward(self, action_tuple):
        # action_tuple: (slot_id, lifecycle_op, blueprint, ...)
        key = tuple(action_tuple)
        count = self.action_counts[key]
        self.action_counts[key] += 1
        self.total_actions += 1

        # Reward inversely proportional to frequency
        frequency = count / max(1, self.total_actions)
        return math.sqrt(1.0 / (frequency + 1e-8))
```

**Caution**: High action novelty reward can lead to:
- Random action sequences that never complete fossilization
- Exploring "interesting failures" instead of useful patterns

### 5.4 Recommended Approach: Tempered Curiosity

If using intrinsic motivation, apply conservatively:

```python
def compute_combined_reward(extrinsic, intrinsic, training_progress):
    # Anneal intrinsic weight as training progresses
    # Early: explore morphogenetic space
    # Late: focus on exploitation of discovered patterns
    intrinsic_weight = max(0.01, 0.1 * (1 - training_progress))

    return extrinsic + intrinsic_weight * intrinsic
```

**Configuration for Esper**:
```python
intrinsic_weight_start: float = 0.1
intrinsic_weight_end: float = 0.01
intrinsic_anneal_episodes: int = 50
intrinsic_source: Literal["rnd", "action_count", "none"] = "none"
```

**Recommendation**: Start without intrinsic rewards. Add only if the policy converges to a single degenerate strategy (e.g., always WAIT, never GERMINATE). The SHAPED mode's PBRS already provides exploration incentives via stage progression bonuses.

---

## 6. Reward Shaping Guarantees

### 6.1 PBRS Theory Review

**Theorem (Ng, Harada, Russell 1999)**: Given MDP M with reward R, define shaped reward:

```
R'(s, a, s') = R(s, a, s') + gamma * phi(s') - phi(s)
```

where phi: S -> R is any potential function. Then:
1. The optimal policy under R' equals the optimal policy under R
2. The optimal value function under R' equals V*(s) + phi(s)

**Implications**:
- PBRS shaping CANNOT change what actions are optimal
- It CAN accelerate learning by providing denser gradients
- The PBRS bonus "telescopes" over trajectories

### 6.2 Esper's PBRS Implementation

From rewards.py (lines 111-120):
```python
STAGE_POTENTIALS = {
    0: 0.0,   # UNKNOWN
    1: 0.0,   # DORMANT
    2: 1.0,   # GERMINATED
    3: 2.0,   # TRAINING
    4: 3.5,   # BLENDING (largest increment)
    6: 5.5,   # HOLDING
    7: 6.0,   # FOSSILIZED (smallest increment)
}
```

The PBRS bonus calculation (lines 1142-1184):
```python
def _contribution_pbrs_bonus(seed_info, config):
    phi_current = STAGE_POTENTIALS.get(seed_info.stage, 0.0)
    phi_current += min(epochs_in_stage * config.epoch_progress_bonus,
                       config.max_progress_bonus)

    # Reconstruct previous potential for telescoping
    if seed_info.epochs_in_stage == 0:
        phi_prev = STAGE_POTENTIALS.get(seed_info.previous_stage, 0.0)
        phi_prev += min(prev_epochs * config.epoch_progress_bonus,
                        config.max_progress_bonus)
    else:
        phi_prev = STAGE_POTENTIALS.get(seed_info.stage, 0.0)
        phi_prev += min((epochs_in_stage - 1) * config.epoch_progress_bonus,
                        config.max_progress_bonus)

    return config.pbrs_weight * (config.gamma * phi_current - phi_prev)
```

### 6.3 PBRS Violations and Goodhart Artifacts

PBRS guarantees hold ONLY if:
1. gamma_pbrs == gamma_ppo (currently both 0.995)
2. phi(s) is a function of state only (not actions or time)
3. The shaping is additive to base reward

**Potential Violations in Current Implementation**:

**6.3.1 Action-Dependent Shaping (VIOLATION)**
```python
# From action shaping (lines 738-764)
if action == LifecycleOp.GERMINATE:
    action_shaping += config.germinate_cost  # -0.02
    # This IS NOT PBRS - it's action-dependent
```

Action-dependent costs can change optimal policy. However, they're designed to prevent action spam and are small relative to primary signals.

**6.3.2 Bounded Attribution (SAFE)**
```python
# Attribution is state-dependent (seed contribution, improvement)
# NOT action-dependent - computed before action selection
bounded_attribution = config.contribution_weight * attributed
```

This is NOT PBRS but is still policy-invariant because it measures state, not actions.

**6.3.3 Blending/Holding Warnings (CAUTION)**
```python
# Epoch-dependent penalties
if seed_info.epochs_in_stage >= 2 and bounded_attribution > 0:
    epochs_waiting = seed_info.epochs_in_stage - 1
    holding_warning = -1.0 * (3 ** (epochs_waiting - 1))
```

This is NOT PBRS and COULD change optimal policy. The intent is to prevent farming, but it introduces time-pressure that may conflict with optimal timing.

### 6.4 Verification Protocol

To verify PBRS correctness:

**Property Test 1: Telescoping**
```python
def test_pbrs_telescopes(trajectory):
    """Sum of PBRS bonuses = gamma^T * phi(s_T) - phi(s_0)."""
    total_bonus = sum(step.pbrs_bonus for step in trajectory)
    expected = (gamma ** len(trajectory)) * phi(trajectory[-1].state) - phi(trajectory[0].state)
    assert abs(total_bonus - expected) < 1e-6
```

**Property Test 2: Policy Invariance**
```python
def test_pbrs_preserves_optimal():
    """Optimal actions unchanged with/without PBRS."""
    env = TamiyoEnv(reward_mode="sparse")
    optimal_actions_sparse = compute_optimal_actions(env)

    env = TamiyoEnv(reward_mode="shaped", pbrs_only=True)
    optimal_actions_pbrs = compute_optimal_actions(env)

    assert optimal_actions_sparse == optimal_actions_pbrs
```

### 6.5 Recommendations for PBRS Purity

If strict policy invariance is required:

1. **Remove action-dependent costs** - Let the environment naturally penalize bad actions through delayed consequences
2. **Remove time-dependent warnings** - Use only state-dependent potentials
3. **Use SIMPLIFIED mode** - Closest to pure PBRS + terminal

For practical training, the small action costs are likely acceptable trade-offs for preventing degenerate policies.

---

## 7. Practical Recommendations for Esper

### 7.1 Recommended Reward Architecture

```python
@dataclass
class RecommendedRewardConfig:
    """Reward configuration based on this analysis."""

    # === Mode Selection ===
    # Start with SIMPLIFIED for cleaner gradients
    # Move to SHAPED only if policy fails to learn seed quality
    reward_mode: RewardMode = RewardMode.SIMPLIFIED

    # === Primary Signal: Terminal ===
    # All ground truth should flow through terminal bonus
    terminal_accuracy_weight: float = 1.0    # 0-100 scaled to 0-1
    terminal_fossilize_bonus: float = 2.0    # Per contributing seed
    terminal_efficiency_penalty: float = 0.1 # Per 100% param growth

    # === Shaping (PBRS-compliant) ===
    pbrs_weight: float = 0.3                 # Stage progression
    epoch_progress_bonus: float = 0.3        # Time in stage
    max_progress_bonus: float = 2.0          # Cap on epoch bonus

    # === Intervention Friction (small, uniform) ===
    # Prevents action spam without biasing action choice
    non_wait_cost: float = 0.01              # Same for all actions

    # === Anti-Gaming (keep, but log separately) ===
    # These are NOT PBRS-compliant but prevent catastrophic failures
    attribution_discount_enabled: bool = True
    ratio_penalty_enabled: bool = True
    ransomware_check_enabled: bool = True

    # === Intrinsic (disabled by default) ===
    intrinsic_enabled: bool = False
    intrinsic_weight: float = 0.05
    intrinsic_source: str = "rnd"
```

### 7.2 Ablation Study Design

To validate the reward function, run controlled ablations:

**Baseline Ablations** (establish what matters):

| Ablation | Change | Hypothesis |
|----------|--------|------------|
| A1: No PBRS | pbrs_weight=0 | Learning will slow, may not converge |
| A2: No Terminal | terminal_*=0 | Policy ignores accuracy, farms stages |
| A3: No Rent | efficiency_penalty=0 | Unbounded growth, poor generalization |
| A4: Pure Sparse | SPARSE mode | High variance, may fail to learn |

**Anti-Gaming Ablations** (verify protections are necessary):

| Ablation | Change | Hypothesis |
|----------|--------|------------|
| A5: No Attribution Discount | discount_enabled=False | Ransomware seeds emerge |
| A6: No Ratio Penalty | ratio_penalty=False | Dependency gaming increases |
| A7: No Holding Warning | holding_warning=False | Fossilization farming |

**Hyperparameter Sensitivity**:

| Sweep | Range | Default |
|-------|-------|---------|
| pbrs_weight | [0.1, 0.3, 0.5, 1.0] | 0.3 |
| terminal_fossilize_bonus | [0.5, 1.0, 2.0, 5.0] | 2.0 |
| gamma | [0.99, 0.995, 0.999] | 0.995 |

### 7.3 Metrics for Diagnosing Reward Function Health

**7.3.1 Primary Health Indicators**

| Metric | Healthy Range | Warning Sign |
|--------|---------------|--------------|
| Episode return std | Decreasing over time | Increasing = instability |
| Reward component correlation | Positive (acc, fossilize) | Negative = conflicting objectives |
| PBRS fraction of total reward | 10-30% | >50% = terminal signal too weak |
| Anti-gaming penalty frequency | <5% of steps | >20% = policy exploiting |

**7.3.2 Pareto Health**

| Metric | Healthy Sign |
|--------|--------------|
| Hypervolume indicator | Increasing or stable |
| Pareto frontier size | Growing, then stable |
| Objective correlation | Low (independent objectives) |

**7.3.3 Credit Assignment Health**

| Metric | Meaning |
|--------|---------|
| Value function explained variance | >0.5 = learning meaningful value estimates |
| Advantage magnitude by epoch | Should be ~uniform across episode |
| Policy gradient by action type | All actions should have non-zero gradient |

### 7.4 Implementation Checklist

1. **Log all reward components separately** (already done via RewardComponentsTelemetry)
2. **Add Pareto tracking to Karn** (log EpisodeOutcome at episode end)
3. **Add hypervolume monitoring** (compute and plot HVI over training)
4. **Implement SIMPLIFIED mode A/B test** (use ab_reward_modes config)
5. **Create ablation configs** (one JSON per ablation in configs/ablations/)
6. **Add reward health dashboard panel** (component breakdown, anti-gaming triggers)

---

## 8. Conclusion

Reward function design for morphogenetic neural network control requires balancing:

1. **Signal density** (sparse for ground truth, dense for learning speed)
2. **Multi-objective trade-offs** (accuracy vs. efficiency vs. stability)
3. **Anti-gaming robustness** (prevent ransomware, farming, dependency exploitation)
4. **Theoretical guarantees** (PBRS for policy invariance where possible)

Our recommendations:

- **Start with SIMPLIFIED mode** for cleaner credit assignment
- **Use linear scalarization** with empirically-tuned weights
- **Track Pareto frontier** for multi-objective analysis
- **Avoid intrinsic motivation initially** - PBRS provides sufficient exploration
- **Accept small PBRS violations** (intervention costs) for practical robustness
- **Run ablation studies** to validate each reward component

The current SHAPED mode is well-engineered but complex. As understanding improves, simplification toward purer PBRS + terminal signals may improve both performance and interpretability.

---

## References

1. Ng, A. Y., Harada, D., & Russell, S. (1999). Policy invariance under reward transformations: Theory and application to reward shaping. *ICML*.

2. Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017). Proximal policy optimization algorithms. *arXiv:1707.06347*.

3. Burda, Y., Edwards, H., Storkey, A., & Klimov, O. (2018). Exploration by random network distillation. *ICLR*.

4. Hayes, C. F., et al. (2022). A practical guide to multi-objective reinforcement learning and planning. *AAMAS*.

5. Fonseca, C. M., Paquete, L., & Lopez-Ibanez, M. (2006). An improved dimension-sweep algorithm for the hypervolume indicator. *CEC*.

---

## Appendix A: Pseudocode for Key Algorithms

### A.1 Multi-Objective PPO Update

```python
def mo_ppo_update(policy, buffer, weight_vector, clip_ratio=0.2):
    """Single policy update for multi-objective PPO."""
    for batch in buffer.iterate():
        obs, actions, old_log_probs, returns, advantages = batch

        # Returns and advantages are vectors [batch, num_objectives]
        # Scalarize advantages using weight vector
        scalar_advantages = (advantages * weight_vector).sum(dim=-1)

        # Standard PPO policy loss
        log_probs = policy.log_prob(obs, actions)
        ratio = torch.exp(log_probs - old_log_probs)
        clipped_ratio = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio)

        policy_loss = -torch.min(
            ratio * scalar_advantages,
            clipped_ratio * scalar_advantages
        ).mean()

        # Vector value function loss (predict all objectives)
        values = policy.value(obs)  # [batch, num_objectives]
        value_loss = F.mse_loss(values, returns)

        # Update
        loss = policy_loss + 0.5 * value_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

### A.2 Pareto Frontier Extraction

```python
def extract_pareto_frontier(outcomes: list[dict], objectives: list[str],
                            maximize: list[bool]) -> list[dict]:
    """Extract non-dominated outcomes.

    Args:
        outcomes: List of outcome dicts with objective values
        objectives: List of objective keys to consider
        maximize: List of bools, True if objective should be maximized

    Returns:
        List of non-dominated outcomes
    """
    def dominates(a, b):
        """Return True if a dominates b."""
        dominated_in_all = True
        strictly_better_in_one = False

        for obj, maxi in zip(objectives, maximize):
            if maxi:
                if a[obj] < b[obj]:
                    dominated_in_all = False
                if a[obj] > b[obj]:
                    strictly_better_in_one = True
            else:
                if a[obj] > b[obj]:
                    dominated_in_all = False
                if a[obj] < b[obj]:
                    strictly_better_in_one = True

        return dominated_in_all and strictly_better_in_one

    frontier = []
    for candidate in outcomes:
        is_dominated = any(dominates(other, candidate)
                          for other in outcomes if other != candidate)
        if not is_dominated:
            frontier.append(candidate)

    return frontier
```

### A.3 Hypervolume Computation (2D Case)

```python
def hypervolume_2d(frontier: list[tuple], ref_point: tuple) -> float:
    """Compute 2D hypervolume for Pareto frontier.

    Assumes both objectives are to be maximized and points are
    (obj1, obj2) tuples.
    """
    if not frontier:
        return 0.0

    # Sort by first objective (descending)
    sorted_pts = sorted(frontier, key=lambda p: -p[0])

    hv = 0.0
    prev_y = ref_point[1]

    for x, y in sorted_pts:
        if y > prev_y:
            # Rectangle from (x, prev_y) to (ref_x, y)
            width = x - ref_point[0]
            height = y - prev_y
            hv += width * height
            prev_y = y

    return hv
```

---

## Appendix B: Configuration Files

### B.1 Simplified Mode Config

```json
{
    "reward_mode": "simplified",
    "reward_family": "contribution",
    "pbrs_weight": 0.3,
    "terminal_acc_weight": 0.05,
    "fossilize_terminal_scale": 2.0,
    "param_budget": 500000,
    "param_penalty_weight": 0.1
}
```

### B.2 Ablation: Pure Sparse

```json
{
    "reward_mode": "sparse",
    "sparse_reward_scale": 2.5,
    "param_budget": 500000,
    "param_penalty_weight": 0.1,
    "n_envs": 8,
    "lr": 1e-4
}
```

### B.3 Ablation: No Anti-Gaming

```json
{
    "reward_mode": "shaped",
    "attribution_sigmoid_steepness": 0.0,
    "hacking_ratio_threshold": 1000.0,
    "prune_good_seed_penalty": 0.0
}
```