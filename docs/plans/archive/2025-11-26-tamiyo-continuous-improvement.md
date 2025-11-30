# Tamiyo Continuous Improvement Roadmap

**Date**: 2025-11-26
**Status**: Planning
**Goal**: Evolve Policy Tamiyo from imitation learning to autonomous improvement

## Context

Policy Tamiyo v1 uses imitation learning - she learns to copy Heuristic Tamiyo's decisions. This has a hard ceiling: she can only be as good as the heuristic she's imitating.

To exceed the heuristic, we need to shift from "what did the expert do?" to "what leads to good outcomes?" This document outlines the progression.

## The Progression

### Phase 1: Imitation Learning (Complete)

**Approach**: Supervised learning on (observation, action) pairs from Heuristic Tamiyo (Kasmina).

**Loss**: `CrossEntropy(predicted_action, heuristic_action)` with class weighting

**Ceiling**: Heuristic performance (can't exceed the teacher)

**Status**: ✅ Complete (2025-11-26)

**Results**:
| Dataset | Episodes | Features | Balanced Accuracy |
|---------|----------|----------|-------------------|
| V1 | 234 | 27 | 98.2% |
| V2 (research telemetry) | 300 | ~288 | 98.9% |

Per-class accuracy (V2):
- WAIT: 98.0%
- GERMINATE: 100%
- ADVANCE: 97.6%
- CULL: 100%

**Key learnings**:
- Class weighting essential (94% WAIT imbalance)
- Effective number of samples weighting (β=0.999) works well
- Optimize for balanced accuracy, not overall accuracy
- V2 rich telemetry improved ADVANCE detection (94% → 98%)

**Data packs**:
- `data/packs/simic_v1_final.json` (17 MB)
- `data/packs/simic_v2_research_2025-11-26.json` (37 MB)

**Value**: Proves the pipeline works, gives a solid initialization for later phases.

---

### Phase 2: Reward-Weighted Imitation

**Approach**: Same supervised setup, but weight examples by outcome quality.

**Key insight**: Not all heuristic decisions are equally good. Some episodes end at 80% accuracy, others at 70%. Weight the good ones higher.

**Implementation options**:

**Option A: Episode-level weighting**
```python
# Weight entire episodes by final accuracy
episode_weight = (episode.final_accuracy - baseline) / scale
for decision in episode.decisions:
    loss += weight * cross_entropy(pred, label)
```

**Option B: Decision-level weighting**
```python
# Weight individual decisions by their reward
decision_weight = decision.outcome.reward
loss += weight * cross_entropy(pred, label)
```

**Option C: Filtering**
```python
# Only train on "good" episodes
good_episodes = [ep for ep in episodes if ep.final_accuracy > threshold]
```

**Recommended**: Start with Option C (simplest), then try Option A.

**Ceiling**: Still bounded by heuristic's action distribution, but biased toward successful patterns.

**Complexity**: Minimal code change - just modify the data loading.

---

### Phase 3: Offline RL

**Approach**: Learn from logged episodes, but optimize for cumulative reward directly.

**Key insight**: We have (state, action, reward, next_state) tuples. This is an offline RL dataset.

**Status**: 🔶 Implemented but Limited by Data (2025-11-26)

**Implementation**: `src/esper/simic_iql.py`

**Results**:

| Method | α/τ | Agreement | Action Distribution |
|--------|-----|-----------|---------------------|
| IQL | τ=0.7 | 12.9% | 70% GERMINATE (bad!) |
| IQL | τ=0.9 | 6.6% | 51% CULL (worse!) |
| CQL | α=5.0 | 86.7% | 90% WAIT (close to behavior) |
| CQL | α=1.0 | 51.1% | 50% WAIT, 49% GERMINATE |
| CQL | α=0.5 | 72.6% | 75% WAIT, 25% GERMINATE |

**Key Finding: IQL Failed, CQL Works (But Can't Improve)**

IQL failed catastrophically because:
- Expectile regression with high τ selects high-variance optimistic estimates
- Rare actions (GERMINATE 2%, CULL 0.5%) have inflated upper tails
- Policy diverges to prefer rare actions based on noise, not signal

CQL works because it explicitly penalizes Q-values for OOD actions.

**Fundamental Limitation: No Counterfactual Data**

Action diversity analysis revealed:
- 87% of state clusters have **only one action** (Kasmina is deterministic)
- Only 13/103 state clusters have counterfactual data
- Kasmina follows simple rules: same state → same action

**Implication**: Offline RL cannot significantly outperform Kasmina because there's no data showing "what would have happened if we chose differently."

**Future Direction: Diverse Training Runs**

The solution is generating data with varied:
- Model architectures (different CNNs, ResNets, etc.)
- Datasets (CIFAR-100, different subsets)
- Hyperparameters (learning rates, batch sizes)

This creates situations where the "right" action varies, giving counterfactual data. The V2 telemetry (gradient health, per-class accuracy, sharpness) then becomes **discriminative** - helping Tamiyo understand *why* intervention helps in one case but not another.

**Methods implemented**:

**CQL (Conservative Q-Learning)** ✅
- Works with α≥5.0 for conservative behavior
- Lower α causes divergence

**IQL (Implicit Q-Learning)** ✅
- Implemented but fails on this dataset
- Would work better with more action diversity

**Complexity**: Moderate - Q-network + V-network (IQL) or CQL penalty term.

**Implementation sketch (IQL)**:
```python
# Expectile loss - asymmetric, penalizes overestimation
def expectile_loss(diff, tau=0.7):
    weight = torch.where(diff > 0, tau, 1 - tau)
    return weight * (diff ** 2)

# V-network: learns to underestimate Q
v_pred = V(states)
q_values = Q(states, actions)
v_loss = expectile_loss(q_values - v_pred, tau=0.7).mean()

# Q-network: uses V as target (not max Q)
v_next = V(next_states)
td_target = rewards + gamma * v_next  # Conservative!
q_loss = ((Q(states, actions) - td_target) ** 2).mean()
```

---

### Phase 4: Online RL

**Approach**: Policy runs live, collects its own experience, optimizes for outcomes.

**Key insight**: Offline RL is limited by the data distribution. Online RL can explore.

**Methods**:

**REINFORCE / Policy Gradient**
- Simplest online method
- High variance, needs many samples
- Good starting point

**PPO (Proximal Policy Optimization)**
- More stable than vanilla policy gradient
- Industry standard for continuous control
- Clipped objective prevents catastrophic updates

**A2C/A3C (Advantage Actor-Critic)**
- Baseline reduces variance
- Can be parallelized (A3C)
- Good middle ground

**Recommended**: Start with REINFORCE for simplicity, upgrade to PPO if variance is an issue.

**Ceiling**: Can discover strategies the heuristic never tried.

**Complexity**: High - need online episode generation, careful hyperparameter tuning.

---

### Phase 5: Self-Play / Curriculum

**Approach**: Policy generates its own training curriculum, progressively harder.

**Concepts**:

**Self-play**: Policy trains against itself or past versions.
- Less relevant for our single-agent setting
- Could be relevant if we had adversarial dynamics

**Curriculum learning**: Start with easy problems, increase difficulty.
- Start with short episodes, easy datasets
- Progress to longer episodes, harder datasets (CIFAR-100, ImageNet)
- Policy learns foundational skills, then transfers

**Automatic curriculum**: Policy or meta-controller selects training problems.
- "Where am I weakest? Train there."
- Requires meta-learning infrastructure

**Recommended**: Manual curriculum first (CIFAR-10 → CIFAR-100 → harder), automate later.

**Ceiling**: Full autonomy - Tamiyo improves herself indefinitely.

**Complexity**: Very high - need infrastructure for curriculum management.

---

## Practical Recommendations

### Immediate Next Step (after overnight run)

Implement **Phase 2, Option C** - filter to good episodes:

```python
def load_good_episodes(data_dir, threshold=75.0):
    dm = DatasetManager(data_dir)
    episodes = []
    for ep_id in dm.list_episodes():
        ep = dm.load_episode(ep_id)
        if ep.final_accuracy >= threshold:
            episodes.append(ep)
    return episodes
```

This is ~10 lines of code and immediately biases toward successful patterns.

### Medium-term (after multi-slot)

Implement **Phase 3 (CQL)** once we have:
- Multi-slot architecture generating richer data
- More action diversity (ADVANCE, CULL)
- Enough episodes to have both good and bad outcomes

### Long-term

**Phase 4 (Online RL)** when we're ready to let Tamiyo explore autonomously. This requires:
- Confidence the policy won't catastrophically fail
- Compute budget for exploration
- Good reward shaping to guide exploration

---

## Reward Shaping Considerations

The reward signal matters enormously for RL. Current reward:

```python
reward = accuracy_change * 10
```

This is sparse and delayed. Consider:

### Insights from yzmir-deep-rl:reward-shaping-engineering

**Episode length rule of thumb**:
- < 20 steps → sparse OK
- 20-50 steps → borderline
- > 50 steps → need shaping

Our episodes are 25-75 epochs, so we're in "need shaping" territory.

**Potential-based shaping** (provably doesn't change optimal policy):
```python
# The theorem: F(s,a,s') = γ * Φ(s') - Φ(s)
# Adding this to reward preserves optimal policy

# For Tamiyo, potential = best accuracy so far
def potential(state):
    return state.best_val_accuracy

# Shaped reward
gamma = 0.99
shaping = gamma * potential(next_state) - potential(state)
total_reward = accuracy_change * 10 + shaping
```

This rewards PROGRESS toward higher accuracy, not just the moment it improves.

**Anti-hacking considerations**:
- "Spam GERMINATE" hack: destabilizes training, negative reward from accuracy drops
- "Never GERMINATE" hack: misses plateau opportunities, no positive reward
- Soft-consequences design should help: invalid actions = wasted turns, no reward

**Denser auxiliary rewards** (use sparingly):
- Small positive for taking valid actions (+0.1)
- Small negative for invalid actions (-0.1) if we enable this later
- Shaping based on intermediate metrics (loss improvement, plateau reduction)

**Validation checklist** (from reward-shaping skill):
- [ ] Task reward clearly specifies success
- [ ] Reward can't be exploited by shortcuts
- [ ] Using potential-based formula F = γΦ(s') - Φ(s)
- [ ] Test on distribution shift (different seeds, model inits)
- [ ] Behavioral inspection (is Tamiyo doing what we expect?)
- [ ] Training stability across runs

**Caution**: Over-shaped rewards can lead to reward hacking. Start simple, add shaping only if learning is too slow.

---

## Success Metrics

| Phase | Metric | Target |
|-------|--------|--------|
| 1 (Imitation) | Accuracy vs heuristic | 90%+ |
| 2 (Weighted) | Final episode accuracy | Higher mean than Phase 1 |
| 3 (Offline RL) | Final episode accuracy | Exceed heuristic outcomes |
| 4 (Online RL) | Final episode accuracy | Discover novel strategies |
| 5 (Curriculum) | Transfer to harder tasks | CIFAR-100 without retraining |

---

## Dependencies

- **Phase 2**: Just needs current data
- **Phase 3**: Needs outcome diversity (good and bad episodes)
- **Phase 4**: Needs stable Phase 3 policy as initialization
- **Phase 5**: Needs Phase 4 working reliably

---

## References

- CQL: "Conservative Q-Learning for Offline Reinforcement Learning" (Kumar et al., 2020)
- Decision Transformer: "Decision Transformer: Reinforcement Learning via Sequence Modeling" (Chen et al., 2021)
- PPO: "Proximal Policy Optimization Algorithms" (Schulman et al., 2017)
- IQL: "Offline Reinforcement Learning with Implicit Q-Learning" (Kostrikov et al., 2021)
