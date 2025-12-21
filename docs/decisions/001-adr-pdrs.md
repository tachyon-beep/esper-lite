# ADR: Use Approximate PBRS Shaping For Simic Rewards (Not Strict PBRS)

**Status**: Accepted
**Date**: 2025-12-01
**System**: Esper / Simic (RL controller for Tamiyo)
**Decision owner**: You, future you, and annoyed future-you

---

## 1. Context

Simic’s reward function `compute_shaped_reward` is currently:

* Accuracy delta primary (`acc_delta * weight`)
* Plus stage bonuses (TRAINING / BLENDING / FOSSILIZED)
* Plus action shaping (GERMINATE, FOSSILIZE, CULL, WAIT)
* Plus compute rent based on extra parameters

Transformer / TinyStories training exposed a failure mode:

* BLENDING often causes a short term drop in accuracy.
* Because the reward is dominated by instantaneous `acc_delta`, the agent learns:

  * BLENDING is painful
  * WAIT is safe
  * So it does as little as possible.

We already have potential based reward shaping (PBRS) helpers:

* `compute_seed_potential(obs: dict) -> float`
* `compute_pbrs_bonus(potential_prev, potential_next, gamma=0.99)`
* Monotone stage potentials for the seed lifecycle

However:

* These PBRS helpers are used only in the **loss based** reward (`compute_loss_reward`).
* The **accuracy based** reward (`compute_shaped_reward`) that Simic actually uses in vectorised PPO does not call them at all.

At the same time:

* The environment is not a nice fixed MDP.
* Actions change the architecture (grafting seeds, adding params).
* We already include non PBRS shaping (stage bonuses, action specific shaping, rent penalties).

So strict Ng style PBRS guarantees (“shaping never changes the optimal policy”) do not strictly apply to this system as implemented.

We want to:

* Give the agent a strong positive signal for advancing seed stages (TRAINING → BLENDING → SHADOWING → HOLDING → FOSSILIZED).
* Avoid the “do nothing” local optimum caused by punishing short term accuracy dips.
* Keep the reward logic understandable to future us.

---

## 2. Decision

We will:

1. **Integrate PBRS style seed stage shaping into `compute_shaped_reward`, but in an approximate way**, using `SeedInfo` and `compute_seed_potential`.

2. **Accept that this shaping is “PBRS flavoured” rather than mathematically strict PBRS**, and that in theory it can shift the optimal policy, while in practice it makes the agent actually learn useful behaviour.

In code terms, inside `compute_shaped_reward` in `src/esper/simic/rewards.py` we will:

* Keep the existing base reward, stage bonuses, action shaping, and terminal bonus.
* Insert a PBRS style block before action specific shaping:

```python
    # Potential based shaping for seed stages (approximate PBRS)
    if seed_info is not None:
        current_obs = {
            "has_active_seed": 1,
            "seed_stage": seed_info.stage,
            "seed_epochs_in_stage": seed_info.epochs_in_stage,
        }
        prev_obs = {
            "has_active_seed": 1,
            "seed_stage": getattr(seed_info, "previous_stage", seed_info.stage),
            "seed_epochs_in_stage": max(0, seed_info.epochs_in_stage - 1),
        }

        phi_prev = compute_seed_potential(prev_obs)
        phi_curr = compute_seed_potential(current_obs)

        # PBRS style term: r' = r + gamma * Phi(s') - Phi(s)
        reward += compute_pbrs_bonus(phi_prev, phi_curr, gamma=0.99)
```

Confidence that this is the right tradeoff: **High (likely correct 80–85 percent)**.

---

## 3. Rationale

### 3.1 Why not strict PBRS?

Textbook PBRS (Ng et al. 1999) says:

* If you shape rewards as
  `F(s, s') = gamma * Phi(s') - Phi(s)`
  and everything else is clean, then the optimal policy under the shaped reward is unchanged.

To get that guarantee here, we would need:

* A Markov environment with fixed dynamics.
* Reward equal to `base_reward(s,a,s') + F(s,s')` only.
* A potential function `Phi` that depends only on the full environment state.

In Esper:

* The environment is non stationary:

  * Actions change the network architecture (new params, new behaviour).
  * Those changes alter how future rewards look.
* `compute_shaped_reward` already includes:

  * Stage bonuses.
  * Action specific shaping.
  * Compute rent.
* `SeedInfo` is a minimal view of the state:

  * We do not include optimiser state, host weights, or full telemetry in `Phi`.

So even if the PBRS term itself were perfectly strict, the overall reward already breaks the letter of the theorem.

Conclusion:

* We cannot honestly claim “strict PBRS policy invariance” for Simic in this system.
* Trying to force everything to obey strict PBRS would add complexity without actually restoring the theoretical guarantee.

### 3.2 Why add approximate PBRS anyway?

Even without the theorem, the PBRS idea is still useful as a design pattern:

* Encode “future potential” of states as a scalar `Phi(s)`:

  * Later stages in the seed lifecycle get higher potential.
  * More time spent in a stage increases potential slightly.
* Add a time discounted difference `gamma * Phi(s') - Phi(s)`:

  * Advancing the lifecycle produces a positive bump.
  * Staying still produces little or no extra reward.

In practice this:

* Gives a **big positive jolt** when a seed moves TRAINING → BLENDING → SHADOWING → … → FOSSILIZED.
* Offsets the short term accuracy dip during BLENDING that was previously interpreted as “this was a mistake”.
* Matches the semantics we actually want:

  * “Advancing a reasonably healthy seed is valuable, even if the next couple of epochs wobble a bit”.

The opportunity cost of trying to keep things strictly PBRS here would be:

* A weaker shaping signal.
* More book keeping about exact state transitions.
* Less room to encode our domain knowledge that stage progression is good.

Given our actual goals and environment, we prefer stronger, pragmatic shaping over a theoretical guarantee that does not really hold.

---

## 4. Alternatives Considered

### 4.1 Do nothing (no PBRS in `compute_shaped_reward`)

*Outcome observed*:

* Agent learns that any action that causes even temporary accuracy dips is bad.
* Finds a local optimum around “WAIT a lot, maybe GERMINATE cautiously, rarely BLEND”.
* On TinyStories / Transformer tasks this shows up as:

  * Poor use of seeds.
  * Worse performance than the heuristic baseline.

**Rejected** because it preserves a known failure mode.

### 4.2 Strict PBRS only, remove other shaping

Idea:

* Strip out stage bonuses and most action specific shaping.
* Use only strict PBRS with a carefully defined `Phi(s)`.

Problems:

* `Phi(s)` would have to encode almost all domain knowledge we currently have in stage and action shaping.
* Environment is still non stationary and architecture changing.
* Would create large engineering churn for limited theoretical gain.

**Rejected** as high complexity, low practical benefit.

### 4.3 Small PBRS term, leave main behaviour to accuracy

Idea:

* Add a very small PBRS term (low weight) on top of current reward.

Problem:

* The observed pathology is strong: BLENDING looks clearly negative on short term `acc_delta`.
* A small PBRS term is unlikely to overpower that without careful tuning.

**Rejected** as unlikely to materially change agent behaviour.

---

## 5. Consequences

### 5.1 Positive

* BLENDING and later lifecycle stages now carry a **strong, explicit positive signal** when the seed advances.
* The agent is less likely to get stuck in a local optimum of “never blend anything because it dips accuracy for three epochs”.
* Reward function now encodes the intent that:

  * “Lifecycle progression of a healthy seed is good, even at the cost of short term noise”.

### 5.2 Negative / Risks

* In principle, the shaped reward may now have a different optimal policy than the pure accuracy based reward:

  * Agent might slightly over value stage transitions.
  * Agent might favour more seeds / faster advancement than a purely accuracy driven optimum.
* We may need to tune:

  * Stage potentials in `compute_seed_potential`.
  * Discount `gamma` for the PBRS term.
  * Interaction with rent penalties.

### 5.3 Mitigations

* Keep stage potentials monotone in lifecycle order, with single direction transitions:

  * No TRAINING → BLENDING → TRAINING loops.
  * Makes reward hacking via cycles unlikely.
* Use empirical evaluation as the arbiter:

  * Compare RL policy against heuristic baseline on final TinyStories loss/perplexity and sample quality.
  * If RL beats heuristic on real metrics, we accept that as “good enough”, even if the strict PBRS guarantee is not satisfied.
* Keep PBRS code local and clearly documented:

  * The ADR you are reading.
  * The PBRS block in `compute_shaped_reward` with explicit comments.

---

## 6. Notes For Future You

If you open `rewards.py` in six months and mutter “why is there this random PBRS blob here?” this is the answer:

* It is there to **stop the RL agent being terrified of BLENDING** because short term accuracy dips.
* It is *not* mathematically strict PBRS in the Ng sense, and that is intentional.
* We accepted a small risk of shifting the theoretical optimum in exchange for a large gain in actually learning sensible lifecycle policies on transformers.

If you later:

* Switch TinyStories to use `compute_loss_reward` instead of `compute_shaped_reward`, or
* Rebalance the stage potentials,

please update this ADR or add a follow up ADR so there is a paper trail.
