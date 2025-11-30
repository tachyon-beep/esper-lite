# Case Study: Emergent Efficiency in Morphogenetic Networks

**Subtitle:** How the Esper Framework Independently Rediscovered Depthwise Separable Convolutions

## 1. Executive Summary

Esper is a morphogenetic training framework that lets an RL controller grow and prune modules during training. During the validation phase of the **Esper** framework on CIFAR-10, the reinforcement learning agent (**Tamiyo**) demonstrated an emergent ability to distinguish between neural architectures based on their "Reward Velocity"—the ratio of accuracy gain to training time.

In other words, Tamiyo was not just asking "Does this blueprint eventually help?" but implicitly asking "How quickly does this blueprint pay me back for the optimisation effort I spend on it?" That distinction only appears once the agent has enough telemetry to see beyond raw accuracy and into the dynamics of training itself.

Without explicit instruction or hardcoded biases, the agent independently "rediscovered" the efficiency principles of **Depthwise Separable Convolutions** (MobileNet [1]) when operating under implicit compute constraints. Under short training horizons and tight budgets, `depthwise` modules consistently emerged as the best trade-off between cost and reward. Under generous horizons and unconstrained compute, Tamiyo rationally switched to favour heavier `conv_enhance` blocks.

This behaviour validates the core hypothesis of the Esper architecture: that structural optimisation is a learnable policy given sufficient telemetry and a reward that reflects the underlying economics of training.

## 2. Experimental Setup

At a high level, the Esper stack combines a host wrapper, a training engine, an RL controller, a telemetry hub, and a small library of blueprints the controller can grow.

| Subsystem           | Role                      | Notes                                                                                                                                                 |
|---------------------|---------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Kasmina (Cells)** | Morphogenetic host wrapper around a task model | In this case study, the task model is a small CNN for CIFAR-10, but Kasmina can equally host transformers, MLPs, or other architectures. It provides `SeedSlot` injection points and manages per-seed lifecycle state such as `TRAINING`, `BLENDING`, and `FOSSILIZED`. |
| **Tolaria (Body)**  | Training execution engine | Runs host + seed forward/backward passes and applies optimisers. Exposes hooks for gradient isolation and blended training regimes.                  |
| **Simic (Gym)**     | RL gym and trainer        | Vectorised PPO environments using inverted control flow. Treats each training run as an RL episode where architectural decisions affect future rewards. |
| **Tamiyo (Brain)**  | Policy / architect        | Decides when to `GERMINATE`, `ADVANCE`, or `CULL` seeds. Receives a structured observation including accuracy deltas, gradient statistics, seed stage, and simple budget signals. |
| **Nissa (Senses)**  | Telemetry hub             | Collects and routes observability signals (accuracy deltas, gradient health, lifecycle metrics, budget signals) and feeds them into Simic/Tamiyo and analytics components. |

### **Blueprint library used in this case study**

- `conv_enhance`: Standard residual-style conv block (~74k params), heavy but expressive.  
- `depthwise`: Depthwise separable block (~4.8k params), lightweight but efficient.  
- `attention`: Squeeze-and-Excitation-style block (~2k params), cheap contextual reweighting.  

The agent was rewarded primarily for **Accuracy Delta** ($\Delta Acc$) and **Terminal Accuracy** in the initial experiments, with no explicit penalty for parameter count. Later iterations introduced a **Compute Rent** term, but the core phenomenon described here was visible even before rent was formalised.

## 3. The Phenomenon: Constraint-Driven Selection

We observed two distinct behavioural regimes based on the "Time Budget" (episode length) and the presence or absence of explicit compute costs. These regimes make the efficiency behaviour very visible: under tight budgets, depthwise seeds shine as fast, efficient upgrades, while under long budgets and free compute, heavy conv blocks naturally take over.

### Regime A: The "Short Horizon" (Implicit Constraints)

In early experimental runs restricted to **~25 epochs**, the agent developed a strong preference for the `depthwise` blueprint.

- **Behaviour:**

  - High germination rate for `depthwise` seeds.
  - Aggressive culling of `conv_enhance` seeds that failed to show quick improvements.
  - Occasional use of `attention` and `norm` as low-risk, low-reward tweaks.

- **Forensic Analysis:**

  - `conv_enhance` (~74k params) requires more epochs to converge. In a short run, its initial gradient noise and warm-up period outweighed its eventual capacity.
  - `depthwise` (~4.8k params) converges rapidly. It generates immediate, stable positive $\Delta Acc$ with much smaller variance.
  - Tamiyo therefore experienced conv blocks as "slow and risky" and depthwise blocks as "fast, reliable upgrades" in this regime.

- **Result:**

  - The agent effectively optimised for **Reward Velocity**: accuracy gain per unit of training time and optimisation effort.
  - It identified that `depthwise` modules provided the most reliable lift per epoch of investment, even though the reward function never referenced depthwise explicitly.

### Regime B: The "Long Horizon" (Unconstrained Capacity)

In production runs extended to **75–100 epochs**, the agent inverted its strategy.

- **Behaviour:**

  - Shifted preference to `conv_enhance` once enough training time was available.
  - Adopted a "monogamous" strategy of planting one heavy seed per environment and nurturing it to a stable blended state.
  - Dramatically reduced use of `CULL`; once a good conv seed was established, Tamiyo mostly issued `ADVANCE` and `WAIT`.

- **Forensic Analysis:**

  - Given sufficient time, the superior representational capacity of the dense convolution outweighed its setup cost.
  - In the absence of an explicit "Energy Tax" (Rent), adding more parameters was always a net positive once training converged.
  - The behaviour matched the objective: maximise terminal accuracy, regardless of capacity.

- **Result:**

  - In a long-horizon, cost-free environment, Tamiyo correctly identified that maximising parameters maximised terminal reward.
  - The same policy network that favoured depthwise under tight time budgets now rationally favoured larger conv blocks when the budget constraint was removed.

## 4. Architectural Validation

### What this tells us about Esper’s design

1. **The Signal-to-Noise Hypothesis**

   The agent could only make these distinctions because **Nissa (Senses)** provided clean, separated telemetry. Instead of a single opaque loss scalar, Nissa exposed:

   - Per-blueprint accuracy deltas and stage transitions.
   - Gradient variance and basic stability indicators.
   - Lightweight budget signals, such as parameter ratios and live seed counts.

   By correlating specific blueprint types with specific gradient variance and loss trajectories, the agent learned to discriminate between "Heavy/Slow" and "Light/Fast" architectures. This is direct evidence that Esper's observability stack is sufficient for structural reasoning.

2. **Structural Taste**

   The agent demonstrated that it is not simply memorising a fixed topology. It is **Domain Adaptive**:

   - Under short horizons, it behaves like a frugal optimiser, prioritising quick wins and low-risk modules.
   - Under long horizons, it behaves like a capacity-maximising architect, willing to invest in heavy modules that only pay off later.

   This shift in strategy, driven purely by boundary conditions and reward signals, is exactly what we mean by **structural taste**—a learned policy over architectures rather than a static, hand-designed network.

## 5. Implementation: The "Rent" Protocol

To operationalise this discovery and make the efficiency pressure explicit rather than accidental, we have introduced a formal **Compute Rent** mechanism into the `Simic` reward function.

$$R_{net} = R_{acc} - \lambda \cdot \left( \frac{P_{total}}{P_{host}} \right)$$

Where:

- $R_{acc}$ is the accuracy-based reward (accuracy deltas plus terminal accuracy).
- $P_{host}$ is the parameter count of the frozen host model.
- $P_{total}$ is the additional parameter mass contributed by fossilised seeds **plus** any currently active seed.
- $\lambda$ is a small rent coefficient (initially set conservatively) that ensures rent is noticeable but not dominant.

By penalising the parameter ratio ($P_{total} / P_{host}$), we explicitly encode the efficiency pressure that was only implicit in Regime A. This has several effects:

- It turns the "short horizon" behaviour into a first-class objective rather than an accident of episode length.
- It forces **Tamiyo** (Builder) and **Emrakul** (Reaper) to constantly evaluate whether a module's accuracy contribution pays for its metabolic cost.
- It nudges the system to default to efficient structures (Depthwise/Attention/Norm) unless the task genuinely demands the extra complexity of heavy conv blocks.

Over time, this should in principle produce architectures that are both high performing and lean, mirroring the human-driven evolution from VGG-style stacks to MobileNet-style efficient networks [1].

## 6. Conclusion

The Esper framework has proven it can learn architectural efficiency principles from first principles. The agent's preference for Depthwise Convolutions under constraint mirrors the human-driven evolution from VGG to MobileNet [1,2], providing high confidence that the system can generalise to new domains where the "right" architectural trade-offs are not known in advance.

Most importantly, this case study shows that **morphogenetic control is a learnable skill**: given the right telemetry and a reward that reflects both benefit and cost, an RL agent can discover and exploit structural regularities that were previously the domain of hand-designed architectures.

[1]: https://arxiv.org/pdf/1704.04861?utm_source=chatgpt.com "arXiv:1704.04861v1 [cs.CV] 17 Apr 2017"
