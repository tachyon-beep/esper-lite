# Esper: Morphogenetic Neural Networks

**Grow capabilities, don't just train weights.**

Esper is a framework for **Morphogenetic AI**—neural networks that dynamically grow, prune, and adapt their own topology during training. Instead of a static architecture, Esper uses a lifecycle-driven approach where "seed" modules are germinated in isolation, trained on residuals, and carefully grafted into a stable host model only when they prove their worth.

## 🚀 Key Features

* **🛡️ Gradient Isolation:** Seeds train in an "incubator" state, learning from the host's errors without destabilizing its existing knowledge (catastrophic forgetting prevention).
* **🧠 Dual-Mode Control:**
  * **Tamiyo (Heuristic):** A rule-based baseline controller for stable, predictable growth.
  * **Simic (RL):** A PPO reinforcement learning agent that learns optimal growth strategies by observing training dynamics.
* **⚡ Vectorized Training:** High-performance, multi-GPU RL environment using CUDA streams and inverted control flow for massive parallel throughput.
* **🔍 Rich Telemetry:** The **Nissa** subsystem provides profile-based diagnostics (gradient health, loss landscape sharpness) for deep debugging.

---

## 🏗️ Architecture

The system is organized into seven decoupled domains:

| Domain | Biological Role | Description |
| :--- | :--- | :--- |
| **Kasmina** | Stem Cells | Pluripotent slots that differentiate into neural modules. Manages grafting mechanics. |
| **Leyline** | DNA/Genome | Shared data contracts, enums (`SeedStage`), tensor schemas — the genetic code. |
| **Tamiyo** | Brain/Cortex | Strategic decision-making logic (heuristic or neural policy). |
| **Tolaria** | Metabolism | Execution engine that converts data into trained weights (energy conversion). |
| **Simic** | Evolution | RL infrastructure (PPO) enabling adaptation through selection pressure. |
| **Nissa** | Sensory Organs | Observability hub — perceives training dynamics and routes telemetry. |
| **Karn** | Memory | Research telemetry system with analytics, health monitoring, TUI, and web dashboard. |

> **📝 Metaphor Note:** Esper uses *body/organism* terminology for system architecture (domains as organs) and *botanical* terminology for seed lifecycle (germinate, graft, cull). Think of it as: "The organism's stem cells undergo a botanical development process."

---

## ⚡ Quick Start

### 1. Installation

Requires Python 3.11+ and PyTorch.

```bash
# Clone and setup
git clone https://github.com/yourusername/esper.git
cd esper

# Recommended: use uv
uv sync
```

### 2\. Run a Heuristic Baseline

Train a CIFAR-10 model where `Tamiyo` (the rule-based system) manages the growth.

```bash
PYTHONPATH=src uv run python -m esper.scripts.train heuristic --task cifar10 --episodes 1
```

### 3\. Train the Brain (Reinforcement Learning)

Train the **Simic** agent using PPO to discover better growth strategies than the heuristic.

```bash
PYTHONPATH=src uv run python -m esper.scripts.train ppo \
    --task cifar10 \
    --episodes 100 \
    --n-envs 4 \
    --device cuda:0 \
    --max-epochs 25 \
    --entropy-coef 0.05
```

-----

## 🌱 The Seed Lifecycle

Esper treats neural modules like living organisms. They must earn their place in the network.

```mermaid
stateDiagram-v2
    [*] --> DORMANT
    DORMANT --> GERMINATED: Germinate Action
    GERMINATED --> TRAINING: Advance (G1)
    TRAINING --> BLENDING: Advance (G2)
    TRAINING --> PRUNED: Prune (Performance Drop)
    BLENDING --> HOLDING: Advance (G3)
    BLENDING --> PRUNED: Prune (Regression)
    HOLDING --> FOSSILIZED: Fossilize (Stability Check)
    FOSSILIZED --> [*]: Terminal Success
    PRUNED --> EMBARGOED: Cleanup
    EMBARGOED --> RESETTING: Cooldown Complete
    RESETTING --> DORMANT: Slot Recycled
```

1. **Germinated:** Module created. Input connected, output detached.
2. **Training:** Module trains on host errors. Host weights frozen relative to this path.
3. **Blending:** Module output is alpha-blended into host stream.
4. **Holding:** Full-amplitude hold (alpha≈1.0). Stable decision point for fossilization.
5. **Fossilized:** Weights permanently integrated. Module becomes part of the "Host" for future seeds.

-----

## 📊 Results (POC)

Preliminary results on CIFAR-10 (ResNet-style Host):

| Approach | Final Accuracy | Notes |
| :--- | :--- | :--- |
| **Static Baseline** | 69.31% | Standard training, no growth. |
| **From-Scratch** | 65.97% | Re-initializing larger model (poor convergence). |
| **Esper (Heuristic)** | **82.16%** | Staged growth managed by Tamiyo. |
| **Esper (PPO)** | *Training...* | Learning to optimize the fossilization timing. |

-----

## 🛠️ Development

**Project Structure:**

```text
src/esper/
├── kasmina/      # Model & Slot mechanics
├── leyline/      # Shared types & contracts
├── tamiyo/       # Decision logic
├── tolaria/      # PyTorch training loops
├── simic/        # RL Algorithms (PPO)
├── nissa/        # Telemetry & Logging
├── karn/         # Research telemetry (TUI, dashboard, analytics)
└── scripts/      # CLI Entry points
```

**Run Tests:**

```bash
uv run pytest -q
```

---

## 📖 CLI Reference

### PPO Training (`esper.scripts.train ppo`)

Train a PPO agent to learn optimal seed lifecycle management.

```bash
PYTHONPATH=src python -m esper.scripts.train ppo [OPTIONS]
```

#### Config-first workflow

All PPO hyperparameters live in `TrainingConfig` (JSON-loadable). CLI flags are
limited to picking a preset and runtime wiring:

| Flag | Default | Description |
|------|---------|-------------|
| `--preset` | `cifar10` | Hyperparameter preset: `cifar10`, `cifar10_stable`, `cifar10_deep`, `cifar10_blind`, `tinystories` |
| `--config-json` | (none) | Path to JSON config (strict: unknown keys fail) |
| `--task` | `cifar10` | Task preset for dataloaders/topology |
| `--device` | `cuda:0` | Primary compute device |
| `--devices` | (none) | Multi-GPU devices (e.g., `cuda:0 cuda:1`) |
| `--amp` | off | Enable AMP (CUDA only) |
| `--num-workers` | (task default) | DataLoader workers per environment |
| `--gpu-preload` | off | Preload dataset to GPU (CIFAR-10 only) |
| `--seed` | (config default) | Override run seed |

#### Hardware & Performance

| Flag | Default | Description |
|------|---------|-------------|
| `--device` | `cuda:0` | Primary compute device |
| `--devices` | (none) | Multi-GPU devices (e.g., `cuda:0 cuda:1`) |
| `--num-workers` | (task default) | DataLoader workers per environment |
| `--gpu-preload` | off | Preload dataset to GPU (CIFAR-10 only, ~0.75GB VRAM) |

#### Checkpointing

| Flag | Default | Description |
|------|---------|-------------|
| `--save` | (none) | Path to save model checkpoint |
| `--resume` | (none) | Path to checkpoint to resume from |

#### Telemetry & Monitoring

| Flag | Default | Description |
|------|---------|-------------|
| `--telemetry-file` | (none) | Save telemetry to JSONL file |
| `--telemetry-dir` | (none) | Save telemetry to timestamped folder |
| `--telemetry-level` | `normal` | Verbosity: `off`, `minimal`, `normal`, `debug` |
| `--telemetry-lifecycle-only` | off | Keep lightweight seed lifecycle telemetry even when ops telemetry is disabled |
| `--no-tui` | off | Disable Rich terminal UI (uses console output instead) |
| `--overwatch` | off | Launch Overwatch TUI for real-time monitoring (replaces Rich TUI) |
| `--sanctum` | off | Launch Sanctum TUI for developer debugging (replaces Rich TUI) |
| `--dashboard` | off | Enable real-time WebSocket dashboard (requires `pip install esper-lite[dashboard]`) |
| `--dashboard-port` | 8000 | Dashboard server port |

**Monitoring Interfaces:**
- **Rich TUI (default)**: Full-screen terminal dashboard showing rewards, policy health (entropy, clip fraction, explained variance, KL divergence), seed states, action distribution, reward components, and losses. Disable with `--no-tui`.
- **`--overwatch` / `--sanctum`**: Textual TUIs for monitoring and developer debugging (mutually exclusive).
- **`--dashboard`**: Web-based dashboard accessible at `http://localhost:8000`. Listens on all network interfaces for remote access (e.g., `http://192.168.1.x:8000` on LAN). Displays clickable links for all available interfaces on startup.

### Heuristic Training (`esper.scripts.train heuristic`)

Run the rule-based Tamiyo controller as a baseline.

```bash
PYTHONPATH=src python -m esper.scripts.train heuristic [OPTIONS]
```

| Flag | Default | Description |
|------|---------|-------------|
| `--episodes` | 1 | Number of episodes |
| `--max-epochs` | 75 | Maximum epochs per episode |
| `--max-batches` | 50 | Batches per epoch (0=all) |
| `--task` | `cifar10` | Task preset |
| `--device` | `cuda:0` | Compute device |
| `--seed` | 42 | Random seed |
| `--slots` | `r0c0 r0c1 r0c2` | Canonical slot IDs to enable (e.g., `r0c0 r0c1 r0c2`) |
| `--max-seeds` | unlimited | Maximum total seeds |
| `--min-fossilize-improvement` | (task default) | Min improvement (%) required to fossilize a seed |

Telemetry flags (`--telemetry-file`, `--telemetry-dir`, `--telemetry-level`) are also available.

### Example Commands

```bash
# CIFAR-10 preset (default hyperparameters)
PYTHONPATH=src python -m esper.scripts.train ppo --preset cifar10 --task cifar10

# CIFAR-10 stable preset (slower, more reliable PPO updates)
PYTHONPATH=src python -m esper.scripts.train ppo --preset cifar10_stable --task cifar10

# Tinystories preset with AMP
PYTHONPATH=src python -m esper.scripts.train ppo \
    --preset tinystories \
    --task tinystories \
    --amp

# Multi-GPU training (deep CIFAR)
PYTHONPATH=src python -m esper.scripts.train ppo \
    --preset cifar10_deep \
    --task cifar10_deep \
    --devices cuda:0 cuda:1

# Load a strict JSON config
PYTHONPATH=src python -m esper.scripts.train ppo \
    --config-json configs/ppo_config.json \
    --task cifar10

# Training with web dashboard (accessible from browser/remote)
PYTHONPATH=src python -m esper.scripts.train ppo \
    --preset cifar10 \
    --dashboard \
    --dashboard-port 8080
```
