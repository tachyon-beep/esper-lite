# Esper-Lite

Morphogenetic neural network training - growing and grafting new capabilities into models during training.

## Overview

Esper-Lite explores whether neural networks can be improved by "growing" new components (seeds) in isolation, then carefully grafting them into the host model. Think of it like neural network surgery: train a specialist module separately, then splice it in without disrupting what the model already knows.

### Core Components

- **Kasmina** - Seed lifecycle management. Handles gradient isolation, alpha blending, and stage transitions.
- **Tamiyo** - Strategic controller. Decides when to germinate seeds, advance them through stages, or cull underperformers.
- **Simic** - Policy learning. Trains Tamiyo to make better decisions by observing outcomes.
- **Leyline** - Shared contracts. Data definitions for inter-component communication.

### Seed Lifecycle

```
DORMANT → GERMINATED → TRAINING → BLENDING → FOSSILIZED
                ↓           ↓          ↓
              CULLED ←──────┴──────────┘
```

1. **Dormant**: Empty slot waiting for a seed
2. **Germinated**: Seed attached, ready to train
3. **Training**: Isolated training with gradient isolation from host
4. **Blending**: Alpha-managed grafting into host model
5. **Fossilized**: Permanently integrated (success)
6. **Culled**: Removed due to poor performance (failure)

## Quick Start

```bash
# Setup
python -m venv .venv
source .venv/bin/activate
pip install torch torchvision

# Run the proof-of-concept (validates core premise)
PYTHONPATH=src .venv/bin/python src/esper/poc.py

# Run Tamiyo-driven training (reactive seed management)
PYTHONPATH=src .venv/bin/python src/esper/poc_tamiyo.py
```

## Policy Learning (Simic)

Train a neural network to imitate Tamiyo's decisions:

```bash
# Generate episodes and train policy
PYTHONPATH=src .venv/bin/python src/esper/simic_overnight.py --episodes 50

# Just train on existing data
PYTHONPATH=src .venv/bin/python src/esper/simic_overnight.py --train-only

# Run live comparison (heuristic vs learned policy)
PYTHONPATH=src .venv/bin/python src/esper/simic_overnight.py --compare
```

## Project Structure

```
src/esper/
├── leyline.py          # Shared contracts (stages, commands, reports)
├── kasmina.py          # Seed management (slots, blueprints, blending)
├── tamiyo.py           # Strategic controller (heuristic policy)
├── simic.py            # Policy learning (data collection, training)
├── poc.py              # Proof-of-concept (fixed schedule)
├── poc_tamiyo.py       # Tamiyo-driven training
└── simic_overnight.py  # Batch training runner

docs/plans/
└── 2025-01-25-policy-tamiyo-design.md  # Current design doc
```

## Results

### POC Results (CIFAR-10)

| Approach | Final Accuracy |
|----------|----------------|
| Baseline (no seeds) | 69.31% |
| Morphogenetic (fixed schedule) | 80.16% |
| Morphogenetic (Tamiyo-driven) | 82.16% |
| From-scratch retraining | 65.97% |

Key finding: Staged training (isolate → blend → fossilize) outperforms both baseline and from-scratch approaches.

## Design Philosophy

- **Small testable stages** - Build incrementally, validate each step
- **Contracts first** - Define data shapes in leyline.py before implementation
- **Prove the premise** - POC before architecture
