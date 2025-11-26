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

## Diverse Data Generation (Datagen)

Generate diverse offline RL training data with full coverage across environments and behavior policies:

```bash
# Preview generation plan (no training)
PYTHONPATH=src .venv/bin/python -m esper.datagen.generate --dry-run

# Generate skeleton episodes (fast, for testing)
PYTHONPATH=src .venv/bin/python -m esper.datagen.generate --skeleton --episodes-per-combo 2

# Full generation with training
PYTHONPATH=src .venv/bin/python -m esper.datagen.generate --episodes-per-combo 10

# Run health checks on existing data
PYTHONPATH=src .venv/bin/python -m esper.datagen.generate --health-check

# Generate for specific configurations
PYTHONPATH=src .venv/bin/python -m esper.datagen.generate \
    --env-ids baseline resnet34 \
    --policy-ids baseline aggressive random
```

### Generation Matrix

The system generates episodes across:
- **13 environment configs**: HostCNN, ResNet-18/34, various learning rates, batch sizes, optimizers
- **11 behavior policies**: baseline, aggressive, conservative, random, early/late interveners, etc.
- **Epsilon variants**: Configurable ε-greedy exploration for each policy

### Data Quality

Health checks ensure dataset quality for offline RL:
- **Action coverage**: All actions represented (≥2% each)
- **Action entropy**: Diverse action distribution
- **Policy diversity**: Multiple behavior policies represented
- **State-action coverage**: Actions vary across similar states

## Project Structure

```
src/esper/
├── leyline.py          # Shared contracts (stages, commands, reports)
├── kasmina.py          # Seed management (slots, blueprints, blending)
├── tamiyo.py           # Strategic controller (heuristic policy)
├── simic.py            # Policy learning (data collection, training)
├── simic_iql.py        # Offline RL training (IQL/CQL)
├── poc.py              # Proof-of-concept (fixed schedule)
├── poc_tamiyo.py       # Tamiyo-driven training
├── simic_overnight.py  # Batch training runner
└── datagen/            # Diverse data generation system
    ├── configs.py      # Environment and policy configurations
    ├── architectures.py # Model factory (HostCNN, ResNet variants)
    ├── policies.py     # Behavior policy wrapper with ε-greedy
    ├── health.py       # Dataset quality checks
    ├── orchestrator.py # Generation matrix and progress tracking
    └── generate.py     # Main CLI for data generation
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
