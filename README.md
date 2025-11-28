# Esper

Morphogenetic neural network training: grow specialist seed modules, graft them into a host model, and keep the host stable while gaining new capabilities.

## What This Project Is

Esper tests staged growth for neural networks—train seeds in isolation, blend them into the host, and fossilize once proven. The goal is to outperform static architectures without destabilizing prior knowledge.

## Architecture Overview

- **Leyline**: Contracts and enums for stages, commands, telemetry, and tensor schemas.
- **Kasmina**: Seed mechanics and host models (blueprints, slots, isolation, blending).
- **Tamiyo**: Heuristic controller that issues commands based on training signals.
- **Simic**: RL policy that learns to improve Tamiyo (PPO/IQL); batch runner lives at `src/esper/simic_overnight.py`.
- **Nissa**: Optional telemetry profiles and outputs.
- Flow: training signals → Tamiyo/Simic action → Kasmina seed update → Leyline reports → back into signals. Keep tensor/vector sizes and stage enums consistent across layers.

## Quick Start

```bash
# Setup (Python 3.11+)
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
pip install torch torchvision   # choose CUDA/CPU build as needed

# Proof-of-concept runs
PYTHONPATH=src python src/esper/poc.py           # fixed schedule
PYTHONPATH=src python src/esper/poc_tamiyo.py    # Tamiyo-controlled

# Policy learning (Simic)
./scripts/train_ppo.sh -e 50 --single            # PPO; GPU recommended
PYTHONPATH=src python src/esper/simic_overnight.py --episodes 50

# Tests
PYTHONPATH=src pytest -q
```

## Seed Lifecycle

```
DORMANT → GERMINATED → TRAINING → BLENDING → SHADOWING → PROBATIONARY → FOSSILIZED
                ↓           ↓          ↓            ↓
              CULLED ←── EMBARGOED ← RESETTING ← DORMANT
```

- Growth path: dormant slot → germinate → isolate/train → blend → shadow → probation → fossilize (success).
- Recovery path: cull → embargoed cooldown → resetting cleanup → dormant reuse.

## Results (CIFAR-10 POC)

| Approach | Final Accuracy |
|----------|----------------|
| Baseline (no seeds) | 69.31% |
| Morphogenetic (fixed schedule) | 80.16% |
| Morphogenetic (Tamiyo-driven) | 82.16% |
| From-scratch retraining | 65.97% |

Key finding: staged growth (isolate → blend → fossilize) outperforms both baseline and from-scratch training.

## Design Principles

- Small, testable stages before full integration.
- Contracts first: define shapes/enums in Leyline, then implement.
- Preserve host competency while adding capabilities.
