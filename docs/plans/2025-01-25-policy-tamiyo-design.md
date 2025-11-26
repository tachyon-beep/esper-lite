# Policy Tamiyo Design

**Date**: 2025-01-25
**Status**: In Progress
**Goal**: Train a neural network to imitate Heuristic Tamiyo's decisions

## Context

Tamiyo is the strategic controller for seed lifecycle in Esper-Lite. Currently implemented as `HeuristicTamiyo` with hand-coded rules for:
- When to germinate seeds (plateau detection)
- When to advance seeds through stages
- When to cull underperforming seeds

**Why learn a policy?**
- Neural networks can detect patterns humans can't see in training telemetry
- A learned policy could generalize beyond the specific heuristics
- Foundation for more sophisticated interventions beyond seed grafting

## Design Decisions

### Approach: Imitation Learning (Phase 1)

We chose imitation learning over reward optimization because:
1. Simpler to validate - just match Heuristic Tamiyo's decisions
2. Needs less data - supervised learning, not RL
3. Proves the pipeline works before getting fancy

Future phases can fine-tune toward outcome optimization.

### Architecture: Simple MLP

```
Input (27) → Linear(64) → ReLU → Linear(32) → ReLU → Linear(4) → Softmax
```

- ~3K parameters
- Input: `TrainingSnapshot.to_vector()` (27 floats)
- Output: probability distribution over 4 actions (WAIT, GERMINATE, ADVANCE, CULL)

Why MLP over recurrent:
- 27-dim input, 4 classes is trivially small
- Start simple, add complexity if it underfits
- LSTM/GRU planned as medium-term upgrade for harder problems

### Training Setup

- **Loss**: CrossEntropyLoss (standard multi-class classification)
- **Optimizer**: Adam, lr=0.001
- **Batching**: Shuffle decision points, standard mini-batches
- **Split**: 80/20 by episode (not by decision point, to avoid leakage)

Class imbalance (WAIT dominates) may be an issue - will add class weights if needed.

### Validation

Two levels:
1. **Fast metrics** (always): Accuracy + confusion matrix
2. **Live comparison** (optional): Run learned policy alongside heuristic, compare decisions

Fast metrics for rented hardware, live comparison for local validation.

## Observation Space (27 dimensions)

From `TrainingSnapshot`:

| Field | Description |
|-------|-------------|
| epoch | Current training epoch |
| global_step | Total training steps |
| train_loss | Training loss this epoch |
| val_loss | Validation loss |
| loss_delta | Change from previous epoch |
| train_accuracy | Training accuracy (%) |
| val_accuracy | Validation accuracy (%) |
| accuracy_delta | Change from previous epoch |
| plateau_epochs | Epochs without improvement |
| best_val_accuracy | Best seen so far |
| best_val_loss | Best loss seen |
| loss_history_5 | Last 5 epochs of loss (5 dims) |
| accuracy_history_5 | Last 5 epochs of accuracy (5 dims) |
| has_active_seed | Whether a seed is active |
| seed_stage | Current seed stage (as int) |
| seed_epochs_in_stage | Epochs in current stage |
| seed_alpha | Current blending alpha |
| seed_improvement | Seed's improvement since stage start |
| available_slots | Open seed slots |

## Action Space (4 discrete)

| Action | Description |
|--------|-------------|
| WAIT | Do nothing this epoch |
| GERMINATE | Start a new seed |
| ADVANCE | Move seed to next stage |
| CULL | Kill underperforming seed |

## Data Pipeline

### Collection (implemented)

```python
from esper.simic import EpisodeCollector, DatasetManager

collector = EpisodeCollector()
collector.start_episode(episode_id, max_epochs)

# During training loop:
collector.record_observation(snapshot)
collector.record_action(action)
collector.record_outcome(outcome)

episode = collector.end_episode(final_accuracy, best_accuracy)

# Save to disk
dm = DatasetManager("data/simic_episodes")
dm.save_episode(episode)
```

### Training Data Prep

```python
episodes = dm.load_all()

# Split by episode to avoid leakage
train_eps = episodes[:int(0.8 * len(episodes))]
val_eps = episodes[int(0.8 * len(episodes)):]

# Flatten to (X, y)
X_train = [obs for ep in train_eps for obs, _, _ in ep.to_training_data()]
y_train = [action.index(1.0) for ep in train_eps for _, action, _ in ep.to_training_data()]
```

## Current Implementation Status

### Completed

- [x] `simic.py` - Data structures, persistence, and policy network
  - `TrainingSnapshot` - 27-dim observation with serialization
  - `ActionTaken` - Action record with serialization
  - `StepOutcome` - Outcome + reward
  - `Episode` - Full trajectory with `save()`/`load()`
  - `EpisodeCollector` - Captures data during training
  - `DatasetManager` - Directory of JSON episodes
  - `PolicyNetwork` - MLP (27→64→32→4) with train/evaluate/predict
  - `print_confusion_matrix` - Pretty-print evaluation results

- [x] `simic_overnight.py` - Full training pipeline
  - Episode generation with Heuristic Tamiyo
  - Policy training with train/val split by episode
  - Evaluation with confusion matrix
  - Live comparison mode (heuristic vs learned, side-by-side)

- [x] `simic_test_collection.py` - Verified collection pipeline

### Pending

- [ ] Run overnight with 50+ episodes
- [ ] Analyze results and iterate on class imbalance if needed

## Test Results

### Collection Test (2025-01-25)

```
Episode ID: test_collection_001
Decision points collected: 15
Final accuracy: 75.73%
Best accuracy: 75.75%
Total reward: 757.30

Training data tuples: 15
First tuple: obs_len=27, act_len=5, reward=541.70
```

### Persistence Test (2025-01-25)

```
Original: test_persist_001, 3 decisions
Loaded:   test_persist_001, 3 decisions
Episodes in dataset: ['test_persist_001']
Training data tuples: 3
Summary: {count: 1, total_decisions: 3, avg_final_accuracy: 65.0}
```

## Next Steps

1. ~~Implement `PolicyNetwork` class in `simic.py`~~ ✓
2. ~~Add training loop~~ ✓
3. Run overnight: `PYTHONPATH=src .venv/bin/python src/esper/simic_overnight.py --episodes 50`
4. Analyze confusion matrix - check for class imbalance issues
5. If GERMINATE/ADVANCE accuracy is low, add class weights
6. Consider LSTM/GRU upgrade for harder problems
