"""Simic Training Module - Phase 2: Reward-Weighted Imitation

This module implements Phase 2 of the Tamiyo improvement roadmap:
- Filter episodes by final accuracy (train on good outcomes)
- Optional episode-level weighting (bias toward better outcomes)
- Support for V1 (27 features) and V2 (~54+ features with telemetry)

Usage:
    # Train on good episodes only (Option C)
    PYTHONPATH=src .venv/bin/python -m esper.simic_train \
        --pack data/packs/simic_v2_research_2025-11-26.json \
        --threshold 75.0

    # Train with episode weighting (Option A)
    PYTHONPATH=src .venv/bin/python -m esper.simic_train \
        --pack data/packs/simic_v2_research_2025-11-26.json \
        --threshold 70.0 \
        --episode-weighting

    # Save trained model
    PYTHONPATH=src .venv/bin/python -m esper.simic_train \
        --pack data/packs/simic_v2_research_2025-11-26.json \
        --save models/tamiyo_phase2.pt
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
from pathlib import Path

import torch
import torch.nn as nn

from esper.simic import SimicAction


# =============================================================================
# Feature Extraction
# =============================================================================

def safe(v, default=0.0, max_val=100.0):
    """Safely convert value to float, handling None/inf/nan."""
    if v is None:
        return default
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return default
    return max(-max_val, min(float(v), max_val))


def obs_to_base_features(obs: dict) -> list[float]:
    """Extract V1-style base features (27 dims) from observation dict."""
    return [
        float(obs['epoch']),
        float(obs['global_step']),
        safe(obs['train_loss'], 10.0),
        safe(obs['val_loss'], 10.0),
        safe(obs['loss_delta'], 0.0),
        obs['train_accuracy'],
        obs['val_accuracy'],
        safe(obs['accuracy_delta'], 0.0),
        float(obs['plateau_epochs']),
        obs['best_val_accuracy'],
        safe(obs['best_val_loss'], 10.0),
        *[safe(v, 10.0) for v in obs['loss_history_5']],
        *obs['accuracy_history_5'],
        float(obs['has_active_seed']),
        float(obs['seed_stage']),
        float(obs['seed_epochs_in_stage']),
        obs['seed_alpha'],
        obs['seed_improvement'],
        float(obs['available_slots']),
    ]


def telemetry_to_features(telem: dict) -> list[float]:
    """Extract V2 telemetry features (27 dims) from telemetry snapshot."""
    features = []

    # Gradient health (5 features)
    gh = telem.get('gradient_health', {})
    features.extend([
        safe(gh.get('overall_norm', 0), 0, 10),
        safe(gh.get('norm_variance', 0), 0, 10),
        float(gh.get('vanishing_layers', 0)),
        float(gh.get('exploding_layers', 0)),
        safe(gh.get('health_score', 1), 1, 1),
    ])

    # Per-class accuracy (10 features) - sorted by class name
    pca = telem.get('per_class_accuracy', {})
    for i in range(10):
        features.append(safe(pca.get(str(i), 50), 50, 100))

    # Class variance (1 feature)
    features.append(safe(telem.get('class_variance', 0), 0, 1000))

    # Sharpness (1 feature)
    features.append(safe(telem.get('sharpness', 0), 0, 100))

    # Gradient stats per layer - just norms (7 features)
    gs = telem.get('gradient_stats', [])
    layer_norms = [safe(g.get('norm', 0), 0, 10) for g in gs[:7]]
    while len(layer_norms) < 7:
        layer_norms.append(0.0)
    features.extend(layer_norms)

    # Red flags as binary (3 features)
    rf = telem.get('red_flags', [])
    features.append(1.0 if 'severe_class_imbalance' in rf else 0.0)
    features.append(1.0 if 'sharp_minimum' in rf else 0.0)
    features.append(1.0 if 'gradient_issues' in rf else 0.0)

    return features  # 27 features


# =============================================================================
# Data Loading
# =============================================================================

def load_pack(path: Path) -> dict:
    """Load a data pack from JSON."""
    with open(path) as f:
        return json.load(f)


def extract_samples(
    pack: dict,
    threshold: float = 0.0,
    use_telemetry: bool = True,
    episode_weighting: bool = False,
) -> tuple[list[list[float]], list[int], list[float]]:
    """Extract (features, labels, weights) from a data pack.

    Args:
        pack: Data pack dictionary.
        threshold: Minimum final_accuracy to include episode.
        use_telemetry: Whether to include V2 telemetry features.
        episode_weighting: Whether to weight samples by episode accuracy.

    Returns:
        Tuple of (features, labels, weights).
    """
    episodes = pack['episodes']

    # Filter by threshold
    filtered = [ep for ep in episodes if ep['final_accuracy'] >= threshold]
    print(f"Filtered: {len(filtered)}/{len(episodes)} episodes above {threshold}% accuracy")

    # Compute episode weights if requested
    if episode_weighting:
        accuracies = [ep['final_accuracy'] for ep in filtered]
        baseline = min(accuracies)
        scale = max(accuracies) - baseline + 1e-6
        ep_weights = {
            ep['episode_id']: (ep['final_accuracy'] - baseline) / scale + 0.5
            for ep in filtered
        }
        print(f"Episode weights: {min(ep_weights.values()):.2f} - {max(ep_weights.values()):.2f}")

    X_all, y_all, w_all = [], [], []

    for ep in filtered:
        telem_hist = ep.get('telemetry_history', [])
        ep_weight = ep_weights.get(ep['episode_id'], 1.0) if episode_weighting else 1.0

        for decision in ep['decisions']:
            # Base features
            features = obs_to_base_features(decision['observation'])

            # Add telemetry if available and requested
            if use_telemetry and telem_hist:
                epoch = decision['observation']['epoch']
                if epoch <= len(telem_hist):
                    telem = telem_hist[epoch - 1]
                    features.extend(telemetry_to_features(telem))
                else:
                    features.extend([0.0] * 27)
            elif use_telemetry:
                features.extend([0.0] * 27)

            X_all.append(features)
            y_all.append(SimicAction[decision['action']['action']].value)
            w_all.append(ep_weight)

    return X_all, y_all, w_all


# =============================================================================
# Model
# =============================================================================

def create_model(input_dim: int, device: str) -> nn.Module:
    """Create policy network for given input dimension."""
    return nn.Sequential(
        nn.Linear(input_dim, 256),
        nn.ReLU(),
        nn.BatchNorm1d(256),
        nn.Dropout(0.3),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.BatchNorm1d(128),
        nn.Dropout(0.3),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, len(SimicAction)),
    ).to(device)


def compute_class_weights(y_train: torch.Tensor, beta: float = 0.999) -> torch.Tensor:
    """Compute class weights using effective number of samples."""
    class_counts = torch.bincount(y_train, minlength=len(SimicAction)).float()
    effective_num = 1.0 - torch.pow(beta, class_counts)
    weights = (1.0 - beta) / (effective_num + 1e-8)
    weights = weights / weights.sum() * len(SimicAction)
    return weights


# =============================================================================
# Training
# =============================================================================

def train(
    X_all: list[list[float]],
    y_all: list[int],
    w_all: list[float],
    device: str = "cuda:0",
    epochs: int = 300,
    patience: int = 40,
    val_split: float = 0.2,
) -> tuple[nn.Module, float]:
    """Train policy network with Phase 2 settings.

    Returns:
        Tuple of (best_model, best_balanced_accuracy).
    """
    # Shuffle and split
    combined = list(zip(X_all, y_all, w_all))
    random.seed(42)
    random.shuffle(combined)
    X_all, y_all, w_all = zip(*combined)

    split = int((1 - val_split) * len(X_all))
    X_train = torch.tensor(X_all[:split], dtype=torch.float32, device=device)
    y_train = torch.tensor(y_all[:split], dtype=torch.long, device=device)
    w_train = torch.tensor(w_all[:split], dtype=torch.float32, device=device)
    X_val = torch.tensor(X_all[split:], dtype=torch.float32, device=device)
    y_val = torch.tensor(y_all[split:], dtype=torch.long, device=device)

    print(f"Train: {len(X_train)}, Val: {len(X_val)}, Features: {X_train.shape[1]}")

    # Model
    model = create_model(X_train.shape[1], device)

    # Class weighting
    class_weights = compute_class_weights(y_train).to(device)
    print(f"Class weights: {dict(zip([a.name for a in SimicAction], [f'{w:.2f}' for w in class_weights.tolist()]))}")

    criterion = nn.CrossEntropyLoss(weight=class_weights, reduction='none')
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=15, factor=0.5)

    best_balanced_acc = 0.0
    best_model = None
    patience_counter = 0

    for epoch in range(epochs):
        # Training
        model.train()
        indices = torch.randperm(len(X_train))
        for i in range(0, len(X_train), 64):
            batch_idx = indices[i:i + 64]
            optimizer.zero_grad()
            outputs = model(X_train[batch_idx])
            losses = criterion(outputs, y_train[batch_idx])
            # Apply episode weights
            weighted_loss = (losses * w_train[batch_idx]).mean()
            weighted_loss.backward()
            optimizer.step()

        # Validation
        model.eval()
        with torch.no_grad():
            val_preds = model(X_val).argmax(dim=1)
            per_class_acc = []
            for action in SimicAction:
                mask = (y_val == action.value)
                if mask.sum() > 0:
                    acc = ((val_preds == action.value) & mask).sum().float() / mask.sum()
                    per_class_acc.append(acc.item())
            balanced_acc = sum(per_class_acc) / len(per_class_acc)

        scheduler.step(-balanced_acc)

        if balanced_acc > best_balanced_acc:
            best_balanced_acc = balanced_acc
            best_model = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1

        if epoch % 50 == 0:
            print(f"Epoch {epoch:3d}: balanced_acc={balanced_acc*100:.1f}% (best={best_balanced_acc*100:.1f}%)")

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

    # Load best model
    model.load_state_dict(best_model)
    return model, best_balanced_acc


def evaluate(model: nn.Module, X_val: torch.Tensor, y_val: torch.Tensor) -> dict:
    """Evaluate model and return per-class metrics."""
    model.eval()
    with torch.no_grad():
        preds = model(X_val).argmax(dim=1)

    results = {}
    for action in SimicAction:
        mask = (y_val == action.value)
        if mask.sum() > 0:
            correct = ((preds == action.value) & mask).sum().item()
            total = mask.sum().item()
            results[action.name] = {'correct': correct, 'total': total, 'accuracy': correct / total}

    overall = (preds == y_val).float().mean().item()
    results['overall'] = overall
    return results


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Train Tamiyo Phase 2")
    parser.add_argument("--pack", required=True, help="Path to data pack JSON")
    parser.add_argument("--threshold", type=float, default=0.0, help="Min final accuracy to include")
    parser.add_argument("--episode-weighting", action="store_true", help="Weight samples by episode accuracy")
    parser.add_argument("--no-telemetry", action="store_true", help="Use only base features (V1 style)")
    parser.add_argument("--save", help="Path to save trained model")
    parser.add_argument("--device", default="cuda:0", help="Device to train on")
    args = parser.parse_args()

    print("=" * 60)
    print("Tamiyo Phase 2: Reward-Weighted Imitation")
    print("=" * 60)
    print(f"Pack: {args.pack}")
    print(f"Threshold: {args.threshold}%")
    print(f"Episode weighting: {args.episode_weighting}")
    print(f"Use telemetry: {not args.no_telemetry}")
    print()

    # Load data
    pack = load_pack(Path(args.pack))
    print(f"Loaded pack: {pack['metadata']['num_episodes']} episodes")

    # Extract samples
    X_all, y_all, w_all = extract_samples(
        pack,
        threshold=args.threshold,
        use_telemetry=not args.no_telemetry,
        episode_weighting=args.episode_weighting,
    )
    print(f"Total samples: {len(X_all)}")
    print()

    # Train
    model, best_acc = train(X_all, y_all, w_all, device=args.device)

    # Final evaluation
    print()
    print("=" * 60)
    print(f"Final Results (balanced_acc={best_acc*100:.1f}%)")
    print("=" * 60)

    # Re-split for final eval display
    combined = list(zip(X_all, y_all))
    random.seed(42)
    random.shuffle(combined)
    X_all, y_all = zip(*combined)
    split = int(0.8 * len(X_all))
    X_val = torch.tensor(X_all[split:], dtype=torch.float32, device=args.device)
    y_val = torch.tensor(y_all[split:], dtype=torch.long, device=args.device)

    results = evaluate(model, X_val, y_val)
    for action in SimicAction:
        if action.name in results:
            r = results[action.name]
            print(f"{action.name:10s}: {r['correct']:4d}/{r['total']:4d} = {r['accuracy']*100:5.1f}%")
    print(f"{'OVERALL':10s}: {results['overall']*100:.1f}%")

    # Save if requested
    if args.save:
        save_path = Path(args.save)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            'model_state_dict': model.state_dict(),
            'input_dim': X_val.shape[1],
            'balanced_accuracy': best_acc,
            'threshold': args.threshold,
            'episode_weighting': args.episode_weighting,
            'use_telemetry': not args.no_telemetry,
        }, save_path)
        print(f"\nModel saved to {save_path}")


if __name__ == "__main__":
    main()
