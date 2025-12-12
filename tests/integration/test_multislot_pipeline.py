"""Integration Tests for Multi-Slot Pipeline.

Tests that verify all components work together end-to-end:
- MorphogeneticModel with multiple slots
- FactoredActorCritic network with correct dimensions
- Feature extraction (obs_to_multislot_features)
- Action masking (compute_action_masks)
- Simple reward computation (compute_simple_reward)

Focus: Integration - verifying components work together, not re-testing individual components.
"""

import torch
import pytest


def test_multislot_model_creation_and_forward():
    """Multi-slot model creation with all 3 slots and forward pass."""
    from esper.kasmina.host import CNNHost, MorphogeneticModel

    # Create host and multi-slot model
    host = CNNHost()
    model = MorphogeneticModel(host, device="cpu", slots=["early", "mid", "late"])

    # Verify slots are created
    assert len(model.seed_slots) == 3
    assert "early" in model.seed_slots
    assert "mid" in model.seed_slots
    assert "late" in model.seed_slots

    # Verify correct channel dimensions from host
    assert model.seed_slots["early"].channels == 32
    assert model.seed_slots["mid"].channels == 64
    assert model.seed_slots["late"].channels == 128

    # Forward pass through model (no seeds yet)
    x = torch.randn(2, 3, 32, 32)
    out = model(x)
    assert out.shape == (2, 10), "Model should output class logits"

    # Verify no active seeds initially
    assert not model.has_active_seed


def test_factored_network_with_correct_dimensions():
    """FactoredActorCritic with correct state/action dimensions from multislot features."""
    from esper.simic.factored_network import FactoredActorCritic
    from esper.simic.features import MULTISLOT_FEATURE_SIZE
    from esper.leyline.factored_actions import NUM_SLOTS, NUM_BLUEPRINTS, NUM_BLENDS, NUM_OPS

    # Create network with multislot feature size
    policy = FactoredActorCritic(
        state_dim=MULTISLOT_FEATURE_SIZE,  # 34 dims
        num_slots=NUM_SLOTS,  # 3
        num_blueprints=NUM_BLUEPRINTS,  # 5
        num_blends=NUM_BLENDS,  # 3
        num_ops=NUM_OPS,  # 4
    )

    # Forward pass
    obs = torch.randn(4, MULTISLOT_FEATURE_SIZE)
    dists, values = policy(obs)

    # Verify output shapes
    assert dists["slot"].probs.shape == (4, NUM_SLOTS)
    assert dists["blueprint"].probs.shape == (4, NUM_BLUEPRINTS)
    assert dists["blend"].probs.shape == (4, NUM_BLENDS)
    assert dists["op"].probs.shape == (4, NUM_OPS)
    assert values.shape == (4,)

    # Sample actions
    actions, log_probs, action_values = policy.get_action_batch(obs)
    assert "slot" in actions
    assert "blueprint" in actions
    assert "blend" in actions
    assert "op" in actions
    assert log_probs.shape == (4,)
    assert action_values.shape == (4,)


def test_feature_extraction_to_network_flow():
    """Feature extraction → Network forward pass pipeline."""
    from esper.simic.features import obs_to_multislot_features, MULTISLOT_FEATURE_SIZE
    from esper.simic.factored_network import FactoredActorCritic

    # Create observation with slot states
    obs = {
        'epoch': 10,
        'global_step': 100,
        'train_loss': 0.5,
        'val_loss': 0.6,
        'loss_delta': -0.1,
        'train_accuracy': 70.0,
        'val_accuracy': 68.0,
        'accuracy_delta': 0.5,
        'plateau_epochs': 2,
        'best_val_accuracy': 70.0,
        'best_val_loss': 0.5,
        'loss_history_5': [0.6, 0.55, 0.5, 0.52, 0.5],
        'accuracy_history_5': [65.0, 66.0, 67.0, 68.0, 68.0],
        'total_params': 100_000,
        'slots': {
            'early': {'is_active': False, 'stage': 0, 'alpha': 0.0, 'improvement': 0.0},
            'mid': {'is_active': True, 'stage': 3, 'alpha': 0.5, 'improvement': 2.5},
            'late': {'is_active': False, 'stage': 0, 'alpha': 0.0, 'improvement': 0.0},
        },
    }

    # Extract features
    features = obs_to_multislot_features(obs)
    assert len(features) == MULTISLOT_FEATURE_SIZE, "Should extract 35 features"

    # Convert to tensor and feed to network
    features_tensor = torch.tensor([features], dtype=torch.float32)
    policy = FactoredActorCritic(state_dim=MULTISLOT_FEATURE_SIZE)

    dists, values = policy(features_tensor)
    assert values.shape == (1,)

    # Verify feature structure
    # Base features: 23
    assert len(features[:23]) == 23
    # Per-slot features: 3 slots * 9 features (4 state + 5 blueprint one-hot) = 27
    assert len(features[23:]) == 27

    # Verify slot state features (first 4 of each 9-dim slot block)
    early_state = features[23:27]
    mid_state = features[32:36]  # 23 + 9 = 32
    late_state = features[41:45]  # 23 + 18 = 41

    assert early_state == [0.0, 0.0, 0.0, 0.0], "Early slot inactive"
    assert mid_state == [1.0, 3.0, 0.5, 2.5], "Mid slot active"
    assert late_state == [0.0, 0.0, 0.0, 0.0], "Late slot inactive"


def test_action_masking_integration():
    """Action masking → Network forward with masks → Valid action sampling."""
    from esper.simic.action_masks import compute_action_masks, MaskSeedInfo
    from esper.simic.factored_network import FactoredActorCritic
    from esper.simic.features import MULTISLOT_FEATURE_SIZE
    from esper.leyline import SeedStage
    from esper.leyline.factored_actions import LifecycleOp

    # Create slot states using MaskSeedInfo (new interface)
    slot_states = {
        "early": None,  # Empty
        "mid": MaskSeedInfo(
            stage=SeedStage.TRAINING.value,
            seed_age_epochs=3,
        ),  # Occupied
        "late": None,  # Empty
    }

    # Compute masks for the mid slot (where our seed is)
    masks_single = compute_action_masks(slot_states, target_slot="mid")

    # Verify masks are correct (NUM_OPS=4 now: WAIT, GERMINATE, CULL, FOSSILIZE)
    assert masks_single["op"][LifecycleOp.WAIT] == True, "WAIT always valid"
    assert masks_single["op"][LifecycleOp.GERMINATE] == True, "Can GERMINATE (other slots empty)"
    assert masks_single["op"][LifecycleOp.CULL] == True, "Can CULL (seed_age >= MIN_CULL_AGE)"
    assert masks_single["op"][LifecycleOp.FOSSILIZE] == False, "Can't FOSSILIZE (not PROBATIONARY)"

    # Create batch masks for network
    batch_size = 4
    masks_batch = {
        "slot": masks_single["slot"].unsqueeze(0).expand(batch_size, -1),
        "blueprint": masks_single["blueprint"].unsqueeze(0).expand(batch_size, -1),
        "blend": masks_single["blend"].unsqueeze(0).expand(batch_size, -1),
        "op": masks_single["op"].unsqueeze(0).expand(batch_size, -1),
    }

    # Forward through network with masks
    policy = FactoredActorCritic(state_dim=MULTISLOT_FEATURE_SIZE)
    obs = torch.randn(batch_size, MULTISLOT_FEATURE_SIZE)

    dists, values = policy(obs, masks=masks_batch)

    # Verify masked actions have zero probability
    # Note: GERMINATE is valid because other slots (early, late) are empty
    from esper.leyline.factored_actions import LifecycleOp
    assert (dists["op"].probs[:, LifecycleOp.GERMINATE] > 0).all(), "GERMINATE should be valid (empty slots)"
    assert (dists["op"].probs[:, LifecycleOp.CULL] > 0).all(), "CULL should be valid"
    assert (dists["op"].probs[:, LifecycleOp.FOSSILIZE] == 0).all(), "FOSSILIZE should be masked (not PROBATIONARY)"


def test_simple_reward_computation_multislot():
    """Simple reward computation with multi-slot contributions."""
    from esper.simic.simple_rewards import compute_simple_reward

    # Test 1: Single slot contribution
    reward_single = compute_simple_reward(
        seed_contributions={"mid": 3.0},
        total_params=100_000,
        host_params=100_000,
        is_terminal=False,
        val_acc=70.0,
    )
    assert reward_single > 0, "Positive contribution should give positive reward"

    # Test 2: Multiple slot contributions
    reward_multi = compute_simple_reward(
        seed_contributions={"early": 1.0, "mid": 1.0, "late": 1.0},
        total_params=100_000,
        host_params=100_000,
        is_terminal=False,
        val_acc=70.0,
    )
    assert abs(reward_single - reward_multi) < 0.01, "Same total contribution = same reward"

    # Test 3: None contributions ignored
    reward_with_none = compute_simple_reward(
        seed_contributions={"early": None, "mid": 2.0, "late": None},
        total_params=100_000,
        host_params=100_000,
        is_terminal=False,
        val_acc=70.0,
    )
    reward_without_none = compute_simple_reward(
        seed_contributions={"mid": 2.0},
        total_params=100_000,
        host_params=100_000,
        is_terminal=False,
        val_acc=70.0,
    )
    assert abs(reward_with_none - reward_without_none) < 0.01, "None contributions ignored"

    # Test 4: Parameter bloat penalty
    reward_small = compute_simple_reward(
        seed_contributions={"mid": 2.0},
        total_params=100_000,
        host_params=100_000,
        is_terminal=False,
        val_acc=70.0,
    )
    reward_bloated = compute_simple_reward(
        seed_contributions={"mid": 2.0},
        total_params=200_000,
        host_params=100_000,
        is_terminal=False,
        val_acc=70.0,
    )
    assert reward_small > reward_bloated, "Bloat should reduce reward"


def test_end_to_end_multislot_lifecycle():
    """Full lifecycle: Model creation → Germinate → Forward → Cull."""
    from esper.kasmina.host import CNNHost, MorphogeneticModel

    host = CNNHost()
    model = MorphogeneticModel(host, device="cpu", slots=["early", "mid", "late"])

    # Initially no active seeds
    assert not model.has_active_seed

    # Germinate in different slots (use actual available blueprints)
    model.germinate_seed("conv_light", "seed_early", slot="early")
    assert model.has_active_seed_in_slot("early")
    assert not model.has_active_seed_in_slot("mid")
    assert not model.has_active_seed_in_slot("late")

    model.germinate_seed("attention", "seed_late", slot="late")
    assert model.has_active_seed_in_slot("early")
    assert not model.has_active_seed_in_slot("mid")
    assert model.has_active_seed_in_slot("late")

    # Forward pass with active seeds
    x = torch.randn(2, 3, 32, 32)
    out = model(x)
    assert out.shape == (2, 10)

    # Verify parameter counts increased
    assert model.active_seed_params > 0

    # Cull a seed
    model.cull_seed(slot="early")
    assert not model.has_active_seed_in_slot("early")
    assert model.has_active_seed_in_slot("late")

    # Model still works after culling
    out = model(x)
    assert out.shape == (2, 10)

    # Cull remaining seed
    model.cull_seed(slot="late")
    assert not model.has_active_seed
    assert model.active_seed_params == 0


def test_multislot_with_all_components_integrated():
    """Integration test combining ALL components: model, policy, features, masks, rewards."""
    from esper.kasmina.host import CNNHost, MorphogeneticModel
    from esper.simic.factored_network import FactoredActorCritic
    from esper.simic.features import obs_to_multislot_features, MULTISLOT_FEATURE_SIZE
    from esper.simic.action_masks import compute_action_masks, MaskSeedInfo
    from esper.simic.simple_rewards import compute_simple_reward
    from esper.leyline import SeedStage

    # 1. Create multi-slot model
    host = CNNHost()
    model = MorphogeneticModel(host, device="cpu", slots=["early", "mid", "late"])

    # 2. Create policy network
    policy = FactoredActorCritic(state_dim=MULTISLOT_FEATURE_SIZE)

    # 3. Germinate seeds (use actual available blueprints)
    model.germinate_seed("conv_light", "seed_early", slot="early")
    model.germinate_seed("attention", "seed_mid", slot="mid")

    # 4. Create observation
    obs = {
        'epoch': 15,
        'global_step': 150,
        'train_loss': 0.4,
        'val_loss': 0.5,
        'loss_delta': -0.15,
        'train_accuracy': 75.0,
        'val_accuracy': 72.0,
        'accuracy_delta': 1.5,
        'plateau_epochs': 1,
        'best_val_accuracy': 74.0,
        'best_val_loss': 0.45,
        'loss_history_5': [0.5, 0.48, 0.45, 0.42, 0.4],
        'accuracy_history_5': [70.0, 71.0, 72.0, 73.0, 75.0],
        'total_params': 150_000,
        'slots': {
            'early': {'is_active': True, 'stage': 2, 'alpha': 0.3, 'improvement': 1.2},
            'mid': {'is_active': True, 'stage': 3, 'alpha': 0.6, 'improvement': 2.1},
            'late': {'is_active': False, 'stage': 0, 'alpha': 0.0, 'improvement': 0.0},
        },
    }

    # 5. Extract features
    features = obs_to_multislot_features(obs)
    features_tensor = torch.tensor([features], dtype=torch.float32)
    assert features_tensor.shape == (1, MULTISLOT_FEATURE_SIZE)

    # 6. Compute action masks
    slot_states = {
        "early": MaskSeedInfo(stage=SeedStage.TRAINING.value, seed_age_epochs=3),
        "mid": MaskSeedInfo(stage=SeedStage.TRAINING.value, seed_age_epochs=5),
        "late": None,
    }
    masks_single = compute_action_masks(slot_states, target_slot="mid")
    masks_batch = {k: v.unsqueeze(0) for k, v in masks_single.items()}

    # 7. Policy forward with masks
    dists, values = policy(features_tensor, masks=masks_batch)
    assert values.shape == (1,)

    # 8. Sample action
    actions, log_probs, action_values = policy.get_action_batch(features_tensor, masks=masks_batch)
    assert "slot" in actions
    assert "blueprint" in actions
    assert "blend" in actions
    assert "op" in actions

    # 9. Compute reward
    reward = compute_simple_reward(
        seed_contributions={"early": 1.2, "mid": 2.1, "late": None},
        total_params=150_000,
        host_params=100_000,
        is_terminal=False,
        val_acc=72.0,
    )
    assert isinstance(reward, float)
    assert reward > 0, "Positive contributions should give positive reward"

    # 10. Forward through model
    x = torch.randn(2, 3, 32, 32)
    out = model(x)
    assert out.shape == (2, 10)

    # Verify all pieces work together
    assert model.has_active_seed
    assert model.active_seed_params > 0
    assert len(features) == MULTISLOT_FEATURE_SIZE
    assert values.shape[0] == 1


def test_multislot_action_execution_flow():
    """Test action sampling → execution → state update flow."""
    from esper.kasmina.host import CNNHost, MorphogeneticModel
    from esper.simic.factored_network import FactoredActorCritic
    from esper.simic.features import MULTISLOT_FEATURE_SIZE
    from esper.leyline.factored_actions import FactoredAction, SlotAction, BlueprintAction, BlendAction, LifecycleOp

    # Setup
    host = CNNHost()
    model = MorphogeneticModel(host, device="cpu", slots=["early", "mid", "late"])
    policy = FactoredActorCritic(state_dim=MULTISLOT_FEATURE_SIZE)

    # Sample action
    obs = torch.randn(1, MULTISLOT_FEATURE_SIZE)
    actions, log_probs, values = policy.get_action_batch(obs)

    # Construct factored action
    factored_action = FactoredAction.from_indices(
        slot_idx=int(actions["slot"][0]),
        blueprint_idx=int(actions["blueprint"][0]),
        blend_idx=int(actions["blend"][0]),
        op_idx=int(actions["op"][0]),
    )

    # Verify action is valid
    assert factored_action.slot_id in ["early", "mid", "late"]
    # Note: factored_actions uses old names but they map to actual blueprints
    assert factored_action.blueprint_id in ["noop", "conv_light", "attention", "norm", "depthwise"]
    assert factored_action.blend_algorithm_id in ["linear", "sigmoid", "gated"]

    # Execute GERMINATE action
    if factored_action.is_germinate and factored_action.blueprint_id != "noop":
        slot_id = factored_action.slot_id
        if not model.has_active_seed_in_slot(slot_id):
            # Map factored action names to actual blueprint names
            blueprint_map = {
                "conv_enhance": "conv_light",  # Map to actual available blueprint
                "attention": "attention",
                "norm": "norm",
                "depthwise": "depthwise",
            }
            actual_blueprint = blueprint_map.get(factored_action.blueprint_id, "conv_light")
            model.germinate_seed(
                actual_blueprint,
                f"test_seed_{slot_id}",
                slot=slot_id,
            )
            assert model.has_active_seed_in_slot(slot_id)

    # Execute CULL action
    elif factored_action.is_cull:
        slot_id = factored_action.slot_id
        if model.has_active_seed_in_slot(slot_id):
            model.cull_seed(slot=slot_id)
            assert not model.has_active_seed_in_slot(slot_id)

    # Model still functional after action
    x = torch.randn(2, 3, 32, 32)
    out = model(x)
    assert out.shape == (2, 10)


def test_multislot_batch_processing():
    """Test batch processing through all components."""
    from esper.simic.factored_network import FactoredActorCritic
    from esper.simic.features import MULTISLOT_FEATURE_SIZE
    from esper.simic.action_masks import compute_batch_masks, MaskSeedInfo
    from esper.leyline import SeedStage
    from esper.leyline.factored_actions import LifecycleOp, NUM_OPS

    batch_size = 8

    # Create batch of observations
    obs_batch = torch.randn(batch_size, MULTISLOT_FEATURE_SIZE)

    # Create batch of slot states
    batch_slot_states = []
    for i in range(batch_size):
        if i % 2 == 0:
            # Even indices: mid slot occupied, early/late empty
            batch_slot_states.append({
                "early": None,
                "mid": MaskSeedInfo(stage=SeedStage.TRAINING.value, seed_age_epochs=3),
                "late": None,
            })
        else:
            # Odd indices: all slots empty
            batch_slot_states.append({
                "early": None,
                "mid": None,
                "late": None,
            })

    # Compute batch masks
    masks = compute_batch_masks(batch_slot_states)
    assert masks["slot"].shape == (batch_size, 3)
    assert masks["blueprint"].shape == (batch_size, 5)
    assert masks["blend"].shape == (batch_size, 3)
    assert masks["op"].shape == (batch_size, NUM_OPS)  # NUM_OPS=4

    # Verify masks are different for even/odd indices
    # Even: has seed, CULL allowed
    # Odd: empty slots, CULL blocked
    for i in range(batch_size):
        if i % 2 == 0:
            # Has seed in mid, empty slots in early/late
            assert masks["op"][i, LifecycleOp.GERMINATE] == True, "Can GERMINATE (empty slots)"
            assert masks["op"][i, LifecycleOp.CULL] == True, "Can CULL (has seed in TRAINING)"
            assert masks["op"][i, LifecycleOp.FOSSILIZE] == False, "Can't FOSSILIZE (not PROBATIONARY)"
        else:
            # All slots empty
            assert masks["op"][i, LifecycleOp.GERMINATE] == True, "Can GERMINATE in empty slot"
            assert masks["op"][i, LifecycleOp.CULL] == False, "Can't CULL in empty slot"

    # Forward through network with batch masks
    policy = FactoredActorCritic(state_dim=MULTISLOT_FEATURE_SIZE)
    dists, values = policy(obs_batch, masks=masks)

    assert values.shape == (batch_size,)
    assert dists["op"].probs.shape == (batch_size, NUM_OPS)  # NUM_OPS=4

    # Sample actions for entire batch
    actions, log_probs, action_values = policy.get_action_batch(obs_batch, masks=masks)
    assert actions["slot"].shape == (batch_size,)
    assert actions["op"].shape == (batch_size,)
    assert log_probs.shape == (batch_size,)
