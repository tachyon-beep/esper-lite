"""Integration tests for heuristic multi-slot runtime."""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader, TensorDataset


def test_heuristic_episode_germinates_and_culls_across_slots(monkeypatch) -> None:
    """Heuristic path should target multiple slots and resolve target_seed_id → slot."""
    from esper.kasmina.host import MorphogeneticModel
    from esper.leyline.actions import build_action_enum
    from esper.runtime import get_task_spec
    from esper.simic.training import run_heuristic_episode
    from esper.tamiyo.decisions import TamiyoDecision

    task_spec = get_task_spec("cifar10")
    Action = build_action_enum(task_spec.topology)

    class ScriptedPolicy:
        def reset(self) -> None:
            pass

        def decide(self, signals, active_seeds):  # noqa: ANN001
            match signals.metrics.epoch:
                case 1:
                    return TamiyoDecision(action=Action.GERMINATE_CONV_LIGHT)
                case 2:
                    return TamiyoDecision(action=Action.GERMINATE_NORM)
                case 3:
                    return TamiyoDecision(action=Action.CULL, target_seed_id="seed_1")
                case _:
                    return TamiyoDecision(action=Action.WAIT)

    # Tiny synthetic dataloaders (1 batch) keep this test fast and hermetic.
    x = torch.randn(4, 3, 32, 32)
    y = torch.randint(0, 10, (4,))
    trainloader = DataLoader(TensorDataset(x, y), batch_size=4, shuffle=False)
    testloader = DataLoader(TensorDataset(x, y), batch_size=4, shuffle=False)

    created: dict[str, object] = {}
    germinate_slots: list[str] = []

    import esper.tolaria as tolaria

    real_create_model = tolaria.create_model

    def create_model_spy(*, task, device, slots):  # noqa: ANN001
        model = real_create_model(task=task, device=device, slots=slots)
        created["model"] = model
        return model

    monkeypatch.setattr(tolaria, "create_model", create_model_spy)

    real_germinate_seed = MorphogeneticModel.germinate_seed

    def germinate_seed_spy(self, blueprint_id, seed_id, *, slot, blend_algorithm_id="sigmoid"):  # noqa: ANN001
        germinate_slots.append(slot)
        return real_germinate_seed(
            self,
            blueprint_id,
            seed_id,
            slot=slot,
            blend_algorithm_id=blend_algorithm_id,
        )

    monkeypatch.setattr(MorphogeneticModel, "germinate_seed", germinate_seed_spy)

    final_acc, action_counts, _ = run_heuristic_episode(
        policy=ScriptedPolicy(),
        trainloader=trainloader,
        testloader=testloader,
        max_epochs=3,
        max_batches=1,
        base_seed=123,
        device="cpu",
        task_spec=task_spec,
        slots=["r0c0", "r0c1", "r0c2"],  # Canonical slot IDs
    )
    assert 0.0 <= final_acc <= 100.0

    assert germinate_slots == ["r0c0", "r0c1"]  # Was ["early", "mid"]
    assert action_counts["GERMINATE"] == 2
    assert action_counts["CULL"] == 1

    model = created["model"]
    assert isinstance(model, MorphogeneticModel)
    assert model.has_active_seed_in_slot("r0c0")  # Was "early"
    assert not model.has_active_seed_in_slot("r0c1")  # Was "mid"
    assert not model.has_active_seed_in_slot("r0c2")  # Was "late"

