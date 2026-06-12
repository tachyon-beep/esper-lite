"""TaskSpec preset coverage."""

import pytest

from esper.runtime import get_task_spec
from esper.runtime.tasks import TaskSpec, VALID_TASKS
from esper.kasmina.blueprints import BlueprintRegistry
from esper.kasmina.host import CNNHost, TransformerHost


def test_cifar_baseline_spec_builds_model_and_loaders():
    spec = get_task_spec("cifar_baseline")

    model = spec.create_model(device="cpu", slots=["r0c1"])
    assert isinstance(model.host, CNNHost)
    assert len(spec.action_enum) == (
        len([s for s in BlueprintRegistry.list_for_topology("cnn") if s.name != "noop"])
        + 4
    )

    trainloader, testloader = spec.create_dataloaders(batch_size=8, mock=True, num_workers=0)
    batch_x, batch_y = next(iter(trainloader))
    assert batch_x.shape[0] == 8
    assert batch_y.shape[0] == 8
    assert len(testloader) > 0


def test_tinystories_spec_builds_model_and_loaders():
    spec = get_task_spec("tinystories")

    model = spec.create_model(device="cpu", slots=["r0c1"])
    assert isinstance(model.host, TransformerHost)
    assert len(spec.action_enum) == (
        len(
            [
                s
                for s in BlueprintRegistry.list_for_topology("transformer")
                if s.name != "noop"
            ]
        )
        + 4
    )

    trainloader, valloader = spec.create_dataloaders(
        mock=True,
        max_train_samples=4,
        max_val_samples=4,
        batch_size=2,
        num_workers=0,
    )
    batch_x, batch_y = next(iter(trainloader))
    assert batch_x.shape[0] == 2
    assert batch_x.shape[1] == spec.block_size
    assert batch_y.shape == batch_x.shape
    assert len(valloader) > 0


def test_cifar_impaired_spec_supports_three_slots():
    spec = get_task_spec("cifar_impaired")

    model = spec.create_model(device="cpu", slots=["r0c0", "r0c1", "r0c2"])
    assert isinstance(model.host, CNNHost)
    assert set(model.seed_slots.keys()) == {"r0c0", "r0c1", "r0c2"}


@pytest.mark.parametrize("task_name", sorted(VALID_TASKS))
def test_all_task_specs_resolve_case_insensitively(task_name: str) -> None:
    spec = get_task_spec(task_name.upper())

    assert spec.name == task_name
    assert spec.host_lr > 0
    assert spec.seed_lr > 0
    assert spec.action_enum is not None


def test_unknown_task_lists_valid_options() -> None:
    with pytest.raises(ValueError, match="Unknown task 'not_a_task'"):
        get_task_spec("not_a_task")


def test_taskspec_create_model_requires_slots_and_forwards_arguments() -> None:
    calls: list[tuple[str, list[str], bool]] = []

    def make_model(
        device: str,
        slots: list[str] | None = None,
        permissive_gates: bool = True,
    ) -> str:
        if slots is None:
            raise AssertionError("TaskSpec should pass slots through")
        calls.append((device, slots, permissive_gates))
        return "model"

    spec = TaskSpec(
        name="custom",
        topology="cnn",
        task_type="classification",
        model_factory=make_model,
        dataloader_factory=lambda: ("train", "test"),
    )

    with pytest.raises(ValueError, match="slots parameter is required"):
        spec.create_model(device="cpu", slots=[])

    assert spec.create_model(device="cpu", slots=["r0c0"], permissive_gates=False) == "model"
    assert calls == [("cpu", ["r0c0"], False)]


def test_taskspec_create_dataloaders_merges_defaults_and_overrides() -> None:
    calls: list[dict[str, int]] = []

    def make_loaders(batch_size: int, num_workers: int) -> tuple[str, str]:
        calls.append({"batch_size": batch_size, "num_workers": num_workers})
        return "train", "test"

    spec = TaskSpec(
        name="custom",
        topology="cnn",
        task_type="classification",
        model_factory=lambda device: "model",
        dataloader_factory=make_loaders,
        dataloader_defaults={"batch_size": 16, "num_workers": 4},
    )

    assert spec.create_dataloaders(batch_size=8) == ("train", "test")
    assert calls == [{"batch_size": 8, "num_workers": 4}]


def test_cifar_task_get_datasets_uses_mock_dataset() -> None:
    spec = get_task_spec("cifar_minimal")
    spec.dataloader_defaults["mock"] = True

    trainset, testset = spec.get_datasets()

    assert len(trainset) > 0
    assert len(testset) > 0


def test_tinystories_task_get_datasets_uses_mock_dataset() -> None:
    spec = get_task_spec("tinystories")
    spec.dataloader_defaults["mock"] = True
    spec.dataloader_defaults["block_size"] = 8
    spec.dataloader_defaults["max_train_samples"] = 3
    spec.dataloader_defaults["max_val_samples"] = 2

    trainset, testset = spec.get_datasets()
    train_x, train_y = trainset[0]

    assert len(trainset) == 3
    assert len(testset) == 2
    assert train_x.shape[0] == 8
    assert train_y.shape == train_x.shape


def test_get_datasets_rejects_unknown_task_family() -> None:
    spec = TaskSpec(
        name="custom",
        topology="cnn",
        task_type="classification",
        model_factory=lambda device: "model",
        dataloader_factory=lambda: ("train", "test"),
    )

    with pytest.raises(NotImplementedError, match="get_datasets not implemented"):
        spec.get_datasets()
