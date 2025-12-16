"""TaskSpec preset coverage."""

from esper.runtime import get_task_spec
from esper.kasmina.blueprints import BlueprintRegistry
from esper.kasmina.host import CNNHost, TransformerHost


def test_cifar10_spec_builds_model_and_loaders():
    spec = get_task_spec("cifar10")

    model = spec.create_model(device="cpu", slots=["r0c1"])
    assert isinstance(model.host, CNNHost)
    assert len(spec.action_enum) == len(BlueprintRegistry.list_for_topology("cnn")) + 3

    trainloader, testloader = spec.create_dataloaders(batch_size=8, mock=True, num_workers=0)
    batch_x, batch_y = next(iter(trainloader))
    assert batch_x.shape[0] == 8
    assert batch_y.shape[0] == 8
    assert len(testloader) > 0


def test_tinystories_spec_builds_model_and_loaders():
    spec = get_task_spec("tinystories")

    model = spec.create_model(device="cpu", slots=["r0c1"])
    assert isinstance(model.host, TransformerHost)
    assert len(spec.action_enum) == len(BlueprintRegistry.list_for_topology("transformer")) + 3

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
