import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from esper.leyline import leyline_pb2
from esper.core import TelemetryEvent
from esper.security.signing import SignatureContext, sign
from esper.tolaria import TolariaTrainer, TrainingLoopConfig, TamiyoClient
from esper.kasmina.seed_manager import KasminaSeedManager

SIGNING_CONTEXT = SignatureContext(secret=b"wp101-integration")


class _TamiyoSeed(TamiyoClient):
    def evaluate_epoch(self, state: leyline_pb2.SystemStatePacket) -> leyline_pb2.AdaptationCommand:
        cmd = leyline_pb2.AdaptationCommand(
            version=1,
            command_id="cmd-seed",
            command_type=leyline_pb2.COMMAND_SEED,
            target_seed_id="seed-integ",
        )
        cmd.seed_operation.operation = leyline_pb2.SEED_OP_GERMINATE
        cmd.seed_operation.blueprint_id = "bp-integ"
        cmd.annotations["training_run_id"] = state.training_run_id or "integration-seed"
        cmd.annotations["feature_coverage"] = "0.5"
        cmd.annotations["blend_mode"] = "CONVEX"
        cmd.annotations["coverage_types"] = '{"node.seed":1}'
        cmd.annotations.setdefault("coverage_map", "{}")
        cmd.annotations.setdefault("mesh_host_layers", "[\"weight\"]")
        cmd.issued_at.GetCurrentTime()
        if "signature" in cmd.annotations:
            del cmd.annotations["signature"]
        cmd.annotations["signature"] = sign(
            cmd.SerializeToString(deterministic=True), SIGNING_CONTEXT
        )
        return cmd


@pytest.mark.integration
def test_control_loop_emits_seed_metrics() -> None:
    torch.manual_seed(42)
    model = nn.Linear(4, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    inputs = torch.randn(16, 4)
    targets = torch.randint(0, 2, (16,))
    dataloader = DataLoader(TensorDataset(inputs, targets), batch_size=4)

    runtime = type(
        "_Runtime",
        (),
        {"fetch_kernel": staticmethod(lambda *_: (nn.Identity(), 1.0))},
    )()

    kasmina = KasminaSeedManager(runtime=runtime, signing_context=SIGNING_CONTEXT)
    kasmina.register_host_model(model)
    kasmina.register_optimizer(optimizer)

    trainer = TolariaTrainer(
        model=model,
        optimizer=optimizer,
        dataloader=dataloader,
        tamiyo=_TamiyoSeed(),
        kasmina=kasmina,
        config=TrainingLoopConfig(
            max_epochs=2,
            gradient_accumulation_steps=1,
            enable_graphs=False,
            device=torch.device("cpu"),
            enable_amp=False,
            enable_tf32=False,
            enable_foreach_optim=False,
        ),
    )

    states = list(trainer.run())
    assert len(states) >= 2

    context = kasmina._seeds.get("seed-integ")
    assert context is not None
    context.alpha = 0.5
    context.pending_events.append(
        TelemetryEvent(
            description="seed_stage",
            attributes={"seed_id": context.seed_id},
        )
    )
    kasmina.finalize_step(step_index=trainer._global_step + 1)
    kasmina_packets = kasmina.drain_telemetry_packets()
    assert kasmina_packets, "Kasmina did not emit telemetry packets"
    alpha_values = [
        metric.value
        for packet in kasmina_packets
        for metric in packet.metrics
        if metric.name == "kasmina.seed.alpha"
    ]
    assert alpha_values
    assert alpha_values[-1] >= 0.0

    seed_events = [
        evt
        for packet in kasmina_packets
        for evt in packet.events
        if evt.description in {"seed_stage", "seed_health"}
    ]
    assert seed_events, "Expected seed_stage or seed_health events from Kasmina"

    assert kasmina.isolation_violations == 0

    trainer.close()
