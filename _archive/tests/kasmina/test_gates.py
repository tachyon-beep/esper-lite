from esper.kasmina.gates import GateInputs, KasminaGates
from esper.leyline import leyline_pb2 as pb


def test_g3_mesh_incomplete_fails() -> None:
    gates = KasminaGates()
    inputs = GateInputs(
        host_params_registered=True,
        interface_checks_ok=True,
        mesh_required_layers=("layer1", "layer2"),
        mesh_available_layers=("layer1",),
    )
    result = gates.evaluate(pb.SEED_GATE_G3_INTERFACE, inputs)
    assert not result.passed
    assert result.reason == "mesh_incomplete"
    assert result.attributes.get("missing_layers") == "layer2"


def test_g4_rejects_fallback_status() -> None:
    gates = KasminaGates()
    inputs = GateInputs(performance_status="fallback")
    result = gates.evaluate(pb.SEED_GATE_G4_SYSTEM_IMPACT, inputs)
    assert not result.passed
    assert result.reason == "fallback_kernel_active"
    assert result.attributes.get("status") == "fallback"
