"""Gate evaluation helpers for Kasmina lifecycle enforcement."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import isnan
from typing import Mapping

from esper.leyline import leyline_pb2


@dataclass(slots=True)
class GateInputs:
    """Inputs considered when evaluating a lifecycle gate."""

    blueprint_id: str | None = None
    parameters: Mapping[str, float] | None = None
    isolation_violations: int = 0
    kernel_attached: bool = False
    last_latency_ms: float = 0.0
    latency_budget_ms: float = 0.0
    fallback_used: bool = False
    host_params_registered: bool = False
    interface_checks_ok: bool = False
    performance_status: str = "nominal"
    telemetry_stage: str | None = None
    expected_stage: str | None = None
    reset_clean: bool = False


@dataclass(slots=True)
class GateResult:
    """Outcome of evaluating a lifecycle gate."""

    gate: int
    passed: bool
    reason: str = ""
    attributes: Mapping[str, str] = field(default_factory=dict)


class KasminaGates:
    """Implements Kasmina lifecycle gate policies."""

    def evaluate(self, gate: int, inputs: GateInputs) -> GateResult:
        if gate == leyline_pb2.SEED_GATE_G0_SANITY:
            return self._evaluate_g0(inputs)
        if gate == leyline_pb2.SEED_GATE_G1_GRADIENT_HEALTH:
            return self._evaluate_g1(inputs)
        if gate == leyline_pb2.SEED_GATE_G2_STABILITY:
            return self._evaluate_g2(inputs)
        if gate == leyline_pb2.SEED_GATE_G3_INTERFACE:
            return self._evaluate_g3(inputs)
        if gate == leyline_pb2.SEED_GATE_G4_SYSTEM_IMPACT:
            return self._evaluate_g4(inputs)
        if gate == leyline_pb2.SEED_GATE_G5_RESET:
            return self._evaluate_g5(inputs)
        return GateResult(
            gate=leyline_pb2.SEED_GATE_UNKNOWN,
            passed=False,
            reason="unsupported gate",
            attributes={"gate": "unknown"},
        )

    def _evaluate_g0(self, inputs: GateInputs) -> GateResult:
        blueprint_id = (inputs.blueprint_id or "").strip()
        params = inputs.parameters or {}
        if not blueprint_id:
            return GateResult(
                gate=leyline_pb2.SEED_GATE_G0_SANITY,
                passed=False,
                reason="missing blueprint id",
                attributes={"blueprint_id": ""},
            )
        for key, value in params.items():
            if value is None or (isinstance(value, float) and isnan(value)):
                return GateResult(
                    gate=leyline_pb2.SEED_GATE_G0_SANITY,
                    passed=False,
                    reason=f"parameter {key} invalid",
                    attributes={"parameter": key},
                )
        return GateResult(
            gate=leyline_pb2.SEED_GATE_G0_SANITY,
            passed=True,
            attributes={"blueprint_id": blueprint_id},
        )

    def _evaluate_g1(self, inputs: GateInputs) -> GateResult:
        if not inputs.kernel_attached:
            return GateResult(
                gate=leyline_pb2.SEED_GATE_G1_GRADIENT_HEALTH,
                passed=False,
                reason="kernel not attached",
                attributes={"kernel_attached": "false"},
            )
        if inputs.isolation_violations:
            return GateResult(
                gate=leyline_pb2.SEED_GATE_G1_GRADIENT_HEALTH,
                passed=False,
                reason="gradient isolation violations present",
                attributes={"violations": str(inputs.isolation_violations)},
            )
        return GateResult(
            gate=leyline_pb2.SEED_GATE_G1_GRADIENT_HEALTH,
            passed=True,
            attributes={"kernel_attached": "true"},
        )

    def _evaluate_g2(self, inputs: GateInputs) -> GateResult:
        if inputs.last_latency_ms and inputs.last_latency_ms > inputs.latency_budget_ms:
            return GateResult(
                gate=leyline_pb2.SEED_GATE_G2_STABILITY,
                passed=False,
                reason="latency budget exceeded",
                attributes={
                    "latency_ms": f"{inputs.last_latency_ms:.2f}",
                    "budget_ms": f"{inputs.latency_budget_ms:.2f}",
                },
            )
        return GateResult(
            gate=leyline_pb2.SEED_GATE_G2_STABILITY,
            passed=True,
            attributes={"fallback_used": str(inputs.fallback_used).lower()},
        )

    def _evaluate_g3(self, inputs: GateInputs) -> GateResult:
        if not inputs.host_params_registered:
            return GateResult(
                gate=leyline_pb2.SEED_GATE_G3_INTERFACE,
                passed=False,
                reason="host model not registered",
                attributes={"host_params_registered": "false"},
            )
        if not inputs.interface_checks_ok:
            return GateResult(
                gate=leyline_pb2.SEED_GATE_G3_INTERFACE,
                passed=False,
                reason="interface checks incomplete",
                attributes={"interface": "failed"},
            )
        return GateResult(
            gate=leyline_pb2.SEED_GATE_G3_INTERFACE,
            passed=True,
            attributes={"interface": "ok"},
        )

    def _evaluate_g4(self, inputs: GateInputs) -> GateResult:
        if inputs.performance_status not in {"nominal", "healthy", "fallback"}:
            return GateResult(
                gate=leyline_pb2.SEED_GATE_G4_SYSTEM_IMPACT,
                passed=False,
                reason=f"performance status {inputs.performance_status}",
                attributes={"status": inputs.performance_status},
            )
        if inputs.isolation_violations:
            return GateResult(
                gate=leyline_pb2.SEED_GATE_G4_SYSTEM_IMPACT,
                passed=False,
                reason="violations recorded",
                attributes={"violations": str(inputs.isolation_violations)},
            )
        return GateResult(
            gate=leyline_pb2.SEED_GATE_G4_SYSTEM_IMPACT,
            passed=True,
            attributes={"status": inputs.performance_status},
        )

    def _evaluate_g5(self, inputs: GateInputs) -> GateResult:
        if not inputs.reset_clean:
            return GateResult(
                gate=leyline_pb2.SEED_GATE_G5_RESET,
                passed=False,
                reason="reset state not clean",
                attributes={},
            )
        return GateResult(
            gate=leyline_pb2.SEED_GATE_G5_RESET,
            passed=True,
            attributes={},
        )
