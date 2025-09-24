import datetime

from google.protobuf import duration_pb2 as _duration_pb2
from google.protobuf import struct_pb2 as _struct_pb2
from google.protobuf import timestamp_pb2 as _timestamp_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class SeedLifecycleStage(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    SEED_STAGE_UNKNOWN: _ClassVar[SeedLifecycleStage]
    SEED_STAGE_DORMANT: _ClassVar[SeedLifecycleStage]
    SEED_STAGE_GERMINATED: _ClassVar[SeedLifecycleStage]
    SEED_STAGE_TRAINING: _ClassVar[SeedLifecycleStage]
    SEED_STAGE_BLENDING: _ClassVar[SeedLifecycleStage]
    SEED_STAGE_SHADOWING: _ClassVar[SeedLifecycleStage]
    SEED_STAGE_PROBATIONARY: _ClassVar[SeedLifecycleStage]
    SEED_STAGE_FOSSILIZED: _ClassVar[SeedLifecycleStage]
    SEED_STAGE_CULLED: _ClassVar[SeedLifecycleStage]
    SEED_STAGE_EMBARGOED: _ClassVar[SeedLifecycleStage]
    SEED_STAGE_RESETTING: _ClassVar[SeedLifecycleStage]
    SEED_STAGE_TERMINATED: _ClassVar[SeedLifecycleStage]

class SeedLifecycleGate(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    SEED_GATE_UNKNOWN: _ClassVar[SeedLifecycleGate]
    SEED_GATE_G0_SANITY: _ClassVar[SeedLifecycleGate]
    SEED_GATE_G1_GRADIENT_HEALTH: _ClassVar[SeedLifecycleGate]
    SEED_GATE_G2_STABILITY: _ClassVar[SeedLifecycleGate]
    SEED_GATE_G3_INTERFACE: _ClassVar[SeedLifecycleGate]
    SEED_GATE_G4_SYSTEM_IMPACT: _ClassVar[SeedLifecycleGate]
    SEED_GATE_G5_RESET: _ClassVar[SeedLifecycleGate]

class HealthStatus(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    HEALTH_STATUS_UNKNOWN: _ClassVar[HealthStatus]
    HEALTH_STATUS_HEALTHY: _ClassVar[HealthStatus]
    HEALTH_STATUS_DEGRADED: _ClassVar[HealthStatus]
    HEALTH_STATUS_UNHEALTHY: _ClassVar[HealthStatus]
    HEALTH_STATUS_CRITICAL: _ClassVar[HealthStatus]

class CircuitBreakerState(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    CIRCUIT_STATE_UNKNOWN: _ClassVar[CircuitBreakerState]
    CIRCUIT_STATE_CLOSED: _ClassVar[CircuitBreakerState]
    CIRCUIT_STATE_OPEN: _ClassVar[CircuitBreakerState]
    CIRCUIT_STATE_HALF_OPEN: _ClassVar[CircuitBreakerState]

class CommandType(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    COMMAND_UNKNOWN: _ClassVar[CommandType]
    COMMAND_SEED: _ClassVar[CommandType]
    COMMAND_ROLLBACK: _ClassVar[CommandType]
    COMMAND_OPTIMIZER: _ClassVar[CommandType]
    COMMAND_CIRCUIT_BREAKER: _ClassVar[CommandType]
    COMMAND_PAUSE: _ClassVar[CommandType]
    COMMAND_EMERGENCY: _ClassVar[CommandType]
    COMMAND_STRUCTURAL_PRUNING: _ClassVar[CommandType]

class SeedOperation(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    SEED_OP_UNKNOWN: _ClassVar[SeedOperation]
    SEED_OP_GERMINATE: _ClassVar[SeedOperation]
    SEED_OP_START_TRAINING: _ClassVar[SeedOperation]
    SEED_OP_START_GRAFTING: _ClassVar[SeedOperation]
    SEED_OP_STABILIZE: _ClassVar[SeedOperation]
    SEED_OP_EVALUATE: _ClassVar[SeedOperation]
    SEED_OP_FINE_TUNE: _ClassVar[SeedOperation]
    SEED_OP_FOSSILIZE: _ClassVar[SeedOperation]
    SEED_OP_CULL: _ClassVar[SeedOperation]
    SEED_OP_CANCEL: _ClassVar[SeedOperation]

class MessagePriority(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    MESSAGE_PRIORITY_UNSPECIFIED: _ClassVar[MessagePriority]
    MESSAGE_PRIORITY_LOW: _ClassVar[MessagePriority]
    MESSAGE_PRIORITY_NORMAL: _ClassVar[MessagePriority]
    MESSAGE_PRIORITY_HIGH: _ClassVar[MessagePriority]
    MESSAGE_PRIORITY_CRITICAL: _ClassVar[MessagePriority]

class DeliveryGuarantee(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    DELIVERY_GUARANTEE_UNKNOWN: _ClassVar[DeliveryGuarantee]
    DELIVERY_GUARANTEE_AT_LEAST_ONCE: _ClassVar[DeliveryGuarantee]
    DELIVERY_GUARANTEE_AT_MOST_ONCE: _ClassVar[DeliveryGuarantee]
    DELIVERY_GUARANTEE_EXACTLY_ONCE: _ClassVar[DeliveryGuarantee]

class TelemetryLevel(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    TELEMETRY_LEVEL_UNSPECIFIED: _ClassVar[TelemetryLevel]
    TELEMETRY_LEVEL_INFO: _ClassVar[TelemetryLevel]
    TELEMETRY_LEVEL_WARNING: _ClassVar[TelemetryLevel]
    TELEMETRY_LEVEL_ERROR: _ClassVar[TelemetryLevel]
    TELEMETRY_LEVEL_CRITICAL: _ClassVar[TelemetryLevel]

class EmergencyLevel(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    EMERGENCY_LEVEL_UNSPECIFIED: _ClassVar[EmergencyLevel]
    EMERGENCY_LEVEL_NOTICE: _ClassVar[EmergencyLevel]
    EMERGENCY_LEVEL_ELEVATED: _ClassVar[EmergencyLevel]
    EMERGENCY_LEVEL_CONSERVATIVE: _ClassVar[EmergencyLevel]
    EMERGENCY_LEVEL_HALT: _ClassVar[EmergencyLevel]

class FieldReportOutcome(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    FIELD_REPORT_OUTCOME_UNSPECIFIED: _ClassVar[FieldReportOutcome]
    FIELD_REPORT_OUTCOME_SUCCESS: _ClassVar[FieldReportOutcome]
    FIELD_REPORT_OUTCOME_NEUTRAL: _ClassVar[FieldReportOutcome]
    FIELD_REPORT_OUTCOME_REGRESSION: _ClassVar[FieldReportOutcome]
    FIELD_REPORT_OUTCOME_ABORTED: _ClassVar[FieldReportOutcome]

class BlueprintTier(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    BLUEPRINT_TIER_UNSPECIFIED: _ClassVar[BlueprintTier]
    BLUEPRINT_TIER_SAFE: _ClassVar[BlueprintTier]
    BLUEPRINT_TIER_EXPERIMENTAL: _ClassVar[BlueprintTier]
    BLUEPRINT_TIER_HIGH_RISK: _ClassVar[BlueprintTier]

class BusMessageType(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    BUS_MESSAGE_TYPE_UNSPECIFIED: _ClassVar[BusMessageType]
    BUS_MESSAGE_TYPE_SYSTEM_STATE: _ClassVar[BusMessageType]
    BUS_MESSAGE_TYPE_ADAPTATION_COMMAND: _ClassVar[BusMessageType]
    BUS_MESSAGE_TYPE_FIELD_REPORT: _ClassVar[BusMessageType]
    BUS_MESSAGE_TYPE_TELEMETRY: _ClassVar[BusMessageType]
    BUS_MESSAGE_TYPE_POLICY_UPDATE: _ClassVar[BusMessageType]
    BUS_MESSAGE_TYPE_KERNEL_PREFETCH_REQUEST: _ClassVar[BusMessageType]
    BUS_MESSAGE_TYPE_KERNEL_PREFETCH_READY: _ClassVar[BusMessageType]
    BUS_MESSAGE_TYPE_KERNEL_PREFETCH_ERROR: _ClassVar[BusMessageType]
    BUS_MESSAGE_TYPE_KERNEL_CATALOG_UPDATE: _ClassVar[BusMessageType]
    BUS_MESSAGE_TYPE_EMERGENCY_SIGNAL: _ClassVar[BusMessageType]
    BUS_MESSAGE_TYPE_BSDS_ISSUED: _ClassVar[BusMessageType]
    BUS_MESSAGE_TYPE_BSDS_FAILED: _ClassVar[BusMessageType]
    BUS_MESSAGE_TYPE_BENCHMARK_REPORT: _ClassVar[BusMessageType]

class HazardBand(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    HAZARD_BAND_UNSPECIFIED: _ClassVar[HazardBand]
    HAZARD_BAND_LOW: _ClassVar[HazardBand]
    HAZARD_BAND_MEDIUM: _ClassVar[HazardBand]
    HAZARD_BAND_HIGH: _ClassVar[HazardBand]
    HAZARD_BAND_CRITICAL: _ClassVar[HazardBand]

class HandlingClass(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    HANDLING_CLASS_UNSPECIFIED: _ClassVar[HandlingClass]
    HANDLING_CLASS_STANDARD: _ClassVar[HandlingClass]
    HANDLING_CLASS_RESTRICTED: _ClassVar[HandlingClass]
    HANDLING_CLASS_QUARANTINE: _ClassVar[HandlingClass]

class ResourceProfile(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    RESOURCE_PROFILE_UNSPECIFIED: _ClassVar[ResourceProfile]
    RESOURCE_PROFILE_CPU: _ClassVar[ResourceProfile]
    RESOURCE_PROFILE_GPU: _ClassVar[ResourceProfile]
    RESOURCE_PROFILE_MEMORY_HEAVY: _ClassVar[ResourceProfile]
    RESOURCE_PROFILE_IO_HEAVY: _ClassVar[ResourceProfile]
    RESOURCE_PROFILE_MIXED: _ClassVar[ResourceProfile]

class Provenance(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    PROVENANCE_UNSPECIFIED: _ClassVar[Provenance]
    PROVENANCE_URABRASK: _ClassVar[Provenance]
    PROVENANCE_CURATED: _ClassVar[Provenance]
    PROVENANCE_HEURISTIC: _ClassVar[Provenance]
    PROVENANCE_EXTERNAL: _ClassVar[Provenance]
SEED_STAGE_UNKNOWN: SeedLifecycleStage
SEED_STAGE_DORMANT: SeedLifecycleStage
SEED_STAGE_GERMINATED: SeedLifecycleStage
SEED_STAGE_TRAINING: SeedLifecycleStage
SEED_STAGE_BLENDING: SeedLifecycleStage
SEED_STAGE_SHADOWING: SeedLifecycleStage
SEED_STAGE_PROBATIONARY: SeedLifecycleStage
SEED_STAGE_FOSSILIZED: SeedLifecycleStage
SEED_STAGE_CULLED: SeedLifecycleStage
SEED_STAGE_EMBARGOED: SeedLifecycleStage
SEED_STAGE_RESETTING: SeedLifecycleStage
SEED_STAGE_TERMINATED: SeedLifecycleStage
SEED_GATE_UNKNOWN: SeedLifecycleGate
SEED_GATE_G0_SANITY: SeedLifecycleGate
SEED_GATE_G1_GRADIENT_HEALTH: SeedLifecycleGate
SEED_GATE_G2_STABILITY: SeedLifecycleGate
SEED_GATE_G3_INTERFACE: SeedLifecycleGate
SEED_GATE_G4_SYSTEM_IMPACT: SeedLifecycleGate
SEED_GATE_G5_RESET: SeedLifecycleGate
HEALTH_STATUS_UNKNOWN: HealthStatus
HEALTH_STATUS_HEALTHY: HealthStatus
HEALTH_STATUS_DEGRADED: HealthStatus
HEALTH_STATUS_UNHEALTHY: HealthStatus
HEALTH_STATUS_CRITICAL: HealthStatus
CIRCUIT_STATE_UNKNOWN: CircuitBreakerState
CIRCUIT_STATE_CLOSED: CircuitBreakerState
CIRCUIT_STATE_OPEN: CircuitBreakerState
CIRCUIT_STATE_HALF_OPEN: CircuitBreakerState
COMMAND_UNKNOWN: CommandType
COMMAND_SEED: CommandType
COMMAND_ROLLBACK: CommandType
COMMAND_OPTIMIZER: CommandType
COMMAND_CIRCUIT_BREAKER: CommandType
COMMAND_PAUSE: CommandType
COMMAND_EMERGENCY: CommandType
COMMAND_STRUCTURAL_PRUNING: CommandType
SEED_OP_UNKNOWN: SeedOperation
SEED_OP_GERMINATE: SeedOperation
SEED_OP_START_TRAINING: SeedOperation
SEED_OP_START_GRAFTING: SeedOperation
SEED_OP_STABILIZE: SeedOperation
SEED_OP_EVALUATE: SeedOperation
SEED_OP_FINE_TUNE: SeedOperation
SEED_OP_FOSSILIZE: SeedOperation
SEED_OP_CULL: SeedOperation
SEED_OP_CANCEL: SeedOperation
MESSAGE_PRIORITY_UNSPECIFIED: MessagePriority
MESSAGE_PRIORITY_LOW: MessagePriority
MESSAGE_PRIORITY_NORMAL: MessagePriority
MESSAGE_PRIORITY_HIGH: MessagePriority
MESSAGE_PRIORITY_CRITICAL: MessagePriority
DELIVERY_GUARANTEE_UNKNOWN: DeliveryGuarantee
DELIVERY_GUARANTEE_AT_LEAST_ONCE: DeliveryGuarantee
DELIVERY_GUARANTEE_AT_MOST_ONCE: DeliveryGuarantee
DELIVERY_GUARANTEE_EXACTLY_ONCE: DeliveryGuarantee
TELEMETRY_LEVEL_UNSPECIFIED: TelemetryLevel
TELEMETRY_LEVEL_INFO: TelemetryLevel
TELEMETRY_LEVEL_WARNING: TelemetryLevel
TELEMETRY_LEVEL_ERROR: TelemetryLevel
TELEMETRY_LEVEL_CRITICAL: TelemetryLevel
EMERGENCY_LEVEL_UNSPECIFIED: EmergencyLevel
EMERGENCY_LEVEL_NOTICE: EmergencyLevel
EMERGENCY_LEVEL_ELEVATED: EmergencyLevel
EMERGENCY_LEVEL_CONSERVATIVE: EmergencyLevel
EMERGENCY_LEVEL_HALT: EmergencyLevel
FIELD_REPORT_OUTCOME_UNSPECIFIED: FieldReportOutcome
FIELD_REPORT_OUTCOME_SUCCESS: FieldReportOutcome
FIELD_REPORT_OUTCOME_NEUTRAL: FieldReportOutcome
FIELD_REPORT_OUTCOME_REGRESSION: FieldReportOutcome
FIELD_REPORT_OUTCOME_ABORTED: FieldReportOutcome
BLUEPRINT_TIER_UNSPECIFIED: BlueprintTier
BLUEPRINT_TIER_SAFE: BlueprintTier
BLUEPRINT_TIER_EXPERIMENTAL: BlueprintTier
BLUEPRINT_TIER_HIGH_RISK: BlueprintTier
BUS_MESSAGE_TYPE_UNSPECIFIED: BusMessageType
BUS_MESSAGE_TYPE_SYSTEM_STATE: BusMessageType
BUS_MESSAGE_TYPE_ADAPTATION_COMMAND: BusMessageType
BUS_MESSAGE_TYPE_FIELD_REPORT: BusMessageType
BUS_MESSAGE_TYPE_TELEMETRY: BusMessageType
BUS_MESSAGE_TYPE_POLICY_UPDATE: BusMessageType
BUS_MESSAGE_TYPE_KERNEL_PREFETCH_REQUEST: BusMessageType
BUS_MESSAGE_TYPE_KERNEL_PREFETCH_READY: BusMessageType
BUS_MESSAGE_TYPE_KERNEL_PREFETCH_ERROR: BusMessageType
BUS_MESSAGE_TYPE_KERNEL_CATALOG_UPDATE: BusMessageType
BUS_MESSAGE_TYPE_EMERGENCY_SIGNAL: BusMessageType
BUS_MESSAGE_TYPE_BSDS_ISSUED: BusMessageType
BUS_MESSAGE_TYPE_BSDS_FAILED: BusMessageType
BUS_MESSAGE_TYPE_BENCHMARK_REPORT: BusMessageType
HAZARD_BAND_UNSPECIFIED: HazardBand
HAZARD_BAND_LOW: HazardBand
HAZARD_BAND_MEDIUM: HazardBand
HAZARD_BAND_HIGH: HazardBand
HAZARD_BAND_CRITICAL: HazardBand
HANDLING_CLASS_UNSPECIFIED: HandlingClass
HANDLING_CLASS_STANDARD: HandlingClass
HANDLING_CLASS_RESTRICTED: HandlingClass
HANDLING_CLASS_QUARANTINE: HandlingClass
RESOURCE_PROFILE_UNSPECIFIED: ResourceProfile
RESOURCE_PROFILE_CPU: ResourceProfile
RESOURCE_PROFILE_GPU: ResourceProfile
RESOURCE_PROFILE_MEMORY_HEAVY: ResourceProfile
RESOURCE_PROFILE_IO_HEAVY: ResourceProfile
RESOURCE_PROFILE_MIXED: ResourceProfile
PROVENANCE_UNSPECIFIED: Provenance
PROVENANCE_URABRASK: Provenance
PROVENANCE_CURATED: Provenance
PROVENANCE_HEURISTIC: Provenance
PROVENANCE_EXTERNAL: Provenance

class HardwareContext(_message.Message):
    __slots__ = ("device_type", "device_id", "total_memory_gb", "available_memory_gb", "temperature_celsius", "utilization_percent", "compute_capability")
    DEVICE_TYPE_FIELD_NUMBER: _ClassVar[int]
    DEVICE_ID_FIELD_NUMBER: _ClassVar[int]
    TOTAL_MEMORY_GB_FIELD_NUMBER: _ClassVar[int]
    AVAILABLE_MEMORY_GB_FIELD_NUMBER: _ClassVar[int]
    TEMPERATURE_CELSIUS_FIELD_NUMBER: _ClassVar[int]
    UTILIZATION_PERCENT_FIELD_NUMBER: _ClassVar[int]
    COMPUTE_CAPABILITY_FIELD_NUMBER: _ClassVar[int]
    device_type: str
    device_id: str
    total_memory_gb: float
    available_memory_gb: float
    temperature_celsius: float
    utilization_percent: float
    compute_capability: int
    def __init__(self, device_type: _Optional[str] = ..., device_id: _Optional[str] = ..., total_memory_gb: _Optional[float] = ..., available_memory_gb: _Optional[float] = ..., temperature_celsius: _Optional[float] = ..., utilization_percent: _Optional[float] = ..., compute_capability: _Optional[int] = ...) -> None: ...

class SeedState(_message.Message):
    __slots__ = ("seed_id", "stage", "gradient_norm", "learning_rate", "layer_depth", "metrics", "age_epochs", "risk_score")
    class MetricsEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: float
        def __init__(self, key: _Optional[str] = ..., value: _Optional[float] = ...) -> None: ...
    SEED_ID_FIELD_NUMBER: _ClassVar[int]
    STAGE_FIELD_NUMBER: _ClassVar[int]
    GRADIENT_NORM_FIELD_NUMBER: _ClassVar[int]
    LEARNING_RATE_FIELD_NUMBER: _ClassVar[int]
    LAYER_DEPTH_FIELD_NUMBER: _ClassVar[int]
    METRICS_FIELD_NUMBER: _ClassVar[int]
    AGE_EPOCHS_FIELD_NUMBER: _ClassVar[int]
    RISK_SCORE_FIELD_NUMBER: _ClassVar[int]
    seed_id: str
    stage: SeedLifecycleStage
    gradient_norm: float
    learning_rate: float
    layer_depth: int
    metrics: _containers.ScalarMap[str, float]
    age_epochs: int
    risk_score: float
    def __init__(self, seed_id: _Optional[str] = ..., stage: _Optional[_Union[SeedLifecycleStage, str]] = ..., gradient_norm: _Optional[float] = ..., learning_rate: _Optional[float] = ..., layer_depth: _Optional[int] = ..., metrics: _Optional[_Mapping[str, float]] = ..., age_epochs: _Optional[int] = ..., risk_score: _Optional[float] = ...) -> None: ...

class SystemStatePacket(_message.Message):
    __slots__ = ("version", "current_epoch", "validation_accuracy", "validation_loss", "timestamp_ns", "hardware_context", "training_metrics", "seed_states", "packet_id", "global_step", "training_loss", "source_subsystem", "training_run_id", "experiment_name")
    class TrainingMetricsEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: float
        def __init__(self, key: _Optional[str] = ..., value: _Optional[float] = ...) -> None: ...
    VERSION_FIELD_NUMBER: _ClassVar[int]
    CURRENT_EPOCH_FIELD_NUMBER: _ClassVar[int]
    VALIDATION_ACCURACY_FIELD_NUMBER: _ClassVar[int]
    VALIDATION_LOSS_FIELD_NUMBER: _ClassVar[int]
    TIMESTAMP_NS_FIELD_NUMBER: _ClassVar[int]
    HARDWARE_CONTEXT_FIELD_NUMBER: _ClassVar[int]
    TRAINING_METRICS_FIELD_NUMBER: _ClassVar[int]
    SEED_STATES_FIELD_NUMBER: _ClassVar[int]
    PACKET_ID_FIELD_NUMBER: _ClassVar[int]
    GLOBAL_STEP_FIELD_NUMBER: _ClassVar[int]
    TRAINING_LOSS_FIELD_NUMBER: _ClassVar[int]
    SOURCE_SUBSYSTEM_FIELD_NUMBER: _ClassVar[int]
    TRAINING_RUN_ID_FIELD_NUMBER: _ClassVar[int]
    EXPERIMENT_NAME_FIELD_NUMBER: _ClassVar[int]
    version: int
    current_epoch: int
    validation_accuracy: float
    validation_loss: float
    timestamp_ns: int
    hardware_context: HardwareContext
    training_metrics: _containers.ScalarMap[str, float]
    seed_states: _containers.RepeatedCompositeFieldContainer[SeedState]
    packet_id: str
    global_step: int
    training_loss: float
    source_subsystem: str
    training_run_id: str
    experiment_name: str
    def __init__(self, version: _Optional[int] = ..., current_epoch: _Optional[int] = ..., validation_accuracy: _Optional[float] = ..., validation_loss: _Optional[float] = ..., timestamp_ns: _Optional[int] = ..., hardware_context: _Optional[_Union[HardwareContext, _Mapping]] = ..., training_metrics: _Optional[_Mapping[str, float]] = ..., seed_states: _Optional[_Iterable[_Union[SeedState, _Mapping]]] = ..., packet_id: _Optional[str] = ..., global_step: _Optional[int] = ..., training_loss: _Optional[float] = ..., source_subsystem: _Optional[str] = ..., training_run_id: _Optional[str] = ..., experiment_name: _Optional[str] = ...) -> None: ...

class CommandSeedOperation(_message.Message):
    __slots__ = ("operation", "blueprint_id", "parameters")
    class ParametersEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: float
        def __init__(self, key: _Optional[str] = ..., value: _Optional[float] = ...) -> None: ...
    OPERATION_FIELD_NUMBER: _ClassVar[int]
    BLUEPRINT_ID_FIELD_NUMBER: _ClassVar[int]
    PARAMETERS_FIELD_NUMBER: _ClassVar[int]
    operation: SeedOperation
    blueprint_id: str
    parameters: _containers.ScalarMap[str, float]
    def __init__(self, operation: _Optional[_Union[SeedOperation, str]] = ..., blueprint_id: _Optional[str] = ..., parameters: _Optional[_Mapping[str, float]] = ...) -> None: ...

class CommandOptimizerAdjustment(_message.Message):
    __slots__ = ("optimizer_id", "hyperparameters")
    class HyperparametersEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: float
        def __init__(self, key: _Optional[str] = ..., value: _Optional[float] = ...) -> None: ...
    OPTIMIZER_ID_FIELD_NUMBER: _ClassVar[int]
    HYPERPARAMETERS_FIELD_NUMBER: _ClassVar[int]
    optimizer_id: str
    hyperparameters: _containers.ScalarMap[str, float]
    def __init__(self, optimizer_id: _Optional[str] = ..., hyperparameters: _Optional[_Mapping[str, float]] = ...) -> None: ...

class CommandCircuitBreaker(_message.Message):
    __slots__ = ("desired_state", "rationale")
    DESIRED_STATE_FIELD_NUMBER: _ClassVar[int]
    RATIONALE_FIELD_NUMBER: _ClassVar[int]
    desired_state: CircuitBreakerState
    rationale: str
    def __init__(self, desired_state: _Optional[_Union[CircuitBreakerState, str]] = ..., rationale: _Optional[str] = ...) -> None: ...

class EmergencySignal(_message.Message):
    __slots__ = ("version", "level", "reason", "origin", "triggered_at", "monotonic_time_ms", "run_id", "attributes", "payload_checksum")
    class AttributesEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: str
        def __init__(self, key: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...
    VERSION_FIELD_NUMBER: _ClassVar[int]
    LEVEL_FIELD_NUMBER: _ClassVar[int]
    REASON_FIELD_NUMBER: _ClassVar[int]
    ORIGIN_FIELD_NUMBER: _ClassVar[int]
    TRIGGERED_AT_FIELD_NUMBER: _ClassVar[int]
    MONOTONIC_TIME_MS_FIELD_NUMBER: _ClassVar[int]
    RUN_ID_FIELD_NUMBER: _ClassVar[int]
    ATTRIBUTES_FIELD_NUMBER: _ClassVar[int]
    PAYLOAD_CHECKSUM_FIELD_NUMBER: _ClassVar[int]
    version: int
    level: EmergencyLevel
    reason: str
    origin: str
    triggered_at: _timestamp_pb2.Timestamp
    monotonic_time_ms: int
    run_id: str
    attributes: _containers.ScalarMap[str, str]
    payload_checksum: bytes
    def __init__(self, version: _Optional[int] = ..., level: _Optional[_Union[EmergencyLevel, str]] = ..., reason: _Optional[str] = ..., origin: _Optional[str] = ..., triggered_at: _Optional[_Union[datetime.datetime, _timestamp_pb2.Timestamp, _Mapping]] = ..., monotonic_time_ms: _Optional[int] = ..., run_id: _Optional[str] = ..., attributes: _Optional[_Mapping[str, str]] = ..., payload_checksum: _Optional[bytes] = ...) -> None: ...

class AdaptationCommand(_message.Message):
    __slots__ = ("version", "command_id", "command_type", "target_seed_id", "execution_deadline_ms", "issued_at", "issued_by", "seed_operation", "optimizer_adjustment", "circuit_breaker", "rollback_payload", "annotations")
    class AnnotationsEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: str
        def __init__(self, key: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...
    VERSION_FIELD_NUMBER: _ClassVar[int]
    COMMAND_ID_FIELD_NUMBER: _ClassVar[int]
    COMMAND_TYPE_FIELD_NUMBER: _ClassVar[int]
    TARGET_SEED_ID_FIELD_NUMBER: _ClassVar[int]
    EXECUTION_DEADLINE_MS_FIELD_NUMBER: _ClassVar[int]
    ISSUED_AT_FIELD_NUMBER: _ClassVar[int]
    ISSUED_BY_FIELD_NUMBER: _ClassVar[int]
    SEED_OPERATION_FIELD_NUMBER: _ClassVar[int]
    OPTIMIZER_ADJUSTMENT_FIELD_NUMBER: _ClassVar[int]
    CIRCUIT_BREAKER_FIELD_NUMBER: _ClassVar[int]
    ROLLBACK_PAYLOAD_FIELD_NUMBER: _ClassVar[int]
    ANNOTATIONS_FIELD_NUMBER: _ClassVar[int]
    version: int
    command_id: str
    command_type: CommandType
    target_seed_id: str
    execution_deadline_ms: int
    issued_at: _timestamp_pb2.Timestamp
    issued_by: str
    seed_operation: CommandSeedOperation
    optimizer_adjustment: CommandOptimizerAdjustment
    circuit_breaker: CommandCircuitBreaker
    rollback_payload: _struct_pb2.Struct
    annotations: _containers.ScalarMap[str, str]
    def __init__(self, version: _Optional[int] = ..., command_id: _Optional[str] = ..., command_type: _Optional[_Union[CommandType, str]] = ..., target_seed_id: _Optional[str] = ..., execution_deadline_ms: _Optional[int] = ..., issued_at: _Optional[_Union[datetime.datetime, _timestamp_pb2.Timestamp, _Mapping]] = ..., issued_by: _Optional[str] = ..., seed_operation: _Optional[_Union[CommandSeedOperation, _Mapping]] = ..., optimizer_adjustment: _Optional[_Union[CommandOptimizerAdjustment, _Mapping]] = ..., circuit_breaker: _Optional[_Union[CommandCircuitBreaker, _Mapping]] = ..., rollback_payload: _Optional[_Union[_struct_pb2.Struct, _Mapping]] = ..., annotations: _Optional[_Mapping[str, str]] = ...) -> None: ...

class KernelPrefetchRequest(_message.Message):
    __slots__ = ("request_id", "blueprint_id", "training_run_id", "issued_at")
    REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
    BLUEPRINT_ID_FIELD_NUMBER: _ClassVar[int]
    TRAINING_RUN_ID_FIELD_NUMBER: _ClassVar[int]
    ISSUED_AT_FIELD_NUMBER: _ClassVar[int]
    request_id: str
    blueprint_id: str
    training_run_id: str
    issued_at: _timestamp_pb2.Timestamp
    def __init__(self, request_id: _Optional[str] = ..., blueprint_id: _Optional[str] = ..., training_run_id: _Optional[str] = ..., issued_at: _Optional[_Union[datetime.datetime, _timestamp_pb2.Timestamp, _Mapping]] = ...) -> None: ...

class KernelArtifactReady(_message.Message):
    __slots__ = ("request_id", "blueprint_id", "artifact_ref", "checksum", "guard_digest", "prewarm_p50_ms", "prewarm_p95_ms")
    REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
    BLUEPRINT_ID_FIELD_NUMBER: _ClassVar[int]
    ARTIFACT_REF_FIELD_NUMBER: _ClassVar[int]
    CHECKSUM_FIELD_NUMBER: _ClassVar[int]
    GUARD_DIGEST_FIELD_NUMBER: _ClassVar[int]
    PREWARM_P50_MS_FIELD_NUMBER: _ClassVar[int]
    PREWARM_P95_MS_FIELD_NUMBER: _ClassVar[int]
    request_id: str
    blueprint_id: str
    artifact_ref: str
    checksum: str
    guard_digest: str
    prewarm_p50_ms: float
    prewarm_p95_ms: float
    def __init__(self, request_id: _Optional[str] = ..., blueprint_id: _Optional[str] = ..., artifact_ref: _Optional[str] = ..., checksum: _Optional[str] = ..., guard_digest: _Optional[str] = ..., prewarm_p50_ms: _Optional[float] = ..., prewarm_p95_ms: _Optional[float] = ...) -> None: ...

class KernelArtifactError(_message.Message):
    __slots__ = ("request_id", "blueprint_id", "reason")
    REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
    BLUEPRINT_ID_FIELD_NUMBER: _ClassVar[int]
    REASON_FIELD_NUMBER: _ClassVar[int]
    request_id: str
    blueprint_id: str
    reason: str
    def __init__(self, request_id: _Optional[str] = ..., blueprint_id: _Optional[str] = ..., reason: _Optional[str] = ...) -> None: ...

class KernelCatalogUpdate(_message.Message):
    __slots__ = ("blueprint_id", "artifact_ref", "checksum", "guard_digest", "compile_ms", "prewarm_ms")
    BLUEPRINT_ID_FIELD_NUMBER: _ClassVar[int]
    ARTIFACT_REF_FIELD_NUMBER: _ClassVar[int]
    CHECKSUM_FIELD_NUMBER: _ClassVar[int]
    GUARD_DIGEST_FIELD_NUMBER: _ClassVar[int]
    COMPILE_MS_FIELD_NUMBER: _ClassVar[int]
    PREWARM_MS_FIELD_NUMBER: _ClassVar[int]
    blueprint_id: str
    artifact_ref: str
    checksum: str
    guard_digest: str
    compile_ms: float
    prewarm_ms: float
    def __init__(self, blueprint_id: _Optional[str] = ..., artifact_ref: _Optional[str] = ..., checksum: _Optional[str] = ..., guard_digest: _Optional[str] = ..., compile_ms: _Optional[float] = ..., prewarm_ms: _Optional[float] = ...) -> None: ...

class MitigationAction(_message.Message):
    __slots__ = ("action_type", "rationale")
    ACTION_TYPE_FIELD_NUMBER: _ClassVar[int]
    RATIONALE_FIELD_NUMBER: _ClassVar[int]
    action_type: str
    rationale: str
    def __init__(self, action_type: _Optional[str] = ..., rationale: _Optional[str] = ...) -> None: ...

class FieldReport(_message.Message):
    __slots__ = ("version", "report_id", "command_id", "training_run_id", "seed_id", "blueprint_id", "outcome", "metrics", "observation_window_epochs", "issued_at", "tamiyo_policy_version", "follow_up_actions", "notes")
    class MetricsEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: float
        def __init__(self, key: _Optional[str] = ..., value: _Optional[float] = ...) -> None: ...
    VERSION_FIELD_NUMBER: _ClassVar[int]
    REPORT_ID_FIELD_NUMBER: _ClassVar[int]
    COMMAND_ID_FIELD_NUMBER: _ClassVar[int]
    TRAINING_RUN_ID_FIELD_NUMBER: _ClassVar[int]
    SEED_ID_FIELD_NUMBER: _ClassVar[int]
    BLUEPRINT_ID_FIELD_NUMBER: _ClassVar[int]
    OUTCOME_FIELD_NUMBER: _ClassVar[int]
    METRICS_FIELD_NUMBER: _ClassVar[int]
    OBSERVATION_WINDOW_EPOCHS_FIELD_NUMBER: _ClassVar[int]
    ISSUED_AT_FIELD_NUMBER: _ClassVar[int]
    TAMIYO_POLICY_VERSION_FIELD_NUMBER: _ClassVar[int]
    FOLLOW_UP_ACTIONS_FIELD_NUMBER: _ClassVar[int]
    NOTES_FIELD_NUMBER: _ClassVar[int]
    version: int
    report_id: str
    command_id: str
    training_run_id: str
    seed_id: str
    blueprint_id: str
    outcome: FieldReportOutcome
    metrics: _containers.ScalarMap[str, float]
    observation_window_epochs: int
    issued_at: _timestamp_pb2.Timestamp
    tamiyo_policy_version: str
    follow_up_actions: _containers.RepeatedCompositeFieldContainer[MitigationAction]
    notes: str
    def __init__(self, version: _Optional[int] = ..., report_id: _Optional[str] = ..., command_id: _Optional[str] = ..., training_run_id: _Optional[str] = ..., seed_id: _Optional[str] = ..., blueprint_id: _Optional[str] = ..., outcome: _Optional[_Union[FieldReportOutcome, str]] = ..., metrics: _Optional[_Mapping[str, float]] = ..., observation_window_epochs: _Optional[int] = ..., issued_at: _Optional[_Union[datetime.datetime, _timestamp_pb2.Timestamp, _Mapping]] = ..., tamiyo_policy_version: _Optional[str] = ..., follow_up_actions: _Optional[_Iterable[_Union[MitigationAction, _Mapping]]] = ..., notes: _Optional[str] = ...) -> None: ...

class BlueprintParameterBounds(_message.Message):
    __slots__ = ("min_value", "max_value")
    MIN_VALUE_FIELD_NUMBER: _ClassVar[int]
    MAX_VALUE_FIELD_NUMBER: _ClassVar[int]
    min_value: float
    max_value: float
    def __init__(self, min_value: _Optional[float] = ..., max_value: _Optional[float] = ...) -> None: ...

class BlueprintDescriptor(_message.Message):
    __slots__ = ("blueprint_id", "name", "tier", "allowed_parameters", "risk", "stage", "quarantine_only", "approval_required", "description")
    class AllowedParametersEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: BlueprintParameterBounds
        def __init__(self, key: _Optional[str] = ..., value: _Optional[_Union[BlueprintParameterBounds, _Mapping]] = ...) -> None: ...
    BLUEPRINT_ID_FIELD_NUMBER: _ClassVar[int]
    NAME_FIELD_NUMBER: _ClassVar[int]
    TIER_FIELD_NUMBER: _ClassVar[int]
    ALLOWED_PARAMETERS_FIELD_NUMBER: _ClassVar[int]
    RISK_FIELD_NUMBER: _ClassVar[int]
    STAGE_FIELD_NUMBER: _ClassVar[int]
    QUARANTINE_ONLY_FIELD_NUMBER: _ClassVar[int]
    APPROVAL_REQUIRED_FIELD_NUMBER: _ClassVar[int]
    DESCRIPTION_FIELD_NUMBER: _ClassVar[int]
    blueprint_id: str
    name: str
    tier: BlueprintTier
    allowed_parameters: _containers.MessageMap[str, BlueprintParameterBounds]
    risk: float
    stage: int
    quarantine_only: bool
    approval_required: bool
    description: str
    def __init__(self, blueprint_id: _Optional[str] = ..., name: _Optional[str] = ..., tier: _Optional[_Union[BlueprintTier, str]] = ..., allowed_parameters: _Optional[_Mapping[str, BlueprintParameterBounds]] = ..., risk: _Optional[float] = ..., stage: _Optional[int] = ..., quarantine_only: bool = ..., approval_required: bool = ..., description: _Optional[str] = ...) -> None: ...

class BusEnvelope(_message.Message):
    __slots__ = ("message_type", "payload", "attributes")
    class AttributesEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: str
        def __init__(self, key: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...
    MESSAGE_TYPE_FIELD_NUMBER: _ClassVar[int]
    PAYLOAD_FIELD_NUMBER: _ClassVar[int]
    ATTRIBUTES_FIELD_NUMBER: _ClassVar[int]
    message_type: BusMessageType
    payload: bytes
    attributes: _containers.ScalarMap[str, str]
    def __init__(self, message_type: _Optional[_Union[BusMessageType, str]] = ..., payload: _Optional[bytes] = ..., attributes: _Optional[_Mapping[str, str]] = ...) -> None: ...

class PolicyUpdate(_message.Message):
    __slots__ = ("version", "policy_id", "training_run_id", "issued_at", "tamiyo_policy_version", "payload")
    VERSION_FIELD_NUMBER: _ClassVar[int]
    POLICY_ID_FIELD_NUMBER: _ClassVar[int]
    TRAINING_RUN_ID_FIELD_NUMBER: _ClassVar[int]
    ISSUED_AT_FIELD_NUMBER: _ClassVar[int]
    TAMIYO_POLICY_VERSION_FIELD_NUMBER: _ClassVar[int]
    PAYLOAD_FIELD_NUMBER: _ClassVar[int]
    version: int
    policy_id: str
    training_run_id: str
    issued_at: _timestamp_pb2.Timestamp
    tamiyo_policy_version: str
    payload: bytes
    def __init__(self, version: _Optional[int] = ..., policy_id: _Optional[str] = ..., training_run_id: _Optional[str] = ..., issued_at: _Optional[_Union[datetime.datetime, _timestamp_pb2.Timestamp, _Mapping]] = ..., tamiyo_policy_version: _Optional[str] = ..., payload: _Optional[bytes] = ...) -> None: ...

class MetricPoint(_message.Message):
    __slots__ = ("name", "value", "unit", "attributes")
    class AttributesEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: str
        def __init__(self, key: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...
    NAME_FIELD_NUMBER: _ClassVar[int]
    VALUE_FIELD_NUMBER: _ClassVar[int]
    UNIT_FIELD_NUMBER: _ClassVar[int]
    ATTRIBUTES_FIELD_NUMBER: _ClassVar[int]
    name: str
    value: float
    unit: str
    attributes: _containers.ScalarMap[str, str]
    def __init__(self, name: _Optional[str] = ..., value: _Optional[float] = ..., unit: _Optional[str] = ..., attributes: _Optional[_Mapping[str, str]] = ...) -> None: ...

class TelemetryEvent(_message.Message):
    __slots__ = ("event_id", "description", "level", "attributes")
    class AttributesEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: str
        def __init__(self, key: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...
    EVENT_ID_FIELD_NUMBER: _ClassVar[int]
    DESCRIPTION_FIELD_NUMBER: _ClassVar[int]
    LEVEL_FIELD_NUMBER: _ClassVar[int]
    ATTRIBUTES_FIELD_NUMBER: _ClassVar[int]
    event_id: str
    description: str
    level: TelemetryLevel
    attributes: _containers.ScalarMap[str, str]
    def __init__(self, event_id: _Optional[str] = ..., description: _Optional[str] = ..., level: _Optional[_Union[TelemetryLevel, str]] = ..., attributes: _Optional[_Mapping[str, str]] = ...) -> None: ...

class TraceSpan(_message.Message):
    __slots__ = ("span_id", "parent_span_id", "name", "start_time", "end_time", "attributes")
    class AttributesEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: str
        def __init__(self, key: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...
    SPAN_ID_FIELD_NUMBER: _ClassVar[int]
    PARENT_SPAN_ID_FIELD_NUMBER: _ClassVar[int]
    NAME_FIELD_NUMBER: _ClassVar[int]
    START_TIME_FIELD_NUMBER: _ClassVar[int]
    END_TIME_FIELD_NUMBER: _ClassVar[int]
    ATTRIBUTES_FIELD_NUMBER: _ClassVar[int]
    span_id: str
    parent_span_id: str
    name: str
    start_time: _timestamp_pb2.Timestamp
    end_time: _timestamp_pb2.Timestamp
    attributes: _containers.ScalarMap[str, str]
    def __init__(self, span_id: _Optional[str] = ..., parent_span_id: _Optional[str] = ..., name: _Optional[str] = ..., start_time: _Optional[_Union[datetime.datetime, _timestamp_pb2.Timestamp, _Mapping]] = ..., end_time: _Optional[_Union[datetime.datetime, _timestamp_pb2.Timestamp, _Mapping]] = ..., attributes: _Optional[_Mapping[str, str]] = ...) -> None: ...

class SystemHealth(_message.Message):
    __slots__ = ("status", "summary", "indicators")
    class IndicatorsEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: str
        def __init__(self, key: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...
    STATUS_FIELD_NUMBER: _ClassVar[int]
    SUMMARY_FIELD_NUMBER: _ClassVar[int]
    INDICATORS_FIELD_NUMBER: _ClassVar[int]
    status: HealthStatus
    summary: str
    indicators: _containers.ScalarMap[str, str]
    def __init__(self, status: _Optional[_Union[HealthStatus, str]] = ..., summary: _Optional[str] = ..., indicators: _Optional[_Mapping[str, str]] = ...) -> None: ...

class TelemetryPacket(_message.Message):
    __slots__ = ("packet_id", "timestamp", "source_subsystem", "level", "metrics", "events", "spans", "system_health")
    PACKET_ID_FIELD_NUMBER: _ClassVar[int]
    TIMESTAMP_FIELD_NUMBER: _ClassVar[int]
    SOURCE_SUBSYSTEM_FIELD_NUMBER: _ClassVar[int]
    LEVEL_FIELD_NUMBER: _ClassVar[int]
    METRICS_FIELD_NUMBER: _ClassVar[int]
    EVENTS_FIELD_NUMBER: _ClassVar[int]
    SPANS_FIELD_NUMBER: _ClassVar[int]
    SYSTEM_HEALTH_FIELD_NUMBER: _ClassVar[int]
    packet_id: str
    timestamp: _timestamp_pb2.Timestamp
    source_subsystem: str
    level: TelemetryLevel
    metrics: _containers.RepeatedCompositeFieldContainer[MetricPoint]
    events: _containers.RepeatedCompositeFieldContainer[TelemetryEvent]
    spans: _containers.RepeatedCompositeFieldContainer[TraceSpan]
    system_health: SystemHealth
    def __init__(self, packet_id: _Optional[str] = ..., timestamp: _Optional[_Union[datetime.datetime, _timestamp_pb2.Timestamp, _Mapping]] = ..., source_subsystem: _Optional[str] = ..., level: _Optional[_Union[TelemetryLevel, str]] = ..., metrics: _Optional[_Iterable[_Union[MetricPoint, _Mapping]]] = ..., events: _Optional[_Iterable[_Union[TelemetryEvent, _Mapping]]] = ..., spans: _Optional[_Iterable[_Union[TraceSpan, _Mapping]]] = ..., system_health: _Optional[_Union[SystemHealth, _Mapping]] = ...) -> None: ...

class EventEnvelope(_message.Message):
    __slots__ = ("event_id", "event_type", "source_subsystem", "created_at", "processing_deadline", "ttl", "payload", "payload_type", "content_encoding", "priority", "routing_keys", "correlation_id", "delivery_guarantee", "max_attempts", "current_attempt")
    EVENT_ID_FIELD_NUMBER: _ClassVar[int]
    EVENT_TYPE_FIELD_NUMBER: _ClassVar[int]
    SOURCE_SUBSYSTEM_FIELD_NUMBER: _ClassVar[int]
    CREATED_AT_FIELD_NUMBER: _ClassVar[int]
    PROCESSING_DEADLINE_FIELD_NUMBER: _ClassVar[int]
    TTL_FIELD_NUMBER: _ClassVar[int]
    PAYLOAD_FIELD_NUMBER: _ClassVar[int]
    PAYLOAD_TYPE_FIELD_NUMBER: _ClassVar[int]
    CONTENT_ENCODING_FIELD_NUMBER: _ClassVar[int]
    PRIORITY_FIELD_NUMBER: _ClassVar[int]
    ROUTING_KEYS_FIELD_NUMBER: _ClassVar[int]
    CORRELATION_ID_FIELD_NUMBER: _ClassVar[int]
    DELIVERY_GUARANTEE_FIELD_NUMBER: _ClassVar[int]
    MAX_ATTEMPTS_FIELD_NUMBER: _ClassVar[int]
    CURRENT_ATTEMPT_FIELD_NUMBER: _ClassVar[int]
    event_id: str
    event_type: str
    source_subsystem: str
    created_at: _timestamp_pb2.Timestamp
    processing_deadline: _duration_pb2.Duration
    ttl: _duration_pb2.Duration
    payload: bytes
    payload_type: str
    content_encoding: str
    priority: MessagePriority
    routing_keys: _containers.RepeatedScalarFieldContainer[str]
    correlation_id: str
    delivery_guarantee: DeliveryGuarantee
    max_attempts: int
    current_attempt: int
    def __init__(self, event_id: _Optional[str] = ..., event_type: _Optional[str] = ..., source_subsystem: _Optional[str] = ..., created_at: _Optional[_Union[datetime.datetime, _timestamp_pb2.Timestamp, _Mapping]] = ..., processing_deadline: _Optional[_Union[datetime.timedelta, _duration_pb2.Duration, _Mapping]] = ..., ttl: _Optional[_Union[datetime.timedelta, _duration_pb2.Duration, _Mapping]] = ..., payload: _Optional[bytes] = ..., payload_type: _Optional[str] = ..., content_encoding: _Optional[str] = ..., priority: _Optional[_Union[MessagePriority, str]] = ..., routing_keys: _Optional[_Iterable[str]] = ..., correlation_id: _Optional[str] = ..., delivery_guarantee: _Optional[_Union[DeliveryGuarantee, str]] = ..., max_attempts: _Optional[int] = ..., current_attempt: _Optional[int] = ...) -> None: ...

class PerformanceBudgets(_message.Message):
    __slots__ = ("epoch_boundary_ms", "tamiyo_inference_ms", "systemstate_assembly_ms", "adaptation_processing_ms", "fast_rollback_ms", "full_rollback_seconds", "protobuf_serialization_us", "message_validation_us", "breaker_timeout_ms", "breaker_reset_seconds", "importance_tracking_overhead_percent", "checkpoint_analysis_minutes_min", "checkpoint_analysis_minutes_max", "structured_validation_seconds", "rollback_coordination_seconds")
    EPOCH_BOUNDARY_MS_FIELD_NUMBER: _ClassVar[int]
    TAMIYO_INFERENCE_MS_FIELD_NUMBER: _ClassVar[int]
    SYSTEMSTATE_ASSEMBLY_MS_FIELD_NUMBER: _ClassVar[int]
    ADAPTATION_PROCESSING_MS_FIELD_NUMBER: _ClassVar[int]
    FAST_ROLLBACK_MS_FIELD_NUMBER: _ClassVar[int]
    FULL_ROLLBACK_SECONDS_FIELD_NUMBER: _ClassVar[int]
    PROTOBUF_SERIALIZATION_US_FIELD_NUMBER: _ClassVar[int]
    MESSAGE_VALIDATION_US_FIELD_NUMBER: _ClassVar[int]
    BREAKER_TIMEOUT_MS_FIELD_NUMBER: _ClassVar[int]
    BREAKER_RESET_SECONDS_FIELD_NUMBER: _ClassVar[int]
    IMPORTANCE_TRACKING_OVERHEAD_PERCENT_FIELD_NUMBER: _ClassVar[int]
    CHECKPOINT_ANALYSIS_MINUTES_MIN_FIELD_NUMBER: _ClassVar[int]
    CHECKPOINT_ANALYSIS_MINUTES_MAX_FIELD_NUMBER: _ClassVar[int]
    STRUCTURED_VALIDATION_SECONDS_FIELD_NUMBER: _ClassVar[int]
    ROLLBACK_COORDINATION_SECONDS_FIELD_NUMBER: _ClassVar[int]
    epoch_boundary_ms: int
    tamiyo_inference_ms: int
    systemstate_assembly_ms: int
    adaptation_processing_ms: int
    fast_rollback_ms: int
    full_rollback_seconds: int
    protobuf_serialization_us: int
    message_validation_us: int
    breaker_timeout_ms: int
    breaker_reset_seconds: int
    importance_tracking_overhead_percent: int
    checkpoint_analysis_minutes_min: int
    checkpoint_analysis_minutes_max: int
    structured_validation_seconds: int
    rollback_coordination_seconds: int
    def __init__(self, epoch_boundary_ms: _Optional[int] = ..., tamiyo_inference_ms: _Optional[int] = ..., systemstate_assembly_ms: _Optional[int] = ..., adaptation_processing_ms: _Optional[int] = ..., fast_rollback_ms: _Optional[int] = ..., full_rollback_seconds: _Optional[int] = ..., protobuf_serialization_us: _Optional[int] = ..., message_validation_us: _Optional[int] = ..., breaker_timeout_ms: _Optional[int] = ..., breaker_reset_seconds: _Optional[int] = ..., importance_tracking_overhead_percent: _Optional[int] = ..., checkpoint_analysis_minutes_min: _Optional[int] = ..., checkpoint_analysis_minutes_max: _Optional[int] = ..., structured_validation_seconds: _Optional[int] = ..., rollback_coordination_seconds: _Optional[int] = ...) -> None: ...

class MemoryBudgets(_message.Message):
    __slots__ = ("model_percent", "optimizer_percent", "gradients_percent", "checkpoints_percent", "telemetry_percent", "morphogenetic_percent", "emergency_percent", "importance_statistics_percent", "pruning_metadata_percent")
    MODEL_PERCENT_FIELD_NUMBER: _ClassVar[int]
    OPTIMIZER_PERCENT_FIELD_NUMBER: _ClassVar[int]
    GRADIENTS_PERCENT_FIELD_NUMBER: _ClassVar[int]
    CHECKPOINTS_PERCENT_FIELD_NUMBER: _ClassVar[int]
    TELEMETRY_PERCENT_FIELD_NUMBER: _ClassVar[int]
    MORPHOGENETIC_PERCENT_FIELD_NUMBER: _ClassVar[int]
    EMERGENCY_PERCENT_FIELD_NUMBER: _ClassVar[int]
    IMPORTANCE_STATISTICS_PERCENT_FIELD_NUMBER: _ClassVar[int]
    PRUNING_METADATA_PERCENT_FIELD_NUMBER: _ClassVar[int]
    model_percent: float
    optimizer_percent: float
    gradients_percent: float
    checkpoints_percent: float
    telemetry_percent: float
    morphogenetic_percent: float
    emergency_percent: float
    importance_statistics_percent: float
    pruning_metadata_percent: float
    def __init__(self, model_percent: _Optional[float] = ..., optimizer_percent: _Optional[float] = ..., gradients_percent: _Optional[float] = ..., checkpoints_percent: _Optional[float] = ..., telemetry_percent: _Optional[float] = ..., morphogenetic_percent: _Optional[float] = ..., emergency_percent: _Optional[float] = ..., importance_statistics_percent: _Optional[float] = ..., pruning_metadata_percent: _Optional[float] = ...) -> None: ...

class SystemLimits(_message.Message):
    __slots__ = ("max_seeds_per_epoch", "max_message_size_bytes", "max_queue_depth", "max_retry_attempts", "max_pause_quota", "max_gc_allocations_per_msg", "max_pruning_ratio_percent", "max_consecutive_pruning_failures", "max_checkpoint_storage_gb", "max_importance_history_days")
    MAX_SEEDS_PER_EPOCH_FIELD_NUMBER: _ClassVar[int]
    MAX_MESSAGE_SIZE_BYTES_FIELD_NUMBER: _ClassVar[int]
    MAX_QUEUE_DEPTH_FIELD_NUMBER: _ClassVar[int]
    MAX_RETRY_ATTEMPTS_FIELD_NUMBER: _ClassVar[int]
    MAX_PAUSE_QUOTA_FIELD_NUMBER: _ClassVar[int]
    MAX_GC_ALLOCATIONS_PER_MSG_FIELD_NUMBER: _ClassVar[int]
    MAX_PRUNING_RATIO_PERCENT_FIELD_NUMBER: _ClassVar[int]
    MAX_CONSECUTIVE_PRUNING_FAILURES_FIELD_NUMBER: _ClassVar[int]
    MAX_CHECKPOINT_STORAGE_GB_FIELD_NUMBER: _ClassVar[int]
    MAX_IMPORTANCE_HISTORY_DAYS_FIELD_NUMBER: _ClassVar[int]
    max_seeds_per_epoch: int
    max_message_size_bytes: int
    max_queue_depth: int
    max_retry_attempts: int
    max_pause_quota: int
    max_gc_allocations_per_msg: int
    max_pruning_ratio_percent: int
    max_consecutive_pruning_failures: int
    max_checkpoint_storage_gb: int
    max_importance_history_days: int
    def __init__(self, max_seeds_per_epoch: _Optional[int] = ..., max_message_size_bytes: _Optional[int] = ..., max_queue_depth: _Optional[int] = ..., max_retry_attempts: _Optional[int] = ..., max_pause_quota: _Optional[int] = ..., max_gc_allocations_per_msg: _Optional[int] = ..., max_pruning_ratio_percent: _Optional[int] = ..., max_consecutive_pruning_failures: _Optional[int] = ..., max_checkpoint_storage_gb: _Optional[int] = ..., max_importance_history_days: _Optional[int] = ...) -> None: ...

class BSDS(_message.Message):
    __slots__ = ("version", "blueprint_id", "risk_score", "hazard_band", "handling_class", "resource_profile", "recommendation", "issued_at", "provenance")
    VERSION_FIELD_NUMBER: _ClassVar[int]
    BLUEPRINT_ID_FIELD_NUMBER: _ClassVar[int]
    RISK_SCORE_FIELD_NUMBER: _ClassVar[int]
    HAZARD_BAND_FIELD_NUMBER: _ClassVar[int]
    HANDLING_CLASS_FIELD_NUMBER: _ClassVar[int]
    RESOURCE_PROFILE_FIELD_NUMBER: _ClassVar[int]
    RECOMMENDATION_FIELD_NUMBER: _ClassVar[int]
    ISSUED_AT_FIELD_NUMBER: _ClassVar[int]
    PROVENANCE_FIELD_NUMBER: _ClassVar[int]
    version: int
    blueprint_id: str
    risk_score: float
    hazard_band: HazardBand
    handling_class: HandlingClass
    resource_profile: ResourceProfile
    recommendation: str
    issued_at: _timestamp_pb2.Timestamp
    provenance: Provenance
    def __init__(self, version: _Optional[int] = ..., blueprint_id: _Optional[str] = ..., risk_score: _Optional[float] = ..., hazard_band: _Optional[_Union[HazardBand, str]] = ..., handling_class: _Optional[_Union[HandlingClass, str]] = ..., resource_profile: _Optional[_Union[ResourceProfile, str]] = ..., recommendation: _Optional[str] = ..., issued_at: _Optional[_Union[datetime.datetime, _timestamp_pb2.Timestamp, _Mapping]] = ..., provenance: _Optional[_Union[Provenance, str]] = ...) -> None: ...

class BlueprintBenchmarkProfile(_message.Message):
    __slots__ = ("name", "p50_latency_ms", "p95_latency_ms", "throughput_samples_per_s")
    NAME_FIELD_NUMBER: _ClassVar[int]
    P50_LATENCY_MS_FIELD_NUMBER: _ClassVar[int]
    P95_LATENCY_MS_FIELD_NUMBER: _ClassVar[int]
    THROUGHPUT_SAMPLES_PER_S_FIELD_NUMBER: _ClassVar[int]
    name: str
    p50_latency_ms: float
    p95_latency_ms: float
    throughput_samples_per_s: float
    def __init__(self, name: _Optional[str] = ..., p50_latency_ms: _Optional[float] = ..., p95_latency_ms: _Optional[float] = ..., throughput_samples_per_s: _Optional[float] = ...) -> None: ...

class BlueprintBenchmark(_message.Message):
    __slots__ = ("version", "blueprint_id", "profiles", "device", "torch_version")
    VERSION_FIELD_NUMBER: _ClassVar[int]
    BLUEPRINT_ID_FIELD_NUMBER: _ClassVar[int]
    PROFILES_FIELD_NUMBER: _ClassVar[int]
    DEVICE_FIELD_NUMBER: _ClassVar[int]
    TORCH_VERSION_FIELD_NUMBER: _ClassVar[int]
    version: int
    blueprint_id: str
    profiles: _containers.RepeatedCompositeFieldContainer[BlueprintBenchmarkProfile]
    device: str
    torch_version: str
    def __init__(self, version: _Optional[int] = ..., blueprint_id: _Optional[str] = ..., profiles: _Optional[_Iterable[_Union[BlueprintBenchmarkProfile, _Mapping]]] = ..., device: _Optional[str] = ..., torch_version: _Optional[str] = ...) -> None: ...

class BSDSIssued(_message.Message):
    __slots__ = ("bsds",)
    BSDS_FIELD_NUMBER: _ClassVar[int]
    bsds: BSDS
    def __init__(self, bsds: _Optional[_Union[BSDS, _Mapping]] = ...) -> None: ...

class BSDSFailed(_message.Message):
    __slots__ = ("blueprint_id", "reason")
    BLUEPRINT_ID_FIELD_NUMBER: _ClassVar[int]
    REASON_FIELD_NUMBER: _ClassVar[int]
    blueprint_id: str
    reason: str
    def __init__(self, blueprint_id: _Optional[str] = ..., reason: _Optional[str] = ...) -> None: ...

class BenchmarkReport(_message.Message):
    __slots__ = ("benchmark",)
    BENCHMARK_FIELD_NUMBER: _ClassVar[int]
    benchmark: BlueprintBenchmark
    def __init__(self, benchmark: _Optional[_Union[BlueprintBenchmark, _Mapping]] = ...) -> None: ...
