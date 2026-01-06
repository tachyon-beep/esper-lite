from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any, Callable, cast

import torch
import torch.amp as torch_amp

from esper.karn.health import HealthMonitor
from esper.kasmina.host import MorphogeneticModel
from esper.leyline import (
    DEFAULT_GOVERNOR_ABSOLUTE_THRESHOLD,
    DEFAULT_GOVERNOR_DEATH_PENALTY,
    DEFAULT_GOVERNOR_HISTORY_WINDOW,
    DEFAULT_GOVERNOR_SENSITIVITY,
    DEFAULT_MIN_PANICS_BEFORE_ROLLBACK,
    GateLevel,
    SeedSlotProtocol,
)
from esper.nissa import BlueprintAnalytics
from esper.simic.attribution import CounterfactualHelper
from esper.simic.telemetry.emitters import apply_slot_telemetry, emit_with_env_context
from esper.tamiyo import SignalTracker
from esper.tolaria import TolariaGovernor

from .parallel_env_state import ParallelEnvState


@dataclass(frozen=True)
class EnvFactory:
    env_device_map: list[str]
    task_spec: Any
    slots: list[str]
    permissive_gates: bool
    auto_forward_gates: frozenset[GateLevel]
    analytics: BlueprintAnalytics
    use_telemetry: bool
    ops_telemetry_enabled: bool
    telemetry_lifecycle_only: bool
    hub: Any
    create_model: Callable[..., Any]
    action_enum: Any
    gpu_preload_augment: bool
    use_grad_scaler: bool
    amp_enabled: bool
    resolved_amp_dtype: torch.dtype | None

    def make_telemetry_callback(
        self, env_idx: int, device: str
    ) -> Callable[[Any], None]:
        if not self.hub:
            return lambda _: None

        def callback(event: Any) -> None:
            emit_with_env_context(self.hub, env_idx, device, event)

        return callback

    def configure_slot_telemetry(
        self,
        env_state: ParallelEnvState,
        inner_epoch: int | None = None,
        global_epoch: int | None = None,
    ) -> None:
        apply_slot_telemetry(
            env_state,
            ops_telemetry_enabled=self.ops_telemetry_enabled,
            lifecycle_only=self.telemetry_lifecycle_only,
            inner_epoch=inner_epoch,
            global_epoch=global_epoch,
        )

    def create_env_state(self, env_idx: int, base_seed: int) -> ParallelEnvState:
        env_device = self.env_device_map[env_idx]
        torch.manual_seed(base_seed + env_idx * 1000)
        random.seed(base_seed + env_idx * 1000)

        model_raw = self.create_model(
            task=self.task_spec,
            device=env_device,
            slots=self.slots,
            permissive_gates=self.permissive_gates,
        )
        assert isinstance(model_raw, MorphogeneticModel)
        model: MorphogeneticModel = model_raw

        model = model.to(memory_format=torch.channels_last)

        telemetry_cb = self.make_telemetry_callback(env_idx, env_device)
        for slot_module in model.seed_slots.values():
            slot = cast(SeedSlotProtocol, slot_module)
            slot.on_telemetry = telemetry_cb
            slot.fast_mode = False
            slot.auto_forward_gates = self.auto_forward_gates
            slot.isolate_gradients = True

        host_params = sum(
            p.numel() for p in model.get_host_parameters() if p.requires_grad
        )
        self.analytics.set_host_params(env_idx, host_params)

        host_optimizer = torch.optim.SGD(
            model.get_host_parameters(),
            lr=self.task_spec.host_lr,
            momentum=0.9,
            weight_decay=5e-4,
        )

        env_device_obj = torch.device(env_device)
        stream = (
            torch.cuda.Stream(device=env_device_obj)  # type: ignore[no-untyped-call]
            if env_device_obj.type == "cuda"
            else None
        )

        augment_generator = None
        if self.gpu_preload_augment:
            augment_generator = torch.Generator(device=env_device)
            augment_generator.manual_seed(base_seed + env_idx * 1009)

        env_scaler = (
            torch_amp.GradScaler("cuda", enabled=self.use_grad_scaler)  # type: ignore[attr-defined]
            if env_device_obj.type == "cuda" and self.use_grad_scaler
            else None
        )

        autocast_enabled = self.amp_enabled and env_device_obj.type == "cuda"

        random_guess_loss = None
        if self.task_spec.task_type == "classification" and self.task_spec.num_classes:
            random_guess_loss = math.log(self.task_spec.num_classes)
        elif self.task_spec.task_type == "lm" and self.task_spec.vocab_size:
            random_guess_loss = math.log(self.task_spec.vocab_size)

        governor = TolariaGovernor(
            model=model,
            sensitivity=DEFAULT_GOVERNOR_SENSITIVITY,
            absolute_threshold=DEFAULT_GOVERNOR_ABSOLUTE_THRESHOLD,
            death_penalty=DEFAULT_GOVERNOR_DEATH_PENALTY,
            history_window=DEFAULT_GOVERNOR_HISTORY_WINDOW,
            min_panics_before_rollback=DEFAULT_MIN_PANICS_BEFORE_ROLLBACK,
            random_guess_loss=random_guess_loss,
        )
        governor.snapshot()

        health_monitor = (
            HealthMonitor(
                store=None,
                emit_callback=telemetry_cb,
            )
            if self.use_telemetry
            else None
        )

        counterfactual_helper = (
            CounterfactualHelper(
                strategy="auto",
                shapley_samples=20,
                emit_callback=telemetry_cb,
                seed=base_seed,
            )
            if self.use_telemetry
            else None
        )

        env_state = ParallelEnvState(
            model=model,
            host_optimizer=host_optimizer,
            signal_tracker=SignalTracker(env_id=env_idx),
            governor=governor,
            health_monitor=health_monitor,
            counterfactual_helper=counterfactual_helper,
            env_device=env_device,
            stream=stream,
            augment_generator=augment_generator,
            scaler=env_scaler,
            seeds_created=0,
            episode_rewards=[],
            action_enum=self.action_enum,
            telemetry_cb=telemetry_cb,
            autocast_enabled=autocast_enabled,
        )
        env_state.prev_slot_alphas = {slot_id: 0.0 for slot_id in self.slots}
        env_state.prev_slot_params = {slot_id: 0 for slot_id in self.slots}
        env_state.init_accumulators(self.slots)
        self.configure_slot_telemetry(env_state)
        return env_state
