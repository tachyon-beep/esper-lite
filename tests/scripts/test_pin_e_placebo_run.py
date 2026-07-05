"""PIN-E placebo run driver (WI-6, esper-lite-94869250f1).

The driver is the only sanctioned launcher for the placebo noise-floor runs:
it builds per-seed train kwargs from configs/config-pin-e-placebo.json, hard
preflights the invariants the measurement depends on, and writes the
pin_e_preflight.json provenance file the analyzer refuses to run without.
"""

import importlib.util
import json
from pathlib import Path

import pytest

from esper.leyline.proof_baselines import (
    FIXED_SCHEDULE_HOLD_PLACEBO_3SLOT_HASH,
    FIXED_SCHEDULE_HOLD_PLACEBO_3SLOT_V1,
)

_DRIVER_PATH = Path(__file__).parents[2] / "scripts" / "pin_e_placebo_run.py"
_SPEC = importlib.util.spec_from_file_location("pin_e_placebo_run", _DRIVER_PATH)
assert _SPEC is not None
assert _SPEC.loader is not None
_DRIVER = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_DRIVER)
DEFAULT_CONFIG = str(Path(__file__).parents[2] / _DRIVER.DEFAULT_CONFIG)
_build_run_kwargs = _DRIVER._build_run_kwargs
_load_base_kwargs = _DRIVER._load_base_kwargs
_preflight = _DRIVER._preflight
_write_preflight = _DRIVER._write_preflight


def _good_kwargs(seed: int = 41) -> dict:
    base = _load_base_kwargs(DEFAULT_CONFIG)
    return _build_run_kwargs(
        base,
        seed=seed,
        device="cpu",
        telemetry_base="/tmp/pin_e_test",
        init_std=1e-3,
    )


def test_build_run_kwargs_pins_the_measurement_invariants():
    kwargs = _good_kwargs()
    assert kwargs["seed_lr_override"] == 0.0
    assert kwargs["proof_baseline_lifecycle_policy"] == "apply_declared_lifecycle_schedule"
    assert kwargs["proof_baseline_schedule_id"] == FIXED_SCHEDULE_HOLD_PLACEBO_3SLOT_V1
    assert kwargs["proof_baseline_schedule_hash"] == FIXED_SCHEDULE_HOLD_PLACEBO_3SLOT_HASH
    assert kwargs["use_telemetry"] is True
    assert kwargs["compile_mode"] == "off"
    assert kwargs["shapley_synergy_scale"] == 0.0
    assert kwargs["slots"] == ["r0c0", "r0c1", "r0c2"]
    assert kwargs["max_seeds"] == 3
    # The declared schedule owns every stage transition: auto-forward gates
    # would advance seeds ahead of the scheduled ADVANCE epochs.
    assert kwargs["auto_forward_g1"] is False
    assert kwargs["auto_forward_g2"] is False
    assert kwargs["auto_forward_g3"] is False
    assert "std0.001_s41" in kwargs["telemetry_dir"]
    assert kwargs["group_id"] == "pin_e_std0.001_s41"
    _preflight(kwargs)  # must not raise on the driver's own output


@pytest.mark.parametrize(
    ("key", "value", "match"),
    [
        ("compile_mode", "default", "compile"),
        ("use_telemetry", False, "telemetry"),
        ("seed_lr_override", 0.01, "seed_lr_override"),
        ("proof_baseline_schedule_id", "fixed-schedule-germinate-r0c0-v1", "schedule"),
        ("shapley_synergy_scale", 0.1, "shapley_synergy_scale"),
        ("auto_forward_g1", True, "auto_forward"),
        ("amp", True, "amp"),
        ("max_seeds", 2, "max_seeds"),
    ],
)
def test_preflight_rejects_broken_invariants(key: str, value, match: str):
    kwargs = _good_kwargs()
    kwargs[key] = value
    with pytest.raises(ValueError, match=match):
        _preflight(kwargs)


def test_preflight_rejects_short_runs():
    kwargs = _good_kwargs()
    kwargs["max_epochs"] = 50  # all-HOLDING window starts at epoch 51
    with pytest.raises(ValueError, match="max_epochs"):
        _preflight(kwargs)


def test_write_preflight_provenance(tmp_path):
    kwargs = _good_kwargs()
    kwargs["telemetry_dir"] = str(tmp_path / "placebo_std0.001_s41")
    path = _write_preflight(kwargs, init_std=1e-3, config_path=DEFAULT_CONFIG)
    payload = json.loads(path.read_text())
    assert payload["schedule_id"] == FIXED_SCHEDULE_HOLD_PLACEBO_3SLOT_V1
    assert payload["schedule_hash"] == FIXED_SCHEDULE_HOLD_PLACEBO_3SLOT_HASH
    assert payload["seed_lr_override"] == 0.0
    assert payload["shapley_synergy_scale"] == 0.0
    assert payload["placebo_init_std"] == 1e-3
    assert payload["seed"] == 41
    assert payload["group_id"] == "pin_e_std0.001_s41"
