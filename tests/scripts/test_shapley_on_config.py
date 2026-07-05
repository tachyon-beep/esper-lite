"""Pin the ON-arm A/B config to the frozen F2 calibration.

The paired OFF/ON A/B's pre-registration (docs/analysis/
2026-07-03-shapley-ab-scoring-preregistration.md) adjudicates against EXACTLY
these knob values; a silent value edit would run a different treatment under
the registered scoring rules. This test is the tripwire (adversarial-review
finding on bb8d3c90; pattern precedent: test_pin_e_placebo_run.py).
"""

import json
from pathlib import Path

from esper.simic.training.config import TrainingConfig

ON_CONFIG = Path("configs/config-3slot-3seed-baseline-shaped-shapley-on.json")
OFF_CONFIG = Path("configs/config-3slot-3seed-baseline-shaped.json")

# The F2 freeze (gate esper-lite-f22a1d48a7 comment #105; PDR-0018/0019).
FROZEN_KNOBS = {
    "shapley_synergy_scale": 1.0,
    "shapley_synergy_noise_floor": 0.28,  # tau, ACCEPT-PROVISIONAL lower bound
    "shapley_synergy_cap": 5.0,
    "shapley_synergy_std_floor": 0.25,
    "shapley_synergy_normalized_cap": 3.0,
}


def test_on_config_pins_the_frozen_f2_knobs():
    config = TrainingConfig.from_dict(json.loads(ON_CONFIG.read_text()))
    for name, frozen in FROZEN_KNOBS.items():
        assert getattr(config, name) == frozen, (
            f"{name} drifted from the frozen F2 calibration: "
            f"{getattr(config, name)} != {frozen} — the A/B pre-registration "
            f"adjudicates the frozen treatment; re-freeze via a new PDR, "
            f"do not edit in place"
        )


def test_on_config_differs_from_off_only_in_the_five_knobs():
    """The paired arms must be identical except the treatment flags."""
    on = json.loads(ON_CONFIG.read_text())
    off = json.loads(OFF_CONFIG.read_text())
    assert {k: v for k, v in on.items() if k not in FROZEN_KNOBS} == off


def test_off_config_is_the_untreated_arm():
    config = TrainingConfig.from_dict(json.loads(OFF_CONFIG.read_text()))
    assert config.shapley_synergy_scale == 0.0
