"""seed_lr_override plumbing (PIN-E WI-4, esper-lite-94869250f1).

The placebo run freezes its seeds by setting seed_lr=0 for the whole run
(requires_grad stays True so G2's gradient measurement is satisfied). TaskSpec
resolves internally by task name, so the override is applied to the resolved
spec via dataclasses.replace. The live-optimizer assertion (lr==0 on
env_state.seed_optimizers built by batch_ops) lives in the WI-5 integration
test — this file covers the pure resolution helper.
"""

import pytest

from esper.runtime import get_task_spec
from esper.simic.training.vectorized import apply_seed_lr_override


def test_none_override_returns_spec_unchanged():
    spec = get_task_spec("cifar_baseline")
    assert apply_seed_lr_override(spec, None) is spec


def test_override_replaces_seed_lr_and_nothing_else():
    spec = get_task_spec("cifar_baseline")
    overridden = apply_seed_lr_override(spec, 0.0)

    assert overridden.seed_lr == 0.0
    assert spec.seed_lr == 0.01  # original untouched (replace = copy)
    assert overridden.name == spec.name
    assert overridden.host_lr == spec.host_lr
    assert overridden.topology == spec.topology
    # __post_init__ re-derives the action enum on the replaced copy.
    assert len(overridden.action_enum) == len(spec.action_enum)


def test_negative_override_is_rejected():
    spec = get_task_spec("cifar_baseline")
    with pytest.raises(ValueError, match="seed_lr_override"):
        apply_seed_lr_override(spec, -0.1)
