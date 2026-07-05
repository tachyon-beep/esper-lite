"""Percentage-point (pp) units guard for the Committed-Shapley coalition eval (WI-9b).

The Committed-Shapley top-up is computed entirely in PERCENTAGE POINTS: the fused-val
unpack in ``VectorizedPPOTrainer._run_fused_val_pass`` does

    acc = 100.0 * correct_counts[cfg_idx] / total          # vectorized_trainer.py ~:1540

and stores ``acc`` into ``committed_shapley_accs`` (~:1601), which feeds
``compute_committed_shapley_topup`` whose ``cap``/``tau`` deadbands are pp-calibrated.
If that ``100.0 *`` scaling were ever dropped (accs returned as fractions in [0, 1]),
the Shapley gap would silently shrink 100x while cap/tau stayed pp-scale — a silent,
high-severity corruption of every credit. These tests pin the convention so such a
refactor fails LOUDLY.

Two complementary guards (per advisor):
  1. Behavioral counts-contract on ``process_fused_val_batch`` — the only thing pinning
     that the fused-val helper returns integer CORRECT COUNTS (not a pre-scaled fraction),
     so the trainer's downstream ``100.0 *`` yields pp. Catches the "helper refactored to
     return a fraction" path.
  2. Source canary on the trainer unpack — directly asserts the ``100.0 *`` pp scaling is
     still applied on line ~1540 and that its result flows into the Shapley table. Catches
     the "delete the ×100 in place" path, which the behavioral test cannot reach without
     constructing the full trainer.

CPU-only, no CUDA requirement.
"""

import inspect
import types

import torch
import torch.nn as nn

from esper.kasmina.host import CNNHost, MorphogeneticModel
from esper.simic.training.counterfactual_eval import process_fused_val_batch
from esper.simic.training.parallel_env_state import ParallelEnvState
from esper.simic.training.vectorized import loss_and_correct
from esper.simic.training.vectorized_trainer import VectorizedPPOTrainer


def _env_state(model: MorphogeneticModel) -> ParallelEnvState:
    """Minimal CPU ParallelEnvState carrying just the model (no trainer)."""
    return ParallelEnvState(
        model=model,
        host_optimizer=torch.optim.SGD(model.parameters(), lr=0.0),
        signal_tracker=None,
        governor=None,
        env_device="cpu",
        stream=None,
    )


def _tiny_model() -> MorphogeneticModel:
    host = CNNHost(n_blocks=3, memory_format=torch.contiguous_format)
    model = MorphogeneticModel(host, slots=["r0c0", "r0c1"])
    model.eval()
    return model


# ---------------------------------------------------------------------------
# (1) Behavioral: process_fused_val_batch returns integer COUNTS, and the
#     documented downstream 100* conversion yields pp in [0, 100].
# ---------------------------------------------------------------------------


def _run(model: MorphogeneticModel, x: torch.Tensor, targets: torch.Tensor):
    task_spec = types.SimpleNamespace(task_type="classification")
    criterion = nn.CrossEntropyLoss(reduction="none")
    return process_fused_val_batch(
        _env_state(model),
        x,
        targets,
        criterion,
        {},  # no alpha overrides: dormant slots pass through
        1,  # num_configs
        task_spec=task_spec,
        loss_and_correct_fn=loss_and_correct,
    )


def test_fused_val_returns_integer_counts_all_correct() -> None:
    """All-correct batch => correct == total (a COUNT, not the fraction 1.0); pp == 100.0."""
    torch.manual_seed(0)
    model = _tiny_model()
    batch = 16
    x = torch.randn(batch, 3, 32, 32)
    with torch.inference_mode():
        predicted = model.fused_forward(x, {}).argmax(dim=1)
    targets = predicted.clone()  # force 100% correct

    _loss, correct_per_config, total_per_config = _run(model, x, targets)

    correct = int(correct_per_config[0].item())
    assert correct == batch, "helper must return the CORRECT COUNT (== batch), not a fraction"
    assert total_per_config == batch
    assert correct > 1, "count must be a real count (>1), so a [0,1] fraction refactor breaks this"

    # Downstream pp conversion, exactly as vectorized_trainer.py:1540.
    acc_pp = 100.0 * correct / total_per_config
    assert acc_pp == 100.0
    assert 0.0 <= acc_pp <= 100.0


def test_fused_val_returns_integer_counts_half_correct() -> None:
    """Half-correct batch => correct == batch/2 (a COUNT); pp == 50.0, in [0, 100]."""
    torch.manual_seed(0)
    model = _tiny_model()
    batch = 16
    x = torch.randn(batch, 3, 32, 32)
    with torch.inference_mode():
        predicted = model.fused_forward(x, {}).argmax(dim=1)
    targets = predicted.clone()
    # Flip the second half to a definitely-wrong class => exactly batch/2 correct.
    targets[batch // 2 :] = (predicted[batch // 2 :] + 1) % 10

    _loss, correct_per_config, total_per_config = _run(model, x, targets)

    correct = int(correct_per_config[0].item())
    expected = batch - batch // 2
    assert correct == expected, "helper must return the CORRECT COUNT, not a fraction"
    assert total_per_config == batch

    acc_pp = 100.0 * correct / total_per_config
    assert acc_pp == 50.0, "pp accuracy must be 50.0 (percentage points), NOT 0.5 (fraction)"
    assert 0.0 <= acc_pp <= 100.0


# ---------------------------------------------------------------------------
# (2) Source canary: the trainer unpack still applies the 100* pp scaling, and
#     the pp value flows into the Committed-Shapley coalition table.
# ---------------------------------------------------------------------------


def test_trainer_unpack_applies_pp_scaling_convention() -> None:
    """Directly pin the ``100.0 *`` pp scaling on the fused-val unpack line.

    This is a convention canary: it fails LOUDLY if the ×100 scaling is dropped (a
    silent 100x Shapley-gap corruption) — the one corruption path the behavioral
    counts-contract test above cannot reach without the full trainer. It will also
    fire on a legitimate rename/extract of that line; that is the intended safe
    direction: re-confirm the accs are still ×100 percentage points before updating
    the expected token below.
    """
    source = inspect.getsource(VectorizedPPOTrainer._run_fused_val_pass)
    collapsed = "".join(source.split())  # whitespace-insensitive

    assert "acc=100.0*correct_counts[cfg_idx]/total" in collapsed, (
        "pp convention guard: the fused-val unpack must scale correct/total by 100.0 "
        "(percentage points). Committed-Shapley cap/tau are pp-calibrated; dropping the "
        "×100 silently corrupts the Shapley gap by 100x."
    )
    # The pp-scaled acc is what populates the coalition table consumed by
    # compute_committed_shapley_topup — pin that the pp value (not a fraction) flows in.
    assert 'committed_shapley_accs[i][cfg["_subset"]]=acc' in collapsed, (
        "the pp-scaled 'acc' must be stored into committed_shapley_accs (the Shapley "
        "coalition table); if this seam moved, re-verify it still carries percentage points."
    )
