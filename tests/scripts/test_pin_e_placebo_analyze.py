"""PIN-E placebo noise-floor analyzer (WI-7, esper-lite-94869250f1).

The analyzer assembles the 2^3 counterfactual factorial logged in
COUNTERFACTUAL_MATRIX_COMPUTED events into the terminal signed (phi - c_paid)
distribution for the three null placebo players, and recommends tau = P99 with
a block-bootstrap CI. These tests pin the Shapley arithmetic against a
hand-computed 2^3 example, the provenance gate, terminal selection, bootstrap
reproducibility, arm grouping, and the exit-code contract (0 ok / 1 provenance
refusal / 2 degenerate).
"""

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from esper.leyline.proof_baselines import (
    FIXED_SCHEDULE_HOLD_PLACEBO_3SLOT_HASH,
    FIXED_SCHEDULE_HOLD_PLACEBO_3SLOT_V1,
)

_ANALYZE_PATH = Path(__file__).parents[2] / "scripts" / "pin_e_placebo_analyze.py"
_SPEC = importlib.util.spec_from_file_location("pin_e_placebo_analyze", _ANALYZE_PATH)
assert _SPEC is not None
assert _SPEC.loader is not None
_A = importlib.util.module_from_spec(_SPEC)
# Register before exec so @dataclass can resolve the module's future annotations.
sys.modules[_SPEC.name] = _A
_SPEC.loader.exec_module(_A)


# --------------------------------------------------------------------------
# Fixture builders
# --------------------------------------------------------------------------
def _matrix_event(env_id, episode_idx, slot_ids, mask_acc, *, strategy="full_factorial"):
    """One COUNTERFACTUAL_MATRIX_COMPUTED event. mask_acc: list[(mask_tuple, acc)]."""
    configs = [{"seed_mask": list(mask), "accuracy": acc} for mask, acc in mask_acc]
    return {
        "event_type": "COUNTERFACTUAL_MATRIX_COMPUTED",
        "group_id": "g",
        "epoch": None,
        "data": {
            "env_id": env_id,
            "slot_ids": list(slot_ids),
            "configs": configs,
            "episode_idx": episode_idx,
            "strategy": strategy,
            "compute_time_ms": 0.0,
        },
    }


# The hand-computed 2^3 worked example. Slots (a, b, c) positional.
#   v(empty)=50  v(a)=52  v(b)=51  v(c)=50
#   v(ab)=54     v(ac)=53 v(bc)=52 v(abc)=56
#
# Shapley phi(s) = 1/3 (v{s}-v{}) + 1/6 [(v{s,t}-v{t}) + (v{s,u}-v{u})]
#                  + 1/3 (v(C)-v{t,u}):
#   phi(a) = 1/3(52-50) + 1/6[(54-51)+(53-50)] + 1/3(56-52)
#          = 1/3(2)     + 1/6[3+3]             + 1/3(4)     = 0.6667 + 1.0 + 1.3333 = 3.0
#   phi(b) = 1/3(51-50) + 1/6[(54-52)+(52-50)] + 1/3(56-53)
#          = 1/3(1)     + 1/6[2+2]             + 1/3(3)     = 0.3333 + 0.6667 + 1.0 = 2.0
#   phi(c) = 1/3(50-50) + 1/6[(53-52)+(52-51)] + 1/3(56-54)
#          = 0          + 1/6[1+1]             + 1/3(2)     = 0.0 + 0.3333 + 0.6667 = 1.0
# c_paid(s) = v{s}-v{}:  c_paid(a)=2, c_paid(b)=1, c_paid(c)=0
# excess(s) = phi - c_paid:  a: 3-2=1, b: 2-1=1, c: 1-0=1  (all exactly 1.0)
# Efficiency: sum phi = 6 == v(abc)-v(empty) = 56-50 = 6.
_WORKED_MASK_ACC = [
    ((False, False, False), 50.0),  # {}
    ((True, False, False), 52.0),   # {a}
    ((False, True, False), 51.0),   # {b}
    ((False, False, True), 50.0),   # {c}
    ((True, True, False), 54.0),    # {a,b}
    ((True, False, True), 53.0),    # {a,c}
    ((False, True, True), 52.0),    # {b,c}
    ((True, True, True), 56.0),     # {a,b,c}
]


def test_worked_example_phi_c_paid_excess_and_efficiency():
    data = _matrix_event(0, 0, ["a", "b", "c"], _WORKED_MASK_ACC)["data"]
    vmap = _A.build_value_map(data)
    per_slot = _A.shapley_paid_excess(vmap, data["slot_ids"])

    phi = {s: per_slot[s][0] for s in ("a", "b", "c")}
    c_paid = {s: per_slot[s][1] for s in ("a", "b", "c")}
    excess = {s: per_slot[s][2] for s in ("a", "b", "c")}

    assert phi["a"] == pytest.approx(3.0, abs=1e-12)
    assert phi["b"] == pytest.approx(2.0, abs=1e-12)
    assert phi["c"] == pytest.approx(1.0, abs=1e-12)
    assert c_paid["a"] == pytest.approx(2.0, abs=1e-12)
    assert c_paid["b"] == pytest.approx(1.0, abs=1e-12)
    assert c_paid["c"] == pytest.approx(0.0, abs=1e-12)
    assert excess["a"] == pytest.approx(1.0, abs=1e-12)
    assert excess["b"] == pytest.approx(1.0, abs=1e-12)
    assert excess["c"] == pytest.approx(1.0, abs=1e-12)

    # Efficiency identity: must not raise, and Sum phi == v(C)-v(empty).
    _A.assert_efficiency(vmap, data["slot_ids"], per_slot)
    assert sum(phi.values()) == pytest.approx(56.0 - 50.0, abs=1e-12)


def test_efficiency_violation_is_a_hard_error():
    # The check is algebraically exact, so it only ever fires on an assembly
    # bug. Corrupt one phi so Sum phi != v(C)-v(empty) and assert it raises --
    # this locks the "hard error, not a warning" contract against a future
    # refactor silently turning the guard into a no-op.
    data = _matrix_event(0, 0, ["a", "b", "c"], _WORKED_MASK_ACC)["data"]
    vmap = _A.build_value_map(data)
    per_slot = dict(_A.shapley_paid_excess(vmap, data["slot_ids"]))
    phi_a, c_paid_a, _ = per_slot["a"]
    per_slot["a"] = (phi_a + 1.0, c_paid_a, phi_a + 1.0 - c_paid_a)  # break the sum
    with pytest.raises(RuntimeError, match="efficiency"):
        _A.assert_efficiency(vmap, data["slot_ids"], per_slot)


# --------------------------------------------------------------------------
# Provenance gate
# --------------------------------------------------------------------------
def _good_preflight(init_std=1e-3):
    return {
        "schedule_id": FIXED_SCHEDULE_HOLD_PLACEBO_3SLOT_V1,
        "schedule_hash": FIXED_SCHEDULE_HOLD_PLACEBO_3SLOT_HASH,
        "seed_lr_override": 0.0,
        "shapley_synergy_scale": 0.0,
        "placebo_init_std": init_std,
        "seed": 41,
        "group_id": "pin_e_std0.001_s41",
    }


def _write_preflight(run_dir: Path, preflight: dict) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "pin_e_preflight.json").write_text(json.dumps(preflight))


def test_check_provenance_accepts_good_preflight(tmp_path):
    run_dir = tmp_path / "placebo_std0.001_s41"
    _write_preflight(run_dir, _good_preflight())
    # Must not raise.
    _A.check_provenance(run_dir, _A.read_preflight(run_dir))


@pytest.mark.parametrize(
    ("field", "bad_value"),
    [
        ("shapley_synergy_scale", 0.1),
        ("seed_lr_override", 0.01),
        ("schedule_id", "fixed-schedule-germinate-r0c0-v1"),
        ("schedule_hash", "deadbeef" * 8),
    ],
)
def test_check_provenance_rejects_each_bad_field(tmp_path, field, bad_value):
    run_dir = tmp_path / "placebo_std0.001_s41"
    preflight = _good_preflight()
    preflight[field] = bad_value
    _write_preflight(run_dir, preflight)
    with pytest.raises(_A.ProvenanceError, match=field):
        _A.check_provenance(run_dir, _A.read_preflight(run_dir))


@pytest.mark.parametrize(
    "field", ["shapley_synergy_scale", "seed_lr_override", "schedule_id", "schedule_hash"]
)
def test_check_provenance_rejects_missing_field(tmp_path, field):
    run_dir = tmp_path / "placebo_std0.001_s41"
    preflight = _good_preflight()
    del preflight[field]
    _write_preflight(run_dir, preflight)
    with pytest.raises(_A.ProvenanceError, match=field):
        _A.check_provenance(run_dir, _A.read_preflight(run_dir))


def test_read_preflight_refuses_when_file_missing(tmp_path):
    run_dir = tmp_path / "no_preflight_here"
    run_dir.mkdir()
    with pytest.raises(_A.ProvenanceError, match=r"pin_e_preflight\.json"):
        _A.read_preflight(run_dir)


# --------------------------------------------------------------------------
# Event assembly + terminal selection
# --------------------------------------------------------------------------
def _flat_masks(slot_ids, acc):
    """A degenerate matrix event: every coalition scores the same accuracy."""
    from itertools import product

    return [(tuple(m), acc) for m in product([False, True], repeat=len(slot_ids))]


def _write_run(run_dir: Path, events, *, preflight=None):
    """Write pin_e_preflight.json + a timestamped telemetry_*/events.jsonl."""
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "pin_e_preflight.json").write_text(
        json.dumps(preflight if preflight is not None else _good_preflight())
    )
    telem = run_dir / "telemetry_2026-07-03_000000"
    telem.mkdir()
    with (telem / "events.jsonl").open("w") as fh:
        for event in events:
            fh.write(json.dumps(event) + "\n")


def test_find_events_file_locates_nested_jsonl(tmp_path):
    run_dir = tmp_path / "placebo_std0.001_s41"
    _write_run(run_dir, [_matrix_event(0, 0, ["a", "b", "c"], _WORKED_MASK_ACC)])
    found = _A.find_events_file(run_dir)
    assert found.name == "events.jsonl"
    assert found.parent.parent == run_dir


def test_load_terminal_skips_k1_k2_and_non_factorial_k3(tmp_path):
    run_dir = tmp_path / "placebo_std0.001_s41"
    events = [
        # k=1 early lifecycle -- skipped.
        _matrix_event(0, 0, ["r0c0"], [((False,), 40.0), ((True,), 40.0)]),
        # k=2 -- skipped.
        _matrix_event(
            0, 0, ["r0c0", "r0c1"], [((False, False), 41.0), ((True, True), 41.0)]
        ),
        # k=3 but only 3 distinct masks (duplicate early-lifecycle) -- skipped.
        _matrix_event(
            0,
            0,
            ["a", "b", "c"],
            [((False, False, False), 42.0), ((True, False, False), 42.0),
             ((True, False, False), 42.0)],
        ),
        # k=3 full factorial -- the only usable terminal.
        _matrix_event(0, 0, ["a", "b", "c"], _WORKED_MASK_ACC),
    ]
    _write_run(run_dir, events)
    records = _A.load_terminal_records(run_dir)
    assert len(records) == 1
    rec = records[0]
    assert (rec.env_id, rec.episode_idx) == (0, 0)
    # Worked-example excess is exactly 1.0 for all three slots.
    for slot in ("a", "b", "c"):
        assert rec.per_slot[slot][2] == pytest.approx(1.0, abs=1e-12)


def test_terminal_is_last_usable_event_per_key(tmp_path):
    run_dir = tmp_path / "placebo_std0.001_s41"
    events = [
        # Earlier full-factorial for (0,0): all-flat -> excess 0. NOT terminal.
        _matrix_event(0, 0, ["a", "b", "c"], _flat_masks(["a", "b", "c"], 50.0)),
        # Later full-factorial for (0,0): worked example -> excess 1.0. Terminal.
        _matrix_event(0, 0, ["a", "b", "c"], _WORKED_MASK_ACC),
    ]
    _write_run(run_dir, events)
    records = _A.load_terminal_records(run_dir)
    assert len(records) == 1
    # Last event won: excess is 1.0, not the earlier flat 0.0.
    assert records[0].per_slot["a"][2] == pytest.approx(1.0, abs=1e-12)


def test_load_terminal_one_record_per_env_episode_key(tmp_path):
    run_dir = tmp_path / "placebo_std0.001_s41"
    events = [
        _matrix_event(0, 0, ["a", "b", "c"], _WORKED_MASK_ACC),
        _matrix_event(1, 5, ["a", "b", "c"], _WORKED_MASK_ACC),
        _matrix_event(0, 0, ["a", "b", "c"], _flat_masks(["a", "b", "c"], 60.0)),
    ]
    _write_run(run_dir, events)
    records = _A.load_terminal_records(run_dir)
    keys = sorted((r.env_id, r.episode_idx) for r in records)
    assert keys == [(0, 0), (1, 5)]


def test_load_terminal_refuses_run_with_no_usable_events(tmp_path):
    run_dir = tmp_path / "placebo_std0.001_s41"
    # Only k=1 events -- no usable k=3 factorial at all.
    _write_run(run_dir, [_matrix_event(0, 0, ["r0c0"], [((False,), 40.0), ((True,), 40.0)])])
    with pytest.raises(RuntimeError, match="no usable"):
        _A.load_terminal_records(run_dir)


# --------------------------------------------------------------------------
# Blocks + block bootstrap
# --------------------------------------------------------------------------
def test_excess_blocks_sorted_with_three_samples_each(tmp_path):
    run_dir = tmp_path / "placebo_std0.001_s41"
    events = [
        _matrix_event(1, 5, ["a", "b", "c"], _WORKED_MASK_ACC),
        _matrix_event(0, 0, ["a", "b", "c"], _WORKED_MASK_ACC),
    ]
    _write_run(run_dir, events)
    blocks = _A.excess_blocks(_A.load_terminal_records(run_dir))
    # Deterministically sorted by (run_dir, env_id, episode_idx).
    keys = [b[0] for b in blocks]
    assert keys == sorted(keys)
    # One block per episode, exactly 3 per-seed samples each.
    assert all(len(b[1]) == 3 for b in blocks)
    assert [b[0][1:] for b in blocks] == [(0, 0), (1, 5)]


def test_bootstrap_p99_ci_is_deterministic_for_fixed_seed():
    rng_state = np.random.default_rng(7)
    sample_blocks = [list(rng_state.normal(size=3)) for _ in range(40)]
    ci_a = _A.bootstrap_p99_ci(sample_blocks)
    ci_b = _A.bootstrap_p99_ci(sample_blocks)
    assert ci_a == ci_b  # same seed -> byte-identical CI
    assert ci_a[0] <= ci_a[1]


def test_bootstrap_p99_ci_rejects_non_uniform_blocks():
    with pytest.raises(RuntimeError, match="3"):
        _A.bootstrap_p99_ci([[0.1, 0.2, 0.3], [0.4, 0.5]])


# --------------------------------------------------------------------------
# End-to-end: discovery, arm grouping, exit codes, report
# --------------------------------------------------------------------------
def test_discover_run_dirs_finds_nested_and_explicit(tmp_path):
    root = tmp_path / "pin_e"
    rd_a = root / "placebo_std0.001_s41"
    rd_b = root / "placebo_std0.0001_s42"
    _write_run(rd_a, [_matrix_event(0, 0, ["a", "b", "c"], _WORKED_MASK_ACC)])
    _write_run(rd_b, [_matrix_event(0, 0, ["a", "b", "c"], _WORKED_MASK_ACC)],
               preflight=_good_preflight(init_std=1e-4))
    # From a parent root: both discovered.
    assert _A.discover_run_dirs([root]) == sorted([rd_a, rd_b], key=str)
    # From an explicit run dir: just that one.
    assert _A.discover_run_dirs([rd_a]) == [rd_a]


def test_main_success_writes_report_and_returns_zero(tmp_path):
    root = tmp_path / "pin_e"
    _write_run(root / "placebo_std0.001_s41",
               [_matrix_event(0, 0, ["a", "b", "c"], _WORKED_MASK_ACC),
                _matrix_event(1, 1, ["a", "b", "c"], _WORKED_MASK_ACC)] + _min_last_actions())
    rc = _A.main([str(root)])
    assert rc == 0
    report_path = root / "pin_e_noise_floor_report.json"
    assert report_path.exists()
    report = json.loads(report_path.read_text())
    # Provenance echo, schedule identity, bootstrap config, per-arm quantiles/CI.
    assert report["schedule_hash"] == FIXED_SCHEDULE_HOLD_PLACEBO_3SLOT_HASH
    assert report["bootstrap"]["seed"] == _A.BOOTSTRAP_SEED
    assert report["overall_degenerate"] is False
    assert len(report["arms"]) == 1
    arm = report["arms"][0]
    assert arm["placebo_init_std"] == pytest.approx(1e-3)
    assert arm["n_samples"] == 6  # 2 episodes x 3 seeds
    assert arm["tau_p99"] == pytest.approx(1.0, abs=1e-12)  # worked-example excess
    assert len(arm["tau_p99_ci"]) == 2
    for key in ("p50", "p90", "p95", "p99", "max"):
        assert key in arm["signed_excess"]
        assert key in arm["abs_excess"]
    assert arm["provenance"][0]["shapley_synergy_scale"] == 0.0


def test_main_returns_one_on_provenance_failure(tmp_path):
    root = tmp_path / "pin_e"
    _write_run(root / "good_s41", [_matrix_event(0, 0, ["a", "b", "c"], _WORKED_MASK_ACC)])
    bad = _good_preflight()
    bad["shapley_synergy_scale"] = 0.5
    _write_run(root / "bad_s42", [_matrix_event(0, 0, ["a", "b", "c"], _WORKED_MASK_ACC)],
               preflight=bad)
    rc = _A.main([str(root)])
    assert rc == 1
    # Refusal happens before analysis -> no report written.
    assert not (root / "pin_e_noise_floor_report.json").exists()


def test_main_returns_two_and_writes_report_on_all_zero(tmp_path):
    root = tmp_path / "pin_e"
    # Flat factorial -> every excess exactly zero -> degenerate floor.
    _write_run(root / "placebo_std0.001_s41",
               [_matrix_event(0, 0, ["a", "b", "c"], _flat_masks(["a", "b", "c"], 44.0)),
                _matrix_event(1, 1, ["a", "b", "c"], _flat_masks(["a", "b", "c"], 44.0))]
               + _min_last_actions())
    rc = _A.main([str(root)])
    assert rc == 2  # PDR-0014 reversal trigger
    report_path = root / "pin_e_noise_floor_report.json"
    assert report_path.exists()  # report written BEFORE the exit-2 return
    report = json.loads(report_path.read_text())
    assert report["overall_degenerate"] is True
    assert report["arms"][0]["non_degeneracy"]["degenerate"] is True
    assert report["arms"][0]["non_degeneracy"]["n_nonzero"] == 0


def test_analyze_groups_arms_by_init_std(tmp_path):
    root = tmp_path / "pin_e"
    _write_run(root / "placebo_std0.001_s41",
               [_matrix_event(0, 0, ["a", "b", "c"], _WORKED_MASK_ACC)] + _min_last_actions())
    _write_run(root / "placebo_std0.0001_s42",
               [_matrix_event(0, 0, ["a", "b", "c"], _WORKED_MASK_ACC)] + _min_last_actions(),
               preflight=_good_preflight(init_std=1e-4))
    report = _A.analyze([root])
    stds = sorted(arm["placebo_init_std"] for arm in report["arms"])
    assert stds == pytest.approx([1e-4, 1e-3])
    assert len(report["arms"]) == 2


def test_analyze_pools_run_dirs_within_one_arm(tmp_path):
    root = tmp_path / "pin_e"
    # Two run dirs, SAME init_std -> one arm pooling both.
    _write_run(root / "placebo_std0.001_s41",
               [_matrix_event(0, 0, ["a", "b", "c"], _WORKED_MASK_ACC)] + _min_last_actions())
    _write_run(root / "placebo_std0.001_s42",
               [_matrix_event(0, 0, ["a", "b", "c"], _WORKED_MASK_ACC)] + _min_last_actions())
    report = _A.analyze([root])
    assert len(report["arms"]) == 1
    assert report["arms"][0]["n_samples"] == 6  # 2 run dirs x 1 episode x 3 seeds


def test_analyze_refuses_when_no_run_dirs(tmp_path):
    with pytest.raises(_A.ProvenanceError, match="no run dir"):
        _A.analyze([tmp_path / "empty"])


def test_analyze_includes_d1_block_beside_flat_d2(tmp_path):
    root = tmp_path / "pin_e"
    events = [
        _matrix_event(0, 0, ["a", "b", "c"], _WORKED_MASK_ACC),
        _last_action_event(0, 6, 1.0, 50.0),
        _last_action_event(0, 6, 2.0, 50.5),
        _last_action_event(0, 3, None, 50.0),  # TRAINING alpha=0
    ]
    _write_run(root / "placebo_std0.001_s41", events)
    report = _A.analyze([root])
    arm = report["arms"][0]
    # D2 stays FLAT at arm top-level (untouched by D1).
    assert "tau_p99" in arm and "signed_excess" in arm and "non_degeneracy" in arm
    # D1 is a sibling block.
    d1 = arm["d1_per_stage_loo"]
    assert d1["source"] == "credited"
    by_name = {row["seed_stage_name"]: row for row in d1["stages"]}
    assert {"HOLDING", "TRAINING"} <= set(by_name)
    assert by_name["TRAINING"]["label"] == "structural zero (alpha=0)"
    assert by_name["HOLDING"]["n_credited"] == 2


# --------------------------------------------------------------------------
# D1: per-stage credited LOO (ANALYTICS_SNAPSHOT last_action)
# --------------------------------------------------------------------------
def _last_action_event(env_id, stage, seed_contribution, val_acc, *, action_name="WAIT",
                       kind="last_action"):
    return {
        "event_type": "ANALYTICS_SNAPSHOT",
        "epoch": 1,
        "group_id": "g",
        "data": {
            "kind": kind,
            "env_id": env_id,
            "episode_idx": 0,
            "action_name": action_name,
            "slot_id": "r0c0",
            "blueprint_id": "placebo",
            "reward_components": {
                "seed_stage": stage,
                "seed_contribution": seed_contribution,
                "val_acc": val_acc,
            },
        },
    }


def _min_last_actions():
    """A minimal realistic last_action pair so analyze()'s D1 pass has input.

    Ignored by the D2 (matrix) path entirely, so D2 numbers are unaffected.
    """
    return [_last_action_event(0, 6, 0.05, 50.0), _last_action_event(0, 6, -0.05, 50.5)]


def test_load_last_action_extracts_fields_and_skips_other_kinds(tmp_path):
    run_dir = tmp_path / "placebo_std0.001_s41"
    events = [
        _matrix_event(0, 0, ["a", "b", "c"], _WORKED_MASK_ACC),  # matrix -> skip
        _last_action_event(0, 6, 0.5, 50.0),                     # keep
        {"event_type": "ANALYTICS_SNAPSHOT", "data": {"kind": "seed_residency"}},  # skip
    ]
    _write_run(run_dir, events)
    recs = _A.load_last_action_records(run_dir)
    assert len(recs) == 1
    r = recs[0]
    assert (r.env_id, r.seed_stage, r.seed_contribution, r.val_acc) == (0, 6, 0.5, 50.0)


def test_load_last_action_refuses_run_with_no_snapshots(tmp_path):
    run_dir = tmp_path / "placebo_std0.001_s41"
    _write_run(run_dir, [_matrix_event(0, 0, ["a", "b", "c"], _WORKED_MASK_ACC)])
    with pytest.raises(RuntimeError, match="no last_action"):
        _A.load_last_action_records(run_dir)


def test_d1_per_stage_hand_computed_stats_labels_and_cov_guard():
    R = _A.LastActionRecord
    recs = []
    # val_acc grid for env 0: {50.0, 50.5, 51.0} -> grid step 0.5.
    # HOLDING(6): contributions 1,2,3 -> mean 2.0, std 1.0 (ddof=1). |mean|>grid -> cov=0.5.
    recs += [R("rd", 0, 6, 1.0, 50.0), R("rd", 0, 6, 2.0, 50.5), R("rd", 0, 6, 3.0, 51.0)]
    # BLENDING(4): 0.1,0.0,-0.1 -> mean 0.0, std 0.1. |mean|<=grid -> cov guarded to null.
    recs += [R("rd", 0, 4, 0.1, 50.0), R("rd", 0, 4, 0.0, 50.5), R("rd", 0, 4, -0.1, 51.0)]
    # TRAINING(3): alpha=0, all credited null -> structural zero.
    recs += [R("rd", 0, 3, None, 50.0), R("rd", 0, 3, None, 50.5)]

    d1 = _A.d1_per_stage(recs)
    assert d1["source"] == "credited"  # credited value, not recomputed
    assert d1["grid_step"] == pytest.approx(0.5)
    rows = {row["seed_stage"]: row for row in d1["stages"]}

    holding = rows[6]
    assert holding["seed_stage_name"] == "HOLDING"
    assert holding["label"] == "credited noise floor"
    assert holding["alpha_zero"] is False
    assert holding["n_snapshots"] == 3 and holding["n_credited"] == 3
    assert holding["mean"] == pytest.approx(2.0)
    assert holding["std"] == pytest.approx(1.0)
    assert holding["cov"] == pytest.approx(0.5)
    assert holding["cov_note"] is None

    blending = rows[4]
    assert blending["mean"] == pytest.approx(0.0, abs=1e-12)
    assert blending["std"] == pytest.approx(0.1)
    assert blending["cov"] is None  # guarded: |mean| below grid step
    assert "grid step" in blending["cov_note"]

    training = rows[3]
    assert training["seed_stage_name"] == "TRAINING"
    assert training["alpha_zero"] is True
    assert training["label"] == "structural zero (alpha=0)"
    assert training["n_snapshots"] == 2 and training["n_credited"] == 0
    assert training["mean"] is None and training["std"] is None and training["cov"] is None


def test_d1_grid_step_is_finest_across_envs():
    # Envs may hold slightly different split sizes (833 vs 834 in the real run);
    # the guard threshold is the finest per-env grid across the arm.
    R = _A.LastActionRecord
    recs = [
        R("rd", 0, 6, 1.0, 50.0), R("rd", 0, 6, 1.0, 50.5),          # env0 step 0.5
        R("rd", 1, 6, 1.0, 50.0), R("rd", 1, 6, 1.0, 50.2), R("rd", 1, 6, 1.0, 50.4),  # env1 step 0.2
    ]
    d1 = _A.d1_per_stage(recs)
    assert d1["grid_step"] == pytest.approx(0.2)


def test_d1_cov_guard_boundary_mean_equals_grid_step():
    R = _A.LastActionRecord
    # grid step 0.5; HOLDING mean exactly 0.5 -> "at or below" -> null.
    recs = [R("rd", 0, 6, 0.5, 50.0), R("rd", 0, 6, 0.5, 50.5), R("rd", 0, 6, 0.5, 51.0)]
    d1 = _A.d1_per_stage(recs)
    row = {r["seed_stage"]: r for r in d1["stages"]}[6]
    assert d1["grid_step"] == pytest.approx(0.5)
    assert row["mean"] == pytest.approx(0.5)
    assert row["cov"] is None  # |mean| == grid step boundary


def test_signal_mask_ignores_float_roundoff_zeros():
    # The 1/3,1/6-weighted Shapley assembly leaves ~1e-17 residuals on excess
    # values that are physically zero. Those must NOT count as signal, else a
    # deterministic-zero (roundoff-corrupted) placebo dodges the exit-2 trigger.
    samples = np.array([0.0, 1.4e-17, -3e-16, 0.05, -0.2])
    mask = _A.signal_mask(samples)
    assert mask.tolist() == [False, False, False, True, True]
    # An all-roundoff arm is degenerate.
    assert not _A.signal_mask(np.array([0.0, 1e-16, -1e-17])).any()
