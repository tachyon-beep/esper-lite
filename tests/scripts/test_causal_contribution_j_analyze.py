"""Per-seed attribution contract for the paired-J analyzer (esper-lite-03a5351609).

The analyzer skips incomplete seed pairs (the pre-registered G1-abort pairing
rule DROPS a pair, so this path is expected at A/B scoring time). The per-seed
report must attribute each Δeff to the seed that produced it — the bug was two
parallel lists (requested ``seeds`` vs complete-pairs-only ``deltas``) zipped
together, mislabeling every delta after a skip and silently truncating the tail.

The fabricated telemetry is minimal but gate-valid: constant controller stride,
identical draw-counts/state-hashes across arms (offset-free parity), no r0c0
germination, no finiteness trips. tot_p = 1e6 params so eff == contrib in
pp/Mparam and the expected deltas are exact.
"""

import importlib.util
import json
import sys
from pathlib import Path

_SCRIPT = (
    Path(__file__).resolve().parents[2] / "scripts" / "causal_contribution_j_analyze.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "causal_contribution_j_analyze", _SCRIPT
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _write_events(run_dir: Path, *, all_off_acc: float, all_on_acc: float) -> None:
    run_dir.mkdir(parents=True)
    events = [
        {
            "event_type": "SEED_FOSSILIZED",
            "data": {
                "params_added": 1_000_000,
                "slot_id": "r0c1",
                "counterfactual": 1.0,
            },
        },
        {"event_type": "SEED_GERMINATED", "data": {"slot_id": "r0c1"}},
        {"event_type": "EPISODE_OUTCOME", "data": {"episode_idx": 10}},
        # Constant stride, identical across arms => stride + parity gates pass.
        *(
            {
                "event_type": "INTERVENTION_STEP",
                "data": {"controller_draw_count": c, "controller_state_hash": h},
            }
            for c, h in [(5, "h1"), (10, "h2"), (15, "h3")]
        ),
        # Late-run (>= 0.8 * max_ep) counterfactual matrix: all-off vs all-on.
        {
            "event_type": "COUNTERFACTUAL_MATRIX_COMPUTED",
            "data": {
                "episode_idx": 9,
                "configs": [
                    {"seed_mask": [False], "accuracy": all_off_acc},
                    {"seed_mask": [True], "accuracy": all_on_acc},
                ],
            },
        },
    ]
    with open(run_dir / "events.jsonl", "w") as f:
        for e in events:
            f.write(json.dumps(e) + "\n")


def _write_pair(base: Path, seed: int, *, control_on: float, suppress_on: float) -> None:
    _write_events(base / f"control_s{seed}", all_off_acc=10.0, all_on_acc=control_on)
    _write_events(
        base / f"suppress_slot_on_s{seed}", all_off_acc=10.0, all_on_acc=suppress_on
    )


def test_deltas_attributed_to_their_own_seed_when_a_pair_is_dropped(
    tmp_path, monkeypatch, capsys
) -> None:
    """Seed 41 incomplete (the dropped-pair scenario); 42/43 have distinct deltas."""
    _write_pair(tmp_path, 42, control_on=60.0, suppress_on=62.0)  # Δeff = +2.000
    _write_pair(tmp_path, 43, control_on=60.0, suppress_on=63.0)  # Δeff = +3.000

    mod = _load_module()
    monkeypatch.setattr(
        sys, "argv", ["causal_contribution_j_analyze.py", str(tmp_path), "41,42,43"]
    )
    rc = mod.main()
    out = capsys.readouterr().out

    assert "seed 41: incomplete" in out
    assert "seed 41: Δeff" not in out, (
        "a skipped seed must not be attributed another seed's delta"
    )
    assert "seed 42: Δeff = +2.000" in out, (
        "seed 42's delta must be reported under seed 42"
    )
    assert "seed 43: Δeff = +3.000" in out, (
        "seed 43's delta must be reported under seed 43 (not truncated off the report)"
    )
    assert "n=2 seeds" in out
    assert rc == 0


def test_deltas_attributed_correctly_with_all_pairs_complete(
    tmp_path, monkeypatch, capsys
) -> None:
    """No-skip regression guard: the normal 100%-complete path stays correct."""
    _write_pair(tmp_path, 42, control_on=60.0, suppress_on=62.0)  # Δeff = +2.000
    _write_pair(tmp_path, 43, control_on=60.0, suppress_on=59.0)  # Δeff = -1.000

    mod = _load_module()
    monkeypatch.setattr(
        sys, "argv", ["causal_contribution_j_analyze.py", str(tmp_path), "42,43"]
    )
    rc = mod.main()
    out = capsys.readouterr().out

    assert "seed 42: Δeff = +2.000" in out
    assert "seed 43: Δeff = -1.000" in out
    assert "n=2 seeds" in out
    assert rc == 0
