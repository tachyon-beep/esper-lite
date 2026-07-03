"""Orphan-run exclusion contract for the Gate-2 Tier-2 analyzer (esper-lite-72fb7074ca).

The analyzer loads ``tier2_{arm}_s{seed}.jsonl`` pairs. The bug: a run was
inserted into ``runs`` BEFORE its pair's completeness was known, so a control
file whose retro pair is missing (an aborted/partial campaign) leaked into
every whole-``runs`` consumer:

  1. the common-batch intersection — an orphan with fewer batches drags the
     endpoint ``read_batch`` down for the COMPLETE pairs;
  2. the Tier-1 pooled screen — the orphan's records pollute the pooled
     dadv/sigma_A ratios feeding the G2a gate line.

The fabricated data makes both surfaces assertable: the complete pair (seed
51) spans batches 0-10; the orphan control (seed 52) stops at batch 8, so a
contaminated endpoint reads batch 8 instead of 10, and a contaminated pooled
screen counts the orphan's 9 extra records.
"""

import importlib.util
import json
import sys
from pathlib import Path

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "gate2_tier2_analyze.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("gate2_tier2_analyze", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _write_run(path: Path, *, n_batches: int, p_eligible: float) -> None:
    records = [
        {
            "batch": b,
            "propensity": {"eligible": {"n": 5, "mean": p_eligible}},
            "adv_base_at_foss": {"n": 2, "mean": 0.05},
            "dadv_retro_at_foss": {"n": 1, "mean": 0.2},
            "dadv_term_at_foss": {"n": 1, "mean": 0.1},
            "sigma_A_raw": 1.0,
        }
        for b in range(n_batches)
    ]
    path.write_text("".join(json.dumps(r) + "\n" for r in records))


def _run_main(mod, tmp_path: Path, monkeypatch, capsys, seeds: list[int]) -> str:
    argv = ["gate2_tier2_analyze.py", "--dir", str(tmp_path), "--seeds"]
    argv += [str(s) for s in seeds]
    monkeypatch.setattr(sys, "argv", argv)
    mod.main()
    return capsys.readouterr().out


def test_orphan_control_run_excluded_from_endpoint_and_pooled_screen(
    tmp_path, monkeypatch, capsys
) -> None:
    """An orphan control (retro pair missing) must not shape any readout."""
    # Complete pair, batches 0..10.
    _write_run(tmp_path / "tier2_control_s51.jsonl", n_batches=11, p_eligible=0.10)
    _write_run(tmp_path / "tier2_retro_s51.jsonl", n_batches=11, p_eligible=0.15)
    # Orphan: control only, stops at batch 8.
    _write_run(tmp_path / "tier2_control_s52.jsonl", n_batches=9, p_eligible=0.10)

    out = _run_main(_load_module(), tmp_path, monkeypatch, capsys, seeds=[51, 52])

    assert "pair for seed 52 incomplete" in out
    # Endpoint: latest batch common to the COMPLETE pair only (10), not the
    # orphan-constrained batch 8.
    assert "PRIMARY @ batch 10" in out, (
        "the endpoint read_batch must be selected over complete pairs only; "
        "an orphan run's shorter batch range must not drag it down"
    )
    # Pooled screen: 11 records x 2 runs of the complete pair = 22 batches;
    # the orphan's 9 records must not be pooled.
    assert "n_batches 22" in out, (
        "the Tier-1 pooled screen must pool complete-pair runs only"
    )


def test_all_pairs_complete_unchanged(tmp_path, monkeypatch, capsys) -> None:
    """No-orphan regression guard: complete pairs behave as before."""
    for seed in (51, 52):
        _write_run(tmp_path / f"tier2_control_s{seed}.jsonl", n_batches=11, p_eligible=0.10)
        _write_run(tmp_path / f"tier2_retro_s{seed}.jsonl", n_batches=11, p_eligible=0.15)

    out = _run_main(_load_module(), tmp_path, monkeypatch, capsys, seeds=[51, 52])

    assert "PRIMARY @ batch 10" in out
    assert "N=2" in out
    assert "n_batches 44" in out  # 11 records x 4 runs
