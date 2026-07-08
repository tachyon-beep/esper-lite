"""End-to-end tests for scripts/stage2_packet.py — the Stage-2 acceptance packet CLI.

Drives the REAL path: fabricated events.jsonl telemetry -> scan_ingestion_integrity ->
create_views -> calibrate_from_spec / packet_from_spec -> rendered text. The §10 freeze
order is exercised at the CLI surface: calibrate reads OFF arms only; score consumes
frozen thresholds.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

_STAGE2_PACKET_PATH = Path(__file__).parents[2] / "scripts" / "stage2_packet.py"
_SPEC = importlib.util.spec_from_file_location("stage2_packet", _STAGE2_PACKET_PATH)
assert _SPEC is not None
assert _SPEC.loader is not None
_STAGE2_PACKET = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_STAGE2_PACKET)
stage2_packet_main = _STAGE2_PACKET.main


_EV_JITTER = (-0.02, 0.0, 0.02, 0.0)
_STD_JITTER = (-0.5, 0.0, 0.5, 0.0)


def _write_run(telemetry_dir: Path, run_name: str, *, seed: int, on: bool, expl: float) -> None:
    """One run's events.jsonl: TRAINING_STARTED + PPO updates + terminal outcomes."""
    run_dir = telemetry_dir / run_name
    run_dir.mkdir()
    events: list[dict] = [
        {
            "event_id": f"start-{run_name}",
            "event_type": "TRAINING_STARTED",
            "timestamp": "2026-07-07T00:00:00+00:00",
            "group_id": "stage2",
            "data": {
                "task": "cifar_baseline",
                "reward_mode": "shaped",
                "seed": seed,
                "n_envs": 1,
                "n_episodes": 5,
                "max_epochs": 150,
                "max_batches": 5,
                "lr": 0.001,
                "gamma": 0.99,
                "gae_lambda": 0.95,
                "ppo_updates_per_batch": 1,
                "recurrent_n_epochs": 1,
                "clip_ratio": 0.2,
                "entropy_coef": 0.01,
                "per_head_advantage_norm": False,
                "return_variance_telemetry": False,
                "value_coef": 0.5,
                "value_warmup_batches": 0,
                "value_coef_start": None,
                "param_budget": 100_000,
                "param_penalty_weight": 0.1,
                "sparse_reward_scale": 1.0,
                "rent_host_params_floor": 200,
                "basic_acc_delta_weight": 5.0,
                "host_params": 100_000,
                "reward_family": "contribution",
                "policy_device": "cpu",
                "slot_ids": ["early"],
                "env_devices": ["cpu"],
                "resume_path": "",
                "start_episode": 0,
                "amp_enabled": False,
                "amp_dtype": None,
                "compile_enabled": False,
                "compile_backend": None,
                "compile_mode": None,
                "plateau_threshold": 0.5,
                "improvement_threshold": 2.0,
                "gradient_telemetry_stride": 10,
                "lstm_hidden_dim": 512,
                "chunk_length": 150,
                "max_seeds": None,
                "permissive_gates": True,
                "auto_forward_g1": False,
                "auto_forward_g2": False,
                "auto_forward_g3": False,
                "disable_pbrs": False,
                "disable_terminal_reward": False,
                "disable_anti_gaming": False,
                "max_grad_norm": None,
                "actor_advantage_source": "total_reconstructed",
            },
        }
    ]
    for b in range(4):
        data: dict = {
            "inner_epoch": 0,
            "batch": b,
            "explained_variance": expl + _EV_JITTER[b],
            "ev_return_variance": 5.0,
            "pre_norm_advantage_std": 1.0,
            "return_std": 3.0 + _STD_JITTER[b],
            "gradient_cv": 0.10,
            "advantage_std_floored": False,
            "advantage_per_head_normalized": False,
        }
        if on:
            data["ev_sum"] = expl + _EV_JITTER[b]
            data["ev_main"] = 0.03
            data["ev_cf"] = 0.10
            data["value_main_target_scale"] = 1.7
            data["cf_value_target_scale"] = 6.3
        events.append(
            {
                "event_id": f"ppo-{run_name}-{b}",
                "event_type": "PPO_UPDATE_COMPLETED",
                "timestamp": "2026-07-07T00:01:00+00:00",
                "epoch": b,
                "group_id": "stage2",
                "data": data,
            }
        )
    for episode_idx in range(5):
        events.append(
            {
                "event_id": f"outcome-{run_name}-{episode_idx}",
                "event_type": "EPISODE_OUTCOME",
                "timestamp": "2026-07-07T00:02:00+00:00",
                "group_id": "stage2",
                "data": {
                    "env_id": 0,
                    "episode_idx": episode_idx,
                    "final_accuracy": 50.0,
                    "param_ratio": 1.1,
                    "germinate_count": 1,
                    "prune_count": 1,
                    "fossilize_count": 0,
                },
            }
        )
    (run_dir / "events.jsonl").write_text(
        "\n".join(json.dumps(event) for event in events) + "\n"
    )


def _write_pairset(telemetry_dir: Path, n: int) -> list[dict]:
    pairs = []
    for seed in range(n):
        _write_run(telemetry_dir, f"on_{seed}", seed=seed, on=True, expl=0.85)
        _write_run(telemetry_dir, f"off_{seed}", seed=seed, on=False, expl=0.80)
        pairs.append(
            {"seed": seed, "on_run_dir": f"on_{seed}", "off_run_dir": f"off_{seed}"}
        )
    return pairs


def test_calibrate_then_score_end_to_end(tmp_path, capsys):
    telemetry_dir = tmp_path / "telemetry"
    telemetry_dir.mkdir()
    pairs = _write_pairset(telemetry_dir, 5)

    # §10 step 2 — calibrate from the OFF arms only.
    calibrate_spec = tmp_path / "calibrate.json"
    calibrate_spec.write_text(
        json.dumps(
            {"off_run_dirs": [f"off_{seed}" for seed in range(5)], "w": 0, "budget": 2}
        )
    )
    rc = stage2_packet_main(
        ["calibrate", "--telemetry-dir", str(telemetry_dir), "--spec", str(calibrate_spec)]
    )
    assert rc == 0
    calibration_text = capsys.readouterr().out
    assert "delta" in calibration_text
    assert "freeze" in calibration_text.lower()

    # §10 step 4 — score against frozen thresholds; packet lands in the output file.
    score_spec = tmp_path / "score.json"
    score_spec.write_text(
        json.dumps(
            {
                "pairs": pairs,
                "host_params": 100_000,
                "n": 5,
                "budget": 2,
                "thresholds": {
                    "delta": 0.05,
                    "eps_rel": 0.10,
                    "tau_acc": 0.3,
                    "delta_param_max": 1e9,
                    "w": 0,
                },
                "g3_ratio_max": 1.5,
                "g4_ratio_max": 2.0,
                "g4_abs_floor": 5,
            }
        )
    )
    output = tmp_path / "packet.md"
    rc = stage2_packet_main(
        [
            "score",
            "--telemetry-dir", str(telemetry_dir),
            "--spec", str(score_spec),
            "--output", str(output),
        ]
    )
    assert rc == 0
    packet = output.read_text()
    assert "# Stage-2 HRA MAJOR-1 acceptance verdict" in packet
    assert "n=5" in packet
    # The clean fixture passes §1 (real telemetry-signature verification end to end).
    lines = packet.splitlines()
    assert lines[2].startswith("**Verdict: REJECT**")
    validity_idx = lines.index("## §1 Validity")
    assert lines[validity_idx + 2] == "VALID — pairing, finiteness, and completeness gates pass."
    assert "## §1 Validity — INVALID" not in packet
    assert "Stage-0 variance gate" in packet  # §0 provenance travels verbatim


def test_corrupt_jsonl_aborts_before_scoring(tmp_path, capsys):
    # FAIL CLOSED: the view layer silently drops malformed lines (ignore_errors=true), so
    # corruption would masquerade as an incomplete-but-scoreable run — the CLI must abort.
    telemetry_dir = tmp_path / "telemetry"
    telemetry_dir.mkdir()
    _write_pairset(telemetry_dir, 5)
    events = telemetry_dir / "on_0" / "events.jsonl"
    events.write_text(events.read_text() + "{not valid json\n")

    spec = tmp_path / "calibrate.json"
    spec.write_text(
        json.dumps({"off_run_dirs": [f"off_{s}" for s in range(5)], "w": 0, "budget": 2})
    )
    rc = stage2_packet_main(
        ["calibrate", "--telemetry-dir", str(telemetry_dir), "--spec", str(spec)]
    )
    assert rc == 2
    assert "not clean" in capsys.readouterr().err


def test_traceback_off_run_log_aborts_calibration_before_freezing_delta(tmp_path, capsys):
    telemetry_dir = tmp_path / "telemetry"
    telemetry_dir.mkdir()
    _write_pairset(telemetry_dir, 5)
    (telemetry_dir / "off_0" / "stderr.log").write_text(
        "Traceback (most recent call last):\nRuntimeError: training failed\n"
    )

    spec = tmp_path / "calibrate.json"
    spec.write_text(
        json.dumps({"off_run_dirs": [f"off_{s}" for s in range(5)], "w": 0, "budget": 2})
    )
    rc = stage2_packet_main(
        ["calibrate", "--telemetry-dir", str(telemetry_dir), "--spec", str(spec)]
    )
    assert rc == 1
    assert "traceback" in capsys.readouterr().err.lower()


def test_traceback_scan_survives_non_utf8_log_bytes(tmp_path, capsys):
    # Training logs routinely carry non-UTF-8 bytes (progress-bar control sequences,
    # truncated multibyte writes, binary spew from a crashing C extension). A stray byte
    # must not crash the §1 scan (a UnicodeDecodeError would propagate out of main, not
    # return an rc); the pure-ASCII crash markers stay detectable through errors="replace".
    telemetry_dir = tmp_path / "telemetry"
    telemetry_dir.mkdir()
    _write_pairset(telemetry_dir, 5)
    (telemetry_dir / "off_0" / "stderr.log").write_bytes(
        b"epoch 3 progress: 45%\n"
        b"\x80\xfe\n"  # raw non-UTF-8 bytes mid-file
        b"Traceback (most recent call last):\n"
        b"RuntimeError: training failed\n"
    )

    spec = tmp_path / "calibrate.json"
    spec.write_text(
        json.dumps({"off_run_dirs": [f"off_{s}" for s in range(5)], "w": 0, "budget": 2})
    )
    rc = stage2_packet_main(
        ["calibrate", "--telemetry-dir", str(telemetry_dir), "--spec", str(spec)]
    )
    assert rc == 1  # (a) did not raise on the non-UTF-8 byte
    err = capsys.readouterr().err
    assert "Traceback (most recent call last):" in err  # (b) marker line still reported
    assert "stderr.log" in err


def test_traceback_unreferenced_run_log_does_not_block_calibration(tmp_path, capsys):
    telemetry_dir = tmp_path / "telemetry"
    telemetry_dir.mkdir()
    _write_pairset(telemetry_dir, 5)
    (telemetry_dir / "on_0" / "stderr.log").write_text(
        "Traceback (most recent call last):\nRuntimeError: training failed\n"
    )

    spec = tmp_path / "calibrate.json"
    spec.write_text(
        json.dumps({"off_run_dirs": [f"off_{s}" for s in range(5)], "w": 0, "budget": 2})
    )
    rc = stage2_packet_main(
        ["calibrate", "--telemetry-dir", str(telemetry_dir), "--spec", str(spec)]
    )
    assert rc == 0
    assert "delta" in capsys.readouterr().out


def test_traceback_paired_run_log_renders_invalid_score_packet(tmp_path):
    telemetry_dir = tmp_path / "telemetry"
    telemetry_dir.mkdir()
    pairs = _write_pairset(telemetry_dir, 5)
    (telemetry_dir / "on_0" / "stderr.log").write_text(
        "[TRAINING ERROR]\nTraceback (most recent call last):\nRuntimeError: training failed\n"
    )

    spec = tmp_path / "score.json"
    spec.write_text(
        json.dumps(
            {
                "pairs": pairs,
                "host_params": 100_000,
                "n": 5,
                "budget": 2,
                "thresholds": {
                    "delta": 0.05,
                    "eps_rel": 0.10,
                    "tau_acc": 0.3,
                    "delta_param_max": 1e9,
                    "w": 0,
                },
                "g3_ratio_max": 1.5,
                "g4_ratio_max": 2.0,
                "g4_abs_floor": 5,
            }
        )
    )
    output = tmp_path / "packet.md"
    rc = stage2_packet_main(
        [
            "score",
            "--telemetry-dir", str(telemetry_dir),
            "--spec", str(spec),
            "--output", str(output),
        ]
    )
    assert rc == 0
    packet = output.read_text()
    assert "## §1 Validity — INVALID" in packet
    assert "traceback" in packet.lower()


def test_calibrate_rejects_on_run_in_off_dirs(tmp_path, capsys):
    telemetry_dir = tmp_path / "telemetry"
    telemetry_dir.mkdir()
    _write_pairset(telemetry_dir, 5)

    spec = tmp_path / "calibrate.json"
    spec.write_text(json.dumps({"off_run_dirs": ["on_0"], "w": 0, "budget": 2}))
    rc = stage2_packet_main(
        ["calibrate", "--telemetry-dir", str(telemetry_dir), "--spec", str(spec)]
    )
    assert rc == 1
    assert "OFF arm carries" in capsys.readouterr().err
