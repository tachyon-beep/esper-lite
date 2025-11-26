from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from scripts.tamiyo_wal_backup import main as backup_main


def _latest_backup(directory: Path) -> Path:
    backups = sorted(directory.iterdir())
    assert backups, "expected a backup directory"
    return backups[-1]


def test_backup_and_restore(tmp_path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    wal = data_dir / "field_reports.log"
    wal.write_text("wal-data")
    retry_index = data_dir / "field_reports.index.json"
    retry_index.write_text("{}")
    windows = data_dir / "field_reports.windows.json"
    windows.write_text("{}")

    backup_dir = tmp_path / "backups"
    backup_main(
        [
            "backup",
            "--wal",
            str(wal),
            "--retry-index",
            str(retry_index),
            "--windows",
            str(windows),
            "--dest",
            str(backup_dir),
        ]
    )

    created = _latest_backup(backup_dir)
    assert (created / wal.name).exists()
    assert (created / retry_index.name).exists()
    assert (created / windows.name).exists()

    restore_dir = tmp_path / "restore"
    backup_main(
        [
            "restore",
            str(created),
            "--wal",
            str(wal),
            "--retry-index",
            str(retry_index),
            "--windows",
            str(windows),
            "--dest",
            str(restore_dir),
        ]
    )

    assert (restore_dir / wal.name).read_text() == "wal-data"
    assert (restore_dir / retry_index.name).read_text() == "{}"
    assert (restore_dir / windows.name).read_text() == "{}"
