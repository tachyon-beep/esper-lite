from __future__ import annotations

import os
import struct

import pytest

from esper.leyline import leyline_pb2
from esper.tamiyo import TamiyoPersistenceError
from esper.tamiyo.persistence import FieldReportStore, FieldReportStoreConfig


def _make_report(index: int) -> leyline_pb2.FieldReport:
    report = leyline_pb2.FieldReport(
        version=1,
        report_id=f"report-{index}",
        outcome=leyline_pb2.FIELD_REPORT_OUTCOME_SUCCESS,
        training_run_id="run-1",
        seed_id="seed-1",
        blueprint_id="bp-1",
        observation_window_epochs=1,
        tamiyo_policy_version="policy",
    )
    report.metrics["loss_delta"] = 0.1 * index
    report.issued_at.GetCurrentTime()
    return report


def test_field_report_store_raises_on_truncated_entry(tmp_path) -> None:
    store_path = tmp_path / "field_reports.log"
    config = FieldReportStoreConfig(path=store_path)
    store = FieldReportStore(config)

    store.append(_make_report(1))

    with store_path.open("ab") as handle:
        handle.write(struct.pack("<I", 32))

    with pytest.raises(TamiyoPersistenceError):
        FieldReportStore(FieldReportStoreConfig(path=store_path))


def test_field_report_store_can_opt_out_of_strict_validation(tmp_path) -> None:
    store_path = tmp_path / "field_reports.log"
    store = FieldReportStore(FieldReportStoreConfig(path=store_path, strict_validation=False))
    store.append(_make_report(1))
    with store_path.open("ab") as handle:
        handle.write(struct.pack("<I", 16))

    reloaded = FieldReportStore(FieldReportStoreConfig(path=store_path, strict_validation=False))
    assert len(reloaded.reports()) == 1
    assert reloaded.load_errors
