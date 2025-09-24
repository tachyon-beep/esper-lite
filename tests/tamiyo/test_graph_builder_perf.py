import os
import statistics
import time

import pytest

from esper.leyline import leyline_pb2
from esper.tamiyo.graph_builder import TamiyoGraphBuilder, TamiyoGraphBuilderConfig


@pytest.mark.perf
def test_graph_builder_small_graph_p95_budget():
    if os.getenv("RUN_PERF_TESTS") != "1":
        pytest.skip("perf test disabled; set RUN_PERF_TESTS=1 to enable")

    # Minimal builder with defaults; no metadata provider to exercise fallbacks
    builder = TamiyoGraphBuilder(TamiyoGraphBuilderConfig(max_layers=4, max_activations=1, max_parameters=1))
    packet = leyline_pb2.SystemStatePacket(version=1, current_epoch=1, training_run_id="run-perf")
    # Add a couple of seeds to exercise seed edges
    for i in range(2):
        seed = packet.seed_states.add()
        seed.seed_id = f"seed-{i}"
        seed.stage = leyline_pb2.SeedLifecycleStage.SEED_STAGE_TRAINING
        seed.learning_rate = 0.01
        seed.layer_depth = 1
        seed.risk_score = 0.2

    # Warm-up
    builder.build(packet)

    durations = []
    runs = 100
    for _ in range(runs):
        start = time.perf_counter()
        _ = builder.build(packet)
        durations.append((time.perf_counter() - start) * 1000.0)

    durations.sort()
    p95 = durations[int(len(durations) * 0.95)]
    # Soft budget for CPU small graphs; adjust if host variance requires
    assert p95 <= 5.0

