"""Committed-Shapley top-up telemetry contract (WI-7).

The credit is NOT a per-step reward additend (it is retro-written into the
rollout buffer pre-GAE), so this event is the ONLY place the books for the
buffer-side delta live. It also feeds the enablement-gate monitors directly
(G-clamp binding rate, per-episode fossilize counts).
"""

from esper.leyline import CommittedShapleyTopUpPayload
from esper.leyline.telemetry import TelemetryEventType


def test_event_type_exists():
    assert TelemetryEventType.COMMITTED_SHAPLEY_TOPUP is not None


def test_payload_round_trips_from_dict():
    payload = CommittedShapleyTopUpPayload(
        env_id=3,
        slot_ids=("r0c0", "r0c1"),
        phi=(12.5, 7.5),
        c_paid=(10.0, 5.0),
        gap=(2.5, 2.5),
        raw=(2.5, 2.5),
        top_up=(2.5, 2.5),
        t_f=(41, 97),
        credit_buf=(2.31, 2.31),
        g=20.0,
        sum_raw=5.0,
        clamp_binding=False,
        std_used=1.08,
        episode_idx=7,
    )
    parsed = CommittedShapleyTopUpPayload.from_dict(
        {
            "env_id": 3,
            "slot_ids": ["r0c0", "r0c1"],
            "phi": [12.5, 7.5],
            "c_paid": [10.0, 5.0],
            "gap": [2.5, 2.5],
            "raw": [2.5, 2.5],
            "top_up": [2.5, 2.5],
            "t_f": [41, 97],
            "credit_buf": [2.31, 2.31],
            "g": 20.0,
            "sum_raw": 5.0,
            "clamp_binding": False,
            "std_used": 1.08,
            "episode_idx": 7,
        }
    )
    assert parsed == payload


def test_payload_dropped_no_std_default_false():
    payload = CommittedShapleyTopUpPayload(
        env_id=0,
        slot_ids=("r0c0",),
        phi=(1.0,),
        c_paid=(1.0,),
        gap=(0.0,),
        raw=(0.0,),
        top_up=(0.0,),
        t_f=(5,),
        credit_buf=(0.0,),
        g=1.0,
        sum_raw=0.0,
        clamp_binding=False,
        std_used=0.0,
        episode_idx=None,
    )
    assert payload.dropped_no_std is False
