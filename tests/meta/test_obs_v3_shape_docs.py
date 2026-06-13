"""Documentation guard for public Obs V3 shape claims."""

from __future__ import annotations

from pathlib import Path

from esper.leyline import DEFAULT_BLUEPRINT_EMBED_DIM, OBS_V3_NON_BLUEPRINT_DIM
from esper.leyline.slot_config import SlotConfig
from esper.tamiyo.policy.features import get_feature_size


ROOT = Path(__file__).parents[2]


def test_public_obs_v3_shape_claims_match_runtime_constants() -> None:
    """README/ROADMAP/specs must track the runtime Obs V3 dimensions."""
    slot_config = SlotConfig.default()
    non_blueprint_dims = get_feature_size(slot_config)
    blueprint_dims = DEFAULT_BLUEPRINT_EMBED_DIM * slot_config.num_slots
    total_dims = non_blueprint_dims + blueprint_dims

    assert non_blueprint_dims == OBS_V3_NON_BLUEPRINT_DIM

    docs = {
        "README.md": (ROOT / "README.md").read_text(),
        "ROADMAP.md": (ROOT / "ROADMAP.md").read_text(),
        "docs/specifications/simic.md": (
            ROOT / "docs/specifications/simic.md"
        ).read_text(),
    }

    for path, text in docs.items():
        assert "113 dims" not in text, path
        assert "125 total" not in text, path
        assert f"{non_blueprint_dims}" in text, path

    assert (
        f"Non-blueprint obs: **{non_blueprint_dims} dims**" in docs["README.md"]
    )
    assert f"Total policy input: **{total_dims} dims**" in docs["README.md"]
    assert (
        f"{non_blueprint_dims} dims + {blueprint_dims} blueprint embed = "
        f"{total_dims} total network input"
        in docs["ROADMAP.md"]
    )
    assert (
        f"features: [batch, seq_len, {non_blueprint_dims}]"
        in docs["docs/specifications/simic.md"]
    )
    assert (
        f"states: [4, 150, {non_blueprint_dims}]"
        in docs["docs/specifications/simic.md"]
    )
