"""Tests for Pareto frontier computation."""

from hypothesis import given, strategies as st
from esper.leyline import EpisodeOutcome
from esper.karn.pareto import extract_pareto_frontier, compute_hypervolume_2d


def make_outcome(acc: float, param: float, stab: float = 1.0) -> EpisodeOutcome:
    """Helper to create test outcomes."""
    return EpisodeOutcome(
        env_id=0, episode_idx=0,
        final_accuracy=acc, param_ratio=param,
        num_fossilized=1, num_contributing_fossilized=1,
        episode_reward=acc / 10, stability_score=stab,
        reward_mode="shaped",
    )


def test_extract_pareto_frontier_simple():
    """Extract non-dominated outcomes."""
    outcomes = [
        make_outcome(80, 0.1),  # Pareto optimal
        make_outcome(70, 0.2),  # Dominated by first
        make_outcome(75, 0.05),  # Pareto optimal (lower param)
    ]
    frontier = extract_pareto_frontier(outcomes)
    assert len(frontier) == 2
    accs = {o.final_accuracy for o in frontier}
    assert 80 in accs
    assert 75 in accs
    assert 70 not in accs


def test_extract_pareto_frontier_all_dominated():
    """Single dominant point returns just that point."""
    outcomes = [
        make_outcome(90, 0.05),  # Dominates all
        make_outcome(80, 0.1),
        make_outcome(70, 0.2),
    ]
    frontier = extract_pareto_frontier(outcomes)
    assert len(frontier) == 1
    assert frontier[0].final_accuracy == 90


def test_extract_pareto_frontier_empty():
    """Empty input returns empty frontier."""
    assert extract_pareto_frontier([]) == []


def test_extract_pareto_frontier_single():
    """Single outcome is always on frontier."""
    single = [make_outcome(50, 0.5)]
    frontier = extract_pareto_frontier(single)
    assert len(frontier) == 1


def test_hypervolume_2d_basic():
    """Compute 2D hypervolume for accuracy vs param_ratio."""
    frontier = [
        make_outcome(80, 0.1),
        make_outcome(70, 0.05),
    ]
    ref_point = (0.0, 1.0)
    hv = compute_hypervolume_2d(frontier, ref_point)
    assert hv > 0


def test_hypervolume_2d_known_value():
    """Verify hypervolume with known expected value."""
    # Single point at (80, 0.2) with ref (0, 1.0)
    # Area = 80 * (1.0 - 0.2) = 80 * 0.8 = 64
    frontier = [make_outcome(80, 0.2)]
    ref_point = (0.0, 1.0)
    hv = compute_hypervolume_2d(frontier, ref_point)
    assert abs(hv - 64.0) < 1e-6, f"Expected 64.0, got {hv}"


def test_hypervolume_2d_two_points():
    """Verify hypervolume with two non-dominated points."""
    # Points: (80, 0.3) and (60, 0.1) with ref (0, 1.0)
    # Sorted by acc descending: [(80, 0.3), (60, 0.1)]
    # Area from (80, 0.3): 80 * (1.0 - 0.3) = 56
    # Area from (60, 0.1): 60 * (0.3 - 0.1) = 12
    # Total = 56 + 12 = 68
    frontier = [make_outcome(80, 0.3), make_outcome(60, 0.1)]
    ref_point = (0.0, 1.0)
    hv = compute_hypervolume_2d(frontier, ref_point)
    assert abs(hv - 68.0) < 1e-6, f"Expected 68.0, got {hv}"


def test_hypervolume_2d_empty():
    """Empty frontier has zero hypervolume."""
    assert compute_hypervolume_2d([], (0.0, 1.0)) == 0.0


def test_hypervolume_2d_nonzero_ref_acc():
    """Verify hypervolume with non-zero reference accuracy.

    This is a regression test for the bug where ref_acc was unpacked but
    not used in the calculation, making the hypervolume only correct
    when ref_acc=0.0.

    Single point at (80, 0.2) with ref (50, 1.0)
    Area = (80 - 50) * (1.0 - 0.2) = 30 * 0.8 = 24
    """
    frontier = [make_outcome(80, 0.2)]
    ref_point = (50.0, 1.0)
    hv = compute_hypervolume_2d(frontier, ref_point)
    assert abs(hv - 24.0) < 1e-6, f"Expected 24.0, got {hv}"


def test_hypervolume_2d_nonzero_ref_acc_two_points():
    """Verify hypervolume with two points and non-zero ref_acc.

    Points: (80, 0.3) and (60, 0.1) with ref (50, 1.0)
    Sorted by acc descending: [(80, 0.3), (60, 0.1)]
    Area from (80, 0.3): (80 - 50) * (1.0 - 0.3) = 30 * 0.7 = 21
    Area from (60, 0.1): (60 - 50) * (0.3 - 0.1) = 10 * 0.2 = 2
    Total = 21 + 2 = 23
    """
    frontier = [make_outcome(80, 0.3), make_outcome(60, 0.1)]
    ref_point = (50.0, 1.0)
    hv = compute_hypervolume_2d(frontier, ref_point)
    assert abs(hv - 23.0) < 1e-6, f"Expected 23.0, got {hv}"


def test_hypervolume_2d_point_below_ref_acc_excluded():
    """Points at or below ref_acc should not contribute to hypervolume.

    Points: (80, 0.2) and (40, 0.1) with ref (50, 1.0)
    Only (80, 0.2) is above ref_acc=50, so:
    Area = (80 - 50) * (1.0 - 0.2) = 30 * 0.8 = 24
    (40, 0.1) is below ref_acc and excluded.
    """
    frontier = [make_outcome(80, 0.2), make_outcome(40, 0.1)]
    ref_point = (50.0, 1.0)
    hv = compute_hypervolume_2d(frontier, ref_point)
    assert abs(hv - 24.0) < 1e-6, f"Expected 24.0, got {hv}"


def test_hypervolume_2d_all_points_below_ref_acc():
    """All points below ref_acc should yield zero hypervolume."""
    frontier = [make_outcome(40, 0.2), make_outcome(30, 0.1)]
    ref_point = (50.0, 1.0)
    hv = compute_hypervolume_2d(frontier, ref_point)
    assert hv == 0.0, f"Expected 0.0 for all points below ref_acc, got {hv}"


def test_hypervolume_2d_point_at_ref_acc_excluded():
    """Point exactly at ref_acc should be excluded (not contribute)."""
    frontier = [make_outcome(50, 0.2)]  # Exactly at ref_acc
    ref_point = (50.0, 1.0)
    hv = compute_hypervolume_2d(frontier, ref_point)
    assert hv == 0.0, f"Expected 0.0 for point exactly at ref_acc, got {hv}"


def test_hypervolume_2d_point_at_ref_param_excluded():
    """Point at or above ref_param should be excluded."""
    frontier = [make_outcome(80, 1.0)]  # At ref_param
    ref_point = (0.0, 1.0)
    hv = compute_hypervolume_2d(frontier, ref_point)
    assert hv == 0.0, f"Expected 0.0 for point at ref_param, got {hv}"


# Property-based tests
@given(st.lists(
    st.tuples(
        st.floats(min_value=0, max_value=100),
        st.floats(min_value=0.01, max_value=1.0),
    ),
    min_size=0, max_size=20
))
def test_pareto_frontier_is_non_dominated(points):
    """Property: all frontier points are non-dominated."""
    outcomes = [make_outcome(acc, param) for acc, param in points]
    frontier = extract_pareto_frontier(outcomes)

    for f_point in frontier:
        for other in outcomes:
            if other is not f_point:
                assert not other.dominates(f_point)


@given(st.lists(
    st.tuples(
        st.floats(min_value=0, max_value=100),
        st.floats(min_value=0.01, max_value=1.0),
    ),
    min_size=0, max_size=20
))
def test_pareto_frontier_covers_all_non_dominated(points):
    """Property: frontier contains all non-dominated points."""
    outcomes = [make_outcome(acc, param) for acc, param in points]
    frontier = extract_pareto_frontier(outcomes)
    frontier_set = set(id(o) for o in frontier)

    for outcome in outcomes:
        is_dominated = any(other.dominates(outcome) for other in outcomes if other is not outcome)
        if not is_dominated:
            assert id(outcome) in frontier_set


@given(st.lists(
    st.tuples(
        st.floats(min_value=1, max_value=100),
        st.floats(min_value=0.01, max_value=0.99),
    ),
    min_size=1, max_size=10
))
def test_hypervolume_is_non_negative(points):
    """Property: hypervolume is always non-negative."""
    outcomes = [make_outcome(acc, param) for acc, param in points]
    frontier = extract_pareto_frontier(outcomes)
    ref_point = (0.0, 1.0)
    hv = compute_hypervolume_2d(frontier, ref_point)
    assert hv >= 0
