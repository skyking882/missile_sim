from aim120_model.tracking import AlphaBetaGate, TrackMode, TrackSolution


def test_alpha_beta_first_measurement_initializes_without_dt_dependency():
    gate = AlphaBetaGate(alpha=0.8, beta=0.05, search_range=10.0)
    assert gate.accepts(100.0, 0.0)
    gate.update(100.0, 0.0)
    assert gate.initialized is True
    assert gate.value == 100.0
    assert gate.rate == 0.0


def test_alpha_beta_update_matches_fixed_formula():
    gate = AlphaBetaGate(alpha=0.8, beta=0.05, search_range=10.0)
    gate.update(100.0, 1.0)
    assert gate.accepts(110.0, 1.0)
    gate.update(110.0, 1.0)
    assert abs(gate.value - 108.0) < 1e-12
    assert abs(gate.rate - 0.5) < 1e-12

    assert gate.accepts(115.0, 2.0)
    gate.update(115.0, 2.0)
    assert abs(gate.value - 113.8) < 1e-12
    assert abs(gate.rate - 0.65) < 1e-12


def test_alpha_beta_rejection_is_inclusive_at_boundary_and_does_not_mutate():
    gate = AlphaBetaGate(alpha=0.8, beta=0.05, search_range=10.0)
    gate.update(100.0, 1.0)
    assert gate.accepts(110.0, 1.0)
    before = (gate.value, gate.rate, gate.initialized)
    assert not gate.accepts(110.000001, 1.0)
    assert (gate.value, gate.rate, gate.initialized) == before


def test_track_solution_copies_vectors_and_reports_age():
    position = [1.0, 2.0, 3.0]
    velocity = [4.0, 5.0, 6.0]
    solution = TrackSolution(position, velocity, 1.0, 1.25, TrackMode.INERTIAL, True, "test")
    position[0] = 99.0
    velocity[0] = 99.0
    assert solution.position == (1.0, 2.0, 3.0)
    assert solution.velocity == (4.0, 5.0, 6.0)
    assert abs(solution.age_s - 0.25) < 1e-12
