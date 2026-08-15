import tempfile
from pathlib import Path

from aim120_model.trajectory import TabulatedTrajectory


def _write_csv(text: str) -> Path:
    directory = Path(tempfile.mkdtemp())
    path = directory / "trajectory.csv"
    path.write_text(text, encoding="utf-8")
    return path


def test_linear_trajectory_interpolates_position_and_constant_velocity():
    path = _write_csv("time_s,x_m,y_m,z_m\n0,0,100,0\n10,10,100,20\n")
    trajectory = TabulatedTrajectory.from_csv(path)
    state = trajectory.state_at(5.0)
    assert state.position == (5.0, 100.0, 10.0)
    assert state.velocity == (1.0, 0.0, 2.0)


def test_hermite_trajectory_matches_node_position_and_velocity():
    path = _write_csv(
        "time_s,x_m,y_m,z_m,vx_mps,vy_mps,vz_mps\n"
        "0,0,100,0,0,0,0\n"
        "10,10,110,20,0,0,0\n"
    )
    trajectory = TabulatedTrajectory.from_csv(path)
    start = trajectory.state_at(0.0)
    middle = trajectory.state_at(5.0)
    end = trajectory.state_at(10.0)
    assert start.position == (0.0, 100.0, 0.0)
    assert start.velocity == (0.0, 0.0, 0.0)
    assert middle.position == (5.0, 105.0, 10.0)
    assert middle.velocity == (1.5, 1.5, 3.0)
    assert end.position == (10.0, 110.0, 20.0)
    assert end.velocity == (0.0, 0.0, 0.0)


def test_trajectory_rejects_partial_velocity_columns_and_non_increasing_time():
    partial = _write_csv("time_s,x_m,y_m,z_m,vx_mps\n0,0,0,0,1\n1,1,0,0,1\n")
    try:
        TabulatedTrajectory.from_csv(partial)
    except ValueError as exc:
        assert "velocity columns" in str(exc)
    else:
        raise AssertionError("partial velocity group was accepted")

    repeated = _write_csv("time_s,x_m,y_m,z_m\n0,0,0,0\n0,1,0,0\n")
    try:
        TabulatedTrajectory.from_csv(repeated)
    except ValueError as exc:
        assert "strictly increasing" in str(exc)
    else:
        raise AssertionError("non-increasing time was accepted")


def test_trajectory_does_not_extrapolate_past_endpoints():
    trajectory = TabulatedTrajectory.from_csv(_write_csv("time_s,x_m,y_m,z_m\n0,0,0,0\n1,1,0,0\n"))
    for time_s in (-0.001, 1.001):
        try:
            trajectory.state_at(time_s)
        except ValueError as exc:
            assert "outside" in str(exc)
        else:
            raise AssertionError("out-of-range query was extrapolated")
