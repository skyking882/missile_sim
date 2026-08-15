from aim120_model.h4_coverage import coverage_report, missing_target_ranges


def test_coverage_reports_gaps_and_adjacent_overlap():
    rows = [
        {"trajectory_id": "G1", "mach": 2.6, "time_s": 0.0},
        {"trajectory_id": "G1", "mach": 4.5, "time_s": 1.0},
        {"trajectory_id": "G2", "mach": 1.1, "time_s": 0.0},
        {"trajectory_id": "G2", "mach": 3.2, "time_s": 1.0},
        {"trajectory_id": "G3", "mach": 0.2, "time_s": 0.0},
        {"trajectory_id": "G3", "mach": 1.4, "time_s": 1.0},
    ]
    report = coverage_report(rows, (0.2, 4.5), 0.3)
    assert report["trajectory_count"] == 3
    assert report["missing_target_ranges"] == []
    assert report["adjacent_overlap"][0]["meets_target_width"]


def test_missing_ranges_are_not_filled_by_interpolation_policy():
    missing = missing_target_ranges([(0.3, 0.8), (1.2, 3.2)], (0.2, 4.5))
    assert missing == [(0.2, 0.3), (0.8, 1.2), (3.2, 4.5)]
