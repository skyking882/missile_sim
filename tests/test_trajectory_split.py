from aim120_model.trajectory_split import explicit_split, leave_one_trajectory_out


def _rows():
    return [
        {"trajectory_id": "G1", "time_s": 0.0},
        {"trajectory_id": "G1", "time_s": 1.0},
        {"trajectory_id": "G2", "time_s": 0.0},
        {"trajectory_id": "G3", "time_s": 0.0},
    ]


def test_leave_one_out_is_trajectory_level():
    folds = leave_one_trajectory_out(_rows())
    assert len(folds) == 3
    assert {fold["validation_trajectory_id"] for fold in folds} == {"G1", "G2", "G3"}
    for fold in folds:
        assert fold["validation_trajectory_id"] not in fold["training_trajectory_ids"]


def test_explicit_split_rejects_overlap_and_keeps_rows_together():
    train, validation = explicit_split(_rows(), ["G1", "G2"], ["G3"])
    assert len(train) == 3
    assert len(validation) == 1
