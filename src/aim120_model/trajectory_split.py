"""Trajectory-level train/validation splitting for H4."""

from __future__ import annotations

from collections import OrderedDict
from typing import Any, Iterable, Mapping, Sequence


def trajectory_id(row: Mapping[str, Any]) -> str:
    return str(row.get("trajectory_id", row.get("case_id", "unknown")))


def group_by_trajectory(rows: Iterable[Mapping[str, Any]]) -> "OrderedDict[str, list[dict[str, Any]]]":
    grouped: "OrderedDict[str, list[dict[str, Any]]]" = OrderedDict()
    for row in rows:
        key = trajectory_id(row)
        grouped.setdefault(key, []).append(dict(row))
    return grouped


def leave_one_trajectory_out(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Return deterministic whole-trajectory folds; no point-level random split."""

    grouped = group_by_trajectory(rows)
    folds: list[dict[str, Any]] = []
    for validation_id in grouped:
        train_ids = [trajectory for trajectory in grouped if trajectory != validation_id]
        folds.append({
            "validation_trajectory_id": validation_id,
            "training_trajectory_ids": train_ids,
            "training_rows": sum(len(grouped[trajectory]) for trajectory in train_ids),
            "validation_rows": len(grouped[validation_id]),
        })
    return folds


def explicit_split(
    rows: Sequence[Mapping[str, Any]],
    training_ids: Sequence[str],
    validation_ids: Sequence[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    training_set = set(str(value) for value in training_ids)
    validation_set = set(str(value) for value in validation_ids)
    if training_set & validation_set:
        raise ValueError("training and validation trajectory IDs must be disjoint")
    grouped = group_by_trajectory(rows)
    unknown = (training_set | validation_set) - set(grouped)
    if unknown:
        raise KeyError("unknown trajectory IDs: " + ", ".join(sorted(unknown)))
    training = [row for key in training_ids for row in grouped[str(key)]]
    validation = [row for key in validation_ids for row in grouped[str(key)]]
    return training, validation


def equal_trajectory_row_weights(rows: Sequence[Mapping[str, Any]]) -> list[float]:
    grouped = group_by_trajectory(rows)
    counts = {key: len(value) for key, value in grouped.items()}
    return [1.0 / max(float(counts[trajectory_id(row)]), 1.0) for row in rows]


__all__ = [
    "equal_trajectory_row_weights",
    "explicit_split",
    "group_by_trajectory",
    "leave_one_trajectory_out",
    "trajectory_id",
]
