"""Small dependency-free helpers shared by the isolated H6 analysis chain."""

from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, List, Mapping, Optional, Sequence, Tuple


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def as_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    if not finite(value):
        return default
    return float(value)


def sha256_file(path: Path) -> Optional[str]:
    path = Path(path)
    if not path.exists() or not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def wrap_angle(angle_rad: float) -> float:
    value = (float(angle_rad) + math.pi) % (2.0 * math.pi) - math.pi
    return value


def unwrap_angles(values: Sequence[float]) -> List[float]:
    if not values:
        return []
    output = [float(values[0])]
    for value in values[1:]:
        delta = wrap_angle(float(value) - output[-1])
        output.append(output[-1] + delta)
    return output


def derivative(values: Sequence[float], times: Sequence[float]) -> List[float]:
    if len(values) != len(times) or len(values) < 2:
        raise ValueError("derivative requires at least two aligned samples")
    for left, right in zip(times, times[1:]):
        if float(right) <= float(left):
            raise ValueError("time must be strictly increasing")
    result: List[float] = []
    result.append((float(values[1]) - float(values[0])) / (float(times[1]) - float(times[0])))
    for index in range(1, len(values) - 1):
        dt = float(times[index + 1]) - float(times[index - 1])
        result.append((float(values[index + 1]) - float(values[index - 1])) / dt)
    result.append((float(values[-1]) - float(values[-2])) / (float(times[-1]) - float(times[-2])))
    return result


def rms(values: Iterable[float]) -> Optional[float]:
    materialized = [float(value) for value in values]
    if not materialized:
        return None
    return math.sqrt(sum(value * value for value in materialized) / len(materialized))


def solve_linear_system(matrix: Sequence[Sequence[float]], vector: Sequence[float]) -> List[float]:
    """Solve a small dense system by pivoted Gaussian elimination."""

    n = len(vector)
    if len(matrix) != n or any(len(row) != n for row in matrix):
        raise ValueError("matrix dimensions do not match vector")
    augmented = [list(float(value) for value in row) + [float(vector[index])] for index, row in enumerate(matrix)]
    for column in range(n):
        pivot = max(range(column, n), key=lambda row: abs(augmented[row][column]))
        if abs(augmented[pivot][column]) < 1.0e-14:
            raise ValueError("singular linear system")
        if pivot != column:
            augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        divisor = augmented[column][column]
        augmented[column] = [value / divisor for value in augmented[column]]
        for row in range(n):
            if row == column:
                continue
            factor = augmented[row][column]
            if abs(factor) < 1.0e-30:
                continue
            augmented[row] = [left - factor * right for left, right in zip(augmented[row], augmented[column])]
    return [augmented[index][-1] for index in range(n)]


def normal_equations(rows: Sequence[Sequence[float]], targets: Sequence[float], ridge: float = 0.0) -> Tuple[List[List[float]], List[float]]:
    if len(rows) != len(targets) or not rows:
        raise ValueError("normal equations require aligned non-empty rows")
    width = len(rows[0])
    if width == 0 or any(len(row) != width for row in rows):
        raise ValueError("design rows must have a common non-zero width")
    matrix = [[0.0 for _ in range(width)] for _ in range(width)]
    rhs = [0.0 for _ in range(width)]
    for row, target in zip(rows, targets):
        for left in range(width):
            rhs[left] += float(row[left]) * float(target)
            for right in range(width):
                matrix[left][right] += float(row[left]) * float(row[right])
    for index in range(width):
        matrix[index][index] += float(ridge)
    return matrix, rhs


def matrix_rank(rows: Sequence[Sequence[float]], tolerance: float = 1.0e-10) -> int:
    """Return the numerical rank of a small dense matrix without NumPy."""

    if not rows:
        return 0
    matrix = [list(float(value) for value in row) for row in rows]
    row_count = len(matrix)
    column_count = len(matrix[0]) if matrix[0] else 0
    rank = 0
    for column in range(column_count):
        pivot = max(range(rank, row_count), key=lambda row: abs(matrix[row][column]), default=rank)
        if pivot >= row_count or abs(matrix[pivot][column]) <= tolerance:
            continue
        matrix[rank], matrix[pivot] = matrix[pivot], matrix[rank]
        divisor = matrix[rank][column]
        matrix[rank] = [value / divisor for value in matrix[rank]]
        for row in range(row_count):
            if row == rank:
                continue
            factor = matrix[row][column]
            if abs(factor) <= tolerance:
                continue
            matrix[row] = [left - factor * right for left, right in zip(matrix[row], matrix[rank])]
        rank += 1
        if rank == row_count:
            break
    return rank


def pearson(x_values: Sequence[float], y_values: Sequence[float]) -> Optional[float]:
    if len(x_values) != len(y_values) or len(x_values) < 2:
        return None
    x_mean = sum(float(value) for value in x_values) / len(x_values)
    y_mean = sum(float(value) for value in y_values) / len(y_values)
    numerator = sum((float(x) - x_mean) * (float(y) - y_mean) for x, y in zip(x_values, y_values))
    denominator = math.sqrt(
        sum((float(x) - x_mean) ** 2 for x in x_values)
        * sum((float(y) - y_mean) ** 2 for y in y_values)
    )
    return numerator / denominator if denominator > 0.0 else None


def mapping_get(mapping: Mapping[str, Any], *keys: str, default: Any = None) -> Any:
    for key in keys:
        if key in mapping:
            return mapping[key]
    return default

