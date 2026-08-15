"""Local H5 zero-active-fin body-alpha-squared drag model.

The model deliberately reports effective drag area (CdA), not a unique
dimensionless drag coefficient or reference area.  ``alpha_rad`` is the only
angle accepted by the model core; degree values are converted at the input
boundary by :func:`alpha_rad_from_row`.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence, Tuple


def _finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def alpha_deg_to_rad(alpha_deg: float) -> float:
    """Convert displayed degrees to radians at the data boundary."""

    return math.radians(float(alpha_deg))


def alpha_rad_from_row(row: Mapping[str, Any]) -> float:
    """Return a row's body angle in radians.

    The explicit ``alpha_rad`` field wins.  Raw UI rows may carry only
    ``alpha_deg``; those are converted once here so no model calculation can
    accidentally square degree values.
    """

    if _finite(row.get("alpha_rad")):
        alpha_rad = float(row["alpha_rad"])
    elif _finite(row.get("alpha_deg")):
        alpha_rad = alpha_deg_to_rad(float(row["alpha_deg"]))
    else:
        raise ValueError("row requires alpha_rad or alpha_deg")
    if not math.isfinite(alpha_rad):
        raise ValueError("alpha must be finite")
    return alpha_rad


@dataclass(frozen=True)
class H4ShapePrior:
    """Frozen H4 local Mach shape used only as a narrow-window prior."""

    mach_knots: Tuple[float, ...]
    cda_knots_m2: Tuple[float, ...]
    reference_mach: float = 1.5

    def __post_init__(self) -> None:
        if len(self.mach_knots) != len(self.cda_knots_m2):
            raise ValueError("H4 Mach and CdA knot lengths differ")
        if len(self.mach_knots) < 2:
            raise ValueError("H4 shape prior requires at least two knots")
        if any(not _finite(value) for value in self.mach_knots + self.cda_knots_m2):
            raise ValueError("H4 shape prior contains non-finite values")
        if any(value <= 0.0 for value in self.cda_knots_m2):
            raise ValueError("H4 shape prior must remain positive")
        if any(right <= left for left, right in zip(self.mach_knots, self.mach_knots[1:])):
            raise ValueError("H4 Mach knots must be strictly increasing")

    @classmethod
    def from_json(cls, path: Path, reference_mach: float = 1.5) -> "H4ShapePrior":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            tuple(float(value) for value in payload["mach_knots"]),
            tuple(float(value) for value in payload["cda_knots_m2"]),
            reference_mach=float(reference_mach),
        )

    def cda_m2(self, mach: float) -> float:
        """Log-linearly interpolate the frozen H4 shape within its support."""

        value = float(mach)
        if value < self.mach_knots[0] or value > self.mach_knots[-1]:
            raise ValueError("Mach is outside the frozen H4 shape support")
        if value == self.mach_knots[0]:
            return self.cda_knots_m2[0]
        if value == self.mach_knots[-1]:
            return self.cda_knots_m2[-1]
        for left_m, right_m, left_cda, right_cda in zip(
            self.mach_knots,
            self.mach_knots[1:],
            self.cda_knots_m2,
            self.cda_knots_m2[1:],
        ):
            if left_m <= value <= right_m:
                fraction = (value - left_m) / (right_m - left_m)
                log_left = math.log(left_cda)
                log_right = math.log(right_cda)
                return math.exp(log_left + fraction * (log_right - log_left))
        raise AssertionError("Mach interpolation interval was not found")

    def delta_m2(self, mach: float) -> float:
        return self.cda_m2(float(mach)) - self.cda_m2(self.reference_mach)


@dataclass(frozen=True)
class BodyAlpha2Parameters:
    """H5 parameters at Mach 1.5.

    ``k_residual_1p5`` is the CxAoA=0 residual alpha-squared coefficient and
    ``s_cx_aoa_1p5`` is the observed coefficient slope per StatShark field
    unit.  Both have units m²/rad².
    """

    cda0_1p5: float
    k_residual_1p5: float
    s_cx_aoa_1p5: float

    def k_for_cx_aoa(self, cx_aoa: float) -> float:
        return self.k_residual_1p5 + float(cx_aoa) * self.s_cx_aoa_1p5

    def as_dict(self) -> dict[str, float]:
        return {
            "CdA0_1p5_m2": float(self.cda0_1p5),
            "K_residual_1p5_m2_per_rad2": float(self.k_residual_1p5),
            "S_CxAoA_1p5_m2_per_rad2_per_field_unit": float(self.s_cx_aoa_1p5),
            "K_alpha2_C0_m2_per_rad2": float(self.k_for_cx_aoa(0.0)),
            "K_alpha2_C9_m2_per_rad2": float(self.k_for_cx_aoa(9.0)),
            "K_alpha2_C18_m2_per_rad2": float(self.k_for_cx_aoa(18.0)),
        }


def body_alpha2_cda(
    mach: float,
    alpha_rad: float,
    cx_aoa: float,
    parameters: BodyAlpha2Parameters,
    h4_shape: Optional[H4ShapePrior] = None,
    use_h4_shape: bool = True,
) -> float:
    """Evaluate ``CdA0_1p5 + H4_shape_delta + K(c)*alpha_rad²``."""

    base_delta = h4_shape.delta_m2(float(mach)) if use_h4_shape and h4_shape else 0.0
    alpha_value = float(alpha_rad)
    return (
        float(parameters.cda0_1p5)
        + base_delta
        + parameters.k_for_cx_aoa(float(cx_aoa)) * alpha_value * alpha_value
    )


def body_alpha_power_cda(
    mach: float,
    alpha_rad: float,
    cx_aoa: float,
    parameters: BodyAlpha2Parameters,
    alpha_power: Optional[float] = 2.0,
    h4_shape: Optional[H4ShapePrior] = None,
    use_h4_shape: bool = True,
) -> float:
    """Evaluate a declared alpha-power diagnostic model.

    H5's production path calls :func:`body_alpha2_cda`.  This helper exists
    only for the read-only M0/M1/M2/M4 competition and synthetic diagnostics.
    ``alpha_power=None`` is the no-alpha M0 model.
    """

    base_delta = h4_shape.delta_m2(float(mach)) if use_h4_shape and h4_shape else 0.0
    alpha_value = abs(float(alpha_rad))
    alpha_term = 0.0 if alpha_power is None else alpha_value ** float(alpha_power)
    return (
        float(parameters.cda0_1p5)
        + base_delta
        + parameters.k_for_cx_aoa(float(cx_aoa)) * alpha_term
    )


def model_basis(
    mach: float,
    alpha_rad: float,
    cx_aoa: float,
    h4_shape: Optional[H4ShapePrior] = None,
    use_h4_shape: bool = True,
) -> Tuple[float, float, float]:
    """Return the linear basis for ``(CdA0, K_residual, S_CxAoA)``."""

    del mach  # The frozen shape is handled as a known offset below.
    del h4_shape, use_h4_shape
    alpha_squared = float(alpha_rad) * float(alpha_rad)
    return 1.0, alpha_squared, float(cx_aoa) * alpha_squared


def shape_delta_for_row(row: Mapping[str, Any], h4_shape: Optional[H4ShapePrior]) -> float:
    if h4_shape is None:
        return 0.0
    if not _finite(row.get("mach")):
        raise ValueError("row requires Mach when an H4 shape prior is used")
    return h4_shape.delta_m2(float(row["mach"]))


def _solve_linear_system(matrix: Sequence[Sequence[float]], vector: Sequence[float]) -> list[float]:
    """Solve a small dense system using pivoted Gaussian elimination."""

    n = len(vector)
    augmented = [list(float(value) for value in row) + [float(vector[index])] for index, row in enumerate(matrix)]
    for column in range(n):
        pivot = max(range(column, n), key=lambda row: abs(augmented[row][column]))
        if abs(augmented[pivot][column]) < 1.0e-15:
            raise ValueError("singular normal-equation matrix")
        if pivot != column:
            augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        divisor = augmented[column][column]
        augmented[column] = [value / divisor for value in augmented[column]]
        for row in range(n):
            if row == column:
                continue
            factor = augmented[row][column]
            if factor == 0.0:
                continue
            augmented[row] = [
                left - factor * right
                for left, right in zip(augmented[row], augmented[column])
            ]
    return [augmented[index][-1] for index in range(n)]


def _weighted_normal_equations(
    rows: Iterable[Mapping[str, Any]],
    h4_shape: Optional[H4ShapePrior],
    use_h4_shape: bool,
) -> Tuple[list[list[float]], list[float], int]:
    normal = [[0.0, 0.0, 0.0] for _ in range(3)]
    rhs = [0.0, 0.0, 0.0]
    count = 0
    for row in rows:
        if not _finite(row.get("observed_cda_m2")):
            continue
        alpha_rad = alpha_rad_from_row(row)
        basis = model_basis(
            float(row.get("mach", 1.5)),
            alpha_rad,
            float(row.get("cx_aoa", 9.0)),
            h4_shape,
            use_h4_shape,
        )
        target = float(row["observed_cda_m2"]) - (
            shape_delta_for_row(row, h4_shape) if use_h4_shape and h4_shape else 0.0
        )
        weight = max(float(row.get("weight", 1.0)), 0.0)
        if weight == 0.0:
            continue
        for left in range(3):
            rhs[left] += weight * basis[left] * target
            for right in range(3):
                normal[left][right] += weight * basis[left] * basis[right]
        count += 1
    return normal, rhs, count


def fit_cda_observations(
    rows: Iterable[Mapping[str, Any]],
    h4_shape: Optional[H4ShapePrior] = None,
    use_h4_shape: bool = True,
) -> dict[str, Any]:
    """Fit the three H5 coefficients to direct effective-CdA observations."""

    materialized = list(rows)
    normal, rhs, count = _weighted_normal_equations(materialized, h4_shape, use_h4_shape)
    if count < 3:
        raise ValueError("at least three finite CdA observations are required")
    coefficients = _solve_linear_system(normal, rhs)
    parameters = BodyAlpha2Parameters(*coefficients)
    residuals: list[float] = []
    for row in materialized:
        if not _finite(row.get("observed_cda_m2")):
            continue
        predicted = body_alpha2_cda(
            float(row.get("mach", 1.5)),
            alpha_rad_from_row(row),
            float(row.get("cx_aoa", 9.0)),
            parameters,
            h4_shape,
            use_h4_shape,
        )
        residuals.append(predicted - float(row["observed_cda_m2"]))
    rmse = math.sqrt(sum(value * value for value in residuals) / len(residuals)) if residuals else None
    return {
        "parameters": parameters,
        "sample_count": count,
        "residual_rmse_m2": rmse,
        "residual_max_abs_m2": max((abs(value) for value in residuals), default=None),
        "normal_matrix": normal,
    }


def predict_rows(
    rows: Iterable[Mapping[str, Any]],
    parameters: BodyAlpha2Parameters,
    h4_shape: Optional[H4ShapePrior] = None,
    use_h4_shape: bool = True,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        item["alpha_rad"] = alpha_rad_from_row(row)
        item["predicted_cda_m2"] = body_alpha2_cda(
            float(row.get("mach", 1.5)),
            item["alpha_rad"],
            float(row.get("cx_aoa", 9.0)),
            parameters,
            h4_shape,
            use_h4_shape,
        )
        output.append(item)
    return output


__all__ = [
    "BodyAlpha2Parameters",
    "H4ShapePrior",
    "alpha_deg_to_rad",
    "alpha_rad_from_row",
    "body_alpha2_cda",
    "body_alpha_power_cda",
    "fit_cda_observations",
    "model_basis",
    "predict_rows",
]
