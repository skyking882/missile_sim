"""Low-g effective drag-area models used by H3.

H3 deliberately models the observable combination ``CdA``.  It does not
pretend that a trajectory can uniquely separate reference area from a hidden
drag coefficient table.  The sparse model is linear in its amplitudes once
the Mach transition locations and widths are frozen, which keeps the first
identification pass low dimensional and auditable.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class SparseCdaParameters:
    """Parameters for the positive sparse ``CdA0(M)`` representation."""

    a_sub_m2: float
    a_sup_m2: float
    a_wave_m2: float
    k_alpha_m2_per_rad2: float
    center: float = 1.0
    width: float = 0.45
    wave_width: float = 0.45


def smooth_step(mach: float, center: float = 1.0, width: float = 0.45) -> float:
    """Return a bounded, smooth transition from the subsonic to supersonic side."""

    safe_width = max(float(width), 1.0e-9)
    return 0.5 * (1.0 + math.tanh((float(mach) - float(center)) / safe_width))


def wave_basis(mach: float, center: float = 1.0, width: float = 0.45) -> float:
    """Return the frozen-width transonic wave-drag basis."""

    safe_width = max(float(width), 1.0e-9)
    return math.exp(-((float(mach) - float(center)) / safe_width) ** 2)


def sparse_cda0(mach: float, params: SparseCdaParameters) -> float:
    """Evaluate the shared zero-AoA effective drag area in square metres."""

    step = smooth_step(mach, params.center, params.width)
    wave = wave_basis(mach, params.center, params.wave_width)
    value = (
        params.a_sub_m2 * (1.0 - step)
        + params.a_sup_m2 * step
        + params.a_wave_m2 * wave
    )
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError("CdA0(M) must be finite and positive")
    return value


def total_cda(
    mach: float,
    alpha_rad: float,
    powered: bool,
    params: SparseCdaParameters,
    burn_delta_m2: float = 0.0,
    burn_transonic_delta_m2: float = 0.0,
    burn_width: float | None = None,
) -> float:
    """Evaluate LG-0, LG-1, or LG-2 depending on burn corrections supplied."""

    alpha = float(alpha_rad)
    value = sparse_cda0(mach, params) + params.k_alpha_m2_per_rad2 * alpha * alpha
    if powered:
        value += float(burn_delta_m2)
        if burn_transonic_delta_m2:
            value += float(burn_transonic_delta_m2) * wave_basis(
                mach,
                params.center,
                params.width if burn_width is None else burn_width,
            )
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError("total effective CdA must be finite and positive")
    return value


def model_basis(
    mach: float,
    alpha_rad: float,
    powered: bool,
    level: str = "LG-0",
    center: float = 1.0,
    width: float = 0.45,
    wave_width: float = 0.45,
) -> tuple[float, ...]:
    """Return the linear basis for the requested low-g model level.

    The first four coefficients are ``A_sub``, ``A_sup``, ``A_wave`` and
    ``K_alpha``.  LG-1 appends a constant powered correction.  LG-2 appends
    both a constant and a frozen-width transonic powered correction.
    """

    step = smooth_step(mach, center, width)
    wave = wave_basis(mach, center, wave_width)
    basis: list[float] = [1.0 - step, step, wave, float(alpha_rad) ** 2]
    if level in {"LG-1", "LG-2"}:
        basis.append(1.0 if powered else 0.0)
    if level == "LG-2":
        basis.append((1.0 if powered else 0.0) * wave_basis(mach, center, width))
    if level not in {"LG-0", "LG-1", "LG-2"}:
        raise ValueError("level must be LG-0, LG-1, or LG-2")
    return tuple(basis)


def parameters_from_coefficients(
    coefficients: Sequence[float],
    level: str = "LG-0",
    center: float = 1.0,
    width: float = 0.45,
    wave_width: float = 0.45,
) -> tuple[SparseCdaParameters, float, float]:
    """Convert fitted linear coefficients to model parameters."""

    if len(coefficients) < 4:
        raise ValueError("at least four coefficients are required")
    params = SparseCdaParameters(
        a_sub_m2=float(coefficients[0]),
        a_sup_m2=float(coefficients[1]),
        a_wave_m2=float(coefficients[2]),
        k_alpha_m2_per_rad2=float(coefficients[3]),
        center=float(center),
        width=float(width),
        wave_width=float(wave_width),
    )
    burn_delta = float(coefficients[4]) if level in {"LG-1", "LG-2"} else 0.0
    burn_transonic = float(coefficients[5]) if level == "LG-2" else 0.0
    return params, burn_delta, burn_transonic


def prediction_from_coefficients(
    mach: float,
    alpha_rad: float,
    powered: bool,
    coefficients: Sequence[float],
    level: str = "LG-0",
    center: float = 1.0,
    width: float = 0.45,
    wave_width: float = 0.45,
) -> float:
    """Evaluate a fitted coefficient vector without hiding the model level."""

    basis = model_basis(mach, alpha_rad, powered, level, center, width, wave_width)
    value = sum(float(a) * float(b) for a, b in zip(coefficients, basis))
    return float(value)


def coefficient_names(level: str) -> tuple[str, ...]:
    names = ["a_sub_m2", "a_sup_m2", "a_wave_m2", "k_alpha_m2_per_rad2"]
    if level in {"LG-1", "LG-2"}:
        names.append("delta_burn_m2")
    if level == "LG-2":
        names.append("delta_burn_transonic_m2")
    if level not in {"LG-0", "LG-1", "LG-2"}:
        raise ValueError("level must be LG-0, LG-1, or LG-2")
    return tuple(names)


def coefficient_bounds(level: str) -> tuple[tuple[float, float], ...]:
    """Conservative bounds for the first sparse identification pass."""

    bounds: list[tuple[float, float]] = [
        (1.0e-7, 0.20),  # A_sub
        (1.0e-7, 0.20),  # A_sup
        (0.0, 0.20),     # A_wave
        (0.0, 5.0),      # K_alpha
    ]
    if level in {"LG-1", "LG-2"}:
        bounds.append((-0.10, 0.10))
    if level == "LG-2":
        bounds.append((-0.10, 0.10))
    if level not in {"LG-0", "LG-1", "LG-2"}:
        raise ValueError("level must be LG-0, LG-1, or LG-2")
    return tuple(bounds)


def model_parameter_dict(
    coefficients: Sequence[float],
    level: str,
    center: float,
    width: float,
    wave_width: float,
) -> dict[str, float | str]:
    """Return JSON-ready parameters for reports."""

    params, burn_delta, burn_transonic = parameters_from_coefficients(
        coefficients, level, center, width, wave_width
    )
    values: dict[str, float | str] = {
        "level": level,
        "a_sub_m2": params.a_sub_m2,
        "a_sup_m2": params.a_sup_m2,
        "a_wave_m2": params.a_wave_m2,
        "k_alpha_m2_per_rad2": params.k_alpha_m2_per_rad2,
        "center": params.center,
        "width": params.width,
        "wave_width": params.wave_width,
    }
    if level in {"LG-1", "LG-2"}:
        values["delta_burn_m2"] = burn_delta
    if level == "LG-2":
        values["delta_burn_transonic_m2"] = burn_transonic
    return values


def fit_linear_bounded(
    rows: Sequence[Mapping[str, Any]],
    level: str,
    center: float = 1.0,
    width: float = 0.45,
    wave_width: float = 0.45,
    prior: Mapping[str, tuple[float, float]] | None = None,
    max_iterations: int = 400,
) -> list[float]:
    """Fit the linear amplitudes using deterministic bounded coordinate descent.

    ``prior`` maps a coefficient name to ``(value, ridge_weight)``.  It is a
    soft constraint used for the existing H2 high-Mach local anchor; it is
    never presented as a new StatShark observation.
    """

    names = coefficient_names(level)
    bounds = coefficient_bounds(level)
    observations: list[tuple[tuple[float, ...], float, float]] = []
    groups: dict[str, int] = {}
    for row in rows:
        group = "powered" if bool(row.get("powered", False)) else "coast"
        groups[group] = groups.get(group, 0) + 1
    for row in rows:
        target = row.get("observed_cda_m2")
        if target is None or not math.isfinite(float(target)):
            continue
        basis = model_basis(
            float(row["mach"]),
            float(row.get("alpha_rad", row.get("angle_of_attack_rad", 0.0))),
            bool(row.get("powered", False)),
            level,
            center,
            width,
            wave_width,
        )
        group = "powered" if bool(row.get("powered", False)) else "coast"
        weight = float(row.get("fit_weight", 1.0)) / max(float(groups.get(group, 1)), 1.0)
        observations.append((basis, float(target), weight))
    if not observations:
        raise ValueError("no finite observed CdA rows available for fitting")

    coefficients = [0.0 for _ in names]
    for index, (lower, upper) in enumerate(bounds):
        coefficients[index] = min(max(0.01, lower), upper)
    prior = prior or {}
    prior_index = {name: index for index, name in enumerate(names)}

    for _ in range(max_iterations):
        max_change = 0.0
        for index, name in enumerate(names):
            numerator = 0.0
            denominator = 0.0
            for basis, target, weight in observations:
                residual_without_current = target - sum(
                    coefficients[k] * basis[k]
                    for k in range(len(coefficients))
                    if k != index
                )
                numerator += weight * basis[index] * residual_without_current
                denominator += weight * basis[index] * basis[index]
            if name in prior:
                prior_value, ridge_weight = prior[name]
                numerator += float(ridge_weight) * float(prior_value)
                denominator += float(ridge_weight)
            if denominator <= 1.0e-18:
                continue
            lower, upper = bounds[index]
            updated = min(max(numerator / denominator, lower), upper)
            max_change = max(max_change, abs(updated - coefficients[index]))
            coefficients[index] = updated
        if max_change < 1.0e-12:
            break
    return coefficients


def evaluate_rows(
    rows: Sequence[Mapping[str, Any]],
    coefficients: Sequence[float],
    level: str,
    center: float = 1.0,
    width: float = 0.45,
    wave_width: float = 0.45,
) -> list[dict[str, Any]]:
    """Add model predictions and residuals without mutating input rows."""

    evaluated: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        target = item.get("observed_cda_m2")
        if target is not None and math.isfinite(float(target)):
            prediction = prediction_from_coefficients(
                float(item["mach"]),
                float(item.get("alpha_rad", item.get("angle_of_attack_rad", 0.0))),
                bool(item.get("powered", False)),
                coefficients,
                level,
                center,
                width,
                wave_width,
            )
            item["predicted_cda_m2"] = prediction
            item["cda_residual_m2"] = float(target) - prediction
        else:
            item["predicted_cda_m2"] = None
            item["cda_residual_m2"] = None
        evaluated.append(item)
    return evaluated


__all__ = [
    "SparseCdaParameters",
    "coefficient_bounds",
    "coefficient_names",
    "evaluate_rows",
    "fit_linear_bounded",
    "model_basis",
    "model_parameter_dict",
    "parameters_from_coefficients",
    "prediction_from_coefficients",
    "smooth_step",
    "sparse_cda0",
    "total_cda",
    "wave_basis",
]
