"""Fit an RM-10 drag-shape prior to the existing H4 effective CdA curve.

This script does not treat RM-10 as an AIM-120 analogue. It compares only the
normalized Mach dependence on their common supersonic interval, then uses the
measured RM-10 transonic shape to make a clearly labelled prior-only extension
below the first supported H4 knot.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
H4_PATH = PROJECT_ROOT / "outputs" / "h4_glide_drag" / "cda_knots_fit.json"
RM10_PATH = (
    PROJECT_ROOT
    / "data"
    / "reference_external"
    / "rm10"
    / "rm10_figure13_composite_digitization.csv"
)
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "rm10_shape_prior"


def log_linear_interp(x: np.ndarray, xp: np.ndarray, yp: np.ndarray) -> np.ndarray:
    """Positive interpolation matching the H4 log-linear model."""
    return np.exp(np.interp(x, xp, np.log(yp)))


def read_rm10(path: Path) -> dict[str, np.ndarray]:
    rows = list(csv.DictReader(path.open("r", encoding="utf-8", newline="")))
    return {
        "mach": np.array([float(row["mach"]) for row in rows]),
        "cd": np.array([float(row["cd_total"]) for row in rows]),
        "sigma": np.array([float(row["cd_uncertainty_approx"]) for row in rows]),
    }


def relative_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    frac = (predicted - actual) / actual
    return {
        "relative_rmse": float(np.sqrt(np.mean(frac**2))),
        "relative_mae": float(np.mean(np.abs(frac))),
        "max_absolute_relative_error": float(np.max(np.abs(frac))),
    }


def main() -> dict:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    h4 = json.loads(H4_PATH.read_text(encoding="utf-8"))
    h4_mach = np.asarray(h4["mach_knots"], dtype=float)
    h4_cda = np.asarray(h4["cda_knots_m2"], dtype=float)
    rm10 = read_rm10(RM10_PATH)

    # Use the clean common supersonic interval only. M=1.2--1.4 comes from a
    # flight envelope rather than the same 8x6 tunnel centerline, so it is not
    # allowed to influence the exponent fit.
    fit_mask = rm10["mach"] >= 1.5
    fit_mach = rm10["mach"][fit_mask]
    fit_rm_cd = rm10["cd"][fit_mask]
    fit_h4_cda = log_linear_interp(fit_mach, h4_mach, h4_cda)

    rm_anchor = float(np.interp(1.5, rm10["mach"], rm10["cd"]))
    h4_anchor = float(log_linear_interp(np.array([1.5]), h4_mach, h4_cda)[0])
    x = np.log(fit_rm_cd / rm_anchor)
    y = np.log(fit_h4_cda / h4_anchor)
    beta = float(np.dot(x, y) / np.dot(x, x))

    baseline = h4_anchor * (fit_rm_cd / rm_anchor)
    exponent_fit = h4_anchor * (fit_rm_cd / rm_anchor) ** beta
    baseline_metrics = relative_metrics(fit_h4_cda, baseline)
    fit_metrics = relative_metrics(fit_h4_cda, exponent_fit)

    # A one-at-a-time digitization sensitivity. This is not a statistical CI;
    # it shows how much beta moves under the manually assigned scan-reading error.
    fit_sigma = rm10["sigma"][fit_mask]
    beta_trials = []
    for index in range(len(fit_rm_cd)):
        for sign in (-1.0, 1.0):
            perturbed = fit_rm_cd.copy()
            perturbed[index] = max(1e-9, perturbed[index] + sign * fit_sigma[index])
            trial_x = np.log(perturbed / rm_anchor)
            denom = float(np.dot(trial_x, trial_x))
            if denom > 0:
                beta_trials.append(float(np.dot(trial_x, y) / denom))

    # Hybrid candidate: H4 is untouched at and above M=1.2. Below it, RM-10 is
    # used only as a normalized transonic prior anchored exactly at the H4 knot.
    prior_mach = rm10["mach"][rm10["mach"] < 1.2]
    rm_at_h4_join = float(np.interp(1.2, rm10["mach"], rm10["cd"]))
    h4_at_join = float(log_linear_interp(np.array([1.2]), h4_mach, h4_cda)[0])
    prior_cda = h4_at_join * np.interp(prior_mach, rm10["mach"], rm10["cd"]) / rm_at_h4_join
    hybrid_mach = np.concatenate([prior_mach, h4_mach])
    hybrid_cda = np.concatenate([prior_cda, h4_cda])
    hybrid_support = ["RM10_shape_prior_only"] * len(prior_mach) + [
        "H4_unchanged"
    ] * len(h4_mach)

    hybrid_path = OUTPUT_DIR / "rm10_h4_hybrid_nodes.csv"
    with hybrid_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["mach", "cda_m2", "support_label"])
        writer.writerows(zip(hybrid_mach, hybrid_cda, hybrid_support))

    comparison_rows = []
    for mach, rm_cd, actual, base, fitted in zip(
        fit_mach, fit_rm_cd, fit_h4_cda, baseline, exponent_fit
    ):
        comparison_rows.append(
            {
                "mach": float(mach),
                "rm10_cd_total": float(rm_cd),
                "h4_cda_m2": float(actual),
                "scaled_rm10_beta_1_cda_m2": float(base),
                "scaled_rm10_exponent_fit_cda_m2": float(fitted),
                "fit_relative_error": float((fitted - actual) / actual),
            }
        )

    result = {
        "schema_version": 1,
        "model_label": "RM10_shape_prior_check_against_H4_effective_small_alpha_CdA",
        "source": {
            "report": "NACA Report 1160 (NACA RM-10)",
            "figure": 13,
            "printed_page": 120,
            "reference_area": "maximum body cross-sectional area",
            "digitization": "manual representative composite from 400 dpi scan",
            "series": "146.5-inch flight envelope center joined to 8x6-foot wind-tunnel centerline at M=1.5",
        },
        "fit": {
            "equation": "CdA_fit(M)=CdA_H4(1.5)*[Cd_RM10(M)/Cd_RM10(1.5)]^beta",
            "mach_range": [float(fit_mach.min()), float(fit_mach.max())],
            "anchor_mach": 1.5,
            "h4_anchor_cda_m2": h4_anchor,
            "rm10_anchor_cd": rm_anchor,
            "beta": beta,
            "beta_digitization_one_at_a_time_min": float(min(beta_trials)),
            "beta_digitization_one_at_a_time_max": float(max(beta_trials)),
            "baseline_beta_1_metrics": baseline_metrics,
            "fitted_beta_metrics": fit_metrics,
            "comparison_rows": comparison_rows,
        },
        "hybrid": {
            "equation_below_m1_2": "CdA_prior(M)=CdA_H4(1.2)*Cd_RM10(M)/Cd_RM10(1.2)",
            "policy_at_or_above_m1_2": "retain H4 unchanged",
            "join_cda_m2": h4_at_join,
            "prior_only_mach_range": [float(prior_mach.min()), float(prior_mach.max())],
            "h4_unchanged_mach_range": [float(h4_mach.min()), float(h4_mach.max())],
            "nodes_csv": str(hybrid_path.relative_to(PROJECT_ROOT)),
        },
        "interpretation_boundary": (
            "RM-10 is a generic slender body with four fins, not an AIM-120. "
            "The absolute RM-10 Cd is not transferred. Values below M1.2 are "
            "a prior-only hypothesis and are not StatShark-identified support."
        ),
    }
    (OUTPUT_DIR / "fit_report.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    # Figure 1: normalized shape comparison and the hybrid candidate.
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.2), constrained_layout=True)
    ax = axes[0]
    dense = np.linspace(1.5, 2.5, 250)
    dense_rm = np.interp(dense, rm10["mach"], rm10["cd"])
    dense_h4 = log_linear_interp(dense, h4_mach, h4_cda)
    ax.plot(dense, dense_h4 / h4_anchor, color="#1565C0", lw=2.6, label="H4 normalized")
    ax.plot(dense, dense_rm / rm_anchor, color="#D97706", lw=2.2, label="RM-10 normalized (β=1)")
    ax.plot(
        dense,
        (dense_rm / rm_anchor) ** beta,
        color="#7C3AED",
        lw=2.2,
        ls="--",
        label=f"RM-10 shape fit (β={beta:.2f})",
    )
    ax.scatter(fit_mach, fit_h4_cda / h4_anchor, s=32, color="#1565C0", zorder=3)
    ax.set_title("Supersonic shape check (anchor M=1.5)")
    ax.set_xlabel("Mach")
    ax.set_ylabel("Normalized drag measure")
    ax.set_xlim(1.5, 2.5)
    ax.grid(True, alpha=0.23)
    ax.legend(frameon=False, fontsize=9)

    ax = axes[1]
    ax.plot(h4_mach, h4_cda, color="#1565C0", lw=2.7, marker="o", label="H4 unchanged")
    ax.plot(
        np.append(prior_mach, 1.2),
        np.append(prior_cda, h4_at_join),
        color="#D97706",
        lw=2.4,
        ls="--",
        marker="o",
        label="RM-10 transonic prior only",
    )
    ax.axvline(1.2, color="#6B7280", lw=1.2, ls=":")
    ax.text(1.225, max(hybrid_cda) * 0.965, "H4 join", color="#4B5563", fontsize=9)
    ax.set_title("Hybrid candidate: prior below M1.2 only")
    ax.set_xlabel("Mach")
    ax.set_ylabel("Effective CdA (m²)")
    ax.set_xlim(0.85, 4.55)
    ax.set_ylim(0.0, max(hybrid_cda) * 1.12)
    ax.grid(True, alpha=0.23)
    ax.legend(frameon=False, fontsize=9)

    fig.suptitle(
        "RM-10 is used as a shape prior, not an absolute AIM-120 drag model",
        fontsize=13,
        fontweight="bold",
    )
    fig.savefig(OUTPUT_DIR / "rm10_h4_shape_fit.png", dpi=180)
    plt.close(fig)
    return result


if __name__ == "__main__":
    summary = main()
    print(json.dumps(summary["fit"], indent=2, ensure_ascii=False))
