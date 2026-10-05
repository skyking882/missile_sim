"""One conditional War Thunder interface hypothesis, on cached 24 Hz rows.

Predeclared alternatives: exported control itself, and control * beta/sin(beta).
beta is hypot of the original local two-axis incidences. No division by control,
fitting, clock shift, alternative angle, new coefficient or forward run.
Compare both derivative stencils in startup/platform/unloading, excluding the
conservative gain-transition derivative halo. Report pitch/yaw separately and
all rows, not only source-saturated controls. A cap diagnostic is independent
of whether the full control mapping explains the observations.
"""
from pathlib import Path
import json
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "outputs/jari_offaxis/inverse_control/inverse_inputs.npz"
OUT = ROOT / "outputs/jari_offaxis/sinc_interface"
CASES = ("jari_259_0", "jari_259_30", "jari_259_60")
WINDOWS = (("startup", 14 / 48, .7), ("platform", .7, 2.3), ("unloading", 2.3, 3.5))
TOL = 1e-6


def stat(v):
    v = np.asarray(v)
    return {"RMS": np.sqrt(np.mean(v * v, axis=0)).tolist(),
        "mean": np.mean(v, axis=0).tolist(),
        "min": np.min(v, axis=0).tolist(), "max": np.max(v, axis=0).tolist(),
        "maximum_absolute": np.max(abs(v), axis=0).tolist()}


def ratio(beta):
    beta = np.asarray(beta)
    if np.any(beta < 0) or np.any(beta >= np.pi / 2):
        raise ValueError("Declared nonnegative beta domain below 90 degrees violated")
    out = np.ones_like(beta, dtype=float)
    nonzero = beta != 0
    out[nonzero] = beta[nonzero] / np.sin(beta[nonzero])
    return out


def main():
    before = (SOURCE.stat().st_size, SOURCE.stat().st_mtime_ns)
    cache = np.load(SOURCE)
    reports, arrays = {}, {}
    for cid in CASES:
        t = cache[cid + "__t"]
        local = cache[cid + "__local_incidence_rad"]
        source_control = cache[cid + "__source_controls_pitch_yaw"]
        beta = np.hypot(local[:, 0], local[:, 1])
        # Diagnose only the declared common original source support.
        support = (t >= 0) & (t <= 3.5 + TOL)
        factor = ratio(beta[support])
        t = t[support]; source_control = source_control[support]
        beta = beta[support]; local = local[support]
        halo = (t >= 14 / 48 - 2 / 24 - TOL) & (t <= 14 / 48 + 2 / 24 + TOL)
        prediction = source_control * factor[:, None]
        a = {"t": t, "local_rad": local, "beta_rad": beta, "sinc_inverse": factor,
            "halo": halo, "linear_prediction": source_control, "sinc_prediction": prediction}
        report = {"axis_order": ["pitch", "yaw"], "source_rows_in_0_to_3p5": len(t),
            "clock24_maximum_error_s": float(np.max(abs(t - np.round(t * 24) / 24))),
            "beta_maximum_deg": float(np.degrees(beta.max())), "windows": {}}
        for w in (3, 5):
            needed = cache[cid + f"__{w}__u_norm_req"][support]
            a[f"needed{w}"] = needed
            a[f"linear_residual{w}"] = source_control - needed
            a[f"sinc_residual{w}"] = prediction - needed
        for name, lo, hi in WINDOWS:
            nominal = (t >= lo - TOL) & (t <= hi + TOL)
            mask = nominal & ~halo
            rows = np.flatnonzero(mask)
            if not len(rows): raise AssertionError("Empty predefined diagnostic window")
            record = {"rows": int(mask.sum()), "halo_excluded_rows": int((nominal & halo).sum()),
                "original_support_s": [float(t[rows[0]]), float(t[rows[-1]])],
                "beta_deg": stat(np.degrees(beta[mask])), "sinc_inverse": stat(factor[mask]),
                "source_controls": stat(source_control[mask]), "stencils": {}}
            for w in (3, 5):
                need = a[f"needed{w}"][mask]
                linear = a[f"linear_residual{w}"][mask]
                corrected = a[f"sinc_residual{w}"][mask]
                lin_stats, cor_stats = stat(linear), stat(corrected)
                cap_one = np.maximum(abs(need) - 1, 0)
                cap_sinc = np.maximum(abs(need) - factor[mask, None], 0)
                record["stencils"][str(w)] = {"needed": stat(need),
                    "linear_prediction_minus_needed": lin_stats,
                    "sinc_prediction_minus_needed": cor_stats,
                    "sinc_RMS_minus_linear_RMS": (np.asarray(cor_stats["RMS"]) - np.asarray(lin_stats["RMS"])).tolist(),
                    "pointwise_absolute_error_improved_rows": np.sum(abs(corrected) < abs(linear), axis=0).tolist(),
                    "pointwise_absolute_error_worsened_rows": np.sum(abs(corrected) > abs(linear), axis=0).tolist(),
                    "cap1_exceeded_rows": np.sum(cap_one > 0, axis=0).tolist(),
                    "cap_sinc_exceeded_rows": np.sum(cap_sinc > 0, axis=0).tolist(),
                    "cap1_excess": stat(cap_one), "cap_sinc_excess": stat(cap_sinc)}
            record["stencil_difference_needed5_minus3"] = stat(a["needed5"][mask] - a["needed3"][mask])
            report["windows"][name] = record
        reports[cid] = report
        arrays.update({cid + "__" + key: value for key, value in a.items()})
    verification = {"all_arrays_finite": all(np.isfinite(a).all() for a in arrays.values()),
        "zero_angle_limit_exact": float(ratio(np.array([0.]))[0]) == 1.,
        "input_cache_size_mtime_unchanged": before == (SOURCE.stat().st_size, SOURCE.stat().st_mtime_ns),
        "original_source_clock_only": max(r["clock24_maximum_error_s"] for r in reports.values()) < TOL,
        "new_integrations": 0, "fits": 0, "clock_shifts": 0, "candidate_count": 1}
    if not all(verification[k] for k in ("all_arrays_finite", "zero_angle_limit_exact", "input_cache_size_mtime_unchanged", "original_source_clock_only")):
        raise AssertionError(verification)
    payload = {"task": "One conditional beta/sin(beta) game-interface hypothesis",
        "formulas": {"linear": "u_pred = exported_control", "candidate": "u_pred = exported_control * beta/sin(beta)",
            "beta": "hypot(original frozen local_pitch, original frozen local_yaw)", "zero": "ratio(0)=1 analytically",
            "domain": "0 <= beta < pi/2 on inspected original source rows", "units": "dimensionless equivalent target normalized by original finsAoa"},
        "scope": "Original 24 Hz source rows, both noncausal 3/5-point angular inverses, gain-transition halo excluded, no fit or free trajectory",
        "classification_rule": "No combined score: compare per-window, per-axis residuals for both stencils; a stronger cap is not evidence that the complete control mapping is identified. All adverse windows remain reported.",
        "limitations": ["D-level mapping assumption; neither exported control units nor frozen passive angular law is independently proven.",
            "beta is a norm of the model local two-axis angles, not independently observed scalar geometry. Near zero beta/sin(beta) tends to one; finite angles use the exact declared expression.",
            "3/5 difference is numerical sensitivity, not a confidence interval; interpolation to 48 Hz is not used.",
            "No division by control, no source-feedback oracle or runtime source controls, and no claims about physical hardware."],
        "cases": reports, "verification": verification}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "results.json").write_text(json.dumps(payload, indent=2) + "\n")
    np.savez_compressed(OUT / "conditional.npz", **arrays)
    print(json.dumps(verification))
    for cid, report in reports.items():
        for name, record in report["windows"].items():
            print(cid, name, "rows", record["rows"], "support", record["original_support_s"],
                "factor", record["sinc_inverse"]["min"], record["sinc_inverse"]["max"])
            for w, r in record["stencils"].items():
                print(w, "RMS linear->sinc", r["linear_prediction_minus_needed"]["RMS"], r["sinc_prediction_minus_needed"]["RMS"],
                    "overcap1->sinc", r["cap1_exceeded_rows"], r["cap_sinc_exceeded_rows"])


if __name__ == "__main__": main()
