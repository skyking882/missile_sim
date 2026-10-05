"""Read-only cached War Thunder pre-peak interface audit; no free integration.

Predeclared attribution: compare source 3/5 inverse with a causal conditional
controller, actual free input, and the frozen plant's free/reference states.
0.3--0.583334 is descriptive; 0.395833--0.5625 excludes the gain halo.
This cannot separately identify an unknown game input and a passive model law.
"""
import dataclasses
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
import numpy as np
import check_r771_lift_direction as geometry
from aim120_model.control import scheduled_pid_values
from aim120_model.r771_game_control import R771GameControl

OUT = ROOT / "outputs/jari_offaxis/prepeak_cause"
DT = 1 / 48
POINTS = (19 / 48, 20 / 48, 22 / 48, 24 / 48, 26 / 48, 27 / 48)


def interp(t, st, a):
    a = np.asarray(a)
    return np.interp(t, st, a) if a.ndim == 1 else np.column_stack([np.interp(t, st, a[:, j]) for j in range(a.shape[1])])


def stats(a, mask):
    a = np.asarray(a)[mask]
    return {"min": np.min(a, axis=0).tolist(), "max": np.max(a, axis=0).tolist(),
            "mean": np.mean(a, axis=0).tolist(), "RMS": np.sqrt(np.mean(a * a, axis=0)).tolist()}


def spans(t, mask):
    ids = np.flatnonzero(mask)
    if not len(ids): return []
    cuts = np.split(ids, np.flatnonzero(np.diff(ids) != 1) + 1)
    return [[float(t[c[0]]), float(t[c[-1]]), len(c)] for c in cuts]


def run():
    inverse = geometry.dec.audit.inverse
    fix = geometry.dec.audit.fix
    drag = geometry.dec.audit.drag
    inputs = inverse.source_inputs()
    iv = np.load(inverse.OUTPUT / "inverse_inputs.npz")
    coeff = drag.DragCoefficients(**json.loads((drag.OUTPUT / "fit.json").read_text())["coefficients"])
    out, arrays = {}, {}
    for cid in geometry.CASES:
        x = inputs[cid]; cfg = x["config"]
        full = json.loads((geometry.OUT / "full_runs" / (cid + ".json")).read_text())
        ticks = [tick for tick in full["ticks"] if tick["time_s"] <= 28 / 48 + 1e-8]
        t = np.arange(len(ticks)) / 48
        limits = np.radians([cfg["aerodynamics"][k] for k in ("horizontal_fin_aoa_limit_deg", "vertical_fin_aoa_limit_deg")])
        st = iv[cid + "__t"]
        a = {"t": t, "source_original_t": st,
             "source_controls": interp(t, st, iv[cid + "__source_controls_pitch_yaw"]),
             "free_held": np.array([v["held_fin_rad"] for v in ticks]),
             "free_request": np.array([v["request_g"] for v in ticks]),
             "free_feedback": np.array([v["feedback_g"] for v in ticks])}
        for w in (3, 5):
            for label, original in (("required_held", "old_held_req_rad"), ("observed_qdot", "omega_dot_rad_s2"), ("source_feedback", "source_feedback_pitch_yaw_G")):
                a[f"source{w}_{label}"] = interp(t, st, iv[cid + f"__{w}__" + original])
        same = geometry.make_model(cfg, coeff); same.initialize(dataclasses.replace(x["initial"]))
        free = geometry.make_model(cfg, coeff); free.initialize(dataclasses.replace(x["initial"]))
        controller = R771GameControl(); memory = [(0., 0., 0.), (0., 0., 0.)]
        rows = []
        for i, tick in enumerate(ticks):
            ts = t[i]; source_state = fix.source_state_at(x["ref"], ts)
            sample = min(full["samples"], key=lambda s: abs(s["time_s"] - tick["time_s"]))
            assert abs(sample["time_s"] - tick["time_s"]) < 1e-8
            free_state = fix.finite.sample_state(sample)
            ef = same.evaluate(source_state, ts, (0., 0.))
            req, details = same.request(source_state, x["target_at"](ts), ts, ef)
            k = same.current_control_k
            held = [0., 0.]; terms = []
            for axis in range(2):
                pid = scheduled_pid_values(cfg["control"]["pid"], ts + DT)
                if tick["gain"] > 1e-12:
                    term = geometry.dec.pid_terms(controller, axis, memory[axis], req[axis] - ef.feedback_g[axis], pid,
                        tick["gain"], k, same.current_force_nonlinear, limits[axis])
                    held[axis], memory[axis] = controller.update_axis(axis, ts, req[axis] - ef.feedback_g[axis], memory[axis], pid,
                        limits[axis], tick["gain"], k, same.current_force_nonlinear)
                    terms.append([term["P_rad"], term["I_provisional_rad"], term["D_rad"]])
                else: terms.append([0., 0., 0.])
            fzero = free.evaluate(free_state, ts, (0., 0.))
            free_speed = np.linalg.norm(np.asarray(free_state.velocity) - free.wind)
            free_ku = free.coefficients.control_gain * (1 + free.coefficients.speed_slope * (free_speed / free.speed0 - 1))
            free_k = free_ku / ((fzero.qbar / fzero.mass) / free.q_over_mass0 * free.coefficients.restoring_gain)
            fsameu = free.evaluate(free_state, ts, tuple(a["free_held"][i]))
            ssameu = same.evaluate(source_state, ts, tuple(a["free_held"][i]))
            # Ku is identical for each axis and is algebraically recovered from
            # the frozen law's unit old-held input, without a parameter fit.
            ku = same.evaluate(source_state, ts, (1., 0.)).angular_acceleration[0] - ef.angular_acceleration[0]
            rows.append({"conditional_held": held, "conditional_request": req, "conditional_feedback": ef.feedback_g,
                "conditional_terms_PID": terms, "conditional_K": k, "free_K": free_k, "Ku_reference": ku,
                "reference_passive_qdot": ef.angular_acceleration, "free_passive_qdot": fzero.angular_acceleration,
                "free_qdot_at_new_held": fsameu.angular_acceleration, "reference_qdot_at_free_held": ssameu.angular_acceleration,
                "free_rates": [free_state.pitch_rate, free_state.yaw_rate], "source_rates": [source_state.pitch_rate, source_state.yaw_rate],
                "free_local_rad": fzero.local_incidence, "source_local_rad": ef.local_incidence,
                "free_high": bool(math.hypot(*fzero.local_incidence) > fix.finite.ALPHA_S),
                "source_high": bool(math.hypot(*ef.local_incidence) > fix.finite.ALPHA_S),
                "free_normalized": free_k * a["free_held"][i] / limits,
                "conditional_normalized": k * np.array(held) / limits})
        for k in rows[0]: a[k] = np.asarray([v[k] for v in rows])
        descriptive = (t >= .3) & (t <= 28 / 48 + 1e-8)
        valid = (t >= 19 / 48 - 1e-8) & (t <= 27 / 48 + 1e-8)
        for w in (3, 5):
            a[f"source{w}_needed_normalized"] = a["conditional_K"][:, None] * a[f"source{w}_required_held"] / limits
            # Residual decomposition uses the same interpolated held input but
            # evaluates the actual source state, so source-law closure includes
            # the documented interpolation commutator (small, measured below).
            a[f"source{w}_controller_residual_qdot"] = a["Ku_reference"][:, None] * (a["conditional_held"] - a[f"source{w}_required_held"])
            a[f"source{w}_free_vs_conditional_input_qdot"] = a["Ku_reference"][:, None] * (a["free_held"] - a["conditional_held"])
            a[f"source{w}_state_shift_qdot"] = a["free_qdot_at_new_held"] - a["reference_qdot_at_free_held"]
            sum_terms = sum(a[f"source{w}_" + k] for k in ("controller_residual_qdot", "free_vs_conditional_input_qdot", "state_shift_qdot"))
            actual = a["free_qdot_at_new_held"] - a[f"source{w}_observed_qdot"]
            a[f"source{w}_decomposition_interpolation_residual"] = actual - sum_terms
        indices = [int(round(p * 48)) for p in POINTS]
        stable = valid & (t >= 20 / 48 - 1e-8)
        out[cid] = {"nonhalo_window_s": [19 / 48, 27 / 48], "nonhalo_rows": int(valid.sum()),
            "descriptive_free_over_target1_pitch_yaw": [spans(t, descriptive & (abs(a["free_normalized"][:, j]) > 1 + 1e-10)) for j in range(2)],
            "nonhalo_free_over_target1_pitch_yaw": [spans(t, valid & (abs(a["free_normalized"][:, j]) > 1 + 1e-10)) for j in range(2)],
            "first_free_R_less_than1_s": float(t[a["free_high"]][0]) if a["free_high"].any() else None,
            "stats": {key: stats(a[key], valid) for key in ("source3_needed_normalized", "source5_needed_normalized", "source_controls", "free_normalized", "conditional_normalized",
                "source5_controller_residual_qdot", "source5_free_vs_conditional_input_qdot", "source5_state_shift_qdot", "source5_decomposition_interpolation_residual")},
            "stencil_difference_needed_normalized": stats(a["source5_needed_normalized"] - a["source3_needed_normalized"], valid),
            "stencil_difference_after_0p416667": stats(a["source5_needed_normalized"] - a["source3_needed_normalized"], stable),
            "inverse5_minus_exported_control_conditional_units": stats(a["source5_needed_normalized"] - a["source_controls"], valid),
            "descriptive_free_normalized": stats(a["free_normalized"], descriptive),
            "points": [{key: a[key][i].tolist() for key in a if a[key].shape[0] == len(t)} for i in indices],
            "new_integrations": 0, "fits": 0}
        arrays.update({cid + "__" + k: v for k, v in a.items()})
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / "diagnostic.npz", **arrays)
    payload = {"cases": out, "classification": "Conditional game-input identification with frozen law; not recovered game internals",
        "predeclared_window": "Descriptive 0.3 to 28/48 s; accepted derivative window 19/48 to 27/48 s excludes conservative gain-transition halo ending 18/48 s.",
        "limits": ["Source derivative estimates at 24 Hz with 3/5 points; 48 Hz interpolation adds no source information.",
            "Frozen law and inverse source input are not independently identified. Agreement with exported controls is suggestive, whose units remain unknown.",
            "Free-state qdot is evaluated using newly issued control at the saved tick; this is instantaneous, not measured interval acceleration.",
            "No source feedback/control is introduced into runtime, no free integration or fit occurs."],
        "verification": {"all_arrays_finite": all(np.isfinite(v).all() for v in arrays.values()),
            "clock48_max_error": float(max(np.max(abs(arrays[cid + '__t'] * 48 - np.arange(len(arrays[cid + '__t'])))) for cid in geometry.CASES)),
            "actual_integrations": 0, "fits": 0}}
    if not payload["verification"]["all_arrays_finite"]: raise AssertionError("Nonfinite cached diagnostic")
    (OUT / "results.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({cid: {k: v for k, v in r.items() if k != "points"} for cid, r in out.items()}, indent=2))


if __name__ == "__main__": run()
