#!/usr/bin/env python3
"""Bounded angle transfer test of the unchanged War Thunder game candidate.

Six predeclared public JARI requests and six distinct actual free integrations.
The current 0/30/60-degree trajectories are reused without reintegration.
No fitting, parameter selection, candidate change, or source-state forcing.
"""
from __future__ import annotations

import copy
import argparse
import dataclasses
import datetime
import json
import math
import os
import shutil
import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src")]
os.environ.setdefault("MPLCONFIGDIR", "/tmp/missle_sim_mpl")
import numpy as np
import check_r771_horizontal_sinc_cap as best
import check_r771_collision_pn_probe as probe
import compare_r771_jari_offaxis as comparison
import fit_r771_jari_capture as fields

OUT = ROOT / "outputs/jari_offaxis/angle_grid"
ENDPOINT = "https://jari-aerospace.top/api/simulate"
NEW_ANGLES = (5, 10, 15, 45, 75, -30)
REUSED = {0: "jari_259_0", 30: "jari_259_30", 60: "jari_259_60"}
WINDOWS = (("startup", 14 / 48, .7), ("platform", .7, 2.3),
           ("unloading", 2.3, 3.5), ("early", 0., 3.5), ("full_common", 0., math.inf))
CHANNELS = ("speed_mps", "alpha_deg", "normal_specific_load_g", "body_yaw_deg",
            "body_pitch_deg", "body_yaw_rate_deg_s", "body_pitch_rate_deg_s")
UNITS = {"speed_mps": "m/s", "alpha_deg": "deg", "normal_specific_load_g": "G",
         "body_yaw_deg": "deg", "body_pitch_deg": "deg", "body_yaw_rate_deg_s": "deg/s",
         "body_pitch_rate_deg_s": "deg/s"}
OTHER_OWNED_FILES = {ROOT / "scripts/analyze_r771_bias_guidance.py", ROOT / "scripts/analyze_r771_bias_control.py"}
OTHER_OWNED_OUTPUTS = {ROOT / "outputs/jari_offaxis/bias_discussion/guidance",
                       ROOT / "outputs/jari_offaxis/bias_discussion/control"}


def cid(angle):
    return "angle_" + ("m" if angle < 0 else "p") + str(abs(angle))


def dump(path, value):
    path.write_text(json.dumps(probe.serializable(value), indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def protected():
    answer = {}
    for folder in ("src", "config", "configs", "schemas", "docs", "scripts", "data", "outputs/jari_offaxis"):
        for path in (ROOT / folder).rglob("*"):
            if not path.is_file() or "__pycache__" in path.parts or ".git" in path.parts:
                continue
            if path == Path(__file__).resolve() or path in OTHER_OWNED_FILES or OUT in path.parents:
                continue
            if any(p in path.parents for p in OTHER_OWNED_OUTPUTS):
                continue
            stat = path.stat()
            answer[str(path)] = (stat.st_size, stat.st_mtime_ns)
    return answer


def payload(angle):
    request = json.loads((probe.OUT / "headon_config_request.json").read_text())
    request["target_position"] = [8000. * math.cos(math.radians(angle)), 6300,
                                   8000. * math.sin(math.radians(angle))]
    return request


def capture(manifest, *, resume_after_dns=False):
    """Unauthenticated documented public endpoint; no cookies or credentials."""
    expected = json.loads((comparison.DEFAULT_MANIFEST.parent / "jari_259_30_response.json").read_text())
    for angle in NEW_ANGLES:
        name = cid(angle)
        request = payload(angle)
        request_path = OUT / (name + "_request.json")
        response_path = OUT / (name + "_response.json")
        if request_path.exists() or response_path.exists():
            prior = [a for a in manifest["attempts"] if a["case_id"] == name]
            allowed = resume_after_dns and not response_path.exists() and len(prior) == 1 and prior[0]["status"] == "transport_error" and "nodename" in prior[0]["message"]
            if not allowed or json.loads(request_path.read_text()) != request:
                raise FileExistsError("No silent resubmission of a predeclared angle request")
        else:
            dump(request_path, request)
        attempt = {"case_id": name, "signed_offaxis_deg": angle, "request_path": str(request_path),
                   "endpoint": ENDPOINT, "started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat()}
        manifest["attempts"].append(attempt)
        dump(OUT / "manifest.json", manifest)
        req = urllib.request.Request(ENDPOINT, data=json.dumps(request).encode(), method="POST",
            headers={"Content-Type": "application/json", "Accept": "application/json", "User-Agent": "WarThunderGameComparison/1.0"})
        try:
            with urllib.request.urlopen(req, timeout=45) as response:
                body = response.read()
                status = response.status
                safe_headers = {k: v for k, v in response.headers.items()
                    if k.lower() in ("content-type", "date", "retry-after", "x-ratelimit-limit", "x-ratelimit-remaining", "x-ratelimit-reset")}
        except urllib.error.HTTPError as exc:
            body = exc.read()
            status = exc.code
            response_path.write_bytes(body)
            attempt.update(status="HTTP_error", http_status=status, response_path=str(response_path))
            manifest.update(status="source_request_blocked", blocking_error={"case_id": name, "http_status": status,
                "response_path": str(response_path), "message": "Actual public endpoint rejection; no alternative mode, quota bypass or retry"})
            dump(OUT / "manifest.json", manifest)
            print(json.dumps({"source_blocked": name, "http_status": status, "response_path": str(response_path)}), flush=True)
            break
        except urllib.error.URLError as exc:
            attempt.update(status="transport_error", message=str(exc.reason), received_HTTP_response=False)
            manifest.update(status="source_transport_blocked", blocking_error={"case_id": name, "message": str(exc.reason)})
            dump(OUT / "manifest.json", manifest)
            print(json.dumps({"source_transport_blocked": name, "message": str(exc.reason)}), flush=True)
            break
        response_path.write_bytes(body)
        run = json.loads(body)
        if status != 200 or run["profile_source"] != expected["profile_source"] or run["backend"] != expected["backend"]:
            attempt.update(status="metadata_mismatch", http_status=status, response_path=str(response_path),
                           actual_profile_source=run["profile_source"], actual_backend=run["backend"])
            dump(OUT / "manifest.json", manifest)
            raise AssertionError("Actual export backend/profile must match existing 2.59 reference")
        if any(run["scenario"][k] != v for k, v in request.items()):
            raise AssertionError("Actual scenario must equal declared request")
        comparison.parse_reference(run)
        attempt.update(status="completed", http_status=status, response_path=str(response_path), safe_response_headers=safe_headers,
                       actual_profile_source=run["profile_source"], actual_backend=run["backend"], source_summary=run["summary"])
        dump(OUT / "manifest.json", manifest)
        print(json.dumps({"actual_source_saved": name, "signed_offaxis_deg": angle, "source_rows": len(run["samples"]),
                          "backend": run["backend"], "response_path": str(response_path)}), flush=True)
    return manifest


def ingest_ui_export(angle, source_path):
    """Preserve bytes downloaded by the normal, actual JARI frontend."""
    if angle not in NEW_ANGLES:
        raise ValueError("Only the six originally declared new angles may be ingested")
    manifest=json.loads((OUT/"manifest.json").read_text())
    if any(a["case_id"]==cid(angle) and a["status"]=="completed"for a in manifest["attempts"]):
        raise FileExistsError("A successful original frontend export must not be replaced")
    wrapper=json.loads(source_path.read_text())
    if wrapper["kind"]!="missile_comparison" or wrapper["selected_missiles"]!=["su_r_77_1.blk"] or wrapper["errors"] or len(wrapper["runs"])!=1:
        raise ValueError("Expected one successful native frontend missile-comparison export")
    run=wrapper["runs"][0]
    expected=json.loads((comparison.DEFAULT_MANIFEST.parent/"jari_259_30_response.json").read_text())
    if run["backend"]!=expected["backend"] or run["profile_source"]!=expected["profile_source"]:
        raise AssertionError("Actual frontend backend/profile metadata differs from existing reference")
    declared=payload(angle)
    if any(run["scenario"][k]!=v for k,v in declared.items()if k!="target_position") or np.linalg.norm(np.asarray(run["scenario"]["target_position"])-declared["target_position"])>1e-8:
        raise AssertionError("Downloaded actual scenario differs from the declared normal form inputs")
    comparison.parse_reference(run)
    name=cid(angle);response_path=OUT/(name+"_response.json")
    if response_path.exists():
        old=json.loads(response_path.read_text())
        if old!={"error":"只接受本站发起的请求"}:
            raise FileExistsError("Existing source bytes may not be silently replaced")
        previous=OUT/(name+"_http403_response.json")
        if previous.exists():raise FileExistsError("Prior rejection artifact already present")
        response_path.rename(previous)
        for a in manifest["attempts"]:
            if a.get("response_path")==str(response_path):a["response_path"]=str(previous)
    raw_path=OUT/(name+"_frontend_download.json")
    if raw_path.exists():raise FileExistsError("Original downloaded wrapper must not be overwritten")
    shutil.copyfile(source_path,raw_path)
    dump(response_path,run)
    request_path=OUT/(name+"_request.json")
    if not request_path.exists():dump(request_path,declared)
    else:
        if json.loads(request_path.read_text())!=declared:raise AssertionError("Previous declared request differs")
    manifest["attempts"].append({"case_id":name,"signed_offaxis_deg":angle,"status":"completed", "route":"normal_JARI_website_calculator_and_JSON_download",
        "response_path":str(response_path),"request_path":str(request_path),"original_frontend_export_path":str(raw_path),"original_download_path":str(source_path),
        "actual_profile_source":run["profile_source"],"actual_backend":run["backend"],"source_summary":run["summary"],
        "captured_utc":datetime.datetime.now(datetime.timezone.utc).isoformat()})
    if "blocking_error"in manifest:manifest["blocking_error_prior_public_POST"]=manifest.pop("blocking_error")
    manifest.update(status="normal_frontend_capture_in_progress",frontend_route="Actual normal Chrome form submissions and JSON download; no source-header spoofing")
    dump(OUT/"manifest.json",manifest)
    print(json.dumps({"actual_source_saved":name,"source_rows":len(run["samples"]),"backend":run["backend"],"original_export_path":str(raw_path)}),flush=True)


def add_rates(observed, prediction, reference, result):
    q = fields.gap.continuous_quaternions(reference["quaternion_xyzw"][:, [3, 0, 1, 2]])
    for width in (3, 5):
        rates = fields.gap.body_rates(q, reference["t"], width)
        observed["body_yaw_rate_" + str(width) + "point_deg_s"] = np.degrees(-rates[:, 1])
        observed["body_pitch_rate_" + str(width) + "point_deg_s"] = np.degrees(rates[:, 2])
    observed["body_pitch_rate_deg_s"] = observed["body_pitch_rate_3point_deg_s"]
    prediction["body_pitch_rate_deg_s"] = np.degrees([s["pitch_rate_rad_s"] for s in result["samples"]])


def stats(values):
    a = np.asarray(values)
    if not len(a):
        raise ValueError("A claimed nonempty comparison window contains no rows")
    return {"n": len(a), "rmse": float(np.sqrt(np.mean(a*a))), "bias": float(np.mean(a)),
            "max_abs": float(np.max(np.abs(a)))}


def metrics(observed, prediction):
    answer = {}
    support = (observed["t"] >= prediction["t"][0] - 1e-9) & (observed["t"] <= prediction["t"][-1] + 1e-9)
    for name, low, high in WINDOWS:
        mask = support & (observed["t"] >= low - 1e-9) & (observed["t"] <= high + 1e-9)
        times = observed["t"][mask]
        if not len(times):
            answer[name] = {"compared_rows": 0, "metrics": None}
            continue
        values = {key: {**stats(np.interp(times, prediction["t"], prediction[key]) - observed[key][mask]), "units": UNITS[key]}
                  for key in CHANNELS}
        # Display-cadence three-point and five-point rate estimates are separate.
        for axis in ("yaw", "pitch"):
            key = "body_" + axis + "_rate_deg_s"
            for width in (3, 5):
                label = "body_" + axis + "_rate_" + str(width) + "point_deg_s"
                values[label] = {**stats(np.interp(times, prediction["t"], prediction[key]) - observed[label][mask]), "units": "deg/s"}
        answer[name] = {"requested_window_s": [low, None if math.isinf(high) else high],
                        "actual_compared_support_s": [float(times[0]), float(times[-1])], "compared_rows": len(times), "metrics": values}
    return {"windows": answer, "unrepresented_source_rows": np.flatnonzero(~support).tolist(),
            "original_time_unchanged": True, "clock_fit": False,
            "matching": "source original export timestamps; linear prediction interpolation, no cadence or clock fit"}


def first_peak(trajectory, key):
    """First local absolute maximum; retains sign for negative-angle yaw."""
    t, signed = np.asarray(trajectory["t"]), np.asarray(trajectory[key])
    value = np.abs(signed)
    ids = [i for i in range(1, len(t)-1) if .3 - 1e-9 <= t[i] <= 3.5 + 1e-9
           and value[i] > value[i-1] and value[i] >= value[i+1] and value[i] > 0]
    if not ids:
        return None
    i = ids[0]
    return {"time_s": float(t[i]), "signed_value": float(signed[i]), "absolute_value": float(value[i]), "units": UNITS[key]}


def events(t, value):
    value = np.asarray(value)
    nonzero = np.flatnonzero(value != 0)
    flips = [int(b) for a, b in zip(nonzero, nonzero[1:]) if value[a] * value[b] < 0]
    return {"first_nonzero_s": float(t[nonzero[0]]) if len(nonzero) else None,
            "first_sign_flip_s": float(t[flips[0]]) if flips else None, "sign_flip_count": len(flips)}


def tick_trace(result, focused_arrays):
    rows = result["ticks"]
    t = np.asarray([row["time_s"] for row in rows])
    trace = {"all_t": t}
    keys = ("gain", "pn_mps2", "loft_g", "loft_weight", "loft_current_weight", "horizontal_loft_gate",
            "horizontal_los_rate_rad_s", "request_g", "slew_original_request_g", "feedback_g", "held_fin_rad",
            "slew_horizontal_state_g", "slew_last_step_g", "slew_active_updates", "nonlinear_control_latched",
            "loft_rate_threshold_rad_s", "loft_los_rate_magnitude_rad_s")
    for key in keys:
        # On the zero-horizontal-rate branch optional horizontal/S110 fields
        # are genuinely absent. Preserve this boundary rather than invent them.
        if all(key in row for row in rows):
            trace[key] = np.asarray([row[key] for row in rows])
        else:
            mask = np.asarray([key in row for row in rows])
            trace[key + "__present_t"] = t[mask]
            trace[key + "__present_values"] = np.asarray([row[key] for row in rows if key in row])
    trace.update(focused_arrays)
    active = [row for row in rows if row["gain"] > 1e-12]
    at = np.asarray([row["time_s"] for row in active])
    summary = {"first_active_s": float(at[0]), "gain_min_max": [float(min(row["gain"] for row in rows)), float(max(row["gain"] for row in rows))],
               "first_high_latch_s": next((row["time_s"] for row in active if row["nonlinear_control_latched"]), None)}
    for axis, label in enumerate(("pitch", "yaw")):
        diagnostics = [row["control_runtime_diagnostics"][axis] for row in active]
        clipping = np.asarray([abs(d["natural_old_rad"]) > d["effective_old_held_limit_rad"] for d in diagnostics])
        aw = np.asarray([d["antiwindup_blocked"] for d in diagnostics])
        summary[label] = {"request": events(at, [row["request_g"][axis] for row in active]),
                          "feedback": events(at, [row["feedback_g"][axis] for row in active]),
                          "held": events(at, [row["held_fin_rad"][axis] for row in active]),
                          "D": events(at, [d["D_old_rad"] for d in diagnostics]),
                          "first_output_clip_s": float(at[np.flatnonzero(clipping)[0]]) if clipping.any() else None,
                          "clipped_active_ticks": int(clipping.sum()),
                          "first_AW_block_s": float(at[np.flatnonzero(aw)[0]]) if aw.any() else None,
                          "AW_blocked_active_ticks": int(aw.sum())}
    summary["record_semantics"] = "Actual freely integrated candidate ticks only; source-state reconstructed fields are kept in separate diagnostic arrays"
    return trace, summary


def threshold_summary(source, cfg):
    t = source["t"]
    rate = np.abs(source["los_bearing_rate_rad_s"])
    threshold = math.radians(abs(float(cfg["guidance"]["loft_omega_max_deg_s"])))
    result = {"raw_omega_max_field": cfg["guidance"]["loft_omega_max_deg_s"],
              "candidate_omega_max_interpretation_rad_s": threshold,
              "candidate_transition_1p3_threshold_rad_s": 1.3*threshold,
              "semantics": "Existing D-grade candidate thresholds; crossing these in a JARI trajectory does not recover JARI internal gating"}
    for name, low, high in WINDOWS:
        mask = (t >= low - 1e-9) & (t <= high + 1e-9)
        if not mask.any():
            result[name] = None
            continue
        s = rate[mask]
        result[name] = {"n": int(mask.sum()), "horizontal_LOS_abs_rad_s_min_max": [float(s.min()), float(s.max())],
                        "fraction_above_candidate_omega_max": float(np.mean(s > threshold)),
                        "fraction_above_candidate_transition": float(np.mean(s > 1.3*threshold)),
                        "source_state_diagnostic_gate_min_max": [float(source["candidate_loft_horizontal_gate"][mask].min()), float(source["candidate_loft_horizontal_gate"][mask].max())]}
    return result


def mirrored(trajectories, results):
    if -30 not in trajectories or 30 not in trajectories:
        return {"status": "unavailable_missing_successful_negative_or_positive_case"}
    report = {"matching": "Exact shared original timestamps only; no fitted clock", "fields": {}}
    for label in ("reference", "candidate"):
        positive = trajectories[30][label]
        negative = trajectories[-30][label]
        t, ip, im = np.intersect1d(positive["t"], negative["t"], return_indices=True)
        if not len(t):
            raise ValueError("Mirror pair has no exact common timestamps")
        data = {"exact_common_rows": len(t), "support_s": [float(t[0]), float(t[-1])], "separate_units": {}}
        for key in CHANNELS:
            sign = -1. if key in ("body_yaw_deg", "body_yaw_rate_deg_s") else 1.
            delta = negative[key][im] - sign * positive[key][ip]
            data["separate_units"][key] = {**stats(delta), "expected_mirror_sign": sign, "units": UNITS[key]}
        report["fields"][label] = data
    pr = trajectories[30]["source_diagnostic"]
    nr = trajectories[-30]["source_diagnostic"]
    t, ip, im = np.intersect1d(pr["t"], nr["t"], return_indices=True)
    controls_p, controls_m = pr["source_controls_raw"][ip], nr["source_controls_raw"][im]
    report["source_controls_raw_unidentified"] = {"exact_common_rows": len(t),
        "index0_same_sign_difference": stats(controls_m[:, 0] - controls_p[:, 0]),
        "index0_opposite_sign_difference": stats(controls_m[:, 0] + controls_p[:, 0]),
        "index1_same_sign_difference": stats(controls_m[:, 1] - controls_p[:, 1]),
        "index1_opposite_sign_difference": stats(controls_m[:, 1] + controls_p[:, 1]),
        "units": "unidentified raw export units; mirror sign is axis evidence, not fin-angle identification"}
    for label in ("source_summary", "candidate_summary"):
        p, m = results[cid(30)][label], results[cid(-30)][label]
        shared = {key: {"positive": p[key], "negative": m[key], "equal": p[key] == m[key]}
                  for key in p if key in m}
        report[label] = shared
    return report


def plot(trajectories, reports):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    order = sorted(trajectories)
    shown = CHANNELS[:5] + ("body_yaw_rate_deg_s",)
    fig, axes = plt.subplots(len(shown), len(order), figsize=(3.1*len(order), 13), sharex=True, squeeze=False)
    for j, angle in enumerate(order):
        for i, key in enumerate(shown):
            ax = axes[i,j]
            for label, color in (("reference", "#137e83"), ("candidate", "#dc8c30")):
                p = trajectories[angle][label]; mask = p["t"] <= 3.5
                ax.plot(p["t"][mask], p[key][mask], color=color, label="JARI" if label == "reference" else "Unchanged best free")
            if key == "body_yaw_rate_deg_s":
                p = trajectories[angle]["reference"]; mask = p["t"] <= 3.5
                ax.plot(p["t"][mask],p["body_yaw_rate_5point_deg_s"][mask],"--",color="#538bb2",lw=.8,label="JARI quat5")
            if i == 0: ax.set_title(f"{angle:+g}°" + (" (cache)" if angle in REUSED else " (new)"))
            if j == 0: ax.set_ylabel(key.replace("body_", "").replace("_", " ") + " (" + UNITS[key] + ")",fontsize=8)
            if i == len(shown)-1: ax.set_xlabel("Original time (s)")
            ax.set_xlim(0,3.5); ax.grid(alpha=.22); ax.axvspan(14/48,.7,color="#ccdce6",alpha=.16)
            ax.tick_params(labelsize=7)
    axes[0,0].legend(fontsize=7)
    fig.suptitle("War Thunder game: angle transfer, original configured loft, no refit\nJARI interim reference versus one unchanged candidate; source-state diagnostics never drive the candidate",fontsize=12)
    fig.tight_layout(rect=(0,0,1,.966));fig.savefig(OUT/"angle_response_grid.png",dpi=145);plt.close(fig)
    fig, axes = plt.subplots(2,3,figsize=(15,8))
    for ax,key in zip(axes.flat,shown):
        for window,_,_ in WINDOWS:
            values=[reports[cid(a)]["free_vs_source_errors"]["windows"][window]["metrics"][key]["rmse"] for a in order]
            ax.plot(order,values,"o-",markersize=4,label=window)
        ax.set_title(key.replace("body_", "").replace("_", " "));ax.set_xlabel("Signed initial off-axis angle (deg)")
        ax.set_ylabel("RMSE ("+UNITS[key]+")");ax.grid(alpha=.25);ax.legend(fontsize=7)
    fig.suptitle("Separate physical errors across angle; no combined score or parameter selection",fontsize=12)
    fig.tight_layout(rect=(0,0,1,.96));fig.savefig(OUT/"errors_vs_angle.png",dpi=145);plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume-after-sandbox-dns", action="store_true", help="Explicit permission-boundary resume; previous DNS attempt had no HTTP response")
    parser.add_argument("--ingest-ui-export", nargs=2, metavar=("ANGLE","DOWNLOADED_JSON"), help="Preserve actual normal-frontend bytes; no API call or free integration")
    parser.add_argument("--analyze-saved-ui",action="store_true",help="Run the original six forward cases once after all actual normal-frontend sources exist")
    args = parser.parse_args()
    if args.ingest_ui_export:
        ingest_ui_export(int(args.ingest_ui_export[0]),Path(args.ingest_ui_export[1]));return
    before = protected()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT/"full_runs").mkdir(exist_ok=True)
    if (OUT/"manifest.json").exists():
        manifest = json.loads((OUT/"manifest.json").read_text())
        if args.analyze_saved_ui:
            successful=[a["signed_offaxis_deg"]for a in manifest["attempts"]if a["status"]=="completed"]
            if sorted(successful)!=sorted(NEW_ANGLES) or list((OUT/"full_runs").glob("*.json")):
                raise ValueError("All six declared successful exports and zero prior new free runs are required")
            manifest["analysis_after_normal_frontend_capture"]=True
        else:
            if not args.resume_after_sandbox_dns or manifest["status"] != "source_transport_blocked" or len(manifest["attempts"]) != 1 or "nodename" not in manifest["attempts"][0]["message"] or list((OUT/"full_runs").glob("*.json")):
                raise FileExistsError("The declared six-source/six-forward run is not silently restartable")
            manifest["permission_boundary_resume"] = "Explicit resume after default sandbox DNS resolution failed before receiving HTTP; identical authorized public endpoint and request"
    else:
        if args.resume_after_sandbox_dns: raise FileNotFoundError("No original DNS failure receipt to resume")
        manifest = {"status":"started", "declared_new_signed_offaxis_deg":NEW_ANGLES, "reused_angles_deg": list(REUSED),
                    "maximum_new_successful_sources":6, "maximum_new_distinct_forward_integrations":6,
                    "loft_state":"original config", "attempts":[], "public_game_reference_only":True}
    if not args.analyze_saved_ui:manifest = capture(manifest, resume_after_dns=args.resume_after_sandbox_dns)
    profiles = json.loads(comparison.SNAPSHOT.read_text())
    coefficients = best.dec.audit.drag.DragCoefficients(**json.loads((best.dec.audit.drag.OUTPUT/"fit.json").read_text())["coefficients"])
    cases = {}; arrays = {}; trajectories = {}; new_forward = 0; failed_forward = []
    sources = [(a["signed_offaxis_deg"],Path(a["response_path"]),False) for a in manifest["attempts"] if a["status"] == "completed"]
    sources += [(angle,comparison.DEFAULT_MANIFEST.parent/(old+"_response.json"),True) for angle,old in REUSED.items()]
    cache = np.load(best.OUT/"trajectories.npz")
    for angle, source_path, reused in sources:
        name = cid(angle);run = json.loads(source_path.read_text()); ref, source_checks = comparison.parse_reference(run)
        profile, bridge = comparison.profile_for(run,{},profiles)
        cfg=copy.deepcopy(profile["_model_config"]);initial,target_at=comparison.free_inputs(run,ref,cfg)
        model=best.make_model(cfg,coefficients)
        if reused:
            path=best.OUT/"full_runs"/(REUSED[angle]+".json")
            result=json.loads(path.read_text());model.initialize(dataclasses.replace(initial))
        else:
            path=OUT/"full_runs"/(name+".json")
            if path.exists():raise FileExistsError("A declared free case must not be silently rerun")
            new_forward+=1
            try:
                result=model.run(dataclasses.replace(initial),target_at,end_time_s=cfg["performance"]["lifetime_s"])
            except ValueError as exc:
                if "forward domain" not in str(exc) and "forward branch" not in str(exc): raise
                failure={"case_id":name,"signed_offaxis_deg":angle,"source_path":str(source_path),"status":"candidate_forward_domain_exit", "actual_error":str(exc), "verified_prediction":False}
                failed_forward.append(failure);cases[name]=failure
                print(json.dumps(failure),flush=True);continue
            dump(path,result)
        prediction=fields.prediction_fields(result,model,run["scenario"])
        observed,source_diag,source_initial=probe.source_geometry(run,ref,cfg)
        add_rates(observed,prediction,ref,result)
        if reused:
            prefix=REUSED[angle]+"__candidate__"
            if not all(np.array_equal(prediction[key],cache[prefix+key]) for key in ("t",*best.PRIMARY,"body_yaw_deg","body_pitch_deg","body_yaw_rate_deg_s")):
                raise AssertionError("Reused best free trajectory must restore exact cached physical arrays")
        focused, focused_arrays=best.verify(model,result,cfg,coefficients,initial,target_at,result)
        focused.pop("physical_baseline_exact")  # self-comparison is not a baseline validation claim
        trace,trace_summary=tick_trace(result,focused_arrays)
        target_error=max(float(np.linalg.norm(np.asarray(s["target"])-(np.asarray(run["scenario"]["target_position"])+s["t"]*np.asarray(run["scenario"]["target_velocity"])))) for s in run["samples"])
        if target_error>=1e-7:raise AssertionError("Source target trajectory differs from the prescribed free target motion")
        geometry_error = np.linalg.norm(np.asarray(run["scenario"]["target_position"]) - np.asarray(payload(angle)["target_position"]))
        if geometry_error > 1e-8 or not np.array_equal(ref["velocity_xyz_mps"][0], [333.3333435058594,0.,0.]):
            raise AssertionError("Shared angle geometry/actual creation velocity must match")
        if any(run["scenario"][key]!=value for key,value in payload(angle).items() if key not in ("speed","target_position")):
            raise AssertionError("All reused/new cases must match the common declared scenario")
        cases[name]={"signed_offaxis_deg":angle,"status":"completed", "source_path":str(source_path),"reused_source_and_free":reused,
            "actual_profile_source":run["profile_source"], "actual_backend":run["backend"], "scenario":run["scenario"],
            "actual_creation_state":source_initial,"source_diagnostics":source_checks,"source_version_bridge":bridge,
            "source_target_linear_motion_max_error_m":target_error,"source_summary":run["summary"],"candidate_summary":result["summary"],
            "planned_target_trigonometry_roundoff_m":float(geometry_error),"scenario_nominal_speed_mps":run["scenario"]["speed"],
            "candidate_full_path":str(path),"raw_PID":cfg["control"]["pid"],"free_vs_source_errors":metrics(observed,prediction),
            "first_absolute_peaks_preserving_sign":{label:{key:first_peak(p,key)for key in CHANNELS[1:]}
                for label,p in (("reference",observed),("candidate",prediction))},
            "source_quaternion3_and5_first_rate_peaks":{str(width):{axis:first_peak({"t":observed["t"],"body_"+axis+"_rate_deg_s":observed["body_"+axis+"_rate_"+str(width)+"point_deg_s"]},"body_"+axis+"_rate_deg_s")for axis in("yaw","pitch")}for width in(3,5)},
            "source_horizontal_LOS_candidate_threshold_diagnostic":threshold_summary(source_diag,cfg),
            "actual_free_tick_events":trace_summary,"focused_verification":focused}
        for label,p in (("reference",observed),("candidate",prediction),("source_diagnostic",source_diag),("tick",trace)):
            for key,value in p.items():
                if key!="source_reported_loft":arrays[name+"__"+label+"__"+key]=np.asarray(value)
        trajectories[angle]={"reference":observed,"candidate":prediction,"source_diagnostic":source_diag}
        np.savez(OUT/"trajectories.npz",**arrays)
        dump(OUT/"results.json",{"status":"partial_completed_cases", "cases":cases,"new_actual_free_forwards":new_forward})
        print(json.dumps({"actual_free_completed":name,"reused":reused,"early_RMSE":{key:cases[name]["free_vs_source_errors"]["windows"]["early"]["metrics"][key]["rmse"]for key in best.PRIMARY},"source_path":str(source_path)}),flush=True)
    mirror=mirrored(trajectories,cases)
    plot(trajectories,cases)
    unchanged=protected()==before
    finite=all(np.isfinite(v).all()for v in arrays.values()if np.issubdtype(v.dtype,np.number))
    if not unchanged or not finite:raise AssertionError("Final protected-size/mtime or finite-array verification failed")
    output={"status":"completed_declared_angle_transfer" if len(sources)==9 and not failed_forward else "partial_or_blocked_angle_transfer",
        "predeclared_new_angles_deg":NEW_ANGLES,"new_POST_attempts":sum(a["status"] != "transport_error" for a in manifest["attempts"]),
        "transport_attempts_including_sandbox_DNS_failure":len(manifest["attempts"]),
        "new_successful_source_cases":sum(a["status"]=="completed"for a in manifest["attempts"]),
        "new_actual_free_forwards":new_forward,"failed_forward_cases":failed_forward,"reused_source_and_free_cases":len(REUSED),
        "new_native_forwards":0,"baseline_forwards":0,"new_fits_or_parameter_scans":0,"cases":cases,
        "positive_negative_30_mirror":mirror,"protected_size_mtime_unchanged":unchanged,"protected_file_count":len(before),
        "all_numeric_arrays_finite":finite,"other_worker_owned_paths_excluded": [str(p)for p in sorted(OTHER_OWNED_FILES|OTHER_OWNED_OUTPUTS)],
        "blocking_error":manifest.get("blocking_error"),
        "rate_semantics":"Quaternion3 and quaternion5 are separate derivative diagnostics (yaw=-bodyY,pitch=bodyZ); raw source omega/controls units unverified and not converted to actuator radians",
        "limitations":["JARI public game simulator is an interim reference, not recovered private game source or captured game telemetry.",
            "New angle cases test an unchanged hypothesis; they are not a new fit or evidence of recovered field units.",
            "Local .27 rocket fields equal the earlier .28 replay parent; JARI global/runtime parity beyond the export metadata remains unverified.",
            "Source-state guidance reconstruction at exported display cadence omits source 48Hz intermediate states; its minimum weight is diagnostic only.",
            "Source normal load is derived from exported velocity with gravity removed and includes normal thrust; endpoint/phase differencing uncertainty remains.",
            "Some source samples.loft are null; configured loft selection is verified but JARI internal activation is not exported.",
            "New native integrations are unnecessary because this test changes no candidate, frozen parameters, or runtime/default files.",
            "Signed-angle mirror uses exact common original times; complete comparison metrics retain unfavorable windows and omit no supported source rows."]}
    dump(OUT/"results.json",output)
    manifest.update(status=output["status"],new_actual_free_forwards=new_forward,reused_free_cases=len(REUSED),failed_forward_cases=failed_forward)
    dump(OUT/"manifest.json",manifest)
    print(json.dumps({"status":output["status"],"new_POST_attempts":output["new_POST_attempts"],"new_source_cases":output["new_successful_source_cases"],
        "new_actual_free_forwards":new_forward,"reused":len(REUSED),"protected_unchanged":unchanged,"results_path":str(OUT/"results.json")}),flush=True)


if __name__=="__main__":
    main()
