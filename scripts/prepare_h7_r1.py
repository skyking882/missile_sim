"""Prepare the isolated H7 R1 payload set; never submits Calculate."""
import copy
import hashlib
import json
import os
from datetime import datetime, timezone

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "data", "raw", "statshark_h7_controller_id")
H6RAW = os.path.join(ROOT, "data", "raw", "statshark_h6_fin_dynamics_recovery")
OUT = os.path.join(ROOT, "outputs", "h7_controller_id")
PLAN = os.path.abspath(os.path.join(ROOT, "..", "plan7.md"))
CONFIG = os.path.join(ROOT, "configs", "h7_controller_experiments.json")
PLANT = os.path.join(ROOT, "outputs", "h6_fin_dynamics_recovery", "effective_yaw_plant_fit.json")
BASE_REQUEST = os.path.join(H6RAW, "requests", "H6R_R1_SMOKE.request.json")
BASE_PARAMS = os.path.join(H6RAW, "model_snapshots", "H6R_SEM_F010_request_payload.json")

MODELS = [
    ("H7_PID_NOM_F010", 0.0086, 0.0565, 0.00025),
    ("H7_PID_P050_F010", 0.0043, 0.0565, 0.00025),
    ("H7_PID_P200_F010", 0.0172, 0.0565, 0.00025),
    ("H7_PID_I050_F010", 0.0086, 0.02825, 0.00025),
    ("H7_PID_I200_F010", 0.0086, 0.113, 0.00025),
    ("H7_PID_D025_F010", 0.0086, 0.0565, 0.0000625),
    ("H7_PID_D400_F010", 0.0086, 0.0565, 0.001),
]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def normalized_payload(params):
    p = copy.deepcopy(params)
    p["finsLatAccel"] = 4.22579
    return p


def main():
    for d in [RAW, "model_snapshots", "requests", "responses", "network_evidence"]:
        os.makedirs(os.path.join(RAW, d) if d != RAW else d, exist_ok=True)
    base_request = json.load(open(BASE_REQUEST, encoding="utf-8"))
    base_params = normalized_payload(json.load(open(BASE_PARAMS, encoding="utf-8"))["Parameters"])
    common = {"StartSpeed": 1775, "LaunchAltitude": 3000, "LaunchAngle": 0, "ClosureRate": 0, "InitialTargetDistance": 10000, "TargetAltitude": 3000, "TargetAzimuth": 25, "TargetCourse": 0, "TargetConstantGTurn": 0, "TargetVerticalCourse": 0, "Timestep": 0.01, "LaunchYaw": 0}
    requests = {}
    for idx, (name, p, i, d) in enumerate(MODELS, 1):
        model_id = "h7r1_%02d_%s" % (idx, name.lower())
        params = copy.deepcopy(base_params)
        params["pids"] = [{"time": 3.4028234663852886e38, "p": p, "i": i, "intgLim": 1, "d": d}]
        params["finsLatAccel"] = 4.22579
        payload = copy.deepcopy(common)
        payload["Missiles"] = [model_id]
        payload["CustomMissiles"] = [{"Id": model_id, "Parameters": params}]
        requests[name] = payload
        write(os.path.join(RAW, "model_snapshots", name + ".request_payload.json"), {"clone_name": name, "model_id": model_id, "save_close_reopen_readback": "pending external UI verification", "payload": {"Id": model_id, "Parameters": params}})
        write(os.path.join(RAW, "requests", name + ".request.json"), payload)
    # Contrast ignores IDs and the explicitly authorized PID fields.
    def strip(obj):
        x = copy.deepcopy(obj)
        x["Missiles"] = ["<ID>"]
        x["CustomMissiles"][0]["Id"] = "<ID>"
        pid = x["CustomMissiles"][0]["Parameters"]["pids"][0]
        for k in ["p", "i", "d"]: pid[k] = "<PID_%s>" % k
        return x
    stripped = {k: strip(v) for k, v in requests.items()}
    contrast = {"pairwise_non_pid_equal": all(stripped[a] == stripped[b] for a in stripped for b in stripped), "authorized_pid_fields": ["p", "i", "d"], "models": list(requests), "readback": "pending external UI verification"}
    write(os.path.join(RAW, "payload_contrast.json"), contrast)
    ledger = {"schema_version": 1, "stage": "H7-R1", "calculate_actions_used": 0, "actions": [], "hard_limit": 1, "status": "prepared_waiting_for_clone_save_reopen_and_captcha", "note": "No Calculate submitted by this preparation script."}
    write(os.path.join(RAW, "calculate_ledger.json"), ledger)
    write(os.path.join(RAW, "frozen_holdout_manifest.json"), {"stage": "H7-R1", "status": "not_applicable_r1_only", "r2_r3_r4_executed": False})
    manifest = {"schema_version": 1, "stage": "H7-R1", "identity": "Lunamax", "generated_at_utc": datetime.now(timezone.utc).isoformat(), "status": "prepared_waiting_for_external_clone_readback", "calculate_count": 0, "calculate_hard_limit": 1, "sources": {p: {"path": p, "sha256": sha256(p)} for p in [PLAN, CONFIG, PLANT]}, "h6r_plant_policy": "frozen_candidate_only_not_refit", "clone_count": len(MODELS), "clone_names": [x[0] for x in MODELS], "scene": common, "payload_contrast_path": os.path.join(RAW, "payload_contrast.json"), "raw_request_snapshots": True, "save_close_reopen_readback": "pending", "captcha_gate": "Verification Required"}
    write(os.path.join(RAW, "session_manifest.json"), manifest)
    print(json.dumps({"raw": RAW, "clones": len(MODELS), "contrast": contrast["pairwise_non_pid_equal"], "calculate": 0, "status": manifest["status"]}, indent=2))


if __name__ == "__main__":
    main()
