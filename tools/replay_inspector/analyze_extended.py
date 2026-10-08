"""Export all retained fields plus explicitly labelled geometry estimates."""
import argparse
import bisect
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics


def sub(a, b): return [x-y for x, y in zip(a, b)]
def dot(a, b): return sum(x*y for x, y in zip(a, b))
def norm(a): return math.sqrt(dot(a, a))
def mul(a, s): return [x*s for x in a]
def angle(a, b): return math.degrees(math.acos(max(-1, min(1, dot(a, b) / (norm(a)*norm(b))))))
def signed16(v): return int(v) - 65536 if int(v) >= 32768 else int(v)


def axes(e):
    # Dagor bank=X, heading=Y, attitude=Z; R_y(heading) R_z(attitude) R_x(bank).
    b, h, a = e
    cb, sb, ch, sh, ca, sa = math.cos(b), math.sin(b), math.cos(h), math.sin(h), math.cos(a), math.sin(a)
    return ([ch*ca, sa, -sh*ca],
            [-ch*sa*cb+sh*sb, ca*cb, sh*sa*cb+ch*sb],
            [ch*sa*sb+sh*cb, -ca*sb, -sh*sa*sb+ch*cb])


def at(track, t):
    times = [p["t_ms"] for p in track]
    if t < times[0] or t > times[-1]:
        return None
    i = bisect.bisect_right(times, t)
    i = min(max(i, 1), len(track)-1)
    a, b = track[i-1], track[i]
    dt = b["t_ms"]-a["t_ms"]
    w = (t-a["t_ms"])/dt
    return ([x+w*(y-x) for x,y in zip(a["xyz"], b["xyz"])], mul(sub(b["xyz"], a["xyz"]), 1000/dt), dt)


def slope(times, values, i):
    # Replay fields can contain a repeated b5 packet timestamp. Keep the raw
    # candidate clock in the export, but widen the finite difference window
    # until a nonzero interval is found for derived rates.
    lo, hi = max(0, i-1), min(len(times)-1, i+1)
    while hi > lo and times[hi] == times[lo]:
        if lo > 0: lo -= 1
        elif hi < len(times)-1: hi += 1
        else: break
    dt = times[hi]-times[lo]
    if dt == 0:
        return [0.0 for _ in values[i]]
    return mul(sub(values[hi], values[lo]), 1000/dt)


def write_table(path, rows):
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows({k: "NA" if v is None else v for k,v in row.items()} for row in rows)


cli = argparse.ArgumentParser(description=__doc__)
cli.add_argument("directory")
cli.add_argument("--target-name", required=True)
args = cli.parse_args()
root = Path(args.directory)
raw = json.loads((root/"raw_extended.json").read_text(encoding="utf-8"))
target = next(u for u in raw["units"] if u["player_internal_name"] == args.target_name)
assert len(raw["rockets"]) == 1, "This analysis is scoped to the supplied single-missile replay"
rocket = raw["rockets"][0]
owner = next(u for u in raw["units"] if u["eid"] == rocket["owner_eid"])
probes = [p for p in raw["weapon_field_probe"] if int(p["eid"]) == rocket["eid"]]
track = rocket["track"]
assert len(probes) == len(track)
assert all(p["time_ms"] == q["t_ms"] and [p[k] for k in ("x","y","z")] == q["xyz"] for p,q in zip(probes,track))
assert all(p["bool2"] == 0 for p in probes), "Missing attitude/kinematics fields need explicit handling"
times = [p["t_ms"] for p in track]
assert all(b>a for a,b in zip(times,times[1:]))
positions = [p["xyz"] for p in track]
velocities = [[signed16(p[f"b7_{axis}_raw_u16"])*4000/65535 for axis in "xyz"] for p in probes]
simulation_times = [p["b5"]*1000/48 for p in probes]
# b5 is retained verbatim; repeated packet clock values are a known gap.
rows = []
sampled_length = 0.0
for i, (p, q, v) in enumerate(zip(probes, track, velocities)):
    t = q["t_ms"]
    target_value = at(target["track"], t)
    owner_value = at(owner["track"], t)
    assert target_value and owner_value, "No extrapolation allowed"
    target_pos, target_vel, gap = target_value
    delta = sub(target_pos, q["xyz"])
    rv = sub(target_vel, v)
    basis = axes(q["euler_raw"])
    vb = [dot(v, axis) for axis in basis]
    fd = slope(times, positions, i)
    fd_sim = slope(simulation_times, positions, i)
    fb = [dot(fd, axis) for axis in basis]
    acceleration = slope(simulation_times, velocities, i)
    tangent = dot(acceleration, v)/norm(v)
    normal = norm(sub(acceleration, mul(v, tangent/norm(v))))
    dt = (times[i]-times[i-1])/1000 if i else None
    segment = math.dist(positions[i],positions[i-1]) if i else 0.0
    sampled_length += segment
    adjacent_speed = segment/dt if dt is not None else None
    row = {
        "replay_time_ms": t, "replay_time_s": t/1000,
        "since_entity_created_s": (t-rocket["created_at_ms"])/1000,
        "simulation_time_s_candidate": p["b5"]/48,
        "flight_time_sf1_s_candidate": p["sf1"],
        "x_m": q["xyz"][0], "y_m": q["xyz"][1], "z_m": q["xyz"][2],
        "dt_s": dt, "derived_world_speed_mps": adjacent_speed,
        "derived_world_speed_kmh": adjacent_speed*3.6 if adjacent_speed is not None else None,
        "sampled_path_length_m": sampled_length,
        "roll_rad_decoded": q["euler_raw"][0], "heading_rad_decoded": q["euler_raw"][1], "pitch_rad_decoded": q["euler_raw"][2],
        "roll_deg": math.degrees(q["euler_raw"][0]), "heading_deg": math.degrees(q["euler_raw"][1]), "pitch_deg": math.degrees(q["euler_raw"][2]),
        **{f"velocity_{axis}_mps_candidate": value for axis,value in zip("xyz",v)},
        "speed_mps_candidate": norm(v), "speed_kmh_candidate": norm(v)*3.6,
        "path_speed_mps_estimate": norm(fd),
        "path_speed_simulation_clock_mps_estimate": norm(fd_sim),
        "flight_path_angle_deg_estimate": math.degrees(math.atan2(v[1], math.hypot(v[0],v[2]))),
        "aoa_deg_estimate_zero_wind": math.degrees(math.atan2(-vb[1],vb[0])),
        "sideslip_deg_estimate_zero_wind": math.degrees(math.atan2(vb[2],math.hypot(vb[0],vb[1]))),
        "total_incidence_deg_estimate_zero_wind": angle(basis[0],v),
        "aoa_from_position_difference_deg_estimate": math.degrees(math.atan2(-fb[1],fb[0])),
        **{f"target_{axis}_m_interpolated": value for axis,value in zip("xyz",target_pos)},
        "target_eid": target["eid"], "target_bracketing_gap_ms": gap,
        "target_slant_distance_m": norm(delta), "target_horizontal_distance_m": math.hypot(delta[0],delta[2]),
        "target_height_difference_m": delta[1],
        "target_closing_speed_mps_estimate": -dot(delta,rv)/norm(delta),
        "target_off_boresight_deg_estimate": angle(basis[0],delta),
        "target_velocity_vector_angle_deg_estimate": angle(v,delta),
        "shooter_distance_m": norm(sub(owner_value[0],q["xyz"])),
        "kinematic_tangential_accel_mps2_estimate": tangent,
        "kinematic_normal_accel_g_estimate": normal/9.80665,
        **{f"angular_rate_{axis}_candidate": signed16(p[f"b8_{axis}_raw_u16"])*30/65535 for axis in "xyz"},
        **{f"raw_{k}": value for k,value in p.items() if k not in {"time_ms","eid","x","y","z"}},
    }
    rows.append(row)
write_table(root/"missile_all_fields.dat", rows)
for unit in raw["units"]:
    unit_rows = [{"time_ms": p["t_ms"], **dict(zip(("x_m","y_m","z_m"),p["xyz"])),
                  **dict(zip(("euler_x","euler_y","euler_z"),p["euler_raw"]))} for p in unit["track"]]
    write_table(root/f"unit_{unit['eid']:08x}_{unit['player_internal_name']}.dat",unit_rows)
if "creation_raw_fields" in rocket:
    (root/"missile_creation_fields.json").write_text(json.dumps(rocket["creation_raw_fields"], indent=2), encoding="utf-8")
if "missile_component_snapshots" in raw:
    (root/"missile_ecs_components.json").write_text(json.dumps(raw["missile_component_snapshots"],indent=2),encoding="utf-8")
if "weapon_field_probe_raw_hex" in raw:
    assert len(raw["weapon_field_probe_raw_hex"]) == len(raw["weapon_field_probe"])
    with (root/"weapon_sync_raw.jsonl").open("w",encoding="utf-8") as f:
        for p, payload in zip(raw["weapon_field_probe"], raw["weapon_field_probe_raw_hex"]):
            expected_bytes = (int(p["end_bit"])+7)//8-int(p["start_bit"])//8
            assert len(bytes.fromhex(payload)) == expected_bytes
            f.write(json.dumps({"fields":p, "byte_slice_hex":payload,
                "first_valid_bit_offset_in_slice":int(p["start_bit"])%8,
                "valid_bit_count":int(p["end_bit"]-p["start_bit"])})+"\n")
def limits(key): return [min(r[key] for r in rows),max(r[key] for r in rows)]
delta_speed = [abs(r["speed_mps_candidate"]-r["path_speed_mps_estimate"]) for r in rows[1:-1]]
delta_aoa = [abs(r["aoa_deg_estimate_zero_wind"]-r["aoa_from_position_difference_deg_estimate"]) for r in rows[1:-1]]
summary = {
    "source":raw["source"], "source_sha256":raw["source_sha256"],
    "source_unchanged":hashlib.sha256(Path(raw["source"]).read_bytes()).hexdigest()==raw["source_sha256"],
    "missile_eid":rocket["eid"], "target_eid":target["eid"], "target_name":target["player_internal_name"],
    "target_selection":"Replay entity name TargetAI_0 and mission assignment; not a decoded missile seeker target ID",
    "samples":len(rows), "column_count":len(rows[0]), "all_sync_packets_have_extended_fields":True,
    "target_samples":len(target["track"]), "target_interpolation":"Linear at missile packet times; no extrapolation",
    "first_distance_m":rows[0]["target_slant_distance_m"], "last_distance_m":rows[-1]["target_slant_distance_m"],
    "distance_m_range":limits("target_slant_distance_m"), "aoa_deg_estimate_range":limits("aoa_deg_estimate_zero_wind"),
    "sideslip_deg_estimate_range":limits("sideslip_deg_estimate_zero_wind"), "speed_mps_candidate_range":limits("speed_mps_candidate"),
    "speed_candidate_vs_position_derivative_median_absolute_difference_mps":statistics.median(delta_speed),
    "speed_candidate_vs_simulation_clock_derivative_median_absolute_difference_mps": statistics.median(abs(r["speed_mps_candidate"]-r["path_speed_simulation_clock_mps_estimate"]) for r in rows[1:-1]),
    "aoa_two_methods_median_absolute_difference_deg":statistics.median(delta_aoa),
    "seeker_skipped_bit_counts":sorted({int(p.get("skipped_bits",0)) for p in probes}),
    "ecs_component_snapshot_count":len(raw.get("missile_component_snapshots",[])),
    "warnings":["AoA is an estimate using decoded attitude, a signed interpretation of b7, and zero wind; not game Sensor AoA.",
        "b7/b8 original unsigned decoding and raw integers are preserved; signed interpretations need additional validation.",
        "Normal acceleration from sampled velocity is kinematic, not available overload or aerodynamic load.",
        "Target center distance at sampled/interpolated times is not a miss distance or impact verdict."]}
creation = rocket.get("creation_raw_fields", {})
if "u12_2" in creation:
    summary["first_velocity_candidate_vs_creation_u12_2_vector_error_mps"] = norm(sub(velocities[0],creation["u12_2"]))
if "creation_time" in creation:
    summary["first_tick_time_minus_creation_time_s"] = probes[0]["b5"]/48-creation["creation_time"]
    summary["sf1_vs_tick_age_max_difference_s"] = max(abs(p["sf1"]-(p["b5"]/48-creation["creation_time"])) for p in probes)
if "u16_1" in creation:
    b,h,a = track[0]["euler_raw"]
    c1,s1,c2,s2,c3,s3 = math.cos(h/2),math.sin(h/2),math.cos(a/2),math.sin(a/2),math.cos(b/2),math.sin(b/2)
    q = [c1*c2*s3+s1*s2*c3, s1*c2*c3+c1*s2*s3, c1*s2*c3-s1*c2*s3, c1*c2*c3-s1*s2*s3]
    rq = creation["u16_1"]
    summary["first_euler_vs_creation_quaternion_abs_dot"] = abs(dot(q,rq))/(norm(q)*norm(rq))
assert all(math.isfinite(v) for row in rows for v in row.values() if isinstance(v,(int,float)))
(root/"extended_summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8")
print(json.dumps(summary,ensure_ascii=False,indent=2))
