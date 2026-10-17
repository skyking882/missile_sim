"""Read-only WRPL trial exporter for WrplReplayParser develop 221be5c.

Run inside WSL with the built PyReplayParser module on PYTHONPATH.
XYZ and timestamps are decoded samples. Speed is a finite-difference estimate,
not a decoded IAS/TAS field. Entity creation is not guaranteed physical launch.
"""
from __future__ import annotations

import argparse
import atexit
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
import time


REVISION = "221be5cac8e772ab41eefb0e7dac35ce6fb5c39b"
UNKNOWN_TIME = 0xFFFFFFFF


def file_hash(path):
    h = hashlib.sha256()
    with path.open("rb") as src:
        for data in iter(lambda: src.read(1024 * 1024), b""):
            h.update(data)
    return h.hexdigest()


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def finite_or_none(value):
    return value if math.isfinite(value) else None


def export(args):
    import PyReplayParser as parser
    atexit.register(parser.ecs.clearPythonCallbacks)

    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    source = Path(args.replay).resolve()
    source_digest = file_hash(source)
    local_package = Path(args.local_package).resolve() if args.local_package else None
    package_files = [{"path": str(p.relative_to(local_package)), "sha256": file_hash(p)}
                     for p in sorted(local_package.rglob("*.blk"))] if local_package else []
    save_json(output / "input.json", {
        "source": str(source), "source_sha256": source_digest,
        "upstream_revision": REVISION, "game_path": args.game,
        "filters": {"weapon": args.weapon, "entity": args.entity},
        "local_package": str(local_package) if local_package else None,
        "local_package_blk_files": package_files,
    })
    registered = {}

    # Retain the original replay entity IDs, including their generations.
    def on_create(event, eid, rocket_component):
        registered[id(rocket_component)] = {
            "eid": eid.get_handle(),
            "rocket": rocket_component,
        }

    parser.ecs.registerEsSystem(on_create, "trial_record_rocket_eid", [], [], [parser.ecs.EventEntityCreatedType])
    parser.initialize(args.game, logfile_path=str(output / "parser.log"), fonts=False, lang=False, mis=False)
    if local_package:
        parser.mountLocalPackage(str(local_package))
    replay = parser.replay.Replay(str(source))
    if not replay.isValid:
        raise RuntimeError("Replay was rejected by the upstream parser")
    state = parser.ParserState(replay)
    started = time.monotonic()
    reader = replay.getReplayReader()
    state.LoadFromReader(reader)
    parser.g_log_flush()
    if state.replay_length_ms == UNKNOWN_TIME:
        raise RuntimeError("No end marker: refusing to label this a complete export")
    log_text = (output / "parser.log").read_text(encoding="utf-8", errors="replace")
    parser_errors = [line for line in log_text.splitlines() if "[E]" in line]

    # Upstream preserves destroyed entities for rewind. Check against all
    # observed creation callbacks so missed or empty trajectories stay visible.
    rockets = list(state.collect_all_rockets())
    rows = []
    total = 0
    missing_eid = 0
    for ordinal, rocket in enumerate(rockets, 1):
        registered_info = registered.get(id(rocket))
        eid = registered_info["eid"] if registered_info else None
        missing_eid += eid is None
        record_id = f"rocket_{eid:08x}" if eid is not None else f"unresolved_{ordinal:04d}"
        weapon = rocket.weapon_obj
        weapon_name = weapon.weapon_name if weapon else ""
        weapon_path = weapon.blk_path if weapon else ""
        if args.weapon and args.weapon.lower() not in (weapon_name + " " + weapon_path).lower():
            continue
        if args.entity and (eid is None or eid != int(args.entity, 0)):
            continue
        owner = rocket.owned_by
        created = rocket.created_at_ms
        destroyed = rocket.destroyed_at_ms
        history = rocket.positions.history
        points = []
        for index in range(len(history)):
            item = history[index]
            pos = item.value.location
            points.append((int(item.time_ms), float(pos.x), float(pos.y), float(pos.z)))
        if not points:
            continue
        derived = []
        positive_dt = []
        valid_speeds = []
        zero_dt = 0
        backwards = 0
        invalid_positions = 0
        distance = 0.0
        prev = None
        for timestamp, x, y, z in points:
            valid = all(math.isfinite(v) for v in (x, y, z))
            invalid_positions += not valid
            speed = None
            dt = None
            if prev is not None:
                dt = (timestamp - prev[0]) / 1000.0
                zero_dt += dt == 0
                backwards += dt < 0
                if dt > 0 and valid and all(math.isfinite(v) for v in prev[1:]):
                    ds = math.dist((x, y, z), prev[1:])
                    speed = ds / dt
                    distance += ds
                    valid_speeds.append(speed)
                    positive_dt.append(dt)
            derived.append([
                timestamp, timestamp / 1000.0,
                (timestamp - created) / 1000.0 if created != UNKNOWN_TIME else None,
                finite_or_none(x), finite_or_none(y), finite_or_none(z), dt,
                speed, speed * 3.6 if speed is not None else None, distance,
            ])
            prev = (timestamp, x, y, z)
        with (output / (record_id + ".dat")).open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle, delimiter="\t")
            writer.writerow([
                "replay_time_ms", "replay_time_s", "since_entity_created_s",
                "x_m", "y_m", "z_m", "dt_s", "derived_world_speed_mps",
                "derived_world_speed_kmh", "sampled_path_length_m",
            ])
            writer.writerows([["NA" if v is None else v for v in row] for row in derived])
        total += len(points)
        rows.append({
            "record_id": record_id, "entity_id": eid,
            "weapon_name": weapon_name or None, "weapon_path": weapon_path or None,
            "owner_entity_id": rocket.ownerEid.get_handle(),
            "owner_unit": owner.unit_name if owner else None,
            "created_at_ms": created if created != UNKNOWN_TIME else None,
            "destroyed_at_ms": destroyed if destroyed != UNKNOWN_TIME else None,
            "samples": len(points), "first_sample_ms": points[0][0], "last_sample_ms": points[-1][0],
            "median_dt_s": statistics.median(positive_dt) if positive_dt else None,
            "max_derived_speed_mps": max(valid_speeds) if valid_speeds else None,
            "sampled_path_length_m": distance, "duplicate_timestamps": zero_dt,
            "backward_timestamps": backwards, "nonfinite_positions": invalid_positions,
        })
    with (output / "projectiles.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        if rows:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    manifest = {
        "source": str(source), "source_sha256": source_digest, "upstream_revision": REVISION,
        "local_patches": ["Linux include casing", "Python callback cleanup before interpreter shutdown", "Explicit local package mounts and CDK include resolver", "Correct std::vector byte allocation in BLK include expansion", "Infer custom unit weapon slots from loaded BLK when shop metadata is absent"],
        "local_package": str(local_package) if local_package else None,
        "local_package_blk_files": package_files,
        "format_version": f"0x{replay.header.magic:08X}",
        "player_no": replay.header.playerNo, "mission": replay.header.mission_file,
        "replay_length_ms": int(state.replay_length_ms), "packets": int(state.current_packet_index),
        "parse_and_export_seconds": time.monotonic() - started,
        "observed_creation_callbacks": len(registered), "rockets_with_history": len(rockets),
        "exported_projectiles": len(rows), "exported_samples": total,
        "unresolved_original_entity_ids": missing_eid,
        "parser_error_count": len(parser_errors),
        "weapon_sync_error_count": sum("Failed to read We payload" in line for line in parser_errors),
        "parser_error_examples": parser_errors[:20],
        "filters": {"weapon": args.weapon, "entity": args.entity},
        "validation": "decoded and structurally checked; not validated against in-game Sensor view",
        "notes": [
            "Rocket is an upstream entity category and is not itself a missile model identifier.",
            "Creation/destruction timestamps are entity events, not validated launch/impact times.",
            "Coordinates use the upstream world frame (meters, Y documented as altitude).",
            "Derived speed is distance between samples divided by time; it is neither a native speed readout nor IAS/TAS.",
            "Orientation, angle of attack, available overload and seeker internals are not exported.",
            "Client replay coverage and precision can be incomplete.",
        ],
        "projectiles": rows,
    }
    save_json(output / "manifest.json", manifest)
    print(json.dumps({k: manifest[k] for k in ["source", "packets", "observed_creation_callbacks", "rockets_with_history", "exported_projectiles", "exported_samples", "unresolved_original_entity_ids"]}, indent=2), flush=True)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--game", required=True)
    cli.add_argument("--replay", required=True)
    cli.add_argument("--output", required=True, help="New output directory; existing directories are refused")
    cli.add_argument("--weapon", help="Case-insensitive substring of decoded weapon name/path")
    cli.add_argument("--entity", help="Original projectile entity ID, decimal or 0x hex")
    cli.add_argument("--local-package", help="Explicit local package root, e.g. content/pkg_local")
    opts = cli.parse_args()
    try:
        export(opts)
    except Exception as error:
        print(f"Export failed: {error}", file=sys.stderr, flush=True)
        raise
