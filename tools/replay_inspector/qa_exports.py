"""Independently check exported files; this does not validate game physics."""
import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path


def check(directory):
    root = Path(directory)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    inventory = list(csv.DictReader((root / "projectiles.csv").open(encoding="utf-8-sig", newline="")))
    issues = []
    source = Path(manifest["source"])
    digest = hashlib.file_digest(source.open("rb"), "sha256").hexdigest()
    if digest != manifest["source_sha256"]:
        issues.append("Source SHA256 changed")
    models = Counter()
    total = duplicates = backwards = nonfinite = after_destroy = before_create = 0
    all_dt = []
    speeds = []
    for item in inventory:
        models[item["weapon_name"] or "unresolved"] += 1
        points = list(csv.DictReader((root / (item["record_id"] + ".dat")).open(encoding="utf-8", newline=""), delimiter="\t"))
        total += len(points)
        if len(points) != int(item["samples"]):
            issues.append(f"Sample count mismatch: {item['record_id']}")
        prev = None
        for point in points:
            t = int(point["replay_time_ms"])
            xyz = tuple(float(point[key]) if point[key] != "NA" else math.nan for key in ("x_m", "y_m", "z_m"))
            nonfinite += not all(math.isfinite(v) for v in xyz)
            before_create += bool(item["created_at_ms"]) and t < int(item["created_at_ms"])
            after_destroy += bool(item["destroyed_at_ms"]) and t > int(item["destroyed_at_ms"])
            if prev:
                dt = (t - prev[0]) / 1000
                duplicates += dt == 0
                backwards += dt < 0
                actual = point["derived_world_speed_mps"]
                if dt > 0:
                    all_dt.append(dt)
                    expected = math.dist(xyz, prev[1]) / dt
                    if actual == "NA" or not math.isclose(float(actual), expected, rel_tol=1e-12, abs_tol=1e-9):
                        issues.append(f"Derived speed mismatch: {item['record_id']} at {t}")
                    if actual != "NA":
                        speeds.append({"speed_mps": float(actual), "dt_s": dt, "record_id": item["record_id"], "weapon_name": item["weapon_name"], "time_ms": t})
                elif actual != "NA":
                    issues.append(f"Nonpositive dt has a speed: {item['record_id']} at {t}")
            prev = (t, xyz)
    if len(inventory) != manifest["exported_projectiles"] or total != manifest["exported_samples"]:
        issues.append("Manifest counts mismatch")
    if len(list(root.glob("rocket_*.dat"))) != len(inventory):
        issues.append("Unexpected trajectory file count")
    if len({r["entity_id"] for r in inventory}) != len(inventory):
        issues.append("Duplicate projectile entity ID")
    all_dt.sort()
    def percentile(q):
        return all_dt[round((len(all_dt) - 1) * q)] if all_dt else None
    result = {
        "source": manifest["source"], "source_sha256_unchanged": digest == manifest["source_sha256"],
        "replay_length_s": manifest["replay_length_ms"] / 1000,
        "projectiles": len(inventory), "samples": total, "models": dict(models.most_common()),
        "unresolved_weapon": models["unresolved"], "unresolved_owner": sum(not r["owner_unit"] for r in inventory),
        "duplicate_time_intervals": duplicates, "backward_time_intervals": backwards,
        "nonfinite_coordinate_samples": nonfinite, "samples_before_entity_creation": before_create,
        "samples_after_entity_destruction": after_destroy,
        "positive_dt_s": {"min": min(all_dt) if all_dt else None, "median": percentile(.5), "p95": percentile(.95), "max": max(all_dt) if all_dt else None},
        "intervals_over_1s": sum(dt > 1 for dt in all_dt),
        "top_derived_speeds_for_review": sorted(speeds, key=lambda r: r["speed_mps"], reverse=True)[:10],
        "structural_issues": issues,
        "validation_scope": "file identity, counts, finite values and arithmetic only; speed/position accuracy and complete missile coverage are not validated",
    }
    (root / "qa.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return not issues


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("directories", nargs="+")
    args = cli.parse_args()
    statuses = [check(path) for path in args.directories]
    raise SystemExit(0 if all(statuses) else 1)
