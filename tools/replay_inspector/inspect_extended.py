"""Inventory replay-backed unit tracks and optional instrumented weapon fields."""
import argparse
import atexit
import hashlib
import json
import math
import struct
from pathlib import Path

import PyReplayParser as parser


def vec(v):
    return [float(v.x), float(v.y), float(v.z)]


def history(obj):
    result = []
    h = obj.positions.history
    for i in range(len(h)):
        row = h[i]
        result.append({"t_ms": int(row.time_ms), "xyz": vec(row.value.location), "euler_raw": vec(row.value.euler)})
    return result


def clean(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {k: clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    return value


cli = argparse.ArgumentParser()
cli.add_argument("--game", required=True)
cli.add_argument("--replay", required=True)
cli.add_argument("--output", required=True)
cli.add_argument("--local-package", help="Optional local content package; defaults to GAME/content/pkg_local when present")
args = cli.parse_args()
local_package = Path(args.local_package) if args.local_package else Path(args.game) / "content/pkg_local"
if args.local_package and not local_package.is_dir():
    cli.error("--local-package must be an existing directory")
out = Path(args.output)
out.mkdir(parents=True, exist_ok=False)
atexit.register(parser.ecs.clearPythonCallbacks)
units_by_ptr, rockets_by_ptr = {}, {}
component_snapshots, last_components = [], {}


def component_value(value):
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (tuple, list)):
        return [component_value(v) for v in value]
    if hasattr(value, "get_handle"):
        return {"entity_id": value.get_handle()}
    if hasattr(value, "to_tuple"):
        return list(value.to_tuple())
    if hasattr(value, "creation_raw_fields"):
        return value.creation_raw_fields
    if hasattr(value, "componentName") and hasattr(value, "data"):
        type_name, data = value.componentName, bytes(value.data)
        formats = {"float": "<f", "double": "<d", "bool": "<?", "Point3": "<3f", "Point2": "<2f", "Point4": "<4f", "TMatrix": "<12f"}
        fmt = formats.get(type_name)
        if fmt and len(data) == struct.calcsize(fmt):
            decoded = struct.unpack(fmt, data)
            return {"type": type_name, "value": list(decoded), "decoded_storage_hex": data.hex()}
        return {"type": type_name, "decoded_storage_hex": data.hex(), "note": "Decoded process storage; not original wire bytes"}
    return {"type": type(value).__name__, "text": str(value)}


def snapshot_components(packet):
    for eid, _ in rockets_by_ptr.values():
        entity = state.mgr.getEntity(eid)
        if entity is None:
            continue
        names = entity.getComponentNames()
        components = {}
        safe_types = {"ecs::EntityId", "Rocket", "ecs::string", "bool", "float", "double", "uint8_t", "uint16_t", "uint32_t", "uint64_t", "int8_t", "int16_t", "int", "int64_t"}
        for kind, name in names:
            if kind in safe_types:
                components[name] = {"type": kind, "value": component_value(entity.getData(name))}
            elif kind == "ecs::Tag":
                components[name] = {"type": kind, "present": True}
            else:
                components[name] = {"type": kind, "not_read": "Upstream generic binding is unsafe for unsupported component types"}
        components = clean(components)
        if components != last_components.get(eid):
            component_snapshots.append({"time_ms": state.curr_time_ms, "eid": eid,
                "template": entity.templateName, "components": components})
            last_components[eid] = components


def on_unit(event, eid, unit__ref):
    if unit__ref.unit:
        unit = unit__ref.unit
        units_by_ptr[id(unit)] = (eid.get_handle(), unit)


def on_rocket(event, eid, rocket_component):
    rockets_by_ptr[id(rocket_component)] = (eid.get_handle(), rocket_component)


parser.ecs.registerEsSystem(on_unit, "extended_unit_ids", [], [], [parser.ecs.EventEntityCreatedType])
parser.ecs.registerEsSystem(on_rocket, "extended_rocket_ids", [], [], [parser.ecs.EventEntityCreatedType])
parser.initialize(args.game, logfile_path=str(out / "parser.log"), fonts=False, lang=False, mis=False)
if local_package.is_dir():
    parser.mountLocalPackage(str(local_package))
if hasattr(parser, "enableWeaponFieldProbe"):
    parser.enableWeaponFieldProbe()
replay = parser.replay.Replay(args.replay)
state = parser.ParserState(replay)
reader = replay.getReplayReader()
state.LoadFromReader(reader, snapshot_components)
if state.replay_length_ms == 0xFFFFFFFF:
    raise RuntimeError("No replay end marker")
unit_rows = []
for _, unit in units_by_ptr.values():
    unit_rows.append({"eid": units_by_ptr.get(id(unit), (None,))[0], "uid": unit.uid,
        "unit_name": unit.unit_name, "player_internal_name": unit.player_internal_name,
        "owner_pid": unit.owner_pid, "spawn_matrix_repr": str(unit.spawn_position),
        "created_at_ms": unit.created_at_ms, "killed_at_ms": unit.killed_at_ms,
        "destroyed_at_ms": unit.destroyed_at_ms, "track": history(unit)})
rocket_rows = []
for rocket in state.collect_all_rockets():
    rocket_rows.append({"eid": rockets_by_ptr.get(id(rocket), (None,))[0],
        "owner_eid": rocket.ownerEid.get_handle(), "eid2": rocket.eid2.get_handle(),
        "weapon_name": rocket.weapon_obj.weapon_name if rocket.weapon_obj else None,
        "created_at_ms": rocket.created_at_ms, "destroyed_at_ms": rocket.destroyed_at_ms,
        "track": history(rocket)})
    if hasattr(rocket, "creation_raw_fields"):
        rocket_rows[-1]["creation_raw_fields"] = rocket.creation_raw_fields
raw = {"source": args.replay, "source_sha256": hashlib.sha256(Path(args.replay).read_bytes()).hexdigest(),
    "backend_revision": "221be5cac8e772ab41eefb0e7dac35ce6fb5c39b",
    "capture_method": "Local backend with optional field retention and raw bit-fragment capture; preserved original decoding",
    "capture_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    "replay_length_ms": state.replay_length_ms, "units": unit_rows, "rockets": rocket_rows,
    "missile_component_snapshots": component_snapshots}
if hasattr(parser, "getWeaponFieldProbe"):
    raw["weapon_field_probe"] = parser.getWeaponFieldProbe()
if hasattr(parser, "getWeaponFieldProbeHex"):
    raw["weapon_field_probe_raw_hex"] = parser.getWeaponFieldProbeHex()
(out / "raw_extended.json").write_text(json.dumps(clean(raw), indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
print(f"Saved {len(rocket_rows)} rocket, {len(unit_rows)} unit tracks and {len(component_snapshots)} component snapshots", flush=True)
print(json.dumps({"units": [{k: v for k, v in u.items() if k != "track"} | {"samples": len(u["track"])} for u in unit_rows],
    "rockets": [{k: v for k, v in r.items() if k != "track"} | {"samples": len(r["track"]), "euler_first": r["track"][:3]} for r in rocket_rows]}, indent=2), flush=True)
