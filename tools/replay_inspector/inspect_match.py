"""Match-scale export of one WRPL replay: what inspect_extended.py keeps for units and missiles, plus the fields the
backend decodes but the earlier exports dropped. Writes OUTPUT/match_export.json.

Needs the backend built with this directory's patches, including retain_fm_sync.py, retain_unit_mpi.py and
expose_kill_fields.py (missing probes are skipped and listed under "probes_missing").

Contents (``status`` per block: decoded = upstream format decoding of a named field; candidate = read from the
stream, meaning or layout not confirmed):
  units            per spawned unit: identity, army (BaseExtReflectable.unitArmyNo, decoded), weapons, the decoded
                   position/Euler track, created/killed/destroyed times (as inspect_extended.py)
  players          MPlayer: name, account id, team (decoded), country, the units each one owned
  battle_messages  kill feed (killer/victim unit, weapon name, death and weapon type), severe/critical damage
                   (decoded; DeathType and WeaponType values follow the upstream enums), awards
  rockets          projectiles with a position track (missiles), as inspect_extended.py, with the ECS template
                   (e.g. "rocket+arh_shell" for active radar homing), plus a compact missile
                   sync series: b7 velocity (candidate scale) and the 607-bit seeker block read with the layout found
                   on the 2026-09-27 sample (candidate track position/velocity)
  countermeasures  projectiles fired by "countermeasure*" launchers: owner, launcher, time, start position,
                   lifetime, shell id (Rocket.uleb_1), spin (u12_3); times and owners decoded, flare/chaff split is a
                   candidate (see the importer)
  fm_probe         per aircraft: sync state rows (weapon counts, sensors, countermeasure controls, target
                   designations, seeker block length) and the decoded-but-dropped velocity; all candidate meanings
  unit_mpi         per-unit MPI messages the parser skips (damage model, effects, ...): time, type, message id,
                   size, first bytes; payloads undecoded
"""
import argparse
import atexit
import hashlib
import json
import math
import struct
from collections import Counter
from pathlib import Path

import PyReplayParser as parser

UNKNOWN = 0xFFFFFFFF


def vec(v):
    return [float(v.x), float(v.y), float(v.z)]


def clean(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {k: clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    return value


def track(obj):
    h = obj.positions.history
    return [{"t_ms": int(h[i].time_ms), "xyz": vec(h[i].value.location), "euler_raw": vec(h[i].value.euler)}
            for i in range(len(h))]


def rv(var):
    """Current value of a ReflectionVar, or None."""
    try:
        return var.data
    except Exception:  # never set
        return None


def bits_of(data):
    return "".join(f"{b:08b}" for b in data)


def read_bits(bit_string, offset, fmt):
    size = struct.calcsize(fmt)*8
    chunk = bit_string[offset:offset+size]
    if len(chunk) < size:
        return None
    return struct.unpack(fmt, int(chunk, 2).to_bytes(size//8, "big"))


cli = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
cli.add_argument("--game", required=True, help="folder with aces.vromfs.bin, game.vromfs.bin, char.vromfs.bin")
cli.add_argument("--replay", required=True)
cli.add_argument("--output", required=True, help="new output directory")
cli.add_argument("--local-package", help="local content package (default GAME/content/pkg_local when present)")
args = cli.parse_args()
local_package = Path(args.local_package) if args.local_package else Path(args.game)/"content/pkg_local"
out = Path(args.output)
out.mkdir(parents=True, exist_ok=False)
atexit.register(parser.ecs.clearPythonCallbacks)
units_by_ptr, rockets_by_ptr = {}, {}


def on_unit(event, eid, unit__ref):
    if unit__ref.unit:
        units_by_ptr[id(unit__ref.unit)] = (eid.get_handle(), unit__ref.unit)


templates = {}


def on_rocket(event, eid, rocket_component):
    rockets_by_ptr[id(rocket_component)] = (eid.get_handle(), rocket_component)
    try:
        templates[eid.get_handle()] = state.mgr.getEntity(eid).templateName
    except Exception:  # entity not reachable at creation time
        pass


parser.ecs.registerEsSystem(on_unit, "match_unit_ids", [], [], [parser.ecs.EventEntityCreatedType])
parser.ecs.registerEsSystem(on_rocket, "match_rocket_ids", [], [], [parser.ecs.EventEntityCreatedType])
parser.initialize(args.game, logfile_path=str(out/"parser.log"), fonts=False, lang=False, mis=False)
if local_package.is_dir():
    parser.mountLocalPackage(str(local_package))
missing = []
for name in ("enableWeaponFieldProbe", "enableFmProbe", "enableUnitMpiProbe"):
    if hasattr(parser, name):
        getattr(parser, name)()
    else:
        missing.append(name)
replay = parser.replay.Replay(args.replay)
if not replay.isValid:
    raise RuntimeError("replay rejected by the parser")
state = parser.ParserState(replay)
state.LoadFromReader(replay.getReplayReader())
if state.replay_length_ms == UNKNOWN:
    raise RuntimeError("No replay end marker")
hdr = replay.header

# -- units ---------------------------------------------------------------------------------------------------------
eid_of_unit = {}
units = []
for eid, unit in units_by_ptr.values():
    eid_of_unit[id(unit)] = eid
    army = None
    if unit.base_data is not None:
        army = rv(unit.base_data.unitArmyNo)
    weapons = [{"weapon_id": w.weapon_id, "weapon_index": w.weapon_index, "weapon_name": w.weapon_name,
                "blk_path": w.blk_path} for w in unit.actual_weapons]
    units.append({"eid": eid, "uid": unit.uid, "unit_name": unit.unit_name,
                  "player_internal_name": unit.player_internal_name, "owner_pid": unit.owner_pid,
                  "unit_type": int(unit.unitType), "army": army, "loadout": unit.loadout_name,
                  "spawn_matrix_repr": str(unit.spawn_position), "created_at_ms": unit.created_at_ms,
                  "killed_at_ms": unit.killed_at_ms, "destroyed_at_ms": unit.destroyed_at_ms,
                  "weapons": weapons,
                  "storage_weapons": [{"launcher": s.launcher, "bullet": s.bullet, "count": s.count}
                                      for s in unit.weapons],
                  "track": track(unit)})


def unit_eid(u):
    return None if u is None else eid_of_unit.get(id(u))


# -- players -------------------------------------------------------------------------------------------------------
players = []
for index, pl in enumerate(state.players):
    uid = rv(pl.uid)
    owned = [unit_eid(u) for u in pl.allOwnedUnits]
    players.append({"index": index, "name": uid.player_name if uid is not None else None,
                    "account_id": uid.player_id if uid is not None else None, "team": rv(pl.team),
                    "country": str(rv(pl.countryId)), "real_nick": rv(pl.realNick),
                    "owned_unit_eids": [e for e in owned if e is not None], "owned_units": len(owned)})

# -- battle messages -----------------------------------------------------------------------------------------------
messages = []
for m in state.battle_messages:
    kind = type(m).__name__
    row = {"kind": kind, "time_ms": m.time_ms}
    if kind == "KillMessage":
        row.update(offender_eid=unit_eid(m.offender_unit), offended_eid=unit_eid(m.offended_unit),
                   offender_vehicle=m.offender_vehicle, used_weapon=m.used_weapon,
                   destroyed_weapon=m.destroyed_weapon, offender_pid=m.offender_pid, victim_pid=m.VictimPid,
                   unit_type=int(m.unitType), burav=bool(m.is_burav_kill),
                   offended_position=vec(m.offended_unit_position) if m.offended_unit else None)
        for k in ("death_type", "weapon_type", "weapon_flags", "extra_str_1", "extra_str_2", "extra_val_3",
                  "extra_str_4"):
            if hasattr(m, k):
                row[k] = getattr(m, k)
    elif kind in ("SevereDamageMessage", "CriticalDamageMessage"):
        row.update(offender_eid=unit_eid(m.offender_unit), offended_eid=unit_eid(m.offended_unit),
                   player_pid=m.player_pid, vehicle=m.vehicle, unit_type=int(m.unitType))
        if kind == "CriticalDamageMessage":
            row["is_fire"] = int(m.is_fire)
    elif kind == "AwardMessage":
        row.update(player_pid=m.player_pid, award=m.award, stage=m.stage)
    messages.append(row)

# -- missile sync series (b7 velocity, seeker block candidate) ------------------------------------------------------
probe = parser.getWeaponFieldProbe() if hasattr(parser, "getWeaponFieldProbe") else []
probe_hex = parser.getWeaponFieldProbeHex() if hasattr(parser, "getWeaponFieldProbeHex") else []
sync = {}
for p, payload in zip(probe, probe_hex if len(probe_hex) == len(probe) else [None]*len(probe)):
    eid = int(p["eid"])
    s16 = lambda v: (int(v)-65536 if int(v) >= 32768 else int(v))*4000/65535  # noqa: E731
    row = [int(p["time_ms"])]
    if p.get("bool2", 1) == 0 and "b7_x_raw_u16" in p:
        row += [s16(p["b7_x_raw_u16"]), s16(p["b7_y_raw_u16"]), s16(p["b7_z_raw_u16"])]
    else:
        row += [None, None, None]
    seek = [None]*8
    nbits = int(p.get("skipped_bits", 0) or 0)
    if payload and nbits >= 299 and "seeker_start_bit" in p:
        full = bits_of(bytes.fromhex(payload))
        start = int(p["seeker_start_bit"])-(int(p["start_bit"])//8)*8
        chunk = full[start:start+nbits]
        pos, vel = read_bits(chunk, 11, "<fff"), read_bits(chunk, 107, "<fff")
        if pos and vel:
            seek = [int(chunk[:11], 2), nbits, *pos, *vel]
    sync.setdefault(eid, []).append(row+seek)

# -- rockets and countermeasures -----------------------------------------------------------------------------------
rockets, countermeasures, untracked = [], [], []
for eid, rc in rockets_by_ptr.values():
    w = rc.weapon_obj
    name = w.weapon_name if w else None
    f = rc.creation_raw_fields if hasattr(rc, "creation_raw_fields") else {}
    owner = rc.ownerEid.get_handle()
    if name and name.startswith("countermeasure"):
        countermeasures.append({"eid": eid, "owner_eid": owner, "launcher": name, "created_at_ms": rc.created_at_ms,
                                "destroyed_at_ms": rc.destroyed_at_ms, "shell_id": f.get("uleb_1"),
                                "weapon_ref": f.get("weapon_ref"), "u1_4": f.get("u1_4"),
                                "start_pos": f.get("starting_pos"), "start_vel": f.get("u12_2"),
                                "spin": f.get("u12_3"), "creation_time": f.get("creation_time")})
        continue
    trk = track(rc)
    row = {"eid": eid, "owner_eid": owner, "eid2": rc.eid2.get_handle(), "weapon_name": name,
           "template": templates.get(eid),
           "weapon_path": w.blk_path if w else None, "created_at_ms": rc.created_at_ms,
           "destroyed_at_ms": rc.destroyed_at_ms, "track": trk}
    if f:
        row["creation_raw_fields"] = f
    if trk:
        row["sync"] = sync.get(eid, [])  # [t_ms, b7 vx, vy, vz, seeker prefix, seeker bits, px, py, pz, vx, vy, vz]
        rockets.append(row)
    else:
        untracked.append({k: v for k, v in row.items() if k != "creation_raw_fields"})

# -- aircraft sync probe --------------------------------------------------------------------------------------------
fm = {}
if hasattr(parser, "getFmProbe"):
    for t, uid, eid, disc, cont in parser.getFmProbe():
        fm.setdefault(eid, {"uid": uid, "rows": [], "velocity": []})["rows"].append([t, disc, cont])
    for t, uid, eid, vx, vy, vz in parser.getFmVelocity():
        fm.setdefault(eid, {"uid": uid, "rows": [], "velocity": []})["velocity"].append(
            [t, round(vx, 2), round(vy, 2), round(vz, 2)])

# -- unhandled per-unit MPI messages ----------------------------------------------------------------------------------
mpi_rows = parser.getUnitMpiProbe() if hasattr(parser, "getUnitMpiProbe") else []
mpi_counts = Counter((r[1], r[4]) for r in mpi_rows)

raw = {"source": args.replay, "source_sha256": hashlib.sha256(Path(args.replay).read_bytes()).hexdigest(),
       "backend_revision": "221be5cac8e772ab41eefb0e7dac35ce6fb5c39b+wt-overlay-exports",
       "capture_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
       "export": "match_export v1", "probes_missing": missing,
       "replay_length_ms": state.replay_length_ms,
       "header": {"mission": hdr.mission_name, "mission_file": hdr.mission_file, "level": hdr.level_path,
                  "player_no": hdr.playerNo, "player_count": hdr.player_count, "game_version": hdr.gameVersion,
                  "environment": hdr.environment, "weather": hdr.weather},
       "units": units, "players": players, "battle_messages": messages, "rockets": rockets,
       "countermeasures": countermeasures, "untracked_rockets": untracked,
       "fm_probe": {str(k): v for k, v in fm.items()},
       "unit_mpi": {"rows": [list(r) for r in mpi_rows],
                    "counts": [[t, mid, n] for (t, mid), n in mpi_counts.most_common()],
                    "columns": ["time_ms", "obj_type", "obj_index", "ext_uid", "message_id", "bytes", "hex24"]},
       "status": {"units": "decoded (army: BaseExtReflectable.unitArmyNo)", "players": "decoded",
                  "battle_messages": "decoded; death_type/weapon_type follow upstream enums",
                  "rockets.sync.b7": "candidate velocity scale", "rockets.sync.seeker": "candidate layout",
                  "countermeasures": "times/owners decoded; flare vs chaff candidate",
                  "fm_probe": "fields decoded by upstream structs, meanings candidate",
                  "unit_mpi": "message ids/sizes only"}}
(out/"match_export.json").write_text(json.dumps(clean(raw), ensure_ascii=False, allow_nan=False), encoding="utf-8")
print(json.dumps({"units": len(units), "players": len(players), "messages": len(messages), "rockets": len(rockets),
                  "countermeasures": len(countermeasures), "untracked_rockets": len(untracked),
                  "fm_aircraft": len(fm), "fm_rows": sum(len(v["rows"]) for v in fm.values()),
                  "unit_mpi_rows": len(mpi_rows), "probes_missing": missing}), flush=True)
