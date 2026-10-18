#!/usr/bin/env python3
"""Read frozen game replay entities using the previously built native parser."""
import argparse,atexit,collections,gzip,hashlib,json,sys
from pathlib import Path
ap=argparse.ArgumentParser();ap.add_argument('session');ap.add_argument('--module',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);args=ap.parse_args()
root=Path.cwd();sys.path.insert(0,str(args.module.resolve()));sys.dont_write_bytecode=True
import PyReplayParser as p
out=args.out;out.mkdir(parents=True,exist_ok=False);units={};rockets={}
def on_unit(event,eid,unit__ref):
 u=unit__ref.unit
 if u:units[eid.get_handle()]=(eid.get_handle(),u)
def on_rocket(event,eid,rocket_component):rockets[eid.get_handle()]=(eid.get_handle(),rocket_component)
p.ecs.registerEsSystem(on_unit,'association_unit_export',[],[],[p.ecs.EventEntityCreatedType]);p.ecs.registerEsSystem(on_rocket,'association_rocket_export',[],[],[p.ecs.EventEntityCreatedType])
p.initialize(str(root/'runtime/game_resources'),logfile_path=str(out/'parser.log'),fonts=False,lang=False,mis=False);atexit.register(p.ecs.clearPythonCallbacks)
p.enableWeaponFieldProbe();p.enableFmProbe();p.enableUnitMpiProbe()
r=p.replay.ServerReplay(str(root/'raw'/args.session));assert r.isValid
state=p.ParserState(r);state.LoadFromReader(r.getReplayReader())
def vec(x):return [float(x.x),float(x.y),float(x.z)]
def track(obj):
 h=obj.positions.history
 return [dict(t_ms=int(h[i].time_ms),xyz=vec(h[i].value.location),euler_raw=vec(h[i].value.euler)) for i in range(len(h))]
def weapons(u):
 out=[]
 try:
  for w in u.actual_weapons:out.append(dict(weapon_id=w.weapon_id,weapon_index=w.weapon_index,weapon_name=w.weapon_name,blk_path=w.blk_path))
 except Exception as e:return {'unavailable':str(e)}
 return out
us=[dict(eid=eid,uid=u.uid,owner_pid=u.owner_pid,unit_name=u.unit_name,created_at_ms=u.created_at_ms,killed_at_ms=u.killed_at_ms,destroyed_at_ms=u.destroyed_at_ms,killed_position=vec(u.killed_position),track=track(u),weapons=weapons(u)) for eid,u in units.values()]
rs=[dict(eid=eid,owner_eid=x.ownerEid.get_handle(),eid2=x.eid2.get_handle(),created_at_ms=x.created_at_ms,destroyed_at_ms=x.destroyed_at_ms,weapon_name=x.weapon_obj.weapon_name if x.weapon_obj else None,launcher_blk=x.weapon_obj.blk_path if x.weapon_obj else None,raw=dict(x.creation_raw_fields),track=track(x)) for eid,x in rockets.values()]
with gzip.open(args.module/'extracted'/args.session/'unit_history.json.gz','rt') as f:prior=json.load(f)
newby={u['eid']:u for u in us};newrock={r['eid']:r for r in rs}
for u in prior['units']:
 assert u['eid'] in newby,('missing unit',u['eid'])
 assert u['track']==newby[u['eid']]['track'],('unit history differs',u['eid'])
for r in prior['rockets']:
 assert r['eid'] in newrock and r['owner_eid']==newrock[r['eid']]['owner_eid'] and r['weapon_name']==newrock[r['eid']]['weapon_name'],('rocket metadata mismatch',r['eid'])
for name,data in [('units',us),('projectiles',rs)]:
 with gzip.open(out/(name+'.json.gz'),'wt') as f:json.dump(data,f,separators=(',',':'),allow_nan=False)
receipt=dict(session=args.session,native_module_sha256=hashlib.sha256(Path(p.__file__).read_bytes()).hexdigest(),python=sys.version,units=len(us),projectiles=len(rs),prior_units_checked=len(prior['units']),prior_rocket_identities_checked=len(prior['rockets']),all_prior_unit_histories_identical=True,projectile_history_count=sum(bool(x['track']) for x in rs),countermeasure_count=sum('countermeasure' in (x['weapon_name'] or '') for x in rs),countermeasure_history_count=sum(bool(x['track']) and 'countermeasure' in (x['weapon_name'] or '') for x in rs),countermeasure_shell_types=dict(collections.Counter(str(x['raw']['shell_type']) for x in rs if 'countermeasure' in (x['weapon_name'] or ''))),state_packet_count=int(state.current_packet_index))
(out/'receipt.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt),flush=True)
