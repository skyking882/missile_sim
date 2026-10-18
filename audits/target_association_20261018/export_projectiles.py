#!/usr/bin/env python3
"""Read-only projectile evidence; aircraft tracks remain the original verified export.
The previous unit callback did not cover all initial aircraft. This script does not
claim to repair it: it uses original aircraft histories, and independently verifies
projectile identity and newly exported creation/lifecycle data.
"""
import argparse,atexit,gzip,hashlib,json,shutil,sys
from pathlib import Path
ap=argparse.ArgumentParser();ap.add_argument('session');ap.add_argument('--module',type=Path,required=True);ap.add_argument('--first-export',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);args=ap.parse_args()
root=Path.cwd();sys.path.insert(0,str(args.module.resolve()));sys.dont_write_bytecode=True
import PyReplayParser as p
out=args.out;out.mkdir(parents=True,exist_ok=False);rockets={}
def on_rocket(event,eid,rocket_component):rockets[eid.get_handle()]=(eid.get_handle(),rocket_component)
p.ecs.registerEsSystem(on_rocket,'association_projectiles_verified',[],[],[p.ecs.EventEntityCreatedType])
p.initialize(str(root/'runtime/game_resources'),logfile_path=str(out/'parser.log'),fonts=False,lang=False,mis=False);atexit.register(p.ecs.clearPythonCallbacks)
r=p.replay.ServerReplay(str(root/'raw'/args.session));assert r.isValid
state=p.ParserState(r);state.LoadFromReader(r.getReplayReader())
def vec(x):return [float(x.x),float(x.y),float(x.z)]
def track(obj):
 h=obj.positions.history
 return [dict(t_ms=int(h[i].time_ms),xyz=vec(h[i].value.location),euler_raw=vec(h[i].value.euler)) for i in range(len(h))]
rs=[dict(eid=eid,owner_eid=x.ownerEid.get_handle(),eid2=x.eid2.get_handle(),created_at_ms=x.created_at_ms,destroyed_at_ms=x.destroyed_at_ms,weapon_name=x.weapon_obj.weapon_name if x.weapon_obj else None,launcher_blk=x.weapon_obj.blk_path if x.weapon_obj else None,raw=dict(x.creation_raw_fields),track=track(x)) for eid,x in rockets.values()]
priorfile=args.module/'extracted'/args.session/'unit_history.json.gz'
with gzip.open(priorfile,'rt') as f:prior=json.load(f)
with gzip.open(args.first_export/args.session/'projectiles.json.gz','rt') as f:first=json.load(f)
nb={x['eid']:x for x in rs};fb={x['eid']:x for x in first};assert nb.keys()==fb.keys()
for r in prior['rockets']:
 assert r['eid'] in nb and all(r[k]==nb[r['eid']][k] for k in ('owner_eid','weapon_name')),r['eid']
for eid in nb:assert nb[eid]==fb[eid],('new projectile export differs',eid)
with gzip.open(out/'projectiles.json.gz','wt') as f:json.dump(rs,f,separators=(',',':'),allow_nan=False)
shutil.copy2(priorfile,out/'original_unit_history.json.gz')
receipt=dict(session=args.session,native_module_sha256=hashlib.sha256(Path(p.__file__).read_bytes()).hexdigest(),projectiles=len(rs),all_projectiles_identical_to_first_export=True,all_prior_projectile_metadata_identical=True,aircraft_source='unchanged original native export; no claim of complete new aircraft callback',original_units=len(prior['units']),original_nonempty_unit_histories=sum(bool(x['track']) for x in prior['units']),original_unit_archive_sha256=hashlib.sha256(priorfile.read_bytes()).hexdigest(),countermeasures=sum('countermeasure' in (x['weapon_name'] or '') for x in rs),countermeasures_with_history=sum(bool(x['track']) for x in rs if 'countermeasure' in (x['weapon_name'] or '')),state_packet_count=int(state.current_packet_index))
(out/'receipt.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt),flush=True)
