#!/usr/bin/env python3
"""Export decompressed packet bytes, retained weapon bit slices, and native unit histories.
The downloaded original streams remain under raw/. Exported JSON is a derived artifact.
"""
import argparse, atexit, collections, gzip, hashlib, json, platform, sys
from pathlib import Path
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1]
cli=argparse.ArgumentParser(description=__doc__)
cli.add_argument('replay',help='one directory under raw/, or an explicit server replay directory')
cli.add_argument('--output',required=True,help='new output directory')
cli.add_argument('--module-dir',type=Path,default=ROOT/'tools/backend/macos-x86_64-cpython39')
cli.add_argument('--game-resources',type=Path,default=ROOT/'runtime/game_resources')
args=cli.parse_args()
sys.path.insert(0,str(args.module_dir.resolve()))
import PyReplayParser as p
rep=Path(args.replay)
if not rep.is_dir():rep=ROOT/'raw'/args.replay
rep=rep.resolve();out=Path(args.output).resolve();out.mkdir(parents=True,exist_ok=False)
units={};rockets={}
def on_unit(event,eid,unit__ref):
    if unit__ref.unit:units[id(unit__ref.unit)]=(eid.get_handle(),unit__ref.unit)
def on_rocket(event,eid,rocket_component):rockets[id(rocket_component)]=(eid.get_handle(),rocket_component)
p.ecs.registerEsSystem(on_unit,'network_server_units',[],[],[p.ecs.EventEntityCreatedType])
p.ecs.registerEsSystem(on_rocket,'network_server_rockets',[],[],[p.ecs.EventEntityCreatedType])
p.initialize(str(args.game_resources.resolve()),logfile_path=str(out/'parser.log'),fonts=False,lang=False,mis=False)
atexit.register(p.ecs.clearPythonCallbacks)
p.enableWeaponFieldProbe()
for name in ['enableFmProbe','enableUnitMpiProbe']:
    if hasattr(p,name):getattr(p,name)()
r=p.replay.ServerReplay(str(rep));assert r.isValid,'server replay rejected'
# First pass retains whole decompressed packet payloads. No semantic field decoder is applied here.
packet_types=collections.Counter();packets=0
with gzip.open(out/'packets.jsonl.gz','wt',encoding='utf-8') as f:
    for packet in r.getReplayReader():
        stream=packet.data;bits=int(stream.GetNumberOfBitsUsed());data=bytes(stream.GetData())
        row=dict(packet_index=packets,packet_time_ms=int(packet.time_ms),packet_type=int(packet.type),payload_bits=bits,payload_hex=data.hex())
        f.write(json.dumps(row,separators=(',',':'))+'\n');packet_types[int(packet.type)]+=1;packets+=1
s=p.ParserState(r);s.LoadFromReader(r.getReplayReader())
fields=[dict(z) for z in p.getWeaponFieldProbe()];hexes=list(p.getWeaponFieldProbeHex());assert len(fields)==len(hexes)
with gzip.open(out/'weapon_raw.jsonl.gz','wt',encoding='utf-8') as f:
    for z,h in zip(fields,hexes):f.write(json.dumps(dict(fields=z,byte_slice_hex=h),separators=(',',':'))+'\n')
def vec(v):return[float(v.x),float(v.y),float(v.z)]
def track(u):
    h=u.positions.history
    return[dict(t_ms=int(h[i].time_ms),xyz=vec(h[i].value.location),euler_raw=vec(h[i].value.euler)) for i in range(len(h))]
assert units,'No native unit histories captured; do not treat this export as complete'
unitrows=[dict(eid=eid,uid=u.uid,owner_pid=u.owner_pid,unit_name=u.unit_name,track=track(u)) for eid,u in units.values()]
rocketrows=[dict(eid=eid,owner_eid=z.ownerEid.get_handle(),weapon_name=z.weapon_obj.weapon_name if z.weapon_obj else None) for eid,z in rockets.values()]
with gzip.open(out/'unit_history.json.gz','wt',encoding='utf-8') as f:json.dump(dict(units=unitrows,rockets=rocketrows),f,separators=(',',':'))
for name in ['getFmProbe','getFmVelocity','getUnitMpiProbe']:
    if hasattr(p,name):
        with gzip.open(out/(name+'.json.gz'),'wt',encoding='utf-8') as f:json.dump(getattr(p,name)(),f,separators=(',',':'))
report=dict(replay=rep.name,module_name=Path(p.__file__).name,module_sha256=hashlib.sha256(Path(p.__file__).read_bytes()).hexdigest(),python=sys.version,architecture=platform.machine(),packet_count=packets,native_packet_index=int(s.current_packet_index),packet_types=dict(packet_types),weapon_states=len(fields),units=len(unitrows),rockets=len(rocketrows),layouts=dict(collections.Counter(str(z.get('skipped_bits')) for z in fields)),original_streams_modified=False)
(out/'summary.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
