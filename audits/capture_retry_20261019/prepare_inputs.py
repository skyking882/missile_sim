"""Bounded extraction from previously completed replay artifacts, no new decoding."""
import gzip,hashlib,json,shutil
from pathlib import Path

def read(path):
    with (gzip.open(path,'rt') if path.suffix=='.gz' else path.open()) as f:return json.load(f)

def main():
    root=Path('previous');out=Path('bounded');out.mkdir(exist_ok=True)
    manifest=read(root/'MANIFEST.json')
    for r in manifest:
        p=root/r['path'];assert p.stat().st_size==r['bytes'] and hashlib.sha256(p.read_bytes()).hexdigest()==r['sha256'],str(p)
    print('VERIFIED_PRIOR_FILES',len(manifest))
    selected=[]
    def retain(p,name):
        dest=out/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest)
        selected.append(dict(path=name,source=str(p),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
    for name in ('anchor_full_histories.json.gz','capture_events.json.gz'):
        p=sorted(root.rglob(name))[0];retain(p,name);d=read(p)
        if name.startswith('anchor'):
            for uid,rs in d.items():print('ANCHOR_RANGE',uid,len(rs),rs[0]['tick'],rs[-1]['tick'],rs[0]['packet_ms'])
    dataset=next(Path('reference').rglob('server_dataset.json.gz'));d=read(dataset)
    infos=d['infos'];(out/'infos.json').write_text(json.dumps(infos,separators=(',',':')))
    selected.append(dict(path='infos.json',source=str(dataset)+'#/infos',bytes=(out/'infos.json').stat().st_size,sha256=hashlib.sha256((out/'infos.json').read_bytes()).hexdigest(),source_sha256=hashlib.sha256(dataset.read_bytes()).hexdigest()))
    for info in infos:
        if info.get('uid') in (118,148):print('CASE_INFO',json.dumps(info))
    for c in d.get('cases',[]):
        if c.get('info',{}).get('uid') in (118,148):print('CASE_SCHEMA',json.dumps({k:(v if k=='info' else (v[:1] if isinstance(v,list) else v)) for k,v in c.items()})[:16000])
    for session in ('2026.10.08_11.49.38','2026.10.08_12.11.04'):
        p=next(p for p in Path('derived').rglob('unit_history.json.gz') if p.parent.name==session);retain(p,f'{session}/{p.name}')
        d=read(p);assert isinstance(d,dict) and 'units' in d
        units=d['units'];print('UNITS',session,len(units),'SAMPLE',json.dumps(units[0])[:1200])
        print('UNIT_IDENTITIES',session,json.dumps([{k:v for k,v in r.items() if k!='track'} for r in units]))
    (out/'SOURCE_MANIFEST.json').write_text(json.dumps(selected,indent=2))
    print('BOUNDED_FILES',json.dumps(selected))
if __name__=='__main__':main()
