#!/usr/bin/env python3
"""Verify bundled files and fixed WRPL headers using only the standard library."""
import argparse, hashlib, json, struct
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()
def main():
    cli=argparse.ArgumentParser(description=__doc__);cli.add_argument('--raw-only',action='store_true');args=cli.parse_args()
    rows=json.loads((ROOT/'metadata/server_download.json').read_text());count=total=0;headers=[]
    expected=set()
    for row in rows:
        indices=[x['index'] for x in row['segments']]
        assert indices == [0]+list(range(1,max(indices)+1,2)), indices
        for seg in row['segments']:
            path=ROOT/seg['path'];expected.add(path.resolve())
            assert path.stat().st_size==seg['bytes'],path
            assert digest(path)==seg['sha256'],path
            with path.open('rb') as f:b=f.read(1234)
            assert len(b)==1234 and b[:8]==bytes.fromhex('e5ac00101c8c0100'),path
            session=struct.unpack_from('<Q',b,732)[0];part=b[740];length=struct.unpack_from('<H',b,742)[0]
            player=struct.unpack_from('<i',b,932)[0];version=struct.unpack_from('<Q',b,1226)[0]
            assert session==int(row['session_id']) and f'{session:016x}'==row['session_hex'],path
            assert part==seg['index'] and player==-2147483648,path
            assert version==563203356491830,path
            headers.append(dict(path=seg['path'],session_id=str(session),part=part,playerNo=player,segmentLengthSec=length,game_version='2.59.0.54'))
            count+=1;total+=len(path.read_bytes())
    assert expected=={p.resolve() for p in (ROOT/'raw').rglob('*.wrpl')}
    checked=0
    if not args.raw_only:
        for line in (ROOT/'SHA256SUMS').read_text().splitlines():
            sha,name=line.split('  ',1);assert digest(ROOT/name)==sha,name;checked+=1
    print(json.dumps(dict(raw_files=count,raw_bytes=total,headers_passed=len(headers),sha256_entries_checked=checked),indent=2))
if __name__=='__main__':main()
