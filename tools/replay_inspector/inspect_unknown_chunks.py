"""Offline bit inspection; candidate interpretations never replace original data."""
import argparse
import csv
import bisect
import hashlib
import json
import math
import statistics
import struct
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / 'outputs' / 'extended_202944_final'
OUT = ROOT / 'outputs' / 'unknown_202944'

def bits(data):
    return ''.join(f'{b:08b}' for b in data)

def aligned_bytes(bit_string):
    # Last partial byte is zero padded on the right for storage, not a value.
    return int(bit_string.ljust((len(bit_string)+7)//8*8, '0'), 2).to_bytes((len(bit_string)+7)//8, 'big')

def read(bit_string, offset, fmt):
    size = struct.calcsize(fmt) * 8
    return struct.unpack(fmt, aligned_bytes(bit_string[offset:offset+size]))

def load():
    raw = [json.loads(x) for x in (SOURCE / 'weapon_sync_raw.jsonl').read_text().splitlines()]
    chunks = []
    for row in raw:
        f = row['fields']
        full = bits(bytes.fromhex(row['byte_slice_hex']))
        start = int(f['seeker_start_bit']) - (int(f['start_bit'])//8)*8
        size = int(f['skipped_bits'])
        assert read(full, start-16, '<H')[0] == size
        assert start+size == row['first_valid_bit_offset_in_slice'] + row['valid_bit_count']
        chunks.append(full[start:start+size])
    with (SOURCE / 'missile_all_fields.dat').open(encoding='utf-8') as stream:
        table = [{k: float(v) if v != 'NA' else None for k,v in r.items()} for r in csv.DictReader(stream, delimiter='\t')]
    assert len(raw) == len(table)
    assert all(r['fields']['time_ms'] == t['replay_time_ms'] for r,t in zip(raw,table))
    return raw, chunks, table

def run():
    raw, chunks, table = load()
    OUT.mkdir(parents=True, exist_ok=True)
    creation = json.loads((SOURCE/'missile_creation_fields.json').read_text())['maybe_sensor_info']
    creation_bits = bits(bytes.fromhex(creation['hex']))[:creation['bit_count']]
    result = {'source': str(SOURCE), 'sample_count':len(chunks), 'chunk_lengths': sorted(set(map(len,chunks))), 'candidate_scan':{}, 'scalar_hits':[]}
    refs = {
        'target_position': [[r[f'target_{axis}_m_interpolated'] for axis in 'xyz'] for r in table],
        'missile_position': [[r[f'{axis}_m'] for axis in 'xyz'] for r in table],
        'missile_velocity': [[r[f'velocity_{axis}_mps_candidate'] for axis in 'xyz'] for r in table],
        'relative_target_position': [[r[f'target_{axis}_m_interpolated']-r[f'{axis}_m'] for axis in 'xyz'] for r in table],
    }
    candidates = []
    for offset in range(len(chunks[0])-96+1):
        values = [read(c,offset,'<fff') for c in chunks]
        if not all(all(math.isfinite(v) and abs(v)<1e8 for v in row) for row in values):
            continue
        for name, reference in refs.items():
            errors = [math.dist(a,b) for a,b in zip(values,reference)]
            score = statistics.median(errors)
            candidates.append((score,name,offset,max(errors),values[0],values[-1]))
    for name in refs:
        result['candidate_scan'][name] = sorted([x for x in candidates if x[1]==name])[:5]
    # Identify plausible float components independently, including non-byte alignment.
    for offset in range(len(chunks[0])-32+1):
        values = [read(c,offset,'<f')[0] for c in chunks]
        if not all(math.isfinite(v) and abs(v)<1e7 for v in values):
            continue
        if max(values)-min(values)<1e-6:
            continue
        for axis in 'xy':
            key = f'target_{axis}_m_interpolated'
            errors=[abs(v-r[key]) for v,r in zip(values,table)]
            if statistics.median(errors)<100:
                result['scalar_hits'].append({'bit_offset':offset,'reference':key,'median_error':statistics.median(errors),'max_error':max(errors),'first':values[0],'last':values[-1]})
    result['constant_bit_count'] = sum(len(set(c[i] for c in chunks))==1 for i in range(len(chunks[0])))
    result['first_chunk_hex'] = aligned_bytes(chunks[0]).hex()
    result['last_chunk_hex'] = aligned_bytes(chunks[-1]).hex()
    result['creation_first_floats_scan'] = []
    for offset in range(len(creation_bits)-96+1):
        val=read(creation_bits,offset,'<fff')
        if all(math.isfinite(x) for x in val) and math.dist(val,refs['target_position'][0])<200:
            result['creation_first_floats_scan'].append({'bit_offset':offset,'value':val,'distance_to_interpolated_target': math.dist(val,refs['target_position'][0])})
    with (OUT/'chunks.jsonl').open('w', encoding='utf-8') as stream:
        for r,c in zip(raw,chunks):
            stream.write(json.dumps({'time_ms':r['fields']['time_ms'], 'bit_count':len(c), 'hex_right_zero_padded':aligned_bytes(c).hex()})+'\n')
    result.update(export_candidates(chunks, table, creation_bits))
    (OUT/'scan.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:result[k] for k in ['sample_count','chunk_lengths','creation_shared_prefix_bits','candidate_error_segments','state_changes','last_six_samples']},ensure_ascii=False,indent=2))

def export_candidates(chunks, table, creation_bits):
    with (SOURCE/'unit_00006c02_TargetAI_0.dat').open(encoding='utf-8') as stream:
        target = [{k:float(v) for k,v in r.items()} for r in csv.DictReader(stream, delimiter='\t')]
    target_times=[r['time_ms'] for r in target]
    rows=[]
    changes=[]
    last_state=None
    for index,(chunk,base) in enumerate(zip(chunks,table)):
        assert bits(aligned_bytes(chunk))[:len(chunk)] == chunk
        p=read(chunk,11,'<fff')
        v=read(chunk,107,'<fff')
        unknown=read(chunk,203,'<fff')
        assert all(math.isfinite(x) for x in (*p,*v,*unknown))
        right=bisect.bisect_right(target_times,base['replay_time_ms'])
        assert 0 < right < len(target)
        a,b=target[right-1:right+1]
        tv=[(b[f'{axis}_m']-a[f'{axis}_m'])/((b['time_ms']-a['time_ms'])/1000) for axis in 'xyz']
        actual=[base[f'target_{axis}_m_interpolated'] for axis in 'xyz']
        missile=[base[f'{axis}_m'] for axis in 'xyz']
        row={'sample_index':index,'replay_time_ms':base['replay_time_ms'],'flight_time_sf1_s_candidate':base['flight_time_sf1_s_candidate'], 'prefix_11_bits_raw':chunk[:11]}
        row.update({f'track_position_{axis}_m_candidate':value for axis,value in zip('xyz',p)})
        row.update({f'track_velocity_{axis}_mps_candidate':value for axis,value in zip('xyz',v)})
        row.update({f'unknown_float_at_bit_{203+i*32}':value for i,value in enumerate(unknown)})
        row.update({f'target_{axis}_m_interpolated':value for axis,value in zip('xyz',actual)})
        row.update({f'target_velocity_{axis}_mps_from_position':value for axis,value in zip('xyz',tv)})
        row.update({'track_position_error_to_target_m_candidate':math.dist(p,actual), 'track_velocity_error_to_target_mps_candidate':math.dist(v,tv),'track_range_from_missile_m_candidate':math.dist(p,missile),'target_slant_distance_m':base['target_slant_distance_m'], 'remaining_bits_from_299_count':len(chunk)-299, 'remaining_bits_from_299_hex_right_zero_padded':aligned_bytes(chunk[299:]).hex()})
        rows.append(row)
        state=(chunk[:11],unknown==(0.,0.,0.))
        if state!=last_state:
            changes.append({'sample_index':index,'replay_time_ms':base['replay_time_ms'],'prefix_11_bits':state[0],'unknown_vector_all_zero':state[1]})
            last_state=state
    with (OUT/'seeker_candidates.dat').open('w',encoding='utf-8',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]),delimiter='\t')
        writer.writeheader();writer.writerows(rows)
    common=0
    for a,b in zip(creation_bits[2:],chunks[0]):
        if a!=b: break
        common+=1
    errors={}
    for name,lo,hi in [('before_state_change',0,18),('after_state_change_before_terminal_departure',18,159),('terminal_departure',159,165),('all',0,165)]:
        segment=rows[lo:hi]
        errors[name]={'sample_count':len(segment)}
        for key in ['track_position_error_to_target_m_candidate','track_velocity_error_to_target_mps_candidate']:
            values=[r[key] for r in segment]
            errors[name][key]={'median':statistics.median(values),'max':max(values)}
    qa={'input_files_sha256':{name:hashlib.sha256((SOURCE/name).read_bytes()).hexdigest() for name in ['weapon_sync_raw.jsonl','missile_all_fields.dat','missile_creation_fields.json','unit_00006c02_TargetAI_0.dat']},'row_count':len(rows),'column_count':len(rows[0]), 'length_prefix_verified_all':True, 'chunk_roundtrip_verified_all':True,'nine_float_candidates_finite_all':True,'creation_shared_prefix_bits_after_shift_2':common,'interpretation_status':'Candidate meanings only; fixed float boundaries supported by one replay. No in-game seeker validation.'}
    (OUT/'qa.json').write_text(json.dumps(qa,ensure_ascii=False,indent=2),encoding='utf-8')
    keys=['sample_index','replay_time_ms','track_range_from_missile_m_candidate','target_slant_distance_m','track_position_error_to_target_m_candidate']
    return {'creation_shared_prefix_bits':common,'candidate_error_segments':errors,'state_changes':changes,'last_six_samples':[{k:r[k] for k in keys} for r in rows[-6:]]}

if __name__=='__main__':
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--source', type=Path, default=SOURCE, help='Extended export directory for the validated 2026-09-27 single-missile replay')
    cli.add_argument('--output', type=Path, default=OUT)
    args=cli.parse_args()
    SOURCE=args.source.resolve()
    OUT=args.output.resolve()
    run()
