#!/usr/bin/env python3
"""Extract the frozen Air RB >=13.0 IR coverage and required game fields.

Reads existing exact-commit source/cache only; never downloads or moves Git.
The persisted extracts retain original file hashes and source-field paths.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath

VERSION = '2.59.0.27'
COMMIT = '4ac255b829a27346b9300fccdc138362c2535fbe'
PREFIX = 'aces.vromfs.bin_u/gamedata/'
WP = 'char.vromfs.bin_u/config/wpcost.blkx'
TAGS = 'char.vromfs.bin_u/config/unittags.blkx'
RANKS = 'gui.vromfs.bin_u/globals/ranks_common_shared.nut'
GLOBAL = 'aces.vromfs.bin_u/config/gameplay.blkx'


def source_refs(value, field=''):
    if isinstance(value, dict):
        for k, v in value.items():
            yield from source_refs(v, field+'/'+k)
    elif isinstance(value, list):
        for i, v in enumerate(value):
            yield from source_refs(v, field+'/'+str(i))
    elif isinstance(value, str) and 'weapons/' in value.lower() and value.lower().endswith('.blk'):
        yield field, str(PurePosixPath('aces.vromfs.bin_u')/value.lower()).removesuffix('.blk')+'.blkx'


def fm_path(value):
    path = str(PurePosixPath(PREFIX+'flightmodels')/value.lstrip('/'))
    return path.removesuffix('.blk')+'.blkx' if path.endswith('.blk') else path+'.blkx'


def build(root):
    cache = root/'outputs/ir_datamine_20261016'
    metadata = json.loads((cache/'source_version.json').read_text())
    if metadata['version'] != VERSION or metadata['commit'] != COMMIT:
        raise ValueError('Source cache must match the frozen commit')
    originals, extracts = {}, {}

    def raw(path):
        for directory in (root/('data/datamine_'+VERSION), cache):
            source = directory/path
            if source.is_file():
                contents = source.read_bytes()
                originals[path] = hashlib.sha256(contents).hexdigest()
                return contents
        raise FileNotFoundError(f'Stage exact {COMMIT}:{path} in {cache}')

    def read(path):
        return json.loads(raw(path))

    def retain(path, data, fields):
        extracts[path] = dict(kind='datamine',evidence_level='A',sha256=originals[path], extraction='selected original fields; values unchanged; semantic mappings not asserted',
                              fields=fields, data=data)

    wp, tags = read(WP), read(TAGS)
    ranks = raw(RANKS).decode()
    if 'economicRank / 3.0 + 1, 0.1' not in ranks or 'REALISTIC] = "Historical"' not in ranks:
        raise ValueError('Frozen client BR conversion/mode mapping changed')
    candidates = {n:v for n,v in wp.items() if isinstance(v,dict)
        and tags.get(n,{}).get('type') == 'aircraft' and v.get('economicRankHistorical',-1) >= 36}
    excluded = {n:dict(reason='unittags.tags.hideBrForVehicle=true; hidden-BR nuclear/killstreak unit',
                      economicRankHistorical=v['economicRankHistorical']) for n,v in candidates.items()
                if tags[n].get('tags',{}).get('hideBrForVehicle')}
    selected = {n:v for n,v in candidates.items() if n not in excluded}
    aircraft, weapons, flare_sources, fms, containers = {}, {}, {}, set(), set()

    def payloads(document):
        entries = document.get('bullet',document)
        for index, bullet in enumerate(entries if isinstance(entries,list) else [entries]):
            r = bullet.get('rocket',bullet.get('ballistic',bullet))
            ir = r.get('signature',{}).get('infraRed',{})
            if r.get('bulletType',bullet.get('bulletType')) in ('flare','flr') and ir.get('emission2',0)>0 and 'emissionProfile' in ir:
                yield index

    def inspect(path, origin, trail=()):
        if path in trail:
            raise ValueError('Cyclic weapon container reference: '+path)
        if '/containers/' in path:
            containers.add(path)
            d = read(path)
            for field, child in source_refs(d):
                if '/containers/' in child or '/rocketguns/' in child:
                    yield from inspect(child,origin+' -> '+path+field,trail+(path,))
            return
        if '/rocketguns/' not in path:
            return
        d = read(path)
        r = d.get('rocket',{})
        if r.get('bulletType') == 'aam' and 'opticalSeeker' in r.get('guidance',{}):
            name = Path(path).stem
            missile_id = r.get('bulletName') or name
            s = r['guidance']['opticalSeeker']
            required = ('rangeMax','fov','angleMax','lockAngleMax','rateMax','prolongationTimeMax')
            unsupported = []
            if r.get('guidanceType') != 'optical':
                unsupported.append('guidanceType='+str(r.get('guidanceType')))
            if s.get('bandMaskToReject',0) not in (0,4):
                unsupported.append('bandMaskToReject='+str(s['bandMaskToReject']))
            if s.get('bandMaskToReject') and 'signalRelRejectedTreshold' not in s:
                unsupported.append('bandMaskToReject present without signalRelRejectedTreshold; rejection mechanism unresolved')
            for k in required:
                if k not in s: unsupported.append('missing '+k)
            for k,v in s.items():
                if k.startswith('rangeBand') and not isinstance(v,(int,float)):
                    unsupported.append('non-scalar '+k)
            mechanism = ('mask_only_unresolved' if s.get('bandMaskToReject') and 'signalRelRejectedTreshold' not in s else
                         'gate+suspension' if s.get('gateWidth') and s.get('bandMaskToReject') else
                         'fov_gate' if s.get('gateWidth') else 'suspension' if s.get('bandMaskToReject') else 'baseline')
            weapons.setdefault(name,dict(missile_id=missile_id, source_path=path, mechanism=mechanism,
                optical_seeker=s, supported=not unsupported, unsupported=unsupported,
                runtime_profile=('missiles/'+missile_id+'.json') if (root/'data/profile_sets'/VERSION/'missiles'/(missile_id+'.json')).is_file() else None,
                runtime_limitation='tvc_excluded' if any(k.startswith('thrustVector') for k in r) else None,
                mounts=[]))['mounts'].append(origin)
            retain(path,dict(rocket={k:r[k] for k in ('bulletName','bulletType','guidanceType','thrustVectoringAngle') if k in r}|
                         {'guidance':{'opticalSeeker':s}}),['rocket.bulletName/bulletType/guidanceType/thrustVectoringAngle','rocket.guidance.opticalSeeker'])
            yield 'ir',name
        indices = list(payloads(d))
        if indices:
            name = Path(path).stem
            flare_sources[name] = dict(source_path=path,payload_indices=indices)
            retain(path,{k:v for k,v in d.items() if k=='bullet'},['bullet (complete actual payloads)'])
            for index in indices:
                yield 'flare',dict(weapon=name,bullet_index=index)

    for name,v in sorted(selected.items()):
        path = PREFIX+'flightmodels/'+name+'.blkx'
        unit = read(path)
        fp = fm_path(unit['fmFile'])
        fms.add(fp)
        retain(path,{'fmFile':unit['fmFile']},['fmFile'])
        ir, flares = set(), set()
        for field,p in source_refs(unit):
            if '/containers/' not in p and '/rocketguns/' not in p:continue
            for kind,value in inspect(p,path+field):
                if kind=='ir':ir.add(value)
                else:flares.add((value['weapon'],value['bullet_index']))
        aircraft[name] = dict(br_air_rb=round(v['economicRankHistorical']/3+1,1),
            economicRankHistorical=v['economicRankHistorical'],unit_source=path,fm_source=fp,
            infrared_weapons=sorted(ir),flare_payloads=[dict(weapon=n,bullet_index=i) for n,i in sorted(flares)])

    exceptions, sparse = {}, {}
    for path in sorted(fms):
        fm = read(path)
        engines = {k:v for k,v in fm.items() if k.startswith('Engine') and k[6:].isdigit()}
        types = {k:dict(Main={a:b for a,b in v['Main'].items() if a in
                    ('Type','MinThrMult','MaxThrMult','AfterburnerBoost','ThrustMax')})
                 for k,v in fm.items() if k.startswith('EngineType')}
        for k,v in types.items():
            if 'VelocityType' not in v['Main']['ThrustMax']:
                if path != PREFIX+'flightmodels/fm/j_8f.blkx':
                    raise ValueError('New missing thrust VelocityType: '+path)
                exceptions[path] = dict(kind='assumed',evidence_level='D',field=k+'.Main.ThrustMax.VelocityType',
                    value='TAS',reason='field absent; explicit TAS interpretation for J-8F numeric velocity table')
            table=v['Main']['ThrustMax']
            heights=[key.split('_')[1] for key in table if key.startswith('Altitude_')]
            speeds=[key.split('_')[1] for key in table if key.startswith('Velocity_')]
            missing={prefix:[f'{prefix}_{i}_{j}' for i in heights for j in speeds if f'{prefix}_{i}_{j}' not in table]
                     for prefix in ('ThrustMaxCoeff','ThrAftMaxCoeff')}
            if any(missing.values()):
                sparse[path]=dict(kind='assumed',evidence_level='D',missing_fields=missing,
                    interpretation='interpolate existing row nodes only; clamp row/domain boundaries; no filled coefficients')
        retain(path,engines|types,['EngineN (actual instances/overrides)','EngineTypeN.Main.Type/MinThrMult/MaxThrMult/AfterburnerBoost/ThrustMax'])
    globals_ = read(GLOBAL)
    keep_s = {k:v for k,v in globals_['sensorsConstants'].items() if any(s in k for s in
                ('engineInfraRedBrightness','engineAfterburnerInfraRedBrightness','airframeInfraRed'))}
    keep_f = {k:v for k,v in globals_['fmConstants'].items() if 'InfraRedBrightness' in k}
    retain(GLOBAL,dict(sensorsConstants=keep_s,fmConstants=keep_f),['fmConstants.*InfraRedBrightness','sensorsConstants.engine[Afterburner]InfraRedBrightness* / airframeInfraRed*'])
    # The three public representative classes remain available even if an aircraft uses a differently named launcher.
    for name in ('countermeasure_split_launcher_jet','countermeasure_split_launcher_jet_bol','countermeasure_split_launcher_jet_only_flare_large'):
        path=PREFIX+'weapons/rocketguns/'+name+'.blkx'; d=read(path)
        retain(path,{'bullet':d['bullet']},['bullet (complete actual payloads)'])
    coverage = dict(schema_version=1,version=VERSION,commit=COMMIT,
        selection=dict(mode='Air RB',minimum_br=13.0,type_field='unittags.type=aircraft',
            rank_field='wpcost.economicRankHistorical',rank_conversion='round(economicRank/3+1, 0.1)',
            br_source=WP,type_source=TAGS,conversion_source=RANKS,
            source_sha256={p:originals[p] for p in (WP,TAGS,RANKS)}),
        aircraft=aircraft,excluded=excluded,weapons=dict(sorted(weapons.items())),flare_sources=flare_sources,
        container_sources={p:dict(sha256=originals[p]) for p in sorted(containers)},
        thrust_velocity_assumptions=exceptions,
        sparse_thrust_tables=sparse,
        counts=dict(aircraft=len(aircraft),fm_files=len(fms),weapon_configs=len(weapons),
            missile_ids=len({v['missile_id'] for v in weapons.values()}),containers=len(containers),excluded=len(excluded)),
        limitations=['IR source-informed mechanisms remain D; no replay calibration',
            'rangeBand3/6 have raw ranges but no additional modeled environmental sources',
            'R-73/R-73E seeker supported; existing TVC flight-model exclusion retained',
            'AAM-3 mask-only rejection lacks threshold and reaction fields; automatic IRCCM mechanism unresolved'])
    coverage['counts'].update(automatic_weapon_configs=sum(v['supported'] for v in weapons.values()),
        unresolved_weapon_configs=sum(not v['supported'] for v in weapons.values()),
        runtime_profile_missile_ids=len({v['missile_id'] for v in weapons.values() if v['runtime_profile']}))
    fields=dict(schema_version=1,version=VERSION,commit=COMMIT,sources=extracts)
    output=root/'data/profile_sets'/VERSION/'infrared';output.mkdir(parents=True,exist_ok=True)
    for name,data in [('coverage.json',coverage),('game_fields.json',fields)]:
        (output/name).write_text(json.dumps(data,ensure_ascii=False,indent=2,sort_keys=True)+'\n')
    return coverage['counts']


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-dir',type=Path,default=Path(__file__).resolve().parents[1])
    print(json.dumps(build(parser.parse_args().project_dir),sort_keys=True))
