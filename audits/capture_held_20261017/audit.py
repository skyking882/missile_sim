#!/usr/bin/env python3
"""Offline War Thunder serialized-state audit. Does not import/change simulation code."""
from __future__ import annotations
import argparse, collections, gzip, hashlib, json, math, platform, struct, sys
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation

RAD=180/math.pi
H=1/48
ALPHA=0.926
TAU=0.078525
ANCHORS={('2026.10.08_11.49.38',2439170):118,('2026.10.08_12.11.04',1612803):148}

def serial(x):
    if isinstance(x,np.ndarray):return serial(x.tolist())
    if isinstance(x,np.generic):return serial(x.item())
    if isinstance(x,dict):return {str(k):serial(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [serial(v) for v in x]
    if isinstance(x,float) and not math.isfinite(x):return None
    return x

def save(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    op=gzip.open if path.suffix=='.gz' else open
    with op(path,'wt',encoding='utf8') as f:json.dump(serial(obj),f,ensure_ascii=False,separators=(',',':'),allow_nan=False)

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()

def unit(n):
    q=np.linalg.norm(n)
    if q<1e-12:raise ValueError('Zero direction')
    return np.asarray(n,dtype=float)/q

def angle(a,b):return float(np.arctan2(np.linalg.norm(np.cross(a,b)),np.dot(a,b)))

def rotate(n,w,t):return Rotation.from_rotvec(-np.asarray(w)*t).apply(n)

def correction(p,n):
    c=np.cross(n,p);l=np.linalg.norm(c);th=np.arctan2(l,n@p)
    if th>math.pi-1e-6:raise ValueError('Antipodal correction')
    return c*(th/l) if l>1e-14 else c

def blend(p,m,alpha=ALPHA):
    c=correction(p,m)
    return unit(Rotation.from_rotvec(-alpha*c).apply(p))

def step(n,w,m,t):
    p=rotate(n,w,t);nn=blend(p,m)
    return nn,np.asarray(w)+correction(p,nn)/(TAU+t/2)

def ray(a):
    y,p=a
    return np.array([math.cos(p)*math.cos(y),math.sin(p),-math.cos(p)*math.sin(y)])

def angles(n):
    n=unit(n)
    return np.array([math.atan2(-n[2],n[0]),math.asin(np.clip(n[1],-1,1))])

def matrix(h,a,b):
    ch,sh=math.cos(h),math.sin(h);ca,sa=math.cos(a),math.sin(a);cb,sb=math.cos(b),math.sin(b)
    return np.array([[ch*ca,-ch*sa*cb+sh*sb,ch*sa*sb+sh*cb],[sa,ca*cb,-ca*sb],[-sh*ca,sh*sa*cb+ch*sb,-sh*sa*sb+ch*cb]])

def parse(raw,maps,index):
    f=raw['fields'];k=int(f['skipped_bits'])
    if k not in (607,639):return None
    data=bytes.fromhex(raw['byte_slice_hex'])
    s=''.join(f'{v:08b}' for v in data)
    off=int(f['seeker_start_bit'])-(int(f['start_bit'])//8)*8
    assert off>=0 and off+k<=len(s),(off,k,len(s),f)
    s=s[off:off+k];v={};ints={}
    for z in maps[str(k)]:
        name,o,w,enc=z['name'],z['start_bit'],z['width'],z['encoding']
        bits=s[o:o+w]
        if enc=='raw_bit':x=int(bits)
        elif enc=='ULEB32':continue
        else:
            packed=bytes(int(bits[j:j+8].ljust(8,'0'),2) for j in range(0,len(bits),8))
            x=struct.unpack({'u8':'<B','i16':'<h','float32':'<f'}[enc],packed)[0]
        ints[name]=x;v[name]=x*z['scale']/32767 if 'scale' in z else x
    assert v['type8']==3 and v['flagC']==1 and v['compression']==1
    vec=lambda name:np.array([v[name+'_'+j] for j in 'xyz'])
    n0=vec('direction');n=unit(n0)
    ac=np.array([(ints[j]+0.5*np.sign(ints[j]))*math.pi/32767 for j in ('angle0','angle1')])
    q=matrix(f['euler_x'],f['euler_y'],f['euler_z'])
    vm=np.array([(int(f['b7_'+a+'_raw_u16'])+32768)%65536-32768 for a in 'xyz'])*4000/65535
    return dict(index=index,eid=int(f['eid']),tick=int(f['b5']),packet_ms=int(f['time_ms']),code=int(v['type8'])*8+int(v['flagA'])*4+int(v['flagB'])*2+int(v['flagC']),timer=v['timer'],timer_int=int(ints['timer']),flag=int(v['flag_after_timer0']),flags=[int(v['flag_after_timer0']),int(v['flag_after_timer1'])],Q=q,p=np.array([f[a] for a in 'xyz']),vm=vm,Ps=vec('position'),Vs=vec('velocity'),unknown=vec('unknown_vector'),n=n,nraw=n0,w=vec('rate'),angles=ac,angle_int=[int(ints[x]) for x in ('angle0','angle1')],m=q@ray(ac),R=v['range'],d=v['scalar_radial'],closing=v['scalar_closing'],last=v['scalar_last'],key=int(v['designation_key']),raw_sha=hashlib.sha256(json.dumps(raw,sort_keys=True,separators=(',',':')).encode()).hexdigest())

def metric(x):
    a=np.asarray(x,dtype=float)
    if not len(a):return {'n':0}
    return dict(n=len(a),rms=float(np.sqrt(np.mean(a*a))),median=float(np.median(a)),p95=float(np.percentile(a,95)),max=float(np.max(a)))

def grouped(rows,keys,values):
    out={}
    for field in keys:
        dd=collections.defaultdict(list)
        for r in rows:dd[str(r[field])].append(r)
        out[field]={k:{v:metric([r[v] for r in rr if r.get(v) is not None]) for v in values} for k,rr in dd.items()}
    return out

def brief(r):return {k:serial(r[k]) for k in ('index','tick','packet_ms','code','timer','flag','Ps','Vs','unknown','n','w','angles','angle_int','R','d','closing','last','p','vm','Q','raw_sha')}

def errors(n,w,b):
    e=np.asarray(w)-b['w'];lam=float(e@b['n']);perp=e-lam*b['n']
    return dict(n_deg=angle(n,b['n'])*RAD,w_deg_s=float(np.linalg.norm(e))*RAD,lambda_error_deg_s=lam*RAD,transverse_deg_s=float(np.linalg.norm(perp))*RAD,P_m=float(np.linalg.norm(b['p']+b['R']*n-b['Ps'])))

def capture_models(a,b,T):
    oldr=a['Ps']-a['p'];R=np.linalg.norm(oldr);ng=unit(oldr)
    geom=np.cross(a['Vs']-a['vm'],ng)/max(R,1e-12)
    seeds={'keep':a['w'],'Qold':a['Q']@a['w'],'Qnew':b['Q']@a['w'],'transport':b['Q']@a['Q'].T@a['w'],'zero':np.zeros(3),'past_geometry':geom}
    out={};m=b['m']
    for name,s in seeds.items():
        out[name+'/reset_only']=(m,s)
        out[name+'/self_h']=step(m,s,m,H)
        out[name+'/self_T']=step(m,s,m,T)
        n,w=m,s.copy()
        for _ in range(b['tick']-a['tick']):n,w=step(n,w,m,H)
        out[name+'/self_ticks']=(n,w)
        out[name+'/carry_n']=step(a['n'],s,m,T)
        npred=unit(a['Ps']+a['Vs']*T-b['p'])
        n=blend(npred,m);w=s+correction(npred,n)/(TAU+T/2)
        out[name+'/past_Ps_correction']=(n,w)
    return out

def main():
    cli=argparse.ArgumentParser();cli.add_argument('--reference',type=Path,required=True);cli.add_argument('--derived',type=Path,required=True);cli.add_argument('--out',type=Path,required=True);args=cli.parse_args();args.out.mkdir(parents=True,exist_ok=False)
    dp=next(args.reference.rglob('server_dataset.json.gz'));mp=next(args.reference.rglob('missile_seeker_fieldmap.json'))
    d=json.load(gzip.open(dp,'rt'));maps=json.loads(mp.read_text())['field_maps']
    info={(z['replay'],int(z['eid'])):z for z in d['infos']};assert len(info)==501
    refs={}
    for c,lab in zip(d['cases'],d['labels']):
        rep,eid=c['info']['replay'],int(c['info']['eid'])
        for j,t in enumerate(c['ticks']):
            refs[(rep,eid,int(t))]=dict(Q=np.array(c['Q'][j]),w=np.array(lab['w'][j-1]) if j else np.array(c['seed'])/RAD,n=np.array(lab['n'][j-1]) if j else None)
    inputs={str(dp):sha(dp),str(mp):sha(mp)}
    cap=[];held=[];returns=[];clips=[];clip_predictions=[];episodes=[];anchors={};inventory=[];seenref=set();maxcheck=collections.defaultdict(float)
    total=0;selected=0;unparsed=0
    files=sorted(args.derived.rglob('weapon_raw.jsonl.gz'));assert len(files)==5
    for path in files:
        rep=path.parent.name;wanted={eid for rr,eid in info if rr==rep};by=collections.defaultdict(list);counts=collections.Counter();inputs[str(path)]=sha(path)
        with gzip.open(path,'rt') as f:
            for idx,line in enumerate(f):
                raw=json.loads(line);total+=1
                eid=int(raw['fields']['eid'])
                if eid not in wanted:continue
                selected+=1;r=parse(raw,maps,idx)
                if r is None:
                    by[eid].append(None);unparsed+=1;continue
                by[eid].append(r);counts['parsed']+=1
                key=(rep,eid,r['tick'])
                if key in refs:
                    ref=refs[key];seenref.add(key)
                    for name in ('Q','w','n'):
                        if ref[name] is not None:maxcheck[name]=max(maxcheck[name],float(np.max(np.abs(ref[name]-r[name]))))
        assert maxcheck['Q']<1e-12 and maxcheck['w']<1e-12 and maxcheck['n']<1e-12,dict(maxcheck)
        for eid,rows in by.items():
            meta=info[(rep,eid)];fam=meta['family'];uid=meta['uid'];occ=0
            valid=[r for r in rows if r is not None]
            if (rep,eid) in ANCHORS:
                anchors[str(ANCHORS[(rep,eid)])]=[brief(r) for r in valid]
            base=dict(replay=rep,eid=eid,uid=uid,family=fam)
            for r in valid:
                if fam!='R-77-1':continue
                bn=angles(r['Q'].T@r['n']);limited=bool(np.any(np.abs(r['angles'])*RAD>=54.99));excess=max(0.,float(np.max(np.abs(bn))*RAD-55))
                clips.append(dict(**base,tick=r['tick'],code=r['code'],flag=r['flag'],fresh=r['timer']==0,limited=limited,n_body_angles_deg=bn*RAD,recorded_angles_deg=r['angles']*RAD,excess_deg=excess,ray_error_deg=angle(r['m'],r['n'])*RAD,clipped_ray_error_deg=angle(r['Q']@ray(np.clip(bn,-55/RAD,55/RAD)),r['m'])*RAD))
            current_episode=None
            for a,b in zip(rows,rows[1:]):
                if a is None or b is None:
                    current_episode=None;continue
                gap=b['tick']-a['tick'];T=gap/48
                if gap<=0 or gap>8:
                    counts['invalid_gap']+=1;current_episode=None;continue
                ident=dict(**base,tick0=a['tick'],tick1=b['tick'],gap=gap,age0=(a['tick']-meta['epoch_tick'])/48,age1=(b['tick']-meta['epoch_tick'])/48,flag0=a['flag'],flag1=b['flag'],flag_pair=str((a['flag'],b['flag'])),timer0=a['timer'],timer1=b['timer'])
                if a['code']==25 and b['code']==31:
                    occ+=1;counts['capture']+=1
                    pred=capture_models(a,b,T)
                    cap.append(dict(**ident,occurrence=occ,original_capture=b['tick']==meta['native_ticks'][0],fresh=b['timer']==0,limited=bool(np.any(np.abs(b['angles'])*RAD>=54.99)),predictions={name:dict(**errors(n,w,b),n=n,w=w) for name,(n,w) in pred.items()},old=brief(a),new=brief(b)))
                if a['code']!=31 or b['code']!=31:
                    current_episode=None;continue
                if b['timer']>0:
                    hp=rotate(a['n'],a['w'],T)
                    pv=unit(a['Ps']+a['Vs']*(a['timer']+T)-b['p'])
                    cache_equal=bool(np.array_equal(a['Ps'],b['Ps']) and np.array_equal(a['Vs'],b['Vs']))
                    he=dict(**ident,stage='held_to_held' if a['timer']>0 else 'fresh_to_held',cache_equal=cache_equal,rate_equal=bool(np.array_equal(a['w'],b['w'])),timer_increment_error_s=abs((b['timer']-a['timer'])-T),rotate_deg=angle(hp,b['n'])*RAD,frozen_world_deg=angle(a['n'],b['n'])*RAD,frozen_body_deg=angle(b['Q']@a['Q'].T@a['n'],b['n'])*RAD,PV_deg=angle(pv,b['n'])*RAD,rate_error_deg_s=float(np.linalg.norm(b['w']-a['w']))*RAD,R_frozen_error_m=abs(b['R']-a['R']),R_CV_error_m=abs(b['R']-np.linalg.norm(a['Ps']+a['Vs']*(a['timer']+T)-b['p'])),d_equal=b['d']==a['d'],last_equal=b['last']==a['last'])
                    held.append(he)
                    if current_episode is None or current_episode['last_index']!=a['index'] or current_episode['flag']!=b['flag']:
                        current_episode=dict(**base,start_tick=a['tick'],last_tick=a['tick'],last_index=a['index'],flag=b['flag'],seed_n=a['n'].copy(),seed_w=a['w'].copy(),seed_Ps=a['Ps'].copy(),seed_Vs=a['Vs'].copy(),n_error_deg=[],w_error_deg_s=[],P_cache_error_m=[],V_cache_error_mps=[])
                        episodes.append(current_episode)
                    ep=current_episode;dt=(b['tick']-ep['start_tick'])/48
                    ep['n_error_deg'].append(angle(rotate(ep['seed_n'],ep['seed_w'],dt),b['n'])*RAD);ep['w_error_deg_s'].append(float(np.linalg.norm(ep['seed_w']-b['w']))*RAD);ep['P_cache_error_m'].append(float(np.linalg.norm(ep['seed_Ps']-b['Ps'])));ep['V_cache_error_mps'].append(float(np.linalg.norm(ep['seed_Vs']-b['Vs'])));ep['last_tick']=b['tick'];ep['last_index']=b['index']
                else:
                    current_episode=None
                    if a['timer']>0:
                        pred={}
                        for label,t in [('continue_save_dt',T),('continue_since_measurement',T+a['timer'])]:
                            n,w=step(a['n'],a['w'],b['m'],t);pred[label]=errors(n,w,b)
                        for label,s in [('reset_keep',a['w']),('reset_Qold',a['Q']@a['w']),('reset_zero',np.zeros(3))]:
                            n,w=step(b['m'],s,b['m'],H);pred[label]=errors(n,w,b)
                        returns.append(dict(**ident,limited=bool(np.any(np.abs(b['angles'])*RAD>=54.99)),predictions=pred,old=brief(a),new=brief(b)))
                    if fam=='R-77-1' and a['timer']==0 and a['flag']==b['flag']==0 and np.any(np.abs(b['angles'])*RAD>=54.99):
                        p=rotate(a['n'],a['w'],T);prior=angles(b['Q'].T@p);ma=b['angles'].copy()
                        for j in range(2):
                            if abs(ma[j])*RAD>=54.99:ma[j]=max(prior[j],55/RAD) if ma[j]>0 else min(prior[j],-55/RAD)
                        pn,pw=step(a['n'],a['w'],b['m'],T)
                        cn,cw=step(a['n'],a['w'],b['Q']@ray(ma),T)
                        clip_predictions.append(dict(**ident,predictions={'as_exact_angle':errors(pn,pw,b),'prediction_only':errors(p,a['w'],b),'censored':errors(cn,cw,b)}))
        inventory.append(dict(replay=rep,entities=len(by),counts=dict(counts)))
        print('COMPLETED',rep,'capture',len(cap),'held',len(held),'return',len(returns),flush=True)
    assert len(seenref)==len(refs),(len(seenref),len(refs))
    def summarize_models(rows):
        if not rows:return {'n':0}
        names=list(rows[0]['predictions'])
        return {name:{key:metric([r['predictions'][name][key] for r in rows]) for key in ('n_deg','w_deg_s','lambda_error_deg_s','transverse_deg_s','P_m')} for name in names}
    def strata(rows):
        return {'all':summarize_models(rows),'R77':summarize_models([r for r in rows if r['family']=='R-77-1']),'R77_unlimited_fresh':summarize_models([r for r in rows if r['family']=='R-77-1' and not r.get('limited',False) and r.get('fresh',True)]),'original_501':summarize_models([r for r in rows if r.get('original_capture')]),'repeated':summarize_models([r for r in rows if r.get('occurrence',0)>1])}
    hs=grouped(held,['family','flag_pair','stage'],['rotate_deg','frozen_world_deg','frozen_body_deg','PV_deg','rate_error_deg_s','R_frozen_error_m','R_CV_error_m'])
    for r in episodes:
        r['steps']=len(r['n_error_deg']);r['duration_s']=(r['last_tick']-r['start_tick'])/48
        r['metrics']={k:metric(r[k]) for k in ('n_error_deg','w_error_deg_s','P_cache_error_m','V_cache_error_mps')}
    cl=grouped(clips,['code','flag','fresh','limited'],['excess_deg','ray_error_deg','clipped_ray_error_deg'])
    summary=dict(scope='Current-angle conditioned reconstruction at capture/hold boundaries; not a fully autonomous observation model',counts=dict(native_raw_records=total,selected_raw_records=selected,unsupported_selected_records=unparsed,reference_checks=len(seenref),captures=len(cap),held_pairs=len(held),return_fresh=len(returns),R77_angle_records=len(clips),R77_clipped_fresh_pairs=len(clip_predictions)),source_hashes=inputs,decoder_reference_max_abs_difference=dict(maxcheck),inventory=inventory,parameters=dict(alpha_eff=ALPHA,tau=TAU,h=H),capture=strata(cap),held=hs,held_cache=dict(cache_equal=sum(r['cache_equal'] for r in held),rate_equal=sum(r['rate_equal'] for r in held),timer_progression_compatible=sum(r['timer_increment_error_s']<2*10/32767 for r in held),d_equal=sum(r['d_equal'] for r in held)),held_episodes=dict(count=len(episodes),flag0_steps=sum(r['steps'] for r in episodes if r['flag']==0),flag0_n=metric([x for r in episodes if r['flag']==0 for x in r['n_error_deg']]),flag1_n=metric([x for r in episodes if r['flag']==1 for x in r['n_error_deg']])),return_to_fresh=strata(returns),clipping=cl,clipping_beyond=dict(at_limit=sum(r['limited'] for r in clips),at_limit_n_beyond_55p1=sum(r['limited'] and r['excess_deg']>.1 for r in clips),at_limit_n_beyond_56=sum(r['limited'] and r['excess_deg']>1 for r in clips)),clipped_prediction=summarize_models(clip_predictions),environment=dict(python=sys.version,numpy=np.__version__,platform=platform.platform()))
    save(args.out/'summary.json',summary);save(args.out/'capture_events.json.gz',cap);save(args.out/'held_pairs.json.gz',held);save(args.out/'held_episodes.json.gz',episodes);save(args.out/'fresh_return_events.json.gz',returns);save(args.out/'clipping_records.json.gz',clips);save(args.out/'clipped_predictions.json.gz',clip_predictions);save(args.out/'anchor_full_histories.json.gz',anchors)
    compact={k:summary[k] for k in ('counts','decoder_reference_max_abs_difference','held_cache','held_episodes','clipping_beyond')}
    for group in ('all','R77','R77_unlimited_fresh','original_501','repeated'):
        vals=summary['capture'][group]
        compact['capture_'+group]=sorted([(k,v['w_deg_s']['rms'],v['n_deg']['rms'],v['lambda_error_deg_s']['rms']) for k,v in vals.items() if isinstance(v,dict) and 'w_deg_s' in v],key=lambda x:x[1])[:10]
    compact['anchor_captures']=[{**{k:r[k] for k in ('replay','uid','occurrence','gap','age0','age1','timer1','flag1')},'best':sorted([(k,v['w_deg_s'],v['n_deg'],v['lambda_error_deg_s']) for k,v in r['predictions'].items()],key=lambda x:x[1])[:6]} for r in cap if (r['replay'],r['eid']) in ANCHORS]
    compact['held_flag_pairs']=hs['flag_pair'];compact['return_R77']=summary['return_to_fresh']['R77'];compact['clip_pred']=summary['clipped_prediction']
    save(args.out/'compact.json',compact)
    print('===COMPACT_RESULT===',json.dumps(serial(compact),ensure_ascii=False,allow_nan=False),flush=True)

if __name__=='__main__':main()
