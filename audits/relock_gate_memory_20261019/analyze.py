#!/usr/bin/env python3
"""Conditional diagnostic gates, no coefficient fitting and no production change."""
from __future__ import annotations
import argparse,bisect,collections,gzip,hashlib,json,math
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
DEG=180/math.pi
EARLY=1-2/48-2*10/32767

def read(p):
    with (gzip.open(p,'rt') if p.suffix=='.gz' else p.open()) as f:return json.load(f)
def clean(x):
    if isinstance(x,np.ndarray):return clean(x.tolist())
    if isinstance(x,np.generic):return clean(x.item())
    if isinstance(x,dict):return {str(k):clean(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [clean(v) for v in x]
    if isinstance(x,float) and not math.isfinite(x):return None
    return x
def save(p,x):
    p.parent.mkdir(parents=True,exist_ok=True)
    with (gzip.open(p,'wt') if p.suffix=='.gz' else p.open('w')) as f:json.dump(clean(x),f,ensure_ascii=False,allow_nan=False,separators=(',',':'))
def vec(x):return np.asarray(x,float)
def unit(x):
    x=vec(x);n=float(np.linalg.norm(x))
    if n<1e-9:raise ValueError('Zero direction')
    return x/n
def angle(a,b):return math.atan2(float(np.linalg.norm(np.cross(a,b))),float(np.dot(a,b)))*DEG
def ray(a):
    y,p=a;return np.array([math.cos(p)*math.cos(y),math.sin(p),-math.cos(p)*math.sin(y)])
def body_angles(n):
    n=unit(n);return np.array([math.atan2(-n[2],n[0]),math.asin(np.clip(n[1],-1,1))])
def propagate(n,w,t):return Rotation.from_rotvec(-vec(w)*t).apply(n)
def metric(v):
    a=np.asarray([x for x in v if x is not None and math.isfinite(x)],float)
    if len(a)==0:return dict(n=0)
    return dict(n=len(a),median=float(np.median(a)),rms=float(np.sqrt(np.mean(a*a))),min=float(a.min()),max=float(a.max()),p95=float(np.percentile(a,95)))

class Target:
    def __init__(self,u):
        rr=[]
        for r in u['track']:
            if rr and r['t_ms']==rr[-1]['t_ms']:
                if r['xyz']!=rr[-1]['xyz']:raise ValueError('Conflicting target duplicate')
                continue
            if rr and r['t_ms']<rr[-1]['t_ms']:raise ValueError('Nonmonotonic target')
            rr.append(r)
        self.rows=rr;self.t=[r['t_ms']/1000 for r in rr];self.name=u['unit_name'];self.eid=u['eid']
    def at(self,r,clock):
        t=r['packet_ms']/1000 if clock=='packet' else r['tick']/48;j=bisect.bisect_right(self.t,t)-1
        if not 0<=j<len(self.t)-1:return None
        dt=self.t[j+1]-self.t[j]
        if not 0<dt<=1:return None
        a,b=self.rows[j:j+2];x=(t-self.t[j])/dt
        return ((1-x)*vec(a['xyz'])+x*vec(b['xyz']),(vec(b['xyz'])-vec(a['xyz']))/dt,dt)

def geometry(r,target,clock):
    k=target.at(r,clock)
    if k is None:return None
    p,v,dt=k;own=vec(r['p']);vm=vec(r['vm']);rel=p-own;R=float(np.linalg.norm(rel))
    if R<=1:return None
    n=rel/R;g=vec(r['Q'])@ray(r['angles']);delta=angle(n,g);rr=float((v-vm)@n);gt=float(v@n)
    limits={}
    for half in (7.5,15.):
        logleg=max(-math.log(2)*(delta/half)**2,math.log(1e-3));logK=4*math.log(16000/R)+2*logleg
        limits[str(half)]=dict(min_rcs_m2=2*math.exp(-logK),margin_ref2_db=10*logK/math.log(10))
    return dict(R=R,n=n,p=p,v=v,rr=rr,ground_radial=gt,delta=delta,g=g,target_bracket_s=dt,limits=limits,ps_error=float(np.linalg.norm(vec(r['Ps'])-p)),look_down=bool(p[1]<own[1]),target_altitude_m=float(p[1]),true_los_rate_deg_s=float(np.linalg.norm(np.cross(v-vm,n))/R)*DEG,
                own_projection=float(vm@n),body_target_angles_deg=body_angles(vec(r['Q']).T@n)*DEG)

def channels(r,g):
    n=unit(r['n']);vs=vec(r['Vs']);vm=vec(r['vm']);rrPV=float((vs-vm)@n);vgPV=float(vs@n)
    out=dict(tick=r['tick'],d_minus_Vs_dot_n=r['d']-vgPV,closing_plus_PV_relative=r['closing']+rrPV,closing_minus_PV_relative=r['closing']-rrPV,last_minus_PV_relative=r['last']-rrPV,R_minus_Ps_distance=r['R']-float(np.linalg.norm(vec(r['Ps'])-vec(r['p']))))
    if g is not None:out.update(ps_error_m=g['ps_error'],d_minus_ground=r['d']-g['ground_radial'],closing_plus_true_relative=r['closing']+g['rr'],saved_R_minus_true=r['R']-g['R'])
    return out

def diagnose(seed,prev,r,target,clock,previous_row=None):
    a=geometry(seed,target,clock);b=geometry(r,target,clock)
    if a is None or b is None:return None
    age=(r['tick']-seed['tick'])/48
    if age<0:return None
    vm=vec(r['vm']);npred=propagate(seed['n'],seed['w'],age);pp=vec(seed['Ps'])+vec(seed['Vs'])*age;pn=unit(pp-vec(r['p']))
    n0=unit(seed['n']);vs=vec(seed['Vs']);vm0=vec(seed['vm'])
    seed_rr=float((vs-vm0)@n0)
    centers={
      'relative_frozen_truth':a['rr'],
      'relative_frozen_PV':seed_rr,
      'closing_frozen':-seed['closing'],
      'target_radial_d_current_own':seed['d']-float(vm@npred),
      'PV_CV':float((vs-vm)@pn),
      'Vs_projected_predicted_LOS':float((vs-vm)@npred),
      'd_current_saved_closing_DIAGNOSTIC':-r['closing'],
    }
    ranges={'range_frozen':seed['R'],'range_linear_PV':seed['R']+seed_rr*age,'range_linear_closing':seed['R']-seed['closing']*age,'range_PV_CV':float(np.linalg.norm(pp-vec(r['p']))),'current_saved_range_DIAGNOSTIC':r['R']}
    if prev is not None and 0<seed['tick']-prev['tick']<=8:
        h=(seed['tick']-prev['tick'])/48
        centers['closing_secant']=-seed['closing']-(seed['closing']-prev['closing'])*age/h
        centers['d_secant_current_own']=seed['d']+(seed['d']-prev['d'])*age/h-float(vm@npred)
        ranges['range_secant']=seed['R']+(seed['R']-prev['R'])*age/h
    di={k:b['rr']-v for k,v in centers.items()};ri={k:b['R']-v for k,v in ranges.items()}
    angpred=angle(npred,b['n']);angaverage=angle(n0,b['n'])/max(age,1/48)
    offset=angle(npred,b['g']);oldclutter=-float(vm0@n0);predclutter=-float(vm@npred)
    gates={}
    for k,v in di.items():
        for lim in (10.,20.,300.):gates[f'doppler/{k}/{lim:g}']=abs(v)<=lim
    for k,v in ri.items():
        for lim in (150.,300.,5000.):gates[f'range/{k}/{lim:g}']=abs(v)<=lim
    for lim in (5.,7.5,15.):gates[f'angle/prediction/{lim:g}']=angpred<=lim
    gates['angle/average_rate30']=angaverage<=30
    gates['angle/prediction_per_tick30']=angpred<=30/48
    gates['notch/instantaneous']=not(b['look_down'] and abs(b['ground_radial'])<50)
    gates['notch/frozen_ground_center']=not(b['look_down'] and abs(b['rr']-oldclutter)<50)
    gates['notch/predicted_LOS_center']=not(b['look_down'] and abs(b['rr']-predclutter)<50)
    common=b['limits']['15.0']['min_rcs_m2']<=2 and 300<=b['R']<=25000 and abs(b['rr'])<=3000 and gates['notch/instantaneous']
    out=dict(index=r['index'],tick=r['tick'],code=r['code'],timer=r['timer'],flag=r['flag'],age_from_seed_s=age,seed_ps_error_m=a['ps_error'],current_ps_error_m=b['ps_error'],distance_m=b['R'],radial_speed_mps=b['rr'],target_ground_radial_mps=b['ground_radial'],target_altitude_m=b['target_altitude_m'],look_down=b['look_down'],beam_offset_deg=b['delta'],signal_limits=b['limits'],common_candidate_eligible=common,true_los_rate_deg_s=b['true_los_rate_deg_s'],body_target_angles_deg=b['body_target_angles_deg'],predicted_direction_error_deg=angpred,average_los_rate_deg_s=angaverage,predicted_direction_vs_recorded_deg=angle(npred,r['n']),doppler_innovation_mps=di,range_innovation_m=ri,predicted_n=npred,notch_separation_frozen_mps=b['rr']-oldclutter,notch_separation_predicted_mps=b['rr']-predclutter,gates=gates,raw_sha=r['raw_sha'],target_bracket_s=b['target_bracket_s'])
    if previous_row is not None and 0<r['tick']-previous_row['tick']<=8:
        old=geometry(previous_row,target,clock);dt=(r['tick']-previous_row['tick'])/48
        if old is not None:
            out['body_los_rate_deg_s']=angle(vec(previous_row['Q']).T@old['n'],vec(r['Q']).T@b['n'])/dt
            out['beam_world_rate_deg_s']=angle(old['g'],b['g'])/dt
            out['body_rotation_rate_deg_s']=float(Rotation.from_matrix(vec(r['Q'])@vec(previous_row['Q']).T).magnitude())*DEG/dt
            gates['angle/body_los_rate60']=out['body_los_rate_deg_s']<=60
            gates['angle/world_beam_rate60']=out['beam_world_rate_deg_s']<=60
    return out

def group_scores(episodes):
    held=[r for e in episodes for r in e['rows'] if r['role']=='held' and r['age_from_seed_s']<EARLY and r['common_candidate_eligible']]
    returns=[r for e in episodes for r in e['rows'] if r['role']=='return' and r['age_from_seed_s']<EARLY and r['flag']==0]
    keys=sorted(set(k for r in held+returns for k in r['gates']))
    table={}
    for k in keys:
        h=[r for r in held if k in r['gates']];f=[r for r in returns if k in r['gates']]
        table[k]=dict(held_eligible_n=len(h),held_rejected=sum(not r['gates'][k] for r in h),held_still_pass=sum(r['gates'][k] for r in h),early_returns_n=len(f),early_returns_rejected=sum(not r['gates'][k] for r in f))
    return dict(episodes=len(episodes),entities=len({(e['replay'],e['eid']) for e in episodes}),early_held_signal_eligible_records=len(held),episodes_with_early_held_signal_eligible=sum(any(r['role']=='held' and r['age_from_seed_s']<EARLY and r['common_candidate_eligible'] for r in e['rows']) for e in episodes),early_return_records=len(returns),gate_results=table)

def main():
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--out',type=Path,required=True);z=p.parse_args();z.out.mkdir(parents=True,exist_ok=False)
    manifest=read(z.input/'MANIFEST.json')
    for e in manifest:
        f=z.input/e['path'];assert f.stat().st_size==e['bytes'] and hashlib.sha256(f.read_bytes()).hexdigest()==e['sha256']
    spells=read(z.input/'episodes.json.gz');fresh=read(z.input/'fresh.json.gz');units=read(z.input/'units.json.gz');anchors=read(z.input/'anchors.json.gz');infos=read(z.input/'infos.json')
    targets={(rep,int(u['eid'])):Target(u) for rep,uu in units.items() for u in uu};outcomes=[];missing=[];ch=[];anchorrows=[]
    for row in fresh:
        r=row['record'];target=targets.get((row['replay'],row['target_eid']));g=None if target is None else geometry(r,target,'packet');v=channels(r,g);v.update(replay=row['replay'],eid=row['eid']);ch.append(v)
    for clock in ('packet','state'):
        for ep in spells:
            target=targets.get((ep['replay'],ep['target_eid']))
            if target is None:missing.append([clock,ep['episode_id'],'target_missing']);continue
            rows=[];previous=ep['seed']
            for role,r in [('held',r) for r in ep['held']]+([] if ep['exit'] is None else [('return' if ep['exit_reason']=='fresh_return' else 'exit',ep['exit'])]):
                v=diagnose(ep['seed'],ep['previous_fresh'],r,target,clock,previous)
                if v is not None:v['role']=role;rows.append(v)
                else:missing.append([clock,ep['episode_id'],r['tick']])
                previous=r
            if not rows:continue
            nearby=rows[0]['seed_ps_error_m']<=50
            outcomes.append(dict(clock=clock,episode_id=ep['episode_id'],replay=ep['replay'],eid=ep['eid'],uid=ep['uid'],strict=ep['strict'],key_stable=ep['key_stable'],seed_tick=ep['seed']['tick'],exit_reason=ep['exit_reason'],nearby_seed_reference=nearby,rows=rows))
    for uid,rs in anchors.items():
        info=next(i for i in infos if str(i['uid'])==uid and (i['replay'],i['eid']) in [('2026.10.08_11.49.38',2439170),('2026.10.08_12.11.04',1612803)])
        target=targets[info['replay'],info['target_eid']]
        if uid=='118':
            seed=next(r for r in rs if r['tick']==11524);prev=next(r for r in rs if r['tick']==11522)
            for clock in ('packet','state'):
                for j,r in enumerate(rs):
                    if not 11524<=r['tick']<=11625:continue
                    v=diagnose(seed,prev,r,target,clock,rs[j-1] if j else None)
                    if v is not None:
                        v.update(uid=118,clock=clock,age_s=(r['tick']-info['epoch_tick'])/48,current_channels=channels(r,geometry(r,target,clock)))
                        if j and r['code']==31 and rs[j-1]['code']==25:
                            v['new_INS_seed_diagnostic']=diagnose(rs[j-1],rs[j-2] if j>=2 else None,r,target,clock,rs[j-1])
                        anchorrow=v;anchorrows.append(anchorrow)
    primary=[e for e in outcomes if e['clock']=='packet' and e['strict'] and e['nearby_seed_reference']]
    groups={'strict_all_packet':group_scores([e for e in outcomes if e['clock']=='packet' and e['strict']]),'strict_nearby_packet':group_scores(primary),'strict_nearby_state':group_scores([e for e in outcomes if e['clock']=='state' and e['strict'] and e['nearby_seed_reference']])}
    cs={k:metric([r.get(k) for r in ch]) for k in ('d_minus_Vs_dot_n','closing_plus_PV_relative','closing_minus_PV_relative','last_minus_PV_relative','R_minus_Ps_distance','d_minus_ground','closing_plus_true_relative','saved_R_minus_true')}
    close=[r for r in ch if r.get('ps_error_m',1e100)<=50];cc={k:metric([r.get(k) for r in close]) for k in ('d_minus_ground','closing_plus_true_relative','saved_R_minus_true')}
    # Explicitly enumerate other-case instances with early positive signal and
    # no conventional notch yet no accepted update; do not assert causal absence.
    comparable=[]
    for e in primary:
        rr=[r for r in e['rows'] if r['role']=='held' and r['common_candidate_eligible'] and r['age_from_seed_s']<EARLY]
        if rr:comparable.append({k:e[k] for k in ('episode_id','replay','eid','uid','seed_tick','exit_reason')}|dict(eligible_held=len(rr),first_candidate=rr[0],return_row=next((r for r in e['rows'] if r['role']=='return'),None)))
    summary=dict(scope='Fixed-output diagnostic acceptance tests, not autonomous observer, not known echo labels. All gate centers and widths are predeclared hypotheses; current saved scalar diagnostics are not predictors.',input_files=len(manifest),counts=dict(spells=len(spells),fresh_records=len(fresh),scored_episode_clock_pairs=len(outcomes),missing=len(missing)),fresh_channel_consistency=cs,nearby_fresh_channel_consistency=cc,groups=groups,comparable=comparable,anchor118=anchorrows,missing=missing)
    save(z.out/'summary.json',summary);save(z.out/'episodes.json.gz',outcomes);save(z.out/'fresh_channels.json.gz',ch);save(z.out/'anchor118.json',anchorrows)
    print('CHANNELS',json.dumps(clean(cs)),flush=True);print('NEARBY_CHANNELS',json.dumps(clean(cc)),flush=True)
    for k,g in groups.items():print('GROUP',k,json.dumps(g),flush=True)
    for r in anchorrow if False else anchorrows:
        if r['clock']=='packet' and r['tick'] in (11524,11526,11532,11571,11573,11575,11615,11617):print('ANCHOR118',json.dumps(clean({k:v for k,v in r.items() if k!='gates'})),flush=True)
    for e in comparable[:8]:print('OTHER_CASE',json.dumps(clean(e)),flush=True)
    print('COUNTS',json.dumps(summary['counts']),flush=True)
if __name__=='__main__':main()
