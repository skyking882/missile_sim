#!/usr/bin/env python3
"""Game replay diagnostic: fixed own trajectory, independent aircraft positions.
No Q/PN/plant integration. No fitted thresholds. Pre-first-TRK INS Ps/Vs are
conditional replay priors, not an independently reconstructed data-link source.
"""
from __future__ import annotations
import argparse,bisect,copy,gzip,hashlib,json,math,sys
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from scipy.spatial.transform import Rotation,Slerp
from retry_clock import RetryClock
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
from aim120_model.radar_seeker import RadarSeekerObserver
from aim120_model.aerodynamics import body_axes_for_state
from aim120_model.tracking import TargetState

ANCHORS={'118':('2026.10.08_11.49.38',2439170),'148':('2026.10.08_12.11.04',1612803)}
C=np.diag([1.,1.,-1.]);H=1/48

def read(p):
    with (gzip.open(p,'rt') if p.suffix=='.gz' else p.open()) as f:return json.load(f)
def clean(x):
    if isinstance(x,np.ndarray):return clean(x.tolist())
    if isinstance(x,np.generic):return clean(x.item())
    if isinstance(x,dict):return {str(k):clean(v) for k,v in x.items()}
    if isinstance(x,(tuple,list)):return [clean(v) for v in x]
    if isinstance(x,float) and not math.isfinite(x):return None
    return x

def save(p,d):
    with (gzip.open(p,'wt') if p.suffix=='.gz' else p.open('w')) as f:json.dump(clean(d),f,ensure_ascii=False,separators=(',',':'),allow_nan=False)

def angle(a,b):
    a=np.asarray(a)/np.linalg.norm(a);b=np.asarray(b)/np.linalg.norm(b)
    return math.degrees(math.atan2(np.linalg.norm(np.cross(a,b)),np.dot(a,b)))

class LoggedRadar(RadarSeekerObserver):
    def detection_range(self,los,center,rcs):
        value=super().detection_range(los,center,rcs)
        self.beam_audit=dict(offset_deg=angle(los,center),detection_range_m=value,axis=list(center))
        return value

def build_frames(history,info,unit,clock):
    rs=history;ts=np.array([r['tick'] for r in rs]);assert np.all(np.diff(ts)>0)
    epoch=int(info['epoch_tick']);first=min(r['tick'] for r in rs if r['code']==31)
    priors=[r for r in rs if r['tick']<first and r['code']==25];assert priors
    pts=[r['tick'] for r in priors]
    # Deduplicate identical timestamp rows explicitly; conflicting positions fail.
    track=[];duplicates=0
    for r in unit['track']:
        if track and r['t_ms']==track[-1]['t_ms']:
            if r['xyz']!=track[-1]['xyz']:raise ValueError('Conflicting target positions at equal timestamp')
            duplicates+=1;continue
        if track and r['t_ms']<track[-1]['t_ms']:raise ValueError('Nonmonotonic target clock')
        track.append(r)
    tt=[r['t_ms']/1000 for r in track];assert len(tt)>1
    rot=Slerp(ts.astype(float),Rotation.from_matrix(np.array([r['Q'] for r in rs])))
    frames=[];savedown_checks=0
    stop=min(int(ts[-1]),epoch+288)
    for tick in range(int(ts[0]),stop+1):
        j=bisect.bisect_right(ts,tick)-1;j=min(j,len(rs)-2);a,b=rs[j:j+2]
        f=(tick-a['tick'])/(b['tick']-a['tick'])
        p=(1-f)*np.array(a['p'])+f*np.array(b['p']);v=(1-f)*np.array(a['vm'])+f*np.array(b['vm']);q=rot(float(tick)).as_matrix()
        packet=((1-f)*a['packet_ms']+f*b['packet_ms'])/1000
        time_target=packet if clock=='packet' else tick/48
        k=bisect.bisect_right(tt,time_target)-1
        if not 0<=k<len(track)-1:raise ValueError('Independent target history does not bracket query')
        dt=tt[k+1]-tt[k];alpha=(time_target-tt[k])/dt
        tp=(1-alpha)*np.array(track[k]['xyz'])+alpha*np.array(track[k+1]['xyz'])
        tv=(np.array(track[k+1]['xyz'])-np.array(track[k]['xyz']))/dt
        z=bisect.bisect_right(pts,tick)-1
        if z<0:raise ValueError('No available prior')
        prior=priors[z];dtp=(tick-prior['tick'])/48
        est=np.array(prior['Ps'])+np.array(prior['Vs'])*dtp
        qq=Rotation.from_matrix(C@q@C).as_quat();quat=[qq[3],qq[0],qq[1],qq[2]]
        own=SimpleNamespace(orientation_quaternion=quat)
        axes=body_axes_for_state(own)
        np.testing.assert_allclose(np.column_stack([axes.forward,axes.up,axes.right]),C@q@C,atol=1e-13,rtol=0)
        if tick in ts:
            orig=rs[bisect.bisect_left(ts,tick)];np.testing.assert_allclose(p,orig['p'],atol=1e-9,rtol=0);savedown_checks+=1
        frames.append(dict(tick=tick,age=(tick-epoch)/48,position=C@p,velocity=C@v,quaternion=quat,target_position=C@tp,target_velocity=C@tv,prior_position=C@est,prior_velocity=C@np.array(prior['Vs']),prior_source_tick=prior['tick'],packet_time_s=packet,target_query_time_s=time_target,target_bracket_s=[tt[k],tt[k+1]],own_bracket_ticks=[a['tick'],b['tick']]))
    meta=dict(first_available_age=(int(ts[0])-epoch)/48,first_available_before_startup=(int(ts[0])-epoch)/48<.75,last_prior_tick=priors[-1]['tick'],independent_target_eid=unit['eid'],independent_target_name=unit['unit_name'],target_duplicate_identical_rows=duplicates,saved_own_checks=savedown_checks,epoch_tick=epoch)
    return clean(frames),meta

def simulate(frames,profile,rcs,width,policy):
    sm=copy.deepcopy(profile['guidance']['sensor_model']);radar=LoggedRadar(sm,lock_range_m=profile['guidance']['lock_range_m'])
    if width=='off_axis_half_power':
        radar.receiver_antenna=(radar.receiver_antenna[0]*2,radar.receiver_antenna[1]);radar.transmitter_antenna=(radar.transmitter_antenna[0]*2,radar.transmitter_antenna[1])
    clock=RetryClock();trace=[];accepted=[];search_captures=[];ever=False;previous_state=None
    for row in frames:
        t=int(row['tick']);age=row['age'];own=SimpleNamespace(position=tuple(row['position']),velocity=tuple(row['velocity']),orientation_quaternion=tuple(row['quaternion']))
        target=TargetState(tuple(row['target_position']),tuple(row['target_velocity']))
        # After a simulated accepted observation use that observer's own cache,
        # never a saved TRK Ps/Vs. Prior handoff is explicit and common to policies.
        if not ever:
            estimate=tuple(row['prior_position']);source='available_pre_capture_INS'
        else:
            elapsed=age-radar._last_detection_time_s
            estimate=tuple(p+v*elapsed for p,v in zip(radar._last_position,radar._last_velocity));source='own_accepted_track_cache'
        radar.search_estimate_position=estimate;radar.require_search_estimate=True
        prior_range=float(np.linalg.norm(np.array(estimate)-np.array(own.position)))
        enabled=age>=radar.startup_delay_s-1e-12;ready=prior_range<=radar.lock_range_m
        actual={}
        def evaluate(searching):
            radar.beam_audit={}
            detection,solution=radar.update(age,H,own,target,target_rcs_m2=rcs)
            actual.update(asdict(detection));actual['beam']=dict(radar.beam_audit)
            actual['searching']=searching;actual['gate_prior_source']=source
            if actual['beam']:
                actual['beam']['margin_db']=40*math.log10(actual['beam']['detection_range_m']/detection.range_m)
            return detection.valid
        if policy=='retry':
            d=clock.step(t,enabled=enabled,search_ready=ready,evaluate=evaluate);state=d.state;was_search=d.searching;attempted=d.attempted;valid=d.accepted;reason=d.reason
        else:
            was_search=not radar._within_track_hold(age)
            attempted=True;valid=evaluate(was_search)
            state='TRACK' if valid else ('HOLD' if radar._within_track_hold(age) else 'RETRY')
            reason='accepted' if valid else actual['reason']
        if valid:
            ever=True;accepted.append(t)
            if was_search:search_captures.append(t)
        r=dict(tick=t,age=age,state=state,state_change=state!=previous_state,attempted=attempted,searching=was_search,accepted=valid,reason=reason,prior_range_m=prior_range,prior_source=source,prior_source_tick=row['prior_source_tick'],gate=actual)
        trace.append(r);previous_state=state
    attempts=[r for r in trace if r['attempted'] and r['searching']]
    changes=[{k:r[k] for k in ('tick','age','state','reason')} for r in trace if r['state_change']]
    summary=dict(policy=policy,rcs_m2=rcs,width=width,first_capture_tick=accepted[0] if accepted else None,search_capture_ticks=search_captures,accepted_observations=len(accepted),first_attempt=attempts[0] if attempts else None,search_attempts=attempts,state_changes=changes)
    return summary,trace

def phases(events):
    result={}
    for name,ee in [('first',[e for e in events if e['original_capture']]),('reacquisition',[e for e in events if not e['original_capture']])]:
        vals=[]
        for phase in range(48):
            formula=sum((e['tick1']-phase)//48-(e['tick0']-phase)//48>0 for e in ee)
            direct=sum(any(t%48==phase for t in range(e['tick0']+1,e['tick1']+1)) for e in ee);assert formula==direct;vals.append(formula)
        result[name]=dict(n=len(ee),phase1_compatible=vals[1],all_phases=vals,not_phase1=len(ee)-vals[1])
    return result

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--input',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);args=ap.parse_args();args.out.mkdir(parents=True,exist_ok=False)
    files=read(args.input/'SOURCE_MANIFEST.json')
    for item in files:
        p=args.input/item['path'];assert p.stat().st_size==item['bytes'] and hashlib.sha256(p.read_bytes()).hexdigest()==item['sha256']
    histories=read(args.input/'anchor_full_histories.json.gz');infos=read(args.input/'infos.json');events=read(args.input/'capture_events.json.gz')
    profile=read(ROOT/'data/profile_sets/2.59.0.27/missiles/su_r_77_1.json')
    output=dict(scope='Fixed replay own motion and independent aircraft position histories; no solver/Q or autonomous data-link reconstruction; no current seeker direction/rate/pointing input. Captures come from attempted full old detector, labels used only after simulation.',phase_audit=phases(events),conditions=[],sources=files)
    for uid,(session,eid) in ANCHORS.items():
        info=next(i for i in infos if i['replay']==session and i['eid']==eid)
        unit=next(u for u in read(args.input/session/'unit_history.json.gz')['units'] if u['eid']==info['target_eid'])
        labels=[e for e in events if e['replay']==session and e['eid']==eid];first=next(e for e in labels if e['original_capture'])
        for timing in ('packet','state'):
            frames,meta=build_frames(histories[uid],info,unit,timing);save(args.out/f'frames_{uid}_{timing}.json.gz',frames)
            for width in ('legacy_full_width','off_axis_half_power'):
                for rcs in (1.,2.):
                    for policy in ('continuous','retry'):
                        summary,trace=simulate(frames,profile,rcs,width,policy)
                        tick=summary['first_capture_tick'];summary.update(uid=int(uid),session=session,target_clock=timing,inputs=meta,first_interval_ticks=[first['tick0'],first['tick1']],within_first_interval=tick is not None and first['tick0']<tick<=first['tick1'],first_capture_age=(tick-info['epoch_tick'])/48 if tick is not None else None)
                        prefix=f'{uid}_{timing}_{width}_{rcs:g}_{policy}';save(args.out/(prefix+'.json.gz'),trace);output['conditions'].append(summary)
                        print('RESULT',json.dumps({k:summary[k] for k in ('uid','target_clock','width','rcs_m2','policy','first_capture_tick','first_capture_age','within_first_interval','search_capture_ticks','accepted_observations')}),flush=True)
                        if timing=='packet' and width=='off_axis_half_power' and rcs==2 and policy=='retry':print('ATTEMPTS',uid,json.dumps(summary['search_attempts']),flush=True);print('CHANGES',uid,json.dumps(summary['state_changes']),flush=True)
    output['verification']=dict(manifest_files=len(files),phase_counts_checked=96,frame_inputs_do_not_contain=['direction','rate','n','w','TRK_time','fresh'])
    save(args.out/'summary.json',output);print('PHASE_AUDIT',json.dumps(output['phase_audit']),flush=True)
if __name__=='__main__':main()
