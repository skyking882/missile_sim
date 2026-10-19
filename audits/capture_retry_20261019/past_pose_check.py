#!/usr/bin/env python3
"""Sensitivity to offline pose interpolation; not an extra fitted model.
Use strictly earlier own-position/attitude samples after the first available row.
Target trajectory is still an offline, independently recorded input.
"""
import argparse,bisect,copy,json,math
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
import replay_check as R

def past_pose(frames,history):
    ticks=[r['tick'] for r in history];out=[];errors=[]
    for frame in frames:
        tick=frame['tick'];j=bisect.bisect_left(ticks,tick)-1
        if j<0:j=0
        old=history[j];elapsed=(tick-old['tick'])/48
        q=np.array(old['Q']);p=np.array(old['p'])+np.array(old['vm'])*elapsed
        if j>0:
            prev=history[j-1];step=(old['tick']-prev['tick'])/48
            w=Rotation.from_matrix(q@np.array(prev['Q']).T).as_rotvec()/step
            q=Rotation.from_rotvec(w*elapsed).as_matrix()@q
        new=copy.deepcopy(frame);quat=Rotation.from_matrix(R.C@q@R.C).as_quat()
        new['position']=(R.C@p).tolist();new['velocity']=(R.C@np.array(old['vm'])).tolist();new['quaternion']=[quat[3],quat[0],quat[1],quat[2]]
        new['own_bracket_ticks']=[old['tick'],old['tick']];new['own_extrapolation_s']=elapsed
        errors.append(dict(tick=tick,position_delta_m=float(np.linalg.norm(np.array(new['position'])-frame['position'])),source_tick=old['tick'],attitude_delta_deg=float((Rotation.from_quat(quat)*Rotation.from_quat([frame['quaternion'][1],frame['quaternion'][2],frame['quaternion'][3],frame['quaternion'][0]]).inv()).magnitude()*180/math.pi)))
        assert old['tick']<tick or tick==frames[0]['tick']
        out.append(new)
    return out,errors

def review_clock(trace):
    last=None;attempted_first=False;count=0
    for r in trace:
        tick=r['tick'];holding=last is not None and tick-last<=48
        if r['attempted']:
            if r['searching']:
                assert not holding
                if attempted_first:assert tick%48==1
                else:attempted_first=True
            else:assert holding
            count+=1
        if r['accepted']:
            assert r['attempted'];last=tick
    return count

def main():
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--baseline',type=Path,required=True);p.add_argument('--out',type=Path,required=True);args=p.parse_args();args.out.mkdir(parents=True,exist_ok=False)
    histories=R.read(args.input/'anchor_full_histories.json.gz');base=R.read(args.baseline/'summary.json');profile=R.read(R.ROOT/'data/profile_sets/2.59.0.27/missiles/su_r_77_1.json')
    results=[];clockchecks=0
    for condition in base['conditions']:
        uid=str(condition['uid']);timing=condition['target_clock'];rcs=condition['rcs_m2'];width=condition['width'];policy=condition['policy']
        frames=R.read(args.baseline/f'frames_{uid}_{timing}.json.gz');new,errors=past_pose(frames,histories[uid]);s,trace=R.simulate(new,profile,rcs,width,policy)
        if policy=='retry':clockchecks+=review_clock(trace)
        first=s['first_capture_tick'];lo,hi=condition['first_interval_ticks']
        r=dict(uid=int(uid),target_clock=timing,rcs_m2=rcs,width=width,policy=policy,baseline_first_tick=condition['first_capture_tick'],past_pose_first_tick=first,first_tick_unchanged=first==condition['first_capture_tick'],past_pose_inside_interval=first is not None and lo<first<=hi,baseline_search_captures=condition['search_capture_ticks'],past_pose_search_captures=s['search_capture_ticks'],first_attempt=s['first_attempt'],maximum_pose_position_difference_m=max(x['position_delta_m'] for x in errors),maximum_pose_attitude_difference_deg=max(x['attitude_delta_deg'] for x in errors))
        results.append(r);R.save(args.out/f'{uid}_{timing}_{width}_{rcs:g}_{policy}.json.gz',trace)
        print('PAST_POSE',json.dumps({k:v for k,v in r.items() if k!='first_attempt'}),flush=True)
    # Poison every input field the simulator is not supposed to read. This does
    # not erase the explicit offline prior-series selection used in build_frames.
    poisoned_checks=0
    for uid in ('118','148'):
        frames=R.read(args.baseline/f'frames_{uid}_packet.json.gz');bad=copy.deepcopy(frames)
        for f in bad:f.update(direction=[float('nan')]*3,rate=[float('nan')]*3,Ps=[float('nan')]*3,Vs=[float('nan')]*3,TRK_time=float('nan'),fresh='poison')
        a,ta=R.simulate(frames,profile,2.,'off_axis_half_power','retry');b,tb=R.simulate(bad,profile,2.,'off_axis_half_power','retry')
        assert a==b and ta==tb;poisoned_checks+=len(frames)
    out=dict(scope='Own-state interpolation sensitivity. Past-only own pose; independent target positions and packet/state query mapping still offline. Same prerecorded INS priors, not an autonomous observer.',conditions=results,independent_attempt_clock_checks=clockchecks,unused_response_field_poison_checks=poisoned_checks,first_tick_unchanged=sum(r['first_tick_unchanged'] for r in results),total_conditions=len(results))
    R.save(args.out/'summary.json',out);print('SENSITIVITY_SUMMARY',json.dumps({k:v for k,v in out.items() if k!='conditions'}),flush=True)
if __name__=='__main__':main()
