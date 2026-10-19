#!/usr/bin/env python3
"""Final source-adjacent positive controls and moving-LOS memory gates."""
import argparse,collections,math
from pathlib import Path
import analyze as A
import review as V

def main():
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--main',type=Path,required=True);p.add_argument('--review',type=Path,required=True);p.add_argument('--adjacency',type=Path,required=True);p.add_argument('--out',type=Path,required=True);z=p.parse_args();z.out.mkdir(parents=True,exist_ok=False)
    fresh=A.read(z.input/'fresh.json.gz');lookup=collections.defaultdict(list)
    for x in fresh:lookup[x['replay'],x['eid'],x['record']['tick']].append(x['record'])
    adj=A.read(z.adjacency/'adjacency.json.gz');ad={(r['replay'],r['index']):r for r in adj}
    old=A.read(z.review/'fresh_pairs.json.gz');valid=[];excluded=collections.Counter()
    for r in old:
        key=r['replay'],r['eid'];ss=lookup[*key,r['tick0']];nn=lookup[*key,r['tick1']]
        if len(ss)!=1 or len(nn)!=1:excluded['ambiguous_tick']+=1;continue
        s,n=ss[0],nn[0]
        if ad[r['replay'],n['index']]['previous_same_eid_index']!=s['index']:
            excluded['intervening_raw_state']+=1;continue
        assert ad[r['replay'],n['index']]['raw_sha']==r['raw_sha']
        c=V.independent_b(s,n);assert abs(c-r['predicted_closing'])<1e-9
        valid.append(r|dict(index0=s['index'],index1=n['index']))
    positive={}
    for clock in ('packet','state'):
        rs=[r for r in valid if r['clock']==clock]
        positive[clock]=dict(n=len(rs),entities=len({(r['replay'],r['eid']) for r in rs}),rejects={str(w):sum(abs(r['innovation_mps'])>w for r in rs) for w in (10,20,300)},absolute_error=V.metric_abs([r['innovation_mps'] for r in rs]),rejected20=[r for r in rs if abs(r['innovation_mps'])>20])
    eps=A.read(z.input/'episodes.json.gz');base=A.read(z.main/'episodes.json.gz');by={(x['clock'],x['episode_id']):x for x in base};predrows=[]
    for e in eps:
        if not e['strict']:continue
        s=e['seed'];past=s;accum=0.
        seq=[('held',r) for r in e['held']]+([] if e['exit'] is None or e['exit_reason']!='fresh_return' else [('return',e['exit'])])
        for role,r in seq:
            T=(r['tick']-s['tick'])/48;oldT=(past['tick']-s['tick'])/48
            pn=A.propagate(s['n'],s['w'],oldT);accum+=V.dot(pn,A.vec(r['vm'])-A.vec(past['vm']))
            pred=s['closing']+s['last']*T+accum
            for clock in ('packet','state'):
                context=by.get((clock,e['episode_id']))
                if context is None:continue
                d=next((d for d in context['rows'] if d['index']==r['index']),None)
                if d is None:continue
                predrows.append(dict(clock=clock,episode_id=e['episode_id'],replay=e['replay'],eid=e['eid'],uid=e['uid'],seed_tick=s['tick'],seed_last=s['last'],tick=r['tick'],role=role,T=T,nearby=context['nearby_seed_reference'],exit_reason=e['exit_reason'],flag=r['flag'],current_ps_error_m=d['current_ps_error_m'],common=d['common_candidate_eligible'],margin_db=d['signal_limits']['15.0']['margin_ref2_db'],predicted_closing=pred,recorded_closing=r['closing'],true_closing=-d['radial_speed_mps'],innovation=pred+d['radial_speed_mps'],closing_error=pred-r['closing'],raw_sha=r['raw_sha']))
            past=r
    gate_results={};examples=[]
    for clock in ('packet','state'):
        rs=[r for r in predrows if r['clock']==clock and r['nearby']]
        held=[r for r in rs if r['role']=='held' and r['T']<A.EARLY and r['common']];returns=[r for r in rs if r['role']=='return' and r['T']<A.EARLY and r['flag']==0]
        gate_results[clock]={}
        for width in (10,20,300):
            gate_results[clock][str(width)]=dict(eligible_held=len(held),held_rejected=sum(abs(r['innovation'])>width for r in held),held_still_pass=sum(abs(r['innovation'])<=width for r in held),early_returns=len(returns),returns_rejected=sum(abs(r['innovation'])>width for r in returns))
        if clock=='packet':
            for eid in sorted({r['episode_id'] for r in held}):
                hr=[r for r in held if r['episode_id']==eid];rejected=[r for r in hr if abs(r['innovation'])>20]
                if rejected:examples.append(dict(episode_id=eid,uid=hr[0]['uid'],eid=hr[0]['eid'],replay=hr[0]['replay'],seed_tick=hr[0]['seed_tick'],seed_slope=hr[0]['seed_last'],eligible_n=len(hr),rejected20=len(rejected),first_rejected20=rejected[0],last_rejected20=rejected[-1],max_innovation=max(abs(r['innovation']) for r in hr),early_return=next((r for r in returns if r['episode_id']==eid),None)))
    anchor=[r for r in predrows if r['clock']=='packet' and r['replay']=='2026.10.08_11.49.38' and r['eid']==2439170 and r['seed_tick']==11524 and r['role']=='held']
    ah=[r for r in anchor if r['common']]
    out=dict(scope='Necessary-gate tests with a retained scalar initialized once from game lastfresh; independent fixed-aircraft motion is an explicit offline diagnostic. Not autonomous reconstruction or proof that closing is the Doppler gate center.',fresh_controls_raw_adjacent=positive,excluded_from_previous_fresh_control=dict(excluded),moving_LOS_gates=gate_results,moving_LOS_examples=examples,anchor118=anchor,anchor118_counts=dict(held=len(anchor),signal_eligible=len(ah),remaining={str(w):sum(abs(r['innovation'])<=w for r in ah) for w in (10,20,300)},scalar_reconstruction=V.metric_abs([r['closing_error'] for r in anchor])),verification=dict(native_adjacency_identifiers=len(adj),fresh_pair_comparisons=len(old),strict_raw_adjacent_pairs=len(valid),same_forward_rules_both_clocks=True))
    A.save(z.out/'summary.json',out);A.save(z.out/'raw_adjacent_fresh_pairs.json.gz',valid);A.save(z.out/'moving_los_predictions.json.gz',predrows)
    print('STRICT_POSITIVES',A.json.dumps(A.clean(positive)),flush=True)
    print('MOVING_LOS_GATES',A.json.dumps(gate_results),flush=True);print('ANCHOR_FINAL',A.json.dumps(out['anchor118_counts']),flush=True)
    for r in anchor:
        if r['tick'] in (11526,11532,11571):print('ANCHOR_MOVING',A.json.dumps(r),flush=True)
    for r in examples:print('EXAMPLE_MOVING',A.json.dumps(r),flush=True)
    print('FINAL_CHECKS',A.json.dumps(out['verification']|dict(excluded=dict(excluded))),flush=True)
if __name__=='__main__':main()
