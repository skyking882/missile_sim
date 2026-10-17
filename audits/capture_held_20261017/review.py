#!/usr/bin/env python3
"""Review saved raw-derived predictions; no new fitting or endpoint-based exclusions."""
import argparse,gzip,json,math
from pathlib import Path
import numpy as np
import audit as A

def read(p):
    with (gzip.open(p,'rt') if p.suffix=='.gz' else p.open()) as f:return json.load(f)

def main():
    c=argparse.ArgumentParser();c.add_argument('--results',type=Path,required=True);c.add_argument('--out',type=Path,required=True);c.add_argument('--reference',type=Path,required=True);a=c.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    s=read(a.results/'summary.json');caps=read(a.results/'capture_events.json.gz');held=read(a.results/'held_pairs.json.gz');episodes=read(a.results/'held_episodes.json.gz');returns=read(a.results/'fresh_return_events.json.gz');clips=read(a.results/'clipping_records.json.gz');anchors=read(a.results/'anchor_full_histories.json.gz')
    # Use the actual case prefix, whose first row is last INS, rather than guessing
    # the meaning of metadata.native_ticks[0]. Preserve the original flag as audited data.
    d=read(next(a.reference.rglob('server_dataset.json.gz')))
    original={(r['info']['replay'],int(r['info']['eid']),int(r['ticks'][1])) for r in d['cases']}
    mislabels=0
    for r in caps:
        correct=(r['replay'],r['eid'],r['tick1']) in original
        mislabels+=correct!=r['original_capture'];r['original_capture_verified']=correct
    names=['Qold/reset_only','Qnew/reset_only','keep/reset_only','transport/reset_only','zero/reset_only','past_geometry/reset_only','Qold/self_h','Qold/self_T','Qold/self_ticks','Qnew/self_h','Qold/carry_n','Qold/past_Ps_correction']
    def models(rr):return {name:{key:A.metric([r['predictions'][name][key] for r in rr]) for key in ('n_deg','w_deg_s','lambda_error_deg_s','transverse_deg_s','P_m')} for name in names}
    groups={'all':caps,'original_verified':[r for r in caps if r['original_capture_verified']],'subsequent':[r for r in caps if not r['original_capture_verified']],'R77':[r for r in caps if r['family']=='R-77-1'],'R77_fresh_unlimited':[r for r in caps if r['family']=='R-77-1' and r['fresh'] and not r['limited']]}
    report=dict(counts=s['counts'],decoder_checks=s['decoder_reference_max_abs_difference'],original_capture_metadata_tag_mismatches=mislabels,actual_original_captures=sum(r['original_capture_verified'] for r in caps),parameters=s['parameters'],capture={g:models(rr) for g,rr in groups.items()})
    hr={}
    for name,rr in [('all',held),('R77',[r for r in held if r['family']=='R-77-1']),('flag00',[r for r in held if r['flag0']==r['flag1']==0]),('flag11',[r for r in held if r['flag0']==r['flag1']==1]),('flag_change',[r for r in held if r['flag0']!=r['flag1']])]:
        hr[name]=dict(n=len(rr),cache_equal=sum(r['cache_equal'] for r in rr),rate_equal=sum(r['rate_equal'] for r in rr),d_equal=sum(r['d_equal'] for r in rr),metrics={k:A.metric([r[k] for r in rr]) for k in ('rotate_deg','frozen_world_deg','frozen_body_deg','PV_deg','rate_error_deg_s','R_frozen_error_m','R_CV_error_m')})
    report['held_pairs']=hr
    er={}
    for flag in (0,1):
        rr=[r for r in episodes if r['flag']==flag]
        alln=[x for r in rr for x in r['n_error_deg']];allw=[x for r in rr for x in r['w_error_deg_s']]
        er[str(flag)]=dict(episodes=len(rr),steps=len(alln),n_deg=A.metric(alln),w_deg_s=A.metric(allw),P_cache_m=A.metric([x for r in rr for x in r['P_cache_error_m']]),V_cache_mps=A.metric([x for r in rr for x in r['V_cache_error_mps']]),max_duration_s=max([r['duration_s'] for r in rr],default=0))
    report['held_episodes']=er
    report['anchors_capture']=[{k:r[k] for k in ('replay','eid','uid','age0','age1','gap','timer0','timer1','flag0','flag1','old','new','predictions')} for r in caps if (r['replay'],r['eid']) in A.ANCHORS]
    report['anchors_held']=[r for r in episodes if (r['replay'],r['eid']) in A.ANCHORS]
    report['anchors_return']=[r for r in returns if (r['replay'],r['eid']) in A.ANCHORS]
    report['return_subgroups']={}
    for name,rr in [('all',returns),('R77',[r for r in returns if r['family']=='R-77-1']),('R77_unlimited_flag00',[r for r in returns if r['family']=='R-77-1' and not r['limited'] and r['flag0']==r['flag1']==0]),('R77_flagchange',[r for r in returns if r['family']=='R-77-1' and r['flag0']!=r['flag1']])]:
        keys=list(rr[0]['predictions']) if rr else []
        report['return_subgroups'][name]={key:{x:A.metric([r['predictions'][key][x] for r in rr]) for x in ('n_deg','w_deg_s','lambda_error_deg_s','P_m')} for key in keys}
    cg={}
    for code in (25,29,31):
        for fresh in (True,False):
            rr=[r for r in clips if r['code']==code and r['fresh']==fresh];sat=[r for r in rr if r['limited']]
            cg[f'code{code}_fresh{fresh}']=dict(n=len(rr),limited=len(sat),limited_and_n_beyond55p1=sum(r['excess_deg']>.1 for r in sat),limited_and_n_beyond56=sum(r['excess_deg']>1 for r in sat),exact_angle_error_deg=A.metric([r['ray_error_deg'] for r in sat]),clipped_angle_error_deg=A.metric([r['clipped_ray_error_deg'] for r in sat]))
    report['clipping']=cg;report['clipping_prediction']=s['clipped_prediction']
    report['failures_kept']=dict(worst_capture=[{k:r[k] for k in ('replay','uid','age0','age1','gap','flag0','flag1','fresh','limited')}|{'Qold_self_h':r['predictions']['Qold/self_h']} for r in sorted(caps,key=lambda r:r['predictions']['Qold/self_h']['w_deg_s'],reverse=True)[:20]],worst_held=[{k:r[k] for k in ('replay','uid','start_tick','last_tick','flag','duration_s','steps','metrics')} for r in sorted(episodes,key=lambda r:r['metrics']['n_error_deg']['max'],reverse=True)[:20]])
    A.save(a.out/'review.json',report);A.save(a.out/'capture_events_verified.json.gz',caps)
    # Compact printout with all anchor first steps; full vectors remain in review.json.
    compact={k:report[k] for k in ('counts','decoder_checks','original_capture_metadata_tag_mismatches','actual_original_captures','held_pairs','held_episodes','return_subgroups','clipping','clipping_prediction')}
    compact['capture']={g:{name:[v['w_deg_s']['rms'],v['n_deg']['rms'],v['lambda_error_deg_s']['rms'],v['P_m']['rms']] for name,v in vals.items() if v['w_deg_s']['n']} for g,vals in report['capture'].items()}
    compact['anchors_capture']=[{k:r[k] for k in ('uid','age0','age1','gap','timer1','flag1')}|{'models':{key:[r['predictions'][key][x] for x in ('w_deg_s','n_deg','lambda_error_deg_s','P_m')] for key in names}} for r in report['anchors_capture']]
    compact['anchors_held']=[{k:r[k] for k in ('uid','start_tick','last_tick','steps','duration_s','flag','metrics')} for r in report['anchors_held']]
    A.save(a.out/'review_compact.json',compact)
    for k,v in compact.items():print('REVIEW',k,json.dumps(A.serial(v),ensure_ascii=False),flush=True)

if __name__=='__main__':main()
