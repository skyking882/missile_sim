#!/usr/bin/env python3
"""Deterministic game-replay INS/TRK provenance diagnostics; no coefficient fitting."""
import argparse, collections, copy, gzip, json, math, sys
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"capture_held_20261017"))
sys.path.insert(0,str(ROOT/"ins_trk_three_models_20261018"))
import audit as A
import range_check as R

H=1/48
TAU=.078525
ALPHA=.85
K=.8
DEG=180/math.pi
def vec(a):return np.asarray(a,float)
def arr(x):
    d=x.copy()
    for k in ("n","w","p","vm","Ps","Vs","Q","angles"):
        if k in d:d[k]=vec(d[k])
    return d
def geo(x):
    u=x["Ps"]-x["p"];d=u@u
    return np.cross(x["Vs"]-x["vm"],u)/d if d>1e-6 else None
def angle(a,b):return A.angle(A.unit(a),A.unit(b))*DEG
def err(x,y):return float(np.linalg.norm(x-y))*DEG
def point(t,a,b):
    f=(t-a["tick"])/(b["tick"]-a["tick"])
    own=a["p"]*(1-f)+b["p"]*f
    target=a["Ps"]+a["Vs"]*((t-a["tick"])*H)
    return A.unit(target-own)
def q_at(t,a,b):
    f=(t-a["tick"])/(b["tick"]-a["tick"])
    r=Rotation.from_matrix(b["Q"]@a["Q"].T).as_rotvec()
    return Rotation.from_rotvec(f*r).as_matrix()@a["Q"]
def stage_step(n,w,m,dt=H):
    p=A.rotate(n,w,dt)
    nn=A.blend(p,m,ALPHA)
    return nn, w+A.correction(p,nn)/(TAU+dt/2)
def reencode_rate(w):
    return np.trunc(vec(w)*32767/15).astype(int)
def clock_phase(oldtick):
    return ((oldtick-1)//48+1)*48+1
def one_event(e):
    a,b=arr(e["old"]),arr(e["new"])
    T=(b["tick"]-a["tick"])*H
    phase=clock_phase(a["tick"])
    eligible=a["tick"]<phase<=b["tick"]
    scheduled=bool(eligible and not e["original_capture"])
    # Never assign an exact capture tick to an unscheduled first-capture event.
    # Its saved tick is a negative-control hypothetical.
    cap=phase if eligible else b["tick"]
    z=point(cap,a,b)
    s=q_at(cap,a,b)@a["w"]
    wcap=R.D(z,K)@s
    # Same seed: compare no further correction with ordinary updates to saved state.
    no_n=A.rotate(z,wcap,(b["tick"]-cap)*H)
    n,w=z,wcap.copy()
    for tick in range(cap+1,b["tick"]+1):
        m=point(tick,a,b)
        n,w=stage_step(n,w,m)
    old_geo=geo(a)
    old_res=float(np.linalg.norm(a["w"]-old_geo))*DEG if old_geo is not None else None
    # Output-conditioned inversion is diagnostic ONLY: it reads saved n and w.
    inferred_seed=b["Q"].T@np.linalg.solve(R.D(b["n"],K),b["w"])
    diff=inferred_seed-a["w"]
    # Conditional rate innovation: uses endpoint n as label, never as independent measurement.
    ni=A.rotate(z,wcap,(b["tick"]-cap)*H)
    c=A.correction(ni,b["n"])
    w_cond=wcap+c/(TAU+max(H,(b["tick"]-cap)*H)/2)
    # Midpoint "changed seed" control using old PV geometric derivative, no true labels.
    gf=geo(dict(Ps=a["Ps"]+a["Vs"]*((cap-a["tick"])*H),
                Vs=a["Vs"],p=a["p"]+(b["p"]-a["p"])*((cap-a["tick"])/(b["tick"]-a["tick"])),
                vm=a["vm"]+(b["vm"]-a["vm"])*((cap-a["tick"])/(b["tick"]-a["tick"]))))
    seed_geo=a["w"]+(gf-old_geo) if gf is not None and old_geo is not None else a["w"]
    zq=q_at(cap,a,b)@seed_geo
    w_geo=R.D(z,K)@zq
    ng,wg=z,w_geo.copy()
    for tick in range(cap+1,b["tick"]+1):
        ng,wg=stage_step(ng,wg,point(tick,a,b))
    return dict(replay=e["replay"],uid=e["uid"],eid=e["eid"],family=e["family"],
        original=e["original_capture"],tick0=a["tick"],tick1=b["tick"],
        gap=b["tick"]-a["tick"],phase=phase,phase_eligible=eligible,
        scheduled_reacq=scheduled,post_tick_count=b["tick"]-cap,
        old_timer=a["timer"],new_timer=b["timer"],flags=[a["flag"],b["flag"]],
        old_rate_from_PV_residual_deg_s=old_res,
        old_rate_norm_deg_s=float(np.linalg.norm(a["w"]))*DEG,
        implied_seed_diff_deg_s=float(np.linalg.norm(diff))*DEG,
        implied_seed_change_parallel_old=float(diff@a["w"]/max(1e-10,float(a["w"]@a["w"]))),
        implied_seed_change_orthogonal_deg_s=float(np.linalg.norm(diff-(diff@a["w"])*a["w"]/max(1e-10,float(a["w"]@a["w"]))))*DEG,
        implied_seed_normalized=diff,
        new_ray_vs_oldPV_deg=angle(point(b["tick"],a,b),b["n"]),
        capture_seed_rate_old=err(wcap,b["w"]),
        capture_seed_direction_deg=angle(no_n,b["n"]),
        ordinary_updates_rate=err(w,b["w"]),
        ordinary_updates_direction_deg=angle(n,b["n"]),
        ordinary_updates_from_geometry_seed_rate=err(wg,b["w"]),
        ordinary_updates_from_geometry_seed_direction_deg=angle(ng,b["n"]),
        current_n_conditioned_endpoint_rate=err(w_cond,b["w"]),
        candidate_seed=wcap, candidate_n=z, model_rate=w, model_n=n,
        old_rate=a["w"],target_rate=b["w"],
        source_old_sha=a["raw_sha"],source_new_sha=b["raw_sha"])
def metric(rr,k):
    z=np.asarray([r[k] for r in rr if r.get(k) is not None],float)
    if not len(z):return {"n":0}
    return dict(n=len(z),rms=float(np.sqrt(np.mean(z*z))),median=float(np.median(z)),
                p95=float(np.percentile(z,95)),max=float(np.max(z)))
def group(rr):
    fields=["old_rate_from_PV_residual_deg_s","implied_seed_diff_deg_s",
            "implied_seed_change_orthogonal_deg_s","new_ray_vs_oldPV_deg",
            "capture_seed_rate_old","ordinary_updates_rate",
            "ordinary_updates_from_geometry_seed_rate","current_n_conditioned_endpoint_rate",
            "capture_seed_direction_deg","ordinary_updates_direction_deg"]
    return {"n":len(rr),"metrics":{k:metric(rr,k) for k in fields}}
def anchor_analysis(root,caps):
    p=list(root.rglob("anchor_full_histories.json.gz"))
    if not p:return {"status":"no_full_histories"}
    histories=R.read(p[0]);out={}
    for uid in ("118","148"):
        series=[arr(x) for x in histories.get(uid,[])]
        event=[e for e in caps if str(e["uid"])==uid and e["original_capture"]]
        res=[]
        for e in event:
            tick=e["tick1"]
            j=next((i for i,x in enumerate(series) if x["tick"]==tick and x["code"]==31),None)
            if j is None:continue
            window=[]
            for i in range(max(0,j-5),min(len(series),j+7)):
                x=series[i];gf=geo(x); v=err(x["w"],gf) if gf is not None else None
                rec=dict(tick=x["tick"],mode=x["code"],timer=x["timer"],
                         wgeo_diff_deg_s=v,rate=vec(x["w"])*DEG,
                         axial_deg_s=float(x["w"]@x["n"])*DEG)
                if i:
                    prev=series[i-1];d=x["tick"]-prev["tick"]
                    if 0<d<=8:
                        rec["rate_delta_deg_s"]=err(x["w"],prev["w"])
                        rec["expected_held_rotation_error_deg"]=angle(A.rotate(prev["n"],prev["w"],d*H),x["n"])
                        rec["saved_n_jump_deg"]=angle(prev["n"],x["n"])
                        rec["n_direction_innovation_deg"]=angle(A.rotate(prev["n"],prev["w"],d*H),x["n"])
                        rec["mode_transition"]=prev["code"]!=x["code"]
                window.append(rec)
            res.append(dict(capture_tick=tick,replay=e["replay"],window=window))
        out[uid]=res
    return out
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--input",type=Path,required=True)
    ap.add_argument("--output",type=Path,required=True)
    z=ap.parse_args()
    z.output.mkdir(parents=True,exist_ok=False)
    mcount=R.verify(z.input)
    path=z.input/"prior/evidence/data/main/evidence/capture_events.json.gz"
    caps=R.read(path)
    assert len(caps)==931
    events=[];guards=0
    for e in caps:
        v=one_event(e)
        # The past-only predictor should not read current target response fields.
        bad=copy.deepcopy(e)
        for name in ("Ps","Vs","n","w"):bad["new"][name]=[float("nan")]*3
        # The whole function contains labelled inversions; check past-only submodel separately.
        a,b=arr(e["old"]),arr(e["new"]);bb=arr(bad["new"]);cap=v["phase"] if v["phase_eligible"] else b["tick"]
        for d in (b,bb):
            p=point(cap,a,d);ss=q_at(cap,a,d)@a["w"];ww=R.D(p,K)@ss
            if d is b:original=(p,ww)
            else:
                for h,k in zip(original,(p,ww)):np.testing.assert_array_equal(h,k)
        guards+=1
        events.append(v)
    near=[r for r in events if R.near(next(e for e in caps if e["uid"]==r["uid"] and e["eid"]==r["eid"] and e["tick1"]==r["tick1"] and e["replay"]==r["replay"]))]
    assert len(near)==244 and sum(r["family"]=="R-77-1" for r in near)==79
    selection={
      "all931":events,"near244":near,
      "near_R77_79":[r for r in near if r["family"]=="R-77-1"],
      "phase_scheduled_reacq":[r for r in events if r["scheduled_reacq"]],
      "phase_scheduled_reacq_late":[r for r in events if r["scheduled_reacq"] and r["post_tick_count"]>0],
      "phase_scheduled_reacq_zero":[r for r in events if r["scheduled_reacq"] and r["post_tick_count"]==0],
      "first_all":[r for r in events if r["original"]],
      "R77_all":[r for r in events if r["family"]=="R-77-1"],
      "old_PV_geo_consistent":[r for r in events if r["old_rate_from_PV_residual_deg_s"] is not None and r["old_rate_from_PV_residual_deg_s"]<.1],
      "old_PV_geo_inconsistent":[r for r in events if r["old_rate_from_PV_residual_deg_s"] is not None and r["old_rate_from_PV_residual_deg_s"]>=.1],
    }
    for lag in range(4):
        selection["reacq_phase_lag_"+str(lag)]=[r for r in events if r["scheduled_reacq"] and r["post_tick_count"]==lag]
    six=[r for r in events if r["uid"] in (94,102,118,129,123,112) and r["tick0"] in (11279,11423,11519,12335,16991,9695)]
    summary={"scope":"No new fitted parameters; past-only conditional prediction with known own motion; inverse seed is output-conditioned diagnostic",
             "frozen":{"k":K,"alpha_h":ALPHA,"tau":TAU,"h":H},
             "events":len(events),"input_hash_verified":mcount,"poison_guards":guards,
             "groups":{name:group(rr) for name,rr in selection.items()},
             "six_cases":six,
             "anchors":anchor_analysis(z.input,caps)}
    R.save(z.output/"summary.json",summary)
    R.save(z.output/"events.json.gz",events)
    print("AUDIT_COUNTS",json.dumps({key:summary[key] for key in ("events","input_hash_verified","poison_guards")}))
    for key in ("near_R77_79","phase_scheduled_reacq_zero","phase_scheduled_reacq_late","old_PV_geo_consistent","old_PV_geo_inconsistent"):
        print("AUDIT_GROUP",key,json.dumps(summary["groups"][key]))
    for r in six:
        print("SIX",json.dumps({k:R.serial(r[k]) for k in ("uid","tick0","tick1","old_timer","old_rate_from_PV_residual_deg_s","implied_seed_diff_deg_s","new_ray_vs_oldPV_deg","capture_seed_rate_old","ordinary_updates_rate","ordinary_updates_from_geometry_seed_rate","ordinary_updates_direction_deg")}))
if __name__=="__main__":main()
