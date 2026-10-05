"""Current horizontal-sinc game controller conditional causal audit.

Predeclared yaw D/AW probes and parent-authorized direct-horizontal-PN/AW
causal followups. No free dynamics, coefficient fit, source runtime input,
new gain/threshold or new PN formula. Source3/5 feedback is diagnostic only.
"""
import copy
import dataclasses
import json
import math
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.dont_write_bytecode=True
sys.path[:0]=[str(ROOT/"scripts"),str(ROOT/"src")]
import numpy as np
import check_r771_horizontal_sinc_cap as current
from aim120_model.control import scheduled_pid_values
from aim120_model.guidance import interpolate_table
from aim120_model.math3d import norm

OUT=ROOT/"outputs/jari_offaxis/horizontal_residual"
DEC=current.dec; AUDIT=DEC.audit; INVERSE=AUDIT.inverse
CASES=current.CASES
KINDS=("current","D_measurement","AW_actual","PN_unslewed","PN_unslewed_AW","source3_feedback","source5_feedback")
FIELDS=("P_old_rad","I_provisional_old_rad","D_old_rad","natural_old_rad","transported_integral_old_rad",
        "effective_old_held_limit_rad","original_AW_limit_old_rad","antiwindup_blocked","next_integral_old_rad","integral_transport_ratio")
WINDOWS=(("first_capture",14/48,.7),("platform",.7,2.3),("unloading",2.3,3.5))


def stats(v,mask):
    v=np.asarray(v)[mask]
    if not len(v):return {"n":0}
    return {"n":len(v),"RMS":np.sqrt(np.mean(v*v,axis=0)).tolist(),"mean":np.mean(v,axis=0).tolist(),
        "min":np.min(v,axis=0).tolist(),"max":np.max(v,axis=0).tolist()}


class Probe(current.HorizontalSincCapControl):
    kind="current"
    feedback=(0.,0.)
    previous_feedback=(0.,0.)
    def update_axis(self,axis,ts,error,memory,pid,limit,gain,k,nonlinear):
        result=super().update_axis(axis,ts,error,memory,pid,limit,gain,k,nonlinear)
        if gain<=1e-12 or axis==0 or self.kind not in ("D_measurement","AW_actual","PN_unslewed_AW"):return result
        d=self.diagnostics[axis]
        derivative=result[1][2]
        if self.kind=="D_measurement":
            raw=-(self.feedback[axis]-self.previous_feedback[axis])*48
            derivative=memory[2]+(1/48)/(1e-9+1/48)*(raw-memory[2])
        derivative_term=pid["d"]*derivative
        natural=d["P_old_rad"]+d["I_provisional_old_rad"]+derivative_term
        post=AUDIT.clip(natural,d["effective_old_held_limit_rad"])
        aw_limit=d["effective_old_held_limit_rad"] if self.kind in ("AW_actual","PN_unslewed_AW") else d["original_AW_limit_old_rad"]
        aw_post=AUDIT.clip(natural,aw_limit)
        blocked=abs(aw_post)>=.99*aw_limit and error*aw_post>0
        next_i=d["transported_integral_old_rad"] if blocked else d["I_provisional_old_rad"]
        result=gain*post,(next_i,error,derivative)
        self.cached[axis]=result
        d.update(D_old_rad=derivative_term,natural_old_rad=natural,next_integral_old_rad=next_i,
            antiwindup_blocked=blocked,held_old_rad=result[0],effective_target_rad=k*result[0],
            normalized_target_at_tick=k*result[0]/limit)
        return result


def analyze(cid,x,iv,coeff):
    cfg=x["config"];st=iv[cid+"__t"];t=np.arange(169)/48
    limits=np.radians([cfg["aerodynamics"][key]for key in ("horizontal_fin_aoa_limit_deg","vertical_fin_aoa_limit_deg")])
    model=current.make_model(cfg,coeff);model.initialize(dataclasses.replace(x["initial"]))
    a={"t":t,"gain":np.array([interpolate_table(ts+1/48,cfg["guidance"]["flight_time_gain_table"])for ts in t])}
    a["halo"]=(t>=14/48-2/24-1e-8)&(t<=14/48+2/24+1e-8)
    for w in (3,5):
        for key,old in (("needed_held","old_held_req_rad"),("source_feedback","source_feedback_pitch_yaw_G"),("source_qdot","omega_dot_rad_s2")):
            a[f"{w}_{key}"]=AUDIT.interp(t,st,iv[cid+f"__{w}__"+old])
    row=[]
    for ts in t:
        s=AUDIT.fix.source_state_at(x["ref"],float(ts));f=model.evaluate(s,float(ts),(0.,0.))
        req,detail=model.request(s,x["target_at"](float(ts)),float(ts),f)
        row.append({"request":req,"feedback":f.feedback_g,"K":model.current_control_k,"beta":math.hypot(*f.local_incidence),
            "nonlinear":model.current_force_nonlinear,"local":f.local_incidence,"source_passive_qdot":f.angular_acceleration,
            "model_lift_acceleration":norm(f.wind_lift_force)/f.mass,"unslewed_request":detail.get("slew_original_request_g",req),
            "loft_G":detail["loft_g"],"horizontal_los_rate":detail.get("horizontal_los_rate_rad_s",0.)})
    a.update({key:np.array([r[key]for r in row])for key in row[0]})
    a["source_export_lift_acceleration"]=np.interp(t,x["ref"]["t"],x["ref"]["acc_lift"])
    controller={kind:Probe()for kind in KINDS};memory={kind:[(0.,0.,0.),(0.,0.,0.)]for kind in KINDS}
    exact=current.HorizontalSincCapControl();exact_mem=[(0.,0.,0.),(0.,0.,0.)]
    for kind in KINDS:
        controller[kind].kind=kind
        for key in ("held",*FIELDS):a[kind+"__"+key]=np.zeros((len(t),2))
        a[kind+"__memory"]=np.zeros((len(t),2,3))
    for w in (3,5):
        a[f"{w}_given_memory_needed_request"]=np.zeros((len(t),2));a[f"{w}_given_memory_status"]=np.zeros((len(t),2),dtype=int)
    max_exact=0.;max_inverse=0.
    for i,ts in enumerate(t):
        if a["gain"][i]<=1e-12:continue
        pid=scheduled_pid_values(cfg["control"]["pid"],float(ts)+1/48);k=float(a["K"][i]);gain=float(a["gain"][i]);nonlin=bool(a["nonlinear"][i])
        prior=copy.deepcopy(memory["current"])
        for kind in KINDS:
            c=controller[kind];c.current_beta_rad=float(a["beta"][i])
            fb=a["feedback"] if not kind.startswith("source") else a[("3" if kind=="source3_feedback" else "5")+"_source_feedback"]
            c.feedback=fb[i];c.previous_feedback=fb[i-1]
            req=a["request"][i].copy()
            if kind in ("PN_unslewed","PN_unslewed_AW"):
                req[1]=a["unslewed_request"][i,1]
                if np.linalg.norm(req)>cfg["guidance"]["maximum_lateral_acceleration_g"]*model.gravity/DEC.G0_MPS2+1e-12:
                    raise ValueError("Unslewed conditional request requires an untested joint-cap interaction")
            for axis in range(2):
                held,memory[kind][axis]=c.update_axis(axis,float(ts),float(req[axis]-fb[i,axis]),memory[kind][axis],pid,float(limits[axis]),gain,k,nonlin)
                a[kind+"__held"][i,axis]=held;a[kind+"__memory"][i,axis]=memory[kind][axis]
                for key in FIELDS:a[kind+"__"+key][i,axis]=c.diagnostics[axis][key]
        exact.current_beta_rad=float(a["beta"][i])
        for axis in range(2):
            output,exact_mem[axis]=exact.update_axis(axis,float(ts),float(a["request"][i,axis]-a["feedback"][i,axis]),exact_mem[axis],pid,float(limits[axis]),gain,k,nonlin)
            max_exact=max(max_exact,abs(output-a["current__held"][i,axis]),max(abs(np.array(exact_mem[axis])-a["current__memory"][i,axis])))
            d=controller["current"].diagnostics[axis];transported=(d["transported_integral_old_rad"],prior[axis][1],prior[axis][2])
            for w in (3,5):
                error,status=AUDIT.error_for_output(a[f"{w}_needed_held"][i,axis],transported,pid,gain,d["effective_old_held_limit_rad"])
                a[f"{w}_given_memory_needed_request"][i,axis]=error+a["feedback"][i,axis];a[f"{w}_given_memory_status"][i,axis]=status
                if status in (1,2):
                    closed=gain*AUDIT.clip(AUDIT.natural_pid(error,transported,pid),d["effective_old_held_limit_rad"])
                    max_inverse=max(max_inverse,abs(closed-a[f"{w}_needed_held"][i,axis]))
    scale=a["K"][:,None]/limits
    for kind in KINDS:
        a[kind+"__normalized"]=scale*a[kind+"__held"]
        for key in ("P_old_rad","I_provisional_old_rad","D_old_rad","natural_old_rad"):
            a[kind+"__norm_"+key]=scale*a[kind+"__"+key]
    for w in (3,5):a[f"{w}_needed_normalized"]=scale*a[f"{w}_needed_held"]
    receipt=json.loads((current.OUT/"full_runs"/(cid+".json")).read_text());tt=np.array([tick["time_s"]for tick in receipt["ticks"]])
    for key,old in (("held","held_fin_rad"),("request","request_g"),("feedback","feedback_g")):
        a["actual_free_"+key]=AUDIT.interp(t,tt,np.array([tick[old]for tick in receipt["ticks"]]))
    a["actual_free_normalized"]=AUDIT.interp(t,tt,np.array([[d["normalized_target_at_tick"]for d in tick["control_runtime_diagnostics"]]if tick["control_runtime_diagnostics"]else [0.,0.]for tick in receipt["ticks"]]))
    report={"axis_order":["pitch","yaw"],"checks":{"current_runtime_exact_max_abs":max_exact,"one_step_inverse_closure_max_abs_rad":max_inverse},"windows":{}}
    active=(a["gain"]>0)&~a["halo"]
    for window,lo,hi in WINDOWS:
        mask=active&(t>=lo-1e-8)&(t<=hi+1e-8);wr={"ticks":int(mask.sum()),"source_interpolated_support_s":[float(t[mask][0]),float(t[mask][-1])],
            "request_G":stats(a["request"],mask),"plant_feedback_G":stats(a["feedback"],mask),
            "lift_magnitude_model_minus_export_mps2":stats(a["model_lift_acceleration"]-a["source_export_lift_acceleration"],mask),
            "current_AW_ticks":np.sum(a["current__antiwindup_blocked"][mask],axis=0).tolist(),
            "current_output_clipped_ticks":np.sum(abs(a["current__natural_old_rad"][mask])>a["current__effective_old_held_limit_rad"][mask],axis=0).tolist(),
            "PID_normalized_terms":{key:stats(a["current__norm_"+key],mask)for key in ("P_old_rad","I_provisional_old_rad","D_old_rad")},"stencils":{}}
        for w in (3,5):
            sw={"feedback_model_minus_source_G":stats(a["feedback"]-a[f"{w}_source_feedback"],mask),"output_error_normalized":{},"given_memory_request_error_G":{},
                "needed_above_output_cap_ticks":np.sum((a[f"{w}_given_memory_status"]==-1)[mask],axis=0).tolist()}
            for kind in KINDS:
                sw["output_error_normalized"][kind]=stats(a[kind+"__normalized"]-a[f"{w}_needed_normalized"],mask)
            for axis in range(2):
                feasible=mask&(a[f"{w}_given_memory_status"][:,axis]==1)
                sw["given_memory_request_error_G"][str(axis)]=stats(a["request"][:,axis]-a[f"{w}_given_memory_needed_request"][:,axis],feasible)
            wr["stencils"][str(w)]=sw
        wr["stencil_difference_needed_normalized"]=stats(a["5_needed_normalized"]-a["3_needed_normalized"],mask)
        report["windows"][window]=wr
    points=[.4166666666666667,.5,.5416666666666666,.6666666666666666,.8333333333333334,1.,1.5,2.,2.5,3.,3.5]
    report["points"]=[{key:a[key][int(round(ts*48))].tolist()for key in ("t","request","feedback","K","beta","5_needed_normalized","3_needed_normalized",
        "current__normalized","actual_free_normalized","current__norm_P_old_rad","current__norm_I_provisional_old_rad","current__norm_D_old_rad",
        "D_measurement__normalized","AW_actual__normalized","PN_unslewed__normalized","PN_unslewed_AW__normalized","source5_feedback__normalized","5_given_memory_needed_request","5_given_memory_status")}for ts in points]
    assert max_exact<1e-12 and max_inverse<1e-12
    return a,report


def saved_free_sequence(cid,cfg,path,end_time=None):
    """Conditional PID replay on saved free signals; this is not a new trajectory."""
    receipt=json.loads(path.read_text())
    ticks=[r for r in receipt["ticks"]if end_time is None or r["time_s"]<=end_time+1e-8]
    controllers={kind:Probe()for kind in ("current","PN_unslewed_AW")}
    memory={kind:[(0.,0.,0.),(0.,0.,0.)]for kind in controllers}
    arrays={"t":np.array([r["time_s"]for r in ticks])}
    for kind,c in controllers.items():
        c.kind=kind
        for key in ("normalized","next_I_target_norm","AW"):
            arrays[kind+"__"+key]=np.zeros((len(ticks),2))
    closure=0.;memory_closure=0.;max_request_delta=0.;oldaw=0;newaw=0;max_natural_cap_ratio=0.
    for i,tick in enumerate(ticks):
        if tick["gain"]<=1e-12:continue
        ds=tick["control_runtime_diagnostics"];ts=tick["time_s"]
        pid=scheduled_pid_values(cfg["control"]["pid"],(round(ts*48)+1)/48)
        unslewed=tick.get("slew_original_request_g",tick["request_g"])
        max_request_delta=max(max_request_delta,abs(tick["request_g"][1]-unslewed[1]))
        for kind,c in controllers.items():
            c.current_beta_rad=ds[0]["beta_rad"]
            for axis in (0,1):
                request=unslewed[axis]if kind=="PN_unslewed_AW" and axis==1 else tick["request_g"][axis]
                d=ds[axis];k=d["K_target_per_old_held"];limit=d["original_old_held_limit_rad"]
                held,memory[kind][axis]=c.update_axis(axis,ts,request-tick["feedback_g"][axis],memory[kind][axis],pid,limit,tick["gain"],k,d["nonlinear_latched"])
                arrays[kind+"__normalized"][i,axis]=k*held/limit
                arrays[kind+"__next_I_target_norm"][i,axis]=k*memory[kind][axis][0]/limit
                arrays[kind+"__AW"][i,axis]=c.diagnostics[axis]["antiwindup_blocked"]
                if kind=="current":
                    closure=max(closure,abs(held-tick["held_fin_rad"][axis]))
                    memory_closure=max(memory_closure,float(np.max(abs(np.array(memory[kind][axis])-tick["pid_memory"][axis]))))
        yaw=ds[1];cap=yaw["effective_old_held_limit_rad"];out=AUDIT.clip(yaw["natural_old_rad"],cap)
        oldaw+=yaw["antiwindup_blocked"];newaw+=abs(out)>=.99*cap and yaw["error_g"]*out>0
        max_natural_cap_ratio=max(max_natural_cap_ratio,abs(yaw["natural_old_rad"])/cap)
    delta=arrays["PN_unslewed_AW__normalized"]-arrays["current__normalized"]
    report={"saved_ticks":len(ticks),"current_receipt_held_closure_rad":closure,"current_receipt_memory_closure":memory_closure,
        "combined_minus_current_max_abs_normalized":np.max(abs(delta),axis=0).tolist(),
        "yaw_unslewed_minus_current_max_abs_G":max_request_delta,"yaw_original_AW_ticks":int(oldaw),
        "yaw_actual_cap_AW_ticks_given_original_memory":int(newaw),"yaw_natural_over_actual_cap_max":max_natural_cap_ratio,
        "windows":{},"points":[]}
    for window,lo,hi in WINDOWS:
        mask=(arrays["t"]>=max(lo,19/48)-1e-8)&(arrays["t"]<=hi+1e-8)
        report["windows"][window]={"output_delta_normalized":stats(delta,mask),
            "I_delta_normalized":stats(arrays["PN_unslewed_AW__next_I_target_norm"]-arrays["current__next_I_target_norm"],mask)}
    for ts in (.4166666666666667,.5,.5416666666666666,.6666666666666666,.8333333333333334,1.,1.5,2.,2.5,3.,3.5):
        i=int(round(ts*48))
        if i<len(ticks):report["points"].append({key:value[i].tolist()for key,value in arrays.items()})
    assert closure<1e-12 and memory_closure<1e-12
    assert np.array_equal(arrays["current__normalized"][:,0],arrays["PN_unslewed_AW__normalized"][:,0])
    return arrays,report


def main():
    inputs=INVERSE.source_inputs();iv=np.load(INVERSE.OUTPUT/"inverse_inputs.npz")
    coeff=AUDIT.drag.DragCoefficients(**json.loads((AUDIT.drag.OUTPUT/"fit.json").read_text())["coefficients"])
    arrays={};reports={};free_reports={};native_reports={}
    for cid in CASES:
        a,r=analyze(cid,inputs[cid],iv,coeff);reports[cid]=r;arrays.update({cid+"__"+key:value for key,value in a.items()})
        a,r=saved_free_sequence(cid,inputs[cid]["config"],current.OUT/"full_runs"/(cid+".json"),3.5)
        free_reports[cid]=r;arrays.update({cid+"__saved_free__"+key:value for key,value in a.items()})
        print(cid,"source and saved-free conditional sequences complete",flush=True)
    harness=AUDIT.fix.finite.harness
    native_inputs={c["id"]:c for c in harness.load_cases(AUDIT.fix.finite.comparison.SNAPSHOT)}
    for cid in current.NATIVE:
        cfg=native_inputs[cid.removeprefix("native_")]["profile"]["_model_config"]
        a,r=saved_free_sequence(cid,cfg,current.OUT/(cid+"_run.json"))
        native_reports[cid]=r;arrays.update({cid+"__saved_free__"+key:value for key,value in a.items()})
    assert all(np.isfinite(v).all()for v in arrays.values())
    OUT.mkdir(parents=True,exist_ok=True);np.savez_compressed(OUT/"conditional.npz",**arrays)
    (OUT/"results.json").write_text(json.dumps({"task":"Current horizontal-sinc conditional residual diagnosis", "cases":reports,
        "saved_free_conditional_sequences":free_reports,"native_saved_free_conditional_sequences":native_reports,
        "predeclared_probes":["yaw D=-Kd*d(feedback)/dt, original filter and all other mappings unchanged","yaw AW uses actual output limit, original .99 rule; all other mappings unchanged"],
        "targeted_followup_probe":"Raw source omega at1/3s is31.44deg/s versus actualfree20.15deg/s while first active output is not cap-bound. Test only removal of extraS110 horizontalrequest ramp with current unslewed PN/caps/PID; reject if added earlyI breaks later source-state agreement. Declared before calculating this third conditional probe.",
        "authorized_shared_cause_check":"Parent explicitly requested direct unslewed horizontalPN plus AW at actual yaw output cap as one coherent PID-interface check after residual-I evidence. No new parameters or gate;110 field meaning remains unidentified.",
        "limits":["Only held inverse is reused from older artifacts; current full-horizontal-PN/controller is recomputed from source states.",
            "24 Hz source state and 3/5-point angular inverse interpolated to controller48Hz; source gain halo excluded, not extra source information.",
            "Source-feedback recursions are diagnostic; no runtime observation feedback or free trajectory is generated.",
            "Given-memory inverse requests are conditional on the selected controller history, not identified native PN requests.",
            "Frozen angular law and unobserved input are jointly nonidentifiable; no passive-law or force coefficient changes considered.",
            "JARI .28 versus local .27 global correspondence remains unverified."],
        "verification":{"all_arrays_finite":True,"new_free_integrations":0,"fits":0,"parameter_changes":0}},indent=2)+"\n")


if __name__=="__main__":main()
