#!/usr/bin/env python3
"""One baseIndSpeed inverse-q output-placement test for the War Thunder game.

Conditional source3/5 and saved-native controls only. Frozen Ku/plant and raw
PID remain unchanged; no field-unit identification or free-pass claim.
"""
from __future__ import annotations
import dataclasses
import inspect
import json
import math
import os
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/"scripts"),str(ROOT/"src")]
sys.dont_write_bytecode=True
os.environ.setdefault("MPLCONFIGDIR","/tmp/missle_sim_mpl")
import numpy as np
import analyze_r771_horizontal_residual as residual
import check_r771_horizontal_sinc_cap as current
from aim120_model.control import base_indicated_speed_schedule, scheduled_pid_values
from aim120_model.r771_game_control import R771GameControl
from aim120_model.single_surface_game import SurfaceCoefficients

OUT=ROOT/"outputs/jari_offaxis/base_speed_inverse"
AUDIT=residual.AUDIT
DENSITY=inspect.signature(base_indicated_speed_schedule).parameters["sea_level_density_kg_m3"].default
FIELDS=("scale_q","effective_Ks","integral_transport_ratio","transported_integral_old_rad","I_provisional_old_rad",
    "P_old_rad","D_old_rad","natural_unscaled_old_rad","natural_old_rad","effective_old_held_limit_rad",
    "original_AW_limit_old_rad","original_counterfactual_saturated_old_rad","antiwindup_blocked","next_integral_old_rad")


class InverseQControl(R771GameControl):
    """Physical K controls caps; K*s transports high-latched output-coordinate I."""
    def __init__(self):
        super().__init__()
        self.previous_high_effective_k=[None,None]

    def update_axis(self,axis,ts,error,memory,pid,limit,gain,k,nonlinear,*,scale_q,beta):
        if axis not in (0,1):raise ValueError("axis must be pitch0 or yaw1")
        if not all(math.isfinite(v)for v in (ts,error,limit,gain,k,scale_q,beta)) or limit<=0 or k<=0 or scale_q<=0 or not 0<=gain<=1:
            raise ValueError("finite positive physical limit/K/scale and gain in[0,1] required")
        if self.last_time[axis]is not None:
            if ts==self.last_time[axis]:return self.cached[axis]
            if ts<self.last_time[axis]:raise ValueError("control clock cannot move backwards")
        if abs(ts*self.controller_hz-round(ts*self.controller_hz))>1e-8:raise ValueError("control memory tick clock")
        if gain<=1e-12:
            if memory!=(0.,0.,0.):raise ValueError("inactive PID memory must be zero")
            return 0.,memory
        factor=current.inverse_sinc(beta)
        self.nonlinear_latched=self.nonlinear_latched or bool(nonlinear)
        integral,previous_error,previous_derivative=memory
        effective_k=k*scale_q;ratio=1.
        if self.nonlinear_latched:
            previous=self.previous_high_effective_k[axis]
            ratio=1. if previous is None else previous/effective_k
            integral=AUDIT.clip(integral*ratio,pid["integral_limit"])
        dt=1/self.controller_hz
        provisional=AUDIT.clip(integral+pid["i"]*error*dt,pid["integral_limit"])
        derivative=previous_derivative+dt/(1e-9+dt)*((error-previous_error)/dt-previous_derivative)
        p,d=pid["p"]*error,pid["d"]*derivative
        unscaled=p+provisional+d;natural=scale_q*unscaled
        aw_limit=min(limit,limit/k)if self.nonlinear_latched else limit
        output_limit=aw_limit if axis==0 else min(limit,limit/k*factor)
        saturated=AUDIT.clip(natural,output_limit);counter=AUDIT.clip(natural,aw_limit)
        blocked=abs(counter)>=.99*aw_limit and error*counter>0
        next_i=integral if blocked else provisional
        result=gain*saturated,(next_i,error,derivative)
        self.last_time[axis]=ts;self.cached[axis]=result
        if self.nonlinear_latched:
            self.previous_high_k[axis]=k
            self.previous_high_effective_k[axis]=effective_k
        self.diagnostics[axis]={"scale_q":scale_q,"effective_Ks":effective_k,"K_target_per_old_held":k,
            "integral_transport_ratio":ratio,"transported_integral_old_rad":integral,"I_provisional_old_rad":provisional,
            "P_old_rad":p,"D_old_rad":d,"natural_unscaled_old_rad":unscaled,"natural_old_rad":natural,
            "effective_old_held_limit_rad":output_limit,"original_AW_limit_old_rad":aw_limit,
            "original_counterfactual_saturated_old_rad":counter,"antiwindup_blocked":blocked,
            "next_integral_old_rad":next_i,"nonlinear_latched":self.nonlinear_latched}
        return result


def qref(cfg):
    # Exact numeric source field, with the one declared /3.6 placement hypothesis.
    raw=float(cfg["control"]["base_indicated_speed_kmh"])
    if not math.isfinite(raw)or raw<=0:raise ValueError("raw baseIndSpeed must be positive")
    return .5*DENSITY*(raw/3.6)**2


def snapshot():
    result={}
    for folder in ("src","config","configs","docs","data","scripts","outputs"):
        for p in (ROOT/folder).rglob("*"):
            if not p.is_file()or "__pycache__"in p.parts or ".git"in p.parts or OUT in p.parents:continue
            rel=str(p.relative_to(ROOT))
            if rel=="scripts/analyze_r771_speed_residual.py"or rel.startswith("outputs/jari_offaxis/speed_residual/"):continue
            st=p.stat();result[str(p)]=[st.st_size,st.st_mtime_ns]
    return result


def replay_signals(cfg,a,*,expected_held,expected_memory):
    """Two own-history recursions on identical saved signals, not integration."""
    controllers={"current":current.HorizontalSincCapControl(),"inverse_q":InverseQControl()}
    memory={kind:[(0.,0.,0.),(0.,0.,0.)]for kind in controllers}
    count=len(a["t"])
    for kind in controllers:
        a[kind+"__held"]=np.zeros((count,2));a[kind+"__memory"]=np.zeros((count,2,3));a[kind+"__AW"]=np.zeros((count,2),bool)
    for key in FIELDS:a["inverse_q__"+key]=np.zeros((count,2))
    checks={"current_held_cache_closure_rad":0.,"current_memory_cache_closure":0.,"candidate_natural_formula_rad":0.,
        "candidate_gain_last_rad":0.,"candidate_next_I_formula":0.,"candidate_effective_coordinate_transport":0.,
        "candidate_physical_cap_formula_rad":0.,"candidate_original_AW_cap_formula_rad":0.,"candidate_sinc_independent":0.}
    previous_effective=[None,None];high=False
    for i,ts in enumerate(a["t"]):
        gain=float(a["gain"][i])
        if gain<=1e-12:continue
        pid=scheduled_pid_values(cfg["control"]["pid"],(round(float(ts)*48)+1)/48)
        k=float(a["K"][i]);s=float(a["scale_q"][i]);beta=float(a["beta"][i]);nonlinear=bool(a["nonlinear"][i])
        high=high or nonlinear
        controllers["current"].current_beta_rad=beta
        for axis in (0,1):
            error=float(a["request"][i,axis]-a["feedback"][i,axis]);limit=float(a["limits"][axis])
            for kind,c in controllers.items():
                before=memory[kind][axis]
                args=(axis,float(ts),error,before,pid,limit,gain,k,nonlinear)
                held,after=c.update_axis(*args,scale_q=s,beta=beta)if kind=="inverse_q"else c.update_axis(*args)
                memory[kind][axis]=after;a[kind+"__held"][i,axis]=held;a[kind+"__memory"][i,axis]=after
                d=c.diagnostics[axis];a[kind+"__AW"][i,axis]=d["antiwindup_blocked"]
                if kind=="current":
                    checks["current_held_cache_closure_rad"]=max(checks["current_held_cache_closure_rad"],abs(held-expected_held[i,axis]))
                    checks["current_memory_cache_closure"]=max(checks["current_memory_cache_closure"],max(abs(np.asarray(after)-expected_memory[i,axis])))
                    continue
                for key in FIELDS:a["inverse_q__"+key][i,axis]=d[key]
                ks=k*s;ratio=1. if not high or previous_effective[axis]is None else previous_effective[axis]/ks
                integral=AUDIT.clip(before[0]*ratio,pid["integral_limit"])if high else before[0]
                provisional=AUDIT.clip(integral+pid["i"]*error/48,pid["integral_limit"])
                derivative=before[2]+(1/48)/(1e-9+1/48)*((error-before[1])*48-before[2])
                natural=s*(pid["p"]*error+provisional+pid["d"]*derivative)
                aw=min(limit,limit/k)if high else limit
                independent_sinc=1/float(np.sinc(beta/math.pi))
                cap=aw if axis==0 else min(limit,limit/k*independent_sinc)
                counter=AUDIT.clip(natural,aw)
                blocked=abs(counter)>=.99*aw and error*counter>0
                next_i=integral if blocked else provisional
                for key,value in (("candidate_natural_formula_rad",abs(natural-d["natural_old_rad"])),
                    ("candidate_gain_last_rad",abs(gain*AUDIT.clip(natural,cap)-held)),
                    ("candidate_next_I_formula",abs(next_i-after[0])),("candidate_effective_coordinate_transport",abs(ratio-d["integral_transport_ratio"])),
                    ("candidate_physical_cap_formula_rad",abs(cap-d["effective_old_held_limit_rad"])),
                    ("candidate_original_AW_cap_formula_rad",abs(aw-d["original_AW_limit_old_rad"])),
                    ("candidate_sinc_independent",abs(independent_sinc-current.inverse_sinc(beta)))):
                    checks[key]=max(checks[key],value)
                if blocked!=d["antiwindup_blocked"]or high!=d["nonlinear_latched"]:raise AssertionError("candidate AW/high lifecycle")
                if high:previous_effective[axis]=ks
                if c.previous_high_effective_k[axis]!=previous_effective[axis]:raise AssertionError("effective-coordinate K lifecycle")
    scale=a["K"][:,None]/a["limits"]
    for kind in controllers:
        a[kind+"__normalized"]=scale*a[kind+"__held"]
        integral=a[kind+"__memory"][:,:,0]
        a[kind+"__I_output_coordinate_norm"]=scale*integral*(a["scale_q"][:,None]if kind=="inverse_q"else 1.)
    if max(checks.values())>1e-12:raise AssertionError(checks)
    return checks


def window_report(a,mask):
    cur=a["current__normalized"];candidate=a["inverse_q__normalized"]
    delta=candidate-cur
    current_rms=np.sqrt(np.mean(cur[mask]**2,axis=0));delta_rms=np.sqrt(np.mean(delta[mask]**2,axis=0))
    return {"ticks":int(mask.sum()),"scale_q":residual.stats(a["scale_q"],mask),
        "current_normalized_output":residual.stats(cur,mask),"candidate_normalized_output":residual.stats(candidate,mask),
        "candidate_minus_current_normalized":residual.stats(delta,mask),
        "output_delta_RMS_over_current_RMS":[None if value==0 else float(delta_rms[j]/value)for j,value in enumerate(current_rms)],
        "zero_current_RMS_ratio_is_undefined":True,
        "raw_memory_difference":residual.stats(a["inverse_q__memory"]-a["current__memory"],mask),
        "effective_integral_output_coordinate_difference":residual.stats(a["inverse_q__I_output_coordinate_norm"]-a["current__I_output_coordinate_norm"],mask),
        "current_AW_axis_ticks":a["current__AW"][mask].sum(axis=0).tolist(),"candidate_AW_axis_ticks":a["inverse_q__AW"][mask].sum(axis=0).tolist()}


def source_case(cid,x,cache,iv,coefficients):
    prefix=cid+"__";keys=("t","gain","halo","request","feedback","K","beta","nonlinear","3_needed_normalized","5_needed_normalized")
    a={key:cache[prefix+key].copy()for key in keys}
    cfg=x["config"];a["limits"]=np.radians([cfg["aerodynamics"][key]for key in ("horizontal_fin_aoa_limit_deg","vertical_fin_aoa_limit_deg")])
    model=current.make_model(cfg,coefficients);model.initialize(dataclasses.replace(x["initial"]))
    pressures=[]
    for ts in a["t"]:
        state=AUDIT.fix.source_state_at(x["ref"],float(ts));f=model.evaluate(state,float(ts),(0.,0.))
        pressures.append(f.qbar)
    a["q_Pa"]=np.asarray(pressures);a["scale_q"]=qref(cfg)/a["q_Pa"]
    checks=replay_signals(cfg,a,expected_held=cache[prefix+"current__held"],expected_memory=cache[prefix+"current__memory"])
    for width in (3,5):
        maximum=0.;original_t=iv[prefix+"t"]
        for i,ts in enumerate(original_t):
            if ts>3.5+1e-8:break
            f=model.evaluate(residual.INVERSE.source_state(x["ref"],i),float(ts),tuple(iv[prefix+str(width)+"__old_held_req_rad"][i]))
            maximum=max(maximum,max(abs(np.asarray(f.angular_acceleration)-iv[prefix+str(width)+"__omega_dot_rad_s2"][i])))
        checks[str(width)+"_inverse_angular_law_closure_rad_s2"]=maximum
        if maximum>1e-10:raise AssertionError(checks)
    report={"checks":checks,"qref_Pa":qref(cfg),"windows":{}}
    active=(a["gain"]>0)&~a["halo"]
    for name,lo,hi in residual.WINDOWS:
        mask=active&(a["t"]>=lo-1e-8)&(a["t"]<=hi+1e-8);item=window_report(a,mask)
        item["stencils"]={}
        for width in (3,5):
            need=a[str(width)+"_needed_normalized"]
            item["stencils"][str(width)]={"current_error":residual.stats(a["current__normalized"]-need,mask),
                "candidate_error":residual.stats(a["inverse_q__normalized"]-need,mask),"needed_normalized":residual.stats(need,mask)}
        item["stencil_difference"]=residual.stats(a["5_needed_normalized"]-a["3_needed_normalized"],mask)
        report["windows"][name]=item
    points=(19/48,.5,.7,1.,1.5,2.,2.5,3.,3.5)
    report["points"]=[{key:a[key][int(round(ts*48))].tolist()for key in ("t","q_Pa","scale_q","current__normalized","inverse_q__normalized","3_needed_normalized","5_needed_normalized")}for ts in points]
    return a,report


def native_case(cid,x,coefficients):
    cfg=x["profile"]["_model_config"];initial=dataclasses.replace(x["_creation_state"])
    qx,qy,qz,qw=x["creation"]["u16_1"];initial.orientation_quaternion=(qw,qx,qy,qz)
    model=current.make_model(cfg,coefficients);model.initialize(initial)
    receipt=json.loads((current.OUT/(cid+"_run.json")).read_text());ticks=receipt["ticks"]
    a={"t":np.array([t["time_s"]for t in ticks]),"gain":np.array([t["gain"]for t in ticks]),
        "request":np.array([t["request_g"]for t in ticks]),"feedback":np.array([t["feedback_g"]for t in ticks]),
        "limits":np.radians([cfg["aerodynamics"][key]for key in ("horizontal_fin_aoa_limit_deg","vertical_fin_aoa_limit_deg")])}
    sample_by_time={s["time_s"]:s for s in receipt["samples"]};rows=[]
    for tick in ticks:
        ts=tick["time_s"];sample=sample_by_time[ts];state=AUDIT.fix.finite.sample_state(sample)
        f=model.evaluate(state,ts,tuple(sample["evaluated_old_held_rad"]))
        model.request(state,x["_target_model"].state_at(ts),ts,f)
        rows.append((f.qbar,model.current_control_k,math.hypot(*f.local_incidence),model.current_force_nonlinear))
    a.update({key:np.array([row[j]for row in rows])for j,key in enumerate(("q_Pa","K","beta","nonlinear"))})
    a["scale_q"]=qref(cfg)/a["q_Pa"]
    checks=replay_signals(cfg,a,expected_held=np.asarray([t["held_fin_rad"]for t in ticks]),expected_memory=np.asarray([t["pid_memory"]for t in ticks]))
    report={"checks":checks,"qref_Pa":qref(cfg),"saved_free_ticks":len(ticks),"windows":{},
        "state_source":"Current horizontal_sinc actual saved free trajectory; no new native forward",
        "creation_quaternion_preserved":True,"load_independent_observation_available":False}
    for name,lo,hi in (*residual.WINDOWS,*AUDIT.fix.finite.harness.WINDOWS):
        mask=(a["gain"]>0)&(a["t"]>=lo-1e-8)&(a["t"]<=hi+1e-8)
        report["windows"][name]=window_report(a,mask)
    return a,report


def plot_source(arrays):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(3,2,figsize=(13,10))
    for row,cid in enumerate(current.CASES):
        a=arrays[cid];mask=(a["gain"]>0)&~a["halo"]
        for axis,ax in enumerate(axes[row]):
            for value,label,color in ((a["current__normalized"],"Current conditional","#ce7b2f"),(a["inverse_q__normalized"],"Inverse-q output","#2465b3"),(a["3_needed_normalized"],"Source3 needed","#929292"),(a["5_needed_normalized"],"Source5 needed","#147378")):
                ax.plot(a["t"][mask],value[mask,axis],label=label,color=color)
            ax.set(title=cid+": "+("pitch"if axis==0 else"yaw"),xlabel="Original time(s)",ylabel="Physical K*held/raw limit")
            ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.suptitle("Conditional baseIndSpeed inverse-q placement; source3/5 activation halo excluded; no free trajectory",fontsize=11)
    fig.tight_layout(rect=(0,0,1,.95));fig.savefig(OUT/"conditional_output.png",dpi=150);plt.close(fig)


def main():
    before=snapshot();OUT.mkdir(parents=True,exist_ok=True)
    protocol={"one_candidate":"qref=.5*existing sea-level density*(raw baseIndSpeed/3.6)^2; s=qref/currentq; natural=s*(P+Iprovisional+D) on both axes",
        "integral_coordinate":"high-only previous(K*s)/(K*s) transport, rawIlim1 and rawKi48 unchanged",
        "physical_boundaries":"Physical K and current yaw-only sinc/pitchoriginal output cap unchanged; AW original cap applied to scaled actual natural; gain-last",
        "screen":"Seek simultaneous nonworsening3/5 needed-yaw agreement in30/60 capture/platform/unload. Strong longitudinal counterevidence:0pitch both stencils worse plus all3 native capture input-difference RMS larger than original pitch input RMS. No free-pass claim from conditional agreement.",
        "free_trigger":"Only cross-condition support without major longitudinal counterevidence permits3JARI actualfree, original strict gate then3native72rules",
        "density_source":"Existing base_indicated_speed_schedule sea_level_density_kg_m3 signature value", "sea_level_density_kg_m3":DENSITY,
        "no_new_runtime_constants_ratio_clips_fallbacks":True,"evidence_grade":"D placement, not raw field unit identification"}
    AUDIT.slew.write_json(OUT/"protocol.json",protocol)
    inputs=residual.INVERSE.source_inputs();cache=np.load(residual.OUT/"conditional.npz");iv=np.load(residual.INVERSE.OUTPUT/"inverse_inputs.npz")
    coefficients=AUDIT.drag.DragCoefficients(**json.loads((AUDIT.drag.OUTPUT/"fit.json").read_text())["coefficients"])
    source_arrays,source_reports,native_reports,saved={},{},{},{}
    for cid in current.CASES:
        a,report=source_case(cid,inputs[cid],cache,iv,coefficients);source_arrays[cid]=a;source_reports[cid]=report
        saved.update({cid+"__"+key:value for key,value in a.items()});print(cid,"source conditional complete",flush=True)
    cases={c["id"]:c for c in AUDIT.fix.finite.harness.load_cases(AUDIT.fix.finite.comparison.SNAPSHOT)}
    for cid in current.NATIVE:
        a,report=native_case(cid,cases[cid.removeprefix("native_")],coefficients);native_reports[cid]=report
        saved.update({cid+"__"+key:value for key,value in a.items()});print(cid,"native saved-state conditional complete",flush=True)
    source0=source_reports["jari_259_0"]["windows"]["first_capture"]
    source0_pitch_worse=all(source0["stencils"][str(w)]["candidate_error"]["RMS"][0]>source0["stencils"][str(w)]["current_error"]["RMS"][0]for w in(3,5))
    native_large={cid:report["windows"]["first_capture"]["output_delta_RMS_over_current_RMS"][0]>1. for cid,report in native_reports.items()}
    longitudinal_refuted=source0_pitch_worse and all(native_large.values())
    cross_rules={cid+"__"+name+"__stencil"+str(w):item["stencils"][str(w)]["candidate_error"]["RMS"][1]<=item["stencils"][str(w)]["current_error"]["RMS"][1]
        for cid in("jari_259_30","jari_259_60")for name,item in source_reports[cid]["windows"].items()for w in(3,5)}
    conditional_support=not longitudinal_refuted and all(cross_rules.values())
    plot_source(source_arrays);np.savez_compressed(OUT/"conditional.npz",**saved)
    protected=before==snapshot();finite=all(np.isfinite(v).all()for v in saved.values())
    if not protected or not finite:raise AssertionError({"protected":protected,"finite":finite})
    profile=json.loads(AUDIT.fix.finite.comparison.SNAPSHOT.read_text())["2.59.0.27/su_r_77_1"]
    report={"task":"War Thunder raw baseIndSpeed inverse-q output placement conditional test","protocol":protocol,
        "raw_baseIndSpeed":profile["control"]["base_indicated_speed_kmh"],"raw_field_source":profile["control"]["parameter_sources"]["base_indicated_speed_kmh"],
        "qref_Pa":qref(inputs["jari_259_0"]["config"]),"source_cases":source_reports,"native_saved_state_cases":native_reports,
        "conditional_cross_yaw_rules":cross_rules,"conditional_longitudinal_counterevidence":{"source0_both_stencils_pitch_error_worse":source0_pitch_worse,"all3_native_capture_delta_RMS_larger_than_current_pitch_RMS":native_large,"refuted":longitudinal_refuted},
        "conditional_support_for_prescribed_free_stage":conditional_support,"actual_free_acceptance_established":False,
        "identification_boundary":"Frozen Ku(V) and other fitted coefficients may already absorb an absent schedule. This test isolates one extra placement at frozen Ku, not native factorization or field units; do not double-count it as physical truth.",
        "verification":{"new_Jari_forward_integrations":0,"new_native_forward_integrations":0,"saved_native_state_replays":3,
            "source3_5_are_diagnostic_only":True,"protected_size_mtime_unchanged":protected,"all_arrays_finite":finite,
            "fits_scans_new_coefficients":0,"core_config_data_docs_changes":0,"default_model_changed":False}}
    AUDIT.slew.write_json(OUT/"results.json",report)
    print(json.dumps({"conditional_support":conditional_support,"longitudinal_refuted":longitudinal_refuted,
        "failed_cross_rules":[k for k,v in cross_rules.items()if not v],"verification":report["verification"],"output":str(OUT)}),flush=True)


if __name__=="__main__":main()
