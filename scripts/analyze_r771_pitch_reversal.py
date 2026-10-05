#!/usr/bin/env python3
"""War Thunder/JARI pitch reversal audit of saved trajectories; no new flight.

One predeclared diagnostic interpretation: raw loft omega .5 replaces the
existing1.3*.5 transition centre. Same-memory and recursively constructed
source-state memories remain conditional diagnostics, not game memories.
"""
from __future__ import annotations
import copy
import dataclasses
import json
import math
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/"scripts"),str(ROOT/"src")]
sys.dont_write_bytecode=True
os.environ.setdefault("MPLCONFIGDIR","/tmp/missle_sim_mpl")
import numpy as np
import check_r771_horizontal_sinc_cap as best
import compare_r771_jari_offaxis as comparison
import analyze_r771_jari_capture_gap as gap
from aim120_model.control import scheduled_pid_values
from aim120_model.guidance import interpolate_table
from aim120_model.math3d import dot, norm
from aim120_model.units import G0_MPS2

OUT=ROOT/"outputs/jari_offaxis/bias_discussion/pitch_reversal"
BASE=ROOT/"outputs/jari_offaxis"
AUDIT=best.dec.audit
DT=1/48
T=np.arange(73)*DT
POINTS=(19/48,.5,28/48,.625,34/48,.75,1.,55/48)
DIAG_FIELDS=("P_old_rad","I_provisional_old_rad","D_old_rad","natural_old_rad",
    "transported_integral_old_rad","I_increment_old_rad","next_integral_old_rad",
    "integral_transport_ratio","effective_old_held_limit_rad","original_AW_limit_old_rad",
    "antiwindup_blocked","nonlinear_latched")


def serial(x):
    if isinstance(x,dict):return {k:serial(v)for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [serial(v)for v in x]
    if isinstance(x,np.ndarray):return x.tolist()
    if isinstance(x,np.generic):return x.item()
    return x


def interp(t,st,x):
    x=np.asarray(x)
    if np.min(t)<st[0]-1e-8 or np.max(t)>st[-1]+1e-8:raise ValueError("No source extrapolation")
    if x.ndim==1:return np.interp(t,st,x)
    return np.column_stack([np.interp(t,st,x[:,i])for i in range(x.shape[1])])


def first_negative(t,x,low=14/48):
    a=np.flatnonzero((t>=low-1e-8)&(np.asarray(x)<0))
    return float(t[a[0]])if len(a)else None


def components(model,f,details,request):
    b=best.dec.outer_boundaries(details["pn_mps2"],details["loft_g"],f.wind_basis,model.gravity,
        float(model.config["guidance"]["maximum_lateral_acceleration_g"])*model.gravity)
    pn=dot(details["pn_mps2"],f.wind_basis.up)/G0_MPS2
    loft=details["loft_g"]*model.gravity*f.wind_basis.up[1]/G0_MPS2
    grav=model.gravity*f.wind_basis.up[1]/G0_MPS2
    return {"PN_pitch_G":pn,"loft_projected_pitch_G":loft,"gravity_pitch_G":grav,
        "boundary_pitch_delta_G":request[0]-pn-loft-grav,
        "pre_gravity_cap_active":b["cap1_active"],"post_gravity_cap_active":b["cap2_active"],
        "loft_world_G":details["loft_g"],"loft_horizontal_gate":details.get("horizontal_loft_gate",1.),
        "loft_minimum_weight":details["loft_weight"],"loft_current_weight":details["loft_current_weight"],
        "total_LOS_rate_deg_s":math.degrees(details["loft_los_rate_magnitude_rad_s"]),
        "request_pitch_G":request[0],"feedback_pitch_G":f.feedback_g[0]}


def pid_terms(controller,ts,request,feedback,memory,pid,limit,gain,k,nonlinear,beta):
    controller.current_beta_rad=beta
    held,next_memory=controller.update_axis(0,ts,request-feedback,memory,pid,limit,gain,k,nonlinear)
    if gain<=1e-12:return {"held_rad":held,"pre_memory":memory,"post_memory":next_memory},next_memory
    d=controller.diagnostics[0]
    alpha=DT/(1e-9+DT)
    coefficient=pid["p"]+pid["i"]*DT+pid["d"]*alpha/DT
    memory_offset=d["transported_integral_old_rad"]-pid["d"]*alpha*memory[1]/DT+pid["d"]*(1-alpha)*memory[2]
    current_error=request-feedback
    closure=d["natural_old_rad"]-(coefficient*current_error+memory_offset)
    if abs(d["I_provisional_old_rad"])>=pid["integral_limit"]:raise ValueError("Unregistered I clamp breaks affine decomposition")
    return {"held_rad":held,"pre_memory":memory,"post_memory":next_memory,
        **{key:d[key]for key in DIAG_FIELDS},"error_G":current_error,
        "current_error_coefficient_rad_per_G":coefficient,"memory_offset_rad":memory_offset,
        "affine_natural_closure_rad":closure,
        "same_memory_zero_output_request_G":feedback-memory_offset/coefficient},next_memory


def inputs():
    grid=json.loads((BASE/"angle_grid/results.json").read_text())
    profiles=json.loads(comparison.SNAPSHOT.read_text())
    result={}
    for cid in ("angle_p0","angle_p5","angle_p10","angle_p15","collision"):
        if cid=="collision":
            source=BASE/"collision_pn_probe/collision_config_response.json"
            free=BASE/"collision_pn_probe/collision_best_free.json"
        else:
            source=Path(grid["cases"][cid]["source_path"])
            free=Path(grid["cases"][cid]["candidate_full_path"])
        run=json.loads(source.read_text());ref,_=comparison.parse_reference(run)
        profile,_=comparison.profile_for(run,{},profiles);cfg=copy.deepcopy(profile["_model_config"])
        initial,target=comparison.free_inputs(run,ref,cfg)
        result[cid]={"source_path":source,"free_path":free,"run":run,"ref":ref,"cfg":cfg,"initial":initial,"target":target}
    return result


def saved_free(x,coeff):
    model=best.make_model(x["cfg"],coeff);model.initialize(dataclasses.replace(x["initial"]))
    controller=best.HorizontalSincCapControl();memory=[(0.,0.,0.),(0.,0.,0.)]
    result=json.loads(x["free_path"].read_text());samples={r["time_s"]:r for r in result["samples"]}
    checks={k:0. for k in ("request_G","feedback_G","held_rad","memory","PID_terms_rad","angular_record_rad_s2","affine_rad")}
    rows=[]
    for tick in result["ticks"]:
        ts=tick["time_s"]
        if ts>1.5+1e-8:break
        s=samples[ts];state=AUDIT.fix.finite.sample_state(s)
        f=model.evaluate(state,ts,tuple(tick["evaluated_old_held_rad"]))
        req,d=model.request(state,x["target"](ts),ts,f)
        checks["request_G"]=max(checks["request_G"],float(np.max(abs(np.asarray(req)-tick["request_g"]))))
        checks["feedback_G"]=max(checks["feedback_G"],float(np.max(abs(np.asarray(f.feedback_g)-tick["feedback_g"]))))
        checks["angular_record_rad_s2"]=max(checks["angular_record_rad_s2"],float(np.max(abs(np.asarray(f.angular_acceleration)-s["angular_acceleration_rad_s2"]))))
        gain=interpolate_table((round(ts*48)+1)/48,x["cfg"]["guidance"]["flight_time_gain_table"])
        pid=scheduled_pid_values(x["cfg"]["control"]["pid"],(round(ts*48)+1)/48)
        limits=np.radians([x["cfg"]["aerodynamics"][k]for k in ("horizontal_fin_aoa_limit_deg","vertical_fin_aoa_limit_deg")])
        row,memory[0]=pid_terms(controller,ts,req[0],f.feedback_g[0],memory[0],pid,float(limits[0]),gain,
            model.current_control_k,model.current_force_nonlinear,math.hypot(*f.local_incidence))
        yaw,memory[1]=controller.update_axis(1,ts,req[1]-f.feedback_g[1],memory[1],pid,float(limits[1]),gain,model.current_control_k,model.current_force_nonlinear)
        checks["held_rad"]=max(checks["held_rad"],abs(row["held_rad"]-tick["held_fin_rad"][0]),abs(yaw-tick["held_fin_rad"][1]))
        checks["memory"]=max(checks["memory"],float(np.max(abs(np.asarray(memory)-tick["pid_memory"]))))
        if gain>0:
            checks["PID_terms_rad"]=max(checks["PID_terms_rad"],max(abs(float(row[k])-float(tick["control_runtime_diagnostics"][0][k]))for k in DIAG_FIELDS))
            checks["affine_rad"]=max(checks["affine_rad"],abs(row["affine_natural_closure_rad"]))
        rows.append({"t":ts,"gain":gain,"pitch_deg":math.degrees(s["pitch_rad"]),**components(model,f,d,req),**row})
    assert max(checks.values())<1e-10,checks
    return rows,{"closure":checks,"saved_actual_free_first_pitch_reversal_s":first_negative(np.array([r["t"]for r in rows]),np.array([r["held_rad"]for r in rows])),
        "points":[min(rows,key=lambda r:abs(r["t"]-t))for t in POINTS]}


def source_kinematics(x):
    ref=x["ref"];t=ref["t"]
    q=gap.continuous_quaternions(ref["quaternion_xyzw"][:,[3,0,1,2]])
    raw=np.column_stack((-ref["omega_xyz_export_units"][:,2],ref["omega_xyz_export_units"][:,1]))
    answer={}
    for width in (3,5):
        br=gap.body_rates(q,t,width)
        rates=np.column_stack((br[:,2],-br[:,1]))
        answer[f"quaternion{width}"]={"rates":rates,"qdot":gap.derivative(rates,t,width),"width":width,
            "derivative_half_support_s":(width-1)/24,"rate_source":"normalized source quaternion differentiated; pitch=bodyOmega_z,yaw=-bodyOmega_y",
            "source_roll_rate_rad_s":br[:,0],"velocity_derivative":gap.derivative(ref["velocity_xyz_mps"],t,width)}
        answer[f"rawomega{width}"]={"rates":raw,"qdot":gap.derivative(raw,t,width),"width":width,
            "derivative_half_support_s":(width//2)/24,"rate_source":"unverified raw export mapping pitch=-omega.z,yaw=omega.y; separate sensitivity, not quaternion-rate proof",
            "source_roll_rate_rad_s":ref["omega_xyz_export_units"][:,0],"velocity_derivative":gap.derivative(ref["velocity_xyz_mps"],t,width)}
    return answer


def conditional(x,kin,coeff):
    cfg=x["cfg"];ref=x["ref"]
    models={name:best.make_model(cfg,coeff)for name in ("baseline","raw_center")}
    for model in models.values():model.initialize(dataclasses.replace(x["initial"]))
    models["raw_center"].transition_coefficient=1.
    controllers={k:best.HorizontalSincCapControl()for k in models}
    memories={k:[(0.,0.,0.),(0.,0.,0.)]for k in models}
    r=interp(T,ref["t"],kin["rates"]);qd=interp(T,ref["t"],kin["qdot"])
    source_dv=interp(T,ref["t"],kin["velocity_derivative"])
    limits=np.radians([cfg["aerodynamics"][key]for key in ("horizontal_fin_aoa_limit_deg","vertical_fin_aoa_limit_deg")])
    source_controls=interp(T,ref["t"],np.asarray([s["controls"]for s in x["run"]["samples"]]))
    rows=[];checks={"angular_inverse_rad_s2":0.,"one_step_PID_inverse_rad":0.,"affine_rad":0.,"same_state_plant_delta":0.,"same_memory_decomposition_rad":0.}
    for i,ts in enumerate(T):
        state=dataclasses.replace(AUDIT.fix.source_state_at(ref,float(ts)),pitch_rate=float(r[i,0]),yaw_rate=float(r[i,1]))
        target=x["target"](float(ts));f=models["baseline"].evaluate(state,float(ts),(0.,0.))
        f_alt=models["raw_center"].evaluate(state,float(ts),(0.,0.))
        checks["same_state_plant_delta"]=max(checks["same_state_plant_delta"],
            float(np.max(abs(np.asarray(f.angular_acceleration)-f_alt.angular_acceleration))),
            float(np.max(abs(np.asarray(f.specific_acceleration)-f_alt.specific_acceleration))))
        fb=f.feedback_g;beta=math.hypot(*f.local_incidence)
        requests={};details={}
        for name,model in models.items():
            requests[name],details[name]=model.request(state,target,float(ts),f)
        model=models["baseline"];k=model.current_control_k
        ku=model.coefficients.control_gain*(1+model.coefficients.speed_slope*(norm(state.velocity)/model.speed0-1))
        needed=(qd[i]-np.asarray(f.angular_acceleration))/ku
        closed=model.evaluate(state,float(ts),tuple(needed)).angular_acceleration
        checks["angular_inverse_rad_s2"]=max(checks["angular_inverse_rad_s2"],float(np.max(abs(np.asarray(closed)-qd[i]))))
        gain=interpolate_table(float(ts)+DT,cfg["guidance"]["flight_time_gain_table"])
        pid=scheduled_pid_values(cfg["control"]["pid"],float(ts)+DT)
        components_by={name:components(m,f,details[name],requests[name])for name,m in models.items()}
        baseline_previous=copy.deepcopy(memories["baseline"])
        trial=copy.deepcopy(controllers["baseline"])
        cf,_=pid_terms(trial,float(ts),requests["raw_center"][0],fb[0],baseline_previous[0],pid,float(limits[0]),gain,k,model.current_force_nonlinear,beta)
        out={}
        for name,c in controllers.items():
            out[name],memories[name][0]=pid_terms(c,float(ts),requests[name][0],fb[0],memories[name][0],pid,float(limits[0]),gain,k,model.current_force_nonlinear,beta)
            _,memories[name][1]=c.update_axis(1,float(ts),requests[name][1]-fb[1],memories[name][1],pid,float(limits[1]),gain,k,model.current_force_nonlinear)
        row={"t":float(ts),"gain":gain,"source_rate_rad_s":r[i],"source_qdot_rad_s2":qd[i],"needed_held_rad":needed,
            "source_velocity_derivative_feedback_pitch_G_diagnostic":dot(tuple(source_dv[i]+np.array([0.,model.gravity,0.])),f.wind_basis.up)/G0_MPS2,
            "source_controls_raw_unidentified_units":source_controls[i],"source_state_interpolated":True,
            "outside_startup_noncausal_derivative_halo":bool(ts>14/48+kin["derivative_half_support_s"]+1e-8),
            "source_derivative_support_s":[float(ts-kin["derivative_half_support_s"]),float(ts+kin["derivative_half_support_s"])],
            "conditional_baseline":out["baseline"],"conditional_raw_center_own_memory":out["raw_center"],
            "raw_center_same_baseline_previous_memory":cf,"signal_components":components_by}
        if gain>0:
            b=out["baseline"];tr=(b["transported_integral_old_rad"],baseline_previous[0][1],baseline_previous[0][2])
            error,status=AUDIT.error_for_output(float(needed[0]),tr,pid,gain,b["effective_old_held_limit_rad"])
            required_req=error+fb[0]
            required_loft=None
            cp=components_by["baseline"]
            if status in (1,2):
                checks["one_step_PID_inverse_rad"]=max(checks["one_step_PID_inverse_rad"],abs(gain*AUDIT.clip(AUDIT.natural_pid(error,tr,pid),b["effective_old_held_limit_rad"])-needed[0]))
                if abs(cp["boundary_pitch_delta_G"])<1e-10 and abs(f.wind_basis.up[1])>1e-8:
                    required_loft=(required_req-cp["PN_pitch_G"]-cp["gravity_pitch_G"])/(model.gravity*f.wind_basis.up[1]/G0_MPS2)
            delta_error=requests["raw_center"][0]-requests["baseline"][0]
            expected=b["current_error_coefficient_rad_per_G"]*delta_error
            checks["same_memory_decomposition_rad"]=max(checks["same_memory_decomposition_rad"],abs((cf["natural_old_rad"]-b["natural_old_rad"])-expected))
            checks["affine_rad"]=max(checks["affine_rad"],abs(b["affine_natural_closure_rad"]),abs(out["raw_center"]["affine_natural_closure_rad"]))
            row.update(inverse_given_baseline_previous_memory={"status":status,"feasible_interior":status==1,
                "needed_request_pitch_G":required_req if status in(1,2)else None,"needed_loft_world_G":required_loft,
                "needed_I_provisional_at_existing_P_D_rad":float(needed[0]/gain-b["P_old_rad"]-b["D_old_rad"]),
                "required_request_is_conditional_on_model_memory":True},
                raw_center_same_memory_change={"request_pitch_G":delta_error,
                    "P_rad":cf["P_old_rad"]-b["P_old_rad"],"I_increment_rad":cf["I_provisional_old_rad"]-b["I_provisional_old_rad"],
                    "D_rad":cf["D_old_rad"]-b["D_old_rad"],"held_rad":cf["held_rad"]-b["held_rad"],
                    "own_history_extra_held_rad":out["raw_center"]["held_rad"]-cf["held_rad"]})
        rows.append(row)
    assert max(checks.values())<1e-10,checks
    active=np.array([r["gain"]>0 and r["outside_startup_noncausal_derivative_halo"]for r in rows])
    need=np.array([r["needed_held_rad"][0]for r in rows]);negative=active&(need<0)
    report={"rate_source":kin["rate_source"],"derivative_half_support_s":kin["derivative_half_support_s"],"checks":checks,
        "first_needed_pitch_reversal_after_halo_s":first_negative(T,np.where(active,need,0.)),
        "source_pitch_negative_supported_ticks":int(negative.sum()),
        "outputs":{},"points":[rows[int(round(t*48))]for t in POINTS]}
    for name in ("conditional_baseline","raw_center_same_baseline_previous_memory","conditional_raw_center_own_memory"):
        u=np.array([row[name]["held_rad"]for row in rows])
        report["outputs"][name]={"first_negative_pitch_s":first_negative(T,u),
            "wrong_positive_sign_when_source_negative_ticks":int(np.sum(negative&(u>=0))),
            "negative_when_source_nonnegative_ticks":int(np.sum(active&(need>=0)&(u<0))),
            "error_rad_on_negative_source_supported_ticks":[float(np.min((u-need)[negative])),float(np.max((u-need)[negative]))]if negative.any()else None}
    return rows,report


def plots(all_data):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(3,2,figsize=(15,11),sharex=True)
    for row,cid in enumerate(("angle_p0","angle_p5","collision")):
        data=all_data[cid];q5=data["quaternion5"];q3=data["quaternion3"]
        t=np.array([r["t"]for r in q5]);mask=t>=.38
        left,right=ax[row]
        for data_q,color,label in ((q3,"#779aa0","needed, quaternion3"),(q5,"#187879","needed, quaternion5")):
            left.plot(t,[r["needed_held_rad"][0]for r in data_q],color=color,label=label)
        for name,color,label in (("conditional_baseline","#dd9135","current conditional memory"),("raw_center_same_baseline_previous_memory","#ae4383","raw0.5 same previous memory"),("conditional_raw_center_own_memory","#6355b3","raw0.5 own conditional memory")):
            left.plot(t,[r[name]["held_rad"]for r in q5],color=color,label=label)
        for name,color,label in (("baseline","#dd9135","current formula request"),("raw_center","#6355b3","raw0.5 formula request")):
            right.plot(t,[r["signal_components"][name]["request_pitch_G"]for r in q5],color=color,label=label)
        right.plot(t,[r.get("inverse_given_baseline_previous_memory",{}).get("needed_request_pitch_G",math.nan)for r in q5],color="#187879",label="needed request, same baseline memory")
        for a in (left,right):
            a.axhline(0.,color="#999",lw=.7);a.axvspan(0.,14/48+4/24,color="#aaa",alpha=.16,label="quaternion5 startup derivative halo")
            a.set_xlim(.3,1.5);a.grid(alpha=.2);a.legend(fontsize=7)
        left.set_ylabel(cid+"\nOld-input radians");right.set_ylabel("Pitch request (G)")
    for a in ax[-1]:a.set_xlabel("Original source time (s)")
    fig.suptitle("Game-only diagnostic: same source states and explicit conditional controller memory\nNo free trajectory prediction; source-derived inverse is noncausal, raw controls have unidentified units",fontsize=12)
    fig.tight_layout(rect=(0,0,1,.95));fig.savefig(OUT/"same_state_pitch_reversal.png",dpi=150);plt.close(fig)


def main():
    files=[ROOT/p for p in ("src/aim120_model/single_surface_game.py","scripts/check_r771_horizontal_sinc_cap.py",
        "scripts/compare_r771_jari_offaxis.py","config/r771_simple_joint_frozen.json")]
    receipt=[(p.stat().st_size,p.stat().st_mtime_ns)for p in files]
    cases=inputs();coeff=AUDIT.drag.DragCoefficients(**json.loads((AUDIT.drag.OUTPUT/"fit.json").read_text())["coefficients"])
    report={"task":"War Thunder/JARI saved-state pitch first-reversal diagnosis","new_forward_integrations":0,"new_fits":0,"new_API_calls":0,
        "predeclared_interpretation":"Only raw0.5 transition centre vs existing1.3*.5; raw fields and horizontal gate unchanged. One interpretation; prior failed I-phase is not retested.",
        "conditional_state_semantics":"Source24Hz snapshots interpolated to48Hz, normalized quaternion interpolation; controller memories are constructed candidates, not observed game state.",
        "primary_inverse":"quaternion3/quaternion5-derived rate then derivative, separate widths with startup halo; original rawomega3/5 mapping is sensitivity only",
        "status_codes":{"0":"inactive","1":"unique interior given specified memory","2":"saturated boundary, nonunique request","-1":"output above existing cap"},"cases":{}}
    raw_data={};arrays={}
    for cid,x in cases.items():
        free,free_report=saved_free(x,coeff)
        cr={"source_path":str(x["source_path"]),"free_path":str(x["free_path"]),"saved_actual_free":free_report,"conditional":{}}
        raw_data[cid]={"actual_free":free}
        for kind,kin in source_kinematics(x).items():
            rows,conditional_report=conditional(x,kin,coeff)
            cr["conditional"][kind]=conditional_report;raw_data[cid][kind]=rows
            for key in ("needed_held_rad","source_rate_rad_s","source_qdot_rad_s2"):
                arrays[cid+"__"+kind+"__"+key]=np.array([r[key]for r in rows])
            for name in ("conditional_baseline","raw_center_same_baseline_previous_memory","conditional_raw_center_own_memory"):
                for key in ("held_rad","P_old_rad","I_provisional_old_rad","D_old_rad","error_G"):
                    arrays[cid+"__"+kind+"__"+name+"__"+key]=np.array([r[name].get(key,0.)for r in rows])
        sensitivity=[]
        for ts in POINTS:
            i=int(round(ts*48))
            sensitivity.append({"t":ts,"outside_combined_quaternion5_halo":bool(ts>14/48+4/24+1e-8),
                "needed_pitch_old_rad_by_method":{k:float(raw_data[cid][k][i]["needed_held_rad"][0])for k in ("quaternion3","quaternion5","rawomega3","rawomega5")},
                "current_conditional_pitch_held_rad_by_method":{k:raw_data[cid][k][i]["conditional_baseline"]["held_rad"]for k in ("quaternion3","quaternion5","rawomega3","rawomega5")}})
        cr["stencil_sensitivity_points"]=sensitivity
        report["cases"][cid]=cr
    arrays["t"]=T
    assert receipt==[(p.stat().st_size,p.stat().st_mtime_ns)for p in files]
    report["critical_inputs_size_mtime_unchanged"]=True
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/"results.json").write_text(json.dumps(serial(report),indent=2)+"\n")
    (OUT/"traces.json").write_text(json.dumps(serial(raw_data))+"\n")
    np.savez(OUT/"conditional_arrays.npz",**arrays)
    plots(raw_data)
    print(json.dumps({c:{"free_first_reversal":r["saved_actual_free"]["saved_actual_free_first_pitch_reversal_s"],
        "quaternion5":r["conditional"]["quaternion5"]["outputs"]}for c,r in report["cases"].items()},indent=2))


if __name__=="__main__":main()
