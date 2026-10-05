#!/usr/bin/env python3
"""Bounded Jari game diagnostic: frozen-plant inverse input and three replays."""
from __future__ import annotations

import copy
import dataclasses
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
import fit_r771_jari_finite_force as finite
from fit_r771_jari_finite_force import comparison, fields, drag, slew
import analyze_r771_jari_capture_gap as gap
import try_r771_jari_trim_target as trim
from aim120_model.aerodynamics import body_axes_for_state, pitch_yaw_from_quaternion
from aim120_model.dynamics import SimState
from aim120_model.math3d import norm, sub
from aim120_model.single_surface_game import SurfaceCoefficients, interpolate_table
from aim120_model.units import G0_MPS2

OUTPUT=ROOT/"outputs/jari_offaxis/inverse_control"
CASES=("jari_259_0","jari_259_30","jari_259_60")
PRIMARY_WIDTH=5
BOUNDARY_TOL=1e-6


def rms(x):return float(np.sqrt(np.mean(np.asarray(x)**2)))


def source_inputs():
    profiles=json.loads(comparison.SNAPSHOT.read_text())
    manifest=json.loads(comparison.DEFAULT_MANIFEST.read_text());out={}
    for e in manifest["references"]:
        run=comparison.split_runs(e["export_path"])[0]
        ref,diagnostics=comparison.parse_reference(run)
        profile,_=comparison.profile_for(run,e,profiles)
        cfg=copy.deepcopy(profile["_model_config"])
        initial,target_at=comparison.free_inputs(run,ref,cfg)
        out[e["id"]]={"run":run,"ref":ref,"config":cfg,"initial":initial,
            "target_at":target_at,"observed":fields.observed_fields(run,ref)}
    return out


def source_state(ref,i):
    qx,qy,qz,qw=ref["quaternion_xyzw"][i];q=(qw,qx,qy,qz)
    pitch,yaw=pitch_yaw_from_quaternion(q);omega=ref["omega_xyz_export_units"][i]
    return SimState(tuple(ref["position_xyz_m"][i]),tuple(ref["velocity_xyz_mps"][i]),
        pitch,yaw,-omega[2],omega[1],float(ref["mass"][i]),orientation_quaternion=q)


def window_mask(t,lo=.5,hi=3.5):return (t>=lo-BOUNDARY_TOL)&(t<=hi+BOUNDARY_TOL)


def outside_duration(t,y,limit,lo,hi):
    """Exact duration outside symmetric cap in the piecewise-linear sequence."""
    total=0.
    for i in range(len(t)-1):
        a=max(float(t[i]),lo);b=min(float(t[i+1]),hi)
        if b<=a:continue
        va=float(np.interp(a,t[i:i+2],y[i:i+2]));vb=float(np.interp(b,t[i:i+2],y[i:i+2]))
        cuts=[0.,1.]
        if va!=vb:
            cuts.extend((z-va)/(vb-va) for z in (-limit,limit) if 0<(z-va)/(vb-va)<1)
        cuts.sort()
        for x1,x2 in zip(cuts[:-1],cuts[1:]):
            if abs(va+(vb-va)*(x1+x2)/2)>limit:total+=(b-a)*(x2-x1)
    return total


def range_and_caps(t,held,target,limits):
    answer={}
    for name,lo,hi in (("interior_0p5_3p5",.5,3.5),("full_original",float(t[0]),float(t[-1]))):
        mask=window_mask(t,lo,hi)
        answer[name]={axis:{"old_held_required_range_deg":np.degrees([held[mask,j].min(),held[mask,j].max()]).tolist(),
            "old_held_cap_deg":math.degrees(limits[j]),
            "old_held_over_cap_duration_s":outside_duration(t,held[:,j],limits[j],lo,hi),
            "equivalent_target_required_range_deg":np.degrees([target[mask,j].min(),target[mask,j].max()]).tolist(),
            "equivalent_normalized_required_range":[float(target[mask,j].min()/limits[j]),float(target[mask,j].max()/limits[j])],
            "equivalent_normalized_over_one_duration_s":outside_duration(t,target[:,j]/limits[j],1.,lo,hi)}
            for j,axis in enumerate(("pitch","yaw"))}
    return answer


def vector_difference(pred,observed,mask):
    residual=pred-observed;pn=np.linalg.norm(pred,axis=1);on=np.linalg.norm(observed,axis=1)
    valid=mask&(pn>.5*G0_MPS2)&(on>.5*G0_MPS2)
    angle=np.degrees(np.arccos(np.clip(np.sum(pred[valid]*observed[valid],axis=1)/(pn[valid]*on[valid]),-1.,1.)))
    return {"vector_RMS_residual_mps2":rms(np.linalg.norm(residual[mask],axis=1)),
        "magnitude_RMS_residual_mps2":rms((pn-on)[mask]),
        "direction_angle_RMS_deg":rms(angle) if len(angle) else None,
        "direction_angle_max_deg":float(angle.max()) if len(angle) else None,
        "direction_rows":int(valid.sum()),"direction_floor_mps2":.5*G0_MPS2}


def prepare(inputs,coefficients):
    arrays={};report={};trim_cache=np.load(trim.OUTPUT/"trajectories.npz")
    for cid,x in inputs.items():
        ref=x["ref"];t=ref["t"];cfg=x["config"];model=finite.make_model(cfg,coefficients)
        model.initialize(dataclasses.replace(x["initial"]))
        c=model.coefficients;rates=np.column_stack((-ref["omega_xyz_export_units"][:,2],ref["omega_xyz_export_units"][:,1]))
        limits=np.radians([cfg["aerodynamics"][k] for k in ("horizontal_fin_aoa_limit_deg","vertical_fin_aoa_limit_deg")])
        source_controls=np.asarray([s["controls"] for s in x["run"]["samples"]])[:,[1,0]]
        local=[];active=[];damping=[];gain=[];forces=[];specific=[];normal=[];feedback=[];up=[];right=[];rolloff=[];aero=[];forward=[];mass=[]
        for i,ts in enumerate(t):
            f=model.evaluate(source_state(ref,i),float(ts),(0.,0.));speed=ref["speed_mps"][i]
            a=(f.qbar/f.mass)/model.q_over_mass0;b=a*model.speed0/speed
            local.append(f.local_incidence);active.append(a);damping.append(b)
            gain.append(c.control_gain*(1+c.speed_slope*(speed/model.speed0-1)))
            forces.append((f.drag_n/f.mass,norm(f.wind_lift_force)/f.mass,f.thrust/f.mass))
            specific.append(f.specific_acceleration);tangent=ref["tangent"][i]
            normal.append(np.asarray(f.specific_acceleration)-np.dot(f.specific_acceleration,tangent)*tangent)
            feedback.append(f.feedback_g);up.append(f.wind_basis.up);right.append(f.wind_basis.right)
            aero.append(f.aerodynamic_force);forward.append(body_axes_for_state(source_state(ref,i)).forward);mass.append(f.mass)
            s=math.hypot(*f.local_incidence);rolloff.append(1/(1+(max(s-finite.ALPHA_S,0)/finite.WIDTH)**2))
        local=np.asarray(local);active=np.asarray(active);damping=np.asarray(damping);gain=np.asarray(gain)
        forces=np.asarray(forces);normal=np.asarray(normal);feedback=np.asarray(feedback);up=np.asarray(up);right=np.asarray(right)
        aero=np.asarray(aero);forward=np.asarray(forward);mass=np.asarray(mass)
        mask=window_mask(t)
        if mask.sum()!=73:raise AssertionError((cid,mask.sum()))
        record={"rows":len(t),"interior_original_rows":int(mask.sum()),"interior_original_support_s":[float(t[mask][0]),float(t[mask][-1])],
            "axis_order":"pitch,yaw","methods":{},"source_roll_rate_0p5_3p5_rad_s_RMS_max_abs":[rms(ref["omega_xyz_export_units"][mask,0]),float(np.max(abs(ref["omega_xyz_export_units"][mask,0])))],
            "scalar_components":{key:{"RMS_residual_mps2":rms((forces[:,j]-ref[key])[mask]),
                "bias_mps2":float((forces[:,j]-ref[key])[mask].mean())} for j,key in enumerate(("acc_drag","acc_lift","acc_thrust"))},
            "component_direction_observability":"Source acc_drag/acc_lift/acc_thrust are scalars; their separate vector directions are not exported or asserted."}
        trim_path=trim.OUTPUT/"full_runs"/(cid+".json");cached=json.loads(trim_path.read_text())
        tt=np.array([s["time_s"] for s in cached["ticks"]]);ut=np.array([s["normalized_control"] for s in cached["ticks"]])
        common=mask&(t>=tt[0])&(t<=tt[-1])
        pid=np.column_stack([np.interp(t[common],tt,ut[:,j]) for j in range(2)])
        tick_t=np.arange(math.floor(t[-1]*48)+1)/48.
        inactive=np.asarray([interpolate_table(ts+1/48.,cfg["guidance"]["flight_time_gain_table"])<=1e-12 for ts in tick_t])
        for width in (3,5):
            observed=gap.derivative(rates,t,width)
            target=local+(observed+damping[:,None]*c.rate_damping*rates)/(active[:,None]*c.restoring_gain)
            held=(observed+active[:,None]*c.restoring_gain*local+damping[:,None]*c.rate_damping*rates)/gain[:,None]
            normalized=target/limits;closed=[];torque=[]
            for i,ts in enumerate(t):
                f=model.evaluate(source_state(ref,i),float(ts),tuple(held[i]));closed.append(f.angular_acceleration)
                torque.append(max(abs(f.inertia*q-l-z) for q,l,z in zip(f.angular_acceleration,f.lever_moment,f.reference_couple)))
            source_specific=gap.derivative(ref["velocity_xyz_mps"],t,width)+np.array([0.,model.gravity,0.])
            source_normal=source_specific-np.sum(source_specific*ref["tangent"],axis=1)[:,None]*ref["tangent"]
            source_aero=source_specific*ref["mass"][:,None]-ref["thrust"][:,None]*forward
            source_feedback=np.column_stack((np.sum(source_normal*up,axis=1),np.sum(source_normal*right,axis=1)))/G0_MPS2
            inactive_held=np.column_stack([np.interp(tick_t[inactive],t,held[:,j]) for j in range(2)])
            method={"caps":range_and_caps(t,held,target,limits),
                "forward_angular_closure_max_abs_rad_s2":float(np.max(abs(np.asarray(closed)-observed))),
                "single_station_torque_closure_max_abs_nm":float(max(torque)),
                "target_norm_minus_source_control_RMS_pitch_yaw": [rms((normalized-source_controls)[mask,j]) for j in range(2)],
                "target_norm_minus_cached_trim_PID_RMS_pitch_yaw":[rms(normalized[common,j]-pid[:,j]) for j in range(2)],
                "force_direction_and_magnitude":vector_difference(normal,source_normal,mask),
                "independently_reconstructed_aerodynamic_vector":{
                    "definition":"F_aero_source=m_source*(dv/dt-raw_gravity_vector)-source_thrust*source_body_forward",
                    "vector_RMS_residual_N":rms(np.linalg.norm((aero-source_aero)[mask],axis=1)),
                    "magnitude_RMS_residual_N":rms((np.linalg.norm(aero,axis=1)-np.linalg.norm(source_aero,axis=1))[mask]),
                    "direction_and_acceleration_residual":vector_difference(aero/mass[:,None],source_aero/ref["mass"][:,None],mask),
                    "assumptions":"Thrust acts along source quaternion forward; exported source mass/thrust and raw world gravity apply; no lift/drag scalar direction invented"},
                "inactive_gain_inverse_demand":{
                    "max_abs_old_held_rad_pitch_yaw":np.max(abs(inactive_held),axis=0).tolist(),
                    "nonzero_ticks_above_1e_10_rad":int(np.sum(np.max(abs(inactive_held),axis=1)>1e-10)),
                    "all_original_inactive_ticks":int(inactive.sum()),
                    "replay_policy":"Preserve original inactive zero, even when noncausal derivative stencil demands input; no anticipation or clock shift"},
                "feedback_RMS_residual_pitch_yaw_G":[rms((feedback-source_feedback)[mask,j]) for j in range(2)],
                "feedback_bias_pitch_yaw_G":[float((feedback-source_feedback)[mask,j].mean()) for j in range(2)],
                "feedback_pitch_sign_disagreement_rows":int(np.sum(mask&(feedback[:,0]*source_feedback[:,0]<0))),
                "source_total_normal_specific_force_method":"Original-clock velocity derivative + raw gravity; tangent removed; includes normal thrust, not acc_lift alone"}
            record["methods"][str(width)+"_point"]=method
            for k,v in {"omega_dot_rad_s2":observed,"old_held_req_rad":held,"target_req_rad":target,"u_norm_req":normalized,
                "source_normal_specific_mps2":source_normal,"source_aerodynamic_force_N":source_aero,"source_feedback_pitch_yaw_G":source_feedback}.items():arrays[cid+f"__{width}__"+k]=v
        record["derivative_stencil_sensitivity_RMS_u_norm_pitch_yaw"]= [rms((arrays[cid+"__5__u_norm_req"]-arrays[cid+"__3__u_norm_req"])[mask,j]) for j in range(2)]
        for k,v in {"t":t,"source_controls_pitch_yaw":source_controls,"rates_rad_s":rates,"local_incidence_rad":local,
            "A":active,"B":damping,"Ku_V":gain,"model_feedback_pitch_yaw_G":feedback,"model_normal_specific_mps2":normal,
            "model_component_drag_lift_thrust_mps2":forces,"rolloff":np.asarray(rolloff),
            "cached_trim_PID_t":tt,"cached_trim_PID_u_norm":ut}.items():arrays[cid+"__"+k]=v
        report[cid]=record
    payload={"status":"inverse_data_prepared_before_any_new_integrations","primary_input_stencil":PRIMARY_WIDTH,
        "cases":report,"fits":0,"source_rows":sum(len(x["ref"]["t"]) for x in inputs.values()),
        "old_held_semantics":"old frozen angular-law actuator input in radians, multiplied by Ku(V); not 40deg target angle",
        "target_semantics":"same-state equivalent trim target in radians; normalized by raw finsAoa only for interface comparison",
        "source_omega_mapping":"pitch=-omega.z,yaw=omega.y in rad/s; no scale/clock shift",
        "limits":"No inverse input clipping; old held cap and equivalent target cap are different constraints",
        "limitations":["3/5 point stencil difference is numerical sensitivity, not a confidence interval",
            "Source end two points and launch/gain-transition stencil halo have lower derivative reliability",
            "Frozen two-axis angular law omits roll dynamics and cross-axis rigid-body terms; source roll is reported",
            "Conditional force direction uses display velocity derivatives; scalar acc_lift direction is not observable"]}
    OUTPUT.mkdir(parents=True,exist_ok=True);np.savez_compressed(OUTPUT/"inverse_inputs.npz",**arrays)
    slew.write_json(OUTPUT/"same_state_inverse.json",payload)
    print(json.dumps({cid:{"caps_5":r["methods"]["5_point"]["caps"],"closure":r["methods"]["5_point"]["forward_angular_closure_max_abs_rad_s2"],"force":r["methods"]["5_point"]["force_direction_and_magnitude"],"feedback_residual_G":r["methods"]["5_point"]["feedback_RMS_residual_pitch_yaw_G"]} for cid,r in report.items()}),flush=True)


class InverseInputGame(finite.FiniteForceGame):
    """Frozen finite-force plant with an exogenous time command, not a PID fit."""
    def __init__(self,config,coefficients,t,held):
        super().__init__(config,request_slew_g_s=110.,drag_coefficients=coefficients,
            coefficients=SurfaceCoefficients(),force_basis="wind_linear",plant_substeps=4,loft_cap_unit="rad_s")
        self.input_t=np.asarray(t);self.input_held=np.asarray(held)
    def request(self,state,target,time_s,evaluation):
        original,details=super().request(state,target,time_s,evaluation)
        self.input_clock=time_s;self.input_axis=0
        return original,{**details,"request_is_diagnostic_only":True,"external_inverse_input":True}
    def _pid(self,error,memory,pid,fin_limit,gain):
        t=self.input_clock
        if not self.input_t[0]-1e-9<=t<=self.input_t[-1]+1e-9:raise ValueError("Inverse input extrapolation forbidden")
        j=self.input_axis;self.input_axis+=1
        # Inverse old held is the actual gain-after input. Original run still
        # keeps it zero while inactive; no double multiplication by gain.
        return float(np.interp(t,self.input_t,self.input_held[:,j])),memory
    def run(self,initial_state,target_at,*,end_time_s):
        result=super().run(initial_state,target_at,end_time_s=end_time_s)
        previous=(0.,0.);bytime={s["time_s"]:s for s in result["samples"]}
        for tick in result["ticks"]:
            tick.update(evaluated_old_held_rad=previous,held_fin_rad_semantics="unclipped old held inverse; original Ku(V) actuator interface; radians")
            bytime[tick["time_s"]]["evaluated_old_held_rad"]=previous;previous=tuple(tick["held_fin_rad"])
        for s in result["samples"]:s.setdefault("evaluated_old_held_rad",previous)
        slew.record_model(result,self)
        result["model"].update(exogenous_inverse_input=True,PID_used=False,primary_stencil=PRIMARY_WIDTH,
            interpolation="linear on unchanged source times; held ZOH at original 48Hz ticks",
            gain_lifecycle="original inactive zero, active inverse is post-gain held; no gain repeated",inverse_clipped=False,
            blind_prediction=False,source_state_injected_after_creation=False,input_support_s=[float(self.input_t[0]),float(self.input_t[-1])])
        if result["summary"]["termination_event"]=="lifetime" and end_time_s<self.config["performance"]["lifetime_s"]:
            result["summary"]["termination_event"]="reference_support_end"
        return result


def saved_closure(model,result):
    residual={"angular_record_rad_s2":0.,"couple_record_nm":0.,"same_station_nm":0.,"aerodynamic_force_N":0.}
    for s in result["samples"]:
        state=finite.sample_state(s);f=model.evaluate(state,s["time_s"],tuple(s["evaluated_old_held_rad"]))
        check={"angular_record_rad_s2":max(abs(a-b) for a,b in zip(f.angular_acceleration,s["angular_acceleration_rad_s2"])),
            "couple_record_nm":max(abs(a-b) for a,b in zip(f.reference_couple,s["reference_couple_nm"])),
            "same_station_nm":max(abs(f.inertia*q-l-z) for q,l,z in zip(f.angular_acceleration,f.lever_moment,f.reference_couple)),
            "aerodynamic_force_N":norm(tuple(a-l-d for a,l,d in zip(f.aerodynamic_force,f.wind_lift_force,f.wind_drag_force)))}
        for k,v in check.items():residual[k]=max(residual[k],v)
    if max(residual.values())>1e-8:raise AssertionError(residual)
    return residual


def plots(cid,x,base,candidate,z):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,3,figsize=(14,8))
    curves=(("Jari reference",x["observed"],"#24566a"),("cached linear force S110",base["request_slew"],"#999"),
        ("cached finite force",base["finite_force"],"#bb762c"),("external inverse old-held input",candidate,"#168c72"))
    for ax,(key,label) in zip(axes.flat,(("speed_mps","Speed (m/s)"),("alpha_deg","AoA (deg)"),
        ("normal_specific_load_g","Normal specific load (G)"),("body_yaw_rate_deg_s","Body yaw rate (deg/s)"),("body_yaw_deg","Body yaw (deg)"))):
        for name,p,col in curves:ax.plot(p["t"],p[key],lw=1.3,color=col,label=name)
        ax.set(xlabel="Original time (s)",ylabel=label,xlim=(0,3.5 if key in ("alpha_deg","body_yaw_rate_deg_s","body_yaw_deg") else x["ref"]["t"][-1]));ax.grid(alpha=.2)
    for _,p,col in curves:axes[1,2].plot(p["t"],p["position_xyz_m"][:,1]-x["ref"]["position_xyz_m"][0,1],color=col,lw=1.3)
    axes[1,2].set(xlabel="Original time (s)",ylabel="Y displacement (m)");axes[1,2].grid(alpha=.2);axes[0,0].legend(fontsize=7)
    fig.suptitle(cid+": frozen plant, unclipped 5-point inverse old-held input; diagnostic replay");fig.tight_layout(rect=(0,0,1,.96));fig.savefig(OUTPUT/(cid+"_overlay.png"),dpi=150);plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(12,7));t=z[cid+"__t"]
    for j,axis in enumerate(("pitch","yaw")):
        for width,col in ((3,"#999"),(5,"#168c72")):axes[0,j].plot(t,z[cid+f"__{width}__u_norm_req"][:,j],color=col,label=f"inverse {width}-point")
        axes[0,j].plot(t,z[cid+"__source_controls_pitch_yaw"][:,j],color="#24566a",label="source controls")
        axes[0,j].plot(z[cid+"__cached_trim_PID_t"],z[cid+"__cached_trim_PID_u_norm"][:,j],color="#bb762c",label="cached trim PID")
        axes[0,j].axhline(1.,color="#aaa",ls=":");axes[0,j].axhline(-1.,color="#aaa",ls=":")
        axes[0,j].set(ylabel=axis+" target normalization",xlim=(0,3.5))
        for width,col in ((3,"#999"),(5,"#168c72")):axes[1,j].plot(t,np.degrees(z[cid+f"__{width}__old_held_req_rad"][:,j]),color=col,label=f"old held {width}-point")
        axes[1,j].set(ylabel=axis+" old held input (deg)",xlim=(0,3.5),xlabel="Original time (s)")
    for ax in axes.flat:ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.suptitle(cid+": frozen-plant inverse input; target and old-held meanings kept separate");fig.tight_layout(rect=(0,0,1,.95));fig.savefig(OUTPUT/(cid+"_required_control.png"),dpi=150);plt.close(fig)


def replay(inputs,coefficients):
    z=np.load(OUTPUT/"inverse_inputs.npz");conditional=json.loads((OUTPUT/"same_state_inverse.json").read_text())
    if conditional["primary_input_stencil"]!=PRIMARY_WIDTH:raise ValueError("Primary inverse stencil changed")
    full=OUTPUT/"full_runs";full.mkdir(exist_ok=True)
    protected=[ROOT/"src/aim120_model/single_surface_game.py",Path(finite.__file__),Path(slew.__file__),Path(drag.__file__),comparison.SNAPSHOT,comparison.FROZEN,
        comparison.DEFAULT_MANIFEST,finite.OUTPUT/"fit.json",finite.OUTPUT/"trajectories.npz",slew.OUTPUT/"trajectories.npz",drag.OUTPUT/"fit.json",trim.OUTPUT/"results.json",trim.OUTPUT/"trajectories.npz",Path(trim.__file__)]
    protected.extend(p for folder in (ROOT/"outputs/jari_offaxis/reference",finite.OUTPUT/"full_runs",slew.OUTPUT/"full_runs",trim.OUTPUT/"full_runs") for p in folder.iterdir() if p.is_file())
    before={str(p):(p.stat().st_size,p.stat().st_mtime_ns) for p in protected}
    fcache=np.load(finite.OUTPUT/"trajectories.npz");scache=np.load(slew.OUTPUT/"trajectories.npz");tcache=np.load(trim.OUTPUT/"trajectories.npz")
    records={};arrays={};runs={};new=0;recovered=0
    for cid in CASES:
        x=inputs[cid];cfg=x["config"];ref=x["ref"];t=z[cid+"__t"];held=z[cid+f"__{PRIMARY_WIDTH}__old_held_req_rad"]
        if not np.array_equal(t,ref["t"]):raise ValueError("Original inverse clock changed")
        end=min(float(t[-1]),float(cfg["performance"]["lifetime_s"]))
        model=InverseInputGame(cfg,coefficients,t,held);path=full/(cid+".json")
        if path.exists():result=json.loads(path.read_text());model.initialize(dataclasses.replace(x["initial"]));recovered+=1
        else:
            result=model.run(dataclasses.replace(x["initial"]),x["target_at"],end_time_s=end);new+=1;slew.write_json(path,result,True)
        predicted=fields.prediction_fields(result,model,x["run"]["scenario"])
        base={"request_slew":drag.selected_arrays(scache,cid+"__selected__"),"finite_force":drag.selected_arrays(fcache,cid+"__finite_force__")}
        trim_prediction=drag.selected_arrays(tcache,cid+"__trim_target__")
        record={"summary":result["summary"],"input_support_s":[float(t[0]),float(t[-1])],"requested_end_s":end,
            "saved_force_and_torque_closure":saved_closure(model,result),
            "metrics_separate_units":{name:drag.pair_metrics(x["observed"],p,predicted,tuple(finite.UNITS),trim.WINDOWS) for name,p in {**base,"cached_trim_target":trim_prediction}.items()},
            "same_time_end":trim.endpoint(x["observed"],{**base,"cached_trim_target":trim_prediction},predicted),
            "events":drag.event_summary(ref,predicted,result,x["run"]["summary"]),
            "response_extrema":{name:trim.response_extrema(p) for name,p in {"reference":x["observed"],"inverse_input":predicted}.items()},
            "body_rate_events":{name:finite.rate_events(p) for name,p in {"reference":x["observed"],"inverse_input":predicted}.items()},
            "full_result_path":str(path),"source_state_injected_after_creation":False}
        record["same_time_end"]["inverse_input"]=record["same_time_end"].pop("trim_target")
        for w in record["metrics_separate_units"].values():
            for m in w.values():
                for key,v in m["metrics"].items():v["units"]=finite.UNITS[key]
        records[cid]=record;runs[cid]=(model,result,x)
        fields.save_arrays(arrays,cid+"__reference",x["observed"]);fields.save_arrays(arrays,cid+"__inverse_input",predicted)
        print(json.dumps({"completed":cid,"summary":result["summary"],"turn_RMS":{k:v["new"]["rmse"] for k,v in record["metrics_separate_units"]["finite_force"]["turn_0p7_3p5"]["metrics"].items()}}),flush=True)
    # One final focused pass on saved runs; no integration or new fit.
    protected_ok=before=={str(p):(p.stat().st_size,p.stat().st_mtime_ns) for p in protected};checks={}
    for cid,(model,result,x) in runs.items():
        ts=np.array([s["time_s"] for s in result["ticks"]]);last=result["samples"][-1]
        sep=norm(sub(x["target_at"](last["time_s"]).position,tuple(last["position_m"])))
        boundary=abs(sep-10.) if result["summary"]["termination_event"]=="proximity_fuse" else None
        fixed=model.config==x["config"] and model.coefficients==SurfaceCoefficients() and model.drag_coefficients==coefficients and model.plant_substeps==4 and model.arm==.4 and model.relative_blend==.5 and model.request_slew_g_s==110. and [model.amplitude_coefficient,model.transition_coefficient]==[.75,1.3]
        noextrap=max(ts)<=model.input_t[-1]+1e-9 and last["time_s"]<=model.input_t[-1]+1e-9
        prescribed=max(abs(np.asarray(tick["held_fin_rad"])-(np.array([np.interp(tick["time_s"],model.input_t,model.input_held[:,j]) for j in range(2)]) if tick["gain"]>1e-12 else np.zeros(2))).max() for tick in result["ticks"])
        checks[cid]={"frozen_parameters":fixed,"controller_clock_48Hz":bool(np.max(abs(ts*48-np.round(ts*48)))<1e-8),
            "no_input_extrapolation":bool(noextrap),"external_ZOH_command_max_abs_error_rad":float(prescribed),
            "fuse_boundary_error_m":boundary,"termination":result["summary"]["termination_event"],
            "saved_arrays_finite":all(np.isfinite(v).all() for k,v in arrays.items() if k.startswith(cid+"__")),"creation_only":result["summary"]["creation_state_only"]}
        if not(fixed and noextrap and checks[cid]["controller_clock_48Hz"] and prescribed<1e-12 and checks[cid]["saved_arrays_finite"] and (boundary is None or boundary<.001)):raise AssertionError(checks[cid])
    verification={"protected_files_size_mtime_unchanged":protected_ok,"core_config_raw_docs_GUI_changes":0,
        "new_integrations_this_invocation":new,"recovered_without_integrating":recovered,"unique_new_free_runs":len(runs),"native_runs":0,
        "baseline_integrations":0,"Jari_API_calls":0,"fits_or_scans":0,"source_rows":conditional["source_rows"],"conditional_arrays_finite":all(np.isfinite(z[k]).all() for k in z.files),
        "source_clock_unchanged":True,"plant_substeps":4,"source_state_injected_after_creation":False,"inverse_clipped":False,"checks":checks}
    if not(protected_ok and new+recovered==3 and len(runs)==3 and conditional["source_rows"]==631 and verification["conditional_arrays_finite"]):raise AssertionError(verification)
    np.savez_compressed(OUTPUT/"trajectories.npz",**arrays)
    for cid in CASES:
        base={"request_slew":drag.selected_arrays(scache,cid+"__selected__"),"finite_force":drag.selected_arrays(fcache,cid+"__finite_force__")}
        prefix=cid+"__inverse_input__"
        plots(cid,inputs[cid],base,{k[len(prefix):]:v for k,v in arrays.items() if k.startswith(prefix)},z)
    slew.write_json(OUTPUT/"results.json",{"status":"three_fixed_plant_inverse_input_replays_complete","same_state":conditional,"cases":records,
        "verification":verification,"composite_score":False,"diagnostic_reference_conditioned_replay":True,"blind_prediction":False,
        "input_definition":"5-point source-omega derivative -> old Ku(V) held inverse at source state; original-time linear interpolation -> 48Hz ZOH; unclipped",
        "limitations":["Correct same-state moment input is not state injection or proof of force correctness",
            "48Hz controls reconstructed from 24Hz display with boundary derivative uncertainty; no launch/gain shift",
            "Original flight-gain lifecycle retained; source derivative halo may demand nonzero input before activation",
            "Roll dynamics omitted in frozen plant; finite-difference force vectors include display-cadence uncertainty",
            "Target normalization above1 does not imply old-held cap exceeded; these are distinct interfaces",
            "No closed-loop or native-game acceptance claimed"]})
    print(json.dumps({"output":str(OUTPUT),"verification":verification}),flush=True)


def main():
    inputs=source_inputs();coefficients=drag.DragCoefficients(**json.loads((drag.OUTPUT/"fit.json").read_text())["coefficients"])
    if len(sys.argv)!=2 or sys.argv[1] not in ("--prepare","--replay"):raise SystemExit("Use --prepare, then --replay; inverse data must be saved before integration")
    if sys.argv[1]=="--prepare":prepare(inputs,coefficients)
    else:replay(inputs,coefficients)


if __name__=="__main__":main()
