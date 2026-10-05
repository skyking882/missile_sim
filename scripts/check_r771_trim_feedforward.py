#!/usr/bin/env python3
"""Two fixed additive static-trim/PID interface hypotheses for a game plant.

Only script-local candidate classes change. A: raw-G PID in old-held radians,
with existing high-angle I transport. B: SI-error PID in normalized target
correction coordinates, without I transport. Both retain Ku and all forces,
clip the total old-held output at raw finsAoa, and use total-output AW.
"""
from __future__ import annotations
import copy
import dataclasses
import json
import math
import os
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.dont_write_bytecode=True
sys.path[:0]=[str(ROOT/"scripts"),str(ROOT/"src")]
os.environ.setdefault("MPLCONFIGDIR","/tmp/missle_sim_mpl")
import numpy as np
import analyze_r771_game_request_inverse as audit
import check_r771_control_authority as authority
from fit_r771_jari_finite_force import comparison, fields, drag, slew, harness
from aim120_model.aerodynamics import body_axes_for_state
from aim120_model.control import scheduled_pid_values
from aim120_model.guidance import interpolate_table
from aim120_model.math3d import dot, norm, sub
from aim120_model.single_surface_game import SurfaceCoefficients
from aim120_model.units import G0_MPS2

OUT=ROOT/"outputs/jari_offaxis/trim_feedforward"
CASES=audit.inverse.CASES
KINDS=("A_old_G","B_target_SI")
WINDOWS=(*audit.WINDOWS,("active",14/48,3.5))
DT=1/48
TERM_KEYS=("trim_old_rad","correction_old_rad","total_prelimit_old_rad","held_old_rad",
    "P_correction_old_rad","I_provisional_correction_old_rad","D_correction_old_rad",
    "I_saved_correction_old_rad","I_saved_coordinate","I_previous_coordinate","I_transported_coordinate",
    "I_increment_coordinate","integral_transport_ratio","antiwindup_blocked","total_output_clipped",
    "derivative_error_per_s","input_error","correction_old_per_coordinate","output_closure_rad")


def lower_branch_magnitude(desired,alpha_s,width):
    peak_s=math.hypot(alpha_s,width)
    peak_f=peak_s/(1+((peak_s-alpha_s)/width)**2)
    capped=min(desired,peak_f)
    if capped<=alpha_s:target_s=capped
    elif capped==peak_f:target_s=peak_s
    else:
        b=2*capped*alpha_s+width*width
        discriminant=max(0.,b*b-4*capped*capped*(alpha_s*alpha_s+width*width))
        target_s=2*capped*(alpha_s*alpha_s+width*width)/(b+math.sqrt(discriminant))
    return target_s,capped,peak_s,peak_f


def trim_solution(model,state,f,request):
    """Static current-state force inverse on the continuous lower branch.

    Current thrust projection is held fixed. No current restoring/damping
    cancellation, observed alpha/control, required-input inverse, or future
    derivative enters this calculation.
    """
    axes=body_axes_for_state(state)
    thrust_normal=np.array([dot(axes.forward,b) for b in (f.wind_basis.up,f.wind_basis.right)])*f.thrust/f.mass
    pressure_slope=f.qbar*model.area*model.coefficients.ca/f.mass
    linear=(np.asarray(request)*G0_MPS2-thrust_normal)/pressure_slope
    desired=float(np.linalg.norm(linear));alpha_s=audit.fix.finite.ALPHA_S;width=audit.fix.finite.WIDTH
    target_s,capped,peak_s,peak_f=lower_branch_magnitude(desired,alpha_s,width)
    attainable=desired<=peak_f+1e-12
    local=linear*(target_s/desired) if desired>0 else np.zeros(2)
    ratio=1/(1+(max(target_s-alpha_s,0)/width)**2)
    achieved=(pressure_slope*local*ratio+thrust_normal)/G0_MPS2
    speed=norm(sub(state.velocity,model.wind));a=(f.qbar/f.mass)/model.q_over_mass0;c=model.coefficients
    ku=c.control_gain*(1+c.speed_slope*(speed/model.speed0-1));k=ku/(a*c.restoring_gain)
    ff=local/k
    expected=(pressure_slope*linear*(capped/desired)+thrust_normal)/G0_MPS2 if desired>0 else thrust_normal/G0_MPS2
    return {"trim_old_rad":ff,"local_target_rad":local,"K":k,"range_capped":not attainable,
        "requested_G":np.asarray(request),"branch_achieved_G":achieved,"branch_request_residual_G":achieved-np.asarray(request),
        "branch_force_closure_G":achieved-expected,"local_peak_rad":peak_s,"linear_peak_rad":peak_f,
        "static_equilibrium_moment_closure_rad_s2":ku*ff-a*c.restoring_gain*local}


class TrimControl:
    def __init__(self,kind):
        self.kind=kind;self.high=False;self.previous_k=[None,None];self.terms=[None,None]

    def step(self,axis,error_g,memory,pid,limit,gain,k,nonlinear,ff):
        old_i,previous_error,previous_derivative=memory
        self.high=self.high or bool(nonlinear)
        if self.kind=="A_old_G":
            ratio=self.previous_k[axis]/k if self.high and self.previous_k[axis] is not None else 1.
            integral=audit.clip(old_i*ratio,pid["integral_limit"]) if self.high else old_i
            error=error_g;conversion=1.
        else:
            ratio=1.;integral=old_i;error=error_g*G0_MPS2;conversion=limit/k
        increment=pid["i"]*error*DT
        provisional=audit.clip(integral+increment,pid["integral_limit"])
        derivative=previous_derivative+DT/(1e-9+DT)*((error-previous_error)/DT-previous_derivative)
        p=pid["p"]*error;d=pid["d"]*derivative
        correction=conversion*(p+provisional+d);total=ff+correction
        clipped=audit.clip(total,limit)
        blocked=abs(clipped)>=.99*limit and error*clipped>0
        next_i=integral if blocked else provisional
        held=gain*clipped
        if self.high:self.previous_k[axis]=k
        self.terms[axis]={"trim_old_rad":ff,"correction_old_rad":correction,"total_prelimit_old_rad":total,
            "held_old_rad":held,"P_correction_old_rad":conversion*p,"I_provisional_correction_old_rad":conversion*provisional,
            "D_correction_old_rad":conversion*d,"I_saved_correction_old_rad":conversion*next_i,
            "I_saved_coordinate":next_i,"I_previous_coordinate":old_i,"I_transported_coordinate":integral,
            "I_increment_coordinate":increment,"integral_transport_ratio":ratio,"antiwindup_blocked":blocked,
            "total_output_clipped":abs(total-clipped)>1e-12,"derivative_error_per_s":derivative,
            "input_error":error,"correction_old_per_coordinate":conversion,
            "output_closure_rad":held-gain*audit.clip(ff+conversion*(p+provisional+d),limit)}
        return held,(next_i,error,derivative)


class TrimFeedforwardGame(audit.fix.ControlFixGame):
    def __init__(self,config,*,trim_kind,**kwargs):
        self.trim_kind=trim_kind
        super().__init__(config,**kwargs)

    def initialize(self,state):
        result=super().initialize(state);self.trim_control=TrimControl(self.trim_kind);self.trim_history={}
        return result

    def request(self,state,target,time_s,evaluation):
        request,details=super().request(state,target,time_s,evaluation)
        self.trim_details=trim_solution(self,state,evaluation,request)
        return request,details

    def _pid(self,error,memory,pid,fin_limit,gain):
        axis=self.current_control_axis;self.current_control_axis+=1
        result=self.trim_control.step(axis,error,memory,pid,fin_limit,gain,self.current_control_k,
            self.current_force_nonlinear,float(self.trim_details["trim_old_rad"][axis]))
        if axis==1:self.trim_history[self.current_control_time]={"solution":copy.deepcopy(self.trim_details),
            "terms":copy.deepcopy(self.trim_control.terms),"high_latched":self.trim_control.high}
        return result

    def run(self,initial_state,target_at,*,end_time_s):
        result=super().run(initial_state,target_at,end_time_s=end_time_s)
        for tick in result["ticks"]:
            tick["trim_feedforward_diagnostics"]=self.trim_history.get(tick["time_s"])
            tick["nonlinear_control_latched"]=bool(tick["trim_feedforward_diagnostics"] and tick["trim_feedforward_diagnostics"]["high_latched"])
        result["model"].update(runtime_controller="script-local TrimControl",trim_interface=self.trim_kind,
            controller_mechanism="static command trim plus PID correction; total raw old-held cap and total-output AW",
            experimental_target_cap="none; raw old-held total cap only",integral_transport="A: previousK/currentK after causal high-angle latch; B: none, normalized correction I",
            raw_integral_limit_units="A old-held radians; B normalized target correction; raw number unchanged",
            memory_integral_units="old-held rad" if self.trim_kind=="A_old_G" else "normalized target correction",
            reference_data_in_runtime=False,static_trim_native_game_identified=False)
        return result


def make_model(cfg,coefficients,kind):
    return TrimFeedforwardGame(cfg,trim_kind=kind,request_slew_g_s=110.,drag_coefficients=coefficients,
        coefficients=SurfaceCoefficients(),force_basis="wind_linear",plant_substeps=4,loft_cap_unit="rad_s")


def snapshot():
    p=authority.snapshot()
    for folder in (authority.OUT,ROOT/"outputs/jari_offaxis/trim_target"):
        p.update({str(f):[f.stat().st_size,f.stat().st_mtime_ns] for f in folder.rglob("*") if f.is_file()})
    return p


def conditional(cid,item,z,baseline,coefficients,*,kinds=KINDS,project_request=False):
    cfg=item["config"];ref=item["ref"];p=cid+"__"
    t=baseline[p+"t"];gain=baseline[p+"flight_gain"];k=baseline[p+"K"]
    limits=baseline[p+"limits_rad"];request=baseline[p+"request_G"];feedback=baseline[p+"feedback_G"]
    nonlinear=baseline[p+"nonlinear_force_active"]
    model=audit.fix.make_model(cfg,coefficients);model.initialize(dataclasses.replace(item["initial"]))
    controllers={kind:TrimControl(kind) for kind in kinds}
    memories={kind:[(0.,0.,0.),(0.,0.,0.)] for kind in kinds}
    a={"t":t,"K":k,"limits_rad":limits,"request_G":request.copy(),"feedback_G":feedback,"flight_gain":gain,
        "current_baseline_held_rad":baseline[p+"transport_target1__held_rad"]}
    if project_request:a["raw_request_G"]=request
    for key in ("trim_old_rad","local_target_rad","branch_achieved_G","branch_request_residual_G","branch_force_closure_G","static_equilibrium_moment_closure_rad_s2"):
        a[key]=np.zeros((len(t),2))
    a["branch_range_capped"]=np.zeros(len(t),bool)
    for kind in kinds:
        for key in TERM_KEYS:a[kind+"__"+key]=np.zeros((len(t),2))
        a[kind+"__memory"]=np.zeros((len(t),2,3))
    checks={"cached_request_max_abs_G":0.,"cached_feedback_max_abs_G":0.,"cached_K_max_abs":0.}
    for i,ts in enumerate(t):
        state=audit.fix.source_state_at(ref,float(ts));f=model.evaluate(state,float(ts),(0.,0.))
        r,_=model.request(state,item["target_at"](float(ts)),float(ts),f)
        checks["cached_request_max_abs_G"]=max(checks["cached_request_max_abs_G"],float(np.max(abs(np.asarray(r)-request[i]))))
        checks["cached_feedback_max_abs_G"]=max(checks["cached_feedback_max_abs_G"],float(np.max(abs(np.asarray(f.feedback_g)-feedback[i]))))
        checks["cached_K_max_abs"]=max(checks["cached_K_max_abs"],abs(model.current_control_k-k[i]))
        solution=trim_solution(model,state,f,r)
        if project_request:a["request_G"][i]=solution["branch_achieved_G"]
        for key in ("trim_old_rad","local_target_rad","branch_achieved_G","branch_request_residual_G","branch_force_closure_G","static_equilibrium_moment_closure_rad_s2"):
            a[key][i]=solution[key]
        a["branch_range_capped"][i]=solution["range_capped"]
        if gain[i]>1e-12:
            pid=scheduled_pid_values(cfg["control"]["pid"],float(ts)+DT)
            for kind,c in controllers.items():
                for axis in range(2):
                    held,memory=c.step(axis,float(a["request_G"][i,axis]-feedback[i,axis]),memories[kind][axis],pid,float(limits[axis]),
                        float(gain[i]),float(k[i]),bool(nonlinear[i]),float(solution["trim_old_rad"][axis]))
                    memories[kind][axis]=memory
                    for key in TERM_KEYS:a[kind+"__"+key][i,axis]=c.terms[axis][key]
                    a[kind+"__memory"][i,axis]=memory
    checks["branch_inverse_force_max_abs_G"]=float(abs(a["branch_force_closure_G"]).max())
    checks["static_equilibrium_moment_closure_max_abs_rad_s2"]=float(abs(a["static_equilibrium_moment_closure_rad_s2"]).max())
    for kind in kinds:
        checks[kind+"__output_closure_max_abs_rad"]=float(abs(a[kind+"__output_closure_rad"]).max())
        checks[kind+"__old_held_cap_excess_rad"]=float(np.maximum(abs(a[kind+"__held_old_rad"])-limits,0).max())
        checks[kind+"__raw_I_limit_excess"]=float(np.maximum(abs(a[kind+"__I_saved_coordinate"])-cfg["control"]["pid"]["integral_limit"],0).max())
    report={"checks":checks,"raw_PID":cfg["control"]["pid"],"windows":{},"branch_range_all_controller_ticks":{}}
    for name,lo,hi in WINDOWS:
        mask=(gain>1e-12)&audit.inverse.window_mask(t,lo,hi)
        report["branch_range_all_controller_ticks"][name]={"active_ticks":int(mask.sum()),
            "range_capped_ticks":int(np.sum(mask&a["branch_range_capped"])),
            "request_residual_RMS_pitch_yaw_G":[audit.inverse.rms(a["branch_request_residual_G"][mask,j]) for j in range(2)],
            "request_residual_max_abs_pitch_yaw_G":abs(a["branch_request_residual_G"][mask]).max(axis=0).tolist()}
    for width in (3,5):
        mask=baseline[p+f"{width}__diagnostic_mask"]
        needed=baseline[p+f"{width}__needed_old_rad"]
        a[f"{width}__mask"]=mask;a[f"{width}__needed_old_rad"]=needed
        for name,lo,hi in WINDOWS:
            valid=mask&audit.inverse.window_mask(t,lo,hi)
            rec={"ticks":int(valid.sum()),"range_capped_ticks":int(np.sum(valid&a["branch_range_capped"])),
                "branch_request_residual_RMS_pitch_yaw_G":[audit.inverse.rms(a["branch_request_residual_G"][valid,j]) for j in range(2)],
                "axes":{}}
            for axis,label in enumerate(("pitch","yaw")):
                factor=k/limits[axis]
                outputs={"current_baseline":a["current_baseline_held_rad"][:,axis],**{kind:a[kind+"__held_old_rad"][:,axis] for kind in kinds}}
                rec["axes"][label]={}
                for kind,held in outputs.items():
                    diff=factor*(held-needed[:,axis])
                    cell={"RMS_normalized_input_error":audit.inverse.rms(diff[valid]),
                        "bias_normalized_input_error":float(diff[valid].mean()),"wrong_sign_ticks":int(np.sum(valid&(held*needed[:,axis]<0))),
                        "held_normalized_range":[float((factor*held)[valid].min()),float((factor*held)[valid].max())]}
                    if kind in kinds:
                        cell.update(trim_normalized_range=[float((factor*a["trim_old_rad"][:,axis])[valid].min()),float((factor*a["trim_old_rad"][:,axis])[valid].max())],
                            correction_normalized_range=[float((factor*a[kind+"__correction_old_rad"][:,axis])[valid].min()),float((factor*a[kind+"__correction_old_rad"][:,axis])[valid].max())],
                            antiwindup_ticks=int(a[kind+"__antiwindup_blocked"][valid,axis].sum()),
                            total_clipping_ticks=int(a[kind+"__total_output_clipped"][valid,axis].sum()))
                    rec["axes"][label][kind]=cell
            report["windows"][f"{width}pt__{name}"]=rec
    report["representative_ticks"]=[]
    for when in (.5,1.5,3.5):
        i=int(np.argmin(abs(t-when)));factor=k[i]/limits
        report["representative_ticks"].append({"time_s":float(t[i]),"range_capped":bool(a["branch_range_capped"][i]),
            "request_G":request[i].tolist(),"feedback_G":feedback[i].tolist(),"trim_normalized":(factor*a["trim_old_rad"][i]).tolist(),
            "needed5_normalized":(factor*a["5__needed_old_rad"][i]).tolist(),
            "baseline_normalized":(factor*a["current_baseline_held_rad"][i]).tolist(),
            "candidates":{kind:{key.replace("_old_rad","_normalized"):(factor*a[kind+"__"+key][i]).tolist()
                for key in ("correction_old_rad","held_old_rad","P_correction_old_rad","I_provisional_correction_old_rad","D_correction_old_rad","I_saved_correction_old_rad")} for kind in kinds}})
    return a,report


class ProjectedTrimGame(TrimFeedforwardGame):
    """One new causal request-conditioning candidate; the B PID is unchanged."""
    def request(self,state,target,time_s,evaluation):
        raw,details=super().request(state,target,time_s,evaluation)
        projected=tuple(float(x) for x in self.trim_details["branch_achieved_G"])
        # The original lower-branch target realizes this projected request.
        # Reuse that target rather than choosing a different inverse branch.
        self.trim_details["unconditioned_request_G"]=np.asarray(raw)
        self.trim_details["requested_G"]=np.asarray(projected)
        details.update(trim_unconditioned_request_G=raw,trim_projected_request_G=projected,
            trim_range_capped=self.trim_details["range_capped"],
            trim_request_residual_G=tuple(float(x) for x in self.trim_details["branch_request_residual_G"]))
        return projected,details

    def run(self,initial_state,target_at,*,end_time_s):
        result=super().run(initial_state,target_at,end_time_s=end_time_s)
        result["model"].update(trim_interface="B_projected",
            request_conditioning="Post-S110 current-state lower-branch projection feeds BOTH static trim and B SI-error PID",
            integral_transport="none; I coordinate is normalized target correction",reference_data_in_runtime=False)
        return result


def make_projected_model(cfg,coefficients):
    return ProjectedTrimGame(cfg,trim_kind="B_target_SI",request_slew_g_s=110.,drag_coefficients=coefficients,
        coefficients=SurfaceCoefficients(),force_basis="wind_linear",plant_substeps=4,loft_cap_unit="rad_s")


def projected_snapshot():
    protected=snapshot()
    protected.update({str(p):[p.stat().st_size,p.stat().st_mtime_ns] for p in OUT.rglob("*") if p.is_file()
        and not p.relative_to(OUT).parts[0].startswith("projected_")})
    return protected


def projected_conditionals(inputs,coefficients):
    baseline=np.load(authority.OUT/"conditional.npz");old=np.load(OUT/"conditional.npz")
    z=np.load(audit.inverse.OUTPUT/"inverse_inputs.npz");reports={};arrays={}
    for cid in CASES:
        a,r=conditional(cid,inputs[cid],z,baseline,coefficients,kinds=("B_projected",),project_request=True)
        r["projection_and_I_by_window"]={};p=cid+"__"
        for name,lo,hi in WINDOWS:
            mask=(a["flight_gain"]>1e-12)&audit.inverse.window_mask(a["t"],lo,hi)
            raw_error=old[p+"B_target_SI__input_error"]
            new_error=a["B_projected__input_error"]
            old_i=old[p+"B_target_SI__I_saved_coordinate"]
            new_i=a["B_projected__I_saved_coordinate"]
            r["projection_and_I_by_window"][name]={"active_ticks":int(mask.sum()),
                "raw_error_mean_pitch_yaw_mps2":raw_error[mask].mean(axis=0).tolist(),
                "projected_error_mean_pitch_yaw_mps2":new_error[mask].mean(axis=0).tolist(),
                "raw_I_end_pitch_yaw":old_i[mask][-1].tolist(),"projected_I_end_pitch_yaw":new_i[mask][-1].tolist(),
                "raw_I_peak_abs_pitch_yaw":abs(old_i[mask]).max(axis=0).tolist(),
                "projected_I_peak_abs_pitch_yaw":abs(new_i[mask]).max(axis=0).tolist()}
        a["previous_B_held_old_rad"]=old[p+"B_target_SI__held_old_rad"]
        a["previous_B_saved_I_coordinate"]=old[p+"B_target_SI__I_saved_coordinate"]
        # Genuine harness route must close with the standalone conditional step.
        m=make_projected_model(inputs[cid]["config"],coefficients);m.initialize(dataclasses.replace(inputs[cid]["initial"]))
        memories=[(0.,0.,0.),(0.,0.,0.)];held=np.zeros((len(a["t"]),2))
        request_closure=0.
        for i,ts in enumerate(a["t"]):
            state=audit.fix.source_state_at(inputs[cid]["ref"],float(ts));f=m.evaluate(state,float(ts),(0.,0.))
            req,_=m.request(state,inputs[cid]["target_at"](float(ts)),float(ts),f)
            request_closure=max(request_closure,float(np.max(abs(np.asarray(req)-a["request_G"][i]))))
            if a["flight_gain"][i]>1e-12:
                pid=scheduled_pid_values(inputs[cid]["config"]["control"]["pid"],float(ts)+DT)
                for axis in range(2):held[i,axis],memories[axis]=m._pid(float(req[axis]-f.feedback_g[axis]),memories[axis],pid,a["limits_rad"][axis],a["flight_gain"][i])
        r["checks"].update(projected_harness_request_max_abs_G=request_closure,
            projected_harness_held_max_abs_rad=float(np.max(abs(held-a["B_projected__held_old_rad"]))))
        reports[cid]=r;arrays.update({p+key:value for key,value in a.items()})
    np.savez_compressed(OUT/"projected_conditional.npz",**arrays)
    return reports,arrays


FREE_WINDOWS=(("startup",14/48,.7),("platform",.7,2.3),("unloading",2.3,3.5),
    ("early_0_3p5",0.,3.5),("burn_0_7",0.,7.),("full_common",0.,math.inf))


def projected_free_checks(model,result,cfg,initial,target_at):
    limits=np.radians([cfg["aerodynamics"][key] for key in ("horizontal_fin_aoa_limit_deg","vertical_fin_aoa_limit_deg")])
    closure=0.;request_closure=0.;increment_closure=0.;max_i=0.;rawcap=0.;capped=0;active=0
    for tick in result["ticks"]:
        diagnostics=tick.get("trim_feedforward_diagnostics")
        if diagnostics is None:continue
        active+=1;solution=diagnostics["solution"];capped+=int(solution["range_capped"])
        request_closure=max(request_closure,float(np.max(abs(np.asarray(tick["request_g"])-solution["branch_achieved_G"]))))
        pid=scheduled_pid_values(cfg["control"]["pid"],tick["time_s"]+DT)
        for axis,terms in enumerate(diagnostics["terms"]):
            closure=max(closure,abs(tick["held_fin_rad"][axis]-terms["held_old_rad"]),abs(terms["output_closure_rad"]))
            error=(tick["request_g"][axis]-tick["feedback_g"][axis])*G0_MPS2
            if diagnostics.get("entry") is None:
                increment_closure=max(increment_closure,abs(terms["I_increment_coordinate"]-pid["i"]*error*DT))
            else:
                increment_closure=max(increment_closure,abs(terms["I_increment_coordinate"]))
            max_i=max(max_i,abs(terms["I_saved_coordinate"]));rawcap=max(rawcap,abs(tick["held_fin_rad"][axis])-limits[axis])
    times=np.array([tick["time_s"] for tick in result["ticks"]]);last=result["samples"][-1]
    boundary=abs(norm(sub(target_at(last["time_s"]).position,tuple(last["position_m"])))-10.) if result["summary"]["termination_event"]=="proximity_fuse" else None
    fixed=model.config==cfg and model.coefficients==SurfaceCoefficients() and model.drag_coefficients is not None and model.relative_blend==.5 and model.request_slew_g_s==110. and [model.amplitude_coefficient,model.transition_coefficient]==[.75,1.3] and model.plant_substeps==4 and model.controller_hz==48
    checks={"PID_output_closure_max_abs_rad":closure,"trim_and_PID_same_request_max_abs_G":request_closure,
        "projected_error_I_increment_closure_max_abs":increment_closure,"I_saved_max_abs_normalized":max_i,
        "raw_old_held_cap_max_excess_rad":max(0.,rawcap),"clock48":bool(abs(times*48-np.round(times*48)).max()<1e-8),
        "frozen_plant_outer_config":fixed,"force_moment_closure":audit.inverse.saved_closure(model,result),
        "proximity_boundary_error_m":boundary,"active_ticks":active,"range_capped_ticks":capped}
    if not(fixed and checks["clock48"] and max(closure,request_closure,increment_closure,rawcap)<1e-10 and max_i<=cfg["control"]["pid"]["integral_limit"]+1e-12 and (boundary is None or boundary<.001)):
        raise AssertionError(checks)
    return checks


def projected_plot(cid,observed,current,prediction,*,prefix="projected_",label="B_projected"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,3,figsize=(14,8))
    curves=(("Jari reference",observed,"#243f56"),("Cached current baseline",current,"#929699"),(label+" free",prediction,"#168477"))
    for ax,(key,ylabel) in zip(axes.flat,(("speed_mps","Speed (m/s)"),("alpha_deg","AoA (deg)"),
        ("normal_specific_load_g","Normal specific load (G)"),("body_yaw_rate_deg_s","Body yaw rate (deg/s)"),("body_yaw_deg","Body yaw (deg)"))):
        for name,p,color in curves:ax.plot(p["t"],p[key],color=color,lw=1.35,label=name)
        ax.set(xlabel="Original clock (s)",ylabel=ylabel,xlim=(0,observed["t"][-1] if key=="speed_mps" else 3.5));ax.grid(alpha=.2)
        if key!="speed_mps":ax.axvline(.7,color="#aaa",lw=.5);ax.axvline(2.3,color="#aaa",lw=.5)
    for _,p,color in curves:axes[1,2].plot(p["t"],p["position_xyz_m"][:,1]-observed["position_xyz_m"][0,1],color=color,lw=1.35)
    axes[1,2].set(xlabel="Original clock (s)",ylabel="Y displacement (m)",xlim=(0,observed["t"][-1]));axes[1,2].grid(alpha=.2)
    axes[0,0].legend(fontsize=8)
    fig.suptitle(cid+": actual free dynamics; "+label+"\nCreation state / external target only; no source feedback or inverse control in runtime")
    fig.tight_layout(rect=(0,0,1,.93));fig.savefig(OUT/(prefix+cid+"_free.png"),dpi=150);plt.close(fig)


def projected_free_jari(inputs,coefficients,*,prefix="projected_",maker=make_projected_model,label="B_projected"):
    current_cache=np.load(audit.fix.OUTPUT/"trajectories.npz");finite_cache=np.load(audit.fix.finite.OUTPUT/"trajectories.npz")
    directory=OUT/(prefix+"full_runs");directory.mkdir(exist_ok=True);reports={};arrays={};new=0;recovered=0
    metric_fields=tuple(audit.fix.finite.UNITS)
    for cid in CASES:
        x=inputs[cid];cfg=copy.deepcopy(x["config"]);initial=dataclasses.replace(x["initial"]);target_at=x["target_at"]
        model=maker(cfg,coefficients);path=directory/(cid+".json")
        if path.exists():result=json.loads(path.read_text());model.initialize(dataclasses.replace(initial));recovered+=1
        else:
            result=model.run(dataclasses.replace(initial),target_at,end_time_s=cfg["performance"]["lifetime_s"])
            slew.write_json(path,result,True);new+=1
        prediction=fields.prediction_fields(result,model,x["run"]["scenario"])
        if result["summary"]["termination_event"]=="numerical_failure" or not all(np.isfinite(v).all() for v in prediction.values()):
            raise ValueError("Actual free candidate numerical failure: "+cid)
        current=drag.selected_arrays(current_cache,cid+"__control_fix__")
        previous=drag.selected_arrays(finite_cache,cid+"__finite_force__")
        metrics={label:drag.pair_metrics(x["observed"],baseline,prediction,metric_fields,FREE_WINDOWS) for label,baseline in (("current_control_fix",current),("finite_force_legacy",previous))}
        for windows in metrics.values():
            for record in windows.values():
                for key,value in record["metrics"].items():value["units"]=audit.fix.finite.UNITS[key]
        peak={}
        for name,lo,hi in FREE_WINDOWS[:3]:
            peak[name]={curve:{key:harness.extremum(p["t"],p[key],"max",lo,hi) for key in ("alpha_deg","normal_specific_load_g")}
                for curve,p in (("reference",x["observed"]),("current_baseline",current),(label,prediction))}
        checks=projected_free_checks(model,result,cfg,initial,target_at)
        reports[cid]={"summary":result["summary"],"metrics_separate_units":metrics,"peaks_by_window":peak,
            "unfiltered_response_extrema":{curve:audit.fix.trim.response_extrema(p) for curve,p in (("reference",x["observed"]),("current_baseline",current),(label,prediction))},
            "Y_displacement_at_3p5_m":{curve:float(np.interp(3.5,p["t"],p["position_xyz_m"][:,1])-p["position_xyz_m"][0,1]) for curve,p in (("reference",x["observed"]),("current_baseline",current),(label,prediction))},
            "events":drag.event_summary(x["ref"],prediction,result,x["run"]["summary"]),
            "body_rate_events":{curve:audit.fix.finite.rate_events(p) for curve,p in (("reference",x["observed"]),("current_baseline",current),(label,prediction))},
            "checks":checks,"full_result_path":str(path)}
        fields.save_arrays(arrays,cid+"__reference",x["observed"]);fields.save_arrays(arrays,cid+"__current_baseline",current)
        fields.save_arrays(arrays,cid+"__"+label,prediction);projected_plot(cid,x["observed"],current,prediction,prefix=prefix,label=label)
        print(json.dumps({"free_completed":cid,"summary":result["summary"],"checks":checks}),flush=True)
    return reports,arrays,new,recovered


def native_screen(reports):
    # Separate channel checks, never an aggregate score. Keep every bad window.
    checks={};material={}
    for cid,record in reports.items():
        improved=False
        for window in ("platform","unloading","early_0_3p5"):
            for key,metric in record["metrics_separate_units"]["current_control_fix"][window]["metrics"].items():
                tolerances={"speed_mps":.5,"alpha_deg":.5,"normal_specific_load_g":.5,"body_yaw_deg":.5,"body_yaw_rate_deg_s":.5}
                base=metric["old"]["rmse"];new=metric["new"]["rmse"]
                checks[cid+"__"+window+"__"+key]=new<=1.25*base+tolerances[key]
                if key in ("speed_mps","alpha_deg","normal_specific_load_g") and new<.90*base:improved=True
        material[cid]=improved
    return {"rule":"Each Jari case has at least10% improvement in speed/AoA/load on platform/unloading/early; every separate channel in those windows <=1.25baseline +(.5m/s,.5deg,.5G,.5deg/s); no aggregate score",
        "no_major_worsening_checks":checks,"some_material_improvement_by_case":material,
        "expand_native":all(checks.values()) and all(material.values())}


def projected_main():
    before=projected_snapshot();inputs=audit.inverse.source_inputs()
    coefficients=drag.DragCoefficients(**json.loads((drag.OUTPUT/"fit.json").read_text())["coefficients"])
    conditional_reports,conditional_arrays=projected_conditionals(inputs,coefficients)
    print(json.dumps({"projected_conditional_complete":True,"checks":{cid:r["checks"] for cid,r in conditional_reports.items()}}),flush=True)
    free_reports,free_arrays,new,recovered=projected_free_jari(inputs,coefficients)
    decision=native_screen(free_reports)
    finite=all(np.isfinite(v).all() for v in (*conditional_arrays.values(),*free_arrays.values()))
    protected=before==projected_snapshot()
    closure=all(v<=1e-10 for r in conditional_reports.values() for v in r["checks"].values())
    if not(finite and protected and closure):raise AssertionError({"finite":finite,"protected":protected,"closure":closure})
    np.savez_compressed(OUT/"projected_trajectories.npz",**free_arrays)
    doc={"task":"B_projected: same feasible total request feeds static trim and unchangedB PID","conditional":conditional_reports,
        "free_cases":free_reports,"native_decision":decision,
        "native_validation_status":"Not expanded: actual0/30 free trajectories have material speed/attitude/load worsening; fixed candidate rejected as shared model",
        "hypothesis":"Post-original-S110 current-state lower-branch radial projection; no new field/gain/cap. B SI-error, normalized correction I, no K transport, total raw old-held cap and conditionalAW unchanged.",
        "limits":["Static range projection holds current thrust direction fixed; not rotating equilibrium or native game semantics.",
            "Reference-state diagnostic uses source states only outside runtime. Free runs use fixed creation state and original external target function.",
            "Original force/moment correspondence and raw-finsAoa-as-old-held cap remain D hypotheses."],
        "verification":{"protected_prior_files_size_mtime_unchanged":protected,"conditional_closure_le1e10":closure,
            "all_arrays_finite":finite,"unique_new_candidate":1,"unique_free_runs":len(free_reports),"new_free_runs_this_invocation":new,"recovered_free_runs_this_invocation":recovered,
            "baseline_runs":0,"fits_scans_API_calls":0,"source_state_injected_after_creation":False,
            "raw_PID_and_plant_and_outer_unchanged":True,"controller_hz":48,"plant_substeps":4}}
    slew.write_json(OUT/"projected_results.json",doc)
    print(json.dumps({"projected_complete":True,"native_decision":decision,"verification":doc["verification"]}),flush=True)


class BranchGatedGame(audit.fix.ControlFixGame):
    """Original controller until first causal force-peak crossing; one switch."""
    def initialize(self,state):
        result=super().initialize(state)
        self.branch_active=False;self.branch_switch_time=None;self.branch_entries=[None,None]
        self.branch_controller=TrimControl("B_target_SI");self.branch_shadow=[(0.,0.),(0.,0.)]
        self.branch_history={};self.branch_pending=False
        return result

    def request(self,state,target,time_s,evaluation):
        raw,details=super().request(state,target,time_s,evaluation)
        solution=trim_solution(self,state,evaluation,raw)
        self.branch_solution=solution;self.branch_raw=raw;self.branch_feedback=evaluation.feedback_g
        peak=math.hypot(audit.fix.finite.ALPHA_S,audit.fix.finite.WIDTH)
        tick=abs(time_s*48-round(time_s*48))<1e-8
        gain=interpolate_table((round(time_s*48)+1)/48,self.config["guidance"]["flight_time_gain_table"])
        self.branch_pending=not self.branch_active and tick and gain>1e-12 and math.hypot(*evaluation.local_incidence)>peak
        use_projected=self.branch_active or self.branch_pending
        effective=tuple(float(x) for x in solution["branch_achieved_G"]) if use_projected else raw
        self.branch_effective=effective
        axes=body_axes_for_state(state)
        thrust_G=np.array([dot(axes.forward,b) for b in (evaluation.wind_basis.up,evaluation.wind_basis.right)])*evaluation.thrust/evaluation.mass/G0_MPS2
        gravity_G=np.array([dot((0.,self.gravity,0.),b) for b in (evaluation.wind_basis.up,evaluation.wind_basis.right)])/G0_MPS2
        requested_norm=float(np.linalg.norm(np.asarray(raw)-thrust_G))
        scale=min(1.,float(np.linalg.norm(np.asarray(effective)-thrust_G))/requested_norm) if use_projected and requested_norm>0 else 1.
        change=np.asarray(effective)-np.asarray(raw)
        gravity_change=(scale-1)*gravity_G
        remainder_change=change-gravity_change
        details.update(branch_uses_projected_request=use_projected,branch_raw_request_G=raw,
            branch_effective_request_G=effective,branch_local_norm_rad=math.hypot(*evaluation.local_incidence),
            branch_threshold_rad=peak,branch_force_range_scale=scale,
            branch_gravity_projection_G=gravity_G,branch_gravity_request_change_G=gravity_change,
            branch_remaining_request_change_G=remainder_change)
        return effective,details

    def _pid(self,error,memory,pid,fin_limit,gain):
        axis=self.current_control_axis
        projected_error=(self.branch_solution["branch_achieved_G"][axis]-self.branch_feedback[axis])*G0_MPS2
        if not self.branch_active or self.branch_switch_time==self.current_control_time:
            previous_e,previous_d=self.branch_shadow[axis]
            derivative=previous_d+DT/(1e-9+DT)*((projected_error-previous_e)/DT-previous_d)
            self.branch_shadow[axis]=(projected_error,derivative)
        if not self.branch_active and not self.branch_pending:
            return super()._pid(error,memory,pid,fin_limit,gain)
        self.current_control_axis+=1
        entering=self.branch_pending or self.branch_switch_time==self.current_control_time
        if entering:
            low_error=float(self.branch_raw[axis]-self.branch_feedback[axis])
            terms=authority.split.pid_terms(self.game_control,axis,memory,low_error,pid,gain,
                self.current_control_k,self.current_force_nonlinear,fin_limit)
            low_held,low_memory=self.game_control.update_axis(axis,self.current_control_time,low_error,memory,pid,fin_limit,gain,
                self.current_control_k,self.current_force_nonlinear)
            if axis==1:self.control_history[self.current_control_time]=copy.deepcopy(self.game_control.diagnostics)
            conversion=fin_limit/self.current_control_k;ff=float(self.branch_solution["trim_old_rad"][axis])
            p=pid["p"]*projected_error;d=pid["d"]*derivative
            low_pre_gain=low_held/gain
            required_i=(low_pre_gain-ff)/conversion-p-d
            initialized_i=audit.clip(required_i,pid["integral_limit"])
            total=ff+conversion*(p+initialized_i+d);clipped=audit.clip(total,fin_limit);held=gain*clipped
            blocked=abs(clipped)>=.99*fin_limit and projected_error*clipped>0
            updated=(initialized_i,projected_error,derivative)
            self.branch_controller.high=True;self.branch_controller.previous_k[axis]=self.current_control_k
            self.branch_controller.terms[axis]={"trim_old_rad":ff,"correction_old_rad":conversion*(p+initialized_i+d),
                "total_prelimit_old_rad":total,"held_old_rad":held,"P_correction_old_rad":conversion*p,
                "I_provisional_correction_old_rad":conversion*initialized_i,"D_correction_old_rad":conversion*d,
                "I_saved_correction_old_rad":conversion*initialized_i,"I_saved_coordinate":initialized_i,
                "I_previous_coordinate":initialized_i,"I_transported_coordinate":initialized_i,"I_increment_coordinate":0.,
                "integral_transport_ratio":1.,"antiwindup_blocked":blocked,"total_output_clipped":abs(total-clipped)>1e-12,
                "derivative_error_per_s":derivative,"input_error":projected_error,"correction_old_per_coordinate":conversion,
                "output_closure_rad":held-gain*audit.clip(total,fin_limit)}
            self.branch_entries[axis]={"axis":axis,"time_s":self.current_control_time,"low_held_rad":low_held,
                "low_pre_gain_postlimit_rad":low_pre_gain,"low_natural_prelimit_rad":terms["prelimit_rad"],
                "required_I_normalized":required_i,"initialized_I_normalized":initialized_i,
                "I_clipped":abs(required_i-initialized_i)>1e-12,"held_jump_rad":held-low_held,
                "projected_current_error_mps2":projected_error,"projected_current_derivative_mps3":derivative,
                "entry_Ki_increment_applied":False,"entry_I_initializes_output_provisional_and_saved_I":True,
                "retired_low_memory_G_oldrad":low_memory}
            self.branch_active=True;self.branch_switch_time=self.current_control_time
        else:
            held,updated=self.branch_controller.step(axis,error,memory,pid,fin_limit,gain,self.current_control_k,
                self.current_force_nonlinear,float(self.branch_solution["trim_old_rad"][axis]))
        if axis==1:
            self.branch_history[self.current_control_time]={"solution":copy.deepcopy(self.branch_solution),
                "terms":copy.deepcopy(self.branch_controller.terms),"high_latched":True,
                "entry":copy.deepcopy(self.branch_entries) if entering else None}
        return held,updated

    def run(self,initial_state,target_at,*,end_time_s):
        result=audit.fix.ControlFixGame.run(self,initial_state,target_at,end_time_s=end_time_s)
        for tick in result["ticks"]:
            tick["trim_feedforward_diagnostics"]=self.branch_history.get(tick["time_s"])
            tick["branch_active"]=self.branch_switch_time is not None and tick["time_s"]>=self.branch_switch_time
        result["model"].update(runtime_controller="script-local BranchGatedGame",trim_interface="B_branch_gated",
            branch_threshold_rad=math.hypot(audit.fix.finite.ALPHA_S,audit.fix.finite.WIDTH),
            branch_switch_time_s=self.branch_switch_time,branch_entry_axis_diagnostics=self.branch_entries,
            branch_lifecycle="Original ControlFix until first active controller tick above frozen force peak; one-way switch",
            bumpless="Match original POST-limit PRE-flightGain output with B projected trim/P/D and initialized normalized I; report rawIlimit clipping/jump",
            derivative_initialization="Causal projected-error shadow derivative during original control; no shadowI/output influence; save current projectederror/D on entry",
            integral_transport="Original low-regime mechanism; B normalized correction I after switch, no transport",
            memory_integral_units="before switch old-held rad; after switch normalized target correction",
            experimental_target_cap="Original controller target1 before switch; total raw old-held cap only after switch",
            reference_data_in_runtime=False,native_game_controller_identified=False)
        return result


def make_branch_model(cfg,coefficients):
    return BranchGatedGame(cfg,request_slew_g_s=110.,drag_coefficients=coefficients,
        coefficients=SurfaceCoefficients(),force_basis="wind_linear",plant_substeps=4,loft_cap_unit="rad_s")


def branch_snapshot():
    protected=snapshot()
    protected.update({str(p):[p.stat().st_size,p.stat().st_mtime_ns] for p in OUT.rglob("*") if p.is_file()
        and not p.relative_to(OUT).parts[0].startswith("branch_")})
    return protected


def baseline_crossings(inputs,coefficients):
    peak=math.hypot(audit.fix.finite.ALPHA_S,audit.fix.finite.WIDTH);reports={}
    for cid,x in inputs.items():
        result=json.loads((audit.fix.OUTPUT/"full_runs"/(cid+".json")).read_text())
        model=audit.fix.make_model(x["config"],coefficients);model.initialize(dataclasses.replace(x["initial"]))
        values=[]
        for sample in result["samples"]:
            f=model.evaluate(audit.fix.finite.sample_state(sample),sample["time_s"],tuple(sample["evaluated_old_held_rad"]))
            values.append((sample["time_s"],math.hypot(*f.local_incidence)))
        crossings=[ts for ts,local in values if local>peak and abs(ts*48-round(ts*48))<1e-8]
        reports[cid]={"threshold_rad":peak,"first_baseline_crossing_s":crossings[0] if crossings else None,
            "max_baseline_local_norm_deg":math.degrees(max(local for _,local in values))}
    return reports


def branch_main():
    before=branch_snapshot();inputs=audit.inverse.source_inputs()
    coefficients=drag.DragCoefficients(**json.loads((drag.OUTPUT/"fit.json").read_text())["coefficients"])
    crossings=baseline_crossings(inputs,coefficients);print(json.dumps({"baseline_crossings":crossings}),flush=True)
    # Reuse general forward harness; only the one causal candidate is integrated.
    reports,arrays,new,recovered=projected_free_jari(inputs,coefficients,prefix="branch_",maker=make_branch_model,label="B_branch_gated")
    for cid,record in reports.items():
        result=json.loads(Path(record["full_result_path"]).read_text())
        cached=json.loads((audit.fix.OUTPUT/"full_runs"/(cid+".json")).read_text())
        switch=result["model"]["branch_switch_time_s"]
        same={key:all(all(sample[k]==original[k] for k in original) for sample,original in zip(result[key],cached[key])) and len(result[key])==len(cached[key])
            for key in ("samples","ticks")} if switch is None else None
        before_switch={key:all(all(sample[k]==original[k] for k in original) for sample,original in zip(result[key],cached[key]) if sample["time_s"]<switch-1e-10)
            for key in ("samples","ticks")} if switch is not None else None
        entries=result["model"]["branch_entry_axis_diagnostics"]
        # New bookkeeping can be reconstructed from saved candidate states;
        # this does not reintegrate or alter the cached forward trajectory.
        diagnostic_model=make_branch_model(inputs[cid]["config"],coefficients)
        diagnostic_model.initialize(dataclasses.replace(inputs[cid]["initial"]))
        by_time={sample["time_s"]:sample for sample in result["samples"]}
        for tick in result["ticks"]:
            if "branch_gravity_request_change_G" in tick:continue
            sample=by_time[tick["time_s"]];state=audit.fix.finite.sample_state(sample)
            f=diagnostic_model.evaluate(state,tick["time_s"],tuple(sample["evaluated_old_held_rad"]))
            axes=body_axes_for_state(state)
            thrust=np.array([dot(axes.forward,b) for b in (f.wind_basis.up,f.wind_basis.right)])*f.thrust/f.mass/G0_MPS2
            gravity=np.array([dot((0.,diagnostic_model.gravity,0.),b) for b in (f.wind_basis.up,f.wind_basis.right)])/G0_MPS2
            raw=np.asarray(tick["branch_raw_request_G"]);effective=np.asarray(tick["branch_effective_request_G"])
            requested_norm=float(np.linalg.norm(raw-thrust))
            scale=min(1.,float(np.linalg.norm(effective-thrust))/requested_norm) if tick["branch_active"] and requested_norm>0 else 1.
            tick["branch_gravity_request_change_G"]=(scale-1)*gravity
            tick["branch_remaining_request_change_G"]=effective-raw-tick["branch_gravity_request_change_G"]
        projection={}
        for window,lo,hi in FREE_WINDOWS[:3]:
            ticks=[tick for tick in result["ticks"] if lo-1e-9<=tick["time_s"]<=hi+1e-9]
            raw=np.array([tick["branch_raw_request_G"] for tick in ticks]);effective=np.array([tick["branch_effective_request_G"] for tick in ticks])
            delta=effective-raw
            gravity_delta=np.array([tick["branch_gravity_request_change_G"] for tick in ticks])
            remainder=np.array([tick["branch_remaining_request_change_G"] for tick in ticks])
            projection[window]={"ticks":len(ticks),"switched_ticks":sum(tick["branch_active"] for tick in ticks),
                "total_request_change_RMS_pitch_yaw_G":[audit.inverse.rms(delta[:,axis]) for axis in range(2)],
                "total_request_change_bias_pitch_yaw_G":delta.mean(axis=0).tolist(),
                "gravity_slice_change_RMS_pitch_yaw_G":[audit.inverse.rms(gravity_delta[:,axis]) for axis in range(2)],
                "gravity_slice_change_bias_pitch_yaw_G":gravity_delta.mean(axis=0).tolist(),
                "remaining_change_RMS_pitch_yaw_G":[audit.inverse.rms(remainder[:,axis]) for axis in range(2)],
                "gravity_plus_remaining_closure_max_abs_G":float(abs(delta-gravity_delta-remainder).max()),
                "gravity_projection_unchanged_before_total_projection":True,
                "allocation":"Affine radial projection about fixed current thrust: known wind-projected world gravity slice uses(s-1)*g; remainder includesPN/loft/priorcaps/S110/thrust subtraction and is not purePN. Diagnostic allocation at final request boundary."}
        record["branch_diagnostics"]={"switch_time_s":switch,"entries":entries,"never_triggered_exact_cached_original_keys":same,
            "never_triggered_exact_cached_summary":result["summary"]==cached["summary"] if switch is None else None,
            "before_switch_exact_cached_original_keys":before_switch,"total_request_projection_by_axis":projection}
        if switch is not None:
            entry_tick=next(tick for tick in result["ticks"] if abs(tick["time_s"]-switch)<1e-12)
            original_tick=next(tick for tick in cached["ticks"] if abs(tick["time_s"]-switch)<1e-12)
            entry_checks={"low_entry_held_equal_original_cached_tick":all(entry["low_held_rad"]==original_tick["held_fin_rad"][axis] for axis,entry in enumerate(entries)),
                "initialized_error_derivative_I_equal_saved_entry_memory":all(tuple(entry_tick["pid_memory"][axis])==(entry["initialized_I_normalized"],entry["projected_current_error_mps2"],entry["projected_current_derivative_mps3"]) for axis,entry in enumerate(entries)),
                "recorded_held_jump_matches_actual":all(abs((entry_tick["held_fin_rad"][axis]-entry["low_held_rad"])-entry["held_jump_rad"])<1e-12 for axis,entry in enumerate(entries))}
            record["branch_diagnostics"]["entry_checks"]=entry_checks
            if not all(entry_checks.values()):raise AssertionError(entry_checks)
        if (same is not None and not all(same.values())) or (before_switch is not None and not all(before_switch.values())):
            raise AssertionError(record["branch_diagnostics"])
    decision=native_screen(reports)
    decision["rule"]="Three Jari cases: each separate speed/AoA/load/yaw/yawrate platform/unloading/early channel <=1.25baseline+unit tolerance. An exactly preserved never-trigger case is allowed; no aggregate score."
    decision["expand_native"]=all(decision["no_major_worsening_checks"].values())
    protected=before==branch_snapshot();finite=all(np.isfinite(v).all() for v in arrays.values())
    if not(protected and finite):raise AssertionError({"protected":protected,"finite":finite})
    np.savez_compressed(OUT/"branch_trajectories.npz",**arrays)
    doc={"task":"One-way causal switch at frozen normal-force peak, bumpless B_projected initialization",
        "baseline_crossings":crossings,"free_cases":reports,"native_decision":decision,
        "hypothesis":"Original controller before local norm>sqrt(alpha_s²+width²), B_projected after; no time/angle labels or new fitted constant",
        "verification":{"protected_prior_files_size_mtime_unchanged":protected,"all_arrays_finite":finite,
            "unique_candidate":1,"unique_free_runs":len(reports),"new_free_runs_this_invocation":new,"recovered_this_invocation":recovered,
            "baseline_runs":0,"fits_scans_API_calls":0,"source_state_injected_after_creation":False,"controller_hz":48,"plant_substeps":4}}
    slew.write_json(OUT/"branch_results.json",doc)
    print(json.dumps({"branch_complete":True,"native_decision":decision,"verification":doc["verification"]}),flush=True)


def screen(reports):
    result={}
    for kind in KINDS:
        checks={};improvement_by_case={}
        for width in (3,5):
            for cid in ("jari_259_30","jari_259_60"):
                improvements=[]
                for axis in ("pitch","yaw"):
                    cells=reports[cid]["windows"][f"{width}pt__active"]["axes"][axis]
                    base=cells["current_baseline"]["RMS_normalized_input_error"];candidate=cells[kind]["RMS_normalized_input_error"]
                    checks[f"{width}pt_{cid}_{axis}_improves"]=candidate<base
                    improvements.append(candidate<=.95*base)
                improvement_by_case[f"{width}pt_{cid}_material_on_one_axis"]=any(improvements)
            cells=reports["jari_259_0"]["windows"][f"{width}pt__active"]["axes"]["pitch"]
            checks[f"{width}pt_0_pitch_noncatastrophic"]=cells[kind]["RMS_normalized_input_error"]<=1.25*cells["current_baseline"]["RMS_normalized_input_error"]+.01
        checks.update(improvement_by_case)
        result[kind]={"checks":checks,"supports_fixed_free_validation":all(checks.values())}
    selected=next((kind for kind in KINDS if result[kind]["supports_fixed_free_validation"]),None)
    return {"rule":"3/5pt active pitch and yaw both improve in30 and60; at least5% on one axis percase;0 pitch <=1.25baseline+.01. All subwindows still reported; no halo veto. A beforeB if both pass.",
        "candidates":result,"selected_for_six_free":selected}


def plot_conditional(cid,a):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(13,8),sharex=True)
    t=a["t"]
    for axis,label in enumerate(("Pitch","Yaw")):
        factor=a["K"]/a["limits_rad"][axis];ax=axes[0,axis]
        for width,style in ((3,":"),(5,"-")):
            ax.plot(t,np.where(a[f"{width}__mask"],factor*a[f"{width}__needed_old_rad"][:,axis],np.nan),color="#243f56",ls=style,label=f"Needed {width}pt")
        ax.plot(t,factor*a["current_baseline_held_rad"][:,axis],color="#82878b",label="Current baseline")
        for kind,color in (("A_old_G","#bf8436"),("B_target_SI","#167f73")):
            ax.plot(t,factor*a[kind+"__held_old_rad"][:,axis],color=color,label=kind)
            axes[1,axis].plot(t,factor*a[kind+"__correction_old_rad"][:,axis],color=color,label=kind+" correction")
        axes[1,axis].plot(t,factor*a["trim_old_rad"][:,axis],color="#243f56",label="Shared static trim")
        ax.set_title(label);ax.set_ylabel("Total held: K old-held / raw finsAoa")
        axes[1,axis].set_ylabel("Trim and correction in same normalized units")
        for row in range(2):
            x=axes[row,axis];x.grid(alpha=.2);x.axhline(0,color="#aaa",lw=.5);x.set_xlim(14/48,3.5)
            x.axvline(.7,color="#aaa",lw=.5);x.axvline(2.3,color="#aaa",lw=.5)
            x.fill_between(t,0,1,where=a["branch_range_capped"],transform=x.get_xaxis_transform(),color="#d99e9e",alpha=.14)
        axes[1,axis].set_xlabel("Original clock (s)")
    axes[0,0].legend(fontsize=7);axes[1,0].legend(fontsize=7)
    fig.suptitle(cid+": static command trim + PID correction; same reference states, separate histories\nRed shading: requested force outside continuous lower branch; no exact trim inverse")
    fig.tight_layout(rect=(0,0,1,.93));fig.savefig(OUT/(cid+"_conditional.png"),dpi=150);plt.close(fig)


def main():
    before=snapshot();inputs=audit.inverse.source_inputs()
    coefficients=drag.DragCoefficients(**json.loads((drag.OUTPUT/"fit.json").read_text())["coefficients"])
    baseline=np.load(authority.OUT/"conditional.npz");z=np.load(audit.inverse.OUTPUT/"inverse_inputs.npz")
    OUT.mkdir(parents=True,exist_ok=True);reports={};arrays={}
    for cid in CASES:
        a,r=conditional(cid,inputs[cid],z,baseline,coefficients);reports[cid]=r
        arrays.update({cid+"__"+key:value for key,value in a.items()});plot_conditional(cid,a)
    decision=screen(reports)
    finite=all(np.isfinite(v).all() for v in arrays.values())
    closure=all(v<=1e-10 for r in reports.values() for v in r["checks"].values())
    protected=before==snapshot()
    alpha_s=audit.fix.finite.ALPHA_S;width=audit.fix.finite.WIDTH;peak=math.hypot(alpha_s,width)
    examples=(0.,alpha_s/2,alpha_s,(alpha_s+peak)/2,peak)
    branch_magnitude_check=max(abs(lower_branch_magnitude(s/(1+(max(s-alpha_s,0)/width)**2),alpha_s,width)[0]-s) for s in examples)
    if not(finite and closure and protected and branch_magnitude_check<1e-10):raise AssertionError({"finite":finite,"closure":closure,"protected":protected,"branch":branch_magnitude_check})
    doc={"task":"War Thunder static trim plus correction interface","cases":reports,"screen":decision,
        "semantics":{"common":"trim=low-branch force inverse/K; total=gain*clip(trim+correction,raw finsAoa); conditionalAW tests total before gain at .99raw cap",
            "A":"error inG; P/I/D coordinate old-held rad; I rawlimit1rad; high-angle I transport previousK/K; correction old-held=PID sum",
            "B":"error in m/s2; P/I/D and I coordinate normalized target correction; raw I limit1; noK transport; correction old-held=(raw finsAoa/K)*PID sum",
            "output_I_phase":"provisionalI contributes to output; savedI afterAW is separate",
            "gain_placement":"entire trim+correction sum after totalcap is multiplied by original flight gain",
            "force_range":"outside low-branch maximum, radial projection to maximum; report cap/residual, not exact inverse",
            "plant":"Original Ku old-held −A Ka local −B Kw omega unchanged; no current restoring/damping cancellation",
            "dedup":"Old trim_target removes Ku and treats PID output as absolute incidence. Here Ku remains and explicit feedforward+correction is added."},
        "limits":["Static trim is a D game hypothesis; current thrust projection held fixed, not a full rotating equilibrium.",
            "Original 2D local-incidence and wind-force/body-moment correspondence remain assumptions; no native field semantics identified.",
            "No source alpha/controls/future derivatives enter candidate runtime. Needed3/5inputs are noncausal diagnostic comparisons only."],
        "verification":{"finite_arrays":finite,"algorithm_branch_cache_closure_le1e10":closure,"old_files_size_mtime_unchanged":protected,
            "lower_branch_at_zero_linear_join_rolloff_peak_max_error_rad":branch_magnitude_check,
            "raw_PID_and_five_plant_coefficients_unchanged":True,"controller_hz":48,"plant_substeps":4,"fits":0,"free_runs":0}}
    np.savez_compressed(OUT/"conditional.npz",**arrays);slew.write_json(OUT/"results.json",doc)
    print(json.dumps({"conditional_complete":True,"screen":decision,"verification":doc["verification"]}),flush=True)


if __name__=="__main__":
    if sys.argv[1:]==["--projected"]:projected_main()
    elif sys.argv[1:]==["--branch-gated"]:branch_main()
    elif not sys.argv[1:]:main()
    else:raise ValueError("Use --projected for the new fixed candidate")
