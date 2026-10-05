"""Cached War Thunder speed-energy attribution; no new trajectory or fit.

Each free trajectory is decomposed on its own states. Cubic Hermite state
interpolation and Gauss quadrature reconstruct the speed-energy integral;
closure versus saved speed measures interpolation/integration uncertainty.
Source force data and same-source-state frozen model evaluation are separate.
"""
import dataclasses
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src")]
import numpy as np
import check_r771_lift_direction as geometry
from aim120_model.aerodynamics import body_axes_for_state, quaternion_derivative, normalize_quaternion, pitch_yaw_from_quaternion
from aim120_model.drag_models import effective_cda0, area_basis
from aim120_model.math3d import dot

OUT = ROOT / "outputs/jari_offaxis/speed_residual"
CASES = ("jari_259_30", "jari_259_0", "jari_259_60")
CHANNELS = ("thrust_projection", "drag_base", "drag_aoa", "gravity")
WINDOWS = (("startup", 0., .7), ("platform", .7, 2.3), ("unloading", 2.3, 3.5), ("late_powered", 3.5, 7.), ("coast", 7., math.inf))


def stats(a):
    a = np.asarray(a)
    return {"n":len(a), "RMS": np.sqrt(np.mean(a*a, axis=0)).tolist(), "mean": np.mean(a, axis=0).tolist(),
        "max_abs": np.max(abs(a), axis=0).tolist()}


def interp(t, source_t, values):
    values=np.asarray(values)
    if values.ndim==1: return np.interp(t,source_t,values)
    return np.column_stack([np.interp(t,source_t,values[:,j]) for j in range(values.shape[1])])


def cumtrap(t, y):
    y=np.asarray(y); out=np.zeros_like(y)
    out[1:]=np.cumsum((y[1:]+y[:-1])*.5*np.diff(t)[:,None],axis=0)
    return out


def hermite(a,b,da,db,h,u):
    return (2*u**3-3*u*u+1)*a+(u**3-2*u*u+u)*h*da+(-2*u**3+3*u*u)*b+(u**3-u*u)*h*db


def terms(model,state,ts):
    f=model.evaluate(state,ts,(0.,0.));v=np.asarray(state.velocity); speed=np.linalg.norm(v); tangent=v/speed
    axes=body_axes_for_state(state); cosine=dot(axes.forward,tangent)
    atm=model.atmosphere.sample(state.position[1]); mach=speed/atm.speed_of_sound_mps
    r0,k=model.drag_coefficients.factors(mach)
    base=f.qbar*effective_cda0(mach,model.config)*r0
    aoa=f.qbar*area_basis(model.config,"caliber_area")*model.config["aerodynamics"]["cx_vs_aoa"]*k*math.sin(f.alpha_total)**2
    energy=np.array([f.thrust/f.mass*cosine,-base/f.mass,-aoa/f.mass,-model.gravity*tangent[1]])
    closure=float(np.dot(f.acceleration,tangent)-energy.sum())
    return energy,np.asarray(f.acceleration),np.array([base,aoa,f.thrust,f.mass,cosine,speed,f.alpha_total]),closure


def free_case(x,coeff,path):
    receipt=json.loads(path.read_text()); model=geometry.make_model(x["config"],coeff);model.initialize(dataclasses.replace(x["initial"]))
    assert model.wind==(0.,0.,0.)
    states=[geometry.dec.audit.fix.finite.sample_state(s) for s in receipt["samples"]]
    t=np.array([s["time_s"] for s in receipt["samples"]]); velocity=np.array([s.velocity for s in states]); speed=np.linalg.norm(velocity,axis=1)
    records=[terms(model,s,ts) for s,ts in zip(states,t)]
    instant=np.array([r[0] for r in records]); quantities=np.array([r[2] for r in records]); max_law=max(abs(r[3]) for r in records)
    # Only evaluate interpolated saved states. No dynamics step/run is called.
    sums={order:np.zeros((len(t),4)) for order in (4,8)}
    nodes={order:np.polynomial.legendre.leggauss(order) for order in sums}
    burn=model.propulsion.burn_time_s
    for i in range(len(t)-1):
        ta,tb=t[i:i+2];h=tb-ta;s0,s1=states[i:i+2]
        left_t=math.nextafter(burn,math.inf) if abs(ta-burn)<1e-8 else ta
        right_t=math.nextafter(burn,-math.inf) if abs(tb-burn)<1e-8 else tb
        a0=terms(model,s0,left_t)[1];a1=terms(model,s1,right_t)[1]
        q0=np.array(s0.orientation_quaternion);q1=np.array(s1.orientation_quaternion)
        if np.dot(q0,q1)<0:q1=-q1
        qd0=np.array(quaternion_derivative(tuple(q0),s0.pitch_rate,s0.yaw_rate));qd1=np.array(quaternion_derivative(tuple(q1),s1.pitch_rate,s1.yaw_rate))
        cuts=sorted(set([ta,tb]+([burn] if ta+1e-8<burn<tb-1e-8 else [])))
        for order,(xx,ww) in nodes.items():
            integral=np.zeros(4)
            for ca,cb in zip(cuts[:-1],cuts[1:]):
                for z,w in zip(xx,ww):
                    ts=(ca+cb)/2+(cb-ca)/2*z;u=(ts-ta)/h
                    v=hermite(np.array(s0.velocity),np.array(s1.velocity),a0,a1,h,u)
                    p=hermite(np.array(s0.position),np.array(s1.position),np.array(s0.velocity),np.array(s1.velocity),h,u)
                    q=normalize_quaternion(tuple(hermite(q0,q1,qd0,qd1,h,u)));pitch,yaw=pitch_yaw_from_quaternion(q)
                    state=dataclasses.replace(s0,position=tuple(p),velocity=tuple(v),orientation_quaternion=q,pitch=pitch,yaw=yaw,
                        pitch_rate=(1-u)*s0.pitch_rate+u*s1.pitch_rate,yaw_rate=(1-u)*s0.yaw_rate+u*s1.yaw_rate)
                    integral+=(cb-ca)/2*w*terms(model,state,float(ts))[0]
            sums[order][i+1]=sums[order][i]+integral
    # Endpoint trapezoid is a lower-order comparison, correcting only the known
    # propulsive discontinuity at the endpoint; it is not used for attribution.
    trap=np.zeros_like(instant)
    for i in range(len(t)-1):
        left=instant[i].copy();right=instant[i+1].copy()
        if abs(t[i]-burn)<1e-8:left=terms(model,states[i],math.nextafter(burn,math.inf))[0]
        if abs(t[i+1]-burn)<1e-8:right=terms(model,states[i+1],math.nextafter(burn,-math.inf))[0]
        trap[i+1]=trap[i]+.5*(t[i+1]-t[i])*(left+right)
    closure=speed-speed[0]-sums[8].sum(axis=1)
    return {"t":t,"speed":speed,"terms":instant,"quantities":quantities,"integral":sums[8],"closure":closure}, {
        "source_path":str(path),"saved_rows":len(t),"energy_identity_max_abs_mps2":max_law,
        "Hermite8_vs_saved_speed_mps":stats(closure),"Hermite4_minus8_max_abs_channel_mps":np.max(abs(sums[4]-sums[8]),axis=0).tolist(),
        "trapezoid_minus_Hermite8_max_abs_channel_mps":np.max(abs(trap-sums[8]),axis=0).tolist(),"new_free_integrations":0}


def source_case(x,coeff):
    inverse=geometry.dec.audit.inverse;gap=inverse.gap;ref=x["ref"];t=ref["t"]
    model=geometry.make_model(x["config"],coeff);model.initialize(dataclasses.replace(x["initial"]))
    evaluated=[terms(model,inverse.source_state(ref,i),float(ts)) for i,ts in enumerate(t)]
    frozen_terms=np.array([r[0]for r in evaluated]);quantities=np.array([r[2]for r in evaluated])
    rows=x["run"]["samples"];base=np.array([s["drag_base_n"] for s in rows]);aoa=np.array([s["drag_aoa_n"]for s in rows]);thrust=ref["thrust"];mass=ref["mass"]
    velocity=ref["velocity_xyz_mps"];speed=np.linalg.norm(velocity,axis=1);cosine=quantities[:,4]
    source_terms=np.column_stack((thrust/mass*cosine,-base/mass,-aoa/mass,-model.gravity*velocity[:,1]/speed))
    # At a sampled thrust step, use the old thrust as the left limit at the
    # right endpoint. Clock uncertainty is the displayed transition bracket.
    source_integral=np.zeros_like(source_terms);frozen_integral=np.zeros_like(frozen_terms)
    source_linear_integral=np.zeros_like(source_terms);source_right_integral=np.zeros_like(source_terms)
    discontinuities=[]
    anchor=1
    # Creation thrust is exported as zero while the first velocity interval is
    # already powered. Anchor this diagnostic at the first nonzero-thrust row;
    # do not pretend the creation scalar is a resolved physical startup history.
    assert thrust[0]==0 and thrust[anchor]>0
    for i in range(anchor,len(t)-1):
        sl,sr=source_terms[i].copy(),source_terms[i+1].copy();fl,fr=frozen_terms[i].copy(),frozen_terms[i+1].copy()
        raw_sl,raw_sr=sl.copy(),sr.copy();right_sl=sl.copy()
        if thrust[i]!=thrust[i+1]:
            sr[0]=thrust[i]/mass[i+1]*cosine[i+1]
            fr[0]=quantities[i,2]/quantities[i+1,3]*cosine[i+1]
            right_sl[0]=thrust[i+1]/mass[i]*cosine[i]
            discontinuities.append([float(t[i]),float(t[i+1])])
        source_integral[i+1]=source_integral[i]+(t[i+1]-t[i])*.5*(sl+sr)
        frozen_integral[i+1]=frozen_integral[i]+(t[i+1]-t[i])*.5*(fl+fr)
        source_linear_integral[i+1]=source_linear_integral[i]+(t[i+1]-t[i])*.5*(raw_sl+raw_sr)
        source_right_integral[i+1]=source_right_integral[i]+(t[i+1]-t[i])*.5*(right_sl+raw_sr)
    derivative={};valid=np.ones(len(t),dtype=bool);valid[:2]=False;valid[-2:]=False
    valid&=t>14/48+2/24+1e-6
    valid&=abs(t-model.propulsion.burn_time_s)>2/24+1e-6
    for width in (3,5):
        dv=gap.derivative(speed,t,width); vector=gap.derivative(velocity,t,width);projected=np.sum(vector*velocity/speed[:,None],axis=1)
        derivative[str(width)]={"speed_derivative_minus_export_energy":dv-source_terms.sum(axis=1),
            "velocity_derivative_projected_minus_export_energy":projected-source_terms.sum(axis=1)}
    record={"anchor_first_valid_thrust_row_s":float(t[anchor]),"creation_thrust_export_N":float(thrust[0]),
        "creation_to_anchor_velocity_change_mps_not_force_reconstructed":float(speed[anchor]-speed[0]),
        "thrust_max_abs_N_after_anchor":float(np.max(abs(quantities[anchor:,2]-thrust[anchor:]))),"mass_max_abs_kg":float(np.max(abs(quantities[:,3]-mass))),
        "source_drag_sum_vs_acc_drag_max_abs_mps2":float(np.max(abs((base+aoa)/mass-ref["acc_drag"]))),
        "source_thrust_over_mass_vs_acc_thrust_max_abs_mps2":float(np.max(abs(thrust/mass-ref["acc_thrust"]))),
        "source_left_hold_integral_vs_saved_speed_mps_after_anchor":stats((speed-speed[anchor]-source_integral.sum(axis=1))[anchor:]),
        "source_linear_integral_vs_saved_speed_mps_after_anchor":stats((speed-speed[anchor]-source_linear_integral.sum(axis=1))[anchor:]),
        "source_right_hold_integral_vs_saved_speed_mps_after_anchor":stats((speed-speed[anchor]-source_right_integral.sum(axis=1))[anchor:]),
        "source_thrust_transition_brackets_s":discontinuities,
        "derivative_checks_halo_and_boundaries_excluded":{w:{k:stats(v[valid])for k,v in a.items()}for w,a in derivative.items()},
        "same_state_force_errors_N":stats(quantities[:,:2]-np.column_stack((base,aoa))),
        "existing_drag_fit_record":None}
    return {"t":t,"speed":speed,"source_terms":source_terms,"frozen_terms":frozen_terms,
        "source_integral":source_integral,"frozen_integral":frozen_integral,"same_state_integral_error":frozen_integral-source_integral,
        "source_linear_integral":source_linear_integral,"source_right_integral":source_right_integral,
        "source_energy_closure":speed-speed[anchor]-source_integral.sum(axis=1),"quantities":quantities,
        "source_base_aoa_N":np.column_stack((base,aoa)),"derivative_valid":valid,
        **{w+"_"+k:v for w,a in derivative.items()for k,v in a.items()}},record


def main():
    inverse=geometry.dec.audit.inverse;drag=geometry.dec.audit.drag;inputs=inverse.source_inputs()
    fit=json.loads((drag.OUTPUT/"fit.json").read_text());coeff=drag.DragCoefficients(**fit["coefficients"])
    reports={};arrays={}
    for cid in CASES:
        x=inputs[cid];free={};checks={}
        for label,folder in (("baseline","lift_direction"),("candidate","sinc_cap")):
            free[label],checks[label]=free_case(x,coeff,ROOT/"outputs/jari_offaxis"/folder/"full_runs"/(cid+".json"))
        source,src_check=source_case(x,coeff);src_check["existing_drag_fit_record"]=fit["same_state_closure"][cid]
        support=(source["t"]>=max(free[k]["t"][0]for k in free))&(source["t"]<=min(free[k]["t"][-1]for k in free))
        t=source["t"][support];vref=source["speed"][support]
        a={"t":t,"reference_speed":vref}
        anchor=src_check["anchor_first_valid_thrust_row_s"]
        for label,data in free.items():
            for key in ("speed","integral","closure"):
                a[label+"_"+key]=interp(t,data["t"],data[key])
            a[label+"_residual"]=a[label+"_speed"]-vref
        a["candidate_minus_baseline_integral"]=a["candidate_integral"]-a["baseline_integral"]
        a["candidate_minus_baseline_speed"]=a["candidate_speed"]-a["baseline_speed"]
        a["source_same_state_integral_error"]=source["same_state_integral_error"][support]
        a["source_energy_closure"]=source["source_energy_closure"][support]
        # Each free-vs-source force integral splits into a common same-source
        # frozen-law mismatch and a state/history difference. This is additive
        # bookkeeping, not a counterfactual trajectory or causal control-only fit.
        for label in free:
            at_anchor=interp(np.array([anchor]),free[label]["t"],free[label]["integral"])[0]
            a[label+"_history_integral_difference"]=a[label+"_integral"]-at_anchor-source["frozen_integral"][support]
        window_report={}
        for name,lo,hi0 in WINDOWS:
            hi=min(hi0,float(t[-1]));mask=(t>=lo-1e-9)&(t<=hi+1e-9)
            if not np.any(mask) or hi<=lo:continue
            increments={key:(interp(np.array([hi]),t,a[key])[0]-interp(np.array([lo]),t,a[key])[0]).tolist()
                for key in ("candidate_minus_baseline_integral","candidate_minus_baseline_speed","source_same_state_integral_error","baseline_history_integral_difference","candidate_history_integral_difference","source_energy_closure")}
            window_report[name]={"boundary_support_s":[lo,hi],"reference_rows":int(mask.sum()),
                "baseline_residual":stats(a["baseline_residual"][mask]),"candidate_residual":stats(a["candidate_residual"][mask]),
                "MSE_change_contribution_to_full":float(np.sum(a["candidate_residual"][mask]**2-a["baseline_residual"][mask]**2)/len(t)),
                "integral_increments_mps":increments}
        points=[]
        for ts in [0.,.5,.7,1.,2.,2.3,3.5,5.,7.,float(t[-1])]:
            if ts>t[-1]:continue
            points.append({"t":ts,**{key:interp(np.array([ts]),t,value)[0].tolist()for key,value in a.items()if key!="t"}})
        reports[cid]={"channel_order":CHANNELS,"checks":checks,"source_checks":src_check,"common_support_s":[float(t[0]),float(t[-1])],
            "source_diagnostic_anchor_s":anchor,"free_residual_at_source_anchor":{label:float(interp(np.array([anchor]),t,a[label+"_residual"])[0])for label in free},
            "full_residual":{k:stats(a[k+"_residual"])for k in free},"windows":window_report,"points":points}
        arrays.update({cid+"__common__"+key:value for key,value in a.items()})
        for label,data in {**free,"source":source}.items():arrays.update({cid+"__"+label+"__"+key:value for key,value in data.items()})
        print(cid,json.dumps({"full":reports[cid]["full_residual"],"checks":checks,"source":src_check}),flush=True)
    assert all(np.isfinite(v).all()for v in arrays.values())
    OUT.mkdir(parents=True,exist_ok=True);np.savez_compressed(OUT/"energy.npz",**arrays)
    payload={"task":"Narrow cached speed residual attribution","cases":reports,
        "equation":"dV/dt=T*cos(total_COM_AoA)/m - D_base/m - D_aoa/m - g*v_y/V; verified zero wind and normal lift",
        "quadrature":"Free: cubic Hermite saved states + 4/8 point Gauss per saved interval; no model step/run. Source: starts at first valid thrust row, native24Hz trapezoid; left-held thrust is primary with linear/right-held transition alternatives explicitly retained.",
        "limits":["Frozen same-source-state mismatch and own-trajectory history terms are additive attribution, not proof of a unique model error or a control-independent outcome.",
            "Source drag_base/drag_aoa/thrust/mass are JARI exported model fields, not recovered game internals.",
            "Local datamine2.59.0.27 versus JARI2.59.0.28 global equivalence remains unverified.",
            "Source force audit begins at 1/24 s because creation thrust=0 is not a resolved startup force history. Endpoint force fields do not determine thrust inside the 6.9583335--7.0000002 s transition interval; reported alternative source integrals bound this ambiguity.",
            "3/5 source derivatives are noncausal diagnostics and stencil differences are not confidence intervals."],
        "verification":{"all_arrays_finite":True,"new_free_integrations":0,"fits":0,"new_coefficients":0,"default_changes":0}}
    (OUT/"results.json").write_text(json.dumps(payload,indent=2)+"\n")


if __name__=="__main__":main()
