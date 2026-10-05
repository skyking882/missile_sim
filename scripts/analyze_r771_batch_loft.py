"""Seven-shot R-77-1 loft and conditional moment audit (no production changes).

All angles/rates are decoded from replay rows. The two control requests are
fixed interpretations of raw fields, not decoded commands. A measured-state
interval fit is conditional and is never presented as a free-flight model.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from try_r771_effective_moment import (
    CONTROL_DT, G, HORIZON, FIT_BOUNDS, HOLD_BOUNDS, full_interval_sign_conflicts,
    raw_parameters, request, scheduled_pid_step, signed16, target_at,
)

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "outputs/r771_replay_batch_20260928"
OUT = BASE / "loft_analysis"
WIN = (0.30, 2.50)
COLORS = {"H110":"#2369a8", "H27.5":"#ec7d28", "H55native":"#5c48a5",
          "H55clone":"#a74796", "L110":"#16856e", "L27.5":"#aa6c1e",
          "L55":"#558a3d"}
NAMES = {"pi28_r771_h_110":"H110", "pi28_r771_h_27p5":"H27.5",
         "pi28_r771_l_110":"L110", "pi28_r771_l_27p5":"L27.5",
         "pi28_r771_l_55":"L55", "su_r_77_1":"H55native",
         "pi28_r771_h_clone":"H55clone"}


def rows_array(case, which, names):
    rows=case[which+"_rows"]
    assert case[which+"_columns"] == names, (case["case"], which)
    a=np.array([[float(row[k]) for k in names] for row in rows],float)
    assert np.isfinite(a).all()
    return a


def load_all():
    payload=json.loads((BASE/"source_payload.json").read_text())
    compact=json.loads((BASE/"batch_compact.json").read_text())
    assert len(payload["cases"])==len(compact["shots"])==7
    rocket=json.loads(payload["parent_parameters"]["native_text"])["rocket"]
    pp={"optional_existing_parent_parameters":{"rocket":rocket,"source_version":"2.59.0.28"}}
    p0=raw_parameters(pp)
    shots={}
    for case,meta in zip(payload["cases"],compact["shots"]):
        assert case["case"]==meta["actual_case"] and len(case["missile_rows"])==meta["sample_count"]
        label=NAMES[case["case"]]
        mcols=case["missile_columns"]
        m=rows_array(case,"missile",mcols)
        target=rows_array(case,"target",case["target_columns"])
        shooter=rows_array(case,"shooter",case["shooter_columns"])
        col=lambda key: m[:,mcols.index(key)]
        release_ms=float(meta["creation_time_ms"])
        t=(col("replay_time_ms")-release_ms)/1000
        assert t[0]>=-1e-9 and np.all(np.diff(t)>0)
        assert np.all(np.diff(target[:,0])>0) and np.all(np.diff(shooter[:,0])>0)
        b5=col("raw_b5")/48-float(case["creation"]["creation_time"])
        assert np.all(np.diff(b5)>=0)
        vel=signed16(np.column_stack([col(f"raw_b7_{axis}_raw_u16") for axis in "xyz"]))*(4000/65535)
        q=-signed16(col("raw_b8_z_raw_u16"))*(30/65535)
        theta=np.unwrap(col("raw_euler_z"))
        speed=np.hypot(vel[:,0],vel[:,1])
        gamma=np.unwrap(np.arctan2(vel[:,1],vel[:,0]))
        target_t=target[:,0]/1000
        target_xy=target[:,1:3]
        target_vxy=np.gradient(target_xy,target_t,axis=0)
        p=dict(p0,finsLatAccel_custom=float(case["finsLatAccel"]),loft_enabled=bool(case["loftEnabled"]),
               schedule_gate="total_pid_output",loft_reference="flight_path",loft_omega_unit="rad_per_s")
        replay={"t":t,"missile_packet0_s":release_ms/1000,"x":col("x_m"),"y":col("y_m"),
                "z":col("z_m"),"theta":theta,"gamma":gamma,"alpha":theta-gamma,
                "omega":q,"speed2":speed,"velocity3":vel,
                "target_t":target_t,"target_xy":target_xy,"target_vxy":target_vxy}
        shots[label]={"case":case,"meta":meta,"p":p,"r":replay,"b5":b5,"shooter":shooter}
    return shots,p0


def interp_at(a,t,v):
    return float(np.interp(a,t,v))


def pose_rotation(bank,heading,pitch):
    cb,sb=math.cos(bank),math.sin(bank)
    ch,sh=math.cos(heading),math.sin(heading)
    cp,sp=math.cos(pitch),math.sin(pitch)
    rx=np.array([[1,0,0],[0,cb,-sb],[0,sb,cb]])
    ry=np.array([[ch,0,sh],[0,1,0],[-sh,0,ch]])
    rz=np.array([[cp,-sp,0],[sp,cp,0],[0,0,1]])
    return ry@rz@rx


def interval_body_pitch_rate(shot):
    case=shot["case"];r=shot["r"]
    poses=[pose_rotation(float(row["raw_euler_x"]),float(row["raw_euler_y"]),float(row["raw_euler_z"]))
           for row in case["missile_rows"]]
    out=[]
    for i in range(len(poses)-1):
        relative=poses[i].T@poses[i+1]
        angle=math.acos(max(-1.0,min(1.0,(np.trace(relative)-1)/2)))
        factor=angle/(2*math.sin(angle)) if angle>1e-7 else .5
        out.append(factor*(relative[1,0]-relative[0,1])/(r["t"][i+1]-r["t"][i]))
    return np.array(out)


def extrema(t,y,lo,hi,kind):
    ids=np.flatnonzero((t>=lo)&(t<=hi))
    if len(ids)==0:return None
    i=int(ids[np.argmax(y[ids]) if kind=="max" else np.argmin(y[ids])])
    return {"time_s":float(t[i]),"value":float(y[i])}


def native_metrics(shot):
    r=shot["r"];t=r["t"];q=r["omega"]
    target=[];los=[];losrate=[];range_m=[]
    for i,ti in enumerate(t):
        xy,vxy=target_at(r,float(ti))
        rel=xy-np.array([r["x"][i],r["y"][i]])
        mv=np.array([r["velocity3"][i,0],r["velocity3"][i,1]])
        rv=vxy-mv
        target.append(xy)
        los.append(math.atan2(rel[1],rel[0]))
        losrate.append((rel[0]*rv[1]-rel[1]*rv[0])/float(rel@rel))
        range_m.append(np.linalg.norm(rel))
    target=np.array(target);los=np.array(los);losrate=np.array(losrate)
    rate_from_pose=interval_body_pitch_rate(shot)
    rate_mid=(q[:-1]+q[1:])/2
    mask=(t[:-1]>=0)&(t[1:]<=2.5)
    e=rate_from_pose[mask]-rate_mid[mask]
    g_interval=((.5*(r["speed2"][:-1]+r["speed2"][1:])*np.diff(r["gamma"])/np.diff(t))
                +G*np.cos(.5*(r["gamma"][:-1]+r["gamma"][1:])))/G
    load_mask=(t[:-1]>=.30)&(t[1:]<=2.5)
    g_abs=np.abs(g_interval[load_mask])
    b5_dt=np.diff(shot["b5"])
    b5_valid=(b5_dt>0)&load_mask
    g_b5=((.5*(r["speed2"][:-1]+r["speed2"][1:])[b5_valid]*np.diff(r["gamma"])[b5_valid]/b5_dt[b5_valid])
          +G*np.cos(.5*(r["gamma"][:-1]+r["gamma"][1:]))[b5_valid])/G
    initial={k:float(r[k][0]) for k in ("theta","gamma","alpha","omega","speed2","x","y","z")}
    initial.update({"theta_deg":math.degrees(initial["theta"]),"gamma_deg":math.degrees(initial["gamma"]),
                    "alpha_deg":math.degrees(initial["alpha"])})
    shooter=shot["shooter"]
    sy0=interp_at(shot["meta"]["creation_time_ms"],shooter[:,0],shooter[:,2])
    sy25=interp_at(shot["meta"]["creation_time_ms"]+2500,shooter[:,0],shooter[:,2])
    position=np.column_stack((r["x"],r["y"],r["z"]))
    vel=r["velocity3"]
    creation=shot["case"]["creation"]
    creation_pos=np.array(creation["starting_pos"],float)
    creation_vel=np.array(creation["u12_2"],float)
    initial.update({"creation_starting_pos_m":creation_pos.tolist(),
                    "creation_u12_2_mps_candidate_initial_velocity":creation_vel.tolist(),
                    "first_position_minus_creation_m":float(np.linalg.norm(position[0]-creation_pos)),
                    "first_b7_velocity_minus_creation_u12_2_mps":float(np.linalg.norm(vel[0]-creation_vel))})
    step_position=np.linalg.norm(np.diff(position,axis=0)-.5*(vel[:-1]+vel[1:])*np.diff(t)[:,None],axis=1)
    valid_b5=b5_dt>0
    step_b5=np.linalg.norm(np.diff(position,axis=0)[valid_b5]-
                           .5*(vel[:-1]+vel[1:])[valid_b5]*b5_dt[valid_b5,None],axis=1)
    last_2p5=int(np.flatnonzero(t<=2.5)[-1])
    def closure(dt,last):
        integrated=np.sum(.5*(vel[:last]+vel[1:last+1])*dt[:last,None],axis=0)
        return float(np.linalg.norm((position[last]-position[0])-integrated))
    metrics={"case":shot["case"]["case"],"n":len(t),"first_packet_age_s":float(t[0]),
             "last_packet_age_s":float(t[-1]),"b5_duplicate_interval_count":int(np.sum(np.diff(shot["b5"])==0)),
             "packet_vs_b5_age_max_abs_s":float(np.max(np.abs(t-shot["b5"]))),
             "position_integral_packet_median_step_error_m":float(np.median(step_position)),
             "position_integral_b5_median_step_error_m_nonduplicate":float(np.median(step_b5)),
             "position_integral_packet_p95_step_error_m":float(np.quantile(step_position,.95)),
             "position_integral_b5_p95_step_error_m_nonduplicate":float(np.quantile(step_b5,.95)),
             "position_integral_packet_max_step_error_m":float(np.max(step_position)),
             "position_integral_b5_max_step_error_m_nonduplicate":float(np.max(step_b5)),
             "position_integral_packet_cumulative_closure_0_2p5_m":closure(np.diff(t),last_2p5),
             "position_integral_b5_cumulative_closure_0_2p5_m":closure(b5_dt,last_2p5),
             "position_integral_packet_cumulative_closure_full_m":closure(np.diff(t),len(t)-1),
             "position_integral_b5_cumulative_closure_full_m":closure(b5_dt,len(t)-1),
             "packet_minus_b5_age_first_last_s":[float(t[0]-shot["b5"][0]),float(t[-1]-shot["b5"][-1])],
             "creation_sim_time_s":float(shot["case"]["creation"]["creation_time"]),
             "release_packet_ms":shot["meta"]["creation_time_ms"],"first":initial,
             "pitch_rate_pose_b8_corr_0_2p5":float(np.corrcoef(rate_from_pose[mask],rate_mid[mask])[0,1]),
             "pitch_rate_pose_b8_rmse_rad_s":float(np.sqrt(np.mean(e*e))),
             "pitch_rate_pose_b8_abs_residual_p95_rad_s":float(np.quantile(np.abs(e),.95)),
             "first_q_peak":extrema(t,q,.35,.60,"max"),
             "first_q_valley":extrema(t,q,.60,.80,"min"),
             "second_q_peak":extrema(t,q,.80,1.08,"max"),
             "alpha_first_peak_deg":extrema(t,np.degrees(r["alpha"]),.35,.75,"max"),
             "height_gain_2p5_m":interp_at(2.5,t,r["y"])-r["y"][0],
             "speed_2p5_mps":interp_at(2.5,t,r["speed2"]),
             "shooter_height_release_m":sy0,"shooter_height_change_2p5_m":sy25-sy0,
             "target_height_release_m":float(target[0,1]),
             "range_initial_m":float(range_m[0]),
             "los_elevation_range_0_2p5_deg":[float(np.min(np.degrees(los[t<=2.5]))),float(np.max(np.degrees(los[t<=2.5])))],
             "los_rate_abs_max_0_2p5_deg_s":float(np.max(np.abs(np.degrees(losrate[t<=2.5])))),
             "normal_load_g_max_abs_0p3_2p5":float(np.max(g_abs)),
             "normal_load_g_p95_abs_0p3_2p5":float(np.quantile(g_abs,.95)),
             "normal_load_b5_clock_g_max_abs_0p3_2p5":float(np.max(np.abs(g_b5))),
             "normal_load_b5_clock_g_p95_abs_0p3_2p5":float(np.quantile(np.abs(g_b5),.95)),
             "fraction_native_intervals_abs_load_over_27p5_g":float(np.mean(g_abs>27.5)),
             "fraction_native_intervals_abs_load_over_27p5_mps2":float(np.mean(g_abs*G>27.5)),
             "fraction_b5_intervals_abs_load_over_27p5_mps2":float(np.mean(np.abs(g_b5)*G>27.5)),
             "max_out_of_plane_velocity_deg":float(np.max(np.degrees(np.arctan2(np.abs(r["velocity3"][:,2]),r["speed2"]))))}
    shot["los_deg"]=np.degrees(los)
    shot["losrate_deg_s"]=np.degrees(losrate)
    shot["range_m"]=np.array(range_m)
    return metrics


def pair_difference(a,b):
    ra=a["r"];rb=b["r"]
    grid=np.arange(.04,2.50001,.01)
    result={}
    for key,scale in (("theta",180/math.pi),("gamma",180/math.pi),("alpha",180/math.pi),
                      ("omega",1),("y",1),("speed2",1)):
        delta=(np.interp(grid,ra["t"],ra[key])-np.interp(grid,rb["t"],rb[key]))*scale
        result[key+"_rms"] = float(np.sqrt(np.mean(delta**2)))
    return result


def diagnostic(shot,mode):
    r=shot["r"];p=dict(shot["p"])
    if mode=="loft_exclusive_vertical" and p["loft_enabled"]:
        # In this fixed candidate, the vertical PN contribution is absent
        # throughout the observed active-loft window. No timed handover.
        p["pn_gain"]=0.0
    pid=(0.0,0.0,0.0);active=False;out=[]
    first=int(math.ceil((r["t"][0]-1e-12)/CONTROL_DT))
    for n in range(first,int(round(2.5/CONTROL_DT))+1):
        t=n*CONTROL_DT
        state=np.array([np.interp(t,r["t"],r[k]) for k in ("theta","omega","gamma","x","y")])
        req=request(t,state,r,p)
        u,pid,_,active=scheduled_pid_step(t,req,pid,p,active)
        out.append({"t":t,"u":float(u),"alpha":float(state[0]-state[2]),"omega":float(state[1]),
                    "a_req_g":req["a_req_g"],"actual_g":req["actual_g"],
                    "pn_for_command_g":req["pn_for_command_g"],"loft_projected_g":req["loft_projected_g"],
                    "loft_raw_g":req["loft_raw_g"],"loft_limited_g":req["loft_limited_g"],
                    "gravity_compensation_g":req["gravity_compensation_g"],"los_elevation_deg":req["los_elevation_deg"],
                    "los_rate_deg_s":req["los_rate_deg_s"],"flight_gain":req["flight_gain"]})
    return out


def interval_features(shot,diag,bounds):
    r=shot["r"];t=r["t"]
    dtimes=np.array([d["t"] for d in diag]);u=np.array([d["u"] for d in diag])
    x=[];y=[];interval=[]
    for i in range(len(t)-1):
        lo,hi=float(t[i]),float(t[i+1])
        if lo<bounds[0]-1e-12 or hi>bounds[1]+1e-12:continue
        cuts=[lo]+[float(v) for v in dtimes if lo<v<hi]+[hi]
        iu=sum((b-a)*u[max(0,np.searchsorted(dtimes,a,side="right")-1)] for a,b in zip(cuts,cuts[1:]))
        ia=.5*(r["alpha"][i]+r["alpha"][i+1])*(hi-lo)
        iq=.5*(r["omega"][i]+r["omega"][i+1])*(hi-lo)
        x.append([iu,-ia,-iq]);y.append(r["omega"][i+1]-r["omega"][i]);interval.append([lo,hi])
    return np.asarray(x),np.asarray(y),interval


def fit_l_chain(shots,diagnostics):
    tr=[];yy=[]
    for name in ("L110","L27.5"):
        x,y,_=interval_features(shots[name],diagnostics[name]["loft_additive"],FIT_BOUNDS)
        tr.append(x);yy.append(y)
    a=np.vstack(tr);b=np.concatenate(yy)
    sv=np.linalg.svd(a,compute_uv=False);scale=np.linalg.norm(a,axis=0)
    upper=np.array([10000.0,300.0,100.0])
    # Exact bounded three-variable least squares: enumerate each coefficient
    # at lower bound, free, or upper bound. No optimizer dependency or sweep.
    best=None
    from itertools import product
    for status in product((-1,0,1),repeat=3):
        coeff_trial=np.array([upper[j] if status[j]==1 else 0.0 for j in range(3)])
        free=[j for j in range(3) if status[j]==0]
        if free:
            rhs=b-a@coeff_trial
            coeff_trial[free]=np.linalg.lstsq(a[:,free]/scale[free],rhs,rcond=None)[0]/scale[free]
        if np.any(coeff_trial < -1e-8) or np.any(coeff_trial > upper+1e-8):continue
        score=float(np.sum((a@coeff_trial-b)**2))
        if best is None or score<best[0]:best=(score,coeff_trial)
    assert best is not None
    coeff=best[1]
    scores={}
    for name in shots:
        scores[name]={}
        for mode in diagnostics[name]:
            for part,bounds in (("train",FIT_BOUNDS),("hold",HOLD_BOUNDS)):
                x,y,interval=interval_features(shots[name],diagnostics[name][mode],bounds)
                pred=x@coeff
                scores[name][mode+"_"+part]={"native_interval_count":len(y),
                    "delta_q_rmse_rad_s":float(np.sqrt(np.mean((pred-y)**2))),
                    "delta_q_mean_bias_rad_s":float(np.mean(pred-y)),
                    "observed_positive_predicted_nonpositive_count":int(np.sum((y>0)&(pred<=0))),
                    "observed_negative_predicted_nonnegative_count":int(np.sum((y<0)&(pred>=0))),
                    "observed_delta_q_sum_rad_s":float(np.sum(y)),"predicted_delta_q_sum_rad_s":float(np.sum(pred))}
    return {"coefficients":{"ku_rad_s2_per_rad":float(coeff[0]),"ka_s_minus2":float(coeff[1]),
                            "kw_s_minus1":float(coeff[2])},
            "training_cases":["L110","L27.5"],"case_holdouts":["L55","H110","H27.5","H55native","H55clone"],
            "train_interval_count":len(b),"rank":int(np.linalg.matrix_rank(a)),
            "condition_number_unscaled":float(sv[0]/sv[-1]),"singular_values":sv.tolist(),
            "boundary_hit":[bool(coeff[j]>=upper[j]-1e-6 or coeff[j]<=1e-9) for j in range(3)],
            "training_delta_q_rmse_rad_s":float(np.sqrt(np.mean((a@coeff-b)**2))),
            "scores":scores,
            "interpretation":"Conditional native-interval delta q only: measured state, measured V, and reconstructed fixed-candidate u are inputs. Not a closed-loop forward or decoded fin history."}


def plot_native(shots):
    order=("H110","H27.5","H55native","H55clone","L110","L27.5","L55")
    fig,axs=plt.subplots(2,3,figsize=(17,9),sharex=True,layout="constrained")
    specs=(("theta","Pitch attitude θ (deg)",180/math.pi),("gamma","Flight path γ (deg)",180/math.pi),
           ("alpha","Signed α = θ−γ (deg)",180/math.pi),("omega","Body pitch rate −b8z (rad/s)",1),
           ("y","Missile height gain (m)",1),("speed2","Planar speed (m/s)",1))
    for ax,(key,title,scale) in zip(axs.flat,specs):
        for name in order:
            r=shots[name]["r"];x=r["t"]
            y=(r[key]-r[key][0] if key=="y" else r[key])*scale
            mask=x<=2.5
            ax.plot(x[mask],y[mask],color=COLORS[name],ls="--" if name.startswith("L") else "-",lw=1.3,
                    marker=".",ms=2.3,label=name)
        ax.axvline(.30,color="0.35",lw=.8,ls=":")
        ax.set(xlim=(0,2.5),title=title,xlabel="packet age from actual release (s)")
        ax.grid(alpha=.2)
    axs[0,0].legend(fontsize=8,ncol=2,loc="upper left")
    fig.suptitle("Seven R-77-1 shots: native decoded kinematics (dots = original samples)",fontsize=15)
    fig.savefig(OUT/"native_dynamics.png",dpi=180)
    plt.close(fig)

    fig,axs=plt.subplots(2,2,figsize=(15,9),sharex=True,layout="constrained")
    for ax,key,title in zip(axs.flat,("los_deg","losrate_deg_s","range_m","z"),
                            ("Target LOS elevation (deg)","Target LOS rate (deg/s)",
                             "Missile–target planar range (m)","Out-of-plane missile displacement (m)")):
        for name in order:
            s=shots[name];r=s["r"];x=r["t"]
            values=(r["z"]-r["z"][0]) if key=="z" else s[key]
            ax.plot(x[x<=2.5],values[x<=2.5],color=COLORS[name],
                    ls="--" if name.startswith("L") else "-",lw=1.2,marker=".",ms=2.3,label=name)
        if key=="los_deg":ax.axhline(-3.5,color="black",lw=.8,ls=":",label="raw targetElevation −3.5°")
        if key=="losrate_deg_s":
            ax.axhline(.5,color="black",lw=.8,ls=":")
            ax.axhline(-.5,color="black",lw=.8,ls=":")
        ax.set(xlim=(0,2.5),title=title,xlabel="packet age from actual release (s)")
        ax.grid(alpha=.2)
    axs[0,0].legend(fontsize=8,ncol=2,loc="best")
    fig.suptitle("Actual target geometry; dotted lines are raw loft values, not established switch laws",fontsize=14)
    fig.savefig(OUT/"native_geometry.png",dpi=180)
    plt.close(fig)


def plot_diagnostic(shots,diags):
    fig,axs=plt.subplots(2,2,figsize=(15,9),sharex=True,layout="constrained")
    for name in ("H110","H27.5","H55native","H55clone"):
        s=shots[name];r=s["r"]
        for mode,style in (("loft_additive","-"),("loft_exclusive_vertical","--")):
            rows=diags[name][mode];t=np.array([v["t"] for v in rows])
            axs[0,0].plot(t,[v["u"] for v in rows],color=COLORS[name],ls=style,lw=1.2,
                          label=name+(" add" if style=="-" else " exclusive"))
        rows=diags[name]["loft_additive"]
        t=np.array([v["t"] for v in rows])
        axs[0,1].plot(t,[v["a_req_g"] for v in rows],color=COLORS[name],lw=1.2,label=name+" request")
        axs[1,0].plot(t,[v["pn_for_command_g"] for v in rows],color=COLORS[name],lw=1.2)
        axs[1,1].plot(r["t"],r["omega"],color=COLORS[name],lw=1.2,marker=".",ms=2.3,label=name)
    axs[0,0].axhline(0,color="black",lw=.7)
    axs[0,0].set_title("Replay-state diagnostic PID output u (rad)")
    axs[0,0].legend(fontsize=7,ncol=2)
    axs[0,1].set_title("Additive total specific-force request (g)")
    axs[1,0].set_title("PN normal contribution before loft addition (g)")
    axs[1,1].set_title("Measured body pitch rate (rad/s)")
    for ax in axs.flat:
        ax.axvline(.30,color="0.3",ls=":",lw=.8)
        ax.set(xlim=(.25,1.4),xlabel="packet age from release (s)")
        ax.grid(alpha=.2)
    fig.suptitle("Fixed raw PID and total timeToGain: two loft combination hypotheses\n"
                 "Diagnostic u uses measured states; both use 17.5(24°−γ) and candidate 0.5 rad/s cap",fontsize=13)
    fig.savefig(OUT/"loft_request_diagnostic.png",dpi=180)
    plt.close(fig)


def plot_conditional_chain(shots,diags,fit):
    coeff=np.array([fit["coefficients"][k] for k in
                    ("ku_rad_s2_per_rad","ka_s_minus2","kw_s_minus1")])
    fig,axs=plt.subplots(3,1,figsize=(14,10),sharex=True,layout="constrained")
    for ax,name in zip(axs,("L55","H110","H55clone")):
        for mode,style,color in (("loft_additive","-","#d36a25"),
                                 ("loft_exclusive_vertical","--","#6d52a5")):
            x,y,interval=interval_features(shots[name],diags[name][mode],(.30,2.45))
            mid=np.mean(np.array(interval),axis=1)
            if mode=="loft_additive":
                ax.plot(mid,y,"o-",color="black",lw=1,ms=3,label="observed native Δq")
            ax.plot(mid,x@coeff,style,color=color,lw=1.2,marker=".",ms=2.5,
                    label="conditional "+("additive" if mode=="loft_additive" else "exclusive"))
        ax.axhline(0,color="0.25",lw=.7)
        ax.axvspan(*FIT_BOUNDS,color="#dbe9f5",alpha=.45)
        ax.axvspan(*HOLD_BOUNDS,color="#f5e4d0",alpha=.45)
        ax.set(title=name+(" (whole-case holdout)" if name=="L55" else " (loft-on cross-case check)"),
               ylabel="native Δq (rad/s)")
        ax.grid(alpha=.2)
    axs[0].legend(ncol=3,fontsize=9)
    axs[-1].set(xlabel="packet age at interval midpoint (s)",xlim=(.30,2.45))
    fig.suptitle("One L-group moment fit applied unchanged across cases\n"
                 "Blue: L110/L27.5 train times; orange: continuous hold times. Measured states and diagnostic u are inputs.",fontsize=13)
    fig.savefig(OUT/"conditional_delta_q_crosscase.png",dpi=180)
    plt.close(fig)


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    shots,p0=load_all()
    metrics={name:native_metrics(s) for name,s in shots.items()}
    pairs={a+"_vs_"+b:pair_difference(shots[a],shots[b]) for a,b in
           (("H27.5","H55clone"),("H55native","H55clone"),("H110","H27.5"),
            ("L110","L27.5"),("L110","L55"))}
    diags={name:{mode:diagnostic(shot,mode) for mode in ("loft_additive","loft_exclusive_vertical")}
           for name,shot in shots.items()}
    conflicts={name:{mode:full_interval_sign_conflicts(shot["r"],rows) for mode,rows in ds.items()}
               for name,(shot,ds) in ((n,(shots[n],diags[n])) for n in shots)}
    conflict_clock_neighborhood={}
    for name,ds in diags.items():
        tolerance=metrics[name]["packet_vs_b5_age_max_abs_s"]
        conflict_clock_neighborhood[name]={}
        for mode,rows in ds.items():
            items=[]
            for c in conflicts[name][mode]:
                lo,hi=c["interval_s"]
                nearby=[v["u"] for v in rows if lo-tolerance-CONTROL_DT<=v["t"]<=hi+tolerance+CONTROL_DT]
                items.append({"interval_packet_s":[lo,hi],"packet_b5_max_age_difference_s":tolerance,
                              "maximum_diagnostic_u_in_expanded_tick_neighborhood_rad":float(max(nearby))})
            conflict_clock_neighborhood[name][mode]=items
    samples={}
    for name,ds in diags.items():
        samples[name]={}
        for mode,rows in ds.items():
            samples[name][mode]={"first_nonzero_u_tick_s":next((v["t"] for v in rows if abs(v["u"])>1e-9),None),
                                 "max_abs_u_rad_0p3_2p5":max(abs(v["u"]) for v in rows),
                                 "loft_cap_active_tick_count":int(sum(abs(v["loft_raw_g"]-v["loft_limited_g"])>1e-7 for v in rows)),
                                 "at_times_s":{str(tt):min(rows,key=lambda v:abs(v["t"]-tt)) for tt in (.32,.40,.44,.84)}}
    fit=fit_l_chain(shots,diags)
    plot_native(shots);plot_diagnostic(shots,diags);plot_conditional_chain(shots,diags,fit)
    result={"source":"seven native replay cases; source_payload.json + batch_compact.json", 
            "time":"Packet age = replay_time_ms minus compact creation_time_ms is the shared observation/target axis used in this script's diagnostic controller. b5 age = raw_b5/48 minus creation.creation_time matches most individual position increments much better, but counter skips/duplicate and endpoint clock drift can make cumulative closure worse. The two clock metrics are reported separately. Neither is declared the decoded game control scheduler; a physical-time controller candidate should use b5 age while target states remain associated with packet timestamps.",
            "decode":"b7 signed16*4000/65535 m/s; body pitch rate = -b8_z signed16*30/65535 rad/s; gamma=atan2(vy,vx); alpha=theta-gamma",
            "raw_fixed":{"source_version":p0["source_version"],"kp":p0["kp"],"ki":p0["ki"],"kd":p0["kd"],
                         "time_gain":p0["time_gain"],"pn_gain":p0["pn_gain"],
                         "loft_elevation_rad":p0["loft_elevation_rad"],
                         "loft_angle_to_accel_mult":p0["loft_angle_to_accel_mult"],
                         "loft_target_elevation_deg":p0["loft_target_elevation_deg"],
                         "loft_omega_max_raw":p0["loft_omega_max_deg_s"],"req_accel_max_g":p0["req_accel_max_g"]},
            "hypotheses":{"common":"u=timeToGain(t)*PID(a_req-a_feedback); a_feedback=(qbar*S_eff*14*alpha+T*sin(alpha))/(m*g); PN=6*closing*LOSrate*timeToHitGain projected on normal; loft=17.5*(24deg-gamma) g capped by candidate 0.5 rad/s * V/g; raw PID unchanged; all requests include gravity compensation",
                          "loft_additive":"vertical request = PN + loft while loftEnabled; no timed capture",
                          "loft_exclusive_vertical":"vertical request = loft while loftEnabled (PN vertical zero); L cases use PN; no timed capture/exit in observed window",
                          "loft_target_elevation_status":"Raw -3.5 deg is not used by either candidate; native LOS elevation geometry can bound threshold but cannot establish field mapping or exit semantics.",
                          "omega_max_status":"Raw 0.5 interpreted as missile turn-rate cap in rad/s here, unverified. Prior 0.5 deg/s target-LOS gate interpretation performed worse on h110; these roles are not interchangeable."},
            "measurements":metrics,"pairwise_rms_0p04_2p5":pairs,
            "replay_state_diagnostic_samples":samples,
            "full_native_interval_sign_conflicts_train":conflicts,
            "sign_conflict_clock_neighborhood":conflict_clock_neighborhood,
            "conditional_l_chain_fit":fit,
            "figures":["native_dynamics.png","native_geometry.png","loft_request_diagnostic.png",
                       "conditional_delta_q_crosscase.png"]}
    (OUT/"results.json").write_text(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False)+"\n")
    print(json.dumps({"cases":list(shots),"normal_load_max_abs_g":{n:round(m["normal_load_g_max_abs_0p3_2p5"],2) for n,m in metrics.items()},
                      "conflict_counts":{n:{mode:len(rows) for mode,rows in mm.items()} for n,mm in conflicts.items()},
                      "conditional_coefficients":fit["coefficients"],"condition":fit["condition_number_unscaled"]},indent=2))


if __name__=="__main__":main()
