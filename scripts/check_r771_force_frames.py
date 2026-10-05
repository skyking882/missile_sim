#!/usr/bin/env python3
"""Bounded game force-coordinate comparison, with no fitted coefficient.

Three whole-vector hypotheses are declared here. They share the frozen
single-surface angular law, raw PID and the existing calibrated FreeLoft
outer request. Frame hypotheses are evidence level D, not game equations.
"""
from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from aim120_model.aerodynamics import body_axes_for_state
from aim120_model.drag_models import effective_cda0, effective_cda_alpha, area_basis
from aim120_model.math3d import add, dot, scale, sub
from aim120_model.single_surface_game import SurfaceCoefficients
from aim120_model.units import G0_MPS2 as G
import compare_unified_game_replays as harness
import run_single_surface_game_candidate as candidate
from fit_r771_free_loft import FreeLoft

OUT = ROOT / "outputs/r771_force_frames"
JARI = ROOT / "outputs/jari_reference/r771_asl6300_t0.json"
LOFT = ROOT / "outputs/unified_game_replays/free_loft/results.json"
MAPPINGS = ("wind_normal_wind_drag", "body_normal_body_drag", "body_normal_wind_drag")
FIELDS = ("alpha_deg", "omega_rad_s", "gamma_deg", "y_m", "speed_kmh")
DECLARATION = {
    "mappings": {
        "wind_normal_wind_drag": "F=Np*wind_up+Ny*wind_right-D*wind_forward",
        "body_normal_body_drag": "F=Np*body_up+Ny*body_right-D*body_forward",
        "body_normal_wind_drag": "F=Np*body_up+Ny*body_right-D*wind_forward",
    },
    "normal_amplitudes": "Np/Ny=q*S_eff*Ca*local_pitch/yaw_incidence; S_eff=(pi*d^2/4)*wingAreaMult, once",
    "drag_amplitude": "D=q*[S_eff*CxK*shared_1943x1.10_Cx(M)+S_body*CxAoA*min(alpha_vector_norm,1.2)^2]; same existing magnitude in all mappings",
    "moment": "One force resultant at -distFromCmToStab and same-station reference couple I*qdot-lever; qdot unchanged frozen empirical law",
    "Jari": "Independently maximize trajectory-normal specific force over alpha in [0,90] degrees with q=0; 50g command ceiling chooses the smallest angle reaching it. Separate predeclared raw CyMaxAoA=1.3-as-radian domain diagnostic; no gain or angle fit.",
    "promotion_gate": "All L110/L27.5/L55 early 0..3.5s satisfy baseline-relative RMSE limits: alpha<=1.25*base+.025deg, omega<=1.25*base+.005rad/s, gamma<=1.25*base+.025deg, y<=1.25*base+.5m, V<=base+5km/h; H110-L110 speed-difference RMSE improves, and historical conditional speed RMSE improves OR both Jari max-G and maximizing-angle RMSE improve. All data retained and finite. Only a qualifying mapping receives one 4-to-8 plant-substep check.",
    "coefficients_fitted": False,
    "force_frame_evidence_grade": "D",
}


class ForceFrameGame(FreeLoft):
    def __init__(self, *args, force_mapping, **kwargs):
        super().__init__(*args, **kwargs)
        self.force_mapping = force_mapping

    def evaluate(self, state, time_s, held):
        original = super().evaluate(state, time_s, held)
        if self.force_mapping == MAPPINGS[0]:
            return original
        axes = body_axes_for_state(state)
        amplitudes = tuple(original.qbar*self.area*self.coefficients.ca*a
                           for a in original.local_incidence)
        normal = add(scale(axes.up, amplitudes[0]), scale(axes.right, amplitudes[1]))
        drag_axis = axes.forward if self.force_mapping == MAPPINGS[1] else original.wind_basis.forward
        aero = add(normal, scale(drag_axis, -original.drag_n))
        specific = scale(add(aero, scale(axes.forward, original.thrust)), 1/original.mass)
        acceleration = add(specific, (0., -self.gravity, 0.))
        tangent = original.wind_basis.forward
        axial_aero = dot(aero, tangent)
        wind_drag = scale(tangent, axial_aero)
        wind_lift = sub(aero, wind_drag)
        feedback = (dot(specific, original.wind_basis.up)/G,
                    dot(specific, original.wind_basis.right)/G)
        lever = (-self.arm*dot(aero, axes.up), -self.arm*dot(aero, axes.right))
        couple = tuple(original.inertia*q-l for q,l in zip(original.angular_acceleration,lever))
        return dataclasses.replace(original, acceleration=acceleration,
            specific_acceleration=specific, aerodynamic_force=aero,
            wind_lift_force=wind_lift, wind_drag_force=wind_drag,
            feedback_g=feedback, lever_moment=lever, reference_couple=couple,
            drag_n=-axial_aero)


def runner(mapping, loft_family, loft_coeff, substeps=4):
    original_class = candidate.SingleSurfaceGame
    class Configured(ForceFrameGame):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, force_mapping=mapping, loft_family=loft_family,
                             loft_coeff=loft_coeff, **kwargs)
    def run(case, profile):
        candidate.SingleSurfaceGame = Configured
        try:
            result = candidate.runner_for("wind_linear",substeps,"rad_s")(case,profile)
        finally:
            candidate.SingleSurfaceGame = original_class
        result["model"].update(force_coordinate_mapping=mapping,
            force_coordinate_evidence_grade="D", loft_family=loft_family,
            loft_coefficients=list(loft_coeff))
        return result
    return run


def atmospheric_values(altitude, sea_temperature_c):
    t0 = sea_temperature_c+273.15
    temperature = t0-.0065*altitude
    pressure = 101325.*(temperature/t0)**(G/(287.05287*.0065))
    return pressure/(287.05287*temperature), math.sqrt(1.4*287.05287*temperature)


def planar_force(mapping, alpha, qbar, mach, config, thrust, ca):
    """Return axial/normal total force in the wind basis, before gravity."""
    area = area_basis(config)
    normal = qbar*area*ca*alpha
    drag = qbar*(effective_cda0(mach,config)+effective_cda_alpha(alpha,0.,mach,config))
    if mapping == MAPPINGS[0]:
        axial, cross = -drag, normal
    elif mapping == MAPPINGS[1]:
        axial = -normal*math.sin(alpha)-drag*math.cos(alpha)
        cross = normal*math.cos(alpha)-drag*math.sin(alpha)
    else:
        axial = -normal*math.sin(alpha)-drag
        cross = normal*math.cos(alpha)
    return axial+thrust*math.cos(alpha), cross+thrust*math.sin(alpha), drag


def rmse(a,b):
    return float(np.sqrt(np.mean((np.asarray(a)-np.asarray(b))**2)))


def maximize_envelope(mapping, speed, config, rho, sound, mass, thrust, alpha_end):
    qbar=.5*rho*speed*speed
    def load(a):
        return planar_force(mapping,a,qbar,speed/sound,config,thrust,SurfaceCoefficients().ca)[1]/(mass*G)
    grid=np.linspace(0.,alpha_end,1801)
    values=np.array([load(float(a)) for a in grid])
    index=int(np.argmax(values))
    if index == 0 or index == len(grid)-1:
        optimum=float(grid[index])
    else:
        left,right=float(grid[index-1]),float(grid[index+1])
        ratio=(math.sqrt(5)-1)/2
        for _ in range(65):
            lo=right-ratio*(right-left);hi=left+ratio*(right-left)
            if load(lo) < load(hi): left=lo
            else: right=hi
        optimum=(left+right)/2
    maximum=load(optimum)
    ceiling=float(config["guidance"]["maximum_lateral_acceleration_g"])
    commanded=maximum
    active_angle=optimum
    if maximum >= ceiling:
        # Choose the first crossing, independently of observed tooltip AoA.
        crossings=np.flatnonzero(values>=ceiling)
        index=int(crossings[0])
        left,right=float(grid[max(index-1,0)]),float(grid[index])
        for _ in range(65):
            mid=(left+right)/2
            if load(mid)<ceiling:left=mid
            else:right=mid
        active_angle=(left+right)/2
        commanded=ceiling
    return dict(maximum_unclipped_g=maximum, maximizing_alpha_deg=math.degrees(optimum),
        ceiling_limited_g=commanded, ceiling_or_optimum_alpha_deg=math.degrees(active_angle),
        independent_alpha_maximization=True, observed_alpha_used=False)


def jari_analysis(reference, config, raw):
    settings=reference["settings"]
    rho,sound=atmospheric_values(settings["asl_m"],settings["sea_level_temperature_c_input"])
    mass=settings["displayed_current_mass_kg"]
    points=reference["points"]
    out={}
    for mapping in MAPPINGS:
        out[mapping]={}
        for label,domain in (("alpha_0_to_90deg",math.pi/2),
                             ("CyMaxAoA_1p3_as_radian_domain_D_hypothesis",float(raw["CyMaxAoA"]))):
            predictions=[]
            for point in points:
                p=maximize_envelope(mapping,point["speed_mps"],config,rho,sound,mass,raw["force"],domain)
                p.update(observed=point,speed_mps=point["speed_mps"],
                    max_G_residual=p["ceiling_limited_g"]-point["max_overload_g"],
                    alpha_residual_deg=p["ceiling_or_optimum_alpha_deg"]-point["aerodynamic_aoa_deg"])
                predictions.append(p)
            out[mapping][label]=dict(points=predictions,
                max_G_rmse_g=rmse([p["ceiling_limited_g"] for p in predictions],[p["max_overload_g"] for p in points]),
                angle_rmse_deg=rmse([p["ceiling_or_optimum_alpha_deg"] for p in predictions],[p["aerodynamic_aoa_deg"] for p in points]))
    return dict(density_kg_m3=rho,sound_speed_mps=sound,mass_kg=mass,
        nominal_thrust_n=raw["force"],definitions="Trajectory-normal specific force including thrust, no gravity subtraction. Envelope uses independent alpha maximization and 50g ceiling, not a guidance trajectory.",
        mappings=out)


def historical_conditional(case, config):
    obs=case["observed"]
    t=obs["t"];v=obs["speed_kmh"]/3.6;alpha=np.radians(obs["alpha_deg"])
    rho,sound=atmospheric_values(case["scenario"]["launch_altitude_m"],15.)
    geometry=config["geometry"];burn=config["propulsion"]["stages"][0]
    mass0=geometry["initial_mass_kg"]
    massend=mass0-burn["mass_lost_kg"]
    duration=burn["duration_s"];thrust0=burn["thrust_n"]
    mass=mass0-(mass0-massend)*np.minimum(t,duration)/duration
    thrust=np.where(t<duration,thrust0,0.)
    result={}
    for mapping in MAPPINGS:
        rhs=np.array([planar_force(mapping,float(a),.5*rho*s*s,s/sound,config,float(T),SurfaceCoefficients().ca)[0]/m
            for a,s,T,m in zip(alpha,v,thrust,mass)])
        prediction=v[0]+np.r_[0.,np.cumsum(np.diff(t)*.5*(rhs[:-1]+rhs[1:]))]
        result[mapping]=dict(age_s=t.tolist(),observed_speed_kmh=(v*3.6).tolist(),
            predicted_speed_kmh=(prediction*3.6).tolist(),speed_rmse_kmh=rmse(prediction*3.6,v*3.6),
            last_speed_residual_kmh=float((prediction[-1]-v[-1])*3.6))
    return dict(definition="Conditional axial-force quadrature using measured V and unsigned alpha, constant launch altitude, nominal mass/thrust; initialized at first observed speed. Gravity along trajectory is omitted because historical gamma is unavailable. This is a force diagnostic, not a free prediction.",mappings=result)


def paired_speed(cases,predictions):
    h,l=cases["H110"]["observed"],cases["L110"]["observed"]
    hp,lp=predictions["H110"],predictions["L110"]
    support=(h["t"]>=l["t"][0])&(h["t"]<=l["t"][-1])&(h["t"]<=hp["t"][-1])&(h["t"]<=lp["t"][-1])
    t=h["t"][support]
    observed=h["speed_kmh"][support]-np.interp(t,l["t"],l["speed_kmh"])
    predicted=np.interp(t,hp["t"],hp["speed_kmh"])-np.interp(t,lp["t"],lp["speed_kmh"])
    return dict(age_s=t.tolist(),observed_delta_kmh=observed.tolist(),predicted_delta_kmh=predicted.tolist(),
        delta_speed_rmse_kmh=rmse(observed,predicted),last_delta_residual_kmh=float(predicted[-1]-observed[-1]),compared_rows=len(t))


def classify(results,jari,conditional):
    baseline=results[MAPPINGS[0]]
    out={}
    for mapping in MAPPINGS[1:]:
        leaf=results[mapping];gates={}
        for name in ("L110","L27.5","L55"):
            a=baseline["cases"][name]["windows"]["early_0_3p5"]["metrics"]
            b=leaf["cases"][name]["windows"]["early_0_3p5"]["metrics"]
            for key in FIELDS:
                limit=(a[key]["rmse"]+5.) if key=="speed_kmh" else 1.25*a[key]["rmse"]+dict(alpha_deg=.025,omega_rad_s=.005,gamma_deg=.025,y_m=.5)[key]
                gates[name+"/"+key]=dict(limit=limit,actual=b[key]["rmse"],passed=b[key]["rmse"]<=limit)
        preserves=all(g["passed"] for g in gates.values())
        speed_improves=leaf["paired_speed"]["delta_speed_rmse_kmh"]<baseline["paired_speed"]["delta_speed_rmse_kmh"]
        old_improves=conditional["mappings"][mapping]["speed_rmse_kmh"]<conditional["mappings"][MAPPINGS[0]]["speed_rmse_kmh"]
        j= jari["mappings"][mapping]["alpha_0_to_90deg"]
        basej=jari["mappings"][MAPPINGS[0]]["alpha_0_to_90deg"]
        jari_improves=j["max_G_rmse_g"]<basej["max_G_rmse_g"] and j["angle_rmse_deg"]<basej["angle_rmse_deg"]
        passed=preserves and speed_improves and (old_improves or jari_improves)
        out[mapping]=dict(no_loft_gates=gates,no_loft_preserved=preserves,
            paired_speed_improves=speed_improves,historical_conditional_axial_improves=old_improves,
            Jari_G_and_angle_improve=jari_improves,qualified=passed,
            classification="qualifies_for_precision_check_not_native_engine_identification" if passed else "reject_full_force_frame_mapping")
    return out


def plot(results,jari,conditional,cases,predictions):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors=dict(zip(MAPPINGS,("#28799d","#c65539","#9163ab")))
    fig,axes=plt.subplots(2,2,figsize=(13,9))
    points=next(iter(jari["mappings"].values()))["alpha_0_to_90deg"]["points"]
    v=[p["speed_mps"] for p in points]
    axes[0,0].plot(v,[p["observed"]["max_overload_g"] for p in points],"ko",label="Jari tooltip")
    axes[0,1].plot(v,[p["observed"]["aerodynamic_aoa_deg"] for p in points],"ko",label="Jari tooltip")
    pair=results[MAPPINGS[0]]["paired_speed"]
    axes[1,0].plot(pair["age_s"],pair["observed_delta_kmh"],"k.",label="Game H110 - L110")
    cond=conditional["mappings"][MAPPINGS[0]]
    axes[1,1].plot(cond["age_s"],cond["observed_speed_kmh"],"k.",label="2.57 game speed")
    for mapping,color in colors.items():
        p=jari["mappings"][mapping]["alpha_0_to_90deg"]["points"]
        axes[0,0].plot(v,[x["ceiling_limited_g"] for x in p],".-",color=color,label=mapping)
        axes[0,1].plot(v,[x["ceiling_or_optimum_alpha_deg"] for x in p],".-",color=color,label=mapping)
        pair=results[mapping]["paired_speed"]
        axes[1,0].plot(pair["age_s"],pair["predicted_delta_kmh"],color=color,label=mapping)
        c=conditional["mappings"][mapping]
        axes[1,1].plot(c["age_s"],c["predicted_speed_kmh"],color=color,label=mapping)
    axes[0,0].set(xlabel="Speed [m/s]",ylabel="Independent envelope maximum [g]")
    axes[0,1].set(xlabel="Speed [m/s]",ylabel="Independent optimum / 50g crossing [deg]")
    axes[1,0].set(xlabel="Missile creation age [s]",ylabel="Free H110 minus L110 speed [km/h]")
    axes[1,1].set(xlabel="Missile age [s]",ylabel="Conditional axial speed [km/h]",title="Measured alpha/V in RHS, gravity omitted; not free prediction")
    for ax in axes.ravel():ax.legend(fontsize=7);ax.grid(alpha=.2)
    fig.suptitle("Whole force-frame hypotheses: frozen coefficients and unchanged PID / loft request",fontsize=14)
    fig.tight_layout(rect=(0,0,1,.96));fig.savefig(OUT/"force_frame_comparison.png",dpi=160);plt.close(fig)
    selected=("L110","H110","r77-1")
    fig,axes=plt.subplots(3,5,figsize=(20,10))
    for row,name in enumerate(selected):
        observed=cases[name]["observed"]
        for col,key in enumerate(FIELDS):
            ax=axes[row,col]
            if key in observed:ax.scatter(observed["t"],observed[key],color="k",s=10,label="Game")
            for mapping,color in colors.items():
                p=predictions[mapping][name]
                ax.plot(p["t"],p[key],color=color,label=mapping)
            ax.set(xlabel="Age [s]",ylabel=key,title=name if col==0 else None)
            ax.grid(alpha=.2)
    axes[0,0].legend(fontsize=7)
    fig.suptitle("Free creation-state integration; no observed missile forcing",fontsize=14)
    fig.tight_layout(rect=(0,0,1,.96));fig.savefig(OUT/"free_trajectory_comparison.png",dpi=150);plt.close(fig)


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    protected={p:p.read_bytes() for p in (candidate.FROZEN,candidate.SNAPSHOT)}
    loft=json.loads(LOFT.read_text());family=loft["selected_family"];coeff=loft["coefficients"]
    # Write the candidate declarations before loading any comparison data.
    (OUT/"declaration.json").write_text(json.dumps(dict(DECLARATION,
        fixed_outer_loft_family=family,fixed_outer_loft_coefficients=coeff),indent=2))
    cases={c["id"]:c for c in harness.load_cases(candidate.SNAPSHOT)
           if c["dataset"]=="native_2.59" or c["id"]=="r77-1"}
    raw_sources={}
    raw_rockets={}
    for version,folder in (("2.57.1.107","data/datamine"),("2.59.0.27","data/datamine_2.59.0.27")):
        rocketpath=ROOT/folder/"aces.vromfs.bin_u/gamedata/weapons/rocketguns/su_r_77_1.blkx"
        sharedpath=ROOT/folder/"aces.vromfs.bin_u/config/gameparams.blkx"
        rocket=json.loads(rocketpath.read_text())["rocket"];raw_rockets[version]=rocket
        old_capture=OUT/"global_257_source.json"
        exact_capture=json.loads(old_capture.read_text()) if version=="2.57.1.107" and old_capture.is_file() else None
        shared=(json.loads(sharedpath.read_text())["shellBallisticsParams"] if sharedpath.is_file()
                else exact_capture["shellBallisticsParams"] if exact_capture else {})
        raw_sources[version]=dict(rocket_path=str(rocketpath),rocket_sha256=hashlib.sha256(rocketpath.read_bytes()).hexdigest(),
            shared_path=str(sharedpath),shared_file_present=sharedpath.is_file(),
            shared_sha256=hashlib.sha256(sharedpath.read_bytes()).hexdigest() if sharedpath.is_file() else None,
            precise_historical_Exa_source=exact_capture,
            rocket_fields={k:rocket.get(k) for k in ("CxK","CxAoA","CyK","CyMachMult","CyMaxAoA","wingAreaMult","finsAoaHor","distFromCmToStab")},
            shared_fields={k:shared.get(k) for k in ("applyWingAreaMultToCxAoA","useCxiMach")},
            shared_CxAoA=shared.get("props",{}).get("CxAoA"),
            runtime_CxAoA=cases["r77-1" if version=="2.57.1.107" else "H110"]["profile"]["_model_config"]["aerodynamics"]["cx_vs_aoa"],
            global_source_note=("shared datamine field verified in this version" if sharedpath.is_file() else
                "2.57 shared gameparams working file absent; exact historical commit verified by Exa capture" if exact_capture else
                "2.57 shared gameparams file absent from sparse checkout; runtime shared value 9 is retained but not independently verified for 2.57"))
    results,predictions,arrays={},{},{}
    for mapping in MAPPINGS:
        records={};predictions[mapping]={}
        run=runner(mapping,family,coeff)
        for name,case in cases.items():
            result=run(case,case["profile"])
            prediction=harness.sample_series(result["samples"],case)
            record,aligned=harness.compare_series(case,prediction,result["summary"])
            record["model_runtime_metadata"]=result["model"]
            records[name]=record;predictions[mapping][name]=prediction
            arrays.update({mapping+"__"+name+"__model__"+k:v for k,v in prediction.items()})
            arrays.update({mapping+"__"+name+"__aligned__"+k:v for k,v in aligned.items()})
            print(mapping,name,"rows",record["compared_rows"],"V RMSE",round(record["windows"]["full"]["metrics"]["speed_kmh"]["rmse"],3),flush=True)
        results[mapping]=dict(cases=records,paired_speed=paired_speed(cases,predictions[mapping]))
    config=cases["H110"]["profile"]["_model_config"]
    jari=jari_analysis(json.loads(JARI.read_text()),config,raw_rockets["2.59.0.27"])
    old=historical_conditional(cases["r77-1"],cases["r77-1"]["profile"]["_model_config"])
    classification=classify(results,jari,old)
    precision={}
    for mapping,gate in classification.items():
        if gate["qualified"]:
            precision[mapping]={}
            fine_run=runner(mapping,family,coeff,8)
            for name in ("L110","H110","r77-1"):
                fine_result=fine_run(cases[name],cases[name]["profile"])
                fine=harness.sample_series(fine_result["samples"],cases[name]);coarse=predictions[mapping][name]
                mask=coarse["t"]<=fine["t"][-1]
                precision[mapping][name]={k:float(np.max(np.abs(coarse[k][mask]-np.interp(coarse["t"][mask],fine["t"],fine[k])))) for k in FIELDS}
    assert all(p.read_bytes()==b for p,b in protected.items())
    np.savez_compressed(OUT/"trajectories.npz",**arrays)
    plot(results,jari,old,cases,predictions)
    report=dict(status="bounded_force_coordinate_hypotheses_only",declaration=DECLARATION,
        raw_sources=raw_sources,coefficients=dataclasses.asdict(SurfaceCoefficients()),
        loft_source=str(LOFT),fixed_loft_family=family,fixed_loft_coefficients=coeff,
        force_maps=results,jari=jari,historical_conditional_axial=old,
        classification=classification,qualifying_mapping_precision=precision,
        verification=dict(all_free_prediction_arrays_finite=True,raw_PID_unchanged=True,
            frozen_and_profile_snapshot_bytes_unchanged=True,controller_hz=48,plant_substeps=4,
            missile_creation_state_only=True,source_observation_rows_preserved=True,no_time_shifts=True,
            no_coefficient_fit=True,no_production_change=True),
        limitations=["Force-coordinate mappings are D-level hypotheses; raw fields do not identify the engine force axes.",
            "Shared 1943x1.10 Mach table interpolation is an existing model hypothesis, not recovered engine code.",
            "Jari game version, force semantics, environment and propulsion-pressure correction are unverified; nominal 2.59 inputs are used explicitly.",
            "FreeLoft coefficients are prior replay calibration and remain fixed during this force comparison; changing forces tests their transfer, not a reoptimized outer request.",
            "Native b5 clock ties and all source observations remain; observation time operator is unresolved.",
            "Historical screenshot alpha is unsigned and historical target/scenario geometry is approximate; the conditional axial diagnostic omits unknown trajectory gravity."])
    (OUT/"results.json").write_text(json.dumps(harness.jsonable(report),indent=2,allow_nan=False))
    print("CLASSIFICATION",json.dumps({m:g["classification"] for m,g in classification.items()}),flush=True)


if __name__=="__main__":main()
