"""Conditional single-surface force comparisons with captured Jari tooltips.

This is an algebraic force check, not a simulation of Jari or game guidance.
Only displayed reference values are read. No plant, profile or PID is changed.
Three predeclared simple force bases are compared at the displayed angle:
wind-linear, body-linear projected to wind normal, and body-sine projected.
The latter two add no body/wing/fin split. Two fixed load conventions are
reported separately; no fitted gravity offset or per-point gain is used.
"""
from __future__ import annotations

import csv
import itertools
import json
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from aim120_model.atmosphere import StandardAtmosphere
import analyze_r771_frozen_loft as loft

OUT = ROOT / "outputs/jari_envelope_comparison"
REFERENCE = ROOT / "outputs/jari_reference/r771_asl6300_t0.json"
G = 9.80665
FORCE_BASES = {
    "wind_linear": lambda a: a,
    "body_linear": lambda a: a * np.cos(a),
    "body_sine": lambda a: np.sin(a) * np.cos(a),
}


def raw_rocket(version):
    folder = "data/datamine" if version == "2.57.1.107" else "data/datamine_2.59.0.27"
    path = ROOT / folder / "aces.vromfs.bin_u/gamedata/weapons/rocketguns/su_r_77_1.blkx"
    return json.loads(path.read_text())["rocket"], str(path.relative_to(ROOT))


def atmosphere(height, sea_temperature_c=15.):
    # Source pressure/lapse/R equal to the existing StandardAtmosphere;
    # T0 differs only by the explicitly displayed site setting.
    t0 = sea_temperature_c + 273.15
    temp = t0 - .0065 * height
    pressure = 101325. * (temp / t0) ** (G / (287.05287 * .0065))
    return pressure / (287.05287 * temp)


def rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def jari_check(reference, raw, coefficient):
    settings = reference["settings"]
    rho = atmosphere(settings["asl_m"], settings["sea_level_temperature_c_input"])
    area = math.pi * raw["caliber"] ** 2 / 4 * raw["wingAreaMult"]
    mass = settings["displayed_current_mass_kg"]
    thrust = raw["force"]
    points = reference["points"]
    v = np.array([p["speed_mps"] for p in points])
    alpha = np.radians([p["aerodynamic_aoa_deg"] for p in points])
    observed = np.array([p["max_overload_g"] for p in points])
    qs = .5 * rho * v ** 2 * area
    thrust_g = thrust * np.sin(alpha) / (mass * G)
    anchor = (v >= 348.) & (v <= 702.)
    result = {}
    for name, basis in FORCE_BASES.items():
        feature = qs * basis(alpha) / (mass * G)
        result[name] = {}
        for gravity in (0., 1.):
            # Each convention has exactly one algebraic coefficient estimate
            # from the four uncapped near-38 degree points. This is calibration,
            # and its transfer residuals are reported without picking a winner.
            demand = observed + gravity - thrust_g
            fitted = float(feature[anchor] @ demand[anchor] / (feature[anchor] @ feature[anchor]))
            inferred = demand / feature
            prediction = coefficient * feature + thrust_g - gravity
            fitted_prediction = fitted * feature + thrust_g - gravity
            intervals = []
            for p in points:
                values = []
                for dv, dn, da in itertools.product((-.05, .05), (-.005, .005), (-.005, .005)):
                    speed = p["speed_mps"] + dv
                    angle = math.radians(p["aerodynamic_aoa_deg"] + da)
                    n = p["max_overload_g"] + dn
                    values.append((mass * G * (n + gravity) - thrust * math.sin(angle)) /
                                  (.5 * rho * speed ** 2 * area * float(basis(angle))))
                intervals.append([min(values), max(values)])
            result[name][str(int(gravity))] = dict(
                effective_coefficient_per_rad=inferred.tolist(),
                display_rounding_only_coefficient_intervals=intervals,
                fixed_coefficient_predictions_g=prediction.tolist(),
                fixed_coefficient_rmse_all8_g=rmse(observed, prediction),
                fixed_coefficient_rmse_anchor4_g=rmse(observed[anchor], prediction[anchor]),
                calibrated_anchor_coefficient=fitted,
                calibrated_anchor_predictions_all8_g=fitted_prediction.tolist(),
                calibrated_anchor_rmse_anchor4_g=rmse(observed[anchor], fitted_prediction[anchor]),
                calibrated_anchor_transfer_rmse_low2_g=rmse(observed[:2], fitted_prediction[:2]),
                calibrated_anchor_transfer_rmse_cap2_g=rmse(observed[-2:], fitted_prediction[-2:]),
            )
    # Literal coefficient-cap candidate only: CyMaxAoA semantic hypothesis.
    # It is not a claim that the actual game applies this q*S cap.
    cap_prediction = (qs * np.minimum(coefficient * alpha, raw["CyMaxAoA"]) +
                      thrust * np.sin(alpha)) / (mass * G)
    return dict(density_kg_m3=rho,standard_ISA_density_kg_m3=StandardAtmosphere().sample(settings["asl_m"]).density_kg_m3,
                reference_area_m2=area,nominal_raw_thrust_n=thrust,mass_kg=mass,
                alpha_rad=alpha.tolist(),speed_mps=v.tolist(),observed_g=observed.tolist(),
                anchor_indices=np.flatnonzero(anchor).tolist(),comparison=result,
                literal_raw_CyMaxAoA_cap_predictions_g=cap_prediction.tolist(),
                literal_raw_cap_rmse_all8_g=rmse(observed,cap_prediction))


def native_check(coefficient):
    raw, _ = loft.old.batch.load_all()
    output = {}
    for name, record in raw.items():
        s = loft.make_loft_shot(record) if name.startswith("H") else loft.old.make_shot(record, "b5")
        obs = s["observed"]
        mask = obs["t"] <= 3.5 + 1e-10
        t, a, gamma, v, y = [obs[k][mask] for k in ("t", "alpha", "gamma", "speed2", "y")]
        p = s["p"]
        m = p["mass0_kg"] - (p["mass0_kg"] - p["mass_end_kg"]) * np.minimum(t,p["time_fire_s"]) / p["time_fire_s"]
        thrust = np.where(t < p["time_fire_s"], p["thrust_n"], 0.)
        density = np.array([loft.model.isa_density(float(h)) for h in y])
        area = math.pi * p["caliber_m"]**2 / 4 * p["wing_area_mult"]
        qs = .5 * density * v**2 * area
        rec = dict(age_s=t.tolist(),observed_gamma_deg=np.degrees(gamma).tolist(),
                   observed_alpha_deg=np.degrees(a).tolist(),observed_altitude_m=y.tolist(),
                   native_rows_retained=int(mask.sum()),zero_length_native_intervals=int(np.count_nonzero(np.diff(t)==0)),
                   force_laws={})
        for label,basis in FORCE_BASES.items():
            normal_g = (qs * coefficient * basis(a) + thrust*np.sin(a)) / (m*G)
            rhs = G*(normal_g-np.cos(gamma))/v
            prediction = gamma[0]+np.r_[0.,np.cumsum(np.diff(t)*.5*(rhs[:-1]+rhs[1:]))]
            rec["force_laws"][label] = dict(predicted_gamma_deg=np.degrees(prediction).tolist(),
                gamma_rmse_deg=rmse(np.degrees(gamma),np.degrees(prediction)),
                final_gamma_error_deg=float(np.degrees(prediction[-1]-gamma[-1])),
                normal_specific_force_g=normal_g.tolist())
        output[name] = rec
    return output


def historical_check(raw, coefficient):
    with (ROOT / "data/replays/r77_1_level_20260824.tsv").open() as stream:
        rows = list(csv.DictReader(stream,delimiter="\t"))
    t, v, a, n = np.array([[float(row[k]) for k in ("missile_flight_time_s","speed_kmh","angle_of_attack_deg","overload_g")] for row in rows]).T
    v = v/3.6
    a = np.radians(a)
    m = raw["mass"] - (raw["mass"]-raw["massEnd"])*np.minimum(t,raw["timeFire"])/raw["timeFire"]
    thrust = np.where(t<raw["timeFire"],raw["force"],0.)
    area = math.pi * raw["caliber"]**2/4*raw["wingAreaMult"]
    qs = .5*StandardAtmosphere().sample(6300.).density_kg_m3*v**2*area
    result = dict(age_s=t.tolist(),observed_g=n.tolist(),alpha_deg=np.degrees(a).tolist(),
                  nominal_fixed_altitude_m=6300.,reference_area_m2=area,force_laws={})
    for label,basis in FORCE_BASES.items():
        inferred=(m*G*n-thrust*np.sin(a))/(qs*basis(a))
        prediction=(qs*coefficient*basis(a)+thrust*np.sin(a))/(m*G)
        result["force_laws"][label] = dict(predicted_g=prediction.tolist(),effective_coefficient=inferred.tolist(),windows={})
        for window,lo,hi in (("all",0.,9.),("early",.2,3.5),("large_aoa",.7,2.3),("late",6.,7.7)):
            mask=(t>=lo)&(t<=hi)
            result["force_laws"][label]["windows"][window] = dict(n=int(mask.sum()),
                rmse_g=rmse(n[mask],prediction[mask]),
                inferred_coefficient_median=float(np.median(inferred[mask])),
                inferred_coefficient_range=[float(inferred[mask].min()),float(inferred[mask].max())])
    return result


def plot(result):
    j, h, native = result["jari"],result["historical_257"],result["native_259"]
    colors = dict(wind_linear="#a653a1",body_linear="#237cb3",body_sine="#d2802a")
    fig,axs=plt.subplots(2,2,figsize=(13,9))
    axs[0,0].plot(j["speed_mps"],j["observed_g"],"ko-",label="Jari tooltip max G")
    for label,color in colors.items():
        values=j["comparison"][label]["0"]
        axs[0,0].plot(j["speed_mps"],values["fixed_coefficient_predictions_g"],".-",color=color,label=label)
        axs[0,1].plot(j["speed_mps"],values["effective_coefficient_per_rad"],"o-",color=color,label=label)
        axs[0,1].plot(j["speed_mps"],j["comparison"][label]["1"]["effective_coefficient_per_rad"],":",color=color)
        axs[1,0].plot(h["age_s"],h["force_laws"][label]["predicted_g"],color=color,label=label)
        axs[1,1].plot(native["H110"]["age_s"],native["H110"]["force_laws"][label]["predicted_gamma_deg"],color=color,label=label)
    axs[0,1].axhline(result["frozen_coefficient"],color=".35",ls="--",label="Frozen 14.6069")
    axs[1,0].plot(h["age_s"],h["observed_g"],"k.",label="2.57 screenshot")
    axs[1,1].plot(native["H110"]["age_s"],native["H110"]["observed_gamma_deg"],"k.",label="2.59 decoded H110")
    axs[0,0].set(xlabel="Speed [m/s]",ylabel="Load [g]",title="Force evaluated at Jari's displayed AoA; not a predicted envelope")
    axs[0,1].set(xlabel="Speed [m/s]",ylabel="Conditional force coefficient",title="Solid: specific load; dotted: horizontal gravity correction +1g")
    axs[1,0].set(xlabel="Missile age [s]",ylabel="Specific normal load [g]",title="2.57: observed speed/AoA, nominal mass/thrust, fixed 6300 m")
    axs[1,1].set(xlabel="Native b5 age [s]",ylabel="Flight-path angle [deg]",title="2.59: integrate gamma with observed states and altitude")
    for ax in axs.ravel():ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.suptitle("Single equivalent-surface force candidates: no parameter retuning or controller changes",fontsize=13)
    fig.tight_layout(rect=(0,0,1,.96))
    fig.savefig(OUT/"force_comparison.png",dpi=160)
    plt.close(fig)


def main():
    reference=json.loads(REFERENCE.read_text())
    frozen_path=ROOT/"config/r771_simple_joint_frozen.json"
    protected=frozen_path.read_bytes()
    coefficient=float(json.loads(protected)["coefficients"][0])
    raw259,path259=raw_rocket("2.59.0.27")
    raw257,path257=raw_rocket("2.57.1.107")
    assert not any(key in r for r in (raw259,raw257) for key in ("CyK","CyMachMult"))
    result=dict(status="conditional_force_analysis_not_game_or_Jari_model_validation",
        reference=str(REFERENCE.relative_to(ROOT)),frozen_coefficient=coefficient,
        formulas={"wind_linear":"L=q*S*C*alpha", "body_linear":"N=q*S*C*alpha; L=N*cos(alpha)",
                  "body_sine":"N=q*S*C*sin(alpha); L=N*cos(alpha)","load":"(L+T*sin(alpha))/(m*g)-fixed_gravity_offset"},
        raw_sources={"2.59.0.27":path259,"2.57.1.107":path257},
        raw_field_comparison={k:{"2.57":raw257.get(k),"2.59":raw259.get(k)} for k in ("force","mass","massEnd","timeFire","caliber","wingAreaMult","finsAoaHor","distFromCmToStab","CyMaxAoA","CyK","CyMachMult")},
        jari=jari_check(reference,raw259,coefficient),
        historical_257=historical_check(raw257,coefficient),native_259=native_check(coefficient),
        limitations=["Jari tooltip max load and aerodynamic AoA are conditionally treated as simultaneous force values; hidden state and site equations are unverified.",
            "Raw nominal thrust, sea-level pressure, ISA lapse and t=0 displayed mass are assumptions for the Jari force inversion; pressure-to-thrust site correction is unverified.",
            "Gravity conventions are fixed separate alternatives, never a fitted offset; envelope geometry itself is unverified.",
            "Display-rounding intervals exclude uncertainty in environment, field meaning, force axes and propulsion.",
            "2.57 screenshot check uses unsigned total AoA/load and nominal constant altitude; altitude history, bank and signed force components are unavailable.",
            "2.59 check forces measured alpha, speed, gamma and altitude into RHS; it does not predict these states or guidance.",
            "No maximum AoA, fin equilibrium, closed-loop recovery, hit or free-speed behavior is validated.",
            "Neither R-77-1 version declares CyK/CyMachMult. Literal CyMaxAoA coefficient cap is only a tested semantic hypothesis."])
    assert frozen_path.read_bytes()==protected
    for group in (result["jari"]["comparison"],):
        for records in group.values():
            for record in records.values():
                assert np.isfinite(record["fixed_coefficient_predictions_g"]).all()
    OUT.mkdir(parents=True,exist_ok=True)
    plot(result)
    result["verification"]={"frozen_config_unchanged":True,"all_reference_points":len(reference["points"]),
                            "anchor_points":4,"native_cases":len(result["native_259"]),
                            "all_formulas_finite":True,"core_or_profile_edits":False}
    (OUT/"results.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps({"output":str(OUT),"Jari_body_linear_anchor":result["jari"]["comparison"]["body_linear"]["0"]["calibrated_anchor_coefficient"],
        "257_body_linear_large_alpha_rmse_g":result["historical_257"]["force_laws"]["body_linear"]["windows"]["large_aoa"]["rmse_g"],
        "259_H110_body_linear_gamma_rmse_deg":result["native_259"]["H110"]["force_laws"]["body_linear"]["gamma_rmse_deg"]}))


if __name__=="__main__":main()
