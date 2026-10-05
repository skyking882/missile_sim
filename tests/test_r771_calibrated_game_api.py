"""Acceptance of the additive calibrated game mode and its GUI contract."""
from __future__ import annotations

import copy
import importlib.util
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from aim120_model.profile_catalog import load_profile_catalog
from aim120_model.public_api import SimulationInputError, _case_from_scenario, simulate, validate_scenario
from aim120_model.r771_calibrated_game import (MODE_ID, ScenarioTarget, creation_state, public_definition)
from aim120_model.single_surface_game import RateAngleLoftGame, SingleSurfaceGame
from missile_gui.library import scan_library


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def profile():
    _, active = load_profile_catalog(ROOT)
    profiles, errors = scan_library(active["profiles_dir"], ROOT,
        expected_source_version=active["source_version"], expected_source_commit=active["source_commit"])
    assert not errors
    return next(p for p in profiles if p["missile_id"] == "su_r_77_1")


def scenario(**changes):
    return {"launch_speed_kmh": 1200., "launch_altitude_m": 6300., "launch_pitch_deg": 0.,
        "launch_heading_deg": 0., "target_speed_kmh": 1200., "target_altitude_m": 6300.,
        "initial_distance_m": 11000., "target_azimuth_deg": 0., "target_heading_deg": 0.,
        "target_vertical_heading_deg": 0., "target_constant_turn_g": 0.,
        "target_course_reference": "launch_axis", "observation_mode": "ideal_truth",
        "max_simulation_time_s": 1.5, "loft_enabled": False, **changes}


@pytest.mark.parametrize("enabled", (False, True))
def test_public_mode_is_free_and_preserves_profile_and_sample_contract(profile, enabled):
    before = copy.deepcopy(profile)
    result = simulate(profile, scenario(loft_enabled=enabled), model_mode=MODE_ID)
    assert profile == before
    assert result["model_mode"] == MODE_ID
    assert result["missile"]["status"] == "Experimental"
    assert result["missile"]["parameter_source"] == public_definition()["parameter_source"]
    assert result["model"]["raw_PID_preserved"] is True
    assert result["model"]["loft_coefficients"] == [.75, 1.3]
    assert result["model"]["raw_loft_fields"] == {"lofting_elevation_deg": 24.,
        "target_elevation_deg": -3.5, "angle_to_acceleration_multiplier": 17.5,
        "loft_omega_max_deg_s": .5}
    assert result["summary"]["loft_enabled"] is enabled
    assert result["summary"]["flight_time_s"] == pytest.approx(1.5)
    assert result["summary"]["termination_detail"] == "reached_scenario_time_limit"
    assert result["summary"]["minimum_distance_m"] <= min(s["distance_to_target_m"] for s in result["samples"])
    assert result["model"]["observation_mode"] == "ideal_truth"
    assert result["model"]["radar_model"] == "not_applicable"
    assert result["markers"][-1]["time_s"] == pytest.approx(1.5)
    for sample in result["samples"]:
        assert sample["loft_active"] is enabled
        for key in ("target_position_m", "target_velocity_mps", "commanded_acceleration_g",
                    "feedback_pitch_g", "feedback_yaw_g", "pitch_alpha_rad", "pitch_rate_rad_s",
                    "yaw_rate_rad_s", "drag_n", "mach", "actual_overload_g", "loft_bias_g"):
            assert key in sample
    json.dumps(result, allow_nan=False)


def test_no_loft_retains_base_dynamics_and_reused_instance_resets_guidance_memory(profile):
    cfg = copy.deepcopy(profile["_model_config"])
    cfg["guidance"]["lofting_enabled"] = False
    case = _case_from_scenario(validate_scenario(scenario()))
    initial = creation_state(cfg, case)
    target = ScenarioTarget(case["initial_conditions"], cfg["atmosphere"]["gravity_mps2"])
    base = SingleSurfaceGame(cfg, loft_cap_unit="rad_s").run(initial, target.state_at, end_time_s=1.5)
    candidate = RateAngleLoftGame(cfg, loft_cap_unit="rad_s")
    actual = candidate.run(initial, target.state_at, end_time_s=1.5)
    assert actual["samples"] == base["samples"]
    assert actual["summary"] == base["summary"]
    cfg["guidance"]["lofting_enabled"] = True
    candidate = RateAngleLoftGame(cfg, loft_cap_unit="rad_s")
    first = candidate.run(initial, target.state_at, end_time_s=1.5)
    second = candidate.run(initial, target.state_at, end_time_s=1.5)
    assert first == second
    evaluation = candidate.evaluate(initial, .123, (0., 0.))
    memory = (candidate.initial_range, candidate.minimum_weight)
    candidate.request(initial, target.state_at(.123), .123, evaluation)
    assert (candidate.initial_range, candidate.minimum_weight) == memory


@pytest.mark.parametrize("changes,match", [
    ({"observation_mode": "sensor_track"}, "理想观测"),
    ({"simulation_dt_s": .02}, "48 Hz"),
    ({"datalink_enabled": False}, "数据链"),
    ({"datalink_disconnect_time_s": 1.}, "数据链"),
])
def test_public_mode_rejects_unimplemented_options(profile, changes, match):
    with pytest.raises(SimulationInputError, match=match):
        simulate(profile, scenario(**changes), model_mode=MODE_ID)


def test_public_mode_rejects_wrong_unit_version_and_unknown_mode(profile):
    other = copy.deepcopy(profile)
    other["missile_id"] = "us_aim_120a"
    with pytest.raises(SimulationInputError, match="只支持 R-77-1"):
        simulate(other, scenario(), model_mode=MODE_ID)
    other = copy.deepcopy(profile)
    other["provenance"]["source_version"] = "2.57.1.107"
    with pytest.raises(SimulationInputError, match="2.59.0.27"):
        simulate(other, scenario(), model_mode=MODE_ID)
    other = copy.deepcopy(profile)
    other["provenance"]["source_commit"] = "different"
    with pytest.raises(SimulationInputError, match="2.59.0.27"):
        simulate(other, scenario(), model_mode=MODE_ID)
    with pytest.raises(SimulationInputError, match="未知模型模式"):
        simulate(profile, scenario(), model_mode="not_a_mode")


def test_public_mode_reports_continuous_proximity_fuse(profile):
    result = simulate(profile, scenario(initial_distance_m=100., target_speed_kmh=0.), model_mode=MODE_ID)
    assert result["summary"]["termination_event"] == "proximity_fuse"
    assert result["summary"]["minimum_distance_m"] <= profile["_model_config"]["guidance"]["proximity_radius_m"]
    assert result["samples"][-1]["time_s"] == result["summary"]["flight_time_s"]
    # The existing kernel intersects each plant-step relative chord, then
    # integrates to that fraction; the returned curved state is within 1 mm.
    assert result["summary"]["terminal_distance_m"] == pytest.approx(
        profile["_model_config"]["guidance"]["proximity_radius_m"], abs=1e-3)


def test_scenario_target_preserves_course_convention_and_turn_is_pure(profile):
    cfg = profile["_model_config"]
    def target(reference, heading=0., turn=0.):
        case = _case_from_scenario(validate_scenario(scenario(target_azimuth_deg=30.,
            target_heading_deg=heading, target_course_reference=reference, target_constant_turn_g=turn)))
        return ScenarioTarget(case["initial_conditions"], cfg["atmosphere"]["gravity_mps2"])
    los = target("relative_to_los")
    axis = target("launch_axis", 30.)
    world = target("absolute_world", 210.)
    for model in (axis, world):
        assert model.state_at(1.).velocity == pytest.approx(los.state_at(1.).velocity)
        assert model.state_at(1.).position == pytest.approx(los.state_at(1.).position)
    curved = target("absolute_world", 0., 3.)
    expected = curved.state_at(2.)
    curved.state_at(1.)
    curved.state_at(3.)
    assert curved.state_at(2.) == expected
    dt = 1e-5
    a, b = curved.state_at(2.-dt), curved.state_at(2.+dt)
    derivative = tuple((y-x)/(2*dt) for x, y in zip(a.position, b.position))
    assert derivative == pytest.approx(expected.velocity, abs=1e-7)


@pytest.mark.parametrize("name", ("H110", "L110"))
def test_promoted_core_exactly_matches_selected_saved_free_loft(name):
    np = pytest.importorskip("numpy")
    reference = ROOT / "outputs/unified_game_replays/free_loft/trajectories.npz"
    snapshot = ROOT / "outputs/unified_game_replays/corrected_input/runtime_profiles.json"
    if not reference.is_file() or not snapshot.is_file():
        pytest.skip("research acceptance artifacts are not included in this checkout")
    helper_path = ROOT / "scripts/compare_unified_game_replays.py"
    spec = importlib.util.spec_from_file_location("calibrated_mode_acceptance", helper_path)
    helper = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(ROOT / "scripts"))
    try:
        spec.loader.exec_module(helper)
    finally:
        sys.path.pop(0)
    case = next(c for c in helper.load_cases(snapshot) if c["id"] == name)
    cfg = copy.deepcopy(case["profile"]["_model_config"])
    cfg["guidance"]["lofting_enabled"] = case["scenario"]["loft_enabled"]
    initial = copy.deepcopy(case["_creation_state"])
    x, y, z, w = case["creation"]["u16_1"]
    initial.orientation_quaternion = (w, x, y, z)
    result = RateAngleLoftGame(cfg, loft_cap_unit="rad_s").run(initial,
        case["_target_model"].state_at, end_time_s=case["scenario"]["max_simulation_time_s"])
    prediction = helper.sample_series(result["samples"], case)
    with np.load(reference) as expected:
        for key, values in prediction.items():
            assert np.array_equal(values, expected[name+"__model__"+key]), key


def test_gui_calibrated_mode_roundtrip_and_restore_general_scenario():
    node = shutil.which("node")
    if not node:
        pytest.skip("Node.js is unavailable for GUI scenario contract check")
    script_path = ROOT / "src/missile_gui/static/app.js"
    setup = r'''
const fs = require("fs"), vm = require("vm"), assert = require("assert/strict");
const values = JSON.parse(process.argv[2]), definition = JSON.parse(process.argv[3]);
const elements = new Map(), fields = [{}];
function element(key) {
  if (!elements.has(key)) elements.set(key, { type:key==="#loft-enabled-toggle"?"checkbox":"text", value:"", checked:false, disabled:false, textContent:"", innerHTML:"",
    classList:{add(){},remove(){},toggle(){}}, querySelectorAll(){return fields;}, checkValidity(){return true;} });
  return elements.get(key);
}
for (const [key,value] of Object.entries(values)) {const e=element(`[name="${key}"]`);e.value=value==null?"":value;e.checked=Boolean(value);}
element("#observation-mode-select").value = "sensor_track";
element("#target-course-reference-select").value = values.target_course_reference;
element("#loft-enabled-toggle").checked = values.loft_enabled;
class FormData {entries(){return Object.entries(values).map(([key])=>[key,element(`[name="${key}"]`).value]);}}
const context = {console, FormData, definition, assert, values,
  document:{querySelector:element,querySelectorAll(){return[];}}, setTimeout,clearTimeout};
vm.createContext(context);
const code=fs.readFileSync(process.argv[1],"utf8").split('$("#missile-select").addEventListener')[0];
vm.runInContext(code,context);
vm.runInContext(`
state.profiles=[{id:"su_r_77_1",name:"R-77-1",country:"USSR",series:"R-77-1",runnable:true,status:"Experimental",parameter_source:definition.parameter_source},
{id:"us_aim_120a",name:"AIM",country:"US",series:"AIM",runnable:true,status:"Experimental"}];
state.experimentalModes=[definition]; state.selected=state.profiles[1];
let exported; download=(name,content)=>{exported=JSON.parse(content);};
setModelMode(definition.mode_id);
assert.equal(state.selected.id,"su_r_77_1");
assert.equal($("#observation-mode-select").value,"ideal_truth");
assert.equal($("#observation-mode-select").disabled,true);
assert.equal($("#missile-select").disabled,true);
assert.equal($("#preset-select").disabled,false);
assert.equal($("#scenario-form").querySelectorAll("fieldset")[0].disabled,false);
applyScenario({...values,initial_distance_m:12345,loft_enabled:true});
$("#observation-mode-select").value="ideal_truth";
$("#loft-enabled-toggle").checked=true;
exportScenario(); assert.equal(exported.model_mode,definition.mode_id);
assert.equal(exported.scenario.initial_distance_m,12345);
assert.equal(exported.scenario.loft_enabled,true);
setModelMode("general");
assert.equal(state.selected.id,"us_aim_120a");
assert.equal($("#observation-mode-select").value,"sensor_track");
assert.equal($("#loft-enabled-toggle").checked,false);
assert.equal(collectScenario().initial_distance_m,values.initial_distance_m);
`,context);
// applyScenario uses named controls; synchronize the select alias in this small DOM.
context.document.querySelector('[name="observation_mode"]').value="sensor_track";
vm.runInContext(`
(async()=>{await importScenario({text:async()=>JSON.stringify(exported)});
assert.equal(state.modelMode,definition.mode_id);
assert.equal(collectScenario().initial_distance_m,12345);
assert.equal(collectScenario().loft_enabled,true);
const bad={...exported,parameter_source:{version:"2.57.1.107",commit:"wrong"}};
await importScenario({text:async()=>JSON.stringify(bad)});
assert.ok($("#error-alert").textContent.includes("已拒绝导入"));
})().catch(e=>{console.error(e);throw e;});
`,context);
'''
    # Native selectors refer to the same select by id and name in a real DOM.
    setup = setup.replace('if (!elements.has(key)) elements.set(key,',
        'if (key === \'[name="observation_mode"]\') key="#observation-mode-select"; '
        'if (key === \'[name="target_course_reference"]\') key="#target-course-reference-select"; '
        'if (key === \'[name="loft_enabled"]\') key="#loft-enabled-toggle"; '
        'if (!elements.has(key)) elements.set(key,')
    result = subprocess.run([node, "-e", setup, str(script_path), json.dumps(scenario()),
        json.dumps(public_definition())], text=True, capture_output=True)
    assert result.returncode == 0, result.stdout+result.stderr
