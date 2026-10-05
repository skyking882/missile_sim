import copy
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from import_datamine_missiles import _global_ballistics_parameters, _pid_schedule, _profile, _stages
from aim120_model.profile_adapter import build_h2_candidate_config


def test_explicit_lift_fields_and_indexed_pid_survive_import(tmp_path):
    raw = json.loads((ROOT / "data/datamine/aces.vromfs.bin_u/gamedata/weapons/rocketguns/us_aim_120a.blkx").read_text())
    rocket = raw["rocket"]
    rocket.update(CyK=5.0, CyMaxAoA=1.3,
                  CyMachMult={"CyMachMult1": [1.0, 1.0], "CyMachMult0": [0.8, 0.815]})
    ap = rocket["guidance"]["guidanceAutopilot"]
    for key in ("accelControlProp", "accelControlIntg", "accelControlDiff", "accelControlIntgLim"):
        ap.pop(key)
    ap["pid0"] = dict(time=2.5, prop=0.000542, intg=0.02081, diff=0.000073, intgLim=1.0)
    ap["pid1"] = dict(time=3.5, prop=0.000542, intg=0.02081, diff=0.000028, intgLim=1.0)
    path = tmp_path / "us_aim_120a.blkx"
    path.write_text(json.dumps(raw))
    result = _profile(raw, rocket, path, [path], tmp_path, "a" * 40, "2.58.0.36",
                      "2026-09-12", "us_aim_120a", "us_aim_120a", [])
    assert result["aerodynamics"]["cy_k"] == 5.0
    assert result["aerodynamics"]["max_cy_at_aoa"] == dict(value=1.3, aoa_rad=None)
    assert result["aerodynamics"]["mach_lift_correction"] == dict(
        model="lookup_table", parameters=[[0.8, 0.815], [1.0, 1.0]])
    assert result["control"]["pid"]["p"] is None
    assert result["control"]["pid_schedule"][1]["d"] == 0.000028
    assert result["control"]["parameter_sources"]["pid_schedule"]["kind"] == "datamine"
    schema = json.loads((ROOT / "schemas/missile_profile.schema.json").read_text())
    from missile_lab.validator import _schema_check
    assert _schema_check(result, schema, "profile", schema) == []
    defaults = json.loads((ROOT / "config/profile_h2_runtime_defaults.json").read_text())
    scheduled_config, assumptions = build_h2_candidate_config(result, defaults)
    assert scheduled_config["control"]["pid"]["schedule"] == result["control"]["pid_schedule"]
    assert any("elapsed seconds since launch" in note for note in assumptions)
    from aim120_model.profile_adapter import _mach_settings
    assumptions = []
    table_settings = _mach_settings(result["aerodynamics"]["mach_lift_correction"], {}, assumptions, "lift")
    assert table_settings == {"model": "lookup_table", "parameters": [[0.8, 0.815], [1.0, 1.0]]}
    assert "not asserted as Gaijin's internal rule" in assumptions[0]


def test_pid_schedule_rejects_missing_or_nonmonotonic_time():
    row = dict(time=2.5, prop=1.0, intg=0.1, diff=0.01, intgLim=1.0)
    with pytest.raises(ValueError, match="strictly increasing"):
        _pid_schedule(dict(pid0=row, pid1=copy.deepcopy(row)))
    row.pop("time")
    with pytest.raises(ValueError, match="missing/nonfinite"):
        _pid_schedule(dict(pid0=row))


def test_nested_impulse_mass_end_is_derived_without_emitting_zero_placeholder():
    stages, _ = _stages({
        "mass": 150.0,
        "propulsion0": {
            "impulse0": {"time": 1.5, "force": 42000.0, "massEnd": 120.0},
            "impulse1": {"time": 0.0, "force": 0.0, "massEnd": 90.0},
        },
    })
    assert len(stages) == 1
    assert stages[0]["duration_s"] == 1.5
    assert stages[0]["thrust_n"] == 42000.0
    assert stages[0]["mass_lost_kg"] == 30.0
    assert stages[0]["parameter_sources"]["mass_lost_kg"]["kind"] == "derived"
    assert stages[0]["parameter_sources"]["mass_lost_kg"]["source_field"] == (
        "rocket.mass - rocket.propulsion0.impulse0.massEnd"
    )


def test_global_cx_aoa_is_reported_separately_from_missile_profiles(tmp_path):
    config_dir = tmp_path / "aces.vromfs.bin_u" / "config"
    config_dir.mkdir(parents=True)
    (config_dir / "gameparams.blkx").write_text(json.dumps({
        "shellBallisticsParams": {
            "applyWingAreaMultToCxAoA": False,
            "useCxiMach": True,
            "props": {"CxAoA": 9.0},
        },
    }))
    parameters = _global_ballistics_parameters(tmp_path)
    assert parameters["CxAoA"]["value"] == 9.0
    assert parameters["applyWingAreaMultToCxAoA"]["value"] is False
    assert parameters["useCxiMach"]["value"] is True
    assert parameters["CxAoA"]["source_field"] == "shellBallisticsParams.props.CxAoA"


@pytest.mark.parametrize("raw_fields_present", [True, False])
def test_raw_loft_fields_survive_import_and_schema_without_unit_claims(tmp_path, raw_fields_present):
    source = ROOT / "data/datamine/aces.vromfs.bin_u/gamedata/weapons/rocketguns/su_r_77_1.blkx"
    raw = json.loads(source.read_text())
    fields = {
        "loft_target_elevation_deg": ("loftTargetElevation", -6.25),
        "loft_angle_to_acceleration_multiplier": ("loftAngleToAccelMult", 13.75),
        "loft_omega_max_deg_s": ("loftTargetOmegaMax", 0.625),
    }
    ap = raw["rocket"]["guidance"]["guidanceAutopilot"]
    for raw_key, value in fields.values():
        if raw_fields_present:
            ap[raw_key] = value
        else:
            ap.pop(raw_key, None)
    path = tmp_path / source.name
    path.write_text(json.dumps(raw))
    profile = _profile(raw, raw["rocket"], path, [path], tmp_path, "a" * 40,
                       "2.57.1.107", "2026-10-01", "su_r_77_1", "su_r_77_1", [])
    for key, (raw_key, value) in fields.items():
        assert profile["guidance"][key] == (value if raw_fields_present else None)
        entry = profile["guidance"]["parameter_sources"][key]
        assert entry["kind"] == "assumed"
        assert entry["source_field"] == (
            f"rocket.guidance.guidanceAutopilot.{raw_key}" if raw_fields_present else None
        )
    from missile_lab.validator import _schema_check
    schema = json.loads((ROOT / "schemas/missile_profile.schema.json").read_text())
    assert _schema_check(profile, schema, "profile", schema) == []
