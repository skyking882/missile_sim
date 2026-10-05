from __future__ import annotations

import json
import copy
from pathlib import Path

import pytest

from aim120_model.profile_catalog import datamine_path_for_profile, load_profile_catalog
from missile_gui.library import public_profile, scan_library


ROOT = Path(__file__).resolve().parents[1]


def test_active_catalog_uses_the_verified_259_profile_set():
    _, active = load_profile_catalog(ROOT)
    assert active["name"] == "2.59.0.27"
    assert active["source_commit"] == "4ac255b829a27346b9300fccdc138362c2535fbe"
    assert active["manifest"]["included_profile_count"] == 122

    profiles, errors = scan_library(
        active["profiles_dir"],
        ROOT,
        expected_source_version=active["source_version"],
        expected_source_commit=active["source_commit"],
    )
    assert errors == []
    assert len(profiles) == 122
    assert all(
        profile["provenance"]["source_version"] == active["source_version"]
        and profile["provenance"]["source_commit"] == active["source_commit"]
        for profile in profiles
    )
    assert sum(bool(profile.get("_model_config")) for profile in profiles) == 118
    scheduled_pid = next(profile for profile in profiles if profile["missile_id"] == "us_aim_120a")
    assert scheduled_pid["_model_config"]["control"]["pid"]["schedule"] == scheduled_pid["control"]["pid_schedule"]


def test_profile_raw_source_resolution_never_falls_back_across_versions():
    old = json.loads((ROOT / "data/profile_sets/2.57.1.107/missiles/su_r_77_1.json").read_text())
    new = json.loads((ROOT / "data/profile_sets/2.59.0.27/missiles/su_r_77_1.json").read_text())
    assert datamine_path_for_profile(ROOT, old).relative_to(ROOT) == Path("data/datamine")
    assert datamine_path_for_profile(ROOT, new).relative_to(ROOT) == Path("data/datamine_2.59.0.27")

    unknown = json.loads(json.dumps(new))
    unknown["provenance"]["source_commit"] = "0" * 40
    try:
        datamine_path_for_profile(ROOT, unknown)
    except ValueError as exc:
        assert "no unique raw datamine checkout" in str(exc)
    else:
        raise AssertionError("unknown profile provenance must fail rather than use another release")


def test_new_explicit_fields_and_global_cx_aoa_are_preserved_with_source():
    profiles, errors = scan_library(ROOT / "data/profile_sets/2.59.0.27/missiles", ROOT)
    assert errors == []
    manifest = json.loads(
        (ROOT / "data/profile_sets/2.59.0.27/data/aam_non_tvc_manifest.json").read_text()
    )
    assert manifest["global_parameters"]["CxAoA"]["value"] == 9.0
    assert manifest["global_parameters"]["applyWingAreaMultToCxAoA"]["value"] is False
    assert manifest["global_parameters"]["useCxiMach"]["value"] is True
    assert manifest["global_parameters"]["CxAoA"]["source_file"] == (
        "aces.vromfs.bin_u/config/gameparams.blkx"
    )
    assert manifest["global_parameters"]["applyWingAreaMultToCxAoA"]["source_field"] == (
        "shellBallisticsParams.applyWingAreaMultToCxAoA"
    )

    aim7 = json.loads((ROOT / "data/profile_sets/2.59.0.27/missiles/us_aim7e2_dogfight_sparrow.json").read_text())
    assert aim7["aerodynamics"]["cy_k"] is not None
    assert aim7["aerodynamics"]["mach_lift_correction"]["model"] == "lookup_table"
    assert aim7["aerodynamics"]["parameter_sources"]["mach_lift_correction"]["kind"] == "datamine"
    aim7_runtime = next(profile for profile in profiles if profile["missile_id"] == "us_aim7e2_dogfight_sparrow")
    assert aim7_runtime["_model_config"]["aerodynamics"]["mach_lift"] == {
        "model": "lookup_table",
        "parameters": aim7["aerodynamics"]["mach_lift_correction"]["parameters"],
    }
    assert any("piecewise-linear interpolation" in note for note in aim7_runtime["_runtime_assumptions"])

    r77 = json.loads((ROOT / "data/profile_sets/2.59.0.27/missiles/su_r_77_1.json").read_text())
    assert r77["aerodynamics"]["cy_k"] is None
    assert r77["aerodynamics"]["mach_lift_correction"]["model"] == "not_declared"
    assert r77["aerodynamics"]["cx_vs_aoa"]["coefficient_per_rad2"] is None
    r77_runtime = next(profile for profile in profiles if profile["missile_id"] == "su_r_77_1")
    assert r77_runtime["_model_config"]["aerodynamics"]["mach_lift"] == {
        "base": 1.0,
        "transonic_peak": 0.1,
        "center": 1.0,
        "width": 0.6,
    }
    public = public_profile(aim7_runtime)
    assert public["parameter_source"] == {
        "version": "2.59.0.27",
        "commit": "4ac255b829a27346b9300fccdc138362c2535fbe",
    }
    gui_source = (ROOT / "src/missile_gui/static/app.js").read_text()
    assert "parameter_source: state.selected?.parameter_source" in gui_source
    assert "为避免同 ID 混用，已拒绝导入" in gui_source
    assert "此旧场景没有参数版本与 commit 记录" in gui_source


def test_open_source_release_allowlist_contains_catalog_and_both_profile_sets_only():
    from scripts.build_open_source_release import FILES

    expected = {
        "config/missile_profile_catalog.json",
        "src/aim120_model/profile_catalog.py",
        "data/profile_sets/2.57.1.107/data/aam_non_tvc_manifest.json",
        "data/profile_sets/2.59.0.27/data/aam_non_tvc_manifest.json",
        "data/profile_sets/2.59.0.27/missiles/su_r_77_1.json",
    }
    assert expected.issubset(FILES)
    assert not any(path.startswith("data/datamine") or "/.git/" in path for path in FILES)


def test_cy_mach_table_uses_linear_knots_interpolation_and_endpoint_clamping():
    from aim120_model.drag_models import mach_multiplier

    settings = {"model": "lookup_table", "parameters": [[0.8, 0.815], [1.0, 1.0], [1.4, 0.9]]}
    assert mach_multiplier(0.8, settings) == 0.815
    assert mach_multiplier(1.0, settings) == 1.0
    assert mach_multiplier(1.2, settings) == 0.95
    assert mach_multiplier(0.2, settings) == 0.815
    assert mach_multiplier(2.0, settings) == 0.9


def test_gaussian_mach_path_is_unchanged():
    import math

    from aim120_model.drag_models import mach_multiplier

    settings = {"base": 1.0, "transonic_peak": 0.1, "center": 1.0, "width": 0.6}
    assert mach_multiplier(1.0, settings) == 1.1
    assert mach_multiplier(1.6, settings) == 1.0 + 0.1 * math.exp(-1.0)


@pytest.mark.parametrize("version", ["2.57.1.107", "2.59.0.27"])
def test_r771_raw_loft_fields_reach_the_actual_library_and_api_config(version, monkeypatch):
    from aim120_model import public_api
    from missile_gui.library import _load_one

    profile = _load_one(ROOT / f"data/profile_sets/{version}/missiles/su_r_77_1.json", ROOT)
    raw_root = datamine_path_for_profile(ROOT, profile)
    raw = json.loads((raw_root / profile["provenance"]["source_files"][0]).read_text())
    ap = raw["rocket"]["guidance"]["guidanceAutopilot"]
    captured = []
    real_simulator = public_api.H2Simulator

    class CapturingSimulator(real_simulator):
        def __init__(self, config):
            captured.append(copy.deepcopy(config))
            super().__init__(config)

    monkeypatch.setattr(public_api, "H2Simulator", CapturingSimulator)
    result = public_api.simulate(profile, dict(
        launch_speed_kmh=1200.0, launch_altitude_m=6500.0,
        launch_pitch_deg=0.0, launch_heading_deg=0.0,
        target_speed_kmh=1200.0, target_altitude_m=6500.0,
        initial_distance_m=8000.0, target_azimuth_deg=0.0,
        target_heading_deg=180.0, target_vertical_heading_deg=0.0,
        target_constant_turn_g=0.0, max_simulation_time_s=0.1,
        loft_enabled=True,
    ), model_mode="legacy_h2")
    assert len(captured) == 1
    runtime = captured[0]["guidance"]
    for dest, source in {
        "lofting_elevation_deg": "loftElevation",
        "target_elevation_deg": "loftTargetElevation",
        "angle_to_acceleration_multiplier": "loftAngleToAccelMult",
        "loft_omega_max_deg_s": "loftTargetOmegaMax",
    }.items():
        assert runtime[dest] == ap[source]
    assert result["missile"]["parameter_source"]["version"] == version


def test_explicit_profile_loft_values_override_shared_defaults_without_raw_files(monkeypatch):
    from aim120_model import profile_adapter

    profile = json.loads((ROOT / "data/profile_sets/2.59.0.27/missiles/su_r_77_1.json").read_text())
    profile["guidance"].update(loft_target_elevation_deg=-6.25,
                               loft_angle_to_acceleration_multiplier=13.75,
                               loft_omega_max_deg_s=0.625)
    def unexpected_raw_read(*args):
        raise AssertionError("complete normalized loft values must not require raw checkout")
    monkeypatch.setattr(profile_adapter, "datamine_path_for_profile", unexpected_raw_read)
    defaults = json.loads((ROOT / "config/profile_h2_runtime_defaults.json").read_text())
    config, _ = profile_adapter.build_h2_candidate_config(profile, defaults)
    assert config["guidance"]["target_elevation_deg"] == -6.25
    assert config["guidance"]["angle_to_acceleration_multiplier"] == 13.75
    assert config["guidance"]["loft_omega_max_deg_s"] == 0.625


@pytest.mark.parametrize("raw_available", [True, False])
def test_absent_raw_loft_fields_retain_legacy_defaults(tmp_path, monkeypatch, raw_available):
    from aim120_model import profile_adapter

    profile = json.loads((ROOT / "data/profile_sets/2.59.0.27/missiles/su_r_77_1.json").read_text())
    for field in ("loft_target_elevation_deg", "loft_angle_to_acceleration_multiplier", "loft_omega_max_deg_s"):
        profile["guidance"].pop(field, None)
    if raw_available:
        path = tmp_path / profile["provenance"]["source_files"][0]
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps({"rocket": {"guidance": {"guidanceAutopilot": {}}}}))
    monkeypatch.setattr(profile_adapter, "datamine_path_for_profile", lambda *args: tmp_path)
    defaults = json.loads((ROOT / "config/profile_h2_runtime_defaults.json").read_text())
    defaults.update(loft_target_elevation_deg=-9.0,
                    loft_angle_to_acceleration_multiplier=11.0,
                    loft_omega_max_deg_s=0.75)
    config, _ = profile_adapter.build_h2_candidate_config(profile, defaults)
    assert config["guidance"]["target_elevation_deg"] == -9.0
    assert config["guidance"]["angle_to_acceleration_multiplier"] == 11.0
    assert config["guidance"]["loft_omega_max_deg_s"] == 0.75


def test_unknown_profile_source_does_not_read_active_loft_fields():
    from aim120_model.profile_adapter import build_h2_candidate_config

    profile = json.loads((ROOT / "data/profile_sets/2.59.0.27/missiles/su_r_77_1.json").read_text())
    profile["provenance"]["source_commit"] = "0" * 40
    defaults = json.loads((ROOT / "config/profile_h2_runtime_defaults.json").read_text())
    config, assumptions = build_h2_candidate_config(profile, defaults)
    assert config["guidance"]["angle_to_acceleration_multiplier"] == profile["guidance"]["loft_angle_to_acceleration_multiplier"]
    assert config["guidance"]["loft_omega_max_deg_s"] == profile["guidance"]["loft_omega_max_deg_s"]
    assert not any("raw datamine checkout" in note for note in assumptions)
