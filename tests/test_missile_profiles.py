import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from missile_lab.validator import validate_project  # noqa: E402


def test_profile_gate_passes_for_batch_import_and_frozen_regression():
    report = validate_project(ROOT)
    assert report["passed"] is True
    assert report["profile_count"] == 120
    assert report["schema_passed"] == 120
    assert report["smoke_passed"] == 120
    assert report["unsupported_profile_count"] == 4
    assert report["frozen_regression_passed"] is True


def test_every_runnable_profile_has_an_observation_toggle_provider():
    from missile_gui.library import scan_library

    profiles, errors = scan_library(ROOT / "missiles", ROOT)
    assert errors == []
    runnable = [profile for profile in profiles if profile.get("_model_config")]
    assert runnable
    for profile in runnable:
        guidance = profile["_model_config"]["guidance"]
        sensor_model = guidance.get("sensor_model")
        assert isinstance(sensor_model, dict)
        if profile["missile_id"] == "us_aim_120a":
            assert "radar_seeker" in sensor_model
        else:
            assert sensor_model["provider"] == "profile_kinematic_v1"


def test_manifest_records_tvc_exclusions_without_profiles():
    manifest = json.loads((ROOT / "data" / "aam_non_tvc_manifest.json").read_text(encoding="utf-8"))
    assert manifest["included_profile_count"] == 120
    assert manifest["excluded_profile_count"] == 4
    assert {item["missile_id"] for item in manifest["excluded"]} == {
        "fr_mica_em",
        "su_r_73",
        "su_r_73e",
        "uk_sraam",
    }
    assert all(item["reason"] == "tvc_excluded" for item in manifest["excluded"])
    assert all("thrust_vector" in item["unsupported_features"] for item in manifest["excluded"])
    assert manifest["bullet_name_fallback_file_count"] == 5
    assert manifest["unsupported_read_only_profile_count"] == 4


def test_unsupported_guidance_is_explicit_and_not_silently_mapped_to_pn():
    for missile_id, family_guidance in {
        "de_x4_ruhrstahl": "command_guidance",
        "fr_aa20": "command_guidance",
        "uk_fireflash": "command_guidance",
        "us_starstreak": "beam_riding",
    }.items():
        profile = json.loads((ROOT / "missiles" / f"{missile_id}.json").read_text(encoding="utf-8"))
        assert profile["model_family"]["guidance"] == family_guidance
        assert profile["runtime"]["implemented"] is False
        assert "不支持" in profile["runtime"]["notes"]


def test_missing_bullet_names_get_safe_unique_ids_and_raw_provenance():
    for missile_id, raw_name in {
        "atam_mistral": "atam_mistral",
        "atam_mistral_a129": "atam_mistral_a129",
        "sws_flz_lwf_ll_64": "sws_flz_lwf_ll_64",
        "us_aim_26b": "us_aim_26b",
        "us_starstreak": "us_starstreak",
    }.items():
        profile = json.loads((ROOT / "missiles" / f"{missile_id}.json").read_text(encoding="utf-8"))
        assert profile["missile_id"] == missile_id
        assert raw_name in profile["provenance"]["notes"]


def test_gui_runtime_opens_supported_profiles_and_keeps_unsupported_guidance_locked():
    sys.path.insert(0, str(ROOT / "src"))
    from missile_gui.library import public_profile, scan_library

    profiles, errors = scan_library(ROOT / "missiles", ROOT)
    assert errors == []
    aim = next(public_profile(profile) for profile in profiles if profile["missile_id"] == "us_aim_120a")
    pl15 = next(public_profile(profile) for profile in profiles if profile["missile_id"] == "cn_pl15")
    fireflash = next(public_profile(profile) for profile in profiles if profile["missile_id"] == "uk_fireflash")
    assert aim["runnable"] is True
    assert pl15["runnable"] is True
    assert pl15["status"] == "Experimental"
    assert aim["runtime_adapter"] == "profile_h2_fin_torque_aoa_v4"
    assert pl15["runtime_adapter"] == "profile_h2_fin_torque_aoa_v4"
    assert pl15["runtime_assumption_count"] > 0
    assert fireflash["runnable"] is False
    assert fireflash["status"] == "Unsupported physics"
    assert fireflash["physics"]["guidance_type"] == "command_guidance"
