from aim120_model.effective_controller import EffectiveControllerEnvelope


MODEL = EffectiveControllerEnvelope(
    gain=0.714070844926948,
    authority_fraction=0.695897709074012,
)


def test_backend_current_g_prediction_is_a_nonnegative_magnitude():
    positive = MODEL.predict_current_g_magnitude(10.0, 42.2579)
    negative = MODEL.predict_current_g_magnitude(-10.0, 42.2579)
    assert positive > 0.0
    assert positive == negative


def test_signed_effective_output_restores_command_sign_only_at_plant_boundary():
    magnitude = MODEL.predict_current_g_magnitude(10.0, 42.2579)
    assert MODEL.predict_signed_effective_output(10.0, 42.2579) == magnitude
    assert MODEL.predict_signed_effective_output(-10.0, 42.2579) == -magnitude
    assert MODEL.predict_signed_effective_output(0.0, 42.2579) == 0.0


def test_authority_envelope_saturates_at_rho_times_fins_lat_accel():
    authority = 16.5
    expected_cap = MODEL.authority_fraction * authority
    assert abs(MODEL.effective_cap_g(authority) - expected_cap) < 1.0e-12
    assert MODEL.predict_current_g_magnitude(100.0, authority) == expected_cap
    switch = MODEL.envelope_switch_command_g(authority)
    assert abs(MODEL.predict_current_g_magnitude(switch, authority) - expected_cap) < 1.0e-12


def test_invalid_authority_is_rejected():
    try:
        MODEL.predict_current_g_magnitude(1.0, -1.0)
    except ValueError:
        return
    raise AssertionError("negative finsLatAccel must be rejected")
