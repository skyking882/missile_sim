"""WT 1b: independent initial cone and supported pointing after own radar loss."""
import copy
import math
import pytest
from test_track_gate_model import observer, missile, target, bundle
from aim120_model.observation import SensorTrackProvider, TrackMode


def provider(**options):
    sensor=dict(observer().sensor_model,datalink=True,inertial_navigation=True,reconnect_datalink=False)
    p=SensorTrackProvider(sensor,lock_range_m=20000.)
    p.radar.enable_track_gate_model({'notch_track_hazard':[],**options})
    return p


def test_initial_acquisition_window_uses_available_estimate_and_can_disable_independently():
    t=target(z=1000.*math.tan(math.radians(6.)))
    o=observer(); assert not o.update(0.,.02,missile(),t)[0].valid
    assert o.last_detection.reason=='acquire_window'
    o.track_gate_estimate=t.position
    assert o.update(.02,.02,missile(),t)[0].valid
    legacy=observer(acquire_half_deg=None,datalink_after_lock=False)
    assert legacy.update(0.,.02,missile(),t)[0].valid


def test_locked_radar_has_priority_then_live_wire_recenters_reacquisition():
    p=provider();m=missile();t=target();aim=target(z=1000.*math.tan(math.radians(6.)))
    p.enable_launcher_support(lambda stamp,truth:('', aim.position,aim.velocity))
    # First initialization cannot acquire target outside the recorded aim cone.
    assert p.update(0.,.02,m,t).mode is TrackMode.DATALINK
    p.launcher_support=lambda stamp,truth:('',t.position,t.velocity)
    assert p.update(.02,.02,m,t).mode is TrackMode.RADAR_TRACK
    p.launcher_support=lambda stamp,truth:('',aim.position,aim.velocity)
    assert p.update(.04,.02,m,t).position==t.position  # wire never replaces radar measurement
    assert p.datalink_connected and p.inertial._position_estimate==t.position
    assert p.update(.06,.02,m,aim).mode is TrackMode.RADAR_COAST
    assert p.radar.track_gate_estimate==aim.position
    assert p.update(1.06,1.,m,aim).mode is TrackMode.RADAR_TRACK
    assert p.datalink_connected


def test_actual_wire_loss_is_permanent_and_center_is_ins_memory():
    p=provider();m=missile();t=target()
    p.enable_launcher_support(lambda stamp,truth:'' if stamp<.05 else 'recorded_loss')
    p.update(0.,.02,m,t);p.update(.02,.02,m,t)
    turned=target(z=1000.*math.tan(math.radians(6.)))
    assert p.update(.06,.04,m,turned).mode is TrackMode.RADAR_COAST
    assert not p.datalink_connected
    assert p.radar.track_gate_estimate==t.position
    p.launcher_support=lambda stamp,truth:('',turned.position,turned.velocity)
    assert p.update(1.04,.98,m,turned).mode is TrackMode.INS_SEARCH
    assert p.radar.last_detection.reason=='reacquire_window'
    assert not p.datalink_connected and p.datalink_lost_reason=='recorded_loss'


def test_explicit_legacy_override_preserves_permanent_self_disconnect():
    p=provider(datalink_after_lock=False,acquire_half_deg=None)
    assert p.update(0.,.02,missile(),target()).mode is TrackMode.RADAR_TRACK
    assert not p.datalink_connected
    assert p.reconnect_datalink is False


def test_disconnect_time_and_user_disabled_remain_effective():
    p=provider();p.datalink_disconnect_time_s=.05
    p.update(0.,.02,missile(),target());p.update(.06,.06,missile(),target())
    assert not p.datalink_connected
    p=provider();p.datalink_enabled=False;p.update(0.,.02,missile(),target())
    assert not p.datalink_connected


def test_same_timestamp_and_snapshot_preserve_wire_and_radar_measurements():
    p=provider();t=target();p.enable_launcher_support(lambda stamp,truth:('',t.position,t.velocity))
    p.update(0.,.02,missile(),t);clone=copy.deepcopy(p)
    for stamp in (.02,.04,.04):
        assert p.update(stamp,.02,missile(),t)==clone.update(stamp,.02,missile(),t)
    assert p.radar.track_gate_summary()==clone.radar.track_gate_summary()
    assert p.datalink_connected and clone.datalink_connected


@pytest.mark.parametrize('options',[{'acquire_half_deg':-1},{'acquire_half_deg':True},{'acquire_half_deg':181},{'datalink_after_lock':1}])
def test_invalid_1b_options_fail(options):
    with pytest.raises(ValueError):observer(**options)
