"""Functional gates for the opt-in causal game-controller experiment."""
import math
from types import SimpleNamespace

import pytest

from aim120_model.r771_game_control import R771GameControl
from aim120_model.single_surface_game import SingleSurfaceGame

PID={"p":.002125,"i":.006481,"d":.000023,"integral_limit":1.}


def test_untriggered_controller_preserves_legacy_output_and_memory_sequence():
    control=R771GameControl();old=(0.,0.,0.);new=old
    for tick,error in enumerate((10.,20.,5.,-12.,-4.,0.,60.,-60.)):
        t=tick/48.;k=20.-tick
        expected,old=SingleSurfaceGame._pid(SimpleNamespace(controller_hz=48),error,old,PID,math.radians(40),1.)
        actual,new=control.update_axis(0,t,error,new,PID,math.radians(40),1.,k,False)
        assert (actual,new)==(expected,old)
    assert not control.nonlinear_latched


def test_constant_integral_target_survives_pressure_change_on_both_axes():
    control=R771GameControl();pid={"p":0.,"i":0.,"d":0.,"integral_limit":1.}
    memories=[(.01,0.,0.),(.02,0.,0.)]
    first=[];second=[]
    for axis in (0,1):
        u,memories[axis]=control.update_axis(axis,0.,0.,memories[axis],pid,.7,1.,10.,True)
        first.append(10*u)
    for axis in (0,1):
        u,memories[axis]=control.update_axis(axis,1/48.,0.,memories[axis],pid,.7,1.,5.,False)
        second.append(5*u)
    assert first==pytest.approx((.1,.2))
    assert second==pytest.approx(first)
    assert memories[0][0]==pytest.approx(.02)
    assert memories[1][0]==pytest.approx(.04)


def test_authority_saturation_blocks_windup_then_allows_error_reversal():
    control=R771GameControl();memory=(0.,0.,0.);limit=math.radians(40)
    for tick in range(12):
        u,memory=control.update_axis(0,tick/48.,100.,memory,PID,limit,1.,20.,True)
        assert abs(u)<=limit and 20*u<=limit+1e-12
        assert memory[0]==0.
    _,memory=control.update_axis(0,12/48.,-1.,memory,PID,limit,1.,20.,False)
    # The unchanged raw derivative term saturates during the error reversal.
    # Holding the reversed error permits unwinding once that kick has passed.
    _,memory=control.update_axis(0,13/48.,-1.,memory,PID,limit,1.,20.,False)
    assert memory[0]<0.


def test_repeated_tick_does_not_advance_memory_and_non_tick_is_rejected():
    control=R771GameControl();memory=(0.,0.,0.)
    first=control.update_axis(0,0.,2.,memory,PID,.7,1.,20.,True)
    assert control.update_axis(0,0.,2.,first[1],PID,.7,1.,20.,True)==first
    with pytest.raises(ValueError,match="only at controller ticks"):
        control.update_axis(0,.001,2.,first[1],PID,.7,1.,20.,True)


def test_initial_entry_is_bumpless_before_any_candidate_bound_is_reached():
    control=R771GameControl();pid={"p":0.,"i":0.,"d":0.,"integral_limit":1.}
    u,memory=control.update_axis(0,0.,0.,(.005,0.,0.),pid,.7,1.,100.,True)
    assert u==.005 and memory==(.005,0.,0.)
