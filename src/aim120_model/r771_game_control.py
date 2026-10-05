"""Opt-in control-memory experiment for the War Thunder R-77-1 game unit.

This module has no replay, file, network, target-history or plant dependency.
The caller supplies current game-plant pressure normalization and an existing
nonlinear-force-regime flag. Raw PID values and old actuator bounds are inputs.
The extra target-angle authority bound is an experimental interface hypothesis,
not an identified native-game cap or a physical fin limit.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Mapping


def _clip(value: float, limit: float) -> float:
    return max(-limit,min(limit,value))


@dataclass
class R771GameControl:
    """Transport only the integral coordinate after a causal regime trigger.

    Memory remains (I_old, previous_G_error, previous_G_error_derivative).
    For K=Ku(V)/(A Ka), entering the regime initializes I_target=K I_old.
    Thereafter I_target is held across a pressure change by transporting
    I_old <- (K_previous/K_current) I_old before the ordinary raw PID update.
    P/D target contributions are K P e and K D e_dot; raw P/D are not retuned.
    Saturation freezes the transported integral, preserving its coordinates.
    """
    controller_hz: int=48
    nonlinear_latched: bool=False
    previous_high_k: list[float|None]=field(default_factory=lambda:[None,None])
    last_time: list[float|None]=field(default_factory=lambda:[None,None])
    cached: list[tuple|None]=field(default_factory=lambda:[None,None])
    diagnostics: list[dict|None]=field(default_factory=lambda:[None,None])

    def update_axis(self,axis:int,time_s:float,error_g:float,memory:tuple[float,float,float],
            pid:Mapping[str,float],old_held_limit_rad:float,gain:float,
            target_per_old_held:float,nonlinear_force_active:bool):
        if axis not in (0,1):raise ValueError("axis must be pitch0 or yaw1")
        k=float(target_per_old_held)
        if not all(math.isfinite(v) for v in (time_s,error_g,k,old_held_limit_rad,gain)) or k<=0 or old_held_limit_rad<=0 or not 0<=gain<=1:
            raise ValueError("finite positive normalization/limit and flight gain in[0,1] required")
        if self.last_time[axis] is not None:
            if time_s==self.last_time[axis]:return self.cached[axis]
            if time_s<self.last_time[axis]:raise ValueError("control clock cannot move backwards")
        if abs(time_s*self.controller_hz-round(time_s*self.controller_hz))>1e-8:
            raise ValueError("control memory may update only at controller ticks")
        if gain<=1e-12:
            # The unchanged calling integrator normally handles this branch.
            if memory!=(0.,0.,0.):raise ValueError("inactive PID memory must be zero")
            return 0.,memory
        self.nonlinear_latched=self.nonlinear_latched or bool(nonlinear_force_active)
        integral,previous_error,previous_derivative=memory
        integral_limit=float(pid["integral_limit"])
        ratio=1.
        if self.nonlinear_latched:
            previous_k=self.previous_high_k[axis]
            # First entry uses the present K, so unit transport adds no jump.
            ratio=1. if previous_k is None else previous_k/k
            integral=_clip(integral*ratio,integral_limit)
            effective_limit=min(old_held_limit_rad,old_held_limit_rad/k)
        else:effective_limit=old_held_limit_rad
        dt=1./self.controller_hz
        provisional=_clip(integral+float(pid["i"])*error_g*dt,integral_limit)
        raw_d=(error_g-previous_error)/dt
        derivative=previous_derivative+dt/(1e-9+dt)*(raw_d-previous_derivative)
        output=float(pid["p"])*error_g+provisional+float(pid["d"])*derivative
        saturated=_clip(output,effective_limit)
        blocked=abs(saturated)>=.99*effective_limit and error_g*saturated>0
        next_i=integral if blocked else provisional
        result=(gain*saturated,(next_i,error_g,derivative))
        self.last_time[axis]=time_s;self.cached[axis]=result
        if self.nonlinear_latched:self.previous_high_k[axis]=k
        self.diagnostics[axis]={"time_s":time_s,"axis":axis,"error_g":error_g,"K_target_per_old_held":k,
            "nonlinear_latched":self.nonlinear_latched,"integral_transport_ratio":ratio,
            "transported_integral_old_rad":integral,"integral_target_rad":k*next_i,
            "P_target_rad":k*float(pid["p"])*error_g,"D_target_rad":k*float(pid["d"])*derivative,
            "target_candidate_limit_rad":old_held_limit_rad,"effective_old_held_limit_rad":effective_limit,
            "original_old_held_limit_rad":old_held_limit_rad,"antiwindup_blocked":blocked,
            "held_old_rad":result[0],"effective_target_rad":k*result[0],
            "normalized_target_at_tick":k*result[0]/old_held_limit_rad,"gain":gain}
        return result
