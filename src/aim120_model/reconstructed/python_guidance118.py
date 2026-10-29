"""Persistent INS, PN, loft and acceleration PID in Python.

Order follows 1452BB8D0 -> 1452A34C0 -> 1452A6570 -> 144001470.
External DL records are independent aircraft inputs; missile answers are unused.
"""
from dataclasses import dataclass, field
from .python_math118 import *
from .resources import loaded_curve


def pn_request(velocity_argument, omega, direction_body, gain, nav, angle_multiplier, elevation_add):
    pitch = math.atan2(direction_body[1], math.hypot(direction_body[0], direction_body[2]))
    yaw = math.atan2(-direction_body[2], direction_body[0])
    warg = [gain*nav*omega[0], gain*(nav*omega[1]-angle_multiplier*yaw),
            gain*(nav*omega[2]-angle_multiplier*(pitch+elevation_add))]
    return cross(velocity_argument, warg)


@dataclass
class INSState:
    valid: bool = False
    p: list = field(default_factory=lambda: [0., 0., 0.])
    v: list = field(default_factory=lambda: [0., 0., 0.])
    error: list = field(default_factory=lambda: [0., 0., 0.])
    channel: int = 0


class INS:
    def __init__(self, params, external):
        self.params, self.external = params, external
        self.state = INSState()  # Original cache constructor, not missing inputs.
        section = 'guidance/inertialGuidance'
        self.drift = params.get(section, 'inertialNavigationDriftSpeed')
        self.gnss = params.get(section, 'gnss')
        self.datalink = params.get(section, 'datalink')
        self.reconnect = params.get(section, 'reconnectDatalink')
        self.max_channels = params.get(section, 'datalinkChannelsMax')
        self.keep_last = params.get(section, 'keepDatalinkChannelsForLast')
        self.mask = params.get('guidance/radarSeeker', 'designationSourceTypeMask')
        self.lag_max = params.gameplay['seeker']['designationLagCompensationMax']
        self.noise = Noise(0x42c5ffb1)  # Recovered startup seed 1443956AB.
        self.last = None

    def selection(self, t, prelaunch=False):
        records, origin, origin_v, _ = self.external.provider(t)
        first, last = self.external.counters(t)
        used = last if self.keep_last else first
        if not prelaunch and (not self.datalink or used >= self.max_channels): return None
        for row in records:
            if not self.mask & (1 << (row['source_id'] & 31)): continue
            if not row['measurement_valid']: continue
            if prelaunch:
                if not row['flag49']: continue
            elif self.reconnect:
                if not row['flag48']: continue
            elif row['channel'] != self.state.channel:
                continue
            if not row['reference_valid']:
                # Native chooses header origin here; no record origin invented.
                reference = origin
            else:
                reference = row['reference_position_native_m']
            velocity = fv(add(row['relative_velocity_native_mps'], origin_v))
            lag = clip(f(t-row['measurement_time_s']), -self.lag_max, self.lag_max)
            position = fv(add(row['target_position_native_m'], mul(velocity, lag)))
            return dict(row=row, p=position, v=velocity, origin=reference, lag=lag,
                        packet=self.external.snapshot(t)['packet_index'])
        return None

    def initialize(self, t):
        selected = self.selection(t, prelaunch=True)
        if selected is None: raise ValueError('No independent prelaunch designation for INS')
        s = self.state
        s.valid, s.p, s.v = True, selected['p'], selected['v']
        s.channel = selected['row']['channel']
        return selected

    def auxiliary(self, physical):
        s = self.state
        if not s.valid: return dict(valid=False, n=[1., 0., 0.], w=[0., 0., 0.], R=0., D=0.)
        r = sub(add(s.p, s.error), physical.p)
        R = norm(r)
        n = unit(r)
        w = mul(cross(sub(s.v, physical.v), r), 1/dot(r, r)) if R else [0., 0., 0.]
        # Active seeker: the auxiliary transmit/receive endpoints coincide.
        return dict(valid=True, n=n, w=w, R=R, D=dot(sub(physical.v, s.v), n))

    def update(self, physical, dt, seeker):
        s = self.state
        if seeker.held:
            self.last = dict(source='held_seeker_cache_unchanged')
            return seeker.n, seeker.w
        old_direction = unit(sub(add(s.p, s.error), physical.p)) if s.valid else [1., 0., 0.]
        if seeker.fresh:
            rd = mul(seeker.n, seeker.R)
            s.p = fv(add(physical.p, rd))
            s.v = fv(add(sub(physical.v, cross(seeker.w, rd)),
                         mul(seeker.n, seeker.radial-dot(physical.v, seeker.n))))
            s.error = [0., 0., 0.]
            s.valid = True
            self.last = dict(source='seeker_measurement')
            return seeker.n, seeker.w
        selected = self.selection(physical.time)
        if selected is not None:
            s.p, s.v = selected['p'], selected['v']
            s.valid = True
            s.channel = selected['row']['channel']
        if not s.valid:
            self.last = dict(source='invalid')
            return [1., 0., 0.], [0., 0., 0.]
        if not self.gnss:
            scale = f(f(f(20.293798446655273)*dt)*self.drift)
            s.error = [f(s.error[i]+f(scale*self.noise.at(
                f(f(physical.time+f(60*old_direction[i]))*f(.30000001192092896))))) for i in range(3)]
        s.p = fv(add(s.p, mul(s.v, dt)))
        r = sub(add(s.p, s.error), physical.p)
        w = mul(cross(sub(s.v, physical.v), r), 1/dot(r, r)) if dot(r, r) > 3.99999987306209e-19 else [0., 0., 0.]
        self.last = dict(source='DL' if selected else 'cache_prediction',
                         packet=selected['packet'] if selected else None,
                         selected_channel=s.channel, lag=selected['lag'] if selected else None)
        return old_direction, fv(w)


@dataclass
class PIDAxis:
    integral: float = 0.
    previous_error: float = 0.
    derivative: float = 0.

    def step(self, error, kp, ki, kd, limit, dt, horizontal):
        e = f(error)
        self.integral = f(clip(f(self.integral+f(f(e*dt)*ki)), -limit, limit))
        self.derivative = f(self.derivative+f(48*f(e-f(self.previous_error+f(dt*self.derivative)))))
        self.previous_error = e
        P, D = f(kp*e), f(kd*self.derivative)
        raw = f(f(P+D)+self.integral) if horizontal else f(f(self.integral+P)+D)
        return f(clip(raw, -1., 1.)), dict(P=P, I=self.integral, D=D, error=e)


class Controller:
    def __init__(self, params):
        self.params = params
        self.raw = params.rocket['guidance']['guidanceAutopilot']
        self.section = 'guidance/guidanceAutopilot'
        self.time_gain = loaded_curve(self.raw, 'timeToGain', timeout=self.raw.get('timeOut',0.))
        self.tgo_gain = loaded_curve(self.raw, 'timeToHitToGain')
        self.axes = [PIDAxis(), PIDAxis()]
        self.control = [0., 0.]  # Native wrapper/control constructor writes.
        self.loft_mode = 1 if self.raw['loftEnabled'] else 0
        self.last = None
        if params.get(self.section, 'altitudeHoldEnabled'):
            raise ValueError('This dry-air port does not implement terrain altitude hold')
        if params.get(self.section, 'limitAoa'):
            raise ValueError('Optional limitAoa motor-capability branch is outside this R-77-1 configuration')
        if not params.get(self.section, 'velFrameReference'):
            raise ValueError('This configuration requires the recovered velocity reference frame')
        if not params.gameplay['accelControl']['useVelocityFrameRefMatrix']:
            raise ValueError('Old quaternion controller frame is outside this configuration')

    def step(self, state, elapsed, dt, direction, omega, velocity_argument, feedback, tgo, range2, allowed):
        p = self.raw
        if not allowed:
            self.last = dict(PID_called=False, reason='wrapper_mode_guard')
            # 1452BC823 clears only wrapper+58/+5C. PID memory is untouched.
            self.control = [0., 0.]
            return list(self.control)
        gain = f(curve(self.time_gain, elapsed)*curve(self.tgo_gain, tgo))
        if gain < .001:
            self.last = dict(PID_called=False, reason='native_gain_guard', gain=gain)
            return [0., 0.]
        r = matrix(state.q)
        local_d = tv(r, direction)
        A = self.params.get(self.section, 'angleToAccelMult')
        N = self.params.get(self.section, 'omegaToAccelMult')
        if self.params.get(self.section, 'purePursuit'): A, N = N, 0.
        elevation_add = self.params.get(self.section, 'elevationAdd')
        request = pn_request(velocity_argument, omega, local_d, gain, N, A, elevation_add)
        if self.loft_mode:
            gate = math.tan(math.radians(p['loftTargetElevation']))
            gate *= abs(gate)
            if direction[1]*abs(direction[1]) < (direction[0]**2+direction[2]**2)*gate or norm(omega) > math.radians(p['loftTargetOmegaMax']):
                self.loft_mode = 0
            else:
                forward = [row[0] for row in r]
                body_pitch = math.atan2(forward[1], math.hypot(forward[0], forward[2]))
                request[1] = max(request[1], p['loftAngleToAccelMult']*(math.radians(p['loftElevation'])-body_pitch))
        forward = unit(state.v)
        z = unit(cross(forward, unit([row[1] for row in r])))
        if norm(z) == 0: raise ValueError('Unsupported degenerate velocity/controller reference')
        y = cross(z, forward)
        basis = [forward, y, z]
        req_ref = [dot(column, request) for column in basis]
        fb_ref = [dot(column, feedback) for column in basis]
        cap = f(p['reqAccelMax']*f(9.8100004196167))
        lateral = math.hypot(req_ref[1], req_ref[2])
        scale = min(1., cap/lateral) if lateral else 1.
        errors = [req_ref[2]*scale-fb_ref[2], req_ref[1]*scale-fb_ref[1]]
        rho, _ = atmosphere(state.p[1])
        speed = norm(state.v)
        base = f(p['baseIndSpeed']*f(.2777777910232544))
        B = base*base*1.225000023841858**2/(rho*rho*speed*speed)
        kp = f(p['accelControlProp']*math.sqrt(B))
        ki = f(p['accelControlIntg']*math.sqrt(B))
        kd = f(p['accelControlDiff']*B)
        values = [self.axes[i].step(errors[i], kp, ki, kd, p['accelControlIntgLim'], dt, i == 0) for i in range(2)]
        self.last = dict(PID_called=True, gain=gain, request=request, feedback=feedback,
                         errors=errors, reference_basis=basis, gains=[kp, ki, kd],
                         terms=[row[1] for row in values], loft_mode=self.loft_mode)
        self.control = [row[0] for row in values]
        return list(self.control)
