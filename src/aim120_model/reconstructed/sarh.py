"""Representative PD SARH observer with an independently supplied illuminator.

The .75 audit establishes the semiActive/provider branch and C090/C640 LOS
observer, but not native carrier association or the complete bistatic echo
formula. Here association, waveform and timestamp are explicit scene inputs.
The point echo uses a reconstructed two-leg inverse-square power law. Neither
an illuminator nor target velocity is manufactured for guidance.
"""
import math

from .types import SensorMeasurement


def _add(a, b): return tuple(x + y for x, y in zip(a, b))
def _sub(a, b): return tuple(x - y for x, y in zip(a, b))
def _mul(a, k): return tuple(x * k for x in a)
def _dot(a, b): return sum(x * y for x, y in zip(a, b))
def _cross(a, b): return (a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0])
def _norm(a): return math.sqrt(_dot(a, a))
def _unit(a):
    length = _norm(a)
    return _mul(a, 1 / length) if length else (0., 0., 0.)


def _forward(q):
    x, y, z, w = q
    return _unit((1-2*(y*y+z*z), 2*(x*y+w*z), 2*(x*z-w*y)))


def _angle(a, b):
    return math.acos(max(-1., min(1., _dot(a, b))))


def _merge(base, overrides):
    result = dict(base)
    for key, value in overrides.items():
        result[key] = _merge(result.get(key, {}), value) if isinstance(value, dict) else value
    return result


class SemiActiveSeeker:
    """PD point-target route; defaults must be verified loader fallbacks.

    ``defaults`` is a radarSeeker dictionary (or guidance/radarSeeker wrapper).
    Missing angular filters, present flags and required channel/receiver fields
    raise a configuration error. A scene timestamp must be within one update
    interval of t. This is the external-scene sampling contract, not a claim
    about native carrier update timing. CW, abs-Doppler and target-ID modes
    require different unreconstructed associations and are rejected.

    Optional response_model exposes point_gain(leg, beam, direction) and
    receiver_power(sigma, tx=(gain, range, gain, range), rx=(...)). It must be
    a pure response model. The own-formula route needs no native runtime.
    """
    def __init__(self, rocket, *, defaults=None, response_model=None):
        self.guidance = rocket['guidance']
        raw = self.guidance['radarSeeker']
        defaults = defaults or {}
        if 'guidance' in defaults: defaults = defaults['guidance']['radarSeeker']
        elif 'radarSeeker' in defaults: defaults = defaults['radarSeeker']
        self.raw = _merge(defaults, raw)
        p = self.raw
        # FF32B0 initializer: active=false; semiActive=!active. A null active
        # in a decoded resource is equivalent to the absent false initializer.
        active = bool(raw.get('active', False))
        if active or not raw.get('semiActive', not active):
            raise ValueError('SemiActiveSeeker requires the semiActive provider branch')
        if p.get('useTargetId', False) or p.get('absDopplerSpeed', False):
            raise NotImplementedError('Target-ID/absolute-Doppler SARH association is not reconstructed')
        if p.get('scanHalfSector', 0) != 0:
            raise NotImplementedError('SARH scanning sectors are not reconstructed')
        for name in ('mainBeamNotchWidth', 'zeroDopplerNotchWidth'):
            if p.get(name, -2.) >= 0:
                raise NotImplementedError('SARH clutter/notch intervals require a reconstructed spectrum')
        try:
            self.alpha, self.beta = float(p['filterAlpha']), float(p['filterBetta'])
            self.prolongation = float(p['prolongationTimeMax'])
            self.reset_time = float(p['trackResetTimeMax'])
            self.angle_limit = math.radians(p['angleMax'])
            self.lock_limit = math.radians(p['lockAngleMax'])
            self.rate = math.radians(p['rateMax'])
            self.angle_gate = math.radians(p['angleGateRate'])
            self.tx, self.rx = p['transmitter'], p['receiver']
            self.distance, self.doppler = p['distance'], p['dopplerSpeed']
            self.has_range = bool(self.distance['presents'])
            self.has_doppler = bool(self.doppler['presents'])
            for config in (self.tx, self.rx):
                config['antenna']['angleHalfSens']; config['antenna']['sideLobesSensitivity']
            self.tx['power']; self.rx['rcs']; self.rx['range']; self.rx['rangeMax']
            for channel in (self.distance, self.doppler):
                if channel['presents']:
                    for key in ('minValue', 'maxValue', 'width', 'signalWidthMin'):
                        channel[key]
        except KeyError as exc:
            raise ValueError(f'Missing verified SARH resource/default field: {exc.args[0]}') from exc
        from .responses import SharedResponses
        self.responses = response_model if response_model is not None else SharedResponses(p)
        self.n, self.w = (1., 0., 0.), (0., 0., 0.)
        self.tracking = False
        self.loss_age = 0.
        self.radial = {}
        self.source = 'none'
        self.last_t = None
        self.range_m = self.closing = None
        self.closure_valid = False

    @staticmethod
    def _gain(beam, direction, antenna):
        # Equal-axis point antenna, as reconstructed in python_seeker118.
        cosine = max(-1., min(1., _dot(beam, direction)))
        if cosine <= 0: return 0.
        half = math.radians(antenna['angleHalfSens'] * .5)
        k = math.log(2) / math.sin(half)**2
        sine2 = max(0., 1-cosine*cosine)
        side = 10**(antenna['sideLobesSensitivity']/10)
        main = math.exp(-k*sine2)
        knee = math.asin(math.sqrt(min(1., -.5*math.log(side)/k)))
        attenuation = math.exp(2*math.log(side)*max(0., math.acos(cosine)-knee)/(math.pi/2-knee))
        return max(main, side*attenuation)

    def _channel(self, name, value, dt):
        channel = self.distance if name == 'distance' else self.doppler
        if not channel['presents']: return 1., value
        period = channel.get('maxUnambiguousValue')
        if period:
            value = math.fmod(value, period)
        if not channel['minValue'] <= value <= channel['maxValue']:
            return 0., value
        if not self.tracking or name not in self.radial:
            return 1., value
        old, derivative = self.radial[name]
        predicted = old + derivative*dt
        section = self.raw.get('distGate' if name == 'distance' else 'dopplerSpeedGate', {})
        # Uniform finite linewidth overlap with the retained observation gate.
        width, signal_width = channel['width'], channel['signalWidthMin']
        overlap = max(0., min(value+signal_width/2, predicted+width/2)
                      - max(value-signal_width/2, predicted-width/2))
        gain = min(1., overlap/signal_width) if signal_width else float(abs(value-predicted) <= width/2)
        if gain == 0: return 0., value
        a, b = section.get('filterAlpha', self.alpha), section.get('filterBetta', self.beta)
        steps = min(48*dt, 10.)
        a, b = 1-(1-a)**steps, 1-(1-b)**(steps*steps)
        residual = value-predicted
        increment = b*residual/dt
        limits = section.get('accelLimit' if name == 'distance' else 'rateLimit')
        if limits and self.loss_age >= section.get('limitTimeOut', math.inf):
            limit = (limits[0]+(limits[1]-limits[0])*min(self.loss_age, 1.))*dt
            if abs(increment) >= limit: return 0., value
        return gain, predicted+a*residual

    def _missing(self, dt, reason, metadata):
        self.loss_age += dt
        held = self.tracking and self.loss_age < self.prolongation
        if self.tracking:
            self.n = _unit(_add(self.n, _mul(_cross(self.n, self.w), dt)))
            for name, (value, derivative) in self.radial.items():
                self.radial[name] = (value+derivative*dt, derivative)
        if not held:
            self.tracking = False
            self.radial.clear()
            self.w = (0., 0., 0.)
            self.range_m = self.closing = None
            self.closure_valid = False
        metadata.update(reason=reason, loss_age_s=self.loss_age,
                        bistatic_doppler_valid=False, illuminator_contract='explicit sampled PD provider')
        return SensorMeasurement(self.n, self.w, held, 'held' if held else 'search',
                                 self.source if held else 'none', self.range_m if held else None,
                                 self.closing if held else None, held and self.range_m is not None,
                                 held and self.closure_valid, False, held, metadata)

    def update(self, t, dt, missile, scene):
        if dt <= 0 or (self.last_t is not None and t <= self.last_t):
            raise ValueError('SARH updates require positive dt and increasing time')
        self.last_t = t
        metadata = {'branch': 'semiActive_pd', 'signal_model': 'shared_resource_scaled_two_leg_response',
                    'signal_model_status': 'derived_approximation',
                    'signal_model_limit': 'Full native bistatic echo normalization is not recovered; point-power law and receiver-reference normalization are derived assumptions',
                    'carrier_association': 'external_scene', 'ground_clutter': 'not_reconstructed'}
        illuminator = scene.illuminator
        if illuminator is None: return self._missing(dt, 'illuminator_absent', metadata)
        if illuminator.waveform.lower() != 'pd':
            raise NotImplementedError('Only explicit PD illumination is reconstructed; CW association is unsupported')
        if not illuminator.illuminating: return self._missing(dt, 'illumination_off', metadata)
        if illuminator.target_identity != scene.target.identity:
            return self._missing(dt, 'illumination_association_mismatch', metadata)
        age = t-illuminator.sample_time_s
        metadata['illumination_age_s'] = age
        if age < -1e-9 or age > dt+1e-9:
            return self._missing(dt, 'illumination_stale', metadata)
        if not scene.target.present: return self._missing(dt, 'target_absent', metadata)
        if scene.chaff:
            raise NotImplementedError('SARH multi-object/chaff association is not reconstructed')
        if scene.target_rcs_m2 is None:
            raise ValueError('PD SARH scene requires an explicit target_rcs_m2')
        rrx = _sub(scene.target.p, missile.p)
        rtx = _sub(scene.target.p, illuminator.position)
        rr, rt = _norm(rrx), _norm(rtx)
        if min(rr, rt) <= 0: return self._missing(dt, 'degenerate_bistatic_geometry', metadata)
        nrx, ntx = _unit(rrx), _unit(rtx)
        txbeam = _unit(illuminator.direction)
        if txbeam == (0., 0., 0.): return self._missing(dt, 'illumination_direction_absent', metadata)
        body = _forward(missile.q)
        beam = self.n if self.tracking else body
        limit = self.angle_limit if self.tracking else self.lock_limit
        if _angle(body, nrx) > limit:
            return self._missing(dt, 'receiver_fov', metadata)
        if _angle(txbeam, ntx) > math.pi/2:
            return self._missing(dt, 'illumination_beam', metadata)
        gt = self._gain(txbeam, ntx, self.tx['antenna'])
        gr = self._gain(beam, nrx, self.rx['antenna'])
        if self.responses is not None:
            gt = self.responses.point_gain('transmitter', txbeam, ntx)
            gr = self.responses.point_gain('receiver', beam, nrx)
            power = self.responses.receiver_power(scene.target_rcs_m2,
                        tx=(gt, rt, gt, rt), rx=(gr, rr, gr, rr))
        else:
            # Raw Tx power is an explicit multiplier; receiver.range/rcs sets
            # the reference sensitivity. This SARH extension is reconstructed,
            # not a claim that the full native bistatic normalization is known.
            power = (self.tx['power']*scene.target_rcs_m2/self.rx['rcs']
                     * self.rx['range']**4/(rt*rt*rr*rr) * gt*gr/.5625)
        baseline = _norm(_sub(illuminator.position, missile.p))
        excess_range = .5*(rt+rr-baseline)
        bistatic_closing = .5*(_dot(_sub(illuminator.velocity, scene.target.v), ntx)
                              + _dot(_sub(missile.v, scene.target.v), nrx))
        rg, observed_r = self._channel('distance', excess_range, dt)
        dg, observed_d = self._channel('dopplerSpeed', bistatic_closing, dt)
        metadata.update(tx_range_m=rt, receiver_path_m=rr, bistatic_excess_range_m=excess_range,
                        bistatic_closing_mps=bistatic_closing, bistatic_doppler_valid=self.has_doppler,
                        antenna_gains=(gt, gr), raw_signal_over_floor=power, channel_gain=(rg, dg),
                        score=power*rg*dg)
        if rr > self.rx['rangeMax'] or power*rg*dg <= 1.:
            return self._missing(dt, 'signal_or_radial_gate', metadata)
        predicted = _unit(_add(self.n, _mul(_cross(self.n, self.w), dt)))
        reset = not self.tracking or self.loss_age > self.reset_time
        if reset:
            new_n, new_w = nrx, (0., 0., 0.)
        else:
            error = _cross(_sub(nrx, predicted), predicted)
            if any(abs(x) > self.angle_gate*dt for x in error):
                return self._missing(dt, 'receiver_angle_rate_gate', metadata)
            error = tuple(max(-self.rate*dt, min(self.rate*dt, x)) for x in error)
            new_n = _unit(_add(predicted, _mul(_cross(predicted, error), self.alpha)))
            new_w = _add(self.w, _mul(error, self.beta/dt))
        # Commit only accepted observations. Neither held LOS nor its rate ever
        # accesses world target P/V. Native world rate sign is d_dot=cross(d,w).
        for name, value, enabled in (('distance', observed_r, self.has_range),
                                     ('dopplerSpeed', observed_d, self.has_doppler)):
            if enabled:
                old, derivative = self.radial.get(name, (value, 0.))
                if self.tracking:
                    section = self.raw.get('distGate' if name == 'distance' else 'dopplerSpeedGate', {})
                    beta = section.get('filterBetta', self.beta)
                    derivative += beta*(value-old-derivative*dt)/dt
                self.radial[name] = (value, derivative)
        self.n, self.w, self.source = new_n, new_w, scene.target.identity
        self.range_m = self.closing = None
        self.closure_valid = False
        if self.has_range:
            # Invert measured excess path using known Tx baseline and measured
            # LOS. The target's true receiver range/velocity is not substituted.
            offset = _sub(illuminator.position, missile.p)
            path = 2*observed_r+baseline
            denominator = 2*(path-_dot(offset, self.n))
            if denominator > 1e-9:
                self.range_m = (path*path-baseline*baseline)/denominator
                tx_direction = _unit(_sub(_mul(self.n, self.range_m), offset))
                projection = 1+_dot(tx_direction, self.n)
                if self.has_doppler and projection > 1e-6:
                    dn = _cross(self.n, self.w)
                    self.closing = (2*observed_d-_dot(_sub(illuminator.velocity, missile.v), tx_direction)
                                    + self.range_m*_dot(dn, tx_direction))/projection
                    self.closure_valid = True
        metadata.update(reason='accepted', loss_age_s=0.,
                        closure_observability='measured_range_and_LOS_rate' if self.closure_valid
                        else 'receiver_radial_closure_unresolved',
                        illuminator_contract='explicit sampled PD provider')
        self.tracking, self.loss_age = True, 0.
        return SensorMeasurement(self.n, self.w, True, 'track', self.source,
                                 self.range_m, self.closing, self.range_m is not None,
                                 self.closure_valid, True, False, metadata)
