"""Causal IR point-source reconstruction; no target kinematics enter guidance.

``SceneSnapshot.target_ir`` and each flare's ``emission`` map band numbers
0..9 to *directional source signature / C0*, C0=24034.501953125. These are
emitted strengths before propagation, not irradiance at the missile. Thus a
normalized flare emission2=1 at its plateau gives (rangeBand2/range)**2.
Aircraft thermal/aspect expansion and flare age/envelope are scene producers.
``emission_profile`` below supplies the recovered finite-age additive envelope.

Recovered .86 blocks: 143FEC8B0/14332B300 (loading), 143FF2A40 (max band),
143FEE994/143FF08A1/143FF214A (gate/outside and rejection), 143FF2338
(centroid), 14332CA60/14332D010 (angular observer/prediction).
This implementation selects their unresolved point-source branch. Propagation
uses inverse square with a hard FOV boundary; native finite-size/edge angular
response, world occlusion, sunlight/background and resolved-target branches
are not reconstructed here. This is not a historical replay reproduction.
"""
import math

from .types import SensorMeasurement


# Getter fallback literals from .86 143FEC8B0 and 14332B300. These are
# initializer defaults, not fitted missile-performance parameters.
NATIVE_DEFAULTS = {
    'range': 4000., 'rangeMax': 10000., 'angleHalfSens': 5.,
    'gateWidth': 90., 'outGateTransparency': 0., 'bandMaskToReject': 0,
    'signalRelRejectedTreshold': .001, 'rejectionReactionTime': 0.,
    'targetsResolution': False, 'scanHalfSector': 0., 'scanPeriod': 1.,
    'boundaryTrack': False, 'mandatoryDesignationSources': False,
    'lockAngleMax': 5., 'angleMax': 20., 'rateMax': 20.,
    'filterAlpha': .8500000238418579, 'filterBetta': .20000000298023224,
    'angleGateRate': 2147440000., 'prolongationTimeMax': 1.,
    'trackResetTimeMax': 1000.,
}


def _add(a, b): return tuple(x+y for x, y in zip(a, b))
def _sub(a, b): return tuple(x-y for x, y in zip(a, b))
def _scale(a, s): return tuple(x*s for x in a)
def _dot(a, b): return sum(x*y for x, y in zip(a, b))
def _cross(a, b):
    return (a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0])
def _unit(a):
    length = math.sqrt(_dot(a, a))
    return _scale(a, 1/length) if length else (1., 0., 0.)
def _angle(a, b): return math.acos(max(-1., min(1., _dot(a, b))))
def _clip(x, lo, hi): return min(max(x, lo), hi)


def _rotate(q, v):
    # Body to world, XYZW. BodyState owns the unit-quaternion contract.
    t = _scale(_cross(q[:3], v), 2.)
    return _add(v, _add(_scale(t, q[3]), _cross(q[:3], t)))


def _slew(current, desired, limit):
    angle = _angle(current, desired)
    if angle <= limit or angle < 1e-12:
        return desired
    tangent = _sub(desired, _scale(current, _dot(current, desired)))
    if _dot(tangent, tangent) < 1e-20:
        reference = (0., 1., 0.) if abs(current[1]) < .9 else (0., 0., 1.)
        tangent = _cross(current, reference)
    return _unit(_add(_scale(current, math.cos(limit)),
                      _scale(_unit(tangent), math.sin(limit))))


def emission_profile(age_s, times):
    """Recovered finite-age addend, four ordered endpoints in seconds.

    A raw three-element emissionProfile [a,b,c] is passed as (0,a,b,c).
    Continuous aircraft sources do not use this finite-age envelope.
    """
    a, b, c, d = times
    if not (a <= b <= c <= d):
        raise ValueError('IR profile endpoints must be ordered')
    if age_s < a or age_s > d:
        return 0.
    if age_s < b:
        return (age_s-a)/(b-a) if b-a > 4e-19 else 0.
    if age_s <= c:
        return 1.
    return 1.-(age_s-c)/(d-c) if d-c > 4e-19 else 1.


class InfraredSeeker:
    """Persistent angular seeker; all ranges/Doppler in outputs are invalid.

    Native alpha/beta formulas use measured direction history, never source
    velocity. Initial search points along body +X. Track/hold steers only from
    the observer prediction. With no designation input this class cannot slew
    to an invisible target outside its acquisition FOV.
    """

    def __init__(self, rocket, *, defaults=None):
        raw = rocket.get('rocket', rocket)['guidance']['opticalSeeker']
        p = dict(NATIVE_DEFAULTS)
        p.update(defaults or {})
        p.update(raw)
        p['fov'] = raw.get('fov', p.get('fov', p['angleHalfSens']))
        if p.get('targetSignatureType', 'infraRed') != 'infraRed':
            raise NotImplementedError('This seeker implements only the infraRed signature branch')
        for key, value in p.items():
            if isinstance(value, (int, float)) and not math.isfinite(value):
                raise ValueError('Nonfinite optical parameter: '+key)
        for key in ('fov', 'angleMax', 'lockAngleMax', 'rateMax', 'rangeMax'):
            if p[key] <= 0:
                raise ValueError('Positive optical parameter required: '+key)
        for key in ('prolongationTimeMax', 'trackResetTimeMax', 'rejectionReactionTime',
                    'outGateTransparency', 'gateWidth', 'angleGateRate'):
            if p[key] < 0:
                raise ValueError('Nonnegative optical parameter required: '+key)
        if p['targetsResolution']:
            raise NotImplementedError('IR resolved-target branch has not been reconstructed')
        if p['scanHalfSector'] > p['fov']/2:
            raise NotImplementedError('IR sector-search scan waveform has not been reconstructed')
        if p['boundaryTrack'] or p['mandatoryDesignationSources']:
            raise NotImplementedError('IR boundary/designation operation requires an explicit adapter')
        mask = int(p['bandMaskToReject'])
        bands = p.get('bandToReject', ())
        if isinstance(bands, int): bands = (bands,)
        for band in bands:
            if not 0 <= band <= 9:
                raise ValueError('IR rejection band must be in 0..9')
            mask |= 1 << band
        self.p, self.mask = p, mask
        # Native band0 falls back to range; other absent bands to zero. Loader
        # clamps R^2 to >=1 even for a zero raw band range.
        self.band_weights = {k: max(1., float(p.get('rangeBand'+str(k),
                                          p['range'] if k == 0 else 0.))**2)
                             for k in range(10)}
        self.half = math.radians(p['fov']/2)
        self.gate = math.sin(self.half)
        self.n = None
        self.w = (0., 0., 0.)
        self.boresight = None
        self.last_time = None
        self.last_seen = None
        self.rejection_age = 0.
        self.has_tracked = False
        self.last_measurement = None

    def update(self, time_s, dt_s, missile, scene):
        if not math.isfinite(time_s) or not math.isfinite(dt_s) or dt_s < 0:
            raise ValueError('IR time must be finite and dt nonnegative')
        if self.last_time is not None and time_s < self.last_time:
            raise ValueError('IR observation time must not decrease')
        if self.last_time == time_s:
            return self.last_measurement
        step = dt_s if self.last_time is None else time_s-self.last_time
        self.last_time = time_s
        p = self.p
        forward = _unit(_rotate(missile.orientation, (1., 0., 0.)))
        if self.boresight is None:
            self.boresight = forward
        predicted = None if self.n is None else _unit(_add(self.n, _scale(_cross(self.n, self.w), step)))
        desired = forward if predicted is None else predicted
        self.boresight = _slew(self.boresight, desired, math.radians(p['rateMax'])*step)
        mechanical = math.radians(p['angleMax'])
        if _angle(forward, self.boresight) > mechanical:
            self.boresight = _slew(forward, self.boresight, mechanical)
        # Rectangular gate projections use a world basis, with missile up as
        # reference (and its right axis at the pole).
        reference = _rotate(missile.orientation, (0., 1., 0.))
        right = _cross(self.boresight, reference)
        if _dot(right, right) < 1e-20:
            right = _rotate(missile.orientation, (0., 0., 1.))
        right = _unit(right)
        up = _unit(_cross(right, self.boresight))
        sources = [(scene.target.identity, scene.target.position, scene.target_ir,
                    scene.target.present)]
        sources.extend((f.identity, f.position, f.emission, f.present) for f in scene.flares)
        gate_samples, outside_samples, visible = [], [], []
        gimbal = mechanical if self.n is not None else math.radians(p['lockAngleMax'])
        for identity, position, emission, present in sources:
            if not present:
                continue
            if any(not isinstance(k, int) or not 0 <= k <= 9 or
                   not math.isfinite(v) or v < 0 for k, v in emission.items()):
                raise ValueError('IR emission requires finite nonnegative normalized bands 0..9')
            r = _sub(position, missile.position)
            r2 = _dot(r, r)
            if r2 <= 0 or r2 > p['rangeMax']**2:
                continue
            n = _scale(r, 1/math.sqrt(r2))
            if (_angle(n, forward) > gimbal+1e-12 or
                    _angle(n, self.boresight) > self.half+1e-12):
                continue
            signal, band = max(((v*self.band_weights[k]/r2, k)
                                for k, v in emission.items()), default=(0., 0),
                               key=lambda x: (x[0], -x[1]))
            if signal <= 0:
                continue
            visible.append(identity)
            rejected = bool((1 << band) & self.mask)
            inside = abs(_dot(n, up)) < self.gate and abs(_dot(n, right)) < self.gate
            weight = 1. if inside else p['outGateTransparency']
            gate_samples.append((identity, n, signal*weight, rejected, band))
            if not inside:
                outside_samples.append((identity, n, signal, rejected, band))
        gate_total = sum(x[2] for x in gate_samples)
        chosen = gate_samples if gate_total > 1. else outside_samples
        selected_set = 'gate' if gate_total > 1. else 'outside'
        total = sum(x[2] for x in chosen)
        rejected = sum(x[2] for x in chosen if x[3])
        if rejected > p['signalRelRejectedTreshold']*total:
            self.rejection_age += step
        else:
            self.rejection_age = 0.
        suspended = self.rejection_age > p['rejectionReactionTime']
        accepted = total > 1. and not suspended
        reason = 'irccm_rejection' if suspended else 'no_signal' if not accepted else ''
        measured = None
        if accepted:
            measured = _unit(tuple(sum(x[2]*x[1][i] for x in chosen) for i in range(3)))
            # Reacquisition resets only after the configured observer reset
            # age; missing input never supplies a geometric truth reset.
            reset = (self.n is None or self.last_seen is None or
                     time_s-self.last_seen > p['trackResetTimeMax'])
            if reset:
                self.n, self.w = measured, (0., 0., 0.)
            elif step > 0:
                residual = _cross(measured, predicted)
                if any(abs(x) > math.radians(p['angleGateRate'])*step for x in residual):
                    accepted, reason = False, 'angular_gate'
                else:
                    limit = math.radians(p['rateMax'])*step
                    correction = tuple(_clip(x, -limit, limit) for x in residual)
                    self.n = _unit(_add(predicted, _scale(_cross(predicted, correction), p['filterAlpha'])))
                    self.w = tuple(_clip(w+p['filterBetta']*e/step, -10., 10.)
                                   for w, e in zip(self.w, correction))
            if accepted:
                self.last_seen = time_s
                self.has_tracked = True
                # Point-source extent is zero; native min(max(extent/(2r),
                # sin(gateWidth/2)),sin(fov/2)) reduces to this lower clamp.
                self.gate = min(math.sin(math.radians(p['gateWidth']/2)), math.sin(self.half))
        if not accepted and predicted is not None:
            self.n = predicted
        age = None if self.last_seen is None else time_s-self.last_seen
        held = not accepted and age is not None and age <= p['prolongationTimeMax']
        valid = accepted or held
        mode = 'track' if accepted else 'suspended' if held and suspended else 'hold' if held else 'lost' if self.has_tracked else 'search'
        output_n = self.n if self.n is not None else self.boresight
        output_w = self.w
        if not valid and self.has_tracked:
            # Search reinitialization restores the full gate after the hold
            # expires. Failure while holding preserves the previous gate.
            self.n, self.w = None, (0., 0., 0.)
            self.boresight = forward
            self.gate = math.sin(self.half)
        result = SensorMeasurement(output_n, output_w, valid, mode,
            'ir_angular_observer' if accepted else 'ir_angular_prediction' if held else 'ir_no_solution',
            fresh=accepted, held=held,
            metadata={'signal': total, 'rejected_signal': rejected,
                      'rejection_age_s': self.rejection_age, 'selected_set': selected_set,
                      'gate_signal': gate_total, 'outside_signal': sum(x[2] for x in outside_samples),
                      'visible_identities': tuple(visible),
                      'selected_bands': {x[0]: x[4] for x in chosen if x[2] > 0},
                      'measured_direction': measured, 'reason': reason,
                      'gate_sine': self.gate, 'observation_age_s': age,
                      'propagation': 'point_source_inverse_square_hard_fov',
                      'unsupported': ('finite_size_edge_response', 'occlusion_sun_background',
                                      'resolved_targets', 'sector_scan_designation')})
        self.last_measurement = result
        return result
