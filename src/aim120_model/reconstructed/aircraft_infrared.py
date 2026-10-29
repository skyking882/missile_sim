"""Aircraft IR inputs assembled from frozen War Thunder FM/global game data.

Explicitly opted-in legacy approximation, not the recovered aircraft producer.
The native B0/B1/Bab engine producer and its conversion to the optical I/C0
contract are not closed. Throttle→thrust, continuous aspect interpolation,
airframe Mach heating and F-16C/200 normalization remain D assumptions.
No real aircraft temperature, materials or weapon performance is imported.
"""
from __future__ import annotations

import bisect
import copy
import math
from dataclasses import dataclass

from ..atmosphere import StandardAtmosphere
from .infrared_data import InfraredGameData
from ..math3d import dot, norm, scale, sub
from .python_math118 import matrix


GLOBAL_PATH = 'aces.vromfs.bin_u/config/gameplay.blkx'
FM_PREFIX = 'aces.vromfs.bin_u/gamedata/flightmodels/fm/'
REFERENCE_AIRCRAFT = 'f_16c_block_50'


def _interp(nodes, value):
    if value <= nodes[0][0]:
        return nodes[0][1]
    if value >= nodes[-1][0]:
        return nodes[-1][1]
    i = bisect.bisect_right([x[0] for x in nodes], value)-1
    a, b = nodes[i:i+2]
    return a[1]+(b[1]-a[1])*(value-a[0])/(b[0]-a[0])


def _coefficient(table, prefix, altitude, speed_kmh):
    heights = sorted((int(k.split('_')[1]), v) for k,v in table.items() if k.startswith('Altitude_'))
    speeds = sorted((int(k.split('_')[1]), v) for k,v in table.items() if k.startswith('Velocity_'))
    rows = []
    for i, height in heights:
        # Some real FM tables are ragged. Interpolate only declared coefficients;
        # clamp to each row's actual domain instead of inventing missing cells.
        nodes = [(speed, table[f'{prefix}_{i}_{j}']) for j,speed in speeds if f'{prefix}_{i}_{j}' in table]
        if not nodes:
            raise ValueError(f'Aircraft thrust table has no {prefix} coefficients at altitude index {i}')
        rows.append((height, _interp(nodes, speed_kmh)))
    return _interp(rows, altitude)


def _engine_main(base, override):
    """EngineN inherits its referenced EngineTypeN, then local fields override."""
    merged = copy.deepcopy(base)
    for k,v in override.items():
        merged[k] = _engine_main(merged[k],v) if isinstance(v,dict) and isinstance(merged.get(k),dict) else copy.deepcopy(v)
    return merged


@dataclass(frozen=True)
class AircraftEngineState:
    """Behavioral input, not an invented thermal coefficient.

    A callable(t) may provide changing controls; all three states are explicit.
    This initial adapter drives all installed engines with the same command.
    No residual engine heat/cooldown time is inferred from the thrust input.
    """
    throttle: float
    afterburner: bool
    running: bool

    def __post_init__(self):
        if not math.isfinite(self.throttle) or not 0 <= self.throttle <= 1:
            raise ValueError('Aircraft throttle must be in [0, 1]')
        if not isinstance(self.afterburner, bool) or not isinstance(self.running, bool):
            raise ValueError('Aircraft afterburner and running states must be booleans')


class AircraftInfraredSignature:
    """Versioned jet-aircraft bands 0/1/7 from thrust, airframe and afterburner.

    Aspect knots come from gameplay: front/side/rear are linearly interpolated
    in cosine. Forward uses the scene body's XYZW orientation, or an explicit
    forward_world vector/callback. Velocity is never substituted for attitude.

    A reference dry F-16C engine from the SAME frozen game data defines one
    normalized engine unit. Airframe uses its global 200 reference. Flare
    emission2 is used directly as band-2 normalized emission. This common
    reference is an assumed closure, not the recovered engine normalization.
    """
    def __init__(self, aircraft_model, *, engine_state=None, forward_world=None,
                 version='2.59.0.27', project_dir=None):
        self.data = InfraredGameData(version, project_dir=project_dir)
        self.aircraft_model = self.data.identifier(aircraft_model)
        if engine_state is None:
            raise ValueError('Legacy FM approximation needs explicit throttle, afterburner and running states')
        self.engine_state = engine_state
        if not callable(self.engine_state) and not isinstance(self.engine_state, AircraftEngineState):
            raise ValueError('engine_state must be AircraftEngineState or callable(t)')
        self.forward_world = forward_world
        global_data = self.data.read(GLOBAL_PATH)
        self.constants = global_data['sensorsConstants']
        self.conversion = global_data['fmConstants']
        self.fm, path, unit_path = self.data.aircraft_fm(self.aircraft_model)
        self.engines = []
        coverage = self.data.coverage()
        velocity_assumptions = coverage['thrust_velocity_assumptions']
        applied_velocity_assumptions = []
        for k, engine in self.fm.items():
            if k.startswith('Engine') and k[6:].isdigit():
                main = _engine_main(self.fm[f"EngineType{engine['Type']}"]['Main'],engine.get('Main',{}))
                if main['Type'] != 'Jet':
                    raise ValueError('Aircraft IR adapter currently supports game Jet engines only')
                velocity_type = main['ThrustMax'].get('VelocityType')
                if velocity_type is None and path in velocity_assumptions:
                    velocity_type = velocity_assumptions[path]['value']
                    applied_velocity_assumptions.append(velocity_assumptions[path])
                if velocity_type != 'TAS':
                    raise ValueError('Aircraft IR thrust table needs TAS or a recorded model-specific interpretation')
                self.engines.append(main)
        if not self.engines:
            raise ValueError('Aircraft FM has no supported engine instances')
        ref_path = FM_PREFIX+REFERENCE_AIRCRAFT+'.blkx'
        ref = self.data.read(ref_path)['EngineType0']['Main']['ThrustMax']
        self.reference_engine_brightness = (ref['ThrustMax0']*ref['ThrustMaxCoeff_0_0']
                                            *self.conversion['thrustKgsToInfraRedBrightness'])
        self.reference_airframe_brightness = self.constants['airframeInfraRedBrightnessMult']
        self.provenance = dict(kind='legacy_fm_approximation', evidence_level='D', aircraft_model=self.aircraft_model,
            native_engine_producer='B0/B1/Bab production path not reconstructed',
            optical_contract_mapping='unverified: legacy F16C/200-normalized bands used as an explicit signature proxy; not established I/C0 units',
            version=version, commit=self.data.commit, sources=[self.data.provenance(path),
                self.data.provenance(unit_path), self.data.provenance(GLOBAL_PATH), self.data.provenance(ref_path)],
            fm_source=path, table_velocity_assumptions=applied_velocity_assumptions,
            sparse_thrust_table=coverage['sparse_thrust_tables'].get(path),
            engine_count=len(self.engines),
            source_fields=dict(engine='EngineN.Type -> EngineTypeN.Main.ThrustMax/MinThrMult/MaxThrMult/AfterburnerBoost',
                conversion='fmConstants.thrustKgsToInfraRedBrightness / afterburnerThrustKgsToInfraRedBrightness',
                aspect='sensorsConstants.engine[Afterburner]InfraRedBrightnessAspectMultFront/Side/Rear',
                airframe='sensorsConstants.airframeInfraRedBrightnessMult/TemperatureMachBase/TemperatureRatioToBrightness*'),
            conversion_values={k:self.conversion[k] for k in ('thrustKgsToInfraRedBrightness',
                'afterburnerThrustKgsToInfraRedBrightness','horsePowersToInfraRedBrightness')},
            aspect_values={k:v for k,v in self.constants.items() if 'BrightnessAspectMult' in k},
            normalization=dict(kind='assumed', reference_aircraft=REFERENCE_AIRCRAFT,
                engine_brightness=self.reference_engine_brightness,
                airframe_brightness=self.reference_airframe_brightness,
                flare='raw signature.infraRed.emission2; unit reference assumed'),
            assumptions=['linear throttle-to-thrust between raw MinThrMult/MaxThrMult',
                'altitude/TAS km/h interpolation using only actual row nodes; row/domain boundary clamping',
                'afterburner excess thrust times raw 4.5 brightness conversion',
                'linear cosine interpolation of game front/side/rear aspect knots',
                'airframe ratio=max(1,(1+.2*M^2)/(1+.2*MachBase^2)), raw exponent and altitude multiplier',
                'no residual engine thermal inertia or radiometric calibration'],
            engine_input=('callable(t)' if callable(self.engine_state) else dict(
                throttle=self.engine_state.throttle, afterburner=self.engine_state.afterburner,
                running=self.engine_state.running)))

    def _aspect(self, prefix, rear_cosine):
        s = self.constants
        return _interp([(-1., s[prefix+'Front']), (0.,s[prefix+'Side']), (1.,s[prefix+'Rear'])],rear_cosine)

    def components(self, time_s, target, missile):
        controls = self.engine_state(time_s) if callable(self.engine_state) else self.engine_state
        if not isinstance(controls, AircraftEngineState):
            raise ValueError('engine_state(t) must return AircraftEngineState')
        q = getattr(target,'orientation',None)
        if q is None:q=getattr(target,'q',None)
        forward = tuple(row[0] for row in matrix(q)) if q is not None else self.forward_world(time_s,target) if callable(self.forward_world) else self.forward_world
        if forward is None:
            raise ValueError('Aircraft IR aspect needs scene body orientation or explicit forward_world; velocity is not attitude')
        forward = tuple(float(x) for x in forward)
        if len(forward) != 3 or not all(math.isfinite(x) for x in forward) or norm(forward) == 0:
            raise ValueError('Aircraft IR aspect needs a finite nonzero body forward direction')
        to_observer = sub(missile.position, target.position)
        rear = 0. if norm(to_observer) == 0 else -dot(scale(forward,1/norm(forward)),scale(to_observer,1/norm(to_observer)))
        dry, ab = 0., 0.
        altitude, speed = target.position[1], norm(target.velocity)
        if controls.running:
            for main in self.engines:
                table = main['ThrustMax']
                power = main['MinThrMult']+(main['MaxThrMult']-main['MinThrMult'])*controls.throttle
                thrust = table['ThrustMax0']*_coefficient(table,'ThrustMaxCoeff',altitude,speed*3.6)*power
                dry += thrust*self.conversion['thrustKgsToInfraRedBrightness']
                if controls.afterburner:
                    full_ab = (table['ThrustMax0']*_coefficient(table,'ThrAftMaxCoeff',altitude,speed*3.6)
                               *main['AfterburnerBoost']*power)
                    ab += max(0.,full_ab-thrust)*self.conversion['afterburnerThrustKgsToInfraRedBrightness']
        dry *= self._aspect('engineInfraRedBrightnessAspectMult', rear)
        ab *= self._aspect('engineAfterburnerInfraRedBrightnessAspectMult', rear)
        s = self.constants
        mach = speed/StandardAtmosphere().sample(altitude).speed_of_sound_mps
        ratio = max(1.,(1+.2*mach**2)/(1+.2*s['airframeInfraRedTemperatureMachBase']**2))
        altitude_mult = _interp([tuple(s['airframeInfraRedTemperatureRatioToBrightnessMult0']),
                                 tuple(s['airframeInfraRedTemperatureRatioToBrightnessMult1'])],altitude)
        airframe = s['airframeInfraRedBrightnessMult']*ratio**s['airframeInfraRedTemperatureRatioToBrightnessPwr']*altitude_mult
        # Source mask bit -> band index is explicit and checked, not assumed by name.
        masks = [s['engineInfraRedBrightnessSourcesMask'],s['airframeInfraRedBrightnessSourcesMask'],
                 s['engineAfterburnerInfraRedBrightnessSourcesMask']]
        if masks != [1,2,128]:
            raise ValueError('Unsupported aircraft IR source-mask mapping in frozen game data')
        return dict(engine_brightness=dry, afterburner_brightness=ab, airframe_brightness=airframe,
                    bands={0:dry/self.reference_engine_brightness,
                           1:airframe/self.reference_airframe_brightness,
                           7:ab/self.reference_engine_brightness})

    def __call__(self, time_s, target, missile):
        return self.components(time_s,target,missile)['bands']


__all__ = ['AircraftEngineState', 'AircraftInfraredSignature']
