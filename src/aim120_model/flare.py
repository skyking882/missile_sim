"""Deterministic game flare entities, independent of radar/chaff reflectors.

Raw fields come from a caller-supplied game weapon file. Their interpretation
as a linear rise/plateau/fall and a small rocket with quadratic drag is level D
(assumed), not recovered game engine code. Units are game SI conventions.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from .math3d import add, scale, norm
from .atmosphere import StandardAtmosphere
from .infrared_data import InfraredGameData


FLARE_WEAPONS = {
    'regular': 'countermeasure_split_launcher_jet',
    'bol': 'countermeasure_split_launcher_jet_bol',
    'large': 'countermeasure_split_launcher_jet_only_flare_large',
}


def _vector(value):
    v = tuple(float(x) for x in value)
    if len(v) != 3 or not all(math.isfinite(x) for x in v):
        raise ValueError("Flare vectors must be finite xyz triples")
    return v


@dataclass(frozen=True)
class FlareSpec:
    mass_kg: float
    mass_end_kg: float
    caliber_m: float
    drag_coefficient: float
    thrust_n: float
    burn_s: float
    lifetime_s: float
    eject_speed_mps: float
    emission_profile_s: tuple[float, float, float]
    emission_band2: float
    density_kg_m3: float | None = None
    integration_step_s: float = .01
    source: str = "caller supplied; dynamics and envelope assumed (D)"
    motion_mode: str = 'rocket'
    use_start_speed: bool = True
    parameter_sources: dict | None = None

    def __post_init__(self):
        positive = (self.mass_kg, self.mass_end_kg, self.caliber_m,
                    self.lifetime_s, self.integration_step_s)
        nonnegative = (self.drag_coefficient, self.thrust_n, self.eject_speed_mps,
                       self.emission_band2, self.burn_s)
        if (not all(math.isfinite(x) and x > 0 for x in positive)
                or not all(math.isfinite(x) and x >= 0 for x in nonnegative)
                or self.mass_end_kg > self.mass_kg):
            raise ValueError("Invalid flare physical fields")
        if self.density_kg_m3 is not None and (not math.isfinite(self.density_kg_m3) or self.density_kg_m3 < 0):
            raise ValueError('Flare density override must be finite and nonnegative')
        if self.motion_mode not in ('rocket', 'ballistic') or not isinstance(self.use_start_speed, bool):
            raise ValueError('Flare needs an explicit rocket/ballistic motion mode and start-speed flag')
        p = tuple(float(x) for x in self.emission_profile_s)
        if len(p) != 3 or not all(math.isfinite(x) for x in p) or not 0 < p[0] <= p[1] < p[2] <= self.lifetime_s:
            raise ValueError("Flare emission profile needs 0 < rise <= fall_start < end <= lifetime")
        object.__setattr__(self, 'emission_profile_s', p)

    @classmethod
    def from_game_weapon(cls, weapon, *, density_kg_m3=None, source=None,
                         version='2.59.0.27', project_dir=None, bullet_index=None):
        """Accept the raw rocket or full launcher; caller accepts D mappings.

        `signature.infraRed.emission2` is retained as band 2 game emission,
        not kelvin, watts, RCS or a calibrated aircraft-to-flare ratio.
        Strings may name regular/bol/large or a precise launcher identifier.
        Nested chaff objects are never searched for a glowing payload. A list
        with multiple real flares requires explicit bullet_index selection.
        Density defaults to the existing runtime StandardAtmosphere at each
        integration state; a numerical override retains the original API.
        """
        data = InfraredGameData(version, project_dir=project_dir)
        provenance = None
        if isinstance(weapon, str):
            name = FLARE_WEAPONS.get(weapon, weapon)
            weapon = data.weapon(name)
            path = 'aces.vromfs.bin_u/gamedata/weapons/rocketguns/'+name+'.blkx'
            provenance = data.provenance(path)
            source = source or f'{version} {path}; dynamics/envelope assumed D'
        candidates = weapon.get('bullet', weapon)
        candidates = candidates if isinstance(candidates, list) else [candidates]
        payloads = []
        for index, candidate in enumerate(candidates):
            r = candidate.get('rocket', candidate.get('ballistic', candidate))
            ir = r.get('signature', {}).get('infraRed', {})
            if (r.get('bulletType', candidate.get('bulletType')) in ('flare', 'flr')
                    and 'emission2' in ir and 'emissionProfile' in ir and ir['emission2'] > 0):
                payloads.append((index,r,'rocket' if 'rocket' in candidate or 'timeFire' in r else 'ballistic'))
        if bullet_index is not None:
            payloads = [x for x in payloads if x[0] == bullet_index]
        if len(payloads) != 1:
            raise ValueError('Expected one emitting flare payload; non-flare/chaff rejected, multiple flares need bullet_index')
        index, r, mode = payloads[0]
        ir = r['signature']['infraRed']
        if mode == 'rocket':
            use_start, eject = r['useStartSpeed'], r['startSpeed'] if r['useStartSpeed'] else 0.
            burn, thrust, end_mass, lifetime = r['timeFire'], r['force'], r['massEnd'], r['timeLife']
        else:
            # Only explicitly typed flare ballistic payloads reach here.
            use_start, eject = True, r['speed']
            burn, thrust, end_mass, lifetime = 0., 0., r['mass'], r['timeLife']
        fields = {k:r[k] for k in ('mass','caliber','CxK')}
        fields.update({k:r[k] for k in ('massEnd','timeFire','force','timeLife','useStartSpeed','startSpeed','speed') if k in r})
        fields['signature.infraRed'] = ir
        metadata = dict(kind='assumed', evidence_level='D', raw_fields=fields,
                        payload_index=index, motion_mode=mode, source=provenance,
                        assumptions=['linear emission rise/plateau/fall', 'quadratic CxK drag and linear burn mass',
                                     'startSpeed is relative ejection when enabled, otherwise carrier velocity',
                                     'fixed initial ejection axis; ISA density unless explicitly overridden'])
        return cls(r['mass'], end_mass, r['caliber'], r['CxK'], thrust,
                   burn, lifetime, eject,
                   tuple(ir['emissionProfile']), ir['emission2'], density_kg_m3,
                   source=source or 'caller raw game payload; semantics assumed D',
                   motion_mode=mode, use_start_speed=use_start, parameter_sources=metadata)

    @classmethod
    def from_game_aircraft(cls, aircraft_model, *, launcher=None, bullet_index=None,
                           version='2.59.0.27', project_dir=None, **overrides):
        """Use an actually mounted flare payload, resolving mixed lists explicitly.

        A plane with several different payload choices requires launcher/index;
        choices are in infrared_coverage()['aircraft'][model]['flare_payloads'].
        This selects the thermal/motion spec; release timing and axis are inputs.
        """
        data = InfraredGameData(version,project_dir=project_dir)
        options = data.aircraft(aircraft_model)['flare_payloads']
        options = [x for x in options if (launcher is None or x['weapon']==launcher)
                   and (bullet_index is None or x['bullet_index']==bullet_index)]
        if len(options) != 1:
            raise ValueError(f'{aircraft_model} needs one actual flare launcher/payload choice; options={options}')
        selected = options[0]
        return cls.from_game_weapon(selected['weapon'],bullet_index=selected['bullet_index'],
            version=version,project_dir=project_dir,**overrides)

    def emission(self, age):
        rise, fall, end = self.emission_profile_s
        if age <= 0 or age >= end:
            return 0.
        envelope = age/rise if age < rise else (end-age)/(end-fall) if age > fall else 1.
        return self.emission_band2*envelope

    def motion(self, age, position, velocity, direction):
        """Fixed-step RK4, recomputed from release: query cadence cannot alter it."""
        state = tuple(position)+tuple(velocity)
        area = math.pi*self.caliber_m**2/4
        def derivative(t, s, thrust_on):
            mass = (self.mass_kg+(self.mass_end_kg-self.mass_kg)*min(t/self.burn_s, 1.)
                    if self.burn_s > 0 else self.mass_kg)
            v = s[3:]
            density = self.density_kg_m3
            if density is None:
                density = StandardAtmosphere().sample(s[1]).density_kg_m3
            drag = scale(v, -.5*density*self.drag_coefficient*area*norm(v)/mass)
            a = add(add(drag, (0., -9.80665, 0.)), scale(direction, self.thrust_n/mass if thrust_on else 0.))
            return v+a
        t = 0.
        while t < age:
            h = min(self.integration_step_s, age-t)
            if t < self.burn_s < t+h:
                h = self.burn_s-t
            on = t < self.burn_s
            k1 = derivative(t, state, on)
            k2 = derivative(t+h/2, tuple(x+h*k/2 for x,k in zip(state,k1)), on)
            k3 = derivative(t+h/2, tuple(x+h*k/2 for x,k in zip(state,k2)), on)
            k4 = derivative(t+h, tuple(x+h*k for x,k in zip(state,k3)), on)
            state = tuple(x+h*(a+2*b+2*c+d)/6 for x,a,b,c,d in zip(state,k1,k2,k3,k4))
            t += h
        return state[:3], state[3:]


@dataclass(frozen=True)
class FlareRelease:
    time_s: float
    identity: str
    direction_world: tuple[float, float, float]
    offset_world_m: tuple[float, float, float] = (0., 0., 0.)

    def __post_init__(self):
        direction, offset = _vector(self.direction_world), _vector(self.offset_world_m)
        if not math.isfinite(self.time_s) or self.time_s < 0 or not isinstance(self.identity, str) or not self.identity:
            raise ValueError("Flare release requires nonnegative time and stable nonempty identity")
        if norm(direction) <= 0:
            raise ValueError("Flare ejection direction must be nonzero")
        object.__setattr__(self, 'direction_world', scale(direction, 1/norm(direction)))
        object.__setattr__(self, 'offset_world_m', offset)


@dataclass(frozen=True)
class FlareState:
    identity: str
    release_time_s: float
    position: tuple[float, float, float]
    velocity: tuple[float, float, float]
    emission_band2: float
    release_position: tuple[float, float, float]
    carrier_velocity: tuple[float, float, float]


class FlaringTarget:
    """Wrap an aircraft; no future release queries and no duplicate entities.

    Ejection direction/offset use world axes, supplied by caller at each release.
    A moving dispenser/body attitude mapping is deliberately not guessed.
    Other target capabilities (including chaff) are delegated unchanged.
    """
    def __init__(self, inner, spec, releases):
        self.inner, self.spec = inner, spec
        self.releases = tuple(sorted(releases, key=lambda x: (x.time_s, x.identity)))
        if len({r.identity for r in self.releases}) != len(self.releases):
            raise ValueError("Flare identities must be unique within a target")
        self._release_states = {}

    def __getattr__(self, name):
        return getattr(object.__getattribute__(self, 'inner'), name)

    def state_at(self, time_s):
        return self.inner.state_at(time_s)

    def describe(self):
        inner = getattr(self.inner, 'describe', None)
        base = inner() if inner is not None else dict(model=type(self.inner).__name__)
        return dict(base, flares=dict(spec_source=self.spec.source, parameter_sources=self.spec.parameter_sources,
            emission_band2=self.spec.emission_band2, emission_profile_s=self.spec.emission_profile_s,
            lifetime_s=self.spec.lifetime_s, density='StandardAtmosphere' if self.spec.density_kg_m3 is None else self.spec.density_kg_m3,
            releases=[dict(time_s=r.time_s, identity=r.identity, direction_world=r.direction_world,
                           offset_world_m=r.offset_world_m) for r in self.releases]))

    def flares_at(self, time_s):
        if not math.isfinite(time_s):
            raise ValueError("Flare time must be finite")
        out = []
        for r in self.releases:
            age = time_s-r.time_s
            if age < 0 or age >= self.spec.lifetime_s:
                continue
            if r.identity not in self._release_states:
                carrier = self.inner.state_at(r.time_s)
                p = add(_vector(carrier.position), r.offset_world_m)
                v = _vector(carrier.velocity)
                self._release_states[r.identity] = p, v
            p0, v0 = self._release_states[r.identity]
            p, v = self.spec.motion(age, p0, add(v0, scale(r.direction_world, self.spec.eject_speed_mps)), r.direction_world)
            out.append(FlareState(r.identity, r.time_s, p, v, self.spec.emission(age), p0, v0))
        return out


def flaring_factory(inner_factory, spec, releases):
    releases = tuple(releases)
    def make(base):
        return FlaringTarget(inner_factory(base) if inner_factory is not None else base, spec, releases)
    return make


__all__ = ['FlareSpec', 'FlareRelease', 'FlareState', 'FlaringTarget', 'flaring_factory', 'FLARE_WEAPONS']
