"""Opt-in War Thunder IR observations and two independently enabled IRCCM rules.

This is a source-informed, deterministic game approximation, not recovered game
code or a calibrated seeker. Raw opticalSeeker field values are evidence A;
their units, full/half angle convention, band mapping, signal competition and
memory rules here remain assumed (D). The default adapter loads aircraft/global
thermal data and band ranges from the same frozen dataset; its normalization is
explicitly assumed. Legacy caller thermal callbacks remain supported.

Sources read 2026-10-16:
* Gaijin, 2018-12-14, https://warthunder.com/en/news/5947-development-air-to-air-missiles-en
  Describes an in-FOV false heat source pulling aim between sources (old version).
* Community wiki (official hosting, not developer specification),
  https://wiki.warthunder.com/mechanics/5026-aerial-infrared-countermeasures-flares
  Describes narrowing and temporary guidance suspension; flare warm-up, aspect.
* CalvinAz, 2023-09-06, player explanation on official forum,
  https://forum.warthunder.com/t/missile-irccm-aim-9m-vs-r-73/24366
  Names R-73 gating and AIM-9M suspension; exact algorithm not established.
  Do not treat this development-server post as proof for 2.59.0.27.

Only angles are sensed by this model. The TrackSolution required by existing PN
uses *explicit truth-assisted range and velocity* of visible signal sources.
These quantities are simulation closure, not IR range/velocity measurements.
No independent axial angular state is fabricated; full_rate_rad_s stays None.
"""
from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Callable

from .math3d import add, sub, scale, norm, dot
from .tracking import TrackMode, TrackSolution


SOURCES = (
    'https://warthunder.com/en/news/5947-development-air-to-air-missiles-en',
    'https://wiki.warthunder.com/mechanics/5026-aerial-infrared-countermeasures-flares',
    'https://forum.warthunder.com/t/missile-irccm-aim-9m-vs-r-73/24366',
)


class InfraredSeekerState(str, Enum):
    SEARCH = 'search'
    TRACK = 'track'
    SUSPENDED = 'suspended'
    MEMORY = 'memory'
    LOST = 'lost'


def _angle(a, b):
    return math.acos(max(-1., min(1., dot(a, b))))


def _unit(v):
    size = norm(v)
    if size <= 1e-12:
        raise ValueError('IR source direction is undefined at contact')
    return scale(v, 1/size)


@dataclass(frozen=True)
class InfraredSeekerConfig:
    """Raw callback configuration, or a versioned game-data assembly factory.

    target_emission(t, target_state, missile_state) returns emission in the same
    arbitrary game-relative units as flare_gain * emission_band2. It may depend
    on throttle/afterburner/aspect via an explicit caller model. No aircraft heat
    profile is guessed. Signals use assumed emission/range², not radar RCS/range⁴.
    signal_rule='centroid' follows the old official qualitative description;
    'strongest' is an alternative D assumption. Neither is version-validated.
    """
    fov_half_deg: float
    gimbal_deg: float
    acquire_gimbal_deg: float
    rate_deg_s: float
    range_max_m: float
    memory_s: float
    target_emission: Callable
    flare_gain: float
    minimum_signal: float
    gated_half_deg: float | None = None
    reject_flare_fraction: float | None = None
    rejection_reaction_s: float = 0.
    signal_rule: str = 'centroid'
    gate_on_loss: str = 'restore_search'
    source: str = 'caller supplied; semantics assumed (D)'
    range_bands_m: tuple[tuple[int, float], ...] = ()
    thermal_provenance: dict | None = None
    weapon_provenance: dict | None = None

    def __post_init__(self):
        positive = (self.fov_half_deg, self.gimbal_deg, self.acquire_gimbal_deg,
                    self.rate_deg_s, self.range_max_m, self.minimum_signal)
        nonnegative = (self.memory_s, self.flare_gain, self.rejection_reaction_s)
        if (not all(math.isfinite(x) and x > 0 for x in positive)
                or not all(math.isfinite(x) and x >= 0 for x in nonnegative)
                or not callable(self.target_emission)):
            raise ValueError('Invalid IR geometry, timing or explicit thermal inputs')
        if (max(self.fov_half_deg, self.gimbal_deg, self.acquire_gimbal_deg) > 180
                or self.acquire_gimbal_deg > self.gimbal_deg):
            raise ValueError('IR angles require acquisition <= gimbal <= 180 degrees')
        if self.gated_half_deg is not None and (not math.isfinite(self.gated_half_deg)
                or not 0 < self.gated_half_deg <= self.fov_half_deg):
            raise ValueError('IR gate must be positive and no wider than search FOV')
        if self.reject_flare_fraction is not None and (not math.isfinite(self.reject_flare_fraction)
                or not 0 <= self.reject_flare_fraction <= 1):
            raise ValueError('IR rejection fraction must be in [0, 1]')
        if self.signal_rule not in ('centroid', 'strongest'):
            raise ValueError('IR signal_rule must be centroid or strongest')
        if self.gate_on_loss not in ('restore_search', 'retain_gate'):
            raise ValueError('IR gate_on_loss must be restore_search or retain_gate')
        bands = tuple((int(k),float(v)) for k,v in self.range_bands_m)
        if (len({k for k,v in bands}) != len(bands)
                or any(k < 0 or not math.isfinite(v) or v < 0 for k,v in bands)):
            raise ValueError('IR range bands need unique nonnegative indices and finite nonnegative ranges')
        object.__setattr__(self, 'range_bands_m', bands)

    @classmethod
    def from_game_weapon(cls, weapon, *, angles_are_full_width=True, target_emission=None,
                         flare_gain=None, minimum_signal=None, source=None,
                         aircraft_model=None, engine_state=None, forward_world=None,
                         launch_aircraft_model=None, version='2.59.0.27', project_dir=None, **assumptions):
        """Read raw 2.59 opticalSeeker without modifying the profile catalog.

        gateWidth and bandMaskToReject are independent features. Mask bit 4 is
        provisionally interpreted as emission2 (D); other masks are unsupported.
        A weapon identifier plus aircraft_model loads all game values. Aircraft
        engine_state/forward_world describe behavior; heat coefficients are read
        automatically. Normalized band signals sum e_b*(rangeBand_b/R)^2 with a
        unit detection threshold (D), capped by rangeMax. No scalar heat input
        is required. The default full-angle convention remains D.

        A caller target_emission callback and explicit gain/floor retain the
        original scalar signal contract; no new band scaling is applied to it.
        """
        from .infrared_data import InfraredGameData
        from .aircraft_infrared import AircraftInfraredSignature
        data = InfraredGameData(version, project_dir=project_dir)
        weapon_provenance = None
        if isinstance(weapon, str):
            name = data.identifier(weapon)
            path = data.mounted_weapon(name,launch_aircraft_model) if launch_aircraft_model else data.weapon_path(name)
            weapon = data.read(path)
            weapon_provenance = data.provenance(path)
            weapon_provenance['launch_aircraft_model'] = launch_aircraft_model
            source = source or f'{version} {path}; field mapping assumed D'
        if not isinstance(angles_are_full_width, bool):
            raise ValueError('Explicit angles_are_full_width boolean is required')
        r = weapon['rocket'] if 'rocket' in weapon else weapon
        if weapon_provenance is not None and (r.get('bulletType') != 'aam' or r.get('guidanceType') != 'optical'):
            raise ValueError('Game-data IR factory requires an optical air-to-air missile payload')
        s = r['guidance']['opticalSeeker']
        if weapon_provenance is not None:
            weapon_provenance['raw_optical_seeker'] = s
        mask = s.get('bandMaskToReject', 0)
        if mask not in (0, 4):
            raise ValueError('Only the assumed emission2/mask4 rejection mapping is supported')
        # AAM-3 has a rejection mask but lacks the 9M threshold/reaction fields.
        # Its mechanism is not inferred from another missile. An explicit caller
        # interpretation may still exercise the approximation, with provenance.
        rejection_override = assumptions.pop('reject_flare_fraction', None)
        if mask and 'signalRelRejectedTreshold' not in s and rejection_override is None:
            raise ValueError('Unresolved IRCCM: bandMaskToReject has no signalRelRejectedTreshold; '
                             'automatic mask-only rejection is unsupported, explicit D override required')
        rejection = rejection_override if rejection_override is not None else s.get('signalRelRejectedTreshold') if mask else None
        if weapon_provenance is not None and rejection_override is not None:
            weapon_provenance['rejection_override'] = dict(kind='assumed',evidence_level='D',value=rejection_override)
        thermal = None
        bands = ()
        if target_emission is None:
            if aircraft_model is None:
                raise ValueError('aircraft_model is required for game-data IR heat; no arbitrary aircraft is substituted')
            target_emission = AircraftInfraredSignature(aircraft_model, engine_state=engine_state,
                forward_world=forward_world, version=version, project_dir=project_dir)
            thermal = target_emission.provenance
            bands = tuple(sorted((int(k[9:]),v) for k,v in s.items() if k.startswith('rangeBand')))
            if not bands:
                raise ValueError('Game-data IR assembly requires explicit optical rangeBand values')
        if flare_gain is None:
            flare_gain = 1.
        if minimum_signal is None:
            if not bands:
                raise ValueError('Legacy scalar target_emission requires an explicit minimum_signal')
            minimum_signal = 1.
        factor = .5 if angles_are_full_width else 1.
        return cls(s['fov']*factor, s['angleMax'], s['lockAngleMax'], s['rateMax'],
            s['rangeMax'], s['prolongationTimeMax'], target_emission, flare_gain, minimum_signal,
            gated_half_deg=s['gateWidth']*factor if 'gateWidth' in s else None,
            reject_flare_fraction=rejection,
            rejection_reaction_s=s.get('rejectionReactionTime', 0.),
            source=source or f'{version} caller raw opticalSeeker; semantics assumed D',
            range_bands_m=bands, thermal_provenance=thermal, weapon_provenance=weapon_provenance, **assumptions)


@dataclass(frozen=True)
class InfraredDetection:
    valid: bool
    reason: str
    selected_identity: str | None
    visible_identities: tuple[str, ...]
    signal: float
    rejected_fraction: float
    fov_half_deg: float


class InfraredTrackProvider:
    provider_name = 'infrared_truth_assisted_v1'

    def __init__(self, config, target):
        if not isinstance(config, InfraredSeekerConfig):
            raise ValueError('infrared_seeker must be InfraredSeekerConfig')
        self.config, self.target = config, target
        self._time = None
        self._last_measurement = None
        self._solution = None
        self._boresight = None
        self._gated = False
        self._reject_since = None
        self.status = 'search'
        self.observation_reject_reason = ''
        self.last_detection = None
        self.first_lock_loss_time_s = None
        self.has_tracked_once = False
        self.provenance = dict(kind='assumed', evidence_level='D', source=config.source,
            sources=list(SOURCES), kinematic_closure='truth_assisted_range_velocity; no IR range measurement',
            signal='emission/range^2', signal_rule=config.signal_rule,
            gate_on_loss=config.gate_on_loss, fov_half_deg=config.fov_half_deg,
            gated_half_deg=config.gated_half_deg, reject_flare_fraction=config.reject_flare_fraction,
            rejection_reaction_s=config.rejection_reaction_s, memory_s=config.memory_s,
            thermal_data=config.thermal_provenance, range_bands_m=dict(config.range_bands_m),
            weapon_data=config.weapon_provenance,
            signal_normalization=('sum(normalized_band_emission * rangeBand^2)/R^2; unit detection threshold; assumed D'
                                  if config.range_bands_m else 'legacy caller emission/R^2'),
            minimum_signal=config.minimum_signal, flare_gain=config.flare_gain,
            limitations=['rangeBand/reference normalization not replay calibrated', 'no sun/background/occlusion',
                         'no spectral/rise-time classifier', 'no independent axial angular state',
                         'rangeBand3/6 retained; no additional environmental emitters modeled'])

    @property
    def seeker_state(self):
        return InfraredSeekerState(self.status)

    @property
    def seeker_display_state(self):
        return dict(search='SRC', track='TRK', suspended='IR SUSPEND', memory='INS', lost='LOST')[self.status]

    def diagnostics(self):
        d = self.last_detection
        return dict(ir_state=self.status, ir_selected_identity=d.selected_identity,
            ir_visible_identities=list(d.visible_identities), ir_signal=d.signal,
            ir_rejected_fraction=d.rejected_fraction, ir_fov_half_deg=d.fov_half_deg,
            guidance_input_source=self._solution.source,
            guidance_input_sample_time_s=self._solution.sample_time_s,
            guidance_full_rate_rad_s=self._solution.full_rate_rad_s)

    def _aim(self, desired, forward, elapsed):
        c = self.config
        # Slew around current line; body-cone filtering below enforces gimbal.
        if self._boresight is None:
            self._boresight = forward
        angle = _angle(self._boresight, desired)
        step = math.radians(c.rate_deg_s)*elapsed
        if angle <= step or angle <= 1e-12:
            self._boresight = desired
        elif step > 0:
            if abs(math.sin(angle)) < 1e-10:
                # Opposite directions cannot define a unique great-circle slew.
                self._boresight = forward
            else:
                self._boresight = _unit(add(scale(self._boresight, math.sin(angle-step)/math.sin(angle)),
                                            scale(desired, math.sin(step)/math.sin(angle))))

    def update(self, time_s, dt_s, missile, target_truth):
        if not math.isfinite(time_s) or not math.isfinite(dt_s) or dt_s < 0:
            raise ValueError('IR time and step must be finite and step nonnegative')
        if self._time is not None and time_s < self._time:
            raise ValueError('IR observation times must be nondecreasing')
        if self._time == time_s:
            return self._solution
        elapsed = 0. if self._time is None else time_s-self._time
        self._time = time_s
        c = self.config
        forward = (math.cos(missile.pitch)*math.cos(missile.yaw), math.sin(missile.pitch),
                   math.cos(missile.pitch)*math.sin(missile.yaw))
        predicted = None
        if self._last_measurement is not None:
            old = self._last_measurement
            predicted = add(old.position, scale(old.velocity, time_s-old.sample_time_s))
        desired = forward if predicted is None else sub(predicted, missile.position)
        self._aim(_unit(desired) if norm(desired) > 1e-12 else forward, forward, elapsed)
        half = c.gated_half_deg if self._gated else c.fov_half_deg
        gimbal = c.gimbal_deg if self.has_tracked_once else c.acquire_gimbal_deg
        emission = c.target_emission(time_s, target_truth, missile)
        band_ranges = dict(c.range_bands_m)
        if isinstance(emission, Mapping):
            if not band_ranges:
                raise ValueError('Band-valued aircraft emission requires optical range_bands_m')
            if any(not math.isfinite(v) or v < 0 for v in emission.values()):
                raise ValueError('Aircraft IR band emission must be finite and nonnegative')
            emission = sum(v*band_ranges.get(k,0.)**2 for k,v in emission.items())
        else:
            emission = float(emission)
            if band_ranges:
                raise ValueError('Game-data band mode needs band-valued target emission; scalar overrides use the legacy configuration')
        if not math.isfinite(emission) or emission < 0:
            raise ValueError('Aircraft IR emission must be finite and nonnegative')
        sources = [('target', target_truth.position, target_truth.velocity, emission, False)]
        flares_at = getattr(self.target, 'flares_at', None)
        if flares_at is not None:
            flares = flares_at(time_s)
            identities = set()
            for f in flares:
                if not isinstance(f.identity, str) or not f.identity or f.identity == 'target' or f.identity in identities:
                    raise ValueError('IR flares need unique stable identities distinct from target')
                identities.add(f.identity)
                if f.release_time_s > time_s:
                    continue
                if not math.isfinite(f.emission_band2) or f.emission_band2 < 0:
                    raise ValueError('Flare IR emission must be finite and nonnegative')
                gain = c.flare_gain*(band_ranges.get(2,0.)**2 if band_ranges else 1.)
                sources.append((f.identity, f.position, f.velocity, gain*f.emission_band2, True))
        visible = []
        for identity, p, v, power, flare in sources:
            relative = sub(p, missile.position)
            distance = norm(relative)
            if distance <= 1e-12 or distance > c.range_max_m:
                continue
            n = scale(relative, 1/distance)
            signal = power/distance**2
            if (signal >= c.minimum_signal and _angle(n, forward) <= math.radians(gimbal)+1e-12
                    and _angle(n, self._boresight) <= math.radians(half)+1e-12):
                visible.append((identity, p, v, signal, flare))
        visible.sort(key=lambda x: x[0])
        total = sum(x[3] for x in visible)
        fraction = sum(x[3] for x in visible if x[4])/total if total else 0.
        rejecting = (c.reject_flare_fraction is not None and any(x[4] for x in visible)
                     and fraction >= c.reject_flare_fraction)
        if rejecting:
            if self._reject_since is None:
                self._reject_since = time_s
        else:
            self._reject_since = None
        suspended = (rejecting and self._last_measurement is not None
                     and time_s-self._reject_since+1e-12 >= c.rejection_reaction_s)
        reason = 'flare_rejection' if suspended else 'no_ir_source' if not visible else ''
        if visible and not suspended:
            strongest = min(visible, key=lambda x: (-x[3], x[0]))
            if c.signal_rule == 'strongest':
                p, v = strongest[1:3]
            else:
                p = tuple(sum(x[1][i]*x[3] for x in visible)/total for i in range(3))
                v = tuple(sum(x[2][i]*x[3] for x in visible)/total for i in range(3))
            self._solution = TrackSolution(p, v, time_s, time_s, TrackMode.INFRARED_TRACK, True,
                                          'ir_truth_assisted_'+c.signal_rule)
            self._last_measurement = self._solution
            self.status = 'track'
            self.has_tracked_once = True
            self._gated = c.gated_half_deg is not None
            selected = strongest[0]
        else:
            selected = None
            if self.status == 'track' and self.first_lock_loss_time_s is None:
                self.first_lock_loss_time_s = time_s
            old = self._last_measurement
            valid = old is not None and time_s-old.sample_time_s <= c.memory_s+1e-12
            if old is None:
                p, v, stamp = missile.position, (0.,0.,0.), time_s
            else:
                p, v, stamp = predicted, old.velocity, old.sample_time_s
            self.status = ('suspended' if suspended else 'memory') if valid else ('lost' if old is not None else 'search')
            self._solution = TrackSolution(p, v, stamp, time_s, TrackMode.INERTIAL if valid else TrackMode.LOST,
                                          valid, 'ir_constant_velocity_memory' if valid else 'ir_no_solution')
            if not valid and c.gate_on_loss == 'restore_search':
                self._gated = False
                self._boresight = forward
        self.observation_reject_reason = reason
        self.last_detection = InfraredDetection(not bool(reason), reason, selected,
            tuple(x[0] for x in visible), total, fraction, half)
        return self._solution


__all__ = ['InfraredSeekerConfig', 'InfraredTrackProvider', 'InfraredDetection']
