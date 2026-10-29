"""Scene inputs and sensor outputs shared by the three seeker branches.

Scene truth generates incident signals; only SensorMeasurement enters guidance.
Coordinates are X forward, Y up, Z right; quaternions are XYZW.
"""
from dataclasses import dataclass, field


@dataclass(frozen=True)
class BodyState:
    position: tuple
    velocity: tuple
    orientation: tuple = (0., 0., 0., 1.)
    present: bool = True
    identity: str = 'target'

    @property
    def p(self): return self.position

    @property
    def v(self): return self.velocity

    @property
    def q(self): return self.orientation


@dataclass(frozen=True)
class IlluminatorState:
    position: tuple
    velocity: tuple
    direction: tuple
    illuminating: bool
    target_identity: str = 'target'
    waveform: str = 'pd'
    sample_time_s: float = 0.


@dataclass(frozen=True)
class SignalSource:
    identity: str
    position: tuple
    velocity: tuple
    emission: dict = field(default_factory=dict)
    rcs_m2: float = 0.
    present: bool = True


@dataclass(frozen=True)
class SceneSnapshot:
    target: BodyState
    illuminator: IlluminatorState | None = None
    target_ir: dict = field(default_factory=dict)
    flares: tuple = ()
    chaff: tuple = ()
    target_rcs_m2: float | None = None


@dataclass
class SensorMeasurement:
    direction: tuple
    angular_rate: tuple
    valid: bool
    mode: str
    source: str
    range_m: float | None = None
    closing_mps: float | None = None
    range_valid: bool = False
    doppler_valid: bool = False
    fresh: bool = False
    held: bool = False
    metadata: dict = field(default_factory=dict)
