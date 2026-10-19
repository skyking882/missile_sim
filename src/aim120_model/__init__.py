"""Auditable standalone AIM-120A-like local model."""

from .config import load_cases, load_model_config
from .effective_controller import EffectiveControllerEnvelope
from .h2_simulator import H2Simulator
from .observation import (
    IdealTruthTrackProvider,
    InertialTrackPropagator,
    KinematicTrackProvider,
    SensorTrackProvider,
    SnsFixedPointProvider,
)
from .radar_seeker import RadarDetection, RadarSeekerObserver, SeekerState
from .simulator import H1Simulator
from .tracking import AlphaBetaGate, TrackMode, TrackSolution, TrackIdentity, ObserverStageSnapshot, DatalinkObservation
from .angular_guidance import AngularObservation, RecordedSeekerSample, RecordedSeekerHistory
from .trajectory import TabulatedTargetModel, TabulatedTrajectory, TrajectoryPoint
from .public_api import simulate
from .infrared_seeker import InfraredSeekerConfig, InfraredTrackProvider
from .flare import FlareSpec, FlareRelease, FlareState, FlaringTarget, flaring_factory
from .aircraft_infrared import AircraftEngineState, AircraftInfraredSignature
from .infrared_data import infrared_coverage, load_infrared_missile_profile

__version__ = "1.1.0"

__all__ = [
    "EffectiveControllerEnvelope",
    "H1Simulator",
    "H2Simulator",
    "IdealTruthTrackProvider",
    "InertialTrackPropagator",
    "KinematicTrackProvider",
    "SensorTrackProvider",
    "SnsFixedPointProvider",
    "RadarDetection",
    "RadarSeekerObserver",
    "SeekerState",
    "AlphaBetaGate",
    "TrackMode",
    "TrackSolution",
    "TrackIdentity",
    "ObserverStageSnapshot",
    "DatalinkObservation",
    "AngularObservation",
    "RecordedSeekerSample",
    "RecordedSeekerHistory",
    "TabulatedTargetModel",
    "TabulatedTrajectory",
    "TrajectoryPoint",
    "__version__",
    "load_cases",
    "load_model_config",
    "simulate",
    "InfraredSeekerConfig",
    "InfraredTrackProvider",
    "FlareSpec",
    "FlareRelease",
    "FlareState",
    "FlaringTarget",
    "flaring_factory",
    "AircraftEngineState",
    "AircraftInfraredSignature",
    "infrared_coverage",
    "load_infrared_missile_profile",
]
