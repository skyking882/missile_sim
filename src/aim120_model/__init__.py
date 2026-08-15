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
from .tracking import AlphaBetaGate, TrackMode, TrackSolution
from .trajectory import TabulatedTargetModel, TabulatedTrajectory, TrajectoryPoint
from .public_api import simulate

__version__ = "1.0.0"

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
    "TabulatedTargetModel",
    "TabulatedTrajectory",
    "TrajectoryPoint",
    "__version__",
    "load_cases",
    "load_model_config",
    "simulate",
]
