"""Explicit unit conversions used at the model boundaries."""

from __future__ import annotations

import math


G0_MPS2 = 9.80665


def kmh_to_mps(value_kmh: float) -> float:
    return float(value_kmh) / 3.6


def mps_to_kmh(value_mps: float) -> float:
    return float(value_mps) * 3.6


def deg_to_rad(value_deg: float) -> float:
    return math.radians(float(value_deg))


def rad_to_deg(value_rad: float) -> float:
    return math.degrees(float(value_rad))


def g_to_mps2(value_g: float, gravity_mps2: float = G0_MPS2) -> float:
    return float(value_g) * gravity_mps2


def mps2_to_g(value_mps2: float, gravity_mps2: float = G0_MPS2) -> float:
    return float(value_mps2) / gravity_mps2

