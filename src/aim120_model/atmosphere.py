"""A dependency-free standard-atmosphere approximation."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class AtmosphereSample:
    altitude_m: float
    temperature_k: float
    pressure_pa: float
    density_kg_m3: float
    speed_of_sound_mps: float


class StandardAtmosphere:
    """ISA troposphere plus an isothermal lower-stratosphere continuation."""

    _T0 = 288.15
    _P0 = 101325.0
    _R = 287.05287
    _GAMMA = 1.4
    _L = 0.0065
    _G = 9.80665
    _T11 = 216.65
    _H11 = 11000.0

    def sample(self, altitude_m: float) -> AtmosphereSample:
        h = max(0.0, float(altitude_m))
        if h <= self._H11:
            temperature = self._T0 - self._L * h
            pressure = self._P0 * (temperature / self._T0) ** (self._G / (self._R * self._L))
        else:
            temperature = self._T11
            p11 = self._P0 * (self._T11 / self._T0) ** (self._G / (self._R * self._L))
            pressure = p11 * math.exp(-self._G * (h - self._H11) / (self._R * temperature))
        density = pressure / (self._R * temperature)
        sound = math.sqrt(self._GAMMA * self._R * temperature)
        return AtmosphereSample(h, temperature, pressure, density, sound)

