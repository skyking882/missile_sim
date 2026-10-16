"""Radar pointing in a rolled body basis; mechanical limits and beam are separate."""
from __future__ import annotations

import math

from .math3d import add, clamp, dot, normalize, scale


def full_cone_half_angle_rad(full_angle_deg):
    """Convert the profile's full circular antenna width to an off-axis radius."""
    return math.radians(full_angle_deg) / 2.0


def limited_antenna_direction(axes, direction, yaw_limits, pitch_limits):
    """Clip body yaw/pitch independently, then reconstruct the world ray."""
    x, y, z = (dot(direction, a) for a in (axes.forward, axes.up, axes.right))
    yaw = clamp(math.atan2(z, x), *yaw_limits)
    pitch = clamp(math.atan2(y, math.hypot(x, z)), *pitch_limits)
    cp = math.cos(pitch)
    return add(add(scale(axes.forward, cp * math.cos(yaw)), scale(axes.up, math.sin(pitch))),
               scale(axes.right, cp * math.sin(yaw)))


def within_circular_beam(direction, center, half_angle):
    """Actual three-dimensional separation, including the boundary."""
    return dot(normalize(direction), normalize(center)) >= math.cos(half_angle) - 1e-12


def validate_launcher_radar_geometry(value):
    if value is None:
        return None
    keys = {'yaw_limits_deg', 'pitch_limits_deg', 'beam_half_angle_deg', 'orientation_quaternion_xyzw'}
    if not isinstance(value, dict) or set(value) != keys:
        raise ValueError('launcher_radar_geometry requires yaw_limits_deg, pitch_limits_deg, beam_half_angle_deg and orientation_quaternion_xyzw')
    result = {}
    for key, bound in [('yaw_limits_deg', 180.), ('pitch_limits_deg', 90.)]:
        limits = value[key]
        if not isinstance(limits, (list, tuple)) or len(limits) != 2 or any(
            isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) or abs(x) > bound for x in limits) or limits[0] > limits[1]:
            raise ValueError(key + ' must be ordered finite degree limits')
        result[key] = tuple(float(x) for x in limits)
    half = value['beam_half_angle_deg']
    if isinstance(half, bool) or not isinstance(half, (int, float)) or not math.isfinite(half) or not 0 < half < 90:
        raise ValueError('launcher beam_half_angle_deg must be between 0 and 90')
    q = value['orientation_quaternion_xyzw']
    if not isinstance(q, (list, tuple)) or len(q) != 4 or any(
        isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) for x in q):
        raise ValueError('launcher orientation_quaternion_xyzw must be a finite quaternion')
    magnitude = math.sqrt(sum(x*x for x in q))
    if magnitude <= 1e-12:
        raise ValueError('launcher orientation quaternion must be nonzero')
    result.update(beam_half_angle_deg=float(half), orientation_quaternion_xyzw=tuple(x/magnitude for x in q))
    return result
