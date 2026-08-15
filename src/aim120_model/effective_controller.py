"""Plan 7 Lite effective mapping from ``aCmdYaw`` to backend ``currentG``.

StatShark reports ``currentG`` as a nonnegative magnitude.  The local plant
uses a signed effective actuator input, so command sign is applied only at
that interface boundary.  Keeping both outputs explicit prevents a magnitude
array from being mistaken for a signed backend state.
"""

from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class EffectiveControllerEnvelope:
    """Static command gain with an independently supplied authority envelope."""

    gain: float
    authority_fraction: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.gain) or self.gain <= 0.0:
            raise ValueError("gain must be positive and finite")
        if not math.isfinite(self.authority_fraction) or self.authority_fraction <= 0.0:
            raise ValueError("authority_fraction must be positive and finite")

    def predict_current_g_magnitude(self, a_cmd_yaw_g: float, fins_lat_accel_g: float) -> float:
        """Predict the nonnegative StatShark ``currentG`` magnitude."""

        command = float(a_cmd_yaw_g)
        authority = float(fins_lat_accel_g)
        if not math.isfinite(command):
            raise ValueError("a_cmd_yaw_g must be finite")
        if not math.isfinite(authority) or authority < 0.0:
            raise ValueError("fins_lat_accel_g must be nonnegative and finite")
        return min(self.gain * abs(command), self.authority_fraction * authority)

    def predict_signed_effective_output(self, a_cmd_yaw_g: float, fins_lat_accel_g: float) -> float:
        """Predict the signed effective input consumed by the local plant."""

        command = float(a_cmd_yaw_g)
        magnitude = self.predict_current_g_magnitude(command, fins_lat_accel_g)
        if command > 0.0:
            return magnitude
        if command < 0.0:
            return -magnitude
        return 0.0

    def effective_cap_g(self, fins_lat_accel_g: float) -> float:
        """Return the predicted ``currentG`` magnitude at full envelope."""

        authority = float(fins_lat_accel_g)
        if not math.isfinite(authority) or authority < 0.0:
            raise ValueError("fins_lat_accel_g must be nonnegative and finite")
        return self.authority_fraction * authority

    def envelope_switch_command_g(self, fins_lat_accel_g: float) -> float:
        """Return ``|aCmdYaw|`` where the command branch reaches the cap."""

        authority = float(fins_lat_accel_g)
        if not math.isfinite(authority) or authority < 0.0:
            raise ValueError("fins_lat_accel_g must be nonnegative and finite")
        return self.authority_fraction * authority / self.gain
