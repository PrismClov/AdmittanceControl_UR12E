import numpy as np


class ContactHold:
    """Latch target motion after external force exceeds a threshold."""

    def __init__(self, force_threshold_n):
        self.force_threshold_n = float(force_threshold_n)
        if not np.isfinite(self.force_threshold_n) or self.force_threshold_n <= 0.0:
            raise ValueError("force threshold must be positive")
        self.active = False
        self.detected_force_n = 0.0

    def update(self, wrench):
        wrench = np.asarray(wrench, dtype=float)
        if wrench.shape != (6,) or not np.all(np.isfinite(wrench)):
            raise ValueError("wrench must contain six finite values")
        force_norm = float(np.linalg.norm(wrench[:3]))
        if force_norm >= self.force_threshold_n:
            self.active = True
            self.detected_force_n = force_norm
        return self.active
