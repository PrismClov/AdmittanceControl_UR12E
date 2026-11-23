import numpy as np


def continuous_deadband(values, deadband):
    """Apply a continuous deadband while preserving the signal sign."""
    values = np.asarray(values, dtype=float)
    deadband = np.asarray(deadband, dtype=float)
    if values.shape != (6,) or deadband.shape != (6,):
        raise ValueError("values and deadband must contain six values")
    if not np.all(np.isfinite(values)) or not np.all(np.isfinite(deadband)):
        raise ValueError("values and deadband must be finite")
    if np.any(deadband < 0.0):
        raise ValueError("deadband values must be non-negative")
    return np.sign(values) * np.maximum(np.abs(values) - deadband, 0.0)
