import numpy as np
from scipy.spatial.transform import Rotation as R


def expand_tcp_motion_limits(values):
    """Return six-axis velocity and acceleration limits.

    Four values are expanded uniformly:
    linear velocity, angular velocity, linear acceleration, angular acceleration.
    Twelve values are used as velocity[0:6] followed by acceleration[0:6].
    """
    values = np.asarray(values, dtype=float)
    if not np.all(np.isfinite(values)):
        raise ValueError("motion limit values must be finite")
    if values.shape == (4,):
        velocity = np.repeat(values[:2], 3)
        acceleration = np.repeat(values[2:], 3)
    elif values.shape == (12,):
        velocity = values[:6]
        acceleration = values[6:]
    else:
        raise ValueError("expected four or twelve motion limit values")
    if np.any(velocity < 0.0) or np.any(acceleration < 0.0):
        raise ValueError("motion limit values must be non-negative")
    return velocity.copy(), acceleration.copy()


class CartesianVelocityAccelerationLimiter:
    """Limit six-axis TCP velocity and its acceleration at the target layer."""

    def __init__(self, max_velocity, max_acceleration, dt):
        self.max_velocity = self._six_axis_array(
            max_velocity,
            "max_velocity",
        )
        self.max_acceleration = self._six_axis_array(
            max_acceleration,
            "max_acceleration",
        )
        if any(value < 0.0 for value in self.max_velocity):
            raise ValueError("max_velocity values must be non-negative")
        if any(value < 0.0 for value in self.max_acceleration):
            raise ValueError("max_acceleration values must be non-negative")
        self.dt = float(dt)
        if not np.isfinite(self.dt) or self.dt <= 0.0:
            raise ValueError("dt must be positive")
        self.last_velocity = np.zeros(6)

    @staticmethod
    def _six_axis_array(values, name):
        array = np.asarray(values, dtype=float)
        if array.shape != (6,):
            raise ValueError(f"{name} must contain six values")
        if not np.all(np.isfinite(array)):
            raise ValueError(f"{name} values must be finite")
        return array

    def update(self, desired_velocity):
        desired = self._six_axis_array(desired_velocity, "desired_velocity")
        limited = np.clip(desired, -self.max_velocity, self.max_velocity)
        max_delta = self.max_acceleration * self.dt
        limited = self.last_velocity + np.clip(
            limited - self.last_velocity,
            -max_delta,
            max_delta,
        )
        self.last_velocity = limited.copy()
        return limited


def integrate_velocity(pose, velocity, dt):
    """Integrate a six-axis velocity using the controller's rotation convention."""
    pose = np.asarray(pose, dtype=float)
    velocity = np.asarray(velocity, dtype=float)
    if pose.shape != (6,) or velocity.shape != (6,):
        raise ValueError("pose and velocity must contain six values")
    target = np.empty(6)
    target[:3] = pose[:3] + velocity[:3] * dt
    current_rotation = R.from_rotvec(pose[3:])
    rotation_delta = R.from_rotvec(velocity[3:] * dt)
    target[3:] = (rotation_delta * current_rotation).as_rotvec()
    return target
