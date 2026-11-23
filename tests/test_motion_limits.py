import unittest

import numpy as np
from scipy.spatial.transform import Rotation as R

from motion_limits import (
    CartesianVelocityAccelerationLimiter,
    expand_tcp_motion_limits,
    integrate_velocity,
)


class CartesianMotionLimiterTest(unittest.TestCase):
    def test_expands_unified_limits(self):
        velocity, acceleration = expand_tcp_motion_limits(
            [0.15, 0.5, 0.8, 1.5]
        )

        np.testing.assert_allclose(velocity, [0.15, 0.15, 0.15, 0.5, 0.5, 0.5])
        np.testing.assert_allclose(acceleration, [0.8, 0.8, 0.8, 1.5, 1.5, 1.5])

    def test_accepts_six_axis_limits(self):
        velocity, acceleration = expand_tcp_motion_limits(
            [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2]
        )

        np.testing.assert_allclose(velocity, [0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
        np.testing.assert_allclose(acceleration, [0.7, 0.8, 0.9, 1.0, 1.1, 1.2])

    def test_clamps_velocity_before_acceleration(self):
        limiter = CartesianVelocityAccelerationLimiter(
            max_velocity=[0.1, 0.1, 0.1, 0.6, 0.6, 0.6],
            max_acceleration=[100.0] * 6,
            dt=0.02,
        )
        limited = limiter.update([0.5, -0.5, 0.0, 0.7, -0.7, 0.0])

        np.testing.assert_allclose(
            limited,
            [0.1, -0.1, 0.0, 0.6, -0.6, 0.0],
        )

    def test_limits_velocity_change_by_acceleration_times_dt(self):
        limiter = CartesianVelocityAccelerationLimiter(
            max_velocity=[1.0] * 6,
            max_acceleration=[0.8] * 6,
            dt=0.02,
        )

        first = limiter.update([1.0] * 6)
        second = limiter.update([1.0] * 6)
        third = limiter.update([0.0] * 6)

        np.testing.assert_allclose(first, [0.016] * 6)
        np.testing.assert_allclose(second, [0.032] * 6)
        np.testing.assert_allclose(third, [0.016] * 6)

    def test_integrates_rotation_with_shortest_path(self):
        pose = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 3.5])
        velocity = np.array([0.1, 0.0, 0.0, 0.0, 0.0, -1.0])

        target = integrate_velocity(pose, velocity, 0.02)
        rotation_delta = (
            R.from_rotvec(target[3:]) * R.from_rotvec(pose[3:]).inv()
        )

        self.assertAlmostEqual(target[0], 0.002)
        self.assertAlmostEqual(rotation_delta.magnitude(), 0.02)


if __name__ == "__main__":
    unittest.main()
