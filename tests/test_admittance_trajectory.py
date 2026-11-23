import unittest

import numpy as np

from admittance_trajectory import AdmittanceTrajectory, TrajectoryLimits


def make_controller():
    limits = TrajectoryLimits(
        max_velocity=(0.05, 0.05, 0.05, 0.2, 0.2, 0.2),
        max_acceleration=(0.5, 0.5, 0.5, 1.5, 1.5, 1.5),
        max_jerk=(10.0, 10.0, 10.0, 30.0, 30.0, 30.0),
        max_tracking_translation=0.02,
        max_tracking_rotation=0.10,
    )
    return AdmittanceTrajectory(
        mass=(0.8, 0.8, 0.8, 0.01, 0.01, 0.01),
        damping=(16, 16, 16, 0.12, 0.12, 0.16),
        stiffness=(80, 80, 80, 1.1, 1.1, 2),
        deadband=(1.5, 1.5, 1.5, 0.015, 0.05, 0.3),
        limits=limits,
    )


class AdmittanceTrajectoryTest(unittest.TestCase):
    def test_closed_loop_force_reaches_static_equilibrium(self):
        controller = make_controller()
        pose = np.zeros(6)
        controller.reset(pose)
        output = None
        for _ in range(3000):
            output = controller.step(pose, [8, 0, 0, 0, 0, 0], 0.005)
            pose = output.target_pose
        self.assertIsNotNone(output)
        self.assertAlmostEqual(pose[0], 0.1, delta=0.002)
        self.assertLess(np.linalg.norm(output.velocity[:3]), 0.002)

    def test_tracking_error_freezes_command(self):
        controller = make_controller()
        controller.reset(np.zeros(6))
        current = np.array([0.05, 0, 0, 0, 0, 0])
        output = controller.step(
            current,
            np.zeros(6),
            0.005,
        )
        self.assertTrue(output.frozen)
        self.assertTrue(np.allclose(output.target_pose, current))

    def test_released_force_returns_to_rest_promptly(self):
        controller = make_controller()
        pose = np.zeros(6)
        controller.reset(pose)

        # Let the command reach a loaded equilibrium while feedback follows it.
        for _ in range(500):
            output = controller.step(pose, [8, 0, 0, 0, 0, 0], 0.02)
            pose = output.target_pose
        self.assertGreater(pose[0], 0.08)

        # Release the force and require both position and velocity to settle.
        settled_after = None
        for cycle in range(500):
            output = controller.step(pose, np.zeros(6), 0.02)
            pose = output.target_pose
            if (
                np.linalg.norm(pose[:3]) < 0.001
                and np.linalg.norm(output.velocity[:3]) < 0.005
            ):
                settled_after = (cycle + 1) * 0.02
                break

        self.assertIsNotNone(settled_after)
        self.assertLess(settled_after, 5.0)
        self.assertTrue(np.allclose(pose, 0.0, atol=0.001))


if __name__ == "__main__":
    unittest.main()
