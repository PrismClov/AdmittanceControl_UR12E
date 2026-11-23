import unittest

import numpy as np

from contact_monitor import ContactForceController


class ContactForceControllerTest(unittest.TestCase):
    def controller(self):
        return ContactForceController(
            detect_force_n=5.0,
            target_force_n=5.0,
            kp=0.001,
            ki=0.0001,
            deadband_n=0.25,
            max_velocity_m_s=0.01,
            dt=0.02,
        )

    def test_retreats_along_normal_when_force_is_too_high(self):
        controller = self.controller()
        active, velocity, normal_force = controller.update(
            [0.0, 0.0, 8.0, 0.0, 0.0, 0.0]
        )

        self.assertTrue(active)
        self.assertAlmostEqual(normal_force, 8.0)
        self.assertGreater(velocity[2], 0.0)
        self.assertEqual(velocity[0], 0.0)
        self.assertEqual(velocity[1], 0.0)

    def test_advances_when_force_is_too_low(self):
        controller = self.controller()
        controller.update([0.0, 0.0, 8.0, 0.0, 0.0, 0.0])
        _, velocity, _ = controller.update(
            [0.0, 0.0, 3.0, 0.0, 0.0, 0.0]
        )

        self.assertLess(velocity[2], 0.0)

    def test_deadband_holds_position(self):
        controller = self.controller()
        controller.update([0.0, 0.0, 8.0, 0.0, 0.0, 0.0])
        _, velocity, _ = controller.update(
            [0.0, 0.0, 5.1, 0.0, 0.0, 0.0]
        )

        np.testing.assert_allclose(velocity, np.zeros(6))


if __name__ == "__main__":
    unittest.main()
