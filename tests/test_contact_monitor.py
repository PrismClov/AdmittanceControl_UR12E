import unittest

import numpy as np

from contact_monitor import ContactForceController


class ContactForceControllerTest(unittest.TestCase):
    def controller(self, release_after_s=0.0):
        return ContactForceController(
            detect_force_n=5.0,
            target_force_n=5.0,
            kp=0.001,
            ki=0.0001,
            deadband_n=0.25,
            max_velocity_m_s=0.01,
            dt=0.02,
            confirm_time_s=0.10,
            max_direction_angle_deg=20.0,
            release_after_s=release_after_s,
            release_time_s=0.10,
        )

    def confirm_contact(self, controller):
        velocity = np.zeros(6)
        for _ in range(5):
            state, _, _ = controller.update(
                [0.0, 0.0, 8.0, 0.0, 0.0, 0.0],
                velocity,
            )
        return state

    def test_requires_stable_contact_before_force_mode(self):
        controller = self.controller()
        state, velocity, force = controller.update(
            [0.0, 0.0, 8.0, 0.0, 0.0, 0.0],
            [0.01, 0.02, -0.01, 0.0, 0.0, 0.0],
        )

        self.assertEqual(state, "confirming")
        self.assertAlmostEqual(force, 8.0)
        self.assertEqual(velocity[2], 0.0)
        self.assertEqual(velocity[0], 0.01)
        self.assertEqual(velocity[1], 0.02)

    def test_retreats_along_normal_when_force_is_too_high(self):
        controller = self.controller()
        self.assertEqual(self.confirm_contact(controller), "contact")
        state, velocity, normal_force = controller.update(
            [0.0, 0.0, 8.0, 0.0, 0.0, 0.0],
            np.zeros(6),
        )

        self.assertEqual(state, "contact")
        self.assertAlmostEqual(normal_force, 8.0)
        self.assertGreater(velocity[2], 0.0)

    def test_contact_mode_preserves_tangential_motion(self):
        controller = self.controller()
        self.confirm_contact(controller)
        _, velocity, _ = controller.update(
            [0.0, 0.0, 5.0, 0.0, 0.0, 0.0],
            [0.03, -0.02, -0.04, 0.0, 0.0, 0.0],
        )

        self.assertAlmostEqual(velocity[0], 0.03)
        self.assertAlmostEqual(velocity[1], -0.02)
        self.assertAlmostEqual(velocity[2], 0.0)

    def test_deadband_holds_normal_position(self):
        controller = self.controller()
        self.confirm_contact(controller)
        _, velocity, _ = controller.update(
            [0.0, 0.0, 5.1, 0.0, 0.0, 0.0],
            np.zeros(6),
        )

        np.testing.assert_allclose(velocity, np.zeros(6))

    def test_contact_confirmation_resets_on_direction_change(self):
        controller = self.controller()
        controller.update([0.0, 0.0, 8.0, 0.0, 0.0, 0.0], np.zeros(6))
        for _ in range(3):
            controller.update([0.0, 0.0, 8.0, 0.0, 0.0, 0.0], np.zeros(6))
        state, _, _ = controller.update(
            [8.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            np.zeros(6),
        )

        self.assertEqual(state, "confirming")
        self.assertEqual(controller.confirmation_time_s, 0.0)

    def test_controlled_release_reaches_hold_state(self):
        controller = self.controller(release_after_s=0.10)
        self.confirm_contact(controller)
        controller.update([0.0, 0.0, 5.0, 0.0, 0.0, 0.0], np.zeros(6))
        states = []
        for _ in range(20):
            state, _, _ = controller.update(
                [0.0, 0.0, 0.1, 0.0, 0.0, 0.0],
                np.zeros(6),
            )
            states.append(state)
        self.assertIn("release", states)
        self.assertEqual(states[-1], "released")


if __name__ == "__main__":
    unittest.main()
