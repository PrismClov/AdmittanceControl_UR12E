import unittest
from types import SimpleNamespace
from pathlib import Path

from roundtrip_admittance_test import (
    DEFAULT_POSE_A,
    DEFAULT_POSE_B,
    build_command,
    pose_argument,
)


class RoundTripAdmittanceTest(unittest.TestCase):
    def test_formats_pose_as_controller_argument(self):
        self.assertEqual(
            pose_argument(DEFAULT_POSE_A),
            "419.893000,113.137000,307.892000,"
            "-174.658000,6.521000,62.576000",
        )

    def test_retract_disables_contact_control(self):
        args = SimpleNamespace(
            robot_ip="192.168.5.5",
            retract_duration=12.0,
            contact_duration=20.0,
            feedback_hz=50.0,
            dt=0.02,
            contact_force=5.0,
            contact_confirm_time=0.1,
            contact_direction_angle=20.0,
            contact_kp=0.001,
            contact_ki=0.0001,
            contact_deadband=0.25,
            contact_max_velocity=0.01,
            contact_release_after=0.0,
        )
        command = build_command(args, DEFAULT_POSE_A, False, Path("a.csv"))

        self.assertIn("--target-pose", command)
        self.assertIn(pose_argument(DEFAULT_POSE_A), command)
        self.assertEqual(command[command.index("--contact-stop-force") + 1], "0")
        self.assertNotIn("--contact-release-after", command)

    def test_contact_leg_targets_pose_b_and_never_releases_by_default(self):
        args = SimpleNamespace(
            robot_ip="192.168.5.5",
            retract_duration=12.0,
            contact_duration=20.0,
            feedback_hz=50.0,
            dt=0.02,
            contact_force=5.0,
            contact_confirm_time=0.1,
            contact_direction_angle=20.0,
            contact_kp=0.001,
            contact_ki=0.0001,
            contact_deadband=0.25,
            contact_max_velocity=0.01,
            contact_release_after=0.0,
        )
        command = build_command(args, DEFAULT_POSE_B, True, Path("b.csv"))

        self.assertIn(pose_argument(DEFAULT_POSE_B), command)
        self.assertEqual(
            command[command.index("--contact-target-force") + 1],
            "5.0",
        )
        self.assertEqual(
            command[command.index("--contact-release-after") + 1],
            "0.0",
        )


if __name__ == "__main__":
    unittest.main()
