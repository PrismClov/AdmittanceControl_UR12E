import unittest
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from local_kinematics import Cr5LocalKinematics, pose_to_transform


URDF = Path(
    "/home/orin/dobot_ws/src/DOBOT_6Axis_ROS2_V4/"
    "dobot_rviz/urdf/cr5_robot.urdf"
)


class LocalKinematicsTest(unittest.TestCase):
    def setUp(self):
        self.joints = np.array(
            [154.1123, -13.3028, 99.1081, 15.4655, -75.6996, 54.9409]
        )
        self.pose = np.array(
            [
                0.388087,
                0.002060,
                0.453670,
                *Rotation.from_euler(
                    "xyz",
                    [162.1019, -1.2528, -171.7833],
                    degrees=True,
                ).as_rotvec(),
            ]
        )
        self.robot = Cr5LocalKinematics(
            URDF,
            self.joints,
            self.pose,
        )

    def test_calibrated_initial_tcp_matches_feedback(self):
        np.testing.assert_allclose(
            self.robot.tcp_fk(np.deg2rad(self.joints)),
            pose_to_transform(self.pose),
            atol=1e-9,
        )

    def test_stationary_target_keeps_joint_target(self):
        target = self.robot.solve_tcp_pose(
            self.pose,
            self.joints,
            0.02,
        )
        np.testing.assert_allclose(target, self.joints, atol=1e-8)

    def test_small_translation_improves_pose_error(self):
        target = self.pose.copy()
        target[0] += 0.001
        q_target = self.robot.solve_tcp_pose(target, self.joints, 0.02)
        initial_error = abs(
            self.robot.tcp_fk(np.deg2rad(self.joints))[0, 3] - target[0]
        )
        next_error = abs(
            self.robot.tcp_fk(np.deg2rad(q_target))[0, 3] - target[0]
        )
        self.assertLess(next_error, initial_error)


if __name__ == "__main__":
    unittest.main()
