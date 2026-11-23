import unittest

import numpy as np

from servo_j import (
    joint_target_error,
    parse_inverse_kinematics,
)


class ServoJHelperTest(unittest.TestCase):
    def test_parse_uses_final_inverse_kinematics_reply(self):
        response = (
            "0,{1,2,3,4,5,6},InverseKin(...);"
            "0,{10.1,-20.2,30.3,-40.4,50.5,-60.6},InverseKin(...);"
        )
        joints = parse_inverse_kinematics(response)
        np.testing.assert_allclose(
            joints,
            [10.1, -20.2, 30.3, -40.4, 50.5, -60.6],
        )

    def test_joint_error_uses_shortest_wrapped_difference(self):
        error = joint_target_error(
            [179, -179, 0, 350, 10, 190],
            [-179, 179, 0, 10, 350, -170],
        )
        np.testing.assert_allclose(error, [-2, 2, 0, -20, 20, 0])

    def test_error_reply_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "error 5"):
            parse_inverse_kinematics(
                "5,{},InverseKin(100,200,300,0,0,0);"
            )

    def test_parses_ros2_braced_joint_reply(self):
        joints = parse_inverse_kinematics(
            "{1.1,2.2,3.3,-4.4,5.5,-6.6}"
        )
        np.testing.assert_allclose(
            joints,
            [1.1, 2.2, 3.3, -4.4, 5.5, -6.6],
        )

if __name__ == "__main__":
    unittest.main()
