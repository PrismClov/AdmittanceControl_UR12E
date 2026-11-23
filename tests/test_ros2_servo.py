import unittest

from ros2_servo import feed_info_to_packet


class Ros2ServoTest(unittest.TestCase):
    def test_maps_feed_info_force_and_state(self):
        source = {
            "i_control": [1, 2, 3, 4, 5, 6],
            "SixForceValue": [10, 20, 30, 40, 50, 60],
            "tool_vector_actual": [100, 200, 300, 10, 20, 30],
            "TCP_speed_actual": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            "q_actual": [1, 2, 3, 4, 5, 6],
            "qd_actual": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            "robot_mode": 5,
            "EnableStatus": 1,
            "ErrorStatus": 0,
            "CollisionStates": 0,
            "controller_timer": 123456,
        }
        packet = feed_info_to_packet(source)
        self.assertEqual(packet["ActualTCPForce"], [[1, 2, 3, 4, 5, 6]])
        self.assertEqual(
            packet["ToolVectorActual"],
            [[100, 200, 300, 10, 20, 30]],
        )
        self.assertEqual(packet["RobotMode"], [5])
        self.assertEqual(packet["TimeStamp"], [123456])


if __name__ == "__main__":
    unittest.main()
