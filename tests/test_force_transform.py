import unittest

import numpy as np
from scipy.spatial.transform import Rotation as R

from force_transform import wrench_tool_to_user


class WrenchToolToUserTest(unittest.TestCase):
    def test_identity_rotation_preserves_wrench(self):
        wrench = [1.0, -2.0, 3.0, 0.1, -0.2, 0.3]
        np.testing.assert_allclose(
            wrench_tool_to_user(wrench, [0.0, 0.0, 0.0]),
            wrench,
        )

    def test_rotates_force_and_torque_about_z(self):
        result = wrench_tool_to_user(
            [1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 90.0],
        )
        np.testing.assert_allclose(
            result,
            [0.0, 1.0, 0.0, -1.0, 0.0, 0.0],
            atol=1e-12,
        )

    def test_round_trip_world_direction_at_changed_pose(self):
        rotation = R.from_euler(
            "xyz",
            [-158.174, 3.350, 63.682],
            degrees=True,
        )
        tool_wrench = np.r_[rotation.inv().apply([1.0, 0.0, 0.0]), 0.0, 0.0, 0.0]
        result = wrench_tool_to_user(
            tool_wrench,
            [-158.174, 3.350, 63.682],
        )
        np.testing.assert_allclose(
            result[:3],
            [1.0, 0.0, 0.0],
            atol=1e-9,
        )


if __name__ == "__main__":
    unittest.main()
