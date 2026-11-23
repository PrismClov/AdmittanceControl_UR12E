import unittest

import numpy as np
from scipy.spatial.transform import Rotation as R

from filter_module import RotationLowPassFilter


class RotationLowPassFilterTest(unittest.TestCase):
    def test_does_not_interpolate_through_rotation_wrap(self):
        initial = np.array([0.0, 3.13, 0.0])
        current = np.array([0.0, -3.13, 0.0])
        low_pass = RotationLowPassFilter(alpha=0.35, initial_value=initial)
        filtered = R.from_rotvec(low_pass.update(current))
        start = R.from_rotvec(initial)
        desired = R.from_rotvec(current)

        self.assertLess((filtered * start.inv()).magnitude(), 0.02)
        self.assertLess((filtered * desired.inv()).magnitude(), 0.02)

    def test_alpha_one_tracks_equivalent_rotation(self):
        low_pass = RotationLowPassFilter(
            alpha=1.0,
            initial_value=np.array([0.0, 3.1, 0.0]),
        )
        result = R.from_rotvec(low_pass.update([0.0, -3.1, 0.0]))
        desired = R.from_rotvec([0.0, -3.1, 0.0])
        self.assertAlmostEqual((result * desired.inv()).magnitude(), 0.0)

    def test_large_rotation_delta_uses_shortest_path(self):
        initial = np.array([0.0, 0.0, 3.5])
        current = np.array([0.0, 0.0, -3.5])
        low_pass = RotationLowPassFilter(alpha=0.5, initial_value=initial)
        filtered = R.from_rotvec(low_pass.update(current))
        start = R.from_rotvec(initial)

        # +3.5 and -3.5 rad are 0.717 rad apart after wrapping to +/-pi.
        shortest_delta = abs(2.0 * np.pi - 7.0)
        self.assertAlmostEqual(
            (filtered * start.inv()).magnitude(),
            0.5 * shortest_delta,
        )


if __name__ == "__main__":
    unittest.main()
