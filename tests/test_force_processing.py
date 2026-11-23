import unittest

import numpy as np

from force_processing import continuous_deadband


class ContinuousDeadbandTest(unittest.TestCase):
    def test_is_continuous_at_deadband_boundary(self):
        values = np.array(
            [-3.0, -1.51, -1.5, 1.5, 1.51, 3.0]
        )
        result = continuous_deadband(values, [1.5] * 6)

        np.testing.assert_allclose(
            result,
            [-1.5, -0.01, 0.0, 0.0, 0.01, 1.5],
        )

    def test_applies_each_axis_deadband(self):
        result = continuous_deadband(
            [2.0, -2.0, 0.5, 0.1, -0.2, 0.3],
            [1.5, 1.5, 1.5, 0.015, 0.05, 0.3],
        )

        np.testing.assert_allclose(
            result,
            [0.5, -0.5, 0.0, 0.085, -0.15, 0.0],
        )

    def test_rejects_negative_deadband(self):
        with self.assertRaises(ValueError):
            continuous_deadband(np.zeros(6), [1.0, 1.0, 1.0, 1.0, 1.0, -1.0])


if __name__ == "__main__":
    unittest.main()
