import tempfile
import unittest
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from tcp_diagnostics import (
    AdmittanceDiagnosticsLogger,
    ServoReplyTracker,
    TcpVelocityEstimator,
)


class TcpVelocityEstimatorTest(unittest.TestCase):
    def test_translation_velocity_and_acceleration(self):
        estimator = TcpVelocityEstimator()
        estimator.reset(np.zeros(6), 10.0)
        velocity, acceleration, elapsed = estimator.update(
            [0.01, 0, 0, 0, 0, 0],
            10.02,
        )
        self.assertAlmostEqual(elapsed, 0.02)
        self.assertAlmostEqual(velocity[0], 0.5)
        self.assertAlmostEqual(acceleration[0], 25.0)

    def test_rotation_velocity_uses_so3_relative_rotation(self):
        estimator = TcpVelocityEstimator()
        start = np.zeros(6)
        end = np.zeros(6)
        end[3:] = Rotation.from_euler(
            "z", 0.2
        ).as_rotvec()
        estimator.reset(start, 1.0)
        velocity, _, _ = estimator.update(end, 1.1)
        self.assertAlmostEqual(velocity[5], 2.0)
        self.assertAlmostEqual(np.linalg.norm(velocity[:5]), 0.0)


class ServoReplyTrackerTest(unittest.TestCase):
    def test_reply_latency_is_correlated_in_send_order(self):
        tracker = ServoReplyTracker()
        tracker.mark_sent(10.0)
        tracker.mark_sent(10.02)
        tracker.mark_reply(10.03)
        self.assertAlmostEqual(tracker.last_latency_s, 0.03)
        self.assertEqual(tracker.pending_count, 1)
        tracker.mark_reply(10.04)
        self.assertAlmostEqual(tracker.last_latency_s, 0.02)
        self.assertEqual(tracker.pending_count, 0)


class DiagnosticsLoggerTest(unittest.TestCase):
    def test_writes_header_and_row(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "diagnostics.csv"
            logger = AdmittanceDiagnosticsLogger(path, ("time", "value"))
            logger.write({"time": 1.0, "value": 2.0})
            logger.close()
            lines = path.read_text(encoding="utf-8").splitlines()
        self.assertEqual(lines, ["time,value", "1.0,2.0"])


if __name__ == "__main__":
    unittest.main()
