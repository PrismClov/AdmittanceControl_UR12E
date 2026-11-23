import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np

from analyze_motion import analyze_file, dominant_frequency


class AnalyzeMotionTest(unittest.TestCase):
    def test_detects_sine_frequency(self):
        sample_hz = 50.0
        time = np.arange(500) / sample_hz
        values = np.sin(2.0 * np.pi * 4.0 * time)
        self.assertAlmostEqual(
            dominant_frequency(values, sample_hz),
            4.0,
            delta=0.25,
        )

    def test_analyzes_current_csv_schema(self):
        fields = [
            "time_s",
            "feedback_sequence",
            "feedback_age_ms",
            "control_dt_s",
            "servo_sent",
            "servo_pending",
            "servo_rtt_ms",
        ]
        for axis in ("x", "y", "z", "rx", "ry", "rz"):
            fields.extend(
                [
                    f"F_raw_{axis}",
                    f"F_filtered_{axis}",
                    f"F_ext_{axis}",
                    f"v_actual_raw_{axis}",
                    f"v_control_{axis}",
                    f"a_actual_{axis}",
                    f"jerk_actual_{axis}",
                    f"a_admittance_raw_{axis}",
                    f"x_target_{axis}",
                    f"x_actual_{axis}",
                    f"x_target_minus_actual_{axis}",
                    f"x_servo_{axis}",
                    f"x_servo_minus_actual_{axis}",
                ]
            )
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "motion.csv"
            with path.open("w", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fields)
                writer.writeheader()
                for index in range(100):
                    row = {field: 0.0 for field in fields}
                    row["time_s"] = index * 0.02
                    row["F_ext_x"] = np.sin(2 * np.pi * 3 * index * 0.02)
                    writer.writerow(row)
            result = analyze_file(path, 5.0, 2.0)
        self.assertEqual(result["rows"], 100)
        self.assertAlmostEqual(result["sample_hz"], 50.0)
        self.assertAlmostEqual(
            result["axes"]["x"]["force_dominant_hz"],
            3.0,
            delta=0.25,
        )


if __name__ == "__main__":
    unittest.main()
