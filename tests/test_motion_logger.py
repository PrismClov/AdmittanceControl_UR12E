import tempfile
import unittest
from pathlib import Path

from motion_logger import MotionCsvLogger


class MotionCsvLoggerTest(unittest.TestCase):
    def test_writes_header_and_rows(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "motion.csv"
            logger = MotionCsvLogger(path, ("time", "force"))
            logger.write({"time": 1.0, "force": 2.0})
            logger.close()
            self.assertEqual(
                path.read_text(encoding="utf-8").splitlines(),
                ["time,force", "1.0,2.0"],
            )


if __name__ == "__main__":
    unittest.main()
