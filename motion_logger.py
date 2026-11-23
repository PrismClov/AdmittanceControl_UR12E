"""CSV logging for admittance and robot-state diagnostics."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Mapping


class MotionCsvLogger:
    def __init__(self, path, fieldnames):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.path.open("w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(
            self._file,
            fieldnames=fieldnames,
            extrasaction="raise",
        )
        self._writer.writeheader()
        self._rows = 0

    def write(self, row: Mapping[str, object]):
        self._writer.writerow(row)
        self._rows += 1
        if self._rows >= 10:
            self._file.flush()
            self._rows = 0

    def close(self):
        if not self._file.closed:
            self._file.flush()
            self._file.close()
