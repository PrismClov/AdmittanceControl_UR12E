"""Measurement-only TCP motion and ServoP timing diagnostics."""

from __future__ import annotations

from collections import deque
import csv
import time
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
from scipy.spatial.transform import Rotation


def _vector6(value: Sequence[float], name: str) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.shape != (6,) or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain six finite values")
    return array


class TcpVelocityEstimator:
    """Estimate TCP velocity and acceleration from feedback poses.

    Angular velocity uses the SO(3) relative rotation
    ``R_current @ R_previous.T`` so it is not affected by Euler-angle jumps.
    """

    def __init__(self):
        self.reset(np.zeros(6), 0.0)

    def reset(self, pose: Sequence[float], timestamp_s: float) -> None:
        self._pose = _vector6(pose, "pose")
        self._timestamp = float(timestamp_s)
        self._velocity = np.zeros(6)
        self._acceleration = np.zeros(6)
        self._initialized = True

    def update(
        self,
        pose: Sequence[float],
        timestamp_s: float,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        current = _vector6(pose, "pose")
        timestamp_s = float(timestamp_s)
        elapsed = timestamp_s - self._timestamp
        if not np.isfinite(elapsed) or elapsed <= 0.0:
            raise ValueError("TCP sample timestamps must increase")
        if elapsed > 1.0:
            raise ValueError("TCP sample interval exceeds one second")

        velocity = np.empty(6)
        velocity[:3] = (current[:3] - self._pose[:3]) / elapsed
        current_rotation = Rotation.from_rotvec(current[3:])
        previous_rotation = Rotation.from_rotvec(self._pose[3:])
        velocity[3:] = (
            current_rotation * previous_rotation.inv()
        ).as_rotvec() / elapsed
        acceleration = (velocity - self._velocity) / elapsed

        self._pose = current
        self._timestamp = timestamp_s
        self._velocity = velocity
        self._acceleration = acceleration
        return velocity.copy(), acceleration.copy(), elapsed


class ServoReplyTracker:
    """Correlate ServoP sends with reply latency using FIFO order."""

    def __init__(self):
        self._pending_sends = deque()
        self.last_send_at = None
        self.last_reply_at = None
        self.last_latency_s = None
        self.latencies = []

    @property
    def pending_count(self) -> int:
        return len(self._pending_sends)

    def mark_sent(self, sent_at: float | None = None) -> float:
        timestamp = time.monotonic() if sent_at is None else float(sent_at)
        self._pending_sends.append(timestamp)
        self.last_send_at = timestamp
        return timestamp

    def mark_reply(self, received_at: float | None = None) -> None:
        timestamp = (
            time.monotonic() if received_at is None else float(received_at)
        )
        if self._pending_sends:
            sent_at = self._pending_sends.popleft()
            latency = max(0.0, timestamp - sent_at)
            self.last_latency_s = latency
            self.latencies.append(latency)
        self.last_reply_at = timestamp


class AdmittanceDiagnosticsLogger:
    """Write named diagnostic rows to CSV."""

    def __init__(self, path: Path, fieldnames: Sequence[str]):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.path.open("w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(
            self._file,
            fieldnames=fieldnames,
            extrasaction="raise",
        )
        self._writer.writeheader()
        self._rows_since_flush = 0

    def write(self, row: Mapping[str, object]) -> None:
        self._writer.writerow(row)
        self._rows_since_flush += 1
        if self._rows_since_flush >= 10:
            self._file.flush()
            self._rows_since_flush = 0

    def close(self) -> None:
        if not self._file.closed:
            self._file.flush()
            self._file.close()
