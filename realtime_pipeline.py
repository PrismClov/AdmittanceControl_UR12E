"""Decoupled robot feedback and ServoP communication threads."""

from __future__ import annotations

from collections import deque
import re
import socket
import threading
import time

import numpy as np


def parse_error_id(message):
    values = re.findall(r"-?\d+", str(message))
    if not values:
        raise RuntimeError(f"无法解析 CR5 回复: {message!r}")
    return int(values[0])


class LatestRobotState:
    """Continuously keep the newest 30004 packet without software downsampling."""

    def __init__(self, feedback, output_hz=50.0):
        if output_hz <= 0.0:
            raise ValueError("output_hz must be positive")
        self._feedback = feedback
        self._output_period = 1.0 / float(output_hz)
        self._next_publish = time.monotonic()
        self._condition = threading.Condition()
        self._raw_packet = None
        self._raw_packet_at = None
        self._packet = None
        self._sequence = 0
        self._received_at = None
        self._published_at = None
        self._published_packet_at = None
        self._receive_intervals = deque(maxlen=512)
        self._published_intervals = deque(maxlen=512)
        self._error = None
        self._stop = threading.Event()
        self._reader_thread = threading.Thread(
            target=self._run,
            name="cr5-feedback-reader",
            daemon=True,
        )
        self._publisher_thread = threading.Thread(
            target=self._publish_loop,
            name="cr5-feedback-publisher",
            daemon=True,
        )

    def start(self):
        self._reader_thread.start()
        self._publisher_thread.start()

    def stop(self):
        self._stop.set()
        with self._condition:
            self._condition.notify_all()
        self._publisher_thread.join(timeout=1.0)
        self._reader_thread.join(timeout=1.0)

    def _run(self):
        while not self._stop.is_set():
            try:
                packet = self._feedback.feedBackData()
                if packet is None:
                    raise RuntimeError("CR5 feedback packet is empty")
            except Exception as exc:
                with self._condition:
                    self._error = exc
                    self._condition.notify_all()
                self._stop.wait(0.05)
                continue
            received_at = time.monotonic()
            with self._condition:
                if self._received_at is not None:
                    interval = received_at - self._received_at
                    if 0.001 <= interval <= 0.1:
                        self._receive_intervals.append(interval)
                self._received_at = received_at
                self._raw_packet = packet
                self._raw_packet_at = received_at
                self._error = None
                self._condition.notify_all()

    def _publish_loop(self):
        next_publish = time.monotonic()
        while not self._stop.is_set():
            delay = next_publish - time.monotonic()
            if delay > 0.0:
                self._stop.wait(delay)
                continue
            with self._condition:
                packet = self._raw_packet
                packet_at = self._raw_packet_at
                if packet is not None:
                    self._packet = packet
                    self._sequence += 1
                    self._published_packet_at = packet_at
                    if self._published_at is not None:
                        published_interval = time.monotonic() - self._published_at
                        if 0.001 <= published_interval <= 0.1:
                            self._published_intervals.append(published_interval)
                    self._published_at = time.monotonic()
                    self._condition.notify_all()
            next_publish += self._output_period

    def wait_for_data(self, timeout=1.0):
        with self._condition:
            self._condition.wait_for(
                lambda: self._packet is not None
                or self._error is not None
                or self._stop.is_set(),
                timeout=timeout,
            )
            if self._error is not None:
                raise RuntimeError(f"CR5 feedback error: {self._error}")
            if self._packet is None:
                raise TimeoutError("timed out waiting for CR5 feedback")
            return self._packet, self._sequence, self._published_packet_at

    def wait_for_next(self, after_sequence, timeout=0.1):
        with self._condition:
            self._condition.wait_for(
                lambda: self._sequence > after_sequence
                or self._error is not None
                or self._stop.is_set(),
                timeout=timeout,
            )
            if self._error is not None:
                raise RuntimeError(f"CR5 feedback error: {self._error}")
            if self._sequence <= after_sequence:
                raise TimeoutError("timed out waiting for next feedback sample")
            return self._packet, self._sequence, self._published_packet_at

    def snapshot(self):
        with self._condition:
            if self._error is not None:
                raise RuntimeError(f"CR5 feedback error: {self._error}")
            if self._packet is None:
                raise TimeoutError("CR5 feedback is not available")
            return self._packet, self._sequence, self._published_packet_at

    def period_stats(self):
        with self._condition:
            intervals = np.asarray(self._receive_intervals, dtype=float)
        if intervals.size == 0:
            raise RuntimeError("not enough feedback timing samples")
        median = float(np.median(intervals))
        return {
            "samples": int(intervals.size),
            "mean_s": float(intervals.mean()),
            "median_s": median,
            "p95_s": float(np.percentile(intervals, 95)),
            "max_s": float(intervals.max()),
            "median_hz": 1.0 / median,
        }

    def published_period_stats(self):
        with self._condition:
            intervals = np.asarray(self._published_intervals, dtype=float)
        if intervals.size == 0:
            raise RuntimeError("not enough published feedback samples")
        median = float(np.median(intervals))
        return {
            "samples": int(intervals.size),
            "mean_s": float(intervals.mean()),
            "median_s": median,
            "p95_s": float(np.percentile(intervals, 95)),
            "max_s": float(intervals.max()),
            "median_hz": 1.0 / median,
        }


class ServoPStream:
    """Independent fixed-rate ServoP sender and reply reader."""

    def __init__(self, sock, period_s=0.03, aheadtime=50.0, gain=500.0):
        if period_s <= 0.0:
            raise ValueError("period_s must be positive")
        self._sock = sock
        self._period = float(period_s)
        self._aheadtime = float(aheadtime)
        self._gain = float(gain)

        self._condition = threading.Condition()
        self._target = None
        self._target_sequence = 0
        self._last_sent_sequence = 0
        self._pending_sends = deque()
        self._reply_buffer = ""
        self._send_lock = threading.Lock()
        self._sender_stop = threading.Event()
        self._receiver_stop = threading.Event()
        self._fault = None
        self.send_intervals = []
        self.reply_latencies = []
        self.sent_count = 0
        self.unknown_reply_count = 0

        self._sender = threading.Thread(
            target=self._send_loop,
            name="cr5-servop-sender",
            daemon=True,
        )
        self._receiver = threading.Thread(
            target=self._receive_loop,
            name="cr5-servop-replies",
            daemon=True,
        )

    @staticmethod
    def command(values, dt, aheadtime, gain):
        return (
            "ServoP("
            f"{values[0]:.6f},{values[1]:.6f},{values[2]:.6f},"
            f"{values[3]:.6f},{values[4]:.6f},{values[5]:.6f},"
            f"t={dt:.6f},aheadtime={aheadtime:.6f},gain={gain:.6f})"
        )

    def start(self):
        self._sender.start()
        self._receiver.start()

    def update_target(self, values):
        values = tuple(float(value) for value in values)
        if len(values) != 6:
            raise ValueError("ServoP target must contain six values")
        with self._condition:
            self._target = values
            self._target_sequence += 1
            self._condition.notify_all()

    def raise_if_fault(self):
        with self._condition:
            if self._fault is not None:
                raise RuntimeError(f"ServoP stream fault: {self._fault}")

    def _send_loop(self):
        next_send = time.monotonic()
        previous_sent_at = None
        last_sequence = -1
        while not self._sender_stop.is_set():
            now = time.monotonic()
            if now < next_send:
                self._sender_stop.wait(next_send - now)
                continue
            with self._condition:
                if self._fault is not None:
                    return
                values = self._target
                sequence = self._target_sequence
            if values is None:
                next_send += self._period
                continue

            sent_at = time.monotonic()
            message = self.command(
                values,
                self._period,
                self._aheadtime,
                self._gain,
            )
            try:
                with self._send_lock:
                    self._sock.sendall(message.encode("utf-8"))
            except OSError as exc:
                with self._condition:
                    self._fault = f"ServoP send failed: {exc}"
                return
            self._pending_sends.append(sent_at)
            self.sent_count += 1
            if previous_sent_at is not None:
                self.send_intervals.append(sent_at - previous_sent_at)
            previous_sent_at = sent_at
            last_sequence = sequence
            self._last_sent_sequence = sequence

            next_send += self._period
            if next_send < time.monotonic() - self._period:
                next_send = time.monotonic()

    def _receive_loop(self):
        old_timeout = self._sock.gettimeout()
        try:
            self._sock.settimeout(0.05)
            while not self._receiver_stop.is_set():
                try:
                    chunk = self._sock.recv(4096)
                except TimeoutError:
                    continue
                except OSError as exc:
                    if not self._receiver_stop.is_set():
                        with self._condition:
                            self._fault = f"ServoP reply failed: {exc}"
                    return
                if not chunk:
                    if not self._receiver_stop.is_set():
                        with self._condition:
                            self._fault = "ServoP dashboard disconnected"
                    return
                received_at = time.monotonic()
                self._reply_buffer += chunk.decode(errors="replace")
                messages = self._reply_buffer.split(";")
                self._reply_buffer = messages.pop()
                for message in messages:
                    if not message.strip():
                        continue
                    if not self._pending_sends:
                        self.unknown_reply_count += 1
                        continue
                    sent_at = self._pending_sends.popleft()
                    self.reply_latencies.append(received_at - sent_at)
                    error = parse_error_id(message)
                    if error != 0:
                        with self._condition:
                            self._fault = (
                                f"ServoP failed with error {error}: {message!r}"
                            )
                        return
        finally:
            self._sock.settimeout(old_timeout)

    def stop(self):
        self._sender_stop.set()
        with self._condition:
            self._condition.notify_all()
        self._sender.join(timeout=1.0)
        try:
            with self._send_lock:
                self._sock.sendall(b"Stop()")
            time.sleep(0.2)
        finally:
            self._receiver_stop.set()
            self._receiver.join(timeout=1.0)
