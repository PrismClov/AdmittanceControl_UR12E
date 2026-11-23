"""ROS 2 service adapter with a single-in-flight ServoP/ServoJ pipeline."""

from __future__ import annotations

import json
import threading
import time
from typing import Sequence

import numpy as np


def feed_info_to_packet(data):
    """Map the ROS 2 FeedInfo JSON names to the original controller packet."""

    return {
        "ActualTCPForce": [data["i_control"]],
        "SixForceValue": [data["SixForceValue"]],
        "ToolVectorActual": [data["tool_vector_actual"]],
        "TCPSpeedActual": [data["TCP_speed_actual"]],
        "QActual": [data["q_actual"]],
        "QDActual": [data["qd_actual"]],
        "RobotMode": [int(data["robot_mode"])],
        "EnableStatus": [int(data["EnableStatus"])],
        "ErrorStatus": [int(data["ErrorStatus"])],
        "CollisionState": [int(data["CollisionStates"])],
        "TimeStamp": [int(data["controller_timer"])],
    }


class Ros2FeedbackReader:
    """Expose Dobot FeedInfo as the packet interface used by the controller."""

    def __init__(self, topic="/dobot_bringup_ros2/msg/FeedInfo"):
        import rclpy
        from std_msgs.msg import String

        self._rclpy = rclpy
        if not rclpy.ok():
            rclpy.init(args=None)
        self.node = rclpy.create_node("admittance_feedback_ros2")
        self._condition = threading.Condition()
        self._packet = None
        self._sequence = 0
        self._consumed = 0
        self._closed = False
        self.node.create_subscription(
            String,
            topic,
            self._on_feed_info,
            20,
        )
        self._executor = rclpy.executors.SingleThreadedExecutor()
        self._executor.add_node(self.node)
        self._thread = threading.Thread(
            target=self._executor.spin,
            name="dobot-ros2-feedback",
            daemon=True,
        )
        self._thread.start()

    def _on_feed_info(self, message) -> None:
        data = json.loads(message.data)
        packet = feed_info_to_packet(data)
        with self._condition:
            self._packet = packet
            self._sequence += 1
            self._condition.notify_all()

    def feedBackData(self):
        with self._condition:
            self._condition.wait_for(
                lambda: self._closed or self._sequence > self._consumed,
                timeout=1.0,
            )
            if self._sequence <= self._consumed:
                return None
            self._consumed = self._sequence
            return self._packet

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        with self._condition:
            self._condition.notify_all()
        self._executor.shutdown(timeout_sec=1.0)
        self._thread.join(timeout=1.0)
        self.node.destroy_node()


class Ros2ServoController:
    """Send ServoP or ServoJ with one in-flight ROS 2 service request."""

    def __init__(self, service_prefix="/dobot_bringup_ros2/srv"):
        import rclpy
        from dobot_msgs_v4.srv import (
            ClearError,
            EnableRobot,
            RobotMode,
            ServoJ,
            ServoP,
            SixForceHome,
            Stop,
        )

        self._rclpy = rclpy
        self._types = {
            "clear_error": ClearError,
            "enable": EnableRobot,
            "robot_mode": RobotMode,
            "servo_j": ServoJ,
            "servo_p": ServoP,
            "six_force_home": SixForceHome,
            "stop": Stop,
        }
        if not rclpy.ok():
            rclpy.init(args=None)
        self.node = rclpy.create_node("admittance_control_ros2")
        self._clients = {
            key: self.node.create_client(cls, f"{service_prefix}/{name}")
            for key, cls, name in (
                ("clear_error", ClearError, "ClearError"),
                ("enable", EnableRobot, "EnableRobot"),
                ("robot_mode", RobotMode, "RobotMode"),
                ("servo_j", ServoJ, "ServoJ"),
                ("servo_p", ServoP, "ServoP"),
                ("six_force_home", SixForceHome, "SixForceHome"),
                ("stop", Stop, "Stop"),
            )
        }

        self._lock = threading.RLock()
        self._latest_target = None
        self._active_stage = None
        self._active_future = None
        self._fault = None
        self._closed = False
        self.started_at = None
        self.last_send_at = None
        self.last_reply_at = None
        self.last_latency_s = None
        self.sent_count = 0
        self.coalesced_count = 0
        self.transient_timeout_count = 0
        self._consecutive_timeout_count = 0
        self._executor = rclpy.executors.SingleThreadedExecutor()
        self._executor.add_node(self.node)
        self._executor_thread = threading.Thread(
            target=self._executor.spin,
            name="dobot-ros2-servo",
            daemon=True,
        )
        self._executor_thread.start()

    @property
    def busy(self) -> bool:
        with self._lock:
            return self._active_stage is not None

    @property
    def pending_count(self) -> int:
        return 1 if self.busy else 0

    @property
    def fault(self):
        with self._lock:
            return self._fault

    def wait_until_ready(self, timeout=5.0) -> None:
        deadline = time.monotonic() + timeout
        unavailable = []
        while time.monotonic() < deadline:
            unavailable = [
                name
                for name, client in self._clients.items()
                if not client.service_is_ready()
            ]
            if not unavailable:
                return
            time.sleep(0.02)
        raise RuntimeError(
            "ROS 2 robot services unavailable: " + ", ".join(unavailable)
        )

    def spin_once(self, timeout_s=0.0) -> None:
        # Service callbacks run in the dedicated executor thread.
        if timeout_s > 0.0:
            time.sleep(timeout_s)

    def call(self, key, request=None, timeout=0.5):
        if request is None:
            request = self._types[key].Request()
        future = self._clients[key].call_async(request)
        deadline = time.monotonic() + timeout
        while not future.done() and time.monotonic() < deadline:
            time.sleep(0.002)
        if not future.done():
            future.cancel()
            raise TimeoutError(f"ROS 2 service timeout: {key}")
        return future.result()

    def submit_joints(self, joint_degrees, dt) -> bool:
        """Keep only the newest joint target and send when the link is idle."""

        with self._lock:
            if self._fault is not None or self._closed:
                return False
            self._latest_target = (
                "joint",
                np.asarray(joint_degrees, dtype=float).copy(),
                float(dt),
            )
            if self._active_stage is not None:
                self.coalesced_count += 1
                return False
            self._start_latest()
            return True

    def submit_pose(self, pose_mm_degrees, dt) -> bool:
        """Keep only the newest Cartesian target for ROS 2 ServoP."""

        with self._lock:
            if self._fault is not None or self._closed:
                return False
            self._latest_target = (
                "pose",
                np.asarray(pose_mm_degrees, dtype=float).copy(),
                float(dt),
            )
            if self._active_stage is not None:
                self.coalesced_count += 1
                return False
            self._start_latest()
            return True

    def prime_pose(self, pose_mm_degrees, dt=0.02, attempts=3) -> None:
        """Warm up the ServoP dashboard connection while holding the pose."""

        values = np.asarray(pose_mm_degrees, dtype=float)
        if values.shape != (6,) or not np.all(np.isfinite(values)):
            raise ValueError("invalid ServoP warm-up target")
        last_result = None
        for _ in range(attempts):
            request = self._types["servo_p"].Request()
            request.a, request.b, request.c = values[:3]
            request.d, request.e, request.f = values[3:]
            request.param_value = [
                f"t={dt:.6f}",
                "aheadtime=50.000000",
                "gain=500.000000",
            ]
            response = self.call("servo_p", request, timeout=2.0)
            last_result = int(response.res)
            if last_result == 0:
                return
            if last_result != -1000:
                raise RuntimeError(f"ROS 2 ServoP warm-up failed: {last_result}")
        raise RuntimeError(
            f"ROS 2 ServoP warm-up did not recover: {last_result}"
        )

    def _start_latest(self) -> None:
        if (
            self._active_stage is not None
            or self._fault is not None
            or not self._latest_target
        ):
            return
        target_kind, values, dt = self._latest_target
        self._latest_target = None
        if values.shape != (6,) or not np.all(np.isfinite(values)):
            self._set_fault(RuntimeError("invalid six-axis servo target"))
            return

        client_key = "servo_j" if target_kind == "joint" else "servo_p"
        request = self._types[client_key].Request()
        request.a, request.b, request.c = values[:3]
        request.d, request.e, request.f = values[3:]
        request.param_value = [
            f"t={dt:.6f}",
            "aheadtime=50.000000",
            "gain=500.000000",
        ]
        self._active_stage = client_key
        self.started_at = time.monotonic()
        self.last_send_at = self.started_at
        self._active_future = self._clients[client_key].call_async(request)
        self._active_future.add_done_callback(self._on_servo)

    def _on_servo(self, future) -> None:
        if self._closed:
            return
        try:
            with self._lock:
                if self._active_stage not in ("servo_j", "servo_p"):
                    return
            response = future.result()
            result = int(response.res)
            if result == -1000 and self._active_stage == "servo_p":
                self.transient_timeout_count += 1
                self._consecutive_timeout_count += 1
                if self._consecutive_timeout_count >= 3:
                    raise RuntimeError(
                        "ROS 2 ServoP failed with three consecutive -1000 replies"
                    )
            elif result != 0:
                raise RuntimeError(
                    f"ROS 2 {self._active_stage} failed: {result}"
                )
            else:
                self._consecutive_timeout_count = 0
            with self._lock:
                self.last_reply_at = time.monotonic()
                if self.last_send_at is not None:
                    self.last_latency_s = self.last_reply_at - self.last_send_at
                self.sent_count += 1
                self._active_stage = None
                self._active_future = None
                self._start_latest()
        except Exception as exc:
            with self._lock:
                self._set_fault(exc)

    def _set_fault(self, exc) -> None:
        self._fault = str(exc)
        self._active_stage = None
        self._active_future = None
        self._latest_target = None

    def prepare_stop(self, timeout=1.0) -> bool:
        """Drop queued targets and let the active service request finish."""

        with self._lock:
            self._latest_target = None
        deadline = time.monotonic() + timeout
        while self.busy and time.monotonic() < deadline:
            time.sleep(0.02)
        return not self.busy

    def raise_if_fault(self) -> None:
        with self._lock:
            if self._fault is not None:
                raise RuntimeError(f"ROS 2 servo fault: {self._fault}")

    def close(self) -> None:
        if self._closed:
            return
        with self._lock:
            self._closed = True
            if self._active_future is not None:
                self._active_future.cancel()
        self._executor.shutdown(timeout_sec=1.0)
        self._executor_thread.join(timeout=1.0)
        self.node.destroy_node()
        if self._rclpy.ok():
            self._rclpy.shutdown()
