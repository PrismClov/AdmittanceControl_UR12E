import argparse
import importlib.util
from pathlib import Path
import re
import threading
import time

import numpy as np
from scipy.spatial.transform import Rotation as R

import filter_module
from admittance_trajectory import AdmittanceTrajectory, TrajectoryLimits
from tcp_diagnostics import (
    AdmittanceDiagnosticsLogger,
    ServoReplyTracker,
    TcpVelocityEstimator,
)


parser = argparse.ArgumentParser(description="CR5 六维力导纳控制")
parser.add_argument("--robot-ip", default="192.168.5.5")
parser.add_argument(
    "--sdk",
    type=Path,
    default=Path(__file__).with_name("sdk") / "dobot_api.py",
)
parser.add_argument("--dt", type=float, default=0.02)
parser.add_argument(
    "--feedback-hz",
    type=float,
    default=50.0,
    help="software output rate for force samples",
)
parser.add_argument("--duration", type=float, default=0.0, help="0 表示持续运行")
parser.add_argument("--execute", action="store_true", help="默认只试运行，不发送 ServoP")
parser.add_argument("--zero-sensor", action="store_true")
parser.add_argument("--timing", action="store_true", help="打印控制周期统计")
parser.add_argument(
    "--log-csv",
    type=Path,
    help="将力、命令、实际位姿、速度和 ServoP 时序写入 CSV",
)
parser.add_argument(
    "--direct-tcp",
    action="store_true",
    help="固定周期直接发送 ServoP，不等待当前命令回包",
)
parser.add_argument(
    "--sync-feedback",
    action="store_true",
    help="每收到一帧六维力反馈执行一次控制，并将反馈 dt 用于导纳积分",
)
parser.add_argument("--max-linear-velocity", type=float, default=0.05)
parser.add_argument("--max-angular-velocity", type=float, default=0.2)
parser.add_argument("--max-linear-acceleration", type=float, default=0.5)
parser.add_argument("--max-angular-acceleration", type=float, default=1.5)
parser.add_argument("--max-linear-jerk", type=float, default=10.0)
parser.add_argument("--max-angular-jerk", type=float, default=30.0)
parser.add_argument("--max-tracking-translation", type=float, default=0.02)
parser.add_argument("--max-tracking-rotation", type=float, default=0.10)
args = parser.parse_args()
if args.dt <= 0.0:
    parser.error("--dt must be positive")
if args.feedback_hz <= 0.0:
    parser.error("--feedback-hz must be positive")
for name in (
    "max_linear_velocity",
    "max_angular_velocity",
    "max_linear_acceleration",
    "max_angular_acceleration",
    "max_linear_jerk",
    "max_angular_jerk",
    "max_tracking_translation",
    "max_tracking_rotation",
):
    if getattr(args, name) <= 0.0:
        parser.error(f"--{name.replace('_', '-')} must be positive")
if args.duration < 0.0:
    parser.error("--duration must be non-negative")
if not args.sdk.is_file():
    parser.error(f"Dobot SDK not found: {args.sdk}")


def load_sdk(path):
    spec = importlib.util.spec_from_file_location("dobot_api_v4", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def require_connected_socket(sock, host, port, name):
    try:
        peer = sock.getpeername()
    except (AttributeError, OSError) as exc:
        raise ConnectionError(
            f"{name} is unreachable at {host}:{port}; "
            "check controller power, Ethernet cable and IP address"
        ) from exc
    if peer[0] != host or peer[1] != port:
        raise ConnectionError(
            f"{name} connected to unexpected endpoint {peer}, expected {host}:{port}"
        )


def parse_error_id(response):
    if response is None or not str(response).strip():
        return 0
    values = re.findall(r"-?\d+", str(response))
    if not values:
        raise RuntimeError(f"无法解析 CR5 回复: {response!r}")
    return int(values[0])


def read_pose(packet):
    values = np.asarray(packet["ToolVectorActual"][0], dtype=float)
    pose = np.empty(6)
    pose[:3] = values[:3] / 1000.0
    pose[3:] = R.from_euler("xyz", values[3:], degrees=True).as_rotvec()
    return pose


def read_force(packet):
    # CR5 left/right compliance is the user-frame Y axis. Its measured sign
    # is opposite to the other axes after conversion to external wrench.
    wrench_sign = np.array([-1.0, 1.0, -1.0, -1.0, 1.0, -1.0])
    return wrench_sign * np.asarray(packet["ActualTCPForce"][0], dtype=float)


def pose_for_servop(pose, reference_rpy_degrees):
    pose = np.asarray(pose, dtype=float)
    values = np.empty(6)
    values[:3] = pose[:3] * 1000.0
    rotation = R.from_rotvec(pose[3:])
    canonical = rotation.as_euler("xyz", degrees=True)
    alternate = np.array(
        [canonical[0] + 180.0, 180.0 - canonical[1], canonical[2] + 180.0]
    )
    reference = np.asarray(reference_rpy_degrees, dtype=float)
    candidates = []
    for candidate in (canonical, alternate):
        candidate = candidate + 360.0 * np.round(
            (reference - candidate) / 360.0
        )
        error = (candidate - reference + 180.0) % 360.0 - 180.0
        candidates.append((float(np.max(np.abs(error))), candidate))
    values[3:] = min(candidates, key=lambda item: item[0])[1]
    return values


def servo_p_command(values, dt, aheadtime=50, gain=500):
    return (
        "ServoP("
        f"{values[0]:.6f},{values[1]:.6f},{values[2]:.6f},"
        f"{values[3]:.6f},{values[4]:.6f},{values[5]:.6f},"
        f"t={dt:.6f},aheadtime={aheadtime:.6f},gain={gain:.6f})"
    )


sdk = load_sdk(args.sdk)
rtde_c = sdk.DobotApiDashboard(args.robot_ip, 29999)
servo_c = rtde_c
rtde_recv = sdk.DobotApiFeedBack(args.robot_ip, 30004)
require_connected_socket(rtde_c.socket_dobot, args.robot_ip, 29999, "CR5 dashboard")
require_connected_socket(rtde_recv.socket_dobot, args.robot_ip, 30004, "CR5 feedback")


def get_packet():
    packet = rtde_recv.feedBackData()
    if packet is None:
        raise RuntimeError("CR5 反馈包缺失")
    return packet


class LatestFeedback:
    """Continuously drain force packets and expose only the newest one."""

    def __init__(self, feedback, feedback_hz):
        self._feedback = feedback
        self._period = 1.0 / feedback_hz
        self._condition = threading.Condition()
        self._packet = None
        self._sequence = 0
        self._received_at = None
        self._error = None
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._run, name="cr5-force-feedback", daemon=True
        )

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop.set()
        with self._condition:
            self._condition.notify_all()
        self._thread.join(timeout=1.0)

    def _run(self):
        next_publish = time.monotonic()
        pending_packet = None
        while not self._stop.is_set():
            try:
                packet = get_packet()
            except Exception as exc:
                with self._condition:
                    self._error = exc
                    self._condition.notify_all()
                self._stop.wait(0.05)
                continue
            pending_packet = packet
            now = time.monotonic()
            if now < next_publish:
                continue
            with self._condition:
                self._packet = pending_packet
                self._sequence += 1
                self._received_at = now
                self._error = None
                self._condition.notify_all()
            next_publish = max(next_publish + self._period, now)

    def get_latest(self, after_sequence=None, timeout=1.0):
        with self._condition:
            if after_sequence is not None:
                self._condition.wait_for(
                    lambda: self._sequence > after_sequence
                    or self._error is not None
                    or self._stop.is_set(),
                    timeout=timeout,
                )
            else:
                self._condition.wait_for(
                    lambda: self._packet is not None
                    or self._error is not None
                    or self._stop.is_set(),
                    timeout=timeout,
                )
            if self._error is not None:
                raise RuntimeError(f"CR5 feedback error: {self._error}")
            if self._packet is None or (
                after_sequence is not None and self._sequence <= after_sequence
            ):
                raise TimeoutError("timed out waiting for a new CR5 feedback packet")
            return (
                self._packet,
                self._sequence,
                self._received_at,
            )


direct_reply_buffer = ""
servo_reply_tracker = ServoReplyTracker()


def drain_direct_replies(raise_on_error=True):
    """Drain completed replies without waiting for the current ServoP."""

    global direct_reply_buffer
    if not args.direct_tcp:
        return
    sock = servo_c.socket_dobot
    old_timeout = sock.gettimeout()
    try:
        sock.setblocking(False)
        while True:
            try:
                chunk = sock.recv(4096)
            except BlockingIOError:
                break
            if not chunk:
                break
            direct_reply_buffer += chunk.decode(errors="replace")
    finally:
        sock.settimeout(old_timeout)

    messages = direct_reply_buffer.split(";")
    direct_reply_buffer = messages.pop()
    received_at = time.monotonic()
    for message in messages:
        if "ServoP(" in message:
            servo_reply_tracker.mark_reply(received_at)
        if raise_on_error:
            if message.strip() and parse_error_id(message) != 0:
                raise RuntimeError(f"Direct ServoP failed: {message!r}")


feedback_reader = LatestFeedback(rtde_recv, args.feedback_hz)
feedback_reader.start()

# 当前 TCP 位姿作为初始位姿，不额外发送初始运动。
packet, feedback_sequence, packet_received_at = feedback_reader.get_latest()
if args.execute:
    if int(packet["RobotMode"][0]) == 9:
        error_id = rtde_c.GetErrorID()
        raise RuntimeError(f"Robot is in error state: {error_id!r}")
    if int(packet["RobotMode"][0]) == 11:
        print("Clearing previous collision state")
        rtde_c.Stop()
        rtde_c.ClearError()
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            packet, feedback_sequence, packet_received_at = (
                feedback_reader.get_latest(after_sequence=feedback_sequence)
            )
            if int(packet["RobotMode"][0]) != 11:
                break
            time.sleep(0.05)
        else:
            raise RuntimeError("CR5 collision state could not be cleared")
    if int(packet["EnableStatus"][0]) != 1:
        response = rtde_c.EnableRobot()
        if parse_error_id(response) != 0:
            raise RuntimeError(f"EnableRobot failed: {response!r}")
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            packet, feedback_sequence, packet_received_at = (
                feedback_reader.get_latest(after_sequence=feedback_sequence)
            )
            if int(packet["EnableStatus"][0]) == 1:
                break
            time.sleep(0.05)
        else:
            raise RuntimeError("CR5 did not become enabled")

x_init = read_pose(packet)
print(x_init)

if args.zero_sensor:
    response = rtde_c.SixForceHome()
    if parse_error_id(response) != 0:
        raise RuntimeError(f"SixForceHome failed: {response!r}")
    time.sleep(1)

"""
传感器零漂补偿
由于传感器零点漂移较严重，所以要进行零漂补偿
滤波器的相关函数见filter_module.py
"""
comp_filter = filter_module.MoveMeanFilter(window_size=15)
for i in range(100):
    packet, feedback_sequence, packet_received_at = (
        feedback_reader.get_latest(after_sequence=feedback_sequence)
    )
    F_ext_comp = comp_filter.update(read_force(packet))

"""
六维力传感器信号滤波
由于传感器噪声较严重，所以要进行信号滤波
滤波器的相关函数见filter_module.py
"""
F_ext_filter = filter_module.RCLowPassFilter(alpha=0.35, initial_value=F_ext_comp)

# 设计导纳控制参数
M = [0.8, 0.8, 0.8, 0.01, 0.01, 0.01]
D = [16, 16, 16, 0.12, 0.12, 0.16]
K = [80, 80, 80, 1.1, 1.1, 2]
dt = args.dt
Deadband = [1.5, 1.5, 1.5, 0.015, 0.05, 0.3]
trajectory = AdmittanceTrajectory(
    M,
    D,
    K,
    Deadband,
    TrajectoryLimits(
        max_velocity=(
            args.max_linear_velocity,
            args.max_linear_velocity,
            args.max_linear_velocity,
            args.max_angular_velocity,
            args.max_angular_velocity,
            args.max_angular_velocity,
        ),
        max_acceleration=(
            args.max_linear_acceleration,
            args.max_linear_acceleration,
            args.max_linear_acceleration,
            args.max_angular_acceleration,
            args.max_angular_acceleration,
            args.max_angular_acceleration,
        ),
        max_jerk=(
            args.max_linear_jerk,
            args.max_linear_jerk,
            args.max_linear_jerk,
            args.max_angular_jerk,
            args.max_angular_jerk,
            args.max_angular_jerk,
        ),
        max_tracking_translation=args.max_tracking_translation,
        max_tracking_rotation=args.max_tracking_rotation,
    ),
)
trajectory.reset(x_init)

# 零漂更新只在小残差、低速和低角速度时进行。
BiasUpdateBeta = 0.001
BiasVelocityGate = [0.005, 0.005, 0.005, 0.02, 0.02, 0.02]
def ExternalForceComp(filtered_force, speed):
    residual = filtered_force - F_ext_comp
    motion_is_small = all(
        abs(speed[i]) < BiasVelocityGate[i] for i in range(6)
    )
    if not motion_is_small:
        return
    for i in range(6):
        if abs(residual[i]) < Deadband[i]:
            F_ext_comp[i] += BiasUpdateBeta * residual[i]


print("Start", "execute=" + str(args.execute))
started = time.monotonic()
next_tick = started
previous_loop_start = None
loop_intervals = []
feedback_ages = []
previous_feedback_timestamp = None
actual_velocity_estimator = TcpVelocityEstimator()
actual_pose_initialized = False
log_fieldnames = (
    "control_time_s",
    "feedback_time_s",
    "feedback_age_ms",
    "control_dt_s",
    "actual_dt_s",
    "F_raw_x_N",
    "F_filtered_x_N",
    "F_ext_x_N",
    "v_virtual_x_mps",
    "v_actual_x_mps",
    "a_actual_x_mps2",
    "x_cmd_m",
    "x_actual_m",
    "x_cmd_minus_actual_m",
    "F_raw_y_N",
    "F_filtered_y_N",
    "F_ext_y_N",
    "v_virtual_y_mps",
    "v_actual_y_mps",
    "a_actual_y_mps2",
    "y_cmd_m",
    "y_actual_m",
    "y_cmd_minus_actual_m",
    "F_raw_z_N",
    "F_filtered_z_N",
    "F_ext_z_N",
    "v_virtual_z_mps",
    "v_actual_z_mps",
    "a_actual_z_mps2",
    "z_cmd_m",
    "z_actual_m",
    "z_cmd_minus_actual_m",
    "F_raw_rx_Nm",
    "F_filtered_rx_Nm",
    "F_ext_rx_Nm",
    "v_virtual_rx_radps",
    "v_actual_rx_radps",
    "a_actual_rx_radps2",
    "rx_cmd_rad",
    "rx_actual_rad",
    "rx_cmd_minus_actual_rad",
    "F_raw_ry_Nm",
    "F_filtered_ry_Nm",
    "F_ext_ry_Nm",
    "v_virtual_ry_radps",
    "v_actual_ry_radps",
    "a_actual_ry_radps2",
    "ry_cmd_rad",
    "ry_actual_rad",
    "ry_cmd_minus_actual_rad",
    "F_raw_rz_Nm",
    "F_filtered_rz_Nm",
    "F_ext_rz_Nm",
    "v_virtual_rz_radps",
    "v_actual_rz_radps",
    "a_actual_rz_radps2",
    "rz_cmd_rad",
    "rz_actual_rad",
    "rz_cmd_minus_actual_rad",
    "frozen",
    "servo_sent",
    "servo_send_time_s",
    "servo_reply_time_s",
    "servo_reply_latency_ms",
    "servo_pending_replies",
)
diagnostics_logger = (
    AdmittanceDiagnosticsLogger(args.log_csv, log_fieldnames)
    if args.log_csv is not None
    else None
)
try:
    while args.duration == 0.0 or time.monotonic() - started < args.duration:
        loop_start = time.monotonic()
        if previous_loop_start is not None:
            loop_intervals.append(loop_start - previous_loop_start)
        previous_loop_start = loop_start
        packet, feedback_sequence, packet_received_at = (
            feedback_reader.get_latest(after_sequence=feedback_sequence)
        )
        feedback_ages.append(time.monotonic() - packet_received_at)
        control_dt = dt
        feedback_timestamp = float(packet["TimeStamp"][0]) * 0.001
        feedback_dt = 0.0
        if (
            previous_feedback_timestamp is not None
            and feedback_timestamp > previous_feedback_timestamp
        ):
            feedback_dt = feedback_timestamp - previous_feedback_timestamp
            if args.sync_feedback:
                control_dt = feedback_dt
        previous_feedback_timestamp = feedback_timestamp
        x = read_pose(packet)
        if not actual_pose_initialized:
            actual_velocity_estimator.reset(x, feedback_timestamp)
            actual_velocity, actual_acceleration = np.zeros(6), np.zeros(6)
            actual_dt = 0.0
            actual_pose_initialized = True
        else:
            actual_velocity, actual_acceleration, actual_dt = (
                actual_velocity_estimator.update(x, feedback_timestamp)
            )
        raw_force = read_force(packet)
        filtered_force = F_ext_filter.update(raw_force)
        F_ext = filtered_force - F_ext_comp
        ExternalForceComp(filtered_force, trajectory.velocity)
        trajectory_step = trajectory.step(x, F_ext, control_dt)
        x_c = trajectory_step.target_pose
        x_servo = x_c
        servo_sent = False
        servo_send_time = None

        if args.execute:
            servo_values = pose_for_servop(
                x_servo, packet["ToolVectorActual"][0][3:]
            )
            if args.direct_tcp:
                drain_direct_replies()
                servo_send_time = servo_reply_tracker.mark_sent()
                servo_c.send_data(servo_p_command(servo_values, control_dt))
                servo_sent = True
            else:
                servo_send_time = servo_reply_tracker.mark_sent()
                response = servo_c.ServoP(
                    *servo_values,
                    t=control_dt,
                    aheadtime=50,
                    gain=500,
                )
                servo_reply_tracker.mark_reply()
                servo_sent = True
                error = parse_error_id(response)
                if error != 0:
                    raise RuntimeError(
                        f"ServoP failed with error {error}: {response!r}"
                    )

        if diagnostics_logger is not None:
            commanded_minus_actual = np.zeros(6)
            commanded_minus_actual[:3] = x_c[:3] - x[:3]
            commanded_minus_actual[3:] = (
                R.from_rotvec(x_c[3:])
                * R.from_rotvec(x[3:]).inv()
            ).as_rotvec()
            row = {
                "control_time_s": loop_start - started,
                "feedback_time_s": feedback_timestamp,
                "feedback_age_ms": (
                    time.monotonic() - packet_received_at
                )
                * 1000.0,
                "control_dt_s": control_dt,
                "actual_dt_s": actual_dt,
                "frozen": int(trajectory_step.frozen),
                "servo_sent": int(servo_sent),
                "servo_send_time_s": (
                    ""
                    if servo_send_time is None
                    else servo_send_time - started
                ),
                "servo_reply_time_s": (
                    ""
                    if servo_reply_tracker.last_reply_at is None
                    else servo_reply_tracker.last_reply_at - started
                ),
                "servo_reply_latency_ms": (
                    ""
                    if servo_reply_tracker.last_latency_s is None
                    else servo_reply_tracker.last_latency_s * 1000.0
                ),
                "servo_pending_replies": servo_reply_tracker.pending_count,
            }
            for index, axis in enumerate(
                ("x", "y", "z", "rx", "ry", "rz")
            ):
                unit_force = "N" if index < 3 else "Nm"
                unit_velocity = "mps" if index < 3 else "radps"
                unit_acceleration = (
                    "mps2" if index < 3 else "radps2"
                )
                unit_position = "m" if index < 3 else "rad"
                row[f"F_raw_{axis}_{unit_force}"] = raw_force[index]
                row[f"F_filtered_{axis}_{unit_force}"] = (
                    filtered_force[index]
                )
                row[f"F_ext_{axis}_{unit_force}"] = F_ext[index]
                row[f"v_virtual_{axis}_{unit_velocity}"] = (
                    trajectory_step.velocity[index]
                )
                row[f"v_actual_{axis}_{unit_velocity}"] = (
                    actual_velocity[index]
                )
                row[f"a_actual_{axis}_{unit_acceleration}"] = (
                    actual_acceleration[index]
                )
                row[f"{axis}_cmd_{unit_position}"] = x_c[index]
                row[f"{axis}_actual_{unit_position}"] = x[index]
                row[f"{axis}_cmd_minus_actual_{unit_position}"] = (
                    commanded_minus_actual[index]
                )
            diagnostics_logger.write(row)

        print(
            "x_c",
            [round(value, 4) for value in x_c],
            "F_ext",
            [round(value, 3) for value in F_ext],
            "v_actual",
            [round(value, 4) for value in actual_velocity],
            "tracking",
            [round(value, 4) for value in trajectory_step.tracking_error],
            "frozen=" + str(trajectory_step.frozen),
            end="\r",
            flush=True,
        )
        if not args.sync_feedback:
            next_tick += dt
            delay = next_tick - time.monotonic()
            if delay > 0.0:
                time.sleep(delay)
            else:
                next_tick = time.monotonic()
except KeyboardInterrupt:
    print("\nInterrupted")
finally:
    if args.execute:
        if args.direct_tcp:
            drain_direct_replies(raise_on_error=False)
        rtde_c.Stop()
    if args.timing and loop_intervals:
        intervals_ms = np.asarray(loop_intervals) * 1000.0
        print(
            "\nTiming:"
            f" cycles={len(loop_intervals)}"
            f" mean_hz={1000.0 / intervals_ms.mean():.3f}"
            f" mean_ms={intervals_ms.mean():.3f}"
            f" median_ms={np.median(intervals_ms):.3f}"
            f" std_ms={intervals_ms.std(ddof=1):.3f}"
            f" p95_ms={np.percentile(intervals_ms, 95):.3f}"
            f" max_ms={intervals_ms.max():.3f}"
        )
        ages_ms = np.asarray(feedback_ages) * 1000.0
        print(
            "Feedback age:"
            f" mean_ms={ages_ms.mean():.3f}"
            f" p95_ms={np.percentile(ages_ms, 95):.3f}"
            f" max_ms={ages_ms.max():.3f}"
        )
        if servo_reply_tracker.latencies:
            reply_ms = np.asarray(servo_reply_tracker.latencies) * 1000.0
            print(
                "ServoP reply:"
                f" samples={len(reply_ms)}"
                f" mean_ms={reply_ms.mean():.3f}"
                f" p95_ms={np.percentile(reply_ms, 95):.3f}"
                f" max_ms={reply_ms.max():.3f}"
                f" pending={servo_reply_tracker.pending_count}"
            )
    if diagnostics_logger is not None:
        diagnostics_logger.close()
        print(f"\nDiagnostics CSV: {diagnostics_logger.path}")
    rtde_c.close()
    feedback_reader.stop()
    rtde_recv.close()
    print("\nStopped")
