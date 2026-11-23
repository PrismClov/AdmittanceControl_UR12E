import argparse
import importlib.util
from pathlib import Path
import re
import time

import numpy as np
from scipy.spatial.transform import Rotation as R

import filter_module
from contact_monitor import ContactHold
from force_processing import continuous_deadband
from force_transform import wrench_tool_to_user
from motion_logger import MotionCsvLogger
from motion_limits import (
    CartesianVelocityAccelerationLimiter,
    expand_tcp_motion_limits,
    integrate_velocity,
)
from realtime_pipeline import LatestRobotState, ServoPStream


DEFAULT_TCP_MAX_VELOCITY = [0.15, 0.15, 0.15, 0.5, 0.5, 0.5]
DEFAULT_TCP_MAX_ACCELERATION = [0.8, 0.8, 0.8, 1.5, 1.5, 1.5]


def parse_six_axis_values(text):
    values = [float(value.strip()) for value in text.split(",")]
    if len(values) != 6:
        raise argparse.ArgumentTypeError("expected six comma-separated values")
    if not np.all(np.isfinite(values)):
        raise argparse.ArgumentTypeError("all values must be finite")
    return values


def positive_mass(text):
    values = parse_six_axis_values(text)
    if any(value <= 0.0 for value in values):
        raise argparse.ArgumentTypeError("mass values must be positive")
    return values


def nonnegative_parameters(text):
    values = parse_six_axis_values(text)
    if any(value < 0.0 for value in values):
        raise argparse.ArgumentTypeError("values must be non-negative")
    return values


def tcp_motion_limits(text):
    values = [float(value.strip()) for value in text.split(",")]
    try:
        return expand_tcp_motion_limits(values)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


parser = argparse.ArgumentParser(description="CR5 六维力导纳控制")
parser.add_argument("--robot-ip", default="192.168.5.5")
parser.add_argument(
    "--sdk",
    type=Path,
    default=Path(__file__).with_name("sdk") / "dobot_api.py",
)
parser.add_argument(
    "--dt",
    type=float,
    default=0.02,
    help="control period in seconds",
)
parser.add_argument(
    "--servo-period",
    type=float,
    default=0.02,
    help="independent ServoP send period in seconds",
)
parser.add_argument(
    "--feedback-hz",
    type=float,
    default=50.0,
    help="software feedback publication rate",
)
parser.add_argument("--duration", type=float, default=0.0, help="0 表示持续运行")
parser.add_argument("--execute", action="store_true", help="默认只试运行，不发送 ServoP")
parser.add_argument("--zero-sensor", action="store_true")
parser.add_argument("--timing", action="store_true", help="打印控制周期统计")
parser.add_argument("--speed-filter-alpha", type=float, default=1.0)
parser.add_argument("--target-filter-alpha", type=float, default=1.0)
parser.add_argument("--force-filter-alpha", type=float, default=0.35)
parser.add_argument(
    "--mass",
    type=positive_mass,
    default=[0.8, 0.8, 0.8, 0.01, 0.01, 0.01],
    metavar="M1,...,M6",
)
parser.add_argument(
    "--damping",
    type=nonnegative_parameters,
    default=[16, 16, 16, 0.12, 0.12, 0.16],
    metavar="D1,...,D6",
)
parser.add_argument(
    "--stiffness",
    type=nonnegative_parameters,
    default=[80, 80, 80, 1.1, 1.1, 2],
    metavar="K1,...,K6",
)
parser.add_argument(
    "--tcp-motion-limits",
    "--motion-limits",
    type=tcp_motion_limits,
    default=None,
    metavar="VLIN,VANG,ALIN,AANG",
    help=(
        "unified TCP limits; use four values for uniform axes or twelve "
        "values for velocity[6] followed by acceleration[6]"
    ),
)
parser.add_argument(
    "--tcp-max-velocity",
    "--max-velocity",
    dest="tcp_max_velocity",
    type=nonnegative_parameters,
    default=None,
    metavar="V1,...,V6",
    help="six-axis TCP velocity limit in m/s and rad/s",
)
parser.add_argument(
    "--tcp-max-acceleration",
    "--max-acceleration",
    dest="tcp_max_acceleration",
    type=nonnegative_parameters,
    default=None,
    metavar="A1,...,A6",
    help="six-axis TCP acceleration limit in m/s^2 and rad/s^2",
)
parser.add_argument(
    "--metrics-csv",
    type=Path,
    help="write six-axis admittance diagnostics to CSV",
)
parser.add_argument(
    "--target-pose",
    type=parse_six_axis_values,
    metavar="X,Y,Z,RX,RY,RZ",
    help="admittance target pose in mm and degrees",
)
parser.add_argument(
    "--contact-stop-force",
    type=float,
    default=5.0,
    help="latch target motion after external force magnitude in N",
)
args = parser.parse_args()
if args.dt <= 0.0:
    parser.error("--dt must be positive")
if args.servo_period <= 0.0:
    parser.error("--servo-period must be positive")
if args.feedback_hz <= 0.0:
    parser.error("--feedback-hz must be positive")
if not 0.0 < args.speed_filter_alpha <= 1.0:
    parser.error("--speed-filter-alpha must be in (0, 1]")
if not 0.0 < args.target_filter_alpha <= 1.0:
    parser.error("--target-filter-alpha must be in (0, 1]")
if not 0.0 < args.force_filter_alpha <= 1.0:
    parser.error("--force-filter-alpha must be in (0, 1]")
if args.duration < 0.0:
    parser.error("--duration must be non-negative")
if args.contact_stop_force <= 0.0:
    parser.error("--contact-stop-force must be positive")
if not args.sdk.is_file():
    parser.error(f"Dobot SDK not found: {args.sdk}")

tcp_max_velocity = DEFAULT_TCP_MAX_VELOCITY.copy()
tcp_max_acceleration = DEFAULT_TCP_MAX_ACCELERATION.copy()
if args.tcp_motion_limits is not None:
    tcp_max_velocity, tcp_max_acceleration = args.tcp_motion_limits
if args.tcp_max_velocity is not None:
    tcp_max_velocity = args.tcp_max_velocity
if args.tcp_max_acceleration is not None:
    tcp_max_acceleration = args.tcp_max_acceleration


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
    pose = np.asarray(packet["ToolVectorActual"][0], dtype=float)
    return wrench_tool_to_user(packet["ActualTCPForce"][0], pose[3:])


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


sdk = load_sdk(args.sdk)
rtde_c = sdk.DobotApiDashboard(args.robot_ip, 29999)
rtde_recv = sdk.DobotApiFeedBack(args.robot_ip, 30004)
require_connected_socket(rtde_c.socket_dobot, args.robot_ip, 29999, "CR5 dashboard")
require_connected_socket(rtde_recv.socket_dobot, args.robot_ip, 30004, "CR5 feedback")

feedback_reader = LatestRobotState(rtde_recv, output_hz=args.feedback_hz)
feedback_reader.start()

# 当前 TCP 位姿作为初始位姿，不额外发送初始运动。
packet, feedback_sequence, packet_received_at = feedback_reader.wait_for_data()
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
                feedback_reader.snapshot()
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
                feedback_reader.snapshot()
            )
            if int(packet["EnableStatus"][0]) == 1:
                break
            time.sleep(0.05)
        else:
            raise RuntimeError("CR5 did not become enabled")

x_init = read_pose(packet)
print(x_init)
if args.target_pose is None:
    x_d = x_init
else:
    target_values = np.asarray(args.target_pose, dtype=float)
    x_d = np.empty(6)
    x_d[:3] = target_values[:3] / 1000.0
    x_d[3:] = R.from_euler(
        "xyz",
        target_values[3:],
        degrees=True,
    ).as_rotvec()
dx_d = [0, 0, 0, 0, 0, 0]
ddx_d = [0, 0, 0, 0, 0, 0]
F_d = [0, 0, 0, 0, 0, 0]

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
    time.sleep(0.02)
    packet, feedback_sequence, packet_received_at = feedback_reader.snapshot()
    F_ext_comp = comp_filter.update(read_force(packet))

"""
六维力传感器信号滤波
由于传感器噪声较严重，所以要进行信号滤波
滤波器的相关函数见filter_module.py
"""
F_ext_filter = filter_module.RCLowPassFilter(
    alpha=args.force_filter_alpha,
    initial_value=F_ext_comp,
)
speed_filter = filter_module.RCLowPassFilter(
    alpha=args.speed_filter_alpha,
    initial_value=np.zeros(6),
)
target_filter = filter_module.RCLowPassFilter(
    alpha=args.target_filter_alpha,
    initial_value=x_init,
)
target_rotation_filter = filter_module.RotationLowPassFilter(
    alpha=args.target_filter_alpha,
    initial_value=x_init[3:],
)
log_axes = ("x", "y", "z", "rx", "ry", "rz")
metric_fields = [
    "time_s",
    "feedback_sequence",
    "feedback_age_ms",
    "control_dt_s",
    "servo_sent",
    "servo_pending",
    "servo_rtt_ms",
    "contact_hold",
    "contact_force_n",
]
for axis in log_axes:
    metric_fields.extend(
        [
            f"F_raw_{axis}",
            f"F_filtered_{axis}",
            f"F_ext_{axis}",
            f"v_actual_raw_{axis}",
            f"v_control_{axis}",
            f"a_actual_{axis}",
            f"jerk_actual_{axis}",
            f"a_admittance_raw_{axis}",
            f"e_F_raw_{axis}",
            f"e_F_deadband_{axis}",
            f"tcp_velocity_raw_{axis}",
            f"tcp_velocity_limited_{axis}",
            f"tcp_acceleration_limited_{axis}",
            f"x_target_{axis}",
            f"x_actual_{axis}",
            f"x_target_minus_actual_{axis}",
            f"x_servo_{axis}",
            f"x_servo_minus_actual_{axis}",
        ]
    )
metrics_logger = (
    MotionCsvLogger(args.metrics_csv, metric_fields)
    if args.metrics_csv is not None
    else None
)

feedback_timing = feedback_reader.period_stats()
publish_timing = feedback_reader.published_period_stats()
dt = args.dt
if not np.isfinite(dt) or not 0.004 <= dt <= 0.1:
    raise RuntimeError(f"invalid aligned control period: {dt}")
print(
    "Raw feedback:"
    f" median_hz={feedback_timing['median_hz']:.3f}"
    f" median_ms={feedback_timing['median_s'] * 1000.0:.3f}"
    f" p95_ms={feedback_timing['p95_s'] * 1000.0:.3f}"
)
print(
    "Software feedback:"
    f" median_hz={publish_timing['median_hz']:.3f}"
    f" median_ms={publish_timing['median_s'] * 1000.0:.3f}"
    f" p95_ms={publish_timing['p95_s'] * 1000.0:.3f}"
)
print(f"Control period: {dt * 1000.0:.3f} ms ({1.0 / dt:.3f} Hz)")

# 设计导纳控制参数
M = args.mass
D = args.damping
K = args.stiffness
Deadband = [1.5, 1.5, 1.5, 0.015, 0.05, 0.3]
motion_limiter = CartesianVelocityAccelerationLimiter(
    tcp_max_velocity,
    tcp_max_acceleration,
    dt,
)
contact_monitor = (
    ContactHold(args.contact_stop_force)
    if args.target_pose is not None
    else None
)

# 零漂更新只在小残差、低速和低角速度时进行。
BiasUpdateBeta = 0.001
BiasVelocityGate = [0.005, 0.005, 0.005, 0.02, 0.02, 0.02]
last_velocity_state = None


def estimate_tcp_speed(pose, sample_time):
    """由相邻位姿计算基坐标/用户坐标下的 TCP 速度和角速度。

    姿态角速度使用 R_current @ R_previous.T，得到与控制器左乘姿态增量
    一致的旋转向量速度。
    """

    global last_velocity_state
    pose = np.asarray(pose, dtype=float)
    sample_time = float(sample_time)
    if last_velocity_state is None:
        last_velocity_state = (sample_time, pose.copy(), np.zeros(6))
        return np.zeros(6)
    previous_time, previous_pose, previous_speed = last_velocity_state
    if sample_time <= previous_time:
        return previous_speed.copy()
    elapsed = sample_time - previous_time
    if elapsed <= 0.0:
        return np.zeros(6)

    speed = np.empty(6)
    speed[:3] = (pose[:3] - previous_pose[:3]) / elapsed
    previous_rotation = R.from_rotvec(previous_pose[3:])
    current_rotation = R.from_rotvec(pose[3:])
    rotation_delta = current_rotation * previous_rotation.inv()
    speed[3:] = rotation_delta.as_rotvec() / elapsed
    last_velocity_state = (sample_time, pose.copy(), speed.copy())
    return speed


def admittance_control(M, D, K, dt, x, dx, F_ext):
    """导纳控制

    参数:
    M     -- 导纳惯性矩阵
    D     -- 导纳阻尼矩阵
    K     -- 导纳刚度矩阵
    dt    -- 离散时间
    x     -- 当前笛卡儿位姿
    dx    -- 当前速度
    F_ext -- 当前外部作用力

    返回:
    目标笛卡儿位姿
    """

    ddx_c = np.zeros(6)
    dx_c = [0] * 6
    deltax_c = [0] * 6
    x_c = [0] * 6

    e = [0] * 6
    de = [0] * 6
    e[0:3] = np.array(x_d[0:3]) - np.array(x[0:3])
    rot_x_d = R.from_rotvec(x_d[3:6])
    rot_x = R.from_rotvec(x[3:6])
    e[3:6] = (rot_x_d * rot_x.inv()).as_rotvec()
    de[0:6] = np.array(dx_d[0:6]) - np.array(dx[0:6])
    e_F_raw = np.asarray(F_d[:6], dtype=float) - np.asarray(
        F_ext[:6],
        dtype=float,
    )
    e_F = continuous_deadband(e_F_raw, Deadband)

    for i in range(6):
        ddx_c[i] = (D[i] * de[i] + K[i] * e[i] - e_F[i]) / M[i] + ddx_d[i]
        dx_c[i] = dx[i] + ddx_c[i] * dt
        deltax_c[i] = dx_c[i] * dt

    x_c[0:3] = np.array(x[0:3]) + np.array(deltax_c[0:3])
    x_c[3:6] = (R.from_rotvec(deltax_c[3:6]) * rot_x).as_rotvec()
    return (
        x_c,
        ddx_c,
        np.asarray(dx_c, dtype=float),
        e_F_raw,
        e_F,
    )


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


servo_stream = None
if args.execute:
    servo_stream = ServoPStream(
        rtde_c.socket_dobot,
        period_s=args.servo_period,
    )
    servo_stream.start()


print("Start", "execute=" + str(args.execute))
started = time.monotonic()
previous_loop_start = None
last_status_print = started
loop_intervals = []
feedback_ages = []
previous_metric_velocity = None
previous_metric_acceleration = None
previous_metric_time = None
contact_notice_printed = False
try:
    while args.duration == 0.0 or time.monotonic() - started < args.duration:
        loop_start = time.monotonic()
        if previous_loop_start is not None:
            loop_intervals.append(loop_start - previous_loop_start)
        previous_loop_start = loop_start
        packet, feedback_sequence, packet_received_at = (
            feedback_reader.wait_for_next(feedback_sequence)
        )
        feedback_ages.append(time.monotonic() - packet_received_at)
        control_dt = dt
        x = read_pose(packet)
        dx_raw = estimate_tcp_speed(x, packet_received_at)
        if previous_metric_time is None:
            actual_acceleration = np.zeros(6)
            actual_jerk = np.zeros(6)
            previous_metric_time = packet_received_at
            previous_metric_velocity = dx_raw.copy()
            previous_metric_acceleration = actual_acceleration.copy()
        else:
            metric_dt = packet_received_at - previous_metric_time
            if metric_dt <= 0.0:
                actual_acceleration = previous_metric_acceleration.copy()
                actual_jerk = np.zeros(6)
            else:
                actual_acceleration = (
                    dx_raw - previous_metric_velocity
                ) / metric_dt
                actual_jerk = (
                    actual_acceleration - previous_metric_acceleration
                ) / metric_dt
                previous_metric_time = packet_received_at
                previous_metric_velocity = dx_raw.copy()
                previous_metric_acceleration = actual_acceleration.copy()
        dx = speed_filter.update(dx_raw)
        filtered_force = F_ext_filter.update(read_force(packet))
        F_ext = filtered_force - F_ext_comp
        ExternalForceComp(filtered_force, dx_raw)
        contact_hold = (
            contact_monitor.update(F_ext)
            if contact_monitor is not None
            else False
        )
        if contact_hold and not contact_notice_printed:
            print(
                "\nContact detected: holding target at current pose, "
                f"force={contact_monitor.detected_force_n:.3f} N"
            )
            contact_notice_printed = True
        (
            _,
            a_admittance_raw,
            v_admittance_raw,
            e_F_raw,
            e_F_deadband,
        ) = admittance_control(
            M,
            D,
            K,
            control_dt,
            x,
            dx,
            F_ext,
        )
        requested_velocity = (
            np.zeros(6) if contact_hold else v_admittance_raw
        )
        v_admittance_limited = motion_limiter.update(requested_velocity)
        a_admittance_limited = motion_limiter.last_acceleration
        x_c = integrate_velocity(x, v_admittance_limited, control_dt)
        filtered_target = target_filter.update(x_c)
        x_servo = np.asarray(x_c, dtype=float).copy()
        x_servo[:3] = filtered_target[:3]
        x_servo[3:] = target_rotation_filter.update(x_c[3:])

        if args.execute:
            servo_values = pose_for_servop(
                x_servo, packet["ToolVectorActual"][0][3:]
            )
            servo_stream.update_target(servo_values)
            servo_stream.raise_if_fault()

        if metrics_logger is not None:
            target_minus_actual = np.zeros(6)
            target_minus_actual[:3] = x_c[:3] - x[:3]
            target_minus_actual[3:] = (
                R.from_rotvec(x_c[3:])
                * R.from_rotvec(x[3:]).inv()
            ).as_rotvec()
            servo_minus_actual = np.zeros(6)
            servo_minus_actual[:3] = x_servo[:3] - x[:3]
            servo_minus_actual[3:] = (
                R.from_rotvec(x_servo[3:])
                * R.from_rotvec(x[3:]).inv()
            ).as_rotvec()
            raw_force = read_force(packet)
            row = {
                "time_s": loop_start - started,
                "feedback_sequence": feedback_sequence,
                "feedback_age_ms": feedback_ages[-1] * 1000.0,
                "control_dt_s": control_dt,
                "servo_sent": (
                    servo_stream.sent_count if servo_stream is not None else 0
                ),
                "servo_pending": (
                    servo_stream.pending_count if servo_stream is not None else 0
                ),
                "servo_rtt_ms": (
                    servo_stream.last_rtt_s * 1000.0
                    if servo_stream is not None
                    and servo_stream.last_rtt_s is not None
                    else ""
                ),
                "contact_hold": int(contact_hold),
                "contact_force_n": (
                    float(np.linalg.norm(F_ext[:3]))
                    if contact_hold
                    else 0.0
                ),
            }
            for index, axis in enumerate(log_axes):
                row[f"F_raw_{axis}"] = raw_force[index]
                row[f"F_filtered_{axis}"] = filtered_force[index]
                row[f"F_ext_{axis}"] = F_ext[index]
                row[f"v_actual_raw_{axis}"] = dx_raw[index]
                row[f"v_control_{axis}"] = dx[index]
                row[f"a_actual_{axis}"] = actual_acceleration[index]
                row[f"jerk_actual_{axis}"] = actual_jerk[index]
                row[f"a_admittance_raw_{axis}"] = a_admittance_raw[index]
                row[f"e_F_raw_{axis}"] = e_F_raw[index]
                row[f"e_F_deadband_{axis}"] = e_F_deadband[index]
                row[f"tcp_velocity_raw_{axis}"] = v_admittance_raw[index]
                row[f"tcp_velocity_limited_{axis}"] = (
                    v_admittance_limited[index]
                )
                row[f"tcp_acceleration_limited_{axis}"] = (
                    a_admittance_limited[index]
                )
                row[f"x_target_{axis}"] = x_c[index]
                row[f"x_actual_{axis}"] = x[index]
                row[f"x_target_minus_actual_{axis}"] = (
                    target_minus_actual[index]
                )
                row[f"x_servo_{axis}"] = x_servo[index]
                row[f"x_servo_minus_actual_{axis}"] = (
                    servo_minus_actual[index]
                )
            metrics_logger.write(row)

        if loop_start - last_status_print >= 0.1:
            print(
                "x_c",
                [round(value, 4) for value in x_c],
                "F_ext",
                [round(value, 3) for value in F_ext],
                end="\r",
                flush=True,
            )
            last_status_print = loop_start
except KeyboardInterrupt:
    print("\nInterrupted")
finally:
    if servo_stream is not None:
        servo_stream.stop()
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
        if servo_stream is not None:
            send_ms = np.asarray(servo_stream.send_intervals) * 1000.0
            rtt_ms = np.asarray(servo_stream.reply_latencies) * 1000.0
            if send_ms.size:
                print(
                    "ServoP send:"
                    f" samples={send_ms.size + 1}"
                    f" mean_ms={send_ms.mean():.3f}"
                    f" p95_ms={np.percentile(send_ms, 95):.3f}"
                    f" max_ms={send_ms.max():.3f}"
                )
            if rtt_ms.size:
                print(
                    "ServoP RTT:"
                    f" samples={rtt_ms.size}"
                    f" mean_ms={rtt_ms.mean():.3f}"
                    f" p95_ms={np.percentile(rtt_ms, 95):.3f}"
                    f" max_ms={rtt_ms.max():.3f}"
                    f" unknown_replies={servo_stream.unknown_reply_count}"
                )
    if metrics_logger is not None:
        metrics_logger.close()
        print(f"Metrics CSV: {metrics_logger.path}")
    rtde_c.close()
    feedback_reader.stop()
    rtde_recv.close()
    print("\nStopped")
