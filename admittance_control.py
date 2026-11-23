import argparse
import importlib.util
from pathlib import Path
import re
import time

import numpy as np
from scipy.spatial.transform import Rotation as R

import filter_module
from realtime_pipeline import LatestRobotState, ServoPStream


parser = argparse.ArgumentParser(description="CR5 六维力导纳控制")
parser.add_argument("--robot-ip", default="192.168.5.5")
parser.add_argument(
    "--sdk",
    type=Path,
    default=Path(__file__).with_name("sdk") / "dobot_api.py",
)
parser.add_argument("--dt", type=float, default=0.02)
parser.add_argument(
    "--servo-period",
    type=float,
    default=0.03,
    help="independent ServoP send period in seconds",
)
parser.add_argument("--duration", type=float, default=0.0, help="0 表示持续运行")
parser.add_argument("--execute", action="store_true", help="默认只试运行，不发送 ServoP")
parser.add_argument("--zero-sensor", action="store_true")
parser.add_argument("--timing", action="store_true", help="打印控制周期统计")
parser.add_argument("--speed-filter-alpha", type=float, default=1.0)
parser.add_argument("--target-filter-alpha", type=float, default=1.0)
args = parser.parse_args()
if args.dt <= 0.0:
    parser.error("--dt must be positive")
if args.servo_period <= 0.0:
    parser.error("--servo-period must be positive")
if not 0.0 < args.speed_filter_alpha <= 1.0:
    parser.error("--speed-filter-alpha must be in (0, 1]")
if not 0.0 < args.target_filter_alpha <= 1.0:
    parser.error("--target-filter-alpha must be in (0, 1]")
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


sdk = load_sdk(args.sdk)
rtde_c = sdk.DobotApiDashboard(args.robot_ip, 29999)
rtde_recv = sdk.DobotApiFeedBack(args.robot_ip, 30004)
require_connected_socket(rtde_c.socket_dobot, args.robot_ip, 29999, "CR5 dashboard")
require_connected_socket(rtde_recv.socket_dobot, args.robot_ip, 30004, "CR5 feedback")

feedback_reader = LatestRobotState(rtde_recv)
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
x_d = x_init
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
F_ext_filter = filter_module.RCLowPassFilter(alpha=0.35, initial_value=F_ext_comp)
speed_filter = filter_module.RCLowPassFilter(
    alpha=args.speed_filter_alpha,
    initial_value=np.zeros(6),
)
target_filter = filter_module.RCLowPassFilter(
    alpha=args.target_filter_alpha,
    initial_value=x_init,
)

# 设计导纳控制参数
M = [0.8, 0.8, 0.8, 0.01, 0.01, 0.01]
D = [16, 16, 16, 0.12, 0.12, 0.16]
K = [80, 80, 80, 1.1, 1.1, 2]
dt = args.dt
Deadband = [1.5, 1.5, 1.5, 0.015, 0.05, 0.3]

# 零漂更新只在小残差、低速和低角速度时进行。
BiasUpdateBeta = 0.001
BiasVelocityGate = [0.005, 0.005, 0.005, 0.02, 0.02, 0.02]
last_velocity_pose = None


def estimate_tcp_speed(pose):
    """由相邻位姿计算基坐标/用户坐标下的 TCP 速度和角速度。

    姿态角速度使用 R_current @ R_previous.T，得到与控制器左乘姿态增量
    一致的旋转向量速度。
    """

    global last_velocity_pose
    now = time.monotonic()
    if last_velocity_pose is None:
        last_velocity_pose = (now, np.array(pose, dtype=float))
        return np.zeros(6)
    previous_time, previous_pose = last_velocity_pose
    elapsed = now - previous_time
    last_velocity_pose = (now, np.array(pose, dtype=float))
    if elapsed <= 0.0:
        return np.zeros(6)

    speed = np.empty(6)
    speed[:3] = (np.array(pose[:3]) - previous_pose[:3]) / elapsed
    previous_rotation = R.from_rotvec(previous_pose[3:])
    current_rotation = R.from_rotvec(pose[3:])
    rotation_delta = current_rotation * previous_rotation.inv()
    speed[3:] = rotation_delta.as_rotvec() / elapsed
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

    ddx_c = [0] * 6
    dx_c = [0] * 6
    deltax_c = [0] * 6
    x_c = [0] * 6

    e = [0] * 6
    de = [0] * 6
    e_F = [0] * 6

    e[0:3] = np.array(x_d[0:3]) - np.array(x[0:3])
    rot_x_d = R.from_rotvec(x_d[3:6])
    rot_x = R.from_rotvec(x[3:6])
    e[3:6] = (rot_x_d * rot_x.inv()).as_rotvec()
    de[0:6] = np.array(dx_d[0:6]) - np.array(dx[0:6])
    e_F[0:6] = np.array(F_d[0:6]) - np.array(F_ext[0:6])
    for i in range(6):
        if abs(e_F[i]) < Deadband[i]:
            e_F[i] = 0

    for i in range(6):
        ddx_c[i] = (D[i] * de[i] + K[i] * e[i] - e_F[i]) / M[i] + ddx_d[i]
        dx_c[i] = dx[i] + ddx_c[i] * dt
        deltax_c[i] = dx_c[i] * dt

    x_c[0:3] = np.array(x[0:3]) + np.array(deltax_c[0:3])
    x_c[3:6] = (R.from_rotvec(deltax_c[3:6]) * rot_x).as_rotvec()
    return x_c


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
next_tick = started
previous_loop_start = None
loop_intervals = []
feedback_ages = []
try:
    while args.duration == 0.0 or time.monotonic() - started < args.duration:
        loop_start = time.monotonic()
        if previous_loop_start is not None:
            loop_intervals.append(loop_start - previous_loop_start)
        previous_loop_start = loop_start
        packet, feedback_sequence, packet_received_at = feedback_reader.snapshot()
        feedback_ages.append(time.monotonic() - packet_received_at)
        control_dt = dt
        x = read_pose(packet)
        dx_raw = estimate_tcp_speed(x)
        dx = speed_filter.update(dx_raw)
        filtered_force = F_ext_filter.update(read_force(packet))
        F_ext = filtered_force - F_ext_comp
        ExternalForceComp(filtered_force, dx_raw)
        x_c = admittance_control(M, D, K, control_dt, x, dx, F_ext)
        x_servo = target_filter.update(x_c)

        if args.execute:
            servo_values = pose_for_servop(
                x_servo, packet["ToolVectorActual"][0][3:]
            )
            servo_stream.update_target(servo_values)
            servo_stream.raise_if_fault()

        print(
            "x_c",
            [round(value, 4) for value in x_c],
            "F_ext",
            [round(value, 3) for value in F_ext],
            end="\r",
            flush=True,
        )
        next_tick += dt
        delay = next_tick - time.monotonic()
        if delay > 0.0:
            time.sleep(delay)
        else:
            next_tick = time.monotonic()
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
    rtde_c.close()
    feedback_reader.stop()
    rtde_recv.close()
    print("\nStopped")
