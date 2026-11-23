import argparse
from pathlib import Path
import threading
import time

import numpy as np
from scipy.spatial.transform import Rotation as R

import filter_module
from local_kinematics import Cr5LocalKinematics
from ros2_servo import Ros2FeedbackReader, Ros2ServoController


parser = argparse.ArgumentParser(description="CR5 六维力导纳控制")
parser.add_argument("--robot-ip", default="192.168.5.5")
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
    "--sync-feedback",
    action="store_true",
    help="每收到一帧六维力反馈执行一次控制，并将反馈 dt 用于导纳积分",
)
parser.add_argument(
    "--servo-command",
    choices=("p", "j"),
    default="j",
    help="p=ROS2 ServoP; j=local kinematics + ROS2 ServoJ（默认）",
)
parser.add_argument(
    "--ros-service-prefix",
    default="/dobot_bringup_ros2/srv",
    help="Dobot ROS2 service prefix",
)
parser.add_argument(
    "--ros-feed-topic",
    default="/dobot_bringup_ros2/msg/FeedInfo",
    help="Dobot ROS2 combined feedback topic",
)
parser.add_argument(
    "--urdf",
    type=Path,
    help="CR5 URDF used for local Jacobian-based ServoJ targets",
)
parser.add_argument("--speed-filter-alpha", type=float, default=1.0)
parser.add_argument("--target-filter-alpha", type=float, default=1.0)
args = parser.parse_args()
if args.dt <= 0.0:
    parser.error("--dt must be positive")
if args.feedback_hz <= 0.0:
    parser.error("--feedback-hz must be positive")
if not 0.0 < args.speed_filter_alpha <= 1.0:
    parser.error("--speed-filter-alpha must be in (0, 1]")
if not 0.0 < args.target_filter_alpha <= 1.0:
    parser.error("--target-filter-alpha must be in (0, 1]")
if args.duration < 0.0:
    parser.error("--duration must be non-negative")


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


ros_servo = Ros2ServoController(args.ros_service_prefix)
ros_servo.wait_until_ready()
rtde_recv = Ros2FeedbackReader(args.ros_feed_topic)


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


def stop_robot():
    response = ros_servo.call("stop", timeout=2.0)
    return int(response.res)


def clear_robot_error():
    response = ros_servo.call("clear_error")
    return int(response.res)


def enable_robot():
    response = ros_servo.call("enable")
    return int(response.res)


def zero_force_sensor():
    response = ros_servo.call("six_force_home")
    return int(response.res)


feedback_reader = LatestFeedback(rtde_recv, args.feedback_hz)
feedback_reader.start()

# 当前 TCP 位姿作为初始位姿，不额外发送初始运动。
packet, feedback_sequence, packet_received_at = feedback_reader.get_latest()
if args.execute:
    if int(packet["RobotMode"][0]) == 9:
        error_id = int(packet["ErrorStatus"][0])
        raise RuntimeError(f"Robot is in error state: {error_id!r}")
    if int(packet["RobotMode"][0]) == 11:
        print("Clearing previous collision state")
        stop_error = stop_robot()
        clear_error = clear_robot_error()
        if stop_error != 0 or clear_error != 0:
            raise RuntimeError(
                f"Could not clear collision state: "
                f"Stop={stop_error}, ClearError={clear_error}"
            )
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
        error = enable_robot()
        if error != 0:
            raise RuntimeError(f"EnableRobot failed with error {error}")
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
local_kinematics = None
if args.servo_command == "j":
    urdf_path = args.urdf
    if urdf_path is None:
        from ament_index_python.packages import get_package_share_directory

        urdf_path = (
            Path(get_package_share_directory("dobot_rviz"))
            / "urdf"
            / "cr5_robot.urdf"
        )
    if not urdf_path.is_file():
        raise FileNotFoundError(f"CR5 URDF not found: {urdf_path}")
    local_kinematics = Cr5LocalKinematics(
        urdf_path,
        packet["QActual"][0],
        x_init,
    )
x_d = x_init
dx_d = [0, 0, 0, 0, 0, 0]
ddx_d = [0, 0, 0, 0, 0, 0]
F_d = [0, 0, 0, 0, 0, 0]

if args.zero_sensor:
    error = zero_force_sensor()
    if error != 0:
        raise RuntimeError(f"SixForceHome failed with error {error}")
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


if args.execute and args.servo_command == "p":
    ros_servo.prime_pose(
        pose_for_servop(x_init, packet["ToolVectorActual"][0][3:]),
        dt,
    )


print("Start", "execute=" + str(args.execute))
started = time.monotonic()
next_tick = started
previous_loop_start = None
loop_intervals = []
feedback_ages = []
previous_feedback_timestamp = None
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
        if args.sync_feedback:
            feedback_timestamp = float(packet["TimeStamp"][0]) * 0.001
            if (
                previous_feedback_timestamp is not None
                and feedback_timestamp > previous_feedback_timestamp
            ):
                control_dt = feedback_timestamp - previous_feedback_timestamp
            previous_feedback_timestamp = feedback_timestamp
        x = read_pose(packet)
        dx_raw = estimate_tcp_speed(x)
        dx = speed_filter.update(dx_raw)
        filtered_force = F_ext_filter.update(read_force(packet))
        F_ext = filtered_force - F_ext_comp
        ExternalForceComp(filtered_force, dx_raw)
        x_c = admittance_control(M, D, K, control_dt, x, dx, F_ext)
        x_servo = target_filter.update(x_c)

        if args.execute:
            if args.servo_command == "j":
                joint_target = local_kinematics.solve_tcp_pose(
                    x_servo,
                    packet["QActual"][0],
                    control_dt,
                )
                ros_servo.submit_joints(joint_target, control_dt)
                ros_servo.spin_once()
                ros_servo.raise_if_fault()
            else:
                servo_values = pose_for_servop(
                    x_servo, packet["ToolVectorActual"][0][3:]
                )
                ros_servo.submit_pose(servo_values, control_dt)
                ros_servo.spin_once()
                ros_servo.raise_if_fault()

        print(
            "x_c",
            [round(value, 4) for value in x_c],
            "F_ext",
            [round(value, 3) for value in F_ext],
            "servo=" + args.servo_command,
            (
                "ros_pending=" + str(ros_servo.pending_count)
                if ros_servo is not None
                else ""
            ),
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
        try:
            if not ros_servo.prepare_stop():
                print("Warning: ROS 2 servo request still active before Stop")
            stop_error = stop_robot()
            if stop_error != 0:
                print(f"Stop failed with error {stop_error}")
        except Exception as exc:
            print(f"Stop request failed: {exc}")
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
        if ros_servo is not None:
            latency_ms = (
                float("nan")
                if ros_servo.last_latency_s is None
                else ros_servo.last_latency_s * 1000.0
            )
            print(
                f"ROS2 Servo{args.servo_command.upper()}:"
                f" sent={ros_servo.sent_count}"
                f" coalesced={ros_servo.coalesced_count}"
                f" pending={ros_servo.pending_count}"
                f" last_latency_ms={latency_ms:.3f}"
            )
    feedback_reader.stop()
    rtde_recv.close()
    ros_servo.close()
    print("\nStopped")
