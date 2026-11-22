import argparse
import importlib.util
from pathlib import Path
import re
import time

import numpy as np
from scipy.spatial.transform import Rotation as R

import filter_module


parser = argparse.ArgumentParser(description="CR5 六维力导纳控制")
parser.add_argument("--robot-ip", default="192.168.5.5")
parser.add_argument(
    "--sdk",
    type=Path,
    default=Path("/home/orin/Desktop/视觉伺服/dobot/sdk/dobot_api.py"),
)
parser.add_argument("--dt", type=float, default=0.02)
parser.add_argument("--duration", type=float, default=0.0, help="0 表示持续运行")
parser.add_argument("--execute", action="store_true", help="默认只试运行，不发送 ServoP")
parser.add_argument("--zero-sensor", action="store_true")
args = parser.parse_args()
if args.dt <= 0.0:
    parser.error("--dt must be positive")
if args.duration < 0.0:
    parser.error("--duration must be non-negative")
if not args.sdk.is_file():
    parser.error(f"Dobot SDK not found: {args.sdk}")


def load_sdk(path):
    spec = importlib.util.spec_from_file_location("dobot_api_v4", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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
if not rtde_recv.socket_dobot:
    raise ConnectionError(f"无法连接 CR5 反馈端口 {args.robot_ip}:30004")


def get_packet():
    packet = rtde_recv.feedBackData()
    if packet is None:
        raise RuntimeError("CR5 反馈包缺失")
    return packet


# 当前 TCP 位姿作为初始位姿，不额外发送初始运动。
packet = get_packet()
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
            packet = get_packet()
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
            packet = get_packet()
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
    F_ext_comp = comp_filter.update(read_force(get_packet()))
    time.sleep(0.02)
F_ext = read_force(get_packet()) - F_ext_comp

"""
六维力传感器信号滤波
由于传感器噪声较严重，所以要进行信号滤波
滤波器的相关函数见filter_module.py
"""
F_ext_filter = filter_module.RCLowPassFilter(alpha=0.35, initial_value=F_ext)

# 设计导纳控制参数
M = [0.8, 0.8, 0.8, 0.01, 0.01, 0.01]
D = [8, 8, 8, 0.006, 0.006, 0.005]
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


print("Start", "execute=" + str(args.execute))
started = time.monotonic()
next_tick = started
try:
    while args.duration == 0.0 or time.monotonic() - started < args.duration:
        packet = get_packet()
        x = read_pose(packet)
        dx = estimate_tcp_speed(x)
        filtered_force = F_ext_filter.update(read_force(packet))
        F_ext = filtered_force - F_ext_comp
        ExternalForceComp(filtered_force, dx)
        x_c = admittance_control(M, D, K, dt, x, dx, F_ext)

        if args.execute:
            response = rtde_c.ServoP(
                *pose_for_servop(x_c, packet["ToolVectorActual"][0][3:]),
                t=dt,
                aheadtime=50,
                gain=500,
            )
            error = parse_error_id(response)
            if error != 0:
                raise RuntimeError(f"ServoP failed with error {error}: {response!r}")

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
    if args.execute:
        rtde_c.Stop()
    rtde_c.close()
    rtde_recv.close()
    print("\nStopped")
