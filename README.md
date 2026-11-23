# AdmittanceControl_CR5

基于本仓库原有六维力导纳控制算法，适配 Dobot CR5。

## 接口

- `/dobot_bringup_ros2/msg/FeedInfo`：读取位姿、关节和六维力
- `/dobot_bringup_ros2/srv/InverseKin`、`ServoJ`：ROS 2 控制服务
- `ServoP` 兼容模式仍直接使用 `192.168.5.5:29999`
- `SixForceHome`：由 `--zero-sensor` 显式触发

默认使用仓库内由跃疆 SDK 同步的文件：

```text
sdk/dobot_api.py
```

也可通过 `--sdk` 指定 `dobot_api.py` 路径。

## 安装

```bash
python3 -m pip install numpy scipy
```

## 试运行

只读取和计算，不发送 `ServoP`：

```bash
python3 admittance_control.py \
  --robot-ip 192.168.5.5 \
  --duration 10
```

## 执行

```bash
python3 admittance_control.py \
  --robot-ip 192.168.5.5 \
  --duration 0 \
  --execute
```

`--duration 0` 表示持续运行，按 `Ctrl+C` 停止。

固定周期直发 `ServoP`：

```bash
python3 admittance_control.py \
  --robot-ip 192.168.5.5 \
  --duration 0 \
  --execute \
  --direct-tcp \
  --timing
```

默认运动接口通过 Dobot ROS 2 驱动调用 `InverseKin + ServoJ`。控制循环
保持 50 Hz；ROS 2 命令使用单一在途请求，忙时只保留最新目标，不排队：

启动 ROS 2 驱动：

```bash
source /opt/ros/humble/setup.bash
source /home/orin/dobot_ws/install/setup.bash

IP_address=192.168.5.5 DOBOT_TYPE=cr5 \
  ros2 launch dobot_bringup_v4 dobot_bringup_ros2.launch.py
```

另一个终端启动导纳控制：

```bash
python3 admittance_control.py \
  --robot-ip 192.168.5.5 \
  --duration 10 \
  --execute \
  --timing
```

临时切回 `ServoP`：

```bash
python3 admittance_control.py \
  --robot-ip 192.168.5.5 \
  --duration 10 \
  --execute \
  --servo-command p \
  --direct-tcp \
  --timing
```

`ServoJ` 使用当前反馈关节作为 `JointNear`，并检查逆解关节是否发生大范围跳变。
六维力、位姿和关节反馈均从 ROS 2 `FeedInfo` 话题读取。
该模式不能与 `--direct-tcp` 同时使用。

直发模式复用唯一的 `29999` TCP 控制连接，不在当前控制周期等待 `ServoP` 回包；
已完成的回包会在后续周期非阻塞读取并检查错误。

按六维力反馈帧同步控制：

```bash
python3 admittance_control.py \
  --robot-ip 192.168.5.5 \
  --duration 10 \
  --execute \
  --direct-tcp \
  --sync-feedback \
  --timing
```

## 控制说明

- 原导纳公式和 `M/D/K/Deadband` 参数保持不变
- 力偏置使用小步长残差更新，仅在六轴速度低于门限时执行
- 角速度由相邻位姿的 `R_current @ R_previous.T` 计算
- 输出姿态使用与左乘姿态增量一致的旋转向量
- `ServoP` 前选择与当前欧拉角连续且矩阵等价的欧拉角分支，避免 Rx 在 ±180° 跳变
- `--execute` 启动时若检测到上次遗留的碰撞状态，会先停止并清除该状态
- `--timing`：退出时打印控制周期均值、抖动、P95 和最大间隔
- `--direct-tcp`：固定周期直接写 Dashboard TCP，不等待当前 `ServoP` 回包
- `--servo-command`：`j` 使用 ROS 2 `InverseKin + ServoJ`（默认），`p` 使用 SDK `ServoP`
- `--ros-service-prefix`：Dobot ROS 2 服务前缀，默认 `/dobot_bringup_ros2/srv`
- `--speed-filter-alpha`：TCP 速度低通系数，默认 `1.0`（关闭）
- `--target-filter-alpha`：ServoP 目标位姿低通系数，默认 `1.0`（关闭）
- `--sync-feedback`：每帧六维力触发一次控制，按反馈时间戳计算 `dt`
- `--feedback-hz`：六维力软件输出频率，默认 `50 Hz`
- 默认固定频率模式会后台持续读取反馈，每次控制只使用最新一帧，避免旧包积压
- 硬件反馈流仍以约 `122 Hz` 到达，`--feedback-hz 50` 是软件降采样
- 没有速度限幅、位姿限幅或循环看门狗
- `ActualTCPForce` 转换为环境外力；用户坐标 Y、Ry 正向修正，Rx 反向修正

## 其他文件

- `filter_module.py`：滑动平均、低通和卡尔曼滤波
- `servoj_example.py`：原 UR `ur_rtde` 示例，CR5 控制不使用
- `try.py`：原 UR 六维力读取示例，CR5 控制不使用
