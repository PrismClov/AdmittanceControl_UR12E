# AdmittanceControl_CR5

基于本仓库原有六维力导纳控制算法，适配 Dobot CR5。

## 接口

- `192.168.5.5:30004`：读取 `ToolVectorActual` 和 `ActualTCPForce`
- `192.168.5.5:29999`：发送 `ServoP`，退出时发送 `Stop`
- `SixForceHome()`：由 `--zero-sensor` 显式触发

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

## 最小优化版

本仓库是独立的最小试验副本，基于原始 CR5 导纳控制，不包含速度导纳、
KF 力估计、jerk 限幅、关节空间控制或自定义轨迹复刻。

相对原始控制只做了以下改动：

- 姿态目标低通改为四元数最短路径插值，避免旋转向量跨 `+pi/-pi`
  时走长路径并生成异常 `ServoP` 目标。
- 增加 `--mass`、`--damping`、`--stiffness` 三个六轴参数入口。
- 对导纳生成的 TCP 目标速度增加六轴速度和加速度限制。

默认参数与当前原始仓库一致：

```text
M = 0.8,0.8,0.8,0.01,0.01,0.01
D = 16,16,16,0.12,0.12,0.16
K = 80,80,80,1.1,1.1,2
```

如需复现最早提交的较软阻尼，可只覆盖 `D`：

```bash
--damping "8,8,8,0.006,0.006,0.005"
```

速度和加速度默认限制：

```text
线速度     0.15 m/s
角速度     0.50 rad/s
线加速度   0.8 m/s^2
角加速度   1.5 rad/s^2
```

只有速度和加速度限制，没有 jerk 限制：

```bash
--tcp-motion-limits "0.15,0.5,0.8,1.5"
```

四个数的顺序是：线速度、角速度、线加速度、角加速度，并分别应用到
`x/y/z` 和 `rx/ry/rz`。确实需要逐轴分别设置时，可传 12 个数：

```bash
--tcp-motion-limits "0.15,0.15,0.15,0.5,0.5,0.5,0.8,0.8,0.8,1.5,1.5,1.5"
```

原先的 `--tcp-max-velocity` 和 `--tcp-max-acceleration` 仍可单独覆盖对应轴。

## 执行

```bash
python3 admittance_control.py \
  --robot-ip 192.168.5.5 \
  --duration 0 \
  --execute
```

`--duration 0` 表示持续运行，按 `Ctrl+C` 停止。

默认执行方式已经拆成三个独立环节：

```bash
python3 admittance_control.py \
  --robot-ip 192.168.5.5 \
  --duration 0 \
  --execute \
  --timing
```

- `30004` 线程按硬件约 122 Hz 读取，但默认只向控制层发布最新帧到 50 Hz。
- 只有原始反馈时间戳更新时才发布新控制帧，错过节拍不会补发旧帧。
- 控制线程由每个新的 50 Hz 软件反馈帧触发，计算周期按 `--dt 0.02`。
- `29999` 独立 ServoP 线程按 `--servo-period 0.02` 发送最新目标，即 50 Hz。
- 独立应答线程解析 ServoP 回复并统计发送周期、RTT 和错误。
- ServoP 发送先登记再发送，限制最大在途请求并检测应答超时。

分析速度滤波和六维力到速度的耦合：

```bash
python3 analyze_motion.py logs/A.csv logs/B.csv logs/C.csv \
  --json-output logs/analysis.json
```

脚本输出六轴力、原始/控制速度、加速度和 Jerk 的 RMS，5 Hz 以上高频 RMS，
主频，跟踪误差，以及力到速度的相干度和增益。

## 控制说明

- 原导纳公式和 `M/D/K/Deadband` 参数保持不变
- 力偏置使用小步长残差更新，仅在六轴速度低于门限时执行
- 角速度由相邻位姿的 `R_current @ R_previous.T` 计算
- 输出姿态使用与左乘姿态增量一致的旋转向量
- `ServoP` 前选择与当前欧拉角连续且矩阵等价的欧拉角分支，避免 Rx 在 ±180° 跳变
- `--execute` 启动时若检测到上次遗留的碰撞状态，会先停止并清除该状态
- `--timing`：退出时打印控制周期均值、抖动、P95 和最大间隔
- `--dt`：控制周期，默认 `0.02 s`
- `--servo-period`：ServoP 独立发送周期，默认 `0.02 s`
- `--feedback-hz`：反馈软件发布频率，默认 `50 Hz`
- `--force-filter-alpha`：六维力低通系数，默认 `0.35`
- `--speed-filter-alpha`：TCP 速度低通系数，默认 `1.0`（关闭）
- `--target-filter-alpha`：ServoP 目标位姿低通系数，默认 `1.0`（关闭）
- `--tcp-motion-limits`：统一设置速度/加速度限制；简写四个数或逐轴十二个数
- `--tcp-max-velocity`：六轴目标速度上限；默认线速度 `0.15 m/s`、角速度 `0.5 rad/s`
- `--tcp-max-acceleration`：六轴目标加速度上限；默认线加速度 `0.8 m/s^2`、角加速度 `1.5 rad/s^2`
- `--metrics-csv`：记录六轴原始力、滤波力、导纳力、实际速度/加速度/Jerk、命令与实际位姿
- 控制周期和 ServoP 发送周期互相独立，始终只发送最新导纳目标
- 只限制速度与加速度，没有 jerk、位姿幅值或循环看门狗限制
- `ActualTCPForce` 按当前 TCP 姿态从 Tool 坐标系旋转到 User/Base 坐标系

## 其他文件

- `filter_module.py`：滑动平均、低通和卡尔曼滤波
- `servoj_example.py`：原 UR `ur_rtde` 示例，CR5 控制不使用
- `try.py`：原 UR 六维力读取示例，CR5 控制不使用
