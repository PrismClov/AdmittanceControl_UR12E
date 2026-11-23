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
- 速度式导纳输出经过速度、加速度和 Jerk 约束后再积分
- 位置与 SO(3) 姿态使用同一约束速度生成连续目标
- 限幅后的实际命令速度写回轨迹状态，避免内部积分饱和
- 编码器反馈与命令跟踪误差超过门限时冻结轨迹推进
- 力偏置使用小步长残差更新，仅在六轴速度低于门限时执行
- 角速度由相邻位姿的 `R_current @ R_previous.T` 计算
- 输出姿态使用与左乘姿态增量一致的旋转向量
- `ServoP` 前选择与当前欧拉角连续且矩阵等价的欧拉角分支，避免 Rx 在 ±180° 跳变
- `--execute` 启动时若检测到上次遗留的碰撞状态，会先停止并清除该状态
- `--timing`：退出时打印控制周期均值、抖动、P95 和最大间隔
- `--direct-tcp`：固定周期直接写 Dashboard TCP，不等待当前 `ServoP` 回包
- `--max-linear-velocity` / `--max-angular-velocity`：默认 `0.05 m/s` / `0.2 rad/s`
- `--max-linear-acceleration` / `--max-angular-acceleration`：默认 `0.5 m/s²` / `1.5 rad/s²`
- `--max-linear-jerk` / `--max-angular-jerk`：默认 `10.0 m/s³` / `30.0 rad/s³`
- `--max-tracking-translation` / `--max-tracking-rotation`：默认 `0.02 m` / `0.10 rad`
- 外力释放且接近初始位姿、速度足够小时，轨迹显式落零以避免残余来回小运动
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
