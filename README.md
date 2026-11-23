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

默认执行方式已经拆成三个独立环节：

```bash
python3 admittance_control.py \
  --robot-ip 192.168.5.5 \
  --duration 0 \
  --execute \
  --timing
```

- `30004` 线程持续读取并维护最新位姿和六维力，不再软件降采样。
- 控制线程按 `--dt` 固定 20 ms 读取最新状态，不等待下一帧序号。
- `29999` 独立 ServoP 线程按 `--servo-period` 默认 30 ms 发送最新目标。
- 独立应答线程解析 ServoP 回复并统计发送周期、RTT 和错误。

## 控制说明

- 原导纳公式和 `M/D/K/Deadband` 参数保持不变
- 力偏置使用小步长残差更新，仅在六轴速度低于门限时执行
- 角速度由相邻位姿的 `R_current @ R_previous.T` 计算
- 输出姿态使用与左乘姿态增量一致的旋转向量
- `ServoP` 前选择与当前欧拉角连续且矩阵等价的欧拉角分支，避免 Rx 在 ±180° 跳变
- `--execute` 启动时若检测到上次遗留的碰撞状态，会先停止并清除该状态
- `--timing`：退出时打印控制周期均值、抖动、P95 和最大间隔
- `--servo-period`：ServoP 独立发送周期，默认 `0.03 s`
- `--speed-filter-alpha`：TCP 速度低通系数，默认 `1.0`（关闭）
- `--target-filter-alpha`：ServoP 目标位姿低通系数，默认 `1.0`（关闭）
- 控制周期和 ServoP 发送周期互相独立，始终只发送最新导纳目标
- 没有速度限幅、位姿限幅或循环看门狗
- `ActualTCPForce` 转换为环境外力；用户坐标 Y、Ry 正向修正，Rx 反向修正

## 其他文件

- `filter_module.py`：滑动平均、低通和卡尔曼滤波
- `servoj_example.py`：原 UR `ur_rtde` 示例，CR5 控制不使用
- `try.py`：原 UR 六维力读取示例，CR5 控制不使用
