# AdmittanceControl_CR5

基于本仓库原有六维力导纳控制算法，适配 Dobot CR5。

## 接口

- `192.168.5.5:30004`：读取 `ToolVectorActual` 和 `ActualTCPForce`
- `192.168.5.5:29999`：发送 `ServoP`，退出时发送 `Stop`
- `SixForceHome()`：由 `--zero-sensor` 显式触发

默认使用：

```text
/home/orin/Desktop/视觉伺服/dobot/sdk/dobot_api.py
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

## 控制说明

- 原导纳公式和 `M/D/K/Deadband` 参数保持不变
- 力偏置使用小步长残差更新，仅在六轴速度低于门限时执行
- 角速度由相邻位姿的 `R_current @ R_previous.T` 计算
- 输出姿态使用与左乘姿态增量一致的旋转向量
- 没有速度限幅、位姿限幅或循环看门狗
- `ActualTCPForce` 转换为环境外力；用户坐标 Y 轴（左右）单独取反

## 其他文件

- `filter_module.py`：滑动平均、低通和卡尔曼滤波
- `servoj_example.py`：原 UR `ur_rtde` 示例，CR5 控制不使用
- `try.py`：原 UR 六维力读取示例，CR5 控制不使用
