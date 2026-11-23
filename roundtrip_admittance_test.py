"""Repeatable A/B admittance and contact-force test."""

from __future__ import annotations

import argparse
import csv
import importlib.util
from pathlib import Path
import shlex
import subprocess
import sys
import time


DEFAULT_POSE_A = (
    419.893,
    113.137,
    307.892,
    -174.658,
    6.521,
    62.576,
)
DEFAULT_POSE_B = (
    429.736,
    112.879,
    58.967,
    -166.527,
    9.889,
    63.774,
)


def parse_pose(text):
    values = tuple(float(value.strip()) for value in text.split(","))
    if len(values) != 6:
        raise argparse.ArgumentTypeError("pose must contain six values")
    return values


def pose_argument(pose):
    return ",".join(f"{value:.6f}" for value in pose)


def build_command(args, target_pose, contact_enabled, metrics_csv):
    command = [
        sys.executable,
        str(Path(__file__).with_name("admittance_control.py")),
        "--robot-ip",
        args.robot_ip,
        "--duration",
        str(args.contact_duration if contact_enabled else args.retract_duration),
        "--execute",
        "--timing",
        "--feedback-hz",
        str(args.feedback_hz),
        "--dt",
        str(args.dt),
        "--servo-period",
        str(args.dt),
        "--mass",
        "0.8,0.8,0.8,0.01,0.01,0.01",
        "--damping",
        "16,16,16,0.12,0.12,0.16",
        "--stiffness",
        "80,80,80,1.1,1.1,2",
        "--tcp-motion-limits",
        "0.15,0.5,0.8,1.5",
        "--target-pose",
        pose_argument(target_pose),
        "--contact-stop-force",
        str(args.contact_force) if contact_enabled else "0",
        "--metrics-csv",
        str(metrics_csv),
    ]
    if contact_enabled:
        command.extend(
            [
                "--contact-target-force",
                str(args.contact_force),
                "--contact-confirm-time",
                str(args.contact_confirm_time),
                "--contact-direction-angle",
                str(args.contact_direction_angle),
                "--contact-kp",
                str(args.contact_kp),
                "--contact-ki",
                str(args.contact_ki),
                "--contact-force-deadband",
                str(args.contact_deadband),
                "--contact-max-velocity",
                str(args.contact_max_velocity),
                "--contact-release-after",
                str(args.contact_release_after),
            ]
        )
    return command


def load_sdk(path):
    spec = importlib.util.spec_from_file_location("dobot_roundtrip", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def require_ready_robot(sdk_path, robot_ip):
    sdk = load_sdk(sdk_path)
    dashboard = sdk.DobotApiDashboard(robot_ip, 29999)
    feedback = sdk.DobotApiFeedBack(robot_ip, 30004)
    try:
        packet = feedback.feedBackData()
        if packet is None:
            raise RuntimeError("CR5 feedback is unavailable")
        mode = int(packet["RobotMode"][0])
        enabled = int(packet["EnableStatus"][0])
        collision = int(packet["CollisionState"][0])
        if mode != 5 or enabled != 1 or collision != 0:
            raise RuntimeError(
                f"CR5 not ready: mode={mode}, enabled={enabled}, "
                f"collision={collision}"
            )
    finally:
        try:
            feedback.socket_dobot.close()
        except Exception:
            pass
        try:
            dashboard.socket_dobot.close()
        except Exception:
            pass


def contains_contact(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return any(
            row.get("contact_state") == "contact"
            for row in csv.DictReader(handle)
        )


def run_command(command):
    print("$", " ".join(shlex.quote(part) for part in command), flush=True)
    completed = subprocess.run(command, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"test leg failed with exit code {completed.returncode}")


def main():
    parser = argparse.ArgumentParser(
        description="Repeatable A/B admittance and 5 N contact-force test"
    )
    parser.add_argument("--robot-ip", default="192.168.5.5")
    parser.add_argument(
        "--sdk",
        type=Path,
        default=Path(__file__).with_name("sdk") / "dobot_api.py",
    )
    parser.add_argument("--cycles", type=int, default=1)
    parser.add_argument("--pose-a", type=parse_pose, default=DEFAULT_POSE_A)
    parser.add_argument("--pose-b", type=parse_pose, default=DEFAULT_POSE_B)
    parser.add_argument("--retract-duration", type=float, default=12.0)
    parser.add_argument("--contact-duration", type=float, default=20.0)
    parser.add_argument("--feedback-hz", type=float, default=50.0)
    parser.add_argument("--dt", type=float, default=0.02)
    parser.add_argument("--contact-force", type=float, default=5.0)
    parser.add_argument("--contact-confirm-time", type=float, default=0.10)
    parser.add_argument("--contact-direction-angle", type=float, default=20.0)
    parser.add_argument("--contact-kp", type=float, default=0.001)
    parser.add_argument("--contact-ki", type=float, default=0.0001)
    parser.add_argument("--contact-deadband", type=float, default=0.25)
    parser.add_argument("--contact-max-velocity", type=float, default=0.01)
    parser.add_argument(
        "--contact-release-after",
        type=float,
        default=0.0,
        help="0 keeps contact force active until the leg ends",
    )
    parser.add_argument(
        "--metrics-dir",
        type=Path,
        default=Path("logs"),
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="actually run the robot; without it, print both leg commands",
    )
    args = parser.parse_args()
    if args.cycles < 1:
        parser.error("--cycles must be at least one")
    if args.retract_duration <= 0.0 or args.contact_duration <= 0.0:
        parser.error("test leg durations must be positive")
    if args.contact_force <= 0.0:
        parser.error("--contact-force must be positive")
    if not args.sdk.is_file():
        parser.error(f"Dobot SDK not found: {args.sdk}")

    args.metrics_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    plan = []
    for cycle in range(1, args.cycles + 1):
        plan.append(
            (
                "retract_to_A",
                build_command(
                    args,
                    args.pose_a,
                    False,
                    args.metrics_dir
                    / f"roundtrip_{stamp}_c{cycle}_to_A.csv",
                ),
                args.metrics_dir
                / f"roundtrip_{stamp}_c{cycle}_to_A.csv",
            )
        )
        plan.append(
            (
                "contact_at_B",
                build_command(
                    args,
                    args.pose_b,
                    True,
                    args.metrics_dir
                    / f"roundtrip_{stamp}_c{cycle}_to_B.csv",
                ),
                args.metrics_dir
                / f"roundtrip_{stamp}_c{cycle}_to_B.csv",
            )
        )

    if not args.execute:
        for _, command, _ in plan:
            print(" ".join(shlex.quote(part) for part in command))
        return

    require_ready_robot(args.sdk, args.robot_ip)
    for leg_name, command, metrics_csv in plan:
        print(f"\n=== {leg_name} ===", flush=True)
        run_command(command)
        if leg_name == "contact_at_B" and not contains_contact(metrics_csv):
            raise RuntimeError(
                f"{leg_name} never reached confirmed contact: {metrics_csv}"
            )
        time.sleep(0.2)
    print(f"\nRound-trip test completed. Logs: {args.metrics_dir.resolve()}")


if __name__ == "__main__":
    main()
