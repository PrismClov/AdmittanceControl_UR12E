"""Local CR5 FK, Jacobian, and differential inverse kinematics."""

from __future__ import annotations

import math
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation


def pose_to_transform(pose: np.ndarray) -> np.ndarray:
    pose = np.asarray(pose, dtype=float)
    if pose.shape != (6,) or not np.all(np.isfinite(pose)):
        raise ValueError("pose must contain six finite values")
    transform = np.eye(4)
    transform[:3, :3] = Rotation.from_rotvec(pose[3:]).as_matrix()
    transform[:3, 3] = pose[:3]
    return transform


def _transform_rpy_xyz(roll, pitch, yaw) -> np.ndarray:
    transform = np.eye(4)
    transform[:3, :3] = Rotation.from_euler(
        "xyz", [roll, pitch, yaw]
    ).as_matrix()
    return transform


class Cr5LocalKinematics:
    """URDF FK/Jacobian and constrained differential IK for TCP targets."""

    def __init__(
        self,
        urdf_path,
        initial_joints_degrees,
        initial_tcp_pose,
        damping=0.02,
        max_joint_velocity=2.0,
        max_follow_error=0.20,
        base_link="base_link",
        tip_link="Link6",
    ):
        self.urdf_path = Path(urdf_path)
        self.damping = float(damping)
        self.max_joint_velocity = float(max_joint_velocity)
        self.max_follow_error = float(max_follow_error)
        self._load_chain(base_link, tip_link)

        q_initial = np.deg2rad(
            np.asarray(initial_joints_degrees, dtype=float)
        )
        if q_initial.shape != (6,) or not np.all(np.isfinite(q_initial)):
            raise ValueError("initial_joints_degrees must contain six values")
        tcp_initial = pose_to_transform(initial_tcp_pose)
        self.end_to_tcp = np.linalg.inv(self.fk(q_initial)) @ tcp_initial
        self.joint_target = q_initial.copy()

    def _load_chain(self, base_link, tip_link) -> None:
        root = ET.parse(self.urdf_path).getroot()
        by_child = {
            joint.find("child").get("link"): joint
            for joint in root.findall("joint")
        }
        chain = []
        link = tip_link
        while link != base_link:
            joint = by_child[link]
            chain.insert(0, joint)
            link = joint.find("parent").get("link")

        self._chain = []
        self.joint_names = []
        self.lower = []
        self.upper = []
        for joint in chain:
            origin = joint.find("origin")
            xyz_text = origin.get("xyz", "0 0 0")
            rpy_text = origin.get("rpy", "0 0 0")
            xyz = np.fromstring(xyz_text, sep=" ")
            rpy = np.fromstring(rpy_text, sep=" ")
            origin_transform = _transform_rpy_xyz(*rpy)
            origin_transform[:3, 3] = xyz
            active = joint.get("type") != "fixed"
            axis = np.zeros(3)
            if active:
                if joint.get("type") != "revolute":
                    raise ValueError("expected bounded revolute joints")
                axis = np.fromstring(joint.find("axis").get("xyz"), sep=" ")
                axis /= np.linalg.norm(axis)
                self.joint_names.append(joint.get("name"))
                limited = joint.find("limit")
                self.lower.append(float(limited.get("lower")))
                self.upper.append(float(limited.get("upper")))
            self._chain.append((origin_transform, axis, active))
        if len(self.joint_names) != 6:
            raise ValueError("expected exactly six active joints")

    def evaluate(self, joints_rad):
        q = np.asarray(joints_rad, dtype=float)
        if q.shape != (6,) or not np.all(np.isfinite(q)):
            raise ValueError("joints_rad must contain six values")
        transform = np.eye(4)
        axes = []
        origins = []
        index = 0
        for origin, axis, active in self._chain:
            transform = transform @ origin
            if active:
                axes.append(transform[:3, :3] @ axis)
                origins.append(transform[:3, 3].copy())
                rotation = np.eye(4)
                rotation[:3, :3] = Rotation.from_rotvec(
                    axis * q[index]
                ).as_matrix()
                transform = transform @ rotation
                index += 1
        jacobian = np.column_stack(
            [
                np.r_[
                    np.cross(axis, transform[:3, 3] - origin),
                    axis,
                ]
                for axis, origin in zip(axes, origins)
            ]
        )
        return transform, jacobian

    def fk(self, joints_rad):
        return self.evaluate(joints_rad)[0]

    def tcp_fk(self, joints_rad):
        return self.fk(joints_rad) @ self.end_to_tcp

    def tcp_jacobian(self, joints_rad):
        end, jacobian = self.evaluate(joints_rad)
        jacobian = jacobian.copy()
        offset = (end[:3, :3] @ self.end_to_tcp[:3, 3])
        jacobian[:3] += np.cross(jacobian[3:], offset[:, None], axis=0)
        return jacobian

    def solve_tcp_pose(self, target_pose, current_joints_degrees, dt):
        current = np.deg2rad(
            np.asarray(current_joints_degrees, dtype=float)
        )
        if current.shape != (6,) or not np.all(np.isfinite(current)):
            raise ValueError("current_joints_degrees must contain six values")
        if not math.isfinite(dt) or dt <= 0.0:
            raise ValueError("dt must be finite and positive")

        if np.max(np.abs(self.joint_target - current)) > self.max_follow_error:
            self.joint_target = current.copy()

        target = pose_to_transform(target_pose)
        actual = self.tcp_fk(self.joint_target)
        error = np.r_[
            target[:3, 3] - actual[:3, 3],
            (
                Rotation.from_matrix(target[:3, :3])
                * Rotation.from_matrix(actual[:3, :3]).inv()
            ).as_rotvec(),
        ]

        jacobian = self.tcp_jacobian(self.joint_target)
        inverse = jacobian.T @ np.linalg.solve(
            jacobian @ jacobian.T
            + self.damping**2 * np.eye(6),
            error,
        )
        max_step = self.max_joint_velocity * dt
        step_norm = float(np.linalg.norm(inverse))
        if step_norm > max_step:
            inverse *= max_step / step_norm
        candidate = self.joint_target + inverse
        if (
            not np.all(np.isfinite(candidate))
            or np.any(candidate < self.lower)
            or np.any(candidate > self.upper)
        ):
            raise RuntimeError("local inverse kinematics reached a joint limit")
        self.joint_target = candidate
        return np.rad2deg(self.joint_target)
