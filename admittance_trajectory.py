"""Velocity-form admittance dynamics with constrained command integration."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np
from scipy.spatial.transform import Rotation


def _vector6(value: Sequence[float], name: str) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.shape != (6,) or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain six finite values")
    return array


def _limit_norm(value: np.ndarray, limit: Sequence[float]) -> np.ndarray:
    result = value.copy()
    limits = np.asarray(limit, dtype=float)
    linear_norm = float(np.linalg.norm(result[:3]))
    angular_norm = float(np.linalg.norm(result[3:]))
    if linear_norm > limits[0]:
        result[:3] *= limits[0] / linear_norm
    if angular_norm > limits[3]:
        result[3:] *= limits[3] / angular_norm
    return result


@dataclass(frozen=True)
class TrajectoryLimits:
    max_velocity: tuple[float, float, float, float, float, float]
    max_acceleration: tuple[float, float, float, float, float, float]
    max_jerk: tuple[float, float, float, float, float, float]
    max_tracking_translation: float
    max_tracking_rotation: float

    def __post_init__(self):
        for name in ("max_velocity", "max_acceleration", "max_jerk"):
            values = _vector6(getattr(self, name), name)
            if np.any(values <= 0.0):
                raise ValueError(f"{name} entries must be positive")
            object.__setattr__(self, name, tuple(values.tolist()))
        if self.max_tracking_translation <= 0.0:
            raise ValueError("max_tracking_translation must be positive")
        if self.max_tracking_rotation <= 0.0:
            raise ValueError("max_tracking_rotation must be positive")


@dataclass(frozen=True)
class TrajectoryStep:
    target_pose: np.ndarray
    velocity: np.ndarray
    acceleration: np.ndarray
    tracking_error: np.ndarray
    frozen: bool


class AdmittanceTrajectory:
    """Integrate admittance in velocity form and feed back limited velocity.

    The mass-damping-stiffness dynamics produce an acceleration request.
    Jerk, acceleration and velocity are limited before the command is
    integrated in SE(3), so the internal state always equals the command that
    was actually sent.
    """

    def __init__(
        self,
        mass: Sequence[float],
        damping: Sequence[float],
        stiffness: Sequence[float],
        deadband: Sequence[float],
        limits: TrajectoryLimits,
    ):
        self.mass = _vector6(mass, "mass")
        self.damping = _vector6(damping, "damping")
        self.stiffness = _vector6(stiffness, "stiffness")
        self.deadband = _vector6(deadband, "deadband")
        if np.any(self.mass <= 0.0):
            raise ValueError("mass entries must be positive")
        if np.any(self.damping < 0.0) or np.any(self.stiffness < 0.0):
            raise ValueError("damping and stiffness must be non-negative")
        self.limits = limits
        self._nominal_pose = np.zeros(6)
        self._command_pose = np.zeros(6)
        self._velocity = np.zeros(6)
        self._acceleration = np.zeros(6)
        self._initialized = False

    @property
    def velocity(self) -> np.ndarray:
        return self._velocity.copy()

    @property
    def command_pose(self) -> np.ndarray:
        return self._command_pose.copy()

    def reset(self, pose: Sequence[float]) -> None:
        pose = _vector6(pose, "pose")
        self._nominal_pose = pose.copy()
        self._command_pose = pose.copy()
        self._velocity = np.zeros(6)
        self._acceleration = np.zeros(6)
        self._initialized = True

    def _tracking_error(self, current_pose: np.ndarray) -> np.ndarray:
        error = np.zeros(6)
        error[:3] = current_pose[:3] - self._command_pose[:3]
        current_rotation = Rotation.from_rotvec(current_pose[3:])
        command_rotation = Rotation.from_rotvec(self._command_pose[3:])
        error[3:] = (
            current_rotation * command_rotation.inv()
        ).as_rotvec()
        return error

    def step(
        self,
        current_pose: Sequence[float],
        external_wrench: Sequence[float],
        dt: float,
    ) -> TrajectoryStep:
        if not self._initialized:
            raise RuntimeError("trajectory.reset(pose) must be called first")
        if not math.isfinite(dt) or not 0.0 < dt <= 0.1:
            raise ValueError("dt must be finite and in (0, 0.1] seconds")

        current = _vector6(current_pose, "current_pose")
        wrench = _vector6(external_wrench, "external_wrench")
        tracking = self._tracking_error(current)
        frozen = bool(
            np.linalg.norm(tracking[:3])
            > self.limits.max_tracking_translation
            or np.linalg.norm(tracking[3:])
            > self.limits.max_tracking_rotation
        )
        if frozen:
            self._velocity = np.zeros(6)
            self._acceleration = np.zeros(6)
            return TrajectoryStep(
                target_pose=self._command_pose.copy(),
                velocity=self._velocity.copy(),
                acceleration=self._acceleration.copy(),
                tracking_error=tracking,
                frozen=True,
            )

        effective_wrench = np.where(
            np.abs(wrench) < self.deadband,
            0.0,
            wrench,
        )
        command_rotation = Rotation.from_rotvec(self._command_pose[3:])
        nominal_rotation = Rotation.from_rotvec(self._nominal_pose[3:])
        offset = np.zeros(6)
        offset[:3] = self._command_pose[:3] - self._nominal_pose[:3]
        offset[3:] = (
            command_rotation * nominal_rotation.inv()
        ).as_rotvec()

        acceleration_request = (
            effective_wrench
            - self.damping * self._velocity
            - self.stiffness * offset
        ) / self.mass
        jerk = (acceleration_request - self._acceleration) / dt
        jerk = _limit_norm(jerk, self.limits.max_jerk)
        acceleration = self._acceleration + jerk * dt
        acceleration = _limit_norm(
            acceleration, self.limits.max_acceleration
        )
        velocity = self._velocity + acceleration * dt
        velocity = _limit_norm(velocity, self.limits.max_velocity)

        target = self._command_pose.copy()
        target[:3] += velocity[:3] * dt
        target_rotation = (
            Rotation.from_rotvec(velocity[3:] * dt) * command_rotation
        )
        target[3:] = target_rotation.as_rotvec()

        self._command_pose = target
        self._velocity = velocity
        self._acceleration = acceleration
        return TrajectoryStep(
            target_pose=target.copy(),
            velocity=velocity.copy(),
            acceleration=acceleration.copy(),
            tracking_error=tracking,
            frozen=False,
        )
