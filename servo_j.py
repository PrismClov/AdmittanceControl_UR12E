"""Inverse-kinematics helpers for the CR5 ServoJ command path."""

from __future__ import annotations

import re
from typing import Sequence

import numpy as np


_IK_REPLY = re.compile(
    r"(-?\d+),\{([^}]*)\},InverseKin\(",
    re.DOTALL,
)
_NUMBER = re.compile(r"-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")


def _vector6(value: Sequence[float], name: str) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.shape != (6,) or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain six finite values")
    return array


def parse_inverse_kinematics(response: str) -> np.ndarray:
    """Return a joint solution from a dashboard or ROS 2 reply."""

    text = str(response)
    matches = list(_IK_REPLY.finditer(text))
    if matches:
        match = matches[-1]
        error_id = int(match.group(1))
        if error_id != 0:
            raise RuntimeError(
                f"InverseKin failed with error {error_id}: {response!r}"
            )
        values = match.group(2)
    elif text.strip().startswith("{") and text.strip().endswith("}"):
        values = text.strip()[1:-1]
    else:
        raise RuntimeError(f"InverseKin returned no joint solution: {response!r}")

    joints = np.asarray(
        [float(value) for value in _NUMBER.findall(values)],
        dtype=float,
    )
    if joints.shape != (6,) or not np.all(np.isfinite(joints)):
        raise RuntimeError(f"Invalid InverseKin solution: {response!r}")
    return joints


def joint_target_error(
    goal_joint_degrees: Sequence[float],
    current_joint_degrees: Sequence[float],
) -> np.ndarray:
    """Return shortest wrapped joint error in degrees."""

    goal = _vector6(goal_joint_degrees, "goal_joint_degrees")
    current = _vector6(current_joint_degrees, "current_joint_degrees")
    return (goal - current + 180.0) % 360.0 - 180.0
