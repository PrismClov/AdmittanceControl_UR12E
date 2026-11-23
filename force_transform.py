import numpy as np
from scipy.spatial.transform import Rotation as R


def wrench_tool_to_user(wrench, rpy_degrees):
    """Rotate a six-axis wrench from the TCP/tool frame to the user frame."""
    wrench = np.asarray(wrench, dtype=float)
    if wrench.shape != (6,):
        raise ValueError("wrench must contain six values")
    if not np.all(np.isfinite(wrench)):
        raise ValueError("wrench values must be finite")
    rotation = R.from_euler("xyz", rpy_degrees, degrees=True)
    transformed = wrench.copy()
    transformed[:3] = rotation.apply(wrench[:3])
    transformed[3:] = rotation.apply(wrench[3:])
    return transformed
