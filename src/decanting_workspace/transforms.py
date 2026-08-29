"""Small, explicit SE(3) helpers shared by scene and robot adapters."""

from __future__ import annotations

import math

import numpy as np


def yaw_matrix(yaw_rad: float) -> np.ndarray:
    """Return a homogeneous rotation about world Z."""

    c = math.cos(float(yaw_rad))
    s = math.sin(float(yaw_rad))
    result = np.eye(4)
    result[:3, :3] = ((c, -s, 0.0), (s, c, 0.0), (0.0, 0.0, 1.0))
    return result


def pose_matrix(
    translation_m: tuple[float, float, float] | np.ndarray,
    *,
    yaw_rad: float = 0.0,
) -> np.ndarray:
    """Return ``World_T_local`` for a translation and Z yaw."""

    result = yaw_matrix(yaw_rad)
    translation = np.asarray(translation_m, dtype=float).reshape(3)
    if not np.all(np.isfinite(translation)):
        raise ValueError("translation must contain finite values")
    result[:3, 3] = translation
    return result


def quaternion_matrix(
    translation_m: tuple[float, float, float] | np.ndarray,
    quaternion_xyzw: tuple[float, float, float, float] | np.ndarray,
) -> np.ndarray:
    """Return ``World_T_local`` from an xyzw quaternion."""

    q = np.asarray(quaternion_xyzw, dtype=float).reshape(4)
    norm = float(np.linalg.norm(q))
    if norm == 0.0 or not np.isfinite(norm):
        raise ValueError("quaternion must be finite and non-zero")
    x, y, z, w = q / norm
    rotation = np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=float,
    )
    result = np.eye(4)
    result[:3, :3] = rotation
    result[:3, 3] = np.asarray(translation_m, dtype=float).reshape(3)
    return result


def transform_xy(
    local_xy_m: tuple[float, float] | np.ndarray,
    origin_xy_m: tuple[float, float] | np.ndarray,
    yaw_rad: float,
) -> tuple[float, float]:
    """Transform a 2-D local point into world coordinates."""

    local = np.asarray(local_xy_m, dtype=float).reshape(2)
    origin = np.asarray(origin_xy_m, dtype=float).reshape(2)
    c = math.cos(float(yaw_rad))
    s = math.sin(float(yaw_rad))
    world = origin + np.array(((c, -s), (s, c))) @ local
    return float(world[0]), float(world[1])


def inverse_transform(transform: np.ndarray) -> np.ndarray:
    """Invert a rigid homogeneous transform without a generic matrix inverse."""

    matrix = np.asarray(transform, dtype=float).reshape(4, 4)
    rotation = matrix[:3, :3]
    result = np.eye(4)
    result[:3, :3] = rotation.T
    result[:3, 3] = -rotation.T @ matrix[:3, 3]
    return result

