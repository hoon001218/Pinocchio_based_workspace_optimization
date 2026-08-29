from __future__ import annotations

import math

import numpy as np
import pytest

from decanting_workspace.transforms import (
    inverse_transform,
    pose_matrix,
    quaternion_matrix,
    transform_xy,
)


def test_candidate_yaw_transform():
    x_m, y_m = transform_xy((1.0, 0.0), (2.0, 3.0), math.pi / 2.0)
    assert (x_m, y_m) == pytest.approx((2.0, 4.0))


def test_pose_inverse_round_trip():
    transform = pose_matrix((1.0, -2.0, 0.5), yaw_rad=0.7)
    np.testing.assert_allclose(inverse_transform(transform) @ transform, np.eye(4), atol=1e-12)


def test_xyzw_frame_conversion():
    transform = quaternion_matrix((1.0, 2.0, 3.0), (0.0, 0.0, 0.0, 1.0))
    np.testing.assert_allclose(transform[:3, :3], np.eye(3))
    np.testing.assert_allclose(transform[:3, 3], (1.0, 2.0, 3.0))

