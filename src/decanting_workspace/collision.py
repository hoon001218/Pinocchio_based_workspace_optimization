"""Collision proxies shared by task evaluation and candidate iteration.

This module intentionally stops short of implementing robot--robot or
robot--environment collision queries.  It only builds the conservative SR
exclusion box used while the UR20 performs a concurrent task.
"""

from __future__ import annotations

import itertools
import math
from typing import Sequence

import numpy as np

from .models import BasePose, BoxPrimitive, SceneSpec
from .robots import RobotBundle, floating_configuration
from .scene import SceneSnapshot, sr_cutting_workspace_footprint_box
from .transforms import transform_xy


def sr_concurrent_exclusion_box(
    spec: SceneSpec,
    snapshot: SceneSnapshot,
    sr_bundle: RobotBundle,
    *,
    arm_q: Sequence[float] | None = None,
    margin_m: float = 0.0,
) -> BoxPrimitive:
    """Bound the configured SR cutting region and the SR at one posture.

    The union is computed once in the SR base frame.  The resulting
    base-aligned box is then placed with ``snapshot.sr_base``, so callers can
    reuse the same calculation pattern for arbitrary base candidates without
    loading visual meshes or using the arm's full mechanical workspace.

    ``arm_q`` defaults to the configured nominal SR posture.  The active
    collision model supplies the robot bounds; the configured maximum total
    height additionally protects a 300 mm geometry model when the same
    evaluation must remain conservative for the 450 mm J3 option.  The SR
    pedestal and its camera poles remain ordinary scene collision boxes and
    are deliberately not duplicated here.
    """

    margin = _nonnegative_finite_scalar(margin_m, "margin_m")
    robot_spec = spec.robots["sr12ia"]
    workspace = robot_spec.workspace
    if workspace is None:
        raise ValueError("SR-12iA workspace specification is missing")

    selected_arm_q = np.asarray(
        robot_spec.nominal_q if arm_q is None else tuple(arm_q),
        dtype=float,
    ).reshape(-1)
    if not np.all(np.isfinite(selected_arm_q)):
        raise ValueError("SR arm_q must contain only finite values")

    # Identity root placement makes every geometry placement base-local for
    # both fixed-base and free-flyer bundles.
    base_local_q = floating_configuration(
        sr_bundle,
        BasePose(0.0, 0.0, 0.0, 0.0),
        selected_arm_q,
    )
    lower, upper = _robot_collision_bounds_base_local(sr_bundle, base_local_q)

    # The current task workspace is an oriented footprint on the pedestal
    # top.  Convert its candidate-world placement back to the common SR base
    # frame before taking the union.  Its small visual thickness is harmless,
    # while its lower face remains exactly at local z=0.
    footprint = sr_cutting_workspace_footprint_box(
        spec,
        snapshot.sr_base,
        snapshot.state.clearance_m,
    )
    footprint_lower, footprint_upper = _aligned_box_bounds_in_base_frame(
        footprint,
        snapshot.sr_base,
    )
    lower = np.minimum(lower, footprint_lower)
    upper = np.maximum(upper, footprint_upper)

    # FANUC's configured total-height options differ only with J3 stroke.  A
    # 300 mm mesh therefore remains safe for a 450 mm study after extending
    # its nominal upper bound to the tallest configured option.  A collision
    # proxy that is already more conservative (the fallback is +10 mm) wins.
    joint3_id = sr_bundle.model.getJointId("joint3")
    if joint3_id >= sr_bundle.model.njoints:
        raise ValueError(f"{sr_bundle.name} has no joint3")
    joint3 = sr_bundle.model.joints[joint3_id]
    if joint3.nq != 1:
        raise ValueError(f"{sr_bundle.name} joint3 is not one-DOF")
    j3_extension_m = float(base_local_q[joint3.idx_q])
    upper[2] = max(
        float(upper[2]),
        max(workspace.total_height_options_m) - j3_extension_m,
    )

    lower -= margin
    upper += margin
    size = upper - lower
    if not np.all(np.isfinite(size)) or np.any(size <= 0.0):
        raise ValueError("SR concurrent exclusion bounds are invalid")

    center_local = (lower + upper) / 2.0
    center_x, center_y = transform_xy(
        center_local[:2],
        snapshot.sr_base.xyz_m[:2],
        snapshot.sr_base.yaw_rad,
    )
    return BoxPrimitive(
        name="sr12ia_concurrent_exclusion",
        role="concurrent_exclusion",
        center_m=(
            center_x,
            center_y,
            snapshot.sr_base.z_m + float(center_local[2]),
        ),
        size_m=tuple(float(value) for value in size),
        yaw_deg=snapshot.sr_base.yaw_deg,
        collision_enabled=True,
    )


def _robot_collision_bounds_base_local(
    bundle: RobotBundle,
    q: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return a base-local AABB of all collision geometry at ``q``."""

    import pinocchio as pin

    if bundle.collision_model.ngeoms == 0:
        raise ValueError(f"{bundle.name} collision model is empty")

    model_data = bundle.model.createData()
    geometry_data = bundle.collision_model.createData()
    pin.forwardKinematics(bundle.model, model_data, q)
    pin.updateGeometryPlacements(
        bundle.model,
        model_data,
        bundle.collision_model,
        geometry_data,
        q,
    )

    lower = np.full(3, np.inf, dtype=float)
    upper = np.full(3, -np.inf, dtype=float)
    for geometry_object, placement in zip(
        bundle.collision_model.geometryObjects,
        geometry_data.oMg,
        strict=True,
    ):
        geometry = geometry_object.geometry
        geometry.computeLocalAABB()
        local_aabb = geometry.aabb_local
        local_lower = np.asarray(local_aabb.min_, dtype=float).reshape(3)
        local_upper = np.asarray(local_aabb.max_, dtype=float).reshape(3)
        if not (
            np.all(np.isfinite(local_lower))
            and np.all(np.isfinite(local_upper))
            and np.all(local_upper >= local_lower)
        ):
            raise ValueError(
                f"invalid local AABB for SR collision geometry {geometry_object.name}"
            )

        corners = np.asarray(
            tuple(
                itertools.product(
                    (local_lower[0], local_upper[0]),
                    (local_lower[1], local_upper[1]),
                    (local_lower[2], local_upper[2]),
                )
            ),
            dtype=float,
        )
        rotation = np.asarray(placement.rotation, dtype=float).reshape(3, 3)
        translation = np.asarray(placement.translation, dtype=float).reshape(3)
        placed_corners = (rotation @ corners.T).T + translation
        lower = np.minimum(lower, placed_corners.min(axis=0))
        upper = np.maximum(upper, placed_corners.max(axis=0))

    return lower, upper


def _aligned_box_bounds_in_base_frame(
    box: BoxPrimitive,
    base: BasePose,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert a box sharing ``base`` yaw into base-local AABB bounds."""

    yaw_delta = math.remainder(math.radians(box.yaw_deg - base.yaw_deg), 2.0 * math.pi)
    if abs(yaw_delta) > 1e-10:
        raise ValueError("cutting workspace box must be aligned with the SR base")

    dx = box.center_m[0] - base.x_m
    dy = box.center_m[1] - base.y_m
    c = math.cos(base.yaw_rad)
    s = math.sin(base.yaw_rad)
    center_local = np.asarray(
        (
            c * dx + s * dy,
            -s * dx + c * dy,
            box.center_m[2] - base.z_m,
        ),
        dtype=float,
    )
    half_size = np.asarray(box.size_m, dtype=float) / 2.0
    return center_local - half_size, center_local + half_size


def _nonnegative_finite_scalar(value: float, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return result
