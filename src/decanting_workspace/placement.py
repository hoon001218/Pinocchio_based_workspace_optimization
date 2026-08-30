"""Candidate robot-installation footprint and pedestal validation."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .models import BoxPrimitive, SceneSpec
from .scene import SceneSnapshot


_CONTACT_TOLERANCE_M = 1e-9
_PEDESTAL_NAMES = ("ur20_pedestal", "sr12ia_pedestal")


@dataclass(frozen=True)
class PlacementIssue:
    """One deterministic reason that a base-placement candidate is invalid."""

    code: str
    objects: tuple[str, ...]
    message: str


@dataclass(frozen=True)
class PlacementReport:
    """Installation validation result for both robot pedestals."""

    valid: bool
    issues: tuple[PlacementIssue, ...]


def validate_base_placement(
    spec: SceneSpec,
    snapshot: SceneSnapshot,
    *,
    clearance_m: float = 0.0,
) -> PlacementReport:
    """Validate both candidate pedestals against the installation geometry.

    Each pedestal is the oriented box materialized by :func:`materialize_scene`,
    spanning from the floor to its candidate base height.  Its footprint must
    remain inside the configured axis-aligned installation rectangle.  Static
    collision boxes and the other pedestal are checked using exact rectangle
    SAT in XY plus interval overlap in Z.  Mere contact is allowed at zero
    clearance.

    A positive clearance creates a rectangular safety envelope around each
    pedestal.  For pedestal/static checks the full clearance is applied to the
    pedestal; for the pedestal pair half is applied to each so their required
    total separation is ``clearance_m``.
    """

    clearance = float(clearance_m)
    if not math.isfinite(clearance) or clearance < 0.0:
        raise ValueError("clearance_m must be finite and non-negative")

    pedestals = _pedestals(snapshot)
    issues: list[PlacementIssue] = []

    for pedestal in pedestals:
        if not _inside_installation_region(spec, pedestal, clearance):
            issues.append(
                PlacementIssue(
                    code="outside_installation_region",
                    objects=(pedestal.name,),
                    message=(
                        f"{pedestal.name} footprint, including {clearance:.6g} m "
                        "clearance, leaves the raw XY installation rectangle"
                    ),
                )
            )

    static_collision_boxes = tuple(
        box for box in spec.static_boxes if box.collision_enabled
    )
    for pedestal in pedestals:
        expanded = _inflate_box(pedestal, clearance)
        for obstacle in static_collision_boxes:
            if _obb_overlap_3d(expanded, obstacle):
                issues.append(
                    PlacementIssue(
                        code="static_box_overlap",
                        objects=(pedestal.name, obstacle.name),
                        message=(
                            f"{pedestal.name} overlaps fixed collision box "
                            f"{obstacle.name}"
                        ),
                    )
                )

    pair_margin = clearance / 2.0
    ur_pedestal = _inflate_box(pedestals[0], pair_margin)
    sr_pedestal = _inflate_box(pedestals[1], pair_margin)
    if _obb_overlap_3d(ur_pedestal, sr_pedestal):
        issues.append(
            PlacementIssue(
                code="pedestal_overlap",
                objects=(pedestals[0].name, pedestals[1].name),
                message="UR20 and SR-12iA pedestal volumes overlap",
            )
        )

    result = tuple(issues)
    return PlacementReport(valid=not result, issues=result)


def _pedestals(snapshot: SceneSnapshot) -> tuple[BoxPrimitive, BoxPrimitive]:
    by_name: dict[str, list[BoxPrimitive]] = {name: [] for name in _PEDESTAL_NAMES}
    for box in snapshot.boxes:
        if box.name in by_name:
            by_name[box.name].append(box)
    for name, matches in by_name.items():
        if len(matches) != 1:
            raise ValueError(
                f"snapshot must contain exactly one {name}; found {len(matches)}"
            )
        pedestal = matches[0]
        if pedestal.role != "pedestal":
            raise ValueError(f"snapshot box {name} must have role 'pedestal'")
        if pedestal.center_m[2] - pedestal.size_m[2] / 2.0 < -_CONTACT_TOLERANCE_M:
            raise ValueError(f"{name} extends below the floor")
    return by_name[_PEDESTAL_NAMES[0]][0], by_name[_PEDESTAL_NAMES[1]][0]


def _inside_installation_region(
    spec: SceneSpec,
    pedestal: BoxPrimitive,
    clearance: float,
) -> bool:
    centre = np.asarray(pedestal.center_m[:2], dtype=float)
    half_x = pedestal.size_m[0] / 2.0
    half_y = pedestal.size_m[1] / 2.0
    axes = _rectangle_axes(pedestal.yaw_deg)
    extent_x = abs(axes[0][0]) * half_x + abs(axes[1][0]) * half_y
    extent_y = abs(axes[0][1]) * half_x + abs(axes[1][1]) * half_y
    lower = np.asarray(spec.installation_region.raw_xy_min_m, dtype=float) + clearance
    upper = np.asarray(spec.installation_region.raw_xy_max_m, dtype=float) - clearance
    return bool(
        centre[0] - extent_x >= lower[0] - _CONTACT_TOLERANCE_M
        and centre[0] + extent_x <= upper[0] + _CONTACT_TOLERANCE_M
        and centre[1] - extent_y >= lower[1] - _CONTACT_TOLERANCE_M
        and centre[1] + extent_y <= upper[1] + _CONTACT_TOLERANCE_M
    )


def _obb_overlap_3d(first: BoxPrimitive, second: BoxPrimitive) -> bool:
    if not _z_intervals_overlap(first, second):
        return False
    return _rectangles_overlap_sat(first, second)


def _z_intervals_overlap(first: BoxPrimitive, second: BoxPrimitive) -> bool:
    first_low = first.center_m[2] - first.size_m[2] / 2.0
    first_high = first.center_m[2] + first.size_m[2] / 2.0
    second_low = second.center_m[2] - second.size_m[2] / 2.0
    second_high = second.center_m[2] + second.size_m[2] / 2.0
    overlap = min(first_high, second_high) - max(first_low, second_low)
    return overlap > _CONTACT_TOLERANCE_M


def _rectangles_overlap_sat(first: BoxPrimitive, second: BoxPrimitive) -> bool:
    first_axes = _rectangle_axes(first.yaw_deg)
    second_axes = _rectangle_axes(second.yaw_deg)
    first_half = np.asarray(first.size_m[:2], dtype=float) / 2.0
    second_half = np.asarray(second.size_m[:2], dtype=float) / 2.0
    centre_delta = np.asarray(second.center_m[:2], dtype=float) - np.asarray(
        first.center_m[:2],
        dtype=float,
    )

    for separating_axis in (*first_axes, *second_axes):
        distance = abs(float(np.dot(centre_delta, separating_axis)))
        first_radius = sum(
            first_half[index]
            * abs(float(np.dot(first_axes[index], separating_axis)))
            for index in range(2)
        )
        second_radius = sum(
            second_half[index]
            * abs(float(np.dot(second_axes[index], separating_axis)))
            for index in range(2)
        )
        penetration = first_radius + second_radius - distance
        if penetration <= _CONTACT_TOLERANCE_M:
            return False
    return True


def _rectangle_axes(yaw_deg: float) -> tuple[np.ndarray, np.ndarray]:
    yaw = math.radians(float(yaw_deg))
    cosine = math.cos(yaw)
    sine = math.sin(yaw)
    return (
        np.array((cosine, sine), dtype=float),
        np.array((-sine, cosine), dtype=float),
    )


def _inflate_box(box: BoxPrimitive, margin: float) -> BoxPrimitive:
    if margin == 0.0:
        return box
    return BoxPrimitive(
        name=box.name,
        role=box.role,
        center_m=box.center_m,
        size_m=tuple(float(size + 2.0 * margin) for size in box.size_m),
        yaw_deg=box.yaw_deg,
        collision_enabled=box.collision_enabled,
    )
