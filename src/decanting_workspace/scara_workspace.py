"""Conservative task-workspace checks for SR-12iA box cutting.

This module deliberately evaluates the configured cutting task domain rather
than a manipulability score.  The SCARA is accepted only when every corner of
the real SKU's placed top face is inside the pedestal-top task footprint, has
at least one joint-limit-valid planar 2R inverse-kinematics branch, and lies in
the selected J3 option's vertical range.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .models import BasePose, SceneSpec, Vec2, Vec3
from .scene import SceneSnapshot, sr_cutting_workspace_footprint_box
from .transforms import quaternion_matrix


_ANGLE_TOLERANCE_RAD = 1e-10
_POSITION_TOLERANCE_M = 1e-10


@dataclass(frozen=True)
class ScaraTargetCheck:
    """Feasibility result for one corner of the placed box's top face."""

    name: str
    world_position_m: Vec3
    base_local_position_m: Vec3
    footprint_local_xy_m: Vec2
    required_j3_m: float
    planar_ik_solutions_rad: tuple[tuple[float, float], ...]
    footprint_ok: bool
    planar_reach_ok: bool
    vertical_ok: bool
    feasible: bool

    @property
    def position_m(self) -> Vec3:
        """Alias for callers that treat each check as a generic target."""

        return self.world_position_m


@dataclass(frozen=True)
class ScaraTaskWorkspaceReport:
    """Aggregate cutting feasibility for all four SKU top-face corners."""

    sku: str
    box_size_m: Vec3
    support_frame_name: str
    clearance_m: float
    j3_stroke_m: float
    tool0_z_range_m: tuple[float, float]
    targets: tuple[ScaraTargetCheck, ...]
    footprint_ok: bool
    planar_reach_ok: bool
    vertical_ok: bool
    feasible: bool

    @property
    def checks(self) -> tuple[ScaraTargetCheck, ...]:
        """Backward-friendly descriptive alias for :attr:`targets`."""

        return self.targets


def evaluate_sr_cutting_workspace(
    spec: SceneSpec,
    snapshot: SceneSnapshot,
    *,
    sku: str | None = None,
    clearance_m: float | None = None,
) -> ScaraTaskWorkspaceReport:
    """Evaluate conservative SR-12iA feasibility for the placed box top.

    ``UncasingLoadFrame`` is interpreted as the box support plane.  Its full
    rotation defines the placed box axes, and the SKU dimensions from the
    configuration define the four top corners; no USD box dimensions enter
    the calculation.  J3 is measured downward from the SCARA arm plane, so
    ``required_j3_m = base_z + arm_plane_height - target_z``.  There is no
    process-tool offset.

    Clearance precedence is the explicit argument, then the snapshot state's
    value, then the configured SR workspace safety clearance.
    """

    robot = spec.robots.get("sr12ia")
    if robot is None or robot.workspace is None:
        raise ValueError("SR-12iA workspace specification is missing")
    workspace = robot.workspace

    selected_sku = snapshot.state.sku if sku is None else str(sku)
    try:
        box = spec.box_skus[selected_sku]
    except KeyError as exc:
        raise ValueError(f"unknown box SKU: {selected_sku}") from exc

    frame_name = "UncasingLoadFrame"
    try:
        support_frame = snapshot.frames[frame_name]
    except KeyError as exc:
        raise ValueError(f"required frame is missing: {frame_name}") from exc

    clearance = _resolve_clearance(spec, snapshot, clearance_m)
    footprint = sr_cutting_workspace_footprint_box(
        spec,
        snapshot.sr_base,
        clearance_m=clearance,
    )

    stroke = float(snapshot.state.sr_j3_stroke_m)
    if not math.isfinite(stroke) or stroke <= 0.0:
        raise ValueError("SR J3 stroke must be finite and positive")
    if not any(
        math.isclose(stroke, option, rel_tol=0.0, abs_tol=1e-9)
        for option in workspace.j3_stroke_options_m
    ):
        raise ValueError(
            f"SR J3 stroke must be one of {workspace.j3_stroke_options_m} m"
        )

    tool0_z_max = snapshot.sr_base.z_m + workspace.arm_plane_height_m
    tool0_z_min = tool0_z_max - stroke
    world_from_box = quaternion_matrix(
        support_frame.translation_m,
        support_frame.rotation_xyzw,
    )

    targets: list[ScaraTargetCheck] = []
    half_x = box.size_m[0] / 2.0
    half_y = box.size_m[1] / 2.0
    for name, sign_x, sign_y in (
        ("negative_x_negative_y", -1.0, -1.0),
        ("positive_x_negative_y", 1.0, -1.0),
        ("negative_x_positive_y", -1.0, 1.0),
        ("positive_x_positive_y", 1.0, 1.0),
    ):
        corner_local = np.array(
            (sign_x * half_x, sign_y * half_y, box.size_m[2], 1.0),
            dtype=float,
        )
        corner_world_array = world_from_box @ corner_local
        corner_world = tuple(float(value) for value in corner_world_array[:3])
        corner_base = _world_to_base(corner_world, snapshot.sr_base)
        footprint_local = _world_to_oriented_box_xy(
            corner_world,
            footprint.center_m,
            math.radians(footprint.yaw_deg),
        )

        footprint_ok = (
            abs(footprint_local[0])
            <= footprint.size_m[0] / 2.0 + _POSITION_TOLERANCE_M
            and abs(footprint_local[1])
            <= footprint.size_m[1] / 2.0 + _POSITION_TOLERANCE_M
        )
        ik_solutions = _planar_ik_solutions(
            corner_base[:2],
            workspace.link_lengths_m,
            workspace.q1_range_deg,
            workspace.q2_range_deg,
        )
        required_j3 = tool0_z_max - corner_world[2]
        vertical_ok = (
            required_j3 >= -_POSITION_TOLERANCE_M
            and required_j3 <= stroke + _POSITION_TOLERANCE_M
        )
        planar_ok = bool(ik_solutions)
        targets.append(
            ScaraTargetCheck(
                name=name,
                world_position_m=corner_world,
                base_local_position_m=corner_base,
                footprint_local_xy_m=footprint_local,
                required_j3_m=required_j3,
                planar_ik_solutions_rad=ik_solutions,
                footprint_ok=footprint_ok,
                planar_reach_ok=planar_ok,
                vertical_ok=vertical_ok,
                feasible=footprint_ok and planar_ok and vertical_ok,
            )
        )

    target_tuple = tuple(targets)
    footprint_ok = all(target.footprint_ok for target in target_tuple)
    planar_ok = all(target.planar_reach_ok for target in target_tuple)
    vertical_ok = all(target.vertical_ok for target in target_tuple)
    return ScaraTaskWorkspaceReport(
        sku=selected_sku,
        box_size_m=box.size_m,
        support_frame_name=frame_name,
        clearance_m=clearance,
        j3_stroke_m=stroke,
        tool0_z_range_m=(tool0_z_min, tool0_z_max),
        targets=target_tuple,
        footprint_ok=footprint_ok,
        planar_reach_ok=planar_ok,
        vertical_ok=vertical_ok,
        feasible=footprint_ok and planar_ok and vertical_ok,
    )


def _resolve_clearance(
    spec: SceneSpec,
    snapshot: SceneSnapshot,
    explicit_clearance_m: float | None,
) -> float:
    workspace = spec.robots["sr12ia"].workspace
    assert workspace is not None
    if explicit_clearance_m is not None:
        clearance = float(explicit_clearance_m)
    elif snapshot.state.clearance_m is not None:
        clearance = float(snapshot.state.clearance_m)
    else:
        clearance = float(workspace.safety_clearance_m)
    if not math.isfinite(clearance) or clearance < 0.0:
        raise ValueError("SR cutting-workspace clearance must be finite and non-negative")
    return clearance


def _world_to_base(world: Vec3, base: BasePose) -> Vec3:
    dx = world[0] - base.x_m
    dy = world[1] - base.y_m
    c = math.cos(base.yaw_rad)
    s = math.sin(base.yaw_rad)
    return (
        c * dx + s * dy,
        -s * dx + c * dy,
        world[2] - base.z_m,
    )


def _world_to_oriented_box_xy(
    world: Vec3,
    center: Vec3,
    yaw_rad: float,
) -> Vec2:
    dx = world[0] - center[0]
    dy = world[1] - center[1]
    c = math.cos(yaw_rad)
    s = math.sin(yaw_rad)
    return c * dx + s * dy, -s * dx + c * dy


def _planar_ik_solutions(
    target_xy_m: Vec2,
    link_lengths_m: Vec2,
    q1_range_deg: Vec2,
    q2_range_deg: Vec2,
) -> tuple[tuple[float, float], ...]:
    """Return all unique 2R branches satisfying both configured ranges."""

    x_m, y_m = target_xy_m
    link_1, link_2 = link_lengths_m
    cosine_q2 = (
        x_m * x_m + y_m * y_m - link_1 * link_1 - link_2 * link_2
    ) / (2.0 * link_1 * link_2)
    if cosine_q2 < -1.0 - _ANGLE_TOLERANCE_RAD or cosine_q2 > 1.0 + _ANGLE_TOLERANCE_RAD:
        return ()
    cosine_q2 = min(1.0, max(-1.0, cosine_q2))

    q1_low, q1_high = (math.radians(value) for value in q1_range_deg)
    q2_low, q2_high = (math.radians(value) for value in q2_range_deg)
    if q1_low > q1_high or q2_low > q2_high:
        raise ValueError("SCARA joint-range lower bounds must not exceed upper bounds")

    principal_q2 = math.acos(cosine_q2)
    solutions: list[tuple[float, float]] = []
    for q2_base in (principal_q2, -principal_q2):
        q1_base = math.atan2(y_m, x_m) - math.atan2(
            link_2 * math.sin(q2_base),
            link_1 + link_2 * math.cos(q2_base),
        )
        for q1 in _equivalent_angles_in_range(q1_base, q1_low, q1_high):
            for q2 in _equivalent_angles_in_range(q2_base, q2_low, q2_high):
                solution = (q1, q2)
                if not any(
                    abs(q1 - prior[0]) <= _ANGLE_TOLERANCE_RAD
                    and abs(q2 - prior[1]) <= _ANGLE_TOLERANCE_RAD
                    for prior in solutions
                ):
                    solutions.append(solution)
    return tuple(solutions)


def _equivalent_angles_in_range(
    angle_rad: float,
    lower_rad: float,
    upper_rad: float,
) -> tuple[float, ...]:
    period = 2.0 * math.pi
    first_turn = math.ceil((lower_rad - angle_rad - _ANGLE_TOLERANCE_RAD) / period)
    last_turn = math.floor((upper_rad - angle_rad + _ANGLE_TOLERANCE_RAD) / period)
    return tuple(
        angle_rad + turn * period
        for turn in range(first_turn, last_turn + 1)
        if lower_rad - _ANGLE_TOLERANCE_RAD
        <= angle_rad + turn * period
        <= upper_rad + _ANGLE_TOLERANCE_RAD
    )
