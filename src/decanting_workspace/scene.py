"""Materialize one immutable scene snapshot from config and scenario state."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping

from .models import (
    BasePose,
    BoxPrimitive,
    FrameSpec,
    SceneSpec,
    SceneState,
)
from .transforms import transform_xy


CORNER_SIGNS: Mapping[str, tuple[float, float]] = {
    "southwest": (-1.0, -1.0),
    "southeast": (1.0, -1.0),
    "northwest": (-1.0, 1.0),
    "northeast": (1.0, 1.0),
}


@dataclass(frozen=True)
class SceneSnapshot:
    """Viewer- and collision-backend-independent scene for one scenario."""

    boxes: tuple[BoxPrimitive, ...]
    frames: Mapping[str, FrameSpec]
    ur_base: BasePose
    sr_base: BasePose
    state: SceneState

    @property
    def collision_boxes(self) -> tuple[BoxPrimitive, ...]:
        return tuple(box for box in self.boxes if box.collision_enabled)


def sr_cutting_workspace_footprint_box(
    spec: SceneSpec,
    base: BasePose,
    clearance_m: float | None = None,
    visual_thickness_m: float = 0.008,
) -> BoxPrimitive:
    """Return the SR cutting-task footprint on its support cube.

    This thin guide is the XY box-cutting target domain, not the SCARA arm's
    full swept envelope.  It follows the SR pedestal top and can be inset by a
    boundary clearance.  Cutting Z is intentionally absent: later task poses
    obtain it from the box top and verify it with J3 limits and collision-aware
    IK.

    Reachability, IK, robot collision geometry, and link sweep remain separate
    constraints; this footprint does not claim that every point is feasible.
    """

    robot = spec.robots["sr12ia"]
    workspace = robot.workspace
    if workspace is None:
        raise ValueError("SR-12iA workspace specification is missing")
    if base.z_m <= 0.0:
        raise ValueError("SR-12iA cutting workspace requires a positive base height")
    if not math.isfinite(visual_thickness_m) or visual_thickness_m <= 0.0:
        raise ValueError("SR cutting-workspace visual thickness must be positive")

    clearance = (
        workspace.safety_clearance_m
        if clearance_m is None
        else float(clearance_m)
    )
    if not math.isfinite(clearance) or clearance < 0.0:
        raise ValueError("SR cutting-workspace clearance must be finite and non-negative")

    size_x = robot.pedestal.size_xy_m[0] - 2.0 * clearance
    size_y = robot.pedestal.size_xy_m[1] - 2.0 * clearance
    if size_x <= 0.0 or size_y <= 0.0:
        raise ValueError("SR cutting-workspace clearance consumes the pedestal footprint")

    center_x, center_y = transform_xy(
        robot.pedestal.center_offset_local_xy_m,
        base.xyz_m[:2],
        base.yaw_rad,
    )
    return BoxPrimitive(
        name="sr12ia_cutting_workspace_footprint",
        role="task_workspace",
        center_m=(center_x, center_y, base.z_m + visual_thickness_m / 2.0),
        size_m=(size_x, size_y, visual_thickness_m),
        yaw_deg=base.yaw_deg,
        collision_enabled=False,
    )


def tote_motion_axis_world_xy(spec: SceneSpec) -> tuple[float, float]:
    """Return the supporting worktable's unit long-axis direction in World XY."""

    support = _tote_motion_support_box(spec)
    yaw = math.radians(support.yaw_deg)
    if support.size_m[0] >= support.size_m[1]:
        return math.cos(yaw), math.sin(yaw)
    return -math.sin(yaw), math.cos(yaw)


def tote_long_axis_offset_range_m(spec: SceneSpec) -> tuple[float, float]:
    """Return offsets that keep the rotated tote within the table's long edges."""

    support = _tote_motion_support_box(spec)
    tote = spec.tote
    frame = spec.frames[tote.representative_frame]
    support_yaw = math.radians(support.yaw_deg)
    c = math.cos(support_yaw)
    s = math.sin(support_yaw)
    dx = frame.translation_m[0] - support.center_m[0]
    dy = frame.translation_m[1] - support.center_m[1]
    frame_local = (c * dx + s * dy, -s * dx + c * dy)
    long_axis_index = 0 if support.size_m[0] >= support.size_m[1] else 1

    relative_yaw = math.radians(tote.representative_yaw_deg - support.yaw_deg)
    if long_axis_index == 0:
        tote_half_extent = (
            abs(math.cos(relative_yaw)) * tote.size_m[0] / 2.0
            + abs(math.sin(relative_yaw)) * tote.size_m[1] / 2.0
        )
    else:
        tote_half_extent = (
            abs(math.sin(relative_yaw)) * tote.size_m[0] / 2.0
            + abs(math.cos(relative_yaw)) * tote.size_m[1] / 2.0
        )
    support_half_extent = support.size_m[long_axis_index] / 2.0
    initial_coordinate = frame_local[long_axis_index]
    return (
        -support_half_extent + tote_half_extent - initial_coordinate,
        support_half_extent - tote_half_extent - initial_coordinate,
    )


def materialize_scene(
    spec: SceneSpec,
    state: SceneState | None = None,
    *,
    ur_base: BasePose | None = None,
    sr_base: BasePose | None = None,
    sr_module_camera_mode: str | None = None,
) -> SceneSnapshot:
    """Create geometry for a lift/SKU/base candidate without mutating the spec."""

    scenario = state or SceneState()
    ur = ur_base or spec.robots["ur20"].nominal_base
    sr = sr_base or spec.robots["sr12ia"].nominal_base
    _validate_state(spec, scenario)
    frames = _scenario_frames(spec, scenario)

    boxes: list[BoxPrimitive] = list(spec.static_boxes)
    boxes.extend(_pallet_and_samples(spec, scenario))
    if scenario.tote_present:
        boxes.append(_representative_tote(spec, frames))
    boxes.append(_pedestal_box("ur20_pedestal", ur, spec.robots["ur20"].pedestal))
    boxes.append(_pedestal_box("sr12ia_pedestal", sr, spec.robots["sr12ia"].pedestal))

    sr_spec = spec.robots["sr12ia"]
    camera_mode = sr_module_camera_mode or sr_spec.module_camera_mode
    if camera_mode not in {"fixed", "with_base"}:
        raise ValueError("sr_module_camera_mode must be 'fixed' or 'with_base'")
    camera_base = sr if camera_mode == "with_base" else sr_spec.nominal_base
    for local_box in sr_spec.module_camera_boxes_local:
        x_m, y_m = transform_xy(
            local_box.center_m[:2], camera_base.xyz_m[:2], camera_base.yaw_rad
        )
        boxes.append(
            BoxPrimitive(
                name=local_box.name,
                role=local_box.role,
                center_m=(x_m, y_m, local_box.center_m[2]),
                size_m=local_box.size_m,
                yaw_deg=camera_base.yaw_deg + local_box.yaw_deg,
                collision_enabled=local_box.collision_enabled,
            )
        )

    return SceneSnapshot(
        boxes=tuple(boxes),
        frames=frames,
        ur_base=ur,
        sr_base=sr,
        state=scenario,
    )


def _pallet_and_samples(spec: SceneSpec, state: SceneState) -> list[BoxPrimitive]:
    pallet = spec.pallet
    pallet_height = pallet.size_m[2]
    pallet_center = (
        pallet.center_xy_m[0],
        pallet.center_xy_m[1],
        state.lift_height_m + pallet_height / 2.0,
    )
    result = [
        BoxPrimitive(
            name="pallet",
            role="pallet",
            center_m=pallet_center,
            size_m=pallet.size_m,
        )
    ]

    sku = spec.box_skus[state.sku]
    box_bottom = state.lift_height_m + pallet_height
    all_alternatives = pallet.corner_samples_are_alternatives and len(state.corner_samples) > 1
    for corner in state.corner_samples:
        sign_x, sign_y = CORNER_SIGNS[corner]
        x_offset = pallet.size_m[0] / 2.0 - sku.size_m[0] / 2.0 - pallet.corner_margin_m
        y_offset = pallet.size_m[1] / 2.0 - sku.size_m[1] / 2.0 - pallet.corner_margin_m
        result.append(
            BoxPrimitive(
                name=f"box_{state.sku}_{corner}",
                role="box_sample_alternative" if all_alternatives else "box",
                center_m=(
                    pallet.center_xy_m[0] + sign_x * x_offset,
                    pallet.center_xy_m[1] + sign_y * y_offset,
                    box_bottom + sku.size_m[2] / 2.0,
                ),
                size_m=sku.size_m,
                collision_enabled=not all_alternatives,
            )
        )
    return result


def _representative_tote(
    spec: SceneSpec,
    frames: Mapping[str, FrameSpec],
) -> BoxPrimitive:
    tote = spec.tote
    frame = frames[tote.representative_frame]
    return BoxPrimitive(
        name="representative_tote",
        role="tote",
        center_m=(
            frame.translation_m[0],
            frame.translation_m[1],
            tote.support_surface_z_m + tote.size_m[2] / 2.0,
        ),
        size_m=tote.size_m,
        yaw_deg=tote.representative_yaw_deg,
    )


def _scenario_frames(
    spec: SceneSpec,
    state: SceneState,
) -> Mapping[str, FrameSpec]:
    frames = dict(spec.frames)
    tote = spec.tote
    source = frames[tote.representative_frame]
    axis_x, axis_y = tote_motion_axis_world_xy(spec)
    offset = state.tote_long_axis_offset_m
    frames[tote.representative_frame] = FrameSpec(
        name=source.name,
        translation_m=(
            source.translation_m[0] + offset * axis_x,
            source.translation_m[1] + offset * axis_y,
            source.translation_m[2],
        ),
        rotation_xyzw=source.rotation_xyzw,
    )
    return frames


def _tote_motion_support_box(spec: SceneSpec) -> BoxPrimitive:
    try:
        return next(
            box
            for box in spec.static_boxes
            if box.name == spec.tote.motion_support_box
        )
    except StopIteration as exc:
        raise ValueError(
            f"unknown tote motion support box: {spec.tote.motion_support_box}"
        ) from exc


def _pedestal_box(name: str, base: BasePose, pedestal: object) -> BoxPrimitive:
    if base.z_m <= 0.0:
        raise ValueError(f"{name} requires a positive base height")
    size_xy = pedestal.size_xy_m
    offset_xy = pedestal.center_offset_local_xy_m
    center_x, center_y = transform_xy(offset_xy, base.xyz_m[:2], base.yaw_rad)
    return BoxPrimitive(
        name=name,
        role="pedestal",
        center_m=(center_x, center_y, base.z_m / 2.0),
        size_m=(size_xy[0], size_xy[1], base.z_m),
        yaw_deg=base.yaw_deg,
    )


def _validate_state(spec: SceneSpec, state: SceneState) -> None:
    low, high = spec.pallet.lift_range_m
    if not math.isfinite(state.lift_height_m) or not low <= state.lift_height_m <= high:
        raise ValueError(f"lift height must be in [{low}, {high}] m")
    if state.sku not in spec.box_skus:
        raise ValueError(f"unknown box SKU: {state.sku}")
    if not state.corner_samples:
        raise ValueError("at least one corner sample is required")
    unknown = set(state.corner_samples) - set(CORNER_SIGNS)
    if unknown:
        raise ValueError(f"unknown pallet corners: {sorted(unknown)}")
    sr_workspace = spec.robots["sr12ia"].workspace
    if sr_workspace is None:
        raise ValueError("SR-12iA workspace specification is missing")
    if not any(abs(state.sr_j3_stroke_m - option) <= 1e-9 for option in sr_workspace.j3_stroke_options_m):
        raise ValueError(
            f"SR J3 stroke must be one of {sr_workspace.j3_stroke_options_m} m"
        )
    if state.clearance_m is not None and (
        not math.isfinite(state.clearance_m) or state.clearance_m < 0.0
    ):
        raise ValueError("clearance must be finite and non-negative")
    tote_offset = state.tote_long_axis_offset_m
    tote_low, tote_high = tote_long_axis_offset_range_m(spec)
    if (
        not math.isfinite(tote_offset)
        or tote_offset < tote_low - 1e-12
        or tote_offset > tote_high + 1e-12
    ):
        raise ValueError(
            "tote long-axis offset must be in "
            f"[{tote_low}, {tote_high}] m"
        )
