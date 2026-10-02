from __future__ import annotations

from dataclasses import replace
import math

import pytest

from decanting_workspace.models import SceneState, load_scene_spec
from decanting_workspace.scene import (
    materialize_scene,
    tote_long_axis_offset_range_m,
    tote_motion_axis_world_xy,
)


def _with_support_axes(spec, *, yaw_deg, swap_dimensions=False):
    support_name = spec.tote.motion_support_box
    boxes = []
    for box in spec.static_boxes:
        if box.name == support_name:
            size = box.size_m
            if swap_dimensions:
                size = size[1], size[0], size[2]
            box = replace(box, size_m=size, yaw_deg=yaw_deg)
        boxes.append(box)
    return replace(spec, static_boxes=tuple(boxes))


@pytest.mark.parametrize("yaw_deg,swap_dimensions", [(90., True), (180., False), (270., True)])
def test_equivalent_support_axes_preserve_signed_offsets_and_asymmetric_limits(
    yaw_deg,
    swap_dimensions,
):
    original = load_scene_spec()
    equivalent = _with_support_axes(
        original,
        yaw_deg=yaw_deg,
        swap_dimensions=swap_dimensions,
    )

    assert tote_motion_axis_world_xy(equivalent) == pytest.approx((1., 0.))
    expected_limits = tote_long_axis_offset_range_m(original)
    assert abs(expected_limits[0]) != pytest.approx(abs(expected_limits[1]))
    assert tote_long_axis_offset_range_m(equivalent) == pytest.approx(expected_limits)

    offset = .10
    snapshot = materialize_scene(equivalent, SceneState(tote_long_axis_offset_m=offset))
    frame_name = original.tote.representative_frame
    frame = snapshot.frames[frame_name]
    source = original.frames[frame_name]
    assert frame.translation_m == pytest.approx(
        (source.translation_m[0] + offset, source.translation_m[1], source.translation_m[2])
    )
    for endpoint in expected_limits:
        materialize_scene(equivalent, SceneState(tote_long_axis_offset_m=endpoint))
    with pytest.raises(ValueError, match="tote"):
        materialize_scene(
            equivalent,
            SceneState(tote_long_axis_offset_m=expected_limits[1] + 1e-6),
        )


@pytest.mark.parametrize("yaw_deg", [90., 270.])
def test_perpendicular_support_axis_points_toward_positive_world_y(yaw_deg):
    spec = _with_support_axes(load_scene_spec(), yaw_deg=yaw_deg)
    assert tote_motion_axis_world_xy(spec) == pytest.approx((0., 1.))


def test_oblique_support_axis_is_invariant_under_half_turn():
    original = load_scene_spec()
    first = _with_support_axes(original, yaw_deg=-30.)
    equivalent = _with_support_axes(original, yaw_deg=150.)
    assert tote_motion_axis_world_xy(first) == pytest.approx(
        (math.cos(math.radians(30.)), -.5)
    )
    assert tote_motion_axis_world_xy(equivalent) == pytest.approx(tote_motion_axis_world_xy(first))
    assert tote_long_axis_offset_range_m(equivalent) == pytest.approx(tote_long_axis_offset_range_m(first))
