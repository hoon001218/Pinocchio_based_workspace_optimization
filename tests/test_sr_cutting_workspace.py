from __future__ import annotations

from dataclasses import replace
import math

import pytest

from decanting_workspace import BasePose, load_scene_spec
from decanting_workspace.scene import sr_cutting_workspace_footprint_box
from decanting_workspace.viewer import CellViewer


class _RecordingNode:
    def __init__(self) -> None:
        self.object = None
        self.transform = None

    def set_object(self, value, *args) -> None:
        self.object = (value, args)

    def set_transform(self, value) -> None:
        self.transform = value


class _RecordingViewer:
    def __init__(self) -> None:
        self.nodes: dict[str, _RecordingNode] = {}

    def __getitem__(self, path: str) -> _RecordingNode:
        return self.nodes.setdefault(path, _RecordingNode())


@pytest.fixture(scope="module")
def spec():
    return load_scene_spec()


def _with_clearance(spec, clearance_m: float):
    sr = spec.robots["sr12ia"]
    assert sr.workspace is not None
    workspace = replace(sr.workspace, safety_clearance_m=clearance_m)
    robots = dict(spec.robots)
    robots["sr12ia"] = replace(sr, workspace=workspace)
    return replace(spec, robots=robots)


def test_cutting_workspace_uses_pedestal_top_not_full_scara_reach(spec):
    base = spec.robots["sr12ia"].nominal_base
    result = sr_cutting_workspace_footprint_box(spec, base)

    # This is the box-opening task plane, not the SCARA's full 0.9 m reach.
    assert result.name == "sr12ia_cutting_workspace_footprint"
    assert result.role == "task_workspace"
    assert result.center_m == pytest.approx(
        (-0.233218148, -7.358607684, 0.558805667)
    )
    assert result.size_m == pytest.approx((1.10, 0.75, 0.008))
    assert result.yaw_deg == pytest.approx(-90.0)
    assert not result.collision_enabled


def test_cutting_workspace_follows_candidate_translation_and_yaw(spec):
    base = BasePose(1.25, -2.0, 0.8, 30.0)
    result = sr_cutting_workspace_footprint_box(spec, base)

    pedestal_offset_x = spec.robots["sr12ia"].pedestal.center_offset_local_xy_m[0]
    expected_center_x = base.x_m + math.cos(base.yaw_rad) * pedestal_offset_x
    expected_center_y = base.y_m + math.sin(base.yaw_rad) * pedestal_offset_x
    assert result.center_m == pytest.approx(
        (expected_center_x, expected_center_y, 0.804)
    )
    assert result.size_m == pytest.approx((1.10, 0.75, 0.008))
    assert result.yaw_deg == pytest.approx(30.0)


def test_clearance_insets_each_pedestal_edge_without_cutter_radius_inset(spec):
    clearance_m = 0.05
    inset_spec = _with_clearance(spec, clearance_m)
    base = BasePose(0.3, -1.7, 0.7, 37.0)
    result = sr_cutting_workspace_footprint_box(inset_spec, base)
    pedestal = inset_spec.robots["sr12ia"].pedestal

    assert result.size_m == pytest.approx(
        (
            pedestal.size_xy_m[0] - 2.0 * clearance_m,
            pedestal.size_xy_m[1] - 2.0 * clearance_m,
            0.008,
        )
    )

    # Transform all guide corners back to pedestal coordinates.  Each must be
    # exactly one clearance inside the corresponding pedestal boundary.
    c = math.cos(base.yaw_rad)
    s = math.sin(base.yaw_rad)
    for sign_x in (-1.0, 1.0):
        for sign_y in (-1.0, 1.0):
            dx = sign_x * result.size_m[0] / 2.0
            dy = sign_y * result.size_m[1] / 2.0
            world_dx = c * dx - s * dy
            world_dy = s * dx + c * dy
            local_dx = c * world_dx + s * world_dy
            local_dy = -s * world_dx + c * world_dy
            assert abs(local_dx) == pytest.approx(
                pedestal.size_xy_m[0] / 2.0 - clearance_m
            )
            assert abs(local_dy) == pytest.approx(
                pedestal.size_xy_m[1] / 2.0 - clearance_m
            )


def test_cutting_workspace_rejects_empty_inset(spec):
    invalid_spec = _with_clearance(spec, 0.375)

    with pytest.raises(ValueError, match=r"(?i)(clearance|footprint|workspace)"):
        sr_cutting_workspace_footprint_box(
            invalid_spec, invalid_spec.robots["sr12ia"].nominal_base
        )


def test_nominal_uncasing_frame_is_inside_cutting_footprint_xy(spec):
    base = spec.robots["sr12ia"].nominal_base
    result = sr_cutting_workspace_footprint_box(spec, base)
    frame = spec.frames["UncasingLoadFrame"]
    dx = frame.translation_m[0] - base.x_m
    dy = frame.translation_m[1] - base.y_m
    c = math.cos(base.yaw_rad)
    s = math.sin(base.yaw_rad)
    local_x = c * dx + s * dy
    local_y = -s * dx + c * dy
    offset_x, offset_y = spec.robots["sr12ia"].pedestal.center_offset_local_xy_m

    assert abs(local_x - offset_x) <= result.size_m[0] / 2.0
    assert abs(local_y - offset_y) <= result.size_m[1] / 2.0

    # At nominal -90 deg yaw the pedestal's local Y span becomes world X,
    # while its local X span becomes world Y.
    assert (
        result.center_m[0] - result.size_m[1] / 2.0,
        result.center_m[0] + result.size_m[1] / 2.0,
    ) == pytest.approx((-0.608218148, 0.141781852))
    assert (
        result.center_m[1] - result.size_m[0] / 2.0,
        result.center_m[1] + result.size_m[0] / 2.0,
    ) == pytest.approx((-7.908607684, -6.808607684))


def test_viewer_replaces_full_scara_reach_guides_with_cutting_workspace(spec):
    recorder = _RecordingViewer()
    viewer = object.__new__(CellViewer)
    viewer.viewer = recorder
    viewer._robots = []

    viewer._render_sr_cutting_workspace(
        spec,
        spec.robots["sr12ia"].nominal_base,
        None,
        show_fill=True,
    )

    root = "decanting/guides/sr12ia/cutting_workspace"
    assert {f"{root}/boundary", f"{root}/volume"} <= set(recorder.nodes)
    obsolete_full_reach_guides = {
        "decanting/guides/sr12ia/reach_samples",
        "decanting/guides/sr12ia/arm_plane",
        "decanting/guides/sr12ia/lowest_tool_plane",
        "decanting/guides/sr12ia/conservative_swept_envelope",
    }
    assert obsolete_full_reach_guides.isdisjoint(recorder.nodes)
