from __future__ import annotations

import pytest

from decanting_workspace import load_scene_spec
from decanting_workspace.viewer import CellViewer


@pytest.fixture(scope="module")
def spec():
    return load_scene_spec()


def test_corrected_dimensions_and_ranges(spec):
    assert spec.pallet.size_m == pytest.approx((1.35, 1.0, 0.15))
    assert spec.pallet.lift_range_m == pytest.approx((0.0, 0.760))
    assert spec.tote.size_m == pytest.approx((0.660, 0.440, 0.300))
    assert len(spec.box_skus) == 5
    assert spec.box_skus["049995"].size_m == pytest.approx((0.517, 0.503, 0.545))
    assert spec.robots["sr12ia"].workspace.total_height_options_m == pytest.approx(
        (0.918, 1.068)
    )
    assert spec.robots["ur20"].suction_proxy is not None
    assert spec.robots["ur20"].suction_proxy.pad_radius_m == pytest.approx(0.075)
    assert spec.robots["sr12ia"].cutter_proxy is not None
    assert spec.robots["sr12ia"].cutter_proxy.size_m == pytest.approx(
        (0.250, 0.090, 0.110)
    )
    cutter_radius, cutter_z_min, cutter_z_max = CellViewer._cutter_swept_extents(
        spec.robots["sr12ia"].cutter_proxy
    )
    assert cutter_radius == pytest.approx(
        ((0.120 + 0.125) ** 2 + 0.045**2) ** 0.5
    )
    assert (cutter_z_min, cutter_z_max) == pytest.approx((-0.110, 0.0))


def test_fixed_usd_proxy_inventory(spec):
    names = {box.name for box in spec.static_boxes}
    assert names == {
        "conveyor_level_1",
        "conveyor_level_2",
        "conveyor_level_3",
        "worktable_far_a",
        "worktable_far_b",
        "worktable_current_a",
        "worktable_current_b",
        "pallet_camera_pole",
        "pallet_camera_arm",
        "tote_camera_pole",
    }
    level_1 = next(box for box in spec.static_boxes if box.name == "conveyor_level_1")
    assert level_1.center_m == pytest.approx((0.7, -5.542254925, 0.61))
    assert level_1.size_m == pytest.approx((0.669039, 9.2, 0.10))


def test_provisional_installation_strip(spec):
    region = spec.installation_region
    assert region.raw_xy_min_m == pytest.approx((-1.477362823, -7.959995))
    assert region.raw_xy_max_m == pytest.approx((0.3654805, -6.663875))
    assert region.raw_xy_max_m[1] - region.raw_xy_min_m[1] == pytest.approx(1.29612)


def test_required_task_frames_are_present(spec):
    assert {
        "ToteLoadFrame",
        "TotePickupFrame",
        "ToteSupplyFrame",
        "UncasingLoadFrame",
        "WasteFrame",
    } <= set(spec.frames)
