from __future__ import annotations

from dataclasses import replace

from decanting_workspace import BasePose, BoxPrimitive, load_scene_spec, materialize_scene
from decanting_workspace.placement import validate_base_placement


def _issue_codes(report):
    return {issue.code for issue in report.issues}


def test_nominal_base_placement_is_valid():
    spec = load_scene_spec()
    snapshot = materialize_scene(spec)

    report = validate_base_placement(spec, snapshot)

    assert report.valid
    assert report.issues == ()


def test_pedestal_footprint_outside_raw_installation_rectangle_is_rejected():
    spec = load_scene_spec()
    nominal = spec.robots["ur20"].nominal_base
    outside = BasePose(
        x_m=spec.installation_region.raw_xy_min_m[0],
        y_m=nominal.y_m,
        z_m=nominal.z_m,
        yaw_deg=nominal.yaw_deg,
    )
    snapshot = materialize_scene(spec, ur_base=outside)

    report = validate_base_placement(spec, snapshot)

    assert not report.valid
    assert "outside_installation_region" in _issue_codes(report)
    assert any(issue.objects == ("ur20_pedestal",) for issue in report.issues)


def test_conveyor_volume_overlap_is_reported():
    spec = load_scene_spec()
    nominal = spec.robots["ur20"].nominal_base
    against_conveyor = BasePose(
        x_m=0.30,
        y_m=nominal.y_m,
        z_m=nominal.z_m,
        yaw_deg=nominal.yaw_deg,
    )
    snapshot = materialize_scene(spec, ur_base=against_conveyor)

    report = validate_base_placement(spec, snapshot)

    assert not report.valid
    assert any(
        issue.code == "static_box_overlap"
        and issue.objects == ("ur20_pedestal", "conveyor_level_1")
        for issue in report.issues
    )


def test_ur_and_sr_pedestal_overlap_is_reported():
    spec = load_scene_spec()
    sr_pedestal = next(
        box for box in materialize_scene(spec).boxes if box.name == "sr12ia_pedestal"
    )
    nominal = spec.robots["ur20"].nominal_base
    overlapping_ur = BasePose(
        x_m=sr_pedestal.center_m[0],
        y_m=sr_pedestal.center_m[1],
        z_m=nominal.z_m,
        yaw_deg=nominal.yaw_deg,
    )
    snapshot = materialize_scene(spec, ur_base=overlapping_ur)

    report = validate_base_placement(spec, snapshot)

    assert not report.valid
    assert "pedestal_overlap" in _issue_codes(report)


def test_static_overlap_depends_on_candidate_mounting_height():
    spec = load_scene_spec()
    nominal = spec.robots["ur20"].nominal_base
    overhead = BoxPrimitive(
        name="height_probe",
        role="obstacle",
        center_m=(nominal.x_m, nominal.y_m, 0.60),
        size_m=(0.10, 0.10, 0.10),
    )
    tote_support = next(
        box for box in spec.static_boxes if box.name == spec.tote.motion_support_box
    )
    isolated_spec = replace(spec, static_boxes=(tote_support, overhead))
    low_base = replace(nominal, z_m=0.50)
    high_base = replace(nominal, z_m=0.60)

    low_report = validate_base_placement(
        isolated_spec,
        materialize_scene(isolated_spec, ur_base=low_base),
    )
    high_report = validate_base_placement(
        isolated_spec,
        materialize_scene(isolated_spec, ur_base=high_base),
    )

    assert low_report.valid
    assert not high_report.valid
    assert any(
        issue.code == "static_box_overlap"
        and issue.objects == ("ur20_pedestal", "height_probe")
        for issue in high_report.issues
    )


def test_touching_static_box_is_allowed_but_positive_clearance_rejects_it():
    spec = load_scene_spec()
    nominal = spec.robots["ur20"].nominal_base
    touching = BoxPrimitive(
        name="touching_probe",
        role="obstacle",
        center_m=(nominal.x_m + 0.20, nominal.y_m, 0.25),
        size_m=(0.10, 0.10, 0.50),
    )
    tote_support = next(
        box for box in spec.static_boxes if box.name == spec.tote.motion_support_box
    )
    isolated_spec = replace(spec, static_boxes=(tote_support, touching))
    snapshot = materialize_scene(isolated_spec)

    touching_report = validate_base_placement(isolated_spec, snapshot)
    clearance_report = validate_base_placement(
        isolated_spec,
        snapshot,
        clearance_m=0.01,
    )

    assert touching_report.valid
    assert not clearance_report.valid
    assert "static_box_overlap" in _issue_codes(clearance_report)
