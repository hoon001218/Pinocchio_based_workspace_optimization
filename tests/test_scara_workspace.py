from __future__ import annotations

from dataclasses import FrozenInstanceError, replace

import pytest

from decanting_workspace.models import SceneState, load_scene_spec
from decanting_workspace.scara_workspace import evaluate_sr_cutting_workspace
from decanting_workspace.scene import materialize_scene


@pytest.fixture(scope="module")
def spec():
    return load_scene_spec()


def test_nominal_box_top_is_xy_reachable_but_above_sr_tool0_plane(spec):
    snapshot = materialize_scene(spec)

    report = evaluate_sr_cutting_workspace(spec, snapshot)

    assert report.sku == "123591"
    assert report.box_size_m == spec.box_skus["123591"].size_m
    assert len(report.targets) == 4
    assert report.footprint_ok
    assert report.planar_reach_ok
    assert not report.vertical_ok
    assert not report.feasible
    assert report.tool0_z_range_m == pytest.approx(
        (0.590805667, 0.890805667)
    )
    assert all(
        target.world_position_m[2] == pytest.approx(0.671464995 + 0.327)
        for target in report.targets
    )
    assert all(target.required_j3_m == pytest.approx(-0.107659328) for target in report.targets)
    assert all(target.footprint_ok for target in report.targets)
    assert all(target.planar_ik_solutions_rad for target in report.targets)
    assert all(not target.vertical_ok for target in report.targets)


def test_reports_are_frozen(spec):
    report = evaluate_sr_cutting_workspace(spec, materialize_scene(spec))

    with pytest.raises(FrozenInstanceError):
        report.feasible = True
    with pytest.raises(FrozenInstanceError):
        report.targets[0].vertical_ok = True


def test_raised_sr_base_makes_nominal_sku_cutting_feasible(spec):
    nominal = spec.robots["sr12ia"].nominal_base
    raised = replace(nominal, z_m=0.8)
    snapshot = materialize_scene(spec, sr_base=raised)

    report = evaluate_sr_cutting_workspace(spec, snapshot)

    assert report.feasible
    assert report.footprint_ok
    assert report.planar_reach_ok
    assert report.vertical_ok
    assert all(target.feasible for target in report.targets)
    assert all(target.required_j3_m == pytest.approx(0.137535005) for target in report.targets)


def test_shifted_sr_base_rejects_box_outside_task_footprint(spec):
    nominal = spec.robots["sr12ia"].nominal_base
    shifted = replace(nominal, x_m=nominal.x_m + 1.0, z_m=0.8)
    snapshot = materialize_scene(spec, sr_base=shifted)

    report = evaluate_sr_cutting_workspace(spec, snapshot)

    assert not report.footprint_ok
    assert any(not target.footprint_ok for target in report.targets)
    assert not report.feasible


def test_450_mm_j3_option_reaches_target_that_300_mm_option_cannot(spec):
    # Put the top face 375 mm below the arm plane.  This cleanly separates the
    # two manufacturer stroke options while leaving all XY inputs unchanged.
    top_z = spec.frames["UncasingLoadFrame"].translation_m[2] + spec.box_skus["123591"].size_m[2]
    arm_plane_height = spec.robots["sr12ia"].workspace.arm_plane_height_m
    base_z = top_z + 0.375 - arm_plane_height
    base = replace(spec.robots["sr12ia"].nominal_base, z_m=base_z)

    short_snapshot = materialize_scene(
        spec,
        SceneState(sr_j3_stroke_m=0.30),
        sr_base=base,
    )
    long_snapshot = materialize_scene(
        spec,
        SceneState(sr_j3_stroke_m=0.45),
        sr_base=base,
    )
    short_report = evaluate_sr_cutting_workspace(spec, short_snapshot)
    long_report = evaluate_sr_cutting_workspace(spec, long_snapshot)

    assert all(target.required_j3_m == pytest.approx(0.375) for target in short_report.targets)
    assert not short_report.vertical_ok
    assert not short_report.feasible
    assert long_report.vertical_ok
    assert long_report.feasible


def test_sku_override_uses_configured_dimensions_not_scene_sample_geometry(spec):
    snapshot = materialize_scene(spec, SceneState(sku="123591"))

    report = evaluate_sr_cutting_workspace(spec, snapshot, sku="000509")

    assert report.sku == "000509"
    assert report.box_size_m == (0.560, 0.480, 0.240)
    expected_top_z = (
        snapshot.frames["UncasingLoadFrame"].translation_m[2] + 0.240
    )
    assert all(
        target.world_position_m[2] == pytest.approx(expected_top_z)
        for target in report.targets
    )
