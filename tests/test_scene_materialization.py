from __future__ import annotations

import pytest

from decanting_workspace import BasePose, SceneState, load_scene_spec, materialize_scene


@pytest.fixture(scope="module")
def spec():
    return load_scene_spec()


def _box(snapshot, name):
    return next(box for box in snapshot.boxes if box.name == name)


def test_lift_height_moves_pallet_and_real_size_boxes(spec):
    state = SceneState(lift_height_m=0.760, sku="000509", corner_samples=("southwest",))
    snapshot = materialize_scene(spec, state)
    pallet = _box(snapshot, "pallet")
    box = _box(snapshot, "box_000509_southwest")

    assert pallet.center_m[2] == pytest.approx(0.835)
    assert box.size_m == pytest.approx((0.560, 0.480, 0.240))
    assert box.center_m[2] - box.size_m[2] / 2.0 == pytest.approx(0.910)
    assert box.collision_enabled


def test_four_corner_boxes_are_alternative_samples(spec):
    snapshot = materialize_scene(spec, SceneState(sku="049995"))
    samples = [box for box in snapshot.boxes if box.name.startswith("box_049995_")]
    assert len(samples) == 4
    assert all(box.role == "box_sample_alternative" for box in samples)
    assert all(not box.collision_enabled for box in samples)


def test_representative_tote_sits_on_first_level(spec):
    tote = _box(materialize_scene(spec), "representative_tote")
    assert tote.center_m[2] - tote.size_m[2] / 2.0 == pytest.approx(0.660)
    assert tote.center_m[:2] == pytest.approx(
        spec.frames[spec.tote.representative_frame].translation_m[:2]
    )


def test_pedestals_follow_candidate_height_and_yaw(spec):
    ur_base = BasePose(-1.0, -7.2, 0.9, 15.0)
    sr_base = spec.robots["sr12ia"].nominal_base
    snapshot = materialize_scene(spec, ur_base=ur_base, sr_base=sr_base)
    ur_pedestal = _box(snapshot, "ur20_pedestal")
    sr_pedestal = _box(snapshot, "sr12ia_pedestal")

    assert ur_pedestal.center_m == pytest.approx((-1.0, -7.2, 0.45))
    assert ur_pedestal.size_m == pytest.approx((0.30, 0.30, 0.9))
    assert ur_pedestal.yaw_deg == pytest.approx(15.0)
    assert sr_pedestal.center_m[0] == pytest.approx(sr_base.x_m)
    assert sr_pedestal.center_m[1] == pytest.approx(sr_base.y_m - 0.284964)
    assert sr_pedestal.center_m[2] == pytest.approx(sr_base.z_m / 2.0)


def test_invalid_scenario_values_fail_early(spec):
    with pytest.raises(ValueError, match="lift height"):
        materialize_scene(spec, SceneState(lift_height_m=0.761))
    with pytest.raises(ValueError, match="unknown box SKU"):
        materialize_scene(spec, SceneState(sku="missing"))
    with pytest.raises(ValueError, match="J3 stroke"):
        materialize_scene(spec, SceneState(sr_j3_stroke_m=0.35))

