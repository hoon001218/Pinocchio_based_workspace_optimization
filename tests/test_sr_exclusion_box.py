from __future__ import annotations

import math

import numpy as np
import pytest

from decanting_workspace import BasePose, SceneState, load_scene_spec
from decanting_workspace.collision import sr_concurrent_exclusion_box
from decanting_workspace.robots import load_robot_bundle
from decanting_workspace.scene import materialize_scene


@pytest.fixture(scope="module")
def sr_assets():
    spec = load_scene_spec()
    bundle = load_robot_bundle(
        "sr12ia",
        spec.robots["sr12ia"].urdf_path,
    )
    return spec, bundle


@pytest.mark.parametrize("stroke_m", (0.30, 0.45))
def test_nominal_exclusion_unions_task_footprint_and_long_j3_proxy(
    sr_assets,
    stroke_m,
):
    spec, bundle = sr_assets
    snapshot = materialize_scene(
        spec,
        SceneState(sr_j3_stroke_m=stroke_m),
    )

    result = sr_concurrent_exclusion_box(spec, snapshot, bundle)

    # Base-local union measured from the configured fallback collision model:
    # task x [-0.265036, 0.834964], task/robot y
    # [-0.375, 0.553457418836231], and robot z [0, 0.978].  The same long
    # conservative proxy covers both configured J3 stroke selections.
    assert result.name == "sr12ia_concurrent_exclusion"
    assert result.role == "concurrent_exclusion"
    assert result.size_m == pytest.approx(
        (1.10, 0.928457418836231, 0.978),
        abs=1e-12,
    )
    assert result.center_m == pytest.approx(
        (-0.14398943858188445, -7.358607684, 1.043805667),
        abs=1e-12,
    )
    assert result.yaw_deg == pytest.approx(-90.0)
    assert result.collision_enabled


def test_exclusion_follows_candidate_base_as_one_oriented_local_box(sr_assets):
    spec, bundle = sr_assets
    base = BasePose(1.25, -2.0, 0.80, 30.0)
    snapshot = materialize_scene(spec, sr_base=base)

    result = sr_concurrent_exclusion_box(spec, snapshot, bundle)

    center_local = np.asarray((0.284964, 0.0892287094181155))
    c = math.cos(base.yaw_rad)
    s = math.sin(base.yaw_rad)
    expected_xy = np.asarray((base.x_m, base.y_m)) + np.asarray(
        (
            c * center_local[0] - s * center_local[1],
            s * center_local[0] + c * center_local[1],
        )
    )
    assert result.center_m == pytest.approx(
        (expected_xy[0], expected_xy[1], base.z_m + 0.489),
        abs=1e-12,
    )
    assert result.size_m == pytest.approx(
        (1.10, 0.928457418836231, 0.978),
        abs=1e-12,
    )
    assert result.yaw_deg == pytest.approx(base.yaw_deg)


def test_margin_expands_every_local_face_without_moving_center(sr_assets):
    spec, bundle = sr_assets
    snapshot = materialize_scene(spec)
    nominal = sr_concurrent_exclusion_box(spec, snapshot, bundle)

    result = sr_concurrent_exclusion_box(
        spec,
        snapshot,
        bundle,
        margin_m=0.02,
    )

    assert result.center_m == pytest.approx(nominal.center_m, abs=1e-12)
    assert result.size_m == pytest.approx(
        np.asarray(nominal.size_m) + 0.04,
        abs=1e-12,
    )


def test_arm_override_recomputes_collision_bounds(sr_assets):
    spec, bundle = sr_assets
    snapshot = materialize_scene(spec)
    nominal = sr_concurrent_exclusion_box(spec, snapshot, bundle)

    straight_arm = sr_concurrent_exclusion_box(
        spec,
        snapshot,
        bundle,
        arm_q=(0.0, 0.0, 0.10, 0.0),
    )

    # The straight arm protrudes beyond the cutting footprint's local +X edge
    # while the configured opposite-elbow nominal pose does not.
    assert straight_arm.size_m[0] > nominal.size_m[0]
    assert straight_arm.center_m != pytest.approx(nominal.center_m)


@pytest.mark.parametrize("margin_m", (-0.001, math.nan, math.inf))
def test_invalid_margin_is_rejected(sr_assets, margin_m):
    spec, bundle = sr_assets
    snapshot = materialize_scene(spec)

    with pytest.raises(ValueError, match="margin_m"):
        sr_concurrent_exclusion_box(
            spec,
            snapshot,
            bundle,
            margin_m=margin_m,
        )


def test_invalid_arm_override_is_rejected(sr_assets):
    spec, bundle = sr_assets
    snapshot = materialize_scene(spec)

    with pytest.raises(ValueError, match="finite"):
        sr_concurrent_exclusion_box(
            spec,
            snapshot,
            bundle,
            arm_q=(0.0, 0.0, math.nan, 0.0),
        )
    with pytest.raises(ValueError, match="expects 4 arm positions"):
        sr_concurrent_exclusion_box(
            spec,
            snapshot,
            bundle,
            arm_q=(0.0,),
        )
