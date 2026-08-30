from __future__ import annotations

import math

import numpy as np
import pytest

from decanting_workspace import load_scene_spec
from decanting_workspace.collision_world import (
    AttachedBox,
    CollisionPhase,
    PinocchioCollisionChecker,
)
from decanting_workspace.kinematics import frame_pose
from decanting_workspace.models import BoxPrimitive
from decanting_workspace.robots import floating_configuration, load_robot_bundle


@pytest.fixture(scope="module")
def ur20_collision_assets():
    spec = load_scene_spec()
    robot_spec = spec.robots["ur20"]
    bundle = load_robot_bundle(
        "ur20",
        robot_spec.urdf_path,
        suction_proxy=robot_spec.suction_proxy,
    )
    q = floating_configuration(
        bundle,
        robot_spec.nominal_base,
        robot_spec.nominal_q,
    )
    return robot_spec, bundle, q


def _box(
    name: str,
    center_m: np.ndarray | tuple[float, float, float],
    size_m: tuple[float, float, float] = (0.02, 0.02, 0.02),
) -> BoxPrimitive:
    return BoxPrimitive(
        name=name,
        role="test_obstacle",
        center_m=tuple(float(value) for value in center_m),
        size_m=size_m,
    )


def _collision_geometry_centers(bundle, q) -> dict[str, np.ndarray]:
    import pinocchio as pin

    model_data = bundle.model.createData()
    geometry_data = bundle.collision_model.createData()
    pin.updateGeometryPlacements(
        bundle.model,
        model_data,
        bundle.collision_model,
        geometry_data,
        q,
    )
    return {
        geometry.name: np.asarray(placement.translation, dtype=float).reshape(3)
        for geometry, placement in zip(
            bundle.collision_model.geometryObjects,
            geometry_data.oMg,
            strict=True,
        )
    }


def _pair_geometry_names(checker: PinocchioCollisionChecker):
    geometries = checker.geometry_model.geometryObjects
    return {
        (
            geometries[int(pair.first)].name,
            geometries[int(pair.second)].name,
        )
        for pair in checker.geometry_model.collisionPairs
    }


def test_checker_creates_nonzero_pairs_for_pairless_urdf(
    ur20_collision_assets,
):
    _, bundle, _ = ur20_collision_assets
    assert len(bundle.collision_model.collisionPairs) == 0

    checker = PinocchioCollisionChecker(
        bundle,
        CollisionPhase(obstacles=(), floor_z_m=0.0),
        include_self_collision=False,
    )

    assert checker.collision_pair_count == bundle.collision_model.ngeoms
    assert checker.collision_pair_count > 0


def test_nominal_configuration_is_clear_with_self_and_floor_pairs(
    ur20_collision_assets,
):
    _, bundle, q = ur20_collision_assets
    checker = PinocchioCollisionChecker(
        bundle,
        CollisionPhase(obstacles=(), floor_z_m=0.0),
        include_self_collision=True,
    )

    report = checker.check(q)

    assert not report.in_collision
    assert report.contacts == ()
    assert checker.is_collision_free(q)
    assert math.isfinite(report.minimum_distance_m)
    assert report.nearest_pair is not None
    # Coal 3.0.4 can report a small negative signed GJK distance for a pair
    # whose collision result remains false.  Collision status must therefore
    # come from collisionResults, not from the distance sign.


def test_named_obstacle_collision_is_reported(ur20_collision_assets):
    _, bundle, q = ur20_collision_assets
    centers = _collision_geometry_centers(bundle, q)
    target = _box(
        "pick_target",
        centers["suction_pad_0"],
        (0.02, 0.02, 0.01),
    )
    checker = PinocchioCollisionChecker(
        bundle,
        CollisionPhase(obstacles=(target,), floor_z_m=None),
        include_self_collision=False,
    )

    report = checker.check(q)

    assert report.in_collision
    assert {(contact.first, contact.second) for contact in report.contacts} == {
        ("suction_pad_0", "pick_target")
    }
    assert report.nearest_pair == ("suction_pad_0", "pick_target")
    assert report.minimum_distance_m < 0.0


def test_allowed_pad_target_does_not_disable_other_link_checks(
    ur20_collision_assets,
):
    _, bundle, q = ur20_collision_assets
    centers = _collision_geometry_centers(bundle, q)
    target = _box(
        "pick_target",
        centers["suction_pad_0"],
        (0.02, 0.02, 0.01),
    )
    arm_blocker = _box(
        "arm_blocker",
        centers["upper_arm_link_0"],
        (0.03, 0.03, 0.03),
    )
    allowed = frozenset({("suction_pad", "pick_target")})

    contact_only = PinocchioCollisionChecker(
        bundle,
        CollisionPhase(
            obstacles=(target,),
            allowed_contacts=allowed,
            floor_z_m=None,
        ),
        include_self_collision=False,
    ).check(q)
    blocked = PinocchioCollisionChecker(
        bundle,
        CollisionPhase(
            obstacles=(target, arm_blocker),
            allowed_contacts=allowed,
            floor_z_m=None,
        ),
        include_self_collision=False,
    ).check(q)

    assert not contact_only.in_collision
    assert contact_only.contacts == ()
    assert blocked.in_collision
    assert {(contact.first, contact.second) for contact in blocked.contacts} == {
        ("upper_arm_link_0", "arm_blocker")
    }


def test_attached_payload_is_checked_against_world_obstacles(
    ur20_collision_assets,
):
    _, bundle, q = ur20_collision_assets
    frame_T_payload = np.eye(4)
    frame_T_payload[2, 3] = 0.10
    payload = AttachedBox(
        name="carried_box",
        size_m=(0.08, 0.08, 0.08),
        frame_name="suction_tcp",
        frame_T_box=frame_T_payload,
    )
    world_T_payload = frame_pose(bundle, q, "suction_tcp") @ frame_T_payload
    blocker = _box("payload_blocker", world_T_payload[:3, 3])

    payload_only = PinocchioCollisionChecker(
        bundle,
        CollisionPhase(
            obstacles=(),
            attached_boxes=(payload,),
            floor_z_m=None,
        ),
        include_self_collision=False,
    ).check(q)
    blocked = PinocchioCollisionChecker(
        bundle,
        CollisionPhase(
            obstacles=(blocker,),
            attached_boxes=(payload,),
            floor_z_m=None,
        ),
        include_self_collision=False,
    ).check(q)

    # Default suction/payload contact is exempt, but the payload remains
    # paired with other robot links and every world obstacle.
    assert not payload_only.in_collision
    assert blocked.in_collision
    assert ("carried_box", "payload_blocker") in {
        (contact.first, contact.second) for contact in blocked.contacts
    }


def test_fixed_base_support_contact_exempts_only_own_base_geometry(
    ur20_collision_assets,
):
    robot_spec, bundle, q = ur20_collision_assets
    base = robot_spec.nominal_base
    own_pedestal = _box(
        "ur20_pedestal",
        (base.x_m, base.y_m, base.z_m / 2.0),
        (0.30, 0.30, base.z_m),
    )
    checker = PinocchioCollisionChecker(
        bundle,
        CollisionPhase(obstacles=(own_pedestal,), floor_z_m=None),
        include_self_collision=False,
    )
    before = np.array(q, copy=True)

    report = checker.check(q)
    pairs = _pair_geometry_names(checker)

    # The IK layer freezes this free-flyer root.  The checker neither changes
    # it nor adds the intentional base-link/support contact as a collision
    # pair.  All moving arm geometries remain paired with the pedestal.
    np.testing.assert_array_equal(q, before)
    assert not report.in_collision
    assert ("base_link_0", "world::ur20_pedestal") not in pairs
    assert ("shoulder_link_0", "world::ur20_pedestal") in pairs
    assert ("suction_pad_0", "world::ur20_pedestal") in pairs
