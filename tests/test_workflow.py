from __future__ import annotations

from dataclasses import replace
import math

import numpy as np
import pytest

from decanting_workspace import SceneState, load_scene_spec, materialize_scene
from decanting_workspace.models import BoxPrimitive
from decanting_workspace.robots import (
    floating_configuration,
    load_robot_bundle,
)
from decanting_workspace.workflow import (
    CoordinationMode,
    Criticality,
    build_ur20_task_sequence,
    expand_process_scenarios,
)


@pytest.fixture(scope="module")
def spec():
    return load_scene_spec()


@pytest.fixture(scope="module")
def ur_bundle(spec):
    ur = spec.robots["ur20"]
    return load_robot_bundle(
        "ur20",
        ur.urdf_path,
        suction_proxy=ur.suction_proxy,
    )


def _scenario(spec, state=None, mode=CoordinationMode.SIMULTANEOUS):
    return expand_process_scenarios(
        spec,
        state or SceneState(corner_samples=("southwest",)),
        (mode,),
    )[0]


def _steps(spec, ur_bundle, state=None, mode=CoordinationMode.SIMULTANEOUS, **kwargs):
    scenario = _scenario(spec, state, mode)
    snapshot = materialize_scene(spec, scenario.scene_state)
    return snapshot, build_ur20_task_sequence(
        spec,
        snapshot,
        scenario,
        ur_bundle,
        **kwargs,
    )


def _check(step, name):
    return next(item for item in step.pose_checks if item.name == name)


def _object(states, name):
    return next(item for item in states if item.name == name)


def _yaw_matrix(angle_rad):
    c = math.cos(angle_rad)
    s = math.sin(angle_rad)
    return np.array(((c, -s, 0.0), (s, c, 0.0), (0.0, 0.0, 1.0)))


def test_four_corners_expand_as_independent_scenes_for_each_mode(spec):
    state = SceneState()
    scenarios = expand_process_scenarios(spec, state)

    assert len(scenarios) == 8
    assert state.corner_samples == (
        "southwest",
        "southeast",
        "northwest",
        "northeast",
    )
    for corner in state.corner_samples:
        matching = [item for item in scenarios if item.corner == corner]
        assert {item.coordination_mode for item in matching} == {
            CoordinationMode.SIMULTANEOUS,
            CoordinationMode.SEQUENTIAL,
        }
        assert all(item.scene_state.corner_samples == (corner,) for item in matching)


def test_steps_one_through_eight_and_criticalities_exist(spec, ur_bundle):
    _, steps = _steps(spec, ur_bundle)

    assert [step.index for step in steps] == list(range(1, 9))
    assert [step.criticality for step in steps] == [
        Criticality.HARD,
        Criticality.HARD,
        Criticality.HARD,
        Criticality.HARD,
        Criticality.HARD,
        Criticality.SKIP,
        Criticality.SOFT,
        Criticality.HARD,
    ]
    assert steps[5].pose_checks == ()
    assert steps[6].waste_fit is not None


def test_pallet_pick_uses_nominal_top_down_rotation_and_uncasing_world_half_turn(
    spec,
    ur_bundle,
):
    import pinocchio as pin

    snapshot, steps = _steps(spec, ur_bundle)
    ur = spec.robots["ur20"]
    q = floating_configuration(ur_bundle, snapshot.ur_base, ur.nominal_q)
    data = ur_bundle.model.createData()
    pin.framesForwardKinematics(ur_bundle.model, data, q)
    nominal_rotation = np.asarray(
        data.oMf[ur_bundle.model.getFrameId("suction_tcp")].rotation
    )
    pick = _check(steps[0], "pallet_box_approach").end_tcp
    place = _check(steps[1], "uncasing_box_preplace").end_tcp

    np.testing.assert_allclose(pick.rotation, nominal_rotation, atol=1e-10)
    np.testing.assert_allclose(pick.rotation[:, 2], (0.0, 0.0, -1.0), atol=1e-8)
    np.testing.assert_allclose(
        place.rotation,
        _yaw_matrix(math.pi) @ pick.rotation,
        atol=1e-9,
    )
    np.testing.assert_allclose(place.rotation[:, 2], (0.0, 0.0, -1.0), atol=1e-8)


def test_top_pick_and_uncasing_positions_use_box_top_and_frame_support_plane(
    spec,
    ur_bundle,
):
    snapshot, steps = _steps(spec, ur_bundle)
    state = snapshot.state
    box = next(
        item
        for item in snapshot.boxes
        if item.name == f"box_{state.sku}_southwest"
    )
    pick = _check(steps[0], "pallet_box_approach").end_tcp
    place = _check(steps[1], "uncasing_box_preplace").end_tcp
    frame = snapshot.frames["UncasingLoadFrame"]

    assert pick.translation_m == pytest.approx(
        (
            box.center_m[0],
            box.center_m[1],
            box.center_m[2] + box.size_m[2] / 2.0,
        )
    )
    assert place.translation_m == pytest.approx(
        (
            frame.translation_m[0],
            frame.translation_m[1],
            frame.translation_m[2] + box.size_m[2],
        )
    )
    placed = _object(steps[1].pose_checks[0].object_states, box.name)
    np.testing.assert_allclose(
        placed.world_T_object.rotation,
        _yaw_matrix(math.pi),
        atol=1e-9,
    )
    assert placed.attached_to_tcp


def test_level_two_tote_is_rotated_and_picked_from_world_negative_x_face(
    spec,
    ur_bundle,
):
    _, steps = _steps(spec, ur_bundle)
    approach = _check(steps[2], "level_2_tote_front_approach")
    contact = approach.end_tcp
    tote = _object(approach.object_states, "representative_tote")

    # Level-2 top is 1.230 m; the 0.300 m tote centre is at 1.380 m.
    assert tote.world_T_object.translation_m[2] == pytest.approx(1.38)
    np.testing.assert_allclose(
        tote.world_T_object.rotation,
        _yaw_matrix(math.pi / 2.0),
        atol=1e-10,
    )
    assert contact.translation_m == pytest.approx(
        (0.683292568 - 0.220, -7.362836361, 1.38)
    )
    np.testing.assert_allclose(
        contact.rotation[:, 2], (1.0, 0.0, 0.0), atol=1e-12
    )
    np.testing.assert_allclose(
        contact.rotation[:, 0], (0.0, 0.0, 1.0), atol=1e-12
    )
    np.testing.assert_allclose(
        np.asarray(contact.translation_m) - np.asarray(approach.start_tcp.translation_m),
        (0.15, 0.0, 0.0),
    )
    lift = _check(steps[2], "level_2_tote_lift")
    assert lift.start_tcp.translation_m == pytest.approx(contact.translation_m)
    np.testing.assert_allclose(
        np.asarray(lift.end_tcp.translation_m)
        - np.asarray(lift.start_tcp.translation_m),
        (0.0, 0.0, 0.15),
    )


def test_table_offset_moves_tote_place_and_push_start_together(spec, ur_bundle):
    offset = 0.50
    state = SceneState(
        corner_samples=("southwest",),
        tote_long_axis_offset_m=offset,
    )
    snapshot, steps = _steps(spec, ur_bundle, state)
    frame = snapshot.frames["ToteLoadFrame"]
    table_place = _check(steps[3], "worktable_tote_preplace").end_tcp
    push_start = _check(steps[7], "filled_tote_push_to_supply").start_tcp

    assert table_place.translation_m == pytest.approx(
        (frame.translation_m[0] - 0.220, frame.translation_m[1], 0.810)
    )
    assert push_start.translation_m == pytest.approx(
        (
            frame.translation_m[0] - 0.220,
            frame.translation_m[1],
            snapshot.frames["ToteSupplyFrame"].translation_m[2],
        )
    )
    table_tote = _object(steps[3].pose_checks[0].object_states, "representative_tote")
    assert table_tote.world_T_object.translation_m == pytest.approx(
        (frame.translation_m[0], frame.translation_m[1], 0.810)
    )


def test_opened_box_uses_nearest_vertical_face_for_candidate_base(spec, ur_bundle):
    snapshot, steps = _steps(spec, ur_bundle)
    box = next(item for item in snapshot.boxes if item.name.startswith("box_"))
    contact = _check(steps[4], "opened_box_front_approach").end_tcp
    frame = snapshot.frames["UncasingLoadFrame"]

    # The nominal UR base lies west of the uncasing box, so its -World-X face
    # is nearest.  The tool +Z therefore points inward along +World-X.
    assert contact.translation_m == pytest.approx(
        (
            frame.translation_m[0] - box.size_m[0] / 2.0,
            frame.translation_m[1],
            frame.translation_m[2] + box.size_m[2] / 2.0,
        )
    )
    np.testing.assert_allclose(
        contact.rotation[:, 2], (1.0, 0.0, 0.0), atol=1e-12
    )
    np.testing.assert_allclose(
        contact.rotation[:, 0], (0.0, 0.0, 1.0), atol=1e-12
    )


def test_sr_keepout_is_active_only_for_simultaneous_tote_steps(spec, ur_bundle):
    keepout = BoxPrimitive(
        name="test_sr_keepout",
        role="dynamic_keepout",
        center_m=(0.0, 0.0, 1.0),
        size_m=(1.0, 1.0, 2.0),
    )
    _, simultaneous = _steps(
        spec,
        ur_bundle,
        mode=CoordinationMode.SIMULTANEOUS,
        sr_keepout=keepout,
    )
    _, sequential = _steps(
        spec,
        ur_bundle,
        mode=CoordinationMode.SEQUENTIAL,
        sr_keepout=keepout,
    )

    assert simultaneous[2].active_extra_obstacles == (keepout,)
    assert simultaneous[3].active_extra_obstacles == (keepout,)
    assert all(not step.active_extra_obstacles for step in simultaneous[4:])
    assert all(not step.active_extra_obstacles for step in sequential)


def test_waste_fit_is_soft_and_release_keeps_front_attachment(spec, ur_bundle):
    state = SceneState(sku="049995", corner_samples=("southwest",))
    _, steps = _steps(spec, ur_bundle, state)
    waste = steps[6]
    check = waste.waste_fit

    assert waste.criticality is Criticality.SOFT
    assert check is not None
    assert check.conveyor_name == "conveyor_level_3"
    assert check.available_width_m == pytest.approx(0.669039)
    assert check.cross_width_options_m == pytest.approx((0.517, 0.503))
    assert check.selected_cross_width_m == pytest.approx(0.503)
    assert check.fits
    release_box = _object(waste.pose_checks[0].object_states, "box_049995_southwest")
    assert release_box.attached_to_tcp
    reconstructed = (
        waste.pose_checks[0].end_tcp.matrix
        @ release_box.tcp_T_object.matrix
    )
    np.testing.assert_allclose(
        reconstructed,
        release_box.world_T_object.matrix,
        atol=1e-9,
    )


def test_push_reaches_exact_supply_frame_and_couples_tote_motion(spec, ur_bundle):
    snapshot, steps = _steps(spec, ur_bundle)
    approach = _check(steps[7], "filled_tote_push_approach")
    push = _check(steps[7], "filled_tote_push_to_supply")
    final = _check(steps[7], "filled_tote_final_pose_at_supply")
    supply = snapshot.frames["ToteSupplyFrame"]

    assert [check.name for check in steps[7].pose_checks] == [
        "filled_tote_push_approach",
        "filled_tote_push_to_supply",
        "filled_tote_final_pose_at_supply",
    ]
    assert push.end_tcp.translation_m == pytest.approx(supply.translation_m)
    assert final.start_tcp.translation_m == pytest.approx(supply.translation_m)
    assert final.end_tcp.translation_m == pytest.approx(supply.translation_m)
    assert not final.include_in_metric_summary
    np.testing.assert_allclose(push.end_tcp.rotation[:, 2], (1.0, 0.0, 0.0))
    np.testing.assert_allclose(
        np.asarray(push.start_tcp.translation_m)
        - np.asarray(approach.start_tcp.translation_m),
        (0.15, 0.0, 0.0),
    )
    pushed_tote = _object(push.object_states, "representative_tote")
    assert pushed_tote.follows_tcp
    assert not pushed_tote.attached_to_tcp
    np.testing.assert_allclose(
        push.end_tcp.matrix @ pushed_tote.tcp_T_object.matrix,
        pushed_tote.world_T_object.matrix,
        atol=1e-9,
    )
    assert (
        "representative_tote",
        spec.tote.motion_support_box,
    ) in push.allowed_contacts
    assert ("representative_tote", "conveyor_level_1") in push.allowed_contacts
    final_tote = _object(final.object_states, "representative_tote")
    assert final_tote.follows_tcp
    assert not final_tote.attached_to_tcp
    np.testing.assert_allclose(
        final.end_tcp.matrix @ final_tote.tcp_T_object.matrix,
        final_tote.world_T_object.matrix,
        atol=1e-9,
    )
    assert final.allowed_contacts == push.allowed_contacts


def test_dynamic_objects_replace_snapshot_alternatives(spec, ur_bundle):
    scenario = _scenario(
        spec,
        SceneState(corner_samples=("southwest",)),
    )
    # Use an all-corner snapshot deliberately: the sequence still declares
    # every alternative suppressed and adds only its scenario box state.
    all_corner_state = replace(
        scenario.scene_state,
        corner_samples=("southwest", "southeast", "northwest", "northeast"),
    )
    snapshot = materialize_scene(spec, all_corner_state)
    steps = build_ur20_task_sequence(spec, snapshot, scenario, ur_bundle)

    assert {
        f"box_{scenario.scene_state.sku}_{corner}"
        for corner in all_corner_state.corner_samples
    } <= set(steps[0].suppress_snapshot_objects)
    assert "representative_tote" in steps[0].suppress_snapshot_objects
    assert [item.name for item in steps[0].object_states] == [
        f"box_{scenario.scene_state.sku}_southwest"
    ]


def test_suction_contact_permissions_use_collision_pad_alias(spec, ur_bundle):
    _, steps = _steps(spec, ur_bundle)
    contact_checks = [
        check
        for step in steps
        for check in step.pose_checks
        if check.contact_object_name is not None
    ]

    assert contact_checks
    assert all(
        any(pair[0] == "suction_pad" for pair in check.allowed_contacts)
        for check in contact_checks
    )
    assert all(
        pair[0] != "suction_tcp"
        for check in contact_checks
        for pair in check.allowed_contacts
    )
