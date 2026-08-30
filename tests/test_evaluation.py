from __future__ import annotations

from dataclasses import dataclass
import math
from types import SimpleNamespace

import numpy as np
import pytest

import decanting_workspace.evaluation as evaluation_module
from decanting_workspace import BasePose, SceneState, load_scene_spec, materialize_scene
from decanting_workspace.collision_world import CollisionReport
from decanting_workspace.evaluation import (
    EvaluationOptions,
    _evaluate_linear_check,
    collision_phase_for_check,
    evaluate_task_sequence,
    interpolate_rigid_poses,
)
from decanting_workspace.kinematics import (
    IKOptions,
    IKResult,
    ManipulabilityMetrics,
    frame_pose,
)
from decanting_workspace.models import BoxPrimitive
from decanting_workspace.robots import floating_configuration, load_robot_bundle
from decanting_workspace.workflow import (
    CoordinationMode,
    Criticality,
    LinearPoseCheck,
    RigidPose,
    TaskStepSpec,
    WasteFitCheck,
    build_ur20_task_sequence,
    expand_process_scenarios,
)


@dataclass(frozen=True)
class _EvaluationAssets:
    spec: object
    bundle: object
    scenario: object
    snapshot: object
    known_q: np.ndarray
    target: RigidPose
    suppressed: tuple[str, ...]


@pytest.fixture(scope="module")
def assets():
    spec = load_scene_spec()
    robot = spec.robots["ur20"]
    bundle = load_robot_bundle(
        "ur20",
        robot.urdf_path,
        suction_proxy=robot.suction_proxy,
    )
    nominal = robot.nominal_base
    # A deliberately non-nominal root makes an accidental base optimization
    # or free-flyer update observable in the returned IK configuration.
    candidate = BasePose(
        nominal.x_m + 0.013,
        nominal.y_m - 0.021,
        nominal.z_m + 0.037,
        nominal.yaw_deg + 11.25,
    )
    state = SceneState(corner_samples=("southwest",))
    scenario = expand_process_scenarios(
        spec,
        state,
        (CoordinationMode.SEQUENTIAL,),
    )[0]
    snapshot = materialize_scene(spec, state, ur_base=candidate)
    known_q = floating_configuration(bundle, candidate, robot.nominal_q)
    target = RigidPose(frame_pose(bundle, known_q, "suction_tcp"))
    suppressed = tuple(box.name for box in snapshot.collision_boxes)
    return _EvaluationAssets(
        spec,
        bundle,
        scenario,
        snapshot,
        known_q,
        target,
        suppressed,
    )


@pytest.fixture(scope="module")
def fast_options():
    return EvaluationOptions(
        cartesian_translation_step_m=1.0,
        cartesian_rotation_step_rad=math.pi,
        joint_interpolation_samples=0,
        ik=IKOptions(max_iterations=30),
    )


def _single_check_step(
    assets: _EvaluationAssets,
    *,
    index: int,
    name: str,
    criticality: Criticality,
    extra_obstacles: tuple[BoxPrimitive, ...] = (),
    waste_fit: WasteFitCheck | None = None,
    include_in_metric_summary: bool = True,
) -> TaskStepSpec:
    check = LinearPoseCheck(
        name=f"{name}_check",
        start_tcp=assets.target,
        end_tcp=assets.target,
        criticality=criticality,
        include_in_metric_summary=include_in_metric_summary,
    )
    return TaskStepSpec(
        index=index,
        name=name,
        criticality=criticality,
        pose_checks=(check,),
        suppress_snapshot_objects=assets.suppressed,
        active_extra_obstacles=extra_obstacles,
        waste_fit=waste_fit,
    )


def _geometry_center(bundle, q, geometry_name: str) -> np.ndarray:
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
    for geometry, placement in zip(
        bundle.collision_model.geometryObjects,
        geometry_data.oMg,
        strict=True,
    ):
        if geometry.name == geometry_name:
            return np.asarray(placement.translation, dtype=float).reshape(3)
    raise AssertionError(f"missing test geometry {geometry_name}")


def _pad_blocker(assets: _EvaluationAssets) -> BoxPrimitive:
    center = _geometry_center(assets.bundle, assets.known_q, "suction_pad_0")
    return BoxPrimitive(
        name="evaluation_pad_blocker",
        role="test_obstacle",
        center_m=tuple(float(value) for value in center),
        size_m=(0.04, 0.04, 0.02),
    )


def test_collision_phase_replaces_world_objects_and_couples_payloads():
    spec = load_scene_spec()
    robot = spec.robots["ur20"]
    bundle = load_robot_bundle(
        "ur20",
        robot.urdf_path,
        suction_proxy=robot.suction_proxy,
    )
    state = SceneState(corner_samples=("southwest",))
    scenario = expand_process_scenarios(
        spec,
        state,
        (CoordinationMode.SEQUENTIAL,),
    )[0]
    snapshot = materialize_scene(spec, state)
    steps = build_ur20_task_sequence(spec, snapshot, scenario, bundle)
    box_name = "box_123591_southwest"

    # The source box from the snapshot is suppressed and reintroduced once at
    # its phase pose, rather than duplicated as a static and dynamic obstacle.
    world_phase = collision_phase_for_check(
        spec,
        snapshot,
        steps[0],
        steps[0].pose_checks[0],
    )
    world_names = [box.name for box in world_phase.obstacles]
    assert world_names.count(box_name) == 1
    assert "representative_tote" not in world_names
    assert world_phase.attached_boxes == ()

    attached_phase = collision_phase_for_check(
        spec,
        snapshot,
        steps[0],
        steps[0].pose_checks[1],
    )
    assert box_name not in {box.name for box in attached_phase.obstacles}
    assert [box.name for box in attached_phase.attached_boxes] == [box_name]
    expected_attachment = steps[0].pose_checks[1].object_states[0].tcp_T_object
    np.testing.assert_allclose(
        attached_phase.attached_boxes[0].frame_T_box,
        expected_attachment.matrix,
    )

    # A pushed tote is not suction-attached in the workflow, but follows_tcp
    # still makes it moving collision geometry on suction_tcp.
    push_check = steps[7].pose_checks[1]
    assert push_check.object_states[0].follows_tcp
    follows_phase = collision_phase_for_check(
        spec,
        snapshot,
        steps[7],
        push_check,
    )
    assert "representative_tote" not in {
        box.name for box in follows_phase.obstacles
    }
    assert [box.name for box in follows_phase.attached_boxes] == [
        "representative_tote"
    ]
    assert ("suction_pad", "representative_tote") in follows_phase.allowed_contacts


def test_pose_interpolation_includes_exact_endpoints_and_zero_motion_sample():
    start_matrix = np.eye(4)
    end_matrix = np.eye(4)
    end_matrix[:3, :3] = np.array(
        ((0.0, -1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0))
    )
    end_matrix[:3, 3] = (0.10, -0.02, 0.03)
    start = RigidPose(start_matrix)
    end = RigidPose(end_matrix)

    samples = interpolate_rigid_poses(
        start,
        end,
        translation_step_m=0.06,
        rotation_step_rad=math.pi / 4.0,
    )

    assert [fraction for fraction, _ in samples] == pytest.approx((0.0, 0.5, 1.0))
    np.testing.assert_array_equal(samples[0][1].matrix, start.matrix)
    np.testing.assert_allclose(samples[-1][1].matrix, end.matrix, atol=1e-15)

    stationary = interpolate_rigid_poses(
        end,
        end,
        translation_step_m=0.01,
        rotation_step_rad=0.01,
    )
    assert len(stationary) == 1
    assert stationary[0][0] == 0.0
    np.testing.assert_array_equal(stationary[0][1].matrix, end.matrix)


def test_joint_collision_rejects_branch_and_tries_next_ik_seed(monkeypatch):
    """A colliding local joint branch must not end the Cartesian check early."""

    import pinocchio as pin

    start_matrix = np.eye(4)
    end_matrix = np.eye(4)
    end_matrix[0, 3] = 0.1
    check = LinearPoseCheck(
        "branch_retry",
        RigidPose(start_matrix),
        RigidPose(end_matrix),
        Criticality.HARD,
    )
    clear = CollisionReport(False, (), 0.1, ("robot", "obstacle"))
    blocked = CollisionReport(
        True,
        (),
        -0.01,
        ("robot", "obstacle"),
    )

    class _Checker:
        def __init__(self):
            self.checked = []

        def check(self, q):
            value = np.asarray(q, dtype=float)
            self.checked.append(value.copy())
            return blocked if np.allclose(value, (1.0, 0.0)) else clear

    checker = _Checker()
    solve_count = 0

    def fake_solve(*args, accept_configuration, **kwargs):
        nonlocal solve_count
        solve_count += 1
        rejected = 0
        if solve_count == 1:
            selected = np.array((0.0, 0.0))
            assert accept_configuration(selected)
        else:
            colliding_branch = np.array((2.0, 0.0))
            assert not accept_configuration(colliding_branch)
            rejected += 1
            selected = np.array((0.0, 2.0))
            assert accept_configuration(selected)
        return IKResult(
            "success",
            selected,
            selected,
            1,
            rejected,
            2,
            0.0,
            0.0,
            0.0,
            rejected,
        )

    metric = ManipulabilityMetrics(
        jacobian=np.zeros((6, 2)),
        translational_singular_values=np.ones(2),
        translational_volume=1.0,
        translational_min_singular_value=1.0,
        translational_condition_number=1.0,
        normalized_singular_values=np.ones(2),
        normalized_volume=1.0,
        normalized_min_singular_value=1.0,
        normalized_condition_number=1.0,
        characteristic_length_m=0.5,
    )
    monkeypatch.setattr(evaluation_module, "solve_frame_ik", fake_solve)
    monkeypatch.setattr(
        evaluation_module,
        "compute_manipulability",
        lambda *args, **kwargs: metric,
    )
    monkeypatch.setattr(
        pin,
        "interpolate",
        lambda model, first, second, fraction: (
            (1.0 - fraction) * np.asarray(first)
            + fraction * np.asarray(second)
        ),
    )

    result, last_arm = _evaluate_linear_check(
        SimpleNamespace(ur_base=BasePose(0.0, 0.0, 0.0, 0.0)),
        SimpleNamespace(model=object()),
        check,
        checker,
        (),
        EvaluationOptions(
            cartesian_translation_step_m=1.0,
            cartesian_rotation_step_rad=math.pi,
            joint_interpolation_samples=1,
        ),
    )

    assert result.status == "success"
    assert len(result.samples) == 2
    assert result.samples[-1].ik.rejected_by_acceptor == 1
    np.testing.assert_allclose(result.samples[-1].ik.q, (0.0, 2.0))
    np.testing.assert_allclose(last_arm, (0.0, 2.0))
    assert any(np.allclose(q, (1.0, 0.0)) for q in checker.checked)


def test_fk_roundtrip_hard_step_has_metrics_and_freezes_candidate_base(
    assets,
    fast_options,
):
    hard = _single_check_step(
        assets,
        index=1,
        name="known_fk",
        criticality=Criticality.HARD,
    )

    result = evaluate_task_sequence(
        assets.spec,
        assets.snapshot,
        assets.scenario,
        assets.bundle,
        (hard,),
        options=fast_options,
    )

    assert result.hard_feasible
    assert result.steps[0].status == "success"
    sample = result.steps[0].checks[0].samples[0]
    assert sample.ik.success
    assert sample.manipulability is not None
    assert sample.manipulability.jacobian.shape == (6, 6)
    assert result.metrics is not None
    assert result.metrics.sample_count == 1
    # Exact equality is intentional: no IK tangent column may update the
    # floating root of the reusable candidate model.
    np.testing.assert_array_equal(sample.ik.q[:7], assets.known_q[:7])


def test_feasibility_only_pose_keeps_ellipsoid_but_not_summary_weight(
    assets,
    fast_options,
):
    weighted = _single_check_step(
        assets,
        index=1,
        name="weighted_pose",
        criticality=Criticality.HARD,
    )
    feasibility_only = _single_check_step(
        assets,
        index=8,
        name="independent_final_pose",
        criticality=Criticality.HARD,
        include_in_metric_summary=False,
    )

    result = evaluate_task_sequence(
        assets.spec,
        assets.snapshot,
        assets.scenario,
        assets.bundle,
        (weighted, feasibility_only),
        options=fast_options,
    )

    assert result.hard_feasible
    final_sample = result.steps[-1].checks[0].samples[0]
    assert final_sample.manipulability is not None
    assert result.metrics is not None
    assert result.metrics.sample_count == 1


def test_colliding_solution_rejects_a_hard_step(assets, fast_options):
    blocked = _single_check_step(
        assets,
        index=1,
        name="blocked_hard",
        criticality=Criticality.HARD,
        extra_obstacles=(_pad_blocker(assets),),
    )

    result = evaluate_task_sequence(
        assets.spec,
        assets.snapshot,
        assets.scenario,
        assets.bundle,
        (blocked,),
        options=fast_options,
    )

    assert not result.hard_feasible
    check = result.steps[0].checks[0]
    assert result.steps[0].status == "failed"
    assert check.status == "solution_in_collision"
    assert check.failed_sample_index == 0
    assert check.samples[0].collision is not None
    assert check.samples[0].collision.in_collision


def test_soft_collision_failure_and_skip_do_not_invalidate_hard_feasibility(
    assets,
    fast_options,
):
    hard = _single_check_step(
        assets,
        index=1,
        name="clear_hard",
        criticality=Criticality.HARD,
    )
    soft = _single_check_step(
        assets,
        index=7,
        name="blocked_soft",
        criticality=Criticality.SOFT,
        extra_obstacles=(_pad_blocker(assets),),
    )
    skipped = TaskStepSpec(
        index=6,
        name="pour_without_pose_check",
        criticality=Criticality.SKIP,
        pose_checks=(),
        suppress_snapshot_objects=assets.suppressed,
    )

    result = evaluate_task_sequence(
        assets.spec,
        assets.snapshot,
        assets.scenario,
        assets.bundle,
        (hard, soft, skipped),
        options=fast_options,
    )

    assert result.hard_feasible
    assert result.steps[0].status == "success"
    assert result.steps[1].status == "failed"
    assert result.steps[1].checks[0].status == "solution_in_collision"
    assert result.steps[2].status == "skipped"
    assert result.steps[2].checks == ()
    assert result.soft_step_count == 1
    assert result.soft_step_successes == 0
    # Only successful/partial HARD samples contribute to objective inputs.
    assert result.metrics is not None
    assert result.metrics.sample_count == 1


def test_failed_waste_dimension_fit_is_a_soft_failure_only(assets, fast_options):
    hard = _single_check_step(
        assets,
        index=1,
        name="clear_hard",
        criticality=Criticality.HARD,
    )
    no_fit = WasteFitCheck(
        conveyor_name="conveyor_level_3",
        available_width_m=0.20,
        clearance_m=0.0,
        yaw_options_deg=(0.0, 90.0),
        cross_width_options_m=(0.39, 0.237),
        selected_yaw_deg=90.0,
        selected_cross_width_m=0.237,
        fits=False,
    )
    soft = _single_check_step(
        assets,
        index=7,
        name="dimension_only_soft_failure",
        criticality=Criticality.SOFT,
        waste_fit=no_fit,
    )

    result = evaluate_task_sequence(
        assets.spec,
        assets.snapshot,
        assets.scenario,
        assets.bundle,
        (hard, soft),
        options=fast_options,
    )

    assert result.hard_feasible
    assert result.steps[1].checks[0].success
    assert result.steps[1].waste_dimension_fits is False
    assert result.steps[1].status == "failed"
    assert result.soft_step_count == 1
    assert result.soft_step_successes == 0
