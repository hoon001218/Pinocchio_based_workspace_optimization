from __future__ import annotations

from dataclasses import replace
import json

import numpy as np
import pytest

import decanting_workspace.precompute as precompute_module
from decanting_workspace.candidates import (
    BaseCandidateEvaluation,
    CoordinationSummary,
)
from decanting_workspace.collision_world import (
    CollisionContact,
    CollisionReport,
)
from decanting_workspace.evaluation import (
    LinearCheckResult,
    MetricSummary,
    PoseSampleResult,
    ScenarioEvaluation,
    StepEvaluation,
)
from decanting_workspace.kinematics import IKResult, ManipulabilityMetrics, frame_pose
from decanting_workspace.models import SceneState, load_scene_spec
from decanting_workspace.placement import PlacementIssue, PlacementReport
from decanting_workspace.precompute import (
    BasePoseGrid,
    CaseGrid,
    iter_case_keys,
    load_precomputed_cache,
    merge_precomputed_caches,
    precompute_cases,
    save_precomputed_cache,
    scene_spec_fingerprint,
)
from decanting_workspace.robots import floating_configuration, load_robot_bundle
from decanting_workspace.workflow import (
    CoordinationMode,
    Criticality,
    ProcessScenario,
    RigidPose,
)


@pytest.fixture(scope="module")
def spec():
    return load_scene_spec()


@pytest.fixture(scope="module")
def ur_bundle(spec):
    robot = spec.robots["ur20"]
    return load_robot_bundle(
        "ur20",
        robot.urdf_path,
        suction_proxy=robot.suction_proxy,
    )


def _single_grid(spec, **changes):
    values = dict(
        ur_bases=BasePoseGrid.single(spec.robots["ur20"].nominal_base),
        sr_bases=BasePoseGrid.single(spec.robots["sr12ia"].nominal_base),
        lift_heights_m=(0.0,),
        tote_offsets_m=(0.0,),
        corners=("southwest",),
        coordination_modes=(CoordinationMode.SEQUENTIAL,),
    )
    values.update(changes)
    return CaseGrid(**values)


def _invalid_evaluation(state, ur_base, sr_base, mode):
    issue = PlacementIssue(
        code="outside_installation_region",
        objects=("ur20_pedestal",),
        message="outside",
    )
    summary = CoordinationSummary(
        mode=mode,
        selected_corners=state.corner_samples,
        hard_feasible_corners=(),
        hard_feasible=False,
        scenarios=(),
        scenario_metrics=(),
        cell_feasible=False,
    )
    return BaseCandidateEvaluation(
        state=state,
        ur_base=ur_base,
        sr_base=sr_base,
        placement=PlacementReport(False, (issue,)),
        sr_workspace=None,
        sr_keepout=None,
        scenarios=(),
        coordination_summaries=(summary,),
    )


def test_explicit_axes_define_a_finite_deterministic_cartesian_product(spec):
    ur = spec.robots["ur20"].nominal_base
    sr = spec.robots["sr12ia"].nominal_base
    grid = CaseGrid(
        ur_bases=BasePoseGrid(
            x_m=(ur.x_m, ur.x_m + 0.01),
            y_m=(ur.y_m,),
            z_m=(ur.z_m,),
            yaw_deg=(ur.yaw_deg,),
        ),
        sr_bases=BasePoseGrid(
            x_m=(sr.x_m,),
            y_m=(sr.y_m,),
            z_m=(sr.z_m,),
            yaw_deg=(sr.yaw_deg, sr.yaw_deg + 1.0),
        ),
        lift_heights_m=(0.0, 0.2),
        tote_offsets_m=(0.0, 0.1),
        corners=("southwest", "northeast"),
        coordination_modes=("simultaneous", "sequential"),
    )

    keys = tuple(iter_case_keys(spec, grid))

    assert grid.case_count == 64
    assert len(keys) == grid.case_count
    assert len({key.stable_id for key in keys}) == grid.case_count
    assert keys[0].coordination_mode is CoordinationMode.SIMULTANEOUS
    assert keys[-1].coordination_mode is CoordinationMode.SEQUENTIAL
    assert {key.corner for key in keys} == {"southwest", "northeast"}


def test_grids_reject_implicit_or_unbounded_continuous_axes(spec):
    with pytest.raises(ValueError, match="at least one explicit grid value"):
        BasePoseGrid((), (0.0,), (0.7,), (0.0,))
    with pytest.raises(ValueError, match="finite"):
        BasePoseGrid((float("inf"),), (0.0,), (0.7,), (0.0,))
    with pytest.raises(ValueError, match="duplicate"):
        _single_grid(spec, lift_heights_m=(0.0, 0.0))

    outside_tote_range = _single_grid(spec, tote_offsets_m=(100.0,))
    with pytest.raises(ValueError, match="tote offset"):
        tuple(iter_case_keys(spec, outside_tote_range))


def test_precompute_calls_candidate_api_once_per_exact_case(spec):
    grid = _single_grid(
        spec,
        corners=("southwest", "northeast"),
        coordination_modes=("simultaneous", "sequential"),
        lift_heights_m=(0.0, 0.1),
    )
    calls = []

    def evaluator(
        received_spec,
        state,
        ur_base,
        sr_base,
        ur_bundle,
        sr_bundle,
        *,
        coordination_modes,
        options,
        approach_distance_m,
        keepout_margin_m,
    ):
        calls.append(
            (
                received_spec,
                state,
                ur_base,
                sr_base,
                coordination_modes,
                approach_distance_m,
                keepout_margin_m,
            )
        )
        return _invalid_evaluation(
            state,
            ur_base,
            sr_base,
            coordination_modes[0],
        )

    cache = precompute_cases(
        spec,
        grid,
        object(),
        object(),
        approach_distance_m=0.2,
        keepout_margin_m=0.03,
        evaluator=evaluator,
    )

    assert len(calls) == grid.case_count == 8
    assert len(cache.cases) == 8
    assert all(case.status == "invalid_placement" for case in cache.cases)
    assert all(len(call[4]) == 1 for call in calls)
    assert {(call[1].lift_height_m, call[1].corner_samples[0]) for call in calls} == {
        (0.0, "southwest"),
        (0.0, "northeast"),
        (0.1, "southwest"),
        (0.1, "northeast"),
    }
    assert all(call[5:] == (0.2, 0.03) for call in calls)


def _detailed_evaluation(spec, state, ur_base, sr_base, ur_bundle, mode):
    robot = spec.robots["ur20"]
    arm_q = np.asarray(robot.nominal_q, dtype=float)
    q = floating_configuration(ur_bundle, ur_base, arm_q)
    solved = frame_pose(ur_bundle, q, "suction_tcp")
    target = RigidPose(solved)
    ik = IKResult(
        status="success",
        q=q,
        arm_q=arm_q,
        iterations=7,
        seed_index=0,
        attempted_seeds=3,
        position_error_m=1e-7,
        orientation_error_rad=2e-7,
        log_residual_norm=3e-7,
        rejected_by_acceptor=1,
    )
    jacobian = np.zeros((6, 6))
    jacobian[:3, :3] = np.diag((3.0, 2.0, 1.0))
    jacobian[3:, 3:] = np.diag((0.6, 0.5, 0.4))
    metrics = ManipulabilityMetrics(
        jacobian=jacobian,
        translational_singular_values=np.array((3.0, 2.0, 1.0)),
        translational_volume=6.0,
        translational_min_singular_value=1.0,
        translational_condition_number=3.0,
        normalized_singular_values=np.array((3.0, 2.0, 1.0, 0.3, 0.25, 0.2)),
        normalized_volume=0.09,
        normalized_min_singular_value=0.2,
        normalized_condition_number=15.0,
        characteristic_length_m=0.5,
    )
    collision = CollisionReport(
        in_collision=False,
        contacts=(CollisionContact("wrist_3", "worktable"),),
        minimum_distance_m=0.123,
        nearest_pair=("wrist_3", "worktable"),
    )
    sample = PoseSampleResult(0.5, target, ik, collision, metrics)
    check = LinearCheckResult(
        name="cached_check",
        criticality=Criticality.HARD,
        status="success",
        samples=(sample,),
        include_in_metric_summary=False,
    )
    step = StepEvaluation(
        index=1,
        name="cached_step",
        criticality=Criticality.HARD,
        status="success",
        checks=(check,),
    )
    scenario = ProcessScenario(state, state.corner_samples[0], mode)
    summary_metrics = MetricSummary(
        sample_count=1,
        translational_volume_min=6.0,
        translational_volume_mean=6.0,
        translational_sigma_min=1.0,
        normalized_volume_min=0.09,
        normalized_volume_mean=0.09,
        normalized_sigma_min=0.2,
        normalized_condition_max=15.0,
    )
    scenario_evaluation = ScenarioEvaluation(
        scenario=scenario,
        steps=(step,),
        hard_feasible=True,
        soft_step_successes=0,
        soft_step_count=0,
        metrics=summary_metrics,
    )
    coordination = CoordinationSummary(
        mode=mode,
        selected_corners=state.corner_samples,
        hard_feasible_corners=state.corner_samples,
        hard_feasible=True,
        scenarios=(scenario_evaluation,),
        scenario_metrics=(summary_metrics,),
        cell_feasible=True,
    )
    return BaseCandidateEvaluation(
        state=state,
        ur_base=ur_base,
        sr_base=sr_base,
        placement=PlacementReport(True, ()),
        sr_workspace=None,
        sr_keepout=None,
        scenarios=(scenario_evaluation,),
        coordination_summaries=(coordination,),
    )


def test_cache_keeps_q_poses_ellipsoid_directions_collision_and_status(
    spec,
    ur_bundle,
):
    grid = _single_grid(spec)

    def evaluator(
        received_spec,
        state,
        ur_base,
        sr_base,
        received_ur_bundle,
        sr_bundle,
        *,
        coordination_modes,
        **kwargs,
    ):
        return _detailed_evaluation(
            received_spec,
            state,
            ur_base,
            sr_base,
            received_ur_bundle,
            coordination_modes[0],
        )

    cache = precompute_cases(
        spec,
        grid,
        ur_bundle,
        object(),
        evaluator=evaluator,
    )
    case = cache.cases[0]
    sample = case.scenarios[0].steps[0].checks[0].samples[0]

    assert case.status == "evaluated"
    assert case.sr_nominal_q == spec.robots["sr12ia"].nominal_q
    assert sample.ik_status == "success"
    assert len(sample.q) == ur_bundle.model.nq
    assert len(sample.arm_q) == 6
    assert sample.target_pose == sample.solved_pose
    assert sample.manipulability is not None
    assert sample.manipulability.translational_axis_lengths == pytest.approx(
        (3.0, 2.0, 1.0)
    )
    np.testing.assert_allclose(
        sample.manipulability.translational_directions_world,
        np.eye(3),
    )
    assert sample.collision is not None
    assert sample.collision.minimum_distance_m == pytest.approx(0.123)
    assert sample.collision.contacts == (("wrist_3", "worktable"),)
    assert case.scenarios[0].steps[0].checks[0].status == "success"
    assert not case.scenarios[0].steps[0].checks[0].include_in_metric_summary


def test_strict_json_round_trip_and_scene_fingerprint_validation(
    spec,
    ur_bundle,
    tmp_path,
):
    grid = _single_grid(spec)

    def evaluator(
        received_spec,
        state,
        ur_base,
        sr_base,
        received_ur_bundle,
        sr_bundle,
        *,
        coordination_modes,
        **kwargs,
    ):
        return _detailed_evaluation(
            received_spec,
            state,
            ur_base,
            sr_base,
            received_ur_bundle,
            coordination_modes[0],
        )

    cache = precompute_cases(
        spec,
        grid,
        ur_bundle,
        object(),
        evaluator=evaluator,
    )
    path = save_precomputed_cache(cache, tmp_path / "nested" / "cases.json")
    loaded = load_precomputed_cache(path, spec=spec)
    compressed_path = save_precomputed_cache(cache, tmp_path / "cases.json.gz")
    compressed = load_precomputed_cache(compressed_path, spec=spec)

    assert loaded == cache
    assert compressed == cache
    assert compressed_path.stat().st_size < path.stat().st_size
    assert loaded.case_for(cache.cases[0].key) == cache.cases[0]
    text = path.read_text(encoding="utf-8")
    assert "NaN" not in text and "Infinity" not in text

    changed_spec = replace(spec, floor_z_m=0.001)
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        load_precomputed_cache(path, spec=changed_spec)

    raw = json.loads(path.read_text(encoding="utf-8"))
    raw["cases"][0]["scenarios"][0]["steps"][0]["checks"][0][
        "include_in_metric_summary"
    ] = "false"
    invalid_bool = tmp_path / "invalid-bool.json"
    invalid_bool.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="include_in_metric_summary.*boolean"):
        load_precomputed_cache(invalid_bool)


def test_workflow_contract_version_changes_cache_fingerprint(spec, monkeypatch):
    original = scene_spec_fingerprint(spec)
    monkeypatch.setattr(
        precompute_module,
        "WORKFLOW_SCHEMA_VERSION",
        precompute_module.WORKFLOW_SCHEMA_VERSION + 1,
    )

    assert scene_spec_fingerprint(spec) != original


def test_robot_fingerprint_is_checkout_path_independent_and_hashes_geometry(
    spec,
    tmp_path,
):
    fingerprints = []
    copied_specs = []
    for directory_name in ("checkout-a", "checkout-b"):
        directory = tmp_path / directory_name
        directory.mkdir()
        mesh = directory / "link.obj"
        mesh.write_text("v 0 0 0\n", encoding="utf-8")
        urdf = directory / "robot.urdf"
        urdf.write_text(
            "<robot name='portable'><link name='base'><visual><geometry>"
            "<mesh filename='link.obj'/></geometry></visual></link></robot>",
            encoding="utf-8",
        )
        robots = dict(spec.robots)
        robots["ur20"] = replace(robots["ur20"], urdf_path=urdf)
        copied = replace(spec, robots=robots)
        copied_specs.append(copied)
        fingerprints.append(scene_spec_fingerprint(copied))

    assert fingerprints[0] == fingerprints[1]

    (tmp_path / "checkout-b" / "link.obj").write_text(
        "v 0 0 1\n",
        encoding="utf-8",
    )
    assert scene_spec_fingerprint(copied_specs[1]) != fingerprints[0]


def test_complete_disjoint_lift_shards_merge_without_sparse_union(spec):
    def evaluator(
        received_spec,
        state,
        ur_base,
        sr_base,
        ur_bundle,
        sr_bundle,
        *,
        coordination_modes,
        **kwargs,
    ):
        return _invalid_evaluation(
            state,
            ur_base,
            sr_base,
            coordination_modes[0],
        )

    low = precompute_cases(
        spec,
        _single_grid(spec, lift_heights_m=(0.0,)),
        object(),
        object(),
        evaluator=evaluator,
    )
    high = precompute_cases(
        spec,
        _single_grid(spec, lift_heights_m=(0.76,)),
        object(),
        object(),
        evaluator=evaluator,
    )

    merged = merge_precomputed_caches(
        (low, high),
        varying_axis="lift_heights_m",
    )

    assert merged.grid.lift_heights_m == (0.0, 0.76)
    assert len(merged.cases) == merged.grid.case_count == 2
    with pytest.raises(ValueError, match="overlap"):
        merge_precomputed_caches((low, low), varying_axis="lift_heights_m")
