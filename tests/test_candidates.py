from __future__ import annotations

from decanting_workspace import BasePose, SceneState, load_scene_spec
from decanting_workspace import candidates
from decanting_workspace.evaluation import MetricSummary, ScenarioEvaluation
from decanting_workspace.models import BoxPrimitive
from decanting_workspace.workflow import CoordinationMode


def _metrics(value: float) -> MetricSummary:
    return MetricSummary(
        sample_count=1,
        translational_volume_min=value,
        translational_volume_mean=value,
        translational_sigma_min=value,
        normalized_volume_min=value,
        normalized_volume_mean=value,
        normalized_sigma_min=value,
        normalized_condition_max=1.0,
    )


def _fake_evaluation(scenario, *, hard_feasible=True, metric_value=1.0):
    # A soft-step failure is represented by zero soft successes; hard grouping
    # must depend only on ScenarioEvaluation.hard_feasible.
    return ScenarioEvaluation(
        scenario=scenario,
        steps=(),
        hard_feasible=hard_feasible,
        soft_step_successes=0,
        soft_step_count=1,
        metrics=_metrics(metric_value),
    )


def test_nominal_candidate_orchestrates_each_corner_and_mode_once(monkeypatch):
    spec = load_scene_spec()
    state = SceneState(corner_samples=("southwest", "northeast"))
    ur_base = spec.robots["ur20"].nominal_base
    sr_base = spec.robots["sr12ia"].nominal_base
    ur_bundle = object()
    sr_bundle = object()
    keepout = BoxPrimitive(
        name="fk_sr_keepout",
        role="concurrent_exclusion",
        center_m=(0.0, 0.0, 1.0),
        size_m=(1.0, 1.0, 2.0),
    )
    keepout_calls = []
    build_calls = []
    evaluate_calls = []

    def fake_keepout(received_spec, snapshot, received_bundle, *, margin_m):
        keepout_calls.append((received_spec, snapshot, received_bundle, margin_m))
        return keepout

    def fake_build(
        received_spec,
        snapshot,
        scenario,
        received_bundle,
        *,
        sr_keepout,
        approach_distance_m,
    ):
        build_calls.append(
            (
                received_spec,
                snapshot,
                scenario,
                received_bundle,
                sr_keepout,
                approach_distance_m,
            )
        )
        return (f"steps-{scenario.corner}-{scenario.coordination_mode.value}",)

    def fake_evaluate(
        received_spec,
        snapshot,
        scenario,
        received_bundle,
        steps,
        *,
        options,
    ):
        evaluate_calls.append(
            (received_spec, snapshot, scenario, received_bundle, steps, options)
        )
        return _fake_evaluation(
            scenario,
            metric_value=float(len(evaluate_calls)),
        )

    monkeypatch.setattr(candidates, "sr_concurrent_exclusion_box", fake_keepout)
    monkeypatch.setattr(candidates, "configure_sr_j3_stroke", lambda *args: None)
    monkeypatch.setattr(
        candidates,
        "evaluate_sr_cutting_workspace",
        lambda *args, **kwargs: type("Workspace", (), {"feasible": True})(),
    )
    monkeypatch.setattr(candidates, "build_ur20_task_sequence", fake_build)
    monkeypatch.setattr(candidates, "evaluate_task_sequence", fake_evaluate)

    result = candidates.evaluate_base_candidate(
        spec,
        state,
        ur_base,
        sr_base,
        ur_bundle,
        sr_bundle,
        approach_distance_m=0.20,
        keepout_margin_m=0.03,
    )

    assert result.installation_valid
    assert result.sr_keepout is keepout
    assert len(keepout_calls) == 1
    assert keepout_calls[0][2] is sr_bundle
    assert keepout_calls[0][3] == 0.03
    assert len(build_calls) == 4
    assert len(evaluate_calls) == 4
    assert all(call[1].state.corner_samples == (call[2].corner,) for call in build_calls)
    assert all(call[3] is ur_bundle for call in build_calls)
    assert all(call[4] is keepout and call[5] == 0.20 for call in build_calls)
    assert len(result.scenarios) == 4
    assert [summary.mode for summary in result.coordination_summaries] == [
        CoordinationMode.SIMULTANEOUS,
        CoordinationMode.SEQUENTIAL,
    ]
    assert all(summary.hard_feasible for summary in result.coordination_summaries)
    assert all(summary.cell_feasible for summary in result.coordination_summaries)
    # The fake soft step fails everywhere; it must not change hard feasibility.
    assert all(
        scenario.soft_step_successes == 0 and scenario.hard_feasible
        for scenario in result.scenarios
    )
    assert [
        metric.translational_volume_min
        for summary in result.coordination_summaries
        for metric in summary.scenario_metrics
    ] == [1.0, 3.0, 2.0, 4.0]


def test_invalid_installation_short_circuits_sr_fk_workflow_and_ik(monkeypatch):
    spec = load_scene_spec()
    state = SceneState(corner_samples=("southwest",))
    nominal = spec.robots["ur20"].nominal_base
    outside = BasePose(
        spec.installation_region.raw_xy_min_m[0],
        nominal.y_m,
        nominal.z_m,
        nominal.yaw_deg,
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("kinematics must not run for invalid placement")

    monkeypatch.setattr(candidates, "sr_concurrent_exclusion_box", forbidden)
    monkeypatch.setattr(candidates, "configure_sr_j3_stroke", forbidden)
    monkeypatch.setattr(candidates, "evaluate_sr_cutting_workspace", forbidden)
    monkeypatch.setattr(candidates, "build_ur20_task_sequence", forbidden)
    monkeypatch.setattr(candidates, "evaluate_task_sequence", forbidden)

    result = candidates.evaluate_base_candidate(
        spec,
        state,
        outside,
        spec.robots["sr12ia"].nominal_base,
        object(),
        object(),
    )

    assert not result.installation_valid
    assert result.sr_keepout is None
    assert result.scenarios == ()
    assert all(
        not summary.hard_feasible and summary.scenarios == ()
        for summary in result.coordination_summaries
    )


def test_hard_feasibility_is_grouped_as_all_corners_per_mode(monkeypatch):
    spec = load_scene_spec()
    state = SceneState(corner_samples=("southwest", "northeast"))
    keepout = BoxPrimitive(
        name="keepout",
        role="concurrent_exclusion",
        center_m=(0.0, 0.0, 1.0),
        size_m=(1.0, 1.0, 2.0),
    )
    monkeypatch.setattr(
        candidates,
        "sr_concurrent_exclusion_box",
        lambda *args, **kwargs: keepout,
    )
    monkeypatch.setattr(candidates, "configure_sr_j3_stroke", lambda *args: None)
    monkeypatch.setattr(
        candidates,
        "evaluate_sr_cutting_workspace",
        lambda *args, **kwargs: type("Workspace", (), {"feasible": True})(),
    )
    monkeypatch.setattr(
        candidates,
        "build_ur20_task_sequence",
        lambda *args, **kwargs: (),
    )

    def fake_evaluate(spec, snapshot, scenario, bundle, steps, *, options):
        feasible = not (
            scenario.coordination_mode is CoordinationMode.SEQUENTIAL
            and scenario.corner == "northeast"
        )
        return _fake_evaluation(scenario, hard_feasible=feasible)

    monkeypatch.setattr(candidates, "evaluate_task_sequence", fake_evaluate)

    result = candidates.evaluate_base_candidate(
        spec,
        state,
        spec.robots["ur20"].nominal_base,
        spec.robots["sr12ia"].nominal_base,
        object(),
        object(),
        coordination_modes=(
            CoordinationMode.SIMULTANEOUS,
            CoordinationMode.SEQUENTIAL,
        ),
    )
    summaries = {summary.mode: summary for summary in result.coordination_summaries}

    assert summaries[CoordinationMode.SIMULTANEOUS].hard_feasible
    assert summaries[CoordinationMode.SIMULTANEOUS].hard_feasible_corners == (
        "southwest",
        "northeast",
    )
    assert not summaries[CoordinationMode.SEQUENTIAL].hard_feasible
    assert summaries[CoordinationMode.SEQUENTIAL].hard_feasible_corners == (
        "southwest",
    )
    assert len(summaries[CoordinationMode.SEQUENTIAL].scenario_metrics) == 2


def test_duplicate_coordination_modes_are_evaluated_once(monkeypatch):
    spec = load_scene_spec()
    state = SceneState(corner_samples=("southwest",))
    keepout = BoxPrimitive(
        name="keepout",
        role="concurrent_exclusion",
        center_m=(0.0, 0.0, 1.0),
        size_m=(1.0, 1.0, 2.0),
    )
    monkeypatch.setattr(
        candidates,
        "sr_concurrent_exclusion_box",
        lambda *args, **kwargs: keepout,
    )
    monkeypatch.setattr(candidates, "configure_sr_j3_stroke", lambda *args: None)
    monkeypatch.setattr(
        candidates,
        "evaluate_sr_cutting_workspace",
        lambda *args, **kwargs: type("Workspace", (), {"feasible": True})(),
    )
    monkeypatch.setattr(
        candidates,
        "build_ur20_task_sequence",
        lambda *args, **kwargs: (),
    )
    monkeypatch.setattr(
        candidates,
        "evaluate_task_sequence",
        lambda spec, snapshot, scenario, bundle, steps, *, options: _fake_evaluation(
            scenario
        ),
    )

    result = candidates.evaluate_base_candidate(
        spec,
        state,
        spec.robots["ur20"].nominal_base,
        spec.robots["sr12ia"].nominal_base,
        object(),
        object(),
        coordination_modes=("sequential", CoordinationMode.SEQUENTIAL),
    )

    assert len(result.scenarios) == 1
    assert len(result.coordination_summaries) == 1
    assert result.coordination_summaries[0].mode is CoordinationMode.SEQUENTIAL


def test_scara_workspace_failure_is_separate_from_ur_hard_result(monkeypatch):
    spec = load_scene_spec()
    state = SceneState(corner_samples=("southwest",))
    keepout = BoxPrimitive(
        name="keepout",
        role="concurrent_exclusion",
        center_m=(0.0, 0.0, 1.0),
        size_m=(1.0, 1.0, 2.0),
    )
    workspace = type("Workspace", (), {"feasible": False})()
    monkeypatch.setattr(
        candidates, "evaluate_sr_cutting_workspace", lambda *args, **kwargs: workspace
    )
    monkeypatch.setattr(candidates, "configure_sr_j3_stroke", lambda *args: None)
    monkeypatch.setattr(
        candidates,
        "sr_concurrent_exclusion_box",
        lambda *args, **kwargs: keepout,
    )
    monkeypatch.setattr(candidates, "build_ur20_task_sequence", lambda *args, **kwargs: ())
    monkeypatch.setattr(
        candidates,
        "evaluate_task_sequence",
        lambda spec, snapshot, scenario, bundle, steps, *, options: _fake_evaluation(
            scenario,
            hard_feasible=True,
        ),
    )

    result = candidates.evaluate_base_candidate(
        spec,
        state,
        spec.robots["ur20"].nominal_base,
        spec.robots["sr12ia"].nominal_base,
        object(),
        object(),
        coordination_modes=(CoordinationMode.SEQUENTIAL,),
    )

    summary = result.coordination_summaries[0]
    assert result.sr_workspace is workspace
    assert summary.hard_feasible
    assert not summary.cell_feasible
