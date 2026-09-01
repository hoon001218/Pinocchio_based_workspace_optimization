from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pytest

from decanting_workspace.models import BasePose
from decanting_workspace.playback_controller import PrecomputedPlaybackController
from decanting_workspace.playback_backend import CachedPlaybackBackend
from decanting_workspace.precompute import (
    CACHE_SCHEMA_VERSION,
    BasePoseGrid,
    CachedCase,
    CachedCollision,
    CachedCoordinationSummary,
    CachedEvaluationSettings,
    CachedLinearCheck,
    CachedManipulabilityEllipsoid,
    CachedPose,
    CachedPoseSample,
    CachedScenario,
    CachedStep,
    CaseGrid,
    CaseKey,
    PrecomputedCache,
)
from decanting_workspace.workflow import CoordinationMode


@dataclass(frozen=True)
class _FakeRender:
    ellipsoid_visible: bool


class _FakePlayback:
    def __init__(self, *, ellipsoid_visible: bool = True) -> None:
        self.ellipsoid_visible = ellipsoid_visible
        self.show_calls: list[tuple[np.ndarray, dict[str, object]]] = []
        self.hide_count = 0

    def show_cached_step(
        self,
        q,
        *,
        tcp_position_m,
        axis_lengths,
        axis_directions_world,
        step_name: str = "",
        successful: bool = True,
        visible: bool = True,
        ellipsoid_scale: float | None = None,
        ellipsoid_wireframe: bool | None = None,
    ) -> _FakeRender:
        self.show_calls.append(
            (
                np.asarray(q, dtype=float).copy(),
                {
                    "tcp_position_m": tcp_position_m,
                    "axis_lengths": axis_lengths,
                    "axis_directions_world": axis_directions_world,
                    "step_name": step_name,
                    "successful": successful,
                    "visible": visible,
                    "ellipsoid_scale": ellipsoid_scale,
                    "ellipsoid_wireframe": ellipsoid_wireframe,
                },
            )
        )
        return _FakeRender(self.ellipsoid_visible and visible)

    def hide_ellipsoid(self) -> None:
        self.hide_count += 1


def _pose() -> CachedPose:
    return CachedPose(
        translation_m=(0.1, 0.2, 0.3),
        rotation=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
    )


def _metric() -> CachedManipulabilityEllipsoid:
    return CachedManipulabilityEllipsoid(
        translational_axis_lengths=(3.0, 2.0, 1.0),
        translational_directions_world=(
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
        ),
        normalized_axis_lengths=(3.0, 2.0, 1.0, 0.3, 0.2, 0.1),
        normalized_directions=tuple(
            tuple(1.0 if row == column else 0.0 for column in range(6))
            for row in range(6)
        ),
        translational_volume=6.0,
        translational_min_singular_value=1.0,
        translational_condition_number=3.0,
        normalized_volume=0.036,
        normalized_min_singular_value=0.1,
        normalized_condition_number=30.0,
        characteristic_length_m=0.5,
    )


def _collision(*, colliding: bool = False) -> CachedCollision:
    return CachedCollision(
        in_collision=colliding,
        contacts=(("ur_link", "obstacle"),) if colliding else (),
        minimum_distance_m=-0.02 if colliding else 0.12,
        nearest_pair=("ur_link", "obstacle"),
    )


def _sample(
    q_value: float,
    *,
    ik_status: str = "success",
    fraction: float = 1.0,
    metric: bool = True,
) -> CachedPoseSample:
    return CachedPoseSample(
        fraction=fraction,
        target_pose=_pose(),
        solved_pose=_pose(),
        ik_status=ik_status,
        q=(q_value, q_value + 0.1, q_value + 0.2),
        arm_q=(q_value,),
        iterations=4,
        seed_index=0,
        attempted_seeds=2,
        position_error_m=1.0e-7,
        orientation_error_rad=2.0e-7,
        log_residual_norm=3.0e-7,
        rejected_by_acceptor=0,
        collision=_collision() if ik_status == "success" else None,
        manipulability=_metric() if metric and ik_status == "success" else None,
    )


@pytest.fixture()
def cached_cases():
    ur_base = BasePose(-1.0, -7.0, 0.7, 90.0)
    sr_base = BasePose(-0.2, -7.1, 0.55, -90.0)
    grid = CaseGrid(
        ur_bases=BasePoseGrid.single(ur_base),
        sr_bases=BasePoseGrid.single(sr_base),
        lift_heights_m=(0.0, 0.1),
        tote_offsets_m=(0.0,),
        corners=("southwest",),
        coordination_modes=(CoordinationMode.SEQUENTIAL,),
    )
    first_key = CaseKey(
        ur_base=ur_base,
        sr_base=sr_base,
        lift_height_m=0.0,
        tote_offset_m=0.0,
        corner="southwest",
        coordination_mode=CoordinationMode.SEQUENTIAL,
        sku="123591",
        sr_j3_stroke_m=0.30,
        clearance_m=None,
    )
    second_key = replace(first_key, lift_height_m=0.1)

    approach = CachedLinearCheck(
        name="approach",
        criticality="hard",
        status="ik_not_found",
        failed_sample_index=1,
        joint_collision=None,
        samples=(
            _sample(1.0, fraction=0.0),
            _sample(1.5, ik_status="not_found", fraction=1.0, metric=False),
        ),
    )
    contact = CachedLinearCheck(
        name="contact",
        criticality="hard",
        status="success",
        failed_sample_index=None,
        joint_collision=None,
        samples=(_sample(2.0, fraction=0.5), _sample(3.0, fraction=1.0)),
    )
    joint_failure = CachedLinearCheck(
        name="blocked_segment",
        criticality="soft",
        status="joint_segment_collision",
        failed_sample_index=0,
        joint_collision=_collision(colliding=True),
        samples=(_sample(4.0),),
    )
    steps = (
        CachedStep(1, "pick", "hard", "failed", None, (approach, contact)),
        CachedStep(2, "optional_waste", "soft", "failed", False, (joint_failure,)),
        CachedStep(6, "pour", "skip", "skipped", None, ()),
    )
    scenario = CachedScenario(
        corner="southwest",
        coordination_mode="sequential",
        hard_feasible=False,
        soft_step_successes=0,
        soft_step_count=1,
        metrics=None,
        steps=steps,
    )
    evaluated = CachedCase(
        key=first_key,
        status="evaluated",
        sr_nominal_q=(0.0, 0.0, 0.1, 0.0),
        installation_valid=True,
        placement_issues=(),
        sr_workspace=None,
        sr_keepout=None,
        scenarios=(scenario,),
        coordination_summaries=(),
    )
    invalid = CachedCase(
        key=second_key,
        status="invalid_placement",
        sr_nominal_q=(0.0, 0.0, 0.1, 0.0),
        installation_valid=False,
        placement_issues=(),
        sr_workspace=None,
        sr_keepout=None,
        scenarios=(),
        coordination_summaries=(),
    )
    settings = CachedEvaluationSettings(
        cartesian_translation_step_m=0.05,
        cartesian_rotation_step_rad=0.1,
        joint_interpolation_samples=3,
        characteristic_length_m=0.5,
        ik_position_tolerance_m=1.0e-6,
        ik_orientation_tolerance_rad=1.0e-6,
        ik_max_iterations=100,
        ik_damping=1.0e-6,
        ik_max_backtracking_steps=10,
        extra_arm_seeds=(),
        approach_distance_m=0.15,
        keepout_margin_m=0.0,
    )
    cache = PrecomputedCache(
        schema_version=CACHE_SCHEMA_VERSION,
        spec_fingerprint="test",
        grid=grid,
        settings=settings,
        cases=(evaluated, invalid),
    )
    return cache, evaluated, invalid


def test_catalog_exposes_finite_axes_and_stable_exact_cases(cached_cases):
    cache, evaluated, invalid = cached_cases
    controller = PrecomputedPlaybackController(cache, _FakePlayback())

    catalog = controller.case_catalog()

    assert catalog.ur_bases == tuple(cache.grid.ur_bases.poses())
    assert catalog.sr_bases == tuple(cache.grid.sr_bases.poses())
    assert catalog.lift_heights_m == (0.0, 0.1)
    assert catalog.coordination_modes == ("sequential",)
    assert [entry.stable_id for entry in catalog.cases] == [
        evaluated.key.stable_id,
        invalid.key.stable_id,
    ]
    assert [entry.status for entry in catalog.cases] == [
        "evaluated",
        "invalid_placement",
    ]
    assert controller.case_for(evaluated.key) is evaluated
    assert controller.case_for(evaluated.key.stable_id) is evaluated
    with pytest.raises(KeyError):
        controller.case_for(evaluated.key.stable_id[:12])
    with pytest.raises(KeyError):
        controller.case_for(replace(evaluated.key, lift_height_m=1.0e-12))


def test_step_defaults_to_last_check_and_last_sample(cached_cases):
    cache, evaluated, _ = cached_cases
    playback = _FakePlayback()
    controller = PrecomputedPlaybackController(cache, playback)

    result = controller.select(evaluated.key, step_index=1)

    assert len(playback.show_calls) == 1
    q, options = playback.show_calls[0]
    np.testing.assert_allclose(q, (3.0, 3.1, 3.2))
    assert options["step_name"] == "1:contact:1"
    assert options["successful"]
    assert options["tcp_position_m"] == (0.1, 0.2, 0.3)
    assert options["axis_lengths"] == (3.0, 2.0, 1.0)
    assert result.status == "success"
    assert result.criticality == "hard"
    assert result.step_index == 1
    assert result.step_name == "pick"
    assert result.step_status == "failed"
    assert result.check_index == 1
    assert result.check_name == "contact"
    assert result.check_status == "success"
    assert result.sample_index == 1
    assert result.sample_fraction == pytest.approx(1.0)
    assert result.ik_status == "success"
    assert result.playback_called
    assert result.ellipsoid_visible
    assert result.collision is not None
    assert result.collision.source == "sample"
    assert not result.collision.in_collision
    assert result.collision.minimum_distance_m == pytest.approx(0.12)
    assert result.metric is not None
    assert result.metric.translational_axis_lengths == pytest.approx((3.0, 2.0, 1.0))
    assert result.metric.translational_volume == pytest.approx(6.0)
    assert result.metric.normalized_condition_number == pytest.approx(30.0)


def test_exact_check_name_and_sample_index_select_requested_q(cached_cases):
    cache, evaluated, _ = cached_cases
    playback = _FakePlayback()
    controller = PrecomputedPlaybackController(cache, playback)

    result = controller.select(
        evaluated.key.stable_id,
        step_index=1,
        check_name="approach",
        sample_index=0,
    )

    np.testing.assert_allclose(playback.show_calls[0][0], (1.0, 1.1, 1.2))
    assert result.check_index == 0
    assert result.check_name == "approach"
    assert result.sample_index == 0
    assert result.status == "ik_not_found"
    # The enclosing check failed later, but this selected sample has successful
    # IK and is therefore still a valid pose to replay.
    assert result.ik_status == "success"
    assert result.playback_called


def test_unsuccessful_ik_never_calls_show_and_hides_stale_ellipsoid(cached_cases):
    cache, evaluated, _ = cached_cases
    playback = _FakePlayback()
    controller = PrecomputedPlaybackController(cache, playback)
    controller.select(evaluated.key, step_index=1)
    previous_show_count = len(playback.show_calls)

    result = controller.select(
        evaluated.key,
        step_index=1,
        check_index=0,
        sample_index=1,
    )

    assert len(playback.show_calls) == previous_show_count
    assert playback.hide_count == 1
    assert result.status == "not_found"
    assert result.ik_status == "not_found"
    assert not result.playback_called
    assert not result.ellipsoid_visible
    assert result.metric is None


def test_skipped_step_and_invalid_case_hide_without_playback(cached_cases):
    cache, evaluated, invalid = cached_cases
    playback = _FakePlayback()
    controller = PrecomputedPlaybackController(cache, playback)

    skipped = controller.select(evaluated.key, step_index=6)
    invalid_result = controller.select(invalid.key.stable_id)

    assert playback.show_calls == []
    assert playback.hide_count == 2
    assert skipped.status == "skipped"
    assert skipped.criticality == "skip"
    assert skipped.step_index == 6
    assert skipped.check_name is None
    assert skipped.ik_status is None
    assert invalid_result.status == "invalid_placement"
    assert invalid_result.case_status == "invalid_placement"
    assert invalid_result.scenario_hard_feasible is None
    assert invalid_result.step_index is None


def test_successful_ik_in_failed_check_replays_and_shows_joint_collision(cached_cases):
    cache, evaluated, _ = cached_cases
    playback = _FakePlayback()
    controller = PrecomputedPlaybackController(cache, playback)

    result = controller.select(evaluated.key, step_index=2)

    assert len(playback.show_calls) == 1
    np.testing.assert_allclose(playback.show_calls[0][0], (4.0, 4.1, 4.2))
    assert result.playback_called
    assert result.status == "joint_segment_collision"
    assert result.criticality == "soft"
    assert result.collision is not None
    assert result.collision.source == "joint_segment"
    assert result.collision.in_collision
    assert result.collision.contacts == (("ur_link", "obstacle"),)
    assert result.collision.minimum_distance_m == pytest.approx(-0.02)


@pytest.mark.parametrize(
    "selection",
    (
        {"step_index": 999},
        {"step_index": 1, "check_index": 99},
        {"step_index": 1, "check_name": "missing"},
        {"step_index": 1, "check_index": 0, "sample_index": 99},
        {"step_index": 1, "check_index": -1},
        {"step_index": 1, "check_index": 0, "check_name": "approach"},
    ),
)
def test_invalid_nested_selection_hides_stale_ellipsoid(
    cached_cases,
    selection,
):
    cache, evaluated, _ = cached_cases
    playback = _FakePlayback()
    controller = PrecomputedPlaybackController(cache, playback)
    controller.select(evaluated.key, step_index=1)

    with pytest.raises((IndexError, KeyError, ValueError)):
        controller.select(evaluated.key, **selection)

    assert playback.hide_count == 1
    assert len(playback.show_calls) == 1


def test_controller_reports_actual_playback_ellipsoid_visibility(cached_cases):
    cache, evaluated, _ = cached_cases
    playback = _FakePlayback(ellipsoid_visible=False)
    controller = PrecomputedPlaybackController(cache, playback)

    result = controller.select(evaluated.key, step_index=1)

    assert result.playback_called
    assert not result.ellipsoid_visible


def test_controller_forwards_ui_ellipsoid_controls(cached_cases):
    cache, evaluated, _ = cached_cases
    playback = _FakePlayback()
    controller = PrecomputedPlaybackController(cache, playback)

    result = controller.select(
        evaluated.key,
        step_index=1,
        ellipsoid_scale=0.27,
        show_ellipsoid=False,
        ellipsoid_wireframe=True,
    )

    options = playback.show_calls[0][1]
    assert options["ellipsoid_scale"] == pytest.approx(0.27)
    assert options["visible"] is False
    assert options["ellipsoid_wireframe"] is True
    assert not result.ellipsoid_visible


def test_unknown_stable_case_selection_hides_stale_ellipsoid(cached_cases):
    cache, evaluated, _ = cached_cases
    playback = _FakePlayback()
    controller = PrecomputedPlaybackController(cache, playback)
    controller.select(evaluated.key, step_index=1)

    with pytest.raises(KeyError):
        controller.select(evaluated.key.stable_id[:8])

    assert len(playback.show_calls) == 1
    assert playback.hide_count == 1


class _FakePresenter:
    def __init__(self) -> None:
        self.prepared = []
        self.presented = []

    def prepare_case(self, case) -> None:
        self.prepared.append(case)

    def present(self, case, selection) -> str:
        self.presented.append((case, selection))
        return "cached_test_pose"


def test_backend_catalog_is_json_ready_and_keeps_nested_task_options(cached_cases):
    import json

    cache, evaluated, invalid = cached_cases
    backend = CachedPlaybackBackend(
        cache,
        PrecomputedPlaybackController(cache, _FakePlayback()),
    )

    catalog = backend.catalog()

    assert catalog["case_count"] == 2
    assert catalog["default_case_id"] == evaluated.key.stable_id
    assert catalog["cell_feasible_case_count"] == 0
    assert catalog["cases"][0]["id"] == evaluated.key.stable_id
    assert catalog["cases"][0]["key"]["coordination_mode"] == "sequential"
    assert catalog["cases"][0]["steps"][0]["checks"][1]["name"] == "contact"
    assert (
        catalog["cases"][0]["steps"][0]["checks"][1][
            "include_in_metric_summary"
        ]
        is True
    )
    assert catalog["cases"][1]["id"] == invalid.key.stable_id
    assert catalog["cases"][1]["steps"] == []
    json.dumps(catalog, allow_nan=False)


def test_backend_prefers_first_full_cell_feasible_case(cached_cases):
    cache, evaluated, invalid = cached_cases
    feasible_summary = CachedCoordinationSummary(
        mode="sequential",
        selected_corners=("southwest",),
        hard_feasible_corners=("southwest",),
        hard_feasible=True,
        cell_feasible=True,
    )
    preferred = replace(
        invalid,
        status="evaluated",
        installation_valid=True,
        coordination_summaries=(feasible_summary,),
    )
    cache = replace(cache, cases=(evaluated, preferred))
    backend = CachedPlaybackBackend(
        cache,
        PrecomputedPlaybackController(cache, _FakePlayback()),
    )

    catalog = backend.catalog()

    assert catalog["default_case_id"] == preferred.key.stable_id
    assert catalog["cell_feasible_case_count"] == 1


def test_backend_selection_forwards_visual_controls_and_formats_status(cached_cases):
    import json

    cache, evaluated, _ = cached_cases
    playback = _FakePlayback()
    presenter = _FakePresenter()
    backend = CachedPlaybackBackend(
        cache,
        PrecomputedPlaybackController(cache, playback),
        presenter=presenter,
    )

    payload = backend.select(
        {
            "case_id": evaluated.key.stable_id,
            "step_index": 1,
            "check_index": 1,
            "sample_index": 1,
            "ellipsoid_scale": 0.31,
            "show_ellipsoid": True,
            "ellipsoid_wireframe": True,
        }
    )
    backend.select(
        {
            "case_id": evaluated.key.stable_id,
            "step_index": 1,
            "check_index": 1,
            "sample_index": 0,
            "show_ellipsoid": False,
        }
    )

    assert presenter.prepared == [evaluated]
    assert len(presenter.presented) == 2
    assert playback.show_calls[0][1]["ellipsoid_scale"] == pytest.approx(0.31)
    assert playback.show_calls[0][1]["ellipsoid_wireframe"] is True
    assert playback.show_calls[1][1]["ellipsoid_wireframe"] is False
    assert payload["pose_kind"] == "cached_test_pose"
    assert payload["step_status"] == "failed"
    assert payload["ik_status"] == "success"
    assert "Translational sigma" in payload["status_text"]
    json.dumps(payload, allow_nan=False)


@pytest.mark.parametrize("value", ("true", 1, None))
def test_backend_rejects_non_boolean_wireframe(cached_cases, value):
    cache, evaluated, _ = cached_cases
    playback = _FakePlayback()
    backend = CachedPlaybackBackend(
        cache,
        PrecomputedPlaybackController(cache, playback),
    )

    with pytest.raises(ValueError, match="ellipsoid_wireframe must be a boolean"):
        backend.select(
            {
                "case_id": evaluated.key.stable_id,
                "ellipsoid_wireframe": value,
            }
        )

    assert playback.show_calls == []
