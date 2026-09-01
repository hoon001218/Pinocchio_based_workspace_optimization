"""Pure selection controller between precomputed cases and MeshCat playback.

The controller contains no HTTP, HTML, or MeshCat-node logic.  It resolves an
exact cached case and one nested step/check/sample, asks an injected playback
port to display only successful IK configurations, and returns immutable data
that a future UI can render directly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from .models import BasePose
from .precompute import (
    CachedCase,
    CachedCollision,
    CachedLinearCheck,
    CachedManipulabilityEllipsoid,
    CachedPoseSample,
    CachedScenario,
    CachedStep,
    CaseKey,
    PrecomputedCache,
)


class PlaybackPort(Protocol):
    """Minimal subset implemented by ``UR20MeshcatPlayback``."""

    def show_cached_step(
        self,
        q: object,
        *,
        tcp_position_m: object,
        axis_lengths: object,
        axis_directions_world: object,
        step_name: str = "",
        successful: bool = True,
        visible: bool = True,
        ellipsoid_scale: float | None = None,
        ellipsoid_wireframe: bool | None = None,
    ) -> object: ...

    def hide_ellipsoid(self) -> None: ...


@dataclass(frozen=True)
class CaseCatalogEntry:
    stable_id: str
    key: CaseKey
    status: str


@dataclass(frozen=True)
class CaseParameterCatalog:
    """Finite selector axes plus exact stable case identifiers."""

    ur_bases: tuple[BasePose, ...]
    sr_bases: tuple[BasePose, ...]
    lift_heights_m: tuple[float, ...]
    tote_offsets_m: tuple[float, ...]
    corners: tuple[str, ...]
    coordination_modes: tuple[str, ...]
    skus: tuple[str, ...]
    sr_j3_strokes_m: tuple[float, ...]
    clearances_m: tuple[float | None, ...]
    cases: tuple[CaseCatalogEntry, ...]


@dataclass(frozen=True)
class CollisionDisplay:
    source: str
    in_collision: bool
    contacts: tuple[tuple[str, str], ...]
    minimum_distance_m: float | None
    nearest_pair: tuple[str, str] | None


@dataclass(frozen=True)
class MetricDisplay:
    translational_axis_lengths: tuple[float, float, float]
    translational_volume: float
    translational_min_singular_value: float
    translational_condition_number: float | None
    normalized_volume: float
    normalized_min_singular_value: float
    normalized_condition_number: float | None


@dataclass(frozen=True)
class PlaybackSelection:
    """Resolved cache hierarchy and display data for one UI selection."""

    case_id: str
    case_key: CaseKey
    status: str
    criticality: str | None
    case_status: str
    scenario_hard_feasible: bool | None
    step_index: int | None
    step_name: str | None
    step_status: str | None
    check_index: int | None
    check_name: str | None
    check_status: str | None
    sample_index: int | None
    sample_fraction: float | None
    ik_status: str | None
    collision: CollisionDisplay | None
    metric: MetricDisplay | None
    playback_called: bool
    ellipsoid_visible: bool


class PrecomputedPlaybackController:
    """Resolve exact cache selections and coordinate playback side effects."""

    def __init__(self, cache: PrecomputedCache, playback: PlaybackPort) -> None:
        self.cache = cache
        self.playback = playback

    def case_catalog(self) -> CaseParameterCatalog:
        """Return deterministic finite axes and one stable ID per exact case."""

        grid = self.cache.grid
        return CaseParameterCatalog(
            ur_bases=tuple(grid.ur_bases.poses()),
            sr_bases=tuple(grid.sr_bases.poses()),
            lift_heights_m=grid.lift_heights_m,
            tote_offsets_m=grid.tote_offsets_m,
            corners=grid.corners,
            coordination_modes=tuple(mode.value for mode in grid.coordination_modes),
            skus=grid.skus,
            sr_j3_strokes_m=grid.sr_j3_strokes_m,
            clearances_m=grid.clearances_m,
            cases=tuple(
                CaseCatalogEntry(case.key.stable_id, case.key, case.status)
                for case in self.cache.cases
            ),
        )

    def case_for(self, selector: CaseKey | str) -> CachedCase:
        """Resolve either an exact physical key or a complete stable ID.

        Stable-ID prefixes and approximate numeric matching are deliberately
        unsupported: a replay must identify the exact case that was computed.
        """

        if isinstance(selector, CaseKey):
            return self.cache.case_for(selector)
        if not isinstance(selector, str):
            raise TypeError("case selector must be a CaseKey or stable-id string")
        for case in self.cache.cases:
            if case.key.stable_id == selector:
                return case
        raise KeyError(selector)

    def select(
        self,
        case_selector: CaseKey | str,
        *,
        step_index: int | None = None,
        check_index: int | None = None,
        check_name: str | None = None,
        sample_index: int | None = None,
        ellipsoid_scale: float | None = None,
        show_ellipsoid: bool = True,
        ellipsoid_wireframe: bool = False,
    ) -> PlaybackSelection:
        """Select and optionally display one cached pose sample.

        With no nested selectors, the final step, its final check, and that
        check's final sample are selected.  Supplying ``step_index`` uses the
        physical process-step index; check/sample indices are zero-based tuple
        positions.  ``check_name`` is an exact alternative to ``check_index``.
        """

        if check_index is not None and check_name is not None:
            self.playback.hide_ellipsoid()
            raise ValueError("select either check_index or check_name, not both")
        try:
            case = self.case_for(case_selector)
            scenario = _scenario_for_case(case)
        except (KeyError, TypeError, ValueError):
            self.playback.hide_ellipsoid()
            raise
        if scenario is None:
            self.playback.hide_ellipsoid()
            return _selection(
                case,
                status=case.status,
                scenario=None,
                step=None,
                check=None,
                check_position=None,
                sample=None,
                sample_position=None,
            )

        try:
            step = _select_step(scenario, step_index)
            if not step.checks:
                self.playback.hide_ellipsoid()
                return _selection(
                    case,
                    status=step.status,
                    scenario=scenario,
                    step=step,
                    check=None,
                    check_position=None,
                    sample=None,
                    sample_position=None,
                )
            check, selected_check_index = _select_check(
                step,
                check_index=check_index,
                check_name=check_name,
            )
            if not check.samples:
                self.playback.hide_ellipsoid()
                return _selection(
                    case,
                    status=check.status,
                    scenario=scenario,
                    step=step,
                    check=check,
                    check_position=selected_check_index,
                    sample=None,
                    sample_position=None,
                )
            sample, selected_sample_index = _select_sample(check, sample_index)
        except (IndexError, KeyError, ValueError):
            # Never leave the previous selection's ellipsoid visible after an
            # invalid UI selection.
            self.playback.hide_ellipsoid()
            raise

        playback_called = sample.ik_status == "success"
        ellipsoid_visible = False
        if playback_called:
            metric = sample.manipulability
            render = self.playback.show_cached_step(
                np.asarray(sample.q, dtype=float),
                tcp_position_m=sample.solved_pose.translation_m,
                axis_lengths=(
                    None if metric is None else metric.translational_axis_lengths
                ),
                axis_directions_world=(
                    None
                    if metric is None
                    else metric.translational_directions_world
                ),
                step_name=f"{step.index}:{check.name}:{selected_sample_index}",
                successful=True,
                visible=bool(show_ellipsoid),
                ellipsoid_scale=ellipsoid_scale,
                ellipsoid_wireframe=ellipsoid_wireframe,
            )
            ellipsoid_visible = bool(
                getattr(render, "ellipsoid_visible", False)
            )
        else:
            self.playback.hide_ellipsoid()

        status = check.status if sample.ik_status == "success" else sample.ik_status
        return _selection(
            case,
            status=status,
            scenario=scenario,
            step=step,
            check=check,
            check_position=selected_check_index,
            sample=sample,
            sample_position=selected_sample_index,
            playback_called=playback_called,
            ellipsoid_visible=ellipsoid_visible,
        )


def _scenario_for_case(case: CachedCase) -> CachedScenario | None:
    if not case.scenarios:
        return None
    expected_mode = case.key.coordination_mode.value
    matches = tuple(
        scenario
        for scenario in case.scenarios
        if scenario.corner == case.key.corner
        and scenario.coordination_mode == expected_mode
    )
    if len(matches) != 1:
        raise ValueError(
            "cached case must contain exactly one matching corner/mode scenario"
        )
    return matches[0]


def _select_step(
    scenario: CachedScenario,
    step_index: int | None,
) -> CachedStep:
    if not scenario.steps:
        raise ValueError("cached scenario has no task steps")
    if step_index is None:
        return scenario.steps[-1]
    step_index = _nonnegative_index(step_index, "step_index")
    matches = tuple(step for step in scenario.steps if step.index == step_index)
    if len(matches) != 1:
        raise KeyError(f"unknown or duplicate process step index: {step_index}")
    return matches[0]


def _select_check(
    step: CachedStep,
    *,
    check_index: int | None,
    check_name: str | None,
) -> tuple[CachedLinearCheck, int]:
    if check_name is not None:
        matches = tuple(
            (index, check)
            for index, check in enumerate(step.checks)
            if check.name == check_name
        )
        if len(matches) != 1:
            raise KeyError(f"unknown or duplicate check name: {check_name}")
        index, check = matches[0]
        return check, index
    index = len(step.checks) - 1 if check_index is None else _nonnegative_index(
        check_index,
        "check_index",
    )
    try:
        return step.checks[index], index
    except IndexError as exc:
        raise IndexError(f"check_index out of range: {index}") from exc


def _select_sample(
    check: CachedLinearCheck,
    sample_index: int | None,
) -> tuple[CachedPoseSample, int]:
    index = len(check.samples) - 1 if sample_index is None else _nonnegative_index(
        sample_index,
        "sample_index",
    )
    try:
        return check.samples[index], index
    except IndexError as exc:
        raise IndexError(f"sample_index out of range: {index}") from exc


def _nonnegative_index(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _selection(
    case: CachedCase,
    *,
    status: str,
    scenario: CachedScenario | None,
    step: CachedStep | None,
    check: CachedLinearCheck | None,
    check_position: int | None,
    sample: CachedPoseSample | None,
    sample_position: int | None,
    playback_called: bool = False,
    ellipsoid_visible: bool = False,
) -> PlaybackSelection:
    return PlaybackSelection(
        case_id=case.key.stable_id,
        case_key=case.key,
        status=status,
        criticality=(
            check.criticality
            if check is not None
            else step.criticality if step is not None else None
        ),
        case_status=case.status,
        scenario_hard_feasible=(
            scenario.hard_feasible if scenario is not None else None
        ),
        step_index=step.index if step is not None else None,
        step_name=step.name if step is not None else None,
        step_status=step.status if step is not None else None,
        check_index=check_position,
        check_name=check.name if check is not None else None,
        check_status=check.status if check is not None else None,
        sample_index=sample_position,
        sample_fraction=sample.fraction if sample is not None else None,
        ik_status=sample.ik_status if sample is not None else None,
        collision=_collision_display(check, sample),
        metric=(
            _metric_display(sample.manipulability)
            if sample is not None and sample.manipulability is not None
            else None
        ),
        playback_called=playback_called,
        ellipsoid_visible=ellipsoid_visible,
    )


def _collision_display(
    check: CachedLinearCheck | None,
    sample: CachedPoseSample | None,
) -> CollisionDisplay | None:
    report: CachedCollision | None = None
    source = ""
    if (
        check is not None
        and check.joint_collision is not None
        and check.joint_collision.in_collision
    ):
        report = check.joint_collision
        source = "joint_segment"
    elif sample is not None and sample.collision is not None:
        report = sample.collision
        source = "sample"
    elif check is not None and check.joint_collision is not None:
        report = check.joint_collision
        source = "joint_segment"
    if report is None:
        return None
    return CollisionDisplay(
        source=source,
        in_collision=report.in_collision,
        contacts=report.contacts,
        minimum_distance_m=report.minimum_distance_m,
        nearest_pair=report.nearest_pair,
    )


def _metric_display(metric: CachedManipulabilityEllipsoid) -> MetricDisplay:
    return MetricDisplay(
        translational_axis_lengths=metric.translational_axis_lengths,
        translational_volume=metric.translational_volume,
        translational_min_singular_value=(
            metric.translational_min_singular_value
        ),
        translational_condition_number=(
            metric.translational_condition_number
        ),
        normalized_volume=metric.normalized_volume,
        normalized_min_singular_value=metric.normalized_min_singular_value,
        normalized_condition_number=metric.normalized_condition_number,
    )
