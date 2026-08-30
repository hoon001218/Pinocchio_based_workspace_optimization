"""High-level base-candidate orchestration immediately before optimization.

This adapter deliberately preserves per-scenario results and metrics.  It
does not assign weights, combine manipulability values, or rank candidates.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

from .collision import sr_concurrent_exclusion_box
from .evaluation import (
    EvaluationOptions,
    MetricSummary,
    ScenarioEvaluation,
    evaluate_task_sequence,
)
from .models import BasePose, BoxPrimitive, SceneSpec, SceneState
from .placement import PlacementReport, validate_base_placement
from .robots import RobotBundle, configure_sr_j3_stroke
from .scara_workspace import (
    ScaraTaskWorkspaceReport,
    evaluate_sr_cutting_workspace,
)
from .scene import materialize_scene
from .workflow import (
    CoordinationMode,
    build_ur20_task_sequence,
    expand_process_scenarios,
)


@dataclass(frozen=True)
class CoordinationSummary:
    """UR Hard results and combined SR/UR feasibility for one policy.

    ``hard_feasible`` concerns the UR process across every selected corner;
    ``cell_feasible`` additionally requires the conservative SR cutting
    workspace to pass.
    """

    mode: CoordinationMode
    selected_corners: tuple[str, ...]
    hard_feasible_corners: tuple[str, ...]
    hard_feasible: bool
    scenarios: tuple[ScenarioEvaluation, ...]
    scenario_metrics: tuple[MetricSummary | None, ...]
    cell_feasible: bool

    @property
    def evaluated_corner_count(self) -> int:
        return len(self.scenarios)


@dataclass(frozen=True)
class BaseCandidateEvaluation:
    """Raw pre-optimizer evaluation for one UR/SR base-pose pair."""

    state: SceneState
    ur_base: BasePose
    sr_base: BasePose
    placement: PlacementReport
    sr_workspace: ScaraTaskWorkspaceReport | None
    sr_keepout: BoxPrimitive | None
    scenarios: tuple[ScenarioEvaluation, ...]
    coordination_summaries: tuple[CoordinationSummary, ...]

    @property
    def installation_valid(self) -> bool:
        return self.placement.valid


def evaluate_base_candidate(
    spec: SceneSpec,
    state: SceneState,
    ur_base: BasePose,
    sr_base: BasePose,
    ur_bundle: RobotBundle,
    sr_bundle: RobotBundle,
    *,
    coordination_modes: Iterable[CoordinationMode | str] = (
        CoordinationMode.SIMULTANEOUS,
        CoordinationMode.SEQUENTIAL,
    ),
    options: EvaluationOptions | None = None,
    approach_distance_m: float = 0.15,
    keepout_margin_m: float = 0.0,
) -> BaseCandidateEvaluation:
    """Evaluate every selected corner independently for one base candidate.

    Installation geometry is validated before SR forward kinematics, task
    construction, or UR IK.  An invalid candidate therefore returns a useful
    placement report without paying any kinematics cost.  For a valid
    candidate, the SR exclusion box is computed once and reused across every
    corner/mode scenario because its bounds depend on the SR base and state,
    not on the active pallet corner.
    """

    approach_distance = _positive_finite(
        approach_distance_m,
        "approach_distance_m",
    )
    keepout_margin = _nonnegative_finite(
        keepout_margin_m,
        "keepout_margin_m",
    )

    placement_snapshot = materialize_scene(
        spec,
        state,
        ur_base=ur_base,
        sr_base=sr_base,
    )
    placement_clearance = 0.0 if state.clearance_m is None else state.clearance_m
    placement = validate_base_placement(
        spec,
        placement_snapshot,
        clearance_m=placement_clearance,
    )
    modes = _normalize_coordination_modes(coordination_modes)
    selected_corners = tuple(state.corner_samples)

    if not placement.valid:
        empty_summaries = tuple(
            CoordinationSummary(
                mode=mode,
                selected_corners=selected_corners,
                hard_feasible_corners=(),
                hard_feasible=False,
                scenarios=(),
                scenario_metrics=(),
                cell_feasible=False,
            )
            for mode in modes
        )
        return BaseCandidateEvaluation(
            state=state,
            ur_base=ur_base,
            sr_base=sr_base,
            placement=placement,
            sr_workspace=None,
            sr_keepout=None,
            scenarios=(),
            coordination_summaries=empty_summaries,
        )

    # The SCARA contributes a conservative task-workspace decision rather than
    # a manipulability objective.  It uses the real SKU top face and the
    # selected J3 option before the UR process is evaluated.
    sr_workspace = evaluate_sr_cutting_workspace(spec, placement_snapshot)

    # Keep the selected 300/450 mm hardware option in the same Pinocchio model
    # that supplies the conservative geometry bounds.  Repeated calls are
    # safe because the bundle retains its native limits.
    configure_sr_j3_stroke(sr_bundle, state.sr_j3_stroke_m)

    # This helper performs Pinocchio FK over the SR collision model.  Keeping
    # it here (rather than falling back to a footprint-only box) ensures the
    # concurrent exclusion includes the actual selected SR posture.
    sr_keepout = sr_concurrent_exclusion_box(
        spec,
        placement_snapshot,
        sr_bundle,
        margin_m=keepout_margin,
    )
    process_scenarios = expand_process_scenarios(spec, state, modes)
    evaluations: list[ScenarioEvaluation] = []
    for scenario in process_scenarios:
        scenario_snapshot = materialize_scene(
            spec,
            scenario.scene_state,
            ur_base=ur_base,
            sr_base=sr_base,
        )
        steps = build_ur20_task_sequence(
            spec,
            scenario_snapshot,
            scenario,
            ur_bundle,
            sr_keepout=sr_keepout,
            approach_distance_m=approach_distance,
        )
        evaluations.append(
            evaluate_task_sequence(
                spec,
                scenario_snapshot,
                scenario,
                ur_bundle,
                steps,
                options=options,
            )
        )

    evaluation_tuple = tuple(evaluations)
    summaries = tuple(
        _coordination_summary(
            mode,
            selected_corners,
            evaluation_tuple,
            sr_workspace_feasible=sr_workspace.feasible,
        )
        for mode in modes
    )
    return BaseCandidateEvaluation(
        state=state,
        ur_base=ur_base,
        sr_base=sr_base,
        placement=placement,
        sr_workspace=sr_workspace,
        sr_keepout=sr_keepout,
        scenarios=evaluation_tuple,
        coordination_summaries=summaries,
    )


def _coordination_summary(
    mode: CoordinationMode,
    selected_corners: tuple[str, ...],
    evaluations: tuple[ScenarioEvaluation, ...],
    *,
    sr_workspace_feasible: bool,
) -> CoordinationSummary:
    grouped = tuple(
        result
        for result in evaluations
        if result.scenario.coordination_mode is mode
    )
    by_corner = {result.scenario.corner: result for result in grouped}
    if len(by_corner) != len(grouped):
        raise ValueError(f"duplicate {mode.value} scenario corner evaluation")
    missing = tuple(corner for corner in selected_corners if corner not in by_corner)
    unexpected = tuple(corner for corner in by_corner if corner not in selected_corners)
    if missing or unexpected:
        raise ValueError(
            f"{mode.value} scenario grouping mismatch: "
            f"missing={missing}, unexpected={unexpected}"
        )

    ordered = tuple(by_corner[corner] for corner in selected_corners)
    feasible_corners = tuple(
        result.scenario.corner for result in ordered if result.hard_feasible
    )
    # ScenarioEvaluation excludes SOFT/SKIP steps from hard_feasible, so step
    # 7 remains diagnostic here even when its dimensional or IK checks fail.
    all_hard_feasible = len(feasible_corners) == len(selected_corners)
    return CoordinationSummary(
        mode=mode,
        selected_corners=selected_corners,
        hard_feasible_corners=feasible_corners,
        hard_feasible=all_hard_feasible,
        scenarios=ordered,
        scenario_metrics=tuple(result.metrics for result in ordered),
        cell_feasible=all_hard_feasible and sr_workspace_feasible,
    )


def _normalize_coordination_modes(
    values: Iterable[CoordinationMode | str],
) -> tuple[CoordinationMode, ...]:
    result: list[CoordinationMode] = []
    for raw in values:
        mode = raw if isinstance(raw, CoordinationMode) else CoordinationMode(str(raw))
        if mode not in result:
            result.append(mode)
    if not result:
        raise ValueError("at least one coordination mode is required")
    return tuple(result)


def _positive_finite(value: float, name: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return result


def _nonnegative_finite(value: float, name: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return result
