"""Precomputed-cache backend and scene presenter for the local MeshCat UI."""

from __future__ import annotations

from dataclasses import asdict
import math
from typing import Mapping, Protocol

import numpy as np

from .kinematics import frame_pose
from .meshcat_playback import UR20MeshcatPlayback
from .models import BasePose, BoxPrimitive, SceneSpec, SceneState
from .playback_controller import PlaybackSelection, PrecomputedPlaybackController
from .precompute import CachedCase, CachedPose, CachedPoseSample, PrecomputedCache
from .robots import (
    RobotBundle,
    configure_sr_j3_stroke,
    floating_configuration,
    load_robot_bundle,
)
from .scene import materialize_scene
from .viewer import CellViewer
from .workflow import ProcessScenario, build_ur20_task_sequence


class SelectionPresenter(Protocol):
    """Optional 3-D side effects around a pure playback selection."""

    def prepare_case(self, case: CachedCase) -> None: ...

    def present(self, case: CachedCase, selection: PlaybackSelection) -> str: ...


class CachedPlaybackBackend:
    """JSON-facing exact cache catalog and selection endpoint."""

    def __init__(
        self,
        cache: PrecomputedCache,
        controller: PrecomputedPlaybackController,
        *,
        presenter: SelectionPresenter | None = None,
    ) -> None:
        self.cache = cache
        self.controller = controller
        self.presenter = presenter
        self._active_case_id: str | None = None

    def catalog(self) -> Mapping[str, object]:
        cases = []
        for case in self.cache.cases:
            scenario = _matching_scenario(case)
            cell_feasible = _case_cell_feasible(case)
            steps = []
            if scenario is not None:
                for step in scenario.steps:
                    steps.append(
                        {
                            "index": step.index,
                            "name": step.name,
                            "criticality": step.criticality,
                            "status": step.status,
                            "checks": [
                                {
                                    "name": check.name,
                                    "status": check.status,
                                    "include_in_metric_summary": (
                                        check.include_in_metric_summary
                                    ),
                                    "samples": [
                                        {
                                            "fraction": sample.fraction,
                                            "ik_status": sample.ik_status,
                                        }
                                        for sample in check.samples
                                    ],
                                }
                                for check in step.checks
                            ],
                        }
                    )
            cases.append(
                {
                    "id": case.key.stable_id,
                    "status": case.status,
                    "installation_valid": case.installation_valid,
                    "scenario_hard_feasible": (
                        None if scenario is None else scenario.hard_feasible
                    ),
                    "sr_workspace_feasible": (
                        None
                        if case.sr_workspace is None
                        else case.sr_workspace.feasible
                    ),
                    "cell_feasible": cell_feasible,
                    "key": _case_key_payload(case),
                    "steps": steps,
                }
            )
        preferred = preferred_cached_case(self.cache)
        return {
            "schema_version": 1,
            "cache_schema_version": self.cache.schema_version,
            "case_count": len(cases),
            "cell_feasible_case_count": sum(
                bool(case["cell_feasible"]) for case in cases
            ),
            "default_case_id": (
                None if preferred is None else preferred.key.stable_id
            ),
            "cases": cases,
        }

    def select(self, request: Mapping[str, object]) -> Mapping[str, object]:
        case_id = request.get("case_id")
        if not isinstance(case_id, str) or not case_id:
            raise ValueError("case_id must be a complete stable-id string")
        case = self.controller.case_for(case_id)
        if self.presenter is not None and case_id != self._active_case_id:
            self.presenter.prepare_case(case)
            self._active_case_id = case_id

        selection = self.controller.select(
            case_id,
            step_index=_optional_index(request.get("step_index"), "step_index"),
            check_index=_optional_index(request.get("check_index"), "check_index"),
            sample_index=_optional_index(request.get("sample_index"), "sample_index"),
            ellipsoid_scale=_optional_positive_float(
                request.get("ellipsoid_scale"),
                "ellipsoid_scale",
            ),
            show_ellipsoid=_boolean(request.get("show_ellipsoid", True), "show_ellipsoid"),
            ellipsoid_wireframe=_boolean(
                request.get("ellipsoid_wireframe", False),
                "ellipsoid_wireframe",
            ),
        )
        pose_kind = "cached_feasible"
        if self.presenter is not None:
            pose_kind = self.presenter.present(case, selection)
        payload = _selection_payload(case, selection, pose_kind)
        payload["status_text"] = _status_text(payload)
        return payload


def preferred_cached_case(cache: PrecomputedCache) -> CachedCase | None:
    """Choose a deterministic UI opening case, preferring full-cell success."""

    if not cache.cases:
        return None
    for case in cache.cases:
        if _case_cell_feasible(case):
            return case
    for case in cache.cases:
        scenario = _matching_scenario(case)
        if (
            case.status == "evaluated"
            and case.installation_valid
            and scenario is not None
            and scenario.hard_feasible
        ):
            return case
    return cache.cases[0]


class MeshcatScenePresenter:
    """Update scene primitives, robot meshes, targets, and process objects."""

    def __init__(
        self,
        spec: SceneSpec,
        cache: PrecomputedCache,
        viewer: CellViewer,
        playback: UR20MeshcatPlayback,
    ) -> None:
        self.spec = spec
        self.cache = cache
        self.viewer = viewer
        self.playback = playback
        self.ur_loaded = viewer.loaded_robot("ur20")
        self.sr_loaded = viewer.loaded_robot("sr12ia")
        self._snapshot = None
        self._workflow = ()

    def prepare_case(self, case: CachedCase) -> None:
        state = _scene_state(case, tote_present=self.cache.grid.tote_present)
        snapshot = materialize_scene(
            self.spec,
            state,
            ur_base=case.key.ur_base,
            sr_base=case.key.sr_base,
        )
        self.viewer.update_scene(
            self.spec,
            snapshot,
            suppress_process_objects=True,
        )
        self._snapshot = snapshot

        sr_arm = list(case.sr_nominal_q)
        if len(sr_arm) < 3:
            raise ValueError("cached SR nominal configuration has no J3 value")
        if sr_arm[2] > case.key.sr_j3_stroke_m + 1.0e-12:
            raise ValueError(
                "cached SR nominal J3 exceeds the evaluated stroke option"
            )
        display_stroke = case.key.sr_j3_stroke_m
        if self.sr_loaded.visual_only:
            if self.sr_loaded.native_j3_stroke_m is None:
                raise ValueError("display-only SR model has no native J3 stroke")
            display_stroke = min(
                display_stroke,
                self.sr_loaded.native_j3_stroke_m,
            )
            if sr_arm[2] > display_stroke + 1.0e-12:
                raise ValueError(
                    "display-only SR model cannot represent cached nominal J3 "
                    f"{sr_arm[2]:.3f} m; native limit is "
                    f"{display_stroke:.3f} m"
                )
        else:
            configure_sr_j3_stroke(
                self.sr_loaded.bundle,
                case.key.sr_j3_stroke_m,
            )
        self.sr_loaded.visualizer.display(
            floating_configuration(self.sr_loaded.bundle, case.key.sr_base, sr_arm)
        )
        self.viewer.render_sr_keepout(
            _cached_keepout(case)
            if case.key.coordination_mode.value == "simultaneous"
            else None
        )
        self._workflow = ()
        if case.scenarios:
            scenario = ProcessScenario(
                state,
                case.key.corner,
                case.key.coordination_mode,
            )
            self._workflow = build_ur20_task_sequence(
                self.spec,
                snapshot,
                scenario,
                self.ur_loaded.bundle,
                sr_keepout=_cached_keepout(case),
                approach_distance_m=self.cache.settings.approach_distance_m,
            )

    def present(self, case: CachedCase, selection: PlaybackSelection) -> str:
        resolved = _selected_sample(case, selection)
        pose_kind = "cached_feasible"
        target = None
        if resolved is not None:
            _, _, sample = resolved
            target = _pose_matrix(sample.target_pose)
            if sample.ik_status != "success":
                self.playback.display_configuration(
                    np.asarray(sample.q, dtype=float)
                )
                self.playback.hide_ellipsoid()
                pose_kind = "diagnostic_best_ik"
        else:
            inherited = _previous_successful_sample(case, selection.step_index)
            if inherited is not None:
                self.playback.display_configuration(
                    np.asarray(inherited[2].q, dtype=float)
                )
                pose_kind = "inherited_previous_step"
            else:
                self.playback.display_configuration(
                    floating_configuration(
                        self.ur_loaded.bundle,
                        case.key.ur_base,
                        self.spec.robots["ur20"].nominal_q,
                    )
                )
                pose_kind = "candidate_nominal"
            self.playback.hide_ellipsoid()

        self.viewer.render_target_pose(
            target,
            successful=(
                selection.ik_status == "success"
                and selection.check_status == "success"
            ),
        )
        object_source = resolved
        if object_source is None:
            object_source = _previous_successful_sample(case, selection.step_index)
        self.viewer.render_process_objects(self._process_objects(object_source))
        return pose_kind

    def _process_objects(
        self,
        resolved: tuple[object, object, CachedPoseSample] | None,
    ) -> tuple[tuple[str, str, tuple[float, ...], np.ndarray], ...]:
        if resolved is None or not self._workflow:
            return ()
        cached_step, cached_check, sample = resolved
        workflow_step = next(
            (item for item in self._workflow if item.index == cached_step.index),
            None,
        )
        if workflow_step is None:
            return ()
        workflow_check = next(
            (item for item in workflow_step.pose_checks if item.name == cached_check.name),
            None,
        )
        if workflow_check is None:
            return ()
        world_T_tcp = _pose_matrix(sample.solved_pose)
        result = []
        for state in workflow_check.object_states:
            if state.attached_to_tcp or state.follows_tcp:
                if state.tcp_T_object is None:
                    continue
                world_T_object = world_T_tcp @ state.tcp_T_object.matrix
            else:
                world_T_object = state.world_T_object.matrix
            role = "tote" if "tote" in state.name.lower() else "box"
            result.append((state.name, role, tuple(state.size_m), world_T_object))
        return tuple(result)


def make_meshcat_backend(
    spec: SceneSpec,
    cache: PrecomputedCache,
    viewer: CellViewer,
    *,
    ellipsoid_scale: float = 0.15,
) -> CachedPlaybackBackend:
    """Construct the controller and presenter around loaded viewer robots."""

    ur_loaded = viewer.loaded_robot("ur20")
    sr_loaded = viewer.loaded_robot("sr12ia")
    if sr_loaded.visual_only:
        evaluation_sr = load_robot_bundle(
            "sr12ia_evaluation_contract",
            spec.robots["sr12ia"].urdf_path,
            floating_base=True,
            load_visual=False,
        )
        _validate_sr_display_contract(evaluation_sr, sr_loaded.bundle)
    playback = UR20MeshcatPlayback(
        viewer.viewer,
        ur_loaded.visualizer,
        ur_loaded.bundle,
        ellipsoid_scale=ellipsoid_scale,
        on_configuration_displayed=viewer.update_ur20_suction_proxy,
    )
    controller = PrecomputedPlaybackController(cache, playback)
    presenter = MeshcatScenePresenter(spec, cache, viewer, playback)
    return CachedPlaybackBackend(cache, controller, presenter=presenter)


def _validate_sr_display_contract(
    evaluation: RobotBundle,
    visual: RobotBundle,
) -> None:
    """Reject a display-only SR URDF that cannot replay cached arm values."""

    expected = _arm_joint_signature(evaluation)
    actual = _arm_joint_signature(visual)
    if actual != expected:
        raise ValueError(
            "display-only SR joint contract differs from the evaluation model: "
            f"expected {expected}, got {actual}"
        )
    for bundle in (evaluation, visual):
        if not bundle.model.existFrame("tool0"):
            raise ValueError(f"{bundle.name} display contract has no tool0 frame")

    arm_dof = evaluation.model.nq - evaluation.arm_configuration_offset
    if arm_dof != 4:
        raise ValueError("SR display contract requires exactly four arm positions")
    base = BasePose(0.0, 0.0, 1.0, 0.0)
    probes = [np.array((0.0, 0.0, 0.10, 0.0), dtype=float)]
    for index, delta in enumerate((0.07, -0.06, 0.04, 0.08)):
        probe = probes[0].copy()
        probe[index] += delta
        probes.append(probe)
    for probe in probes:
        evaluation_pose = frame_pose(
            evaluation,
            floating_configuration(evaluation, base, probe),
            "tool0",
        )
        visual_pose = frame_pose(
            visual,
            floating_configuration(visual, base, probe),
            "tool0",
        )
        if not np.allclose(evaluation_pose, visual_pose, atol=1.0e-9):
            raise ValueError(
                "display-only SR tool0 kinematics differ from the evaluation model"
            )


def _arm_joint_signature(bundle: RobotBundle) -> tuple[tuple[object, ...], ...]:
    entries = []
    for joint_id in range(1, bundle.model.njoints):
        joint = bundle.model.joints[joint_id]
        if joint.idx_q < bundle.arm_configuration_offset:
            continue
        entries.append(
            (
                str(bundle.model.names[joint_id]),
                int(joint.idx_q - bundle.arm_configuration_offset),
                int(joint.nq),
                int(joint.nv),
                str(joint.shortname()),
            )
        )
    return tuple(entries)


def _matching_scenario(case: CachedCase):
    matches = tuple(
        scenario
        for scenario in case.scenarios
        if scenario.corner == case.key.corner
        and scenario.coordination_mode == case.key.coordination_mode.value
    )
    if not matches:
        return None
    if len(matches) != 1:
        raise ValueError("cached case has duplicate matching scenarios")
    return matches[0]


def _case_cell_feasible(case: CachedCase) -> bool:
    """Return the cached combined UR-hard/SR-workspace decision for this key."""

    return any(
        summary.mode == case.key.coordination_mode.value
        and summary.cell_feasible
        for summary in case.coordination_summaries
    )


def _selected_sample(case: CachedCase, selection: PlaybackSelection):
    scenario = _matching_scenario(case)
    if (
        scenario is None
        or selection.step_index is None
        or selection.check_index is None
        or selection.sample_index is None
    ):
        return None
    step = next((item for item in scenario.steps if item.index == selection.step_index), None)
    if step is None:
        return None
    try:
        check = step.checks[selection.check_index]
        sample = check.samples[selection.sample_index]
    except IndexError:
        return None
    return step, check, sample


def _previous_successful_sample(case: CachedCase, before_step: int | None):
    scenario = _matching_scenario(case)
    if scenario is None:
        return None
    candidates = []
    for step in scenario.steps:
        if before_step is not None and step.index >= before_step:
            continue
        for check in step.checks:
            for sample in check.samples:
                if sample.ik_status == "success":
                    candidates.append((step, check, sample))
    return candidates[-1] if candidates else None


def _scene_state(case: CachedCase, *, tote_present: bool) -> SceneState:
    return SceneState(
        lift_height_m=case.key.lift_height_m,
        sku=case.key.sku,
        corner_samples=(case.key.corner,),
        tote_present=tote_present,
        tote_long_axis_offset_m=case.key.tote_offset_m,
        sr_j3_stroke_m=case.key.sr_j3_stroke_m,
        clearance_m=case.key.clearance_m,
    )


def _cached_keepout(case: CachedCase) -> BoxPrimitive | None:
    box = case.sr_keepout
    if box is None:
        return None
    return BoxPrimitive(
        name=box.name,
        role=box.role,
        center_m=box.center_m,
        size_m=box.size_m,
        yaw_deg=box.yaw_deg,
    )


def _pose_matrix(pose: CachedPose) -> np.ndarray:
    result = np.eye(4)
    result[:3, :3] = np.asarray(pose.rotation, dtype=float)
    result[:3, 3] = np.asarray(pose.translation_m, dtype=float)
    return result


def _case_key_payload(case: CachedCase) -> dict[str, object]:
    key = case.key
    return {
        "ur_base": asdict(key.ur_base),
        "sr_base": asdict(key.sr_base),
        "lift_height_m": key.lift_height_m,
        "tote_offset_m": key.tote_offset_m,
        "corner": key.corner,
        "coordination_mode": key.coordination_mode.value,
        "sku": key.sku,
        "sr_j3_stroke_m": key.sr_j3_stroke_m,
        "clearance_m": key.clearance_m,
    }


def _selection_payload(
    case: CachedCase,
    selection: PlaybackSelection,
    pose_kind: str,
) -> dict[str, object]:
    summary = case.coordination_summaries[0] if case.coordination_summaries else None
    scenario = _matching_scenario(case)
    cached_step = (
        None
        if scenario is None or selection.step_index is None
        else next(
            (step for step in scenario.steps if step.index == selection.step_index),
            None,
        )
    )
    return {
        "case_id": selection.case_id,
        "case_status": selection.case_status,
        "installation_valid": case.installation_valid,
        "scenario_hard_feasible": selection.scenario_hard_feasible,
        "cell_feasible": None if summary is None else summary.cell_feasible,
        "step_index": selection.step_index,
        "step_name": selection.step_name,
        "step_status": selection.step_status,
        "criticality": selection.criticality,
        "waste_dimension_fits": (
            None if cached_step is None else cached_step.waste_dimension_fits
        ),
        "check_index": selection.check_index,
        "check_name": selection.check_name,
        "check_status": selection.check_status,
        "sample_index": selection.sample_index,
        "sample_fraction": selection.sample_fraction,
        "ik_status": selection.ik_status,
        "pose_kind": pose_kind,
        "ellipsoid_visible": selection.ellipsoid_visible,
        "metric": None if selection.metric is None else asdict(selection.metric),
        "collision": None if selection.collision is None else asdict(selection.collision),
        "placement_issues": [asdict(issue) for issue in case.placement_issues],
        "sr_workspace": (
            None if case.sr_workspace is None else asdict(case.sr_workspace)
        ),
    }


def _status_text(payload: Mapping[str, object]) -> str:
    lines = [
        f"Case: {payload['case_status']}",
        f"Scenario hard feasible: {payload['scenario_hard_feasible']}",
        f"Cell feasible: {payload['cell_feasible']}",
        f"Step {payload['step_index']}: {payload['step_name']} [{payload['step_status']}]",
        f"Check: {payload['check_name']} [{payload['check_status']}]",
        f"IK: {payload['ik_status']} / pose: {payload['pose_kind']}",
        f"Manipulability ellipsoid: {'visible' if payload['ellipsoid_visible'] else 'hidden'}",
    ]
    metric = payload.get("metric")
    if isinstance(metric, Mapping):
        sigma = metric.get("translational_axis_lengths")
        lines.append(f"Translational sigma: {sigma}")
        lines.append(
            "Volume / min sigma / condition: "
            f"{metric.get('translational_volume')} / "
            f"{metric.get('translational_min_singular_value')} / "
            f"{metric.get('translational_condition_number')}"
        )
    collision = payload.get("collision")
    if isinstance(collision, Mapping) and collision.get("in_collision"):
        lines.append(f"Collision ({collision.get('source')}): {collision.get('contacts')}")
    workspace = payload.get("sr_workspace")
    if isinstance(workspace, Mapping):
        lines.append(
            "SR workspace footprint / planar / vertical: "
            f"{workspace.get('footprint_ok')} / "
            f"{workspace.get('planar_reach_ok')} / "
            f"{workspace.get('vertical_ok')}"
        )
    if payload.get("waste_dimension_fits") is not None:
        lines.append(f"Waste dimension fits: {payload.get('waste_dimension_fits')}")
    return "\n".join(lines)


def _optional_index(value: object, name: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer or null")
    return value


def _optional_positive_float(value: object, name: str) -> float | None:
    if value is None:
        return None
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return result


def _boolean(value: object, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a boolean")
    return value
