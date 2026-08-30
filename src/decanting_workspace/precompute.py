"""Finite case enumeration and replay-oriented candidate-result caches.

There is no meaningful finite set of "all" continuous robot placements.  A
precomputation therefore starts from explicit, non-empty values for every
continuous axis.  The Cartesian product is deterministic and its keys retain
the physical values (not fragile loop indices), so a later optimizer or UI can
look up exactly the case that was evaluated.

The cache is intentionally lossier than the live Pinocchio objects but keeps
everything needed to replay a result: target and solved TCP poses, full and
arm robot configurations, translational and normalized manipulability-
ellipsoid axes/directions, collision reports, and the status hierarchy.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import gzip
import hashlib
import json
import math
from pathlib import Path
from typing import Callable, Iterable, Iterator, Mapping, Sequence

import numpy as np

from .candidates import BaseCandidateEvaluation, evaluate_base_candidate
from .collision_world import CollisionReport
from .evaluation import EvaluationOptions, MetricSummary
from .kinematics import ManipulabilityMetrics, frame_pose
from .models import BasePose, BoxPrimitive, SceneSpec, SceneState
from .robots import RobotBundle
from .scene import CORNER_SIGNS, tote_long_axis_offset_range_m
from .workflow import CoordinationMode, WORKFLOW_SCHEMA_VERSION


CACHE_SCHEMA_VERSION = 2


@dataclass(frozen=True)
class BasePoseGrid:
    """Explicit Cartesian axes for one robot base pose grid."""

    x_m: tuple[float, ...]
    y_m: tuple[float, ...]
    z_m: tuple[float, ...]
    yaw_deg: tuple[float, ...]

    def __post_init__(self) -> None:
        for name in ("x_m", "y_m", "z_m", "yaw_deg"):
            values = _finite_unique_axis(getattr(self, name), name)
            if name == "z_m" and any(value <= 0.0 for value in values):
                raise ValueError("base z_m grid values must be positive")
            object.__setattr__(self, name, values)

    @classmethod
    def single(cls, pose: BasePose) -> "BasePoseGrid":
        """Create a one-pose grid without weakening explicit-axis semantics."""

        return cls(
            x_m=(pose.x_m,),
            y_m=(pose.y_m,),
            z_m=(pose.z_m,),
            yaw_deg=(pose.yaw_deg,),
        )

    @property
    def pose_count(self) -> int:
        return len(self.x_m) * len(self.y_m) * len(self.z_m) * len(self.yaw_deg)

    def poses(self) -> Iterator[BasePose]:
        for x_m in self.x_m:
            for y_m in self.y_m:
                for z_m in self.z_m:
                    for yaw_deg in self.yaw_deg:
                        yield BasePose(x_m, y_m, z_m, yaw_deg)


@dataclass(frozen=True)
class CaseGrid:
    """Finite axes that define every candidate-evaluation case.

    SKU, J3 option, and clearance are included because changing any of them
    changes geometry or feasibility.  ``None`` clearance preserves the scene
    configuration's default.  Tote presence is fixed metadata rather than an
    axis because the current decanting sequence requires the representative
    tote.
    """

    ur_bases: BasePoseGrid
    sr_bases: BasePoseGrid
    lift_heights_m: tuple[float, ...]
    tote_offsets_m: tuple[float, ...]
    corners: tuple[str, ...]
    coordination_modes: tuple[CoordinationMode | str, ...]
    skus: tuple[str, ...] = ("123591",)
    sr_j3_strokes_m: tuple[float, ...] = (0.30,)
    clearances_m: tuple[float | None, ...] = (None,)
    tote_present: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "lift_heights_m",
            _finite_unique_axis(self.lift_heights_m, "lift_heights_m"),
        )
        object.__setattr__(
            self,
            "tote_offsets_m",
            _finite_unique_axis(self.tote_offsets_m, "tote_offsets_m"),
        )
        object.__setattr__(
            self,
            "sr_j3_strokes_m",
            _positive_unique_axis(self.sr_j3_strokes_m, "sr_j3_strokes_m"),
        )
        clearances = tuple(
            None if value is None else _nonnegative_finite(value, "clearances_m")
            for value in _nonempty_tuple(self.clearances_m, "clearances_m")
        )
        if len(set(clearances)) != len(clearances):
            raise ValueError("clearances_m must not contain duplicate values")
        object.__setattr__(self, "clearances_m", clearances)

        corners = tuple(str(value) for value in _nonempty_tuple(self.corners, "corners"))
        if len(set(corners)) != len(corners):
            raise ValueError("corners must not contain duplicate values")
        object.__setattr__(self, "corners", corners)

        modes: list[CoordinationMode] = []
        for raw in _nonempty_tuple(self.coordination_modes, "coordination_modes"):
            mode = raw if isinstance(raw, CoordinationMode) else CoordinationMode(str(raw))
            if mode in modes:
                raise ValueError("coordination_modes must not contain duplicates")
            modes.append(mode)
        object.__setattr__(self, "coordination_modes", tuple(modes))

        skus = tuple(str(value) for value in _nonempty_tuple(self.skus, "skus"))
        if len(set(skus)) != len(skus):
            raise ValueError("skus must not contain duplicate values")
        object.__setattr__(self, "skus", skus)

    @property
    def case_count(self) -> int:
        axes = (
            self.ur_bases.pose_count,
            self.sr_bases.pose_count,
            len(self.lift_heights_m),
            len(self.tote_offsets_m),
            len(self.corners),
            len(self.coordination_modes),
            len(self.skus),
            len(self.sr_j3_strokes_m),
            len(self.clearances_m),
        )
        result = 1
        for length in axes:
            result *= length
        return result


@dataclass(frozen=True)
class CaseKey:
    """Physical values uniquely identifying one cached evaluation."""

    ur_base: BasePose
    sr_base: BasePose
    lift_height_m: float
    tote_offset_m: float
    corner: str
    coordination_mode: CoordinationMode
    sku: str
    sr_j3_stroke_m: float
    clearance_m: float | None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "coordination_mode",
            (
                self.coordination_mode
                if isinstance(self.coordination_mode, CoordinationMode)
                else CoordinationMode(str(self.coordination_mode))
            ),
        )
        object.__setattr__(
            self,
            "lift_height_m",
            _finite(self.lift_height_m, "lift_height_m"),
        )
        object.__setattr__(
            self,
            "tote_offset_m",
            _finite(self.tote_offset_m, "tote_offset_m"),
        )
        object.__setattr__(
            self,
            "sr_j3_stroke_m",
            _positive_finite(self.sr_j3_stroke_m, "sr_j3_stroke_m"),
        )
        if self.clearance_m is not None:
            object.__setattr__(
                self,
                "clearance_m",
                _nonnegative_finite(self.clearance_m, "clearance_m"),
            )
        if self.corner not in CORNER_SIGNS:
            raise ValueError(f"unknown pallet corner: {self.corner}")
        if not self.sku:
            raise ValueError("sku must not be empty")

    @property
    def stable_id(self) -> str:
        encoded = json.dumps(
            _json_compatible(asdict(self)),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class CachedPose:
    """JSON-safe rigid transform with World_T_frame semantics."""

    translation_m: tuple[float, float, float]
    rotation: tuple[tuple[float, float, float], ...]


@dataclass(frozen=True)
class CachedCollision:
    in_collision: bool
    contacts: tuple[tuple[str, str], ...]
    minimum_distance_m: float | None
    nearest_pair: tuple[str, str] | None


@dataclass(frozen=True)
class CachedManipulabilityEllipsoid:
    """SVD axes and column directions for UI ellipsoid rendering."""

    translational_axis_lengths: tuple[float, float, float]
    translational_directions_world: tuple[tuple[float, float, float], ...]
    normalized_axis_lengths: tuple[float, ...]
    normalized_directions: tuple[tuple[float, ...], ...]
    translational_volume: float
    translational_min_singular_value: float
    translational_condition_number: float | None
    normalized_volume: float
    normalized_min_singular_value: float
    normalized_condition_number: float | None
    characteristic_length_m: float


@dataclass(frozen=True)
class CachedPoseSample:
    fraction: float
    target_pose: CachedPose
    solved_pose: CachedPose
    ik_status: str
    q: tuple[float, ...]
    arm_q: tuple[float, ...]
    iterations: int
    seed_index: int
    attempted_seeds: int
    position_error_m: float
    orientation_error_rad: float
    log_residual_norm: float
    rejected_by_acceptor: int
    collision: CachedCollision | None
    manipulability: CachedManipulabilityEllipsoid | None


@dataclass(frozen=True)
class CachedLinearCheck:
    name: str
    criticality: str
    status: str
    failed_sample_index: int | None
    joint_collision: CachedCollision | None
    samples: tuple[CachedPoseSample, ...]
    include_in_metric_summary: bool = True


@dataclass(frozen=True)
class CachedStep:
    index: int
    name: str
    criticality: str
    status: str
    waste_dimension_fits: bool | None
    checks: tuple[CachedLinearCheck, ...]


@dataclass(frozen=True)
class CachedMetricSummary:
    sample_count: int
    translational_volume_min: float
    translational_volume_mean: float
    translational_sigma_min: float
    normalized_volume_min: float
    normalized_volume_mean: float
    normalized_sigma_min: float
    normalized_condition_max: float | None


@dataclass(frozen=True)
class CachedScenario:
    corner: str
    coordination_mode: str
    hard_feasible: bool
    soft_step_successes: int
    soft_step_count: int
    metrics: CachedMetricSummary | None
    steps: tuple[CachedStep, ...]


@dataclass(frozen=True)
class CachedScaraTarget:
    name: str
    world_position_m: tuple[float, float, float]
    required_j3_m: float
    planar_ik_solutions_rad: tuple[tuple[float, float], ...]
    footprint_ok: bool
    planar_reach_ok: bool
    vertical_ok: bool
    feasible: bool


@dataclass(frozen=True)
class CachedScaraWorkspace:
    feasible: bool
    footprint_ok: bool
    planar_reach_ok: bool
    vertical_ok: bool
    tool0_z_range_m: tuple[float, float]
    targets: tuple[CachedScaraTarget, ...]


@dataclass(frozen=True)
class CachedCoordinationSummary:
    mode: str
    selected_corners: tuple[str, ...]
    hard_feasible_corners: tuple[str, ...]
    hard_feasible: bool
    cell_feasible: bool


@dataclass(frozen=True)
class CachedPlacementIssue:
    code: str
    objects: tuple[str, ...]
    message: str


@dataclass(frozen=True)
class CachedBox:
    name: str
    role: str
    center_m: tuple[float, float, float]
    size_m: tuple[float, float, float]
    yaw_deg: float


@dataclass(frozen=True)
class CachedCase:
    key: CaseKey
    status: str
    sr_nominal_q: tuple[float, ...]
    installation_valid: bool
    placement_issues: tuple[CachedPlacementIssue, ...]
    sr_workspace: CachedScaraWorkspace | None
    sr_keepout: CachedBox | None
    scenarios: tuple[CachedScenario, ...]
    coordination_summaries: tuple[CachedCoordinationSummary, ...]


@dataclass(frozen=True)
class CachedEvaluationSettings:
    cartesian_translation_step_m: float
    cartesian_rotation_step_rad: float
    joint_interpolation_samples: int
    characteristic_length_m: float
    ik_position_tolerance_m: float
    ik_orientation_tolerance_rad: float
    ik_max_iterations: int
    ik_damping: float
    ik_max_backtracking_steps: int
    extra_arm_seeds: tuple[tuple[float, ...], ...]
    approach_distance_m: float
    keepout_margin_m: float


@dataclass(frozen=True)
class PrecomputedCache:
    schema_version: int
    spec_fingerprint: str
    grid: CaseGrid
    settings: CachedEvaluationSettings
    cases: tuple[CachedCase, ...]

    def __post_init__(self) -> None:
        if self.schema_version != CACHE_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported precompute cache schema: {self.schema_version}"
            )
        if len(self.cases) != self.grid.case_count:
            raise ValueError(
                "cache case count does not match the declared finite grid"
            )
        ids = tuple(case.key.stable_id for case in self.cases)
        if len(set(ids)) != len(ids):
            raise ValueError("cache contains duplicate case keys")

    def case_for(self, key: CaseKey) -> CachedCase:
        """Return an exact physical-key match or raise ``KeyError``."""

        stable_id = key.stable_id
        for case in self.cases:
            if case.key.stable_id == stable_id and case.key == key:
                return case
        raise KeyError(stable_id)


CandidateEvaluator = Callable[..., BaseCandidateEvaluation]


def iter_case_keys(spec: SceneSpec, grid: CaseGrid) -> Iterator[CaseKey]:
    """Yield the validated finite Cartesian product in deterministic order."""

    _validate_grid_against_scene(spec, grid)
    for ur_base in grid.ur_bases.poses():
        for sr_base in grid.sr_bases.poses():
            for sku in grid.skus:
                for lift_height in grid.lift_heights_m:
                    for tote_offset in grid.tote_offsets_m:
                        for corner in grid.corners:
                            for mode in grid.coordination_modes:
                                for stroke in grid.sr_j3_strokes_m:
                                    for clearance in grid.clearances_m:
                                        yield CaseKey(
                                            ur_base=ur_base,
                                            sr_base=sr_base,
                                            lift_height_m=lift_height,
                                            tote_offset_m=tote_offset,
                                            corner=corner,
                                            coordination_mode=mode,
                                            sku=sku,
                                            sr_j3_stroke_m=stroke,
                                            clearance_m=clearance,
                                        )


def precompute_cases(
    spec: SceneSpec,
    grid: CaseGrid,
    ur_bundle: RobotBundle,
    sr_bundle: RobotBundle,
    *,
    options: EvaluationOptions | None = None,
    approach_distance_m: float = 0.15,
    keepout_margin_m: float = 0.0,
    evaluator: CandidateEvaluator = evaluate_base_candidate,
) -> PrecomputedCache:
    """Evaluate and cache every key in an explicit finite case grid."""

    settings = options or EvaluationOptions()
    approach = _positive_finite(approach_distance_m, "approach_distance_m")
    keepout = _nonnegative_finite(keepout_margin_m, "keepout_margin_m")
    cases: list[CachedCase] = []
    for key in iter_case_keys(spec, grid):
        state = SceneState(
            lift_height_m=key.lift_height_m,
            sku=key.sku,
            corner_samples=(key.corner,),
            tote_present=grid.tote_present,
            tote_long_axis_offset_m=key.tote_offset_m,
            sr_j3_stroke_m=key.sr_j3_stroke_m,
            clearance_m=key.clearance_m,
        )
        evaluation = evaluator(
            spec,
            state,
            key.ur_base,
            key.sr_base,
            ur_bundle,
            sr_bundle,
            coordination_modes=(key.coordination_mode,),
            options=settings,
            approach_distance_m=approach,
            keepout_margin_m=keepout,
        )
        cases.append(_cache_candidate(spec, key, evaluation, ur_bundle))

    return PrecomputedCache(
        schema_version=CACHE_SCHEMA_VERSION,
        spec_fingerprint=scene_spec_fingerprint(spec),
        grid=grid,
        settings=_cache_settings(settings, approach, keepout),
        cases=tuple(cases),
    )


def save_precomputed_cache(cache: PrecomputedCache, path: str | Path) -> Path:
    """Write deterministic strict JSON, optionally gzip-compressed."""

    target = Path(path).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = _json_compatible(asdict(cache))
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    if target.suffix.lower() == ".gz":
        with gzip.open(target, "wt", encoding="utf-8", newline="\n") as stream:
            stream.write(encoded + "\n")
    else:
        target.write_text(encoded + "\n", encoding="utf-8")
    return target


def load_precomputed_cache(
    path: str | Path,
    *,
    spec: SceneSpec | None = None,
) -> PrecomputedCache:
    """Load a cache, optionally rejecting data from a different scene spec."""

    source = Path(path).resolve()
    if source.suffix.lower() == ".gz":
        with gzip.open(source, "rt", encoding="utf-8") as stream:
            text = stream.read()
    else:
        text = source.read_text(encoding="utf-8")
    raw = json.loads(
        text,
        parse_constant=lambda value: (_raise_invalid_json_constant(value)),
    )
    if not isinstance(raw, Mapping):
        raise ValueError("precompute cache root must be an object")
    cache = _cache_from_mapping(raw)
    if spec is not None:
        expected = scene_spec_fingerprint(spec)
        if cache.spec_fingerprint != expected:
            raise ValueError("precompute cache scene fingerprint mismatch")
    return cache


def merge_precomputed_caches(
    caches: Sequence[PrecomputedCache],
    *,
    varying_axis: str,
) -> PrecomputedCache:
    """Merge complete, disjoint shards split along exactly one grid axis.

    This intentionally refuses arbitrary sparse unions.  Each input cache must
    have identical scene/model fingerprints, evaluation settings, base grids,
    and every case-grid axis other than ``varying_axis``.
    """

    shards = tuple(caches)
    if not shards:
        raise ValueError("at least one precomputed cache shard is required")
    allowed_axes = {
        "lift_heights_m",
        "tote_offsets_m",
        "corners",
        "coordination_modes",
        "skus",
        "sr_j3_strokes_m",
        "clearances_m",
    }
    if varying_axis not in allowed_axes:
        raise ValueError(f"unsupported cache shard axis: {varying_axis}")
    first = shards[0]
    merged_axis: list[object] = []
    merged_cases: list[CachedCase] = []
    comparable_fields = tuple(
        name
        for name in (
            "ur_bases",
            "sr_bases",
            "lift_heights_m",
            "tote_offsets_m",
            "corners",
            "coordination_modes",
            "skus",
            "sr_j3_strokes_m",
            "clearances_m",
            "tote_present",
        )
        if name != varying_axis
    )
    for shard in shards:
        if shard.spec_fingerprint != first.spec_fingerprint:
            raise ValueError("cache shard scene/model fingerprints differ")
        if shard.settings != first.settings:
            raise ValueError("cache shard evaluation settings differ")
        for field_name in comparable_fields:
            if getattr(shard.grid, field_name) != getattr(first.grid, field_name):
                raise ValueError(
                    f"cache shard grid differs outside {varying_axis}: {field_name}"
                )
        for value in getattr(shard.grid, varying_axis):
            if value in merged_axis:
                raise ValueError(
                    f"cache shard {varying_axis} values overlap: {value!r}"
                )
            merged_axis.append(value)
        merged_cases.extend(shard.cases)

    merged_grid = replace(first.grid, **{varying_axis: tuple(merged_axis)})
    return PrecomputedCache(
        schema_version=CACHE_SCHEMA_VERSION,
        spec_fingerprint=first.spec_fingerprint,
        grid=merged_grid,
        settings=first.settings,
        cases=tuple(merged_cases),
    )


def scene_spec_fingerprint(spec: SceneSpec) -> str:
    """Hash scene semantics, workflow contract, and both evaluation URDFs."""

    # Absolute checkout paths are deliberately excluded so an unchanged cache
    # remains portable.  Parsed semantics and file-content hashes still make
    # model/config changes invalidate it.
    payload = asdict(spec)
    payload.pop("source_path", None)
    for robot in payload["robots"].values():
        robot.pop("urdf_path", None)
    payload = _json_compatible(payload)
    robot_assets: dict[str, str] = {}
    for name, robot in sorted(spec.robots.items()):
        robot_assets[name] = hashlib.sha256(robot.urdf_path.read_bytes()).hexdigest()
    canonical = json.dumps(
        {
            "scene": payload,
            "urdf_sha256": robot_assets,
            "workflow_schema_version": WORKFLOW_SCHEMA_VERSION,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _cache_candidate(
    spec: SceneSpec,
    key: CaseKey,
    evaluation: BaseCandidateEvaluation,
    ur_bundle: RobotBundle,
) -> CachedCase:
    if evaluation.state.corner_samples != (key.corner,):
        raise ValueError("candidate result corner does not match its cache key")
    if evaluation.ur_base != key.ur_base or evaluation.sr_base != key.sr_base:
        raise ValueError("candidate result base pose does not match its cache key")

    scenarios = tuple(
        _cache_scenario(result, ur_bundle) for result in evaluation.scenarios
    )
    summaries = tuple(
        CachedCoordinationSummary(
            mode=summary.mode.value,
            selected_corners=tuple(summary.selected_corners),
            hard_feasible_corners=tuple(summary.hard_feasible_corners),
            hard_feasible=bool(summary.hard_feasible),
            cell_feasible=bool(summary.cell_feasible),
        )
        for summary in evaluation.coordination_summaries
    )
    if evaluation.installation_valid:
        status = "evaluated"
    else:
        status = "invalid_placement"
    return CachedCase(
        key=key,
        status=status,
        sr_nominal_q=tuple(float(value) for value in spec.robots["sr12ia"].nominal_q),
        installation_valid=evaluation.installation_valid,
        placement_issues=tuple(
            CachedPlacementIssue(issue.code, tuple(issue.objects), issue.message)
            for issue in evaluation.placement.issues
        ),
        sr_workspace=_cache_sr_workspace(evaluation.sr_workspace),
        sr_keepout=_cache_box(evaluation.sr_keepout),
        scenarios=scenarios,
        coordination_summaries=summaries,
    )


def _cache_scenario(result: object, ur_bundle: RobotBundle) -> CachedScenario:
    return CachedScenario(
        corner=result.scenario.corner,
        coordination_mode=result.scenario.coordination_mode.value,
        hard_feasible=bool(result.hard_feasible),
        soft_step_successes=int(result.soft_step_successes),
        soft_step_count=int(result.soft_step_count),
        metrics=_cache_metric_summary(result.metrics),
        steps=tuple(_cache_step(step, ur_bundle) for step in result.steps),
    )


def _cache_step(step: object, ur_bundle: RobotBundle) -> CachedStep:
    return CachedStep(
        index=int(step.index),
        name=str(step.name),
        criticality=step.criticality.value,
        status=str(step.status),
        waste_dimension_fits=step.waste_dimension_fits,
        checks=tuple(_cache_check(check, ur_bundle) for check in step.checks),
    )


def _cache_check(check: object, ur_bundle: RobotBundle) -> CachedLinearCheck:
    return CachedLinearCheck(
        name=str(check.name),
        criticality=check.criticality.value,
        status=str(check.status),
        failed_sample_index=check.failed_sample_index,
        joint_collision=_cache_collision(check.joint_collision),
        samples=tuple(_cache_sample(sample, ur_bundle) for sample in check.samples),
        include_in_metric_summary=bool(check.include_in_metric_summary),
    )


def _cache_sample(sample: object, ur_bundle: RobotBundle) -> CachedPoseSample:
    configuration = np.asarray(sample.ik.q, dtype=float).reshape(-1)
    solved_world_T_tcp = frame_pose(ur_bundle, configuration, "suction_tcp")
    return CachedPoseSample(
        fraction=float(sample.fraction),
        target_pose=_cache_pose(sample.target.matrix),
        solved_pose=_cache_pose(solved_world_T_tcp),
        ik_status=str(sample.ik.status),
        q=_finite_vector(configuration, "IK q"),
        arm_q=_finite_vector(sample.ik.arm_q, "IK arm_q"),
        iterations=int(sample.ik.iterations),
        seed_index=int(sample.ik.seed_index),
        attempted_seeds=int(sample.ik.attempted_seeds),
        position_error_m=float(sample.ik.position_error_m),
        orientation_error_rad=float(sample.ik.orientation_error_rad),
        log_residual_norm=float(sample.ik.log_residual_norm),
        rejected_by_acceptor=int(sample.ik.rejected_by_acceptor),
        collision=_cache_collision(sample.collision),
        manipulability=_cache_manipulability(sample.manipulability),
    )


def _cache_pose(matrix_like: Sequence[Sequence[float]] | np.ndarray) -> CachedPose:
    matrix = np.asarray(matrix_like, dtype=float).reshape(4, 4)
    if not np.all(np.isfinite(matrix)):
        raise ValueError("cached pose must be finite")
    return CachedPose(
        translation_m=tuple(float(value) for value in matrix[:3, 3]),
        rotation=tuple(
            tuple(float(value) for value in row) for row in matrix[:3, :3]
        ),
    )


def _cache_collision(report: CollisionReport | None) -> CachedCollision | None:
    if report is None:
        return None
    distance = float(report.minimum_distance_m)
    return CachedCollision(
        in_collision=bool(report.in_collision),
        contacts=tuple((contact.first, contact.second) for contact in report.contacts),
        minimum_distance_m=distance if math.isfinite(distance) else None,
        nearest_pair=None if report.nearest_pair is None else tuple(report.nearest_pair),
    )


def _cache_manipulability(
    metrics: ManipulabilityMetrics | None,
) -> CachedManipulabilityEllipsoid | None:
    if metrics is None:
        return None
    jacobian = np.asarray(metrics.jacobian, dtype=float)
    if jacobian.ndim != 2 or jacobian.shape[0] != 6 or not np.all(np.isfinite(jacobian)):
        raise ValueError("manipulability Jacobian must be a finite 6xN matrix")
    translational_u, translational_s, _ = np.linalg.svd(
        jacobian[:3, :], full_matrices=True
    )
    normalized_jacobian = np.vstack(
        (jacobian[:3, :], metrics.characteristic_length_m * jacobian[3:, :])
    )
    normalized_u, normalized_s, _ = np.linalg.svd(
        normalized_jacobian, full_matrices=True
    )
    return CachedManipulabilityEllipsoid(
        translational_axis_lengths=tuple(float(value) for value in translational_s),
        translational_directions_world=_direction_columns(translational_u),
        normalized_axis_lengths=tuple(float(value) for value in normalized_s),
        normalized_directions=_direction_columns(normalized_u),
        translational_volume=float(metrics.translational_volume),
        translational_min_singular_value=float(
            metrics.translational_min_singular_value
        ),
        translational_condition_number=_finite_or_none(
            metrics.translational_condition_number
        ),
        normalized_volume=float(metrics.normalized_volume),
        normalized_min_singular_value=float(metrics.normalized_min_singular_value),
        normalized_condition_number=_finite_or_none(
            metrics.normalized_condition_number
        ),
        characteristic_length_m=float(metrics.characteristic_length_m),
    )


def _direction_columns(matrix_like: np.ndarray) -> tuple[tuple[float, ...], ...]:
    matrix = np.array(matrix_like, dtype=float, copy=True)
    # An ellipsoid axis is unchanged by a sign flip.  Canonical signs make the
    # JSON stable across otherwise equivalent SVD implementations.
    for column in range(matrix.shape[1]):
        vector = matrix[:, column]
        nonzero = np.flatnonzero(np.abs(vector) > 1e-12)
        if nonzero.size and vector[int(nonzero[0])] < 0.0:
            matrix[:, column] *= -1.0
    return tuple(
        tuple(float(value) for value in matrix[:, column])
        for column in range(matrix.shape[1])
    )


def _cache_metric_summary(metrics: MetricSummary | None) -> CachedMetricSummary | None:
    if metrics is None:
        return None
    return CachedMetricSummary(
        sample_count=int(metrics.sample_count),
        translational_volume_min=float(metrics.translational_volume_min),
        translational_volume_mean=float(metrics.translational_volume_mean),
        translational_sigma_min=float(metrics.translational_sigma_min),
        normalized_volume_min=float(metrics.normalized_volume_min),
        normalized_volume_mean=float(metrics.normalized_volume_mean),
        normalized_sigma_min=float(metrics.normalized_sigma_min),
        normalized_condition_max=_finite_or_none(metrics.normalized_condition_max),
    )


def _cache_sr_workspace(report: object | None) -> CachedScaraWorkspace | None:
    if report is None:
        return None
    return CachedScaraWorkspace(
        feasible=bool(report.feasible),
        footprint_ok=bool(report.footprint_ok),
        planar_reach_ok=bool(report.planar_reach_ok),
        vertical_ok=bool(report.vertical_ok),
        tool0_z_range_m=tuple(float(value) for value in report.tool0_z_range_m),
        targets=tuple(
            CachedScaraTarget(
                name=target.name,
                world_position_m=tuple(target.world_position_m),
                required_j3_m=float(target.required_j3_m),
                planar_ik_solutions_rad=tuple(
                    tuple(float(value) for value in solution)
                    for solution in target.planar_ik_solutions_rad
                ),
                footprint_ok=bool(target.footprint_ok),
                planar_reach_ok=bool(target.planar_reach_ok),
                vertical_ok=bool(target.vertical_ok),
                feasible=bool(target.feasible),
            )
            for target in report.targets
        ),
    )


def _cache_box(box: BoxPrimitive | None) -> CachedBox | None:
    if box is None:
        return None
    return CachedBox(
        name=box.name,
        role=box.role,
        center_m=tuple(box.center_m),
        size_m=tuple(box.size_m),
        yaw_deg=float(box.yaw_deg),
    )


def _cache_settings(
    options: EvaluationOptions,
    approach_distance_m: float,
    keepout_margin_m: float,
) -> CachedEvaluationSettings:
    return CachedEvaluationSettings(
        cartesian_translation_step_m=options.cartesian_translation_step_m,
        cartesian_rotation_step_rad=options.cartesian_rotation_step_rad,
        joint_interpolation_samples=options.joint_interpolation_samples,
        characteristic_length_m=options.characteristic_length_m,
        ik_position_tolerance_m=options.ik.position_tolerance_m,
        ik_orientation_tolerance_rad=options.ik.orientation_tolerance_rad,
        ik_max_iterations=options.ik.max_iterations,
        ik_damping=options.ik.damping,
        ik_max_backtracking_steps=options.ik.max_backtracking_steps,
        extra_arm_seeds=tuple(tuple(float(value) for value in seed) for seed in options.extra_arm_seeds),
        approach_distance_m=approach_distance_m,
        keepout_margin_m=keepout_margin_m,
    )


def _validate_grid_against_scene(spec: SceneSpec, grid: CaseGrid) -> None:
    unknown_skus = tuple(sku for sku in grid.skus if sku not in spec.box_skus)
    if unknown_skus:
        raise ValueError(f"unknown box SKUs in case grid: {unknown_skus}")
    unknown_corners = tuple(corner for corner in grid.corners if corner not in CORNER_SIGNS)
    if unknown_corners:
        raise ValueError(f"unknown pallet corners in case grid: {unknown_corners}")
    lift_low, lift_high = spec.pallet.lift_range_m
    if any(not lift_low <= value <= lift_high for value in grid.lift_heights_m):
        raise ValueError(f"lift grid values must be in [{lift_low}, {lift_high}] m")
    tote_low, tote_high = tote_long_axis_offset_range_m(spec)
    if any(not tote_low <= value <= tote_high for value in grid.tote_offsets_m):
        raise ValueError(f"tote offset grid values must be in [{tote_low}, {tote_high}] m")
    workspace = spec.robots["sr12ia"].workspace
    if workspace is None:
        raise ValueError("SR-12iA workspace specification is missing")
    for stroke in grid.sr_j3_strokes_m:
        if not any(math.isclose(stroke, option, abs_tol=1e-9) for option in workspace.j3_stroke_options_m):
            raise ValueError(
                f"SR J3 grid values must be selected from {workspace.j3_stroke_options_m} m"
            )


def _cache_from_mapping(raw: Mapping[str, object]) -> PrecomputedCache:
    schema_version = int(raw.get("schema_version", -1))
    if schema_version != CACHE_SCHEMA_VERSION:
        raise ValueError(f"unsupported precompute cache schema: {schema_version}")
    grid_raw = _mapping(raw, "grid")
    grid = CaseGrid(
        ur_bases=_base_grid_from_mapping(_mapping(grid_raw, "ur_bases")),
        sr_bases=_base_grid_from_mapping(_mapping(grid_raw, "sr_bases")),
        lift_heights_m=_float_tuple(grid_raw.get("lift_heights_m")),
        tote_offsets_m=_float_tuple(grid_raw.get("tote_offsets_m")),
        corners=_string_tuple(grid_raw.get("corners")),
        coordination_modes=_string_tuple(grid_raw.get("coordination_modes")),
        skus=_string_tuple(grid_raw.get("skus")),
        sr_j3_strokes_m=_float_tuple(grid_raw.get("sr_j3_strokes_m")),
        clearances_m=tuple(
            None if value is None else float(value)
            for value in _sequence(grid_raw.get("clearances_m"), "clearances_m")
        ),
        tote_present=bool(grid_raw.get("tote_present")),
    )
    settings = CachedEvaluationSettings(**_settings_kwargs(_mapping(raw, "settings")))
    cases = tuple(_case_from_mapping(item) for item in _mapping_sequence(raw, "cases"))
    return PrecomputedCache(
        schema_version=schema_version,
        spec_fingerprint=str(raw.get("spec_fingerprint", "")),
        grid=grid,
        settings=settings,
        cases=cases,
    )


def _case_from_mapping(raw: Mapping[str, object]) -> CachedCase:
    key_raw = _mapping(raw, "key")
    key = CaseKey(
        ur_base=_base_from_mapping(_mapping(key_raw, "ur_base")),
        sr_base=_base_from_mapping(_mapping(key_raw, "sr_base")),
        lift_height_m=float(key_raw["lift_height_m"]),
        tote_offset_m=float(key_raw["tote_offset_m"]),
        corner=str(key_raw["corner"]),
        coordination_mode=CoordinationMode(str(key_raw["coordination_mode"])),
        sku=str(key_raw["sku"]),
        sr_j3_stroke_m=float(key_raw["sr_j3_stroke_m"]),
        clearance_m=None if key_raw.get("clearance_m") is None else float(key_raw["clearance_m"]),
    )
    return CachedCase(
        key=key,
        status=str(raw["status"]),
        sr_nominal_q=_float_tuple(raw.get("sr_nominal_q")),
        installation_valid=bool(raw["installation_valid"]),
        placement_issues=tuple(
            CachedPlacementIssue(
                code=str(item["code"]),
                objects=_string_tuple(item.get("objects")),
                message=str(item["message"]),
            )
            for item in _mapping_sequence(raw, "placement_issues")
        ),
        sr_workspace=_sr_workspace_from_value(raw.get("sr_workspace")),
        sr_keepout=_box_from_value(raw.get("sr_keepout")),
        scenarios=tuple(_scenario_from_mapping(item) for item in _mapping_sequence(raw, "scenarios")),
        coordination_summaries=tuple(
            CachedCoordinationSummary(
                mode=str(item["mode"]),
                selected_corners=_string_tuple(item.get("selected_corners")),
                hard_feasible_corners=_string_tuple(item.get("hard_feasible_corners")),
                hard_feasible=bool(item["hard_feasible"]),
                cell_feasible=bool(item["cell_feasible"]),
            )
            for item in _mapping_sequence(raw, "coordination_summaries")
        ),
    )


def _scenario_from_mapping(raw: Mapping[str, object]) -> CachedScenario:
    return CachedScenario(
        corner=str(raw["corner"]),
        coordination_mode=str(raw["coordination_mode"]),
        hard_feasible=bool(raw["hard_feasible"]),
        soft_step_successes=int(raw["soft_step_successes"]),
        soft_step_count=int(raw["soft_step_count"]),
        metrics=_metric_from_value(raw.get("metrics")),
        steps=tuple(_step_from_mapping(item) for item in _mapping_sequence(raw, "steps")),
    )


def _step_from_mapping(raw: Mapping[str, object]) -> CachedStep:
    return CachedStep(
        index=int(raw["index"]),
        name=str(raw["name"]),
        criticality=str(raw["criticality"]),
        status=str(raw["status"]),
        waste_dimension_fits=(None if raw.get("waste_dimension_fits") is None else bool(raw["waste_dimension_fits"])),
        checks=tuple(_check_from_mapping(item) for item in _mapping_sequence(raw, "checks")),
    )


def _check_from_mapping(raw: Mapping[str, object]) -> CachedLinearCheck:
    return CachedLinearCheck(
        name=str(raw["name"]),
        criticality=str(raw["criticality"]),
        status=str(raw["status"]),
        failed_sample_index=None if raw.get("failed_sample_index") is None else int(raw["failed_sample_index"]),
        joint_collision=_collision_from_value(raw.get("joint_collision")),
        samples=tuple(_sample_from_mapping(item) for item in _mapping_sequence(raw, "samples")),
        include_in_metric_summary=_required_bool(
            raw,
            "include_in_metric_summary",
        ),
    )


def _sample_from_mapping(raw: Mapping[str, object]) -> CachedPoseSample:
    return CachedPoseSample(
        fraction=float(raw["fraction"]),
        target_pose=_pose_from_mapping(_mapping(raw, "target_pose")),
        solved_pose=_pose_from_mapping(_mapping(raw, "solved_pose")),
        ik_status=str(raw["ik_status"]),
        q=_float_tuple(raw.get("q")),
        arm_q=_float_tuple(raw.get("arm_q")),
        iterations=int(raw["iterations"]),
        seed_index=int(raw["seed_index"]),
        attempted_seeds=int(raw["attempted_seeds"]),
        position_error_m=float(raw["position_error_m"]),
        orientation_error_rad=float(raw["orientation_error_rad"]),
        log_residual_norm=float(raw["log_residual_norm"]),
        rejected_by_acceptor=int(raw["rejected_by_acceptor"]),
        collision=_collision_from_value(raw.get("collision")),
        manipulability=_manipulability_from_value(raw.get("manipulability")),
    )


def _pose_from_mapping(raw: Mapping[str, object]) -> CachedPose:
    translation = _float_tuple(raw.get("translation_m"))
    if len(translation) != 3:
        raise ValueError("cached pose translation must contain three values")
    rotation = tuple(_float_tuple(row) for row in _sequence(raw.get("rotation"), "rotation"))
    if len(rotation) != 3 or any(len(row) != 3 for row in rotation):
        raise ValueError("cached pose rotation must be 3x3")
    return CachedPose(translation, rotation)


def _collision_from_value(value: object) -> CachedCollision | None:
    if value is None:
        return None
    raw = _expect_mapping(value, "collision")
    nearest = raw.get("nearest_pair")
    return CachedCollision(
        in_collision=bool(raw["in_collision"]),
        contacts=tuple(
            tuple(_string_pair(pair, "collision contact"))
            for pair in _sequence(raw.get("contacts"), "contacts")
        ),
        minimum_distance_m=None if raw.get("minimum_distance_m") is None else float(raw["minimum_distance_m"]),
        nearest_pair=None if nearest is None else _string_pair(nearest, "nearest_pair"),
    )


def _manipulability_from_value(value: object) -> CachedManipulabilityEllipsoid | None:
    if value is None:
        return None
    raw = _expect_mapping(value, "manipulability")
    return CachedManipulabilityEllipsoid(
        translational_axis_lengths=tuple(_float_tuple(raw.get("translational_axis_lengths"))),
        translational_directions_world=tuple(_float_tuple(row) for row in _sequence(raw.get("translational_directions_world"), "translational_directions_world")),
        normalized_axis_lengths=_float_tuple(raw.get("normalized_axis_lengths")),
        normalized_directions=tuple(_float_tuple(row) for row in _sequence(raw.get("normalized_directions"), "normalized_directions")),
        translational_volume=float(raw["translational_volume"]),
        translational_min_singular_value=float(raw["translational_min_singular_value"]),
        translational_condition_number=_optional_float(raw.get("translational_condition_number")),
        normalized_volume=float(raw["normalized_volume"]),
        normalized_min_singular_value=float(raw["normalized_min_singular_value"]),
        normalized_condition_number=_optional_float(raw.get("normalized_condition_number")),
        characteristic_length_m=float(raw["characteristic_length_m"]),
    )


def _metric_from_value(value: object) -> CachedMetricSummary | None:
    if value is None:
        return None
    raw = _expect_mapping(value, "metrics")
    return CachedMetricSummary(
        sample_count=int(raw["sample_count"]),
        translational_volume_min=float(raw["translational_volume_min"]),
        translational_volume_mean=float(raw["translational_volume_mean"]),
        translational_sigma_min=float(raw["translational_sigma_min"]),
        normalized_volume_min=float(raw["normalized_volume_min"]),
        normalized_volume_mean=float(raw["normalized_volume_mean"]),
        normalized_sigma_min=float(raw["normalized_sigma_min"]),
        normalized_condition_max=_optional_float(raw.get("normalized_condition_max")),
    )


def _sr_workspace_from_value(value: object) -> CachedScaraWorkspace | None:
    if value is None:
        return None
    raw = _expect_mapping(value, "sr_workspace")
    return CachedScaraWorkspace(
        feasible=bool(raw["feasible"]),
        footprint_ok=bool(raw["footprint_ok"]),
        planar_reach_ok=bool(raw["planar_reach_ok"]),
        vertical_ok=bool(raw["vertical_ok"]),
        tool0_z_range_m=tuple(_float_tuple(raw.get("tool0_z_range_m"))),
        targets=tuple(
            CachedScaraTarget(
                name=str(item["name"]),
                world_position_m=tuple(_float_tuple(item.get("world_position_m"))),
                required_j3_m=float(item["required_j3_m"]),
                planar_ik_solutions_rad=tuple(
                    tuple(_float_tuple(solution))
                    for solution in _sequence(item.get("planar_ik_solutions_rad"), "planar_ik_solutions_rad")
                ),
                footprint_ok=bool(item["footprint_ok"]),
                planar_reach_ok=bool(item["planar_reach_ok"]),
                vertical_ok=bool(item["vertical_ok"]),
                feasible=bool(item["feasible"]),
            )
            for item in _mapping_sequence(raw, "targets")
        ),
    )


def _box_from_value(value: object) -> CachedBox | None:
    if value is None:
        return None
    raw = _expect_mapping(value, "sr_keepout")
    return CachedBox(
        name=str(raw["name"]),
        role=str(raw["role"]),
        center_m=tuple(_float_tuple(raw.get("center_m"))),
        size_m=tuple(_float_tuple(raw.get("size_m"))),
        yaw_deg=float(raw["yaw_deg"]),
    )


def _settings_kwargs(raw: Mapping[str, object]) -> dict[str, object]:
    return {
        "cartesian_translation_step_m": float(raw["cartesian_translation_step_m"]),
        "cartesian_rotation_step_rad": float(raw["cartesian_rotation_step_rad"]),
        "joint_interpolation_samples": int(raw["joint_interpolation_samples"]),
        "characteristic_length_m": float(raw["characteristic_length_m"]),
        "ik_position_tolerance_m": float(raw["ik_position_tolerance_m"]),
        "ik_orientation_tolerance_rad": float(raw["ik_orientation_tolerance_rad"]),
        "ik_max_iterations": int(raw["ik_max_iterations"]),
        "ik_damping": float(raw["ik_damping"]),
        "ik_max_backtracking_steps": int(raw["ik_max_backtracking_steps"]),
        "extra_arm_seeds": tuple(_float_tuple(seed) for seed in _sequence(raw.get("extra_arm_seeds"), "extra_arm_seeds")),
        "approach_distance_m": float(raw["approach_distance_m"]),
        "keepout_margin_m": float(raw["keepout_margin_m"]),
    }


def _base_grid_from_mapping(raw: Mapping[str, object]) -> BasePoseGrid:
    return BasePoseGrid(
        x_m=_float_tuple(raw.get("x_m")),
        y_m=_float_tuple(raw.get("y_m")),
        z_m=_float_tuple(raw.get("z_m")),
        yaw_deg=_float_tuple(raw.get("yaw_deg")),
    )


def _base_from_mapping(raw: Mapping[str, object]) -> BasePose:
    return BasePose(
        float(raw["x_m"]),
        float(raw["y_m"]),
        float(raw["z_m"]),
        float(raw["yaw_deg"]),
    )


def _json_compatible(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _json_compatible(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_compatible(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, CoordinationMode):
        return value.value
    if isinstance(value, np.generic):
        return value.item()
    return value


def _finite_unique_axis(values: Iterable[float], name: str) -> tuple[float, ...]:
    result = tuple(_finite(value, name) for value in _nonempty_tuple(values, name))
    if len(set(result)) != len(result):
        raise ValueError(f"{name} must not contain duplicate values")
    return result


def _positive_unique_axis(values: Iterable[float], name: str) -> tuple[float, ...]:
    result = _finite_unique_axis(values, name)
    if any(value <= 0.0 for value in result):
        raise ValueError(f"{name} values must be positive")
    return result


def _nonempty_tuple(values: Iterable[object], name: str) -> tuple[object, ...]:
    result = tuple(values)
    if not result:
        raise ValueError(f"{name} must contain at least one explicit grid value")
    return result


def _finite(value: float, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} values must be finite")
    return result


def _positive_finite(value: float, name: str) -> float:
    result = _finite(value, name)
    if result <= 0.0:
        raise ValueError(f"{name} must be positive")
    return result


def _nonnegative_finite(value: float, name: str) -> float:
    result = _finite(value, name)
    if result < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return result


def _finite_vector(values: Sequence[float] | np.ndarray, name: str) -> tuple[float, ...]:
    result = tuple(float(value) for value in np.asarray(values, dtype=float).reshape(-1))
    if not all(math.isfinite(value) for value in result):
        raise ValueError(f"{name} must contain finite values")
    return result


def _finite_or_none(value: float) -> float | None:
    result = float(value)
    return result if math.isfinite(result) else None


def _optional_float(value: object) -> float | None:
    return None if value is None else float(value)


def _mapping(mapping: Mapping[str, object], key: str) -> Mapping[str, object]:
    return _expect_mapping(mapping.get(key), key)


def _required_bool(mapping: Mapping[str, object], key: str) -> bool:
    if key not in mapping or not isinstance(mapping[key], bool):
        raise ValueError(f"{key} must be a boolean")
    return mapping[key]


def _expect_mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    return value


def _sequence(value: object, name: str) -> Sequence[object]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{name} must be an array")
    return value


def _mapping_sequence(mapping: Mapping[str, object], key: str) -> tuple[Mapping[str, object], ...]:
    return tuple(_expect_mapping(value, key) for value in _sequence(mapping.get(key), key))


def _float_tuple(value: object) -> tuple[float, ...]:
    return tuple(float(item) for item in _sequence(value, "numeric array"))


def _string_tuple(value: object) -> tuple[str, ...]:
    return tuple(str(item) for item in _sequence(value, "string array"))


def _string_pair(value: object, name: str) -> tuple[str, str]:
    result = _string_tuple(value)
    if len(result) != 2:
        raise ValueError(f"{name} must contain two names")
    return result[0], result[1]


def _raise_invalid_json_constant(value: str) -> None:
    raise ValueError(f"invalid non-finite JSON number: {value}")
