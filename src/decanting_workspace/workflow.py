"""UR20 process-task geometry for one decanting-cell scenario.

The module deliberately stops before numerical IK.  It turns a validated
scene snapshot into immutable, World-referenced ``suction_tcp`` pose checks,
including the moving-object transforms required by a later collision backend.
No visualization or optimizer state is created here.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
import math
from typing import Iterable, Sequence

import numpy as np

from .models import BoxPrimitive, SceneSpec, SceneState
from .robots import RobotBundle, floating_configuration
from .scene import CORNER_SIGNS, SceneSnapshot
from .transforms import quaternion_matrix, transform_xy


_EPS = 1.0e-10
_WORLD_X = np.array((1.0, 0.0, 0.0))
_WORLD_Z = np.array((0.0, 0.0, 1.0))

# Bump whenever the generated task/check contract changes.  Precomputed
# caches include this value in their scene/model fingerprint so a UI cannot
# silently replay a stale task sequence after a workflow edit.
WORKFLOW_SCHEMA_VERSION = 3


class Criticality(Enum):
    """Whether failure of one task check invalidates a base candidate."""

    HARD = "hard"
    SOFT = "soft"
    SKIP = "skip"


class CoordinationMode(Enum):
    """Whether UR20 tote handling overlaps the SR cutting operation."""

    SIMULTANEOUS = "simultaneous"
    SEQUENTIAL = "sequential"


@dataclass(frozen=True, eq=False)
class RigidPose:
    """Immutable homogeneous pose with ``World_T_frame`` semantics."""

    world_T_frame: np.ndarray

    def __post_init__(self) -> None:
        matrix = np.asarray(self.world_T_frame, dtype=float).reshape(4, 4).copy()
        if not np.all(np.isfinite(matrix)):
            raise ValueError("pose matrix must contain finite values")
        if not np.allclose(matrix[3], (0.0, 0.0, 0.0, 1.0), atol=1e-10):
            raise ValueError("pose matrix must have a homogeneous final row")
        rotation = matrix[:3, :3]
        if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-8):
            raise ValueError("pose rotation must be orthonormal")
        if not math.isclose(float(np.linalg.det(rotation)), 1.0, abs_tol=1e-8):
            raise ValueError("pose rotation must be right-handed")
        matrix.setflags(write=False)
        object.__setattr__(self, "world_T_frame", matrix)

    @property
    def translation_m(self) -> tuple[float, float, float]:
        return tuple(float(value) for value in self.world_T_frame[:3, 3])

    @property
    def rotation(self) -> np.ndarray:
        """Return a read-only view of ``World_R_frame``."""

        result = self.world_T_frame[:3, :3]
        result.setflags(write=False)
        return result

    @property
    def matrix(self) -> np.ndarray:
        """Alias used by Pinocchio/Coal adapters."""

        return self.world_T_frame


@dataclass(frozen=True)
class ObjectState:
    """Dynamic process-object state for reconstructing a collision phase.

    ``tcp_T_object`` is present whenever an object follows the TCP.  For an
    actual suction grasp ``attached_to_tcp`` is true; during the tote push the
    transform is still supplied but ``attached_to_tcp`` remains false and
    ``follows_tcp`` identifies the kinematic contact constraint.
    """

    name: str
    size_m: tuple[float, float, float]
    world_T_object: RigidPose
    attached_to_tcp: bool = False
    tcp_T_object: RigidPose | None = None
    follows_tcp: bool = False

    def __post_init__(self) -> None:
        if len(self.size_m) != 3 or any(
            not math.isfinite(value) or value <= 0.0 for value in self.size_m
        ):
            raise ValueError("object size must contain three positive finite values")
        if (self.attached_to_tcp or self.follows_tcp) and self.tcp_T_object is None:
            raise ValueError("a TCP-coupled object requires tcp_T_object")


@dataclass(frozen=True)
class LinearPoseCheck:
    """One straight pose segment evaluated at least at both endpoints."""

    name: str
    start_tcp: RigidPose
    end_tcp: RigidPose
    criticality: Criticality
    object_states: tuple[ObjectState, ...] = ()
    payload_name: str | None = None
    contact_object_name: str | None = None
    allowed_contacts: tuple[tuple[str, str], ...] = ()
    include_in_metric_summary: bool = True


@dataclass(frozen=True)
class WasteFitCheck:
    """Dimension-only box fit through the third-level conveyor width."""

    conveyor_name: str
    available_width_m: float
    clearance_m: float
    yaw_options_deg: tuple[float, ...]
    cross_width_options_m: tuple[float, ...]
    selected_yaw_deg: float
    selected_cross_width_m: float
    fits: bool


@dataclass(frozen=True)
class TaskStepSpec:
    """All task targets and dynamic objects for one numbered process step."""

    index: int
    name: str
    criticality: Criticality
    pose_checks: tuple[LinearPoseCheck, ...]
    object_states: tuple[ObjectState, ...] = ()
    suppress_snapshot_objects: tuple[str, ...] = ()
    active_extra_obstacles: tuple[BoxPrimitive, ...] = ()
    waste_fit: WasteFitCheck | None = None


@dataclass(frozen=True)
class ProcessScenario:
    """One physical corner alternative and one coordination policy."""

    scene_state: SceneState
    corner: str
    coordination_mode: CoordinationMode

    def __post_init__(self) -> None:
        if self.corner not in CORNER_SIGNS:
            raise ValueError(f"unknown pallet corner: {self.corner}")
        if self.scene_state.corner_samples != (self.corner,):
            raise ValueError("process scenario state must contain exactly its corner")


def expand_process_scenarios(
    spec: SceneSpec,
    state: SceneState | None = None,
    coordination_modes: Iterable[CoordinationMode | str] = (
        CoordinationMode.SIMULTANEOUS,
        CoordinationMode.SEQUENTIAL,
    ),
) -> tuple[ProcessScenario, ...]:
    """Expand the selected pallet corners and coordination modes.

    SKU, lift, tote offset, and SR-stroke sampling remain outer-loop concerns;
    the supplied ``SceneState`` represents one selection of those axes.  Four
    selected corners become four independent physical scenes rather than four
    simultaneous collision boxes.
    """

    base_state = state or SceneState()
    if base_state.sku not in spec.box_skus:
        raise ValueError(f"unknown box SKU: {base_state.sku}")
    if not base_state.corner_samples:
        raise ValueError("at least one pallet corner is required")
    unknown = set(base_state.corner_samples) - set(CORNER_SIGNS)
    if unknown:
        raise ValueError(f"unknown pallet corners: {sorted(unknown)}")

    modes: list[CoordinationMode] = []
    for raw_mode in coordination_modes:
        mode = (
            raw_mode
            if isinstance(raw_mode, CoordinationMode)
            else CoordinationMode(str(raw_mode))
        )
        if mode not in modes:
            modes.append(mode)
    if not modes:
        raise ValueError("at least one coordination mode is required")

    scenarios: list[ProcessScenario] = []
    for corner in base_state.corner_samples:
        single_corner_state = replace(base_state, corner_samples=(corner,))
        for mode in modes:
            scenarios.append(
                ProcessScenario(
                    scene_state=single_corner_state,
                    corner=corner,
                    coordination_mode=mode,
                )
            )
    return tuple(scenarios)


def build_ur20_task_sequence(
    spec: SceneSpec,
    snapshot: SceneSnapshot,
    scenario: ProcessScenario,
    ur_bundle: RobotBundle,
    *,
    sr_keepout: BoxPrimitive | None = None,
    approach_distance_m: float = 0.15,
) -> tuple[TaskStepSpec, ...]:
    """Build hard/soft UR20 checks for process steps 1 through 8.

    The returned poses all target ``suction_tcp``.  Geometry is constructed
    without solving IK so a loaded UR bundle can be reused across every base
    candidate and scenario iteration.
    """

    distance = float(approach_distance_m)
    if not math.isfinite(distance) or distance <= 0.0:
        raise ValueError("approach distance must be finite and positive")
    _validate_snapshot_scenario(snapshot, scenario)

    box = _find_scenario_box(snapshot, scenario)
    box_pose_pallet = _box_pose(box)
    tote_size = spec.tote.size_m
    tote_name = "representative_tote"
    suppressed = tuple(
        item.name
        for item in snapshot.boxes
        if item.role in {"box", "box_sample_alternative", "tote"}
    )
    # Include logical names even when a phase object was absent in the input
    # snapshot (for example a compute-only state with tote_present=False).
    suppressed = tuple(dict.fromkeys((*suppressed, box.name, tote_name)))

    pick_rotation = _nominal_suction_rotation(spec, snapshot, ur_bundle)
    pick_contact = _pose(
        pick_rotation,
        np.asarray(box.center_m) + np.array((0.0, 0.0, box.size_m[2] / 2.0)),
    )
    pick_approach = _translated(pick_contact, _WORLD_Z * distance)
    tcp_T_box_top = _relative(pick_contact, box_pose_pallet)
    pick_lift = _translated(pick_contact, _WORLD_Z * distance)
    lifted_box_pose = _compose(pick_lift, tcp_T_box_top)

    pallet_box_world = _world_object(box.name, box.size_m, box_pose_pallet)
    pallet_box_attached = _coupled_object(
        box.name,
        box.size_m,
        lifted_box_pose,
        tcp_T_box_top,
        attached=True,
    )
    pallet_support = ((box.name, "pallet"),)
    # ``suction_pad`` is a logical collision alias resolved by the later
    # collision-world adapter for either primitive or official UR geometry.
    box_suction_contact = (("suction_pad", box.name),)
    step1 = TaskStepSpec(
        index=1,
        name="pick_box_from_pallet",
        criticality=Criticality.HARD,
        pose_checks=(
            LinearPoseCheck(
                "pallet_box_approach",
                pick_approach,
                pick_contact,
                Criticality.HARD,
                (pallet_box_world,),
                contact_object_name=box.name,
                allowed_contacts=(*pallet_support, *box_suction_contact),
            ),
            LinearPoseCheck(
                "pallet_box_lift",
                pick_contact,
                pick_lift,
                Criticality.HARD,
                (pallet_box_attached,),
                payload_name=box.name,
                contact_object_name=box.name,
                allowed_contacts=(*pallet_support, *box_suction_contact),
            ),
        ),
        object_states=(pallet_box_world,),
        suppress_snapshot_objects=suppressed,
    )

    # A world-Z half turn is deliberately pre-multiplied.  This preserves the
    # fixed top-centre grasp and rotates the box itself exactly 180 degrees,
    # independent of candidate-base yaw.
    uncasing_frame = _frame_pose(snapshot, "UncasingLoadFrame")
    uncasing_box_rotation = _yaw_rotation(math.pi) @ box_pose_pallet.rotation
    uncasing_box_center = _transform_point(
        uncasing_frame, (0.0, 0.0, box.size_m[2] / 2.0)
    )
    box_pose_uncasing = _pose(uncasing_box_rotation, uncasing_box_center)
    uncasing_tcp = _compose(box_pose_uncasing, _inverse(tcp_T_box_top))
    expected_uncasing_rotation = _yaw_rotation(math.pi) @ pick_rotation
    if not np.allclose(uncasing_tcp.rotation, expected_uncasing_rotation, atol=1e-8):
        raise RuntimeError("uncasing target does not preserve the top-centre grasp")
    uncasing_approach = _translated(uncasing_tcp, _WORLD_Z * distance)
    uncasing_retreat = _translated(uncasing_tcp, _WORLD_Z * distance)
    uncasing_box_attached = _coupled_object(
        box.name,
        box.size_m,
        box_pose_uncasing,
        tcp_T_box_top,
        attached=True,
    )
    uncasing_box_world = _world_object(box.name, box.size_m, box_pose_uncasing)
    step2 = TaskStepSpec(
        index=2,
        name="place_box_at_uncasing",
        criticality=Criticality.HARD,
        pose_checks=(
            LinearPoseCheck(
                "uncasing_box_preplace",
                uncasing_approach,
                uncasing_tcp,
                Criticality.HARD,
                (uncasing_box_attached,),
                payload_name=box.name,
                contact_object_name=box.name,
                allowed_contacts=box_suction_contact,
            ),
            LinearPoseCheck(
                "uncasing_box_release_retreat",
                uncasing_tcp,
                uncasing_retreat,
                Criticality.HARD,
                (uncasing_box_world,),
                contact_object_name=box.name,
                allowed_contacts=box_suction_contact,
            ),
        ),
        object_states=(uncasing_box_attached,),
        suppress_snapshot_objects=suppressed,
    )

    conveyor_l2 = _static_box(spec, "conveyor_level_2")
    pickup_frame = _frame_pose(snapshot, "TotePickupFrame")
    tote_rotation = _yaw_rotation(math.radians(spec.tote.representative_yaw_deg))
    tote_pickup_center = np.array(
        (
            pickup_frame.translation_m[0],
            pickup_frame.translation_m[1],
            _box_top_z(conveyor_l2) + tote_size[2] / 2.0,
        )
    )
    tote_pose_pickup = _pose(tote_rotation, tote_pickup_center)
    tote_half_world_x = _oriented_half_extent(tote_rotation, tote_size, _WORLD_X)
    tote_pick_contact_point = tote_pickup_center - _WORLD_X * tote_half_world_x
    front_positive_x_rotation = _front_grasp_rotation(_WORLD_X)
    tote_pick_contact = _pose(front_positive_x_rotation, tote_pick_contact_point)
    tote_pick_approach = _translated(tote_pick_contact, -_WORLD_X * distance)
    tcp_T_tote = _relative(tote_pick_contact, tote_pose_pickup)
    # After a normal-direction approach the tote is lifted vertically from
    # the conveyor.  A second horizontal retreat would drag the tote across
    # its support surface and does not match the requested "front grasp, then
    # lift" operation.
    tote_lift = _translated(tote_pick_contact, _WORLD_Z * distance)
    tote_lift_pose = _compose(tote_lift, tcp_T_tote)
    tote_pickup_world = _world_object(tote_name, tote_size, tote_pose_pickup)
    tote_pickup_attached = _coupled_object(
        tote_name,
        tote_size,
        tote_lift_pose,
        tcp_T_tote,
        attached=True,
    )
    tote_l2_support = ((tote_name, conveyor_l2.name),)
    tote_suction_contact = (("suction_pad", tote_name),)
    concurrent_obstacles = _concurrent_obstacles(
        spec, snapshot, scenario, sr_keepout
    )
    step3 = TaskStepSpec(
        index=3,
        name="pick_empty_tote_from_level_2",
        criticality=Criticality.HARD,
        pose_checks=(
            LinearPoseCheck(
                "level_2_tote_front_approach",
                tote_pick_approach,
                tote_pick_contact,
                Criticality.HARD,
                (uncasing_box_world, tote_pickup_world),
                contact_object_name=tote_name,
                allowed_contacts=(*tote_l2_support, *tote_suction_contact),
            ),
            LinearPoseCheck(
                "level_2_tote_lift",
                tote_pick_contact,
                tote_lift,
                Criticality.HARD,
                (uncasing_box_world, tote_pickup_attached),
                payload_name=tote_name,
                allowed_contacts=(*tote_l2_support, *tote_suction_contact),
            ),
        ),
        object_states=(uncasing_box_world, tote_pickup_world),
        suppress_snapshot_objects=suppressed,
        active_extra_obstacles=concurrent_obstacles,
    )

    tote_pose_table = _table_tote_pose(spec, snapshot)
    tote_table_tcp = _compose(tote_pose_table, _inverse(tcp_T_tote))
    tote_table_approach = _translated(tote_table_tcp, _WORLD_Z * distance)
    tote_table_retreat = _translated(tote_table_tcp, -_WORLD_X * distance)
    tote_table_attached = _coupled_object(
        tote_name,
        tote_size,
        tote_pose_table,
        tcp_T_tote,
        attached=True,
    )
    tote_table_world = _world_object(tote_name, tote_size, tote_pose_table)
    tote_table_support = ((tote_name, spec.tote.motion_support_box),)
    step4 = TaskStepSpec(
        index=4,
        name="place_tote_on_worktable",
        criticality=Criticality.HARD,
        pose_checks=(
            LinearPoseCheck(
                "worktable_tote_preplace",
                tote_table_approach,
                tote_table_tcp,
                Criticality.HARD,
                (uncasing_box_world, tote_table_attached),
                payload_name=tote_name,
                contact_object_name=tote_name,
                allowed_contacts=(*tote_table_support, *tote_suction_contact),
            ),
            LinearPoseCheck(
                "worktable_tote_release_retreat",
                tote_table_tcp,
                tote_table_retreat,
                Criticality.HARD,
                (uncasing_box_world, tote_table_world),
                contact_object_name=tote_name,
                allowed_contacts=(*tote_table_support, *tote_suction_contact),
            ),
        ),
        object_states=(uncasing_box_world, tote_table_attached),
        suppress_snapshot_objects=suppressed,
        active_extra_obstacles=concurrent_obstacles,
    )

    box_face_point, box_outward_normal = _nearest_vertical_face(
        box_pose_uncasing,
        box.size_m,
        snapshot.ur_base.xyz_m[:2],
    )
    box_inward_normal = -box_outward_normal
    box_front_rotation = _front_grasp_rotation(box_inward_normal)
    box_front_contact = _pose(box_front_rotation, box_face_point)
    box_front_approach = _translated(
        box_front_contact, box_outward_normal * distance
    )
    tcp_T_box_front = _relative(box_front_contact, box_pose_uncasing)
    box_front_outside = _translated(
        box_front_contact, box_outward_normal * distance
    )
    box_front_lift = _translated(box_front_outside, _WORLD_Z * distance)
    box_front_lift_pose = _compose(box_front_lift, tcp_T_box_front)
    box_front_attached = _coupled_object(
        box.name,
        box.size_m,
        box_front_lift_pose,
        tcp_T_box_front,
        attached=True,
    )
    step5 = TaskStepSpec(
        index=5,
        name="pick_opened_box_from_uncasing",
        criticality=Criticality.HARD,
        pose_checks=(
            LinearPoseCheck(
                "opened_box_front_approach",
                box_front_approach,
                box_front_contact,
                Criticality.HARD,
                (uncasing_box_world, tote_table_world),
                contact_object_name=box.name,
                allowed_contacts=(*tote_table_support, *box_suction_contact),
            ),
            LinearPoseCheck(
                "opened_box_front_retreat",
                box_front_contact,
                box_front_outside,
                Criticality.HARD,
                (
                    _coupled_object(
                        box.name,
                        box.size_m,
                        _compose(box_front_outside, tcp_T_box_front),
                        tcp_T_box_front,
                        attached=True,
                    ),
                    tote_table_world,
                ),
                payload_name=box.name,
                contact_object_name=box.name,
                allowed_contacts=(*tote_table_support, *box_suction_contact),
            ),
            LinearPoseCheck(
                "opened_box_lift",
                box_front_outside,
                box_front_lift,
                Criticality.HARD,
                (box_front_attached, tote_table_world),
                payload_name=box.name,
                allowed_contacts=(*tote_table_support, *box_suction_contact),
            ),
        ),
        object_states=(uncasing_box_world, tote_table_world),
        suppress_snapshot_objects=suppressed,
    )

    step6 = TaskStepSpec(
        index=6,
        name="pour_box_contents_into_tote",
        criticality=Criticality.SKIP,
        pose_checks=(),
        object_states=(box_front_attached, tote_table_world),
        suppress_snapshot_objects=suppressed,
    )

    waste_fit = _waste_fit_check(spec, snapshot, box_pose_uncasing, box.size_m)
    waste_frame = _frame_pose(snapshot, "WasteFrame")
    waste_rotation = _yaw_rotation(math.radians(waste_fit.selected_yaw_deg))
    waste_center = _transform_point(
        waste_frame, (0.0, 0.0, box.size_m[2] / 2.0)
    )
    waste_box_pose = _pose(waste_rotation, waste_center)
    waste_tcp = _compose(waste_box_pose, _inverse(tcp_T_box_front))
    waste_box_attached = _coupled_object(
        box.name,
        box.size_m,
        waste_box_pose,
        tcp_T_box_front,
        attached=True,
    )
    step7 = TaskStepSpec(
        index=7,
        name="discard_empty_box_to_level_3",
        criticality=Criticality.SOFT,
        pose_checks=(
            LinearPoseCheck(
                "waste_release_pose",
                waste_tcp,
                waste_tcp,
                Criticality.SOFT,
                (waste_box_attached, tote_table_world),
                payload_name=box.name,
                allowed_contacts=(*tote_table_support, *box_suction_contact),
            ),
        ),
        object_states=(box_front_attached, tote_table_world),
        suppress_snapshot_objects=suppressed,
        waste_fit=waste_fit,
    )

    supply_frame = _frame_pose(snapshot, "ToteSupplyFrame")
    table_center = np.asarray(tote_pose_table.translation_m)
    push_z = supply_frame.translation_m[2]
    tote_bottom = table_center[2] - tote_size[2] / 2.0
    tote_top = table_center[2] + tote_size[2] / 2.0
    if not tote_bottom - _EPS <= push_z <= tote_top + _EPS:
        raise ValueError("ToteSupplyFrame Z is outside the tote side face")
    push_start_point = np.array(
        (table_center[0] - tote_half_world_x, table_center[1], push_z)
    )
    push_start = _pose(front_positive_x_rotation, push_start_point)
    push_approach = _translated(push_start, -_WORLD_X * distance)
    push_end = _pose(
        front_positive_x_rotation,
        np.asarray(supply_frame.translation_m),
    )
    if push_end.translation_m[0] <= push_start.translation_m[0] + _EPS:
        raise ValueError("ToteSupplyFrame must be beyond the rear tote face toward +X")
    tcp_T_tote_push = _relative(push_start, tote_pose_table)
    tote_level_1_support = ((tote_name, "conveyor_level_1"),)
    pushed_tote_pose = _compose(push_end, tcp_T_tote_push)
    pushed_tote = _coupled_object(
        tote_name,
        tote_size,
        pushed_tote_pose,
        tcp_T_tote_push,
        attached=False,
        follows=True,
    )
    step8 = TaskStepSpec(
        index=8,
        name="push_filled_tote_to_level_1",
        criticality=Criticality.HARD,
        pose_checks=(
            LinearPoseCheck(
                "filled_tote_push_approach",
                push_approach,
                push_start,
                Criticality.HARD,
                (tote_table_world,),
                contact_object_name=tote_name,
                allowed_contacts=(
                    *tote_table_support,
                    *tote_level_1_support,
                    *tote_suction_contact,
                ),
            ),
            LinearPoseCheck(
                "filled_tote_push_to_supply",
                push_start,
                push_end,
                Criticality.HARD,
                (pushed_tote,),
                contact_object_name=tote_name,
                # The tote remains supported by/slides over the worktable
                # while it follows the pushing TCP.
                allowed_contacts=(
                    *tote_table_support,
                    *tote_level_1_support,
                    *tote_suction_contact,
                ),
            ),
            LinearPoseCheck(
                "filled_tote_final_pose_at_supply",
                push_end,
                push_end,
                Criticality.HARD,
                (pushed_tote,),
                contact_object_name=tote_name,
                allowed_contacts=(
                    *tote_table_support,
                    *tote_level_1_support,
                    *tote_suction_contact,
                ),
                include_in_metric_summary=False,
            ),
        ),
        object_states=(tote_table_world,),
        suppress_snapshot_objects=suppressed,
    )

    return (step1, step2, step3, step4, step5, step6, step7, step8)


def _validate_snapshot_scenario(
    snapshot: SceneSnapshot,
    scenario: ProcessScenario,
) -> None:
    state = scenario.scene_state
    snapshot_state = snapshot.state
    fields = (
        "sku",
        "lift_height_m",
        "tote_long_axis_offset_m",
        "sr_j3_stroke_m",
    )
    for field in fields:
        if getattr(snapshot_state, field) != getattr(state, field):
            raise ValueError(f"snapshot and process scenario differ in {field}")


def _find_scenario_box(
    snapshot: SceneSnapshot,
    scenario: ProcessScenario,
) -> BoxPrimitive:
    name = f"box_{scenario.scene_state.sku}_{scenario.corner}"
    try:
        return next(box for box in snapshot.boxes if box.name == name)
    except StopIteration as exc:
        raise ValueError(f"scenario box is absent from snapshot: {name}") from exc


def _nominal_suction_rotation(
    spec: SceneSpec,
    snapshot: SceneSnapshot,
    bundle: RobotBundle,
) -> np.ndarray:
    import pinocchio as pin

    if not bundle.model.existFrame("suction_tcp"):
        raise ValueError("UR20 bundle must contain the suction_tcp frame")
    q = floating_configuration(
        bundle,
        snapshot.ur_base,
        spec.robots["ur20"].nominal_q,
    )
    data = bundle.model.createData()
    pin.framesForwardKinematics(bundle.model, data, q)
    frame_id = bundle.model.getFrameId("suction_tcp")
    rotation = np.asarray(data.oMf[frame_id].rotation, dtype=float).copy()
    if float(np.dot(rotation[:, 2], -_WORLD_Z)) < 1.0 - 1.0e-6:
        raise ValueError("UR20 nominal suction orientation is not top-down")
    return rotation


def _concurrent_obstacles(
    spec: SceneSpec,
    snapshot: SceneSnapshot,
    scenario: ProcessScenario,
    supplied: BoxPrimitive | None,
) -> tuple[BoxPrimitive, ...]:
    if scenario.coordination_mode is CoordinationMode.SEQUENTIAL:
        return ()
    if supplied is not None:
        if not supplied.collision_enabled:
            raise ValueError("SR concurrent keepout must be collision-enabled")
        return (supplied,)

    # Default conservative task-cell prism.  A caller with sampled SR link
    # sweep bounds can replace it through ``sr_keepout`` without changing the
    # workflow contract.
    robot = spec.robots["sr12ia"]
    workspace = robot.workspace
    if workspace is None:
        raise ValueError("SR-12iA workspace specification is missing")
    try:
        option_index = next(
            index
            for index, stroke in enumerate(workspace.j3_stroke_options_m)
            if abs(stroke - scenario.scene_state.sr_j3_stroke_m) <= 1e-9
        )
    except StopIteration as exc:
        raise ValueError("unsupported SR J3 stroke in process scenario") from exc
    clearance = scenario.scene_state.clearance_m or 0.0
    center_x, center_y = transform_xy(
        robot.pedestal.center_offset_local_xy_m,
        snapshot.sr_base.xyz_m[:2],
        snapshot.sr_base.yaw_rad,
    )
    z_min = spec.floor_z_m
    z_max = snapshot.sr_base.z_m + workspace.total_height_options_m[option_index]
    return (
        BoxPrimitive(
            name="sr12ia_concurrent_task_keepout",
            role="dynamic_keepout",
            center_m=(center_x, center_y, (z_min + z_max) / 2.0),
            size_m=(
                robot.pedestal.size_xy_m[0] + 2.0 * clearance,
                robot.pedestal.size_xy_m[1] + 2.0 * clearance,
                z_max - z_min,
            ),
            yaw_deg=snapshot.sr_base.yaw_deg,
            collision_enabled=True,
        ),
    )


def _nearest_vertical_face(
    object_pose: RigidPose,
    size_m: Sequence[float],
    base_xy_m: Sequence[float],
) -> tuple[np.ndarray, np.ndarray]:
    size = np.asarray(size_m, dtype=float)
    center = np.asarray(object_pose.translation_m)
    rotation = object_pose.rotation
    base_xy = np.asarray(base_xy_m, dtype=float).reshape(2)
    candidates: list[tuple[float, np.ndarray, np.ndarray]] = []
    for local_outward, half_extent in (
        (np.array((1.0, 0.0, 0.0)), size[0] / 2.0),
        (np.array((-1.0, 0.0, 0.0)), size[0] / 2.0),
        (np.array((0.0, 1.0, 0.0)), size[1] / 2.0),
        (np.array((0.0, -1.0, 0.0)), size[1] / 2.0),
    ):
        outward = rotation @ local_outward
        point = center + outward * half_extent
        distance_squared = float(np.sum((point[:2] - base_xy) ** 2))
        candidates.append((distance_squared, point, outward))
    _, point, outward = min(candidates, key=lambda item: item[0])
    return point, outward / np.linalg.norm(outward)


def _front_grasp_rotation(inward_normal_world: Sequence[float]) -> np.ndarray:
    tool_z = np.asarray(inward_normal_world, dtype=float).reshape(3)
    norm = float(np.linalg.norm(tool_z))
    if norm <= _EPS:
        raise ValueError("front-grasp inward normal must be non-zero")
    tool_z /= norm
    if abs(float(np.dot(tool_z, _WORLD_Z))) > 1.0e-8:
        raise ValueError("front-grasp normal must be horizontal")
    tool_x = _WORLD_Z.copy()
    tool_y = np.cross(tool_z, tool_x)
    tool_y /= np.linalg.norm(tool_y)
    return np.column_stack((tool_x, tool_y, tool_z))


def _waste_fit_check(
    spec: SceneSpec,
    snapshot: SceneSnapshot,
    box_pose: RigidPose,
    box_size_m: Sequence[float],
) -> WasteFitCheck:
    conveyor = _static_box(spec, "conveyor_level_3")
    short_index = 0 if conveyor.size_m[0] <= conveyor.size_m[1] else 1
    available = conveyor.size_m[short_index]
    clearance = snapshot.state.clearance_m or 0.0
    available_after_clearance = available - 2.0 * clearance
    base_yaw = math.atan2(box_pose.rotation[1, 0], box_pose.rotation[0, 0])
    yaw_options = (base_yaw, base_yaw + math.pi / 2.0)
    cross_widths: list[float] = []
    for yaw in yaw_options:
        relative = yaw - math.radians(conveyor.yaw_deg)
        if short_index == 0:
            width = (
                abs(math.cos(relative)) * box_size_m[0]
                + abs(math.sin(relative)) * box_size_m[1]
            )
        else:
            width = (
                abs(math.sin(relative)) * box_size_m[0]
                + abs(math.cos(relative)) * box_size_m[1]
            )
        cross_widths.append(float(width))
    selected_index = int(np.argmin(cross_widths))
    selected_width = cross_widths[selected_index]
    return WasteFitCheck(
        conveyor_name=conveyor.name,
        available_width_m=available_after_clearance,
        clearance_m=clearance,
        yaw_options_deg=tuple(math.degrees(value) for value in yaw_options),
        cross_width_options_m=tuple(cross_widths),
        selected_yaw_deg=math.degrees(yaw_options[selected_index]),
        selected_cross_width_m=selected_width,
        fits=selected_width <= available_after_clearance + _EPS,
    )


def _table_tote_pose(spec: SceneSpec, snapshot: SceneSnapshot) -> RigidPose:
    frame = snapshot.frames[spec.tote.representative_frame]
    center = (
        frame.translation_m[0],
        frame.translation_m[1],
        spec.tote.support_surface_z_m + spec.tote.size_m[2] / 2.0,
    )
    return _pose(
        _yaw_rotation(math.radians(spec.tote.representative_yaw_deg)),
        center,
    )


def _frame_pose(snapshot: SceneSnapshot, name: str) -> RigidPose:
    try:
        frame = snapshot.frames[name]
    except KeyError as exc:
        raise ValueError(f"required workflow frame is missing: {name}") from exc
    return RigidPose(quaternion_matrix(frame.translation_m, frame.rotation_xyzw))


def _static_box(spec: SceneSpec, name: str) -> BoxPrimitive:
    try:
        return next(box for box in spec.static_boxes if box.name == name)
    except StopIteration as exc:
        raise ValueError(f"required static box is missing: {name}") from exc


def _box_top_z(box: BoxPrimitive) -> float:
    # Scene boxes only rotate about Z, so their vertical extent is unchanged.
    return box.center_m[2] + box.size_m[2] / 2.0


def _box_pose(box: BoxPrimitive) -> RigidPose:
    return _pose(_yaw_rotation(math.radians(box.yaw_deg)), box.center_m)


def _world_object(
    name: str,
    size_m: Sequence[float],
    pose: RigidPose,
) -> ObjectState:
    return ObjectState(name, tuple(float(value) for value in size_m), pose)


def _coupled_object(
    name: str,
    size_m: Sequence[float],
    world_pose: RigidPose,
    tcp_T_object: RigidPose,
    *,
    attached: bool,
    follows: bool = False,
) -> ObjectState:
    return ObjectState(
        name,
        tuple(float(value) for value in size_m),
        world_pose,
        attached_to_tcp=attached,
        tcp_T_object=tcp_T_object,
        follows_tcp=follows,
    )


def _oriented_half_extent(
    rotation: np.ndarray,
    size_m: Sequence[float],
    axis_world: np.ndarray,
) -> float:
    half = np.asarray(size_m, dtype=float) / 2.0
    return float(np.sum(np.abs(axis_world @ rotation) * half))


def _yaw_rotation(angle_rad: float) -> np.ndarray:
    c = math.cos(float(angle_rad))
    s = math.sin(float(angle_rad))
    return np.array(((c, -s, 0.0), (s, c, 0.0), (0.0, 0.0, 1.0)))


def _pose(rotation: np.ndarray, translation_m: Sequence[float]) -> RigidPose:
    matrix = np.eye(4)
    matrix[:3, :3] = np.asarray(rotation, dtype=float).reshape(3, 3)
    matrix[:3, 3] = np.asarray(translation_m, dtype=float).reshape(3)
    return RigidPose(matrix)


def _translated(pose: RigidPose, world_offset_m: Sequence[float]) -> RigidPose:
    matrix = pose.matrix.copy()
    matrix[:3, 3] += np.asarray(world_offset_m, dtype=float).reshape(3)
    return RigidPose(matrix)


def _inverse(pose: RigidPose) -> RigidPose:
    rotation = pose.rotation
    translation = np.asarray(pose.translation_m)
    matrix = np.eye(4)
    matrix[:3, :3] = rotation.T
    matrix[:3, 3] = -rotation.T @ translation
    return RigidPose(matrix)


def _compose(parent: RigidPose, child: RigidPose) -> RigidPose:
    return RigidPose(parent.matrix @ child.matrix)


def _relative(parent: RigidPose, child: RigidPose) -> RigidPose:
    return _compose(_inverse(parent), child)


def _transform_point(pose: RigidPose, local_point: Sequence[float]) -> np.ndarray:
    return (
        pose.rotation @ np.asarray(local_point, dtype=float).reshape(3)
        + np.asarray(pose.translation_m)
    )
