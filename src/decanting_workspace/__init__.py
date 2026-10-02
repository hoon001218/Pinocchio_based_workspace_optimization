"""Dimension-based decanting-cell workspace model."""

from .candidates import (
    BaseCandidateEvaluation,
    CoordinationSummary,
    evaluate_base_candidate,
)
from .collision import sr_concurrent_exclusion_box
from .evaluation import (
    EvaluationOptions,
    MetricSummary,
    ScenarioEvaluation,
    evaluate_task_sequence,
)
from .kinematics import (
    IKOptions,
    IKResult,
    ManipulabilityMetrics,
    arm_frame_jacobian,
    compute_manipulability,
    frame_pose,
    solve_frame_ik,
)

from .models import (
    BasePose,
    BoxPrimitive,
    MeshObstacle,
    SceneSpec,
    SceneState,
    load_scene_spec,
)
from .scene import (
    SceneSnapshot,
    materialize_scene,
    sr_cutting_workspace_footprint_box,
    tote_long_axis_offset_range_m,
    tote_motion_axis_world_xy,
)
from .placement import PlacementIssue, PlacementReport, validate_base_placement
from .scara_workspace import (
    ScaraTargetCheck,
    ScaraTaskWorkspaceReport,
    evaluate_sr_cutting_workspace,
)
from .precompute import (
    BasePoseGrid,
    CaseGrid,
    CaseKey,
    PrecomputedCache,
    load_precomputed_cache,
    merge_precomputed_caches,
    precompute_cases,
    save_precomputed_cache,
)
from .meshcat_playback import UR20MeshcatPlayback
from .playback_controller import PrecomputedPlaybackController
from .workflow import (
    CoordinationMode,
    Criticality,
    ProcessScenario,
    TaskStepSpec,
    build_ur20_task_sequence,
    expand_process_scenarios,
)

__all__ = [
    "BasePose",
    "BasePoseGrid",
    "BaseCandidateEvaluation",
    "BoxPrimitive",
    "MeshObstacle",
    "CaseGrid",
    "CaseKey",
    "CoordinationMode",
    "CoordinationSummary",
    "Criticality",
    "EvaluationOptions",
    "IKOptions",
    "IKResult",
    "ManipulabilityMetrics",
    "MetricSummary",
    "PlacementIssue",
    "PlacementReport",
    "ProcessScenario",
    "PrecomputedCache",
    "PrecomputedPlaybackController",
    "ScaraTargetCheck",
    "ScaraTaskWorkspaceReport",
    "SceneSnapshot",
    "SceneSpec",
    "SceneState",
    "ScenarioEvaluation",
    "TaskStepSpec",
    "UR20MeshcatPlayback",
    "arm_frame_jacobian",
    "build_ur20_task_sequence",
    "compute_manipulability",
    "evaluate_base_candidate",
    "evaluate_sr_cutting_workspace",
    "evaluate_task_sequence",
    "expand_process_scenarios",
    "frame_pose",
    "load_scene_spec",
    "load_precomputed_cache",
    "materialize_scene",
    "merge_precomputed_caches",
    "precompute_cases",
    "save_precomputed_cache",
    "solve_frame_ik",
    "sr_concurrent_exclusion_box",
    "sr_cutting_workspace_footprint_box",
    "tote_long_axis_offset_range_m",
    "tote_motion_axis_world_xy",
    "validate_base_placement",
]
