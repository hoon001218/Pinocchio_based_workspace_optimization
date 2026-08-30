"""Command-line report for one pre-optimization base candidate."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path
import sys

from .candidates import BaseCandidateEvaluation, evaluate_base_candidate
from .evaluation import EvaluationOptions
from .kinematics import IKOptions
from .models import BasePose, SceneState, default_config_path, load_scene_spec
from .robots import load_robot_bundle, resolve_official_ur20
from .scene import CORNER_SIGNS
from .workflow import CoordinationMode, WORKFLOW_SCHEMA_VERSION


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate one UR20/SR-12iA base pair with Pinocchio IK, Coal "
            "collision checks, and raw Jacobian-ellipsoid metrics."
        )
    )
    parser.add_argument("--config", type=Path, default=default_config_path())
    parser.add_argument(
        "--ur-base",
        nargs=4,
        type=float,
        metavar=("X_M", "Y_M", "Z_M", "YAW_DEG"),
        help="UR20 base candidate; defaults to the nominal config pose.",
    )
    parser.add_argument(
        "--sr-base",
        nargs=4,
        type=float,
        metavar=("X_M", "Y_M", "Z_M", "YAW_DEG"),
        help="SR-12iA base candidate; defaults to the nominal config pose.",
    )
    parser.add_argument("--lift-height-mm", type=float, default=0.0)
    parser.add_argument("--sku", default="123591")
    parser.add_argument(
        "--corner",
        choices=("all", *CORNER_SIGNS),
        default="all",
        help="Evaluate four independent alternatives or one pallet corner.",
    )
    parser.add_argument("--tote-table-offset-mm", type=float, default=0.0)
    parser.add_argument(
        "--sr-j3-stroke-mm",
        type=float,
        choices=(300.0, 450.0),
        default=300.0,
    )
    parser.add_argument("--clearance-mm", type=float)
    parser.add_argument(
        "--coordination",
        choices=("both", "simultaneous", "sequential"),
        default="both",
    )
    parser.add_argument("--approach-mm", type=float, default=150.0)
    parser.add_argument("--cartesian-step-mm", type=float, default=50.0)
    parser.add_argument("--cartesian-rotation-step-deg", type=float, default=5.0)
    parser.add_argument("--joint-interpolation-samples", type=int, default=3)
    parser.add_argument("--characteristic-length-mm", type=float, default=500.0)
    parser.add_argument("--keepout-margin-mm", type=float, default=0.0)
    parser.add_argument("--position-tolerance-mm", type=float, default=0.001)
    parser.add_argument("--orientation-tolerance-deg", type=float, default=0.0001)
    parser.add_argument("--ik-max-iterations", type=int, default=500)
    parser.add_argument(
        "--ur20-source",
        choices=("official", "primitive"),
        default="official",
        help="Official collision mesh is the evaluation default.",
    )
    parser.add_argument(
        "--ur20-urdf",
        type=Path,
        help="Explicit UR20 URDF; overrides --ur20-source.",
    )
    parser.add_argument(
        "--sr12ia-urdf",
        type=Path,
        help=(
            "Explicit SR model for concurrent exclusion FK. The bundled "
            "conservative 300/450 proxy is used by default."
        ),
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        help="Write a compact result without per-sample q/J matrices.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        spec = load_scene_spec(args.config)
        corners = tuple(CORNER_SIGNS) if args.corner == "all" else (args.corner,)
        state = SceneState(
            lift_height_m=args.lift_height_mm / 1000.0,
            sku=str(args.sku),
            corner_samples=corners,
            tote_present=True,
            tote_long_axis_offset_m=args.tote_table_offset_mm / 1000.0,
            sr_j3_stroke_m=args.sr_j3_stroke_mm / 1000.0,
            clearance_m=(
                None
                if args.clearance_mm is None
                else args.clearance_mm / 1000.0
            ),
        )
        ur_base = (
            BasePose.from_sequence(args.ur_base, "--ur-base")
            if args.ur_base
            else spec.robots["ur20"].nominal_base
        )
        sr_base = (
            BasePose.from_sequence(args.sr_base, "--sr-base")
            if args.sr_base
            else spec.robots["sr12ia"].nominal_base
        )

        urdf = args.ur20_urdf
        ur_package_dirs: tuple[Path, ...] = ()
        if urdf is None and args.ur20_source == "official":
            urdf, ur_package_dirs = resolve_official_ur20()
        if urdf is None:
            urdf = spec.robots["ur20"].urdf_path
        ur_bundle = load_robot_bundle(
            "ur20",
            urdf,
            package_dirs=ur_package_dirs,
            floating_base=True,
            load_visual=False,
            suction_proxy=spec.robots["ur20"].suction_proxy,
        )
        sr_bundle = load_robot_bundle(
            "sr12ia",
            args.sr12ia_urdf or spec.robots["sr12ia"].urdf_path,
            floating_base=True,
            load_visual=False,
        )

        ik_options = IKOptions(
            position_tolerance_m=args.position_tolerance_mm / 1000.0,
            orientation_tolerance_rad=math.radians(
                args.orientation_tolerance_deg
            ),
            max_iterations=args.ik_max_iterations,
        )
        evaluation_options = EvaluationOptions(
            cartesian_translation_step_m=args.cartesian_step_mm / 1000.0,
            cartesian_rotation_step_rad=math.radians(
                args.cartesian_rotation_step_deg
            ),
            joint_interpolation_samples=args.joint_interpolation_samples,
            characteristic_length_m=args.characteristic_length_mm / 1000.0,
            ik=ik_options,
        )
        modes = _coordination_modes(args.coordination)
        result = evaluate_base_candidate(
            spec,
            state,
            ur_base,
            sr_base,
            ur_bundle,
            sr_bundle,
            coordination_modes=modes,
            options=evaluation_options,
            approach_distance_m=args.approach_mm / 1000.0,
            keepout_margin_m=args.keepout_margin_mm / 1000.0,
        )
        report = compact_candidate_report(result)
        report["evaluation_settings"] = {
            "ur20_urdf": str(Path(urdf).resolve()),
            "sr12ia_urdf": str(
                Path(args.sr12ia_urdf or spec.robots["sr12ia"].urdf_path).resolve()
            ),
            "approach_distance_m": args.approach_mm / 1000.0,
            "cartesian_translation_step_m": args.cartesian_step_mm / 1000.0,
            "cartesian_rotation_step_rad": math.radians(
                args.cartesian_rotation_step_deg
            ),
            "joint_interpolation_samples": args.joint_interpolation_samples,
            "characteristic_length_m": args.characteristic_length_mm / 1000.0,
            "keepout_margin_m": args.keepout_margin_mm / 1000.0,
            "ik_position_tolerance_m": args.position_tolerance_mm / 1000.0,
            "ik_orientation_tolerance_rad": math.radians(
                args.orientation_tolerance_deg
            ),
            "ik_max_iterations": args.ik_max_iterations,
        }
        _print_report(report)
        if args.output_json:
            output = args.output_json.resolve()
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(
                json.dumps(report, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            print(f"Saved candidate report: {output}")
        return 0
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


def compact_candidate_report(result: BaseCandidateEvaluation) -> dict:
    """Remove heavy q/J arrays while retaining every decision input."""

    return {
        "schema_version": 1,
        "workflow_schema_version": WORKFLOW_SCHEMA_VERSION,
        "units": "m_rad",
        "interpretation": (
            "IK not_found is a deterministic search result, not a global "
            "infeasibility proof; no candidate objective is applied."
        ),
        "state": {
            "sku": result.state.sku,
            "lift_height_m": result.state.lift_height_m,
            "corners": list(result.state.corner_samples),
            "tote_long_axis_offset_m": result.state.tote_long_axis_offset_m,
            "sr_j3_stroke_m": result.state.sr_j3_stroke_m,
            "clearance_m": result.state.clearance_m,
        },
        "bases": {
            "ur20": asdict(result.ur_base),
            "sr12ia": asdict(result.sr_base),
        },
        "placement": {
            "valid": result.placement.valid,
            "issues": [asdict(issue) for issue in result.placement.issues],
        },
        "sr_cutting_workspace": (
            None if result.sr_workspace is None else asdict(result.sr_workspace)
        ),
        "sr_concurrent_exclusion": (
            None if result.sr_keepout is None else asdict(result.sr_keepout)
        ),
        "coordination": [
            {
                "mode": summary.mode.value,
                "ur20_hard_feasible": summary.hard_feasible,
                "cell_feasible": summary.cell_feasible,
                "selected_corners": list(summary.selected_corners),
                "hard_feasible_corners": list(summary.hard_feasible_corners),
                "scenarios": [
                    _scenario_report(scenario) for scenario in summary.scenarios
                ],
            }
            for summary in result.coordination_summaries
        ],
    }


def _scenario_report(result: object) -> dict:
    return {
        "corner": result.scenario.corner,
        "hard_feasible": result.hard_feasible,
        "soft_step_successes": result.soft_step_successes,
        "soft_step_count": result.soft_step_count,
        "metrics": None if result.metrics is None else asdict(result.metrics),
        "steps": [
            {
                "index": step.index,
                "name": step.name,
                "criticality": step.criticality.value,
                "status": step.status,
                "waste_dimension_fits": step.waste_dimension_fits,
                "checks": [
                    {
                        "name": check.name,
                        "status": check.status,
                        "include_in_metric_summary": (
                            check.include_in_metric_summary
                        ),
                        "sample_count": len(check.samples),
                        "failed_sample_index": check.failed_sample_index,
                        "failure_contacts": _failure_contacts(check),
                        "last_pose_error": (
                            None
                            if not check.samples
                            else {
                                "position_m": check.samples[-1].ik.position_error_m,
                                "orientation_rad": check.samples[-1].ik.orientation_error_rad,
                            }
                        ),
                        "sample_metrics": [
                            _sample_metric_report(sample)
                            for sample in check.samples
                        ],
                    }
                    for check in step.checks
                ],
            }
            for step in result.steps
        ],
    }


def _failure_contacts(check: object) -> list[dict[str, str]]:
    report = check.joint_collision
    if report is None and check.samples:
        report = check.samples[-1].collision
    if report is None:
        return []
    return [asdict(contact) for contact in report.contacts]


def _sample_metric_report(sample: object) -> dict:
    metric = sample.manipulability
    result = {
        "fraction": sample.fraction,
        "ik_status": sample.ik.status,
        "position_error_m": sample.ik.position_error_m,
        "orientation_error_rad": sample.ik.orientation_error_rad,
        "manipulability": None,
    }
    if metric is not None:
        result["manipulability"] = {
            "translational_singular_values": [
                float(value) for value in metric.translational_singular_values
            ],
            "translational_volume": metric.translational_volume,
            "translational_min_singular_value": (
                metric.translational_min_singular_value
            ),
            "translational_condition_number": (
                metric.translational_condition_number
            ),
            "normalized_singular_values": [
                float(value) for value in metric.normalized_singular_values
            ],
            "normalized_volume": metric.normalized_volume,
            "normalized_min_singular_value": metric.normalized_min_singular_value,
            "normalized_condition_number": metric.normalized_condition_number,
            "characteristic_length_m": metric.characteristic_length_m,
        }
    return result


def _coordination_modes(value: str) -> tuple[CoordinationMode, ...]:
    if value == "both":
        return (CoordinationMode.SIMULTANEOUS, CoordinationMode.SEQUENTIAL)
    return (CoordinationMode(value),)


def _print_report(report: dict) -> None:
    placement = report["placement"]
    print(f"Installation valid: {placement['valid']}")
    for issue in placement["issues"]:
        print(f"  - {issue['code']}: {issue['message']}")
    sr_workspace = report["sr_cutting_workspace"]
    if sr_workspace is not None:
        print(
            "SR cutting workspace: "
            f"feasible={sr_workspace['feasible']}; "
            f"footprint={sr_workspace['footprint_ok']}; "
            f"planar={sr_workspace['planar_reach_ok']}; "
            f"vertical={sr_workspace['vertical_ok']}"
        )
    for summary in report["coordination"]:
        feasible = ", ".join(summary["hard_feasible_corners"]) or "none"
        print(
            f"{summary['mode']}: ur20_hard={summary['ur20_hard_feasible']}; "
            f"cell_feasible={summary['cell_feasible']}; "
            f"feasible_corners={feasible}"
        )
        for scenario in summary["scenarios"]:
            failed = [
                f"{step['index']}:{step['name']}"
                for step in scenario["steps"]
                if step["criticality"] == "hard" and step["status"] == "failed"
            ]
            print(
                f"  {scenario['corner']}: hard={scenario['hard_feasible']}; "
                f"failed_hard_steps={','.join(failed) or 'none'}"
            )


if __name__ == "__main__":
    raise SystemExit(main())
