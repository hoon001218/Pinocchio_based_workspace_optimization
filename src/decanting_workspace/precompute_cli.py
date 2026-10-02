"""CLI for finite, YAML-defined decanting case precomputation."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np
import yaml

from .evaluation import EvaluationOptions
from .kinematics import IKOptions
from .models import SceneSpec, default_config_path, load_scene_spec
from .precompute import (
    BasePoseGrid,
    CaseGrid,
    PrecomputedCache,
    iter_case_keys,
    precompute_cases,
    save_precomputed_cache,
)
from .robots import RobotBundle, load_robot_bundle, resolve_official_ur20


@dataclass(frozen=True)
class PrecomputeDependencies:
    """Injectable I/O and heavy-computation boundary used by :func:`main`."""

    spec_loader: Callable[..., SceneSpec]
    official_ur20_resolver: Callable[..., tuple[Path, tuple[Path, ...]]]
    robot_loader: Callable[..., RobotBundle]
    precomputer: Callable[..., PrecomputedCache]
    cache_saver: Callable[[PrecomputedCache, str | Path], Path]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Precompute an explicit finite base/task grid and save the complete "
            "replay-oriented result cache as strict JSON."
        )
    )
    parser.add_argument("--config", type=Path, default=default_config_path())
    parser.add_argument(
        "--usd",
        type=Path,
        help=(
            "Read environment geometry directly from a local USD/USDA/USDC "
            "scene; task and robot settings still come from --config."
        ),
    )
    parser.add_argument(
        "--grid",
        type=Path,
        required=True,
        help=(
            "YAML case grid. There is intentionally no implicit default grid; "
            "all continuous axes must be explicit."
        ),
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        required=True,
        help="Destination for the strict precomputed-cache JSON.",
    )
    parser.add_argument(
        "--max-cases",
        type=_positive_int,
        default=10_000,
        help="Abort before robot loading when the exact Cartesian product is larger.",
    )
    parser.add_argument(
        "--only-sku",
        action="append",
        default=[],
        help="Precompute only this SKU from the grid; may be repeated for shards.",
    )
    parser.add_argument(
        "--only-lift-height-mm",
        action="append",
        type=float,
        default=[],
        help=(
            "Precompute only this exact lift value from the grid; may be "
            "repeated for resumable/parallel shards."
        ),
    )
    parser.add_argument(
        "--ur20-source",
        choices=("official", "primitive"),
        default="official",
        help="Use the pinned official collision meshes or bundled primitive proxy.",
    )
    parser.add_argument(
        "--ur20-urdf",
        type=Path,
        help="Explicit UR20 URDF; overrides --ur20-source.",
    )
    parser.add_argument(
        "--sr12ia-urdf",
        type=Path,
        help="Explicit SR collision model; defaults to the configured proxy.",
    )

    parser.add_argument("--approach-mm", type=float, default=150.0)
    parser.add_argument("--cartesian-step-mm", type=float, default=50.0)
    parser.add_argument(
        "--cartesian-rotation-step-deg",
        type=float,
        default=5.0,
    )
    parser.add_argument("--joint-interpolation-samples", type=int, default=3)
    parser.add_argument("--characteristic-length-mm", type=float, default=500.0)
    parser.add_argument("--keepout-margin-mm", type=float, default=0.0)
    parser.add_argument("--position-tolerance-mm", type=float, default=0.001)
    parser.add_argument(
        "--orientation-tolerance-deg",
        type=float,
        default=0.0001,
    )
    parser.add_argument("--ik-max-iterations", type=int, default=500)
    parser.add_argument("--ik-damping", type=float, default=1e-6)
    parser.add_argument("--ik-max-backtracking-steps", type=int, default=14)
    parser.add_argument(
        "--extra-arm-seed-deg",
        action="append",
        nargs=6,
        type=float,
        metavar=("J1", "J2", "J3", "J4", "J5", "J6"),
        default=[],
        help="Additional deterministic UR20 IK seed; may be supplied repeatedly.",
    )
    return parser


def load_case_grid_yaml(path: str | Path) -> CaseGrid:
    """Load the explicit version-1 metre-based case-grid YAML schema.

    Expected top-level keys are ``ur_base``, ``sr_base``, ``skus``,
    ``lift_heights_m``, ``tote_offsets_m``, ``corners``,
    ``coordination_modes``, ``sr_j3_strokes_m``, and ``clearances_m``.
    ``ur_bases``/``sr_bases`` are accepted as aliases because the Python data
    model uses those plural field names.
    """

    source = Path(path).resolve()
    if not source.is_file():
        raise FileNotFoundError(f"case-grid YAML not found: {source}")
    raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    root = _mapping_value(raw, "case-grid root")
    if root.get("schema_version") != 1:
        raise ValueError("case-grid schema_version must be 1")
    if root.get("units") != "m":
        raise ValueError("case-grid units must be 'm'")
    payload = (
        _mapping_value(root.get("grid"), "grid") if "grid" in root else root
    )
    ur_raw = _aliased_mapping(payload, "ur_base", "ur_bases")
    sr_raw = _aliased_mapping(payload, "sr_base", "sr_bases")
    tote_present = payload.get("tote_present", True)
    if not isinstance(tote_present, bool):
        raise ValueError("tote_present must be a boolean")

    return CaseGrid(
        ur_bases=_base_pose_grid(ur_raw, "ur_base"),
        sr_bases=_base_pose_grid(sr_raw, "sr_base"),
        skus=tuple(str(value) for value in _sequence(payload, "skus")),
        lift_heights_m=_float_axis(payload, "lift_heights_m"),
        tote_offsets_m=_float_axis(payload, "tote_offsets_m"),
        corners=tuple(str(value) for value in _sequence(payload, "corners")),
        coordination_modes=tuple(
            str(value) for value in _sequence(payload, "coordination_modes")
        ),
        sr_j3_strokes_m=_float_axis(payload, "sr_j3_strokes_m"),
        clearances_m=tuple(
            None if value is None else _finite_float(value, "clearances_m")
            for value in _sequence(payload, "clearances_m")
        ),
        tote_present=tote_present,
    )


def main(
    argv: list[str] | None = None,
    *,
    dependencies: PrecomputeDependencies | None = None,
) -> int:
    args = build_parser().parse_args(argv)
    deps = dependencies or _default_dependencies()
    try:
        spec = (
            deps.spec_loader(args.config)
            if args.usd is None
            else deps.spec_loader(args.config, usd_path=args.usd)
        )
        grid = load_case_grid_yaml(args.grid)
        grid = _filter_grid(grid, args.only_sku, args.only_lift_height_mm)
        case_count = grid.case_count
        print(f"Exact case count: {case_count}")
        if case_count > args.max_cases:
            raise ValueError(
                f"case grid has {case_count} cases, exceeding --max-cases "
                f"{args.max_cases}"
            )

        # Trigger precompute's complete scene/grid validation without
        # materializing the Cartesian product or loading either robot.
        next(iter_case_keys(spec, grid), None)

        urdf = args.ur20_urdf
        ur_package_dirs: tuple[Path, ...] = ()
        if urdf is None and args.ur20_source == "official":
            urdf, ur_package_dirs = deps.official_ur20_resolver()
        if urdf is None:
            urdf = spec.robots["ur20"].urdf_path
        sr_urdf = args.sr12ia_urdf or spec.robots["sr12ia"].urdf_path

        # Make the cache fingerprint describe the models that were actually
        # evaluated rather than the fallback paths in the source config.
        evaluation_spec = _with_robot_paths(spec, Path(urdf), Path(sr_urdf))
        ur_bundle = deps.robot_loader(
            "ur20",
            urdf,
            package_dirs=ur_package_dirs,
            floating_base=True,
            load_visual=False,
            suction_proxy=evaluation_spec.robots["ur20"].suction_proxy,
        )
        sr_bundle = deps.robot_loader(
            "sr12ia",
            sr_urdf,
            floating_base=True,
            load_visual=False,
        )
        options = _evaluation_options(args)
        cache = deps.precomputer(
            evaluation_spec,
            grid,
            ur_bundle,
            sr_bundle,
            options=options,
            approach_distance_m=args.approach_mm / 1000.0,
            keepout_margin_m=args.keepout_margin_mm / 1000.0,
        )
        output = deps.cache_saver(cache, args.output_json)
        print(f"Saved {case_count} precomputed cases: {Path(output).resolve()}")
        return 0
    except (FileNotFoundError, OSError, RuntimeError, ValueError, yaml.YAMLError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


def _evaluation_options(args: argparse.Namespace) -> EvaluationOptions:
    ik = IKOptions(
        position_tolerance_m=args.position_tolerance_mm / 1000.0,
        orientation_tolerance_rad=math.radians(args.orientation_tolerance_deg),
        max_iterations=args.ik_max_iterations,
        damping=args.ik_damping,
        max_backtracking_steps=args.ik_max_backtracking_steps,
    )
    return EvaluationOptions(
        cartesian_translation_step_m=args.cartesian_step_mm / 1000.0,
        cartesian_rotation_step_rad=math.radians(
            args.cartesian_rotation_step_deg
        ),
        joint_interpolation_samples=args.joint_interpolation_samples,
        characteristic_length_m=args.characteristic_length_mm / 1000.0,
        ik=ik,
        extra_arm_seeds=tuple(
            tuple(float(value) for value in np.deg2rad(seed))
            for seed in args.extra_arm_seed_deg
        ),
    )


def _filter_grid(
    grid: CaseGrid,
    only_skus: Sequence[str],
    only_lift_heights_mm: Sequence[float],
) -> CaseGrid:
    result = grid
    if only_skus:
        requested = tuple(dict.fromkeys(str(value) for value in only_skus))
        missing = tuple(value for value in requested if value not in grid.skus)
        if missing:
            raise ValueError(f"--only-sku values are absent from the grid: {missing}")
        result = replace(result, skus=requested)
    if only_lift_heights_mm:
        requested_m = tuple(
            dict.fromkeys(float(value) / 1000.0 for value in only_lift_heights_mm)
        )
        selected = []
        for requested in requested_m:
            match = next(
                (
                    value
                    for value in grid.lift_heights_m
                    if math.isclose(value, requested, abs_tol=1e-12)
                ),
                None,
            )
            if match is None:
                raise ValueError(
                    "--only-lift-height-mm value is absent from the grid: "
                    f"{requested * 1000.0}"
                )
            selected.append(match)
        result = replace(result, lift_heights_m=tuple(selected))
    return result


def _with_robot_paths(
    spec: SceneSpec,
    urdf: Path,
    sr_urdf: Path,
) -> SceneSpec:
    robots = dict(spec.robots)
    robots["ur20"] = replace(robots["ur20"], urdf_path=urdf.resolve())
    robots["sr12ia"] = replace(robots["sr12ia"], urdf_path=sr_urdf.resolve())
    return replace(spec, robots=robots)


def _default_dependencies() -> PrecomputeDependencies:
    # Construct at call time so normal module-level monkeypatching remains
    # effective in addition to explicit dependency injection.
    return PrecomputeDependencies(
        spec_loader=load_scene_spec,
        official_ur20_resolver=resolve_official_ur20,
        robot_loader=load_robot_bundle,
        precomputer=precompute_cases,
        cache_saver=save_precomputed_cache,
    )


def _base_pose_grid(raw: Mapping[str, object], name: str) -> BasePoseGrid:
    return BasePoseGrid(
        x_m=_float_axis(raw, "x_m", prefix=name),
        y_m=_float_axis(raw, "y_m", prefix=name),
        z_m=_float_axis(raw, "z_m", prefix=name),
        yaw_deg=_float_axis(raw, "yaw_deg", prefix=name),
    )


def _aliased_mapping(
    raw: Mapping[str, object],
    primary: str,
    alias: str,
) -> Mapping[str, object]:
    present = tuple(key for key in (primary, alias) if key in raw)
    if len(present) != 1:
        raise ValueError(f"case grid must contain exactly one of {primary!r}, {alias!r}")
    return _mapping_value(raw[present[0]], present[0])


def _mapping_value(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value


def _sequence(raw: Mapping[str, object], key: str) -> Sequence[object]:
    value = raw.get(key)
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{key} must be an explicit sequence")
    return value


def _float_axis(
    raw: Mapping[str, object],
    key: str,
    *,
    prefix: str | None = None,
) -> tuple[float, ...]:
    name = key if prefix is None else f"{prefix}.{key}"
    return tuple(_finite_float(value, name) for value in _sequence(raw, key))


def _finite_float(value: object, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} values must be numeric") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} values must be finite")
    return result


def _positive_int(value: str) -> int:
    try:
        result = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if result <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return result


if __name__ == "__main__":
    raise SystemExit(main())
