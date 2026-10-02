"""Prepare and validate a relocatable decanting-workspace checkout."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
import json
from pathlib import Path, PureWindowsPath
import sys
import tempfile

from .asset_prep import (
    FANUC_3D_CONTENT_SHARING_AGREEMENT_URL,
    default_sr12ia_output_urdf,
    download_sr12ia_geometry,
    prepare_sr12ia_assets,
    validate_prepared_sr12ia_asset,
)
from .models import SceneSpec, default_config_path, load_scene_spec
from .paths import portable_repository_reference, repository_path
from .precompute import PrecomputedCache, load_precomputed_cache
from .robots import resolve_official_ur20, validate_vendored_ur20_asset


@dataclass(frozen=True)
class SetupDependencies:
    """Injectable boundaries for deterministic setup tests."""

    spec_loader: Callable[..., SceneSpec]
    ur20_validator: Callable[[], Path]
    ur20_resolver: Callable[..., tuple[Path, tuple[Path, ...]]]
    sr12ia_path: Callable[[], Path]
    sr12ia_validator: Callable[[str | Path | None], dict[str, object]]
    sr12ia_preparer: Callable[..., Path]
    sr12ia_downloader: Callable[..., Path]
    cache_loader: Callable[..., PrecomputedCache]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare licensed local assets and verify that this checkout can "
            "replay its tracked Pinocchio result caches on the current PC."
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
    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "--sr12ia-usd",
        type=Path,
        help="authorized local Fanuc/sr12ia/payloads/geometries.usd",
    )
    source.add_argument(
        "--accept-fanuc-license",
        action="store_true",
        help=(
            "review and accept the linked FANUC agreement, then download the "
            "pinned NVIDIA geometry when a valid local bundle is absent"
        ),
    )
    parser.add_argument(
        "--refresh-sr12ia",
        action="store_true",
        help="rebuild SR geometry even when an existing bundle validates",
    )
    parser.add_argument(
        "--skip-cache-validation",
        action="store_true",
        help="prepare robot assets without validating tracked result caches",
    )
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    dependencies: SetupDependencies | None = None,
) -> int:
    args = build_parser().parse_args(argv)
    deps = dependencies or _default_dependencies()
    try:
        if args.refresh_sr12ia and not (
            args.sr12ia_usd is not None or args.accept_fanuc_license
        ):
            raise ValueError(
                "--refresh-sr12ia requires --sr12ia-usd or "
                "--accept-fanuc-license"
            )

        spec = (
            deps.spec_loader(args.config)
            if args.usd is None
            else deps.spec_loader(args.config, usd_path=args.usd)
        )
        print(f"OK scene configuration: {_portable_path(spec.source_path)}")

        vendored_urdf = deps.ur20_validator()
        resolved_urdf, _ = deps.ur20_resolver()
        if Path(vendored_urdf).resolve() != Path(resolved_urdf).resolve():
            raise ValueError("default official UR20 resolver is not using the vendored bundle")
        print(f"OK official UR20 bundle: {_portable_path(vendored_urdf)}")

        sr_urdf = _ensure_sr12ia_asset(args, deps)
        provenance = deps.sr12ia_validator(sr_urdf)
        print(
            "OK FANUC SR-12iA local bundle: "
            f"{_portable_path(sr_urdf)} "
            f"(source {str(provenance.get('source_sha256', 'unknown'))[:12]}...)"
        )

        if not args.skip_cache_validation:
            evaluation_spec = _with_robot_paths(
                spec,
                Path(resolved_urdf),
                spec.robots["sr12ia"].urdf_path,
            )
            for cache_path in _tracked_cache_paths():
                cache = deps.cache_loader(cache_path, spec=evaluation_spec)
                print(
                    f"OK precomputed cache: {_portable_path(cache_path)} "
                    f"({len(cache.cases)} cases)"
                )
            _validate_candidate_report_paths()
            print("OK candidate reports contain no checkout-specific model paths")

        print(
            "Setup complete. This checkout is ready for MeshCat playback "
            "and live evaluation."
        )
        return 0
    except (FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        if "FANUC" in str(exc) or "SR-12iA" in str(exc):
            print(
                "Review the FANUC agreement before downloading: "
                f"{FANUC_3D_CONTENT_SHARING_AGREEMENT_URL}",
                file=sys.stderr,
            )
        return 2


def _ensure_sr12ia_asset(
    args: argparse.Namespace,
    deps: SetupDependencies,
) -> Path:
    target = deps.sr12ia_path()
    if args.sr12ia_usd is not None:
        return deps.sr12ia_preparer(args.sr12ia_usd, target.parent)

    if not args.refresh_sr12ia:
        try:
            deps.sr12ia_validator(target)
            return target
        except (FileNotFoundError, OSError, ValueError):
            pass

    if not args.accept_fanuc_license:
        raise FileNotFoundError(
            "prepared FANUC SR-12iA mesh is missing or stale; provide "
            "--sr12ia-usd PATH or rerun with --accept-fanuc-license"
        )

    with tempfile.TemporaryDirectory(prefix="sr12ia_download_") as directory:
        source = deps.sr12ia_downloader(
            Path(directory) / "geometries.usd",
            accept_fanuc_license=True,
        )
        return deps.sr12ia_preparer(source, target.parent)


def _with_robot_paths(spec: SceneSpec, urdf: Path, sr_urdf: Path) -> SceneSpec:
    robots = dict(spec.robots)
    robots["ur20"] = replace(robots["ur20"], urdf_path=urdf.resolve())
    robots["sr12ia"] = replace(robots["sr12ia"], urdf_path=sr_urdf.resolve())
    return replace(spec, robots=robots)


def _project_root() -> Path:
    return repository_path()


def _portable_path(path: str | Path) -> str:
    return portable_repository_reference(path)


def _tracked_cache_paths() -> tuple[Path, Path]:
    outputs = _project_root() / "outputs"
    return (
        outputs / "precomputed_working_cases.json.gz",
        outputs / "precomputed_nominal_cases.json.gz",
    )


def _validate_candidate_report_paths() -> None:
    outputs = _project_root() / "outputs"
    for name in (
        "working_candidate_full_path.json",
        "nominal_candidate_all_corners.json",
    ):
        path = outputs / name
        try:
            report = json.loads(path.read_text(encoding="utf-8"))
            settings = report["evaluation_settings"]
        except (FileNotFoundError, json.JSONDecodeError, KeyError, TypeError) as exc:
            raise ValueError(f"invalid tracked candidate report: {path}") from exc
        for key in ("ur20_urdf", "sr12ia_urdf"):
            value = settings.get(key)
            if not isinstance(value, str) or not value:
                raise ValueError(f"candidate report has invalid {key}: {path}")
            if Path(value).is_absolute() or PureWindowsPath(value).is_absolute():
                raise ValueError(
                    f"candidate report embeds an absolute model path ({key}): {path}"
                )


def _default_dependencies() -> SetupDependencies:
    return SetupDependencies(
        spec_loader=load_scene_spec,
        ur20_validator=validate_vendored_ur20_asset,
        ur20_resolver=resolve_official_ur20,
        sr12ia_path=default_sr12ia_output_urdf,
        sr12ia_validator=validate_prepared_sr12ia_asset,
        sr12ia_preparer=prepare_sr12ia_assets,
        sr12ia_downloader=download_sr12ia_geometry,
        cache_loader=load_precomputed_cache,
    )


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
