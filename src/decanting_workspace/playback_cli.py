"""Run the precomputed-result control panel with a live MeshCat viewer."""

from __future__ import annotations

import argparse
from collections.abc import Callable
from dataclasses import dataclass, replace
import math
from pathlib import Path
import sys
import webbrowser

from .asset_prep import (
    default_sr12ia_output_urdf,
    validate_prepared_sr12ia_asset,
)
from .meshcat_ui import MeshcatControlServer
from .models import SceneSpec, SceneState, default_config_path, load_scene_spec
from .playback_backend import make_meshcat_backend, preferred_cached_case
from .precompute import PrecomputedCache, load_precomputed_cache
from .robots import resolve_official_ur20
from .scene import materialize_scene
from .viewer import CellViewer


@dataclass(frozen=True)
class PlaybackDependencies:
    spec_loader: Callable[[str | Path | None], SceneSpec]
    cache_loader: Callable[..., PrecomputedCache]
    official_ur20_resolver: Callable[..., tuple[Path, tuple[Path, ...]]]
    viewer_factory: Callable[[], CellViewer]
    backend_factory: Callable[..., object]
    server_factory: Callable[..., MeshcatControlServer]
    browser_open: Callable[[str], object]
    prepared_sr12ia_resolver: Callable[[], Path] | None = None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Open a live MeshCat UI that replays exact precomputed Pinocchio "
            "cases, task poses, and manipulability ellipsoids without rerunning IK."
        )
    )
    parser.add_argument("--config", type=Path, default=default_config_path())
    parser.add_argument(
        "--cache",
        type=Path,
        default=default_working_cache_path(),
        help=(
            "Precomputed cache to replay. Defaults to the repository's "
            "verified commissioning setup."
        ),
    )
    parser.add_argument(
        "--ur20-source",
        choices=("official", "primitive"),
        default="official",
        help="Must match the URDF used by decanting-precompute.",
    )
    parser.add_argument(
        "--ur20-urdf",
        type=Path,
        help="Explicit evaluation/visual URDF; overrides --ur20-source.",
    )
    parser.add_argument(
        "--sr12ia-urdf",
        type=Path,
        help=(
            "Explicit SR evaluation URDF used for cache fingerprint validation. "
            "It is also displayed unless --sr12ia-visual-urdf is supplied."
        ),
    )
    parser.add_argument(
        "--sr12ia-visual-urdf",
        type=Path,
        help=(
            "Display-only SR URDF. By default playback uses the locally prepared "
            "FANUC/Isaac mesh without changing the cached evaluation model."
        ),
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=_port, default=8765)
    parser.add_argument("--ellipsoid-scale", type=_positive_float, default=0.15)
    parser.add_argument("--show-collisions", action="store_true")
    parser.add_argument("--hide-frames", action="store_true")
    parser.add_argument("--open", action="store_true")
    return parser


def main(
    argv: list[str] | None = None,
    *,
    dependencies: PlaybackDependencies | None = None,
) -> int:
    args = build_parser().parse_args(argv)
    deps = dependencies or _default_dependencies()
    server = None
    try:
        source_spec = deps.spec_loader(args.config)
        urdf = args.ur20_urdf
        package_dirs: tuple[Path, ...] = ()
        if urdf is None and args.ur20_source == "official":
            urdf, package_dirs = deps.official_ur20_resolver()
        if urdf is None:
            urdf = source_spec.robots["ur20"].urdf_path
        evaluation_sr_urdf = (
            args.sr12ia_urdf or source_spec.robots["sr12ia"].urdf_path
        )
        spec = _with_robot_paths(
            source_spec,
            Path(urdf),
            Path(evaluation_sr_urdf),
        )
        cache = deps.cache_loader(args.cache, spec=spec)
        if not cache.cases:
            raise ValueError("precomputed cache contains no cases")

        first = preferred_cached_case(cache)
        if first is None:  # guarded by the non-empty check above
            raise ValueError("precomputed cache contains no selectable cases")
        state = SceneState(
            lift_height_m=first.key.lift_height_m,
            sku=first.key.sku,
            corner_samples=(first.key.corner,),
            tote_present=cache.grid.tote_present,
            tote_long_axis_offset_m=first.key.tote_offset_m,
            sr_j3_stroke_m=first.key.sr_j3_stroke_m,
            clearance_m=first.key.clearance_m,
        )
        snapshot = materialize_scene(
            spec,
            state,
            ur_base=first.key.ur_base,
            sr_base=first.key.sr_base,
        )
        visual_sr_urdf = args.sr12ia_visual_urdf
        using_prepared_sr_mesh = False
        if visual_sr_urdf is None and args.sr12ia_urdf is not None:
            visual_sr_urdf = args.sr12ia_urdf
        if visual_sr_urdf is None:
            try:
                resolver = (
                    deps.prepared_sr12ia_resolver
                    or _resolve_prepared_sr12ia_visual
                )
                visual_sr_urdf = resolver()
                using_prepared_sr_mesh = True
            except (FileNotFoundError, OSError, ValueError) as exc:
                visual_sr_urdf = Path(evaluation_sr_urdf)
                print(
                    "WARNING: prepared FANUC SR-12iA mesh is unavailable "
                    f"({exc}); displaying the evaluation model.",
                    file=sys.stderr,
                )
        display_only_sr_model = (
            Path(visual_sr_urdf).resolve()
            != Path(evaluation_sr_urdf).resolve()
        )

        viewer = deps.viewer_factory()
        viewer.render(
            spec,
            snapshot,
            ur20_urdf=urdf,
            ur20_package_dirs=package_dirs,
            sr12ia_urdf=visual_sr_urdf,
            sr12ia_visual_only=display_only_sr_model,
            show_collisions=args.show_collisions,
            show_frames=not args.hide_frames,
        )
        backend = deps.backend_factory(
            spec,
            cache,
            viewer,
            ellipsoid_scale=args.ellipsoid_scale,
        )
        server = deps.server_factory(
            (args.host, args.port),
            backend,
            viewer.url(),
        )
        print(f"MeshCat URL: {viewer.url()}")
        print(f"Playback UI: {server.control_url}")
        print(f"Loaded exact cases: {len(cache.cases)}")
        print(f"SR-12iA visual model: {Path(visual_sr_urdf).resolve()}")
        visual_native_stroke = 0.300 if using_prepared_sr_mesh else None
        if display_only_sr_model and hasattr(viewer, "loaded_robot"):
            visual_native_stroke = viewer.loaded_robot(
                "sr12ia"
            ).native_j3_stroke_m
        if visual_native_stroke is not None and any(
            case.key.sr_j3_stroke_m > visual_native_stroke + 1.0e-12
            for case in cache.cases
        ):
            print(
                "WARNING: the display-only SR model has a shorter native J3 "
                "stroke than at least one cached evaluation option. Feasibility "
                "remains based on the evaluation model; playback rejects any "
                "cached nominal pose beyond the visual model's native limit.",
                file=sys.stderr,
            )
        if display_only_sr_model and args.show_collisions:
            print(
                "WARNING: displayed SR collision geometry belongs to the visual "
                "surrogate, not the cached evaluation collision model.",
                file=sys.stderr,
            )
        if args.open:
            deps.browser_open(server.control_url)
        print("Press Ctrl+C to stop the playback UI.")
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
        return 0
    except (FileNotFoundError, KeyError, OSError, RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    finally:
        if server is not None:
            server.server_close()


def _with_robot_paths(spec: SceneSpec, urdf: Path, sr_urdf: Path) -> SceneSpec:
    robots = dict(spec.robots)
    robots["ur20"] = replace(robots["ur20"], urdf_path=urdf.resolve())
    robots["sr12ia"] = replace(robots["sr12ia"], urdf_path=sr_urdf.resolve())
    return replace(spec, robots=robots)


def default_working_cache_path() -> Path:
    """Return the repository-owned, full-path verified playback cache."""

    return (
        Path(__file__).resolve().parents[2]
        / "outputs"
        / "precomputed_working_cases.json.gz"
    )


def _default_dependencies() -> PlaybackDependencies:
    return PlaybackDependencies(
        spec_loader=load_scene_spec,
        cache_loader=load_precomputed_cache,
        official_ur20_resolver=resolve_official_ur20,
        prepared_sr12ia_resolver=_resolve_prepared_sr12ia_visual,
        viewer_factory=CellViewer,
        backend_factory=make_meshcat_backend,
        server_factory=MeshcatControlServer,
        browser_open=webbrowser.open,
    )


def _resolve_prepared_sr12ia_visual() -> Path:
    """Return the validated, locally generated FANUC display URDF."""

    path = default_sr12ia_output_urdf()
    validate_prepared_sr12ia_asset(path)
    return path.resolve()


def _port(value: str) -> int:
    try:
        result = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("port must be an integer") from exc
    if not 0 <= result <= 65535:
        raise argparse.ArgumentTypeError("port must be in [0, 65535]")
    return result


def _positive_float(value: str) -> float:
    try:
        result = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be numeric") from exc
    if not math.isfinite(result) or result <= 0.0:
        raise argparse.ArgumentTypeError("must be finite and positive")
    return result


if __name__ == "__main__":
    raise SystemExit(main())
